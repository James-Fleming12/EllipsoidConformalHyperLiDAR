import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from common.sync_batchnorm.batchnorm import convert_model
from modules.losses.Lovasz_Softmax import Lovasz_softmax
from modules.losses.boundary_loss import BoundaryLoss
from modules.scheduler.warmupLR import warmupLR
from modules.scheduler.cosine import CosineAnnealingWarmUpRestarts
from common.avgmeter import AverageMeter
from modules.ioueval import iouEval
from modules.trainer import save_checkpoint

class BeamDensityEstimator(nn.Module):
    """
    Computes a multi-scale beam density map on a spherical projection image
    for a range-image pipeline.

    For every pixel (row, col) in an H×W range image the estimator returns a
    4-dimensional density vector obtained by convolving 1-D Gaussian kernels
    (σ ∈ {10, 30, 50, 70}) over binary beam-presence vectors and combining
    horizontal and vertical responses divided by r².

    Parameters
    ----------
    proj_H, proj_W : int
        Height and width of the spherical range image.
    f_min, f_max : float
        Minimum / maximum vertical field-of-view of the *projection* image
        (degrees).  Set to [-30, 15] to cover all common sensors.
    sensor_beam_config : dict with keys 'Vb', 'f_min_sensor', 'f_max_sensor', 'Hb'
        Sensor-specific beam configuration used to build the binary indicator
        vectors B_h and B_v.
    sigmas : list[float]
        Standard deviations for the Gaussian kernels.
    """

    def __init__(
        self,
        proj_H: int,
        proj_W: int,
        f_min: float = -30.0,
        f_max: float = 15.0,
        sensor_beam_config: dict = None,
        sigmas=(10.0, 30.0, 50.0, 70.0),
    ):
        super().__init__()
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.f_min = f_min
        self.f_max = f_max
        self.sigmas = list(sigmas)
        self.n_sigmas = len(sigmas)

        if sensor_beam_config is None:
            sensor_beam_config = {
                "Vb": 64,
                "f_min_sensor": -24.8,
                "f_max_sensor": 2.0,
                "Hb": 2048,
            }

        Hb = sensor_beam_config["Hb"]
        Vb = sensor_beam_config["Vb"]
        f_min_s = sensor_beam_config["f_min_sensor"]
        f_max_s = sensor_beam_config["f_max_sensor"]

        # Horizontal beam azimuths → projected column indices
        azimuth_beams = np.array(
            [2 * np.pi * i / Hb for i in range(1, Hb + 1)]
        )  # radians
        col_indices = np.floor(azimuth_beams / (2 * np.pi) * proj_W).astype(int) % proj_W

        Bh = np.zeros(proj_W, dtype=np.float32)
        Bh[col_indices] = 1.0

        # Vertical beam elevations (degrees) → projected row indices
        elev_beams_deg = np.array(
            [
                (f_max_s - f_min_s) * j / Vb + f_min_s
                for j in range(1, Vb + 1)
            ]
        )
        row_indices = np.floor(
            (elev_beams_deg - f_min) / (f_max - f_min) * proj_H
        ).astype(int)
        row_indices = np.clip(row_indices, 0, proj_H - 1)

        Bv = np.zeros(proj_H, dtype=np.float32)
        Bv[row_indices] = 1.0

        def _convolve_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
            full   = np.convolve(signal, kernel, mode='full')
            offset = (len(kernel) - 1) // 2
            return full[offset: offset + len(signal)]

        Bh_smooth = []
        Bv_smooth = []
        for sigma in sigmas:
            ks = int(6 * sigma) | 1          # odd kernel size ≈ 6σ
            half = ks // 2
            xs = np.arange(-half, half + 1, dtype=np.float32)
            g = np.exp(-0.5 * (xs / sigma) ** 2)
            g /= g.sum()
            Bh_smooth.append(_convolve_same(Bh, g))  # always len proj_W
            Bv_smooth.append(_convolve_same(Bv, g))  # always len proj_H

        # [n_sigmas, proj_H, proj_W]  – outer product per scale
        density_map = np.zeros(
            (self.n_sigmas, proj_H, proj_W), dtype=np.float32
        )
        for k in range(self.n_sigmas):
            # outer product: Bv_smooth[:,None] * Bh_smooth[None,:]
            density_map[k] = (
                Bv_smooth[k][:, None] * Bh_smooth[k][None, :]
            )

        # Register as non-trainable buffer; shape [n_sigmas, H, W]
        self.register_buffer(
            "density_map", torch.from_numpy(density_map)
        )

    def forward(self, range_img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        range_img : torch.Tensor, shape [B, C, H, W]
            The spherical range image.  Channel 0 is expected to be the
            radial distance r (in metres), already normalised or raw.

        Returns
        -------
        density : torch.Tensor, shape [B, n_sigmas, H, W]
            Per-pixel multi-scale beam density D_i = sqrt(B̂h * B̂v / r²).
        """
        # r: [B, 1, H, W]  – radial distance from channel 0
        r = range_img[:, 0:1, :, :].clamp(min=1e-3)

        # density_map: [n_sigmas, H, W] → [1, n_sigmas, H, W]
        dm = self.density_map.unsqueeze(0)           # broadcast over batch

        density = torch.sqrt(dm / (r ** 2 + 1e-6))  # [B, n_sigmas, H, W]
        return density


# ---------------------------------------------------------------------------
# Density Soft Clipping
# ---------------------------------------------------------------------------

class DensitySoftClipper:
    """
    Maintains running 10th / 90th percentile statistics of density values
    using reservoir sampling and applies tanh-based soft clipping.

    Call `update(density)` during training to accumulate statistics and
    `clip(density)` to apply the transformation described in Eq. (6).
    """

    def __init__(self, reservoir_size: int = 1000, n_channels: int = 4):
        self.reservoir_size = reservoir_size
        self.n_channels = n_channels
        self._reservoir = [[] for _ in range(n_channels)]
        self._count = 0

        # Percentile statistics (initialised to neutral values)
        self.m = np.zeros(n_channels, dtype=np.float32)
        self.l = np.ones(n_channels, dtype=np.float32)

    # ---- reservoir sampling update ----------------------------------------
    @torch.no_grad()
    def update(self, density: torch.Tensor):
        """
        density : [B, n_sigmas, H, W] or [B, n_sigmas, N_points]
        """
        flat = density.detach().cpu().float()
        # flatten spatial dims → [B*H*W, n_sigmas]
        flat = flat.permute(0, 2, 3, 1).reshape(-1, self.n_channels).numpy() \
            if flat.dim() == 4 else \
            flat.permute(0, 2, 1).reshape(-1, self.n_channels).numpy()

        for ch in range(self.n_channels):
            vals = flat[:, ch].tolist()
            for v in vals:
                self._count += 1
                if len(self._reservoir[ch]) < self.reservoir_size:
                    self._reservoir[ch].append(v)
                else:
                    j = np.random.randint(0, self._count)
                    if j < self.reservoir_size:
                        self._reservoir[ch][j] = v

        # recompute percentiles
        for ch in range(self.n_channels):
            if len(self._reservoir[ch]) >= 10:
                arr = np.array(self._reservoir[ch])
                p90 = np.percentile(arr, 90)
                p10 = np.percentile(arr, 10)
                self.m[ch] = (p90 + p10) / 2.0
                self.l[ch] = max((p90 - p10) / 2.0, 1e-6)

    def clip(self, density: torch.Tensor) -> torch.Tensor:
        """
        Apply Eq. (6): D^c = tanh((D - m) / l) * l + m

        density : [B, n_sigmas, H, W]
        Returns clipped density of same shape.
        """
        device = density.device
        m = torch.tensor(self.m, dtype=torch.float32, device=device)
        l = torch.tensor(self.l, dtype=torch.float32, device=device)
        # reshape for broadcast: [1, n_sigmas, 1, 1]
        m = m.view(1, -1, 1, 1)
        l = l.view(1, -1, 1, 1)
        return torch.tanh((density - m) / l) * l + m

class DensityAwareEmbedding(nn.Module):
    """
    Range-image adaptation of the density-aware embedding module (Sec. 3.4).

    Instead of point-wise and voxel-wise branches, we have pixel-wise and
    local-region (pooled) branches that mirror the architecture for 2-D
    feature maps.

    Parameters
    ----------
    in_channels : int
        Number of channels of the backbone feature map F.
    density_channels : int
        Number of Gaussian scales (= n_sigmas, default 4).
    out_channels : int
        Output channel count (default 32, matching DDFE paper).
    """

    def __init__(
        self,
        in_channels: int,
        density_channels: int = 4,
        out_channels: int = 32,
    ):
        super().__init__()

        # pixel-wise attention branch: f_p  (1D Conv ↔ 1×1 Conv2D here)
        self.pixel_attn = nn.Sequential(
            nn.Conv2d(density_channels, density_channels, kernel_size=1),
            nn.BatchNorm2d(density_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(density_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # local-region attention branch: f_v  (uses avg-pooled neighbourhood)
        self.region_attn = nn.Sequential(
            nn.Conv2d(density_channels, density_channels, kernel_size=1),
            nn.BatchNorm2d(density_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(density_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # local pooling kernel (mimics voxel aggregation for point-wise max pool)
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_local_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # final projection to out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, features: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : [B, C, H, W]   backbone feature map
        density  : [B, D, H, W]   clipped density (D = density_channels)

        Returns
        -------
        out : [B, out_channels, H, W]   density-discriminative features
        """
        # pixel-wise attention
        p_attn = self.pixel_attn(density)           # [B, C, H, W]
        F_pixel = p_attn * features                 # Eq. (7) analogue

        # region-wise attention (avg-pooled density → local descriptor)
        d_region = self.local_pool(density)
        r_attn = self.region_attn(d_region)         # [B, C, H, W]
        F_region = r_attn * features
        F_region_max = self.max_local_pool(F_region)

        # concatenate and project  →  Eq. (8) analogue
        out = self.proj(torch.cat([F_pixel, F_region_max], dim=1))
        return out

class DDFETrainer:
    """
    Trainer for range-image semantic segmentation enhanced with the Density
    Discriminative Feature Embedding (DDFE) principles from
    "Rethinking LiDAR Domain Generalization: Single Source as Multiple
    Density Domains" (Kim et al., ECCV 2024).
    """

    # ---- sensor beam configs for common datasets --------------------------
    SENSOR_CONFIGS = {
        "SemanticKITTI": {"Vb": 64, "f_min_sensor": -24.8, "f_max_sensor": 2.0,  "Hb": 2048},
        "Waymo":         {"Vb": 64, "f_min_sensor": -17.6, "f_max_sensor": 2.4,  "Hb": 2560},
        "nuScenes":      {"Vb": 32, "f_min_sensor": -30.0, "f_max_sensor": 10.0, "Hb": 1080},
    }

    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        self.info = {
            "train_loss": 0, "train_acc": 0, "train_iou": 0,
            "valid_loss": 0, "valid_acc": 0, "valid_iou": 0,
            "best_train_iou": 0, "best_val_iou": 0,
        }

        # ---- data ----------------------------------------------------------
        from dataset.kitti.parser import Parser
        self.parser = Parser(
            root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=None,
            labels=self.DATA["labels"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            max_points=self.ARCH["dataset"]["max_points"],
            batch_size=self.ARCH["train"]["batch_size"],
            workers=self.ARCH["train"]["workers"],
            gt=True,
            shuffle_train=True,
        )

        # ---- loss weights --------------------------------------------------
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)
        for x_cl, w in enumerate(self.loss_w):
            if DATA["learning_ignore"][x_cl]:
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        # ---- backbone model ------------------------------------------------
        with torch.no_grad():
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.model = HarDNet(
                    self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"]
                )
            elif self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.model = ResNet_34(
                    self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"]
                )
                if self.ARCH["train"]["act"] == "Hardswish":
                    self._convert_activation(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    self._convert_activation(self.model, nn.SiLU())
            elif self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.model = ResNet_34(
                    self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"]
                )
                if self.ARCH["train"]["act"] == "Hardswish":
                    self._convert_activation(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    self._convert_activation(self.model, nn.SiLU())

        # ---- DDFE modules --------------------------------------------------
        sensor_name = self.ARCH["dataset"].get("sensor_name", "SemanticKITTI")
        sensor_cfg = self.SENSOR_CONFIGS.get(
            sensor_name, self.SENSOR_CONFIGS["SemanticKITTI"]
        )

        proj_H = self.ARCH["dataset"]["sensor"]["img_prop"]["height"]
        proj_W = self.ARCH["dataset"]["sensor"]["img_prop"]["width"]

        self.beam_density_estimator = BeamDensityEstimator(
            proj_H=proj_H,
            proj_W=proj_W,
            f_min=-30.0,
            f_max=15.0,
            sensor_beam_config=sensor_cfg,
        )

        self.density_clipper = DensitySoftClipper(
            reservoir_size=self.ARCH["train"].get("density_reservoir", 1000),
            n_channels=4,   # 4 Gaussian scales
        )

        # The encoder feature dimension must match the backbone's output channels.
        # Adjust `encoder_channels` in ARCH if your backbone differs.
        encoder_ch = self.ARCH["train"].get("encoder_channels", 128)
        self.density_embedding = DensityAwareEmbedding(
            in_channels=encoder_ch,
            density_channels=4,
            out_channels=32,
        )

        # DDFE augmentation probability
        self.ddfe_aug_prob = self.ARCH["train"].get("ddfe_aug_prob", 0.5)

        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        ddfe_params = sum(
            p.numel()
            for m in [self.beam_density_estimator, self.density_embedding]
            for p in m.parameters()
            if p.requires_grad
        )
        print(f"Backbone parameters: {pytorch_total_params / 1e6:.3f} M")
        print(f"DDFE parameters:     {ddfe_params / 1e3:.1f} k  "
              f"(+{100 * ddfe_params / max(pytorch_total_params, 1):.2f}%)")

        self.tb_logger = SummaryWriter(log_dir=self.log, flush_secs=20)

        # ---- GPU setup -----------------------------------------------------
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
            self.beam_density_estimator.cuda()
            self.density_embedding.cuda()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model).cuda()
            self.model = convert_model(self.model).cuda()
            self.model_single = self.model.module
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()
            # DDFE modules are lightweight; keep them on a single GPU
            # (density maps are computed per-batch before scattering)

        # ---- losses --------------------------------------------------------
        self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        self.bd = BoundaryLoss().to(self.device)

        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()
            self.ls = nn.DataParallel(self.ls).cuda()

        # ---- optimiser & scheduler -----------------------------------------
        # Include DDFE embedding parameters in the optimiser.
        all_params = list(self.model.parameters()) + \
                     list(self.density_embedding.parameters())

        if self.ARCH["train"]["scheduler"] == "consine":
            length = self.parser.get_train_size()
            d = self.ARCH["train"]["consine"]
            self.optimizer = optim.SGD(
                all_params,
                lr=d["min_lr"],
                momentum=self.ARCH["train"]["momentum"],
                weight_decay=self.ARCH["train"]["w_decay"],
            )
            self.scheduler = CosineAnnealingWarmUpRestarts(
                optimizer=self.optimizer,
                T_0=d["first_cycle"] * length,
                T_mult=d["cycle"],
                eta_max=d["max_lr"],
                T_up=d["wup_epochs"] * length,
                gamma=d["gamma"],
            )
        else:
            self.optimizer = optim.SGD(
                all_params,
                lr=self.ARCH["train"]["decay"]["lr"],
                momentum=self.ARCH["train"]["momentum"],
                weight_decay=self.ARCH["train"]["w_decay"],
            )
            steps_per_epoch = self.parser.get_train_size()
            up_steps = int(
                self.ARCH["train"]["decay"]["wup_epochs"] * steps_per_epoch
            )
            final_decay = self.ARCH["train"]["decay"]["lr_decay"] ** (
                1 / steps_per_epoch
            )
            self.scheduler = warmupLR(
                optimizer=self.optimizer,
                lr=self.ARCH["train"]["decay"]["lr"],
                warmup_steps=up_steps,
                momentum=self.ARCH["train"]["momentum"],
                decay=final_decay,
            )

        # ---- optional weight loading ---------------------------------------
        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(
                path + "/SENet", map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(w_dict["state_dict"], strict=True)
            print("dict epoch:", w_dict["epoch"])
            print("info", w_dict["info"])

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _convert_activation(model, act):
        for child_name, child in model.named_children():
            if isinstance(child, nn.LeakyReLU):
                setattr(model, child_name, act)
            else:
                DDFETrainer._convert_activation(child, act)

    @staticmethod
    def _drop_intensity(in_vol: torch.Tensor) -> torch.Tensor:
        """
        DDFE Sec. 3.1: omit the intensity channel to improve domain
        generalisation.  Assumes channel layout [range, x, y, z, intensity, …];
        channel index 4 (intensity) is zeroed out.
        """
        out = in_vol.clone()
        if out.shape[1] > 4:
            out[:, 4, :, :] = 0.0   # zero intensity channel in-place
        return out

    @staticmethod
    def _to_spherical_input(in_vol: torch.Tensor) -> torch.Tensor:
        """
        DDFE Sec. 3.1: encode (x, y, z) channels as spherical coordinates
        (cos θ, sin θ, φ, r).

        Assumes channel layout: [r, x, y, z, …].
        Replaces channels 1-3 (x, y, z) with (cos θ, sin θ, φ).
        Channel 0 (r) is kept as-is.
        """
        out = in_vol.clone()
        r   = out[:, 0:1, :, :].clamp(min=1e-6)
        x   = out[:, 1:2, :, :]
        y   = out[:, 2:3, :, :]
        z   = out[:, 3:4, :, :]

        # azimuth θ and elevation φ
        theta = torch.atan2(y, x)          # azimuth
        phi   = torch.asin((z / r).clamp(-1, 1))  # elevation

        out[:, 1, :, :] = torch.cos(theta).squeeze(1)
        out[:, 2, :, :] = torch.sin(theta).squeeze(1)
        out[:, 3, :, :] = phi.squeeze(1)
        return out

    def _density_augmentation(self, in_vol: torch.Tensor) -> torch.Tensor:
        """
        DDFE Sec. 3.5: apply density augmentation with probability
        self.ddfe_aug_prob.

        Two techniques are combined (both applied independently):
        1. enhanced-Mix3D: mix two random range images in the batch with a
           small random translation and rotation (simulates a wider density
           range).
        2. Beam dropout: randomly zero out entire horizontal scan rows to
           simulate a sparser LiDAR sensor.
        """
        B, C, H, W = in_vol.shape
        out = in_vol.clone()

        for b in range(B):
            # ---- enhanced-Mix3D -------------------------------------------
            if np.random.rand() < self.ddfe_aug_prob:
                # pick a random partner in the batch
                partner = np.random.randint(0, B)
                # random horizontal translation (range-image column shift)
                shift = np.random.randint(-W // 4, W // 4)
                # random row flip to add rotational variation
                flip  = np.random.rand() < 0.5
                partner_vol = out[partner].clone()
                if shift != 0:
                    partner_vol = torch.roll(partner_vol, shift, dims=2)
                if flip:
                    partner_vol = torch.flip(partner_vol, dims=[2])
                # blend upper and lower halves (different density regions)
                split = H // 2
                out[b, :, :split, :] = partner_vol[:, :split, :]

            # ---- beam dropout (row masking) --------------------------------
            if np.random.rand() < self.ddfe_aug_prob:
                # drop 10-30 % of scan rows
                n_drop = int(np.random.uniform(0.1, 0.3) * H)
                drop_rows = np.random.choice(H, n_drop, replace=False)
                out[b, :, drop_rows, :] = 0.0

        return out

    def calculate_estimate(self, epoch, iter):
        estimate = int(
            (self.data_time_t.avg + self.batch_time_t.avg)
            * (
                self.parser.get_train_size() * self.ARCH["train"]["max_epochs"]
                - (iter + 1 + epoch * self.parser.get_train_size())
            )
        ) + int(
            self.batch_time_e.avg
            * self.parser.get_valid_size()
            * (self.ARCH["train"]["max_epochs"] - epoch)
        )
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        sm = plt.cm.ScalarMappable(cmap=cmap)
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        depth = (
            cv2.normalize(
                depth, None, alpha=0, beta=1,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
            ) * 255.0
        ).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, DDFETrainer.get_mpl_colormap("viridis")
        ) * mask[..., None]
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return out_img.astype(np.uint8)

    # -----------------------------------------------------------------------
    # Public training API
    # -----------------------------------------------------------------------

    def train(self, epochs=None):
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")

        self.evaluator = iouEval(
            self.parser.get_n_classes(), self.device, self.ignore_class
        )

        if self.path is not None:
            acc, iou, loss, _ = self.validate(
                val_loader=self.parser.get_valid_set(),
                model=self.model,
                criterion=self.criterion,
                evaluator=self.evaluator,
                class_func=self.parser.get_xentropy_class_string,
                color_fn=self.parser.to_color,
                save_scans=self.ARCH["train"]["save_scans"],
            )

        max_epochs = epochs if epochs is not None else self.ARCH["train"]["max_epochs"]

        for epoch in range(self.epoch, max_epochs):
            acc, iou, loss = self.train_epoch(
                train_loader=self.parser.get_train_set(),
                model=self.model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                epoch=epoch,
                evaluator=self.evaluator,
                scheduler=self.scheduler,
                color_fn=self.parser.to_color,
                report=self.ARCH["train"]["report_batch"],
                show_scans=self.ARCH["train"]["show_scans"],
            )

            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou

            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "density_embedding": self.density_embedding.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "info": self.info,
                "scheduler": self.scheduler.state_dict(),
                "density_clipper_m": self.density_clipper.m.tolist(),
                "density_clipper_l": self.density_clipper.l.tolist(),
            }
            save_checkpoint(state, self.log, suffix="")

            if self.info["train_iou"] > self.info["best_train_iou"]:
                print("Best mean iou in training set so far, save model!")
                self.info["best_train_iou"] = self.info["train_iou"]
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                print("*" * 80)
                acc, iou, loss, _ = self.validate(
                    val_loader=self.parser.get_valid_set(),
                    model=self.model,
                    criterion=self.criterion,
                    evaluator=self.evaluator,
                    class_func=self.parser.get_xentropy_class_string,
                    color_fn=self.parser.to_color,
                    save_scans=self.ARCH["train"]["save_scans"],
                )
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou

            if self.info["valid_iou"] > self.info["best_val_iou"]:
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info["best_val_iou"] = self.info["valid_iou"]
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

        print("Finished Training")

    def train_epoch(
        self, train_loader, model, criterion, optimizer, epoch,
        evaluator, scheduler, color_fn, report=10, show_scans=False,
    ):
        losses = AverageMeter()
        acc    = AverageMeter()
        iou    = AverageMeter()
        bd_m   = AverageMeter()
        train_time = []

        if self.gpu:
            torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler("cuda")
        model.train()
        self.density_embedding.train()

        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                _, _, _, _, _, _, _, _, _) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            self.data_time_t.update(time.time() - end)
            in_vol = self._drop_intensity(in_vol)
            in_vol = self._to_spherical_input(in_vol)
            in_vol = self._density_augmentation(in_vol)

            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()

            with torch.no_grad():
                density = self.beam_density_estimator(in_vol)   # [B, 4, H, W]

            # Update reservoir statistics & clip
            self.density_clipper.update(density)
            density_c = self.density_clipper.clip(density)      # [B, 4, H, W]

            start = time.time()

            with torch.amp.autocast("cuda"):
                if self.ARCH["train"]["aux_loss"]:
                    output, aux_list, z8 = model(in_vol)
                else:
                    output = model(in_vol)
                density_feat = self.density_embedding(output, density_c)

                if not hasattr(self, "_ddfe_head") or \
                        self._ddfe_head_in != density_feat.shape[1]:
                    self._ddfe_head_in = density_feat.shape[1]
                    self._ddfe_head = nn.Conv2d(
                        density_feat.shape[1],
                        output.shape[1],
                        kernel_size=1,
                    ).to(self.device)
                    nn.init.zeros_(self._ddfe_head.weight)
                    nn.init.zeros_(self._ddfe_head.bias)

                output = output + self._ddfe_head(density_feat)

                if self.ARCH["train"]["aux_loss"]:
                    lamda = self.ARCH["train"]["lamda"]
                    bdlosss = (self.bd(output, proj_labels.long()) + lamda * self.bd(z8, proj_labels.long()))
                    loss_m0 = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output, proj_labels.long())

                    loss_m8 = criterion(torch.log(z8.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z8, proj_labels.long())
                    loss_m = loss_m0 + lamda * loss_m8 + bdlosss
                else:
                    bdlosss = self.bd(output, proj_labels.long())
                    loss_m = (
                        criterion(torch.log(output.clamp(min=1e-8)), proj_labels)
                        + 1.5 * self.ls(output, proj_labels.long())
                        + bdlosss
                    )

            optimizer.zero_grad()
            loss = loss_m.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, _ = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            bd_m.update(bdlosss.mean().item(), in_vol.size(0))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - start
            train_time.append(res)
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            for g in self.optimizer.param_groups:
                lr = g["lr"]

            scheduler.step()

        print(
            "Mean CNN training time: {:.4f}s\tstd: {:.4f}s".format(
                np.mean(train_time), np.std(train_time)
            )
        )
        return acc.avg, iou.avg, losses.avg

    def validate(
        self, val_loader, model, criterion, evaluator, class_func,
        color_fn, save_scans,
    ):
        losses  = AverageMeter()
        jaccs   = AverageMeter()
        wces    = AverageMeter()
        acc     = AverageMeter()
        iou     = AverageMeter()
        rand_imgs = []
        validation_time = []

        model.eval()
        self.density_embedding.eval()
        evaluator.reset()

        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                    _, _, _, _, _, _, _, _, _) in tqdm(
                enumerate(val_loader), total=len(val_loader)
            ):
                # DDFE pre-processing (NO augmentation at validation time)
                in_vol = self._drop_intensity(in_vol)
                in_vol = self._to_spherical_input(in_vol)

                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # Density estimation + clipping (no reservoir update)
                density = self.beam_density_estimator(in_vol)
                density_c = self.density_clipper.clip(density)

                start = time.time()

                if self.ARCH["train"]["aux_loss"]:
                    output_list = model(in_vol)
                    output = output_list[0]
                else:
                    output = model(in_vol)

                # Apply DDFE density-aware embedding
                if hasattr(self, "_ddfe_head"):
                    density_feat = self.density_embedding(output, density_c)
                    output = output + self._ddfe_head(density_feat)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                validation_time.append(time.time() - start)

                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce  = criterion(log_out, proj_labels)
                loss = wce + jacc

                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))
                wces.update(wce.mean().item(), in_vol.size(0))

                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc

        print(
            "Mean validation time: {:.4f}s\tstd: {:.4f}s".format(
                np.mean(validation_time), np.std(validation_time)
            )
        )
        return acc.avg, iou.avg, losses.avg, rand_imgs