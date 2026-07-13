import json
import math
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.kitti.parser import Parser
from modules.HDC_utils import KNNModel

import os

DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
MODEL_DIR = "logs/kitti_pretrain"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

CORRUPTIONS = ["fog", "snow", "motion", "beam", "crosstalk", "echo", "cross_sensor"]
SEVERITY = 3
RANKS_BY_MODE = {
    "ellipsoid": (0, 4, 8, 16, 32, 64, 128, 256),
    "residual":  (0, 4, 8, 16, 32, 64, 128, 256),
    "anti":      (0, 4, 8, 16, 32, 64, 128, 256),
    "subspace":  (1, 2, 4, 8, 16, 32, 64, 128, 256),
}
COVERAGE = 0.90
MAX_PER_CLASS = 5000
RADIUS_QUANTILE = 0.99
OUT = "rank_sweep.json"

@torch.no_grad()
def score_subspace(H, mu, V, d, mode="ellipsoid"):
    delta = H.float() - mu
    sq = delta.pow(2).sum(dim=1)
    if V.shape[1] == 0:
        return sq.clamp_min(0).sqrt()
    proj_sq = (delta @ V).pow(2).sum(dim=1)

    if mode == "ellipsoid":      # paper: shrink high-var dirs
        out = sq - (1.0 - 1.0 / d) * proj_sq
    elif mode == "subspace":     # score ONLY in the high-var subspace
        out = proj_sq
    elif mode == "residual":     # score ONLY orthogonal to it (the r->limit of above)
        out = sq - proj_sq
    elif mode == "anti":         # AMPLIFY the high-var dirs
        out = sq + 9.0 * proj_sq
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return out.clamp_min(0).sqrt()

@torch.no_grad()
def fit_ellipsoid(Y, d, rank, coverage=COVERAGE, mode="ellipsoid"):
    """Fit to source hypervectors Y of ONE class. rank=0 -> ball at the centroid."""
    Y = Y.float()
    n = Y.shape[0]
    mu = Y.mean(dim=0)
    delta = Y - mu

    if rank == 0:
        V = torch.zeros(d, 0, device=Y.device)
    else:
        q = max(1, int(min(rank, n - 1, d)))
        _, S, Vfull = torch.pca_lowrank(delta, q=q, center=False, niter=4)
        V = Vfull[:, :rank].contiguous()

    R = torch.quantile(score_subspace(Y, mu, V, d, mode=mode), coverage).item()
    return {"mu": mu, "V": V, "R": R, "r": V.shape[1], "d": d}

def log_volume(e):
    """log vol(E), up to the constant log(c_d) shared by every set.
    vol = c_d * R^d * det(M)^{1/2}, and log det(M) = r * log(d), since high-variance
    dirs contribute log(d) each and low-variance dirs contribute log(1) = 0."""
    return e["d"] * math.log(max(e["R"], 1e-12)) + 0.5 * e["r"] * math.log(e["d"])

@torch.no_grad()
def collect_source(model, loader, device, max_per_class=MAX_PER_CLASS):
    buckets = {c: [] for c in range(model.num_classes)}
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device).view(-1)
        enc, idx, _ = model.encode(x)
        h = F.normalize(enc)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any():
            continue
        h, lab = h[v], lab[v]
        for c in lab.unique().tolist():
            if counts[c] >= max_per_class:
                continue
            hc = h[lab == c]
            take = min(hc.shape[0], max_per_class - counts[c])
            buckets[c].append(hc[:take].cpu())
            counts[c] += take
        if all(counts[c] >= max_per_class for c in range(model.num_classes)):
            break
    return {c: torch.cat(t) for c, t in buckets.items() if t}

@torch.no_grad()
def collect_target(model, loader, device, max_per_class=MAX_PER_CLASS):
    """(H, preds, correct) for one corrupted stream, using the frozen model."""
    H, P, C = [], [], []
    protos = F.normalize(model.classify.weight)
    counts = {c: 0 for c in range(model.num_classes)}
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device).view(-1)
        if x.shape[1] == 0: continue
        enc, idx, _ = model.encode(x)
        h = F.normalize(enc)
        lab = y[idx] if idx is not None else y
        v = (lab >= 0) & (lab < model.num_classes)
        if not v.any():
            continue
        h, lab = h[v], lab[v]
        preds = (h.to(protos.dtype) @ protos.T).argmax(dim=1)
        
        for c in lab.unique().tolist():
            if counts[c] >= max_per_class:
                continue
            mask = (lab == c)
            hc = h[mask]
            predc = preds[mask]
            
            take = min(hc.shape[0], max_per_class - counts[c])
            H.append(hc[:take].cpu())
            P.append(predc[:take].cpu())
            C.append((predc[:take] == c).cpu())
            counts[c] += take
            
        # We can stop if we have collected enough points for all valid classes
        # Ignoring class 0 (unlabeled) which might not be used
        if all(counts[c] >= max_per_class for c in range(1, 17)):
            break
            
    if not H:
        return torch.zeros(0, model.hd_dim), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.bool)
        
    return torch.cat(H), torch.cat(P), torch.cat(C)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))

    parser = Parser(
        root=DATA_DIR,
        train_sequences=DATA["split"]["train"],
        valid_sequences=DATA["split"]["valid"],
        test_sequences=None,
        labels=DATA["labels"], color_map=DATA.get("color_map", {}),
        learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
        batch_size=1, workers=0, gt=True, shuffle_train=False,
    )
    clean_ds = parser.validloader.dataset

    print(f"Loading pretrained model from {PRETRAINED}...")
    model = KNNModel(ARCH, MODEL_DIR, "rp", 0, 0, NUM_CLASSES, device)
    sd = torch.load(PRETRAINED, map_location=device, weights_only=False)
    sd = sd.state_dict() if isinstance(sd, torch.nn.Module) else sd
    if "subclusters" in sd and hasattr(model, "subclusters"):
        n_sub = sd["subclusters"].shape[0]
        if model.subclusters.shape[0] != n_sub:
            model.subclusters = torch.nn.Parameter(
                torch.zeros(n_sub, model.hd_dim, device=device))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    d = model.hd_dim
    print("Model loaded successfully.")

    print("\nCollecting source (clean) hypervectors...")
    src = collect_source(model, DataLoader(clean_ds, batch_size=1, num_workers=0), device)
    src = {c: v.to(device) for c, v in src.items()}
    print(f"  classes with data: {sorted(src.keys())}")

    tgt = {}
    for cond in CORRUPTIONS:
        print(f"Collecting target hypervectors: {cond} sev {SEVERITY}...")
        try:
            SEVERITY_MAP = {1: 'light', 3: 'moderate', 5: 'heavy'}
            sev_str = SEVERITY_MAP.get(SEVERITY, 'moderate')
            corruption_root = os.path.join(KITTIC_DIR, cond, sev_str)
            seq_dir = os.path.join(corruption_root, "sequences")
            seq_08 = os.path.join(seq_dir, "08")
            if not os.path.exists(seq_08):
                os.makedirs(seq_dir, exist_ok=True)
                if not os.path.islink(seq_08):
                    os.symlink("..", seq_08)
            
            parser_obj = Parser(
                root=corruption_root,
                train_sequences=DATA["split"]["valid"],
                valid_sequences=DATA["split"]["valid"],
                test_sequences=None,
                labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
                sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
                batch_size=1, workers=0, gt=True, shuffle_train=False,
            )
            ld = parser_obj.get_valid_set()
            H, P, C = collect_target(model, ld, device)
            acc = C.float().mean().item()
            print(f"  n={len(C)}  pseudo-label acc={acc:.3f}")
            if acc < 0.2:
                print(f"  !! WARNING: accuracy is {acc:.3f}, suspiciously low "
                      f"(possible label misalignment). Excluding it from evaluation.")
                continue
            tgt[cond] = (H, P, C)
        except Exception as e:
            print(f"  SKIPPED ({type(e).__name__}: {e})")

    results = {}
    modes = ["ellipsoid", "subspace", "residual", "anti"]
    
    for mode in modes:
        print("\n" + "=" * 78)
        print(f"MODE: {mode.upper()}")
        print(f"{'rank':>5} {'meanAUROC':>10} {'log_vol':>11}   per-corruption AUROC")
        print("=" * 78)
        
        results[mode] = {}
        for r in RANKS_BY_MODE[mode]:
            ells = {c: fit_ellipsoid(Y, d, rank=r, mode=mode) for c, Y in src.items()}
            mlv = float(np.mean([log_volume(e) for e in ells.values()]))

            per = {}
            per_classes = {}
            for cond, (H, P, C) in tgt.items():
                Hd, Pd = H.to(device), P.to(device)
                s = torch.full((Hd.shape[0],), -1e9, device=device)
                for c in Pd.unique().tolist():
                    if c not in ells:
                        continue
                    m = Pd == c
                    e = ells[c]

                    s[m] = -score_subspace(Hd[m], e["mu"], e["V"], d, mode=mode) / max(e["R"], 1e-8)
                
                # Compute per-class AUROC, then average
                correct_all = C.numpy().astype(int)
                scores_all = s.cpu().numpy()
                preds_all = Pd.cpu().numpy()
                
                auroc_list = []
                for c in np.unique(preds_all):
                    if c not in ells:
                        continue
                    mask_c = preds_all == c
                    if mask_c.sum() < 50:
                        continue
                    correct_c = correct_all[mask_c]
                    scores_c = scores_all[mask_c]
                    if len(np.unique(correct_c)) == 2:
                        auroc_list.append(roc_auc_score(correct_c, scores_c))
                
                if not auroc_list:
                    continue
                per[cond] = float(np.mean(auroc_list))
                per_classes[cond] = len(auroc_list)

            mean_auroc = float(np.mean(list(per.values()))) if per else float("nan")
            results[mode][r] = {"mean_auroc": mean_auroc, "mean_log_volume": mlv,
                          "per_corruption": per}

            tag = "   <-- BALL BASELINE" if r == 0 else ""
            detail = " ".join(f"{k[:8]}={v:.3f}({per_classes[k]}c)" for k, v in per.items())
            print(f"{r:>5} {mean_auroc:>10.4f} {mlv:>11.1f}   {detail}{tag}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")

    print("\n" + "=" * 78)
    print("SUMMARY OF RESULTS")
    print("=" * 78)
    for mode in modes:
        mode_results = results[mode]
        base = results["ellipsoid"][0]["mean_auroc"]
        best_r = max(mode_results, key=lambda k: mode_results[k]["mean_auroc"])
        best = mode_results[best_r]["mean_auroc"]
        delta = best - base
        print(f"[{mode.upper():<9}] ball(r=0): {base:.4f}  |  best(r={best_r:<3}): {best:.4f}  |  diff: {delta:+.4f}")

if __name__ == "__main__":
    main()