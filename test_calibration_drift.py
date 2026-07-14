import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_knn_model

KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
NUM_CLASSES = 17

# fog is EXCLUDED: precision is 0-6% at every coverage, i.e. below chance for 17
# classes. That is a label-misalignment pathology, not a hard corruption, and no
# conclusion can be drawn from it until the ground truth is fixed.
CORRUPTIONS = ["snow", "wet_ground", "motion_blur", "beam_missing", "cross_sensor"]
SEVERITIES = ["light", "moderate", "heavy"]

TARGET_COVERAGE = 0.50      # admit the top 50% most conformal points
TARGET_SETSIZE = 1.0        # E[|C(x)|] we drive toward in the label-free ACI
GAMMA = 0.05                # ACI step size
LR = 0.01
N_FRAMES = 300
OUT = "calibration_drift.json"


# ------------------------------------------------------------------ the gates

class StaticGate:
    """Source-calibrated fixed threshold. This is what ConformalHDC does."""
    def __init__(self, q):
        self.q = q
    def threshold(self, sims, preds):
        return self.q
    def update(self, sims, preds, correct=None):
        pass


class QuantileGate:
    """Per-frame quantile. Admits the top `cov` fraction of THIS frame.

    Adaptive and label-free, but note it maintains a fixed ADMISSION RATE, not a
    coverage guarantee. It cannot collapse and cannot blow open -- but it also cannot
    tell you that a frame is entirely OOD (it will still dutifully admit its "best" 50%).
    That is the weakness aci_setsize is meant to fix.
    """
    def __init__(self, cov):
        self.cov = cov
    def threshold(self, sims, preds):
        s = sims.gather(1, preds.unsqueeze(1)).squeeze(1)
        return torch.quantile(s.float(), 1.0 - self.cov).item()
    def update(self, sims, preds, correct=None):
        pass


class ACISetSizeGate:
    """THE PROPOSED METHOD. Label-free ACI driven by prediction-set size.

    C(x) = { y : sim(x,y) >= q }.  We drive q so that E[|C(x)|] -> 1:
        mean |C| > 1  -> too loose  -> raise q
        mean |C| < 1  -> too tight  -> lower q
    No ground truth anywhere. The set size IS the miscoverage surrogate.
    """
    def __init__(self, q0, target=TARGET_SETSIZE, gamma=GAMMA):
        self.q = q0
        self.target = target
        self.gamma = gamma
        self.hist = []
    def threshold(self, sims, preds):
        return self.q
    def update(self, sims, preds, correct=None):
        set_size = (sims >= self.q).sum(dim=1).float().mean().item()
        self.hist.append(set_size)
        # error > 0 means sets are too BIG -> tighten (raise q)
        err = set_size - self.target
        self.q = float(np.clip(self.q + self.gamma * err * 0.1, -1.0, 1.0))


class ACIOracleGate:
    """UPPER BOUND ONLY -- uses ground-truth miscoverage. NOT DEPLOYABLE.

    This is standard ACI (Gibbs & Candes). It exists in this experiment purely to show
    the ceiling that the label-free surrogate is chasing. If aci_setsize lands close to
    aci_oracle, the surrogate is a good one. Never report this as a method.
    """
    def __init__(self, q0, alpha=1.0 - TARGET_COVERAGE, gamma=GAMMA):
        self.q = q0
        self.alpha = alpha
        self.gamma = gamma
    def threshold(self, sims, preds):
        return self.q
    def update(self, sims, preds, correct=None):
        if correct is None:
            return
        s = sims.gather(1, preds.unsqueeze(1)).squeeze(1)
        admitted = s >= self.q
        # miscoverage among ADMITTED points: how often did we admit a wrong label?
        err = 1.0 - correct[admitted].float().mean().item() if admitted.any() else 1.0
        self.q = float(np.clip(self.q + self.gamma * (err - self.alpha) * 0.1, -1.0, 1.0))


# ------------------------------------------------------------------ runner

@torch.no_grad()
def run_arm(model, loader, device, arm, gate, src_proto, n_frames=N_FRAMES):
    """Stream through the corruption, adapting. Returns per-frame trajectories."""
    model.classify.weight.data = src_proto.clone()      # reset prototypes

    firing, precision, setsize, thresh_hist = [], [], [], []

    for i, batch in enumerate(loader):
        if i >= n_frames:
            break
        x = batch[0].to(device)
        y = batch[2].to(device).view(-1)
        if x.shape[1] == 0:
            continue

        enc, _, _ = model.encode(x)
        valid = torch.any(
            x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1]) != 0, dim=1)
        if not valid.any():
            continue

        h = F.normalize(enc[valid], dim=1).to(model.classify.weight.dtype)
        protos = F.normalize(model.classify.weight, dim=1)
        sims = h @ protos.T
        preds = sims.argmax(dim=1)
        labels = y[valid]
        correct = preds == labels

        if arm == "frozen":
            firing.append(0.0)
            precision.append(float("nan"))
            setsize.append(float("nan"))
            thresh_hist.append(float("nan"))
            continue

        if arm == "ungated":
            admit = torch.ones_like(preds, dtype=torch.bool)
            q = float("nan")
        else:
            q = gate.threshold(sims, preds)
            s = sims.gather(1, preds.unsqueeze(1)).squeeze(1)
            admit = s >= q

        firing.append(admit.float().mean().item())
        precision.append(correct[admit].float().mean().item() if admit.any()
                         else float("nan"))
        setsize.append((sims >= q).sum(dim=1).float().mean().item()
                       if not np.isnan(q) else float("nan"))
        thresh_hist.append(q)

        # ---- the prototype update (identical across arms; only the GATE differs) ----
        if admit.any():
            for c in preds[admit].unique().tolist():
                m = (preds == c) & admit
                pull = h[m].mean(dim=0)
                w = model.classify.weight[c] + LR * pull
                model.classify.weight[c] = F.normalize(w.unsqueeze(0), dim=1).squeeze(0)

        if arm not in ("ungated",):
            gate.update(sims, preds, correct)

    return {"firing": firing, "precision": precision,
            "setsize": setsize, "threshold": thresh_hist}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))

    model = set_knn_model(ARCH, "logs/kitti_pretrain", "rp", 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(PRETRAINED, map_location=device), strict=False)
    model.to(device).eval()
    src_proto = model.classify.weight.data.clone()

    # ---- calibrate the STATIC threshold on CLEAN SOURCE data ------------------
    # This is exactly what ConformalHDC would do, and it is the thing under test.
    print("Calibrating static threshold on clean source...")
    p = Parser(root=DATA_DIR, train_sequences=DATA["split"]["train"],
               valid_sequences=DATA["split"]["valid"], test_sequences=None,
               labels=DATA["labels"], color_map=DATA.get("color_map", {}),
               learning_map=DATA["learning_map"],
               learning_map_inv=DATA["learning_map_inv"],
               sensor=ARCH["dataset"]["sensor"],
               max_points=ARCH["dataset"]["max_points"],
               batch_size=1, workers=0, gt=True, shuffle_train=False)
    src_scores = []
    with torch.no_grad():
        for i, batch in enumerate(p.validloader):
            if i >= 50:
                break
            x = batch[0].to(device)
            if x.shape[1] == 0:
                continue
            enc, _, _ = model.encode(x)
        valid = torch.any(
            x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1]) != 0, dim=1)
        h = F.normalize(enc[valid], dim=1).to(model.classify.weight.dtype)
        sims = h @ F.normalize(model.classify.weight, dim=1).T
        src_scores.append(sims.max(dim=1).values.cpu())
    src_scores = torch.cat(src_scores)
    Q_STATIC = torch.quantile(src_scores.float(), 1.0 - TARGET_COVERAGE).item()
    print(f"  static threshold (source {int(TARGET_COVERAGE*100)}% coverage) = "
          f"{Q_STATIC:.4f}")
    print(f"  source similarity range: min={src_scores.min():.3f} "
          f"median={src_scores.median():.3f} max={src_scores.max():.3f}")

    results = {}
    for corr in CORRUPTIONS:
        for sev in SEVERITIES:
            root = os.path.join(KITTIC_DIR, corr, sev)
            s08 = os.path.join(root, "sequences", "08")
            if not os.path.exists(s08):
                os.makedirs(os.path.dirname(s08), exist_ok=True)
                try:
                    os.symlink("..", s08)
                except FileExistsError:
                    pass
            try:
                pp = Parser(root=root, train_sequences=DATA["split"]["valid"],
                            valid_sequences=DATA["split"]["valid"], test_sequences=None,
                            labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=1, workers=0, gt=True, shuffle_train=False)
                loader = pp.validloader
            except Exception as e:
                print(f"skip {corr}/{sev}: {e}")
                continue

            key = f"{corr}/{sev}"
            results[key] = {}
            print(f"\n=== {key} ===")
            print(f"{'arm':>12} {'fire%':>8} {'prec%':>8} {'|C|':>6} {'q_end':>8}")

            arms = {
                "frozen":      None,
                "ungated":     None,
                "static":      StaticGate(Q_STATIC),
                "quantile":    QuantileGate(TARGET_COVERAGE),
                "aci_setsize": ACISetSizeGate(Q_STATIC),
                "aci_oracle":  ACIOracleGate(Q_STATIC),
            }
            for arm, gate in arms.items():
                tr = run_arm(model, loader, device, arm, gate, src_proto)
                results[key][arm] = tr
                f = np.nanmean(tr["firing"]) * 100
                pr = np.nanmean(tr["precision"]) * 100
                ss = np.nanmean(tr["setsize"])
                qe = tr["threshold"][-1] if tr["threshold"] else float("nan")
                print(f"{arm:>12} {f:>8.1f} {pr:>8.1f} {ss:>6.2f} {qe:>8.3f}")

                # THE HEADLINE CHECK, printed inline so it cannot be missed
                if arm == "static":
                    early = np.nanmean(tr["firing"][:20]) * 100
                    late = np.nanmean(tr["firing"][-20:]) * 100
                    print(f"{'':>12}   static firing: first20={early:.1f}%  "
                          f"last20={late:.1f}%   <-- COLLAPSE?")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")

    print("""
================================================================================
HOW TO READ THIS -- and what each outcome means for the paper
================================================================================
1. static firing rate << 50% (the requested coverage), and FALLING with severity
   -> THE CLAIM IS CONFIRMED. Static conformal calibration is vacuous under shift.
      This is the refutation of ConformalHDC-for-TTA, and it is your premise.
      The severity axis is the key evidence: firing should fall monotonically
      light -> moderate -> heavy.

2. static firing rate ~= 50% and stable
   -> THE CLAIM IS FALSE. The shift does not move the score distribution enough to
      decalibrate a static threshold, and the paper has no premise. STOP and rethink
      before writing anything.

3. aci_setsize holds its firing rate AND its precision where static collapses
   -> the label-free mechanism works. This is the contribution.

4. aci_setsize ~= aci_oracle
   -> the set-size surrogate is as good as true-miscoverage feedback. This is the
      strongest possible result: you recovered ACI's behaviour WITHOUT labels.

5. aci_setsize ~= quantile
   -> the ACI machinery buys nothing over a per-frame quantile. Then the honest paper
      is "per-frame quantile calibration", which is simpler and still correct -- but
      you must NOT dress it up as ACI. Say what it is.

6. ungated collapses (low precision, mIoU tanks)
   -> confirms that gating is load-bearing at all, which the earlier TTA disaster
      already suggested.

NOTE ON aci_oracle: it uses ground-truth labels. It is a CEILING, not a method.
Never report it as deployable. It is in the table only to show what the label-free
surrogate is chasing.
""")


if __name__ == "__main__":
    main()
