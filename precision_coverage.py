import sys
import torch
import torch.nn.functional as F
import yaml
from dataset.kitti.parser import Parser
from modules.HDC_utils import set_knn_model

DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
KITTIC_DIR = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C"
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS = "config/labels/semantic-kitti-all.yaml"
PRETRAINED = "logs/kitti_pretrain/hdc_sub.pth"
BANK_PATH = "/mnt/alpha/jmfleming/knn_bank.pt"
NUM_CLASSES = 17

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = yaml.safe_load(open(CONFIG_ARCH))
    DATA = yaml.safe_load(open(CONFIG_LABELS))
    
    print("Loading Model...")
    model = set_knn_model(ARCH, "logs/kitti_pretrain", 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(PRETRAINED, map_location=device), strict=False)
    model.to(device)
    
    print(f"Loading Bank from {BANK_PATH}...")
    model.bank = torch.load(BANK_PATH, map_location=device)
    
    corruptions = ["snow/heavy", "fog/heavy", "wet_ground/heavy"]
    coverages = [0.10, 0.25, 0.50, 0.75]
    num_frames = 20
    
    for corr in corruptions:
        tgt_dir = f"{KITTIC_DIR}/{corr}"
        print(f"\n==========================================")
        print(f"Evaluating Precision-Coverage on {corr}")
        print(f"==========================================")
        
        tgt_parser = Parser(
            root=tgt_dir, train_sequences=DATA["split"]["valid"],
            valid_sequences=DATA["split"]["valid"], test_sequences=None,
            labels=DATA["labels"], color_map=DATA.get("color_map", {}),
            learning_map=DATA["learning_map"], learning_map_inv=DATA["learning_map_inv"],
            sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
            batch_size=1, workers=4, gt=True, shuffle_train=False
        )
        
        loader = tgt_parser.validloader
        
        stats = {
            cov: {"knn_correct": 0, "knn_total": 0, "proto_correct": 0, "proto_total": 0}
            for cov in coverages
        }
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_frames:
                    break
                    
                proj_in = batch[0].to(device)
                proj_labels = batch[2].to(device).view(-1)
                
                enc, _, _ = model.encode(proj_in)
                valid = torch.any(proj_in.permute(0, 2, 3, 1).contiguous().reshape(-1, proj_in.shape[1]) != 0, dim=1)
                if not torch.any(valid): continue
                
                enc_norm = F.normalize(enc[valid], dim=1).to(model.classify.weight.dtype)
                
                logits = model.classify(enc_norm)
                preds = logits.argmax(dim=1)
                true_labels = proj_labels.view(-1)[valid]
                correct = (preds == true_labels)
                
                # KNN Confidences (higher is better)
                knn_conf = model.get_confidence(enc_norm, preds)
                
                # Prototype Similarities (higher is better)
                normalized_prototypes = F.normalize(model.classify.weight, dim=1)
                sims = F.linear(enc_norm, normalized_prototypes)
                proto_sims, _ = sims.max(dim=1)
                
                for cov in coverages:
                    # Per-frame quantile thresholding
                    # If we want top `cov` fraction, we take the `(1 - cov)` quantile.
                    # e.g., cov=0.25 (top 25%) -> quantile 0.75
                    q_val = 1.0 - cov
                    
                    k_thresh = torch.quantile(knn_conf.float(), q_val).item()
                    k_admit = knn_conf > k_thresh
                    
                    p_thresh = torch.quantile(proto_sims.float(), q_val).item()
                    p_admit = proto_sims > p_thresh
                    
                    stats[cov]["knn_correct"] += correct[k_admit].sum().item()
                    stats[cov]["knn_total"] += k_admit.sum().item()
                    
                    stats[cov]["proto_correct"] += correct[p_admit].sum().item()
                    stats[cov]["proto_total"] += p_admit.sum().item()
                    
        print(f"{'Coverage':<10} | {'KNN Precision':<15} | {'Proto Precision':<15}")
        print("-" * 45)
        for cov in coverages:
            k_prec = (stats[cov]["knn_correct"] / max(1, stats[cov]["knn_total"])) * 100
            p_prec = (stats[cov]["proto_correct"] / max(1, stats[cov]["proto_total"])) * 100
            print(f"{cov*100:>5.1f}%     | {k_prec:>13.1f}% | {p_prec:>13.1f}%")

if __name__ == "__main__":
    main()
