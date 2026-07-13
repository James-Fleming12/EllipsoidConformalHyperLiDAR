import numpy as np
import torch
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import Model, EllipsoidModel
from modules.trainer import DGLSSTrainer, Trainer
from modules.Basic_HD import BasicHD, EllipsoidTrainer
from modules.ioueval import iouEval

from dataset.export_semantickitti import KittiConverter

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_HDC_EPOCHS = 10
FEATURE_EXTRACTOR_EPOCHS = 80

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

def convert_dataset():
    converter = KittiConverter(
        nusc_dir=NU_DATA_DIR,
        nusc_skitti_dir=DATA_DIR,
        lidar_name='LIDAR_TOP',
        nusc_version='v1.0-trainval'
    )

    converter.nuscenes_gt_to_semantickitti()

    print("Conversion Complete: Output Saved to ")

def train_extractor(ARCH, DATA, epochs=FEATURE_EXTRACTOR_EPOCHS, data_dir=None, return_trainer=False, resume_path=None):
    trainer = Trainer(ARCH, DATA, data_dir if data_dir else DATA_DIR, LOG_DIR, path=resume_path) # saves in "/logs/SENet_..."
    trainer.train(epochs=epochs)

    if return_trainer:
        return trainer

def train_dglss(ARCH, DATA, dist_type="standard", epochs=FEATURE_EXTRACTOR_EPOCHS):
    trainer = DGLSSTrainer(ARCH, DATA, DATA_DIR, LOG_DIR, dist_type=dist_type) # saves in "/logs/SENet_..."
    trainer.train(epochs=epochs)

def train_hdc(ARCH, DATA, epochs=MAX_HDC_EPOCHS, data_dir=None, return_extractor=False) -> EllipsoidModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=data_dir if data_dir else DATA_DIR,
                        train_sequences=DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        labels=DATA["labels"],
                        color_map=DATA["color_map"],
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=ARCH["train"]["batch_size"],
                        workers=ARCH["train"]["workers"],
                        gt=True,
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()
    val_loader = parser.get_valid_set() # val_loader is empty???

    ignore = []
    for cl, ign in DATA['learning_ignore'].items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)

    trainer = EllipsoidTrainer(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR, None)

    trainer.train(dataloader, trainer.model, None)

    for i in range(epochs - 1):
        trainer.retrain(dataloader, trainer.model, i+1, None)
        # Save checkpoint after each epoch so training can be picked up if interrupted
        torch.save(trainer.model, HDC_SAVE_PATH)

    model: EllipsoidModel = trainer.model
    torch.save(model, HDC_SAVE_PATH)

    if return_extractor: return model, trainer

    return model

def test_hdc_model(model, dataloader, return_detailed=False) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_accuracies = []
    class_correct = torch.zeros(model.num_classes, device=device)
    class_total = torch.zeros(model.num_classes, device=device)

    class_intersection = torch.zeros(model.num_classes, device=device)
    class_union = torch.zeros(model.num_classes, device=device)
    
    global_correct = 0
    global_total = 0
    model.eval()
    
    with torch.no_grad():
        for _, batch_data in enumerate(dataloader):
            proj_in = batch_data[0].to(device)
            proj_labels = batch_data[2].to(device)
            logits, _, indices, _ = model(proj_in, PERCENTAGE=None, is_wrong=None)

            # top_two = torch.topk(logits, k=2, dim=1).values
            # margin = top_two[:, 0] - top_two[:, 1]
            # print(f"Mean Confidence Margin: {margin.mean().item()}")

            # mask = torch.ones_like(logits, dtype=torch.bool)
            # mask.scatter_(1, logits.argmax(1, keepdim=True), False)
            # rcv = logits[mask].view(logits.size(0), -1).var(dim=1)
            # print(f"Residual Class Variance: {rcv.mean().item()}")
            
            predictions = torch.argmax(logits, dim=1)
            proj_labels_flat = proj_labels.view(-1)
            selected_labels = proj_labels_flat[indices]
            
            batch_correct = ((predictions == selected_labels) & (selected_labels > 0)).sum().item()
            batch_total = (selected_labels > 0).sum().item()
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
            all_accuracies.append(batch_accuracy)
            global_correct += batch_correct
            global_total += batch_total
            
            valid_eval_mask = (selected_labels > 0)
            for class_id in range(model.num_classes):
                class_mask = (selected_labels == class_id)
                pred_mask = (predictions == class_id) & valid_eval_mask
                
                if class_mask.any():
                    class_correct[class_id] += (predictions[class_mask] == class_id).sum().item()
                    class_total[class_id] += class_mask.sum().item()

                intersection = (class_mask & pred_mask).sum().item()
                union = (class_mask | pred_mask).sum().item()
                
                class_intersection[class_id] += intersection
                class_union[class_id] += union
    
    global_accuracy = global_correct / global_total if global_total > 0 else 0
    mean_batch_accuracy = np.mean(all_accuracies) if all_accuracies else 0
    
    per_class_accuracy = {}
    per_class_iou = {}
    valid_ious = []
    
    for class_id in range(model.num_classes):
        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = (class_correct[class_id] / class_total[class_id]).item()
        else:
            per_class_accuracy[class_id] = 0.0
        
        # Calculate IoU for each class (Exclude Class 0 which is typically the ignored/unlabeled class)
        if class_union[class_id] > 0:
            iou = (class_intersection[class_id] / class_union[class_id]).item()
            per_class_iou[class_id] = iou
            if class_id > 0:
                valid_ious.append(iou)
        else:
            per_class_iou[class_id] = 0.0

    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    print(f"\n{'='*60}")
    print("Training Set Accuracy Results")
    print(f"{'='*60}")
    print(f"Global Accuracy: {global_accuracy:.4f} ({global_correct}/{global_total})")
    print(f"Mean Batch Accuracy: {mean_batch_accuracy:.4f}")
    print(f"mIOU: {miou:.4f}")
    print()
    print("Per-Class Accuracies:")
    for class_id in sorted(range(model.num_classes)):
        if class_total[class_id] > 0:
            acc = per_class_accuracy[class_id]
            iou = per_class_iou[class_id]
            correct = int(class_correct[class_id].item())
            total = int(class_total[class_id].item())
            print(f"  Class {class_id}: Acc={acc:.4f} ({correct}/{total}), IoU={iou:.4f}")
        else:
            print(f"  Class {class_id}: No samples")

    if return_detailed:
        detailed_stats = {
            "per_class_acc": per_class_accuracy,
            "per_class_iou": per_class_iou,
            "class_total": {i: int(class_total[i].item()) for i in range(model.num_classes)},
            "class_correct": {i: int(class_correct[i].item()) for i in range(model.num_classes)},
            "class_intersection": {i: int(class_intersection[i].item()) for i in range(model.num_classes)},
            "class_union": {i: int(class_union[i].item()) for i in range(model.num_classes)}
        }
        return global_accuracy, miou, detailed_stats

    return global_accuracy, miou

def test_hdc_model_debug(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_correct = torch.zeros(model.num_classes, device=device)
    class_total = torch.zeros(model.num_classes, device=device)
    class_sim_sum = torch.zeros(model.num_classes, device=device)
    class_sim_sq = torch.zeros(model.num_classes, device=device)

    pred_hist = torch.zeros(model.num_classes, device=device)

    global_correct = 0
    global_total = 0

    model.eval()
    with torch.no_grad():
        for proj_in, _, proj_labels, *_ in dataloader:
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device).view(-1)

            logits, sims, indices, _ = model(proj_in)
            predictions = torch.argmax(logits, dim=1)

            selected_labels = proj_labels[indices]

            global_correct += (predictions == selected_labels).sum().item()
            global_total += selected_labels.numel()

            for c in range(model.num_classes):
                mask = selected_labels == c
                if mask.any():
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (predictions[mask] == c).sum().item()

                    s = sims[mask]
                    class_sim_sum[c] += s.sum().item()
                    class_sim_sq[c] += (s ** 2).sum().item()

            for p in predictions:
                pred_hist[p] += 1

    print("\n[Accuracy + Similarity Diagnostics]")
    for c in range(model.num_classes):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c]
            mean_sim = class_sim_sum[c] / class_total[c]
            var_sim = (
                class_sim_sq[c] / class_total[c] - mean_sim ** 2
            ).clamp(min=0)
            std_sim = torch.sqrt(var_sim)

            collapse_flag = "⚠ COLLAPSE" if abs(mean_sim.item() - 0.5) < 1e-3 else ""

            print(
                f"  Class {c:2d}: acc={acc:.4f}, "
                f"sim μ={mean_sim:.4f}, σ={std_sim:.4f} {collapse_flag}"
            )
        else:
            print(f"  Class {c:2d}: no samples")

    # Prediction entropy (global collapse detector)
    pred_dist = pred_hist / pred_hist.sum().clamp(min=1)
    entropy = -(pred_dist * torch.log2(pred_dist + 1e-8)).sum()

    print("\n[Prediction Entropy]")
    print(f"  Entropy: {entropy:.4f} (max = log2({model.num_classes}) ≈ {np.log2(model.num_classes):.2f})")

    print("\n[Prediction Distribution]")
    for c in range(model.num_classes):
        print(
            f"  Class {c:2d}: {int(pred_hist[c].item()):6d} "
            f"({100 * pred_dist[c].item():5.2f}%)"
        )

    print(
        f"\nGlobal Accuracy: {global_correct / max(global_total,1):.4f} "
        f"({global_correct}/{global_total})"
    )

def test_orig(ARCH, DATA) -> Model:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=DATA_DIR,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=True)
    
    dataloader = parser.get_train_set()
    val_loader = parser.get_valid_set()

    ignore = []
    for cl, ign in DATA['learning_ignore'].items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)

    evaluator = iouEval(NUM_CLASSES, device, ignore)

    trainer = BasicHD(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR, None)

    trainer.train(dataloader, trainer.model, None)

    for i in range(1):
        trainer.retrain(dataloader, trainer.model, i+1, None)

    model: Model = trainer.model

    test_hdc_model(model, dataloader)

    # trainer.validate(val_loader, model, evaluator)

    return model

def init_sub(ARCH, DATA):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=DATA_DIR,
                        train_sequences=DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        labels=DATA["labels"],
                        color_map=DATA["color_map"],
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=ARCH["train"]["batch_size"],
                        workers=ARCH["train"]["workers"],
                        gt=True,
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()

    model: EllipsoidModel = EllipsoidModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model = torch.load(HDC_SAVE_PATH, weights_only=False)

    model.init_subclusters(dataloader)
    test_hdc_model(model, dataloader)

    torch.save(model.state_dict(), HDC_SUB_PATH)

    print(f"Subcluster Initialized Model saved to {HDC_SUB_PATH}")

def test_inference(ARCH, DATA):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=DATA_DIR,
                        train_sequences=DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        labels=DATA["labels"],
                        color_map=DATA["color_map"],
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=ARCH["train"]["batch_size"],
                        workers=ARCH["train"]["workers"],
                        gt=True,
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()

    model: EllipsoidModel = EllipsoidModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    images, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = next(iter(dataloader))

    image = images[0].to(device).unsqueeze(0)

    model.inference_update(image)

def main():
    try:
        # ARCH = yaml.safe_load(open("config/arch/senet-1024p.yml", 'r'))
        ARCH = yaml.safe_load(open("config/arch/senet-2048p-gen.yml", 'r')) # higher res
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    # convert_dataset()

    ARCH["train"]["batch_size"] = 16

    train_extractor(ARCH, DATA)
    # DDFEtrain_extractor(ARCH, DATA)

    ARCH["train"]["batch_size"] = 2

    hdc = train_hdc(ARCH, DATA)
    init_sub(ARCH, DATA)
    # test_inference(ARCH, DATA)

    # test_orig(ARCH, DATA)

if __name__=="__main__":
    main()