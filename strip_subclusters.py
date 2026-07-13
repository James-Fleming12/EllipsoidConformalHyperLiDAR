import torch
import os

def strip_subclusters(pretrained_path):
    print(f"Loading checkpoint from: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    removed = False
    keys_to_remove = []
    
    # Identify keys to remove
    for k in state_dict.keys():
        if "subclusters" in k:
            keys_to_remove.append(k)
            
    if len(keys_to_remove) > 0:
        for k in keys_to_remove:
            del state_dict[k]
            print(f"Removed key: {k}")
        
        # Save it back, optionally backing up the original
        backup_path = pretrained_path + ".bak"
        if not os.path.exists(backup_path):
            os.rename(pretrained_path, backup_path)
            print(f"Backed up original to: {backup_path}")
            
        torch.save(state_dict, pretrained_path)
        print(f"Successfully saved stripped checkpoint to: {pretrained_path}")
    else:
        print("No 'subclusters' keys found in the checkpoint. It is already clean!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strip subclusters from old DensityModel checkpoints")
    parser.add_argument("--path", type=str, default="logs/kitti_pretrain/hdc_sub.pth", help="Path to the checkpoint")
    args = parser.parse_args()
    
    strip_subclusters(args.path)
