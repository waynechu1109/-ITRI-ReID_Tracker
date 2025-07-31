import torch

# === Input and output filenames ===
pth_tar_path = 'osnet_ain_ms_d_c.pth.tar'     # Input: full checkpoint file
pt_path = 'osnet_ain_x1_0_ms_d_c.pt'          # Output: only state_dict

# === Load checkpoint ===
checkpoint = torch.load(pth_tar_path, map_location='cpu')

# === Check structure ===
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("Extracted 'state_dict' from checkpoint.")
else:
    state_dict = checkpoint
    print("Checkpoint is already a state_dict.")

# === Save as .pt ===
torch.save(state_dict, pt_path)
print(f"Saved state_dict to {pt_path}")
