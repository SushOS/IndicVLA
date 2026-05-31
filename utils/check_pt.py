import torch
d = torch.load('D:/HumanML3d/amass_smpl.pt', weights_only=False, map_location='cpu')
print(f"Motions      : {len(d['motion_files'])}")
print(f"Total frames : {sum(d['motion_num_frames']).item()}")
print(f"Keys         : {list(d.keys())}")
# Estimate file size
import os
size_mb = os.path.getsize('D:/HumanML3d/amass_smpl.pt') / 1024 / 1024
print(f"File size    : {size_mb:.1f} MB")
