"""
Pix2Pix GAN for Satellite-to-Map Image Translation
Week 6 Assignment
"""

# ============================================================
# Step 3: Load and Preprocess the Satellite-to-Map Dataset
# Commit: "Loaded and preprocessed image datasets for Pix2Pix GAN"
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Transform: resize to 256x256, convert to tensor, normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class SatMapDataset(Dataset):
    """Paired satellite and map image dataset.
    Expects ./data/satellite/ and ./data/map/ folders.
    Falls back to synthetic data if folders are not found.
    """
    def __init__(self, sat_dir="./data/satellite", map_dir="./data/map", size=200):
        self.use_synthetic = not (os.path.exists(sat_dir) and os.path.exists(map_dir))
        self.size = size

        if not self.use_synthetic:
            self.sat_imgs = sorted([
                os.path.join(sat_dir, f) for f in os.listdir(sat_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.map_imgs = sorted([
                os.path.join(map_dir, f) for f in os.listdir(map_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.size = min(len(self.sat_imgs), len(self.map_imgs))
            print(f"Loaded {self.size} image pairs from disk.")
        else:
            print("Data folders not found — using synthetic data for demo.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.use_synthetic:
            # Generate synthetic paired images (random noise as placeholder)
            sat = torch.rand(3, 256, 256) * 2 - 1
            mp  = torch.rand(3, 256, 256) * 2 - 1
        else:
            sat = transform(Image.open(self.sat_imgs[idx]).convert("RGB"))
            mp  = transform(Image.open(self.map_imgs[idx]).convert("RGB"))
        return sat, mp

# DataLoader
dataset    = SatMapDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Quick sanity check — display one sample pair
sat_sample, map_sample = dataset[0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow((sat_sample.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1))
axes[0].set_title("Input: Satellite")
axes[0].axis("off")
axes[1].imshow((map_sample.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1))
axes[1].set_title("Target: Map")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("sample_input.jpg", dpi=150)
plt.close()
print("Saved sample_input.jpg")
