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



# ============================================================
# Step 4: Implement Pix2Pix Generator (U-Net) and Discriminator (PatchGAN)
# Commit: "Implemented Pix2Pix Generator and Discriminator models"
# ============================================================

class UNetBlock(nn.Module):
    """Single encoder or decoder block for U-Net."""
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        layers = []
        if down:
            layers += [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        else:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if use_bn:
            layers += [nn.BatchNorm2d(out_ch)]
        if dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.ReLU(True) if not down else nn.LeakyReLU(0.2, True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """U-Net Generator: encodes input then decodes with skip connections."""
    def __init__(self):
        super().__init__()
        # Encoder (downsampling)
        self.e1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = UNetBlock(64,  128)
        self.e3 = UNetBlock(128, 256)
        self.e4 = UNetBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(True))

        # Decoder (upsampling) — input channels doubled due to skip connections
        self.d1 = UNetBlock(512,  512, down=False, dropout=True)
        self.d2 = UNetBlock(1024, 256, down=False)
        self.d3 = UNetBlock(512,  128, down=False)
        self.d4 = UNetBlock(256,   64, down=False)

        # Final output layer
        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()   # Output in [-1, 1]
        )

    def forward(self, x):
        e1 = self.e1(x)          # 64 x 128 x 128
        e2 = self.e2(e1)         # 128 x 64 x 64
        e3 = self.e3(e2)         # 256 x 32 x 32
        e4 = self.e4(e3)         # 512 x 16 x 16
        bn = self.bottleneck(e4) # 512 x 8 x 8

        # Decode with skip connections (concatenate encoder features)
        d1 = self.d1(bn)                           # 512 x 16
        d2 = self.d2(torch.cat([d1, e4], dim=1))   # 256 x 32
        d3 = self.d3(torch.cat([d2, e3], dim=1))   # 128 x 64
        d4 = self.d4(torch.cat([d3, e2], dim=1))   # 64 x 128
        return self.out(torch.cat([d4, e1], dim=1)) # 3 x 256 x 256


class Discriminator(nn.Module):
    """PatchGAN Discriminator: classifies 70x70 image patches as real or fake."""
    def __init__(self):
        super().__init__()
        # Input: concatenated source + target (6 channels)
        self.model = nn.Sequential(
            nn.Conv2d(6,   64,  4, 2, 1),         nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,  128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1,   4, 1, 1),          nn.Sigmoid()
        )

    def forward(self, src, tgt):
        # Concatenate source and target along channel axis
        return self.model(torch.cat([src, tgt], dim=1))


# Instantiate models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

G = Generator().to(device)
D = Discriminator().to(device)
print("Generator and Discriminator initialized.")



# ============================================================
# Step 5: Train Pix2Pix GAN
# Commit: "Trained Pix2Pix GAN for satellite-to-map translation"
# ============================================================

# Loss functions
adversarial_loss = nn.BCELoss()
l1_loss          = nn.L1Loss()
LAMBDA_L1        = 10  # Weight for L1 pixel-wise loss

# Optimizers (Adam with lr=0.0002, β1=0.5 as in original paper)
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Track losses per epoch for visualization
g_losses, d_losses = [], []

EPOCHS = 10
print(f"\nStarting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    epoch_g, epoch_d = [], []

    for sat_imgs, map_imgs in dataloader:
        sat_imgs = sat_imgs.to(device)
        map_imgs = map_imgs.to(device)

        batch_size = sat_imgs.size(0)
        real_label = torch.ones(batch_size, 1, 30, 30).to(device)
        fake_label = torch.zeros(batch_size, 1, 30, 30).to(device)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        fake_map  = G(sat_imgs)
        real_pred = D(sat_imgs, map_imgs)
        fake_pred = D(sat_imgs, fake_map.detach())

        d_real_loss = adversarial_loss(real_pred, real_label)
        d_fake_loss = adversarial_loss(fake_pred, fake_label)
        d_loss = (d_real_loss + d_fake_loss) * 0.5

        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        optimizer_G.zero_grad()
        fake_map  = G(sat_imgs)
        fake_pred = D(sat_imgs, fake_map)

        g_adv  = adversarial_loss(fake_pred, real_label)   # Fool discriminator
        g_l1   = l1_loss(fake_map, map_imgs)               # Pixel-wise accuracy
        g_loss = g_adv + LAMBDA_L1 * g_l1

        g_loss.backward()
        optimizer_G.step()

        epoch_d.append(d_loss.item())
        epoch_g.append(g_loss.item())

    avg_d = np.mean(epoch_d)
    avg_g = np.mean(epoch_g)
    d_losses.append(avg_d)
    g_losses.append(avg_g)
    print(f"Epoch [{epoch+1}/{EPOCHS}]  D Loss: {avg_d:.4f}  G Loss: {avg_g:.4f}")

print("Training complete.")
