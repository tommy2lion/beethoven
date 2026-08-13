"""
Phase 2: U-Net for Music Source Separation
============================================
The first deep learning model in the Beethoven project.

Architecture: Simple 4-level U-Net on spectrograms
  - Input:  (B, 1, F, T)   mixture log-spectrogram
  - Output: (B, C, F, T)   masks for C instruments

Key design:
  - Encoder: Conv2D + BatchNorm + ReLU + MaxPool (downsample 2x)
  - Decoder: ConvTranspose2D + Skip Connections (upsample 2x)
  - Output:  Sigmoid masks applied to input spectrogram
  - Loss:    L1 on masked spectrograms + temporal continuity loss

Your insight "时序连续性" is baked into:
  1) CNN architecture: local receptive fields across time
  2) Skip connections: preserve fine temporal structure
  3) Temporal continuity loss: explicitly penalize frame-to-frame jumps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent dir for dataset import
sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_dataloader, INSTRUMENTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ========== Model ==========

class DownBlock(nn.Module):
    """Encoder block: Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU -> MaxPool"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """Decoder block: Upsample -> Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),  # *2 for skip connection
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch (padding if needed)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for spectrogram mask prediction.

    Input:  (B, 1, F, T)   F=n_fft//2+1, T=time frames
    Output: (B, C, F, T)   C=number of instruments (masks in 0~1)
    """

    def __init__(self, n_freqs=513, n_instruments=3, base_ch=32):
        super().__init__()
        self.n_instruments = n_instruments

        # Encoder (4 levels)
        self.enc1 = DownBlock(1, base_ch)       # 513 -> 256
        self.enc2 = DownBlock(base_ch, base_ch * 2)   # 256 -> 128
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4) # 128 -> 64
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8) # 64 -> 32

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, 3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
        )

        # Decoder (4 levels)
        self.dec4 = UpBlock(base_ch * 16, base_ch * 8)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch)

        # Output
        self.out = nn.Sequential(
            nn.Conv2d(base_ch, n_instruments, 1),
            nn.Sigmoid(),  # masks in [0, 1]
        )

    def forward(self, x):
        """
        x: (B, 1, F, T)  mixture spectrogram
        returns: (B, C, F, T) masks
        """
        # Encoder
        x1, skip1 = self.enc1(x)   # skip1: (B, base_ch, F, T)
        x2, skip2 = self.enc2(x1)  # skip2: (B, base_ch*2, F/2, T/2)
        x3, skip3 = self.enc3(x2)  # skip3: (B, base_ch*4, F/4, T/4)
        x4, skip4 = self.enc4(x3)  # skip4: (B, base_ch*8, F/8, T/8)

        # Bottleneck
        b = self.bottleneck(x4)    # (B, base_ch*16, F/16, T/16)

        # Decoder with skip connections
        d4 = self.dec4(b, skip4)   # (B, base_ch*8, F/8, T/8)
        d3 = self.dec3(d4, skip3)  # (B, base_ch*4, F/4, T/4)
        d2 = self.dec2(d3, skip2)  # (B, base_ch*2, F/2, T/2)
        d1 = self.dec1(d2, skip1)  # (B, base_ch, F, T)

        # Output masks
        masks = self.out(d1)       # (B, C, F, T)
        return masks


# ========== Loss Functions ==========

def l1_spectral_loss(predictions, targets):
    """L1 loss between predicted and target spectrograms."""
    return F.l1_loss(predictions, targets)


def temporal_continuity_loss(masks):
    """
    Explicit temporal smoothness constraint.
    Penalizes large changes between adjacent time frames.

    This is YOUR idea: "相邻时刻的分解结果必须与前后时刻形成时序呼应"

    Args:
        masks: (B, C, F, T) predicted masks
    """
    # diff between frame t and t+1 along the time axis (dim=-1)
    diff = masks[:, :, :, 1:] - masks[:, :, :, :-1]
    return torch.mean(diff ** 2)


def combined_loss(pred_masks, mix_spec, target_specs, lambda_tc=0.1):
    """
    Combined loss: L1 + temporal continuity.

    pred_masks: (B, C, F, T)
    mix_spec:   (B, 1, F, T)
    target_specs: (B, C, F, T)

    Apply masks to mixture to get separated spectrograms,
    then compare with ground truth.
    """
    # Apply masks: separated = mask * mixture (broadcast mixture over C channels)
    sep_specs = pred_masks * mix_spec

    # L1 loss on separated spectrograms
    loss_l1 = l1_spectral_loss(sep_specs, target_specs)

    # Temporal continuity loss on masks (your idea!)
    loss_tc = temporal_continuity_loss(pred_masks)

    total = loss_l1 + lambda_tc * loss_tc
    return total, loss_l1.item(), loss_tc.item()


# ========== Training ==========

def train_one_epoch(model, dataloader, optimizer, lambda_tc=0.1):
    """Train for one epoch, return average losses."""
    model.train()
    total_loss = 0
    total_l1 = 0
    total_tc = 0
    n_batches = 0

    for batch in dataloader:
        mix = batch['mixture'].to(DEVICE)       # (B, 1, F, T)
        targets = batch['targets'].to(DEVICE)   # (B, C, F, T)

        optimizer.zero_grad()

        # Forward: predict masks
        masks = model(mix)  # (B, C, F, T)

        # Loss
        loss, l1_val, tc_val = combined_loss(masks, mix, targets, lambda_tc)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_l1 += l1_val
        total_tc += tc_val
        n_batches += 1

    return (total_loss / n_batches,
            total_l1 / n_batches,
            total_tc / n_batches)


def validate(model, dataloader):
    """Validate and compute average loss + per-instrument SDR."""
    model.eval()
    total_loss = 0
    total_l1 = 0
    total_tc = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            mix = batch['mixture'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)

            masks = model(mix)
            loss, l1_val, tc_val = combined_loss(masks, mix, targets)

            total_loss += loss.item()
            total_l1 += l1_val
            total_tc += tc_val
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_l1 = total_l1 / n_batches
    avg_tc = total_tc / n_batches

    return avg_loss, avg_l1, avg_tc


def main():
    print("=" * 60)
    print("Phase 2: U-Net for Music Source Separation")
    print("=" * 60)

    # Hyperparameters
    BATCH_SIZE = 2
    EPOCHS = 100
    LR = 1e-3
    LAMBDA_TC = 0.1  # temporal continuity weight

    # Data
    data_dir = Path(__file__).parent.parent / "samples" / "synthetic"
    train_loader = create_dataloader(data_dir, batch_size=BATCH_SIZE, shuffle=True)
    # Use same data for validation (small dataset)
    val_loader = create_dataloader(data_dir, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = UNet(n_freqs=513, n_instruments=len(INSTRUMENTS), base_ch=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: U-Net ({n_params:,} parameters)")
    print(f"Channels: {len(INSTRUMENTS)} instruments ({', '.join(INSTRUMENTS)})")
    print(f"Training: {len(train_loader.dataset)} scenes, {EPOCHS} epochs")
    print(f"Device: {DEVICE}")
    print(f"Loss: L1 + {LAMBDA_TC} * temporal_continuity")
    print()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_l1, train_tc = train_one_epoch(
            model, train_loader, optimizer, LAMBDA_TC)
        val_loss, val_l1, val_tc = validate(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       Path(__file__).parent.parent / "samples" / "unet_best.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"Train: {train_loss:.4f} (L1={train_l1:.4f} TC={train_tc:.4f})  "
                  f"Val: {val_loss:.4f} (L1={val_l1:.4f} TC={val_tc:.4f})")

    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: samples/unet_best.pth")
    print(f"{'='*60}")

    # Quick test: separate one sample and play it back
    print("\nRunning separation on a test sample...")
    model.eval()
    sample = val_loader.dataset[0]
    mix_spec = sample['mixture'].unsqueeze(0).to(DEVICE)  # (1, 1, F, T)
    with torch.no_grad():
        masks = model(mix_spec)  # (1, C, F, T)

    # Convert back to audio
    import librosa
    import soundfile as sf
    from dataset import SR, HOP_LENGTH

    mix_wav, _ = librosa.load(
        data_dir / f"scene_{sample['scene']:02d}_mix.wav", sr=SR)

    out_dir = Path(__file__).parent.parent / "samples" / "separated" / "unet"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, inst in enumerate(INSTRUMENTS):
        mask = masks[0, i].cpu().numpy()
        # Apply mask to mixture STFT
        D = librosa.stft(mix_wav, n_fft=1024, hop_length=HOP_LENGTH)
        # Interpolate mask to match STFT time frames
        from scipy.ndimage import zoom
        mask_resized = zoom(mask, (1, D.shape[1] / mask.shape[1]), order=1)
        mask_resized = np.minimum(mask_resized[:, :D.shape[1]], 1.0)
        sep_mag = np.abs(D) * mask_resized
        sep_wav = librosa.istft(sep_mag * np.exp(1j * np.angle(D)),
                                hop_length=HOP_LENGTH)
        sf.write(out_dir / f"unet_{inst}.wav", sep_wav, SR)
        print(f"  Saved: unet_{inst}.wav")

    print(f"\nPhase 2 complete! Listen to results in {out_dir}")


if __name__ == "__main__":
    main()
