"""
Phase 3: Teacher-Student Training with Chunked Audio
======================================================
Key changes from Phase 2:
  - Use Demucs as "teacher" to generate stem training data
  - CHUNK audio into short segments (~6s) to fit in memory
  - Bigger model: U-Net++ with residual blocks
  - 4 output instruments: vocals, bass, drums, other
  - Train on ANY song the user has (no MUSDB18 needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import subprocess, sys, os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# --- Config ---
SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_FREQS = N_FFT // 2 + 1  # 513
CHUNK_SECONDS = 6.0       # Train on 6-second chunks
CHUNK_SAMPLES = int(SR * CHUNK_SECONDS)
CHUNK_FRAMES = CHUNK_SAMPLES // HOP_LENGTH + 1  # ~517 frames per chunk
INSTRUMENTS = ['vocals', 'bass', 'drums', 'other']
N_INSTRUMENTS = 4


# ========== Model: U-Net++ (compact) ==========

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ResBlock(out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.net(x)
        return self.pool(s), s


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.net = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ResBlock(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [0, dw, 0, dh])
        return self.net(torch.cat([x, skip], dim=1))


class UNetPlusPlus(nn.Module):
    """4-level U-Net with ResBlocks. ~3.4M params."""

    def __init__(self, n_freqs=N_FREQS, n_instruments=N_INSTRUMENTS):
        super().__init__()
        b = 24  # base channels
        self.enc1 = Down(1, b)
        self.enc2 = Down(b, b * 2)
        self.enc3 = Down(b * 2, b * 4)
        self.enc4 = Down(b * 4, b * 8)

        self.bot = nn.Sequential(
            nn.Conv2d(b * 8, b * 16, 3, padding=1),
            nn.BatchNorm2d(b * 16), nn.ReLU(inplace=True),
            ResBlock(b * 16),
        )

        self.dec4 = Up(b * 16, b * 8)
        self.dec3 = Up(b * 8, b * 4)
        self.dec2 = Up(b * 4, b * 2)
        self.dec1 = Up(b * 2, b)

        self.out = nn.Sequential(
            nn.Conv2d(b, n_instruments, 1), nn.Sigmoid())

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.bot(x4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out(d1)


# ========== Chunked Dataset ==========

def run_demucs(audio_path):
    """Ensure Demucs teacher output exists."""
    out_root = Path(__file__).parent.parent / "samples" / "separated"
    stem_dir = out_root / "htdemucs" / audio_path.stem

    if all((stem_dir / f"{i}.wav").exists() for i in INSTRUMENTS):
        return stem_dir

    print(f"  Running Demucs teacher on {audio_path.name}...")
    subprocess.run([sys.executable, "-m", "demucs", "-o", str(out_root),
                    str(audio_path)], capture_output=True, check=True)
    return stem_dir


def spec_from_file(path, sr=SR, n_fft=N_FFT, hop=HOP_LENGTH):
    """Load audio and return log-spec."""
    y, _ = librosa.load(path, sr=sr)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    return np.log1p(np.abs(D)).astype(np.float32)


def chunk_spec(spec, chunk_frames=CHUNK_FRAMES, hop_frames=256):
    """Split a spectrogram (F, T) into overlapping chunks (F, chunk_frames)."""
    chunks = []
    T = spec.shape[1]
    for start in range(0, T - chunk_frames + 1, hop_frames):
        chunks.append(spec[:, start:start + chunk_frames])
    if chunks:
        return np.stack(chunks, axis=0)
    # Pad if too short
    pad = np.zeros((spec.shape[0], chunk_frames))
    pad[:, :spec.shape[1]] = spec
    return pad[np.newaxis, ...]


class ChunkedDemucsDataset(Dataset):
    """Load audio files, run Demucs, chunk into short segments."""

    def __init__(self, audio_dir):
        self.audio_dir = Path(audio_dir)
        self.files = []
        for ext in ['*.mp3', '*.wav']:
            self.files.extend(self.audio_dir.glob(ext))
        self.files = [f for f in self.files
                      if 'separated' not in str(f)
                      and 'synthetic' not in str(f)]
        if not self.files:
            samples = self.audio_dir / "samples"
            for ext in ['*.mp3', '*.wav']:
                self.files.extend(samples.glob(ext))
            self.files = [f for f in self.files
                          if 'separated' not in str(f)
                          and 'synthetic' not in str(f)]

        # Generate chunks
        self.chunks = []  # list of (mix_chunk, targets_chunk)
        total_chunks = 0
        for fpath in self.files:
            print(f"  Processing {fpath.name}...")
            try:
                mix_spec = spec_from_file(fpath)
                stem_dir = run_demucs(fpath)
                stem_specs = []
                for inst in INSTRUMENTS:
                    sp = spec_from_file(stem_dir / f"{inst}.wav")
                    stem_specs.append(sp)
                targets = np.stack(stem_specs, axis=0)  # (C, F, T)
            except Exception as e:
                print(f"    Skipping: {e}")
                continue

            # Chunk both
            mix_chunks = chunk_spec(mix_spec)
            tgt_chunks = chunk_spec(targets[0])  # just for shape
            n_chunks = min(len(mix_chunks), len(chunk_spec(targets[0].reshape(N_FREQS, -1))))

            # Actually do it properly
            T = mix_spec.shape[1]
            start = 0
            c = 0
            while start + CHUNK_FRAMES <= T + CHUNK_FRAMES // 2:
                end = min(start + CHUNK_FRAMES, T)
                if end - start < CHUNK_FRAMES // 2:
                    break

                mix_chunk = mix_spec[:, start:end]
                tgt_chunk = targets[:, :, start:end]

                # Pad if needed
                if mix_chunk.shape[1] < CHUNK_FRAMES:
                    pad_w = CHUNK_FRAMES - mix_chunk.shape[1]
                    mix_chunk = np.pad(mix_chunk, ((0, 0), (0, pad_w)))
                    tgt_chunk = np.pad(tgt_chunk, ((0, 0), (0, 0), (0, pad_w)))

                self.chunks.append((mix_chunk, tgt_chunk))
                c += 1
                start += HOP_LENGTH * 6  # ~1.5s stride

            total_chunks += c
            print(f"    {c} chunks")

        print(f"[Dataset] {total_chunks} chunks from {len(self.files)} files")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        mix, tgt = self.chunks[idx]
        return {
            'mixture': torch.from_numpy(mix[np.newaxis, ...].copy()),  # (1, F, T)
            'targets': torch.from_numpy(tgt.copy()),                    # (C, F, T)
        }


# ========== Training ==========

def main():
    print("=" * 60)
    print("Phase 3: Chunked Teacher-Student Training")
    print("=" * 60)

    model = UNetPlusPlus().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: U-Net++ ({n_params:,} params)")
    print(f"Output: {INSTRUMENTS}")

    dataset = ChunkedDemucsDataset(Path(__file__).parent.parent)
    if len(dataset) == 0:
        print("[ERROR] No audio files found! Put .mp3 in project root.")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    n_epochs = 30

    print(f"\nTraining: {len(dataset)} chunks, {n_epochs} epochs, {DEVICE}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for batch in loader:
            mix = batch['mixture'].to(DEVICE)
            tgt = batch['targets'].to(DEVICE)

            opt.zero_grad()
            masks = model(mix)
            sep = masks * mix
            loss_l1 = F.l1_loss(sep, tgt)
            loss_tc = torch.mean((masks[:, :, :, 1:] - masks[:, :, :, :-1])**2)
            loss = loss_l1 + 0.05 * loss_tc
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}  Loss: {total_loss/len(loader):.4f}")

    # Save
    save_path = Path(__file__).parent.parent / "samples" / "unetpp_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[OK] Model saved: {save_path}")

    # Test on 光辉岁月
    print("\nTesting on 光辉岁月...")
    model.eval()
    test_path = Path(__file__).parent.parent / "BEYOND+-+光辉岁月.mp3"
    if test_path.exists():
        y, _ = librosa.load(test_path, sr=SR)
        # Process in chunks to avoid memory issues
        D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag, phase = np.abs(D), np.angle(D)
        log_spec = np.log1p(mag)
        T = log_spec.shape[1]

        out_dir = Path(__file__).parent.parent / "samples" / "separated" / "unetpp"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Process full song one chunk at a time
        all_masks = np.zeros((N_INSTRUMENTS, N_FREQS, T))
        chunk_overlap = np.zeros((N_INSTRUMENTS, N_FREQS, T))

        for start in range(0, T, CHUNK_FRAMES // 2):
            end = min(start + CHUNK_FRAMES, T)
            chunk = log_spec[:, start:end]
            if chunk.shape[1] < 100:
                break

            # Pad to CHUNK_FRAMES
            pad_w = CHUNK_FRAMES - chunk.shape[1]
            inp = np.pad(chunk, ((0, 0), (0, max(0, pad_w))))
            x = torch.from_numpy(inp[np.newaxis, np.newaxis, ...]).float().to(DEVICE)

            with torch.no_grad():
                mask_out = model(x)[0].cpu().numpy()
                if pad_w > 0:
                    mask_out = mask_out[:, :, :end - start]
                all_masks[:, :, start:end] += mask_out[:, :, :end - start]
                chunk_overlap[:, :, start:end] += 1

        # Average overlapping regions
        all_masks = all_masks / np.maximum(chunk_overlap, 1)
        all_masks = np.clip(all_masks, 0, 1)

        # Reconstruct
        for i, inst in enumerate(INSTRUMENTS):
            mask_rs = zoom(all_masks[i], (1, D.shape[1] / all_masks.shape[2]), order=1)
            mask_rs = mask_rs[:, :D.shape[1]]
            mask_rs = np.clip(mask_rs, 0, 1)
            sep = librosa.istft(mag * mask_rs * np.exp(1j * phase), hop_length=HOP_LENGTH)
            sf.write(out_dir / f"unetpp_{inst}.wav", sep, SR)
            rms = np.sqrt(np.mean(sep**2))
            print(f"  {inst}: RMS={rms:.4f}")

        print(f"\nResults saved to {out_dir}")

    print(f"\nPhase 3 complete!")


if __name__ == "__main__":
    main()
