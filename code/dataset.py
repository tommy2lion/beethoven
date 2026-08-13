"""
PyTorch Dataset for music source separation.
Works with our synthetic data, and can be extended to MUSDB18 later.

Each sample returns:
  - mixture spectrogram: (1, F, T)  - input to the model
  - target spectrograms: (C, F, T)  - one per instrument
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
INSTRUMENTS = ['piano', 'violin', 'bass']
N_INSTRUMENTS = len(INSTRUMENTS)

class InstrumentSeparationDataset(Dataset):
    """Load synthetic multi-instrument mixtures and their ground-truth stems."""

    def __init__(self, data_dir, instruments=INSTRUMENTS,
                 sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.data_dir = Path(data_dir)
        self.instruments = instruments
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Find all scene indices
        self.scenes = set()
        for f in self.data_dir.glob("scene_*_mix.wav"):
            idx = int(f.stem.split('_')[1])
            self.scenes.add(idx)
        self.scenes = sorted(self.scenes)

        if len(self.scenes) == 0:
            raise FileNotFoundError(f"No scene_*_mix.wav files found in {data_dir}")

        print(f"[Dataset] Found {len(self.scenes)} scenes in {data_dir}")

    def __len__(self):
        return len(self.scenes)

    def _load_spectrogram(self, path):
        """Load audio file and return log-magnitude spectrogram."""
        y, _ = librosa.load(path, sr=self.sr)
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        # Log scale: log(1 + mag) to compress dynamic range
        log_spec = np.log1p(magnitude)
        return log_spec.astype(np.float32)

    def __getitem__(self, idx):
        scene = self.scenes[idx]

        # Load mixture
        mix_path = self.data_dir / f"scene_{scene:02d}_mix.wav"
        mix_spec = self._load_spectrogram(mix_path)

        # Load each instrument stem
        target_specs = []
        for inst in self.instruments:
            path = self.data_dir / f"scene_{scene:02d}_{inst}.wav"
            if path.exists():
                spec = self._load_spectrogram(path)
            else:
                # Silent instrument (not present in this scene)
                spec = np.zeros_like(mix_spec)
            target_specs.append(spec)

        # Stack: (C, F, T) where C = number of instruments
        targets = np.stack(target_specs, axis=0)
        # Add channel dim for mixture: (1, F, T)
        mix_spec = mix_spec[np.newaxis, ...]

        return {
            'mixture': torch.from_numpy(mix_spec),
            'targets': torch.from_numpy(targets),
            'scene': scene,
        }


def collate_batch(batch):
    """Collate function: handles variable-length spectrograms by padding."""
    # Find max T in batch
    max_T = max(b['mixture'].shape[-1] for b in batch)
    n_instruments = batch[0]['targets'].shape[0]

    mixtures = []
    targets = []
    scenes = []

    for b in batch:
        mix = b['mixture']
        tgt = b['targets']
        T = mix.shape[-1]

        # Pad to max_T
        if T < max_T:
            pad_size = max_T - T
            mix = torch.nn.functional.pad(mix, (0, pad_size))
            tgt = torch.nn.functional.pad(tgt, (0, pad_size))

        mixtures.append(mix)
        targets.append(tgt)
        scenes.append(b['scene'])

    return {
        'mixture': torch.stack(mixtures),
        'targets': torch.stack(targets),
        'scene': scenes,
    }


def create_dataloader(data_dir, batch_size=2, shuffle=True):
    """Create a DataLoader for our synthetic dataset."""
    dataset = InstrumentSeparationDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )
    return dataloader


# ---------- Quick test ----------
if __name__ == "__main__":
    print("=" * 50)
    print("Testing dataset pipeline...")
    print("=" * 50)

    data_dir = Path(__file__).parent.parent / "samples" / "synthetic"
    dataset = InstrumentSeparationDataset(data_dir)

    print(f"\nDataset size: {len(dataset)} samples")
    sample = dataset[0]
    print(f"Mixture shape: {sample['mixture'].shape}")    # (1, F, T)
    print(f"Targets shape: {sample['targets'].shape}")    # (C, F, T)
    print(f"Scene index: {sample['scene']}")
    print(f"Instruments: {dataset.instruments}")

    # Test DataLoader
    dataloader = create_dataloader(data_dir, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    print(f"\nBatch mixture shape: {batch['mixture'].shape}")    # (B, 1, F, T)
    print(f"Batch targets shape: {batch['targets'].shape}")      # (B, C, F, T)
    print(f"Batch scenes: {batch['scene']}")

    print(f"\n{'='*50}")
    print(f"Dataset pipeline ready for Phase 2!")
    print(f"{'='*50}")
