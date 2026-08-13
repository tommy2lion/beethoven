"""
Spectrogram visualization: original mix vs separated stems
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
MIX_FILE = SAMPLES_DIR / "test_song.wav"
SEP_DIR = SAMPLES_DIR / "separated" / "htdemucs" / "test_song"

# Also display audio info
def audio_info(path, label):
    y, sr = librosa.load(path, sr=None)
    duration = len(y) / sr
    print(f"  {label:25s}: {duration:.1f}s, {sr}Hz, "
          f"peak={np.max(np.abs(y)):.3f}, "
          f"rms={np.sqrt(np.mean(y**2)):.4f}")
    return y, sr

def plot_stft(audio_path, title, ax, sr=22050, max_freq=8000):
    y, _ = librosa.load(audio_path, sr=sr)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    mag = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(mag, sr=sr, hop_length=512,
                                    x_axis='s', y_axis='hz',
                                    ax=ax, vmin=-60, vmax=0)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0, max_freq)
    return img

def main():
    if not MIX_FILE.exists():
        print(f"[ERROR] No mix file at {MIX_FILE}")
        return

    print("=" * 50)
    print("Audio file information:")
    print("=" * 50)
    audio_info(MIX_FILE, "Original (mix)")

    stems = {"Original (mix)": MIX_FILE}
    if SEP_DIR.exists():
        for f in sorted(SEP_DIR.iterdir()):
            if f.suffix == '.wav':
                label = f.stem.replace("test_song_", "").title()
                stems[f"Demucs - {label}"] = f
                audio_info(f, f"Demucs - {label}")

    n = len(stems)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n))

    if n == 1:
        axes = [axes]

    print("\nRendering spectrograms...")
    for ax, (title, path) in zip(axes, stems.items()):
        img = plot_stft(path, title, ax)

    fig.colorbar(img, ax=list(axes), label='dB', shrink=0.6, pad=0.02)
    plt.tight_layout()

    out_path = SAMPLES_DIR / "spectrogram_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Spectrogram saved: {out_path}")

    print("\n" + "=" * 50)
    print("Phase 0 checkpoint - what just happened:")
    print("=" * 50)
    print("  1. Demucs (Meta's SOTA model) separated the mix into 4 stems")
    print("  2. The spectrograms show what each stem captured")
    print("  3. Open samples/spectrogram_comparison.png to see the result")
    print()
    print("  Key observations to make:")
    print("  - Which stems sound clean vs. which have leakage?")
    print("  - Are low vs high frequency instruments separated differently?")
    print("  - Does the 'vocals' stem contain anything for this orchestral piece?")
    print("=" * 50)

if __name__ == "__main__":
    main()
