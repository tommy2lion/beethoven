"""
Phase 1 - Step 1: Synthesize multi-instrument training data
============================================================
Generate controlled mixtures where we know the exact ground truth.

Three synthetic "instruments" with distinct timbres:
  - Piano-like:  rich harmonics, percussive decay
  - Violin-like:  sustained, vibrato
  - Bass:         pure low frequencies

This lets us test separation algorithms against known ground truth.
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SR = 22050  # sample rate
DURATION = 8.0  # seconds per sample
DATA_DIR = Path(__file__).parent.parent / "samples" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Instrument generators ----------

def piano_note(freq, duration, sr=SR):
    """Piano-like: harmonics with exponential decay."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    harmonics = [1, 2, 3, 4, 5, 6, 8]
    weights = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.01]
    note = np.zeros_like(t)
    for h, w in zip(harmonics, weights):
        note += w * np.sin(2 * np.pi * freq * h * t)
    # Percussive decay (fast attack, exponential decay)
    envelope = np.exp(-t * 6)  # short sustain
    note *= envelope
    return note / np.max(np.abs(note) + 1e-10) * 0.3

def violin_note(freq, duration, sr=SR):
    """Violin-like: sustained with vibrato."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Slow attack
    attack = np.minimum(1.0, t / 0.05)
    # Vibrato: frequency modulation at ~5-6 Hz
    vibrato = 0.05 * np.sin(2 * np.pi * 5.5 * t)
    modulated = freq * (1 + vibrato)
    # Phase accumulation for FM synthesis
    phase = 2 * np.pi * np.cumsum(modulated) / sr
    # Rich harmonics but softer than piano
    harmonics = [1, 2, 3, 4, 5]
    weights = [1.0, 0.4, 0.15, 0.05, 0.02]
    note = np.zeros_like(t)
    for h, w in zip(harmonics, weights):
        note += w * np.sin(h * phase)
    # Sustain envelope (slow decay)
    envelope = attack * np.exp(-t * 1.5)
    note *= envelope
    return note / np.max(np.abs(note) + 1e-10) * 0.3

def bass_note(freq, duration, sr=SR):
    """Bass: simple low sine with slight harmonics."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mostly fundamental + a bit of 3rd harmonic
    note = np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    envelope = np.exp(-t * 3)
    note *= envelope
    return note / np.max(np.abs(note) + 1e-10) * 0.3

# ---------- Note mapping ----------
# Simple melodies to make it sound musical
NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
    'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99,
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
    'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
}

# ---------- Scenes ----------
SCENES = [
    {  # Scene 0: Simple - piano plays C major scale, violin holds one note
        'piano':  [('C4', 0.0), ('D4', 0.5), ('E4', 1.0), ('F4', 1.5),
                   ('G4', 2.0), ('A4', 2.5), ('B4', 3.0), ('C5', 3.5)],
        'violin': [('G4', 0.0)],  # held note
        'bass':   [('C3', 0.0), ('G3', 2.0), ('C3', 4.0)],
    },
    {  # Scene 1: Two instruments playing, bass supporting
        'piano':  [('E4', 0.0), ('G4', 0.5), ('A4', 1.0), ('G4', 1.5),
                   ('E4', 2.0), ('F4', 2.5), ('G4', 3.0), ('C5', 3.5)],
        'violin': [('C5', 0.0), ('B4', 1.5), ('A4', 3.0)],
        'bass':   [('C3', 0.0), ('E3', 1.5), ('G3', 3.0)],
    },
    {  # Scene 2: Dense - all three active, overlapping
        'piano':  [('C4', 0.0), ('E4', 0.3), ('G4', 0.6), ('C5', 0.9),
                   ('D4', 2.0), ('F4', 2.3), ('A4', 2.6), ('D5', 2.9)],
        'violin': [('G4', 0.0), ('A4', 1.0), ('G4', 2.0), ('F4', 3.0), ('E4', 4.0)],
        'bass':   [('C3', 0.0), ('G3', 1.0), ('A3', 2.0), ('F3', 3.0), ('E3', 4.0)],
    },
    {  # Scene 3: Busy
        'piano':  [('C4', 0.0), ('E4', 0.4), ('G4', 0.8), ('B4', 1.2),
                   ('C5', 1.6), ('E5', 2.0), ('G5', 2.4), ('E5', 2.8),
                   ('C5', 3.2), ('G4', 3.6), ('E4', 4.0), ('C4', 4.4)],
        'violin': [('C5', 0.0), ('G4', 1.0), ('E4', 2.0), ('G4', 3.0), ('C5', 4.0)],
        'bass':   [('C3', 0.0), ('G3', 1.2), ('E3', 2.4), ('G3', 3.6)],
    },
]

# ---------- Helper ----------
NOTE_DUR = 0.5  # seconds per note
INSTRUMENTS = ['piano', 'violin', 'bass']
NOTE_LEN = int(NOTE_DUR * SR)

def render_scene(scene, idx):
    """Render a scene to mixture + individual stems."""
    n_samples = int(DURATION * SR)
    mix = np.zeros(n_samples)
    stems = {}

    for inst_name in INSTRUMENTS:
        track = np.zeros(n_samples)
        if inst_name in scene:
            for note_name, start_time in scene[inst_name]:
                start_sample = int(start_time * SR)
                if start_sample >= n_samples:
                    continue
                freq = NOTE_FREQS[note_name]
                # Generate note with appropriate instrument generator
                if inst_name == 'piano':
                    note = piano_note(freq, NOTE_DUR)
                elif inst_name == 'violin':
                    note = violin_note(freq, NOTE_DUR)
                elif inst_name == 'bass':
                    note = bass_note(freq, NOTE_DUR)
                # Trim or pad to fit
                end = min(start_sample + len(note), n_samples)
                note = note[:end - start_sample]
                track[start_sample:start_sample + len(note)] += note
        stems[inst_name] = track
        mix += track

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix /= peak * 1.1
        for inst in stems:
            stems[inst] /= peak * 1.1

    return mix, stems

def main():
    print("=" * 50)
    print("Phase 1 - Synthetic Data Generation")
    print("=" * 50)

    for i, scene in enumerate(SCENES):
        print(f"\nScene {i}: ", end='')
        mix, stems = render_scene(scene, i)

        # Save mixture
        sf.write(DATA_DIR / f"scene_{i:02d}_mix.wav", mix, SR)
        print(f"mix ", end='')

        # Save individual stems
        for inst in INSTRUMENTS:
            sf.write(DATA_DIR / f"scene_{i:02d}_{inst}.wav", stems[inst], SR)
            print(f"{inst} ", end='')

        # Quick stats
        rms_mix = np.sqrt(np.mean(mix**2))
        active_insts = [inst for inst in INSTRUMENTS if np.sqrt(np.mean(stems[inst]**2)) > 0.001]
        print(f"  [{', '.join(active_insts)}, {rms_mix:.3f} RMS]")

    print(f"\n{'='*50}")
    print(f"Generated {len(SCENES)} scenes in {DATA_DIR}")
    print(f"Each scene: mix.wav + 3 individual instrument tracks")
    print(f"{'='*50}")

    # Generate a spectrogram preview
    print("\nGenerating spectrogram preview...")
    import matplotlib.pyplot as plt
    import librosa.display

    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    scene_idx = 1  # Show scene 1

    mix, _ = librosa.load(DATA_DIR / f"scene_{scene_idx:02d}_mix.wav", sr=SR)
    titles = ['Full Mix'] + INSTRUMENTS

    for ax, title in zip(axes, titles):
        if title == 'Full Mix':
            y = mix
        else:
            y, _ = librosa.load(DATA_DIR / f"scene_{scene_idx:02d}_{title}.wav", sr=SR)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=SR, hop_length=512,
                                        x_axis='s', y_axis='hz', ax=ax)
        ax.set_title(title)
        ax.set_ylim(0, 4000)

    fig.colorbar(img, ax=axes, label='dB', shrink=0.6)
    plt.tight_layout()
    preview_path = DATA_DIR / "scene_preview.png"
    plt.savefig(preview_path, dpi=150)
    print(f"Saved: {preview_path}")

if __name__ == "__main__":
    main()
