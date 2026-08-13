"""
Approach 2: Generative Source Separation (构想二)
=================================================
"逐乐器重演"—— Hybrid pipeline:

  1. Demucs 粗分 → 4/6 个独立声轨
  2. 对每个声轨做音高检测（现在是单音，容易了）
  3. 用目标乐器音色重新合成
  4. 对比重演结果与原轨，筛选置信度高的片段
  5. 输出：重演音频 + MIDI 乐谱

This is fundamentally different from pure masking:
  - Demucs: "在频谱上挖出属于你的部分"
  - 构想二: "我听了你的旋律，用我的音色重新弹一遍"
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import subprocess, sys
import warnings
warnings.filterwarnings('ignore')

import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SR = 22050
HOP_LENGTH = 256

# ========== 1. Instrument Synthesizer ==========

def synthesize_note(instrument, freq, duration, sr=SR):
    """
    Synthesize a single note with instrument-specific timbre.
    This is the core of "用你的音色重新演奏"。
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if len(t) == 0:
        return np.array([])

    params = INSTRUMENT_PARAMS.get(instrument, INSTRUMENT_PARAMS['piano'])
    harmonics = params['harmonics']
    weights = np.array(params['weights'])
    decay = params['decay']
    has_vibrato = params.get('vibrato', False)

    if has_vibrato:
        vibrato = 0.05 * np.sin(2 * np.pi * 5.5 * t)
        mod_freq = freq * (1 + vibrato)
        phase = 2 * np.pi * np.cumsum(mod_freq) / sr
        note = np.zeros_like(t)
        for h, w in zip(harmonics, weights):
            note += w * np.sin(h * phase)
    else:
        note = np.zeros_like(t)
        for h, w in zip(harmonics, weights):
            note += w * np.sin(2 * np.pi * freq * h * t)

    # Envelope
    if instrument == 'piano':
        envelope = np.exp(-t * decay)  # fast decay
    elif instrument == 'drums':
        envelope = np.exp(-t * 20) + np.random.randn(len(t)) * 0.01  # noise burst
    else:
        attack = np.minimum(1.0, t / 0.02)
        envelope = attack * np.exp(-t * decay)

    note = note * envelope
    peak = np.max(np.abs(note))
    return note / peak * 0.3 if peak > 0 else note


INSTRUMENT_PARAMS = {
    'piano':  {'harmonics': [1,2,3,4,5,6,8], 'weights': [1,.6,.3,.15,.08,.04,.02], 'decay': 5},
    'guitar': {'harmonics': [1,2,3,4,5],      'weights': [1,.4,.2,.1,.05],        'decay': 4},
    'bass':   {'harmonics': [1,3],            'weights': [1,.2],                   'decay': 3},
    'vocals': {'harmonics': [1,2,3,4,5],      'weights': [1,.3,.1,.05,.02],       'decay': 2, 'vibrato': True},
    'violin': {'harmonics': [1,2,3,4],        'weights': [1,.5,.2,.08],            'decay': 1.2, 'vibrato': True},
    'flute':  {'harmonics': [1,2,3],          'weights': [1,.2,.05],               'decay': 0.5},
    'drums':  {'harmonics': [1,2,3],          'weights': [1,.5,.25],               'decay': 20},
}


# ========== 2. Note Detection on a Stem ==========

def detect_notes_stem(audio_path, min_freq=60, max_freq=2000):
    """
    Fast note detection using spectral peak picking.
    Much faster than pyin, works on polyphonic audio.
    Returns [(start_s, end_s, freq_hz, confidence), ...]
    """
    y, sr = librosa.load(audio_path, sr=SR)

    # Check if stem has meaningful content
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    if np.max(rms) < 0.001:
        return [], y, sr

    # STFT
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.times_like(D, sr=sr, hop_length=HOP_LENGTH)

    # Onset detection for note segmentation
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH,
                                         backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=HOP_LENGTH)

    if len(onset_times) < 2:
        # Use uniform segmentation
        seg_len = int(sr * 0.2 / HOP_LENGTH)  # 200ms per segment
        onset_times = times[::max(1, seg_len)]

    # For each segment, find dominant pitch(es)
    notes = []
    f_min_bin = max(0, np.argmax(freqs >= min_freq) - 1)
    f_max_bin = min(len(freqs), np.argmax(freqs >= max_freq) + 1)

    for i in range(len(onset_times)):
        start = onset_times[i]
        end = onset_times[i+1] if i+1 < len(onset_times) else times[-1]

        sf = max(0, int(start * sr / HOP_LENGTH))
        ef = min(D.shape[1], int(end * sr / HOP_LENGTH) + 1)
        if ef - sf < 1:
            continue

        # Average spectrum in this segment
        seg_spec = np.mean(D[f_min_bin:f_max_bin, sf:ef], axis=1)

        # Find spectral peaks
        from scipy.signal import find_peaks
        peaks, props = find_peaks(seg_spec, height=np.max(seg_spec)*0.1,
                                  distance=3, prominence=0.02)

        if len(peaks) > 0:
            # Take the strongest peak
            strongest = peaks[np.argmax(props['peak_heights'])]
            pitch = freqs[f_min_bin + strongest]
            confidence = props['peak_heights'][np.argmax(props['peak_heights'])] / (np.max(seg_spec) + 1e-10)

            if pitch > 60:
                notes.append((start, end, pitch, min(1.0, confidence)))

    # Group consecutive similar-pitch notes
    if len(notes) > 1:
        grouped = [notes[0]]
        for n in notes[1:]:
            if (abs(n[2] - grouped[-1][2]) / grouped[-1][2] < 0.05
                and n[0] - grouped[-1][1] < 0.05):
                # Same pitch, merge
                grouped[-1] = (grouped[-1][0], n[1], grouped[-1][2],
                              max(grouped[-1][3], n[3]))
            else:
                grouped.append(n)
        notes = grouped

    return notes, y, sr


# ========== 3. Resynthesis + Confidence Filter ==========

def resynthesize_stem(notes, instrument, duration, sr=SR):
    """Re-perform the notes using target instrument timbre."""
    n_s = int(duration * sr)
    track = np.zeros(n_s)
    for start, end, pitch, conf in notes:
        d = min(end - start, 3.0)
        if d < 0.05:
            continue
        note = synthesize_note(instrument, pitch, d, sr)
        ss = int(start * sr)
        note = note[:min(len(note), n_s - ss)]
        track[ss:ss+len(note)] += note * conf * 0.8
    return track


def confidence_filter(track, original, threshold=0.2, window_s=0.1, sr=SR):
    """Keep only segments where resynthesis matches original."""
    ws = int(window_s * sr)
    n = min(len(track), len(original))
    out = np.zeros(n)

    for s in range(0, n - ws, ws // 2):
        e = min(s + ws, n)
        t_seg = track[s:e]
        o_seg = original[s:e]
        rt = np.sqrt(np.mean(t_seg**2) + 1e-10)
        ro = np.sqrt(np.mean(o_seg**2) + 1e-10)

        if rt > 0.001 and ro > 0.001:
            corr = np.sum(t_seg * o_seg) / (rt * ro * len(t_seg))
            vr = min(rt/ro, ro/rt)
            if corr * vr > threshold:
                out[s:e] = t_seg[:e-s]
    return out


# ========== 4. MIDI export ==========

def save_midi(notes, instrument, output_path, sr=SR):
    """Export detected notes as a MIDI file using simple byte-level writing."""
    try:
        # Use midiutil with a more robust approach
        from midiutil import MIDIFile

        GM_PROGRAMS = {
            'piano': 0, 'guitar': 25, 'bass': 34,
            'violin': 41, 'flute': 73, 'vocals': 54, 'drums': 0,
            'other': 1,
        }

        mf = MIDIFile(1, deinterleave=False)
        mf.addTrackName(0, 0, instrument)
        mf.addTempo(0, 0, 120)
        mf.addProgramChange(0, 0, 0, GM_PROGRAMS.get(instrument, 0))

        for start, end, pitch, conf in notes:
            midi_note = int(round(librosa.hz_to_midi(pitch)))
            midi_note = max(0, min(127, midi_note))
            duration = max(0.05, end - start)
            volume = min(127, int(conf * 80 + 30))
            try:
                mf.addNote(0, 0, midi_note, start, duration, volume)
            except Exception:
                pass

        with open(output_path, 'wb') as f:
            mf.writeFile(f)
        return True
    except Exception as e:
        print(f"    MIDI export failed: {e}")
        return False


# ========== Main Pipeline ==========

def main():
    print("=" * 60)
    print("Approach 2: Generative Source Separation")
    print("=" * 60)
    print()
    print("Pipeline: Demucs → Note Detection → Resynthesis → Filter → MIDI")
    print()

    test_file = Path(__file__).parent.parent / "BEYOND+-+光辉岁月.mp3"
    if not test_file.exists():
        print("[ERROR] No test file found")
        return

    # Step 0: Ensure Demucs 6-stem is available
    demucs_dir = (Path(__file__).parent.parent / "samples" / "separated"
                  / "htdemucs_6s" / "BEYOND+-+光辉岁月")
    if not demucs_dir.exists():
        print("Running Demucs 6-stem first...")
        subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs_6s",
                        "-o", str(Path(__file__).parent.parent / "samples" / "separated"),
                        str(test_file)], capture_output=True)

    instruments = ['vocals', 'bass', 'drums', 'guitar', 'piano', 'other']
    output_dir = Path(__file__).parent.parent / "samples" / "separated" / "approach2"
    midi_dir = output_dir / "midi"
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(exist_ok=True)

    # Load original mix for confidence comparison
    y_mix, _ = librosa.load(test_file, sr=SR)
    duration = len(y_mix) / SR

    results = {}
    for inst in instruments:
        stem_path = demucs_dir / f"{inst}.wav"
        if not stem_path.exists():
            continue

        print(f"\n[{inst}]")
        print(f"  Step 1: Detecting notes from Demucs stem...")

        # Step 1: Note detection on the Demucs stem
        notes, y_stem, sr = detect_notes_stem(stem_path)
        print(f"  → {len(notes)} notes detected")

        if len(notes) == 0:
            results[inst] = {'notes': 0, 'rms': 0}
            continue

        # Show a few notes
        for n in notes[:3]:
            note_name = librosa.hz_to_note(n[2])
            print(f"    {note_name:6s}  {n[2]:.0f}Hz  t={n[0]:.2f}-{n[1]:.2f}s  conf={n[3]:.2f}")

        # Step 2: Resynthesize
        print(f"  Step 2: Resynthesizing with {inst} timbre...")
        track = resynthesize_stem(notes, inst, duration, sr)
        rms_raw = np.sqrt(np.mean(track**2))
        print(f"  → Raw RMS = {rms_raw:.4f}")

        # Step 3: Confidence filter against original mix
        print(f"  Step 3: Confidence filtering...")
        filtered = confidence_filter(track, y_mix, threshold=0.15)

        # Also filter against Demucs stem
        filtered_vs_stem = confidence_filter(track, y_stem, threshold=0.15)

        rms_filt = np.sqrt(np.mean(filtered**2))
        rms_filt_s = np.sqrt(np.mean(filtered_vs_stem**2))

        # Save
        sf.write(output_dir / f"approach2_{inst}.wav", filtered, sr)
        sf.write(output_dir / f"approach2_{inst}_vs_stem.wav", filtered_vs_stem, sr)

        # Step 4: Export MIDI
        midi_path = midi_dir / f"{inst}.mid"
        midi_ok = save_midi(notes, inst, midi_path)
        if midi_ok:
            print(f"  → MIDI saved: {midi_path.name}")

        print(f"  → Final RMS:      {rms_filt:.4f}")
        print(f"  → vs Demucs stem: {rms_filt_s:.4f}")
        results[inst] = {'notes': len(notes), 'rms': rms_filt}

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Instrument':<12} {'Notes':<8} {'RMS':<10}")
    print("-" * 30)
    for inst, r in results.items():
        print(f"{inst:<12} {r['notes']:<8} {r['rms']:<10.4f}")
    print()
    print(f"MIDI files: {midi_dir}")
    print(f"Audio files: {output_dir}")
    print(f"\nKey difference from Demucs:")
    print(f"  Demucs:    filters the spectrum")
    print(f"  Approach 2: transcribes → resynthesizes")
    print(f"  This approach produces MIDI as a byproduct!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
