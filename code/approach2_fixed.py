"""
Approach 2 - Generous version
==============================
The core idea is correct (transcribe → resynthesize), but with
our basic sine-wave synth, the audio will never sound realistic.

Strategy: Output TWO versions for each instrument:
  A) Raw synth:  What our synth sounds like (recognizeable notes, robotic timbre)
  B) Hybrid:     Demucs stem volume-gated by our synth energy
                 (uses real recorded audio, shaped by our detection)

This is closest to your original Bilibili-inspired idea:
  "用钢琴演奏一遍 → 删掉不像的 → 拼起来"
Here: "用我们的合成器检测一遍 → 用Demucs的真实音色替换合成器部分"
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.signal import find_peaks
import warnings; warnings.filterwarnings('ignore')
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SR = 22050; HOP = 256

INST_PARAMS = {
    'piano':{'harm':[1,2,3,4,5,6,8],'wt':[1,.6,.3,.15,.08,.04,.02],'decay':5},
    'guitar':{'harm':[1,2,3,4,5],'wt':[1,.4,.2,.1,.05],'decay':4},
    'bass':{'harm':[1,3],'wt':[1,.2],'decay':3},
    'vocals':{'harm':[1,2,3,4,5],'wt':[1,.3,.1,.05,.02],'decay':2,'vib':True},
    'drums':{'harm':[1,2,3],'wt':[1,.5,.25],'decay':20},
}

def synth(instr, freq, dur):
    t = np.linspace(0, dur, int(SR*dur), endpoint=False)
    if len(t)==0: return np.array([])
    p = INST_PARAMS.get(instr, INST_PARAMS['piano'])
    if p.get('vib'):
        vib = 0.05*np.sin(2*np.pi*5.5*t)
        ph = 2*np.pi*np.cumsum(freq*(1+vib))/SR
        note = sum(w*np.sin(h*ph) for h,w in zip(p['harm'],p['wt']))
    else:
        note = sum(w*np.sin(2*np.pi*freq*h*t) for h,w in zip(p['harm'],p['wt']))
    atk = np.minimum(1.0, t/0.02) if instr!='piano' else 1.0
    env = atk * np.exp(-t*p['decay'])
    if instr=='drums': env += np.random.randn(len(t))*0.01
    note *= env; pk = np.max(np.abs(note))
    return note/pk*0.3 if pk>0 else note


def detect_dense(y_stem):
    """Dense note detection."""
    D = np.abs(librosa.stft(y_stem, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    times = librosa.times_like(D, sr=SR, hop_length=HOP)

    onsets = librosa.onset.onset_detect(y=y_stem, sr=SR, hop_length=HOP,
                                         backtrack=True, wait=1)
    ot = librosa.frames_to_time(onsets, sr=SR, hop_length=HOP)
    if len(ot) < 2:
        ot = times[::max(1, int(0.15*SR/HOP))]

    notes = []
    fb = max(0, np.argmax(freqs>=60)-1)
    fe = min(len(freqs), np.argmax(freqs>=2000)+1)

    for i in range(len(ot)):
        sf_ = max(0, int(ot[i]*SR/HOP))
        ef = min(D.shape[1], int((ot[i+1] if i+1<len(ot) else times[-1])*SR/HOP)+1)
        if ef-sf_ < 1: continue
        seg = np.mean(D[fb:fe, sf_:ef], axis=1)
        peaks, props = find_peaks(seg, height=np.max(seg)*0.05, distance=2)
        for idx in peaks:
            pitch = freqs[fb+idx]
            conf = props['peak_heights'][peaks.tolist().index(idx)]/(np.max(seg)+1e-10)
            if pitch > 60 and conf > 0.05:
                notes.append((ot[i], ot[i+1] if i+1<len(ot) else times[-1], pitch, min(1,conf)))

    if notes:
        notes.sort()
        merged = [notes[0]]
        for n in notes[1:]:
            if abs(n[2]-merged[-1][2])/merged[-1][2] < 0.06 and n[0]-merged[-1][1] < 0.08:
                merged[-1] = (merged[-1][0], max(merged[-1][1],n[1]), merged[-1][2], max(merged[-1][3],n[3]))
            else:
                merged.append(n)
        notes = merged
    return notes


def resynth_raw(notes, inst, duration):
    """Raw resynthesis (robotic but complete - no silence gaps)."""
    n_s = int(duration * SR)
    track = np.zeros(n_s)
    for start, end, pitch, conf in notes:
        d = min(end-start, 3.0)
        if d < 0.05: continue
        note = synth(inst, pitch, d)
        ss = int(start*SR)
        note = note[:min(len(note), n_s-ss)]
        if len(note):
            track[ss:ss+len(note)] += note * conf * 0.6
    # Normalize to a reasonable level
    pk = np.max(np.abs(track))
    if pk > 0: track /= pk * 1.5
    return track


def hybrid_with_demucs(raw_synth, demucs_stem, blend=0.3):
    """
    Hybrid: blend raw synth with Demucs stem.
    Where synth has energy → mostly use Demucs stem (real sound)
    Where synth is silent → fade out Demucs stem (reducing interference)
    """
    n = min(len(raw_synth), len(demucs_stem))
    out = np.zeros(n)
    ws = int(0.05 * SR)  # 50ms window

    for s in range(0, n-ws, ws//2):
        e = min(s+ws, n)
        rt = np.sqrt(np.mean(raw_synth[s:e]**2)+1e-10)
        rd = np.sqrt(np.mean(demucs_stem[s:e]**2)+1e-10)

        # Energy-based blend: more synth energy → more Demucs
        synth_energy = np.clip(rt * 10, 0, 1)
        # Blend: mostly Demucs where synth detected, fade otherwise
        weight = synth_energy * (1 - blend) + blend * 0.1
        out[s:e] = demucs_stem[s:e] * weight

    return out


def main():
    print("="*50)
    print("Approach 2 - Two-output version")
    print("="*50)

    song = 'ヨルシカ-花に亡霊'
    song_path = Path(__file__).parent.parent / f"{song}.mp3"
    demucs_dir = Path(__file__).parent.parent / "samples" / "separated" / "htdemucs_6s" / song

    out_dir = Path(__file__).parent.parent / "samples" / "separated" / "approach2_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    instruments = ['vocals','bass','drums','guitar','piano','other']
    y_mix,_ = librosa.load(song_path, sr=SR)
    duration = len(y_mix)/SR

    for inst in instruments:
        print(f"\n--- {inst} ---")
        y_stem,_ = librosa.load(demucs_dir/f"{inst}.wav", sr=SR)
        stem_rms = np.sqrt(np.mean(y_stem**2))

        if stem_rms < 0.001:
            print(f"  Silent, skip")
            continue

        # 1. Detect notes (from Demucs stem)
        notes = detect_dense(y_stem)
        print(f"  Notes: {len(notes)}")

        # 2. Raw synth (A) - robotic but all notes present
        raw = resynth_raw(notes, inst, duration)
        sf.write(out_dir/f"synth_{inst}.wav", raw, SR)
        print(f"  Raw synth RMS: {np.sqrt(np.mean(raw**2)):.4f}")

        # 3. Hybrid blend (B) - uses real audio from Demucs
        hybrid = hybrid_with_demucs(raw, y_stem)
        sf.write(out_dir/f"hybrid_{inst}.wav", hybrid, SR)
        hybrid_rms = np.sqrt(np.mean(hybrid**2))
        print(f"  Hybrid RMS: {hybrid_rms:.4f}  (Demucs: {stem_rms:.4f})")

        # Note range
        pitches = [n[2] for n in notes]
        if pitches:
            print(f"  Range: {librosa.hz_to_note(min(pitches))} - {librosa.hz_to_note(max(pitches))}")
        else:
            print(f"  No notes detected")

    print(f"\nSaved to {out_dir}")
    print()
    print("Two versions per instrument:")
    print("  synth_*.wav  - Pure MIDI-like synthesis (robotic, all notes)")
    print("  hybrid_*.wav - Demucs audio shaped by our detection")
    print()
    print("Listen to synth first (hear the notes clearly),")
    print("then hybrid (hear them with real instrument timbre).")


if __name__ == "__main__":
    main()
