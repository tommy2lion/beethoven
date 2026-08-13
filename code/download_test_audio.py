"""用 librosa 自带的示例音频或合成音频作为测试文件"""

import sys
# 确保 stdout 不会因为 unicode 崩溃
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)
test_file = SAMPLES_DIR / "test_song.wav"

print("Loading librosa example audio...")
try:
    y, sr = librosa.load(librosa.example('brahms'), sr=22050, duration=15)
    sf.write(test_file, y, sr)
    print(f"[OK] Saved: {test_file} ({len(y)/sr:.1f}s, {sr}Hz)")
except Exception as e:
    print(f"librosa example failed: {e}")
    print("Generating synthetic test audio...")
    sr = 22050
    duration = 10
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 简化的多乐器模拟
    melody = (0.5 * np.sin(2 * np.pi * 440 * t) +
              0.25 * np.sin(2 * np.pi * 880 * t) +
              0.125 * np.sin(2 * np.pi * 1320 * t))
    vibrato = 0.1 * np.sin(2 * np.pi * 5 * t)
    violin = 0.3 * np.sin(2 * np.pi * (660 + vibrato * 50) * t)

    bass = 0.3 * np.sin(2 * np.pi * 110 * t)

    beat = np.zeros_like(t)
    beat[::int(sr // 2)] = 0.8
    beat = np.convolve(beat, np.hanning(256), mode='same')

    mix = melody + violin + bass + beat
    mix = mix / np.max(np.abs(mix)) * 0.9
    sf.write(test_file, mix, sr)
    print(f"[OK] Synthetic test audio: {test_file}")

print(f"\nFile size: {test_file.stat().st_size / 1024:.1f} KB")
