"""
Phase 0 - First Separation Experience
======================================
Run Meta's Demucs (SOTA model) on real music,
then visualize and listen to the separated stems.
"""

import subprocess, os, sys
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
OUTPUT_DIR = SAMPLES_DIR / "separated"
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use the .wav file we generated
test_file = SAMPLES_DIR / "test_song.wav"

if not test_file.exists():
    print(f"[ERROR] No test file found at {test_file}")
    print("Run download_test_audio.py first.")
    sys.exit(1)

print(f"Input: {test_file}")
print(f"Size: {test_file.stat().st_size / 1024:.1f} KB")
print()

# Run Demucs
print("=" * 50)
print("Running Demucs (htdemucs model)...")
print("=" * 50)

result = subprocess.run([
    sys.executable, "-m", "demucs",
    "--two-stems=vocals",        # separate into vocals + accompaniment
    "-o", str(OUTPUT_DIR),
    str(test_file)
], capture_output=True, text=True, encoding='utf-8', errors='replace')

# Check output directory for the model used
model_name = "htdemucs"
for d in OUTPUT_DIR.iterdir():
    if d.is_dir():
        model_name = d.name
        break

# Check results
separated_dir = OUTPUT_DIR / model_name / test_file.stem
if separated_dir.exists():
    print(f"\n[OK] Demucs finished! Output in: {separated_dir}")
    print("\nGenerated files:")
    for f in sorted(separated_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:<30s} {size_mb:.1f} MB")
else:
    print("STDOUT:", result.stdout[:1000])
    print("STDERR:", result.stderr[:1000])
    print("\n[ERROR] No output found. See above.")

print()
print("=" * 50)
print("What to do next:")
print("  1. Listen to the separated files")
print("  2. Run: python code/visualize.py  (see spectrograms)")
print("=" * 50)
