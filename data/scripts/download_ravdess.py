"""
Download RAVDESS emotional speech audio dataset and extract 52-dim features.

Downloads from Zenodo (free, no auth required):
  https://zenodo.org/record/1188976

RAVDESS emotions → Our 5 speech-cognitive-load classes:
  neutral/calm → NORMAL_FOCUSED
  happy/surprise → FAST_ENERGIZED
  sad → SLOW_LABORED
  angry/fearful → ERRATIC_SPEECH
  disgust → RAPID_SCATTERED

Output:
  data/processed/audio_features_real.npy  (N, 52) float32
  data/processed/audio_labels_real.npy    (N,) int64
"""

import os
import sys
import glob
import zipfile
import urllib.request
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAVDESS_DIR = PROJECT_ROOT / "data" / "ravdess"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# RAVDESS emotion → our class mapping
EMOTION_TO_CLASS = {
    1: 2,  # neutral  → NORMAL_FOCUSED
    2: 2,  # calm     → NORMAL_FOCUSED
    3: 3,  # happy    → FAST_ENERGIZED
    4: 1,  # sad      → SLOW_LABORED
    5: 0,  # angry    → ERRATIC_SPEECH
    6: 0,  # fearful  → ERRATIC_SPEECH
    7: 4,  # disgust  → RAPID_SCATTERED
    8: 3,  # surprise → FAST_ENERGIZED
}

CLASS_NAMES = ["ERRATIC_SPEECH", "SLOW_LABORED", "NORMAL_FOCUSED", "FAST_ENERGIZED", "RAPID_SCATTERED"]


def download_ravdess():
    """Download RAVDESS speech audio from Zenodo."""
    RAVDESS_DIR.mkdir(parents=True, exist_ok=True)
    
    if list(RAVDESS_DIR.glob("Actor_*")):
        print("✅ RAVDESS already downloaded")
        return
    
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = RAVDESS_DIR / "Audio_Speech_Actors_01-24.zip"
    
    print("📥 Downloading RAVDESS speech audio dataset (~580MB)...")
    
    def progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        sys.stdout.write(f'\r   {pct:.1f}%')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, str(zip_path), reporthook=progress)
    print(f"\n   ✅ Downloaded")
    
    print("   Extracting...")
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(str(RAVDESS_DIR))
    print(f"   ✅ Extracted to {RAVDESS_DIR}")


def extract_features_from_file(audio_path, sr=16000):
    """Extract 52-dim feature vector using librosa."""
    import librosa
    
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=5.0)
    except Exception as e:
        print(f"   ⚠️ Failed: {audio_path}: {e}")
        return None
    
    if len(y) < sr * 0.5:
        return None
    
    features = np.zeros(52, dtype=np.float32)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    features[0:13] = np.mean(mfccs, axis=1)
    features[13:26] = np.std(mfccs, axis=1)
    
    mfcc_delta = librosa.feature.delta(mfccs)
    features[26:39] = np.mean(mfcc_delta, axis=1)
    
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    features[39:42] = np.mean(mfcc_delta2[:3], axis=1)
    
    features[42] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features[43] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    features[44] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features[45] = float(np.mean(librosa.feature.rms(y=y)))
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    features[46] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    features[47] = float(np.var(pitch_values)) if len(pitch_values) > 0 else 0.0
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features[48] = float(np.squeeze(tempo))
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    duration = len(y) / sr
    if duration > 0 and len(onset_frames) > 1:
        onsets_per_sec = len(onset_frames) / duration
        features[49] = (onsets_per_sec * 60) / 1.5
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        intervals = np.diff(onset_times)
        features[50] = float(np.var(intervals)) if len(intervals) > 0 else 0.0
    
    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    features[51] = float(np.sum(rms_frames < 0.01) / len(rms_frames))
    
    return features


def parse_ravdess_filename(filepath):
    """Parse RAVDESS filename → emotion code."""
    parts = Path(filepath).stem.split('-')
    if len(parts) >= 3:
        return int(parts[2])
    return None


def main():
    download_ravdess()
    
    print("\n🔬 Extracting 52-dim audio features from RAVDESS...")
    audio_files = sorted(glob.glob(str(RAVDESS_DIR / "**" / "*.wav"), recursive=True))
    print(f"   Found {len(audio_files)} audio files")
    
    features_list = []
    labels_list = []
    
    for i, path in enumerate(audio_files):
        emotion = parse_ravdess_filename(path)
        if emotion is None or emotion not in EMOTION_TO_CLASS:
            continue
        
        our_class = EMOTION_TO_CLASS[emotion]
        feats = extract_features_from_file(path)
        
        if feats is not None:
            features_list.append(feats)
            labels_list.append(our_class)
        
        if (i + 1) % 100 == 0:
            print(f"   [{i+1}/{len(audio_files)}] Extracted {len(features_list)} features")
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_DIR / "audio_features_real.npy"), X)
    np.save(str(OUTPUT_DIR / "audio_labels_real.npy"), y)
    
    # Also save as the default audio features for the training script
    np.save(str(OUTPUT_DIR / "audio_features.npy"), X)
    np.save(str(OUTPUT_DIR / "audio_labels.npy"), y)
    
    print(f"\n✅ Extracted {len(features_list)} feature vectors → {OUTPUT_DIR}")
    print(f"   X={X.shape}, y={y.shape}")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == cls_id))
        print(f"   {cls_id} ({cls_name}): {count} samples")


if __name__ == "__main__":
    main()
