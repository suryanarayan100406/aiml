"""
============================================================================
  ANI Flow Optimizer — Audio Model Training (Google Colab)
  Model: XGBoost trained on RAVDESS real speech audio features
  
  HOW TO USE:
    1. Open Google Colab (colab.research.google.com)
    2. GPU is NOT required for this model (CPU is fine)
    3. Paste this entire script into a cell and run
    4. Download the output files from /content/ani_models/
  
  DATASET: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech)
    - 1440 speech audio files from 24 actors
    - 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprise
    - Mapped to our 5 speech-cognitive-load classes
  
  OUTPUT FILES:
    - speech_classifier.onnx
    - speech_classifier.pkl  
    - speech_scaler.pkl
    - audio_metrics.json
    - audio_features_real.npy (extracted features for meta-classifier)
    - audio_labels_real.npy
============================================================================
"""

# ──────────────────────────────────────────────────────────────
# Step 0: Install dependencies
# ──────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("librosa>=0.10.1")
install("xgboost>=2.0.0")
install("scikit-learn>=1.4.0")
install("onnxmltools>=1.12.0")
install("skl2onnx>=1.16.0")
install("onnxruntime>=1.17.0")
install("onnx>=1.15.0")
install("shap>=0.44.0")

import os, json, shutil, zipfile, glob
import numpy as np
import joblib
import librosa
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Step 1: Configuration
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/content/ani_models")
RAVDESS_DIR = Path("/content/ravdess")
OUTPUT_DIR.mkdir(exist_ok=True)

# RAVDESS emotion codes → Our 5 speech-cognitive-load classes
# RAVDESS emotions: 01=neutral, 02=calm, 03=happy, 04=sad,
#                   05=angry, 06=fearful, 07=disgust, 08=surprised
# Our classes: 0=ERRATIC, 1=SLOW_LABORED, 2=NORMAL_FOCUSED, 3=FAST_ENERGIZED, 4=RAPID_SCATTERED
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

print("=" * 60)
print("🎙️  ANI Audio Model — XGBoost Speech Cognitive Load Classifier")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Step 2: Download RAVDESS Dataset
# ──────────────────────────────────────────────────────────────
RAVDESS_DIR.mkdir(parents=True, exist_ok=True)

# RAVDESS has 24 actors, each in a separate zip
RAVDESS_BASE_URL = "https://zenodo.org/record/1188976/files"
ACTOR_ZIPS = [f"Audio_Speech_Actors_01-24.zip"]

zip_path = RAVDESS_DIR / "Audio_Speech_Actors_01-24.zip"

if not list(RAVDESS_DIR.glob("Actor_*")):
    print("\n📥 Downloading RAVDESS speech audio dataset (~580MB)...")
    print("   Source: Zenodo (Ryerson Audio-Visual Database)")
    
    import urllib.request
    url = f"{RAVDESS_BASE_URL}/Audio_Speech_Actors_01-24.zip"
    
    def progress_hook(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        mb = count * block_size / 1024 / 1024
        sys.stdout.write(f'\r   {pct:.1f}% ({mb:.0f}MB)')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, str(zip_path), reporthook=progress_hook)
    print(f"\n   ✅ Downloaded RAVDESS")
    
    print("   Extracting...")
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(str(RAVDESS_DIR))
    print(f"   ✅ Extracted to {RAVDESS_DIR}")
else:
    print(f"✅ RAVDESS already downloaded: {RAVDESS_DIR}")

# ──────────────────────────────────────────────────────────────
# Step 3: Extract 52-dim Audio Features using Librosa
# ──────────────────────────────────────────────────────────────
print(f"\n🔬 Extracting 52-dimensional audio features from RAVDESS...")
print(f"   Feature vector: 13 MFCC means + 13 MFCC stds + 13 MFCC deltas")
print(f"                 + 3 MFCC delta-deltas + spectral + pitch + tempo + WPM + silence")

def extract_features(audio_path, sr=16000):
    """Extract 52-dimensional feature vector from audio file, matching our pipeline."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=5.0)
    except Exception as e:
        print(f"   ⚠️ Failed to load {audio_path}: {e}")
        return None
    
    if len(y) < sr * 0.5:  # Skip files shorter than 0.5s
        return None
    
    features = np.zeros(52, dtype=np.float32)
    
    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    
    # Features 0-12: MFCC means
    features[0:13] = np.mean(mfccs, axis=1)
    
    # Features 13-25: MFCC standard deviations
    features[13:26] = np.std(mfccs, axis=1)
    
    # Features 26-38: MFCC delta means
    mfcc_delta = librosa.feature.delta(mfccs)
    features[26:39] = np.mean(mfcc_delta, axis=1)
    
    # Features 39-41: MFCC delta-delta means (first 3)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    features[39:42] = np.mean(mfcc_delta2[:3], axis=1)
    
    # Feature 42: Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features[42] = np.mean(spectral_centroid)
    
    # Feature 43: Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    features[43] = np.mean(spectral_rolloff)
    
    # Feature 44: Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features[44] = np.mean(zcr)
    
    # Feature 45: RMS energy
    rms = librosa.feature.rms(y=y)
    features[45] = np.mean(rms)
    
    # Feature 46-47: Pitch (F0) mean and variance
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    if len(pitch_values) > 0:
        features[46] = np.mean(pitch_values)
        features[47] = np.var(pitch_values)
    else:
        features[46] = 0.0
        features[47] = 0.0
    
    # Feature 48: Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features[48] = float(np.squeeze(tempo))
    
    # Feature 49-50: WPM proxy and variance
    # Use onset detection as proxy for speech rate / words per minute
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    duration = len(y) / sr
    if duration > 0 and len(onset_frames) > 1:
        onsets_per_sec = len(onset_frames) / duration
        features[49] = (onsets_per_sec * 60) / 1.5  # Approx WPM
        
        # Variance of inter-onset intervals
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        intervals = np.diff(onset_times)
        features[50] = np.var(intervals) if len(intervals) > 0 else 0.0
    else:
        features[49] = 0.0
        features[50] = 0.0
    
    # Feature 51: Silence ratio
    frame_length = 2048
    hop_length = 512
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    silence_threshold = 0.01
    silence_ratio = np.sum(rms_frames < silence_threshold) / len(rms_frames)
    features[51] = silence_ratio
    
    return features


def parse_ravdess_filename(filepath):
    """Parse RAVDESS filename to extract emotion label.
    Format: 03-01-05-01-01-02-12.wav
    Fields: modality-vocal_channel-emotion-intensity-statement-repetition-actor
    """
    parts = Path(filepath).stem.split('-')
    if len(parts) >= 3:
        emotion = int(parts[2])
        return emotion
    return None


# Find all audio files
audio_files = sorted(glob.glob(str(RAVDESS_DIR / "**" / "*.wav"), recursive=True))
print(f"   Found {len(audio_files)} audio files")

# Extract features
all_features = []
all_labels = []
skipped = 0

for audio_path in tqdm(audio_files, desc="   Extracting features"):
    emotion = parse_ravdess_filename(audio_path)
    if emotion is None or emotion not in EMOTION_TO_CLASS:
        skipped += 1
        continue
    
    our_class = EMOTION_TO_CLASS[emotion]
    feats = extract_features(audio_path)
    
    if feats is not None:
        all_features.append(feats)
        all_labels.append(our_class)
    else:
        skipped += 1

X = np.array(all_features, dtype=np.float32)
y = np.array(all_labels, dtype=np.int64)

print(f"\n   ✅ Extracted features: X={X.shape}, y={y.shape}")
print(f"   Skipped: {skipped} files")
print(f"   Class distribution: {dict(Counter(y))}")
for cls_id, cls_name in enumerate(CLASS_NAMES):
    count = np.sum(y == cls_id)
    print(f"     {cls_id} ({cls_name}): {count} samples")

# Save features for meta-classifier training later
np.save(str(OUTPUT_DIR / "audio_features_real.npy"), X)
np.save(str(OUTPUT_DIR / "audio_labels_real.npy"), y)
print(f"   ✅ Saved features to {OUTPUT_DIR}/audio_features_real.npy")

# ──────────────────────────────────────────────────────────────
# Step 4: Handle Class Imbalance 
# ──────────────────────────────────────────────────────────────
print(f"\n⚖️  Handling class imbalance...")

# Check if any class has very few samples — if so, use SMOTE or oversampling
class_counts = Counter(y)
min_count = min(class_counts.values())
max_count = max(class_counts.values())

if max_count / max(min_count, 1) > 3:
    print(f"   High imbalance detected (ratio: {max_count/max(min_count,1):.1f}x)")
    print(f"   Using random oversampling to balance classes...")
    
    # Simple random oversampling
    target_count = max_count
    X_balanced = []
    y_balanced = []
    
    for cls in range(5):
        cls_mask = y == cls
        cls_X = X[cls_mask]
        cls_count = len(cls_X)
        
        if cls_count == 0:
            print(f"   ⚠️ Class {cls} ({CLASS_NAMES[cls]}) has 0 samples! Adding synthetic noise samples.")
            # Generate from class mean of nearest class
            synthetic = np.random.randn(50, 52).astype(np.float32) * np.std(X, axis=0) + np.mean(X, axis=0)
            X_balanced.append(synthetic)
            y_balanced.extend([cls] * 50)
            continue
        
        X_balanced.append(cls_X)
        y_balanced.extend([cls] * cls_count)
        
        if cls_count < target_count:
            # Oversample with small noise
            extra_needed = target_count - cls_count
            indices = np.random.choice(cls_count, extra_needed, replace=True)
            oversampled = cls_X[indices] + np.random.randn(extra_needed, 52).astype(np.float32) * 0.05
            X_balanced.append(oversampled)
            y_balanced.extend([cls] * extra_needed)
    
    X = np.vstack(X_balanced)
    y = np.array(y_balanced, dtype=np.int64)
    
    # Shuffle
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    
    print(f"   ✅ Balanced dataset: X={X.shape}, y={y.shape}")
    print(f"   New class distribution: {dict(Counter(y))}")
else:
    print(f"   Classes are reasonably balanced (ratio: {max_count/max(min_count,1):.1f}x)")

# ──────────────────────────────────────────────────────────────
# Step 5: Train XGBoost Classifier
# ──────────────────────────────────────────────────────────────
print(f"\n🚀 Training XGBoost classifier...")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
)

# 5-fold cross-validation
print(f"\n📊 5-Fold Stratified Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
print(f"   CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train on full data
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

print(f"\n📋 Training Classification Report:")
print(classification_report(y, y_pred, target_names=CLASS_NAMES))

# Feature importance
feature_names = [f"mfcc_mean_{i}" for i in range(13)] + \
                [f"mfcc_std_{i}" for i in range(13)] + \
                [f"mfcc_delta_{i}" for i in range(13)] + \
                ["mfcc_dd_0", "mfcc_dd_1", "mfcc_dd_2"] + \
                ["spectral_centroid", "spectral_rolloff", "zcr", "rms",
                 "pitch_mean", "pitch_var", "tempo", "wpm_mean", "wpm_var", "silence_ratio"]

importances = model.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]
print(f"\n🔍 Top 10 Feature Importances:")
for idx in top_indices:
    print(f"   {feature_names[idx]:25s}: {importances[idx]:.4f}")

# SHAP explainability
try:
    import shap
    print(f"\n🔬 Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled[:100])
    print(f"   ✅ SHAP values computed for 100 samples")
except Exception as e:
    print(f"   ⚠️ SHAP skipped: {e}")

# ──────────────────────────────────────────────────────────────
# Step 6: Save Models
# ──────────────────────────────────────────────────────────────
print(f"\n💾 Saving models...")

joblib.dump(model, str(OUTPUT_DIR / "speech_classifier.pkl"))
joblib.dump(scaler, str(OUTPUT_DIR / "speech_scaler.pkl"))
print(f"   ✅ speech_classifier.pkl saved")
print(f"   ✅ speech_scaler.pkl saved")

# ──────────────────────────────────────────────────────────────
# Step 7: Export to ONNX
# ──────────────────────────────────────────────────────────────
print(f"\n📦 Exporting to ONNX...")

onnx_path = str(OUTPUT_DIR / "speech_classifier.onnx")
exported = False

# Method 1: onnxmltools (best for XGBoost)
try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as FTT
    
    onnx_model = convert_xgboost(
        model,
        initial_types=[("input", FTT([None, X_scaled.shape[1]]))]
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"   ✅ ONNX exported via onnxmltools: {onnx_path}")
    exported = True
except Exception as e1:
    print(f"   ⚠️ onnxmltools failed: {e1}")

# Method 2: skl2onnx fallback
if not exported:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        onnx_model = convert_sklearn(
            model, "speech_classifier",
            [("input", FloatTensorType([None, X_scaled.shape[1]]))]
        )
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"   ✅ ONNX exported via skl2onnx: {onnx_path}")
        exported = True
    except Exception as e2:
        print(f"   ⚠️ skl2onnx also failed: {e2}")

if not exported:
    print(f"   ❌ ONNX export failed. Only .pkl saved.")
    print(f"   The frontend will use the demo audio classifier as fallback.")

# ──────────────────────────────────────────────────────────────
# Step 8: Save Metrics & Verify
# ──────────────────────────────────────────────────────────────
metrics = {
    "cv_f1_macro_mean": float(cv_scores.mean()),
    "cv_f1_macro_std": float(cv_scores.std()),
    "training_accuracy": float(np.mean(y_pred == y)),
    "training_f1_macro": float(f1_score(y, y_pred, average='macro')),
    "dataset": "RAVDESS",
    "total_samples": int(len(y)),
    "class_distribution": {CLASS_NAMES[i]: int(np.sum(y == i)) for i in range(5)},
    "class_names": CLASS_NAMES,
    "feature_vector_dim": 52,
    "onnx_exported": exported,
}

metrics_path = OUTPUT_DIR / "audio_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

# Quick ONNX inference test
if exported:
    print(f"\n🧪 Quick ONNX inference test...")
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 52).astype(np.float32)
    output = session.run(None, {input_name: dummy})
    print(f"   Input: {input_name} [1, 52]")
    print(f"   Output: label={output[0]}, probabilities shape={output[1].shape if len(output) > 1 else 'N/A'}")
    print(f"   ✅ ONNX inference works!")

print(f"\n✅ AUDIO MODEL TRAINING COMPLETE!")
print(f"   CV F1 (macro): {metrics['cv_f1_macro_mean']:.4f} ± {metrics['cv_f1_macro_std']:.4f}")
print(f"   Training Accuracy: {metrics['training_accuracy']:.4f}")
print(f"\n   Output files in: {OUTPUT_DIR}")
print(f"   - speech_classifier.onnx")
print(f"   - speech_classifier.pkl")
print(f"   - speech_scaler.pkl")
print(f"   - audio_metrics.json")
print(f"   - audio_features_real.npy")
print(f"   - audio_labels_real.npy")

print("\n" + "=" * 60)
print("🎉 Audio model ready! Download files from /content/ani_models/")
print("=" * 60)
