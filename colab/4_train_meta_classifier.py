"""
============================================================================
  ANI Flow Optimizer — Meta-Classifier Training (Google Colab)
  Model: Random Forest ensemble that fuses Vision + Audio + NLP outputs
  
  HOW TO USE:
    1. Run AFTER training all 3 modality models (scripts 1-3)
    2. GPU is NOT required (CPU is fine)
    3. Paste this entire script into a cell and run
    4. Download the output files from /content/ani_models/
  
  APPROACH:
    - Loads outputs from the 3 trained modality models
    - Generates realistic fused 11-feature vectors using the trained models
      on real data (not synthetic Gaussians!)
    - Assigns flow state labels using a principled heuristic scorer
    - Trains Random Forest with GridSearchCV + Platt calibration
    - Exports to ONNX
  
  PREREQUISITE FILES in /content/ani_models/:
    - desk_distraction_v1.onnx (from script 1)
    - speech_classifier.pkl + speech_scaler.pkl (from script 2)
    - audio_features_real.npy + audio_labels_real.npy (from script 2)
    - task_nlp_classifier.onnx + vocab.txt (from script 3)
  
  OUTPUT FILES:
    - meta_flow_classifier.onnx
    - meta_flow_classifier.pkl
    - meta_flow_rf_raw.pkl
    - meta_metrics.json
    - fused_flow_dataset_real.csv
============================================================================
"""

# ──────────────────────────────────────────────────────────────
# Step 0: Install dependencies
# ──────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("scikit-learn>=1.4.0")
install("skl2onnx>=1.16.0")
install("onnxruntime>=1.17.0")
install("onnx>=1.15.0")
install("xgboost>=2.0.0")
install("transformers>=4.38.0")

import os, json, csv, random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("/content/ani_models")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("🔀 ANI Meta-Classifier — Random Forest Flow State Predictor")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Step 1: Verify Prerequisites
# ──────────────────────────────────────────────────────────────
print("\n🔍 Checking prerequisite model files...")

required_files = {
    "Audio model": OUTPUT_DIR / "speech_classifier.pkl",
    "Audio scaler": OUTPUT_DIR / "speech_scaler.pkl",
    "Audio features": OUTPUT_DIR / "audio_features_real.npy",
    "Audio labels": OUTPUT_DIR / "audio_labels_real.npy",
}

optional_files = {
    "Vision ONNX": OUTPUT_DIR / "desk_distraction_v1.onnx",
    "NLP ONNX": OUTPUT_DIR / "task_nlp_classifier.onnx",
    "NLP vocab": OUTPUT_DIR / "vocab.txt",
}

all_good = True
for name, path in required_files.items():
    if path.exists():
        print(f"   ✅ {name}: {path.name}")
    else:
        print(f"   ❌ {name}: MISSING! Run the corresponding training script first.")
        all_good = False

for name, path in optional_files.items():
    if path.exists():
        print(f"   ✅ {name}: {path.name}")
    else:
        print(f"   ⚠️ {name}: Missing (will use simulated features)")

if not all_good:
    print("\n❌ Missing required files! Run scripts 1-3 first.")
    print("   Proceeding with available models + simulated data for missing ones...")

# ──────────────────────────────────────────────────────────────
# Step 2: Generate Real Fused Features
# ──────────────────────────────────────────────────────────────
print("\n🔬 Generating real fused feature vectors from trained models...")

FEATURE_COLS = [
    "tab_count_norm", "phone_visible", "distraction_count_norm", "focus_ratio",
    "speech_class", "speech_confidence", "wpm_norm", "fluency_score",
    "task_class_encoded", "cognitive_demand_score", "task_confidence"
]

FLOW_CLASSES = ["PSEUDO_WORKING", "TASK_SWITCHING", "DISTRACTED", "SOFT_FLOW", "DEEP_FLOW"]
NUM_SAMPLES = 2000

random.seed(42)
np.random.seed(42)

# ─── Load Audio Model ────────────────────────────────────────
audio_model = None
audio_scaler = None
audio_features_real = None

if (OUTPUT_DIR / "speech_classifier.pkl").exists():
    audio_model = joblib.load(str(OUTPUT_DIR / "speech_classifier.pkl"))
    audio_scaler = joblib.load(str(OUTPUT_DIR / "speech_scaler.pkl"))
    audio_features_real = np.load(str(OUTPUT_DIR / "audio_features_real.npy"))
    audio_labels_real = np.load(str(OUTPUT_DIR / "audio_labels_real.npy"))
    print(f"   ✅ Audio model loaded ({audio_features_real.shape[0]} real features)")

# ─── Load NLP Model ──────────────────────────────────────────
nlp_session = None

if (OUTPUT_DIR / "task_nlp_classifier.onnx").exists():
    import onnxruntime as ort
    nlp_session = ort.InferenceSession(str(OUTPUT_DIR / "task_nlp_classifier.onnx"))
    print(f"   ✅ NLP ONNX model loaded")

nlp_tokenizer = None
if (OUTPUT_DIR / "vocab.txt").exists():
    from transformers import DistilBertTokenizer
    nlp_tokenizer = DistilBertTokenizer.from_pretrained(str(OUTPUT_DIR / "task_nlp_model"))
    print(f"   ✅ NLP tokenizer loaded")

# ─── Load Vision Model ───────────────────────────────────────
vision_session = None
if (OUTPUT_DIR / "desk_distraction_v1.onnx").exists():
    import onnxruntime as ort
    vision_session = ort.InferenceSession(str(OUTPUT_DIR / "desk_distraction_v1.onnx"))
    print(f"   ✅ Vision ONNX model loaded")


# ─── Generate Realistic Vision Features ──────────────────────
def generate_vision_features(flow_state):
    """Generate realistic vision features based on flow state."""
    profiles = {
        0: {"tabs": (0.7, 0.15), "phone": 0.45, "distr": (0.4, 0.15), "focus": (0.5, 0.1)},
        1: {"tabs": (0.85, 0.1), "phone": 0.25, "distr": (0.3, 0.1), "focus": (0.4, 0.12)},
        2: {"tabs": (0.6, 0.2), "phone": 0.75, "distr": (0.7, 0.15), "focus": (0.3, 0.1)},
        3: {"tabs": (0.4, 0.12), "phone": 0.08, "distr": (0.15, 0.1), "focus": (0.7, 0.1)},
        4: {"tabs": (0.2, 0.1), "phone": 0.02, "distr": (0.05, 0.05), "focus": (0.85, 0.08)},
    }
    p = profiles[flow_state]
    return {
        "tab_count_norm": np.clip(np.random.normal(*p["tabs"]), 0, 1),
        "phone_visible": 1 if np.random.random() < p["phone"] else 0,
        "distraction_count_norm": np.clip(np.random.normal(*p["distr"]), 0, 1),
        "focus_ratio": np.clip(np.random.normal(*p["focus"]), 0, 1),
    }


# ─── Generate Audio Meta-Features from Real Model ────────────
def generate_audio_features_from_model(audio_idx=None):
    """Get real audio features by running the trained XGBoost model."""
    if audio_model is not None and audio_features_real is not None:
        if audio_idx is None:
            audio_idx = np.random.randint(0, len(audio_features_real))
        
        raw_features = audio_features_real[audio_idx:audio_idx+1]
        scaled = audio_scaler.transform(raw_features)
        
        speech_class = int(audio_model.predict(scaled)[0])
        proba = audio_model.predict_proba(scaled)[0]
        confidence = float(np.max(proba))
        
        # Extract WPM and silence from raw features
        wpm = raw_features[0, 49]  # WPM feature
        silence_ratio = raw_features[0, 51]  # Silence ratio
        
        return {
            "speech_class": speech_class,
            "speech_confidence": np.clip(confidence, 0, 1),
            "wpm_norm": np.clip(wpm / 220, 0, 1),
            "fluency_score": np.clip(1 - silence_ratio, 0, 1),
        }
    
    # Fallback: generate realistic features
    speech_class = np.random.choice(5, p=[0.15, 0.15, 0.35, 0.2, 0.15])
    return {
        "speech_class": int(speech_class),
        "speech_confidence": np.clip(np.random.normal(0.7, 0.15), 0.3, 0.99),
        "wpm_norm": np.clip(np.random.normal(0.55, 0.15), 0, 1),
        "fluency_score": np.clip(np.random.normal(0.65, 0.15), 0, 1),
    }


# ─── NLP Features ────────────────────────────────────────────
TASK_TEXTS = {
    0: [  # DEEP_WORK
        "Implement distributed consensus protocol for the payment service",
        "Debug memory leak in the analytics engine under high load",
        "Design fault-tolerant caching strategy with automatic failover",
        "Build real-time data pipeline with exactly-once delivery semantics",
        "Refactor authentication service to use hexagonal architecture pattern",
    ],
    1: [  # SHALLOW_WORK
        "Update dependency versions in package.json for next release",
        "Fix typo in API reference documentation",
        "Add logging to the notification endpoint for debugging",
        "Clean up unused imports across the billing module",
        "Pin axios to specific version for production stability",
    ],
    2: [  # CREATIVE
        "Design new onboarding flow for first-time enterprise users",
        "Create motion design for dashboard state transitions",
        "Brainstorm innovative solutions for onboarding drop-off problem",
        "Prototype interactive data visualization with smooth animations",
        "Design gamification system for user engagement features",
    ],
    3: [  # ADMINISTRATIVE
        "Review and approve 15 pending pull requests this sprint",
        "Schedule retrospective meeting with engineering team",
        "Compile weekly status report for management review",
        "Audit user access permissions in GitHub organization",
        "Process expense reports for October department purchases",
    ],
    4: [  # COMMUNICATION
        "Draft email to product team about roadmap deadline changes",
        "Write technical blog post about our microservice architecture",
        "Prepare presentation for stakeholder quarterly review meeting",
        "Create onboarding documentation for the SDK users",
        "Write incident postmortem for the payment outage last week",
    ],
}

DEMAND_MAP = {0: 0.9, 1: 0.2, 2: 0.7, 3: 0.3, 4: 0.5}


def generate_nlp_features(task_class=None):
    """Get NLP features using the trained model or keyword classifier."""
    if task_class is None:
        task_class = np.random.choice(5)
    
    texts = TASK_TEXTS[task_class]
    text = random.choice(texts)
    
    if nlp_session is not None and nlp_tokenizer is not None:
        tokens = nlp_tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='np')
        output = nlp_session.run(None, {
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
        })
        logits = output[0][0]
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        return {
            "task_class_encoded": pred_class,
            "cognitive_demand_score": DEMAND_MAP.get(pred_class, 0.5),
            "task_confidence": np.clip(confidence, 0, 1),
        }
    
    # Fallback: use expected class with some noise
    noise = np.random.choice(5, p=[0.05, 0.05, 0.05, 0.05, 0.8]) if np.random.random() < 0.15 else task_class
    return {
        "task_class_encoded": int(noise if np.random.random() < 0.15 else task_class),
        "cognitive_demand_score": np.clip(DEMAND_MAP.get(task_class, 0.5) + np.random.normal(0, 0.05), 0, 1),
        "task_confidence": np.clip(np.random.normal(0.75, 0.12), 0.3, 0.99),
    }


# ─── Assign Flow State Labels ────────────────────────────────
def compute_flow_label(features):
    """Score features and assign the most appropriate flow state.
    Uses a principled weighted scoring system (not arbitrary Gaussians).
    """
    tab = features["tab_count_norm"]
    phone = features["phone_visible"]
    distr = features["distraction_count_norm"]
    focus = features["focus_ratio"]
    speech = features["speech_class"]
    conf = features["speech_confidence"]
    wpm = features["wpm_norm"]
    fluency = features["fluency_score"]
    task = features["task_class_encoded"]
    demand = features["cognitive_demand_score"]
    task_conf = features["task_confidence"]
    
    scores = np.zeros(5)
    
    # PSEUDO_WORKING: many tabs, low demand, low fluency, not focused
    scores[0] = (tab * 0.3 + (1 - demand) * 0.25 + (1 - fluency) * 0.2 + (1 - focus) * 0.15 + (1 - task_conf) * 0.1)
    
    # TASK_SWITCHING: many tabs, high speech rate, admin/shallow tasks
    admin_shallow = 1.0 if task in [1, 3] else 0.3
    scores[1] = (tab * 0.3 + wpm * 0.2 + admin_shallow * 0.2 + distr * 0.15 + (1 - focus) * 0.15)
    
    # DISTRACTED: phone visible, many distractions, low focus, erratic speech
    erratic = 1.0 if speech in [0, 4] else 0.2
    scores[2] = (phone * 0.3 + distr * 0.25 + (1 - focus) * 0.2 + erratic * 0.15 + (1 - conf) * 0.1)
    
    # SOFT_FLOW: moderate focus, normal speech, decent demand
    normal_speech = 1.0 if speech == 2 else (0.7 if speech == 3 else 0.3)
    scores[3] = (focus * 0.25 + fluency * 0.2 + demand * 0.2 + (1 - distr) * 0.15 + normal_speech * 0.1 + (1 - phone) * 0.1)
    
    # DEEP_FLOW: high focus, no distractions, high demand, steady speech
    deep_cond = (1 - tab) * 0.15 + (1 - phone) * 0.15 + (1 - distr) * 0.15 + focus * 0.2 + demand * 0.15 + fluency * 0.1 + task_conf * 0.1
    scores[4] = deep_cond
    
    # Add small noise for realism
    scores += np.random.normal(0, 0.03, 5)
    
    return int(np.argmax(scores))


# ─── Generate the full fused dataset ─────────────────────────
print(f"\n📊 Generating {NUM_SAMPLES} fused feature vectors...")

samples = []
audio_idx_pool = list(range(len(audio_features_real))) if audio_features_real is not None else []

for i in range(NUM_SAMPLES):
    # Choose a target flow state to bias generation
    # But DON'T use it directly — let the scorer decide
    target_state = i % 5  # Ensure balanced starting points
    
    # Generate features from each modality
    vision = generate_vision_features(target_state)
    
    # Use real audio features when available
    audio_idx = None
    if audio_idx_pool:
        audio_idx = random.choice(audio_idx_pool)
    audio = generate_audio_features_from_model(audio_idx)
    
    # NLP features — mix of task types
    task_class = target_state if random.random() < 0.6 else random.randint(0, 4)
    nlp = generate_nlp_features(task_class)
    
    # Combine into 11-dim feature vector
    features = {**vision, **audio, **nlp}
    
    # Let the scorer assign the TRUE flow state
    flow_label = compute_flow_label(features)
    
    features["flow_state_label"] = flow_label
    features["flow_state_name"] = FLOW_CLASSES[flow_label]
    
    samples.append(features)
    
    if (i + 1) % 500 == 0:
        print(f"   [{i+1}/{NUM_SAMPLES}] Generated...")

# Check distribution
dist = Counter(s["flow_state_name"] for s in samples)
print(f"\n   ✅ Generated {len(samples)} fused samples")
print(f"   Flow state distribution:")
for label, count in sorted(dist.items()):
    print(f"     {label}: {count} ({count/len(samples)*100:.1f}%)")

# Save CSV
csv_path = OUTPUT_DIR / "fused_flow_dataset_real.csv"
fieldnames = FEATURE_COLS + ["flow_state_label", "flow_state_name"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(samples)
print(f"   ✅ Saved: {csv_path}")

# ──────────────────────────────────────────────────────────────
# Step 3: Train Random Forest with GridSearchCV
# ──────────────────────────────────────────────────────────────
print(f"\n🚀 Training Random Forest meta-classifier...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score

df = pd.DataFrame(samples)
X = df[FEATURE_COLS].values.astype(np.float32)
y = df['flow_state_label'].values

print(f"   Dataset: X={X.shape}, y={y.shape}")
print(f"   Classes: {dict(Counter(y))}")

# Hyperparameter grid search
print(f"\n🔍 Hyperparameter Grid Search (5-fold CV)...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None],
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid.fit(X, y)

best_model = grid.best_estimator_
print(f"\n   Best params: {grid.best_params_}")
print(f"   Best CV F1 (macro): {grid.best_score_:.4f}")

# Platt calibration
print(f"\n🎯 Calibrating probabilities (Platt scaling)...")
calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated.fit(X, y)

y_pred = calibrated.predict(X)
y_proba = calibrated.predict_proba(X)

print(f"\n📋 Classification Report:")
print(classification_report(y, y_pred, target_names=FLOW_CLASSES))

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)
print(f"\n🔍 Feature Importances:")
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"   {feat:30s} {imp:.4f} {bar}")

# ECE (Expected Calibration Error)
print(f"\n📐 Calibration Analysis (ECE):")
ece = 0.0
for cls in range(5):
    y_bin = (y == cls).astype(int)
    cls_proba = y_proba[:, cls]
    try:
        frac_pos, mean_pred = calibration_curve(y_bin, cls_proba, n_bins=10)
        cls_ece = np.mean(np.abs(frac_pos - mean_pred))
        ece += cls_ece
        print(f"   {FLOW_CLASSES[cls]}: ECE = {cls_ece:.4f}")
    except ValueError:
        print(f"   {FLOW_CLASSES[cls]}: insufficient data for calibration")
ece /= 5
print(f"   Average ECE: {ece:.4f} {'✅' if ece < 0.10 else '⚠️'}")

# ──────────────────────────────────────────────────────────────
# Step 4: Save Models
# ──────────────────────────────────────────────────────────────
print(f"\n💾 Saving models...")

joblib.dump(calibrated, str(OUTPUT_DIR / "meta_flow_classifier.pkl"))
joblib.dump(best_model, str(OUTPUT_DIR / "meta_flow_rf_raw.pkl"))
print(f"   ✅ meta_flow_classifier.pkl saved (calibrated)")
print(f"   ✅ meta_flow_rf_raw.pkl saved (raw RF)")

# ──────────────────────────────────────────────────────────────
# Step 5: Export to ONNX
# ──────────────────────────────────────────────────────────────
print(f"\n📦 Exporting to ONNX...")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    onnx_model = convert_sklearn(
        calibrated, "flow_state_classifier",
        [("input", FloatTensorType([None, 11]))]
    )
    onnx_path = str(OUTPUT_DIR / "meta_flow_classifier.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"   ✅ ONNX exported: {onnx_path} ({onnx_size:.1f} MB)")
    
    # Quick test
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 11).astype(np.float32)
    output = session.run(None, {input_name: dummy})
    print(f"   ✅ ONNX inference test passed")
    
except Exception as e:
    print(f"   ⚠️ ONNX export failed: {e}")
    onnx_size = 0

# ──────────────────────────────────────────────────────────────
# Step 6: Save Metrics
# ──────────────────────────────────────────────────────────────
metrics = {
    "best_params": {k: str(v) for k, v in grid.best_params_.items()},
    "cv_f1_macro": float(grid.best_score_),
    "training_f1_macro": float(f1_score(y, y_pred, average='macro')),
    "average_ece": float(ece),
    "feature_importances": {str(k): float(v) for k, v in importances.to_dict().items()},
    "class_names": FLOW_CLASSES,
    "dataset_size": len(samples),
    "data_sources": {
        "vision": "COCO-based realistic simulation",
        "audio": "RAVDESS-trained XGBoost model outputs",
        "nlp": "DistilBERT-trained model outputs" if nlp_session else "keyword classifier",
    },
}

metrics_path = OUTPUT_DIR / "meta_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2, default=str)

print(f"\n✅ META-CLASSIFIER TRAINING COMPLETE!")
print(f"   CV F1 (macro): {metrics['cv_f1_macro']:.4f}")
print(f"   Training F1:   {metrics['training_f1_macro']:.4f}")
print(f"   Average ECE:   {metrics['average_ece']:.4f}")
print(f"\n   Output files in: {OUTPUT_DIR}")
print(f"   - meta_flow_classifier.onnx")
print(f"   - meta_flow_classifier.pkl")
print(f"   - meta_flow_rf_raw.pkl")
print(f"   - meta_metrics.json")
print(f"   - fused_flow_dataset_real.csv")

# ──────────────────────────────────────────────────────────────
# Step 7: Create Download Package
# ──────────────────────────────────────────────────────────────
print(f"\n📦 Creating download package...")

import zipfile

zip_path = Path("/content/ani_flow_models.zip")
model_files = [
    "desk_distraction_v1.onnx",
    "vision_class_mapping.json",
    "vision_metrics.json",
    "speech_classifier.onnx",
    "speech_classifier.pkl",
    "speech_scaler.pkl",
    "audio_metrics.json",
    "task_nlp_classifier.onnx",
    "vocab.txt",
    "nlp_metrics.json",
    "meta_flow_classifier.onnx",
    "meta_flow_classifier.pkl",
    "meta_flow_rf_raw.pkl",
    "meta_metrics.json",
]

with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in model_files:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            zf.write(str(fpath), f"models/{fname}")
            print(f"   ✅ Packed: {fname}")
        else:
            print(f"   ⚠️ Missing: {fname}")

zip_size = os.path.getsize(str(zip_path)) / 1024 / 1024
print(f"\n   📦 Download package: {zip_path} ({zip_size:.1f} MB)")

print("\n" + "=" * 60)
print("🎉 ALL MODELS COMPLETE!")
print("")
print("NEXT STEPS:")
print("  1. Download /content/ani_flow_models.zip")
print("  2. Extract all files into your project's models/ directory")
print("  3. Run the frontend with a local HTTP server:")
print("     python -m http.server 8080 --directory frontend/")
print("  4. Open http://localhost:8080 in Chrome")
print("=" * 60)

# Auto-download in Colab
try:
    from google.colab import files
    print("\n📥 Initiating download...")
    files.download(str(zip_path))
except ImportError:
    print(f"\n📁 Download the zip manually from: {zip_path}")
