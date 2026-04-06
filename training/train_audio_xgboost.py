"""
Train XGBoost speech cognitive load classifier on 52-dim audio features.

Classes: ERRATIC(0), SLOW_LABORED(1), NORMAL_FOCUSED(2), FAST_ENERGIZED(3), RAPID_SCATTERED(4)
Output: models/speech_classifier.pkl + speech_classifier.onnx
"""
import os, sys, argparse, json
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

CLASS_NAMES = ["ERRATIC_SPEECH", "SLOW_LABORED", "NORMAL_FOCUSED", "FAST_ENERGIZED", "RAPID_SCATTERED"]

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost audio model")
    parser.add_argument("--data", default=None, help="Path to audio_features.npy")
    parser.add_argument("--labels", default=None, help="Path to audio_labels.npy")
    parser.add_argument("--export-onnx", action="store_true", default=True)
    args = parser.parse_args()

    proc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    if args.data is None:
        args.data = os.path.join(proc_dir, "audio_features.npy")
    if args.labels is None:
        args.labels = os.path.join(proc_dir, "audio_labels.npy")

    for p in [args.data, args.labels]:
        if not os.path.exists(p):
            print(f"❌ Not found: {p}")
            print("   Run: python data/scripts/generate_synthetic_audio.py")
            sys.exit(1)

    print("=" * 60)
    print("🎙️  XGBoost Speech Cognitive Load Classifier Training")
    print("=" * 60)

    X = np.load(args.data)
    y = np.load(args.labels)
    print(f"✅ Loaded data: X={X.shape}, y={y.shape}")
    print(f"   Classes: {np.unique(y, return_counts=True)}")

    # Normalize
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

    # Cross-validation
    print("\n📊 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
    print(f"   CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full data
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    print("\n📋 Training Classification Report:")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))

    # Feature importance
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    feature_names = [f"feat_{i}" for i in range(52)]
    feature_names[42] = "spectral_centroid"
    feature_names[43] = "spectral_rolloff"
    feature_names[44] = "zcr"
    feature_names[45] = "rms"
    feature_names[46] = "pitch_mean"
    feature_names[47] = "pitch_var"
    feature_names[48] = "tempo"
    feature_names[49] = "wpm_mean"
    feature_names[50] = "wpm_var"
    feature_names[51] = "silence_ratio"

    print("\n🔍 Top 10 Feature Importances:")
    for idx in top_indices:
        print(f"   {feature_names[idx]:20s}: {importances[idx]:.4f}")

    # SHAP analysis
    try:
        import shap
        print("\n🔬 Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled[:100])
        print("   SHAP values computed for first 100 samples")

        # Safely compute per-feature importance regardless of SHAP output shape
        sv = np.array(shap_values)
        if sv.ndim == 3:
            # shape (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
            if sv.shape[0] == len(CLASS_NAMES):
                shap_importance = np.mean(np.abs(sv), axis=(0, 1))  # → (n_features,)
            else:
                shap_importance = np.mean(np.abs(sv), axis=(0, 2))  # → (n_features,)
        elif sv.ndim == 2:
            shap_importance = np.mean(np.abs(sv), axis=0)
        else:
            shap_importance = np.zeros(X.shape[1])

        shap_importance = np.asarray(shap_importance).flatten()
        top_shap = np.argsort(shap_importance)[::-1][:5]
        print("   Top 5 SHAP features:")
        for idx in top_shap:
            i = int(idx)
            print(f"     {feature_names[i]}: {float(shap_importance[i]):.4f}")
    except ImportError:
        print("   ⚠️ SHAP not installed, skipping explainability analysis")
    except Exception as e:
        print(f"   ⚠️ SHAP explainability skipped due to error: {e}")

    # Save models
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "speech_classifier.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "speech_scaler.pkl"))
    print(f"\n✅ Model saved: {model_dir}/speech_classifier.pkl")
    print(f"   Scaler saved: {model_dir}/speech_scaler.pkl")

    # ONNX export
    if args.export_onnx:
        onnx_path = os.path.join(model_dir, "speech_classifier.onnx")
        exported = False
        # Method 1: skl2onnx (works for many sklearn-compatible models)
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            onnx_model = convert_sklearn(
                model, "speech_classifier",
                [("input", FloatTensorType([None, X_scaled.shape[1]]))]
            )
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"   ONNX exported (skl2onnx): {onnx_path}")
            exported = True
        except Exception as e1:
            print(f"   ⚠️ skl2onnx export failed: {e1}")
        # Method 2: onnxmltools fallback
        if not exported:
            try:
                from onnxmltools import convert_xgboost
                from onnxmltools.convert.common.data_types import FloatTensorType as FTT
                onnx_model = convert_xgboost(model, initial_types=[("input", FTT([None, X_scaled.shape[1]]))])
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                print(f"   ONNX exported (onnxmltools): {onnx_path}")
                exported = True
            except Exception as e2:
                print(f"   ⚠️ onnxmltools export also failed: {e2}")
        if not exported:
            print("   ❌ All ONNX export methods failed. Only .pkl saved.")

    # Save evaluation metrics
    metrics = {
        "cv_f1_macro_mean": float(cv_scores.mean()),
        "cv_f1_macro_std": float(cv_scores.std()),
        "training_accuracy": float(np.mean(y_pred == y)),
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(model_dir, "audio_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Audio model training complete!")

if __name__ == "__main__":
    main()
