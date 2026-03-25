"""
Train Random Forest meta-classifier for final flow state prediction.
Fuses outputs from vision, audio, and NLP models into 11-feature vector.

Classes: PSEUDO_WORKING(0), TASK_SWITCHING(1), DISTRACTED(2), SOFT_FLOW(3), DEEP_FLOW(4)
Output: models/meta_flow_classifier.pkl + meta_flow_classifier.onnx
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASS_NAMES = ["PSEUDO_WORKING","TASK_SWITCHING","DISTRACTED","SOFT_FLOW","DEEP_FLOW"]
FEATURE_COLS = [
    "tab_count_norm","phone_visible","distraction_count_norm","focus_ratio",
    "speech_class","speech_confidence","wpm_norm","fluency_score",
    "task_class_encoded","cognitive_demand_score","task_confidence"
]

def main():
    parser = argparse.ArgumentParser(description="Train meta-classifier")
    parser.add_argument("--data", default=None)
    parser.add_argument("--export-onnx", action="store_true", default=True)
    args = parser.parse_args()

    proc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    if args.data is None:
        args.data = os.path.join(proc_dir, "fused_flow_dataset.csv")

    if not os.path.exists(args.data):
        print(f"❌ Not found: {args.data}")
        print("   Run: python data/scripts/generate_fused_dataset.py")
        sys.exit(1)

    print("=" * 60)
    print("🔀 Random Forest Meta-Classifier Training")
    print("=" * 60)

    df = pd.read_csv(args.data)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['flow_state_label'].values

    print(f"✅ Loaded: {len(df)} samples, {X.shape[1]} features")
    print(f"   Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Grid search
    print("\n🔍 Hyperparameter Grid Search (5-fold CV)...")
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

    # Calibrate probabilities (Platt scaling)
    print("\n🎯 Calibrating probabilities (Platt scaling)...")
    calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
    calibrated.fit(X, y)

    y_pred = calibrated.predict(X)
    y_proba = calibrated.predict_proba(X)

    print("\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    print("\n🔍 Feature Importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"   {feat:30s} {imp:.4f} {bar}")

    # Calibration check (ECE)
    print("\n📐 Calibration Analysis (ECE):")
    ece = 0.0
    for cls in range(5):
        y_bin = (y == cls).astype(int)
        cls_proba = y_proba[:, cls]
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, cls_proba, n_bins=10)
            cls_ece = np.mean(np.abs(frac_pos - mean_pred))
            ece += cls_ece
            print(f"   {CLASS_NAMES[cls]}: ECE = {cls_ece:.4f}")
        except ValueError:
            print(f"   {CLASS_NAMES[cls]}: insufficient data for calibration")
    ece /= 5
    print(f"   Average ECE: {ece:.4f} {'✅' if ece < 0.10 else '⚠️'}")

    # Save
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(calibrated, os.path.join(model_dir, "meta_flow_classifier.pkl"))
    joblib.dump(best_model, os.path.join(model_dir, "meta_flow_rf_raw.pkl"))
    print(f"\n✅ Saved: {model_dir}/meta_flow_classifier.pkl")

    # ONNX export
    if args.export_onnx:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            onnx_model = convert_sklearn(
                calibrated, "flow_state_classifier",
                [("input", FloatTensorType([None, 11]))]
            )
            onnx_path = os.path.join(model_dir, "meta_flow_classifier.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"   ONNX exported: {onnx_path}")
        except Exception as e:
            print(f"   ⚠️ ONNX export failed: {e}")

    # Metrics JSON
    metrics = {
        "best_params": grid.best_params_,
        "cv_f1_macro": float(grid.best_score_),
        "training_f1_macro": float(f1_score(y, y_pred, average='macro')),
        "average_ece": float(ece),
        "feature_importances": importances.to_dict(),
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(model_dir, "meta_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n✅ Meta-classifier training complete!")

if __name__ == "__main__":
    main()
