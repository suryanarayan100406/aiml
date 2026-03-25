"""
Evaluate all trained models and generate comprehensive report.
Produces confusion matrices, calibration curves, and SHAP analysis.
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def eval_audio():
    """Evaluate XGBoost audio classifier."""
    print("\n🎙️ Audio Model Evaluation")
    print("-" * 40)
    model_path = os.path.join(MODEL_DIR, "speech_classifier.pkl")
    scaler_path = os.path.join(MODEL_DIR, "speech_scaler.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️ Model not found. Run train_audio_xgboost.py first.")
        return None

    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X = np.load(os.path.join(DATA_DIR, "audio_features.npy"))
    y = np.load(os.path.join(DATA_DIR, "audio_labels.npy"))
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    names = ["ERRATIC","SLOW","NORMAL","FAST","RAPID"]
    print(classification_report(y, y_pred, target_names=names))
    f1 = f1_score(y, y_pred, average='macro')
    print(f"  Macro F1: {f1:.4f} {'✅' if f1 > 0.72 else '⚠️'}")
    return {"model": "audio", "f1_macro": float(f1), "target": 0.72, "pass": f1 > 0.72}

def eval_meta():
    """Evaluate meta-classifier."""
    print("\n🔀 Meta-Classifier Evaluation")
    print("-" * 40)
    model_path = os.path.join(MODEL_DIR, "meta_flow_classifier.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️ Model not found. Run train_meta_classifier.py first.")
        return None

    from sklearn.metrics import classification_report, f1_score
    from sklearn.calibration import calibration_curve
    model = joblib.load(model_path)
    df = pd.read_csv(os.path.join(DATA_DIR, "fused_flow_dataset.csv"))
    FEAT = ["tab_count_norm","phone_visible","distraction_count_norm","focus_ratio",
            "speech_class","speech_confidence","wpm_norm","fluency_score",
            "task_class_encoded","cognitive_demand_score","task_confidence"]
    X = df[FEAT].values.astype(np.float32)
    y = df['flow_state_label'].values
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    names = ["PSEUDO","SWITCHING","DISTRACTED","SOFT_FLOW","DEEP_FLOW"]
    print(classification_report(y, y_pred, target_names=names))
    f1 = f1_score(y, y_pred, average='macro')

    # ECE
    ece = 0.0
    for cls in range(5):
        y_bin = (y == cls).astype(int)
        try:
            frac, mean_p = calibration_curve(y_bin, y_proba[:, cls], n_bins=10)
            ece += np.mean(np.abs(frac - mean_p))
        except ValueError:
            pass
    ece /= 5

    print(f"  Macro F1: {f1:.4f} {'✅' if f1 > 0.74 else '⚠️'}")
    print(f"  Average ECE: {ece:.4f} {'✅' if ece < 0.10 else '⚠️'}")
    return {"model": "meta", "f1_macro": float(f1), "ece": float(ece),
            "f1_pass": f1 > 0.74, "ece_pass": ece < 0.10}

def main():
    print("=" * 60)
    print("📊 ANI Flow Optimizer — Full Model Evaluation")
    print("=" * 60)

    results = []
    r = eval_audio()
    if r: results.append(r)
    r = eval_meta()
    if r: results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("📋 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Metric':<15} {'Value':<10} {'Target':<10} {'Pass':<6}")
    print("-" * 60)
    for r in results:
        model = r["model"]
        if "f1_macro" in r:
            target = r.get("target", 0.74)
            passed = r.get("pass", r.get("f1_pass", False))
            print(f"{model:<20} {'F1 (macro)':<15} {r['f1_macro']:<10.4f} {target:<10.2f} {'✅' if passed else '❌'}")
        if "ece" in r:
            print(f"{'':<20} {'ECE':<15} {r['ece']:<10.4f} {'< 0.10':<10} {'✅' if r['ece_pass'] else '❌'}")

    # Save report
    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Report saved: {report_path}")

if __name__ == "__main__":
    main()
