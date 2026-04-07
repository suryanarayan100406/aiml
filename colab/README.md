# 🧠 ANI Flow Optimizer — Colab Training Scripts

## How to Use

Each script is **self-contained** — paste the entire file into a Google Colab cell and run.

### Training Order

**Run in this order** (script 4 depends on outputs from 1-3):

| # | Script | GPU? | Time | Description |
|---|--------|------|------|-------------|
| 1 | `1_train_vision_yolov8.py` | ✅ T4+ | ~15 min | Downloads COCO desk images, fine-tunes YOLOv8-nano |
| 2 | `2_train_audio_xgboost.py` | ❌ CPU | ~5 min | Downloads RAVDESS audio, extracts features, trains XGBoost |
| 3 | `3_train_nlp_distilbert.py` | ✅ T4+ | ~10 min | Generates augmented task data, fine-tunes DistilBERT |
| 4 | `4_train_meta_classifier.py` | ❌ CPU | ~2 min | Uses outputs from 1-3, trains RF meta-classifier, creates zip |

### Step-by-Step

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. For scripts 1 & 3: Set GPU runtime (`Runtime → Change runtime type → T4 GPU`)
4. Copy-paste each script into a cell and run (`Shift+Enter`)
5. After running script 4, it will auto-download `ani_flow_models.zip`
6. Extract the zip into your project's `models/` directory

### Output Files

All trained models are saved to `/content/ani_models/` on Colab:

```
/content/ani_models/
├── desk_distraction_v1.onnx      # Vision model (YOLOv8-nano, ~12MB)
├── vision_class_mapping.json     # Vision class config
├── vision_metrics.json           # Vision eval metrics
├── speech_classifier.onnx        # Audio model (XGBoost, ~450KB)
├── speech_classifier.pkl         # Audio model (sklearn format)
├── speech_scaler.pkl             # Audio feature scaler
├── audio_metrics.json            # Audio eval metrics
├── audio_features_real.npy       # Extracted RAVDESS features
├── audio_labels_real.npy         # RAVDESS labels
├── task_nlp_classifier.onnx      # NLP model (DistilBERT, ~260MB)
├── vocab.txt                     # WordPiece vocabulary
├── nlp_metrics.json              # NLP eval metrics
├── meta_flow_classifier.onnx     # Meta-classifier (RF, ~7MB)
├── meta_flow_classifier.pkl      # Meta-classifier (sklearn format)
├── meta_flow_rf_raw.pkl          # Raw RF (uncalibrated)
├── meta_metrics.json             # Meta-classifier eval metrics
└── fused_flow_dataset_real.csv   # Generated fused dataset
```

### Datasets Used

| Model | Dataset | Source | Size |
|-------|---------|--------|------|
| Vision | COCO 2017 val subset | [cocodataset.org](https://cocodataset.org) | ~400 images |
| Audio | RAVDESS | [Zenodo](https://zenodo.org/record/1188976) | ~1440 files |
| NLP | Template-generated + augmented | Generated in-script | ~8000 samples |
| Meta | Model outputs on real data | Generated from 1-3 | ~2000 samples |
