# 📂 Sample Datasets — ANI Flow Optimizer

This directory contains **sample data files** that demonstrate the structure and format of the datasets used to train each of the four models in the ANI Flow Optimizer.

> ⚠️ **These are illustrative samples, not the full training datasets.** Actual datasets are downloaded automatically during training via the Colab scripts in `colab/`.

---

## Files

| File | Model | Description |
|------|-------|-------------|
| `sample_coco_annotations.json` | Vision (YOLOv8) | COCO 2017 annotation format with desk-object category mapping |
| `sample_ravdess_features.csv` | Audio (XGBoost) | Extracted 52-dim feature vectors from RAVDESS WAV files |
| `sample_task_descriptions.csv` | NLP (DistilBERT) | Task text samples with 5-class labels and cognitive demand scores |
| `sample_fused_vectors.csv` | Meta-Classifier (RF) | 11-dim fused feature vectors with flow state labels |

---

## How to Get Full Datasets

### 1. COCO 2017 (Vision)
```bash
# Auto-downloaded in colab/1_train_vision_yolov8.py
# Source: https://cocodataset.org/
# License: CC BY 4.0
```

### 2. RAVDESS (Audio)
```bash
# Auto-downloaded in colab/2_train_audio_xgboost.py
# Source: https://zenodo.org/record/1188976
# License: CC BY-NC-SA 4.0
```

### 3. Synthetic Tasks (NLP)
```python
# Generated in-script by colab/3_train_nlp_distilbert.py
# Uses 120+ templates per class with 4 augmentation techniques
```

### 4. Fused Features (Meta)
```python
# Generated in-script by colab/4_train_meta_classifier.py
# Uses output from all three trained models above
```

---

## Column Descriptions

### RAVDESS Features (`sample_ravdess_features.csv`)
- `mfcc_0` through `mfcc_12` — Mel-Frequency Cepstral Coefficients (means)
- `spectral_centroid` — Center of mass of the audio spectrum (Hz)
- `spectral_rolloff` — Frequency below which 85% of energy is contained (Hz)
- `zcr` — Zero Crossing Rate (how often signal crosses zero)
- `rms_energy` — Root Mean Square of audio amplitude
- `pitch_mean/var` — Fundamental frequency (F0) statistics
- `tempo` — Estimated beats per minute
- `onset_rate` — Onset detection rate (speech onsets per second)
- `silence_ratio` — Proportion of low-energy frames

### Fused Vectors (`sample_fused_vectors.csv`)
- Features indexed 0-3: Vision output
- Features indexed 4-7: Audio output
- Features indexed 8-10: NLP output
- `flow_state_label`: Target class (0-4)

---

*These samples are for documentation and demonstration purposes only.*
