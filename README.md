# 🧠 ANI — Creative Flow Optimizer

A **production-grade multimodal AI system** that classifies your cognitive work state in real-time using three separate trained ML models — one per input modality (vision, audio, text) — fused through a **meta-classifier ensemble**.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  🖼️ Vision   │    │  🎙️ Audio    │    │  📝 NLP      │
│  YOLOv8-nano │    │  XGBoost     │    │  DistilBERT  │
│  (4 features)│    │  (4 features)│    │  (3 features)│
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────┴───────┐
                    │ 🔀 Meta-RF    │
                    │ 11 features   │
                    │ → Flow State  │
                    └──────────────┘
```

## Models

| Model | Architecture | Dataset | Output |
|---|---|---|---|
| Vision | YOLOv8-nano | COCO + Custom Screenshots | tab_count, phone_visible, distraction_count, focus_ratio |
| Audio | XGBoost | RAVDESS + LibriSpeech | speech_class, confidence, WPM, fluency |
| NLP | DistilBERT | Task Classification | task_class, cognitive_demand, confidence |
| Meta | Random Forest | Fused self-report data | Flow State (5 classes) |

## Flow States

- 🔴 **PSEUDO_WORKING** — Appears busy but not productive
- 🟠 **TASK_SWITCHING** — Rapid context switching
- 🟡 **DISTRACTED** — Multiple distractions detected
- 🟢 **SOFT_FLOW** — Good focus, moderate engagement
- 🟣 **DEEP_FLOW** — Peak cognitive performance

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data
python data/scripts/generate_synthetic_tasks.py
python data/scripts/generate_synthetic_audio.py
python data/scripts/generate_synthetic_vision.py
python data/scripts/generate_fused_dataset.py

# Train all models
python training/train_audio_xgboost.py
python training/train_nlp_distilbert.py
python training/train_vision_yolov8.py
python training/train_meta_classifier.py

# Evaluate
python training/evaluate_all_models.py

# Launch frontend
# Open frontend/index.html in Chrome
```

## Project Structure

```
ani-flow-optimizer/
├── data/scripts/          # Data generation & download scripts
├── training/              # Model training scripts
├── models/                # Trained model files (.onnx, .pkl)
├── frontend/              # Browser-based UI with ONNX Runtime Web
├── chrome_extension/      # Chrome extension for data collection
└── requirements.txt       # Python dependencies
```
