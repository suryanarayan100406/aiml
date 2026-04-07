# 🧠 ANI — Creative Flow Optimizer

> **A production-grade multimodal AI system that classifies your cognitive work state in real-time using four custom-trained ML models — one per input modality (vision, audio, text) — fused through a meta-classifier ensemble, all running entirely in the browser via ONNX Runtime Web.**

<p align="center">
  <img src="https://img.shields.io/badge/Models-4%2F4%20Loaded-brightgreen" alt="Models Status">
  <img src="https://img.shields.io/badge/Inference-Browser%20(ONNX%20Runtime%20Web)-blue" alt="Inference Engine">
  <img src="https://img.shields.io/badge/Framework-Vanilla%20JS-yellow" alt="Framework">
  <img src="https://img.shields.io/badge/Training-Google%20Colab-orange" alt="Training">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Flow States](#-flow-states)
- [Models Deep Dive](#-models-deep-dive)
- [Quick Start](#-quick-start)
- [Training on Google Colab](#-training-on-google-colab)
- [Running the Frontend](#-running-the-frontend)
- [Chrome Extension](#-chrome-extension)
- [Project Structure](#-project-structure)
- [Dashboard Features](#-dashboard-features)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)

---

## 🔍 Overview

ANI (Adaptive Neural Intelligence) is a **cognitive flow state analyzer** that monitors your workspace in real-time through three sensory channels — **what you see (vision)**, **how you sound (audio)**, and **what you're working on (text)** — and fuses these signals into a single, calibrated prediction of your current cognitive state.

Unlike simple screen-time trackers, ANI uses actual trained deep learning models to understand the *quality* of your engagement:

- **Are you in deep flow?** ANI sees focused workspace, hears steady speech, and detects demanding tasks.
- **Are you pseudo-working?** ANI notices multiple distractions, erratic speech patterns, and shallow tasks.
- **Are you task-switching?** ANI detects rapid context changes across modalities.

All inference runs **entirely in your browser** — no data leaves your machine.

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        Browser (ONNX Runtime Web)                 │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │
│  │  🖼️ Vision    │   │  🎙️ Audio     │   │  📝 NLP       │          │
│  │  YOLOv8-nano │   │  XGBoost     │   │  DistilBERT  │          │
│  │  12 MB       │   │  2.4 MB      │   │  256 MB      │          │
│  │  (4 features)│   │  (4 features)│   │  (3 features)│          │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │
│         │                  │                   │                  │
│         │    ┌─────────────┴───────────────┐   │                  │
│         └────┤    Feature Fusion (11-dim)   ├──┘                  │
│              └─────────────┬───────────────┘                      │
│                            │                                      │
│                   ┌────────┴────────┐                             │
│                   │ 🔀 Meta-Classifier│                            │
│                   │ Random Forest    │                             │
│                   │ + Platt Scaling  │                             │
│                   │ 9.9 MB           │                             │
│                   └────────┬────────┘                             │
│                            │                                      │
│                   ┌────────┴────────┐                             │
│                   │   Flow State    │                             │
│                   │ (5 classes +    │                             │
│                   │  probabilities) │                             │
│                   └─────────────────┘                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   ANI Guardian (Adaptive Alerts)              │ │
│  │  Distraction warnings · Flow suggestions · Work quality      │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
         ▲                                           ▲
         │                                           │
    ┌────┴─────┐                              ┌──────┴──────┐
    │ Webcam   │                              │   Chrome    │
    │ Feed     │                              │  Extension  │
    └──────────┘                              │ (tab data)  │
                                              └─────────────┘
```

### Data Flow

1. **Vision Pipeline** — Captures webcam frames → resizes to 640×640 → runs YOLOv8 ONNX model → extracts 4 environmental features (phone visibility, monitor count, distraction count, focus ratio)
2. **Audio Pipeline** — Captures microphone stream → computes 52-dim librosa-style features (MFCCs, chroma, spectral) using Web Audio API → runs XGBoost ONNX model → extracts 4 speech features (class, confidence, WPM, fluency)
3. **NLP Pipeline** — Receives task description text (from browser tab title or manual input) → tokenizes with WordPiece → runs DistilBERT ONNX model → extracts 3 task features (class, cognitive demand, confidence)
4. **Meta-Classifier** — Concatenates all 11 features → runs calibrated Random Forest → outputs 5-class flow state with calibrated probabilities

---

## 🎯 Flow States

ANI classifies your cognitive state into one of five levels:

| State | Emoji | Description | Example |
|-------|-------|-------------|---------|
| **PSEUDO_WORKING** | 🔴 | Appears busy but not productive | Scrolling social media with code editor open |
| **TASK_SWITCHING** | 🟠 | Rapid context switching | Jumping between 10+ tabs every minute |
| **DISTRACTED** | 🟡 | Multiple distractions present | Phone visible, erratic speech, shallow task |
| **SOFT_FLOW** | 🟢 | Good focus, moderate engagement | Writing docs with steady pace |
| **DEEP_FLOW** | 🟣 | Peak cognitive performance | Complex coding, minimal distractions, focused speech |

---

## 🤖 Models Deep Dive

### 1. Vision — YOLOv8-nano (Desk Distraction Detector)

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv8-nano + custom detection head |
| **Training Dataset** | COCO 2017 desk-relevant subset (~389 images) |
| **Classes** | `phone`, `monitor`, `work_tool`, `distraction` |
| **Input** | 640×640 RGB screenshot |
| **Output** | 4 features: `phone_visible`, `monitor_count`, `distraction_count`, `focus_ratio` |
| **Format** | ONNX (opset 12) |
| **Size** | ~12 MB |
| **mAP@0.5** | 0.427 |
| **Training** | 50 epochs on T4 GPU |

The vision model detects workspace objects from webcam/screenshots. It's fine-tuned from the official YOLOv8-nano checkpoint on a curated COCO subset containing desk-relevant objects. The raw detections are post-processed into 4 normalized features before being sent to the meta-classifier.

### 2. Audio — XGBoost (Speech Pattern Classifier)

| Property | Value |
|----------|-------|
| **Architecture** | XGBoost (300 trees, max depth 6) |
| **Training Dataset** | RAVDESS emotional speech corpus (1,440 files) |
| **Classes** | `Erratic`, `Slow/Labored`, `Normal/Focused`, `Fast/Energized`, `Rapid/Scattered` |
| **Input** | 52-dim feature vector (13 MFCCs + 12 chroma + spectral features) |
| **Output** | 4 features: `speech_class`, `confidence`, `wpm_norm`, `fluency_score` |
| **Format** | ONNX |
| **Size** | ~2.4 MB |
| **CV F1 (macro)** | 0.646 |

The audio model classifies speech patterns from microphone input. Raw audio is processed in the browser using Web Audio API to extract 52 librosa-style features per window. The XGBoost model was trained on RAVDESS emotional speech data, where emotions were mapped to speech-energy patterns relevant to cognitive state analysis.

### 3. NLP — DistilBERT (Task Classifier)

| Property | Value |
|----------|-------|
| **Architecture** | DistilBERT-base-uncased + classification head |
| **Training Dataset** | 5,044 augmented task descriptions |
| **Classes** | `DEEP_WORK`, `SHALLOW_WORK`, `CREATIVE`, `ADMINISTRATIVE`, `COMMUNICATION` |
| **Input** | Text (max 128 WordPiece tokens) |
| **Output** | 3 features: `task_class`, `cognitive_demand`, `confidence` |
| **Format** | ONNX (opset 18, single file) |
| **Size** | ~256 MB |
| **Accuracy** | 94.7% |
| **F1 (macro)** | 0.947 |
| **Augmentation** | Word dropout, synonym replacement, typos, cross-category noise |

The NLP model classifies the type of task you're working on from text descriptions (e.g., browser tab titles). It uses a fine-tuned DistilBERT with a WordPiece tokenizer implemented in pure JavaScript for browser inference. Each task class is mapped to a cognitive demand score (e.g., `DEEP_WORK` → 0.9, `SHALLOW_WORK` → 0.2).

### 4. Meta-Classifier — Random Forest (Flow State Fusion)

| Property | Value |
|----------|-------|
| **Architecture** | Random Forest (300 trees, max depth 6) + Platt Scaling calibration |
| **Training Dataset** | 2,000 simulated multimodal feature vectors |
| **Input** | 11-dim fused vector (4 vision + 4 audio + 3 NLP) |
| **Output** | 5-class flow state + calibrated probabilities |
| **Format** | ONNX |
| **Size** | ~9.9 MB |
| **CV F1 (macro)** | 0.697 |
| **ECE** | 0.092 |

The meta-classifier fuses all 11 features from the three upstream models into a single cognitive flow state prediction. It uses Platt scaling for probability calibration, ensuring that reported confidence values are meaningful (e.g., 80% confidence means the model is correct ~80% of the time).

**Top Feature Importances:**
1. `cognitive_demand_score` (18.1%)
2. `tab_count_norm` (15.2%)
3. `phone_visible` (15.2%)
4. `focus_ratio` (13.9%)
5. `distraction_count_norm` (13.6%)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (for serve.py)
- **Google Chrome** (recommended browser)
- **Google Colab account** (free, for training models)

### 1. Clone the repository

```bash
git clone https://github.com/suryanarayan100406/aiml.git
cd aiml/ani-flow-optimizer
```

### 2. Train models on Google Colab

See [Training on Google Colab](#-training-on-google-colab) below for detailed instructions.

### 3. Place trained models

After training, download the output files and place them in the `models/` directory:

```
models/
├── desk_distraction_v1.onnx       # Vision model (~12 MB)
├── speech_classifier.onnx         # Audio model (~2.4 MB)
├── task_nlp_classifier.onnx       # NLP model (~256 MB)
├── meta_flow_classifier.onnx      # Meta classifier (~9.9 MB)
├── vision_class_mapping.json      # Vision class metadata
├── vocab.txt                      # DistilBERT vocabulary (30k tokens)
└── *_metrics.json                 # Training metrics for each model
```

### 4. Start the server

```bash
python serve.py
```

### 5. Open the dashboard

Navigate to **http://localhost:8080/frontend/** in Chrome.

You should see the ANI dashboard with "🟢 4/4 models loaded" in the bottom-left corner.

---

## 📚 Training on Google Colab

Each script in `colab/` is **self-contained** — paste the entire file into a Colab cell and run. Each script handles its own dependency installation, data download, training, evaluation, and ONNX export.

### Training Order

> ⚠️ **Script 4 must be run LAST** — it depends on outputs from scripts 1-3.

| # | Script | GPU | Time | Output | Key Metric |
|---|--------|-----|------|--------|------------|
| 1 | `1_train_vision_yolov8.py` | ✅ T4+ | ~15 min | `desk_distraction_v1.onnx` | mAP@0.5: 0.43 |
| 2 | `2_train_audio_xgboost.py` | ❌ CPU | ~5 min | `speech_classifier.onnx` | F1: 0.65 |
| 3 | `3_train_nlp_distilbert.py` | ✅ T4+ | ~10 min | `task_nlp_classifier.onnx` | Acc: 94.7% |
| 4 | `4_train_meta_classifier.py` | ❌ CPU | ~2 min | `meta_flow_classifier.onnx` | F1: 0.70 |

### Step-by-Step Instructions

1. **Open Google Colab** → `File` → `New notebook`
2. **Set GPU runtime**: `Runtime` → `Change runtime type` → `T4 GPU` (for scripts 1 and 3)
3. **Paste the script**: Copy the entire content of a `colab/*.py` file into a code cell
4. **Run the cell**: Press `Shift+Enter` — the script handles everything automatically
5. **Download outputs**: After each script completes, download the generated files:
   - Script 1: `desk_distraction_v1.onnx`, `vision_class_mapping.json`, `vision_metrics.json`
   - Script 2: `speech_classifier.onnx`, `audio_metrics.json`
   - Script 3: `task_nlp_classifier.onnx`, `vocab.txt`, `nlp_metrics.json`
   - Script 4: `meta_flow_classifier.onnx`, `meta_metrics.json` (or download the full zip)
6. **Place files**: Copy all downloaded files to the `models/` directory

### Training Data Details

| Model | Dataset | Source | Size | Preprocessing |
|-------|---------|--------|------|---------------|
| Vision | COCO 2017 desk subset | COCO API (auto-downloaded) | 389 images | Filtered for laptop, phone, book, keyboard, mouse |
| Audio | RAVDESS | Zenodo (auto-downloaded) | 1,440 wav files | 52-dim librosa features (MFCCs + chroma + spectral) |
| NLP | Synthetic task descriptions | Generated in-script | 5,044 samples | Word dropout, synonym replace, typos |
| Meta | Simulated feature vectors | Generated from model outputs | 2,000 vectors | Uses real model predictions where available |

---

## 💻 Running the Frontend

### Development Server

```bash
python serve.py
```

This starts a local HTTP server at `http://localhost:8080` with:
- **CORS headers** enabled for ONNX Runtime WebAssembly
- **Correct MIME types** for `.onnx`, `.wasm`, `.json`, `.js` files
- **Cache-Control headers** to prevent stale JS during development
- **Model status display** showing which models are found on disk

### What Happens on Load

1. **Loading overlay** appears with progress bar
2. **ONNX Runtime Web** is loaded from CDN
3. **All 4 models** are downloaded and initialized (may take 30-60s for the 256MB NLP model)
4. **Dashboard appears** with "🟢 4/4 models loaded" status
5. **Demo mode** activates automatically if any models are missing

### Browser Requirements

- **Chrome 90+** (recommended) or any browser supporting WebAssembly
- **Webcam access** (optional, for vision features)
- **Microphone access** (optional, for audio features)

---

## 🔌 Chrome Extension

ANI includes an optional Chrome extension (`chrome_extension/`) that provides real tab data to the dashboard.

### Features
- Tracks active tab count and tab switching frequency
- Sends tab title to the NLP model for task classification
- Monitors focus patterns and context switching behavior

### Installation
1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" → Select the `chrome_extension/` folder
4. Click the ANI icon in the toolbar to connect

---

## 📁 Project Structure

```
ani-flow-optimizer/
│
├── 📂 colab/                          # Google Colab training scripts
│   ├── 1_train_vision_yolov8.py       # YOLOv8 fine-tuning on COCO desk subset
│   ├── 2_train_audio_xgboost.py       # XGBoost on RAVDESS audio features
│   ├── 3_train_nlp_distilbert.py      # DistilBERT on augmented task descriptions
│   ├── 4_train_meta_classifier.py     # RF meta-classifier + ONNX packaging
│   ├── ANI_Flow_Training.ipynb        # Combined training notebook
│   └── README.md                      # Colab-specific instructions
│
├── 📂 frontend/                       # Browser-based dashboard
│   ├── index.html                     # Main application (single page)
│   ├── css/
│   │   └── styles.css                 # Dark theme UI stylesheet
│   └── js/
│       ├── inference_pipeline.js      # ONNX model orchestrator (load → run → fuse)
│       ├── ui_controller.js           # Dashboard state management & rendering
│       ├── vision_preprocessor.js     # Webcam capture & YOLOv8 pre/post-processing
│       ├── audio_extractor.js         # Web Audio API → 52-dim feature extraction
│       ├── nlp_tokenizer.js           # WordPiece tokenizer (pure JS, no dependencies)
│       ├── ani_guardian.js            # Adaptive alert system (distraction warnings)
│       └── user_profile.js           # Session history & user preferences
│
├── 📂 models/                         # Trained model files (git-ignored)
│   ├── desk_distraction_v1.onnx       # Vision: YOLOv8-nano (12 MB)
│   ├── speech_classifier.onnx        # Audio: XGBoost (2.4 MB)
│   ├── task_nlp_classifier.onnx      # NLP: DistilBERT (256 MB)
│   ├── meta_flow_classifier.onnx     # Meta: Random Forest (9.9 MB)
│   ├── vocab.txt                     # DistilBERT vocabulary (30k tokens)
│   ├── vision_class_mapping.json     # Vision class metadata
│   └── *_metrics.json                # Training metrics for each model
│
├── 📂 chrome_extension/              # Chrome extension for real tab data
│   ├── manifest.json                 # Extension manifest (Manifest V3)
│   ├── background.js                 # Tab tracking service worker
│   ├── content.js                    # Page content extraction
│   ├── popup.html/js                 # Extension popup UI
│   └── data_logger.js                # Session data logging
│
├── 📂 training/                      # Local training scripts (alternative to Colab)
│   ├── download_coco_desk_subset.py  # COCO data downloader
│   ├── train_vision_yolov8.py        # Vision model training
│   ├── train_audio_xgboost.py        # Audio model training
│   ├── train_nlp_distilbert.py       # NLP model training
│   ├── train_meta_classifier.py      # Meta-classifier training
│   ├── feature_extraction.py         # Feature engineering utilities
│   ├── evaluate_all_models.py        # Cross-model evaluation
│   └── user_calibration.py           # Runtime user calibration
│
├── 📂 data/                          # Training data (git-ignored)
│   ├── coco_desk_subset/             # Downloaded COCO images + annotations
│   ├── processed/                    # Preprocessed features
│   └── scripts/                      # Data generation scripts
│
├── serve.py                          # Local HTTP server (CORS + MIME types)
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── .gitignore                        # Git exclusions
└── README.md                         # This file
```

---

## 📊 Dashboard Features

### 🏠 Dashboard Panel
- **Current Flow State** — Large radial gauge showing your active cognitive state with animated transitions
- **Vision Card** — Real-time desk environment analysis (tabs, phone, distractions, focus ratio)
- **Audio Card** — Speech pattern metrics with live waveform visualizer (speech class, WPM, fluency, confidence) and microphone controls
- **NLP Card** — Task classification (task type, cognitive demand, confidence)
- **Session Timeline** — Time-series chart of flow state changes over your work session
- **Top Features** — Live feature importance breakdown showing which signals drive the current prediction

### ▶️ Session Panel
- Start/stop recording sessions
- Real-time inference loop (runs all 4 models every 2 seconds)
- Session duration tracking

### 🧩 Models Panel
- Individual model cards with architecture details
- Live load status badges (`Loaded ✅` / `Not loaded`)
- Model metadata (input shape, output classes, target metrics, ONNX format)

### 📜 History Panel
- Past session logs with flow state distributions
- Session duration and average flow quality

### ⚙️ Settings Panel
- Demo mode toggle (use synthetic data when models are unavailable)
- Model path configuration
- Audio sensitivity controls
- Guardian alert preferences

### 🛡️ ANI Guardian
- **Distraction alerts** — Warns when sustained distractions exceed threshold
- **Flow protection** — Suppresses notifications during deep flow
- **Work quality score** — Continuous assessment of cognitive engagement
- **Adaptive suggestions** — Context-aware tips based on current state

---

## 🔧 Technical Details

### Browser Inference Stack

| Layer | Technology |
|-------|-----------|
| Runtime | ONNX Runtime Web 1.17+ (WebAssembly backend) |
| Model Format | ONNX (opset 12-18) |
| Vision Capture | `navigator.mediaDevices.getUserMedia()` + Canvas |
| Audio Processing | Web Audio API (`AudioWorklet` / `ScriptProcessorNode`) |
| NLP Tokenization | Pure JS WordPiece tokenizer (no external dependencies) |
| UI Framework | Vanilla JS + CSS (no React/Vue/Angular) |
| Typography | Inter + JetBrains Mono (Google Fonts) |

### Feature Vector Format

The 11-dimensional feature vector passed to the meta-classifier:

```
Index  Feature                Source    Range
──────────────────────────────────────────────
0      tab_count_norm         Vision    [0, 1]
1      phone_visible          Vision    {0, 1}
2      distraction_count_norm Vision    [0, 1]
3      focus_ratio            Vision    [0, 1]
4      speech_class           Audio     {0,1,2,3,4}
5      speech_confidence      Audio     [0, 1]
6      wpm_norm               Audio     [0, 1]
7      fluency_score          Audio     [0, 1]
8      task_class_encoded     NLP       {0,1,2,3,4}
9      cognitive_demand_score NLP       [0, 1]
10     task_confidence        NLP       [0, 1]
```

### Performance

| Model | Browser Load Time | Inference Time (per frame) |
|-------|------------------|--------------------------|
| Vision (YOLOv8) | ~2s | ~150ms |
| Audio (XGBoost) | ~0.5s | ~5ms |
| NLP (DistilBERT) | ~30-60s | ~50ms |
| Meta (RF) | ~1s | ~2ms |
| **Total Pipeline** | ~35-65s (first load) | **~210ms per cycle** |

---

## ❓ Troubleshooting

### Models show "Not loaded"

1. Ensure all `.onnx` files are in the `models/` directory
2. Hard-refresh the browser with `Ctrl+Shift+R` to clear cache
3. Check browser console (`F12`) for model loading errors
4. Verify the server shows "✅ All models found!" at startup

### NLP model fails to load

The NLP model must be exported as a **single ONNX file** (not split format). If you see a `task_nlp_classifier.onnx.data` file alongside the `.onnx` file, re-export using the fix in `colab/3_train_nlp_distilbert.py` which merges external data.

### Webcam/Microphone not working

- Chrome requires **HTTPS or localhost** for media access
- Ensure you're accessing via `http://localhost:8080` (not a file:// URL)
- Check browser permissions for camera and microphone

### Slow first load

The NLP model is ~256MB. First load downloads this from the server. Subsequent loads may be cached by the browser. Consider quantizing the model for faster loads.

### Port 8080 already in use

```bash
# Windows
Get-Process -Name python | Stop-Process -Force

# Then restart
python serve.py
```

---

## 📝 License

This project is part of the AI/ML coursework and is provided for educational purposes.

---

## 🙏 Acknowledgments

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — Vision backbone
- **[Hugging Face Transformers](https://huggingface.co/)** — DistilBERT model
- **[RAVDESS Dataset](https://zenodo.org/record/1188976)** — Emotional speech data
- **[COCO Dataset](https://cocodataset.org/)** — Object detection data
- **[ONNX Runtime Web](https://onnxruntime.ai/)** — Browser inference engine
- **[XGBoost](https://xgboost.readthedocs.io/)** — Gradient boosting framework
