# 🧠 ANI Creative Flow Optimizer — Technical Documentation

> **Complete system documentation covering architecture, models, datasets, training pipelines, inference engine, and deployment.**

**Version:** 3.0.0  
**Last Updated:** April 2026  
**Author:** Suryanarayan (AI/ML Coursework)  
**Repository:** [github.com/suryanarayan100406/aiml](https://github.com/suryanarayan100406/aiml)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Flow & Pipeline](#3-data-flow--pipeline)
4. [Model 1 — Vision (YOLOv8-nano)](#4-model-1--vision-yolov8-nano)
5. [Model 2 — Audio (XGBoost)](#5-model-2--audio-xgboost)
6. [Model 3 — NLP (DistilBERT)](#6-model-3--nlp-distilbert)
7. [Model 4 — Meta-Classifier (Random Forest)](#7-model-4--meta-classifier-random-forest)
8. [Model 5 — Screen Classifier (MobileNetV3-Small)](#8-model-5--screen-classifier-mobilenetv3-small)
9. [Voice State Engine (DSP)](#9-voice-state-engine-dsp)
10. [ANI Guardian (Decision Engine)](#10-ani-guardian-decision-engine)
11. [Chrome Extension](#11-chrome-extension)
12. [Frontend Architecture](#12-frontend-architecture)
13. [Deployment & Setup](#13-deployment--setup)
14. [Datasets Reference](#14-datasets-reference)
15. [Performance Benchmarks](#15-performance-benchmarks)
16. [Future Roadmap](#16-future-roadmap)

---

## 1. Project Overview

### What is ANI?

**ANI (Adaptive Neural Intelligence)** is a **multimodal cognitive flow state analyzer** that monitors a user's workspace in real-time through three sensory channels:

| Channel | What It Monitors | Technology |
|---------|-----------------|------------|
| 👁️ **Vision** | Desk environment — phone, monitors, distractions | YOLOv8-nano via webcam |
| 🖥️ **Screen** | Screen content productivity — coding vs distraction | MobileNetV3-Small via screen share |
| 🎙️ **Audio** | Voice energy, tone, and activity patterns | XGBoost + Web Audio DSP |
| 📝 **Text** | Task type & cognitive demand from tab/task titles | DistilBERT transformer |

These signals are **fused into an 11-dimensional feature vector** (screen productivity feeds into `focus_ratio`) and passed through a **calibrated Random Forest meta-classifier** to produce a final prediction of the user's cognitive state.

### Five Flow States

| # | State | Emoji | Description | Example Scenario |
|---|-------|-------|-------------|-----------------|
| 0 | **PSEUDO_WORKING** | 🔴 | Appears busy but not productive | Scrolling social media with VS Code open |
| 1 | **TASK_SWITCHING** | 🟠 | Rapid context switching | Jumping between 10+ tabs every minute |
| 2 | **DISTRACTED** | 🟡 | Multiple distractions present | Phone visible, erratic voice, shallow task |
| 3 | **SOFT_FLOW** | 🟢 | Good focus, moderate engagement | Writing documentation with steady pace |
| 4 | **DEEP_FLOW** | 🟣 | Peak cognitive performance | Complex coding, minimal distractions |

### Key Design Principles

1. **Privacy-First**: All inference runs entirely in the browser — **no data leaves the machine**.
2. **Language-Agnostic**: Audio analysis uses DSP features (energy, pitch, spectral centroid), not speech recognition.
3. **Graceful Degradation**: Each modality can operate independently; missing models trigger demo mode.
4. **ONNX Portable**: All models are exported to ONNX format for universal browser/edge inference.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     BROWSER (ONNX Runtime Web + WebAssembly)                 │
│                                                                              │
│  ┌─────── VISION PIPELINE ──────┐                                           │
│  │                               │                                           │
│  │  Webcam ──► YOLOv8-nano       │  ┌────────────────┐  ┌────────────────┐   │
│  │             (640×640, 12 MB)   │  │  🎙️ AUDIO       │  │  📝 NLP         │   │
│  │             ↓                 │  │                │  │                │   │
│  │         desk objects          │  │  XGBoost       │  │  DistilBERT    │   │
│  │         (phone, monitor,      │  │  ONNX: 2.4 MB  │  │  ONNX: 256 MB  │   │
│  │          tools, distractions) │  │  + DSP Engine   │  │                │   │
│  │                               │  │                │  │  Input:        │   │
│  │  Screen ──► MobileNetV3-S     │  │  Input:        │  │  128 WordPiece  │   │
│  │  Capture    (224×224, ~5 MB)   │  │  52-dim vector  │  │  tokens        │   │
│  │             ↓                 │  │                │  │                │   │
│  │         productivity score    │  │  Output:       │  │  Output:       │   │
│  │         (code/docs/chat/      │  │  4 features    │  │  3 features    │   │
│  │          distraction/neutral) │  │  (class, conf, │  │  (class, demand│   │
│  │              ↓                │  │   energy, act) │  │   confidence)  │   │
│  │         → replaces focus_ratio│  │                │  │                │   │
│  └──────────────┬────────────────┘  └───────┬────────┘  └───────┬────────┘   │
│                 │                            │                   │            │
│                 └────────────┐               │       ┌───────────┘            │
│                              ▼               ▼       ▼                        │
│                 ┌───────────────────────────────────────┐                     │
│                 │      FEATURE FUSION (11-dim)          │                     │
│                 │   [4 vision + 4 audio + 3 nlp]        │                     │
│                 │   (focus_ratio from Screen Classifier  │                     │
│                 │    when screen share is active)        │                     │
│                 └──────────────────┬────────────────────┘                     │
│                                    │                                          │
│                           ┌────────┴────────┐                                 │
│                           │  🔀 META-CLF     │                                 │
│                           │  Random Forest   │                                 │
│                           │  + Platt Scaling │                                 │
│                           │  ONNX: 9.9 MB   │                                 │
│                           └────────┬────────┘                                 │
│                                    │                                          │
│                           ┌────────┴────────┐                                 │
│                           │   FLOW STATE    │──▶ ANI Guardian (Decision Tree) │
│                           │   5 classes +   │──▶ Dashboard UI (Charts)        │
│                           │   probabilities │──▶ Session History              │
│                           └─────────────────┘                                 │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  🛡️ ANI GUARDIAN — Adaptive Alert Engine                              │    │
│  │  Decision Tree + Pomodoro Timer + Context-Aware Suggestions          │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
         ▲               ▲                ▲                        ▲
         │               │                │                        │
    ┌────┴─────┐  ┌──────┴──────┐   ┌─────┴─────┐           ┌─────┴─────┐
    │ Webcam   │  │ Screen      │   │ Microphone│           │  Chrome   │
    │ Feed     │  │ Share       │   │ Stream    │           │ Extension │
    └──────────┘  └─────────────┘   └───────────┘           └───────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Runtime** | ONNX Runtime Web 1.17+ (WASM) | Browser-side model inference |
| **Vision Capture** | `getUserMedia()` + Canvas | Webcam frame acquisition |
| **Audio Processing** | Web Audio API `AnalyserNode` | Real-time DSP feature extraction |
| **NLP Tokenizer** | Pure JS WordPiece | 30k-token DistilBERT vocabulary |
| **UI Framework** | Vanilla JS + CSS | Zero-dependency frontend |
| **Training** | Google Colab (free T4 GPU) | Model training in the cloud |
| **Server** | Python `http.server` | Local HTTP with CORS headers |

---

## 3. Data Flow & Pipeline

### Inference Cycle (every 2-5 seconds)

```
1. CAPTURE
   ├── Webcam → 640×640 RGB tensor (NCHW format)
   ├── Screen → 224×224 RGB tensor (ImageNet normalized)
   ├── Microphone → 2048-sample time-domain buffer
   └── Chrome Extension → Active tab title + tab count

2. EXTRACT FEATURES
   ├── Vision (Webcam): YOLOv8 → detections → [tab_count_norm, phone_visible,
   │                                            distraction_count_norm, focus_ratio]
   ├── Screen: MobileNetV3 → [productivity_class, confidence, productivity_score]
   │           → replaces focus_ratio when screen share is active
   ├── Audio: Web Audio DSP → 52-dim librosa-style vector
   │          XGBoost → [speech_class, confidence, wpm_norm, fluency_score]
   │          DSP Engine → [energyLevel, tone, activityPercent]
   └── NLP: WordPiece → DistilBERT → [task_class, cognitive_demand, confidence]

3. FUSE (11-dimensional feature vector)
   Index  Feature                Source    Range
   ─────────────────────────────────────────────
   0      tab_count_norm         Vision    [0, 1]
   1      phone_visible          Vision    {0, 1}
   2      distraction_count_norm Vision    [0, 1]
   3      focus_ratio            Vision    [0, 1]
   4      speech_class           Audio     {0,1,2,3,4}
   5      speech_confidence      Audio     [0, 1]
   6      wpm_norm (energy)      Audio     [0, 1]
   7      fluency_score          Audio     [0, 1]
   8      task_class_encoded     NLP       {0,1,2,3,4}
   9      cognitive_demand_score NLP       [0, 1]
   10     task_confidence        NLP       [0, 1]

4. CLASSIFY
   Meta-classifier → 5-class flow state + calibrated probabilities

5. RESPOND
   ANI Guardian → Context-aware message + action items
   UI Controller → Dashboard charts + metric cards
```

---

## 4. Model 1 — Vision (YOLOv8-nano)

### Specification

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv8-nano (Ultralytics) with custom detection head |
| **Base Model** | `yolov8n.pt` (pretrained on full COCO) |
| **Fine-tuned On** | COCO 2017 desk-relevant subset |
| **Classes (4)** | `phone`, `monitor`, `work_tool`, `distraction` |
| **Input Shape** | `[1, 3, 640, 640]` — RGB, NCHW format |
| **Output Shape** | `[1, 8, 8400]` — 4 box coords + 4 class scores per anchor |
| **ONNX Opset** | 12 |
| **Model Size** | ~12 MB |
| **Confidence Threshold** | 0.35 |
| **NMS IoU Threshold** | 0.45 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640×640 |
| Optimizer | AdamW |
| Learning Rate | 0.001 → 0.01 (cosine) |
| Augmentation | Default Ultralytics mosaic, flip, scale |
| Early Stopping | Patience = 15 |
| Train/Val Split | 80% / 20% |

### Dataset: COCO 2017 Desk Subset

| Property | Value |
|----------|-------|
| **Source** | [COCO 2017 Validation Set](https://cocodataset.org/) |
| **Original Size** | 5,000 images |
| **Filtered Size** | ~389 images (desk-relevant objects only) |
| **Annotation Format** | YOLO format (center_x, center_y, width, height) |
| **Auto-Downloaded** | Yes, via COCO API in the training script |

**COCO Category → ANI Class Mapping:**

| COCO Category ID | COCO Name | ANI Class ID | ANI Class Name |
|:---:|---|:---:|---|
| 77 | Cell Phone | 0 | `phone` |
| 73 | Laptop | 1 | `monitor` |
| 72 | TV/Monitor | 1 | `monitor` |
| 76 | Keyboard | 2 | `work_tool` |
| 74 | Mouse | 2 | `work_tool` |
| 75 | Remote | 3 | `distraction` |
| 84 | Book | 3 | `distraction` |
| 47 | Cup | 3 | `distraction` |
| 44 | Bottle | 3 | `distraction` |

### Output Feature Extraction

The raw YOLO detections are post-processed into 4 normalized features:

```
tab_count_norm         = min(monitor_count / 3, 1.0)
phone_visible          = 1 if any phone detection with conf > 0.35, else 0
distraction_count_norm = min(distraction_count / 5, 1.0)
focus_ratio            = 1 - (distraction_count / total_detections)  [clamped 0.3-1.0]
```

### Key Metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.427 |
| mAP@0.5:0.95 | ~0.25 |
| Inference Time (Browser) | ~150ms per frame |
| Load Time | ~2s |

---

## 5. Model 2 — Audio (XGBoost)

### Specification

| Property | Value |
|----------|-------|
| **Architecture** | XGBoost Gradient Boosted Trees |
| **Hyperparameters** | 300 trees, max depth 6, lr=0.05 |
| **Input** | 52-dimensional feature vector |
| **Output** | 5 speech classes + probabilities |
| **ONNX Export** | via `onnxmltools` |
| **Model Size** | ~2.4 MB |

### Speech Classes

| Class ID | Class Name | Description | Mapped From (RAVDESS) |
|:---:|---|---|---|
| 0 | `ERRATIC_SPEECH` | Irregular, stressed speech | Angry, Fearful |
| 1 | `SLOW_LABORED` | Slow, deliberate, struggling | Sad |
| 2 | `NORMAL_FOCUSED` | Steady, focused engagement | Neutral, Calm |
| 3 | `FAST_ENERGIZED` | Energetic, fast-paced speech | Happy, Surprised |
| 4 | `RAPID_SCATTERED` | Very fast, scattered, agitated | Disgust |

### 52-Dimensional Feature Vector

```
Index     Feature                    Extraction Method
───────────────────────────────────────────────────────
0-12      MFCC means (13 coeffs)    librosa.feature.mfcc() → mean
13-25     MFCC std devs             librosa.feature.mfcc() → std
26-38     MFCC delta means          librosa.feature.delta(mfcc) → mean
39-41     MFCC delta-delta (3)      librosa.feature.delta(mfcc, order=2) → mean
42        Spectral centroid         librosa.feature.spectral_centroid()
43        Spectral rolloff (85%)    librosa.feature.spectral_rolloff()
44        Zero crossing rate        librosa.feature.zero_crossing_rate()
45        RMS energy                librosa.feature.rms()
46        Pitch (F0) mean           librosa.piptrack()
47        Pitch (F0) variance       librosa.piptrack()
48        Tempo (BPM)               librosa.beat.beat_track()
49        WPM proxy (onset rate)    librosa.onset.onset_detect()
50        Inter-onset variance      np.diff(onset_times)
51        Silence ratio             RMS < 0.01 frame ratio
```

### Dataset: RAVDESS

| Property | Value |
|----------|-------|
| **Full Name** | Ryerson Audio-Visual Database of Emotional Speech and Song |
| **Source** | [Zenodo Record 1188976](https://zenodo.org/record/1188976) |
| **Size** | 1,440 speech audio files (WAV, 16-bit, 48kHz) |
| **Actors** | 24 professional actors (12 male, 12 female) |
| **Emotions** | 8: neutral, calm, happy, sad, angry, fearful, disgust, surprise |
| **Statements** | 2 lexically-matched sentences per emotion |
| **Intensity** | 2 levels: normal and strong |
| **License** | CC BY-NC-SA 4.0 |
| **Auto-Downloaded** | Yes, via Zenodo URL in training script |

**RAVDESS Filename Format:**
```
03-01-05-01-01-02-12.wav
 │  │  │  │  │  │  │
 │  │  │  │  │  │  └── Actor (01-24)
 │  │  │  │  │  └───── Repetition (01-02)
 │  │  │  │  └──────── Statement (01-02)
 │  │  │  └─────────── Emotion intensity (01=normal, 02=strong)
 │  │  └────────────── Emotion (01-08)
 │  └───────────────── Vocal channel (01=speech, 02=song)
 └──────────────────── Modality (01=full-AV, 02=video, 03=audio)
```

### Class Imbalance Handling

The training script detects class imbalance and applies **random oversampling with noise injection**:
- If `max_class_count / min_class_count > 3x`, oversampling activates
- Minority class samples are duplicated with `±0.05` Gaussian noise added
- Empty classes get 50 synthetic samples from the dataset mean + std

### Key Metrics

| Metric | Value |
|--------|-------|
| CV F1 (macro, 5-fold) | 0.646 ± 0.03 |
| Training Accuracy | ~0.95 |
| Inference Time (Browser) | ~5ms |
| Load Time | ~0.5s |

---

## 6. Model 3 — NLP (DistilBERT)

### Specification

| Property | Value |
|----------|-------|
| **Architecture** | DistilBERT-base-uncased + 5-class linear head |
| **Base Model** | `distilbert-base-uncased` (HuggingFace) |
| **Total Params** | 66M (only ~3.4M trainable) |
| **Frozen Layers** | All except `transformer.layer.5` + `classifier` |
| **Input** | Text string → 128 WordPiece tokens |
| **Output** | 5 task classes + logits |
| **ONNX Opset** | 14 |
| **Model Size** | ~256 MB (single ONNX file, no external data) |

### Task Classes

| Class ID | Task Type | Cognitive Demand Score | Example |
|:---:|---|:---:|---|
| 0 | `DEEP_WORK` | 0.90 | "Implement gradient descent for the payment module" |
| 1 | `SHALLOW_WORK` | 0.20 | "Fix typo in API reference documentation" |
| 2 | `CREATIVE` | 0.70 | "Design new onboarding flow for enterprise users" |
| 3 | `ADMINISTRATIVE` | 0.30 | "Review and approve 12 pending pull requests" |
| 4 | `COMMUNICATION` | 0.50 | "Draft email to engineering team about deadline" |

### Dataset: Synthetic Augmented Task Descriptions

| Property | Value |
|----------|-------|
| **Source** | Generated from 120+ templates per class |
| **Base Samples** | 2,000 (400 per class) |
| **After Augmentation** | ~5,044 - 8,000+ samples |
| **Cross-Category Noise** | 5% hard negatives |
| **Template Variables** | 30+ fill-value categories |

**Augmentation Techniques (4 methods):**

| Technique | Probability | Description |
|-----------|:-----------:|-------------|
| Word Dropout | 10% per word | Randomly removes words from sentence |
| Synonym Replacement | 15% per word | Swaps words with WordNet synonyms |
| Character Typos | 2% per char | Swap, delete, insert, or replace characters |
| Word Swap | Every 3rd sample | Swap two adjacent words |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 8 (+ early stopping, patience 3) |
| Batch Size | 16 |
| Max Sequence Length | 128 tokens |
| Learning Rate | 2e-5 |
| Warmup Steps | 200 |
| Weight Decay | 0.01 |
| FP16 | Yes (on GPU) |
| Metric for Best Model | F1 (macro) |
| Train/Val Split | 80% / 20% (stratified) |

### Browser Tokenizer

A **pure JavaScript WordPiece tokenizer** is included (no external dependencies):
- Loads `vocab.txt` (30,522 tokens from DistilBERT)
- Implements proper `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` handling
- Supports `##` subword continuation tokens
- Falls back to hash-based tokenization if vocab fails to load

### Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 94.7% |
| F1 (macro) | 0.947 |
| Inference Time (Browser) | ~50ms |
| Load Time | ~30-60s (256MB download) |

---

## 7. Model 4 — Meta-Classifier (Random Forest)

### Specification

| Property | Value |
|----------|-------|
| **Architecture** | Random Forest + Platt Calibration |
| **Hyperparameter Search** | GridSearchCV (5-fold) |
| **Best Config** | ~300 trees, max depth 6, balanced class weights |
| **Input** | 11-dimensional fused feature vector |
| **Output** | 5 flow state classes + calibrated probabilities |
| **Calibration** | Platt Scaling (sigmoid, 5-fold CV) |
| **ONNX Export** | via `skl2onnx` |
| **Model Size** | ~9.9 MB |

### Dataset: Simulated Fused Features

| Property | Value |
|----------|-------|
| **Size** | 2,000 samples |
| **Generation Method** | Uses trained models from scripts 1-3 |
| **Vision Features** | Profile-based simulation per flow state |
| **Audio Features** | Real RAVDESS features run through trained XGBoost |
| **NLP Features** | Real task texts run through trained DistilBERT |
| **Labels** | Assigned via principled weighted scoring heuristic |

### Label Assignment Heuristic

The meta-classifier labels are NOT arbitrary — they're computed using a weighted scoring system:

```python
# PSEUDO_WORKING: many tabs, low demand, low fluency
scores[0] = tab*0.3 + (1-demand)*0.25 + (1-fluency)*0.2 + (1-focus)*0.15

# TASK_SWITCHING: many tabs, high speech rate, admin tasks
scores[1] = tab*0.3 + wpm*0.2 + admin_task*0.2 + distr*0.15

# DISTRACTED: phone, distractions, low focus, erratic speech
scores[2] = phone*0.3 + distr*0.25 + (1-focus)*0.2 + erratic*0.15

# SOFT_FLOW: moderate focus, normal speech, decent demand
scores[3] = focus*0.25 + fluency*0.2 + demand*0.2 + (1-distr)*0.15

# DEEP_FLOW: high focus, no distractions, high demand, steady speech
scores[4] = (1-tab)*0.15 + (1-phone)*0.15 + focus*0.2 + demand*0.15
```

### Feature Importances (from Random Forest)

| Rank | Feature | Importance |
|:---:|---|:---:|
| 1 | `cognitive_demand_score` | 18.1% |
| 2 | `tab_count_norm` | 15.2% |
| 3 | `phone_visible` | 15.2% |
| 4 | `focus_ratio` | 13.9% |
| 5 | `distraction_count_norm` | 13.6% |
| 6 | `task_confidence` | 8.1% |
| 7 | `fluency_score` | 5.9% |
| 8 | `speech_confidence` | 4.2% |
| 9 | `wpm_norm` | 3.2% |
| 10 | `speech_class` | 1.6% |
| 11 | `task_class_encoded` | 1.0% |

### Key Metrics

| Metric | Value |
|--------|-------|
| CV F1 (macro) | 0.697 |
| Average ECE (calibration error) | 0.092 |
| Inference Time (Browser) | ~2ms |
| Load Time | ~1s |

---

## 8. Model 5 — Screen Classifier (MobileNetV3-Small)

### Specification

| Property | Value |
|----------|-------|
| **Architecture** | MobileNetV3-Small (torchvision) + custom classifier head |
| **Base Model** | `mobilenet_v3_small` (pretrained on ImageNet) |
| **Training Method** | CLIP knowledge distillation on synthetic screenshots |
| **Classes (5)** | `PRODUCTIVE_CODE`, `PRODUCTIVE_DOCS`, `COMMUNICATION`, `DISTRACTION`, `NEUTRAL` |
| **Input Shape** | `[1, 3, 224, 224]` — RGB, NCHW, ImageNet normalized |
| **Output Shape** | `[1, 5]` — class logits (softmaxed in browser) |
| **ONNX Opset** | 14 |
| **Model Size** | ~5 MB |
| **Normalization** | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

### Screen Productivity Classes

| Class ID | Category | Productivity Score | Examples |
|:---:|---|:---:|---|
| 0 | `PRODUCTIVE_CODE` | 0.95 | VS Code, terminals, IDEs, Jupyter |
| 1 | `PRODUCTIVE_DOCS` | 0.80 | Google Docs, Excel, Notion, Word |
| 2 | `COMMUNICATION` | 0.50 | Slack, Email, Teams, Zoom |
| 3 | `DISTRACTION` | 0.10 | YouTube, Twitter, Reddit, news sites |
| 4 | `NEUTRAL` | 0.40 | File manager, system settings, search |

### Training Pipeline (Knowledge Distillation)

```
Step 1: Generate 2,500 synthetic screenshots (500/class)
        └── Programmatic generation with class-specific visual features:
            - Code: dark bg, line numbers, syntax highlight blocks, sidebars
            - Docs: white bg, headings, paragraph blocks, toolbars
            - Chat: message bubbles, contact lists, input bars
            - Distraction: colorful cards, video thumbnails, social feeds
            - Neutral: file lists, settings panels, search bars

Step 2: (Optional) CLIP ViT-B/32 label verification
        └── Verify synthetic labels agree with CLIP zero-shot classification

Step 3: Fine-tune MobileNetV3-Small
        └── Freeze early layers, train last 3 blocks + classifier
        └── Classifier head: 576 → 1024 → 5 (with Hardswish + Dropout)

Step 4: Export to ONNX
        └── screen_classifier.onnx (~5 MB)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 15 (+ early stopping, patience 5) |
| Batch Size | 32 |
| Image Size | 224×224 |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealing (η_min=1e-6) |
| Frozen Layers | features[0-8] (early blocks) |
| Trainable Layers | features[-3:] + classifier |
| Augmentation | RandomCrop, HFlip, ColorJitter, Grayscale |
| Train/Val Split | 80% / 20% |

### Integration: Focus Ratio Override (Option A)

The screen classifier feeds into the existing 11-dimension pipeline **without retraining the meta-classifier**:

```
When screen share is active:
    focus_ratio = screen_productivity_score   ← from MobileNetV3
When screen share is inactive:
    focus_ratio = computed from YOLO detections  ← existing logic
```

This means:
- **PRODUCTIVE_CODE detected** → `focus_ratio = 0.95` → pushes toward DEEP_FLOW
- **DISTRACTION detected** → `focus_ratio = 0.10` → pushes toward DISTRACTED
- No retraining of the meta-classifier (script 4) is needed

### Prediction Smoothing

To avoid flickering between classes:
- Maintains a buffer of the last 5 predictions
- Uses **confidence-weighted majority voting** to determine final class
- Clips max confidence at 0.99 to avoid overconfidence

### Fallback: Heuristic Classification

When the ONNX model is not available, a color-analysis heuristic runs:
- **Dark screenshots** (>60% dark pixels) → likely code/IDE
- **Bright screenshots** (>60% bright pixels) → likely docs
- **Colorful screenshots** (high saturation variance) → likely distraction
- **Moderate** → neutral

---

## 9. Voice State Engine (DSP)

### Overview

The Voice State Engine replaced the original WPM-based speech analysis (which was unreliable due to browser limitations). It uses **direct DSP features** from the Web Audio API's `AnalyserNode`:

### Three Voice Metrics

| Metric | Derived From | Levels | Purpose |
|--------|-------------|--------|---------|
| **Energy** | RMS amplitude | Silent → Quiet → Active → Energized | How loud/active the voice is |
| **Tone** | Pitch (F0) + Spectral Centroid | Calm → Neutral → Animated → Stressed | Emotional quality of voice |
| **Activity %** | Rolling boolean history (30 frames) | 0-100% | Percentage of time speaking |

### Energy Level Thresholds

```
RMS < 0.008  →  Silent
RMS < 0.03   →  Quiet
RMS < 0.08   →  Active
RMS >= 0.08  →  Energized
```

### Tone Classification

```
                    Pitch < 150Hz  →  Calm
150Hz ≤ Pitch < 250Hz             →  Neutral
250Hz ≤ Pitch < 350Hz             →  Animated
Pitch ≥ 350Hz OR Centroid > 3kHz  →  Stressed
```

### Hard Gating (Override Logic)

To prevent the XGBoost model from producing false positives on ambient noise:

```javascript
// If RMS is too low for real speech, force "Silent"
if (rms < 0.008) { label = 1; /* Silent */ }

// If RMS is loud but model says "slow", override to "energized"
else if (rms > 0.08 && label < 3) { label = 3; /* Energized */ }
```

---

## 10. ANI Guardian (Decision Engine)

### Overview

The ANI Guardian is a **personality-driven decision engine** that generates contextual coaching messages based on the ML predictions and raw signal analysis.

### Decision Tree Branches

```
Branch 1: tabs > 10 + erratic speech + complex task
  → "TASK SWITCHING OVERLOAD" (severity: HIGH)

Branch 2: phone visible + many distractions (≥3)
  → "DISTRACTED PHONE" (severity: HIGH)

Branch 3: phone visible + low focus (<40%)
  → "DISTRACTED PHONE MILD" (severity: MEDIUM)

Branch 4: slow speech + complex task
  → "STRUGGLING" (severity: MEDIUM, mood: supportive)

Branch 5: many tabs + admin task
  → "PSEUDO WORKING" (severity: MEDIUM)

Branch 6: high focus + no phone + ≤1 distraction + steady speech + complex task
  → "DEEP FLOW" (severity: NONE, mood: happy)

Branch 7: decent focus + no phone + good fluency
  → "SOFT FLOW" (severity: LOW)

Default: falls back to ML prediction class
```

### Pomodoro Focus Timer

When the Guardian detects sustained poor states (3+ consecutive), it offers a 25-minute Pomodoro focus sprint that mutes all alerts.

---

## 11. Chrome Extension

### Manifest V3 Extension

| Property | Value |
|----------|-------|
| **Name** | ANI Flow Data Collector |
| **Version** | 1.1.0 |
| **Permissions** | `tabs`, `activeTab`, `storage`, `alarms` |
| **Communication** | `window.postMessage()` bridge |

### Data Provided

| Field | Description |
|-------|-------------|
| `tabCount` | Total number of open tabs |
| `tabCountNorm` | Normalized tab count (0-1, /30) |
| `distractionTabs` | Tabs classified as distracting |
| `activeTab.title` | Current active tab's title |
| `activeTab.url` | Current active tab's URL |
| `categories` | Tab classification breakdown |
| `productivityScore` | Computed productivity metric |
| `switchRate` | Tab switching frequency |

### Tab Classification Categories

The extension classifies tabs into: `work`, `social`, `entertainment`, `news`, `shopping`, `communication`, `distraction`.

---

## 12. Frontend Architecture

### File Structure

```
frontend/
├── index.html                 # Single-page application (ALL panels)
├── css/
│   └── styles.css             # Dark theme design system (~2000 lines)
└── js/
    ├── inference_pipeline.js  # ONNX orchestrator — load, run, fuse (5 models)
    ├── ui_controller.js       # State machine — panels, charts, toasts
    ├── vision_preprocessor.js # Webcam/screen → 640×640 NCHW tensor
    ├── screen_classifier.js   # Screen → 224×224 → MobileNetV3 productivity
    ├── audio_extractor.js     # Web Audio → 52-dim + Voice State DSP
    ├── nlp_tokenizer.js       # WordPiece tokenizer (pure JS)
    ├── ani_guardian.js        # Decision tree + Pomodoro timer
    └── user_profile.js        # LocalStorage session persistence
```

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| 🏠 **Dashboard** | Flow gauge, 4 metric cards (Vision, Screen, Audio, NLP), timeline chart, top features |
| ▶️ **Session** | Start/stop recording, inference loop, task text input |
| 🧩 **Models** | Individual model cards with load status badges |
| 📜 **History** | Past session logs with flow distributions |
| ⚙️ **Settings** | Demo mode, audio sensitivity, guardian preferences |
| 🔍 **Live Diagnostics** | Real-time feature values for debugging |

---

## 13. Deployment & Setup

### Prerequisites

- Python 3.8+
- Google Chrome 90+
- Google Colab account (free, for training)

### Quick Start

```bash
# 1. Clone
git clone https://github.com/suryanarayan100406/aiml.git
cd aiml/ani-flow-optimizer

# 2. Train models on Google Colab (scripts in colab/)
#    Order: 1_vision → 2_audio → 3_nlp → 4_meta → 5_screen

# 3. Place trained .onnx files in models/

# 4. Start server
python serve.py

# 5. Open http://localhost:8080/frontend/ in Chrome
```

### Server (serve.py)

The Python HTTP server provides:
- CORS headers (`Access-Control-Allow-Origin: *`)
- COEP/COOP headers for SharedArrayBuffer support
- Correct MIME types for `.onnx`, `.wasm`, `.json` files
- Cache-busting headers for development
- Model file presence check at startup

---

## 14. Datasets Reference

### Summary Table

| Dataset | Used By | Source | Size | License | Location |
|---------|---------|--------|------|---------|----------|
| **COCO 2017** | Vision (YOLOv8) | [cocodataset.org](https://cocodataset.org/) | ~389 images | CC BY 4.0 | Auto-downloaded in Colab |
| **RAVDESS** | Audio (XGBoost) | [Zenodo 1188976](https://zenodo.org/record/1188976) | 1,440 WAV files | CC BY-NC-SA 4.0 | Auto-downloaded in Colab |
| **Synthetic Tasks** | NLP (DistilBERT) | Generated in-script | ~5,044 samples | N/A | Generated during training |
| **Simulated Fusion** | Meta-Classifier | Generated from models 1-3 | 2,000 samples | N/A | Generated during training |
| **Synthetic Screenshots** | Screen (MobileNetV3) | Generated in-script | 2,500 images | N/A | Generated during training |

> **Note:** All datasets are automatically downloaded or generated during training. No manual dataset setup is required.

### Detailed Dataset Descriptions

See the `documents/datasets/` directory for sample data files:
- `sample_coco_annotations.json` — Example COCO annotation structure
- `sample_ravdess_features.csv` — Example extracted audio features
- `sample_task_descriptions.csv` — Example NLP training data
- `sample_fused_vectors.csv` — Example meta-classifier training data
- `sample_screen_screenshots.json` — Example screen classifier training metadata

---

## 15. Performance Benchmarks

### Model Performance

| Model | Training Metric | Value | Browser Inference | Load Time |
|-------|----------------|-------|-------------------|-----------|
| Vision (YOLOv8) | mAP@0.5 | 0.427 | ~150ms | ~2s |
| Screen (MobileNetV3) | Val F1 (macro) | TBD* | ~30ms | ~1s |
| Audio (XGBoost) | CV F1 (macro) | 0.646 | ~5ms | ~0.5s |
| NLP (DistilBERT) | Accuracy | 94.7% | ~50ms | ~30-60s |
| Meta (RF) | CV F1 (macro) | 0.697 | ~2ms | ~1s |
| **Full Pipeline** | — | — | **~240ms total** | **~36-66s** |

> *Screen classifier metrics are populated after first Colab training run.

### Resource Usage (Browser)

| Resource | Approximate Usage |
|----------|------------------|
| Total ONNX model size | ~285 MB (12 + 5 + 2.4 + 256 + 9.9) |
| Peak memory (all loaded) | ~420 MB |
| WebAssembly memory | ~128 MB |
| CPU utilization (inference) | ~5-15% (one core) |

---

## 16. Future Roadmap

1. **Meta-Classifier Retraining** — Retrain on real Voice State features (Energy/Tone/Activity) instead of WPM
2. **Screen Classifier Enhancement** — Train on real screenshot datasets (e.g., web-screenshots corpus) instead of synthetic
3. **Model Quantization** — INT8 quantize the DistilBERT model to reduce size from 256MB to ~64MB
4. **WebGPU Inference** — Migrate from WASM to WebGPU when ONNX Runtime adds stable support
5. **Calibration Tuning** — Fine-tune the DSP energy/tone thresholds based on user feedback
6. **Multi-Language Support** — Already language-agnostic for audio; extend NLP templates
7. **Mobile Responsive** — Adapt the dashboard for tablet/mobile form factors

---

*This documentation is kept in sync with the codebase. Updated after each significant change.*
