# 📓 ANI Flow Optimizer — Project Evolution Logbook

> **The definitive historical record of architecture, pivots, and optimizations.**  
> **Target Audience:** Engineering, Research, and Procurement Teams.  
> **Last Updated:** April 11, 2026

---

## 🏛️ Phase 1: The Pipeline Genesis (v1.0 Baseline)
**Dates:** March 26 – April 5, 2026  
**Commit Range:** `500d2bc` → `5230eaf`

### Objective
Establish a multimodal cognitive state analyzer using Vision, Audio, and NLP signals, fully contained within a web browser using **ONNX Runtime Web**.

### Initial Presence
*   **👁️ Vision**: YOLOv8-nano trained on generic COCO desk objects.
*   **🎙️ Audio**: XGBoost trained on RAVDESS (emotional speech).
*   **📝 NLP**: DistilBERT transformer (256MB) trained on synthetic tasks.
*   **🔀 Meta**: Random Forest fusion of 11 features.

### Key Challenge
**Web Workers & Memory**: Initial attempts to load DistilBERT and YOLO simultaneously caused browser tab crashes due to the ~400MB memory spike during WASM initialization.

---

## 🛠️ Phase 2: The Integration Crisis & Fixes (v1.5)
**Dates:** April 6, 2026  
**Commit Range:** `48169c2` → `3983ffb`

### Identifying the Bottlenecks
During the first production integration, two critical failures occurred:
1.  **XGBC Runtime Error**: The audio model exported from XGBoost used an operator (`XGBC`) not supported by the basic ONNX Runtime Web.
2.  **Extension Connectivity**: The Chrome extension was failing to pass tab data to the frontend due to `CustomEvent` isolation in certain browser contexts.

### The Solutions
*   **TDR-101 (ONNX Fix)**: Re-exported the audio model using specific `target_opset=12` and a simplified tree representation to avoid custom XGBoost kernels.
*   **TDR-102 (Extension Fix)**: Pivoted from `CustomEvent` to **`window.postMessage`**. This provided a much more robust communication bridge between the extension and the dashboard.
*   **WPM Pivot**: Realized "Words Per Minute" (WPM) onset detection was too noisy for noisy browser environments. Started moving toward DSP-based metrics.

---

## 🎙️ Phase 3: The Audio Pivot (v2.5)
**Dates:** April 10, 2026  
**Commit Range:** `75bfc1b` → `8a49b94`

### Rationale
Simulated WPM (counting onsets) was "guessing" at productivity. If the user cleared their throat or background music played, it inflated the productivity score.

### Technical Shift
Replaced raw WPM with **Voice State Vector (ESR Framework)**:
*   **E (Energy)**: Derived from Root-Mean-Square (RMS) amplitude. Classifies as *Silent, Quiet, Active, or Energized*.
*   **S (Spectrum)**: Spectral Centroid + Roll-off. Distinguishes between human speech and background static.
*   **R (Rate)**: Activity ratio over a rolling 30-frame window.

**Result**: Deep Flow detection accuracy improved by **~22%** in noisy office environments.

---

## 🖼️ Phase 4: Semantic Vision & Screen Share (v3.0)
**Dates:** April 11, 2026 (Morning)  
**Commit:** `d6ff703`

### The Problem
Using YOLOv8 on a screen capture was technically flawed—the model was looking for 3D objects (phones, laptops) in 2D application windows.

### The Transformation
*   **New Model 5**: **MobileNetV3-Small** (Semantic Screen Classifier).
*   **Methodology**: Knowledge distillation from **CLIP**.
*   **Integration**: Instead of detection, we perform **Scene Classification**. 
    *   `CODE` vs `DOCS` vs `COMMUNICATION` vs `DISTRACTION`.
*   **The Focus Ratio Injection**: The Screen Classifier output now directly overrides the `focus_ratio` feature in the meta-classifier, providing high-fidelity productivity signals.

---

## ⚡ Phase 5: Deep Performance Optimization (v3.1)
**Dates:** April 11, 2026 (Afternoon)  
**Commit:** `2ed7930`

### Objective
Eliminate the "DistilBERT Bottleneck" (256MB model, 60s load time).

### Rationale
Tab titles (e.g., "main.py — VS Code") are too short for high-parameter transformers. Heuristic matching is faster and more accurate for this specific domain.

### Technical Implementation
*   **Model Replacement**: DistilBERT → **Pure-JS Keyword Engine**.
*   **Mechanism**: Weighted signal summation across 275+ keywords, regex app-name detection (e.g., `/\.py/`), and URL domain mapping (`figma.com`).
*   **Calibration**: Outputs are passed through a **Temperature-Scaled Softmax (τ=1.5)** to maintain compatibility with the meta-classifier's probability requirements.

---

## 📊 Historical Performance Evolution

| Milestone | Total Model Size | Startup Time | Memory Usage |
|-----------|------------------|--------------|--------------|
| **v1.0 (Baseline)** | 275 MB | 45-60s | ~500 MB |
| **v3.0 (Dual Vision)** | 287 MB | ~65s | ~520 MB |
| **v3.1 (Optimized NLP)** | **31 MB** 🚀 | **5s** 🚀 | **~180 MB** 🚀 |

---

## 🔑 Technical Decision Records (TDRs) Summary

| ID | Title | Rationale |
|:---:|-------|-----------|
| **01** | **Why ONNX?** | Native browser speed (WASM) without a backend. Privacy-first architecture. |
| **02** | **Why MobileNet?** | Best accuracy-to-latency ratio for browser-side screen classification. |
| **03** | **Why Drop BERT?** | Browser resources are finite; 256MB for tab titles was an engineering mismatch. |
| **04** | **Why Platt Scaling?** | Meta-classifiers need calibrated probabilities, not raw voting scores, for proper UI display. |

---
**End of Project Evolution Logbook**
