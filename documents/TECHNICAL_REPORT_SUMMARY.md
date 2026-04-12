# ANI Flow Optimizer: Technical Report Summary

This document provides a comprehensive overview of the data pipeline and machine learning architecture for the ANI Flow Optimizer, formatted for inclusion in a formal project report.

---

## 1. Data Collection Methods

The project utilizes a hybrid data collection strategy, combining curated public datasets with synthetic data generation to ensure high performance in niche productivity contexts.

*   **Computer Vision (Object Detection)**: 
    *   **Source**: COCO (Common Objects in Context) dataset subset.
    *   **Focus**: Identification of "distraction triggers" (e.g., smartphones, game controllers) and "context markers" (e.g., laptops, keyboards).
*   **Computer Vision (Screen Classification)**: 
    *   **Method**: Knowledge Distillation from the **CLIP (Contrastive Language-Image Pretraining)** model.
    *   **Generation**: Thousands of synthetic screenshots were generated representing five distinct categories (Code, Docs, Communication, Distraction, Neutral) to train a lightweight mobile-ready classifier.
*   **Audio Intelligence**: 
    *   **Source**: **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song).
    *   **Utility**: Trained models to identify vocal energy, tone, and speech activity as proxies for cognitive load and stress levels.
*   **NLP & Task Analysis**: 
    *   **Method**: Heuristic and rule-based mapping.
    *   **Dataset**: A proprietary mapping of over 200 creative software domains (e.g., Figma, Blender, VS Code) and productivity-related keywords was curated to classify application usage patterns.

---

## 2. Data Preprocessing Methods

Prior to inference, all signals undergo rigorous preprocessing to ensure consistency and minimize computational overhead in the browser environment.

### A. Vision Pipeline
*   **Normalization**: Images are resized to **224x224** (for Screen Classification) or **640x640** (for YOLO Object Detection) and pixel values are normalized to a [0, 1] range.
*   **Motion Filtering**: Successive frames are compared using pixel-diffing to reduce redundant inference calls when the user's environment is static.

### B. Audio Pipeline
*   **Feature Extraction**: Raw audio streams are processed using the WebAudio API to extract:
    *   **RMS Energy**: Measuring volume density.
    *   **Zero-Crossing Rate**: Detecting speech vs. background noise.
    *   **Frequency Centroids**: Understanding the "tone" of the activity.
*   **Thresholding**: Noise-gating is applied to filter out ambient background hum before features are passed to the classifier.

### C. Metadata Pipeline
*   **Domain Extraction**: Full URLs are stripped to extract the primary TLD (e.g., `github.com`) to prevent privacy leaks.
*   **Tokenization**: Keyword frequency is calculated for active window titles, focusing on high-intent verbs (e.g., "coding", "designing", "writing").

---

## 3. Model Training Methods

The architecture relies on a "Hierarchical Fusion" approach, where specialized lightweight models feed their outputs into a master "Meta-Classifier."

*   **Vision (YOLOv8)**: Trained using Supervised Learning on a filtered desk-object dataset. Optimization involved converting the PyTorch model to **ONNX Quantized FP16** to ensure accessibility on low-power device hardware.
*   **Screen Classifier (MobileNetV3-Small)**: Trained via **Knowledge Distillation**. By using CLIP as a "teacher" model to label synthetic screenshots, ANI's student model learned to identify complex UI patterns with a footprint of only **0.28MB**.
*   **Audio (XGBoost/Random Forest)**: Trained on the RAVDESS-extracted features to perform multi-class classification (Valence/Activation). The models were optimized for high recall on "High Activation" states.
*   **Meta-Classifier (Random Forest)**: 
    *   **Method**: Supervised Learning on **2,000 fused feature vectors**.
    *   **Fusion**: It takes the softmax outputs of the Vision, Audio, and NLP models as input features to arrive at a final "Flow State" prediction.

---

## 4. Model Accuracy & Performance Metrics

The system demonstrates high reliability across varied environments, with specific emphasis on preventing false positives for "Distracted" states.

| Model Component | Metric Type | Accuracy / Value | Note |
|:--- |:--- |:--- |:--- |
| **Meta-Classifier** | Average F1 Macro | **~84%** (Training) | Measures consistency of Deep Flow detection. |
| **Meta-Classifier** | Cross-Val F1 | **~70%** | Robustness across unseen user behaviors. |
| **Screen Classifier** | Top-1 Accuracy | **~100%** | Exceptionally high on synthetic UI patterns. |
| **Vision (YOLO)** | Mean AP (mAP) | **~85%** | Reliable tracking of desk distractions. |
| **Audio Energy** | Precision | **~80%** | Stability in distinguishing work from casual talk. |

### Performance Benchmarks
*   **Inference Latency**: Average **15-40ms** per model on modern hardware.
*   **Model Size**: Total combined binary size **< 15MB** (excluding unused legacy models), ensuring fast load times in the browser.
