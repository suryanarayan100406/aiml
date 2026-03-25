# 🧠 ANI Creative Flow Optimizer — Usage Guide

This guide explains how to run, install, and use the ANI Creative Flow Optimizer project.

## 1. Quick Start (Demo Mode)
If you just want to see the UI in action without running any Python code or ML models:
1. Open Chrome.
2. Press `Ctrl + O` (or `Cmd + O` on Mac).
3. Select `frontend/index.html` from this repository.
4. Click **Start Session**. In "Demo Mode", the dashboard will simulate cognitive load inference realistically based on your microphone input and task description!

---

## 2. Installing the Chrome Extension
The Chrome extension is required if you want to log real-world usage data to calibrate the meta-classifier.

**How to Install:**
1. Open Chrome and go to `chrome://extensions/`.
2. Turn on **Developer mode** (toggle switch in the top right corner).
3. Click the **Load unpacked** button.
4. Select the `chrome_extension` folder located inside this project directory.
5. The extension will now appear in your browser toolbar. Click the `ANI` icon to log your focus score and export your collected CSV data!

---

## 3. Running the Full Machine Learning Pipeline
If you want to train the actual ONNX models using the Python backend:

### Prerequisites:
Ensure you have Python 3.10+ installed. Then, open a terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### Step 3.1: Generate Synthetic Data
Run the data generators to create the training datasets:
```bash
python data/scripts/generate_synthetic_tasks.py
python data/scripts/generate_synthetic_audio.py
python data/scripts/generate_synthetic_vision.py
python data/scripts/generate_fused_dataset.py
```

### Step 3.2: Train the Models
Execute the training scripts. These scripts will train the models and export them to `.pkl` and `.onnx` formats in the `models/` directory:
```bash
python training/train_audio_xgboost.py
python training/train_meta_classifier.py
```
*(Note: Visual and NLP training scripts require larger libraries like `torch` and `ultralytics` which may take longer to run)*

### Step 3.3: Use the Frontend with Real Models
Once the `.onnx` models are generated in the `models/` directory, open `frontend/index.html` again. The application will detect the ONNX models and switch off Demo Mode, running full client-side inference using your local WebAssembly engine!

---

## Troubleshooting
- **Missing Models Warning:** If the frontend says "Models offline", it means the `.onnx` files are missing from the `models/` folder. It will seamlessly fall back to Demo Mode.
- **Microphone/Screen errors:** Ensure you grant Chrome the necessary permissions when clicking "Start Session".
- **Chrome Extension Icon Error:** If you see `Could not load icon`, ensure the `generate_icons.py` script ran successfully to populate the `chrome_extension/icons/` folder.
