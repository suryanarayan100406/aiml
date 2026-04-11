@echo off
echo ================================================
echo  Copying Models + Pushing to GitHub
echo ================================================

cd /d "c:\Users\samai\Desktop\codes backup\aiml\ani-flow-optimizer"

echo.
echo [1/3] Copying model files to models/...
copy /Y "output\5\screen_classifier.onnx" "models\screen_classifier.onnx"
copy /Y "output\5\screen_classifier.onnx.data" "models\screen_classifier.onnx.data"
copy /Y "output\5\screen_class_mapping.json" "models\screen_class_mapping.json"
copy /Y "output\5\screen_metrics.json" "models\screen_metrics.json"

echo.
echo [2/3] Staging all files...
git add -A
git status

echo.
echo [3/3] Committing and pushing...
git commit -m "feat: replace 256MB DistilBERT with lightweight keyword classifier (Model 3)

BREAKING CHANGE: task_nlp_classifier.onnx is no longer loaded
  - Replaced 66M-param DistilBERT ONNX with pure-JS keyword engine
  - Model size: 256 MB -> 0 MB (zero model files)
  - Load time: 30-60s -> 0s (instantaneous)
  - Inference: ~50ms -> <1ms
  - Total pipeline: ~287 MB -> ~31 MB ONNX footprint

New NLP Classifier (nlp_tokenizer.js):
  - 275+ weighted keywords across 5 task classes
  - URL domain hints (github.com -> DEEP_WORK, figma.com -> CREATIVE, etc.)
  - Regex app name detection (file.py - VS Code -> DEEP_WORK)
  - Distraction domain detection (youtube.com, reddit.com -> low demand)
  - Temperature-scaled softmax (tau=1.5) for calibrated probabilities

Pipeline changes:
  - inference_pipeline.js: removed NLP ONNX model loading
  - serve.py: removed task_nlp_classifier.onnx from expected models
  - TECHNICAL_DOCUMENTATION.md: v3.1 - complete NLP section rewrite

Previous DistilBERT training script preserved in colab/3_train_nlp_distilbert.py"

git push origin main

echo.
echo ================================================
echo  DONE! Push complete.
echo ================================================
pause
