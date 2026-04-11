@echo off
echo ================================================
echo  Copying Screen Classifier + Pushing to GitHub
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
git commit -m "feat: add trained MobileNetV3 screen classifier (Model 5)

Model Files:
- screen_classifier.onnx + .onnx.data (~6.1MB total)
- screen_class_mapping.json (5 classes + productivity scores)
- screen_metrics.json (val F1=1.0, 10 epochs, all classes perfect)

New Code:
- colab/5_train_screen_classifier.py (training pipeline + onnxscript fix)
- frontend/js/screen_classifier.js (browser ONNX inference module)

Pipeline Updates:
- Webcam frames -> YOLO (desk detection, unchanged)
- Screen captures -> MobileNetV3 (productivity classification, NEW)
- Screen productivity score overrides focus_ratio (Option A)

UI Updates:
- New Screen card: activity, productivity, confidence, per-class bars
- index.html version bumped to v8

Server Updates:
- serve.py: added .data MIME type, screen_classifier.onnx to startup check

Documentation:
- TECHNICAL_DOCUMENTATION.md v3.0.0 audit-corrected:
  model size 5->6.1MB, frozen layers clarified, actual training
  results added, resource totals updated, stale notes removed"

git push origin main

echo.
echo ================================================
echo  DONE! Everything pushed to GitHub.
echo ================================================
pause
