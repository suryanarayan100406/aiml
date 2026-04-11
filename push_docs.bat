@echo off
echo ================================================
echo  Pushing Screen Classifier to GitHub...
echo ================================================

cd /d "c:\Users\samai\Desktop\codes backup\aiml\ani-flow-optimizer"

git add -A
git status
echo.
echo Committing...
git commit -m "feat: add MobileNetV3-Small screen productivity classifier (Model 5)

New Model:
- colab/5_train_screen_classifier.py: CLIP-distilled MobileNetV3-Small
  training pipeline with synthetic screenshot generation
- frontend/js/screen_classifier.js: Browser ONNX inference module
  with prediction smoothing and heuristic fallback

Pipeline Changes:
- Webcam frames -> YOLO (desk object detection, unchanged)
- Screen captures -> MobileNetV3 (productivity classification, NEW)
- Screen productivity score replaces focus_ratio (Option A, no retrain)

UI Updates:
- New Screen card with activity, productivity, confidence metrics
- Per-class probability bars (Code/Docs/Chat/Distract/Neutral)
- Script version bumped to v8

Documentation:
- Updated TECHNICAL_DOCUMENTATION.md to v3.0.0
- Added Model 5 section with full specification
- Updated architecture diagram showing split pipeline
- Added sample_screen_screenshots.json dataset metadata"

echo.
echo Pushing...
git push origin main

echo.
echo ================================================
echo  DONE! Screen classifier pushed to GitHub.
echo ================================================
pause
