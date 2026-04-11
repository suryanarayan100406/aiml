@echo off
echo ================================================
echo  Copying Screen Classifier to models/
echo ================================================

cd /d "c:\Users\samai\Desktop\codes backup\aiml\ani-flow-optimizer"

copy /Y "output\5\screen_classifier.onnx" "models\screen_classifier.onnx"
copy /Y "output\5\screen_classifier.onnx.data" "models\screen_classifier.onnx.data"
copy /Y "output\5\screen_class_mapping.json" "models\screen_class_mapping.json"
copy /Y "output\5\screen_metrics.json" "models\screen_metrics.json"

echo.
echo Files copied! Now staging and pushing to GitHub...
echo.

git add -A
git status
git commit -m "feat: add trained MobileNetV3 screen classifier (Model 5)

- screen_classifier.onnx + .onnx.data (~6MB total)
- screen_class_mapping.json (5 classes + productivity scores)
- screen_metrics.json (val F1=1.0, 10 epochs)
- Added onnxscript dependency fix in training script
- New screen_classifier.js browser inference module
- Updated inference_pipeline.js for webcam/screen split routing
- Updated index.html with screen analysis card (v8)
- Updated ui_controller.js for screen card rendering
- Updated TECHNICAL_DOCUMENTATION.md to v3.0.0"

git push origin main

echo.
echo ================================================
echo  DONE! All pushed to GitHub.
echo ================================================
pause
