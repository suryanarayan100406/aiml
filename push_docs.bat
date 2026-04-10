@echo off
echo ================================================
echo  Pushing documentation to GitHub...
echo ================================================

cd /d "c:\Users\samai\Desktop\codes backup\aiml\ani-flow-optimizer"

git add -A
git status
echo.
echo Committing...
git commit -m "docs: add comprehensive technical documentation and sample datasets

- Created documents/TECHNICAL_DOCUMENTATION.md with full system architecture
- Added documents/datasets/ with sample data for all 4 models:
  - sample_coco_annotations.json (Vision/COCO)
  - sample_ravdess_features.csv (Audio/RAVDESS)
  - sample_task_descriptions.csv (NLP/Synthetic)
  - sample_fused_vectors.csv (Meta-classifier)
- Added datasets README with column descriptions and download instructions
- Documentation covers: models, training, DSP engine, Guardian, extension, deployment"

echo.
echo Pushing...
git push origin main

echo.
echo ================================================
echo  DONE! Documentation pushed to GitHub.
echo ================================================
pause
