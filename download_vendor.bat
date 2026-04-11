@echo off
echo Downloading Live2D vendor libraries...
mkdir "frontend\vendor" 2>nul

echo Downloading pixi.min.js...
curl -L -o "frontend\vendor\pixi.min.js" "https://cdnjs.cloudflare.com/ajax/libs/pixi.js/7.4.0/pixi.min.js"

echo Downloading live2dcubismcore.min.js...
curl -L -o "frontend\vendor\live2dcubismcore.min.js" "https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js"

echo Downloading cubism4.min.js...
curl -L -o "frontend\vendor\cubism4.min.js" "https://cdn.jsdelivr.net/npm/pixi-live2d-display/dist/cubism4.min.js"

echo.
echo Done! Check frontend\vendor\ for the files.
dir "frontend\vendor\"
pause
