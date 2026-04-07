@echo off
:: start.bat — Launch ANI Flow Optimizer
:: Automatically handles environment setup, starting the backend, and opening the dashboard

echo =====================================================================
echo 🧠 ANI — Creative Flow Optimizer
echo =====================================================================
echo.

:: 1. Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.12+ to run this application.
    pause
    exit /b 1
)

:: 2. Start the backend server
echo [INFO] Starting local WebSockets and HTTP server...
start "ANI Backend Server" cmd /c "python serve.py"

:: Give the server a few seconds to initialize
echo [INFO] Waiting for server to initialize (3 seconds)...
timeout /t 3 /nobreak >nul

:: 3. Open the dashboard in default browser
echo [INFO] Opening ANI Dashboard in your web browser...
start http://localhost:8080/frontend/

echo.
echo =====================================================================
echo ✅ System is running!
echo.
echo 🔧 NEXT STEPS:
echo 1. Open Chrome and go to chrome://extensions/
echo 2. Enable "Developer mode" in the top right
echo 3. Click "Load unpacked" and select the 'chrome_extension' folder
echo 4. Grant microphone and webcam permissions when prompted
echo.
echo Close this window to keep the server running in the background.
echo =====================================================================
pause
