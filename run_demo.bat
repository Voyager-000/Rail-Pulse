@echo off
echo ===================================================
echo 🚂 RailPulse - AI Railway Risk Intelligence Platform
echo ===================================================
echo.
echo Starting FastAPI Backend Server...
start "RailPulse API" cmd /c "uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo Waiting for API to start...
timeout /t 3 /nobreak >nul

echo Opening Dashboard...
start "" "dashboard\index.html"

echo.
echo ✅ RailPulse is now running!
echo You can view the live dashboard in your browser.
echo Close this window to stop the API server.
pause
