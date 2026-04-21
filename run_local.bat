@echo off
REM ============================================================
REM  MGRS-HFA Local Training Script
REM  Runs the full pipeline. Will use GPU if available.
REM ============================================================

echo ============================================================
echo  MGRS-HFA: Multimodal Graph-based Recommendation System
echo  Running on MovieLens 1M
echo ============================================================
echo.

REM Activate virtual environment
call "%~dp0venv\Scripts\activate.bat"

REM Install/verify dependencies
echo [SETUP] Installing dependencies...
pip install -r "%~dp0requirements.txt" --quiet

REM Run the training
echo.
echo [RUN] Starting training...
echo.
python "%~dp0main.py" --epochs 100 --k 10 --include_cold_start

echo.
echo ============================================================
echo  Training complete! Results saved to: results\
echo ============================================================
pause
