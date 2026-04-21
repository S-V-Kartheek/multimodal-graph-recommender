@echo off
REM ============================================================
REM  MGRS-HFA Full Setup + Run Script
REM  Creates venv, installs deps, and runs training
REM ============================================================

echo ============================================================
echo  MGRS-HFA Setup and Training
echo ============================================================
echo.

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo         Make sure Python 3.9+ is installed and in PATH.
        pause
        exit /b 1
    )
    echo [SETUP] Virtual environment created.
)

REM Activate venv
call venv\Scripts\activate.bat
echo [SETUP] Virtual environment activated.

REM Install PyTorch (Will use GPU if available, else CPU)
echo [SETUP] Installing PyTorch...
pip install torch --quiet

REM Install PyTorch Geometric and extensions
echo [SETUP] Installing PyTorch Geometric...
pip install torch-geometric torch-scatter torch-sparse --quiet

REM Install remaining dependencies
echo [SETUP] Installing other dependencies...
pip install pandas numpy scikit-learn matplotlib tqdm requests --quiet

echo.
echo [SETUP] All dependencies installed.
echo.

REM Run training
echo ============================================================
echo  Starting MGRS-HFA Training (100 epochs)
echo  NOTE: This will use GPU if torch detects CUDA, else CPU.
echo ============================================================
echo.

python main.py --epochs 100 --k 10 --include_cold_start

echo.
echo ============================================================
echo  Done! Check the results\ folder for outputs:
echo    - training_loss_ml1m.png
echo    - metrics_ml1m.png
echo    - metrics_over_epochs_ml1m.png
echo    - loss_components_ml1m.png
echo ============================================================
pause
