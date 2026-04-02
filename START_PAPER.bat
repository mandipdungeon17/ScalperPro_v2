@echo off
title ScalperPro v2 - Paper Trading
color 0A
echo.
echo  =====================================================
echo   SCALPER PRO v2 - Paper Trading Mode
echo   Alerts: @StockNiftyAlertBot on Telegram
echo  =====================================================
echo.

cd /d D:\Trading\ScalperPro_v2

:: ── Step 1: Check Python ──────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

:: ── Step 2: Install / verify dependencies ─────────────────────────
echo [1/3] Checking dependencies...
pip install -q -r scalper\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo       Done.

:: ── Step 3: Verify credentials loaded from .env ───────────────────
echo [2/3] Verifying credentials...
python -m scalper.verify_env
if errorlevel 1 (
    echo.
    echo Open scalper\.env and fill in missing credentials.
    pause
    exit /b 1
)

:: ── Step 4: Start the bot ─────────────────────────────────────────
echo [3/3] Starting ScalperPro v2...
echo.
python -m scalper.main_v2 --mode paper --scan 30

echo.
echo Session ended.
pause
