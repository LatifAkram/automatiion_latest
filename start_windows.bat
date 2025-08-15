@echo off
echo 🎯 SUPER-OMEGA Windows Startup
echo ================================

echo ℹ️ Choose startup method:
echo 1. Minimal (Web server + Playwright only - SAFEST for Windows)
echo 2. Simple (Core automation - RECOMMENDED for Windows)
echo 3. Full (With AI features - may have compatibility issues)
echo.

set /p choice="Enter choice (1, 2, or 3): "

if "%choice%"=="1" (
    echo 🚀 Starting Minimal SUPER-OMEGA...
    python start_minimal_windows.py
) else if "%choice%"=="2" (
    echo 🚀 Starting Simple SUPER-OMEGA...
    python start_simple_windows.py
) else if "%choice%"=="3" (
    echo 🚀 Starting Full SUPER-OMEGA...
    python windows_startup_fix.py
) else (
    echo ❌ Invalid choice, starting Minimal version...
    python start_minimal_windows.py
)

pause