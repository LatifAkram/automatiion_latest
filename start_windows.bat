@echo off
echo 🎯 SUPER-OMEGA Windows Startup
echo ================================

echo ℹ️ Choose startup method:
echo 1. Simple (Core automation only - RECOMMENDED for Windows)
echo 2. Full (With AI features - may have compatibility issues)
echo.

set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo 🚀 Starting Simple SUPER-OMEGA...
    python start_simple_windows.py
) else if "%choice%"=="2" (
    echo 🚀 Starting Full SUPER-OMEGA...
    python windows_startup_fix.py
) else (
    echo ❌ Invalid choice, starting Simple version...
    python start_simple_windows.py
)

pause