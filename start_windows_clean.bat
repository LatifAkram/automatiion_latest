@echo off
echo SUPER-OMEGA Windows Startup (Clean Version)
echo ============================================

echo Choose startup method:
echo 1. Clean startup (recommended)
echo 2. Direct server only
echo 3. Fixed startup script
echo.

set /p choice="Enter choice (1, 2, or 3): "

if "%choice%"=="1" (
    echo Starting SUPER-OMEGA with clean script...
    python start_simple_windows_clean.py
) else if "%choice%"=="2" (
    echo Starting direct server...
    python start_server_direct.py
) else if "%choice%"=="3" (
    echo Starting with fixed script...
    python start_windows_fixed.py
) else (
    echo Invalid choice, starting with clean script...
    python start_simple_windows_clean.py
)

pause