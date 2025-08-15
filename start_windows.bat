@echo off
echo 🎯 SUPER-OMEGA Windows Startup
echo ================================

echo 📦 Installing dependencies...
pip install fastapi uvicorn aiohttp websockets psutil playwright

echo 🌐 Installing Playwright browsers...
python -m playwright install chromium

echo 🚀 Starting SUPER-OMEGA Live Console...
python windows_startup_fix.py

pause