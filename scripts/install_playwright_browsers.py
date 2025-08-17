#!/usr/bin/env python3
"""
Install Playwright Browsers
- Installs Playwright Python package browsers and OS deps via CLI
- Safe to run multiple times
"""

import shutil
import subprocess
import sys


def main() -> int:
    # Prefer python -m playwright install --with-deps when available
    python_exec = sys.executable or 'python3'
    cmds = [
        [python_exec, '-m', 'playwright', 'install', '--with-deps'],
        [python_exec, '-m', 'playwright', 'install']  # fallback without --with-deps
    ]
    for cmd in cmds:
        try:
            print(f"Running: {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=True)
            if proc.returncode == 0:
                print("✅ Playwright browsers installed")
                return 0
        except Exception as e:
            print(f"⚠️ Install attempt failed: {e}")
            continue
    print("❌ Could not install Playwright browsers. Ensure 'playwright' is installed and rerun.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())