#!/usr/bin/env python3
import subprocess
import sys

print("🚀 Launching IC Light v2 Complete System...")
print("🌟 راه‌اندازی سیستم کامل IC Light v2...")

try:
    subprocess.run([sys.executable, "ic_light_fixed.py"], check=True)
except Exception as e:
    print(f"Error: {e}")
    print("Please run: python ic_light_fixed.py")
