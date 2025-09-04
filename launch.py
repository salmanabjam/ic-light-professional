#!/usr/bin/env python3
import subprocess
import sys

print("🚀 Launching IC Light v2 Complete System...")
print("🌟 راه‌اندازی سیستم کامل IC Light v2...")

try:
    # Try Colab compatible version first
    subprocess.run([sys.executable, "ic_light_colab_compatible.py"], check=True)
except Exception as e:
    print(f"Colab version failed: {e}")
    try:
        # Fallback to complete version
        subprocess.run([sys.executable, "ic_light_complete_fixed.py"], check=True)
    except Exception as e2:
        print(f"Error: {e2}")
        print("Please run manually:")
        print("python ic_light_colab_compatible.py")
        print("or")
        print("python ic_light_complete_fixed.py")
