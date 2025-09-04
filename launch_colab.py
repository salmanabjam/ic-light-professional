#!/usr/bin/env python3
"""
🚀 IC Light v2 - Ultimate Google Colab Launcher
🎯 One-click solution for Google Colab execution
"""

import subprocess
import sys
import os

def install_requirements():
    """Install essential packages for Google Colab"""
    print("🔧 Installing requirements for Google Colab...")
    
    requirements = [
        "torch==2.0.1",
        "torchvision==0.15.2", 
        "diffusers==0.24.0",
        "transformers==4.35.0",
        "accelerate==0.24.0",
        "gradio==3.50.0",
        "opencv-python-headless==4.8.1.78",
        "Pillow==10.0.1",
        "scipy==1.11.4",
        "safetensors==0.4.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", req], 
                      capture_output=True)
    
    print("✅ Requirements installed!")

def main():
    """Main launcher function"""
    print("🌟 IC Light v2 - Google Colab Launcher")
    print("🚀 Starting installation and setup...")
    
    # Install requirements
    install_requirements()
    
    # Launch the native version
    print("🎯 Launching IC Light v2...")
    try:
        # Import and run the native version
        import ic_light_colab_native
        ic_light_colab_native.main()
    except ImportError:
        print("❌ Could not import native version")
        print("🔄 Trying direct execution...")
        os.system("python ic_light_colab_native.py")

if __name__ == "__main__":
    main()
