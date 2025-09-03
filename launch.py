#!/usr/bin/env python3
"""
IC Light Launch Script
Easy deployment for Google Colab and local environments
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core ML dependencies
    packages = [
        "torch>=2.1.0",
        "torchvision",
        "torchaudio",
        "diffusers>=0.27.2",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "gradio>=4.0.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "opencv-python>=4.8.0",
        "scipy>=1.11.0"
    ]
    
    # Install packages
    for package in packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            
    # Try to install XFormers (optional)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "xformers>=0.0.22", "--quiet"
        ])
        print("✅ xformers (GPU acceleration)")
    except subprocess.CalledProcessError:
        print("⚠️ XFormers not installed (GPU acceleration disabled)")


def setup_environment():
    """Setup environment variables and directories"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        "models",
        "outputs", 
        "cache",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created {directory}/")
    
    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TRANSFORMERS_CACHE"] = "./cache"
    os.environ["HF_HOME"] = "./cache"
    
    print("✅ Environment configured")


def check_gpu():
    """Check GPU availability and setup"""
    print("🔍 Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Check for compatible GPU
            if gpu_memory < 4:
                print("⚠️ Warning: GPU memory < 4GB, may cause issues")
            
            return True
        else:
            print("⚠️ No GPU available, using CPU (slower)")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def launch_colab():
    """Launch in Google Colab environment"""
    print("🚀 Launching IC Light in Google Colab...")
    
    # Google Colab specific setup
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except ImportError:
        print("⚠️ Not in Google Colab environment")
    
    # Import and launch
    try:
        from ic_light import create_app
        
        app = create_app(
            model_type="fc",
            device="auto",
            enable_analytics=True,
            theme="default"
        )
        
        # Launch with public URL in Colab
        app.launch(
            share=True,
            server_name="0.0.0.0", 
            server_port=7860,
            debug=False
        )
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        print("Try installing the package with: pip install -e .")


def launch_local(args):
    """Launch in local environment"""
    print("🚀 Launching IC Light locally...")
    
    try:
        from ic_light import create_app
        
        app = create_app(
            model_type=args.model_type,
            device=args.device,
            enable_analytics=args.analytics,
            theme=args.theme
        )
        
        app.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            debug=args.debug
        )
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        print("Make sure all dependencies are installed")


def main():
    """Main launch function"""
    parser = argparse.ArgumentParser(description="IC Light Launcher")
    
    parser.add_argument(
        "--model-type",
        choices=["fc", "fbc", "fcon"],
        default="fc",
        help="IC Light model variant"
    )
    
    parser.add_argument(
        "--device", 
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    
    parser.add_argument(
        "--theme",
        choices=["default", "soft", "monochrome"],
        default="default", 
        help="UI theme"
    )
    
    parser.add_argument(
        "--no-analytics",
        dest="analytics",
        action="store_false",
        help="Disable analytics"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install dependencies only"
    )
    
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Google Colab mode"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install:
        install_dependencies()
        return
    
    # Setup environment
    setup_environment()
    
    # Check GPU
    check_gpu()
    
    # Launch application
    if args.colab or "COLAB_GPU" in os.environ:
        launch_colab()
    else:
        launch_local(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 IC Light launcher stopped")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)
