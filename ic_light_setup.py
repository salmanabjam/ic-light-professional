#!/usr/bin/env python3
"""
IC Light Setup Script for Google Colab
Quick setup and launch script for IC Light relighting application
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🎨 IC Light - Image Relighting Setup")
    print("=" * 60)
    print()

def check_environment():
    """Check system requirements"""
    print("🔍 Checking system environment...")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"   Python: {python_version}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA: {'✅ Available' if cuda_available else '❌ Not available'}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   GPU: {gpu_name}")
        
        # Check VRAM
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   VRAM: {gpu_memory:.1f} GB")
        
        if gpu_memory < 15:
            print("   ⚠️  Warning: Less than 16GB VRAM may cause memory issues")
    else:
        print("   ❌ GPU required for optimal performance")
    
    print()

def setup_repository():
    """Setup IC Light repository"""
    print("📦 Setting up IC Light repository...")
    
    # Change to content directory (Colab standard)
    os.chdir('/content')
    
    # Remove existing directory
    if os.path.exists('IC-Light'):
        print("   Removing existing IC-Light directory...")
        subprocess.run(['rm', '-rf', 'IC-Light'], check=True)
    
    # Clone repository
    print("   Cloning IC Light repository...")
    subprocess.run([
        'git', 'clone', 
        'https://github.com/lllyasviel/IC-Light.git'
    ], check=True)
    
    # Change to IC-Light directory
    os.chdir('/content/IC-Light')
    print("   ✅ Repository setup complete")
    print()

def install_dependencies():
    """Install required dependencies"""
    print("📚 Installing dependencies...")
    
    # Core PyTorch with CUDA
    print("   Installing PyTorch with CUDA...")
    subprocess.run([
        'pip', 'install', 'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu121'
    ], check=True)
    
    # Required packages
    packages = [
        'diffusers==0.27.2',
        'transformers==4.36.2', 
        'opencv-python',
        'safetensors',
        'pillow==10.2.0',
        'einops',
        'peft',
        'gradio==3.41.2',
        'protobuf==3.20',
        'accelerate',
        'xformers'
    ]
    
    print("   Installing IC Light dependencies...")
    for package in packages:
        print(f"     Installing {package}...")
        subprocess.run(['pip', 'install', package], check=True)
    
    print("   ✅ Dependencies installed")
    print()

def optimize_environment():
    """Set environment optimizations"""
    print("⚡ Optimizing environment...")
    
    # Set environment variables
    optimizations = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_LAUNCH_BLOCKING': '0',
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"   Set {key}={value}")
    
    print("   ✅ Environment optimized")
    print()

def verify_installation():
    """Verify installation is working"""
    print("✅ Verifying installation...")
    
    try:
        # Test imports
        import torch
        import diffusers
        import transformers
        import gradio
        
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Diffusers: {diffusers.__version__}")
        print(f"   Transformers: {transformers.__version__}")
        print(f"   Gradio: {gradio.__version__}")
        
        # Test GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            test_tensor = torch.randn(1, 3, 256, 256).to(device)
            print(f"   GPU Test: ✅ Tensor created on {device}")
        
        print("   ✅ Installation verified")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    except Exception as e:
        print(f"   ❌ Verification error: {e}")
        return False
    
    print()
    return True

def launch_demo(demo_type='text'):
    """Launch IC Light demo"""
    print(f"🚀 Launching IC Light {demo_type} demo...")
    
    if demo_type == 'text':
        script_name = 'gradio_demo.py'
        print("   Features: Text-conditioned relighting")
    elif demo_type == 'background':
        script_name = 'gradio_demo_bg.py'
        print("   Features: Background-conditioned relighting")
    else:
        print("   ❌ Unknown demo type")
        return
    
    print(f"   Script: {script_name}")
    print("   🔗 Gradio interface will open shortly...")
    print("   ⏳ Model download may take a few minutes on first run")
    print()
    
    # Launch the demo
    try:
        subprocess.run(['python', script_name], check=True)
    except KeyboardInterrupt:
        print("\n   ⏹️  Demo stopped by user")
    except Exception as e:
        print(f"   ❌ Error launching demo: {e}")

def main():
    """Main setup function"""
    print_header()
    
    try:
        # Setup steps
        check_environment()
        setup_repository()
        install_dependencies()
        optimize_environment()
        
        if not verify_installation():
            print("❌ Installation verification failed!")
            return
        
        print("🎉 Setup complete!")
        print()
        
        # Ask user which demo to launch
        print("Choose demo to launch:")
        print("1. Text-conditioned relighting (recommended)")
        print("2. Background-conditioned relighting")
        print("3. Skip demo launch")
        
        while True:
            try:
                choice = input("\nEnter choice (1-3): ").strip()
                
                if choice == '1':
                    launch_demo('text')
                    break
                elif choice == '2':
                    launch_demo('background')
                    break
                elif choice == '3':
                    print("\n💡 To launch later, run:")
                    print("   python gradio_demo.py (text-conditioned)")
                    print("   python gradio_demo_bg.py (background-conditioned)")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
                    
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Setup completed without launching demo")
                break
                
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Please check the error message and try again")

if __name__ == "__main__":
    main()
