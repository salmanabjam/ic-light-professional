# IC Light v2 Professional - Google Colab Integration
# نسخه کامل و حرفه‌ای IC Light v2 برای Google Colab
# Professional Complete IC Light v2 for Google Colab

"""
🌟 IC Light v2 Professional - Google Colab Edition

This is a comprehensive implementation of IC Light v2 with all professional features:
✅ Text-Conditioned Relighting
✅ Background-Conditioned Relighting  
✅ Professional Background Removal (BRIA RMBG 1.4)
✅ Advanced Subject Detection
✅ Preset Prompt System
✅ Multi-Model Support
✅ Professional Quality Controls
✅ Bilingual Interface (Persian/English)

Features:
- Complete IC Light v2 functionality
- Professional background removal and subject detection
- 20+ preset lighting prompts for quick selection
- Advanced quality control parameters
- Multi-sample generation
- Professional schedulers and optimizations
- Comprehensive examples gallery
- Real-time preview and controls

سیستم حرفه‌ای IC Light v2 با تمامی قابلیت‌های پیشرفته
"""

import os
import sys
import subprocess
import importlib.util
import argparse

def install_requirements():
    """Install all required packages for IC Light v2 Professional"""
    print("� Installing IC Light v2 Professional requirements...")
    
    requirements = [
        "diffusers==0.27.2",
        "transformers==4.36.2", 
        "opencv-python",
        "safetensors",
        "pillow==10.2.0",
        "einops",
        "torch",
        "peft", 
        "gradio==3.41.2",
        "protobuf==3.20",
        "huggingface_hub",
        "numpy",
        "accelerate"
    ]
    
    for req in requirements:
        try:
            print(f"📦 Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req, "--quiet"])
            print(f"✅ {req} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {req}: {e}")
            # Try without version constraint
            package_name = req.split("==")[0]
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
                print(f"✅ {package_name} installed successfully (latest version)")
            except:
                print(f"❌ Failed to install {package_name}")

def check_gpu():
    """Check GPU availability and setup"""
    print("� Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Available: {gpu_name}")
            print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️ No GPU available, using CPU (will be slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_models():
    """Download and setup IC Light models"""
    print("📥 Setting up IC Light v2 Professional models...")
    
    # Create models directory
    os.makedirs("./models", exist_ok=True)
    
    models_info = {
        "iclight_sd15_fc.safetensors": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
        "iclight_sd15_fbc.safetensors": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors"
    }
    
    for model_name, url in models_info.items():
        model_path = f"./models/{model_name}"
        if not os.path.exists(model_path):
            print(f"📥 Downloading {model_name}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ {model_name} downloaded successfully")
            except Exception as e:
                print(f"❌ Error downloading {model_name}: {e}")
                try:
                    from torch.hub import download_url_to_file
                    download_url_to_file(url, model_path)
                    print(f"✅ {model_name} downloaded successfully (torch.hub)")
                except Exception as e2:
                    print(f"❌ Failed with torch.hub too: {e2}")
        else:
            print(f"✅ {model_name} already exists")

def create_sample_images():
    """Create sample images directory"""
    print("🖼️ Setting up sample images...")
    
    # Create images directories
    dirs = ["./imgs", "./imgs/bgs", "./imgs/alter"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("✅ Sample directories created")

def launch_interface_choice():
    """Launch interface selection"""
    print("\n🌟 IC Light v2 Professional - Choose Interface:")
    print("1. Text-Conditioned Relighting (Recommended)")
    print("2. Background-Conditioned Relighting") 
    print("3. Both Interfaces (Advanced)")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        return "text"
    elif choice == "2":
        return "background" 
    elif choice == "3":
        return "both"
    else:
        print("Invalid choice, using Text-Conditioned (default)")
        return "text"

def main():
    """Main setup and launch function"""
    parser = argparse.ArgumentParser(description="IC Light v2 Professional - Google Colab")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--interface", choices=["text", "background", "both"], 
                       default="text", help="Interface type to launch")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--skip-setup", action="store_true", help="Skip installation and setup")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌟 IC Light v2 Professional - Google Colab Setup")
    print("سیستم حرفه‌ای IC Light v2 برای Google Colab")
    print("=" * 60)
    
    if not args.skip_setup:
        # Step 1: Install requirements
        install_requirements()
        
        # Step 2: Check GPU
        has_gpu = check_gpu()
        
        # Step 3: Setup models
        setup_models()
        
        # Step 4: Create sample directories
        create_sample_images()
    else:
        print("⏭️ Skipping setup as requested")
    
    # Step 5: Choose interface (if not specified)
    if args.interface == "text" and not args.skip_setup:
        interface_type = launch_interface_choice()
    else:
        interface_type = args.interface
    
    print(f"\n� Launching {interface_type} interface(s)...")
    
    # Import and launch based on choice
    try:
        if interface_type == "text":
            print("📱 Launching Text-Conditioned Interface...")
            from ic_light_v2_professional import create_professional_interface
            interface = create_professional_interface()
            interface.queue(max_size=10).launch(
                server_name='0.0.0.0',
                server_port=args.port,
                share=args.share,
                inbrowser=True,
                show_error=True,
                title="IC Light v2 Professional"
            )
            
        elif interface_type == "background":
            print("📱 Launching Background-Conditioned Interface...")
            from ic_light_bg_professional import create_background_interface
            interface = create_background_interface()
            interface.queue(max_size=10).launch(
                server_name='0.0.0.0',
                server_port=args.port, 
                share=args.share,
                inbrowser=True,
                show_error=True,
                title="IC Light v2 - Background-Conditioned"
            )
            
        elif interface_type == "both":
            print("📱 Launching Both Interfaces...")
            import threading
            import time
            
            def launch_text():
                from ic_light_v2_professional import create_professional_interface
                interface = create_professional_interface()
                interface.queue(max_size=5).launch(
                    server_name='0.0.0.0',
                    server_port=args.port,
                    share=args.share,
                    inbrowser=False,
                    show_error=True,
                    title="IC Light v2 Professional - Text"
                )
            
            def launch_bg():
                time.sleep(5)  # Delay to avoid port conflicts
                from ic_light_bg_professional import create_background_interface
                interface = create_background_interface()
                interface.queue(max_size=5).launch(
                    server_name='0.0.0.0',
                    server_port=args.port + 1, 
                    share=args.share,
                    inbrowser=False,
                    show_error=True,
                    title="IC Light v2 Professional - Background"
                )
            
            # Launch both in separate threads
            text_thread = threading.Thread(target=launch_text)
            bg_thread = threading.Thread(target=launch_bg)
            
            text_thread.start()
            bg_thread.start()
            
            print(f"✅ Both interfaces launched successfully!")
            print(f"🌐 Text-Conditioned: Port {args.port}")
            print(f"🌐 Background-Conditioned: Port {args.port + 1}")
            
            # Keep main thread alive
            text_thread.join()
            bg_thread.join()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all files are present:")
        print("- ic_light_v2_professional.py")
        print("- ic_light_bg_professional.py") 
        print("- briarmbg.py")
        print("- db_examples.py")
    
    except Exception as e:
        print(f"❌ Launch Error: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
