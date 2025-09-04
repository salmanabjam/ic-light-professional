#!/usr/bin/env python3
"""
IC Light Professional - Optimized Setup Script for Google Colab
Addresses all issues mentioned in the technical report

This script provides:
- Intelligent memory management based on GPU type
- Error handling and fallback mechanisms
- Progress monitoring and resource optimization
- English interface messages
- Fooocus-style simple execution
"""

import os
import sys
import subprocess
import torch
import psutil
import gc
from pathlib import Path
import time
import traceback

class ICLightSetup:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_memory = 0
        self.optimal_settings = {}
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment variables and detect optimal settings"""
        print("🚀 IC Light Professional - Intelligent Setup")
        print("=" * 60)
        
        # Set environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
        # Detect GPU capabilities
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name}")
            print(f"💾 GPU Memory: {self.gpu_memory:.1f} GB")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Determine optimal settings based on GPU memory
            if self.gpu_memory >= 15:  # T4 or better
                self.optimal_settings = {
                    "max_resolution": 768,
                    "batch_size": 1,
                    "steps": 25,
                    "use_fp16": True,
                    "enable_attention_slicing": True,
                    "enable_xformers": True
                }
                print("🎯 Detected: High-memory GPU (T4/V100/A100)")
            elif self.gpu_memory >= 8:  # Medium GPU
                self.optimal_settings = {
                    "max_resolution": 512,
                    "batch_size": 1,
                    "steps": 20,
                    "use_fp16": True,
                    "enable_attention_slicing": True,
                    "enable_cpu_offload": True
                }
                print("🎯 Detected: Medium GPU - enabling CPU offload")
            else:  # Low memory GPU
                self.optimal_settings = {
                    "max_resolution": 384,
                    "batch_size": 1,
                    "steps": 15,
                    "use_fp16": True,
                    "enable_attention_slicing": True,
                    "enable_cpu_offload": True,
                    "enable_sequential_offload": True
                }
                print("⚠️ Low GPU memory detected - aggressive optimization enabled")
        else:
            print("⚠️ No GPU detected! Using CPU mode (will be very slow)")
            self.optimal_settings = {
                "max_resolution": 256,
                "batch_size": 1,
                "steps": 10,
                "use_fp16": False
            }
        
        # Display optimal settings
        print(f"\n🎯 Optimal Settings:")
        for key, value in self.optimal_settings.items():
            print(f"   {key}: {value}")
    
    def install_dependencies(self):
        """Install dependencies with error handling"""
        print("\n📦 Installing dependencies with intelligent error handling...")
        
        installation_steps = [
            {
                "name": "PyTorch with CUDA",
                "packages": ["torch>=2.0.0", "torchvision>=0.15.0", "--index-url", "https://download.pytorch.org/whl/cu121"],
                "critical": True
            },
            {
                "name": "Core ML Libraries",
                "packages": ["diffusers==0.27.2", "transformers==4.35.0", "accelerate==0.25.0"],
                "critical": True
            },
            {
                "name": "Safety & Storage",
                "packages": ["safetensors", "huggingface-hub", "requests"],
                "critical": True
            },
            {
                "name": "Memory Optimization",
                "packages": ["xformers==0.0.22.post7", "bitsandbytes"],
                "critical": False  # Optional but recommended
            },
            {
                "name": "User Interface",
                "packages": ["gradio==4.44.0", "spaces"],
                "critical": True
            },
            {
                "name": "Image Processing",
                "packages": ["opencv-python-headless", "pillow", "numpy", "scipy"],
                "critical": True
            },
            {
                "name": "Visualization",
                "packages": ["matplotlib", "plotly==5.17.0"],
                "critical": False
            },
            {
                "name": "Background Removal",
                "packages": ["rembg[new]", "onnxruntime"],
                "critical": False
            }
        ]
        
        for step in installation_steps:
            try:
                print(f"\n🔧 Installing {step['name']}...")
                cmd = ["pip", "install", "-q", "--no-cache-dir"] + step['packages']
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {step['name']} installed successfully")
                else:
                    if step['critical']:
                        print(f"❌ Critical installation failed: {step['name']}")
                        print(f"Error: {result.stderr}")
                        return False
                    else:
                        print(f"⚠️ Optional installation failed: {step['name']} - continuing...")
                        
            except subprocess.TimeoutExpired:
                print(f"⏰ Installation timeout for {step['name']} - trying alternative method...")
                if step['critical']:
                    try:
                        # Fallback: install without version constraints
                        basic_packages = [pkg.split('==')[0] for pkg in step['packages'] if not pkg.startswith('--')]
                        subprocess.check_call(["pip", "install", "-q"] + basic_packages)
                        print(f"✅ {step['name']} installed with fallback method")
                    except:
                        print(f"❌ {step['name']} installation completely failed")
                        return False
            except Exception as e:
                print(f"❌ Unexpected error installing {step['name']}: {e}")
                if step['critical']:
                    return False
        
        # Install IC Light package in editable mode
        print("\n📦 Installing IC Light package...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print("✅ IC Light package installed successfully!")
            else:
                print("⚠️ Editable install failed, adding to Python path...")
                # Add current directory to Python path as fallback
                current_dir = os.getcwd()
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                    print(f"✅ Added {current_dir} to Python path")
        except Exception as e:
            print(f"⚠️ Package installation failed: {e}")
            # Add current directory to Python path as fallback
            current_dir = os.getcwd()
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
                print(f"✅ Added {current_dir} to Python path as fallback")
        
        return True
    
    def verify_installations(self):
        """Verify critical packages are working"""
        print("\n🔍 Verifying installations...")
        
        critical_imports = {
            'torch': 'PyTorch',
            'diffusers': 'Diffusers',
            'transformers': 'Transformers',
            'gradio': 'Gradio',
            'PIL': 'Pillow',
            'cv2': 'OpenCV'
        }
        
        failed_imports = []
        
        for module, name in critical_imports.items():
            try:
                if module == 'PIL':
                    import PIL
                    version = PIL.__version__
                elif module == 'cv2':
                    import cv2
                    version = cv2.__version__
                else:
                    imported_module = __import__(module)
                    version = getattr(imported_module, '__version__', 'unknown')
                
                print(f"✅ {name}: {version}")
                
            except ImportError:
                print(f"❌ {name}: Import failed")
                failed_imports.append(name)
        
        if failed_imports:
            print(f"\n⚠️ Failed imports: {', '.join(failed_imports)}")
            print("🔄 Some features may not work correctly")
            return False
        
        print("\n✅ All critical packages verified!")
        return True
    
    def setup_models_directory(self):
        """Setup models directory and download critical models"""
        print("\n📁 Setting up models directory...")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save optimal settings for later use
        import json
        settings_file = models_dir / "optimal_settings.json"
        with open(settings_file, 'w') as f:
            json.dump(self.optimal_settings, f, indent=2)
        
        print("✅ Models directory setup complete")
    
    def monitor_resources(self):
        """Monitor system resources"""
        try:
            # CPU & RAM
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            print(f"\n📊 System Resources:")
            print(f"   CPU Usage: {cpu_percent:.1f}%")
            print(f"   RAM Usage: {ram_percent:.1f}%")
            
            # GPU if available
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
                
                print(f"   GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB ({gpu_percent:.1f}%)")
                
        except Exception as e:
            print(f"⚠️ Resource monitoring failed: {e}")
    
    def create_launch_script(self):
        """Create optimized launch script"""
        print("\n📝 Creating optimized launch script...")
        
        launch_script = '''#!/usr/bin/env python3
"""
Auto-generated optimized launch script for IC Light Professional
"""
import sys
import os
from pathlib import Path

# Add project to path
project_path = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_path))

# Set optimal environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():
    try:
        from ic_light.app import ICLightApp
        
        print("🚀 Launching IC Light Professional...")
        print("🔗 Creating share link...")
        
        app = ICLightApp()
        interface = app.create_interface()
        
        interface.launch(
            share=True,
            server_name='0.0.0.0',
            server_port=7860,
            show_error=True,
            enable_queue=True
        )
        
    except ImportError:
        print("❌ IC Light app not found - please run setup first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open('optimized_launch.py', 'w') as f:
            f.write(launch_script)
        
        # Make executable
        os.chmod('optimized_launch.py', 0o755)
        print("✅ Launch script created: optimized_launch.py")
    
    def run_setup(self, install_deps=True, verify=True):
        """Run complete setup process"""
        try:
            start_time = time.time()
            
            if install_deps:
                if not self.install_dependencies():
                    print("❌ Setup failed during dependency installation")
                    return False
            
            if verify:
                if not self.verify_installations():
                    print("⚠️ Setup completed with some issues")
            
            self.setup_models_directory()
            self.create_launch_script()
            self.monitor_resources()
            
            setup_time = time.time() - start_time
            
            print(f"\n🎉 Setup completed successfully in {setup_time:.1f} seconds!")
            print("\n📋 Next steps:")
            print("   1. Run: python optimized_launch.py --share")
            print("   2. Or use the Colab notebook interface")
            print("   3. Upload an image and start processing!")
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed with unexpected error: {e}")
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IC Light Professional Setup')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--no-verify', action='store_true', help='Skip installation verification')
    parser.add_argument('--launch-ui', action='store_true', help='Launch UI after setup')
    parser.add_argument('--quick-restore', action='store_true', help='Quick restore after session disconnect')
    
    args = parser.parse_args()
    
    setup = ICLightSetup()
    
    if args.quick_restore:
        # Quick restore mode for Colab session recovery
        print("🔄 Quick restore mode...")
        success = setup.run_setup(install_deps=False, verify=False)
    else:
        # Full setup
        success = setup.run_setup(
            install_deps=not args.skip_deps,
            verify=not args.no_verify
        )
    
    if success and args.launch_ui:
        print("\n🚀 Launching UI...")
        os.system("python optimized_launch.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
