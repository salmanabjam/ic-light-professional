#!/usr/bin/env python3
"""
IC Light Professional Demo Script
Quick test to verify installation and functionality
"""

import sys
import time
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def print_header():
    """Print demo header"""
    print("=" * 60)
    print("🌟 IC Light Professional Demo")
    print("=" * 60)
    print()

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"), 
        ("transformers", "Transformers"),
        ("gradio", "Gradio"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly")
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available\n")
    return True

def check_gpu():
    """Check GPU availability"""
    print("🔍 Checking GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name}")
            print(f"📊 VRAM: {gpu_memory:.1f}GB")
            
            if gpu_memory >= 8:
                print("🎯 Status: Ready for high-quality processing")
            elif gpu_memory >= 4:
                print("🎯 Status: Ready for standard processing")
            else:
                print("⚠️ Status: Limited VRAM, may need CPU offload")
                
            return True
        else:
            print("⚠️ No GPU available - CPU mode")
            return False
            
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def test_ic_light_import():
    """Test IC Light package import"""
    print("\n🔍 Testing IC Light import...")
    
    try:
        # Test package structure
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Test main package
        import ic_light
        print(f"✅ ic_light package - v{getattr(ic_light, '__version__', 'unknown')}")
        
        # Test components
        from ic_light.models.ic_light_model import ICLightModel
        print("✅ ICLightModel")
        
        from ic_light.utils.image_processor import ImageProcessor  
        print("✅ ImageProcessor")
        
        from ic_light.app import ICLightApp
        print("✅ ICLightApp")
        
        from ic_light.ui.components import ComponentFactory
        print("✅ UI Components")
        
        print("✅ All components imported successfully\n")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        print("📝 Traceback:")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation (without loading weights)"""
    print("🔍 Testing model creation...")
    
    try:
        from ic_light.models.ic_light_model import ICLightModel
        
        # Create model instance (without loading weights)
        model = ICLightModel.__new__(ICLightModel)
        model.model_type = "fc"
        model.device = "cpu"  # Use CPU for demo
        
        print("✅ Model structure created")
        print(f"   - Type: {model.model_type}")
        print(f"   - Device: {model.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_image_processor():
    """Test image processor functionality"""
    print("\n🔍 Testing image processor...")
    
    try:
        from ic_light.utils.image_processor import ImageProcessor
        from PIL import Image
        import numpy as np
        
        # Create processor
        processor = ImageProcessor(
            device="cpu",
            enable_background_removal=False  # Skip BG removal for demo
        )
        
        print("✅ ImageProcessor created")
        
        # Test with dummy image
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        # Test resize
        resized = processor.resize_with_aspect(dummy_image, (512, 512))
        print(f"✅ Image resize: {dummy_image.size} -> {resized.size}")
        
        # Test enhancement
        enhanced = processor.enhance_image(
            dummy_image, 
            brightness=1.1, 
            contrast=1.1
        )
        print("✅ Image enhancement")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processor test failed: {e}")
        return False

def test_ui_components():
    """Test UI components"""
    print("\n🔍 Testing UI components...")
    
    try:
        from ic_light.ui.components import (
            ModernTheme, 
            ComponentFactory, 
            ExamplePrompts,
            PresetManager
        )
        
        # Test theme
        css = ModernTheme.get_custom_css()
        print(f"✅ Theme CSS ({len(css)} chars)")
        
        # Test component factory
        header = ComponentFactory.create_header("Demo", "Test")
        print("✅ Component factory")
        
        # Test prompts
        prompt = ExamplePrompts.get_random_prompt("portrait")
        print(f"✅ Example prompts: '{prompt[:50]}...'")
        
        # Test presets
        preset_names = PresetManager.get_preset_names()
        print(f"✅ Presets: {len(preset_names)} available")
        
        return True
        
    except Exception as e:
        print(f"❌ UI components test failed: {e}")
        return False

def test_app_creation():
    """Test application creation (without launch)"""
    print("\n🔍 Testing app creation...")
    
    try:
        from ic_light import create_app
        
        # Test create_app function
        print("✅ create_app function available")
        
        # Note: We don't actually create the app to avoid loading models
        print("✅ App creation interface ready")
        
        return True
        
    except Exception as e:
        print(f"❌ App creation test failed: {e}")
        return False

def run_quick_demo():
    """Run a quick demonstration"""
    print("🚀 Running quick demonstration...")
    
    try:
        from ic_light.ui.components import ExamplePrompts, PresetManager
        
        # Show example prompts
        print("\n📝 Example Prompts:")
        for category in ["portrait", "creative", "artistic"]:
            prompt = ExamplePrompts.get_random_prompt(category)
            print(f"   {category.title()}: {prompt}")
        
        # Show presets
        print("\n🎛️ Available Presets:")
        for preset_name in PresetManager.get_preset_names()[:3]:
            preset = PresetManager.get_preset(preset_name)
            print(f"   {preset_name}: {preset.get('prompt', 'N/A')[:60]}...")
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("📋 Usage Instructions")
    print("=" * 60)
    print()
    print("🚀 Launch Options:")
    print()
    print("1. Professional Interface:")
    print("   python launch.py --share")
    print()
    print("2. Google Colab:")
    print("   Open IC_Light_Professional_Colab.ipynb")
    print() 
    print("3. Python API:")
    print("   from ic_light import create_app")
    print("   app = create_app()")
    print("   app.launch()")
    print()
    print("🎯 Next Steps:")
    print("   - Upload an image")
    print("   - Write a lighting prompt")
    print("   - Choose lighting direction") 
    print("   - Click 'Process Image'")
    print()

def main():
    """Main demo function"""
    print_header()
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("Dependencies", check_dependencies),
        ("GPU Status", check_gpu),
        ("IC Light Import", test_ic_light_import),
        ("Model Creation", test_model_creation),
        ("Image Processor", test_image_processor),
        ("UI Components", test_ui_components),
        ("App Creation", test_app_creation),
        ("Quick Demo", run_quick_demo)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"🧪 {test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
        print()
    
    # Results
    total_time = time.time() - start_time
    
    print("=" * 60)
    print("📊 Demo Results")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print()
    
    if failed == 0:
        print("🎉 All tests passed! IC Light Professional is ready to use.")
        print_usage_instructions()
    else:
        print("⚠️ Some tests failed. Please check the requirements and setup.")
        print("\n🔧 Troubleshooting:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Install the package: pip install -e .")
        print("   3. Check GPU drivers if using CUDA")
        print("   4. Restart Python environment")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        traceback.print_exc()
