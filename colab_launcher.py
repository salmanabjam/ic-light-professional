#!/usr/bin/env python3
"""
IC Light Professional - Google Colab Launcher
Simple launcher script that handles all setup and launch automatically

Usage in Google Colab:
    !python colab_launcher.py
    !python colab_launcher.py --share
    !python colab_launcher.py --share --debug
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Setup the environment and fix common issues"""
    print("🚀 IC Light Professional - Colab Launcher")
    print("=" * 50)
    
    # Fix directory structure if needed
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if we're in a nested directory and fix it
    if "ic-light-professional" in current_dir:
        # Find the root ic-light-professional directory
        parts = current_dir.split(os.sep)
        try:
            # Find the first occurrence of ic-light-professional
            idx = parts.index("ic-light-professional")
            # Construct path up to the first ic-light-professional
            root_path = os.sep.join(parts[:idx+1])
            
            if os.path.exists(os.path.join(root_path, "optimized_setup.py")):
                os.chdir(root_path)
                print(f"🔧 Fixed nested directories, now in: {os.getcwd()}")
            else:
                print("⚠️ Could not find project files in expected location")
        except ValueError:
            print("⚠️ Could not parse directory structure")
    
    # Ensure Python path includes current directory
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
        print("✅ Added current directory to Python path")

def install_package():
    """Install the IC Light package"""
    print("\n📦 Installing IC Light package...")
    
    try:
        # Try editable install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode == 0:
            print("✅ Package installed successfully!")
            return True
        else:
            print(f"⚠️ Editable install failed: {result.stderr}")
            # Try alternative method
            print("🔄 Trying alternative installation...")
            
            # Create __init__.py files if missing
            init_files = [
                "ic_light/__init__.py",
                "ic_light/models/__init__.py",
                "ic_light/utils/__init__.py",
                "ic_light/ui/__init__.py"
            ]
            
            for init_file in init_files:
                if os.path.exists(init_file):
                    continue
                    
                os.makedirs(os.path.dirname(init_file), exist_ok=True)
                with open(init_file, 'w') as f:
                    if "models" in init_file:
                        f.write('from .ic_light_model import ICLightModel\n__all__ = ["ICLightModel"]')
                    elif "utils" in init_file:
                        f.write('from .image_processor import ImageProcessor\n__all__ = ["ImageProcessor"]')
                    elif "ui" in init_file:
                        f.write('from .components import *')
                    else:
                        f.write('"""IC Light Package"""')
                print(f"✅ Created/updated {init_file}")
            
            return True
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def run_setup():
    """Run the optimized setup script"""
    print("\n🔧 Running optimized setup...")
    
    try:
        result = subprocess.run(
            [sys.executable, "optimized_setup.py", "--quick-restore"],
            capture_output=False,  # Show output in real time
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Setup completed successfully!")
            return True
        else:
            print("⚠️ Setup had some issues but continuing...")
            return True  # Continue anyway
            
    except subprocess.TimeoutExpired:
        print("⏰ Setup timed out, but continuing...")
        return True
    except Exception as e:
        print(f"⚠️ Setup error: {e}, but continuing...")
        return True

def launch_app(share=False, debug=False, port=7860):
    """Launch the IC Light application"""
    print("\n🚀 Launching IC Light Professional...")
    
    # Prepare launch command
    cmd = [sys.executable, "easy_launch.py"]
    
    if share:
        cmd.append("--share")
        print("🌐 Share link will be generated!")
        
    if debug:
        cmd.append("--debug")
        print("🐛 Debug mode enabled!")
        
    cmd.extend(["--port", str(port)])
    
    try:
        # Launch the application
        subprocess.run(cmd, check=False)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Launch error: {e}")
        
        # Try fallback launch
        print("\n🔄 Attempting fallback launch...")
        try:
            # Direct Python execution
            fallback_code = """
import sys
import os
sys.path.insert(0, os.getcwd())

from ic_light.app import ICLightApp

app = ICLightApp()
interface = app.create_interface()

launch_kwargs = {
    'server_name': '0.0.0.0',
    'server_port': %d,
    'share': %s,
    'show_error': True
}

interface.launch(**launch_kwargs)
""" % (port, str(share).lower())

            with open("_temp_launch.py", "w") as f:
                f.write(fallback_code)
            
            subprocess.run([sys.executable, "_temp_launch.py"], check=False)
            
        except Exception as e2:
            print(f"❌ Fallback launch also failed: {e2}")
            print("\n📖 Manual steps:")
            print("1. Check if all files are present")
            print("2. Try: python optimized_setup.py")
            print("3. Try: python easy_launch.py --share")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="IC Light Professional - Colab Launcher")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup and go directly to launch")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    if not args.skip_setup:
        # Install package
        if not install_package():
            print("❌ Package installation failed, but continuing...")
        
        # Run setup
        if not run_setup():
            print("❌ Setup failed, but attempting to launch anyway...")
    
    # Launch application
    launch_app(
        share=args.share, 
        debug=args.debug, 
        port=args.port
    )

if __name__ == "__main__":
    main()
