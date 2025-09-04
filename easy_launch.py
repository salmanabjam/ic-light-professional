#!/usr/bin/env python3
"""
IC Light Professional - Easy Launch Script (Like Fooocus)
Similar to entry_with_update.py - Simple & Fast Launch

Usage:
    python launch.py --share          # Launch with public share link
    python launch.py --colab          # Launch in Google Colab mode
    python launch.py --share --colab  # Both share and colab mode
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def parse_arguments():
    """Parse command line arguments similar to Fooocus"""
    parser = argparse.ArgumentParser(description='IC Light Professional Launcher')
    
    parser.add_argument('--share', action='store_true', 
                       help='Create public share link (like Fooocus --share)')
    parser.add_argument('--colab', action='store_true',
                       help='Optimize for Google Colab environment')
    parser.add_argument('--always-high-vram', action='store_true',
                       help='Use high VRAM mode (like Fooocus)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to')
    
    return parser.parse_args()


def setup_environment():
    """Setup the environment for IC Light Professional"""
    print("🚀 IC Light Professional - Easy Launch")
    print("="*50)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Set environment variables for optimal performance
    os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
    os.environ['GRADIO_SERVER_PORT'] = '7860'
    
    print(f"📁 Working directory: {current_dir}")
    print("✅ Environment setup complete!")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'diffusers', 'transformers', 'gradio', 
        'PIL', 'cv2', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Please install requirements: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are satisfied!")
    return True


def launch_ic_light(args):
    """Launch IC Light Professional application"""
    try:
        from ic_light.app import ICLightApp
        
        print("\n🌟 Starting IC Light Professional...")
        print("🔗 Interface will be available shortly...")
        
        # Create and configure the app
        app = ICLightApp()
        interface = app.create_interface()
        
        # Configure launch parameters
        launch_kwargs = {
            'server_name': args.host,
            'server_port': args.port,
            'share': args.share,
            'quiet': False,
            'show_error': True,
            'favicon_path': None,
        }
        
        # Special settings for Colab
        if args.colab:
            launch_kwargs['debug'] = True
            launch_kwargs['enable_queue'] = True
            print("🔧 Google Colab mode enabled!")
        
        if args.share:
            print("🌐 Public share link will be generated!")
            print("🔗 You can share this link with anyone!")
        
        print(f"🖥️  Server starting on {args.host}:{args.port}")
        print("="*50)
        
        # Launch the interface
        interface.launch(**launch_kwargs)
        
    except ImportError as e:
        print(f"❌ Error importing IC Light: {e}")
        print("📦 Please ensure the ic_light package is properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)


def main():
    """Main entry point - Similar to Fooocus entry_with_update.py"""
    args = parse_arguments()
    
    # Print startup banner
    print("\n" + "="*60)
    print("🌟 IC Light Professional - Easy Launch")
    print("🎯 Professional AI Image Relighting")
    print("🔗 Just like Fooocus - Simple & Fast!")
    print("="*60)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n📦 Installing missing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Launch the application
    print("\n🚀 Launching IC Light Professional...")
    launch_ic_light(args)


if __name__ == "__main__":
    main()
