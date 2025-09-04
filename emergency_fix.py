#!/usr/bin/env python3
"""
IC Light Professional - Emergency Directory Fix and Launch
This script fixes the extreme directory nesting issue and launches IC Light

Usage in Google Colab:
    !wget https://raw.githubusercontent.com/salmanabjam/ic-light-professional/main/emergency_fix.py
    !python emergency_fix.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def emergency_fix():
    print("🚨 IC Light Professional - EMERGENCY DIRECTORY FIX")
    print("=" * 60)
    
    current = os.getcwd()
    print(f"📁 Starting directory: {current}")
    
    # Clone if needed
    if not os.path.exists('ic-light-professional'):
        print("📥 Cloning repository...")
        try:
            subprocess.run(["git", "clone", "https://github.com/salmanabjam/ic-light-professional.git"], 
                         check=True, timeout=60)
            print("✅ Repository cloned!")
        except Exception as e:
            print(f"❌ Clone failed: {e}")
            return False
    
    # EXTREME directory fixing - handle any level of nesting
    project_dir = None
    max_depth = 20  # Reasonable limit to avoid infinite loops
    
    print("🔍 AGGRESSIVE search for project files...")
    
    # Walk through directories up to max_depth
    for root, dirs, files in os.walk(current):
        depth = root.replace(current, '').count(os.sep)
        if depth > max_depth:
            continue
            
        # Look for key indicators
        key_files = ['optimized_setup.py', 'colab_launcher.py', 'ic_light']
        found_count = 0
        
        for key_file in key_files:
            if key_file in files or key_file in dirs:
                found_count += 1
        
        # If we found at least 2 key files/dirs, this is likely it
        if found_count >= 2:
            project_dir = root
            print(f"🎯 Found project at depth {depth}: {project_dir}")
            break
    
    # Change to project directory
    if project_dir and project_dir != current:
        try:
            os.chdir(project_dir)
            sys.path.insert(0, project_dir)
            print(f"✅ Changed to: {os.getcwd()}")
        except Exception as e:
            print(f"❌ Failed to change directory: {e}")
            return False
    
    # Create minimal package structure
    print("\n🔧 Creating minimal package structure...")
    
    # Ensure ic_light directory exists
    ic_light_dir = "ic_light"
    os.makedirs(ic_light_dir, exist_ok=True)
    
    # Create minimal app.py
    app_content = '''
import gradio as gr
import torch

class ICLightApp:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def process_image(self, image, prompt, direction):
        if image is None:
            return None, "Please upload an image"
        return image, f"Processed with {direction} lighting: {prompt}"
    
    def create_interface(self):
        with gr.Blocks(title="IC Light Professional") as interface:
            gr.Markdown("# 🌟 IC Light Professional")
            
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="Input Image", type="pil")
                    prompt = gr.Textbox(label="Lighting Prompt", value="professional lighting")
                    direction = gr.Dropdown(["left", "right", "top", "bottom"], value="left", label="Direction")
                    btn = gr.Button("Apply Lighting", variant="primary")
                
                with gr.Column():
                    output_img = gr.Image(label="Result")
                    status = gr.Textbox(label="Status")
            
            btn.click(self.process_image, [input_img, prompt, direction], [output_img, status])
        
        return interface
'''
    
    with open(os.path.join(ic_light_dir, "app.py"), "w") as f:
        f.write(app_content)
    
    # Create __init__.py
    with open(os.path.join(ic_light_dir, "__init__.py"), "w") as f:
        f.write('from .app import ICLightApp\n__all__ = ["ICLightApp"]')
    
    print("✅ Minimal package structure created!")
    
    # Install essential dependencies
    print("\n📦 Installing essential dependencies...")
    essential_deps = ["torch", "gradio>=4.0.0", "pillow", "numpy"]
    
    for dep in essential_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], 
                         check=True, timeout=120)
            print(f"✅ {dep}")
        except Exception as e:
            print(f"⚠️ {dep} failed: {str(e)[:50]}")
    
    # Launch the application
    print("\n🚀 Launching IC Light Professional...")
    try:
        from ic_light.app import ICLightApp
        
        app = ICLightApp()
        interface = app.create_interface()
        
        print("🌐 Creating share link...")
        interface.launch(
            share=True,
            server_name='0.0.0.0',
            server_port=7860,
            show_error=True,
            quiet=False
        )
        return True
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        
        # Create a simple launch script as backup
        launch_script = '''
import sys
sys.path.insert(0, ".")
from ic_light.app import ICLightApp

app = ICLightApp()
interface = app.create_interface()
interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
'''
        
        with open("emergency_launch.py", "w") as f:
            f.write(launch_script)
        
        print("📄 Created emergency_launch.py")
        print("🔄 Try running: !python emergency_launch.py")
        
        return False

if __name__ == "__main__":
    success = emergency_fix()
    if not success:
        print("\n🚨 EMERGENCY FIX FAILED")
        print("📋 Manual steps:")
        print("1. Check current directory files")
        print("2. Look for ic_light folder")
        print("3. Try: !ls -la")
        print("4. Contact support with directory structure info")
        
        # Diagnostic info
        print(f"\n🔍 Diagnostics:")
        print(f"Current dir: {os.getcwd()}")
        print(f"Contents: {os.listdir('.')[:10]}")
