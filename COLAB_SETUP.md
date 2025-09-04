# 🚀 IC Light Professional - Super Easy Setup

**Simple commands like Fooocus - Just copy & paste!**

## Method 1: Super Easy Setup (Recommended)

```bash
# 🎯 One-Click Setup (Just like Fooocus!)
%cd /content
!git clone https://github.com/salmanabjam/ic-light-professional.git
%cd /content/ic-light-professional
!python easy_launch.py --share --colab
```

## Method 2: Step-by-step (Like your Fooocus example)

```bash
# Install Git support (like your pygit2 command)
!pip install pygit2==1.15.1

# Navigate to content directory
%cd /content

# Clone the repository (like Fooocus clone)
!git clone https://github.com/salmanabjam/ic-light-professional.git

# Navigate to project directory
%cd /content/ic-light-professional

# Launch with share link (like Fooocus --share)
!python easy_launch.py --share --always-high-vram
```

## Method 3: Full Control Setup

```bash
# Install core dependencies first
!pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
!pip install diffusers==0.27.2 transformers==4.35.0 accelerate==0.25.0
!pip install gradio==4.44.0 opencv-python-headless pillow requests

# Navigate and clone
%cd /content
!git clone https://github.com/salmanabjam/ic-light-professional.git
%cd /content/ic-light-professional

# Install remaining requirements
!pip install -r requirements.txt

# Launch the application
!python easy_launch.py --share --colab --always-high-vram
```

## 🎯 Available Launch Options

- `--share` : Creates public link (like Fooocus --share)
- `--colab` : Optimizes for Google Colab
- `--always-high-vram` : High VRAM mode (like Fooocus)
- `--port 7860` : Custom port (default 7860)

## 🌟 After Launch

You'll get:
- ✅ Professional Gradio interface
- ✅ Public share link for easy access
- ✅ GPU-accelerated processing
- ✅ Background removal tools
- ✅ Advanced analytics dashboard

**🎉 Just like Fooocus - Simple, Fast & Professional!**
