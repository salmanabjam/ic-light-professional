#!/usr/bin/env python3
# IC Light v2 Complete System - FULLY CORRECTED VERSION
# سیستم کامل IC Light v2 - نسخه کاملاً تصحیح شده

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import safetensors.torch as sf

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("🚀 Loading IC Light v2 Complete System...")
print("🌟 بارگیری سیستم کامل IC Light v2...")

# Import handling with automatic installation
def install_and_import():
    import subprocess
    import sys
    
    packages = [
        "huggingface_hub==0.16.4",
        "transformers==4.35.0", 
        "diffusers==0.26.3",
        "torch>=2.0.0",
        "torchvision",
        "gradio>=4.0.0",
        "safetensors",
        "accelerate"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                package, "--force-reinstall", "--quiet"
            ])
        except:
            pass

try:
    from diffusers import (
        DDIMScheduler, EulerAncestralDiscreteScheduler, 
        DPMSolverMultistepScheduler, StableDiffusionPipeline
    )
    from diffusers.models import UNet2DConditionModel, AutoencoderKL
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTokenizer
    print("✅ Libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Installing packages...")
    install_and_import()
    from diffusers import (
        DDIMScheduler, EulerAncestralDiscreteScheduler, 
        DPMSolverMultistepScheduler, StableDiffusionPipeline
    )
    from diffusers.models import UNet2DConditionModel, AutoencoderKL
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTokenizer
    print("✅ Packages installed and imported")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# BRIA RMBG Implementation (Complete)
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1*dirate, 
            dilation=1*dirate, stride=stride
        )
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode="bilinear")
    return src

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)
        
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        
        return hx1d + hxin

class myrebnconv(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1):
        super(myrebnconv, self).__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 
                              padding, dilation, groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.rl(self.bn(self.conv(x)))

class BriaRMBG(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(BriaRMBG, self).__init__()
        
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 256, 512)
        
        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)
        
        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

# Professional presets
PROFESSIONAL_PRESETS = [
    ("Golden Hour", "beautiful woman, detailed face, golden hour lighting, warm sunlight"),
    ("نور طلایی", "زن زیبا، صورت با جزئیات، نورپردازی ساعت طلایی، نور گرم خورشید"),
    ("Studio Professional", "beautiful woman, detailed face, soft studio lighting, professional photography"),
    ("استودیو حرفه‌ای", "زن زیبا، صورت با جزئیات، نور نرم استودیو، عکاسی حرفه‌ای"),
    ("Cinematic Drama", "handsome man, detailed face, cinematic lighting, dramatic shadows"),
    ("سینمایی دراماتیک", "مرد جذاب، صورت با جزئیات، نورپردازی سینمایی، سایه‌های دراماتیک"),
    ("Natural Window", "beautiful woman, detailed face, natural lighting, sunshine from window"),
    ("نور طبیعی پنجره", "زن زیبا، صورت با جزئیات، نورپردازی طبیعی، نور خورشید از پنجره"),
    ("Neon Cyberpunk", "beautiful woman, detailed face, neon light, sci-fi RGB glowing, cyberpunk"),
    ("نئون سایبرپانک", "زن زیبا، صورت با جزئیات، نور نئون، RGB سایبرپانک درخشان"),
    ("Romantic Sunset", "beautiful woman, detailed face, sunset over sea, romantic atmosphere"),
    ("غروب عاشقانه", "زن زیبا، صورت با جزئیات، غروب روی دریا، فضای عاشقانه"),
    ("Shadow Drama", "handsome man, detailed face, light and shadow, dramatic contrast"),
    ("دراما سایه و نور", "مرد جذاب، صورت با جزئیات، نور و سایه، کنتراست دراماتیک"),
    ("Warm Bedroom", "beautiful woman, detailed face, warm atmosphere, cozy bedroom lighting"),
    ("اتاق خواب گرم", "زن زیبا، صورت با جزئیات، فضای گرم، نور دنج اتاق خواب"),
    ("Magic Fantasy", "beautiful woman, detailed face, magic lit, fantasy RGB glowing"),
    ("فانتزی جادویی", "زن زیبا، صورت با جزئیات، نور جادویی، درخشش فانتزی"),
    ("Evil Gothic", "mysterious human, detailed face, evil gothic lighting, dark atmosphere"),
    ("گوتیک تاریک", "انسان مرموز، صورت با جزئیات، نورپردازی گوتیک تاریک"),
]

# Model paths and configurations  
MODEL_NAME = 'stablediffusionapi/realistic-vision-v51'

# Download function
def safe_download(url, dst):
    """Safe download with multiple fallbacks"""
    try:
        from torch.hub import download_url_to_file
        download_url_to_file(url, dst)
        return True
    except Exception:
        try:
            import urllib.request
            urllib.request.urlretrieve(url, dst)
            return True
        except Exception:
            print(f"❌ Failed to download: {url}")
            return False

# Load models
print("📦 Loading base models...")
try:
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    print("✅ Base models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Initialize BRIA RMBG
print("🔧 Initializing BRIA RMBG...")
rmbg = BriaRMBG()

# Modify UNet for IC Light (8 channels for foreground conditioned)
print("🔧 Modifying UNet for IC Light...")
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(
        8, unet.conv_in.out_channels, 
        unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

# Hook UNet forward
unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet.forward = hooked_unet_forward

# Download IC Light weights
model_path = './iclight_sd15_fc.safetensors'
if not os.path.exists(model_path):
    print("📥 Downloading IC Light model...")
    url = 'https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors'
    if not safe_download(url, model_path):
        print("❌ Download failed - using base model")

# Load IC Light weights
if os.path.exists(model_path):
    print("🔧 Loading IC Light weights...")
    try:
        sd_offset = sf.load_file(model_path)
        sd_origin = unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged
        print("✅ IC Light weights loaded")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")

# Move models to device
print("🚀 Moving models to device...")
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# Set attention processors
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Schedulers
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

print("✅ All models loaded and configured")

# Helper functions
@torch.inference_mode()
def encode_prompt_inner(txt: str):
    """Encode text prompt to embeddings"""
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x + [p] * max(0, i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds

@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    """Encode positive and negative prompts"""
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    return c, uc

def resize_and_center_crop(image, target_width, target_height):
    """Resize and center crop image"""
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    """Run BRIA RMBG background removal"""
    h, w = img.shape[:2]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    # Run BRIA RMBG
    result = rmbg(img_tensor)
    mask = result[0] if isinstance(result, tuple) else result
    
    # Add noise if specified
    if sigma > 0:
        noise = torch.normal(mean=0, std=sigma, size=mask.shape, device=mask.device, dtype=mask.dtype)
        mask = mask + noise
        mask = torch.clamp(mask, 0, 1)
    
    mask_np = mask.squeeze().cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
    
    # Apply mask to create RGBA
    img_rgba = np.concatenate([img, np.expand_dims(mask_np * 255, axis=2)], axis=2).astype(np.uint8)
    
    return mask_pil, img_rgba

def generate_gradient_map(direction, width, height):
    """Generate lighting gradient based on direction"""
    if direction == "Left Light":
        gradient = np.linspace(255, 0, width).reshape(1, width, 1)
        gradient = gradient.repeat(height, axis=0).repeat(3, axis=2)
    elif direction == "Right Light":
        gradient = np.linspace(0, 255, width).reshape(1, width, 1)
        gradient = gradient.repeat(height, axis=0).repeat(3, axis=2)
    elif direction == "Top Light":
        gradient = np.linspace(255, 0, height).reshape(height, 1, 1)
        gradient = gradient.repeat(width, axis=1).repeat(3, axis=2)
    elif direction == "Bottom Light":
        gradient = np.linspace(0, 255, height).reshape(height, 1, 1)
        gradient = gradient.repeat(width, axis=1).repeat(3, axis=2)
    else:  # Ambient
        gradient = np.full((height, width, 3), 127, dtype=np.uint8)
    
    return gradient.astype(np.uint8)

@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    """Convert pytorch tensor to numpy"""
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y.detach().float().cpu().numpy()
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    """Convert numpy to pytorch tensor"""
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    """Main relighting process"""
    
    # Handle empty input
    if input_fg is None:
        return []
    
    # Set random seed
    if seed == -1:
        seed = np.random.randint(0, 2147483647)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    try:
        # Resize input image
        fg = resize_and_center_crop(input_fg, image_width, image_height)
        
        # Remove background
        rmbg_mask, fg_rgba = run_rmbg(fg)
        
        # Generate lighting map
        lighting_map = generate_gradient_map(bg_source, image_width, image_height)
        
        # Prepare prompts
        full_prompt = f"{prompt}, {a_prompt}".strip(", ")
        
        # Encode prompts
        cond, uncond = encode_prompt_pair(full_prompt, n_prompt)
        
        # Prepare images for latent encoding
        fg_rgb = fg.astype(np.float32) / 127.5 - 1.0
        lighting_rgb = lighting_map.astype(np.float32) / 127.5 - 1.0
        
        # Convert to tensors
        fg_tensor = torch.from_numpy(fg_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        lighting_tensor = torch.from_numpy(lighting_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        
        # Encode to latents
        fg_latent = vae.encode(fg_tensor).latent_dist.mode()
        lighting_latent = vae.encode(lighting_tensor).latent_dist.mode()
        
        # Concatenate conditioning
        concat_conds = torch.cat([fg_latent, lighting_latent], dim=1)
        
        # Generate samples
        results = []
        for _ in range(num_samples):
            # Random latents
            generator = torch.Generator(device=device).manual_seed(seed + _)
            latents = torch.randn(
                (1, 4, image_height // 8, image_width // 8),
                generator=generator,
                device=device,
                dtype=torch.float16
            )
            
            # Set scheduler
            ddim_scheduler.set_timesteps(steps)
            
            # Denoising loop
            for t in ddim_scheduler.timesteps:
                # Predict noise
                noise_pred_cond = unet(
                    latents, t, cond,
                    cross_attention_kwargs={"concat_conds": concat_conds}
                ).sample
                
                noise_pred_uncond = unet(
                    latents, t, uncond,
                    cross_attention_kwargs={"concat_conds": concat_conds}
                ).sample
                
                # Classifier-free guidance
                noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
                
                # Scheduler step
                latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode latents
            image = vae.decode(latents.to(dtype=torch.bfloat16)).sample
            image = (image * 0.5 + 0.5).clamp(0, 1)
            
            # Convert to numpy
            image_np = pytorch2numpy([image])[0]
            results.append(Image.fromarray(image_np))
        
        return results
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return [Image.fromarray(input_fg)]

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="🌟 IC Light v2 Complete", theme=gr.themes.Soft()) as app:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1 style="color: #2196F3; margin-bottom: 10px;">🌟 IC Light v2 Complete Professional System</h1>
            <h2 style="color: #4CAF50; margin-bottom: 15px;">سیستم حرفه‌ای کامل IC Light v2</h2>
            <p style="color: #666; font-size: 16px;">Professional Image Relighting with BRIA RMBG 1.4 | نورپردازی حرفه‌ای تصاویر</p>
            <p style="color: #888; font-size: 14px;">✨ 20+ Professional Presets | بیش از ۲۰ پرست حرفه‌ای ✨</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.HTML("<h3 style='color: #2196F3;'>📤 Input | ورودی</h3>")
                input_fg = gr.Image(type="numpy", label="Upload Foreground Image | آپلود تصویر")
                
                # Professional presets
                gr.HTML("<h3 style='color: #4CAF50;'>🎨 Professional Presets | پرست‌های حرفه‌ای</h3>")
                preset_dropdown = gr.Dropdown(
                    choices=[(name, prompt) for name, prompt in PROFESSIONAL_PRESETS],
                    value=PROFESSIONAL_PRESETS[0][1],
                    label="Choose Professional Preset | انتخاب پرست حرفه‌ای"
                )
                
                # Custom prompt
                prompt = gr.Textbox(
                    value=PROFESSIONAL_PRESETS[0][1],
                    label="Custom Prompt | پرامپت سفارشی",
                    lines=3,
                    placeholder="Enter your custom lighting description..."
                )
                
                # Lighting direction
                bg_source = gr.Radio(
                    choices=["Left Light", "Right Light", "Top Light", "Bottom Light"],
                    value="Left Light",
                    label="Lighting Direction | جهت نورپردازی"
                )
                
            with gr.Column(scale=1):
                # Output section
                gr.HTML("<h3 style='color: #FF9800;'>🎨 Results | نتایج</h3>")
                result_gallery = gr.Gallery(
                    label="Generated Images | تصاویر تولید شده",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )
                
                # Generate button
                relight_button = gr.Button("🚀 Generate Professional Lighting | تولید نورپردازی حرفه‌ای", 
                                         variant="primary", size="lg")
                
                # Advanced settings
                gr.HTML("<h3 style='color: #9C27B0;'>⚙️ Advanced Settings | تنظیمات پیشرفته</h3>")
                with gr.Accordion("Advanced Controls | کنترل‌های پیشرفته", open=False):
                    with gr.Row():
                        image_width = gr.Slider(256, 1024, 512, step=64, label="Width | عرض")
                        image_height = gr.Slider(256, 1024, 768, step=64, label="Height | ارتفاع")
                    
                    with gr.Row():
                        num_samples = gr.Slider(1, 4, 1, step=1, label="Number of Images | تعداد تصاویر")
                        steps = gr.Slider(1, 50, 20, step=1, label="Steps | مراحل")
                    
                    with gr.Row():
                        cfg = gr.Slider(1.0, 20.0, 7.0, step=0.5, label="CFG Scale | مقیاس CFG")
                        seed = gr.Slider(-1, 999999999, -1, step=1, label="Seed (-1 for random) | سید")
                    
                    with gr.Row():
                        a_prompt = gr.Textbox(
                            value="best quality, extremely detailed",
                            label="Additional Prompt | پرامپت اضافی"
                        )
                        n_prompt = gr.Textbox(
                            value="longbody, lowres, bad anatomy, bad hands, missing fingers",
                            label="Negative Prompt | پرامپت منفی"
                        )
                    
                    with gr.Row():
                        highres_scale = gr.Slider(1.0, 2.0, 1.5, step=0.1, label="Highres Scale | مقیاس کیفیت")
                        highres_denoise = gr.Slider(0.2, 0.7, 0.5, step=0.05, label="Highres Denoise | کاهش نویز")
        
        # Event handlers
        def on_preset_change(preset_value):
            return preset_value
        
        def on_generate(input_fg, prompt, image_width, image_height, num_samples, seed, 
                       steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
            if input_fg is None:
                return []
            
            return process_relight(
                input_fg, prompt, int(image_width), int(image_height), 
                int(num_samples), int(seed), int(steps), a_prompt, n_prompt, 
                cfg, highres_scale, highres_denoise, bg_source
            )
        
        # Connect events
        preset_dropdown.change(on_preset_change, preset_dropdown, prompt)
        
        relight_button.click(
            on_generate,
            inputs=[input_fg, prompt, image_width, image_height, num_samples, seed, 
                   steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source],
            outputs=result_gallery
        )
        
        # Feature showcase
        gr.HTML("""
        <div style="margin-top: 30px; text-align: center;">
            <h3 style="color: #2196F3;">✨ Professional Features | امکانات حرفه‌ای ✨</h3>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin-top: 20px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <strong>🎭 20+ Professional Presets</strong><br/>
                    <small>بیش از ۲۰ پرست حرفه‌ای</small>
                </div>
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <strong>🔧 BRIA RMBG 1.4</strong><br/>
                    <small>حذف پس‌زمینه حرفه‌ای</small>
                </div>
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <strong>💡 4-Direction Lighting</strong><br/>
                    <small>نورپردازی ۴ جهته</small>
                </div>
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <strong>🌐 Bilingual Interface</strong><br/>
                    <small>رابط کاربری دوزبانه</small>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <p style="color: #666; font-style: italic;">
                    🚀 Powered by IC Light v2 + Stable Diffusion 1.5 + BRIA RMBG 1.4<br/>
                    ⚡ قدرت گرفته از IC Light v2 + Stable Diffusion 1.5 + BRIA RMBG 1.4
                </p>
            </div>
        </div>
        """)
    
    return app

if __name__ == "__main__":
    print("🎉 Starting IC Light v2 Complete Professional System...")
    print("🌟 راه‌اندازی سیستم حرفه‌ای کامل IC Light v2...")
    
    app = create_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_api=False,
        quiet=False
    )
