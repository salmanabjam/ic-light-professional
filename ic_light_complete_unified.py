#!/usr/bin/env python3
# IC Light v2 Complete - All Features in One Interface
# سیستم کامل IC Light v2 - تمام امکانات در یک واسط

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageOps
import safetensors.torch as sf
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.hub import download_url_to_file
from huggingface_hub import PyTorchModelHubMixin

# Set environment variables for optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("🚀 Loading IC Light v2 Complete System...")
print("🌟 نظام IC Light v2 کامل - تمام امکانات در یک واسط")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# BRIA RMBG Implementation
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    return src

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
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

class BriaRMBG(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_dim=320, num_frames=8):
        super(BriaRMBG, self).__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage1 = RSU7(64, 32, 64)
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
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        hx = x
        hxin = self.conv_in(hx)
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)
        d2 = _upsample_like(d2, d1)
        d3 = _upsample_like(d3, d1)
        d4 = _upsample_like(d4, d1)
        d5 = _upsample_like(d5, d1)
        d6 = _upsample_like(d6, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

# Professional preset prompts
PROFESSIONAL_PRESETS = [
    ("Golden Hour Portrait", "beautiful woman, detailed face, golden time, warm sunlight, golden hour lighting"),
    ("نور طلایی عکاسی", "زن زیبا، صورت با جزئیات، نور طلایی، نور گرم خورشید، نورپردازی ساعت طلایی"),
    ("Studio Lighting", "beautiful woman, detailed face, soft studio lighting, professional portrait"),
    ("نور استودیو حرفه‌ای", "زن زیبا، صورت با جزئیات، نور نرم استودیو، عکس پرتره حرفه‌ای"),
    ("Cinematic Moody", "handsome man, detailed face, cinematic lighting, dramatic shadows, moody atmosphere"),
    ("سینمایی تیره", "مرد جذاب، صورت با جزئیات، نورپردازی سینمایی، سایه‌های دراماتیک"),
    ("Natural Window Light", "beautiful woman, detailed face, sunshine from window, natural lighting"),
    ("نور طبیعی پنجره", "زن زیبا، صورت با جزئیات، نور خورشید از پنجره، نورپردازی طبیعی"),
    ("Neon Cyberpunk", "beautiful woman, detailed face, neon light, city, sci-fi RGB glowing, cyberpunk"),
    ("نئون سایبرپانک", "زن زیبا، صورت با جزئیات، نور نئون، شهر، نور RGB سایبرپانک"),
    ("Sunset Romance", "beautiful woman, detailed face, sunset over sea, romantic lighting, warm colors"),
    ("عاشقانه غروب", "زن زیبا، صورت با جزئیات، غروب روی دریا، نور عاشقانه، رنگ‌های گرم"),
    ("Shadow Drama", "handsome man, detailed face, light and shadow, dramatic contrast, film noir"),
    ("دراما سایه", "مرد جذاب، صورت با جزئیات، نور و سایه، کنتراست دراماتیک"),
    ("Warm Bedroom", "beautiful woman, detailed face, warm atmosphere, at home, bedroom, cozy lighting"),
    ("اتاق خواب گرم", "زن زیبا، صورت با جزئیات، فضای گرم، در خانه، اتاق خواب"),
    ("Magic Fantasy", "beautiful woman, detailed face, magic lit, fantasy lighting, ethereal glow"),
    ("جادویی فانتزی", "زن زیبا، صورت با جزئیات، نور جادویی، نورپردازی فانتزی"),
    ("Gothic Dark", "handsome man, detailed face, evil, gothic, Yharnam, dark atmosphere"),
    ("تیره گوتیک", "مرد جذاب، صورت با جزئیات، شیطانی، گوتیک، فضای تیره"),
    ("Window Shadow", "beautiful woman, detailed face, shadow from window, soft contrast"),
    ("سایه پنجره", "زن زیبا، صورت با جزئیات، سایه از پنجره، کنتراست نرم")
]

# Background examples for background-conditioned mode
BACKGROUND_EXAMPLES = [
    "Left Light - نور چپ",
    "Right Light - نور راست", 
    "Top Light - نور بالا",
    "Bottom Light - نور پایین"
]

# Load models
print("📦 Loading Stable Diffusion base model...")
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

print("📦 Loading BRIA RMBG 1.4...")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Download IC Light models
print("📥 Setting up IC Light models...")
fc_model_path = './iclight_sd15_fc.safetensors'
fbc_model_path = './iclight_sd15_fbc.safetensors'

if not os.path.exists(fc_model_path):
    print("📥 Downloading IC Light FC model...")
    download_url_to_file(
        url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors',
        dst=fc_model_path
    )

if not os.path.exists(fbc_model_path):
    print("📥 Downloading IC Light FBC model...")
    download_url_to_file(
        url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors',
        dst=fbc_model_path
    )

# Load FC model
print("🔧 Loading IC Light FC weights...")
sd_offset = sf.load_file(fc_model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Move models to device
print("🚀 Optimizing models...")
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# Optimizations
from diffusers.models.attention_processor import AttnProcessor2_0
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Schedulers
ddim_scheduler = DDIMScheduler.from_pretrained(sd15_name, subfolder="scheduler")
dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(sd15_name, subfolder="scheduler", algorithm_type="sde-dpmsolver++", solver_order=2)
euler_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(sd15_name, subfolder="scheduler")

# Helper functions
def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(original_width * scale_factor)
    resized_height = int(original_height * scale_factor)
    pil_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    pil_image = pil_image.crop((left, top, right, bottom))
    return pil_image

def run_rmbg(img):
    H, W = img.shape[:2]
    image_size = (1024, 1024)
    input_images = img.astype(np.float32) / 255
    input_images = torch.from_numpy(input_images).permute(2, 0, 1).unsqueeze(0).to(device)
    input_images = F.interpolate(input_images, size=image_size, mode='bilinear', align_corners=False)
    with torch.no_grad():
        preds = rmbg(input_images)[0]
    pred = preds[0].squeeze()
    pred_pil = Image.fromarray((pred.cpu().numpy() * 255).astype(np.uint8), mode='L')
    pred_pil = pred_pil.resize((W, H), Image.LANCZOS)
    foreground = img.copy().astype(np.float32)
    alpha = np.array(pred_pil).astype(np.float32) / 255
    alpha = alpha[:, :, None]
    foreground = foreground * alpha + 255 * (1 - alpha)
    foreground = foreground.astype(np.uint8)
    return foreground, pred_pil

def encode_prompt_pair(positive_prompt, negative_prompt):
    max_length = tokenizer.model_max_length
    positive_ids = tokenizer(positive_prompt, truncation=False, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
    negative_ids = tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
    positive_embs = text_encoder(positive_ids).last_hidden_state
    negative_embs = text_encoder(negative_ids).last_hidden_state
    return positive_embs, negative_embs

class BGSource:
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light" 
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

def process_image(input_image, prompt, bg_source, image_width, image_height, num_samples, seed, steps, cfg_scale, denoise, lowres_denoise, bg_image=None, use_background_mode=False):
    try:
        # Process input image
        input_fg = resize_and_center_crop(input_image, image_width, image_height)
        input_fg = np.array(input_fg)
        
        # Background removal
        input_fg, matting = run_rmbg(input_fg)
        
        # Generate conditioning
        if use_background_mode and bg_image is not None:
            # Background-conditioned mode
            bg_image = resize_and_center_crop(bg_image, image_width, image_height)
            bg_image = np.array(bg_image).astype(np.float32) / 127.5 - 1.0
            bg_latent = vae.encode(torch.from_numpy(bg_image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)).latent_dist.mode()
            
            # Foreground conditioning
            input_fg_rgb = input_fg.astype(np.float32) / 127.5 - 1.0
            fg_latent = vae.encode(torch.from_numpy(input_fg_rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)).latent_dist.mode()
            
            concat_conds = torch.cat([fg_latent, bg_latent], dim=1)
        else:
            # Text-conditioned mode
            if bg_source == BGSource.LEFT:
                gradient = np.linspace(255, 0, image_width).reshape(1, image_width, 1).repeat(image_height, axis=0).repeat(3, axis=2).astype(np.uint8)
            elif bg_source == BGSource.RIGHT:
                gradient = np.linspace(0, 255, image_width).reshape(1, image_width, 1).repeat(image_height, axis=0).repeat(3, axis=2).astype(np.uint8)
            elif bg_source == BGSource.TOP:
                gradient = np.linspace(255, 0, image_height).reshape(image_height, 1, 1).repeat(image_width, axis=1).repeat(3, axis=2).astype(np.uint8)
            elif bg_source == BGSource.BOTTOM:
                gradient = np.linspace(0, 255, image_height).reshape(image_height, 1, 1).repeat(image_width, axis=1).repeat(3, axis=2).astype(np.uint8)
            else:
                gradient = np.full((image_height, image_width, 3), 127, dtype=np.uint8)
            
            gradient = gradient.astype(np.float32) / 127.5 - 1.0
            gradient_latent = vae.encode(torch.from_numpy(gradient).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)).latent_dist.mode()
            
            input_fg_rgb = input_fg.astype(np.float32) / 127.5 - 1.0
            fg_latent = vae.encode(torch.from_numpy(input_fg_rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)).latent_dist.mode()
            
            concat_conds = torch.cat([fg_latent, gradient_latent], dim=1)
        
        # Text encoding
        conds, unconds = encode_prompt_pair(prompt, "")
        
        # Generation loop
        results = []
        generator = torch.manual_seed(seed)
        
        for i in range(num_samples):
            latents = torch.randn((1, 4, image_height // 8, image_width // 8), generator=generator, dtype=torch.float16, device=device)
            
            # Diffusion
            ddim_scheduler.set_timesteps(steps)
            for t in ddim_scheduler.timesteps:
                with torch.no_grad():
                    noise_pred_cond = unet(latents, t, conds, cross_attention_kwargs={"concat_conds": concat_conds}).sample
                    noise_pred_uncond = unet(latents, t, unconds, cross_attention_kwargs={"concat_conds": concat_conds}).sample
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                    latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode
            with torch.no_grad():
                decoded_img = vae.decode(latents.to(dtype=torch.bfloat16)).sample
            decoded_img = (decoded_img * 0.5 + 0.5).clamp(0, 1)
            decoded_img = decoded_img.permute(0, 2, 3, 1).cpu().numpy()
            result_img = numpy_to_pil(decoded_img)[0]
            results.append(result_img)
        
        return results[0] if len(results) == 1 else results
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return None

# Gradio interface
def create_interface():
    with gr.Blocks(title="🌟 IC Light v2 Complete - تمام امکانات", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #2196F3; margin-bottom: 10px;">🌟 IC Light v2 Complete System</h1>
            <h2 style="color: #4CAF50; margin-bottom: 15px;">سیستم کامل IC Light v2 - تمام امکانات</h2>
            <p style="color: #666; font-size: 16px;">Professional Image Relighting with All Features | نورپردازی حرفه‌ای تصاویر با تمام امکانات</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Input - ورودی</h3>")
                input_image = gr.Image(type="numpy", label="Upload Image | تصویر را آپلود کنید")
                
                gr.HTML("<h3>💡 Lighting Control - کنترل نور</h3>")
                mode_choice = gr.Radio(
                    choices=[
                        ("Text Prompts (متن)", "text"),
                        ("Background Image (تصویر پس‌زمینه)", "background")
                    ],
                    value="text",
                    label="Mode | حالت"
                )
                
                # Text mode controls
                with gr.Group(visible=True) as text_controls:
                    preset_choice = gr.Dropdown(
                        choices=[(f"{name} | {prompt[:50]}...", prompt) for name, prompt in PROFESSIONAL_PRESETS],
                        label="Professional Presets | پرست‌های حرفه‌ای",
                        value=PROFESSIONAL_PRESETS[0][1]
                    )
                    
                    prompt = gr.Textbox(
                        value=PROFESSIONAL_PRESETS[0][1],
                        label="Custom Prompt | پرامپت سفارشی",
                        lines=3
                    )
                    
                    bg_source = gr.Radio(
                        choices=[BGSource.LEFT, BGSource.RIGHT, BGSource.TOP, BGSource.BOTTOM],
                        value=BGSource.LEFT,
                        label="Light Direction | جهت نور"
                    )
                
                # Background mode controls  
                with gr.Group(visible=False) as bg_controls:
                    bg_image = gr.Image(type="numpy", label="Background Image | تصویر پس‌زمینه")
                    bg_prompt = gr.Textbox(
                        value="beautiful woman, detailed face, natural lighting",
                        label="Subject Description | توضیحات سوژه",
                        lines=2
                    )
                
            with gr.Column(scale=1):
                gr.HTML("<h3>🎨 Results - نتایج</h3>")
                output_images = gr.Gallery(
                    label="Generated Images | تصاویر تولید شده",
                    columns=2,
                    rows=2,
                    height=600
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate | تولید", variant="primary", size="lg")
                
                gr.HTML("<h3>⚙️ Advanced Settings - تنظیمات پیشرفته</h3>")
                with gr.Accordion("Settings | تنظیمات", open=False):
                    with gr.Row():
                        image_width = gr.Slider(256, 1024, 512, step=64, label="Width | عرض")
                        image_height = gr.Slider(256, 1024, 768, step=64, label="Height | ارتفاع")
                    
                    with gr.Row():
                        num_samples = gr.Slider(1, 4, 1, step=1, label="Samples | نمونه‌ها")
                        seed = gr.Slider(-1, 999999999, -1, step=1, label="Seed | سید")
                    
                    with gr.Row():
                        steps = gr.Slider(1, 100, 25, step=1, label="Steps | مراحل")
                        cfg_scale = gr.Slider(1.0, 32.0, 2.0, step=0.1, label="CFG Scale")
                    
                    with gr.Row():
                        denoise = gr.Slider(0.1, 1.0, 0.9, step=0.1, label="Denoise | کاهش نویز")
                        lowres_denoise = gr.Slider(0.1, 1.0, 0.5, step=0.1, label="Lowres Denoise")
        
        # Event handlers
        def on_preset_change(preset_value):
            return preset_value
        
        def on_mode_change(mode):
            return gr.update(visible=(mode == "text")), gr.update(visible=(mode == "background"))
        
        def on_generate(input_img, mode, preset, custom_prompt, bg_src, bg_img, bg_prompt_val, width, height, samples, seed_val, steps_val, cfg, denoise_val, lowres_denoise_val):
            if input_img is None:
                return None
            
            # Use custom prompt if provided, otherwise use preset
            final_prompt = custom_prompt if custom_prompt.strip() else preset
            if mode == "background":
                final_prompt = bg_prompt_val
            
            # Handle seed
            actual_seed = seed_val if seed_val >= 0 else np.random.randint(0, 999999999)
            
            result = process_image(
                input_image=input_img,
                prompt=final_prompt,
                bg_source=bg_src,
                image_width=int(width),
                image_height=int(height),
                num_samples=int(samples),
                seed=actual_seed,
                steps=int(steps_val),
                cfg_scale=cfg,
                denoise=denoise_val,
                lowres_denoise=lowres_denoise_val,
                bg_image=bg_img if mode == "background" else None,
                use_background_mode=(mode == "background")
            )
            
            return result if isinstance(result, list) else [result] if result else None
        
        # Connect events
        preset_choice.change(on_preset_change, preset_choice, prompt)
        mode_choice.change(on_mode_change, mode_choice, [text_controls, bg_controls])
        
        generate_btn.click(
            on_generate,
            inputs=[
                input_image, mode_choice, preset_choice, prompt, bg_source, 
                bg_image, bg_prompt, image_width, image_height, num_samples, 
                seed, steps, cfg_scale, denoise, lowres_denoise
            ],
            outputs=output_images
        )
        
        # Example gallery
        gr.HTML("""
        <div style="margin-top: 30px; text-align: center;">
            <h3>✨ Features | امکانات ✨</h3>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
                <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <h4>🎭 Text-Conditioned</h4>
                    <p>20+ Professional Presets<br/>پرست‌های حرفه‌ای</p>
                </div>
                <div style="background: #f0fff0; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <h4>🖼️ Background-Conditioned</h4>
                    <p>Custom Background Lighting<br/>نورپردازی پس‌زمینه سفارشی</p>
                </div>
                <div style="background: #fff8dc; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <h4>🔧 BRIA RMBG 1.4</h4>
                    <p>Professional Background Removal<br/>حذف پس‌زمینه حرفه‌ای</p>
                </div>
                <div style="background: #ffe4e1; padding: 15px; border-radius: 10px; min-width: 200px;">
                    <h4>🌐 Bilingual</h4>
                    <p>English + Persian Support<br/>پشتیبانی انگلیسی + فارسی</p>
                </div>
            </div>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    print("🎉 Starting IC Light v2 Complete System...")
    print("🌟 راه‌اندازی سیستم کامل IC Light v2...")
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        quiet=False,
        inbrowser=True
    )
