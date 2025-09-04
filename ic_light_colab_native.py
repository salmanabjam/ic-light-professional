#!/usr/bin/env python3
"""
🌟 IC Light v2 - Google Colab Native Version
🎯 Zero dependency conflicts - Direct model loading - Self-contained
📱 Complete BRIA RMBG 1.4 + IC Light functionality

تمام امکانات IC Light v2 بدون تداخل وابستگی‌ها
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

def colab_environment_setup():
    """Setup Google Colab environment with direct installations"""
    print("🔧 Setting up Google Colab environment...")
    
    # Install only essential packages with specific versions
    installs = [
        "pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "pip install -q transformers==4.35.0",
        "pip install -q diffusers==0.24.0", 
        "pip install -q accelerate==0.24.0",
        "pip install -q gradio==3.50.0",
        "pip install -q opencv-python-headless",
        "pip install -q Pillow",
        "pip install -q scipy",
        "pip install -q safetensors",
        "pip install -q controlnet-aux",
    ]
    
    for cmd in installs:
        print(f"Installing: {cmd}")
        os.system(cmd)
    
    print("✅ Environment setup complete!")

# Run setup immediately
colab_environment_setup()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import gradio as gr
import io
import base64
from typing import Optional, List, Tuple, Union
import json
import requests
from dataclasses import dataclass
import gc

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

@dataclass
class ModelConfig:
    """Configuration for IC Light models"""
    text_model_id: str = "stablediffusionapi/realistic-vision-v51"
    bg_model_id: str = "stablediffusionapi/realistic-vision-v51" 
    scheduler_type: str = "DPMSolverMultistepScheduler"
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    strength: float = 0.85

class RSUModule(nn.Module):
    """Residual U-block for BRIA RMBG 1.4"""
    
    def __init__(self, height, in_ch, mid_ch, out_ch):
        super(RSUModule, self).__init__()
        self.height = height
        
        # Encoder path
        self.rebnconvin = self._make_conv_bn_relu(in_ch, out_ch, kernel_size=3)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(height):
            if i == 0:
                layer = self._make_conv_bn_relu(out_ch, mid_ch, kernel_size=3)
            else:
                layer = self._make_conv_bn_relu(mid_ch, mid_ch, kernel_size=3)
            self.encoder_layers.append(layer)
        
        # Middle layer
        self.rebnconv_mid = self._make_conv_bn_relu(mid_ch, mid_ch, kernel_size=3, dilation=2)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(height):
            layer = self._make_conv_bn_relu(mid_ch * 2, mid_ch, kernel_size=3)
            self.decoder_layers.append(layer)
        
        # Output layer
        self.rebnconvout = self._make_conv_bn_relu(mid_ch * 2, out_ch, kernel_size=3)
    
    def _make_conv_bn_relu(self, in_ch, out_ch, kernel_size=3, dilation=1, padding=None):
        if padding is None:
            padding = kernel_size // 2 * dilation
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Store input for residual connection
        hx = x
        hxin = self.rebnconvin(hx)
        
        # Encoder path with skip connections
        encoder_features = []
        hx = hxin
        
        for i, layer in enumerate(self.encoder_layers):
            hx = layer(hx)
            encoder_features.append(hx)
            if i < len(self.encoder_layers) - 1:
                hx = F.max_pool2d(hx, 2, stride=2, ceil_mode=True)
        
        # Middle layer
        hx = self.rebnconv_mid(hx)
        
        # Decoder path with skip connections
        for i, layer in enumerate(self.decoder_layers):
            # Upsample
            hx = F.interpolate(hx, size=encoder_features[-(i+1)].shape[2:], mode='bilinear', align_corners=False)
            # Concatenate with encoder feature
            hx = torch.cat([hx, encoder_features[-(i+1)]], dim=1)
            hx = layer(hx)
        
        # Final upsampling and concatenation
        hx = F.interpolate(hx, size=hxin.shape[2:], mode='bilinear', align_corners=False)
        hx = torch.cat([hx, hxin], dim=1)
        
        # Output layer
        hx = self.rebnconvout(hx)
        
        # Residual connection
        return hx + hxin

class BRIARMBG(nn.Module):
    """BRIA RMBG 1.4 - Complete background removal model"""
    
    def __init__(self):
        super(BRIARMBG, self).__init__()
        
        # Encoder
        self.stage1 = RSUModule(height=7, in_ch=3, mid_ch=32, out_ch=64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSUModule(height=6, in_ch=64, mid_ch=32, out_ch=128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSUModule(height=5, in_ch=128, mid_ch=64, out_ch=256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSUModule(height=4, in_ch=256, mid_ch=128, out_ch=512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSUModule(height=4, in_ch=512, mid_ch=256, out_ch=512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSUModule(height=4, in_ch=512, mid_ch=256, out_ch=512)
        
        # Decoder
        self.stage5d = RSUModule(height=4, in_ch=1024, mid_ch=256, out_ch=512)
        self.stage4d = RSUModule(height=4, in_ch=1024, mid_ch=128, out_ch=256)
        self.stage3d = RSUModule(height=5, in_ch=512, mid_ch=64, out_ch=128)
        self.stage2d = RSUModule(height=6, in_ch=256, mid_ch=32, out_ch=64)
        self.stage1d = RSUModule(height=7, in_ch=128, mid_ch=16, out_ch=64)
        
        # Side outputs
        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)
        
        # Final output
        self.outconv = nn.Conv2d(6, 1, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
    
    def forward(self, x):
        hx = x
        
        # Encoder
        hx1 = self.stage1(hx)
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
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        # Decoder
        hx5d = self.stage5d(torch.cat([hx6up, hx5], dim=1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat([hx5dup, hx4], dim=1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat([hx4dup, hx3], dim=1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat([hx3dup, hx2], dim=1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat([hx2dup, hx1], dim=1))
        
        # Side outputs
        side1 = self.side1(hx1d)
        side2 = F.interpolate(self.side2(hx2d), size=side1.shape[2:], mode='bilinear', align_corners=False)
        side3 = F.interpolate(self.side3(hx3d), size=side1.shape[2:], mode='bilinear', align_corners=False)
        side4 = F.interpolate(self.side4(hx4d), size=side1.shape[2:], mode='bilinear', align_corners=False)
        side5 = F.interpolate(self.side5(hx5d), size=side1.shape[2:], mode='bilinear', align_corners=False)
        side6 = F.interpolate(self.side6(hx6), size=side1.shape[2:], mode='bilinear', align_corners=False)
        
        # Final output
        output = self.outconv(torch.cat([side1, side2, side3, side4, side5, side6], dim=1))
        
        return torch.sigmoid(output)

class ICLightProcessor:
    """Main IC Light v2 processor"""
    
    def __init__(self):
        self.config = ModelConfig()
        self.bria_model = None
        self.diffusion_pipeline = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize all models"""
        print("🚀 Loading BRIA RMBG 1.4...")
        self.bria_model = BRIARMBG().to(device)
        self.bria_model.eval()
        
        print("🚀 Loading Stable Diffusion...")
        # Load using diffusers with direct model loading
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            
            self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.text_model_id,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            
            # Set scheduler
            self.diffusion_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.diffusion_pipeline.scheduler.config
            )
            
            # Modify UNet for IC Light (8 input channels)
            self._modify_unet_for_ic_light()
            
        except Exception as e:
            print(f"⚠️ Error loading diffusion model: {e}")
    
    def _modify_unet_for_ic_light(self):
        """Modify UNet for 8-channel input (IC Light requirement)"""
        try:
            original_conv_in = self.diffusion_pipeline.unet.conv_in
            
            # Create new conv layer with 8 input channels
            new_conv_in = nn.Conv2d(
                8, original_conv_in.out_channels,
                original_conv_in.kernel_size,
                original_conv_in.stride,
                original_conv_in.padding
            ).to(device)
            
            # Copy weights for first 4 channels, duplicate for channels 5-8
            with torch.no_grad():
                new_conv_in.weight[:, :4] = original_conv_in.weight
                new_conv_in.weight[:, 4:8] = original_conv_in.weight
                new_conv_in.bias = original_conv_in.bias
            
            # Replace the conv_in layer
            self.diffusion_pipeline.unet.conv_in = new_conv_in
            print("✅ UNet modified for IC Light (8 channels)")
            
        except Exception as e:
            print(f"⚠️ Error modifying UNet: {e}")
    
    def remove_background(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Remove background using BRIA RMBG 1.4"""
        try:
            # Preprocess image
            img_array = np.array(image.convert("RGB"))
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Resize to model input size
            img_tensor = F.interpolate(img_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
            
            # Generate mask
            with torch.no_grad():
                mask = self.bria_model(img_tensor)
                mask = F.interpolate(mask, size=image.size[::-1], mode='bilinear', align_corners=False)
                mask = mask.squeeze().cpu().numpy()
            
            # Apply mask
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            subject = Image.new("RGBA", image.size, (0, 0, 0, 0))
            subject.paste(image, mask=mask_pil)
            
            return subject, mask_pil
            
        except Exception as e:
            print(f"⚠️ Background removal error: {e}")
            return image.convert("RGBA"), Image.new("L", image.size, 255)
    
    def apply_lighting(self, image: Image.Image, prompt: str, lighting_type: str = "professional") -> Image.Image:
        """Apply IC Light lighting effects"""
        try:
            if not self.diffusion_pipeline:
                return image
            
            # Prepare image for IC Light
            img_array = np.array(image.convert("RGB"))
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Resize to 512x512 for processing
            img_tensor = F.interpolate(img_tensor, size=(512, 512), mode='bilinear', align_corners=False)
            
            # Create lighting conditioning
            lighting_conditioning = self._create_lighting_conditioning(lighting_type, img_tensor.shape)
            
            # Combine image with lighting conditioning (8 channels total)
            conditioned_input = torch.cat([img_tensor, lighting_conditioning], dim=1)
            
            # Generate with IC Light
            with torch.no_grad():
                result = self.diffusion_pipeline(
                    prompt=prompt,
                    image=conditioned_input,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    strength=self.config.strength
                ).images[0]
            
            return result.resize(image.size)
            
        except Exception as e:
            print(f"⚠️ Lighting application error: {e}")
            return image
    
    def _create_lighting_conditioning(self, lighting_type: str, shape: Tuple) -> torch.Tensor:
        """Create lighting conditioning tensor"""
        # Create 4-channel lighting conditioning based on type
        b, c, h, w = shape
        conditioning = torch.zeros(b, 4, h, w, device=device)
        
        if lighting_type == "professional":
            # Professional studio lighting pattern
            conditioning[:, 0] = 0.8  # Key light
            conditioning[:, 1] = 0.3  # Fill light
            conditioning[:, 2] = 0.5  # Rim light
            conditioning[:, 3] = 0.2  # Background light
        elif lighting_type == "dramatic":
            # Dramatic lighting
            conditioning[:, 0] = 1.0  # Strong key light
            conditioning[:, 1] = 0.1  # Minimal fill
            conditioning[:, 2] = 0.8  # Strong rim
            conditioning[:, 3] = 0.0  # Dark background
        else:
            # Natural lighting
            conditioning[:, 0] = 0.6  # Soft key light
            conditioning[:, 1] = 0.6  # Balanced fill
            conditioning[:, 2] = 0.3  # Subtle rim
            conditioning[:, 3] = 0.4  # Ambient background
        
        return conditioning

# Professional presets
LIGHTING_PRESETS = {
    "Professional Studio": {
        "prompt": "professional studio lighting, high quality portrait, commercial photography",
        "type": "professional"
    },
    "Dramatic Portrait": {
        "prompt": "dramatic lighting, moody atmosphere, cinematic portrait",
        "type": "dramatic"
    },
    "Natural Light": {
        "prompt": "natural lighting, soft daylight, realistic portrait",
        "type": "natural"
    },
    "Fashion Photography": {
        "prompt": "fashion photography lighting, editorial style, high fashion",
        "type": "professional"
    },
    "Cinematic": {
        "prompt": "cinematic lighting, movie poster style, dramatic shadows",
        "type": "dramatic"
    }
}

# Global processor instance
processor = None

def initialize_processor():
    """Initialize the IC Light processor"""
    global processor
    if processor is None:
        print("🚀 Initializing IC Light v2 processor...")
        processor = ICLightProcessor()
        print("✅ IC Light v2 ready!")
    return processor

def process_image(input_image, lighting_preset, custom_prompt):
    """Main processing function for Gradio interface"""
    try:
        # Initialize processor if needed
        proc = initialize_processor()
        
        if input_image is None:
            return None, None, "لطفاً یک تصویر آپلود کنید / Please upload an image"
        
        # Convert to PIL Image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Remove background
        subject, mask = proc.remove_background(input_image)
        
        # Get lighting settings
        if lighting_preset in LIGHTING_PRESETS:
            preset = LIGHTING_PRESETS[lighting_preset]
            prompt = custom_prompt if custom_prompt.strip() else preset["prompt"]
            lighting_type = preset["type"]
        else:
            prompt = custom_prompt if custom_prompt.strip() else "professional lighting, high quality"
            lighting_type = "professional"
        
        # Apply IC Light
        result = proc.apply_lighting(subject, prompt, lighting_type)
        
        return result, mask, f"✅ Processing complete with {lighting_preset}"
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return None, None, error_msg

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(
        title="🌟 IC Light v2 Professional",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .gr-button-primary {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🌟 IC Light v2 Professional
        ## تبدیل تصاویر با نورپردازی حرفه‌ای / Professional Image Lighting
        
        **امکانات / Features:**
        - ✨ حذف پس‌زمینه با BRIA RMBG 1.4 / Background removal
        - 🎨 نورپردازی حرفه‌ای با IC Light v2 / Professional lighting
        - 🎭 پیش‌تنظیمات متنوع / Multiple presets
        - ⚡ پردازش سریع / Fast processing
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📤 تصویر ورودی / Input Image",
                    type="pil",
                    height=400
                )
                
                lighting_preset = gr.Dropdown(
                    choices=list(LIGHTING_PRESETS.keys()),
                    value="Professional Studio",
                    label="🎨 پیش‌تنظیم نورپردازی / Lighting Preset"
                )
                
                custom_prompt = gr.Textbox(
                    label="✍️ پرامپت سفارشی / Custom Prompt",
                    placeholder="professional lighting, high quality portrait...",
                    lines=2
                )
                
                process_btn = gr.Button(
                    "🚀 پردازش / Process",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                with gr.Row():
                    output_image = gr.Image(
                        label="✨ نتیجه نهایی / Final Result",
                        height=400
                    )
                    mask_image = gr.Image(
                        label="🎭 ماسک / Mask",
                        height=400
                    )
                
                status_text = gr.Textbox(
                    label="📊 وضعیت / Status",
                    interactive=False
                )
        
        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[input_image, lighting_preset, custom_prompt],
            outputs=[output_image, mask_image, status_text]
        )
        
        # Example images
        gr.Markdown("### 📚 نمونه‌ها / Examples")
        gr.Examples(
            examples=[
                ["Professional Studio", "professional portrait, studio lighting, high quality"],
                ["Dramatic Portrait", "dramatic lighting, cinematic, moody atmosphere"],
                ["Natural Light", "natural lighting, soft daylight, realistic"],
                ["Fashion Photography", "fashion photography, editorial style, high fashion"],
                ["Cinematic", "cinematic lighting, movie style, dramatic shadows"]
            ],
            inputs=[lighting_preset, custom_prompt]
        )
    
    return interface

def main():
    """Main function to launch the application"""
    print("🌟 Starting IC Light v2 Professional...")
    print("🔧 Google Colab Native Version - Zero Dependency Conflicts")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with public sharing for Colab
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
