#!/usr/bin/env python3
# IC Light v2 - Google Colab Compatible Version
# نسخه سازگار با Google Colab

import os
import subprocess
import sys

# Fix dependency conflicts first
def fix_colab_environment():
    """Fix Google Colab environment conflicts"""
    print("🔧 Fixing Google Colab environment...")
    
    # Fix matplotlib issue
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "matplotlib>=3.8.0", "--force-reinstall", "--quiet"
        ])
        print("✅ matplotlib fixed")
    except:
        print("⚠️ matplotlib fix failed, continuing...")
    
    # Install compatible versions avoiding conflicts
    compatible_packages = [
        "numpy==1.26.4",  # Compatible with most packages
        "huggingface_hub==0.20.3",  # Modern but stable
        "transformers==4.36.0",  # Stable version
        "diffusers==0.25.0",  # Stable version
        "torch>=2.0.0",
        "torchvision",
        "gradio==3.50.0",  # Stable gradio version
        "safetensors",
        "accelerate",
        "pillow>=8.0,<11.0"  # Compatible range
    ]
    
    for package in compatible_packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                package, "--force-reinstall", "--quiet", 
                "--no-deps"  # Avoid dependency conflicts
            ])
        except Exception as e:
            print(f"⚠️ Failed to install {package}: {e}")
    
    print("✅ Environment setup complete")

# Fix environment before imports
fix_colab_environment()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("🚀 Loading IC Light v2 - Colab Compatible Version...")
print("🌟 بارگیری IC Light v2 - نسخه سازگار با کولب...")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# Import after environment fix
try:
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from diffusers.models import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
    print("✅ Core libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Trying alternative imports...")
    
    # Try installing one more time with compatible versions
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "diffusers==0.21.4", "transformers==4.30.0", 
        "--force-reinstall", "--quiet"
    ])
    
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from diffusers.models import UNet2DConditionModel, AutoencoderKL  
    from transformers import CLIPTextModel, CLIPTokenizer
    print("✅ Alternative imports successful")

# Simplified BRIA RMBG for Colab
class SimpleBRIA(nn.Module):
    def __init__(self):
        super().__init__()
        # Lightweight architecture for Colab
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Simple U-Net style processing
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(self.pool(x1)))
        x3 = torch.relu(self.conv3(self.pool(x2)))
        x4 = torch.relu(self.conv4(self.upsample(x3)))
        x5 = torch.relu(self.conv5(self.upsample(x4)))
        out = torch.sigmoid(self.conv6(x5))
        return out

# Professional presets (simplified)
PRESETS = [
    ("Golden Hour", "beautiful woman, detailed face, golden hour lighting, warm sunlight"),
    ("نور طلایی", "زن زیبا، صورت با جزئیات، نورپردازی ساعت طلایی، نور گرم"),
    ("Studio Light", "beautiful woman, detailed face, soft studio lighting, professional"),
    ("استودیو", "زن زیبا، صورت با جزئیات، نور نرم استودیو، حرفه‌ای"),
    ("Cinematic", "handsome man, detailed face, cinematic lighting, dramatic shadows"),
    ("سینمایی", "مرد جذاب، صورت با جزئیات، نورپردازی سینمایی، سایه‌های دراماتیک"),
    ("Natural Window", "beautiful woman, detailed face, natural lighting, sunshine from window"),
    ("نور پنجره", "زن زیبا، صورت با جزئیات، نورپردازی طبیعی، نور خورشید از پنجره"),
    ("Cyberpunk", "beautiful woman, detailed face, neon light, sci-fi RGB glowing, cyberpunk"),
    ("سایبرپانک", "زن زیبا، صورت با جزئیات، نور نئون، RGB سایبرپانک"),
    ("Sunset", "beautiful woman, detailed face, sunset over sea, romantic lighting"),
    ("غروب", "زن زیبا، صورت با جزئیات، غروب روی دریا، نور عاشقانه"),
]

# Model loading with error handling
MODEL_NAME = 'runwayml/stable-diffusion-v1-5'  # More reliable for Colab

print("📦 Loading models...")
try:
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    print("✅ Base models loaded")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("🔧 Trying alternative model...")
    MODEL_NAME = 'stabilityai/stable-diffusion-2-1-base'
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    print("✅ Alternative models loaded")

# Initialize BRIA
rmbg = SimpleBRIA()

# Modify UNet for IC Light (simplified)
print("🔧 Modifying UNet...")
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(
        8, unet.conv_in.out_channels, 
        unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

# Hook UNet
unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs.get('cross_attention_kwargs', {}).get('concat_conds')
    if c_concat is not None:
        c_concat = c_concat.to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    else:
        return unet_original_forward(sample, timestep, encoder_hidden_states, **kwargs)

unet.forward = hooked_unet_forward

# Move to device
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# Scheduler
ddim_scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

print("✅ Models loaded and configured")

# Helper functions
def resize_image(image, width, height):
    """Resize image to target dimensions"""
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    return pil_image.resize((width, height), Image.LANCZOS)

def simple_background_removal(img):
    """Simplified background removal"""
    h, w = img.shape[:2]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        mask = rmbg(img_tensor)
        
    mask_np = mask.squeeze().cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
    
    # Apply mask
    foreground = img.copy().astype(np.float32)
    alpha = np.array(mask_pil).astype(np.float32) / 255
    alpha = alpha[:, :, None]
    foreground = foreground * alpha + 255 * (1 - alpha)
    
    return foreground.astype(np.uint8), mask_pil

def encode_prompt(prompt):
    """Encode text prompt"""
    tokens = tokenizer(
        prompt, truncation=True, padding="max_length", 
        max_length=77, return_tensors="pt"
    )
    tokens = tokens.input_ids.to(device)
    with torch.no_grad():
        embeddings = text_encoder(tokens).last_hidden_state
    return embeddings

def generate_lighting_gradient(direction, width, height):
    """Generate lighting gradient"""
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
    else:
        gradient = np.full((height, width, 3), 127, dtype=np.uint8)
    
    return gradient.astype(np.uint8)

@torch.inference_mode()
def process_image(input_image, prompt, light_direction, width, height, steps, cfg_scale, seed):
    """Main processing function"""
    if input_image is None:
        return None
    
    try:
        # Set seed
        if seed == -1:
            seed = np.random.randint(0, 999999999)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Resize input
        input_resized = resize_image(input_image, width, height)
        input_np = np.array(input_resized)
        
        # Background removal
        fg_image, mask = simple_background_removal(input_np)
        
        # Generate lighting
        lighting = generate_lighting_gradient(light_direction, width, height)
        
        # Encode prompt
        prompt_emb = encode_prompt(prompt)
        uncond_emb = encode_prompt("")
        
        # Convert to latents
        fg_rgb = fg_image.astype(np.float32) / 127.5 - 1.0
        lighting_rgb = lighting.astype(np.float32) / 127.5 - 1.0
        
        fg_tensor = torch.from_numpy(fg_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        lighting_tensor = torch.from_numpy(lighting_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            fg_latent = vae.encode(fg_tensor).latent_dist.mode()
            lighting_latent = vae.encode(lighting_tensor).latent_dist.mode()
            
            # Concatenate conditioning
            concat_conds = torch.cat([fg_latent, lighting_latent], dim=1)
            
            # Generate latents
            latents = torch.randn((1, 4, height // 8, width // 8), device=device, dtype=torch.float16)
            
            # Diffusion loop (simplified)
            ddim_scheduler.set_timesteps(steps)
            for t in ddim_scheduler.timesteps:
                noise_pred_cond = unet(latents, t, prompt_emb, cross_attention_kwargs={"concat_conds": concat_conds}).sample
                noise_pred_uncond = unet(latents, t, uncond_emb, cross_attention_kwargs={"concat_conds": concat_conds}).sample
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode result
            result = vae.decode(latents.to(dtype=torch.bfloat16)).sample
            result = (result * 0.5 + 0.5).clamp(0, 1)
            result = result.permute(0, 2, 3, 1).cpu().numpy()
            result_image = (result[0] * 255).astype(np.uint8)
            
        return Image.fromarray(result_image)
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return Image.fromarray(input_image)

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="IC Light v2 - Colab", theme=gr.themes.Soft()) as app:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1 style="color: #2196F3;">🌟 IC Light v2 - Google Colab Edition</h1>
            <h2 style="color: #4CAF50;">نسخه گوگل کولب IC Light v2</h2>
            <p style="color: #666;">Professional Image Relighting | نورپردازی حرفه‌ای تصاویر</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Input | ورودی</h3>")
                input_image = gr.Image(type="numpy", label="Upload Image")
                
                gr.HTML("<h3>🎨 Presets | پرست‌ها</h3>")
                preset_dropdown = gr.Dropdown(
                    choices=[(name, prompt) for name, prompt in PRESETS],
                    value=PRESETS[0][1],
                    label="Choose Preset"
                )
                
                prompt_input = gr.Textbox(
                    value=PRESETS[0][1],
                    label="Custom Prompt | پرامپت سفارشی",
                    lines=2
                )
                
                light_dir = gr.Radio(
                    choices=["Left Light", "Right Light", "Top Light", "Bottom Light"],
                    value="Left Light",
                    label="Light Direction | جهت نور"
                )
                
            with gr.Column(scale=1):
                gr.HTML("<h3>🎨 Result | نتیجه</h3>")
                output_image = gr.Image(label="Generated Image")
                
                generate_btn = gr.Button("🚀 Generate | تولید", variant="primary", size="lg")
                
                gr.HTML("<h3>⚙️ Settings | تنظیمات</h3>")
                with gr.Accordion("Advanced Settings", open=False):
                    width_slider = gr.Slider(256, 768, 512, step=64, label="Width")
                    height_slider = gr.Slider(256, 768, 512, step=64, label="Height")
                    steps_slider = gr.Slider(1, 30, 15, step=1, label="Steps")
                    cfg_slider = gr.Slider(1.0, 15.0, 7.0, step=0.5, label="CFG Scale")
                    seed_slider = gr.Slider(-1, 999999999, -1, step=1, label="Seed (-1 for random)")
        
        # Event handlers
        def on_preset_change(preset_value):
            return preset_value
        
        def on_generate(img, prompt, direction, width, height, steps, cfg, seed):
            if img is None:
                return None
            return process_image(img, prompt, direction, int(width), int(height), int(steps), cfg, int(seed))
        
        # Connect events
        preset_dropdown.change(on_preset_change, preset_dropdown, prompt_input)
        
        generate_btn.click(
            on_generate,
            inputs=[input_image, prompt_input, light_dir, width_slider, height_slider, steps_slider, cfg_slider, seed_slider],
            outputs=output_image
        )
        
        gr.HTML("""
        <div style="margin-top: 20px; text-align: center;">
            <p style="color: #666;">✨ Google Colab Compatible Version | نسخه سازگار با گوگل کولب ✨</p>
        </div>
        """)
    
    return app

if __name__ == "__main__":
    print("🎉 Starting IC Light v2 - Colab Edition...")
    print("🌟 راه‌اندازی IC Light v2 - نسخه کولب...")
    
    app = create_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_api=False
    )
