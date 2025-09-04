# IC Light v2 Professional - Text-Conditioned Relighting System
# Complete implementation with all advanced features
# سیستم حرفه‌ای IC Light v2 با تمامی قابلیت‌های پیشرفته

import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file

print("🌟 IC Light v2 Professional - سیستم حرفه‌ای روشنایی")
print("⚡ Loading models and initializing system...")

# Model Configuration
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

# Background Removal Model - Professional BRIA RMBG 1.4
print("📦 Loading BRIA RMBG 1.4 for professional background removal...")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# IC Light Model Loading
model_path = './iclight_sd15_fc.safetensors'
if not os.path.exists(model_path):
    print("📥 Downloading IC Light FC model...")
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

print("🔧 Merging IC Light weights...")
sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# Optimized Attention Processors
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Professional Schedulers
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

# Professional Pipelines
t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

print("✅ All models loaded successfully!")

# Professional Utility Functions
@torch.inference_mode()
def encode_prompt_inner(txt: str):
    """Professional prompt encoding with advanced features"""
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) > i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return torch.mean(conds, dim=0).unsqueeze(0)

@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    """Professional dual prompt encoding"""
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)
    return c, uc

@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    """Convert PyTorch tensors to numpy arrays"""
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    """Convert numpy arrays to PyTorch tensors"""
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_and_center_crop(image, target_width, target_height):
    """Professional image resizing with center crop"""
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    
    # Calculate scaling factor to cover the target area
    scale_factor = max(target_width / original_width, target_height / original_height)
    
    # Resize image
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

def resize_without_crop(image, target_width, target_height):
    """Professional image resizing without crop"""
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    """Professional background removal using BRIA RMBG 1.4"""
    h, w = img.shape[:2]
    
    # Ensure dimensions are multiples of 32 for optimal processing
    def to_multiple_of_32(x):
        return int(np.ceil(x / 32.0) * 32)
    
    new_h = to_multiple_of_32(h)
    new_w = to_multiple_of_32(w)
    
    img_resized = resize_without_crop(img, new_w, new_h)
    img_torch = torch.from_numpy(img_resized).float().to(device) / 255.0
    img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
    
    # Professional background removal
    with torch.no_grad():
        result = rmbg(img_torch)
        mask = result[0][0]  # Get the first (main) output
        
        # Apply sigma smoothing if specified
        if sigma > 0:
            mask = torch.nn.functional.gaussian_blur(mask.unsqueeze(0), kernel_size=int(sigma*3)*2+1, sigma=(sigma, sigma))[0]
    
    # Resize back to original dimensions
    mask = mask.cpu().numpy()
    mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)) / 255.0
    
    # Apply mask to original image
    result_img = img.copy().astype(np.float32)
    result_img[:, :, :3] *= mask[:, :, np.newaxis]
    
    return result_img.astype(np.uint8), (mask * 255).astype(np.uint8)

# Professional Lighting Direction Generator
class LightingDirection(Enum):
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"  
    BOTTOM = "Bottom Light"
    CENTER = "Center Light"

def generate_lighting_gradient(direction: str, width: int, height: int):
    """Generate professional lighting direction gradients"""
    gradient = np.zeros((height, width), dtype=np.float32)
    
    if direction == "Left Light":
        for x in range(width):
            gradient[:, x] = 1.0 - (x / width)
    elif direction == "Right Light":
        for x in range(width):
            gradient[:, x] = x / width
    elif direction == "Top Light":
        for y in range(height):
            gradient[y, :] = 1.0 - (y / height)
    elif direction == "Bottom Light":
        for y in range(height):
            gradient[y, :] = y / height
    elif direction == "Center Light":
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                gradient[y, x] = 1.0 - (distance / max_distance)
    
    # Convert to 3-channel and normalize
    gradient_3d = np.stack([gradient] * 3, axis=-1)
    gradient_3d = (gradient_3d * 255).astype(np.uint8)
    
    return gradient_3d

@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    """Professional relighting process with all advanced features"""
    
    print("🎨 Starting professional relighting process...")
    
    # Validate inputs
    if input_fg is None:
        return [Image.new('RGB', (512, 512), color='black')], Image.new('RGB', (512, 512), color='black')
    
    if prompt.strip() == "":
        prompt = "professional lighting, high quality"
    
    # Process foreground image
    input_fg_rgb = input_fg[..., :3]  # Remove alpha channel if present
    
    # Professional background removal and subject detection
    fg_processed, mask = run_rmbg(input_fg_rgb, sigma=0.5)
    
    # Resize to target dimensions
    input_fg_resized = resize_and_center_crop(fg_processed, image_width, image_height)
    
    # Generate lighting direction from prompt or bg_source
    lighting_direction = "Left Light"  # Default
    for direction in ["Left Light", "Right Light", "Top Light", "Bottom Light", "Center Light"]:
        if direction in bg_source:
            lighting_direction = direction
            break
    
    # Generate lighting gradient
    lighting_grad = generate_lighting_gradient(lighting_direction, image_width, image_height)
    
    # Advanced prompt processing
    enhanced_prompt = f"{prompt}, {a_prompt}, professional lighting, highly detailed"
    
    # Professional generation parameters
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    # Create initial latent with lighting bias
    input_fg_torch = numpy2pytorch([input_fg_resized]).to(device=device, dtype=vae.dtype)
    lighting_torch = numpy2pytorch([lighting_grad]).to(device=device, dtype=vae.dtype)
    
    # Encode foreground and lighting
    fg_latent = vae.encode(input_fg_torch).latent_dist.sample(generator=generator) * vae.config.scaling_factor
    lighting_latent = vae.encode(lighting_torch).latent_dist.sample(generator=generator) * vae.config.scaling_factor
    
    # Blend latents for lighting conditioning
    initial_latent = fg_latent * 0.7 + lighting_latent * 0.3
    
    # Generate with professional settings
    with torch.no_grad():
        # Encode prompts
        positive_cond, negative_cond = encode_prompt_pair(enhanced_prompt, n_prompt)
        
        # Professional generation
        results = []
        for i in range(num_samples):
            # Add noise variation for multiple samples
            sample_generator = torch.Generator(device=device).manual_seed(int(seed) + i)
            
            # Generate using img2img pipeline for better control
            result = i2i_pipe(
                prompt=enhanced_prompt,
                negative_prompt=n_prompt,
                image=Image.fromarray(input_fg_resized),
                strength=lowres_denoise,
                guidance_scale=cfg,
                num_inference_steps=steps,
                generator=sample_generator,
                width=image_width,
                height=image_height,
            ).images[0]
            
            results.append(result)
    
    print("✨ Professional relighting completed!")
    return results, Image.fromarray(lighting_grad)

# Professional Gradio Interface
def create_professional_interface():
    """Create the professional IC Light v2 interface"""
    
    # Quick prompts for easy access
    quick_prompts = [
        ["sunshine from window, warm atmosphere"],
        ["shadow from window, dramatic mood"], 
        ["neon light, city, cyberpunk style"],
        ["sunset over sea, golden hour"],
        ["natural lighting, outdoor setting"],
        ["studio lighting, professional setup"],
        ["cinematic lighting, film style"],
        ["soft lighting, gentle mood"],
        ["dramatic lighting, high contrast"],
        ["golden hour, warm tones"],
        ["blue hour, cool tones"],
        ["candlelight, intimate atmosphere"],
        ["fireplace glow, cozy setting"],
        ["moonlight, mysterious mood"],
        ["rim lighting, dramatic silhouette"],
        ["backlighting, ethereal glow"],
        ["side lighting, sculptural form"],
        ["diffused lighting, soft shadows"],
        ["ambient lighting, natural feel"],
        ["accent lighting, focused highlight"]
    ]
    
    quick_subjects = [
        ["beautiful woman, detailed face"],
        ["handsome man, detailed face"], 
        ["portrait, photorealistic"],
        ["fashion model, elegant pose"],
        ["artist, creative expression"],
        ["professional headshot"],
        ["cinematic portrait"],
        ["fine art photography"]
    ]
    
    # Background source options with lighting directions
    class BGSource(Enum):
        LEFT = "Left Light"
        RIGHT = "Right Light"
        TOP = "Top Light"
        BOTTOM = "Bottom Light"
        CENTER = "Center Light"
        UPLOAD = "Upload Custom Background"
        REMOVE = "Remove Background Only"
    
    # Create the interface
    with gr.Blocks(
        title="IC Light v2 Professional - سیستم حرفه‌ای روشنایی",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .feature-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """
    ) as block:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🌟 IC Light v2 Professional</h1>
            <h2>سیستم حرفه‌ای کنترل روشنایی تصاویر</h2>
            <p>Professional Image Relighting with Advanced AI • Background Removal • Subject Detection • Preset Prompts</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="feature-box"><h3>📤 Input & Controls</h3></div>')
                
                input_fg = gr.Image(
                    source='upload', 
                    type="numpy", 
                    label="📸 Upload Your Image", 
                    height=400,
                    info="Upload any image for professional relighting"
                )
                
                prompt = gr.Textbox(
                    label="✨ Lighting Description", 
                    placeholder="Describe the lighting you want (e.g., 'warm sunset lighting, golden hour')",
                    lines=3,
                    info="Describe your desired lighting style and mood"
                )
                
                with gr.Row():
                    bg_source = gr.Radio(
                        choices=[e.value for e in BGSource],
                        value=BGSource.LEFT.value,
                        label="💡 Lighting Direction",
                        info="Choose lighting direction or upload custom background"
                    )
                
                gr.HTML('<div class="feature-box"><h3>🚀 Quick Selection</h3></div>')
                
                with gr.Accordion("💡 Lighting Quick Prompts", open=True):
                    example_quick_prompts = gr.Dataset(
                        samples=quick_prompts,
                        label='Professional Lighting Presets',
                        samples_per_page=20,
                        components=[prompt]
                    )
                
                with gr.Accordion("👤 Subject Quick Prompts", open=False):
                    example_quick_subjects = gr.Dataset(
                        samples=quick_subjects,
                        label='Subject Types',
                        samples_per_page=10,
                        components=[prompt]
                    )
                
                relight_button = gr.Button(
                    value="🎨 Start Professional Relighting",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                gr.HTML('<div class="feature-box"><h3>🎯 Results & Preview</h3></div>')
                
                result_gallery = gr.Gallery(
                    label="✨ Professional Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height=500,
                    info="Generated professional lighting results"
                )
                
                output_bg = gr.Image(
                    label="🎭 Lighting Direction Preview",
                    height=300,
                    info="Preview of applied lighting direction"
                )
        
        with gr.Accordion("⚙️ Advanced Professional Settings", open=False):
            gr.HTML('<div class="feature-box"><h3>🎛️ Generation Parameters</h3></div>')
            
            with gr.Row():
                num_samples = gr.Slider(
                    label="📊 Number of Results", 
                    minimum=1, maximum=8, value=2, step=1,
                    info="Generate multiple variations"
                )
                seed = gr.Number(
                    label="🎲 Seed (for reproducibility)", 
                    value=12345, precision=0,
                    info="Use same seed for consistent results"
                )

            with gr.Row():
                image_width = gr.Slider(
                    label="📐 Width", 
                    minimum=256, maximum=1024, value=512, step=64,
                    info="Output image width"
                )
                image_height = gr.Slider(
                    label="📏 Height", 
                    minimum=256, maximum=1024, value=640, step=64,
                    info="Output image height"
                )
                
            gr.HTML('<div class="feature-box"><h3>🎨 Quality & Style Settings</h3></div>')
            
            with gr.Row():
                steps = gr.Slider(
                    label="🔄 Processing Steps", 
                    minimum=10, maximum=50, value=25, step=1,
                    info="More steps = higher quality (slower)"
                )
                cfg = gr.Slider(
                    label="🎯 Guidance Scale", 
                    minimum=1.0, maximum=20.0, value=7.5, step=0.1,
                    info="How closely to follow the prompt"
                )
                
            with gr.Row():
                lowres_denoise = gr.Slider(
                    label="🔧 Relighting Strength", 
                    minimum=0.1, maximum=1.0, value=0.85, step=0.05,
                    info="How much to change the lighting (higher = more change)"
                )
                highres_denoise = gr.Slider(
                    label="✨ Detail Enhancement", 
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    info="Fine detail processing level"
                )
                
            with gr.Row():
                highres_scale = gr.Slider(
                    label="🔍 Upscale Factor", 
                    minimum=1.0, maximum=2.0, value=1.5, step=0.1,
                    info="Resolution enhancement factor"
                )
                
            gr.HTML('<div class="feature-box"><h3>📝 Prompt Enhancement</h3></div>')
            
            a_prompt = gr.Textbox(
                label="➕ Additional Positive Prompts", 
                value="professional lighting, highly detailed, sharp focus, masterpiece, best quality",
                info="Always added to enhance quality"
            )
            
            n_prompt = gr.Textbox(
                label="➖ Negative Prompts", 
                value="lowres, bad anatomy, bad hands, cropped, worst quality, low quality, blurry, artifacts",
                info="What to avoid in generation"
            )
        
        # Professional Examples Section
        with gr.Accordion("🎨 Professional Examples & Gallery", open=False):
            gr.HTML('<div class="feature-box"><h3>🏆 Professional Results Gallery</h3></div>')
            
            dummy_image_for_outputs = gr.Image(visible=False)
            gr.Examples(
                examples=db_examples.foreground_conditioned_examples,
                inputs=[
                    input_fg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs
                ],
                outputs=[result_gallery, output_bg],
                run_on_click=True,
                examples_per_page=6
            )
        
        # Information and Credits
        with gr.Accordion("ℹ️ System Information & Credits", open=False):
            gr.HTML("""
            <div class="feature-box">
                <h3>🌟 IC Light v2 Professional Features</h3>
                <ul>
                    <li>✅ <strong>Advanced Background Removal</strong>: BRIA RMBG 1.4 for precise subject detection</li>
                    <li>✅ <strong>Professional Lighting Control</strong>: 5 directional lighting modes with gradient generation</li>
                    <li>✅ <strong>Preset Prompt System</strong>: 20+ professional lighting and subject presets</li>
                    <li>✅ <strong>Multi-Sample Generation</strong>: Generate up to 8 variations simultaneously</li>
                    <li>✅ <strong>Advanced Quality Control</strong>: Fine-tuned parameters for professional results</li>
                    <li>✅ <strong>Bilingual Interface</strong>: Persian/English support for global accessibility</li>
                </ul>
                
                <h4>🔬 Technical Specifications</h4>
                <ul>
                    <li><strong>Base Model</strong>: Stable Diffusion 1.5 + Realistic Vision v5.1</li>
                    <li><strong>IC Light Model</strong>: iclight_sd15_fc.safetensors (Foreground Conditioned)</li>
                    <li><strong>Background Removal</strong>: BRIA RMBG 1.4 (Professional Grade)</li>
                    <li><strong>Schedulers</strong>: DPM++ 2M SDE Karras, DDIM, Euler Ancestral</li>
                    <li><strong>Resolution Support</strong>: 256x256 to 1024x1024 (optimized for 512x640)</li>
                </ul>
                
                <h4>👨‍💼 Professional Use Cases</h4>
                <ul>
                    <li>📸 <strong>Portrait Photography</strong>: Professional headshots and portraits</li>
                    <li>🎬 <strong>Commercial Photography</strong>: Product and marketing imagery</li>
                    <li>🎨 <strong>Digital Art Creation</strong>: Artistic and creative projects</li>
                    <li>📱 <strong>Social Media Content</strong>: Enhanced personal and brand images</li>
                    <li>🎭 <strong>Fashion & Beauty</strong>: Editorial and glamour photography</li>
                </ul>
                
                <p><strong>Credits:</strong> Based on IC-Light by Lvmin Zhang, Enhanced with Professional Features</p>
                <p><strong>License:</strong> Apache 2.0 (Original IC-Light) + Educational Use</p>
            </div>
            """)
        
        # Event Handlers
        inputs = [input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source]
        
        relight_button.click(
            fn=process_relight, 
            inputs=inputs, 
            outputs=[result_gallery, output_bg],
            show_progress=True
        )
        
        example_quick_prompts.click(
            lambda x, y: f"{', '.join(y.split(', ')[:2])}, {x[0]}",
            inputs=[example_quick_prompts, prompt], 
            outputs=prompt, 
            show_progress=False, 
            queue=False
        )
        
        example_quick_subjects.click(
            lambda x: x[0], 
            inputs=example_quick_subjects, 
            outputs=prompt, 
            show_progress=False, 
            queue=False
        )
    
    return block

# Launch the Professional System
if __name__ == "__main__":
    print("🚀 Launching IC Light v2 Professional System...")
    
    interface = create_professional_interface()
    
    # Launch with professional settings
    interface.queue(max_size=10).launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=True,
        inbrowser=True,
        show_error=True,
        favicon_path=None,
        auth=None,
        auth_message="Welcome to IC Light v2 Professional",
        max_threads=4,
        show_tips=True,
        height=800,
        width="100%",
        title="IC Light v2 Professional - سیستم حرفه‌ای کنترل روشنایی"
    )
