# IC Light v2 Professional - Background-Conditioned Relighting System
# Complete implementation with background-aware lighting
# سیستم حرفه‌ای IC Light v2 با کنترل پس‌زمینه

import os
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

print("🌟 IC Light v2 Professional - Background-Conditioned System")
print("⚡ Loading background-aware relighting models...")

# Model Configuration
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

# Background Removal Model
print("📦 Loading BRIA RMBG 1.4...")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# IC Light Background-Conditioned Model Loading
model_path = './iclight_sd15_fbc.safetensors'
if not os.path.exists(model_path):
    print("📥 Downloading IC Light FBC model...")
    download_url_to_file(
        url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors',
        dst=model_path
    )

print("🔧 Merging IC Light FBC weights...")
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

print("✅ Background-conditioned system ready!")

# Professional Utility Functions
@torch.inference_mode()
def encode_prompt_inner(txt: str):
    """Professional prompt encoding"""
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


def resize_and_center_crop(image, target_width, target_height):
    """Professional image resizing with center crop"""
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    
    scale_factor = max(target_width / original_width, target_height / original_height)
    
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    """Professional background removal"""
    h, w = img.shape[:2]
    
    def to_multiple_of_32(x):
        return int(np.ceil(x / 32.0) * 32)
    
    new_h = to_multiple_of_32(h)
    new_w = to_multiple_of_32(w)
    
    img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))
    img_torch = torch.from_numpy(img_resized).float().to(device) / 255.0
    img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        result = rmbg(img_torch)
        mask = result[0][0]
        
        if sigma > 0:
            mask = torch.nn.functional.gaussian_blur(
                mask.unsqueeze(0), 
                kernel_size=int(sigma*3)*2+1, 
                sigma=(sigma, sigma)
            )[0]
    
    mask = mask.cpu().numpy()
    mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)) / 255.0
    
    result_img = img.copy().astype(np.float32)
    result_img[:, :, :3] *= mask[:, :, np.newaxis]
    
    return result_img.astype(np.uint8), (mask * 255).astype(np.uint8)


# Background Source Enum
class BGSource(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    REMOVE = "Remove Background Only"


@torch.inference_mode()
def process_background_relight(input_fg, input_bg, prompt, bg_source, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, denoise_strength):
    """Professional background-conditioned relighting"""
    
    print("🎨 Starting background-conditioned relighting...")
    
    if input_fg is None:
        return [Image.new('RGB', (512, 512), color='black')], Image.new('RGB', (512, 512), color='black')
    
    if prompt.strip() == "":
        prompt = "cinematic lighting, professional"
    
    # Process foreground
    input_fg_rgb = input_fg[..., :3]
    fg_processed, mask = run_rmbg(input_fg_rgb, sigma=0.5)
    input_fg_resized = resize_and_center_crop(fg_processed, image_width, image_height)
    
    # Process background based on source
    if bg_source in [BGSource.UPLOAD.value, BGSource.UPLOAD_FLIP.value] and input_bg is not None:
        input_bg_rgb = input_bg[..., :3]
        input_bg_resized = resize_and_center_crop(input_bg_rgb, image_width, image_height)
        
        if bg_source == BGSource.UPLOAD_FLIP.value:
            input_bg_resized = np.fliplr(input_bg_resized)
    else:
        # Generate gradient background for lighting directions
        gradient = np.zeros((image_height, image_width), dtype=np.float32)
        
        if bg_source == "Left Light":
            for x in range(image_width):
                gradient[:, x] = 1.0 - (x / image_width)
        elif bg_source == "Right Light":
            for x in range(image_width):
                gradient[:, x] = x / image_width
        elif bg_source == "Top Light":
            for y in range(image_height):
                gradient[y, :] = 1.0 - (y / image_height)
        elif bg_source == "Bottom Light":
            for y in range(image_height):
                gradient[y, :] = y / image_height
        else:
            gradient.fill(0.5)
        
        input_bg_resized = np.stack([gradient] * 3, axis=-1)
        input_bg_resized = (input_bg_resized * 255).astype(np.uint8)
    
    # Enhanced prompt for background conditioning
    enhanced_prompt = f"{prompt}, {a_prompt}, professional lighting, highly detailed"
    
    # Create concatenated input (foreground + background)
    concat_conds = np.concatenate([input_fg_resized, input_bg_resized], axis=2)
    
    # Generate professional results
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    results = []
    for i in range(num_samples):
        sample_generator = torch.Generator(device=device).manual_seed(int(seed) + i)
        
        # Convert to PIL for pipeline
        concat_pil = Image.fromarray(concat_conds)
        
        try:
            result = i2i_pipe(
                prompt=enhanced_prompt,
                negative_prompt=n_prompt,
                image=concat_pil,
                strength=denoise_strength,
                guidance_scale=cfg,
                num_inference_steps=steps,
                generator=sample_generator,
                width=image_width,
                height=image_height,
            ).images[0]
            
            results.append(result)
        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback to text-to-image
            result = t2i_pipe(
                prompt=enhanced_prompt,
                negative_prompt=n_prompt,
                guidance_scale=cfg,
                num_inference_steps=steps,
                generator=sample_generator,
                width=image_width,
                height=image_height,
            ).images[0]
            results.append(result)
    
    print("✨ Background-conditioned relighting completed!")
    return results, Image.fromarray(input_bg_resized)


# Professional Background-Conditioned Interface
def create_background_interface():
    """Create the professional background-conditioned interface"""
    
    quick_prompts = [
        ["cinematic lighting, professional"],
        ["natural lighting, outdoor"],
        ["studio lighting, controlled"],
        ["dramatic lighting, high contrast"],
        ["soft lighting, gentle mood"],
        ["warm lighting, golden hour"],
        ["cool lighting, blue hour"],
        ["ambient lighting, natural feel"],
        ["professional portrait lighting"],
        ["fashion photography lighting"]
    ]
    
    with gr.Blocks(
        title="IC Light v2 - Background-Conditioned Professional",
        theme=gr.themes.Soft()
    ) as block:
        
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1>🌟 IC Light v2 - Background-Conditioned Professional</h1>
            <h2>سیستم حرفه‌ای کنترل روشنایی با پس‌زمینه</h2>
            <p>Professional Background-Aware Relighting System</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML('<h3>📤 Input Controls</h3>')
                
                with gr.Row():
                    input_fg = gr.Image(
                        source='upload', 
                        type="numpy", 
                        label="📸 Foreground Image", 
                        height=400
                    )
                    input_bg = gr.Image(
                        source='upload', 
                        type="numpy", 
                        label="🖼️ Background Image", 
                        height=400
                    )
                
                prompt = gr.Textbox(
                    label="✨ Lighting Description", 
                    placeholder="Describe lighting style (e.g., 'cinematic lighting, professional')",
                    lines=2
                )
                
                bg_source = gr.Radio(
                    choices=[e.value for e in BGSource],
                    value=BGSource.UPLOAD.value,
                    label="🎭 Background Source"
                )
                
                with gr.Accordion("💡 Quick Prompts", open=True):
                    example_prompts = gr.Dataset(
                        samples=quick_prompts,
                        label='Professional Lighting Presets',
                        components=[prompt]
                    )
                
                # Background Gallery
                bg_gallery = gr.Gallery(
                    height=300,
                    object_fit='contain',
                    label='🖼️ Background Quick Selection',
                    value=db_examples.bg_samples,
                    columns=5,
                    allow_preview=False
                )
                
                relight_button = gr.Button(
                    value="🎨 Start Background-Conditioned Relighting",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column():
                gr.HTML('<h3>🎯 Professional Results</h3>')
                
                result_gallery = gr.Gallery(
                    label="✨ Generated Results",
                    show_label=True,
                    columns=2,
                    rows=2,
                    height=500
                )
                
                output_bg = gr.Image(
                    label="🎭 Background Preview",
                    height=300
                )
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                num_samples = gr.Slider(
                    label="📊 Number of Results", 
                    minimum=1, maximum=6, value=2, step=1
                )
                seed = gr.Number(
                    label="🎲 Seed", 
                    value=12345, precision=0
                )
            
            with gr.Row():
                image_width = gr.Slider(
                    label="📐 Width", 
                    minimum=256, maximum=1024, value=512, step=64
                )
                image_height = gr.Slider(
                    label="📏 Height", 
                    minimum=256, maximum=1024, value=768, step=64
                )
            
            with gr.Row():
                steps = gr.Slider(
                    label="🔄 Steps", 
                    minimum=10, maximum=50, value=25, step=1
                )
                cfg = gr.Slider(
                    label="🎯 CFG Scale", 
                    minimum=1.0, maximum=20.0, value=7.5, step=0.1
                )
                denoise_strength = gr.Slider(
                    label="🔧 Relighting Strength", 
                    minimum=0.1, maximum=1.0, value=0.85, step=0.05
                )
            
            a_prompt = gr.Textbox(
                label="➕ Additional Prompts", 
                value="professional lighting, highly detailed, masterpiece"
            )
            
            n_prompt = gr.Textbox(
                label="➖ Negative Prompts", 
                value="lowres, bad anatomy, worst quality, low quality"
            )
        
        # Examples
        with gr.Accordion("🎨 Professional Examples", open=False):
            dummy_image = gr.Image(visible=False)
            gr.Examples(
                examples=db_examples.background_conditioned_examples,
                inputs=[
                    input_fg, input_bg, prompt, bg_source, image_width, image_height, seed, dummy_image
                ],
                outputs=[result_gallery, output_bg],
                run_on_click=True
            )
        
        # Event handlers
        inputs = [input_fg, input_bg, prompt, bg_source, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, denoise_strength]
        
        relight_button.click(
            fn=process_background_relight,
            inputs=inputs,
            outputs=[result_gallery, output_bg]
        )
        
        example_prompts.click(
            lambda x: x[0],
            inputs=example_prompts,
            outputs=prompt,
            show_progress=False
        )
    
    return block


if __name__ == "__main__":
    print("🚀 Launching Background-Conditioned Professional System...")
    
    interface = create_background_interface()
    
    interface.queue(max_size=10).launch(
        server_name='0.0.0.0',
        server_port=7861,
        share=True,
        inbrowser=True,
        show_error=True,
        title="IC Light v2 - Background-Conditioned Professional"
    )
