# IC Light Technical Implementation Guide

## Architecture Overview

IC Light is a sophisticated image relighting system built on Stable Diffusion 1.5 architecture with custom modifications for consistent light transport. The system implements the principle of "Imposing Consistent Light" through mathematically consistent HDR-based light transport.

## Core Components

### 1. Base Architecture
```python
# Core Pipeline Components:
- Stable Diffusion 1.5 (stablediffusionapi/realistic-vision-v51)
- Modified UNet2DConditionModel (8-channel input instead of 4)
- AutoencoderKL (VAE) for latent space encoding/decoding
- CLIPTextModel for text conditioning
- BriaRMBG 1.4 for background removal
```

### 2. Model Variants

#### ic_light_sd15_fc.safetensors (Foreground Conditioned)
- **Input channels**: 8 (4 original + 4 foreground condition)
- **Conditioning**: Text + Foreground mask
- **Usage**: Default relighting model
- **Performance**: Optimal for single-subject portraits

#### ic_light_sd15_fbc.safetensors (Foreground + Background Conditioned)
- **Input channels**: 12 (4 original + 4 foreground + 4 background)
- **Conditioning**: Text + Foreground + Background
- **Usage**: Complex scene relighting
- **Performance**: Better context understanding

#### ic_light_sd15_fcon.safetensors (With Offset Noise)
- **Training**: Include offset noise during training
- **Performance**: Slightly lower than fc model
- **Usage**: Alternative for specific use cases

### 3. Background Removal System (BriaRMBG)

```python
class BriaRMBG Architecture:
├── Encoder Path
│   ├── RSU7 (7-layer residual U-block)
│   ├── RSU6 (6-layer residual U-block) 
│   ├── RSU5 (5-layer residual U-block)
│   ├── RSU4 (4-layer residual U-block)
│   └── RSU4F (4-layer residual U-block with dilation)
├── Decoder Path
│   ├── RSU4F → RSU4 → RSU5 → RSU6 → RSU7
│   └── Skip connections from encoder
└── Side Outputs
    └── 6 side prediction heads for multi-scale supervision
```

## Mathematical Foundation

### Consistent Light Transport Theory

IC Light implements the mathematical principle:
```
In HDR space: blend(appearances) = appearance(blend(lights))
```

This means:
1. **Appearance Mixture**: Blending different light source appearances
2. **Light Source Mixture**: Mixing light sources then rendering
3. **Mathematical Equivalence**: Both approaches yield identical results in HDR

### Implementation in Latent Space
```python
def impose_light_consistency(latent_features, light_conditions):
    """
    Enforce consistency constraint in latent space
    using MLPs to map between light conditions
    """
    # Multi-scale feature processing
    features = []
    for scale in range(num_scales):
        feat = extract_features(latent_features, scale)
        light_feat = condition_on_light(feat, light_conditions[scale])
        features.append(light_feat)
    
    # Consistency enforcement
    consistent_features = enforce_consistency(features)
    return merge_multi_scale(consistent_features)
```

## Technical Implementation Details

### 1. Image Processing Pipeline

```python
def process_image_pipeline(input_image, prompt, lighting_direction):
    """Complete IC Light processing pipeline"""
    
    # Step 1: Background Removal
    foreground, alpha_mask = remove_background(input_image)
    
    # Step 2: Prepare Conditioning
    text_embedding = encode_text(prompt)
    light_condition = generate_light_condition(lighting_direction)
    
    # Step 3: Latent Encoding
    latent_fg = vae.encode(foreground).latent_dist.mode()
    latent_light = vae.encode(light_condition).latent_dist.mode()
    
    # Step 4: Concatenate Conditions
    concat_conditions = torch.cat([latent_fg, latent_light], dim=1)
    
    # Step 5: Diffusion Process
    result_latent = diffusion_pipeline(
        prompt_embeds=text_embedding,
        cross_attention_kwargs={'concat_conds': concat_conditions}
    )
    
    # Step 6: Decode Result
    result_image = vae.decode(result_latent).sample
    
    return postprocess(result_image)
```

### 2. Light Direction Implementation

```python
class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light" 
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

def generate_initial_latent(bg_source, width, height):
    """Generate initial latent based on lighting direction"""
    
    if bg_source == BGSource.LEFT:
        # White to black gradient (left to right)
        gradient = np.linspace(255, 0, width)
        image = np.tile(gradient, (height, 1))
        
    elif bg_source == BGSource.RIGHT:
        # Black to white gradient (left to right)
        gradient = np.linspace(0, 255, width)
        image = np.tile(gradient, (height, 1))
        
    elif bg_source == BGSource.TOP:
        # White to black gradient (top to bottom)
        gradient = np.linspace(255, 0, height)[:, None]
        image = np.tile(gradient, (1, width))
        
    elif bg_source == BGSource.BOTTOM:
        # Black to white gradient (top to bottom)
        gradient = np.linspace(0, 255, height)[:, None]
        image = np.tile(gradient, (1, width))
        
    else:  # NONE
        # Neutral gray
        image = np.full((height, width), 127)
    
    # Convert to 3-channel RGB
    return np.stack((image,) * 3, axis=-1).astype(np.uint8)
```

### 3. UNet Modification

```python
def modify_unet_for_ic_light(unet, input_channels=8):
    """Modify UNet input layer for IC Light conditioning"""
    
    with torch.no_grad():
        # Create new input convolution layer
        old_conv = unet.conv_in
        new_conv = torch.nn.Conv2d(
            input_channels,  # 8 for fc, 12 for fbc
            old_conv.out_channels,
            old_conv.kernel_size,
            old_conv.stride,
            old_conv.padding
        )
        
        # Initialize weights
        new_conv.weight.zero_()
        new_conv.weight[:, :4, :, :].copy_(old_conv.weight)  # Copy original weights
        new_conv.bias = old_conv.bias
        
        # Replace layer
        unet.conv_in = new_conv
    
    return unet
```

### 4. Multi-Resolution Processing

```python
def multi_resolution_processing(image, prompt, settings):
    """Process image at multiple resolutions for better quality"""
    
    # Low resolution pass
    low_res_result = process_at_resolution(
        image, prompt, 
        width=settings.base_width,
        height=settings.base_height,
        strength=settings.lowres_denoise
    )
    
    # High resolution refinement
    if settings.highres_scale > 1.0:
        target_width = int(settings.base_width * settings.highres_scale)
        target_height = int(settings.base_height * settings.highres_scale)
        
        # Upscale low-res result
        upscaled = resize_image(low_res_result, target_width, target_height)
        
        # Refine at high resolution
        high_res_result = process_at_resolution(
            upscaled, prompt,
            width=target_width,
            height=target_height, 
            strength=settings.highres_denoise
        )
        
        return high_res_result
    
    return low_res_result
```

## Performance Optimizations

### 1. Memory Management
```python
# GPU Memory Optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use attention slicing for large images
unet.set_attention_slice("auto")

# Enable memory efficient attention
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())
```

### 2. Mixed Precision Training
```python
# Use appropriate dtypes for each component
text_encoder = text_encoder.to(dtype=torch.float16)
vae = vae.to(dtype=torch.bfloat16)
unet = unet.to(dtype=torch.float16)
rmbg = rmbg.to(dtype=torch.float32)  # Segmentation needs fp32
```

### 3. Batch Processing
```python
def batch_process_images(images, prompts, batch_size=4):
    """Process multiple images efficiently"""
    
    results = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        
        # Process batch
        batch_results = process_batch(batch_images, batch_prompts)
        results.extend(batch_results)
        
        # Clear cache between batches
        torch.cuda.empty_cache()
    
    return results
```

## Quality Control Parameters

### 1. Diffusion Parameters
```python
DEFAULT_SETTINGS = {
    'num_inference_steps': 25,    # Balance quality/speed
    'guidance_scale': 2.0,        # Lower than typical SD (7.5)
    'eta': 0.0,                   # DDIM deterministic
    'scheduler': 'DDIM',          # Recommended scheduler
}
```

### 2. Resolution Guidelines
```python
SUPPORTED_RESOLUTIONS = [
    256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024
]

RECOMMENDED_RATIOS = [
    (512, 512),   # Square - optimal for portraits
    (512, 768),   # Portrait orientation  
    (768, 512),   # Landscape orientation
    (640, 640),   # Square - good balance
]
```

### 3. Prompt Engineering
```python
def optimize_prompt(base_prompt, lighting_style):
    """Optimize prompt for better results"""
    
    # Add quality tokens
    quality_tokens = "best quality, detailed, high resolution"
    
    # Add negative prompt
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    
    # Combine elements
    full_prompt = f"{base_prompt}, {lighting_style}, {quality_tokens}"
    
    return full_prompt, negative_prompt
```

## Model Loading and Caching

```python
class ICLightModelManager:
    """Efficient model loading and caching"""
    
    def __init__(self, cache_dir="./models"):
        self.cache_dir = cache_dir
        self.loaded_models = {}
    
    def load_model(self, model_type="fc"):
        """Load specific IC Light model with caching"""
        
        if model_type in self.loaded_models:
            return self.loaded_models[model_type]
        
        model_paths = {
            "fc": "iclight_sd15_fc.safetensors",
            "fbc": "iclight_sd15_fbc.safetensors", 
            "fcon": "iclight_sd15_fcon.safetensors"
        }
        
        model_path = os.path.join(self.cache_dir, model_paths[model_type])
        
        # Download if not exists
        if not os.path.exists(model_path):
            self.download_model(model_type, model_path)
        
        # Load model
        model = self.setup_pipeline(model_path, model_type)
        self.loaded_models[model_type] = model
        
        return model
    
    def download_model(self, model_type, local_path):
        """Download model from Hugging Face"""
        from torch.hub import download_url_to_file
        
        urls = {
            "fc": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
            "fbc": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors",
            "fcon": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fcon.safetensors"
        }
        
        print(f"Downloading {model_type} model...")
        download_url_to_file(urls[model_type], local_path)
        print(f"✅ Model downloaded: {local_path}")
```

## Error Handling and Validation

```python
def validate_input_image(image):
    """Validate input image meets requirements"""
    
    if image is None:
        raise ValueError("No image provided")
    
    # Check format
    if not isinstance(image, (np.ndarray, Image.Image)):
        raise TypeError("Image must be numpy array or PIL Image")
    
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Check dimensions
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB (H, W, 3)")
    
    # Check size limits
    h, w = image.shape[:2]
    if h < 256 or w < 256:
        raise ValueError("Image too small (minimum 256x256)")
    if h > 1024 or w > 1024:
        print("Warning: Large images may cause memory issues")
    
    return image

def safe_process_image(image, prompt, **kwargs):
    """Process image with error handling"""
    
    try:
        # Validate inputs
        image = validate_input_image(image)
        
        # Process with timeout
        with timeout(300):  # 5 minute timeout
            result = process_image_pipeline(image, prompt, **kwargs)
        
        return result
        
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU out of memory. Try reducing image size.")
        torch.cuda.empty_cache()
        return None
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return None
```

This technical guide provides the complete implementation details for IC Light, covering the mathematical foundation, architectural components, and practical implementation considerations for building the system in Google Colab or other environments.
