# تحلیل کامل اپلیکیشن IC Light و راهنمای ساخت در Google Colab

## معرفی IC Light

IC Light (Imposing Consistent Light) یک ابزار پیشرفته AI برای تغییر نورپردازی تصاویر است که توسط Lvmin Zhang (سازنده ControlNet) توسعه داده شده است. این ابزار قادر به تغییر نورپردازی پرتره‌ها و تصاویر با استفاده از دو روش اصلی است:

### ویژگی‌های اصلی:
1. **Text-Conditioned Relighting**: کنترل نور با استفاده از متن توضیحی
2. **Background-Conditioned Relighting**: کنترل نور با استفاده از تصویر پس‌زمینه مرجع
3. **قابلیت تغییر جهت نور**: چپ، راست، بالا، پایین
4. **کیفیت بالا**: پشتیبانی از اندازه‌های مختلف تصویر
5. **سرعت بالا**: تولید تصویر در 10-20 ثانیه

## معماری سیستم

### ساختار کلی:
```
IC Light System
├── Frontend (iclightai.com)
│   ├── Image Upload Interface
│   ├── Light Direction Control
│   ├── Text Prompt Input
│   └── Result Display
├── Backend Processing
│   ├── Background Removal (BRIA RMBG 1.4)
│   ├── Stable Diffusion Pipeline
│   ├── IC Light Models
│   └── Post-processing
└── Models
    ├── iclight_sd15_fc.safetensors (Foreground Conditioned)
    ├── iclight_sd15_fbc.safetensors (Foreground + Background)
    └── iclight_sd15_fcon.safetensors (With Offset Noise)
```

### مدل‌های موجود:

#### 1. iclight_sd15_fc.safetensors
- **کاربری**: مدل پیش‌فرض برای Relighting
- **شرایط**: متن + Foreground
- **ویژگی**: قابلیت کنترل با Initial Latent

#### 2. iclight_sd15_fbc.safetensors  
- **کاربری**: Relighting با شرایط پیچیده‌تر
- **شرایط**: متن + Foreground + Background
- **مزایا**: کنترل دقیق‌تر نورپردازی

#### 3. iclight_sd15_fcon.safetensors
- **کاربری**: مدل آموزش دیده با Offset Noise
- **عملکرد**: کمی ضعیف‌تر از مدل اصلی

## جزئیات فنی

### Dependencies اصلی:
```
diffusers==0.27.2
transformers==4.36.2
opencv-python
safetensors
pillow==10.2.0
einops
torch
peft
gradio==3.41.2
protobuf==3.20
```

### معماری شبکه عصبی:

#### Background Removal - BriaRMBG:
```python
class BriaRMBG(nn.Module, PyTorchModelHubMixin):
    - RSU7, RSU6, RSU5, RSU4 modules
    - Multi-scale feature extraction
    - Encoder-decoder architecture
    - Side outputs for better segmentation
```

#### Stable Diffusion Pipeline:
```python
# Core Components:
- UNet2DConditionModel (modified input channels)
- AutoencoderKL (VAE)
- CLIPTextModel (text encoder)
- DDIMScheduler/EulerAncestralDiscreteScheduler
```

### فرآیند پردازش:

1. **Input Processing**:
   ```python
   input_fg = load_image(user_upload)
   input_fg, matting = run_rmbg(input_fg)  # Background removal
   ```

2. **Light Direction Setup**:
   ```python
   # تولید Initial Latent بر اساس جهت نور
   if bg_source == BGSource.LEFT:
       gradient = np.linspace(255, 0, image_width)
   elif bg_source == BGSource.RIGHT:
       gradient = np.linspace(0, 255, image_width)
   # ... similar for TOP, BOTTOM
   ```

3. **Diffusion Process**:
   ```python
   latents = t2i_pipe(
       prompt_embeds=conds,
       negative_prompt_embeds=unconds,
       width=image_width,
       height=image_height,
       cross_attention_kwargs={'concat_conds': concat_conds}
   )
   ```

## نحوه پیاده‌سازی در Google Colab

### 1. نوت‌بوک اصلی (Text-Conditioned):

```python
# سلول 1: نصب Dependencies
%cd /content
!git clone -b dev https://github.com/camenduru/IC-Light
%cd /content/IC-Light
!pip install diffusers==0.27.2 gradio==3.50.2

# سلول 2: اجرای اپلیکیشن
!python gradio_demo.py
```

### 2. نوت‌بوک Background-Conditioned:

```python
# سلول 1: Setup
%cd /content
!git clone -b dev https://github.com/camenduru/IC-Light
%cd /content/IC-Light
!pip install diffusers==0.27.2 gradio==3.50.2

# سلول 2: اجرا
!python gradio_demo_bg.py
```

### 3. پیکربندی سخت‌افزاری:
- **GPU**: T4 یا بالاتر (حداقل 16GB VRAM)
- **RAM**: حداقل 12GB
- **Storage**: حداقل 10GB فضای خالی

## ساختار فایل‌های اصلی

### gradio_demo.py (Text-Conditioned Interface):
```python
# Core Functions:
- process(): پردازش اصلی تصویر
- run_rmbg(): حذف پس‌زمینه
- encode_prompt_pair(): انکود کردن پرامپت‌ها
- BGSource enum: تعریف جهات نور

# UI Components:
- input_fg: آپلود تصویر
- prompt: ورودی متن
- bg_source: انتخاب جهت نور
- Advanced settings: CFG, Steps, Denoise
```

### gradio_demo_bg.py (Background-Conditioned Interface):
```python
# Additional Features:
- input_bg: آپلود تصویر پس‌زمینه
- process_relight(): پردازش با پس‌زمینه
- process_normal(): پردازش معمولی
- Background gallery: گالری پس‌زمینه‌های پیش‌فرض
```

### briarmbg.py (Background Removal):
```python
# Key Classes:
- REBNCONV: Convolution block با Batch Norm
- RSU7, RSU6, RSU5, RSU4: U-Net residual blocks
- BriaRMBG: مدل اصلی حذف پس‌زمینه
```

## پارامترهای کنترلی

### تنظیمات اصلی:
- **Image Width/Height**: 256-1024 (مضرب 64)
- **Steps**: 1-100 (پیش‌فرض: 25)
- **CFG Scale**: 1.0-32.0 (پیش‌فرض: 2.0)
- **Seed**: عدد تصادفی برای reproducibility

### تنظیمات پیشرفته:
- **Highres Scale**: 1.0-3.0 (پیش‌فرض: 1.5)
- **Highres Denoise**: 0.1-1.0 (پیش‌فرض: 0.5)
- **Lowres Denoise**: 0.1-1.0 (پیش‌فرض: 0.9)

## پرامپت‌های پیشنهادی

### موضوعات:
- "beautiful woman, detailed face"
- "handsome man, detailed face"

### نورپردازی:
- "sunshine from window"
- "neon light, city" 
- "sunset over sea"
- "golden time"
- "sci-fi RGB glowing, cyberpunk"
- "natural lighting"
- "warm atmosphere, at home, bedroom"
- "magic lit"
- "evil, gothic, Yharnam"
- "light and shadow"
- "shadow from window"
- "soft studio lighting"

## فرمت‌های پشتیبانی شده

### ورودی:
- **تصاویر**: JPG, JPEG, PNG, WEBP
- **اندازه**: 256×256 تا 1024×1024
- **نسبت**: ترجیحاً 1:1 (مربع)

### خروجی:
- **فرمت**: PNG با کیفیت بالا
- **زمان**: 10-20 ثانیه
- **تعداد**: 1-12 تصویر همزمان

## مزایای فنی IC Light

### 1. Consistent Light Transport:
```python
# در فضای HDR، انتقال نور مستقل است
appearance_mixed = blend_appearances(light_sources)
light_source_mixed = mix_light_sources(sources)
# این دو روش معادل هستند
```

### 2. Normal Map Generation:
- قابلیت استخراج Normal Map از Relighting
- عدم نیاز به داده Normal Map در آموزش
- تولید خودکار از طریق consistency

### 3. Multi-Scale Processing:
- پردازش در رزولوشن‌های مختلف
- Hierarchical feature extraction
- Progressive refinement

## چالش‌ها و محدودیت‌ها

### محدودیت‌های فنی:
1. **VRAM**: نیاز به حداقل 16GB
2. **Processing Time**: 10-20 ثانیه برای هر تصویر
3. **Model Size**: مدل‌ها حدود 2-3GB حجم دارند

### محدودیت‌های کاربری:
1. **Image Quality**: بهتر با تصاویر با کیفیت بالا
2. **Face Recognition**: بهینه برای پرتره‌ها
3. **Lighting Angles**: محدود به 4 جهت اصلی

## راهنمای نصب کامل برای Google Colab

### نوت‌بوک کامل:

```python
# =============================================
# IC Light - Complete Google Colab Setup
# =============================================

# سلول 1: بررسی GPU
!nvidia-smi

# سلول 2: Clone Repository
%cd /content
!git clone https://github.com/lllyasviel/IC-Light.git
%cd /content/IC-Light

# سلول 3: Install Requirements
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt

# سلول 4: Download Models (اختیاری - خودکار دانلود می‌شود)
!mkdir -p models
!wget -O models/iclight_sd15_fc.safetensors \
  https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors

# سلول 5: Run Text-Conditioned Demo
!python gradio_demo.py

# سلول 6: Run Background-Conditioned Demo (در سلول جداگانه)
# !python gradio_demo_bg.py
```

### تنظیمات بهینه:
```python
# برای بهبود عملکرد
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

## مقایسه با سایر ابزارها

### مزایای IC Light:
1. **Consistency**: نورپردازی سازگار و طبیعی
2. **Speed**: سرعت بالا نسبت به روش‌های سنتی
3. **Quality**: کیفیت عالی در پرتره‌ها
4. **Flexibility**: دو حالت Text و Background conditioned

### مقایسه با:
- **Total Relighting**: کیفیت مشابه، سرعت بالاتر
- **SwitchLight**: کنترل بیشتر اما پیچیدگی بالاتر
- **GeoWizard**: متفاوت در approach و کاربرد

## نتیجه‌گیری

IC Light یک ابزار قدرتمند برای Relighting با معماری پیشرفته است که:
- بر روی Stable Diffusion ساخته شده
- از تکنولوژی Consistent Light Transport استفاده می‌کند  
- قابلیت اجرا در Google Colab را دارد
- برای کاربرد تجاری و شخصی مناسب است

این اپلیکیشن می‌تواند در زمینه‌های مختلفی مانند عکاسی پرتره، رسانه‌های اجتماعی، تجارت الکترونیک و پست‌پروداکشن مورد استفاده قرار گیرد.
