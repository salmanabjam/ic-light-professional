# IC Light Google Colab Implementation

Complete implementation and documentation for IC Light (Imposing Consistent Light) - an advanced AI-powered image relighting application.

## 📁 Files Overview

### 📋 Documentation Files
- **`IC_Light_Complete_Analysis_FA.md`** - Comprehensive Persian analysis and documentation
- **`IC_Light_Technical_Implementation_Guide.md`** - Technical implementation details
- **`README.md`** - This overview file

### 💻 Implementation Files  
- **`IC_Light_Complete_Google_Colab.ipynb`** - Complete Jupyter notebook for Google Colab
- **`ic_light_setup.py`** - Python setup script for quick installation

## 🎯 What is IC Light?

IC Light is a revolutionary AI tool for image relighting that can:
- **Transform lighting** in portraits and images using text descriptions
- **Relight with reference images** using background-conditioned models
- **Control lighting direction** (Left, Right, Top, Bottom)
- **Maintain consistency** through mathematical light transport principles
- **Process in real-time** (10-20 seconds per image)

## ✨ Key Features

### 🔧 Technical Capabilities
- Built on **Stable Diffusion 1.5** architecture
- **Modified UNet** with 8/12 channel inputs for conditioning
- **BriaRMBG 1.4** for automatic background removal
- **HDR-consistent light transport** for realistic results
- **Multi-resolution processing** for high-quality outputs

### 🎨 Creative Features
- **Text-conditioned relighting**: "sunset over sea", "neon city lights", etc.
- **Background-conditioned relighting**: Use reference images for lighting
- **Multiple lighting directions**: Left, Right, Top, Bottom illumination
- **Advanced controls**: CFG scale, inference steps, denoising strength
- **Batch processing**: Handle multiple images efficiently

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Open `IC_Light_Complete_Google_Colab.ipynb` in Google Colab
2. Run all cells sequentially
3. Use the Gradio interface that appears

### Option 2: Python Setup Script
```bash
python ic_light_setup.py
```

### Option 3: Manual Installation
```bash
# Clone repository
git clone https://github.com/lllyasviel/IC-Light.git
cd IC-Light

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.27.2 gradio==3.41.2

# Run demo
python gradio_demo.py  # Text-conditioned
# OR
python gradio_demo_bg.py  # Background-conditioned
```

## 📊 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (T4/V100/A100)
- **RAM**: 12GB+ system memory
- **Storage**: 10GB+ free space for models

### Software
- **Python**: 3.8-3.11
- **CUDA**: 11.7 or 12.1
- **PyTorch**: 2.0+
- **Google Colab**: Recommended environment

## 🎨 Usage Examples

### Text Prompts
```
Subject + Lighting Style:
- "beautiful woman, detailed face, sunshine from window"
- "handsome man, detailed face, neon light, city"
- "cat, detailed fur, golden hour lighting"
- "Buddha statue, natural lighting"
```

### Lighting Directions
- **Left Light**: Illumination from left side
- **Right Light**: Illumination from right side  
- **Top Light**: Illumination from above
- **Bottom Light**: Illumination from below

### Advanced Settings
- **CFG Scale**: 1.0-32.0 (default: 2.0)
- **Steps**: 1-100 (default: 25)
- **Resolution**: 256x256 to 1024x1024
- **Seed**: For reproducible results

## 📐 Model Architecture

### Available Models
1. **iclight_sd15_fc.safetensors** - Foreground Conditioned (default)
2. **iclight_sd15_fbc.safetensors** - Foreground + Background Conditioned
3. **iclight_sd15_fcon.safetensors** - With Offset Noise

### Processing Pipeline
```
Input Image → Background Removal → Light Conditioning → 
Stable Diffusion → Post-processing → Output Image
```

### Mathematical Foundation
Based on the principle: `blend(appearances) = appearance(blend(lights))`
- Ensures consistent light transport in HDR space
- Enables realistic relighting effects
- Allows normal map generation from lighting consistency

## 🔧 Advanced Configuration

### Environment Variables
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

### Memory Optimization
```python
# Enable attention slicing
unet.set_attention_slice("auto")

# Use efficient attention
unet.set_attn_processor(AttnProcessor2_0())

# Mixed precision
text_encoder = text_encoder.to(dtype=torch.float16)
vae = vae.to(dtype=torch.bfloat16)
```

## 🎯 Use Cases

### Professional Applications
- **Portrait Photography**: Enhanced lighting for professional shoots
- **E-commerce**: Product image improvement
- **Film/TV**: Post-production lighting adjustment
- **Architecture**: Building visualization with different lighting

### Creative Applications  
- **Social Media**: Instagram-worthy lighting effects
- **Art Creation**: Artistic lighting experiments
- **Gaming**: Character lighting for game development
- **Education**: Learning about light and photography

## 🔍 Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce image resolution, clear GPU cache |
| Slow processing | Ensure GPU is being used, reduce steps |
| Poor quality | Use higher resolution input, clear face visibility |
| Model loading errors | Check internet connection, restart notebook |

### Performance Tips
- Use square images (1:1 ratio) for best results
- Ensure clear face visibility in portraits
- Experiment with different CFG scale values
- Try various prompt combinations
- Monitor GPU memory usage

## 📚 Resources

### Original Project
- **Paper**: [Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization](https://openreview.net/forum?id=u1cQYxRI1H)
- **GitHub**: https://github.com/lllyasviel/IC-Light
- **Hugging Face**: https://huggingface.co/lllyasviel/ic-light

### Related Work
- **Total Relighting**: Portrait background replacement
- **SwitchLight**: Physics-driven relighting  
- **GeoWizard**: Geometry-aware processing

## 👥 Credits

### Authors
- **Lvmin Zhang** - Creator of IC Light and ControlNet
- **Anyi Rao** - Research contributor
- **Maneesh Agrawala** - Research supervision

### Implementation
- **Original Implementation**: lllyasviel/IC-Light
- **Google Colab Version**: camenduru/IC-Light-jupyter  
- **This Documentation**: Comprehensive analysis and setup

## 📄 License

- **IC Light Models**: CreativeML Open RAIL-M
- **Code**: Apache 2.0
- **BriaRMBG**: Non-commercial use (replace for commercial use)

## 🚀 Getting Started

1. **Read the documentation** in `IC_Light_Complete_Analysis_FA.md` (Persian)
2. **Review technical details** in `IC_Light_Technical_Implementation_Guide.md` 
3. **Open the notebook** `IC_Light_Complete_Google_Colab.ipynb` in Google Colab
4. **Run all cells** and start experimenting!

---

**Ready to transform your images with AI-powered lighting? Let's get started! 🌟**
