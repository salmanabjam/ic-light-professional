"""
UI Components Module
Modern Gradio interface components and themes
"""

import gradio as gr
from typing import Dict, List, Tuple, Optional
import numpy as np


class ModernTheme:
    """Professional theme configuration for Gradio"""
    
    @staticmethod
    def get_custom_css() -> str:
        """Get custom CSS for professional appearance"""
        return """
        /* Professional IC Light Theme */
        .gradio-container {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(102, 126, 234, 0.1);
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            backdrop-filter: blur(10px);
        }
        
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
        }
        
        .metric-box:hover {
            transform: translateY(-5px);
        }
        
        .status-success {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .status-error {
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        /* Button styles */
        .primary-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .primary-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Tab styles */
        .tab-nav {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px !important;
            margin-bottom: 1rem !important;
        }
        
        /* Progress bar */
        .progress-bar {
            background: linear-gradient(90deg, #667eea, #764ba2) !important;
            border-radius: 10px !important;
        }
        
        /* Input styles */
        .gr-textbox, .gr-dropdown, .gr-slider {
            border-radius: 8px !important;
            border: 2px solid rgba(102, 126, 234, 0.2) !important;
            transition: border-color 0.3s ease !important;
        }
        
        .gr-textbox:focus, .gr-dropdown:focus, .gr-slider:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Image containers */
        .image-container {
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header {
                padding: 1.5rem;
            }
            
            .feature-card {
                padding: 1rem;
            }
        }
        """


class ComponentFactory:
    """Factory for creating standardized UI components"""
    
    @staticmethod
    def create_header(title: str, subtitle: str, description: str = "") -> str:
        """Create professional header HTML"""
        return f"""
        <div class="main-header">
            <h1>{title}</h1>
            <h3>{subtitle}</h3>
            {f'<p><em>{description}</em></p>' if description else ''}
        </div>
        """
    
    @staticmethod
    def create_feature_card(title: str, content: str = "") -> str:
        """Create feature card HTML"""
        return f"""
        <div class="feature-card">
            <h3>{title}</h3>
            {f'<p>{content}</p>' if content else ''}
        </div>
        """
    
    @staticmethod
    def create_metric_box(title: str, value: str, unit: str = "") -> str:
        """Create metric display box"""
        return f"""
        <div class="metric-box">
            <h4>{title}</h4>
            <h2>{value} {unit}</h2>
        </div>
        """
    
    @staticmethod
    def create_status_display(message: str, status_type: str = "info") -> str:
        """Create status message display"""
        css_class = f"status-{status_type}"
        return f'<div class="{css_class}">{message}</div>'
    
    @staticmethod
    def create_image_upload(
        label: str = "Upload Image",
        height: int = 300,
        **kwargs
    ) -> gr.Image:
        """Create standardized image upload component"""
        return gr.Image(
            label=label,
            type="pil",
            height=height,
            elem_classes=["image-container"],
            **kwargs
        )
    
    @staticmethod
    def create_primary_button(
        text: str,
        **kwargs
    ) -> gr.Button:
        """Create primary action button"""
        return gr.Button(
            text,
            variant="primary",
            size="lg",
            elem_classes=["primary-btn"],
            **kwargs
        )
    
    @staticmethod
    def create_parameter_slider(
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float = 0.1,
        **kwargs
    ) -> gr.Slider:
        """Create standardized parameter slider"""
        return gr.Slider(
            label=label,
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            **kwargs
        )
    
    @staticmethod
    def create_prompt_textbox(
        label: str,
        placeholder: str = "",
        lines: int = 3,
        value: str = "",
        **kwargs
    ) -> gr.Textbox:
        """Create standardized prompt input"""
        return gr.Textbox(
            label=label,
            placeholder=placeholder,
            lines=lines,
            value=value,
            **kwargs
        )
    
    @staticmethod
    def create_info_panel(info_dict: Dict) -> gr.JSON:
        """Create information display panel"""
        return gr.JSON(
            label="Processing Information",
            value=info_dict
        )


class ExamplePrompts:
    """Collection of example prompts for different lighting scenarios"""
    
    PORTRAIT_LIGHTING = [
        "professional studio lighting, soft key light from left",
        "natural window lighting, warm and diffused",
        "dramatic side lighting, high contrast shadows",
        "beauty lighting setup with ring light",
        "golden hour portrait lighting, warm tones",
        "corporate headshot lighting, clean and professional"
    ]
    
    CREATIVE_LIGHTING = [
        "cinematic film noir lighting, dramatic shadows",
        "neon cyberpunk lighting, blue and pink tones",
        "vintage golden hour lighting, nostalgic mood",
        "editorial fashion lighting, high contrast",
        "moody atmospheric lighting, mysterious ambiance",
        "commercial product lighting, clean and bright"
    ]
    
    ARTISTIC_LIGHTING = [
        "rembrandt lighting style, classical portrait",
        "chiaroscuro lighting, strong light-dark contrast",
        "soft impressionist lighting, painting-like quality",
        "abstract artistic lighting, creative shadows",
        "minimalist lighting setup, clean and simple",
        "experimental lighting, artistic expression"
    ]
    
    @classmethod
    def get_random_prompt(cls, category: str = "portrait") -> str:
        """Get random prompt from specified category"""
        category_map = {
            "portrait": cls.PORTRAIT_LIGHTING,
            "creative": cls.CREATIVE_LIGHTING,
            "artistic": cls.ARTISTIC_LIGHTING
        }
        
        prompts = category_map.get(category, cls.PORTRAIT_LIGHTING)
        return np.random.choice(prompts)
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get list of all prompt categories"""
        return ["portrait", "creative", "artistic"]


class PresetManager:
    """Manage lighting presets for quick selection"""
    
    PRESETS = {
        "Portrait - Natural": {
            "prompt": "natural daylight from window, soft and flattering",
            "negative_prompt": "harsh shadows, overexposed, artificial lighting",
            "lighting_direction": "left",
            "guidance_scale": 2.0,
            "steps": 25
        },
        "Portrait - Studio": {
            "prompt": "professional studio lighting, key light and fill light",
            "negative_prompt": "amateur lighting, harsh shadows, uneven",
            "lighting_direction": "left",
            "guidance_scale": 2.5,
            "steps": 30
        },
        "Creative - Dramatic": {
            "prompt": "dramatic cinematic lighting, high contrast",
            "negative_prompt": "flat lighting, low contrast, boring",
            "lighting_direction": "left",
            "guidance_scale": 3.0,
            "steps": 35
        },
        "Product - Commercial": {
            "prompt": "clean commercial lighting, evenly lit, no shadows",
            "negative_prompt": "dark shadows, uneven lighting, amateur",
            "lighting_direction": "top",
            "guidance_scale": 1.5,
            "steps": 20
        },
        "Artistic - Moody": {
            "prompt": "moody atmospheric lighting, artistic shadows",
            "negative_prompt": "bright lighting, commercial look, generic",
            "lighting_direction": "right",
            "guidance_scale": 2.5,
            "steps": 30
        }
    }
    
    @classmethod
    def get_preset_names(cls) -> List[str]:
        """Get list of available preset names"""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset(cls, name: str) -> Dict:
        """Get preset parameters by name"""
        return cls.PRESETS.get(name, {})
    
    @classmethod
    def apply_preset(cls, name: str) -> Tuple:
        """Get preset values as tuple for Gradio update"""
        preset = cls.get_preset(name)
        if not preset:
            return None
        
        return (
            preset.get("prompt", ""),
            preset.get("negative_prompt", ""),
            preset.get("lighting_direction", "left"),
            preset.get("guidance_scale", 2.0),
            preset.get("steps", 25)
        )
