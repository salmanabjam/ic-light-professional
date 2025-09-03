"""
IC Light Application - Professional Gradio Interface
Advanced image relighting application with modern UI components
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from typing import Optional, Tuple, List, Dict
import threading
import queue
import logging

from .models.ic_light_model import ICLightModel
from .utils.image_processor import ImageProcessor


class ICLightApp:
    """
    Professional IC Light Application
    
    Features:
    - Modern Gradio 4.0+ interface
    - Advanced visualization with Plotly
    - Real-time processing
    - Batch operations
    - Professional analytics
    """
    
    def __init__(
        self,
        model_type: str = "fc",
        device: str = "auto",
        enable_analytics: bool = True,
        theme: str = "default"
    ):
        self.model_type = model_type
        self.device = device
        self.enable_analytics = enable_analytics
        self.theme = theme
        
        # Initialize components
        print("🚀 Initializing IC Light Application...")
        self._setup_logging()
        self._initialize_models()
        self._setup_analytics()
        
        # Processing queue for batch operations
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        print("✅ IC Light Application ready!")
        
    def _setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize IC Light model
            self.ic_light_model = ICLightModel(
                model_type=self.model_type,
                device=self.device,
                enable_xformers=True,
                enable_cpu_offload=torch.cuda.device_count() < 2
            )
            
            # Initialize image processor
            self.image_processor = ImageProcessor(
                device=self.device,
                enable_background_removal=True,
                high_quality=True
            )
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
            
    def _setup_analytics(self):
        """Setup analytics tracking"""
        if self.enable_analytics:
            self.analytics = {
                "processing_times": [],
                "popular_prompts": {},
                "lighting_directions": {},
                "user_ratings": [],
                "error_counts": {}
            }
        else:
            self.analytics = None
            
    def process_single_image(
        self,
        input_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        lighting_direction: str,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        width: int,
        height: int,
        remove_background: bool,
        enhance_image: bool,
        brightness: float,
        contrast: float,
        saturation: float,
        progress=gr.Progress()
    ) -> Tuple[Image.Image, Image.Image, str, Dict]:
        """Process single image with IC Light"""
        
        start_time = time.time()
        
        try:
            # Update progress
            progress(0.1, desc="Preprocessing image...")
            
            # Validate inputs
            if input_image is None:
                return None, None, "❌ Please upload an image", {}
            
            if not prompt.strip():
                return None, None, "❌ Please provide a text prompt", {}
            
            # Preprocess image
            processed_img = input_image.copy()
            
            # Remove background if requested
            if remove_background:
                progress(0.2, desc="Removing background...")
                processed_img = self.image_processor.remove_background(
                    processed_img
                )
            
            # Enhance image if requested
            if enhance_image:
                progress(0.3, desc="Enhancing image...")
                processed_img = self.image_processor.enhance_image(
                    processed_img,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation
                )
            
            # Resize image
            progress(0.4, desc="Resizing image...")
            processed_img = self.image_processor.resize_with_aspect(
                processed_img, (width, height)
            )
            
            # Apply IC Light
            progress(0.5, desc="Applying IC Light relighting...")
            
            result_image = self.ic_light_model.process_image(
                input_image=processed_img,
                prompt=prompt,
                lighting_direction=lighting_direction,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed if seed >= 0 else None,
                width=width,
                height=height
            )
            
            # Create visualization
            progress(0.8, desc="Creating visualization...")
            
            viz_image = self.image_processor.create_lighting_visualization(
                original=input_image,
                processed=result_image,
                lighting_direction=lighting_direction
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update analytics
            if self.analytics:
                self._update_analytics(
                    prompt=prompt,
                    lighting_direction=lighting_direction,
                    processing_time=processing_time
                )
            
            # Create info
            info = {
                "processing_time": f"{processing_time:.2f}s",
                "model_type": self.model_type,
                "steps": num_steps,
                "guidance_scale": guidance_scale,
                "resolution": f"{width}x{height}",
                "seed": seed if seed >= 0 else "Random"
            }
            
            progress(1.0, desc="Complete!")
            
            success_msg = f"✅ Processing complete in {processing_time:.2f}s"
            
            return result_image, viz_image, success_msg, info
            
        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            self.logger.error(error_msg)
            
            if self.analytics:
                self._update_error_count(str(e))
            
            return None, None, error_msg, {}
            
    def batch_process_images(
        self,
        images: List[Image.Image],
        prompt: str,
        settings: Dict,
        progress=gr.Progress()
    ) -> List[Image.Image]:
        """Process multiple images in batch"""
        
        if not images:
            return []
        
        results = []
        total = len(images)
        
        for i, image in enumerate(images):
            progress((i + 1) / total, desc=f"Processing image {i+1}/{total}")
            
            try:
                result = self.ic_light_model.process_image(
                    input_image=image,
                    prompt=prompt,
                    **settings
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Batch processing error on image {i}: {e}")
                results.append(image)  # Return original on error
                
        return results
        
    def create_analytics_dashboard(self) -> Tuple[go.Figure, go.Figure]:
        """Create analytics dashboard with Plotly"""
        
        if not self.analytics or not self.analytics["processing_times"]:
            # Empty dashboard
            fig1 = go.Figure()
            fig1.add_annotation(
                text="No analytics data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            fig2 = go.Figure()
            fig2.add_annotation(
                text="No usage data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            return fig1, fig2
        
        # Processing times chart
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Processing Times', 'Popular Prompts',
                'Lighting Directions', 'User Ratings'
            ],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Processing times
        times = self.analytics["processing_times"]
        fig1.add_trace(
            go.Scatter(
                y=times[-50:],  # Last 50 processes
                mode='lines+markers',
                name='Processing Time',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Popular prompts
        if self.analytics["popular_prompts"]:
            prompts = list(self.analytics["popular_prompts"].keys())[:10]
            counts = [self.analytics["popular_prompts"][p] for p in prompts]
            
            fig1.add_trace(
                go.Bar(x=counts, y=prompts, orientation='h', name='Prompts'),
                row=1, col=2
            )
        
        # Lighting directions
        if self.analytics["lighting_directions"]:
            directions = list(self.analytics["lighting_directions"].keys())
            counts = list(self.analytics["lighting_directions"].values())
            
            fig1.add_trace(
                go.Pie(labels=directions, values=counts, name="Directions"),
                row=2, col=1
            )
        
        # User ratings
        if self.analytics["user_ratings"]:
            fig1.add_trace(
                go.Histogram(
                    x=self.analytics["user_ratings"],
                    nbinsx=5,
                    name="Ratings"
                ),
                row=2, col=2
            )
        
        fig1.update_layout(
            height=600,
            title_text="IC Light Analytics Dashboard",
            showlegend=False
        )
        
        # Performance metrics
        fig2 = go.Figure()
        
        if times:
            # Calculate metrics
            avg_time = np.mean(times)
            median_time = np.median(times)
            
            # Timeline chart
            fig2.add_trace(go.Scatter(
                x=list(range(len(times))),
                y=times,
                mode='lines',
                name='Processing Times',
                line=dict(color='green', width=2)
            ))
            
            # Add average line
            fig2.add_hline(
                y=avg_time,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_time:.2f}s"
            )
            
            fig2.update_layout(
                title="Performance Timeline",
                xaxis_title="Process Number",
                yaxis_title="Processing Time (s)",
                height=400
            )
        
        return fig1, fig2
        
    def _update_analytics(
        self,
        prompt: str,
        lighting_direction: str,
        processing_time: float
    ):
        """Update analytics data"""
        if not self.analytics:
            return
        
        # Processing times
        self.analytics["processing_times"].append(processing_time)
        
        # Popular prompts
        prompt_key = prompt.lower()[:50]  # Truncate for privacy
        self.analytics["popular_prompts"][prompt_key] = (
            self.analytics["popular_prompts"].get(prompt_key, 0) + 1
        )
        
        # Lighting directions
        self.analytics["lighting_directions"][lighting_direction] = (
            self.analytics["lighting_directions"].get(lighting_direction, 0) + 1
        )
        
    def _update_error_count(self, error: str):
        """Update error analytics"""
        if not self.analytics:
            return
            
        error_key = error[:100]  # Truncate error message
        self.analytics["error_counts"][error_key] = (
            self.analytics["error_counts"].get(error_key, 0) + 1
        )
        
    def add_user_rating(self, rating: int):
        """Add user rating to analytics"""
        if self.analytics and 1 <= rating <= 5:
            self.analytics["user_ratings"].append(rating)
            
    def create_interface(self) -> gr.Blocks:
        """Create modern Gradio interface"""
        
        # Custom CSS for professional look
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif !important;
        }
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .feature-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        """
        
        with gr.Blocks(
            title="IC Light Professional",
            css=custom_css,
            theme=gr.themes.Soft() if self.theme == "soft" else None
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🌟 IC Light Professional</h1>
                <p>Advanced AI Image Relighting with Consistent Lighting</p>
                <p><em>Powered by Stable Diffusion & Modern ML</em></p>
            </div>
            """)
            
            with gr.Tabs():
                
                # Main Processing Tab
                with gr.Tab("🎨 Image Relighting", elem_id="main-tab"):
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="feature-card"><h3>Input Settings</h3></div>')
                            
                            input_image = gr.Image(
                                label="Upload Image",
                                type="pil",
                                height=300
                            )
                            
                            prompt = gr.Textbox(
                                label="Lighting Prompt",
                                placeholder="Describe the desired lighting...",
                                lines=3,
                                value="beautiful cinematic lighting, warm ambient light"
                            )
                            
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="What to avoid...",
                                lines=2,
                                value="harsh shadows, overexposed, underexposed, blurry"
                            )
                            
                            with gr.Row():
                                lighting_direction = gr.Dropdown(
                                    label="Light Direction",
                                    choices=["left", "right", "top", "bottom", "none"],
                                    value="left"
                                )
                                
                                model_variant = gr.Dropdown(
                                    label="Model Type",
                                    choices=["fc", "fbc", "fcon"],
                                    value=self.model_type,
                                    interactive=False
                                )
                            
                        with gr.Column(scale=2):
                            gr.HTML('<div class="feature-card"><h3>Results</h3></div>')
                            
                            with gr.Row():
                                output_image = gr.Image(
                                    label="Relit Image",
                                    height=400
                                )
                                
                                viz_image = gr.Image(
                                    label="Analysis Visualization",
                                    height=400
                                )
                            
                            status_text = gr.Textbox(
                                label="Status",
                                value="Ready for processing",
                                interactive=False
                            )
                    
                    # Advanced Settings
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<h4>Generation Parameters</h4>')
                                
                                num_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=10,
                                    maximum=50,
                                    value=25,
                                    step=5
                                )
                                
                                guidance_scale = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=1.0,
                                    maximum=7.5,
                                    value=2.0,
                                    step=0.5
                                )
                                
                                seed = gr.Number(
                                    label="Seed (-1 for random)",
                                    value=-1,
                                    precision=0
                                )
                                
                            with gr.Column():
                                gr.HTML('<h4>Image Processing</h4>')
                                
                                with gr.Row():
                                    width = gr.Slider(
                                        label="Width",
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64
                                    )
                                    
                                    height = gr.Slider(
                                        label="Height", 
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64
                                    )
                                
                                remove_bg = gr.Checkbox(
                                    label="Remove Background",
                                    value=False
                                )
                                
                                enhance_img = gr.Checkbox(
                                    label="Enhance Image",
                                    value=False
                                )
                            
                            with gr.Column():
                                gr.HTML('<h4>Enhancement Settings</h4>')
                                
                                brightness = gr.Slider(
                                    label="Brightness",
                                    minimum=0.5,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.1
                                )
                                
                                contrast = gr.Slider(
                                    label="Contrast",
                                    minimum=0.5,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.1
                                )
                                
                                saturation = gr.Slider(
                                    label="Saturation",
                                    minimum=0.5,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.1
                                )
                    
                    # Processing Buttons
                    with gr.Row():
                        process_btn = gr.Button(
                            "🚀 Process Image",
                            variant="primary",
                            size="lg"
                        )
                        
                        clear_btn = gr.Button(
                            "🗑️ Clear All",
                            variant="secondary"
                        )
                        
                        random_prompt_btn = gr.Button(
                            "🎲 Random Prompt",
                            variant="secondary"
                        )
                    
                    # Processing info
                    with gr.Row():
                        info_json = gr.JSON(
                            label="Processing Information",
                            visible=False
                        )
                    
                # Analytics Tab
                with gr.Tab("📊 Analytics", elem_id="analytics-tab"):
                    
                    gr.HTML('<div class="feature-card"><h3>Performance Analytics</h3></div>')
                    
                    with gr.Row():
                        refresh_analytics = gr.Button("🔄 Refresh Analytics")
                        
                        with gr.Column():
                            gr.HTML('<div class="metric-box"><h4>Total Processes</h4><h2 id="total-processes">0</h2></div>')
                        
                        with gr.Column():
                            gr.HTML('<div class="metric-box"><h4>Avg Time</h4><h2 id="avg-time">0.0s</h2></div>')
                        
                        with gr.Column():
                            gr.HTML('<div class="metric-box"><h4>Success Rate</h4><h2 id="success-rate">100%</h2></div>')
                    
                    analytics_plot1 = gr.Plot(label="Usage Statistics")
                    analytics_plot2 = gr.Plot(label="Performance Timeline")
                    
                    # User feedback
                    with gr.Row():
                        gr.HTML('<h4>Rate Your Experience</h4>')
                        user_rating = gr.Radio(
                            choices=[1, 2, 3, 4, 5],
                            label="Rating (1-5 stars)",
                            value=5
                        )
                        submit_rating = gr.Button("Submit Rating")
                
                # Help Tab
                with gr.Tab("❓ Help & Examples", elem_id="help-tab"):
                    
                    gr.HTML("""
                    <div class="feature-card">
                        <h3>🎯 How to Use IC Light</h3>
                        <ol>
                            <li><strong>Upload an image:</strong> Choose any portrait or object photo</li>
                            <li><strong>Write a prompt:</strong> Describe the lighting you want</li>
                            <li><strong>Select direction:</strong> Choose where the light comes from</li>
                            <li><strong>Adjust settings:</strong> Fine-tune advanced parameters</li>
                            <li><strong>Process:</strong> Click the process button and wait</li>
                        </ol>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<h4>💡 Example Prompts</h4>')
                            example_prompts = gr.Textbox(
                                value="""• "warm golden hour lighting, soft shadows"
• "dramatic studio lighting, high contrast"
• "natural daylight from window"
• "cinematic blue hour lighting"
• "soft ring light for portraits"
• "harsh industrial lighting"
• "romantic candlelight ambiance"
• "bright commercial photography lighting"
""",
                                lines=10,
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.HTML('<h4>⚙️ Parameter Guide</h4>')
                            param_guide = gr.Textbox(
                                value="""• Steps: Higher = better quality, slower
• Guidance: 1-3 for subtle, 4-7 for dramatic
• Seed: Same seed = same result
• Background removal: For isolated objects
• Enhancement: Adjust brightness/contrast
• Resolution: Higher = better quality, slower
""",
                                lines=10,
                                interactive=False
                            )
            
            # Event handlers
            def random_prompt():
                prompts = [
                    "warm cinematic lighting, golden hour",
                    "dramatic studio lighting with soft shadows",
                    "natural daylight streaming through window",
                    "romantic candlelight ambiance",
                    "professional portrait lighting",
                    "moody blue hour lighting",
                    "bright commercial photography setup",
                    "soft ring light for beauty shots"
                ]
                return np.random.choice(prompts)
                
            def clear_all():
                return None, None, None, "Ready for processing", {}
                
            def update_analytics():
                return self.create_analytics_dashboard()
                
            def submit_user_rating(rating):
                self.add_user_rating(rating)
                return "Thank you for your feedback!"
            
            # Connect events
            process_btn.click(
                self.process_single_image,
                inputs=[
                    input_image, prompt, negative_prompt, lighting_direction,
                    num_steps, guidance_scale, seed, width, height,
                    remove_bg, enhance_img, brightness, contrast, saturation
                ],
                outputs=[output_image, viz_image, status_text, info_json]
            )
            
            clear_btn.click(
                clear_all,
                outputs=[input_image, output_image, viz_image, status_text, info_json]
            )
            
            random_prompt_btn.click(
                random_prompt,
                outputs=[prompt]
            )
            
            refresh_analytics.click(
                update_analytics,
                outputs=[analytics_plot1, analytics_plot2]
            )
            
            submit_rating.click(
                submit_user_rating,
                inputs=[user_rating],
                outputs=[status_text]
            )
        
        return interface
        
    def launch(
        self,
        share: bool = False,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        debug: bool = False
    ):
        """Launch the Gradio application"""
        
        interface = self.create_interface()
        
        print(f"🌐 Launching IC Light Professional on http://{server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=debug,
            show_error=True,
            quiet=False
        )
