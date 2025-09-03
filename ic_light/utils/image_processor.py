"""
Advanced Image Processing for IC Light
Background removal, image enhancement, and preprocessing utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Tuple, Union, List
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import io


class ImageProcessor:
    """
    Professional Image Processing for IC Light
    
    Features:
    - Background removal with BriaRMBG
    - Image enhancement and filtering
    - Multi-resolution support
    - Batch processing
    - Visual analytics
    """
    
    def __init__(
        self,
        device: str = "auto",
        enable_background_removal: bool = True,
        high_quality: bool = True
    ):
        self.device = self._setup_device(device)
        self.enable_background_removal = enable_background_removal
        self.high_quality = high_quality
        
        # Initialize background removal model
        if enable_background_removal:
            self._setup_background_removal()
            
        # Processing parameters
        self.default_size = (512, 512)
        self.max_size = (1024, 1024) if high_quality else (768, 768)
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup processing device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
        
    def _setup_background_removal(self):
        """Initialize background removal model (BriaRMBG)"""
        try:
            from transformers import pipeline
            
            print("🔧 Loading background removal model...")
            
            self.bg_remover = pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                device=0 if self.device.type == "cuda" else -1
            )
            
            print("✅ Background removal ready")
            
        except Exception as e:
            print(f"⚠️ Background removal not available: {e}")
            self.bg_remover = None
            
    def remove_background(
        self,
        image: Union[Image.Image, np.ndarray],
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """
        Remove background from image using RMBG
        
        Args:
            image: Input image
            return_mask: Whether to return mask separately
            
        Returns:
            Processed image (and mask if requested)
        """
        if not hasattr(self, 'bg_remover') or self.bg_remover is None:
            print("⚠️ Background removal not available")
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            return (image, Image.new('L', image.size, 255)) if return_mask else image
            
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            
        try:
            # Process with RMBG
            result = self.bg_remover(image)
            
            if isinstance(result, list) and len(result) > 0:
                mask = result[0]['mask']
                
                # Create RGBA image
                rgba_image = image.convert('RGBA')
                rgba_array = np.array(rgba_image)
                mask_array = np.array(mask)
                
                # Apply mask to alpha channel
                rgba_array[:, :, 3] = mask_array
                processed_image = Image.fromarray(rgba_array, 'RGBA')
                
                if return_mask:
                    return processed_image, mask
                return processed_image
            else:
                print("⚠️ Background removal failed")
                return (image, Image.new('L', image.size, 255)) if return_mask else image
                
        except Exception as e:
            print(f"⚠️ Background removal error: {e}")
            return (image, Image.new('L', image.size, 255)) if return_mask else image
            
    def enhance_image(
        self,
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
    ) -> Image.Image:
        """Apply image enhancements"""
        
        # Brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            
        # Contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
            
        # Saturation
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
            
        # Sharpness
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
            
        return image
        
    def resize_with_aspect(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        method: str = "lanczos"
    ) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        
        # Get original dimensions
        orig_w, orig_h = image.size
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize methods
        resize_methods = {
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST
        }
        
        resample = resize_methods.get(method, Image.Resampling.LANCZOS)
        
        # Resize image
        resized = image.resize((new_w, new_h), resample)
        
        # Create canvas and center image
        canvas = Image.new('RGB', target_size, (0, 0, 0))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        
        if resized.mode == 'RGBA':
            canvas = canvas.convert('RGBA')
            
        canvas.paste(resized, (offset_x, offset_y))
        
        return canvas
        
    def create_lighting_visualization(
        self,
        original: Image.Image,
        processed: Image.Image,
        lighting_direction: str,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """Create professional visualization comparing results"""
        
        # Setup matplotlib with professional style
        plt.style.use('default')
        
        # Create figure with advanced layout
        fig = plt.figure(figsize=(16, 8), facecolor='white')
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Processed image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(processed)
        ax2.set_title('IC Light Result', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Lighting direction visualization
        ax3 = fig.add_subplot(gs[0, 2])
        self._draw_lighting_diagram(ax3, lighting_direction)
        ax3.set_title(f'Lighting: {lighting_direction.title()}', 
                     fontsize=14, fontweight='bold')
        
        # Histogram comparison
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_histogram_comparison(ax4, original, processed)
        
        # Color analysis
        ax5 = fig.add_subplot(gs[1, :2])
        self._plot_color_analysis(ax5, original, processed)
        
        # Lighting intensity profile
        ax6 = fig.add_subplot(gs[1, 2:])
        self._plot_lighting_profile(ax6, processed, lighting_direction)
        
        plt.tight_layout()
        
        # Save or return
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Convert to PIL Image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        viz_image = Image.open(buffer)
        
        plt.close(fig)
        
        return viz_image
        
    def _draw_lighting_diagram(self, ax, direction: str):
        """Draw lighting direction diagram"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Draw object (circle)
        circle = patches.Circle((5, 5), 2, facecolor='lightgray', 
                              edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Draw light source based on direction
        light_positions = {
            'left': (1, 5),
            'right': (9, 5),
            'top': (5, 9),
            'bottom': (5, 1),
            'none': (5, 5)
        }
        
        if direction.lower() in light_positions:
            lx, ly = light_positions[direction.lower()]
            
            # Light source
            light = patches.Circle((lx, ly), 0.5, facecolor='yellow', 
                                 edgecolor='orange', linewidth=2)
            ax.add_patch(light)
            
            # Light rays
            if direction.lower() != 'none':
                for i in range(5):
                    angle = np.pi * i / 4 - np.pi/2
                    if direction.lower() == 'left':
                        end_x = 4 + 0.5 * np.cos(angle)
                        end_y = 5 + 0.5 * np.sin(angle)
                    elif direction.lower() == 'right':
                        end_x = 6 + 0.5 * np.cos(angle + np.pi)
                        end_y = 5 + 0.5 * np.sin(angle + np.pi)
                    elif direction.lower() == 'top':
                        end_x = 5 + 0.5 * np.cos(angle + np.pi/2)
                        end_y = 4 + 0.5 * np.sin(angle + np.pi/2)
                    else:  # bottom
                        end_x = 5 + 0.5 * np.cos(angle - np.pi/2)
                        end_y = 6 + 0.5 * np.sin(angle - np.pi/2)
                        
                    ax.plot([lx, end_x], [ly, end_y], 'orange', 
                           linewidth=1, alpha=0.7)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
    def _plot_histogram_comparison(self, ax, original: Image.Image, 
                                 processed: Image.Image):
        """Plot RGB histogram comparison"""
        
        # Convert images to arrays
        orig_array = np.array(original.convert('RGB'))
        proc_array = np.array(processed.convert('RGB'))
        
        # Calculate histograms
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist_orig, bins = np.histogram(orig_array[:,:,i], bins=50, 
                                         range=(0, 255), density=True)
            hist_proc, _ = np.histogram(proc_array[:,:,i], bins=50, 
                                      range=(0, 255), density=True)
            
            # Plot histograms
            ax.plot(bins[:-1], hist_orig, color=color, alpha=0.5, 
                   linestyle='--', label=f'Original {color.title()}')
            ax.plot(bins[:-1], hist_proc, color=color, alpha=0.8, 
                   label=f'Processed {color.title()}')
        
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.set_title('RGB Histogram Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_color_analysis(self, ax, original: Image.Image, 
                           processed: Image.Image):
        """Plot color distribution analysis"""
        
        # Convert to LAB color space for better analysis
        orig_array = np.array(original.convert('RGB'))
        proc_array = np.array(processed.convert('RGB'))
        
        # Calculate mean colors
        orig_mean = np.mean(orig_array.reshape(-1, 3), axis=0)
        proc_mean = np.mean(proc_array.reshape(-1, 3), axis=0)
        
        # Color temperature estimation (simplified)
        orig_temp = self._estimate_color_temperature(orig_mean)
        proc_temp = self._estimate_color_temperature(proc_mean)
        
        # Create bar chart
        categories = ['Red', 'Green', 'Blue', 'Color Temp (K)']
        orig_values = list(orig_mean) + [orig_temp]
        proc_values = list(proc_mean) + [proc_temp]
        
        # Normalize color temp for visualization
        orig_values_norm = orig_values[:3] + [orig_temp / 100]
        proc_values_norm = proc_values[:3] + [proc_temp / 100]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, orig_values_norm, width, label='Original', 
               alpha=0.7)
        ax.bar(x + width/2, proc_values_norm, width, label='Processed', 
               alpha=0.7)
        
        ax.set_xlabel('Color Components')
        ax.set_ylabel('Intensity / Temperature (K/100)')
        ax.set_title('Color Analysis Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add values as text
        for i, (orig, proc) in enumerate(zip(orig_values_norm, 
                                           proc_values_norm)):
            ax.text(i - width/2, orig + 5, f'{orig_values[i]:.0f}', 
                   ha='center', fontsize=8)
            ax.text(i + width/2, proc + 5, f'{proc_values[i]:.0f}', 
                   ha='center', fontsize=8)
        
    def _plot_lighting_profile(self, ax, image: Image.Image, direction: str):
        """Plot lighting intensity profile"""
        
        # Convert to grayscale for intensity analysis
        gray = np.array(image.convert('L'))
        h, w = gray.shape
        
        # Create profile based on direction
        if direction.lower() == 'left' or direction.lower() == 'right':
            profile = np.mean(gray, axis=0)  # Horizontal profile
            x_axis = np.arange(w)
            ax.set_xlabel('Horizontal Position')
        else:  # top or bottom
            profile = np.mean(gray, axis=1)  # Vertical profile
            x_axis = np.arange(h)
            ax.set_xlabel('Vertical Position')
        
        # Plot profile
        ax.plot(x_axis, profile, linewidth=2, color='blue')
        ax.fill_between(x_axis, profile, alpha=0.3, color='blue')
        
        # Add expected direction indicator
        if direction.lower() == 'left':
            ax.axvline(x=w*0.2, color='red', linestyle='--', alpha=0.7, 
                      label='Expected bright area')
        elif direction.lower() == 'right':
            ax.axvline(x=w*0.8, color='red', linestyle='--', alpha=0.7, 
                      label='Expected bright area')
        elif direction.lower() == 'top':
            ax.axvline(x=h*0.2, color='red', linestyle='--', alpha=0.7, 
                      label='Expected bright area')
        elif direction.lower() == 'bottom':
            ax.axvline(x=h*0.8, color='red', linestyle='--', alpha=0.7, 
                      label='Expected bright area')
        
        ax.set_ylabel('Average Intensity')
        ax.set_title(f'Lighting Profile - {direction.title()} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    def _estimate_color_temperature(self, rgb_mean: np.ndarray) -> float:
        """Estimate color temperature from RGB values (simplified)"""
        
        r, g, b = rgb_mean / 255.0
        
        # Simplified color temperature estimation
        # This is a rough approximation
        if r > g and r > b:
            # Warm light
            temp = 3000 + (g - b) * 1000
        elif b > r and b > g:
            # Cool light
            temp = 6000 + (b - r) * 2000
        else:
            # Neutral
            temp = 5000
            
        return max(2000, min(10000, temp))
        
    def batch_process(
        self,
        images: List[Union[str, Image.Image]],
        operation: str = "remove_background",
        **kwargs
    ) -> List[Image.Image]:
        """Process multiple images in batch"""
        
        results = []
        
        for i, img in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}...")
            
            # Load image if path
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            
            # Apply operation
            if operation == "remove_background":
                result = self.remove_background(img, **kwargs)
            elif operation == "enhance":
                result = self.enhance_image(img, **kwargs)
            elif operation == "resize":
                result = self.resize_with_aspect(img, **kwargs)
            else:
                result = img
                
            results.append(result)
            
        return results
