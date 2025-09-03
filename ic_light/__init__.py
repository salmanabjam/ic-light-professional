"""
IC Light Professional - Main Package
Advanced Image Relighting Tool for Google Colab
"""

__version__ = "1.0.0"
__author__ = "IC Light Team"
__email__ = "contact@iclight.dev"
__description__ = "Professional AI-powered image relighting tool"

# Import main components with error handling
try:
    from .app import ICLightApp
    from .models.ic_light_model import ICLightModel
    from .utils.image_processor import ImageProcessor
    
    __all__ = [
        "ICLightApp",
        "ICLightModel", 
        "ImageProcessor",
        "create_app"
    ]
    
    def create_app(model_type="fc", device="auto", **kwargs):
        """
        Quick start function to create IC Light application
        
        Args:
            model_type: Model variant (fc, fbc, fcon)
            device: Computing device (auto, cpu, cuda)
            **kwargs: Additional arguments for ICLightApp
            
        Returns:
            ICLightApp instance ready to launch
        """
        return ICLightApp(model_type=model_type, device=device, **kwargs)
    
except ImportError as e:
    print(f"⚠️ Warning: IC Light components loading: {e}")
    
    __all__ = ["create_app"]
    
    def create_app(*args, **kwargs):
        """Fallback function when components are not available"""
        raise ImportError(
            "IC Light components not ready. Please install dependencies first."
        )
