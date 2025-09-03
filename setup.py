"""
IC Light - Professional Image Relighting Tool
Setup configuration for the package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ic-light-colab",
    version="1.0.0",
    author="IC Light Team",
    author_email="contact@iclight.dev",
    description="Professional AI-powered image relighting tool for Google Colab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/IC-Light-Colab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.27.2",
        "transformers>=4.36.2",
        "accelerate>=0.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "gradio>=4.0.0",
        "spaces>=0.19.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "streamlit>=1.28.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.17.0",
        "einops>=0.7.0",
        "xformers>=0.0.22",
        "controlnet-aux>=0.0.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "xformers>=0.0.22",
        ]
    },
    entry_points={
        "console_scripts": [
            "ic-light=ic_light.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ic_light": ["assets/*", "models/*", "templates/*"],
    },
)
