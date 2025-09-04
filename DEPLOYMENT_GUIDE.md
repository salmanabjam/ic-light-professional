# 🚀 GitHub Deployment Guide

## Quick Setup for GitHub & Google Colab

Follow these steps to upload your IC Light Professional project to GitHub and use it in Google Colab:

### Step 1: Initialize Git Repository

```bash
# Navigate to project directory
cd "c:\Ai\Projects\BRAINixIDEX\IC Light Google Colab"

# Initialize Git
git init
git add .
git commit -m "Initial commit: IC Light Professional v1.0.0"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click "New repository" or go to [https://github.com/new](https://github.com/new)
3. Repository name: `ic-light-professional` (or your preferred name)
4. Description: `Professional AI Image Relighting with IC Light - Google Colab Ready`
5. Make it **Public** (required for Colab access)
6. Don't initialize with README (we already have one)
7. Click "Create repository"

### Step 3: Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME and YOUR_REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 4: Access in Google Colab

1. **Direct Colab URL**: 
   ```
   https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/IC_Light_Professional_Colab.ipynb
   ```

2. **Or manually**:
   - Open [Google Colab](https://colab.research.google.com/)
   - File → Open notebook → GitHub tab
   - Enter your repository URL
   - Select `IC_Light_Professional_Colab.ipynb`

### Step 5: Run in Colab

1. Make sure you're using GPU runtime:
   - Runtime → Change runtime type → GPU (T4, V100, or A100)
2. Run all cells in sequence
3. Access your application via the public Gradio URL

## 📱 Alternative: Automated Deployment

For automated deployment, run:

```bash
python deploy_to_github.py
```

This script will:
- Initialize Git repository
- Prompt for GitHub username and repository name
- Push everything to GitHub
- Provide direct Colab links

## 🔧 Repository Structure

Your GitHub repository will contain:

```
ic-light-professional/
├── ic_light/                          # Main package
├── IC_Light_Professional_Colab.ipynb  # Colab notebook
├── launch.py                          # Launch script
├── setup.py                          # Package setup
├── requirements.txt                   # Dependencies
├── README.md                         # Documentation
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore file
```

## 🎯 Usage in Google Colab

Once deployed, users can:

1. **Open Colab**: Click the Colab badge in your repository
2. **Run Setup**: Execute all cells to install dependencies
3. **Launch Interface**: Get a public Gradio URL
4. **Start Processing**: Upload images and apply lighting effects

## 🌟 Features Available

- ✅ Professional Gradio 4.0+ interface
- ✅ Latest PyTorch 2.1+ with GPU optimization
- ✅ IC Light models (FC, FBC, FCON)
- ✅ Background removal with BriaRMBG 1.4
- ✅ Advanced analytics with Plotly
- ✅ Batch processing capabilities
- ✅ Real-time performance monitoring

## 🎉 Ready to Deploy!

Your IC Light Professional application is ready for GitHub and Google Colab deployment. Users will be able to access a professional-grade AI image relighting tool with just one click!

---

**Need help?** Check the main README.md for detailed documentation and troubleshooting.
