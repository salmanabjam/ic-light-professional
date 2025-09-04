# 🚀 IC Light v2 Professional - Updated Google Colab Commands
## دستورات به‌روزرسانی شده برای Google Colab

---

## ⚠️ **SOLUTION FOR "Directory Already Exists" ERROR**
### **راه‌حل برای خطای "دایرکتوری از قبل وجود دارد"**

If you get the error: `fatal: destination path 'ic-light-professional' already exists and is not an empty directory.`

اگر این خطا را دریافت کردید: `fatal: destination path 'ic-light-professional' already exists and is not an empty directory.`

---

## 🔧 **METHOD 1: CLEAN RESTART (RECOMMENDED)**
### **روش ۱: شروع تمیز (پیشنهادی)**

```python
# Clean and restart with fresh installation
!rm -rf ic-light-professional
!git clone https://github.com/salmanabjam/ic-light-professional.git
!cd ic-light-professional && python colab_launcher.py --share
```

**Persian Explanation:**
این دستور ابتدا دایرکتوری قدیمی را حذف کرده، سپس نسخه جدید را دانلود و اجرا می‌کند.

---

## 🔄 **METHOD 2: UPDATE EXISTING INSTALLATION**
### **روش ۲: به‌روزرسانی نصب موجود**

```python
# Update existing installation
!cd ic-light-professional && git pull origin main
!cd ic-light-professional && python colab_launcher.py --share
```

**Persian Explanation:**
این دستور نصب موجود را به‌روزرسانی کرده و اجرا می‌کند.

---

## 🆕 **METHOD 3: FORCE FRESH INSTALLATION**
### **روش ۳: نصب جدید اجباری**

```python
# Force fresh installation with timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%H%M%S")
!rm -rf ic-light-professional ic-light-professional-*
!git clone https://github.com/salmanabjam/ic-light-professional.git ic-light-professional-{timestamp}
!cd ic-light-professional-{timestamp} && python colab_launcher.py --share
```

**Persian Explanation:**
این دستور با timestamp منحصر به فرد، دایرکتوری جدیدی ایجاد می‌کند.

---

## 🎯 **COMPLETE ONE-LINE SOLUTIONS**
### **راه‌حل‌های کامل تک‌خطی**

### **Quick Text-Conditioned System:**
```python
!rm -rf ic-light-professional && git clone https://github.com/salmanabjam/ic-light-professional.git && cd ic-light-professional && python ic_light_v2_professional.py
```

### **Quick Background-Conditioned System:**
```python
!rm -rf ic-light-professional && git clone https://github.com/salmanabjam/ic-light-professional.git && cd ic-light-professional && python ic_light_bg_professional.py
```

### **Complete Auto-Setup with Share Link:**
```python
!rm -rf ic-light-professional && git clone https://github.com/salmanabjam/ic-light-professional.git && cd ic-light-professional && python colab_launcher.py --share
```

### **Both Interfaces (Advanced):**
```python
!rm -rf ic-light-professional && git clone https://github.com/salmanabjam/ic-light-professional.git && cd ic-light-professional && python colab_launcher.py --interface both --share
```

---

## 🛠️ **TROUBLESHOOTING COMMANDS**
### **دستورات عیب‌یابی**

### **Check if directory exists:**
```python
!ls -la | grep ic-light
```

### **Remove all IC Light directories:**
```python
!rm -rf ic-light* IC-Light* IC_Light*
```

### **Check available space:**
```python
!df -h
```

### **Check GPU status:**
```python
!nvidia-smi
```

---

## 📋 **STEP-BY-STEP INSTALLATION GUIDE**
### **راهنمای نصب گام‌به‌گام**

### **Step 1: Clean Environment**
```python
# Clean any existing installations
!rm -rf ic-light-professional IC-Light ic_light*
print("✅ Environment cleaned")
```

### **Step 2: Clone Repository**
```python
# Clone the professional repository
!git clone https://github.com/salmanabjam/ic-light-professional.git
print("✅ Repository cloned")
```

### **Step 3: Navigate and Check**
```python
# Navigate to directory and check files
!cd ic-light-professional && ls -la
print("✅ Files verified")
```

### **Step 4: Launch System**
```python
# Launch with your preferred method
!cd ic-light-professional && python colab_launcher.py --share
```

---

## 🎨 **AVAILABLE INTERFACES**
### **رابط‌های کاربری موجود**

### **1. Text-Conditioned Relighting (Default)**
- ✅ 20+ preset prompts
- ✅ Professional quality controls
- ✅ Multi-sample generation
- ✅ Bilingual interface

```python
!cd ic-light-professional && python ic_light_v2_professional.py
```

### **2. Background-Conditioned Relighting**
- ✅ Background image upload
- ✅ Gradient generation
- ✅ Advanced background processing
- ✅ Professional controls

```python
!cd ic-light-professional && python ic_light_bg_professional.py
```

### **3. Complete Launcher (All Options)**
- ✅ Interface selection
- ✅ Auto-setup
- ✅ Model downloading
- ✅ Error handling

```python
!cd ic-light-professional && python colab_launcher.py --share
```

---

## ⚡ **PERFORMANCE OPTIMIZATION**
### **بهینه‌سازی عملکرد**

### **For Better Performance:**
```python
# Set environment variables for optimal performance
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Then run your preferred command
!rm -rf ic-light-professional && git clone https://github.com/salmanabjam/ic-light-professional.git && cd ic-light-professional && python colab_launcher.py --share
```

---

## 🌟 **FEATURES SUMMARY**
### **خلاصه ویژگی‌ها**

Your IC Light v2 Professional system includes:

✅ **Complete IC Light v2 functionality**  
✅ **20+ professional preset prompts**  
✅ **Advanced background removal (BRIA RMBG 1.4)**  
✅ **Multi-sample generation (1-4 samples)**  
✅ **Professional quality controls**  
✅ **Bilingual Persian/English interface**  
✅ **Google Colab optimized**  
✅ **Auto-model download**  
✅ **Professional error handling**  

سیستم شما شامل تمامی ویژگی‌های IC Light v2 با کیفیت حرفه‌ای است.

---

## 📞 **SUPPORT**
### **پشتیبانی**

If you encounter any issues:
اگر با مشکلی مواجه شدید:

1. **Try Method 1 (Clean Restart)** first
2. **Check GPU availability** with `!nvidia-smi`
3. **Verify internet connection**
4. **Check available disk space** with `!df -h`

**Repository:** https://github.com/salmanabjam/ic-light-professional

---

**🎉 Ready to create amazing relit images! / آماده خلق تصاویر شگفت‌انگیز!**
