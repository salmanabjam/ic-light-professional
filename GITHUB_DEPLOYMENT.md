# 🚀 GitHub Deployment Guide - IC Light Professional

## مراحل آپلود به GitHub

### مرحله 1: ایجاد Repository در GitHub
1. به وب‌سایت [GitHub.com](https://github.com) بروید
2. روی "New Repository" کلیک کنید  
3. نام repository را وارد کنید: `ic-light-professional`
4. توضیحات: `Advanced IC Light image relighting for Google Colab`
5. Repository را **Public** بگذارید
6. **تیک "Initialize this repository with a README" را نزنید**
7. روی "Create Repository" کلیک کنید

### مرحله 2: آپلود فایل‌های پروژه
پس از ایجاد repository، دو گزینه دارید:

#### گزینه A: استفاده از Git Command Line (پیشنهادی)
```bash
# اگر Git نصب است
git remote add origin https://github.com/YOUR_USERNAME/ic-light-professional.git
git branch -M main  
git push -u origin main
```

#### گزینه B: آپلود از طریق وب‌سایت GitHub
1. در صفحه repository جدید، روی "uploading an existing file" کلیک کنید
2. تمام فایل‌های پروژه را drag & drop کنید
3. Commit message بنویسید: "Add IC Light Professional application"
4. روی "Commit changes" کلیک کنید

### مرحله 3: بروزرسانی README
پس از آپلود، فایل README.md را ویرایش کنید و `YOUR_GITHUB_USERNAME` را با username واقعی خود جایگزین کنید.

## مراحل اجرا در Google Colab

### مرحله 1: باز کردن در Colab
1. به repository GitHub خود بروید
2. روی فایل `IC_Light_Professional_Colab.ipynb` کلیک کنید  
3. روی دکمه "Open in Colab" کلیک کنید

### مرحله 2: اجرای Notebook
1. در Google Colab، Runtime > Change runtime type
2. Hardware accelerator را روی "GPU" قرار دهید
3. تمام cell ها را به ترتیب اجرا کنید (Ctrl+F9)
4. منتظر بمانید تا interface آماده شود

### مرحله 3: استفاده از برنامه
1. لینک عمومی ایجاد می‌شود (مثل: https://xxxxx.gradio.live)
2. تصویر خود را آپلود کنید
3. prompt نوشته و تنظیمات را اعمال کنید
4. روی "Process Image" کلیک کنید

## 🎯 مزایای این Implementation

### ✅ آماده برای Production
- **Professional Package Structure**: ساختار حرفه‌ای Python
- **Latest Libraries**: آخرین نسخه کتابخانه‌ها (PyTorch 2.1+, Gradio 4.0+)
- **Modern UI**: رابط کاربری مدرن با analytics
- **GPU Optimized**: بهینه‌سازی برای GPU های Google Colab

### ✅ قابلیت‌های پیشرفته
- **Real-time Analytics**: نمایش metrics با Plotly
- **Background Removal**: حذف پس‌زمینه با BriaRMBG
- **Batch Processing**: پردازش چندین تصویر همزمان
- **Multiple Models**: سه variant مختلف (FC, FBC, FCON)

### ✅ Easy Deployment
- **One-Click Colab**: یک کلیک در Google Colab
- **Public URL**: لینک عمومی برای اشتراک‌گذاری
- **Error Handling**: مدیریت خطاها و troubleshooting
- **Complete Documentation**: مستندات کامل فارسی و انگلیسی

## 🔧 تنظیمات مهم

### برای Google Colab
```python
# در config.ini
[COLAB]
public_url = true
server_name = 0.0.0.0
server_port = 7860
enable_cpu_offload = true
max_resolution = 768
```

### برای استفاده محلی
```python
# در config.ini  
[LOCAL]
public_url = false
server_name = 127.0.0.1
server_port = 7860
max_resolution = 1024
```

## 📊 System Requirements

### Google Colab Free
- ✅ T4 GPU (16GB VRAM)
- ✅ 12GB RAM
- ✅ تمام قابلیت‌ها فعال

### Google Colab Pro/Pro+
- 🚀 V100/A100 GPU (40GB+ VRAM)  
- 🚀 High-RAM runtime
- 🚀 بهترین عملکرد

## 🎉 آماده برای استفاده!

پس از آپلود به GitHub، برنامه شما آماده است:

1. **GitHub Repository**: کد کامل با مستندات
2. **Google Colab Ready**: یک کلیک برای اجرا
3. **Professional Interface**: رابط حرفه‌ای با analytics
4. **Latest Technology**: آخرین کتابخانه‌ها و بهینه‌سازی‌ها

---

**🌟 IC Light Professional - Ready to Transform Your Images! 🌟**
