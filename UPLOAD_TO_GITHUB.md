# 🎯 گام به گام آپلود به GitHub برای Google Colab

## مراحل آپلود پروژه IC Light Professional

### گام 1: ایجاد Repository در GitHub

1. به [GitHub.com](https://github.com) بروید و وارد شوید
2. روی دکمه **"New"** یا **"+"** کلیک کنید
3. **Repository name**: `ic-light-professional` (یا هر نام دیگری)
4. **Description**: `Professional AI Image Relighting - Google Colab Ready`
5. حتماً **Public** انتخاب کنید (برای Colab ضروری است)
6. README, .gitignore و License را انتخاب نکنید (از قبل داریم)
7. **"Create repository"** کلیک کنید

### گام 2: GitHub آدرس کپی کنید

بعد از ایجاد repository، آدرس `.git` را کپی کنید:
```
https://github.com/USERNAME/REPO-NAME.git
```

### گام 3: آپلود فایل‌ها

در PowerShell اجرا کنید:

```powershell
# Remote اضافه کردن (آدرس GitHub خودتان را جایگزین کنید)
git remote add origin https://github.com/USERNAME/REPO-NAME.git

# آپلود فایل‌ها
git push -u origin master
```

### گام 4: تست در Google Colab

1. آدرس مستقیم Colab:
   ```
   https://colab.research.google.com/github/USERNAME/REPO-NAME/blob/master/IC_Light_Professional_Colab.ipynb
   ```

2. یا دستی:
   - [Google Colab](https://colab.research.google.com) باز کنید
   - File → Open notebook → GitHub
   - آدرس repository خود را وارد کنید
   - فایل `IC_Light_Professional_Colab.ipynb` را انتخاب کنید

## 🚀 نتیجه

بعد از آپلود موفق:
- ✅ Repository شما آماده استفاده در Colab
- ✅ کاربران با یک کلیک می‌توانند از برنامه استفاده کنند
- ✅ همه کتابخانه‌های مدرن نصب می‌شود
- ✅ رابط کاربری حرفه‌ای Gradio
- ✅ پردازش تصاویر با هوش مصنوعی IC Light

## 📞 در صورت مشکل

اگر با مشکل مواجه شدید، فایل `DEPLOYMENT_GUIDE.md` را بخوانید یا از طریق Issues در GitHub سوال بپرسید.

---

**آماده آپلود! 🎉**
