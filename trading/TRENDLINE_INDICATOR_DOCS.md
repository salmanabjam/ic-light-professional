# Advanced Trendline Indicator v6 - Documentation
# اندیکاتور پیشرفته ترند لاین نسخه 6 - مستندات

## English Documentation

### Overview
The Advanced Trendline Indicator v6 is a sophisticated Pine Script indicator that automatically detects and draws the most important trendlines on your TradingView charts. It features a fully modular architecture designed for easy extension and customization.

### Key Features

1. **Automatic Trendline Detection**
   - Automatically identifies significant support and resistance levels
   - Draws trendlines based on pivot points
   - Calculates trendline strength based on number of touches

2. **Dynamic & Static Trendlines**
   - Dynamic lines: Lines with significant slope (trending)
   - Static lines: Horizontal or near-horizontal support/resistance
   - Toggle visibility of each type independently

3. **Fully Modular Architecture**
   - 11 distinct modules for easy maintenance
   - Each module handles specific functionality
   - Easy to extend with new features

4. **Intelligent Algorithm**
   - Ranks trendlines by strength (number of pivot touches)
   - Shows only the most significant lines
   - Automatically filters weak trendlines

5. **Rich Customization**
   - Configurable pivot detection parameters
   - Adjustable line colors, styles, and widths
   - Flexible extension options (none, left, right, both)
   - Control maximum number of lines displayed

### Installation

1. Open TradingView
2. Go to Pine Editor (ALT + E)
3. Click "New" to create a new indicator
4. Copy and paste the entire code from `advanced_trendline_indicator_v6.pine`
5. Click "Add to Chart"

### Configuration Parameters

#### Pivot Detection
- **Pivot Left Bars** (default: 10): Number of bars to the left for pivot detection
- **Pivot Right Bars** (default: 10): Number of bars to the right for pivot detection
- **Max Stored Pivots** (default: 20): Maximum number of pivot points to store

#### Trendline Display
- **Show Resistance Lines**: Toggle resistance line visibility
- **Show Support Lines**: Toggle support line visibility
- **Show Dynamic Lines**: Toggle dynamic (sloped) line visibility
- **Show Static Lines**: Toggle static (horizontal) line visibility

#### Visual Settings
- **Resistance Color**: Color for resistance lines (default: red)
- **Support Color**: Color for support lines (default: green)
- **Dynamic Line Style**: Style for dynamic lines (Solid/Dashed/Dotted)
- **Static Line Width**: Width for static lines (1-5)
- **Dynamic Line Width**: Width for dynamic lines (1-5)

#### Advanced Settings
- **Min Trendline Strength** (default: 3): Minimum pivot touches required
- **Extend Lines**: How to extend lines (None/Right/Left/Both)
- **Max Lines Per Type** (default: 10): Maximum lines per category

### Module Structure

1. **Configuration & Settings**: User inputs and parameters
2. **Utility Functions**: Helper functions for conversions
3. **Pivot Detection**: Identifies swing highs and lows
4. **Data Storage**: Manages pivot point history
5. **Trendline Type**: Custom type definitions
6. **Trendline Manager**: Manages line collections
7. **Drawing Functions**: Creates and styles lines
8. **Detection Algorithm**: Finds optimal trendlines
9. **Main Execution**: Orchestrates the analysis
10. **Visual Enhancements**: Displays info and markers
11. **Alerts**: Extension point for alert conditions

### How to Extend

#### Adding a New Module
```pine
// =============================================================================
// MODULE 12: Your New Feature
// =============================================================================

// Add your custom functionality here
f_yourNewFunction() =>
    // Implementation
    na
```

#### Adding Alert Conditions
```pine
// In MODULE 11: Alerts section
priceAboveResistance = close > line.get_y1(resistanceLines[0])
alertcondition(priceAboveResistance, "Price Above Resistance", "Price crossed resistance!")
```

#### Customizing Detection Logic
Modify the `f_findBestTrendlines()` function in MODULE 8 to implement your own trendline detection algorithm.

### Best Practices

1. **Start with default settings** and adjust based on your timeframe
2. **Higher timeframes**: Use larger pivot detection values (15-20)
3. **Lower timeframes**: Use smaller pivot detection values (5-10)
4. **Reduce visual clutter**: Decrease max lines per type if chart is too busy
5. **Strength threshold**: Increase for only the strongest trendlines

### Troubleshooting

**Issue**: Too many lines on chart
- **Solution**: Increase "Min Trendline Strength" or decrease "Max Lines Per Type"

**Issue**: No lines appearing
- **Solution**: Decrease "Min Trendline Strength" or wait for more pivot points to form

**Issue**: Lines not extending correctly
- **Solution**: Check "Extend Lines" setting

---

## مستندات فارسی

### معرفی کلی
اندیکاتور پیشرفته ترند لاین نسخه 6 یک اندیکاتور پیچیده Pine Script است که به طور خودکار مهم‌ترین خطوط روند را روی چارت TradingView شما شناسایی و رسم می‌کند. این اندیکاتور دارای معماری کاملاً ماژولار است که برای توسعه و سفارشی‌سازی آسان طراحی شده است.

### ویژگی‌های کلیدی

1. **تشخیص خودکار خطوط روند**
   - شناسایی خودکار سطوح مهم حمایت و مقاومت
   - رسم خطوط روند بر اساس نقاط پیووت
   - محاسبه قدرت خط روند بر اساس تعداد برخوردها

2. **خطوط روند دینامیک و استاتیک**
   - خطوط دینامیک: خطوط با شیب قابل توجه (روند دار)
   - خطوط استاتیک: حمایت/مقاومت افقی یا نزدیک به افقی
   - امکان نمایش/عدم نمایش هر نوع به صورت مستقل

3. **معماری کاملاً ماژولار**
   - 11 ماژول مجزا برای نگهداری آسان
   - هر ماژول عملکرد خاصی را مدیریت می‌کند
   - امکان افزودن ویژگی‌های جدید به راحتی

4. **الگوریتم هوشمند**
   - رتبه‌بندی خطوط روند بر اساس قدرت (تعداد برخورد با پیووت)
   - نمایش فقط مهم‌ترین خطوط
   - فیلتر خودکار خطوط ضعیف

5. **سفارشی‌سازی غنی**
   - پارامترهای قابل تنظیم برای تشخیص پیووت
   - رنگ، استایل و ضخامت قابل تنظیم
   - گزینه‌های انعطاف‌پذیر برای امتداد خطوط
   - کنترل حداکثر تعداد خطوط نمایش داده شده

### نحوه نصب

1. TradingView را باز کنید
2. به Pine Editor بروید (ALT + E)
3. روی "New" کلیک کنید تا یک اندیکاتور جدید بسازید
4. کل کد را از فایل `advanced_trendline_indicator_v6.pine` کپی و پیست کنید
5. روی "Add to Chart" کلیک کنید

### پارامترهای پیکربندی

#### تشخیص پیووت
- **Pivot Left Bars** (پیش‌فرض: 10): تعداد کندل‌های سمت چپ برای تشخیص پیووت
- **Pivot Right Bars** (پیش‌فرض: 10): تعداد کندل‌های سمت راست برای تشخیص پیووت
- **Max Stored Pivots** (پیش‌فرض: 20): حداکثر تعداد نقاط پیووت ذخیره شده

#### نمایش خطوط روند
- **Show Resistance Lines**: نمایش/عدم نمایش خطوط مقاومت
- **Show Support Lines**: نمایش/عدم نمایش خطوط حمایت
- **Show Dynamic Lines**: نمایش/عدم نمایش خطوط دینامیک (شیب‌دار)
- **Show Static Lines**: نمایش/عدم نمایش خطوط استاتیک (افقی)

#### تنظیمات بصری
- **Resistance Color**: رنگ خطوط مقاومت (پیش‌فرض: قرمز)
- **Support Color**: رنگ خطوط حمایت (پیش‌فرض: سبز)
- **Dynamic Line Style**: استایل خطوط دینامیک (Solid/Dashed/Dotted)
- **Static Line Width**: ضخامت خطوط استاتیک (1-5)
- **Dynamic Line Width**: ضخامت خطوط دینامیک (1-5)

#### تنظیمات پیشرفته
- **Min Trendline Strength** (پیش‌فرض: 3): حداقل برخورد پیووت مورد نیاز
- **Extend Lines**: نحوه امتداد خطوط (None/Right/Left/Both)
- **Max Lines Per Type** (پیش‌فرض: 10): حداکثر خطوط در هر دسته

### ساختار ماژول‌ها

1. **پیکربندی و تنظیمات**: ورودی‌های کاربر و پارامترها
2. **توابع کمکی**: توابع کمکی برای تبدیل‌ها
3. **تشخیص پیووت**: شناسایی سقف‌ها و کف‌های نوسانی
4. **ذخیره‌سازی داده**: مدیریت تاریخچه نقاط پیووت
5. **نوع خط روند**: تعاریف نوع سفارشی
6. **مدیر خط روند**: مدیریت مجموعه خطوط
7. **توابع رسم**: ایجاد و استایل‌دهی خطوط
8. **الگوریتم تشخیص**: یافتن بهترین خطوط روند
9. **اجرای اصلی**: هماهنگی تحلیل
10. **بهبودهای بصری**: نمایش اطلاعات و نشانگرها
11. **هشدارها**: نقطه توسعه برای شرایط هشدار

### نحوه توسعه

#### افزودن ماژول جدید
```pine
// =============================================================================
// MODULE 12: ویژگی جدید شما
// =============================================================================

// عملکرد سفارشی خود را اینجا اضافه کنید
f_yourNewFunction() =>
    // پیاده‌سازی
    na
```

#### افزودن شرایط هشدار
```pine
// در بخش MODULE 11: Alerts
priceAboveResistance = close > line.get_y1(resistanceLines[0])
alertcondition(priceAboveResistance, "قیمت بالای مقاومت", "قیمت از مقاومت عبور کرد!")
```

#### سفارشی‌سازی منطق تشخیص
تابع `f_findBestTrendlines()` در MODULE 8 را ویرایش کنید تا الگوریتم تشخیص خط روند خود را پیاده‌سازی کنید.

### بهترین روش‌ها

1. **با تنظیمات پیش‌فرض شروع کنید** و بر اساس تایم‌فریم تنظیم کنید
2. **تایم‌فریم‌های بالاتر**: از مقادیر بزرگ‌تر برای تشخیص پیووت استفاده کنید (15-20)
3. **تایم‌فریم‌های پایین‌تر**: از مقادیر کوچک‌تر برای تشخیص پیووت استفاده کنید (5-10)
4. **کاهش شلوغی بصری**: حداکثر خطوط در هر نوع را کاهش دهید
5. **آستانه قدرت**: برای نمایش فقط قوی‌ترین خطوط روند افزایش دهید

### عیب‌یابی

**مشکل**: خطوط زیادی روی چارت
- **راه‌حل**: "Min Trendline Strength" را افزایش یا "Max Lines Per Type" را کاهش دهید

**مشکل**: هیچ خطی ظاهر نمی‌شود
- **راه‌حل**: "Min Trendline Strength" را کاهش دهید یا منتظر تشکیل نقاط پیووت بیشتر بمانید

**مشکل**: خطوط به درستی امتداد نمی‌یابند
- **راه‌حل**: تنظیم "Extend Lines" را بررسی کنید

---

## Technical Specifications | مشخصات فنی

- **Pine Script Version**: v6 (Latest) | نسخه 6 (آخرین نسخه)
- **Max Lines**: 500 | حداکثر خطوط: 500
- **Chart Type**: Overlay | نوع چارت: همپوشانی
- **Modules**: 11 specialized modules | 11 ماژول تخصصی
- **Type Safety**: Full type annotations | امنیت نوع: با تمام حاشیه‌نویسی‌ها

## Version History | تاریخچه نسخه

- **v6.0.0**: Initial release with modular architecture | انتشار اولیه با معماری ماژولار

## Support | پشتیبانی

For issues or questions, please refer to the code comments or TradingView Pine Script documentation.

برای مشکلات یا سوالات، لطفاً به توضیحات کد یا مستندات Pine Script در TradingView مراجعه کنید.
