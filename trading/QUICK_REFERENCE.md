# Quick Reference - Advanced Trendline Indicator v6
# راهنمای سریع - اندیکاتور پیشرفته ترند لاین نسخه 6

## Quick Start | شروع سریع

### Installation Steps | مراحل نصب
```
1. Open TradingView → باز کردن TradingView
2. Press ALT + E (Pine Editor) → فشار دادن ALT + E (ویرایشگر پاین)
3. New Indicator → اندیکاتور جدید
4. Paste Code → چسباندن کد
5. Save + Add to Chart → ذخیره + افزودن به چارت
```

## Default Settings | تنظیمات پیش‌فرض

| Setting | Value | Purpose |
|---------|-------|---------|
| Pivot Left/Right Bars | 10 | Pivot detection sensitivity |
| Max Stored Pivots | 20 | Memory for pivot points |
| Min Trendline Strength | 3 | Quality filter |
| Max Lines Per Type | 10 | Display limit |

## Common Use Cases | موارد استفاده رایج

### 1. Day Trading (15m-1h charts)
```
Pivot Left/Right: 5-7
Min Strength: 2-3
Max Lines: 15
```

### 2. Swing Trading (4h-D charts)
```
Pivot Left/Right: 10-15
Min Strength: 3-4
Max Lines: 10
```

### 3. Long-term Analysis (D-W charts)
```
Pivot Left/Right: 15-20
Min Strength: 4-5
Max Lines: 5-8
```

## Color Coding | کدگذاری رنگ

- 🔴 **Red Lines** = Resistance (مقاومت)
- 🟢 **Green Lines** = Support (حمایت)
- ⚫ **Solid Lines** = Static/Horizontal (افقی/استاتیک)
- ⚡ **Dashed Lines** = Dynamic/Sloped (شیب‌دار/دینامیک)

## Troubleshooting | عیب‌یابی

### Problem: Too Many Lines
**Solution:**
- Increase "Min Trendline Strength" to 4-5
- Decrease "Max Lines Per Type" to 5-7

### Problem: No Lines Showing
**Solution:**
- Decrease "Min Trendline Strength" to 2
- Increase "Max Stored Pivots" to 30
- Wait for more price action

### Problem: Lines Too Weak
**Solution:**
- Increase "Min Trendline Strength"
- Adjust pivot detection bars

## Tips & Tricks | نکات و ترفندها

### ✅ Do's
- Start with defaults
- Adjust based on timeframe
- Combine with volume analysis
- Use info table for statistics

### ❌ Don'ts
- Don't use too many lines
- Don't ignore line strength
- Don't use on very low timeframes (<5m)
- Don't mix with too many other indicators

## Keyboard Shortcuts | میانبرهای صفحه‌کلید

| Action | Shortcut |
|--------|----------|
| Open Pine Editor | ALT + E |
| Save Code | CTRL + S |
| Add to Chart | Click button |
| Remove from Chart | Right-click indicator |

## Extension Points | نقاط توسعه

### Module 11: Alerts
Add custom alert conditions:
```pine
// Price crossing resistance
alertcondition(ta.crossover(close, resistance),
    "Resistance Break", "Price broke resistance!")
```

### Module 8: Custom Algorithm
Modify detection logic for specific needs:
```pine
// Add your custom trendline detection
// Modify f_findBestTrendlines() function
```

### New Modules
Add after Module 11:
```pine
// =============================================================================
// MODULE 12: Your Feature
// =============================================================================
```

## Performance Tips | نکات عملکرد

- **Reduce lag**: Lower max lines and pivots
- **More accuracy**: Increase pivot detection bars
- **Cleaner chart**: Use only support OR resistance
- **Better signals**: Increase min strength

## Info Table Explained | توضیح جدول اطلاعات

Top-right table shows:
- **Resistance Lines**: Current red lines displayed
- **Support Lines**: Current green lines displayed
- **Pivot Highs**: Stored swing highs
- **Pivot Lows**: Stored swing lows

## Best Timeframes | بهترین تایم‌فریم‌ها

| Timeframe | Recommended | Notes |
|-----------|-------------|-------|
| 1m-5m | ❌ Not recommended | Too noisy |
| 15m-1h | ✅ Good | Day trading |
| 4h-D | ✅✅ Excellent | Swing trading |
| W-M | ✅ Good | Long-term |

## Parameter Optimization | بهینه‌سازی پارامترها

### For Volatile Markets
```
Pivot Bars: Lower (5-7)
Min Strength: Lower (2)
Max Lines: Higher (15)
```

### For Stable Markets
```
Pivot Bars: Higher (15-20)
Min Strength: Higher (4-5)
Max Lines: Lower (5-8)
```

## Common Patterns | الگوهای رایج

### Strong Trendline
- ✅ Multiple touches (4+)
- ✅ Clear bounces
- ✅ Consistent slope

### Weak Trendline
- ❌ Few touches (2)
- ❌ Price breaks through
- ❌ Inconsistent angle

## Support Resources | منابع پشتیبانی

1. **Full Documentation**: TRENDLINE_INDICATOR_DOCS.md
2. **Pine Script v6 Docs**: https://www.tradingview.com/pine-script-docs/v6/
3. **Code Comments**: Inline documentation in script

## Updates & Versioning | به‌روزرسانی‌ها و نسخه‌بندی

- Current Version: **6.0.0**
- Last Updated: **2025-10-30**
- Compatible with: **Pine Script v6**

---

## Quick Command Reference | مرجع سریع دستورات

```pine
// Core Functions
f_detectPivotHigh()     // Detect swing highs
f_detectPivotLow()      // Detect swing lows
f_drawTrendline()       // Draw line
f_findBestTrendlines()  // Find optimal lines

// Manager Methods
manager.addLine()       // Add line to collection
manager.clearAll()      // Remove all lines

// Utility Functions
f_getLineStyle()        // Get line style
f_getExtension()        // Get extension type
f_calculateSlope()      // Calculate slope
f_isValidTrendline()    // Check validity
```

---

**Remember**: This is a tool to help identify trends. Always confirm with other analysis methods!

**به یاد داشته باشید**: این یک ابزار کمکی برای شناسایی روندهاست. همیشه با روش‌های دیگر تأیید کنید!
