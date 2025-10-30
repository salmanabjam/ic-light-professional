# Trading Indicators for TradingView

This directory contains professional Pine Script indicators for TradingView.

## 📊 Available Indicators

### Advanced Trendline Indicator v6

**File**: `advanced_trendline_indicator_v6.pine`

A sophisticated trendline detection indicator that automatically identifies and draws the most important support and resistance lines on your charts.

#### Features

- ✅ **Automatic Detection**: Identifies support/resistance automatically
- ✅ **Dynamic & Static Lines**: Both trending and horizontal lines
- ✅ **Modular Architecture**: 11 specialized modules for easy extension
- ✅ **Intelligent Ranking**: Shows only the strongest trendlines
- ✅ **Fully Customizable**: Colors, styles, widths, and more
- ✅ **Pine Script v6**: Latest version with full type safety
- ✅ **Performance Optimized**: Handles up to 500 lines efficiently

#### Quick Start

1. **Copy the code** from `advanced_trendline_indicator_v6.pine`
2. **Open TradingView** and press `ALT + E` (Pine Editor)
3. **Create new indicator** and paste the code
4. **Save and add to chart**

#### Documentation

- **Full Documentation**: [TRENDLINE_INDICATOR_DOCS.md](TRENDLINE_INDICATOR_DOCS.md)
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Language**: Bilingual (English & Persian/فارسی)

#### Configuration

```
Default Settings:
- Pivot Detection: 10 left, 10 right bars
- Min Strength: 3 pivot touches
- Max Lines: 10 per type
- Colors: Red (resistance), Green (support)
```

#### Use Cases

| Trading Style | Timeframe | Settings |
|--------------|-----------|----------|
| Day Trading | 15m - 1h | Pivot: 5-7, Strength: 2-3 |
| Swing Trading | 4h - D | Pivot: 10-15, Strength: 3-4 |
| Position Trading | D - W | Pivot: 15-20, Strength: 4-5 |

## 🎯 How to Use

### Step 1: Installation

```bash
1. Open TradingView
2. Go to Pine Editor (ALT + E)
3. Click "New" → "Indicator"
4. Copy/paste the code
5. Click "Add to Chart"
```

### Step 2: Configuration

Adjust settings based on your trading style:

- **Short-term traders**: Lower pivot bars, higher max lines
- **Long-term traders**: Higher pivot bars, lower max lines
- **Volatile markets**: Lower strength threshold
- **Stable markets**: Higher strength threshold

### Step 3: Interpretation

- **Red lines** = Resistance levels (potential reversal points)
- **Green lines** = Support levels (potential bounce points)
- **Solid lines** = Static/horizontal levels
- **Dashed lines** = Dynamic/sloped trendlines

## 📐 Module Structure

The indicator is built with a modular architecture:

```
Module 1:  Configuration & Settings
Module 2:  Utility Functions
Module 3:  Pivot Detection
Module 4:  Data Storage
Module 5:  Trendline Type Definition
Module 6:  Trendline Manager
Module 7:  Drawing Functions
Module 8:  Detection Algorithm
Module 9:  Main Execution
Module 10: Visual Enhancements
Module 11: Alerts (Extension Point)
```

## 🔧 Customization Examples

### Change Colors

```pine
// In MODULE 1: Configuration
resistanceColor = input.color(color.blue, "Resistance Color", group="Visual Settings")
supportColor = input.color(color.orange, "Support Color", group="Visual Settings")
```

### Add Alerts

```pine
// In MODULE 11: Alerts
priceAboveResistance = close > line.get_y1(resistanceLines[0])
alertcondition(priceAboveResistance, "Breakout", "Price broke resistance!")
```

### Modify Detection Logic

```pine
// In MODULE 8: Main Algorithm
// Customize f_findBestTrendlines() function
// Add your own filtering or ranking logic
```

## 📚 Documentation Files

| File | Purpose | Language |
|------|---------|----------|
| `advanced_trendline_indicator_v6.pine` | Main indicator code | Pine Script v6 |
| `TRENDLINE_INDICATOR_DOCS.md` | Complete documentation | English + Persian |
| `QUICK_REFERENCE.md` | Quick start guide | English + Persian |
| `README.md` | This overview | English |

## 🌟 Key Algorithms

### Pivot Detection
- Uses TradingView's built-in `ta.pivothigh()` and `ta.pivotlow()`
- Configurable left/right bars for sensitivity
- Stores historical pivots for analysis

### Trendline Strength Calculation
- Counts pivot touches within tolerance (ATR-based)
- Ranks trendlines by number of confirming pivots
- Filters weak lines automatically

### Line Management
- Custom TrendlineManager type for efficient handling
- Automatic cleanup of old lines
- Dynamic line limit enforcement

## 💡 Tips & Best Practices

### For Better Results

1. **Start with defaults** - Adjust based on observation
2. **Match timeframe** - Higher TF = larger pivot bars
3. **Reduce clutter** - Limit max lines if chart is busy
4. **Combine with volume** - Validate breakouts
5. **Wait for confirmation** - Don't trade on lines alone

### Common Mistakes to Avoid

1. ❌ Using too many lines (max 5-10 recommended)
2. ❌ Not adjusting for timeframe
3. ❌ Ignoring line strength (touchPoints)
4. ❌ Trading against strong trendlines
5. ❌ Mixing too many other indicators

## 🔒 Code Quality

- ✅ **Type Safe**: Full Pine Script v6 type annotations
- ✅ **Well Commented**: Every function documented
- ✅ **Modular**: Easy to extend and maintain
- ✅ **Optimized**: Efficient array operations
- ✅ **Clean Code**: Follows Pine Script best practices

## 📈 Performance

- **Line Limit**: 500 maximum (Pine Script constraint)
- **Execution Speed**: Fast (optimized algorithms)
- **Memory Usage**: Efficient (capped arrays)
- **Compatibility**: All TradingView plans

## 🌐 Language Support

All documentation available in:
- **English** 🇬🇧🇺🇸
- **Persian/فارسی** 🇮🇷

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 6.0.0 | 2025-10-30 | Initial release with modular architecture |

## 🤝 Contributing

To extend this indicator:

1. Add new modules after Module 11
2. Follow the existing naming convention
3. Document all functions with `@function` tags
4. Test thoroughly before using in live trading

## ⚠️ Disclaimer

This indicator is for educational and informational purposes only. It is not financial advice. Always:

- Do your own research
- Use proper risk management
- Test on demo account first
- Never risk more than you can afford to lose

## 📧 Support

For questions or issues:
- Check the documentation files
- Review code comments
- Consult TradingView Pine Script v6 documentation

---

## فارسی (Persian)

### اندیکاتور پیشرفته ترند لاین نسخه 6

این یک اندیکاتور حرفه‌ای برای شناسایی خودکار خطوط روند است.

#### ویژگی‌ها

- تشخیص خودکار حمایت و مقاومت
- خطوط دینامیک و استاتیک
- معماری ماژولار برای توسعه آسان
- رتبه‌بندی هوشمند خطوط روند
- کاملاً قابل سفارشی‌سازی

#### نصب سریع

1. کد را از فایل کپی کنید
2. TradingView را باز کرده و ALT + E بزنید
3. اندیکاتور جدید بسازید و کد را بچسبانید
4. ذخیره کنید و به چارت اضافه کنید

#### مستندات

- مستندات کامل در فایل `TRENDLINE_INDICATOR_DOCS.md`
- راهنمای سریع در فایل `QUICK_REFERENCE.md`

---

**Made with ❤️ for TradingView Community**
**ساخته شده با ❤️ برای جامعه TradingView**
