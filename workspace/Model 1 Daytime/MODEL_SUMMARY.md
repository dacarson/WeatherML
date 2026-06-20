# Model 1 Daytime — Wide+Deep+Residual (No Illuminance Filter)

## Overview

Variant of Model 1 using the same architecture and feature set but without the illuminance > 1400 lux daytime filter. This exposes the model to nighttime samples, which may include near-zero solar readings and a different temperature dynamic. Results are nearly identical to the filtered Model 1, suggesting the daytime filter had minimal impact on accuracy.

## Features (12)

| Feature | Description |
|---------|-------------|
| `illuminance` | Ambient light level (lux) |
| `solar_radiation` | Solar irradiance (W/m²) |
| `uv` | UV index |
| `relative_humidity` | Current humidity (%) |
| `station_pressure` | Station pressure (hPa) |
| `wind_avg` | Average wind speed |
| `wind_gust` | Wind gust speed |
| `day_of_year` | Day of year (1–366), raw scalar |
| `time_of_day` | Time in hours (0–24), raw scalar |
| `temperature_delta` | 15-minute rolling linear-regression slope of temperature |
| `temp_lag1` | Temperature from 1 minute ago |
| `humidity_lag1` | Humidity from 1 minute ago |

All features scaled to [0, 1] using per-feature min/max bounds with ±5% padding.

## Architecture

Identical to Model 1:

```
Input (12) ─┬─ Dense(16)  ────────────────────────────────────────┐
             │                                                       │
             └─ Dense(128, relu) → Dropout(0.3) → Dense(64, relu) ─┤  Concatenate
                                             └─ Dense(64) (shortcut)┤
                                                  Add → Dense(32, relu) ┘
                                                                     │
                                                        Concatenate(wide, deep)
                                                                     │
                                                   Dense(1) × 3 (t1hr, t2hr, t3hr)
```

- Optimizer: Adam lr=1e-5
- Loss: MSE per output
- Early stopping: patience=5
- Targets: absolute temperature (scaled to training min/max range)

## Results (10 runs)

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 (best) | 0.004022 | 0.012984 | 50 | 32.9 KB |
| Run 10 | 0.004244 | 0.013566 | 28 | 32.9 KB |

Results are essentially identical to the filtered Model 1, indicating the daytime filter did not meaningfully change accuracy for this architecture.

### Top Feature Importances (Run 1, permutation)

1. `temp_lag1` — 0.0823
2. `solar_radiation` — 0.0197
3. `illuminance` — 0.0147
4. `time_of_day` — 0.0085
5. `uv` — 0.0060

## Key Notes

- No illuminance threshold filter applied (full 24-hour data used)
- Same architecture and features as Model 1
- Predicts absolute temperatures, not changes
- Edge TPU quantized model produced (`weather_model_1_best_edgetpu.tflite`)
- Comparison with Model 1 shows that the daytime filter had negligible effect on validation loss
