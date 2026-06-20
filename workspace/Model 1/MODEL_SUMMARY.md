# Model 1 — Wide+Deep+Residual (Daytime Filter)

## Overview

First production model: a flat (single-timestep) feed-forward network predicting absolute future temperatures at +1hr, +2hr, and +3hr horizons. Training data is filtered to daytime-only rows (illuminance > 1400 lux) to reduce noise from nighttime readings where solar sensors are uninformative.

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
| `day_of_year` | Day of year (1–366), raw |
| `time_of_day` | Time in hours (0–24), raw |
| `temperature_delta` | 15-minute rolling linear-regression slope of temperature |
| `temp_lag1` | Temperature from 1 minute ago |
| `humidity_lag1` | Humidity from 1 minute ago |

All features scaled to [0, 1] using per-feature min/max bounds with ±5% padding.

## Architecture

```
Input (12) ─┬─ Dense(16)  ─────────────────────────────────────────┐
             │                                                        │
             └─ Dense(128, relu) → Dropout(0.3) → Dense(64, relu) ──┤  Concatenate
                                               └─ Dense(64) (shortcut)┤
                                                    Add → Dense(32, relu) ┘
                                                                      │
                                                         Concatenate(wide, deep)
                                                                      │
                                                    ┌────────────────┤
                                               Dense(1) × 3 (t1hr, t2hr, t3hr)
```

- Optimizer: Adam lr=1e-5
- Loss: MSE per output
- Early stopping: patience=5
- Targets: absolute temperature (scaled to training min/max with ±5% + 15°C headroom)

## Results (10 runs)

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 (best) | 0.004022 | 0.012984 | 50 | 32.9 KB |
| Run 10 | 0.004244 | 0.013566 | 28 | 32.9 KB |

### Top Feature Importances (Run 1, permutation)

1. `temp_lag1` — 0.0823 (dominant by a large margin)
2. `illuminance` — 0.0083
3. `time_of_day` — 0.0080
4. `solar_radiation` — 0.0078
5. `uv` — 0.0058

`temp_lag1` is massively more important than any other feature, confirming that short-term temperature persistence is the primary predictive signal.

## Key Notes

- Data filter: only illuminance > 1400 lux rows used (daytime training)
- Predicts absolute temperatures, not changes
- `time_of_day` and `day_of_year` are encoded as raw scalars, not cyclically
- Edge TPU quantized model produced (`weather_model_1_best_edgetpu.tflite`)
- 10 independent training runs were performed for robustness estimation
