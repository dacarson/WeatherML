# Model 1 Diffs — Wide+Deep+Residual with Temperature Differences

## Overview

Key evolution from Model 1: instead of predicting absolute future temperatures, this model predicts temperature *changes* from the current reading (temp_diff_1hr/2hr/3hr = future_temp − current_temp). This framing is physically more natural — the model learns how much the temperature will rise or fall, rather than an absolute value. Also introduces extended lag features (30/60/120 min), higher-order cyclical time encoding, multi-variable delta features, and Numba-accelerated rolling slope computation for Raspberry Pi deployment.

## Features (28–31, depending on optional sensors)

| Feature | Description |
|---------|-------------|
| `illuminance_delta` | 15-min rolling slope of illuminance |
| `solar_radiation_delta` | 15-min rolling slope of solar radiation |
| `uv` | UV index |
| `wind_avg` | Average wind speed |
| `wind_gust` | Wind gust speed |
| `day_of_year_sin/cos` | Seasonal cyclical encoding |
| `time_of_day_sin/cos` | Diurnal cyclical encoding (first harmonic) |
| `time_of_day_sin2/cos2` | Diurnal cyclical encoding (second harmonic) |
| `temperature_delta` | 15-min rolling slope of temperature |
| `pressure_delta` | 15-min rolling slope of station pressure |
| `humidity_delta` | 15-min rolling slope of relative humidity |
| `temp_lag30/60/120` | Temperature 30, 60, 120 minutes ago |
| `humidity_lag30/60/120` | Humidity 30, 60, 120 minutes ago |
| `wind_avg_lag30` | Wind avg 30 min ago |
| `wind_gust_lag30` | Wind gust 30 min ago |
| `uv_lag30` | UV 30 min ago |
| `pressure_lag30` | Pressure 30 min ago |
| `wind_direction_sin/cos` | (optional) Wind direction cyclical encoding |
| `wind_lull` | (optional) Minimum wind speed |
| `rain_accumulated` | (optional) Accumulated precipitation |

## Architecture

Wide+deep+residual with an Edge TPU-compatible feature interaction path:

```
Input (28+) ─┬─ interaction_embed(Dense(16,relu)) → square(Multiply) → Concat → Dense(32,relu)
              ├─ Dense(16) (wide)
              └─ Dense(128, relu) → Dropout(0.3) → Dense(64, relu) ─┐
                                              └─ Dense(64) (shortcut) ┤ Add → Dense(32, relu)
                                                                      │
                                           Concatenate(wide, deep, interaction)
                                                                      │
                                               Dense(1) × 3 (diff_1hr, diff_2hr, diff_3hr)
```

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Batch size: 256
- Early stopping: patience=5
- Targets: temperature differences scaled to [−1, 1] using global min/max ±2°C padding
- Numba-accelerated rolling slopes for RPi5 performance

## Results (10 runs)

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 (best) | 0.010793 | 0.021837 | 77 | 39.4 KB |
| Run 10 | 0.010968 | 0.022086 | 72 | 39.4 KB |

### Top Feature Importances (Run 1, permutation)

1. `time_of_day_sin` — 0.01181
2. `day_of_year_cos` — 0.00279
3. `time_of_day_cos` — 0.00276
4. `time_of_day_sin2` — 0.00235
5. `time_of_day_cos2` — 0.00181
6. `humidity_lag30` — 0.00129
7. `temp_lag60` — 0.00102

Time-of-day features dominate; temperature lag features rank lower than in Model 1 which used a single 1-step lag.

## Key Notes

- **First model in the series to predict temperature differences** rather than absolute temperatures; this framing became standard for all subsequent models
- Higher val_loss (~0.011) than Model 1 (~0.004) — note the targets are *different* (differences vs absolute), so these are not directly comparable
- Extended lag features (30/60/120 min) replaced the single 1-min lag (`temp_lag1`)
- Higher-order cyclical encoding (`sin2/cos2`) added for richer time representation
- Feature interaction path added (same as Model 5 series)
- Numba JIT compilation for rolling slopes enables deployment on Raspberry Pi 5
- Edge TPU compiled model available (`weather_model_1_diff_best_edgetpu.tflite`)
