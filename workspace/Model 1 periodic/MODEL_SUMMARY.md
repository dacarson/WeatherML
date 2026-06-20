# Model 1 Periodic — Cyclical Time Encoding

## Overview

Variant of Model 1 that replaces the raw scalar `time_of_day` (0–24) and `day_of_year` (1–366) features with sine/cosine cyclical encodings. Raw scalars treat noon (12) as twice as far from midnight as 6am, which misrepresents the continuous cyclical nature of time. Cyclical encoding places midnight and 11:59pm as adjacent points in a unit circle, which is physically correct. This was the first model to test whether cyclical time features improve predictions.

## Features (14)

| Feature | Description |
|---------|-------------|
| `illuminance` | Ambient light level (lux) |
| `solar_radiation` | Solar irradiance (W/m²) |
| `uv` | UV index |
| `relative_humidity` | Current humidity (%) |
| `station_pressure` | Station pressure (hPa) |
| `wind_avg` | Average wind speed |
| `wind_gust` | Wind gust speed |
| `time_of_day_sin` | sin(2π × time / 24) |
| `time_of_day_cos` | cos(2π × time / 24) |
| `day_of_year_sin` | sin(2π × day / 366) |
| `day_of_year_cos` | cos(2π × day / 366) |
| `temperature_delta` | 15-minute rolling linear-regression slope |
| `temp_lag1` | Temperature from 1 minute ago |
| `humidity_lag1` | Humidity from 1 minute ago |

Cyclical features are bounded to [-1, 1]; all others use per-feature min/max with ±5% padding.

## Architecture

Same wide+deep+residual architecture as Model 1:

```
Input (14) ─┬─ Dense(16)  ────────────────────────────────────────┐
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
- Loss: MSE
- Early stopping: patience=5
- Targets: absolute temperatures

## Results (10 runs)

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 (best) | 0.004373 | 0.013573 | 41 | 33.2 KB |
| Run 10 | — | — | — | — |

### Top Feature Importances (Run 1, permutation)

1. `temp_lag1` — 0.0737 (still dominant)
2. `illuminance` — 0.0261
3. `relative_humidity` — 0.0216
4. `humidity_lag1` — 0.0189
5. `solar_radiation` — 0.0179
6. `time_of_day_sin` — 0.0078

## Key Notes

- Cyclical encoding (`sin/cos`) for time and day replaced raw scalars compared to Model 1
- val_loss (0.004373) is similar to Model 1 (0.004022) — cyclical encoding had little impact at this scale
- `temp_lag1` remains dominant, as in all Model 1 variants
- `illuminance` and `relative_humidity` rank higher with cyclical time than in the non-cyclical version, suggesting these features share variance with raw time_of_day
- Cyclical time encoding was adopted as standard in all subsequent models (Model 1 diffs, Model 5+)
