# Model 1a — Single-Horizon Output (1hr Only)

## Overview

Simplified variant of Model 1 predicting only the +1hr temperature target (single output head). Reduces complexity versus the 3-output Model 1 to investigate whether a dedicated single-horizon model learns better than a shared-head multi-horizon model. The architecture is otherwise identical to Model 1.

## Features (12)

Same as Model 1:

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
| `temperature_delta` | 15-minute rolling linear-regression slope |
| `temp_lag1` | Temperature from 1 minute ago |
| `humidity_lag1` | Humidity from 1 minute ago |

## Architecture

Same wide+deep+residual architecture as Model 1, but with a single output head:

```
Input (12) ─┬─ Dense(16) ──────────────────────────────────────────┐
             │                                                        │
             └─ Dense(128, relu) → Dropout(0.3) → Dense(64, relu) ──┤  Concatenate
                                               └─ Dense(64) (shortcut)┤
                                                    Add → Dense(32, relu) ┘
                                                                      │
                                                        Concatenate(wide, deep)
                                                                      │
                                                           Dense(1, linear) ← temp_t+1hr only
```

- Optimizer: Adam
- Loss: MSE (single output)
- Single target: `temp_t+1hr` (absolute temperature)
- Targets scaled to [−1, 1] using training min/max

## Results (2 runs)

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 | 0.003164 | 0.041769 | 8 | 32.3 KB |
| Run 2 | — | — | — | — |

Note: val_mae of 0.0418 on a [−1, 1] scale seems high for only 8 epochs — early stopping may have fired very early, suggesting the model converged quickly to a local minimum.

### Top Feature Importances (Run 1, permutation)

1. `temp_lag1` — 0.0467 (dominant)
2. `illuminance` — 0.0031
3. `solar_radiation` — 0.0028
4. `relative_humidity` — 0.0021
5. `uv` — 0.0017

## Key Notes

- Single-output model for +1hr only
- val_loss (0.003164) is slightly better than Model 1's run1 (0.004022) for the 1hr target, as expected since the model specializes on one horizon
- Very early stopping (epoch 8) suggests rapid convergence — possibly the 1hr target is easier than multi-horizon
- Model 1a was not extended to other horizons or further developed; the multi-output approach of Model 1/1 diffs was preferred
