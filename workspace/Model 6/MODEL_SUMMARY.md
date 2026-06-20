# Model 6 — Solar Radiation Forecasting with Marine Layer Features

## Overview

First model targeting a non-temperature output: predicts solar radiation differences at +30, +60, and +90 minutes. Solar radiation forecasting is important for anticipating photovoltaic output and diurnal heating cycles. Unique to this model is a marine layer heuristic feature set: hand-crafted meteorological features that capture coastal fog behavior (particularly relevant to San Francisco's marine layer). Uses 50+ features — the largest feature set in the project.

## Features (50+)

### Core Sensor Features
| Feature | Description |
|---------|-------------|
| `illuminance` | Ambient light level |
| `solar_radiation` | Current solar irradiance |
| `uv` | UV index |
| `relative_humidity` | Current humidity |
| `station_pressure` | Station pressure |
| `wind_avg` | Average wind speed |
| `wind_gust` | Wind gust speed |
| `temperature` | Current temperature |

### Cyclical Time Features
| Feature | Description |
|---------|-------------|
| `time_of_day_sin/cos` | Diurnal cyclical encoding |
| `day_of_year_sin/cos` | Annual cyclical encoding |

### Rolling Slope Features (Numba JIT, 15-min window)
| Feature | Description |
|---------|-------------|
| `temperature_delta` | Temperature slope |
| `illuminance_delta` | Illuminance slope |
| `solar_radiation_delta` | Solar radiation slope |
| `pressure_delta` | Pressure slope |
| `humidity_delta` | Humidity slope |

### Lag Features (multiple time scales)
| Feature | Lags |
|---------|------|
| `temp_lag` | 30, 60, 120 min |
| `humidity_lag` | 30, 60, 120 min |
| `pressure_lag` | 30, 60, 120 min |
| `solar_lag` | 30, 60, 120 min |
| `illuminance_lag` | 30, 60, 120 min |

### Marine Layer Heuristic Features (derived)
| Feature | Description |
|---------|-------------|
| `fog_likelihood` | High humidity + low solar → fog score |
| `marine_push_score` | West wind + humidity + pressure gradient → marine push |
| `solar_clear_sky_ratio` | `solar / theoretical_clear_sky_solar` (0–1) |
| `marine_layer_depth_proxy` | Temperature inversion proxy from humidity gradient |
| `delta_fog_to_clear` | Rate of change in `solar_clear_sky_ratio` |

### Targets (3)
- `solar_diff_30min`: solar radiation change in next 30 minutes
- `solar_diff_60min`: solar radiation change in next 60 minutes
- `solar_diff_90min`: solar radiation change in next 90 minutes

## Architecture

Wide+Deep+Residual with interaction path — same structural pattern as Model 5:

```
Input (50+)
      │
   ┌──┴────────────────────────────────────────────────────────┐
   │                                                            │
interaction_path:                                          Wide+Deep+Residual:
Dense(16) → sq → Concat → Dense(32)                           Dense(16) (wide)
                                                               Dense(128, relu) → Dropout(0.3)
                                                                  └─ Dense(64) → Add(shortcut) → Dense(32, relu)
   └────────────────────────────────────────────────────────────┘
                              Concatenate(interaction, wide, deep)
                                             │
                      Dense(1) × 3 (solar_diff_30min, solar_diff_60min, solar_diff_90min)
```

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Batch size: 256
- Target scaled by ±5% padded min/max
- Numba JIT slopes: `@njit(parallel=True)`

## Results

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 | 0.018472 | 0.032081 | 14 | 67.4 KB |
| Run 5 | 0.023863 | 0.038542 | 12 | 67.4 KB |

val_loss did not improve meaningfully across runs (Run 5 is slightly worse), suggesting the model converged to a local minimum around epoch 12–14.

### Top Feature Importances (Run 1, permutation)

1. `solar_lag30` — 0.00581 (dominant)
2. `illuminance_lag30` — 0.00423
3. `solar_clear_sky_ratio` — 0.00317
4. `fog_likelihood` — 0.00198
5. `time_of_day_sin` — 0.00177
6. `solar_radiation_delta` — 0.00143

Solar-specific lag features and the marine layer heuristics (`solar_clear_sky_ratio`, `fog_likelihood`) rank highly, validating the domain-specific feature engineering approach.

## Key Notes

- **First non-temperature target**: predicts solar radiation change, not temperature
- Marine layer heuristics are the most notable feature engineering contribution — hand-crafted meteorological domain knowledge encoded as derived features
- `solar_clear_sky_ratio` is particularly valuable: it normalizes raw solar values against a theoretical clear-sky model, isolating cloud/fog attenuation independent of time-of-day
- val_loss 0.018 is comparable to early temperature models (Model 2/3), which makes sense — solar radiation is highly variable (cloud intermittency) and harder to predict
- The marine layer features (`fog_likelihood`, `marine_push_score`) improve performance in the SF coastal environment but would need recalibration for other sites
- Model size (67.4KB) is reasonable but the large feature count (50+) may make inference preprocessing expensive on embedded hardware
