# Model 5 — Flat Feature + Interaction Path (Diffs)

## Overview

Extends the Model 1 diffs approach with two key additions: (1) a `build_feature_interaction_path` module that learns non-linear pairwise interactions between all input features, and (2) Numba JIT-compiled rolling slopes for multiple sensor channels (not just temperature). Targets temperature differences at +1/2/3hr. Uses 28 features — the largest flat feature set to date.

## Features (28)

| Feature | Description |
|---------|-------------|
| `time_of_day_sin/cos` | Diurnal cyclical encoding |
| `day_of_year_sin/cos` | Annual cyclical encoding |
| `temp_lag30/60/120` | Temperature 30, 60, 120 minutes ago |
| `humidity_lag30/60/120` | Humidity 30, 60, 120 minutes ago |
| `pressure_lag30/60/120` | Pressure 30, 60, 120 minutes ago |
| `wind_gust` | Wind gust speed |
| `wind_avg` | Average wind speed |
| `uv` | UV index |
| `temperature_delta` | 15-min rolling slope of temperature (Numba JIT) |
| `illuminance_delta` | 15-min rolling slope of illuminance (Numba JIT) |
| `solar_radiation_delta` | 15-min rolling slope of solar radiation (Numba JIT) |
| `pressure_delta` | 15-min rolling slope of pressure (Numba JIT) |
| `humidity_delta` | 15-min rolling slope of humidity (Numba JIT) |

All Numba slopes computed via `@njit(parallel=True)` for multi-core performance.

## Architecture

```
Input (28)
      │
   ┌──┴──────────────────────────────────────────────────────┐
   │                                                          │
interaction_path:                                         Wide+Deep+Residual:
Dense(16) → Multiply(self) → Concat([raw,sq]) → Dense(32)    Dense(16) (wide)
                                                              Dense(128, relu) → Dropout(0.3) → Dense(64, relu)
                                                                   └── Dense(64, shortcut) → Add → Dense(32, relu)
   └──────────────────────────────────────────────────────────┘
                            Concatenate(interaction, wide, deep)
                                           │
                               Dense(1) × 3 (diff_1hr, diff_2hr, diff_3hr)
```

`build_feature_interaction_path`:
```python
embed = Dense(16, relu)(input)
squared = Multiply()([embed, embed])  # element-wise self-interaction
concat = Concatenate()([input, squared])
return Dense(32, relu)(concat)
```

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Batch size: 256
- Targets: temperature differences (°C difference, not absolute)
- Scalers: per-feature min/max with ±5% padding; target scaled to range + ±5%

## Results

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 | 0.010794 | 0.019958 | 12 | 50.7 KB |
| Run 5 | 0.010968 | 0.020043 | 14 | 50.7 KB |

### Top Feature Importances (Run 1, permutation)

1. `time_of_day_sin` — 0.01726 (dominant)
2. `time_of_day_cos` — 0.00733
3. `day_of_year_sin` — 0.00204
4. `temp_lag30` — 0.00184
5. `wind_gust` — 0.00140

## Key Notes

- val_loss ~0.011 — worse than Model 1 absolute (0.004) but targets are fundamentally different (diffs vs absolute)
- The interaction path adds a non-linear polynomial feature-mixing layer that lets the model discover pairwise interactions without hand-crafting them
- Time-of-day cyclical features dominate even in a diffs model — the day/night cycle strongly influences temperature change rates
- This model established the 28-feature "diffs" configuration and Numba multi-slope pipeline, which was carried forward into Model 5 new arch
- Extended lags (30/60/120 min for temp, humidity, and pressure) provide multi-scale temporal context without a sequence model
