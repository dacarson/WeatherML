# Model 1 Pi — Raspberry Pi Optimized (Diffs Variant)

## Overview

Raspberry Pi 5 deployment version of the Model 1 diffs architecture. Identical to the standard Model 1 diffs model in features and architecture, but with performance optimizations for the RPi5's ARM CPU: Numba JIT-compiled rolling slopes, multi-core parallelism via `multiprocessing.set_start_method("fork")`, and TensorFlow thread configuration matched to available cores.

## Features (13)

| Feature | Description |
|---------|-------------|
| `time_of_day_sin/cos` | Diurnal cyclical encoding |
| `day_of_year` | Day of year (raw scalar) |
| `temp_lag30` | Temperature 30 min ago |
| `humidity_lag30` | Humidity 30 min ago |
| `wind_gust` | Wind gust speed |
| `wind_avg` | Average wind speed |
| `uv` | UV index |
| `temperature_delta` | 15-min rolling slope (Numba) |
| `illuminance_delta` | 15-min rolling slope of illuminance |
| `solar_radiation_delta` | 15-min rolling slope of solar radiation |
| `pressure_delta` | 15-min rolling slope of pressure |
| `humidity_delta` | 15-min rolling slope of humidity |

Note: only 30-minute lags (not 60/120), fewer lag features than the full diffs model.

## Architecture

Same wide+deep+residual with interaction path as Model 1 diffs:

```
Input (13) ─ interaction_embed(Dense(16)) → square → Concat → Dense(32)
           ─┬─ Dense(16) (wide)
            └─ Dense(128, relu) → Dropout(0.3) → Dense(64, relu) + shortcut(Dense(64))
                                                     Add → Dense(32, relu)
                                                           Concatenate → Dense(1) × 3
```

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Batch size: 256
- Targets: temperature differences (diff_1hr/2hr/3hr)
- CPU-optimized: `set_intra_op_parallelism_threads(cores)`, `set_inter_op_parallelism_threads(cores)`
- Numba `@njit(parallel=True)` for all rolling slope computations

## Results (10 runs)

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 | 0.011311 | 0.022396 | 14 | 33.0 KB |
| Run 10 | — | — | — | — |

### Top Feature Importances (Run 1, permutation)

1. `time_of_day_sin` — 0.01726
2. `time_of_day_cos` — 0.00733
3. `day_of_year` — 0.00204
4. `temp_lag30` — 0.00184
5. `wind_gust` — 0.00140

## Key Notes

- Designed for deployment on Raspberry Pi 5 with Coral Edge TPU (PCIe M.2)
- Fewer lag features than the full diffs model (30-min only, vs 30/60/120) to reduce inference latency
- Numba JIT + parallel for-loops provides multi-core rolling slope computation on RPi5's 4-core ARM Cortex-A76
- Results similar to Model 1 diffs (val_loss ~0.011) confirming the architecture is sound
- Edge TPU quantized model available (`weather_model_1_diff_best_edgetpu.tflite`)
- Uses `"fork"` multiprocessing start method for stability on Linux/ARM
