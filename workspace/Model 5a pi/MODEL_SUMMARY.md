# Model 5a Pi — Raspberry Pi Deployment Version of Model 5a

## Overview

Raspberry Pi 5 port of Model 5a clean, targeting CPU inference. The architecture is identical to Model 5a new arch (sequence-flattened wide+deep+residual), but with Numba JIT-compiled slopes and the same multiprocessing configuration as prior Pi models. Does not include the AveragePooling1D fix from Model 5a clean — this is a direct port of the original 4,860-dim architecture, which runs on CPU only (no Edge TPU acceleration on the Pi).

## Features (27 per timestep, 180 timesteps → 4,860 flattened)

Same 27 features as Model 5 new arch and Model 5a clean:

| Category | Features |
|----------|----------|
| Cyclical time | `time_of_day_sin/cos`, `day_of_year_sin/cos` |
| Current sensors | `relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `uv`, `illuminance`, `solar_radiation` |
| Slopes (Numba JIT) | `temperature_delta`, `illuminance_delta`, `solar_radiation_delta`, `pressure_delta`, `humidity_delta` |
| Lags | `temp_lag30/60/120`, `humidity_lag30/60/120`, `pressure_lag30/60/120` |

## Architecture

```
Input (SEQ_LEN=180, n_features=27)
      │
Reshape → (4,860,)    ← No AveragePooling1D (CPU deployment, no Edge TPU)
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
                                Dense(1) × 3 (diff_1hr, diff_2hr, diff_3hr)
```

RPi5 performance optimizations:
- Numba `@njit(parallel=True)` for all rolling slope computations
- `mp.set_start_method("fork")` for stable multiprocessing on Linux/ARM
- `set_intra_op_parallelism_threads(cores)` and `set_inter_op_parallelism_threads(cores)` for all CPU cores

- Optimizer: Adam lr=1e-5
- Batch size: 256
- Early stopping: patience=10
- `timeseries_dataset_from_array` for windowed training

## Results

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 | 0.000682 | 0.002355 | 15 | 787.74 KB |
| Run 2 | 0.006502 | 0.009413 | 28 | 45.4 KB |

Run 2 shows significantly worse val_loss (0.006502) and a much smaller model (45.4KB vs 787.74KB), suggesting run 2 used a different configuration (likely with a different SEQ_LEN or feature set that produced a smaller architecture).

### Top Feature Importances (Run 1, permutation)

1. `time_of_day_sin` — 0.000168
2. `day_of_year_sin` — 0.000088
3. `time_of_day_cos` — 0.000074
4. `relative_humidity` — 0.000063

Same importance ranking as Model 5 new arch / 5a clean.

## Key Notes

- Run 1 val_loss (0.000682) is very close to Model 5 new arch (0.000706) — confirms architecture is sound on Pi data pipeline
- Run 2 anomaly (45.4KB, 0.006502): the tiny model size suggests a different architecture branch may have run; investigation recommended if run 2 weights need to be understood
- This model runs via TFLite CPU inference on the RPi5 — no Edge TPU acceleration; the 4,860-dim input that overflows Edge TPU SRAM is fine for CPU TFLite
- For Edge TPU deployment on Pi+Coral: use the AveragePooling1D version from Model 5a clean
