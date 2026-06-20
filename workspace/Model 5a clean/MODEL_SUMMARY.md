# Model 5a Clean — AveragePooling1D + Gap Invalidation (Edge TPU Compatible)

## Overview

The production-quality evolution of Model 5 new arch, solving the Edge TPU SRAM overflow problem while maintaining high accuracy. The key insight: applying `AveragePooling1D(pool_size=6, strides=6)` before flattening reduces the 180-step sequence to 30 steps, bringing the flattened dimension from 4,860 to 810 — well within the Edge TPU's 8MB SRAM limit. Also adds gap invalidation (nulling targets that span sensor data gaps) and `ReduceLROnPlateau` for more stable training.

## Features (27 per timestep, 180 timesteps → pool to 30 → 810 flattened)

Same as Model 5 new arch:

| Feature | Description |
|---------|-------------|
| `time_of_day_sin/cos` | Diurnal cyclical encoding |
| `day_of_year_sin/cos` | Annual cyclical encoding |
| `relative_humidity` | Current humidity |
| `station_pressure` | Station pressure |
| `wind_avg` | Average wind speed |
| `wind_gust` | Wind gust speed |
| `uv` | UV index |
| `illuminance` | Ambient light level |
| `solar_radiation` | Solar irradiance |
| `temperature_delta` | 15-min rolling slope of temperature |
| `illuminance_delta` | 15-min rolling slope of illuminance |
| `solar_radiation_delta` | 15-min rolling slope of solar radiation |
| `pressure_delta` | 15-min rolling slope of pressure |
| `humidity_delta` | 15-min rolling slope of humidity |
| `temp_lag30/60/120` | Temperature 30, 60, 120 minutes ago |
| `humidity_lag30/60/120` | Humidity 30, 60, 120 minutes ago |
| `pressure_lag30/60/120` | Pressure 30, 60, 120 minutes ago |

## Architecture

```
Input (SEQ_LEN=180, n_features=27)
      │
AveragePooling1D(pool_size=6, strides=6)   ← (180, 27) → (30, 27)
Reshape → (810,)                            ← Flatten pooled sequence
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

Additional training features:
- **Gap invalidation**: if any sensor reading in a training window crosses a data gap, all 3 target values are set to NaN and masked during loss computation
- **ReduceLROnPlateau**: patience=5, factor=0.5, min_lr=1e-7
- Early stopping: patience=10

## Results (Multiple Experiments)

| Experiment | Config | val_loss | Model Size | Edge TPU |
|------------|--------|----------|------------|----------|
| Exp 1 (reference) | Full 27 features, no pooling | 0.000706 | 787.74 KB | No (SRAM overflow) |
| dense_wide_run1 | 27 features, no pooling | 0.000373 | 84.9 KB | No (SRAM overflow) |
| avgpool_run1 | AveragePooling1D(6,6) | 0.000508 | 90.6 KB | Yes |
| no_tod_run1 | No time-of-day features | 0.000555 | 83.1 KB | Yes |

### Experiment Log Summary (from MODEL_5A_CLEAN_EXPERIMENT_LOG.md)

| Exp | Key Change | val_loss | Finding |
|-----|-----------|----------|---------|
| 1 | Baseline (no pooling) | 0.000706 | Reference; SRAM overflow on Edge TPU |
| 2 | Add AveragePooling1D(6,6) | 0.000508 | 28% accuracy degradation; Edge TPU compatible |
| 3 | AveragePooling(2,2) | ~0.000450 | Less pooling, better accuracy, still under threshold |
| 4 | Gap invalidation only | 0.000373 | Best accuracy — gap targets were biasing loss down |
| 5 | Gap invalidation + pooling | ~0.000520 | Confirms pooling costs accuracy |
| 6 | Remove `day_of_year` features | Similar | Time features not critical for diffs |
| 7 | Remove `time_of_day` features | 0.000555 | Minor accuracy loss; confirmed TOD dominance was artifact |
| 8 | Full ablation of temporal | Similar | Core physics (pressure, humidity, solar) sufficient |

### Top Feature Importances (dense_wide_run1, permutation)

1. `time_of_day_sin` — 0.000168 (dominant)
2. `day_of_year_sin` — 0.000088
3. `time_of_day_cos` — 0.000074
4. `relative_humidity` — 0.000063

## Key Notes

- **Best accuracy across all models**: val_loss=0.000373 (84.9KB, dense_wide_run1)
- **Best Edge TPU compatible**: val_loss=0.000508 (90.6KB, avgpool_run1) — fully accelerated on Edge TPU
- AveragePooling1D(6,6) reduces 180→30 steps: input dimension 27×30=810, safely under the Edge TPU SRAM ~1,660 threshold
- Gap invalidation was the single biggest accuracy improvement: prior models were unwittingly trained on targets that spanned sensor data gaps, which introduced a systematic downward bias in loss metrics
- Time-of-day feature importance was initially suspicious (model appeared too dependent on it); Experiment 7 confirmed this was an artifact of gap correlation — the model was not over-relying on TOD
- This model family (5a clean) defines the production architecture for SF deployment
