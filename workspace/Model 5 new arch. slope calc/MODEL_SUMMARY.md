# Model 5 New Arch — Sequence-Flattened Wide+Deep+Residual

## Overview

Major architectural shift: instead of hand-crafted lag features, a 180-minute sliding window of raw sensor data is flattened into a single large input vector and fed to a wide+deep+residual network. This "sequence-as-flat-features" approach gives the model access to every minute of the past 3 hours without the Conv1D complexity that caused Model 2/3 to fail. The result is a 4,860-dimension input vector (180 × 27 features) — which causes an Edge TPU SRAM overflow problem.

## Features (27 per timestep, 180 timesteps → 4,860 flattened)

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

Note: slope features are commented out in the script (`# train_df['temperature_delta'] = ...`), suggesting this run used raw values without deltas.

## Architecture

```
Input (SEQ_LEN=180, n_features=27)
      │
Reshape → (4,860,)    ← Flatten entire sequence to 1D
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

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Batch size: 256
- `timeseries_dataset_from_array` for memory-efficient windowed batching
- SF-only CSV: `train_data_sf.csv` / `val_data_sf.csv`

## Results

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 (SF) | 0.000706 | 0.002388 | 18 | 787.74 KB |
| Run 1 PS | 0.003954 | 0.006889 | — | — |

The SF model achieves val_loss=0.000706 — a dramatic improvement over all prior flat models (1,000× better than Model 5, 15× better than Model 1).

### Top Feature Importances (Run 1, permutation)

1. `time_of_day_sin` — 0.000168 (still top, but much smaller relative to total loss)
2. `day_of_year_sin` — 0.000088
3. `time_of_day_cos` — 0.000074
4. `relative_humidity` — 0.000063

## Key Notes

- **Edge TPU SRAM overflow**: The 4,860-dim input causes the FC layer's input tensor to require ~4,800 bytes × 4,860 dims ≈ 23MB, which far exceeds the Edge TPU's 8MB SRAM. The model compiles but runs on CPU fallback, not on the TPU.
- The ~1,660-dim SRAM threshold (4,800 bytes/dim × 1,660 ≈ 8MB) means any flat model with SEQ_LEN × n_features > 1,660 will overflow Edge TPU SRAM
- Solution explored in Model 5a: AveragePooling1D(pool_size=6, strides=6) reduces 180 steps to 30, giving 30 × 27 = 810 dims — safely under the threshold
- Palm Springs (run1_ps) val_loss is 5.6× worse (0.003954), consistent with Palm Springs being more arid/predictable with less marine layer variance
- Model size 787.74KB is 9× larger than Model 5 (50.7KB) due to the 4,860-dim input layer weights
