# Model 2 — Dilated Conv1D with Residual Blocks

## Overview

First temporal sequence model: instead of using hand-crafted lag features, this model ingests a raw 180-minute sliding window of sensor readings and uses a stack of dilated Conv1D blocks to learn temporal patterns automatically. The dilated residual architecture gives the network a large effective receptive field (covering the full 180-step window) without a proportional increase in parameters.

## Features (15)

| Feature | Description |
|---------|-------------|
| `illuminance` | Ambient light level |
| `solar_radiation` | Solar irradiance |
| `uv` | UV index |
| `relative_humidity` | Current humidity |
| `station_pressure` | Station pressure |
| `wind_avg` | Average wind speed |
| `wind_gust` | Wind gust speed |
| `temperature_delta` | 15-min rolling slope |
| `temp_lag1` | Temperature 1 min ago |
| `humidity_lag1` | Humidity 1 min ago |
| `sin_time_of_day` | sin(2π × time / 24) |
| `cos_time_of_day` | cos(2π × time / 24) |
| `day_of_year` | Raw day of year scalar |
| `delta_minutes` | Minutes since last reading (gap detection) |
| `is_gap` | Binary: 1 if gap > 1.5 min |

Input is a 3D tensor of shape `(batch, 180, 15)` — a 180-minute window with 15 features per timestep.

## Architecture

```
Input (180, 15)
      │
Conv1D(32, k=3, same) → BatchNorm → ReLU
      │
ResidualDilatedBlock(32, k=3, dilation=1)
ResidualDilatedBlock(32, k=3, dilation=2)
ResidualDilatedBlock(32, k=3, dilation=4)
ResidualDilatedBlock(32, k=3, dilation=8)
      │
GlobalAveragePooling1D
Dropout(0.3)
Dense(64, relu, L1=1e-5)
Dropout(0.3)
Dense(32, relu, L1=1e-5)
BatchNorm
      │
Dense(1) × 3  (t1hr, t2hr, t3hr)
```

Each `ResidualDilatedBlock` is:
```
x → Conv1D(same, dilation) → BN → ReLU → Conv1D(same, dilation) → BN → Add(shortcut) → ReLU
```

- Optimizer: Adam lr=1e-4
- Loss: MSE per output
- Batch size: 32
- Early stopping: patience=5
- Window size: 180 minutes
- Targets: absolute temperatures, scaled to [0, 1] using fixed range [0°C, 50°C]

## Results

| Run | val_loss | val_mae | Best Epoch | avg_mae_celsius |
|-----|----------|---------|------------|-----------------|
| Run 1 | 0.018693 | 0.031361 | 13 | 1.75°C |

### Feature Importance
Feature importance was not computed (empty list in results).

## Key Notes

- val_loss (0.018693) is much worse than Model 1 (0.004022), despite using 180x more temporal context
- The model stopped at epoch 13 (early stopping), suggesting difficulty learning from the raw window
- Dilated receptive field covers 1 + (3-1)×(1+2+4+8) = ~29 steps, not the full 180 — the dilations do not span the full 3-hour window
- MSB/LSB separate feature file (`conv1d_quantized.tflite`) suggests quantization experiments were also performed
- The poor result relative to flat models led to moving away from raw Conv1D toward either: (a) flat models with hand-crafted lags (Model 5), or (b) improved Conv1D architectures with flattened sequences (Model 5 new arch.)
- Includes gap detection features (`delta_minutes`, `is_gap`) — an early attempt at handling irregular time sampling
