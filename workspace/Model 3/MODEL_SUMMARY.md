# Model 3 — Conv1D TPU-Optimized (Edge TPU Architecture)

## Overview

Conv1D architecture redesigned specifically for Edge TPU compatibility. Uses strided convolutions and a single dilated layer instead of residual blocks, targeting a ~27-step effective receptive field. The architecture avoids operations that are problematic for Edge TPU quantization. Notably, it uses a shorter 90-minute window (vs 180 in Model 2) and a simpler 8-feature input set.

## Features (8)

| Feature | Description |
|---------|-------------|
| `temp_avg_15min` | 15-min rolling average of temperature (shifted 1 step back) |
| `temperature_delta` | 15-min rolling slope |
| `sin_time_of_day` | sin(2π × time / 24) |
| `cos_time_of_day` | cos(2π × time / 24) |
| `illuminance` | Ambient light level |
| `solar_radiation` | Solar irradiance |
| `station_pressure` | Station pressure |
| `relative_humidity` | Current humidity |

Input is `(batch, 90, 8)` — a 90-minute window with 8 features per timestep.

## Architecture

```
Input (90, 8)
      │
Conv1D(32, k=3, stride=1, relu)          ← Block 1: local features
Conv1D(32, k=3, stride=2, relu)          ← Block 2: expand receptive field (downsamples to 45 steps)
Conv1D(32, k=5, stride=2, relu)          ← Block 3: larger kernel (downsamples to 23 steps)
Conv1D(32, k=3, dilation=2, relu)        ← Block 4: dilation for broader context
      │
Reshape(-1,)                             ← Flatten all temporal positions
Dense(64, relu)
Dense(32, relu)
Dense(16, relu)
      │
Dense(1) × 3  (t1hr, t2hr, t3hr)
```

Effective temporal coverage: strides reduce to ~23 steps; dilation=2 extends to ~27 steps. Does **not** cover the full 90-minute window depth.

- Optimizer: Adam lr=1e-3
- Loss: MSE per output
- Batch size: 32
- Early stopping: patience=5
- Targets: absolute temperatures, scaled to [0, 1] using fixed range [0°C, 50°C]
- Checkpoints saved (`./checkpoints/best_{name}.weights.h5`)

## Results

| Run | val_loss | val_mae | Best Epoch | avg_mae_celsius | val_rmse_celsius |
|-----|----------|---------|------------|-----------------|------------------|
| Run 1 | 0.021779 | 0.033171 | 1 | 2.95°C | 4.26°C |
| Run 2 | 0.022750 | 0.033969 | 1 | 3.02°C | 4.35°C |

Both runs stopped at epoch 1 with `best_epoch=1`, indicating the model failed to meaningfully learn from the data — validation loss was already at its minimum after a single epoch of training.

### Feature Importances

All feature importances are negative (~−0.006 each), meaning permuting any feature *improved* validation loss, which is a diagnostic sign of a broken or undertrained model.

## Key Notes

- **Model failed to train** — both runs stopped at epoch 1, val_loss ~0.022, far worse than Model 1 (~0.004)
- Negative feature importances confirm the model is not using features meaningfully
- Root cause: high learning rate (1e-3) may have caused divergence after epoch 1; model hit the early-stopping minimum immediately
- Short window (90 min) and minimal features may have been insufficient for the temporal modeling task
- The Flatten approach (Reshape after Conv1D) creates a large input to the Dense layers, which may have caused instability
- Architecture was not continued; lessons incorporated into Model 5 new arch approach
