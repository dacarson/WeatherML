# Model 4 — INT16 MSB/LSB Encoding

## Overview

Explores an alternative approach to INT8 quantization: instead of using TFLite's post-training quantization, represent each float feature as two bytes (MSB and LSB of its INT16 encoding) fed as separate inputs. This gives the network access to full 16-bit precision via integer inputs that are naturally quantization-friendly. The hypothesis was that this encoding might improve Edge TPU compatibility while preserving predictive accuracy.

## Features (12 base → 24 encoded)

Base features (12):

| Feature | Description |
|---------|-------------|
| `illuminance` | Ambient light level |
| `solar_radiation` | Solar irradiance |
| `uv` | UV index |
| `relative_humidity` | Current humidity |
| `station_pressure` | Station pressure |
| `wind_avg` | Average wind speed |
| `wind_gust` | Wind gust speed |
| `day_of_year` | Day of year (1–366) |
| `time_of_day` | Time in hours (0–24) |
| `temperature_delta` | 15-min rolling slope |
| `temp_lag1` | Temperature 1 min ago |
| `humidity_lag1` | Humidity 1 min ago |

Each float is min-max scaled to [0, 1], multiplied by 65535, cast to `uint16`, then split into MSB (high byte) and LSB (low byte), both normalized to [0, 1]. This yields 24 input values per sample.

## Architecture

```
Input (24)
      │
LayerNormalization
Dense(64, relu, L2=1e-4)
Dense(32, relu, L2=1e-4)
      │
Wide path:  Dense(16)   ──────────────────────────────────────────┐
Deep path:  Dense(128) → Dropout(0.3) → Dense(64) + shortcut(Dense(64))
                                              Add → Dense(32, relu) ┤
                                                     Concatenate(wide, deep)
                                                                    │
                                                        Dense(1) × 3 (t1hr, t2hr, t3hr)
```

Key differences from Model 1:
- `LayerNormalization` at the input (instead of per-feature min/max scaling)
- Smaller Dense(64/32) initial projection (vs 128/64 in Model 1) to keep model compact
- L2 regularization (1e-4) on all layers
- Same 3-output wide+deep+residual structure

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Batch size: 256
- Early stopping: patience=10

## Results

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 | 0.017228 | 0.028576 | 99 | 53.2 KB |

`best_epoch=99` equals the maximum epochs configured — the model never converged within the epoch limit, suggesting it was still training when stopped.

### Top Feature Importances (Run 1, permutation)

1. `time_of_day_msb` — 0.00568
2. `time_of_day_lsb` — 0.00381
3. `temp_lag1_msb` — 0.00290
4. `temp_lag1_lsb` — 0.00254
5. `humidity_lag1_msb` — 0.00227

Time-of-day and temp_lag1 dominate, consistent with other models, but both MSB and LSB contribute — the model correctly leverages both bytes of the encoding.

## Key Notes

- **val_loss (0.017228) is 4× worse than Model 1 (0.004022)** — MSB/LSB encoding significantly degraded performance
- Never converged (hit 99 epochs, early stopping patience=10)
- Hypothesis failure: the MSB/LSB trick adds complexity without benefit; the network has to reconstruct the original float from two separate correlated bytes, which is a harder learning problem than operating on scaled floats directly
- This approach was abandoned; see Model 1 INT for a related experiment that also failed
- The L2 regularization and LayerNorm may also have constrained model capacity too aggressively for the 24-input problem
