# Model 1 PS — Wide+Deep+Residual with QAT Experiments

## Overview

Variant of Model 1 investigating Quantization-Aware Training (QAT) and the impact of training without the daytime illuminance filter. The primary experiments here focused on whether QAT could improve deployed INT8 accuracy compared to post-training quantization (PTQ). Multiple quantization formats were evaluated: standard INT8, float16, dynamic range, and an improved INT8 approach.

## Features (12)

Same feature set as Model 1:

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

Same wide+deep+residual architecture as Model 1. Float baseline trained first, then QAT fine-tuning applied.

```
Input (12) ─┬─ Dense(16)  ────────────────────────────────────────┐
             │                                                       │
             └─ Dense(128, relu) → Dropout(0.3) → Dense(64, relu) ─┤  Concatenate
                                             └─ Dense(64) (shortcut)┤
                                                  Add → Dense(32, relu) ┘
                                                                     │
                                                        Concatenate(wide, deep)
                                                                     │
                                                   Dense(1) × 3 (t1hr, t2hr, t3hr)
```

## Results

### Standard Training (10 runs)

| Run | val_loss | val_mae | Best Epoch |
|-----|----------|---------|------------|
| Best run | 0.006164 | 0.015718 | 38 |

### QAT Experiment (`results_qat.json`)

| Metric | Value |
|--------|-------|
| Float MAE (1hr/2hr/3hr) | 0.0217 / 0.0308 / 0.0387 |
| QAT MAE (1hr/2hr/3hr) | 0.0217 / 0.0308 / 0.0387 |
| Best float val_loss | 0.006455 |
| Training samples | 480,080 |
| Validation samples | 440,278 |
| Float epochs trained | 26 |

QAT and float model produced identical MAE scores, indicating QAT did not introduce quantization error relative to the float model for this architecture.

### Quantization Format Comparison

All four formats (standard INT8, float16, dynamic range, improved INT8) produced valid models.

### Top Feature Importances (best run, permutation)

1. `temp_lag1` — 0.318 (dominant)
2. `time_of_day` — 0.025
3. `illuminance` — 0.014
4. `solar_radiation` — 0.010
5. `uv` — 0.007

## Key Notes

- Significantly more training/validation samples than other Model 1 variants (480K/440K vs ~200K), suggesting this ran on a larger combined or expanded dataset
- QAT produced identical MAE to float, meaning quantization error was negligible for this model size/architecture
- val_loss (~0.006) is higher than Model 1 (~0.004), possibly due to different dataset composition
- Multiple quantization scripts included: `train_model_qat.py`, `train_model_qat_simple.py`, `train_model_qat_tf.py`
