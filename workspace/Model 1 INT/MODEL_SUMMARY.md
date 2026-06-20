# Model 1 INT — MSB/LSB 16-bit Precision Encoding

## Overview

Experimental variant exploring INT8 quantization accuracy by representing each normalized feature as two INT8 values: a Most Significant Byte (MSB) and a Least Significant Byte (LSB). This effectively encodes each scalar in 16-bit precision using two INT8 inputs, doubling the input width to 24 features. The hypothesis was that richer numerical precision could improve predictions from the limited INT8 quantization format.

## Features

The same 12 base features as Model 1 are used:

`illuminance`, `solar_radiation`, `uv`, `relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `day_of_year`, `time_of_day`, `temperature_delta`, `temp_lag1`, `humidity_lag1`

Each feature is first scaled to [0, 1], then converted to a 16-bit integer (0–65535), and split into MSB and LSB bytes. Each byte is then normalized to [0, 1] (÷255). This produces **24 input features** (2 × 12).

Input ordering: `[feature1_LSB, feature1_MSB, feature2_LSB, feature2_MSB, ...]`

## Architecture

Smaller than Model 1 to account for the doubled input width and added complexity:

```
Input (24) → LayerNormalization
           ─┬─ Dense(16) ─────────────────────────────────────────────┐
            │                                                           │
            └─ Dense(64, relu, L2) → Dropout(0.2) → Dense(32, relu, L2)┤  Concatenate
                                                └─ Dense(32) (shortcut)┤
                                                  Add → Dense(16, relu, L2) ┘
                                                                        │
                                                           Concatenate(wide, deep)
                                                                        │
                                                      Dense(1) × 3 (t1hr, t2hr, t3hr)
```

- Optimizer: Adam lr=1e-5 (legacy)
- Loss: MSE
- L2 regularization (1e-4) on all dense layers
- LayerNormalization on input
- Early stopping: patience=5

## Results (2 runs)

| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Run 1 | 0.01723 | 0.02704 | 99 | 21.1 KB |
| Run 2 | — | — | — | — |

The val_loss of 0.01723 is substantially worse than Model 1 (~0.004), despite requiring more epochs to train.

## Key Notes

- MSB/LSB encoding did **not** improve accuracy — val_loss is ~4× worse than Model 1
- The model reached max epochs (99) without early stopping, suggesting convergence issues
- Smaller architecture (Dense(64) vs Dense(128)) may have been inadequate to learn the MSB/LSB decomposition
- This approach was not pursued further in later models
- Conceptual issue: the MSB/LSB split is artificial (re-encoding a float → int → bytes); the model has to learn to "reassemble" precision that is already present in float32 during training
