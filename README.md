# WeatherML

A machine learning project that uses historical weather data from a **Tempest Weather Station** to predict future temperatures, quantizes the models to INT8 TFLite, compiles them for the **Coral Edge TPU**, and deploys inference on a **Raspberry Pi**.

This README is a tutorial tracing the journey from a simple dense baseline to a ~15× accuracy improvement over that baseline — and the lessons learned along the way.

---

## Hardware & Deployment Pipeline

```
MacBook Pro                      Docker (x86)              Raspberry Pi
──────────────────               ─────────────────         ──────────────────────
train_model.py                   edgetpu_compiler          Coral Edge TPU
  └─ Keras model (.keras)   →    └─ _edgetpu.tflite   →    └─ Inference_InfluxDB_Writer.py
       └─ INT8 TFLite (.tflite)                                  └─ writes to InfluxDB
```

1. **Train** — Python + TensorFlow on MacBook Pro (CPU and GPU)
2. **Quantize** — INT8 TFLite via representative-dataset quantization
3. **Compile** — `edgetpu_compiler` in a Docker container (see `edgetpu-x86-compiler.sh`)
4. **Deploy** — Copy `_edgetpu.tflite` + scaler JSON files to Raspberry Pi
5. **Infer** — `Inference_InfluxDB_Writer.py` reads live data from InfluxDB, runs the model on the Coral TPU, writes predictions back to InfluxDB

---

## Data

One-minute observations from a Tempest Weather Station, exported from InfluxDB using `export_influx_to_csv*.py`.

| File | Split | Location | Period |
|------|-------|----------|--------|
| `workspace/train_data_sf.csv` | Training | San Francisco, CA | Apr 9 2023 – Apr 6 2025 |
| `workspace/val_data_sf.csv` | Validation | San Francisco, CA | Apr 7 2025 – Apr 7 2026 |
| `workspace/train_data_ps.csv` | Training | Palm Springs, CA | Oct 10 2023 – Apr 6 2025 |
| `workspace/val_data_ps.csv` | Validation | Palm Springs, CA | Apr 7 2025 – Apr 7 2026 |

Raw features: `temperature`, `relative_humidity`, `station_pressure`, `solar_radiation`, `illuminance`, `uv`, `wind_avg`, `wind_gust`, `wind_lull`, `wind_direction`, `rain_accumulated`, `day_of_year`, `time_of_day`

> Data files are not checked in (60–140 MB each). Regenerate them with the export scripts.

---

## Model Evolution — A Tutorial

All model directories live under `workspace/`. Each contains a training script, result JSON files, scaler JSON files, and (after training) compiled `.tflite` artifacts.

### Step 1 — The Baseline: Dense Wide-Deep Model

**Directory**: `workspace/Model 1/`

The first model is a **wide-and-deep dense network** that takes a snapshot of 12 current weather features and predicts temperature at **+1hr, +2hr, and +3hr** simultaneously (three output heads). Training data is filtered to daytime-only rows (`illuminance > 1400 lux`).

**Architecture**
```
Input (12 features)
  ├─ Wide branch:  Dense(16)                           ← memorization
  └─ Deep branch:  Dense(128,relu) → Dropout(0.3)
                   → Dense(64,relu) → residual Dense(64) → Add()
                   → Dense(32,relu)                    ← generalization
Concatenate([wide, deep]) → Dense(1) × 3  ← outputs: temp_t+1hr, temp_t+2hr, temp_t+3hr
```

**Features (12)**: `illuminance`, `solar_radiation`, `uv`, `relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `day_of_year` (raw scalar), `time_of_day` (raw scalar), `temperature_delta` (15-sample rolling slope), `temp_lag1`, `humidity_lag1`

**Key design choices**:
- **Daytime-only filter** (`illuminance > 1400 lux`) to focus on the most predictable regime
- Per-feature min/max scaling with ±5% padding and domain bounds (e.g. humidity 0–100)
- Multi-run training to pick the best of 10 random initializations; `temp_lag1` dominates across all runs

**Results** (Run 1 best): val_loss = 0.004022, val_mae = 0.012984, model size 32.9 KB, best epoch 50

**Feature importance** (top 5): `temp_lag1` (0.0823) ≫ `illuminance` (0.0083) > `time_of_day` (0.0080) > `solar_radiation` (0.0078) > `uv` (0.0058)

**Lesson learned**: `temp_lag1` (the previous minute's temperature) dominates feature importance by a large margin — the model is learning "temperature barely changes in one minute." This is a hint that we are predicting the wrong thing.

---

### Step 1a — Feature Exploration: Variants of Model 1

Several variants explored specific questions without changing the core architecture.

| Variant | Directory | Question asked | Finding |
|---------|-----------|----------------|---------|
| **Model 1a** | `Model 1a/` | Single-output (+1hr only) vs. multi-output | val_loss 0.003164 — slightly better for 1hr alone, but multi-output adds no cost and provides all three horizons |
| **Model 1 Daytime** | `Model 1 Daytime/` | Remove the daytime filter; train on 24-hour data | Same val_loss (0.004022) as the filtered baseline; the illuminance filter had negligible effect |
| **Model 1 INT** | `Model 1 INT/` | INT-style quantization encoding | Worse (val_loss 0.0094 vs 0.0040) |
| **Model 1 Periodic** | `Model 1 periodic/` | Replace scalar `time_of_day` / `day_of_year` with sin/cos cyclic pairs | Slightly worse (val_loss 0.004373 vs 0.004022); cyclic encoding is the right idea but not yet impactful |
| **Model 1 Diffs** | `Model 1 diffs/` | Predict temperature *change* and expand features to 28–31 inputs with multi-horizon lags, slopes, double harmonics | Higher loss at this stage (val_loss 0.010793); techniques here are assembled later in Model 5a |
| **Model 1 Combined** | `Model 1 combined/` | Combine diff + cyclic features | No win |
| **Model 1 Pi** | `Model 1 pi/` | Can training run on the Raspberry Pi itself? | Yes, with Numba-accelerated slope calculation and all CPU threads |
| **Model 1 PS** | `Model 1 PS/` | Does the model generalise to Palm Springs climate? | Hard failures at temperatures outside the SF training distribution |

> **Note on directory naming**: `workspace/Model 1/` has the daytime filter applied; `workspace/Model 1 Daytime/` is the no-filter (full 24-hour) variant. The names are counterintuitive.

**Lesson learned**: Cyclic encoding (`sin`/`cos`) for `time_of_day` and `day_of_year` will become important in later models. The multi-horizon lags and difference targets introduced in Model 1 Diffs are key building blocks — they just need the right architecture to show their value.

---

### Step 1b — Predicting Differences with Richer Features: Model 1 Diffs

**Directory**: `workspace/Model 1 diffs/`

Model 1 Diffs makes two structural changes at once: it switches the prediction target from **absolute temperature to temperature change** (`temp_t+Xhr − temp_now`), and expands the feature set to **28–31 inputs** (depending on optional sensors) covering multi-horizon lags, Numba-computed rolling slopes, and double-harmonic time encoding.

**Architecture** (same wide-deep, 3 output heads, adds explicit interaction path):
```
Input (28+)
  ├─ Interaction path: Dense(16,relu) → Multiply(self) → Concat → Dense(32,relu)
  ├─ Wide branch:  Dense(16)
  └─ Deep branch:  Dense(128,relu) → Dropout(0.3)
                   → Dense(64,relu) → residual Dense(64) → Add()
                   → Dense(32,relu)
Concatenate([wide, deep, interaction]) → Dense(1) × 3  ← diff_1hr, diff_2hr, diff_3hr
```

**Key changes from Model 1**:
- **Difference targets**: `temp_diff_1hr/2hr/3hr = temp_t+Xhr − temp_now`, scaled to `[−1, 1]` with ±2°C padding
- **Multi-horizon lags** at 30/60/120 min for temperature and humidity, plus 30 min for wind avg/gust, UV, pressure
- **Numba-accelerated rolling slopes** (15-min window) for temperature, pressure, humidity, illuminance, solar radiation — first use of `@njit(parallel=True)` for Pi deployment
- **Double-harmonic time encoding**: `time_of_day_sin2/cos2 = sin/cos(4π × t/24)` for sub-daily patterns
- **Interaction path**: dense self-multiplication layer that lets the model learn pairwise feature products

**Results** (Run 1 best): val_loss = 0.010793, val_mae = 0.021837, model size 39.4 KB, best epoch 77

**Feature importance**: `time_of_day_sin` now leads (0.01181), with `day_of_year_cos`, `time_of_day_cos`, and second harmonics filling out the top 5. Temperature lag features rank lower — the 30/60/120 min lags are less predictive than `temp_lag1` was in the flat snapshot model.

**Lesson learned**: Predicting differences is worse here (0.0109 vs 0.0040) because removing `temp_lag1` costs more than the difference target gains. But all the techniques introduced — Numba slopes, double harmonics, difference targets, multi-horizon lags, interaction path — are carried forward and collectively produce the breakthrough in Model 5a.

---

### Step 2 — Temporal Sequences: Conv1D

**Directories**: `workspace/Model 2/`, `workspace/Model 3/`

Instead of a snapshot, what if the model sees a **window of time**? Model 2 feeds a **180-minute sliding window** of 15 features per timestep through a dilated Conv1D residual network.

**Model 2 architecture**:
```
Input (180 steps × 15 features)
  Conv1D(32, kernel=3, same) → BatchNorm → ReLU
  4× ResidualDilatedBlock: Conv1D(32, dilation=[1,2,4,8]) + BN + Add
  GlobalAveragePooling1D
  Dropout(0.3) → Dense(64,relu) → Dropout(0.3) → Dense(32,relu) → BatchNorm
  Dense(1) × 3  ← temp_t+1hr, temp_t+2hr, temp_t+3hr
```

**Features (15)**: raw sensor readings plus `sin_time_of_day`, `cos_time_of_day`, `temperature_delta`, `temp_lag1`, `humidity_lag1`, `delta_minutes`, `is_gap`

Note: the dilated receptive field covers only ~29 steps (dilation 1+2+4+8 with kernel 3), not the full 180-minute window.

**Model 3** (`workspace/Model 3/`) simplified the window to **90 minutes** and 8 features for Edge TPU size targets. Both runs stopped at epoch 1 with negative feature importances — the model failed to learn meaningfully.

**Results**:
| Model | val_loss | val_mae | Size | Note |
|-------|----------|---------|------|------|
| Model 2 | 0.018693 | 0.031361 | 65 KB | Stopped at epoch 13 |
| Model 3 | 0.021779 | 0.033171 | 79 KB | Stopped at epoch 1; negative importances |

Both are worse than Model 1, despite more complexity.

**Lesson learned**: Conv1D doesn't automatically beat hand-crafted lag features. GlobalAveragePooling discards positional information. The dilated receptive field doesn't cover the full 3-hour window. Explicitly providing lag features to a dense model outperforms letting convolutions discover them implicitly — at least at this scale.

---

### Step 3 — Precision Engineering: INT16 Hybrid

**Directory**: `workspace/Model 4/`

Model 4 experiments with representing each input feature as two values — a **least-significant byte (LSB) and most-significant byte (MSB)** — to approximate INT16 precision within an INT8 TFLite model. This doubles the input width from 12 to 24 features and adds LayerNormalization and L2 regularisation.

**Architecture**:
```
Input (24 encoded features)
  LayerNormalization
  Dense(64,relu,L2) → Dense(32,relu,L2)
  ├─ Wide branch:  Dense(16)
  └─ Deep branch:  Dense(128,relu) → Dropout(0.3) → Dense(64,relu) + shortcut → Add → Dense(32,relu)
Concatenate([wide, deep]) → Dense(1) × 3
```

**Results**: val_loss = 0.017228, val_mae = 0.028576, model size 53.2 KB — worse than Model 1. The model never converged (hit 99-epoch cap); it was still training when stopped.

**Feature importance**: `time_of_day_msb` (#1, 0.00568) and `temp_lag1_msb` (#3, 0.00290) — both MSB and LSB of each feature contribute, confirming the encoding is being used, but the overall accuracy is still poor.

> **Model 4a** (`workspace/Model 4a/`) was an incomplete follow-up applying per-feature precision analysis (INT32 for wide-range features like `illuminance`). It was never run to completion. Both Model 4 and 4a were abandoned.

**Lesson learned**: The LSB/MSB trick adds complexity without benefit. INT8 quantization is precise enough for these weather features after proper domain-aware scaling. The interaction path design from Model 4 is worth keeping.

---

### Step 4 — Predicting Differences with an Interaction Path

**Directory**: `workspace/Model 5/`

This is the first model to combine all of the techniques developed in the Model 1 variants: difference targets, multi-horizon lags, rolling slopes, double-harmonic time encoding, and the pairwise interaction path — into a single 28-feature flat model.

**Architecture** (same wide-deep + interaction path as Model 1 Diffs, 28 features):
```
Input (28 features)
  ├─ Interaction path: Dense(16,relu) → Multiply(self) → Concat → Dense(32,relu)
  ├─ Wide branch:  Dense(16)
  └─ Deep branch:  Dense(128,relu) → Dropout(0.3) → Dense(64,relu) + shortcut → Dense(32,relu)
Concatenate([interaction, wide, deep]) → Dense(1) × 3  ← diff_1hr, diff_2hr, diff_3hr
```

**Features (28)**: `time_of_day_sin/cos`, `day_of_year_sin/cos`, `temp_lag30/60/120`, `humidity_lag30/60/120`, `pressure_lag30/60/120`, `wind_gust`, `wind_avg`, `uv`, rolling slopes (temperature, illuminance, solar, pressure, humidity) via Numba JIT, plus current sensor readings

**Results** (Run 1): val_loss = 0.010794, val_mae = 0.019958, model size 50.7 KB, best epoch 12

**Feature importance**: `time_of_day_sin` dominates (0.01726), with `time_of_day_cos` second (0.00733). The multi-horizon lag features rank in the lower half — the rolling slope and interaction features are absorbing gradient that the lags need.

**Lesson learned**: Despite having lags at 30/60/120 minutes AND slope features AND an interaction path, `time_of_day_sin` still dominates feature importance. The lags are present but the model is not attending to them — the slope and delta features are providing easier gradient paths that suppress the lag signal.

---

### Step 5 — The Breakthrough: Sequence Architecture

**Directories**: `workspace/Model 5 new arch. slope calc/`, `workspace/Model 5a pi/`

This step achieves a **~15× reduction in val_loss** over Model 5. The key architectural change is moving from a flat feature snapshot to a **180-minute sliding window** fed directly to the model — with the 180-step sequence flattened into a single 4,860-dimension input vector (180 × 27 features per timestep).

**Architecture**:
```
Input (SEQ_LEN=180 timesteps, 27 features per timestep)
  Reshape → (4,860,)    ← entire 3-hour sequence as one flat vector
  ├─ Interaction path: Dense(16,relu) → Multiply(self) → Concat → Dense(32,relu)
  ├─ Wide branch:  Dense(16)
  └─ Deep branch:  Dense(128,relu) → Dropout(0.3) → Dense(64,relu) + shortcut → Dense(32,relu)
Concatenate([interaction, wide, deep]) → Dense(1) × 3  ← diff_1hr, diff_2hr, diff_3hr
```

The model now has direct access to every sensor reading from the past 3 hours. The explicit lag features (`temp_lag30/60/120`, `humidity_lag30/60/120`, `pressure_lag30/60/120`) are retained as dedicated positional anchors within the flat vector — they appear at consistent offsets in the 4,860-dim input, giving the model direct gradient paths to historical temperature values without requiring it to discover them within the raw temporal sequence.

Training uses `timeseries_dataset_from_array` for memory-efficient windowed batching.

**Features (27 per timestep)**: `time_of_day_sin/cos`, `day_of_year_sin/cos`, current sensor readings (`relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `uv`, `illuminance`, `solar_radiation`), rolling slopes (temperature, illuminance, solar, pressure, humidity via Numba JIT), explicit lags `temp_lag30/60/120`, `humidity_lag30/60/120`, `pressure_lag30/60/120`

**Results**:
| Run | val_loss | val_mae | Best Epoch | Model Size |
|-----|----------|---------|------------|------------|
| Model 5 new arch Run 1 (SF) | 0.000706 | 0.002388 | 18 | 787.74 KB |
| Model 5a Pi Run 1 | **0.000682** | **0.002355** | 15 | **787.74 KB** |

**Feature importance** (Model 5a Pi, permutation): `time_of_day_sin` (0.000168) > `day_of_year_sin` (0.000088) > `time_of_day_cos` (0.000074) > `relative_humidity` (0.000063). Note: importance values are small in absolute terms because val_loss is now so low (~0.000682) that permuting any single feature has limited impact on an already near-floor loss.

> **Edge TPU note**: The 4,860-dim input (4,800 bytes/dim × 4,860 ≈ 23 MB) overflows the Edge TPU's 8 MB SRAM. This model runs on CPU TFLite, not on the TPU. See Model 5a clean below for the Edge TPU-compatible version.

**Raspberry Pi deployment variant**: `workspace/Model 5a pi/` packages the training script and inference tooling for Pi-side workflows, with Numba JIT slopes and all-CPU-core parallelism.

**Lesson learned**: A 180-minute window with explicit lag anchors gives the model the full temperature trajectory it needs. The same architecture that failed as a flat snapshot (Model 5, val_loss 0.0108) achieves a ~15× reduction in val_loss when the model can see all 180 minutes of history at once.

---

### Step 5a — Edge TPU Compatible: Model 5a Clean

**Directory**: `workspace/Model 5a clean/`

The production-quality evolution of Model 5a that solves the Edge TPU SRAM overflow and further improves accuracy through gap-aware training.

**Key changes**:
- **`AveragePooling1D(pool_size=6, strides=6)`** before flattening: reduces 180 timesteps → 30 timesteps, bringing flattened input from 4,860 to 810 dims — safely under the Edge TPU's ~1,660-dim SRAM threshold
- **Gap invalidation**: target values that span a sensor data gap are nulled before training, removing a systematic downward bias in loss that had been inflating prior results
- **`ReduceLROnPlateau`**: patience=5, factor=0.5, min_lr=1e-7 for more stable convergence

**Results** (multiple experiments):
| Experiment | Configuration | val_loss | Model Size | Edge TPU |
|------------|---------------|----------|------------|----------|
| Exp 1 (reference) | No pooling | 0.000706 | 787.74 KB | No (SRAM overflow) |
| dense_wide_run1 | No pooling + gap fix | **0.000373** | 84.9 KB | No (SRAM overflow) |
| avgpool_run1 | AveragePooling1D(6,6) + gap fix | 0.000508 | 90.6 KB | **Yes** |
| no_tod_run1 | No time-of-day features | 0.000555 | 83.1 KB | Yes |

**Feature importance** (dense_wide_run1): `illuminance` (0.0237) > `time_of_day_sin` (0.0224) > `time_of_day_cos` (0.0215) > `solar_radiation` (0.0212) — features are distributed much more evenly than in earlier models; `temp_lag120` ranks 9th (0.0147).

**Lesson learned**: Gap invalidation alone (Exp 4) was the single biggest accuracy improvement, cutting val_loss from 0.000706 to 0.000373. Prior models were unwittingly trained on targets that spanned sensor outages, which introduced a systematic bias. The Edge TPU-compatible model (avgpool, 90.6 KB) trades ~36% accuracy for full TPU acceleration; with the gap fix, 0.000508 is still dramatically better than the flat Model 5 (0.010794).

---

### Step 6 — Conv2D Architecture Experiments

**Directory**: `workspace/Model 5b Conv2D/`

With a strong baseline (Model 5a clean, val_loss 0.000373), this step experiments with a **Conv2D architecture** that treats the 180-minute × n-feature input as a 2D grid and uses convolutional blocks to learn temporal and cross-feature patterns without flattening.

**Architecture** (Exp 35 — current production configuration):
```
Input: (180, n_features)
  ├─ Anchor path: input[:, -1, :]  ← current-timestep slice (STRIDED_SLICE, Edge TPU ✅)
  │    → Dense(32, L2) → ReLU6 → anchor(32)
  └─ Conv2D path:
       Reshape(180, n_features, 1)
       → Conv2D(96, k=(3,1)) → BN → ReLU6   ← short temporal patterns
       → Conv2D(96, k=(7,1)) → BN → ReLU6   ← medium temporal patterns
       → Conv2D(96, k=(15,1)) → BN → ReLU6  ← longer temporal patterns
       → Conv2D(96, k=(1, n_features)) → BN → ReLU6  ← feature mixing
       → GlobalAveragePooling2D → (96,)
       → Dense(64, L2) → ReLU6 → conv_context(64)
Concatenate([conv_context(64), anchor(32)]) → Dense(32, L2) → ReLU6 → Dense(1) × 3
```

All operations are Edge TPU-compatible (STRIDED_SLICE anchor, Conv2D, BN, ReLU6, GAP, Dense — no custom ops or dynamic shapes).

**Explicit features** (Exp 37, 28 total): 12 raw station readings + 6 cyclical time encodings + `temp_delta_1` + `temp_lag60` + `temp_lag120` + `temp_lag180` + 6 rolling slopes

**Experiment history** (key milestones):

| Exp | Key Change | float val_loss | Quant MAE (1/2/3hr °C) | Edge TPU |
|-----|-----------|---------------|------------------------|---------|
| 24 | Best Conv1D result | 0.001343 | 0.61/1.16/2.63 | ❌ |
| 27 | Switch to Conv2D + GAP (first PTQ success) | 0.0027 | 1.57/2.21/2.63 | ✅ |
| 28 | + skip path (temp, lag60, lag120 → Dense) | 0.0028 | 1.12/1.63/2.01 | ✅ |
| 29 | + 6 slope features | 0.0026 | 0.82/1.49/1.63 | ✅ |
| **32** | **ReduceLROnPlateau + L2 + no Dropout** | **0.0024** ⭐ | 0.67/1.39/1.71 | ✅ |
| 34 | Replace GAP with multi-point extraction | 0.0104 ❌ | — | ✅ |
| 35 | Revert GAP + wider filters (64→96) | 0.002368 | not quantized | ✅ |
| 36 | Saliency analysis (no training) | — | — | — |
| 37 | Add `temp_lag180` (28th feature) | in progress | — | ✅ |

**Key findings**:
- The anchor path (last-timestep skip connection) is essential: without it, GlobalAveragePooling dilutes single-timestep anchor values 180× and lag features rank last
- `ReduceLROnPlateau` moved best epoch from 15 → 46 and narrowed train/val gap from ~10× to ~3.1× — the best regularisation intervention across all Conv2D experiments
- PTQ post-training quantization (without QAT) consistently degrades MAE by 3–4× vs. float; QAT is required to close this gap
- Best float val_loss is **0.0024** (Exp 32), approximately 3.5× worse than Model 5a clean (0.000373)

**Lesson learned**: Simpler architectures generalise better at this data scale. The wide-deep-interaction dense model (Model 5a clean) outperforms the Conv2D architecture despite 28 experiments across varied filter counts, skip connections, regularisation strategies, and pooling approaches. At ~500 K training samples, feature engineering dominates architectural exploration.

---

### Step 7 — A New Prediction Target: Solar Radiation

**Directory**: `workspace/Model 6/`

Model 6 applies the same wide-deep-interaction architecture to a different prediction task: **solar radiation change** at +30min, +60min, +90min. This is a complementary signal useful for forecasting cloud cover changes.

**Feature additions** specific to solar context:
- `solar_clear_sky_ratio`, `clear_sky_deficit`, `solar_illuminance_ratio`
- Solar variability stats: `solar_radiation_{variance,change,mean,std}_30min`
- `fog_likelihood`, `fog_indicator` (computed from humidity + solar ratio)
- `marine_push_score`, `marine_push_flag` (coastal fog/marine layer signals)

**Features**: 48 total (the most feature-engineered model so far)

**Results**: val_loss = 0.0185, val_mae = 0.025, model size ~48 KB. Top features: `time_of_day_sin` and `solar_radiation_mean_30min`.

---

## Performance Summary

| Model | Architecture | Predicts | val_loss | val_mae | Size |
|-------|-------------|----------|----------|---------|------|
| Model 1 | Wide-deep dense | Absolute temp +1/2/3hr | 0.004022 | 0.012984 | 32.9 KB |
| Model 1a | Wide-deep dense (1 head) | Absolute temp +1hr | 0.003164 | 0.041769 | 32.3 KB |
| Model 2 | Dilated Conv1D | Absolute temp +1/2/3hr | 0.018693 | 0.031361 | 65 KB |
| Model 3 | Conv1D TPU-optimised | Absolute temp +1/2/3hr | 0.021779 | 0.033171 | 79 KB |
| Model 4 | INT16 MSB/LSB dense | Absolute temp +1/2/3hr | 0.017228 | 0.028576 | 53.2 KB |
| Model 5 | Wide-deep + interaction | Temp diff +1/2/3hr | 0.010794 | 0.019958 | 50.7 KB |
| Model 5a Pi | Sequence-flattened wide-deep | Temp diff +1/2/3hr | 0.000682 | 0.002355 | 787.74 KB |
| **Model 5a clean (avgpool)** | **Sequence-flattened + pooling** | **Temp diff +1/2/3hr** | **0.000508** | — | **90.6 KB** ✅ Edge TPU |
| Model 5a clean (no pool) | Sequence-flattened | Temp diff +1/2/3hr | 0.000373 | 0.002781 | 84.9 KB |
| Model 5b Conv2D (Exp 32 best) | Conv2D + anchor skip | Temp diff +1/2/3hr | 0.0024 | — | ~193 KB ✅ Edge TPU |
| Model 6 | Wide-deep + interaction | Solar diff +30/60/90min | 0.0185 | 0.025 | 48 KB |

> All metrics are normalised (targets scaled to approximately `[−1, 1]`).  
> Model 5a clean avgpool is the current production deployment (Edge TPU, 0.000508 val_loss).  
> Model 5a clean no-pool achieves the best absolute accuracy (0.000373) but requires CPU TFLite.

---

## Key Learnings

1. **Predict differences, not absolute values.** Models 5+ predict temperature *change* from current. It's a simpler target and consistently outperforms absolute prediction.

2. **Provide 180 minutes of history, not a snapshot.** The single biggest accuracy jump — from Model 5 (val_loss 0.0108) to Model 5a (val_loss 0.000682), roughly 15× — came from giving the model access to the full 3-hour temporal window, not from any feature substitution alone.

3. **Explicit lag features are essential anchor points.** Even with a 180-minute window, explicit lag scalars (`temp_lag30/60/120`) are retained as dedicated input features. They provide direct, consistent gradient paths to historical temperature values that the model cannot reliably discover by position within the raw sequence.

4. **Gap-aware training matters.** Target values that span a sensor data gap silently corrupt training. Invalidating cross-gap targets (using a 10-minute gap threshold, not 90 seconds) improved Model 5a clean from val_loss 0.000706 to 0.000373 — the single largest gain in the clean experiments.

5. **Cyclical time encoding with harmonics.** `sin`/`cos` of `time_of_day` and `day_of_year` — and their double harmonics (`sin2`, `cos2`) — are consistently high-importance features. Raw scalars are inferior.

6. **Simpler architectures generalise better** at this data scale. The sequence-flattened wide-deep model (Model 5a clean, 0.000373) outperformed 28+ Conv2D experiments despite Conv2D's stronger theoretical temporal inductive bias. At ~500 K training samples, feature engineering dominates architectural complexity.

7. **INT8 quantization is viable** with minimal accuracy loss when features are properly scaled with domain-aware bounds. `StandardScaler` suppresses meaningful seasonal variance; per-feature min/max with physical bounds preserves it.

8. **Gap-aware windowing**: the gap detection threshold matters enormously. At 90 s the function fires ~9,000 times on single missed readings and invalidates the majority of training data; at 600 s (10 minutes) it fires ~53 times on real outages and removes only ~0.9% of rows.

9. **Pre-normalise before the inference loop.** In `Inference_InfluxDB_Writer.py`, normalise the entire feature matrix once (vectorised NumPy) before the loop. Per-window normalisation inside the loop saturates CPU at ~3,240 Python ops per prediction — even though the TPU finishes inference in ~0.55 ms.

10. **Climate generalisation fails hard at distribution boundaries.** A model trained on San Francisco data clips predictions at the upper end of the SF training distribution when asked to forecast Palm Springs temperatures (~45°C). This is an out-of-distribution failure, not a capacity failure. Separate models with separate scalers per climate are the simplest fix.

---

## Training on Raspberry Pi 5

The Pi variants (`workspace/Model 1 pi/train_model_pi.py` and `workspace/Model 5a pi/train_model.py`) can train directly on the RPi5 against a local InfluxDB database, skipping the Mac → CSV → Mac pipeline entirely. The differences from the Mac scripts are non-trivial; this section documents each one and why it matters.

### 1. Entry point guard + `main()` wrapper

**Change**: Wrap all top-level code in `def main()` and add `if __name__ == "__main__": main()`.

**Why**: Multiprocessing on Linux re-imports the module in each worker process. Without the guard, top-level code (TF imports, data loading) executes again in every worker, causing crashes or silent hangs.

### 2. Force multiprocessing fork method

**Change** (top of `main()`, before any TF import):
```python
import multiprocessing as mp
mp.set_start_method("fork", force=True)
```

**Why**: Some TF ARM builds default to `"spawn"`, which requires pickling all state. Forcing `"fork"` on Linux matches the behaviour TF expects on the RPi.

### 3. Pin TensorFlow threads to all CPU cores

**Change**:
```python
cores = multiprocessing.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(cores)
tf.config.threading.set_inter_op_parallelism_threads(1)  # Model 5a Pi
tf.config.set_soft_device_placement(True)
```

**Why**: TF's default thread count on ARM often underestimates available cores. The RPi5 has 4 cores. Setting `inter_op=1` focuses all cores on one operation at a time rather than splitting them across concurrent ops — faster for sequential batch training.

### 4. Disable XLA JIT and layout optimizer

**Change** (Model 5a Pi only):
```python
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
```

**Why**: XLA JIT is tuned for GPUs and x86 SIMD. On ARM Cortex-A76 (RPi5) it adds compilation overhead without runtime benefit.

### 5. Replace scipy rolling slope with Numba JIT

**Change**: Replace `scipy.stats.linregress` inside `pandas.rolling().apply()` with:

```python
from numba import njit, prange

@njit(parallel=True)
def rolling_slope_numba(data, window):
    n = len(data)
    slopes = np.full(n, np.nan)
    x = np.arange(window)
    x_mean = np.mean(x)
    denom = np.sum((x - x_mean) ** 2)
    for i in prange(window - 1, n):
        y = data[i - window + 1:i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = np.mean(y)
        slopes[i] = np.sum((x - x_mean) * (y - y_mean)) / denom
    return slopes
```

**Why**: `pandas.rolling().apply()` with a Python lambda is ~50× slower than the Numba version. On 800 K rows the scipy path takes minutes; the Numba path takes seconds because `prange` distributes across all 4 RPi5 cores and the JIT compiles to NEON SIMD.

### 6. Cast feature arrays to float32

```python
X_train = X_train.values.astype(np.float32)
```

**Why**: NumPy/pandas default to `float64`. `float32` halves the memory footprint on the RPi's limited RAM.

### 7. Load data directly from InfluxDB instead of CSV

**Change**: Connect to the local InfluxDB instance with time-window–paginated queries.

**Why**: The RPi hosts the live InfluxDB database. Time-window pagination avoids InfluxDB v1.x's `max-select-series` limit that triggers when using `LIMIT/OFFSET` on large measurements.

### 8. Gap-aware target invalidation

```python
def _invalidate_targets_crossing_gaps(df, label, tol_s=600):
    # Nulls temp_t+1/2/3hr for rows whose future lookup crosses a gap > 10 min
    ...

DIFF_CLIP = 12.0  # °C — removes physically implausible outlier targets
df[diff_targets] = df[diff_targets].clip(-DIFF_CLIP, DIFF_CLIP)
```

**Why**: `df['temp_t+1hr'] = df['temperature'].shift(-60)` is purely positional. A Wi-Fi outage creates a gap where pre-gap rows get post-gap temperatures as targets — silent data leakage. `tol_s=600` (10 minutes) avoids over-triggering on the ~9,000 single missed-minute readings that are harmless.

### Summary of required changes

| Change | Model 1 Pi | Model 5a Pi | Why |
|--------|-----------|-------------|-----|
| `main()` wrapper + entry guard | yes | yes | Linux multiprocessing safety |
| `mp.set_start_method("fork")` | yes | yes | TF thread pool compatibility |
| Pin intra/inter-op thread count | yes | yes | Use all RPi5 cores |
| Disable XLA JIT + layout optimizer | — | yes | No benefit on ARM |
| Numba `@njit(parallel=True)` slope | yes | yes | ~50× faster than scipy/pandas |
| Float32 explicit cast | yes | yes | Halve memory footprint |
| Load from InfluxDB (not CSV) | yes | yes | Local DB; avoid large CSV export |
| Gap-aware target invalidation (tol_s=600) | — | yes | Prevent cross-gap data leakage |
| Target clipping (±12°C) | — | yes | Remove physically implausible outliers |

```bash
pip install numba influxdb tensorflow flatbuffers
```

---

## Running a Model

### Train
```bash
cd workspace/Model\ 5a\ clean/
python train_model.py
```

### Compile for Edge TPU
```bash
# Uses the x86 compiler container (required on Apple Silicon and Raspberry Pi):
./edgetpu-x86-compiler.sh weather_model_5a_best.tflite
# Produces: weather_model_5a_best_edgetpu.tflite
```

### Deploy to Raspberry Pi
```bash
scp workspace/Model\ 5a\ clean/weather_model_5a_best_edgetpu.tflite pi@raspberrypi:~/
scp workspace/Model\ 5a\ clean/input_scaler_5a.json pi@raspberrypi:~/
scp workspace/Model\ 5a\ clean/target_scaler_5a.json pi@raspberrypi:~/
```

### Run inference
```bash
# On the Raspberry Pi:
python Inference_InfluxDB_Writer.py
```

---

## Additional Notes

- See `dual-edge-tpu-fix.md` for a short guide to fixing dual Coral Edge TPU detection/runtime issues on Raspberry Pi deployments.

---

## Related Project: SolarChargeML

**Directory**: `workspace/SolarChargeML/`

A separate initiative that lives in this repo because it reuses the same Tempest weather station data (`solar_radiation`, `illuminance`, `uv`, wind, etc.) as a feature source — but it doesn't predict temperature. It predicts **excess solar power 5 minutes ahead**, as a candidate drop-in replacement for the average+slope heuristic in a companion project,
[ChargePoint-SunPower-ChargeManager](https://github.com/dacarson/ChargePoint-SunPower-ChargeManager)'s `solar_charge_controller.py`, which throttles EV charging amperage to track available solar excess (biasing toward grid draw off-peak, toward export on-peak) through passing clouds, fog, and unscheduled house loads.

**Method**: `sklearn.ensemble.HistGradientBoostingRegressor` (gradient-boosted trees, not TensorFlow — chosen for lightweight CPU inference on the same Raspberry Pi that runs the charge controller), trained on 30-second-resolution InfluxDB features joining PVS6 solar/load data with this repo's WeatherFlow station data, daytime-only. Predicts the *delta* from current excess rather than the raw value — see the experiment log for why a 1-minute grid and three earlier target/feature reformulations all failed to beat the heuristic before finer time resolution did.

**Status**: offline backtest beats the existing heuristic by 5.6% MAE (and naive persistence by 11.1%). A read-only shadow validator is now running live on the deployed Raspberry Pi, logging what the model would predict alongside the real controller's own predictions for direct comparison — not yet controlling real charging.

Full detail: `workspace/SolarChargeML/SOLARCHARGE_PLAN.md` (rationale, data sources, deployment notes) and `SOLARCHARGE_EXPERIMENT_LOG.md` (full run-by-run history).

---

## Project Structure

```
.
├── Dockerfile.tpu                          # Docker image for training
├── edgetpu-x86-compiler.sh                 # Compiles .tflite → _edgetpu.tflite
├── dual-edge-tpu-fix.md                    # Notes on dual Edge TPU setup/fix
├── run_dev.sh                              # Launches Docker dev container
└── workspace/
    ├── export_influx_to_csv*.py            # Export weather data from InfluxDB
    ├── create_combined_data.py             # Merge multi-location datasets
    ├── cron/                               # Scheduled inference jobs
    ├── Model 1/                            # Wide-deep baseline (daytime filter, 3 outputs)
    ├── Model 1a/                           # Single-output (+1hr only) variant
    ├── Model 1 Daytime/                    # Full 24-hour data, no filter
    ├── Model 1 INT/                        # INT quantization variant
    ├── Model 1 Periodic/                   # Cyclic time encoding
    ├── Model 1 diffs/                      # Predict temp change; 28-31 features
    ├── Model 1 combined/                   # Diff + cyclic features
    ├── Model 1 Pi/                         # Pi-optimised training
    ├── Model 1 PS/                         # Palm Springs dataset
    ├── Model 2/                            # Dilated Conv1D (180-step window)
    ├── Model 3/                            # Conv1D TPU-optimised (90-step window)
    ├── Model 4/                            # INT16 MSB/LSB hybrid encoding
    ├── Model 4a/                           # Hybrid precision (incomplete)
    ├── Model 5/                            # Flat feature vector, temp diff targets
    ├── Model 5 new arch. slope calc/       # Sequence-flattened; first 0.000706 result
    ├── Model 5a pi/                        # Pi port of Model 5a (0.000682)
    ├── Model 5a clean/                     # BEST MODEL — gap fix + Edge TPU pool (0.000373 / 0.000508)
    ├── Model 5b Conv2D/                    # Conv2D architecture experiments (best 0.0024, Exp 32)
    ├── Model 6/                            # Solar radiation prediction
    ├── Forecaster_1/                       # Weather condition classifier (F1 0.523, complete)
    ├── Forecaster_2/                       # Hierarchical redesign (in progress)
    └── SolarChargeML/                      # EV charge optimization (related project, see below)
```

---

## License

MIT
