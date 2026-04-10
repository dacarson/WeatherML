# WeatherML Experiment Specification

## Overview

This project trains ML models on Tempest Weather Station data to predict future temperatures at **+1hr**, **+2hr**, and **+3hr** horizons. Models are trained on a MacBook Pro, quantized to INT8 TFLite, compiled for the Coral Edge TPU, and deployed for inference on a Raspberry Pi.

### Strategic goal (raw-sequence inference on Edge TPU)

The **overall product goal** is to avoid **pre-calculating derived features** (for example explicit lag columns such as `temp_lag30`, `humidity_lag60`, or other hand-built transforms in Python or in the inference path). Instead, the system should feed the model a **single 180-minute window** of appropriately scaled **raw (or minimally processed) station readings**—i.e. the same time span models like Model 5 already use as context—and let the network learn temporal structure internally.

**Success criterion**: validation quality **at least as good as Model 5a** (see Model 5 new arch. in this spec: strong `val_loss` / `val_mae` on the multi-horizon temperature-difference task), while meeting the “no pre-computed lag features” constraint in training and at deploy time, **and the compiled model must run on the Coral Edge TPU**.

**Edge TPU constraints** (hard requirements for any candidate architecture):
- All ops must be INT8-quantizable and Edge TPU-compilable (no LSTM, GRU, or standard attention — these are not supported)
- Supported ops: `Conv1D`, `Conv2D`, `DepthwiseConv2D`, `Dense`, `ReLU`, `ReLU6`, `GlobalAveragePooling`, `Add`, `Concatenate`, `Reshape`, `BatchNormalization`
- Model must survive `edgetpu_compiler` with all ops mapped to the TPU (no CPU-fallback ops)
- INT8 quantization via representative dataset must not cause significant accuracy degradation
- Practical model size limit: Edge TPU has 8 MB of SRAM; compiled model + parameters must fit

Note: cyclical encodings of calendar time (e.g. `sin`/`cos` of time-of-day) are a gray area—count as “hand features” unless the model consumes raw timestamps and learns periodicity. The north-star is **minimal feature engineering**: longest term, ideally only normalization + stacking the 180-minute tensor.

---

## Data

| File | Location | Description |
|------|----------|-------------|
| `train_data.csv` | `workspace/` | San Francisco — training, April 9 2023–April 8 2024 |
| `val_data.csv` | `workspace/` | San Francisco — validation, April 9 2024–April 8 2025 |
| `train_data_ps.csv` | `workspace/` | Palm Springs — training data |
| `val_data_ps.csv` | `workspace/` | Palm Springs — validation data |

**Raw input features** (from Tempest station): `temperature`, `relative_humidity`, `station_pressure`, `solar_radiation`, `illuminance`, `uv`, `wind_avg`, `wind_gust`, `wind_lull`, `wind_direction`, `rain_accumulated`, `timestamp`, `day_of_year`, `time_of_day`

**Derived features** used by models: lag values (30/60/120 min), delta/slope values, cyclical encodings (`sin`/`cos` of time-of-day and day-of-year), gap indicators.

---

## Deployment Pipeline

1. **Train** — Python + TensorFlow on MacBook Pro, outputs a Keras `.keras` model
2. **Quantize** — Convert to INT8 TFLite (representative dataset quantization)
3. **Compile** — `edgetpu_compiler` (via Docker container) produces `_edgetpu.tflite`
4. **Deploy** — Copy `_edgetpu.tflite` + scaler JSON files to Raspberry Pi
5. **Infer** — `Inference_InfluxDB_Writer.py` reads live data, runs model on Coral Edge TPU, writes predictions to InfluxDB

---

## Model Catalog

Models are numbered in rough chronological/experimental order. Each directory under `workspace/` contains the training script, scaler JSON files, TFLite artifacts, and per-run results JSON.

Metrics reported are the best run's normalized `val_loss` (MSE) and `val_mae`. All targets are scaled to roughly `[-1, 1]` before training.

---

### Model 1 — Dense-Wide Baseline

**Directory**: `workspace/Model 1/`  
**Target**: Absolute temperature at t+1hr (single output)  
**Architecture**:
- Input: 12 features
- Wide branch: `Dense(16)`
- Deep branch: `Dense(128, relu)` → `Dropout(0.3)` → `Dense(64, relu)` → residual shortcut `Dense(64)` → `Add()` → `Dense(32, relu)`
- Output: `Concatenate([wide, deep])` → `Dense(1, linear)`

**Features**: `illuminance`, `solar_radiation`, `uv`, `relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `day_of_year`, `time_of_day`, `temperature_delta`, `temp_lag1`, `humidity_lag1`

**Training**: Adam lr=1e-5, MSE loss, 50 epochs, batch 32, min/max feature scaling ±5% padding

**Results** (best run):

| Metric | Value |
|--------|-------|
| val_loss | 0.0040 |
| val_mae | 0.0130 |
| Model size | ~33 KB |
| Top feature | `temp_lag1` (0.082) |

---

### Model 1 INT — Dense-Wide INT Quantization Variant

**Directory**: `workspace/Model 1 INT/`  
**Target**: Absolute temperature at t+1hr  
**Architecture**: Same as Model 1, with INT8/INT16 quantization focus during conversion

**Results** (best run):

| Metric | Value |
|--------|-------|
| val_loss | 0.0094 |
| val_mae | 0.0202 |
| Model size | ~33 KB |
| Top feature | `temp_lag1` (0.181) |

---

### Model 1 Daytime — Daytime-Only Variant

**Directory**: `workspace/Model 1 Daytime/`  
**Target**: Absolute temperature at t+1hr, trained only on daytime records  
**Architecture**: Same as Model 1

**Results** (best of 10 runs):

| Metric | Value |
|--------|-------|
| val_loss | 0.0040 |
| val_mae | 0.0130 |
| Model size | ~33 KB |
| Top feature | `temp_lag1` (0.078) |

---

### Model 1 Periodic — Cyclic Time Encoding Variant

**Directory**: `workspace/Model 1 periodic/`  
**Target**: Absolute temperature at t+1hr  
**Architecture**: Same as Model 1, but `time_of_day` replaced with `sin`/`cos` cyclic encodings  
**Training**: 10 runs

**Results** (best of 10 runs):

| Metric | Value |
|--------|-------|
| val_loss | 0.0045 |
| val_mae | 0.0137 |
| Model size | ~33 KB |
| Top feature | `temp_lag1` (0.075) |

---

### Model 1 Diffs — Temperature Difference Variant

**Directory**: `workspace/Model 1 diffs/`  
**Target**: Temperature **change** from current (diff_1hr, diff_2hr, diff_3hr)  
**Architecture**: Same wide-deep structure, 3 outputs  
**Features**: Adds cyclical encodings and delta features; removes raw `temp_lag1`  
**Training**: 10 runs

**Results** (best of 10 runs):

| Metric | Value |
|--------|-------|
| val_loss | 0.0114 |
| val_mae | 0.0225 |
| Model size | ~34 KB |
| Top features | `time_of_day_sin` (0.011), `time_of_day_cos` (0.005) |

---

### Model 1 Combined — Diff + Cyclic Features

**Directory**: `workspace/Model 1 combined/`  
**Target**: Temperature differences with combined diff + cyclic feature sets  
**Architecture**: Same wide-deep, expanded feature set with lag features, cyclic encodings, and seasonal interaction terms  
**Training**: 5 runs

**Results** (best of 5 runs):

| Metric | Value |
|--------|-------|
| val_loss | 0.0222 |
| val_mae | 0.0316 |
| Model size | ~36 KB |
| Top features | `time_of_day_sin` (0.041), `time_of_day_cos` (0.028) |

---

### Model 1a — Extended Training Variant

**Directory**: `workspace/Model 1a/`  
**Target**: Absolute temperature at t+1hr  
**Architecture**: Identical to Model 1  
**Training**: 100 epochs with early stopping (patience=5)

**Results** (best run):

| Metric | Value |
|--------|-------|
| val_loss | 0.00316 |
| val_mae | 0.0418 |
| Best epoch | 8 |
| Model size | ~32 KB |

---

### Model 1 Pi — Raspberry Pi Optimized

**Directory**: `workspace/Model 1 pi/`  
**Target**: Temperature differences (diff_1hr, diff_2hr, diff_3hr)  
**Architecture**: Same as Model 1 Diffs, optimized for Pi deployment  
**Training**: 10 runs

**Results** (best of 10 runs):

| Metric | Value |
|--------|-------|
| val_loss | 0.0114 |
| val_mae | 0.0225 |
| Model size | ~34 KB |

---

### Model 1 PS — Palm Springs Dataset

**Directory**: `workspace/Model 1 PS/`  
**Target**: Absolute temperature at t+1hr, trained on Palm Springs data  
**Architecture**: Same as Model 1

---

### Model 2 — Conv1D Dilated Residual

**Directory**: `workspace/Model 2/`  
**Target**: Absolute temperature at t+1hr, t+2hr, t+3hr (3 outputs)  
**Architecture**:
- Input: 180-minute window × 15 features (time-series)
- `Conv1D(32, kernel=3, padding='same')`
- 4× Residual dilated blocks: `Conv1D(32, kernel=3, dilation=[1,2,4,8])` + BatchNorm + Add
- `GlobalAveragePooling1D()`
- `Dense(64, relu, L1=1e-5)` → `Dropout(0.3)` → `Dense(32, relu)` → `BatchNorm`
- 3 output heads (linear)

**Features (15)**: `illuminance`, `solar_radiation`, `uv`, `relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `temperature_delta`, `temp_lag1`, `humidity_lag1`, `sin_time_of_day`, `cos_time_of_day`, `day_of_year`, `delta_minutes`, `is_gap`

**Training**: Adam lr=1e-4, MSE, 100 epochs, batch 32, early stopping patience=5

**Results**:

| Metric | Value |
|--------|-------|
| val_loss | 0.0187 |
| val_mae | 0.0314 |
| MAE (denormalized) | ~1.75°C |
| Model size | ~65 KB (quantized) |

---

### Model 3 — Conv1D TPU-Optimized

**Directory**: `workspace/Model 3/`  
**Target**: Temperatures at t+1hr, t+2hr, t+3hr  
**Architecture**: Similar to Model 2 but simplified for Edge TPU compatibility
- Input: 90-minute window (reduced from 180 min)
- `Conv1D(32)` + dilated residual blocks `[1,2,4,8]`
- `GlobalAveragePooling1D()` → `Dense(64, relu)` → `Dropout(0.3)` → `Dense(32, relu)` → `BatchNorm`
- 3 output heads

**Features (8)**: `temp_avg_15min`, `temperature_delta`, `sin_time_of_day`, `cos_time_of_day`, `illuminance`, `solar_radiation`, `station_pressure`, `relative_humidity`

**Training**: CPU-only (GPU disabled), Adam lr=1e-4, MSE, 100 epochs, batch 32

**Results**:

| Metric | Value |
|--------|-------|
| val_loss | 0.0218 |
| val_mae | 0.0332 |
| MAE (denormalized) | ~2.95°C |
| Best epoch | 1 (early stopping) |
| Model size | ~79 KB (quantized) |

**Note**: Overfitting observed; simplified feature set led to higher errors than Model 2.

---

### Model 4 — INT16 Hybrid Precision

**Directory**: `workspace/Model 4/`  
**Target**: Temperature at t+1hr (single output)  
**Architecture**:
- Input: 24 features (12 base features split into LSB/MSB pairs for INT16 representation)
- Wide branch: `Dense(16)`
- Deep branch: `Dense(128, relu)` → `Dropout(0.3)` → `Dense(64, relu)` → residual `Dense(64)` → `Add()` → `Dense(32, relu)`
- Interaction layer: `Dense(16, relu)` → element-wise `Multiply()` → `Concatenate()` → `Dense(32, relu)`
- Output: `Concatenate([wide, deep, interaction])` → `Dense(1, linear)`

**Features (12 base)**: `illuminance`, `solar_radiation`, `uv`, `relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `day_of_year`, `time_of_day`, `temperature_delta`, `temp_lag1`, `humidity_lag1`

**Training**: Adam lr=1e-5, MSE, 99 epochs, batch 32

**Results**:

| Metric | Value |
|--------|-------|
| val_loss | 0.0172 |
| val_mae | 0.0270 |
| Model size | ~21 KB |
| Top feature | `temp_lag1` (0.126) |

---

### Model 4a — Hybrid Precision Experimental

**Directory**: `workspace/Model 4a/`  
**Status**: Incomplete/experimental. Training script present (`train_hybrid_precision_model.py`) but no completed results.

---

### Model 5 — Temperature Difference (Wide-Deep-Interaction)

**Directory**: `workspace/Model 5/`  
**Target**: Temperature **change** at +1hr, +2hr, +3hr (`diff_1hr`, `diff_2hr`, `diff_3hr`)  
**Architecture**:
- Input: 28 features
- Wide: `Dense(16)`
- Deep: `Dense(128, relu)` → `Dropout(0.3)` → `Dense(64, relu)` → residual → `Dense(32, relu)`
- Interaction: `Dense(16, relu)` → pairwise `Multiply()` → `Concatenate()` → `Dense(32, relu)`
- Output: `Concatenate([wide, deep, interaction])` → 3× `Dense(1, linear)`

**Features (28)**: delta features (`illuminance_delta`, `solar_radiation_delta`, `temperature_delta`, `pressure_delta`, `humidity_delta`), multi-horizon lags (`temp_lag{30,60,120}`, `humidity_lag{30,60,120}`, `wind_{avg,gust}_lag30`, `uv_lag30`, `pressure_lag30`), cyclical (`time_of_day_{sin,cos,sin2,cos2}`, `day_of_year_{sin,cos}`), wind (`wind_avg`, `wind_gust`, `wind_direction_{sin,cos}`, `wind_lull`), `uv`, `rain_accumulated`

**Training**: Adam lr=1e-5, MSE, ~77 epochs, batch 256, target scaled to `[-1, 1]` with ±2°C padding. 5 runs.

**Results** (best run):

| Metric | Value |
|--------|-------|
| val_loss | 0.0108 |
| val_mae | 0.0219 |
| Model size | ~39 KB |
| Top features | `time_of_day_sin` (0.012), `day_of_year_cos` (0.003) |

---

### Model 5 (new arch. slope calc) — Best Performing Model

**Directory**: `workspace/Model 5 new arch. slope calc/`  
**Target**: Temperature change at +1hr, +2hr, +3hr  
**Architecture**: Same wide-deep-interaction as Model 5  
**Key innovation**: Replaces delta features with Numba-accelerated **slope calculations** over multiple time windows; adds higher-order cyclical terms

**Features (27)**: Similar to Model 5 but delta features replaced with computed slopes; includes `temp_lag120` explicitly; higher-order time harmonics

**Training**: Adam lr=1e-5, MSE, ~97 epochs, batch 256. 2 runs.

**Results** (best run):

| Metric | Value |
|--------|-------|
| **val_loss** | **0.000682** |
| **val_mae** | **0.00445** |
| Best epoch | 97 |
| Model size | ~788 KB |
| Top features | `temp_lag120` (0.093), `time_of_day_cos2` (0.088), `time_of_day_cos` (0.075) |

> **Best performing model to date.** val_loss is ~15× lower than Model 5 and ~60× lower than Model 1. The larger model size (788 KB vs ~34 KB) is a trade-off for this accuracy gain.

---

### Model 5b Conv2D — 2D Convolution with Lag Features

**Directory**: `workspace/Model 5b Conv2D/`  
**Target**: Temperature change at +1hr, +2hr, +3hr  
**Architecture**:
- Input: 180-minute window × n_features, reshaped for `Conv2D`
- Conv2D branch: multiple convolution blocks for temporal-feature interactions
- Dense branch: learned patterns via dense layers
- Lag extraction branch: explicit lag values at 30/60/120 min
- Output: `Concatenate([conv_branch, dense_branch, lag_branch])` → 3 output heads

**Features (30)**: `temperature`, `temp_delta_1`, `uv`, `wind_{avg,gust,lull}`, `solar_radiation`, `illuminance`, `relative_humidity`, `station_pressure`, cyclical time/day/wind encodings, multi-horizon lags, `rain_accumulated`

**Training**: Adam lr=5e-4 (later reverted to 1e-5), weighted Huber loss (weights 1.0/1.3/1.9 for 1hr/2hr/3hr), 100 epochs, batch 512, gap-aware windowing

**Results**:

| Metric | Value |
|--------|-------|
| val_loss | 0.00775 |
| val_mae | 0.0159 |
| Best epoch | 73 |
| Model size | ~844 KB |
| Top features | `time_of_day_cos` (0.242), `uv_lag30` (0.232) |

**Note**: See `MODEL_5B_EXPERIMENT_LOG.md` for detailed optimization history. Gap-aware windowing prevents training on windows that span data collection gaps.

---

### Model 6 — Solar Radiation Change Prediction

**Directory**: `workspace/Model 6/`  
**Target**: Solar radiation **change** at +30min, +60min, +90min  
**Architecture**: Same wide-deep-interaction as Model 5  
**Input**: 48 features

**Features (48)**: base weather features, extensive solar context (`solar_radiation_deviation`, `solar_clear_sky_ratio`, `clear_sky_deficit`, `solar_illuminance_ratio`), solar variability stats (`solar_radiation_{variance,change,mean,std}_30min`), fog indicators (`fog_likelihood`, `fog_indicator`), marine push indicators (`marine_push_score`, `marine_push_flag`), UV/humidity stats

**Training**: Adam lr=1e-5, MSE, ~18 epochs, batch 32, target scaled to `[-1, 1]` with ±10 W/m² padding. 5 runs.

**Results** (best run):

| Metric | Value |
|--------|-------|
| val_loss | 0.0185 |
| val_mae | 0.0249 |
| Model size | ~48 KB |
| Top features | `time_of_day_sin` (0.021), `solar_radiation_mean_30min` (0.014) |

**Note**: Different prediction task — solar radiation, not temperature. Useful as a complementary signal.

---

## Summary Comparison

| Model | Target | val_loss | val_mae | Size |
|-------|--------|----------|---------|------|
| Model 1 | Absolute temp +1hr | 0.0040 | 0.0130 | 33 KB |
| Model 1a | Absolute temp +1hr | 0.00316 | 0.0418 | 32 KB |
| Model 1 INT | Absolute temp +1hr | 0.0094 | 0.0202 | 33 KB |
| Model 1 Daytime | Absolute temp +1hr | 0.0040 | 0.0130 | 33 KB |
| Model 1 Periodic | Absolute temp +1hr | 0.0045 | 0.0137 | 33 KB |
| Model 1 Diffs | Temp diff +1/2/3hr | 0.0114 | 0.0225 | 34 KB |
| Model 1 Combined | Temp diff +1/2/3hr | 0.0222 | 0.0316 | 36 KB |
| Model 2 | Absolute temp +1/2/3hr | 0.0187 | 0.0314 | 65 KB |
| Model 3 | Absolute temp +1/2/3hr | 0.0218 | 0.0332 | 79 KB |
| Model 4 | Absolute temp +1hr | 0.0172 | 0.0270 | 21 KB |
| Model 5 | Temp diff +1/2/3hr | 0.0108 | 0.0219 | 39 KB |
| **Model 5 slope calc** | **Temp diff +1/2/3hr** | **0.000682** | **0.00445** | **788 KB** |
| Model 5b Conv2D | Temp diff +1/2/3hr | 0.00775 | 0.0159 | 844 KB |
| Model 6 | Solar diff +30/60/90min | 0.0185 | 0.0249 | 48 KB |

---

## Key Learnings

1. **Predict differences, not absolute values** — Models 5+ predict temperature *change* from current, which is a simpler target and consistently outperforms absolute temperature prediction.

2. **Slope/rate features beat delta features** — The single biggest accuracy jump (Model 5 → Model 5 slope calc, 15× improvement) came from replacing raw deltas with Numba-computed slopes over multiple windows.

3. **Explicit lag features are essential** — `temp_lag1`, `temp_lag30/60/120` dominate feature importance across all models. Conv layers do not learn these implicitly as well as providing them explicitly.

4. **Cyclical time encoding matters** — `time_of_day_sin/cos` and `day_of_year_sin/cos` are consistently high-importance features. Raw `time_of_day` as a scalar is inferior.

5. **Model size vs. accuracy trade-off** — The best model (788 KB) is ~24× larger than early models (33 KB). Edge TPU deployment requires careful compilation and on-device memory budget checks.

6. **INT8 quantization is viable** — All models successfully quantize to INT8 TFLite with minimal accuracy degradation.

7. **Gap-aware windowing** (Model 5b) — Windows spanning data collection gaps corrupt training. Explicitly detecting and dropping these windows prevents subtle data integrity issues.

8. **Pre-normalize before the inference loop** — In `Inference_InfluxDB_Writer.py`, always normalize the entire feature matrix once (vectorized numpy) *before* the prediction loop, and pre-extract targets and temperature as plain numpy arrays. Do **not** normalize per-window inside the loop:
   ```python
   # CORRECT: normalize once
   scaled_data = np.clip((df[FEATURE_ORDER].values.astype(np.float32) - f_mins) / f_denoms, 0.0, 1.0)
   targets_arr = df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].values.astype(np.float32)
   temp_arr = df['temperature'].values.astype(np.float32)
   # then in the loop:
   scaled_window = scaled_data[window_start:i+1]   # cheap numpy slice
   ```
   Without this, each iteration runs a Python `for` loop over all features × `SEQ_LEN` rows (e.g. 18 features × 180 steps = 3240 Python ops per prediction), saturating the CPU despite the TPU completing inference in ~0.55 ms. Pre-normalizing reduces the per-iteration work to a single numpy stride, cutting CPU usage dramatically.

9. **InfluxDB `max-select-point` is a scan limit, not a result limit** — The server counts every point *examined* before applying `LIMIT`, so a query like `LIMIT 100000` will still fail if the measurement contains more than 100k points. The correct pattern (matching Model 5's approach) is to use bounded time-range queries:
   - Query `SELECT FIRST(...)` (1 point) to find the dataset start, then fetch `WHERE time >= start AND time <= start + QUERY_BATCH_SIZE + EXTRA_SAMPLES minutes`
   - After `BATCH_SIZE` inferences, exit with code 88 so `run_with_restart.py` relaunches and advances the window via `progress_diff.json`
   - This keeps each query well within the server scan limit (~70 days ≈ 100k points at 1 obs/min)
