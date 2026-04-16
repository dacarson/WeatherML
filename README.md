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

1. **Train** — Python + TensorFlow on MacBook Pro (CPU-only)
2. **Quantize** — INT8 TFLite via representative-dataset quantization
3. **Compile** — `edgetpu_compiler` in a Docker container (see `edgetpu-x86-compiler.sh`)
4. **Deploy** — Copy `_edgetpu.tflite` + scaler JSON files to Raspberry Pi
5. **Infer** — `Inference_InfluxDB_Writer.py` reads live data from InfluxDB, runs the model on the Coral TPU, writes predictions back to InfluxDB

---

## Data

One-minute observations from a Tempest Weather Station, exported from InfluxDB using `export_influx_to_csv*.py`.

| File | Description |
|------|-------------|
| `workspace/train_data.csv` | San Francisco — April 9 2023–April 8 2024 |
| `workspace/val_data.csv` | San Francisco — April 9 2024–April 8 2025 |
| `workspace/train_data_ps.csv` | Palm Springs — training |
| `workspace/val_data_ps.csv` | Palm Springs — validation |

Raw features: `temperature`, `relative_humidity`, `station_pressure`, `solar_radiation`, `illuminance`, `uv`, `wind_avg`, `wind_gust`, `wind_lull`, `wind_direction`, `rain_accumulated`, `day_of_year`, `time_of_day`

> Data files are not checked in (60–140 MB each). Regenerate them with the export scripts.

---

## Model Evolution — A Tutorial

All model directories live under `workspace/`. Each contains a training script, result JSON files, scaler JSON files, and (after training) compiled `.tflite` artifacts.

### Step 1 — The Baseline: Dense Wide-Deep Model

**Directory**: `workspace/Model 1/`

The first model is intentionally simple: a **wide-and-deep dense network** that takes a snapshot of 12 current weather features and predicts the temperature **1 hour ahead** (absolute value).

**Architecture**
```
Input (12 features)
  ├─ Wide branch:  Dense(16)                           ← memorization
  └─ Deep branch:  Dense(128,relu) → Dropout(0.3)
                   → Dense(64,relu) → residual Dense(64) → Add()
                   → Dense(32,relu)                    ← generalization
Concatenate([wide, deep]) → Dense(1)  ← single output: temp_t+1hr
```

**Features**: `illuminance`, `solar_radiation`, `uv`, `relative_humidity`, `station_pressure`, `wind_avg`, `wind_gust`, `day_of_year`, `time_of_day`, `temperature_delta` (15-sample rolling slope), `temp_lag1`, `humidity_lag1`

**Key design choices**:
- **Daytime-only filter** (`illuminance > 1400 lux`) to focus on the most predictable regime
- Per-feature min/max scaling with ±5% padding and domain bounds (e.g. humidity 0–100)
- Multi-run training (`run_with_restart.py`) to pick the best of N random initializations

**Results**: val_loss = 0.0040, val_mae = 0.013, model size ~33 KB

**Lesson learned**: `temp_lag1` (the previous minute's temperature) dominates feature importance by far. The model is essentially learning "temperature doesn't change much in one minute." That's a hint we're predicting the wrong thing.

---

### Step 1a — Feature Exploration: Variants of Model 1

Several variants explored specific questions without changing the core architecture:

| Variant | Question asked | Finding |
|---------|---------------|---------|
| **Model 1a** (`Model 1a/`) | Does training longer help? (100 epochs with early stopping) | Minimal gain; early stopping fires at epoch 8 |
| **Model 1 Daytime** (`Model 1 Daytime/`) | Daytime-only data at all hours? | No improvement over full data |
| **Model 1 INT** (`Model 1 INT/`) | Can INT16-style quantization help? | Worse (val_loss 0.0094 vs 0.0040) |
| **Model 1 Periodic** (`Model 1 periodic/`) | Replace `time_of_day` scalar with sin/cos cyclic encoding? | Slightly worse on this model, but the technique matters later |
| **Model 1 Diffs** (`Model 1 diffs/`) | Predict temperature *change* instead of absolute temp? | Higher loss at this stage, but sets up the key insight for Model 5 |
| **Model 1 Combined** (`Model 1 combined/`) | Combine diff + cyclic features? | No win; feature interaction matters more than volume |
| **Model 1 Pi** (`Model 1 pi/`) | Can training run on the Raspberry Pi itself? | Yes, with Numba-accelerated slope calculation and all CPU threads |
| **Model 1 PS** (`Model 1 PS/`) | Does the model generalize to Palm Springs climate? | Works, but needs retraining on local data |

**Lesson learned**: `sin`/`cos` encoding of `time_of_day` and `day_of_year` will become important. Scalars like `time_of_day = 23.5` have no notion of "close to midnight"; cyclical encodings fix that.

---

### Step 2 — Temporal Sequences: Conv1D

**Directories**: `workspace/Model 2/`, `workspace/Model 3/`

Instead of a snapshot, what if the model sees a **window of time**? Model 2 feeds a **180-minute sliding window** of 15 features per timestep through a dilated Conv1D residual network.

**Model 2 architecture** (see `workspace/Model 2/train_conv1d_model.md`):
```
Input (180 steps × 15 features)
  Conv1D(32, kernel=3, same)
  4× residual dilated blocks: Conv1D(32, dilation=[1,2,4,8]) + BatchNorm + Add
  GlobalAveragePooling1D()
  Dense(64,relu) → Dropout(0.3) → Dense(32,relu) → BatchNorm
  3 output heads: temp_t+1hr, temp_t+2hr, temp_t+3hr
```

This is the first **multi-output** model — it predicts +1hr, +2hr, and +3hr simultaneously.

**Model 3** (`workspace/Model 3/`) simplified the window to **90 minutes** and the feature set to 8 features to reduce size for Edge TPU. See `workspace/Model 3/train_conv1d_tpu_model.md` for details.

**Results**:
| Model | val_loss | val_mae | Size |
|-------|----------|---------|------|
| Model 2 | 0.0187 | 0.0314 | 65 KB |
| Model 3 | 0.0218 | 0.0332 | 79 KB |

Both are *worse* than Model 1, despite more complexity.

**Lesson learned**: Conv1D doesn't automatically beat hand-crafted lag features. The global average pooling loses positional information the lag features provided explicitly. Explicitly providing `temp_lag1`, `temp_lag30`, etc. as features to a dense model outperforms letting convolutions discover the pattern implicitly — at least at this scale.

---

### Step 3 — Precision Engineering: INT16 Hybrid

**Directory**: `workspace/Model 4/`

Model 4 experiments with representing each input feature as two values — a **least-significant byte (LSB) and most-significant byte (MSB)** — to approximate INT16 precision within an INT8 TFLite model. This doubles the input width from 12 to 24 features and adds an interaction branch.

**Architecture addition** (on top of Model 1's wide-deep):
```
Interaction branch: Dense(16,relu) → element-wise Multiply() → Concatenate() → Dense(32,relu)
Output: Concatenate([wide, deep, interaction]) → Dense(1)
```

**Results**: val_loss = 0.0172 — worse than Model 1.

**Lesson learned**: The LSB/MSB trick adds complexity without benefit. INT8 quantization is precise enough for these weather features after proper scaling. The interaction branch is worth keeping for later models, though.

> Model 4a (`workspace/Model 4a/`) was an incomplete follow-up experiment along the same lines.

---

### Step 4 — Predicting Differences, Not Absolutes

**Directory**: `workspace/Model 5/`

This is the first real architectural insight. Instead of predicting **absolute temperature** at t+1hr, predict the **temperature change** from the current reading (`diff_1hr = temp_t+1hr − temp_now`).

**Why this helps**:
- Temperature changes are much smaller in scale than absolute temperatures
- The target distribution is tighter and easier to fit
- The model no longer needs to "remember" the current temperature — the inference script adds the prediction back

**Key feature changes** from Model 1:
- Replace `temp_lag1` / `humidity_lag1` with multi-horizon lags: `temp_lag{30,60,120}`, `humidity_lag{30,60,120}`, plus wind/UV/pressure lags at 30 min
- Add higher-order cyclical terms: `time_of_day_{sin,cos,sin2,cos2}`, `day_of_year_{sin,cos}` (28 features total)
- Cyclical time encoding with **double harmonics** (`sin2`, `cos2`) for within-day patterns
- Target scaled to `[−1, 1]` with ±2°C padding

**Results**: val_loss = 0.0108, val_mae = 0.022 — comparable to Model 1, not yet better.

**Lesson learned**: The difference target alone isn't enough. The lag features are better (30/60/120 min instead of 1 min), but the delta features (raw differences) are noisy. The key improvement is still ahead.

---

### Step 5 — The Breakthrough: Slope Features Replace Deltas

**Directory**: `workspace/Model 5 new arch. slope calc/`

This model achieves a **~15× reduction in val_loss** over Model 5 and a **~60× reduction** over Model 1. The architecture is nearly identical to Model 5 — the key change is in **feature engineering**.

**The critical change**: Replace raw delta features (`illuminance_delta`, `solar_radiation_delta`, `pressure_delta`, `humidity_delta`) with **Numba-accelerated linear regression slopes** computed over multiple rolling windows.

```python
# Instead of: delta = value[t] - value[t-1]   (noisy)
# Use: slope over a rolling window             (stable trend)
@njit(parallel=True)
def rolling_slope_numba(arr, window):
    ...  # linear regression slope over `window` samples
```

Numba is used here for two reasons:
1. `scipy.stats.linregress` is too slow for the large dataset
2. Training can run on the Raspberry Pi itself, where Numba's parallel JIT uses all CPU cores

**Features (27)**: slopes replace deltas; `temp_lag120` (2 hours ago) is the single most important feature; double harmonic cyclical time encoding.

**Results**:
| Metric | Value |
|--------|-------|
| val_loss | **0.000682** |
| val_mae | **0.00445** |
| Best epoch | 97 |
| Model size | ~788 KB |
| Top feature | `temp_lag120` (importance 0.093) |

**Model architecture diagram**: `workspace/Model 5 new arch. slope calc/weather_model_architecture.png`

**Lesson learned**: **Slopes beat deltas.** A single-point difference is dominated by measurement noise. A slope over 15–30 samples captures the real trend and is far more predictive. The Numba JIT makes this computationally feasible everywhere.

---

### Step 6 — Conv2D Architecture Experiments

**Directory**: `workspace/Model 5b Conv2D/`

With a strong baseline (Model 5a), this step experiments with a **Conv2D architecture** that treats the input as a 2D grid: time steps × features, looking for spatial feature interactions.

**Architecture** (see `workspace/Model 5b Conv2D/MODEL_5B_EXPERIMENT_LOG.md`):
```
Input: 180-minute window × n_features
  ├─ Conv2D branch: temporal-feature convolution blocks
  ├─ Dense branch: wide-deep learned patterns
  └─ Lag extraction branch: explicit 30/60/120-min lag values
Concatenate([conv, dense, lag]) → 3 output heads
```

**New technique — gap-aware windowing**: Training windows that span a data collection gap (e.g. a Wi-Fi outage) corrupt the sequence. Model 5b explicitly detects and drops these windows.

Despite extensive hyperparameter search (see `MODEL_5B_EXPERIMENT_LOG.md` and `ARCHITECTURE_IMPROVEMENT_PROPOSALS.md`), the Conv2D approach could not match Model 5a's accuracy.

**Results**: val_loss = 0.00775, val_mae = 0.016 — ~11× worse than Model 5a.

**Key finding**: Simpler architectures generalize better at this data scale. Model 5a's wide-deep-interaction dense network with well-engineered lag and slope features outperforms more complex convolutional architectures.

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

**Results**: val_loss = 0.0185, val_mae = 0.025, model size ~48 KB

This model is not trying to beat Model 5a on temperature — it's solving a different problem. Top features are `time_of_day_sin` and `solar_radiation_mean_30min`.

---

## Performance Summary

| Model | Predicts | val_loss | val_mae | Size |
|-------|----------|----------|---------|------|
| Model 1 | Absolute temp +1hr | 0.0040 | 0.0130 | 33 KB |
| Model 1a | Absolute temp +1hr | 0.00316 | 0.0418 | 32 KB |
| Model 2 | Absolute temp +1/2/3hr | 0.0187 | 0.0314 | 65 KB |
| Model 3 | Absolute temp +1/2/3hr | 0.0218 | 0.0332 | 79 KB |
| Model 4 | Absolute temp +1hr | 0.0172 | 0.0270 | 21 KB |
| Model 5 | Temp diff +1/2/3hr | 0.0108 | 0.0219 | 39 KB |
| **Model 5a** (slope calc) | **Temp diff +1/2/3hr** | **0.000682** | **0.00445** | **788 KB** |
| Model 5b Conv2D | Temp diff +1/2/3hr | 0.00775 | 0.0159 | 844 KB |
| Model 6 | Solar diff +30/60/90min | 0.0185 | 0.0249 | 48 KB |

> All metrics are normalized (targets scaled to roughly `[−1, 1]`). Model 5a is the current best.

---

## Key Learnings

1. **Predict differences, not absolute values.** Models 5+ predict temperature *change* from current. It's a simpler target and consistently outperforms absolute prediction.

2. **Slope features beat delta features.** The single biggest accuracy jump (Model 5 → Model 5a, ~15×) came from replacing raw one-step deltas with Numba-computed linear regression slopes over rolling windows.

3. **Explicit lag features are essential.** `temp_lag1`, `temp_lag30/60/120` dominate feature importance. Conv layers do not implicitly learn multi-horizon lags as well as providing them directly.

4. **Cyclical time encoding with harmonics matters.** `sin`/`cos` of `time_of_day` and `day_of_year` — and their double harmonics (`sin2`, `cos2`) — are consistently high-importance. Raw scalars are inferior.

5. **Simpler architectures generalize better** at this data scale. The wide-deep-interaction dense model (Model 5a) outperformed dilated Conv1D (Models 2/3) and Conv2D (Model 5b).

6. **INT8 quantization is viable** with minimal accuracy loss when features are properly scaled.

7. **Gap-aware windowing** prevents corrupted training sequences. Windows spanning data collection gaps must be explicitly detected and dropped.

8. **Pre-normalize before the inference loop.** In `Inference_InfluxDB_Writer.py`, normalize the entire feature matrix once (vectorized NumPy) before the loop. Per-window normalization inside the loop saturates CPU at ~3,240 Python ops per prediction — even though the TPU finishes inference in ~0.55 ms.

---

## Running a Model

### Build the Docker environment
```bash
docker build -f Dockerfile.tpu -t tpu-dev .
./run_dev.sh   # or: docker run -it --rm -v $(pwd):/workspace tpu-dev bash
```

### Train
```bash
# Inside Docker or directly on macOS/Pi:
cd workspace/Model\ 5\ new\ arch.\ slope\ calc/
python train_model.py
```

### Compile for Edge TPU
```bash
# Uses the x86 compiler container:
./edgetpu-x86-compiler.sh weather_model_5a_best.tflite
```

### Deploy to Raspberry Pi
```bash
scp workspace/Model\ 5\ new\ arch.\ slope\ calc/weather_model_5a_best_edgetpu.tflite pi@raspberrypi:~/
scp workspace/Model\ 5\ new\ arch.\ slope\ calc/input_scaler_5a.json pi@raspberrypi:~/
scp workspace/Model\ 5\ new\ arch.\ slope\ calc/target_scaler_5a.json pi@raspberrypi:~/
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

## Project Structure

```
.
├── Dockerfile.tpu                     # Docker image for training
├── edgetpu-x86-compiler.sh            # Compiles .tflite → _edgetpu.tflite
├── dual-edge-tpu-fix.md               # Notes on dual Edge TPU setup/fix
├── run_dev.sh                         # Launches Docker dev container
├── SPEC.md                            # Detailed model catalog and spec
└── workspace/
    ├── export_influx_to_csv*.py       # Export weather data from InfluxDB
    ├── create_combined_data.py        # Merge multi-location datasets
    ├── extract_scaler_params.py       # Dump scaler parameters to JSON
    ├── check.py                       # Data validation utilities
    ├── Model 1/                       # Dense wide-deep baseline
    ├── Model 1a/                      # Extended training variant
    ├── Model 1 Daytime/               # Daytime-only filter
    ├── Model 1 INT/                   # INT quantization variant
    ├── Model 1 Periodic/              # Cyclic time encoding
    ├── Model 1 diffs/                 # Predict temp change
    ├── Model 1 combined/              # Diff + cyclic features
    ├── Model 1 Pi/                    # Pi-optimized training
    ├── Model 1 PS/                    # Palm Springs dataset
    ├── Model 2/                       # Conv1D dilated residual
    ├── Model 3/                       # Conv1D TPU-optimized
    ├── Model 4/                       # INT16 hybrid precision
    ├── Model 4a/                      # Hybrid precision experimental
    ├── Model 5/                       # Temp diff prediction
    ├── Model 5 new arch. slope calc/  # BEST MODEL — slope features
    ├── Model 5b Conv2D/               # Conv2D architecture experiment
    └── Model 6/                       # Solar radiation prediction
```

---

## License

MIT
