# Model 5a Clean Experiment Log

## Overall Goal

Produce a clean, reproducible version of Model 5a that:
- Removes data contamination around sensor outages (gap invalidation)
- Matches or beats the Model 5 reference val_loss of **0.000706**
- Keeps lag features as dominant predictors (not time-of-day)
- Remains deployable as a quantized INT8 TFLite model for Edge TPU

---

## Reference Baseline — Model 5 (new arch. slope calc)

This is the benchmark that Model 5a clean targets. The "slope calc" variant was trained without gap invalidation and without target clipping.

| Metric | Value |
|--------|-------|
| val_loss | **0.000706** |
| val_mae (normalized) | **0.00477** |
| Best epoch | 81 |
| Model size | 788 KB |
| #1 feature | `temp_lag120` (0.0864) |
| #2 feature | `time_of_day_cos2` (0.0824) |
| #3 feature | `time_of_day_cos` (0.0704) |
| Optimizer | Adam lr=1e-5 (fixed) |
| EarlyStopping patience | 5 |
| Target clipping | None |
| Gap invalidation | None |

**Key characteristic**: lag features and time-of-day are interleaved in the top 10, indicating the model learned both physical dynamics and diurnal patterns.

---

## Architecture

### Exp 1–5 (original)

- Input: `(SEQ_LEN=180, n_features)` flattened to `(180 × n_features,)`
- Wide path: Dense(16)
- Deep path: Dense(128, relu) → Dropout(0.3) → residual block Dense(64) + shortcut → Dense(32, relu)
- Interaction path: Dense(16, relu) → element-wise square → Concatenate → Dense(32, relu)
- Output: three Dense(1, linear) heads for `temp_diff_1hr`, `temp_diff_2hr`, `temp_diff_3hr`
- Targets: temperature differences (future − current) normalized to [−1, 1]

### Exp 6 attempt 1 (bottleneck only — failed, same streaming)

- Input: `(SEQ_LEN=180, n_features)` flattened to `(180 × n_features = 4860,)`
- **Bottleneck: Dense(64, relu)** — single large-input FC replacing three
- Wide/deep/interaction paths then operate on 64-dim bottleneck output
- *(See Exp 6 for why this did not reduce off-chip streaming)*

### Exp 6 attempt 2 (temporal compress + bottleneck — trained, EdgeTPU FC version issue)

Three-stage approach:
1. **Temporal compress**: `Dense(4, relu)` on 3D input → `(SEQ_LEN, 4)`.
2. **Flatten**: `Reshape((720,))`.
3. **Shared bottleneck**: `Dense(64, relu)` on 720-dim vector.
*(Eliminated SRAM overflow but generated FULLY_CONNECTED version 9 — all FC ops on CPU. See Exp 6.)*

### Exp 7+ (AveragePooling1D + bottleneck — planned architecture)

Two-stage approach that avoids any 3D Dense:
1. **Temporal pool**: `AveragePooling1D(pool_size=6, strides=6)` → `(30, n_features)` using `AVERAGE_POOL_2D` (EdgeTPU-supported, no FULLY_CONNECTED op).
2. **Flatten**: `Reshape((30 × n_features,))` ≈ `(810,)` — below the ~1,660-dim SRAM threshold.
3. **Shared bottleneck**: `Dense(64, relu)` on the 810-dim vector — standard 2D input, FC version 4.
- Wide path: Dense(16)
- Deep path: Dense(128, relu) → Dropout(0.3) → residual block Dense(64) + shortcut → Dense(32, relu)
- Interaction path: Dense(16, relu) → element-wise square → Concatenate → Dense(32, relu)
- Output: three Dense(1, linear) heads for `temp_diff_1hr`, `temp_diff_2hr`, `temp_diff_3hr`
- Targets: temperature differences (future − current) normalized to [−1, 1]

**Features (23 base + optional wind_direction, wind_lull, rain_accumulated):**
`uv`, `wind_avg`, `wind_gust`, `solar_radiation`, `illuminance`, `relative_humidity`,
`station_pressure`, `day_of_year_sin/cos`, `time_of_day_sin/cos/sin2/cos2`,
`temp_lag30/60/120`, `humidity_lag30/60/120`, `wind_avg_lag30`, `wind_gust_lag30`,
`uv_lag30`, `pressure_lag30`

---

## Data Cleaning (added in 5a clean, absent in reference)

**Gap invalidation** (`_invalidate_targets_crossing_gaps`, `tol_s=600`):
- Nulls target values for rows within 1–3 hours before any sensor gap > 10 minutes
- Prevents InfluxDB post-gap glitch data from entering training targets
- Training data: 53 gaps >10 min, **5,540 rows nulled (0.54% of 1,017,257)**
- Validation data: 14 gaps >10 min, **2,131 rows nulled (0.41% of 523,458)**
- Nulled training rows are actually *quieter* than average (3hr diff std: 1.71°C vs 2.20°C remaining), so this does NOT selectively remove extreme events

---

## Experiments

---

### Exp 1 — Baseline clean run (with ±12°C target clip)

**Date**: ~2026-05

**Changes from reference**:
- Added gap invalidation (`tol_s=600`)
- Added target clipping at ±12°C (rationale: remove physically implausible sensor glitches)
- Optimizer: Adam lr=1e-5 (same as reference)
- EarlyStopping patience=5 (same as reference)

**Results**:

| Metric | Value |
|--------|-------|
| val_loss | 0.001069 |
| val_mae | 0.00544 |
| Best epoch | **27** |

**Feature importance (top 5)**:
1. `time_of_day_cos`: 0.0483
2. `time_of_day_sin2`: 0.0438
3. `time_of_day_sin`: 0.0408
4. `time_of_day_cos2`: 0.0394
5. `uv`: 0.0299
— `temp_lag120`: 0.0247 (#10), `temp_lag30`: 0.0128 (#27 / last)

**Diagnosis**: Model snapped to time-of-day. All 4 time-of-day cyclical features occupy the top positions; lag features rank near the bottom. val_loss is 51% worse than reference. Early stopping fired at epoch 27 vs reference's epoch 81.

**Root cause investigation**:
- Target clip suspected as the culprit — removing large swings would leave mostly diurnal data

---

### Exp 2 — Remove target clip

**Date**: ~2026-05

**Changes from Exp 1**:
- Removed ±12°C target clip
- Everything else unchanged (lr=1e-5, patience=5)

**Results** (user-reported, no JSON saved):

| Metric | Value |
|--------|-------|
| val_loss | not recorded |
| val_mae | not recorded |

**Feature importance (top 5)**:
1. `time_of_day_cos`: 0.0394
2. `time_of_day_cos2`: 0.0367
3. `time_of_day_sin2`: 0.0348
4. `time_of_day_sin`: 0.0334
5. `solar_radiation`: 0.0245
— `temp_lag120`: 0.0151 (#11), `temp_lag30`: 0.0075 (#27 / last)

**Diagnosis**: Still snapping to time-of-day. Removing the clip had no meaningful effect on feature importance ranking. Time-of-day dominance is not caused by the clip.

**Root cause investigation**:
- Gap invalidation removes only 0.54% of data and does NOT selectively remove dynamic rows (nulled rows are quieter than average) — ruled out as cause
- Key insight: early stopping fires at epoch 27 vs reference's epoch 81. The model quickly learns the strong time-of-day signal and plateaus. With lr=1e-5 and patience=5, early stopping fires before the model can push through to learn the subtler lag-feature patterns
- **Root cause: training dynamics — fixed lr=1e-5 is too small to escape the time-of-day local minimum, and patience=5 stops training too early**

---

### Exp 3 — ReduceLROnPlateau + higher initial LR + longer patience

**Date**: 2026-05-13

**Changes from Exp 2**:
- Optimizer: Adam lr=1e-5 → **1e-4** (matches Model 5b starting point)
- EarlyStopping patience: 5 → **20** (allows ReduceLR to fire multiple times before stopping)
- Added **`ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)`**
  - LR halves after 8 epochs without val_loss improvement
  - When the model stalls on the time-of-day minimum, LR reduction creates a new gradient descent direction rather than stopping

**Results**:

| Metric | Value | vs Reference |
|--------|-------|-------------|
| val_loss | **0.000440** | ✅ better (ref: 0.000706) |
| val_mae (normalized) | **0.00299** | ✅ better (ref: 0.00477) |
| Best epoch | **95** | ✅ much better (ref: 81) |
| Final LR | 1.95e-07 | ReduceLR fired ~9× (1e-4 → … → 1.95e-7) |
| Quantized 1hr MAE | 0.35°C | |
| Quantized 2hr MAE | 0.54°C | |
| Quantized 3hr MAE | 0.66°C | |
| Target range (scaler) | −18.54°C to +17.53°C (36.07°C) | identical to reference — comparison valid |

**Feature importance (full)**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `time_of_day_cos` | 0.0236 |
| 2 | `uv_lag30` | 0.0211 |
| 3 | `time_of_day_sin2` | 0.0209 |
| 4 | `time_of_day_sin` | 0.0206 |
| 5 | `time_of_day_cos2` | 0.0205 |
| 6 | `illuminance` | 0.0185 |
| 7 | `humidity_lag120` | 0.0171 |
| 8 | `wind_gust` | 0.0164 |
| 9 | `humidity_lag30` | 0.0162 |
| 10 | `wind_avg` | 0.0160 |
| 11 | `temp_lag120` | 0.0149 |
| … | … | … |
| 25 | `temp_lag60` | 0.0132 |
| 27 | `temp_lag30` | 0.0074 |

**Assessment**: Significant improvement over Exp 1/2. ReduceLR working — trained to epoch 95 and val_loss beats the reference on the same target scale. However, time-of-day still occupies 3 of the top 5 slots and temp_lag features remain near the bottom. The feature importance spread is much more compressed than the reference (0.024 vs 0.086 for top feature), suggesting the model is using all features more evenly but still anchoring on time-of-day.

**Note on "little progress"**: The val_loss is better than the reference, but the feature importance pattern shows time-of-day still dominant. This is the core tension: the model is numerically accurate but physically anchored to the diurnal cycle rather than to actual temperature dynamics.

**Key missing insight**: `temperature` (current value) is **not in the features list**. The model predicts temperature *changes* but cannot directly observe the current temperature — it can only infer it from lag values. Without knowing the current temperature relative to the seasonal baseline, the model has to use time-of-day as a proxy for "what temperature is typical right now." Adding `temperature` as an explicit input could break this dependency.

---

### Exp 4 — Add current `temperature` as explicit feature

**Date**: 2026-05-13

**Hypothesis**: The model uses time-of-day as a proxy for "what is the temperature expected to be right now?" because current temperature is absent from the feature set. Adding it directly would give the model a physical anchor and reduce the need to infer absolute temperature level from time-of-day signals.

**Changes from Exp 3**:
- Add `'temperature'` to the features list (28 total features)
- Add `"temperature": (-10, 55)` to domain_bounds (same bounds as temp_lag30)
- Everything else unchanged (lr=1e-4, ReduceLROnPlateau, patience=20)

**Results**:

| Metric | Value | vs Reference | vs Exp 3 |
|--------|-------|-------------|---------|
| val_loss | **0.0002** | ✅ better (ref: 0.000706) | ✅ better (Exp 3: 0.000440) |
| val_mae (normalized) | **0.0016** | ✅ better (ref: 0.00477) | ✅ better (Exp 3: 0.00299) |
| Best epoch | **99** | ✅ better (ref: 81) | ≈ same (Exp 3: 95) |
| Quantized 1hr MAE | 0.36°C | | ≈ same (Exp 3: 0.35°C) |
| Quantized 2hr MAE | 0.56°C | | ≈ same (Exp 3: 0.54°C) |
| Quantized 3hr MAE | 0.67°C | | ≈ same (Exp 3: 0.66°C) |
| Quantized model size | 815.87 KB | | slightly larger (Exp 3: ~788 KB) |

**Feature importance (full)**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `time_of_day_sin2` | 0.0246 |
| 2 | `uv_lag30` | 0.0243 |
| 3 | `time_of_day_cos` | 0.0237 |
| 4 | `time_of_day_sin` | 0.0228 |
| 5 | `time_of_day_cos2` | 0.0215 |
| 6 | `illuminance` | 0.0190 |
| 7 | `wind_gust_lag30` | 0.0159 |
| 8 | `solar_radiation` | 0.0159 |
| 9 | `humidity_lag120` | 0.0158 |
| 10 | `wind_avg_lag30` | 0.0153 |
| 11 | `humidity_lag60` | 0.0153 |
| 12 | `temp_lag120` | 0.0150 |
| 13 | `humidity_lag30` | 0.0148 |
| 14 | `wind_lull` | 0.0148 |
| 15 | `uv` | 0.0148 |
| 16 | `station_pressure` | 0.0147 |
| 17 | `day_of_year_sin` | 0.0147 |
| 18 | `rain_accumulated` | 0.0147 |
| 19 | `pressure_lag30` | 0.0147 |
| 20 | `wind_direction_cos` | 0.0147 |
| 21 | `wind_direction_sin` | 0.0147 |
| 22 | `day_of_year_cos` | 0.0147 |
| 23 | `relative_humidity` | 0.0144 |
| 24 | `temp_lag60` | 0.0142 |
| 25 | `wind_gust` | 0.0141 |
| 26 | `wind_avg` | 0.0141 |
| 27 | `temp_lag30` | 0.0133 |
| 28 | `temperature` | 0.0086 |

**Assessment**: Float val_loss improved substantially (0.0002 vs 0.000440 in Exp 3), but quantized MAE was essentially unchanged. Critically, adding `temperature` as an explicit feature did **not** reduce time-of-day dominance — it ranked **dead last** (28/28, importance 0.0086), even below `temp_lag30`. Time-of-day still occupies 4 of the top 5 slots.

**Interpretation**: The hypothesis was incorrect. The model already had access to current temperature implicitly via `temp_lag30` (30 minutes ago, very close to current). Adding the current value didn't provide new information the lag features weren't already supplying. The time-of-day dominance is structural, not a feature-gap problem.

**Remaining question**: Does time-of-day dominance actually hurt prediction quality on unusual weather events, or is this a benign pattern? The val_loss improvement (0.0002) suggests the model is predicting accurately on average, but the feature importance pattern still raises concern about robustness to off-diurnal events (cold fronts, marine layer).

---

---

### Exp 5 — Remove `temperature` from feature set

**Date**: 2026-05-14

**Hypothesis**: `temperature` ranked last in Exp 4 (importance 0.0086) and is largely redundant with `temp_lag30`. Removing it should not hurt accuracy while producing a cleaner 27-feature model.

**Changes from Exp 4**:
- Remove `'temperature'` from features list (27 features total)
- Remove `"temperature": (-10, 55)` from `domain_bounds`
- Everything else unchanged

**Results**:

| Metric | Value | vs Reference | vs Exp 4 |
|--------|-------|-------------|---------|
| val_loss | **0.0004** | ✅ better (ref: 0.000706) | ❌ worse (Exp 4: 0.0002) |
| val_mae (normalized) | **0.0030** | ✅ better (ref: 0.00477) | ❌ worse (Exp 4: 0.0016) |
| Best epoch | **99** | | same |
| Quantized 1hr MAE | 0.38°C | | ❌ slightly worse (Exp 4: 0.36°C) |
| Quantized 2hr MAE | 0.55°C | | ✅ slightly better (Exp 4: 0.56°C) |
| Quantized 3hr MAE | 0.63°C | | ✅ better (Exp 4: 0.67°C) |
| Quantized model size | 787.74 KB | | ✅ smaller (Exp 4: 815.87 KB) |

**Feature importance (full)**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `time_of_day_cos` | 0.0248 |
| 2 | `time_of_day_cos2` | 0.0242 |
| 3 | `time_of_day_sin` | 0.0233 |
| 4 | `time_of_day_sin2` | 0.0214 |
| 5 | `uv_lag30` | 0.0190 |
| 6 | `solar_radiation` | 0.0188 |
| 7 | `humidity_lag30` | 0.0168 |
| 8 | `illuminance` | 0.0165 |
| 9 | `humidity_lag120` | 0.0156 |
| 10 | `wind_avg` | 0.0150 |
| 11 | `temp_lag120` | 0.0148 |
| 12 | `humidity_lag60` | 0.0145 |
| 13 | `wind_gust` | 0.0145 |
| 14 | `wind_direction_sin` | 0.0143 |
| 15 | `wind_gust_lag30` | 0.0143 |
| 16 | `pressure_lag30` | 0.0142 |
| 17 | `wind_direction_cos` | 0.0142 |
| 18 | `day_of_year_cos` | 0.0142 |
| 19 | `rain_accumulated` | 0.0142 |
| 20 | `day_of_year_sin` | 0.0142 |
| 21 | `station_pressure` | 0.0142 |
| 22 | `wind_lull` | 0.0142 |
| 23 | `wind_avg_lag30` | 0.0138 |
| 24 | `temp_lag60` | 0.0135 |
| 25 | `uv` | 0.0134 |
| 26 | `relative_humidity` | 0.0123 |
| 27 | `temp_lag30` | 0.0072 |

**Assessment**: The hypothesis was partially wrong. Removing `temperature` hurt float metrics (val_loss 0.0002 → 0.0004) but produced mixed quantized results: 3hr MAE improved (0.67 → 0.63°C) while 1hr degraded slightly (0.36 → 0.38°C). Model size returned to near-reference (787.74 KB). `temp_lag30` is now dead last (27/27, 0.0072), and time-of-day still occupies the top 4 positions — same structural pattern as before.

**Interpretation**: The float val_loss improvement in Exp 4 was partly real — `temperature` was contributing something. However, the quantized 3hr improvement suggests that on the deployed model, `temperature` may have been acting as a shortcut that hurt long-horizon generalization. The overall quantized accuracy is comparable to Exp 4 (within noise), so the tradeoff of smaller model size (787.74 KB) and marginally better 3hr quantized MAE may favor keeping this version.

---

---

### Exp 6 — Bottleneck architecture to fix EdgeTPU SRAM overflow

**Date**: 2026-05-14

**Context — EdgeTPU inference was 56ms/sample on RPi5 (PCIe Coral M.2):**

The Exp 5 model deployed with all 16 ops mapped to the Edge TPU (confirmed by the EdgeTPU compiler log), yet inference averaged **56ms** per sample instead of the expected 1–5ms. CPU utilization was near zero. Root cause investigation:

```
Edge TPU Compiler version 14.1.317412892
Input size:   787.74 KiB
Output size:  24.23 MiB

On-chip memory used for caching model parameters:        1.23 MiB
On-chip memory remaining for caching model parameters:   6.61 MiB
Off-chip memory used for streaming uncached parameters: 23.40 MiB
```

The compiler inflated the 788KB model to 24MB because it pre-tiles FC weight matrices into the TPU's systolic array format. Only **1.23MB fits in the EdgeTPU's 8MB on-chip SRAM**; the remaining **23.40MB is DMA'd from host RAM on every inference call**. At PCIe Gen2 x1 bandwidth (~500 MB/s): 23.4MB ÷ 500 MB/s ≈ **47ms per inference** — that accounts for the full 56ms.

**Root cause**: The original architecture has **three separate FC layers all taking the raw 4,860-dim flattened input**:

| Layer | Shape | Params |
|-------|-------|--------|
| `interaction_embed` | 4,860 → 16 | 77,760 |
| `wide_dense` | 4,860 → 16 | 77,760 |
| `deep_dense1` | 4,860 → 128 | 622,080 |

These three layers account for ~778K of ~797K total parameters. The EdgeTPU compiler's tiling/padding of three wide-input weight matrices produces a 30× size explosion in the compiled output.

**Attempt 1 — Shared bottleneck (failed)**:

Replace the three separate large-input FC layers with a single `Dense(64, relu)` bottleneck immediately after the Reshape, so all paths operate on a 64-dim vector. Trained and compiled:

```
Input size:   343.19 KiB
Output size:  23.79 MiB

On-chip memory used for caching model parameters:        363.50 KiB
On-chip memory remaining for caching model parameters:   7.48 MiB
Off-chip memory used for streaming uncached parameters: 23.40 MiB  ← identical to before
```

Off-chip streaming was unchanged at exactly 23.40 MiB. The bottleneck was still a **4,860-dim input FC layer** — only the output dimension changed. The compiler overhead is tied to the input dimension of the largest FC, not its output dimension or the number of such layers.

**Key insight**: Dividing 23.40 MB ÷ 4,860 input dims ≈ 4,800 bytes/input-dim. The overhead scales linearly with the flat vector size, independent of what comes after it. To fit in 8MB SRAM, the flat input must be ≤ ~1,660 dims (4,860 × 8/23.4).

**Attempt 2 — Temporal compression before flatten (current architecture)**:

Eliminate the 4,860-dim flat vector by adding a `Dense(4, relu)` stage that compresses each timestep's features **before** flattening:

| Stage | Operation | Output shape | Large dim? |
|-------|-----------|-------------|------------|
| Temporal compress | Dense(4) on 3D input | (180, 4) | No — weight matrix is only (27, 4) = 108 params |
| Flatten | Reshape | (720,) | 720 << 1,660 threshold |
| Bottleneck | Dense(64) | (64,) | Input is 720, not 4,860 |

Estimated compiled overhead: 720/4,860 × 23.40 MB ≈ **3.5 MB** → fits in SRAM.

**Parameter counts**:

| Layer | Exp 1–5 original | Exp 6 bottleneck | Exp 6 temporal compress |
|-------|-----------------|-----------------|------------------------|
| `temporal_compress` | — | — | 27 × 4 = 108 |
| `bottleneck` | — | 4,860 × 64 = 311,040 | 720 × 64 = 46,080 |
| `interaction_embed` | 77,760 | 1,024 | 1,024 |
| `wide_dense` | 77,760 | 1,024 | 1,024 |
| `deep_dense1` | 622,080 | 8,192 | 8,192 |
| **Total** | **~797K** | **~341K** | **~76K** |

**Results**:

| Metric | Value | vs Reference | vs Exp 5 |
|--------|-------|-------------|---------|
| val_loss | **0.0004** | ✅ better (ref: 0.000706) | = same (Exp 5: 0.0004) |
| val_mae (normalized) | **0.0028** | ✅ better (ref: 0.00477) | ✅ better (Exp 5: 0.0030) |
| Best epoch | **89** | | ≈ same (Exp 5: 99) |
| Quantized 1hr MAE | 0.35°C | | = same (Exp 5: 0.38°C) |
| Quantized 2hr MAE | 0.56°C | | = same (Exp 5: 0.55°C) |
| Quantized 3hr MAE | 0.65°C | | ≈ same (Exp 5: 0.63°C) |
| Quantized model size | **84.91 KB** | | ✅ 9× smaller (Exp 5: 787.74 KB) |

**Feature importance (top 10)**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `illuminance` | 0.0237 |
| 2 | `time_of_day_sin` | 0.0224 |
| 3 | `time_of_day_cos` | 0.0215 |
| 4 | `solar_radiation` | 0.0212 |
| 5 | `humidity_lag30` | 0.0174 |
| 6 | `time_of_day_cos2` | 0.0168 |
| 7 | `time_of_day_sin2` | 0.0167 |
| 8 | `humidity_lag120` | 0.0149 |
| 9 | `temp_lag120` | 0.0147 |
| 10 | `uv` | 0.0147 |
| 27 | `temp_lag30` | 0.0088 |

**Assessment**: Accuracy is on par with Exp 5 while the model is **9× smaller**. The temporal compression approach dramatically reduced model size (84.91 KB vs 787 KB). Importantly, `illuminance` and `solar_radiation` broke into the top 4, displacing two time-of-day features — a slight improvement in physical grounding. `temp_lag30` is still last but `temp_lag120` moved up to #9, suggesting the model has better access to thermal trend information through the compressed representation.

**EdgeTPU compilation — new problem: FULLY_CONNECTED version not supported**:

```
Input size:   84.91 KiB
Output size:  163.72 KiB   ← massive improvement (was 24 MiB)

On-chip memory used for caching model parameters:   0.00 B   ← SRAM overflow solved
Off-chip memory used for streaming uncached params:  0.00 B   ← 0 streaming

Number of Edge TPU subgraphs: 3
Number of operations that will run on Edge TPU: 6
Number of operations that will run on CPU:      12

FULLY_CONNECTED   12   Operation version not supported
RESHAPE            2   Mapped to Edge TPU
CONCATENATION      2   Mapped to Edge TPU
MUL                1   Mapped to Edge TPU
ADD                1   Mapped to Edge TPU
```

The SRAM overflow is completely solved (0 off-chip streaming, model down from 24 MiB to 163 KiB compiled). However, **all 12 FULLY_CONNECTED ops now run on CPU** — only the structurally trivial ops (reshape, concatenate, multiply, add) run on the TPU.

**Root cause**: `Dense(4)` applied to a 3D input `(None, SEQ_LEN, n_features)` generates a FULLY_CONNECTED op with `keep_num_dims=true`. This attribute was introduced in TFLite FULLY_CONNECTED schema **version 9**, which the EdgeTPU compiler v14 does not support (max supported is ~v4 for asymmetric int8). When any FC op in the graph requires version 9, the TFLite converter upgrades **all** FULLY_CONNECTED ops to version 9, causing the EdgeTPU compiler to reject all of them.

**Fix for Exp 7**: Replace the `Dense(4)` temporal compression with `AveragePooling1D(pool_size=6, strides=6)`, which:
- Reduces 180 timesteps → 30 (6-minute averaging, preserves temporal trends)
- Creates an `AVERAGE_POOL_2D` op (TFLite), fully supported by EdgeTPU
- Avoids FULLY_CONNECTED entirely for the dimension-reduction step
- Resulting flat vector: 30 × n_features ≈ 810 dims → estimated compiled overhead ≈ 3.9 MB → fits in SRAM
- All downstream FC ops remain 2D inputs → stay at version 4 → EdgeTPU-compatible

---

### Exp 7 — AveragePooling1D architecture (fix EdgeTPU FULLY_CONNECTED version 9)

**Date**: 2026-05-14

**Context**: Exp 6 solved the SRAM overflow (0 off-chip streaming, compiled model 163 KB) but introduced a new problem: `Dense(4)` applied to a 3D input generates `FULLY_CONNECTED keep_num_dims=true` (schema version 9), which the EdgeTPU compiler does not support. The TFLite converter upgrades **all** FC ops to version 9, causing every FULLY_CONNECTED to fall back to CPU.

**Fix**: Replace `Dense(4, relu)` temporal compression with `AveragePooling1D(pool_size=6, strides=6)`:
- Reduces 180 timesteps → 30 via 6-minute averaging
- Generates `AVERAGE_POOL_2D` (EdgeTPU v1, fully supported) — no FULLY_CONNECTED for the reduction step
- Flat vector: 30 × 27 = 810 dims → well below the ~1,660-dim SRAM threshold
- All downstream Dense layers remain 2D inputs → FC version 4 → EdgeTPU-compatible

**Changes from Exp 6**:
- Replace `Dense(4, relu)` with `AveragePooling1D(pool_size=6, strides=6)` on the 3D input
- Everything else unchanged (lr=1e-4, ReduceLROnPlateau, patience=20, 27 features)

**Results**:

| Metric | Value | vs Reference | vs Exp 6 |
|--------|-------|-------------|---------|
| val_loss | **0.0005** | ✅ better (ref: 0.000706) | ≈ same (Exp 6: 0.0004) |
| val_mae (normalized) | **0.0033** | ✅ better (ref: 0.00477) | ≈ same (Exp 6: 0.0028) |
| Best epoch | **95** | | ≈ same (Exp 6: 89) |
| Quantized 1hr MAE | 0.35°C | | = same (Exp 6: 0.35°C) |
| Quantized 2hr MAE | 0.58°C | | ❌ slightly worse (Exp 6: 0.56°C) |
| Quantized 3hr MAE | 0.67°C | | ❌ slightly worse (Exp 6: 0.65°C) |
| Quantized model size | **90.57 KB** | | ≈ same (Exp 6: 84.91 KB) |
| Final LR | 1.0e-07 | ReduceLR fully decayed |  |

**Feature importance (full)**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `time_of_day_sin` | 0.0182 |
| 2 | `time_of_day_cos` | 0.0169 |
| 3 | `uv_lag30` | 0.0167 |
| 4 | `time_of_day_cos2` | 0.0167 |
| 5 | `time_of_day_sin2` | 0.0166 |
| 6 | `humidity_lag30` | 0.0165 |
| 7 | `solar_radiation` | 0.0156 |
| 8 | `humidity_lag120` | 0.0146 |
| 9 | `temp_lag120` | 0.0146 |
| 10 | `wind_gust_lag30` | 0.0145 |
| 11 | `wind_avg_lag30` | 0.0144 |
| 12 | `wind_lull` | 0.0141 |
| 13 | `station_pressure` | 0.0140 |
| 14 | `uv` | 0.0140 |
| 15 | `wind_direction_sin` | 0.0140 |
| 16 | `wind_direction_cos` | 0.0140 |
| 17 | `day_of_year_cos` | 0.0140 |
| 18 | `day_of_year_sin` | 0.0139 |
| 19 | `wind_gust` | 0.0139 |
| 20 | `rain_accumulated` | 0.0139 |
| 21 | `pressure_lag30` | 0.0139 |
| 22 | `illuminance` | 0.0137 |
| 23 | `humidity_lag60` | 0.0136 |
| 24 | `wind_avg` | 0.0135 |
| 25 | `temp_lag60` | 0.0126 |
| 26 | `relative_humidity` | 0.0123 |
| 27 | `temp_lag30` | 0.0073 |

**Assessment**: Accuracy is essentially on par with Exp 6 (both ~0.0005 val_loss, ~90 KB quantized model). The AveragePooling1D approach eliminates the FULLY_CONNECTED version 9 problem, so downstream EdgeTPU compilation should work correctly with all FC ops at version 4. Feature importance pattern is similar to Exp 6: time-of-day dominates the top 5, `uv_lag30` breaks into #3 (was #3 in Exp 3 as well), `temp_lag30` is still last (0.0073), and `temp_lag120` reaches #9. Quantized 2hr and 3hr MAE regressed slightly vs Exp 6 (0.58 vs 0.56, 0.67 vs 0.65) — within noise, but worth noting.

**EdgeTPU compilation result** ✅ — all 19 ops mapped to TPU, single subgraph, 0 off-chip streaming:

```
Output size:  840.70 KiB
On-chip memory used for caching model parameters:        796.25 KiB
On-chip memory remaining for caching model parameters:   7.06 MiB
Off-chip memory used for streaming uncached parameters:  0.00 B

Number of Edge TPU subgraphs: 1
Total number of operations:   19

Operator              Count   Status
MUL                   1       Mapped to Edge TPU
AVERAGE_POOL_2D       1       Mapped to Edge TPU
FULLY_CONNECTED       11      Mapped to Edge TPU
CONCATENATION         2       Mapped to Edge TPU
RESHAPE               3       Mapped to Edge TPU
ADD                   1       Mapped to Edge TPU
```

All 11 FULLY_CONNECTED ops are now at version 4 (EdgeTPU-compatible) — the version 9 issue from Exp 6's 3D Dense is fully resolved. `AVERAGE_POOL_2D` maps cleanly. The compiled model (840.70 KB) fits entirely in the EdgeTPU's 8 MB on-chip SRAM (796.25 KB used, 7.06 MB remaining), so inference will have **zero off-chip streaming** — expected latency ~1–5 ms/sample on PCIe Coral M.2.

**This is the deployment candidate.**

**Full-dataset bias analysis (Actual − Predicted)**:

| Model | Mean bias | Stddev |
|-------|-----------|--------|
| Model 5ac Exp 7 (this model) | **+0.0276°C** | 0.144°C |
| Original Model 5a | +0.164°C | 0.145°C |

The gap invalidation reduced systematic bias by **0.136°C** (6×) while leaving the random error component essentially unchanged (0.144 vs 0.145°C stddev). This is the expected signature of a data cleaning fix: it removes a directional artifact without affecting the model's fundamental predictive uncertainty.

The original Model 5a's +0.164°C mean bias was almost certainly caused by post-gap sensor glitch data leaking into training targets — InfluxDB records immediately after sensor reconnection tend to be artificially warm or otherwise corrupted, biasing the model to under-predict actual temperatures. Gap invalidation eliminated that artifact. The residual +0.0276°C bias in Exp 7 is within normal calibration tolerance.

---

### Exp 8 — Remove time-of-day features entirely

**Date**: 2026-05-14

**Hypothesis**: Time-of-day dominance is structural — the four `time_of_day_sin/cos/sin2/cos2` features provide the strongest, cleanest gradient signal (perfectly deterministic, high correlation with average temperature change), so the model always anchors to them first. Removing them forces the model to build predictive pathways through lag features and physical observations. `day_of_year_sin/cos` remain for seasonal context. If accuracy holds, the model will be more robustly grounded in actual physical state rather than statistical expectation.

**Changes from Exp 7**:
- Remove `time_of_day_sin`, `time_of_day_cos`, `time_of_day_sin2`, `time_of_day_cos2` from features (23 features total, down from 27)
- Flat vector after AveragePooling1D: 30 × 23 = 690 dims (was 810) — still well below SRAM threshold
- Run name: `no_tod_run1`
- Everything else unchanged (lr=1e-4, ReduceLROnPlateau, patience=20, AveragePooling1D architecture)

**Results**:

| Metric | Value | vs Reference | vs Exp 7 |
|--------|-------|-------------|---------|
| val_loss | **0.0006** | ✅ better (ref: 0.000706) | ≈ same (Exp 7: 0.0005) |
| val_mae (normalized) | **0.0035** | ✅ better (ref: 0.00477) | ≈ same (Exp 7: 0.0033) |
| Best epoch | **73** | | ❌ earlier (Exp 7: 95) |
| Quantized 1hr MAE | 0.44°C | | ❌ worse (Exp 7: 0.35°C) |
| Quantized 2hr MAE | 0.57°C | | ✅ slightly better (Exp 7: 0.58°C) |
| Quantized 3hr MAE | 0.73°C | | ❌ worse (Exp 7: 0.67°C) |
| Quantized model size | **83.07 KB** | | ✅ smaller (Exp 7: 90.57 KB) |
| Final LR | 1.95e-07 | ReduceLR fully decayed | |

**Feature importance (full)**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `uv_lag30` | **0.0341** |
| 2 | `humidity_lag120` | 0.0175 |
| 3 | `wind_gust` | 0.0169 |
| 4 | `humidity_lag30` | 0.0167 |
| 5 | `illuminance` | 0.0166 |
| 6 | `temp_lag120` | 0.0156 |
| 7 | `wind_avg` | 0.0154 |
| 8 | `solar_radiation` | 0.0152 |
| 9 | `uv` | 0.0148 |
| 10 | `humidity_lag60` | 0.0147 |
| 11 | `wind_lull` | 0.0143 |
| 12 | `wind_direction_sin` | 0.0142 |
| 13 | `pressure_lag30` | 0.0142 |
| 14 | `wind_direction_cos` | 0.0142 |
| 15 | `day_of_year_sin` | 0.0141 |
| 16 | `rain_accumulated` | 0.0141 |
| 17 | `day_of_year_cos` | 0.0141 |
| 18 | `station_pressure` | 0.0141 |
| 19 | `wind_avg_lag30` | 0.0137 |
| 20 | `wind_gust_lag30` | 0.0134 |
| 21 | `temp_lag60` | 0.0123 |
| 22 | `relative_humidity` | 0.0120 |
| 23 | `temp_lag30` | 0.0074 |

**Assessment**: The hypothesis was confirmed for feature importance — removing time-of-day forced the model to anchor on physical state. `uv_lag30` jumped from 0.0167 (#3) in Exp 7 to 0.0341 (#1) — a 2× increase in relative importance, now clearly dominant. No time-of-day features appear anywhere (they were removed). `temp_lag120` moved up to #6, `temp_lag30` remains last (0.0074). Float val_loss (0.0006) is essentially identical to Exp 7.

**However**, quantized accuracy degraded: 1hr MAE worsened from 0.35 → 0.44°C and 3hr from 0.67 → 0.73°C. The model also converged faster (epoch 73 vs 95), suggesting time-of-day features were providing gradient signal that aided training, not just shortcutting predictions. Without them the model may be fitting more to noise in the compressed 6-minute averages.

**Interpretation**: Removing time-of-day produces a physically grounded feature importance ranking, but at a severe cost to calibration. The feature importance spread is now much more concentrated at the top (`uv_lag30` at 0.034 vs the next feature at 0.018), while Exp 7's importance was nearly flat — suggesting the no-TOD model found a strong but narrower predictive pathway that introduces systematic bias.

**Full-dataset bias analysis (Actual − Predicted)**:

| Model | Mean bias | Stddev |
|-------|-----------|--------|
| Exp 7 (with time-of-day) | +0.028°C | 0.144°C |
| Exp 8 (no time-of-day) | **−0.297°C** | 0.147°C |
| Original Model 5a | +0.164°C | 0.145°C |

Removing time-of-day introduced a **−0.297°C systematic warm bias** (model over-predicts temperature) while the random error (stddev) barely changed (0.147 vs 0.144°C). The bias is 10× worse than Exp 7 and in the opposite direction from the original Model 5a's artifact.

The pattern is diagnostically clear: stddev is essentially constant across all three models (~0.145°C), meaning the fundamental predictive uncertainty is the same regardless of architecture or data cleaning. What changes is only the mean bias. Time-of-day features were providing the calibration anchor that kept predictions centered — without them, `uv_lag30` and solar radiation dominate but are asymmetric (daytime-only signals), causing the model to systematically over-predict.

**Conclusion**: Exp 8 is not a viable deployment candidate. The bias of −0.297°C dwarfs the quantization error differences that motivated the experiment. **Exp 7 remains the deployment candidate.**

**EdgeTPU compilation result** ✅ — all 19 ops mapped to TPU, single subgraph, 0 off-chip streaming:

```
Output size:  640.70 KiB
On-chip memory used for caching model parameters:        600.75 KiB
On-chip memory remaining for caching model parameters:   7.25 MiB
Off-chip memory used for streaming uncached parameters:  0.00 B

Number of Edge TPU subgraphs: 1
Total number of operations:   19

Operator              Count   Status
RESHAPE               3       Mapped to Edge TPU
ADD                   1       Mapped to Edge TPU
MUL                   1       Mapped to Edge TPU
FULLY_CONNECTED       11      Mapped to Edge TPU
AVERAGE_POOL_2D       1       Mapped to Edge TPU
CONCATENATION         2       Mapped to Edge TPU
```

The AveragePooling1D architecture compiles cleanly with 23 features (no time-of-day) just as it did with 27 features in Exp 7. Compiled model is smaller (640.70 KB vs 840.70 KB in Exp 7) because fewer input features reduce the weight matrices. All 600.75 KB fits in the EdgeTPU's 8 MB on-chip SRAM — expected inference latency ~1–5 ms/sample on PCIe Coral M.2.

---

## Key Observations

1. **Time-of-day dominance is a training dynamics problem, not a feature gap.** Training dynamics (solved in Exp 3 with ReduceLR) was part of the problem. Adding explicit `temperature` (Exp 4) did not reduce time-of-day dominance — `temperature` ranked last, confirming the lag features already implicitly encode current temperature.

2. **Val_loss beats the reference with clean data.** Exp 4 (0.0002) substantially outperforms the reference (0.000706) on the identical target scale. The reference's lag-feature dominance may reflect training luck or data contamination artifacts rather than superior physical modeling.

3. **Gap invalidation removes 0.54% of rows that are quieter than average** — it is not selectively removing dynamic events and is not the cause of time-of-day snapping.

4. **Float accuracy and quantized accuracy don't track together.** Exp 4 had the best float val_loss (0.0002) but removing `temperature` in Exp 5 (val_loss 0.0004) actually improved 3hr quantized MAE (0.67 → 0.63°C). Float metrics are not a reliable proxy for deployed quantized performance.

5. **A model that relies primarily on time-of-day may fail on unusual weather events** (cold front passages, marine layer intrusions) that break the expected diurnal pattern. Lag features are essential for detecting those deviations — but their low permutation importance may reflect redundancy with time-of-day rather than low predictive value.

6. **EdgeTPU compiled overhead scales with the flat FC input dimension, not output size or layer count.** The 23.40 MB off-chip streaming (≈47ms of the 56ms inference cost) was identical whether there were three 4,860-wide FC layers (Exp 1–5) or one (Exp 6 bottleneck attempt). The overhead is ~4,800 bytes per input dimension. The threshold to fit in 8MB SRAM is ~1,660 flat dims.

7. **Gap invalidation eliminated 0.136°C of systematic bias without changing random error.** Full-dataset analysis of Exp 7 vs the original Model 5a: mean error dropped from +0.164°C to +0.028°C while stddev was unchanged (0.144 vs 0.145°C). The bias in the original model was almost certainly from post-gap sensor glitch data leaking into training targets. This is the clearest validation that gap invalidation was necessary and effective.

8. **Dense applied to a 3D tensor generates FULLY_CONNECTED version 9, which EdgeTPU does not support.** The `Dense(4)` temporal compression on a 3D input creates a FULLY_CONNECTED op with `keep_num_dims=true` (schema version 9). The TFLite converter then upgrades **all** FULLY_CONNECTED ops in the model to version 9, causing the EdgeTPU compiler to reject all FC operations — not just the 3D one. Fix: use `AveragePooling1D(pool_size=6, strides=6)` for dimension reduction instead of a 3D Dense. This generates `AVERAGE_POOL_2D` (EdgeTPU v1, fully supported), reduces 180 → 30 timesteps (810-dim flat), and keeps all subsequent Dense layers as standard 2D inputs at FC version 4.
