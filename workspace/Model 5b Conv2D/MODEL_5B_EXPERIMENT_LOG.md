# Model 5b Experiment Log

## ⛔ PROJECT CONCLUDED (2026-06-18)

**Conclusion: Model 5b beat Model 5a clean (5ac) in live INT8 deployment by Exp 37. The Conv2D layer itself adds no value — the Dense anchor path does the work.**

40 experiments across Conv1D and Conv2D architectures produced consistent evidence: the Dense anchor path (explicit lag features → Dense(32)) does all the useful work. The Conv2D path never learned complementary representations — it either learned the same thing as the anchor (structural redundancy) or shortcutted to diurnal signals (time-of-day encodings). However, the overall system (Conv2D + Dense anchor + explicit lags, Exp 37) deployed on Coral TPU via PTQ INT8 and outperformed Model 5a clean at all timeframes < 1 year in live data.

**Important correction on the val_loss comparison**: The raw val_loss numbers (5b Exp37: 0.002117 vs 5a: 0.000682) were used throughout this project as if directly comparable, but they are not — the two models use different target scaler ranges (5b: 28.45°C; 5a: 36.07°C). Converting to actual °C MAE: Exp37 float ≈ **0.174°C** vs Model 5a float ≈ **0.161°C** — marginally worse, not 3×. The float model was close to Model 5a; the INT8 deployed model beat Model 5a clean in practice.

**Key confirmed findings carried forward to Model 5c:**
- Model 5b (Exp 37, INT8 on Coral) beat Model 5a clean in live deployment at < 1 year timeframes ✅
- Float accuracy comparable to Model 5a is achievable with explicit lag features + Dense anchor (Exp 37 float: ~0.174°C MAE)
- PTQ caused accuracy degradation vs. float but did not prevent deployment-quality improvement over 5ac; offline PTQ validation (500-sample output collapse) was not representative of real accuracy
- Explicit temperature lag features (`temp_lag60/120/180`) are essential — the architecture cannot learn them implicitly through Conv2D+GAP
- Conv2D adds no value beyond Dense anchor; the Dense path with explicit lags is the effective model
- Pressure lag features (Zambretti insight: 3-hr tendency) are a likely gap in the current feature set — TFT can discover the optimal timeframes
- Feature discovery via TFT attention is the path to finding additional high-value lags for Model 5c

**Succeeded by**: [Model 5c TFT](../Model%205c%20TFT/MODEL_5C_PLAN.md) — TFT-based feature discovery, with Track A (everything model) and Track B (optimized TPU Dense model with TFT-discovered features).

---

## Overall Goal

**Match or beat Model 5a (val_loss=0.000682, val_mae=0.00445) using a Conv1D architecture with explicit temperature lag features, while remaining Edge TPU-compilable.**

### Retired goal (2026-04-30, after Exp 25)

The original goal was to achieve Model 5a accuracy with *no pre-computed lag features*, feeding only a raw 180-minute window. After 25 experiments this goal is retired. `temperature` ranked 17th–19th out of 19 features in every experiment while `time_of_day` and `solar_radiation` dominated, causing predictions to track the expected diurnal curve rather than actual temperature and producing a visible phase lag vs. reality. The best float result (Exp 24: val_loss=0.001343) remained 2× short of Model 5a and the gap is structural — diurnal signals are too strong for the Conv1D to learn implicit temperature anchoring under the Edge TPU op constraints. **Exp 26 onwards use `temp_lag60` and `temp_lag120` as explicit input features.**

### Priority Order

**Phase 1 — Float accuracy first (current focus)**
Get float val_loss ≤ 0.000682 before worrying about quantization. Do not let quantization failures drive architectural decisions until the float target is met.

**Phase 2 — Quantization (after float target is met)**
Once float val_loss ≤ 0.000682, apply QAT (Quantization-Aware Training) to produce a deployable INT8 model. Post-training quantization (PTQ) has failed consistently across Exp 12–24, and QAT (Exp 25) also failed to fix the near-constant output collapse — the root cause is unbounded activations in the `SliceTimestep`/`SliceFeatures` ops that are excluded from QAT wrapping. A different quantization strategy will be needed once float accuracy is solved.

### Constraints
- Feed the model a single `(180, n_features)` tensor of appropriately scaled station readings
- `temp_lag60` and `temp_lag120` are explicit input features (Exp 26+); `temp_delta_1` is also retained
- Cyclical time encodings (`sin`/`cos` of time-of-day, day-of-year) are included
- The trained model must survive `edgetpu_compiler` with all ops mapped to the TPU (INT8 quantized, no CPU-fallback ops, fits in 8 MB Edge TPU SRAM)
- All ops must be from the supported set: `Conv1D`, `Conv2D`, `DepthwiseConv2D`, `Dense`, `ReLU`, `ReLU6`, `GlobalAveragePooling`, `Add`, `Concatenate`, `Reshape`, `BatchNormalization`

**Reference**: `SPEC.md` — "Strategic goal (Conv1D on Edge TPU with explicit lag features)"

---

## Model 5a Reference Baseline

Model 5a is the benchmark this project targets. These numbers are the concrete reference for all accuracy comparisons.

| Metric | Value |
|--------|-------|
| val_loss | **0.000682** |
| val_mae (normalized) | **0.00445** |
| Best epoch | 97 |
| Model size | 788 KB |
| Top feature | `temp_lag120` (importance 0.093) |
| Architecture | Wide-deep-interaction dense, no Conv layers |
| Target scaler (`target_scaler_5a.json`) | min=−18.54°C, max=17.53°C, range=36.07°C |
| INT8 step size | 0.141°C/step (36.07°C ÷ 256) |
| **Deployed StdDev (4.5 years of live data)** | **0.145°C ≈ 1.0 INT8 step** |

The deployed StdDev of exactly 1 INT8 step is a useful calibration point: it shows how close a well-quantized model can come to the quantization floor, and sets the practical lower bound for what Model 5b should aim for.

---

## INT8 Output Precision & StdDev Analysis

### Current target_scaler_5b.json

`target_scaler_5b.json` is **recalculated every training run** (line 598 of `train_model_conv2D.py`) from the global min/max of all three diff targets (`temp_diff_1hr/2hr/3hr`) across the training data after gap detection, with a fixed ±2°C padding. It will produce identical results each run as long as the training data does not change — the computation is fully deterministic.

| Metric | Value |
|--------|-------|
| Current scaler | min=−13.36°C, max=15.09°C, range=**28.45°C** |
| INT8 step size | **0.111°C/step** (28.45°C ÷ 256) |
| Quantization noise floor (theoretical) | 0.032°C StdDev |

The tighter range compared to Model 5a (28.45°C vs 36.07°C) is an advantage: each INT8 step is finer (0.111°C vs 0.141°C), giving better output resolution if the model is accurate enough to exploit it.

### ⚠️ Grafana StdDev Is Time-Range Dependent — Use 30 Days as Primary Metric

Grafana automatically downsamples data when viewing long time ranges, averaging many raw points into each display bucket. **This averaging mathematically reduces the variance of the error**, making long-range stddev numbers look far better than actual model performance.

### May 2026 Snapshot (Exp32 float deployed)

| Time Window | Model 5a (INT8) | Model 5ac (INT8) | Model 5b Exp32 (float) |
|---|---|---|---|
| All time (2023–now) | 0.146°C ⚠️ | 0.147°C ⚠️ | 0.335°C ⚠️ |
| 2 years | 0.140°C ⚠️ | 0.145°C ⚠️ | 0.303°C ⚠️ |
| 1 year | 0.449°C | 0.465°C | 0.389°C |
| 6 months | 0.853°C | 0.860°C | 0.548°C |
| **30 days** ✅ | **0.988°C** | **1.00°C** | **0.607°C** |
| 2 days | 0.641°C | 0.635°C | 0.433°C |
| 24 hours | 0.612°C | 0.607°C | 0.336°C |

### June 2026 Snapshot (Exp37 INT8 deployed on Coral TPU)

| Time Window | Model 5a (INT8) | Model 5ac (INT8) | Model 5b Exp37 (INT8 TPU) |
|---|---|---|---|
| 6 months | 0.891°C | 0.899°C | **0.720°C** ✅ |
| **30 days** ✅ | 1.24°C | 1.26°C | **0.930°C** ✅ |
| 7 days | 1.30°C | 1.32°C | **0.922°C** ✅ |

**Mean bias (June 2026):**

| Model | 6 months | 30 days | 7 days |
|---|---|---|---|
| Model 5a | +0.161°C | +0.157°C | +0.117°C |
| Model 5ac | −0.286°C | −0.298°C | −0.341°C |
| **Model 5b Exp37** | **−0.078°C** | **−0.063°C** | **−0.014°C** |

**"All time" and "2 years" stddev numbers are misleading** — the apparent 5a advantage inverts completely at real timescales. **Model 5b Exp37 INT8 beats both 5a and 5ac at every measured timeframe**, with 19–26% lower StdDev and dramatically better bias (nearly unbiased vs 5a's +0.16°C warmth bias and 5ac's −0.30°C cold bias).

Note: the 30-day StdDev varies with the measurement window (5a: 0.988°C in May vs 1.24°C in June; 5ac: 1.00°C vs 1.26°C). The 6-month number is more stable and is the better primary metric for cross-model comparison.

**Evaluation protocol going forward:**
- **Primary: 30 days** — near-raw data resolution with enough statistical power; use this as the main comparison metric
- **Secondary: 6 months** — captures seasonal variation; some Grafana aggregation present but still meaningful
- **Ignore: "all time" and "2 years"** for stddev comparisons — Grafana averaging artifacts dominate

Mean bias is stable across all time windows (not affected by aggregation): 5a ~+0.16°C, 5ac ~−0.30°C, 5b ~+0.18°C.

### StdDev Benchmarks

| Scenario | StdDev (30-day) | StdDev (all-time⚠️) | INT8 steps (30-day) |
|----------|--------|--------|-----------|
| Model 5a deployed (INT8 TPU) | 0.988°C | 0.145°C | ~9 |
| Model 5ac (Model 5a Clean) deployed (INT8 TPU) | 1.00°C | 0.147°C | ~9 |
| Model 5b Exp 32 (float, live deployment) | **0.607°C** | 0.335°C | ~5.5 |
| **Model 5b Exp 37 (INT8 TPU, live deployment)** | **0.720°C** (6-month) / **0.930°C** (30-day) / **0.922°C** (7-day) | — | ~6.5 (6-month) |
| Model 5b with QAT, if float target met | **~0.10–0.14°C est.** | — | ~0.9–1.3 |
| Theoretical INT8 floor (perfect model) | 0.032°C | — | 0.3 |

**Conclusion**: Exp37 INT8 on Coral TPU outperforms both 5a and 5ac at every measured timeframe by 19–26% StdDev, with dramatically better bias (−0.07°C vs 5a's +0.16°C and 5ac's −0.30°C). The offline PTQ validation (500-sample output collapse) was not representative of real deployment quality. The finer INT8 step size (0.111°C vs 5a's 0.141°C) and multi-horizon explicit lag features are the likely contributors.

At the 6-month timescale (the most stable metric), **5b Exp37 INT8 (0.720°C) beats 5a (0.891°C) by 19% and 5ac (0.899°C) by 20%**.  
At 30 days, **5b float Exp32 (0.607°C) remains the best result**, ~35% better than 5a — the float model's advantage over INT8 remains real.

### ⚠️ Palm Springs Data Warning

The current `target_scaler_5b.json` is computed from **San Francisco training data only**. Palm Springs has substantially different climate characteristics:

| | San Francisco | Palm Springs |
|--|--|--|
| Temperature range (training data) | −4.6 to 34.4°C | 3.2 to **50.1°C** |
| Diurnal swing | Moderate (marine influence, fog) | Extreme (desert, 20–30°C/day typical) |
| Training data diff range (1/2/3hr, full dataset) | −14.9 to 15.5°C | −13.4 to 16.7°C |

If Palm Springs data is included in training, the combined diff range expands and the target scaler range widens. Estimated impact on INT8 precision:

| Training data | Approx. scaler range | INT8 step |
|---|---|---|
| SF only (current) | 28.45°C | 0.111°C/step |
| PS only (estimated) | ~34°C | ~0.133°C/step |
| SF + PS combined (estimated) | ~35–36°C | ~0.137–0.140°C/step |

A combined SF+PS scaler would degrade INT8 output resolution by roughly 25%, pushing the achievable StdDev floor from ~0.111°C toward ~0.138°C — close to Model 5a's step size and erasing the precision advantage of the tighter SF-only scaler.

**Mitigations to consider before adding PS data:**
1. **Separate models per climate** — train a PS-specific model with its own scaler; keeps both models at their optimal INT8 precision
2. **Climate-aware scaler clipping** — pad with a fixed worst-case ±2°C for the *expected deployment site* rather than the training extreme, accepting the small risk of out-of-range clipping for truly anomalous events
3. **Evaluate whether combined training actually improves SF accuracy** — PS data may not generalize to SF and could hurt the SF model despite widening the scaler

---

This document tracks all experiments, changes, and results for Model 5b Conv2D.

## Baseline: Original Model 5b (Before Optimizations)

**Date**: Initial state  
**Configuration**:
- Learning rate: 5e-4
- Loss: Weighted Huber loss (weights: 1.0, 1.3, 1.9 for 1hr/2hr/3hr)
- Architecture: Full capacity (wide=24, deep=192→96→64→48, interaction=24→48, lag=48, final=256)
- Batch size: 64 (training), 256 (validation)
- CPU threads: 1 intra_op, 2 inter_op (GPU mode)

**Results**:
- val_loss: 0.0094
- val_mae: 0.0156
- Best epoch: 17
- Model size: 1356.52 KB

**Notes**: User reported "This model came out better" - suggesting it was performing well initially.

---

## Experiment 1: Attempt to Match Model 5a's Approach

**Date**: First optimization attempt  
**Goal**: Apply Model 5a's successful training configuration to Model 5b

**Changes Made**:
1. Learning rate: 5e-4 → 1e-5 (50x reduction)
2. Loss: Weighted Huber → MSE (removed weighting)
3. Architecture: Reduced capacity significantly
   - Wide: 24 → 16
   - Deep: 192 → 128 → 64 → 32
   - Interaction: 24→48 → 16→32
   - Lag extraction: 48 → 32, individual lags 24→16
   - Removed final 256-unit merged layer

**Results**:
- val_loss: 0.0079 (worse than baseline)
- val_mae: 0.0158 (worse than baseline)
- Best epoch: 89

**Outcome**: ❌ **FAILED** - Model performance degraded  
**Analysis**: Architecture was simplified too much. Model 5b's lag extraction branch needs more capacity to be effective.

---

## Experiment 2: Partial Capacity Restoration

**Date**: Second optimization attempt  
**Goal**: Restore some capacity while keeping Model 5a's training approach

**Changes Made**:
1. Kept: Learning rate 1e-5, MSE loss
2. Architecture: Increased capacity moderately
   - Wide: 16 → 20
   - Deep: 128 → 160 → 80→48→40
   - Interaction: 16→32 → 20→40
   - Lag extraction: 32 → 40, individual lags 16 → 20
   - Restored final merged layer: 128 units

**Results**:
- val_loss: 0.0067 (still worse than baseline)
- val_mae: 0.0152 (slightly better than Exp 1, but worse than baseline)
- Best epoch: 100

**Outcome**: ❌ **FAILED** - Still worse than original  
**Analysis**: Still not enough capacity. User reported "This model seems to be even more worse. We seem to be going backwards."

---

## Experiment 3: Revert to Original Configuration

**Date**: Third attempt  
**Goal**: Restore original Model 5b configuration that was working

**Changes Made**:
1. Reverted all architecture changes to original:
   - Wide: 24
   - Deep: 192 → 96 → 64 → 48
   - Interaction: 24 → 48
   - Lag extraction: 24 per lag, final 48
   - Final merged layer: 256 units
2. Reverted training config:
   - Learning rate: 1e-5 → 5e-4
   - Loss: MSE → Weighted Huber (1.0, 1.3, 1.9)
3. **Kept GPU optimizations**:
   - Batch size: 64 → 512 (training and validation)
   - CPU threads: Optimized for data loading (half cores for intra_op, quarter for inter_op)

**Results**:
- val_loss: ~0.0094 (back to baseline level)
- val_mae: ~0.0156 (back to baseline level)

**Outcome**: ✅ **RESTORED** - Back to original performance  
**Analysis**: Original configuration was working. GPU optimizations (batch size, CPU threads) were kept as they improve training efficiency without affecting model quality.

---

## Experiment 4: Optimized Model 5b (Current)

**Date**: Final optimization attempt  
**Goal**: Combine Model 5a's stable training approach with Model 5b's architectural advantages

**Changes Made**:
1. **Training Configuration** (from Model 5a):
   - Learning rate: 5e-4 → 1e-5 (Model 5a's successful rate)
   - Loss: Weighted Huber → MSE (Model 5a's approach)
   - Removed loss weighting

2. **Architecture** (kept Model 5b's full capacity):
   - Wide: 24
   - Deep: 192 → 96 → 64 → 48 (full residual block)
   - Interaction: embedding_dim=24, projection_dim=48
   - Lag extraction: 24 per lag, final 48
   - Final merged layer: 256 units

3. **Training Enhancements**:
   - Early stopping: patience=10 (allows longer training like Model 5a's 97 epochs)
   - Learning rate scheduling: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-7)
   - Kept GPU optimizations: batch size 512, optimized CPU threads

**Expected Advantages**:
- Stable training from Model 5a's proven 1e-5 LR + MSE
- Better architecture with explicit lag extraction (Model 5b's advantage)
- Fine-tuning via LR scheduling for the complex architecture
- Better GPU utilization (batch size 512)

**Results**:
- val_loss: 0.0071 (worse than Model 5a's 0.00068)
- val_mae: 0.0151 (worse than Model 5a's 0.00445)
- Best epoch: 99

**Outcome**: ❌ **FAILED** - Still much worse than Model 5a  
**Analysis**: Learning rate of 1e-5 is too low for Model 5b's complex architecture. The model needs more learning capacity to train the lag extraction branch effectively.

---

## Experiment 5: Moderate Learning Rate (Current)

**Date**: Latest optimization attempt  
**Goal**: Find a learning rate that works for Model 5b's complex architecture

**Changes Made**:
1. **Learning Rate**: 1e-5 → 2e-5 (moderate rate - 2x Model 5a, but 25x lower than original 5e-4)
   - Model 5a's 1e-5 works for simple architecture
   - Original Model 5b's 5e-4 might be too high
   - Try middle ground: 2e-5 with warmup and scheduling

2. **Training Configuration**:
   - Loss: MSE (kept from Experiment 4)
   - Learning rate warmup: 1e-5 → 2e-5 over 5 epochs
   - LR scheduling: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
   - Early stopping: patience=10

3. **Architecture**: Kept full Model 5b architecture (no changes)

**Expected Advantages**:
- Moderate LR gives complex architecture room to learn
- Warmup provides stability
- LR scheduling fine-tunes after initial convergence

**Results**:
- val_loss: 0.0062 (slight improvement from Exp 4's 0.0071, but still much worse than Model 5a's 0.00068)
- val_mae: 0.0143 (slight improvement from Exp 4's 0.0151)
- Best epoch: 99

**Outcome**: ⚠️ **MINOR IMPROVEMENT** - Better than Exp 4, but still far from Model 5a  
**Analysis**: Moderate learning rate (2e-5) helped slightly, but the fundamental issue remains: Model 5b's lag extraction branch is redundant since lag features are already explicitly in the input. The model is trying to learn the same information twice.

---

## Experiment 6: Simplified Architecture (Current)

**Date**: Latest optimization attempt  
**Goal**: Remove redundant lag extraction and use Model 5a's proven architecture with Model 5b's additional features

**Key Insight**: Lag features (temp_lag30, temp_lag60, temp_lag120, etc.) are already explicitly in the input. The lag extraction branch was redundant and wasted model capacity trying to extract what's already provided.

**Changes Made**:
1. **Architecture Simplification**:
   - **Removed**: Entire lag extraction branch (convolutions, ExtractLagLayer, lag projections)
   - **Adopted**: Model 5a's proven simple architecture:
     - Wide: 24 → 16 (Model 5a size)
     - Deep: 192→96→64→48 → 128→64→32 (Model 5a structure)
     - Interaction: 24→48 → 16→32 (Model 5a size)
     - Removed final 256-unit merged layer (Model 5a goes directly to output)

2. **Training Configuration**:
   - Learning rate: 2e-5 → 1e-5 (Model 5a's proven rate)
   - Loss: MSE (Model 5a's approach)
   - LR scheduling: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-7)

3. **Kept Model 5b's Advantages**:
   - Additional input features: `temperature` and `temp_delta_1` (Model 5a doesn't have these)
   - Per-horizon target scaling (theoretically better than global)
   - GPU optimizations: batch size 512, optimized CPU threads

**Expected Advantages**:
- Model 5a's proven architecture (simple, effective)
- More input information than Model 5a (temperature, temp_delta_1)
- No wasted capacity on redundant lag extraction
- Model 5a's proven training config (1e-5 LR, MSE)
- Should outperform Model 5a because we have more input features

**Results**:
- val_loss: 0.0077 (worse than Experiment 5's 0.0062, worse than Model 5a's 0.00068)
- val_mae: 0.0159 (worse than Experiment 5's 0.0143, worse than Model 5a's 0.00445)
- Best epoch: 73
- Model size: 844.02 KB (smaller than Exp 5's 1356.52 KB due to simplified architecture)

**Outcome**: ❌ **FAILED** - Performed worse than Experiment 5  
**Analysis**: 
- The simplified architecture (Model 5a's size) is too small to effectively handle Model 5b's 29 input features (vs Model 5a's 20 features)
- Model 5a's architecture (wide=16, deep=128→64→32) was optimized for 20 features
- Model 5b has 9 additional features that need more capacity to process effectively
- Experiment 5's larger architecture (wide=24, deep=192→96→64→48) had enough capacity for the 29 features
- **Key insight**: Architecture size must match input feature complexity. More features need more capacity, even if some features are redundant.

**Hypothesis**: The architecture needs to be sized appropriately for the number of input features. Model 5a's small architecture works for 20 features, but Model 5b's 29 features need more capacity (like Experiment 5 had).

---

## Summary of Key Learnings

### What Worked ✅
1. **GPU Optimizations**: Increasing batch size to 512 and optimizing CPU threads improved GPU utilization without affecting model quality
2. **Original Architecture**: Model 5b's full capacity architecture with lag extraction is well-designed

### What Didn't Work ❌
1. **Over-simplification**: Reducing architecture capacity too much hurt performance
2. **Mismatched training/architecture**: Using Model 5a's low LR (1e-5) with reduced capacity didn't work well
3. **Too low learning rate**: 1e-5 works for Model 5a's simple architecture but is too low for Model 5b's complex architecture
4. **Redundant lag extraction**: The lag extraction branch was trying to learn what's already explicitly in the input (lag features), wasting capacity

### Current Strategy 🎯 (Next Steps)
**Experiment 5 was the best so far** (val_loss=0.0062, val_mae=0.0143):
- Moderate learning rate (2e-5) - works well for complex architecture
- Full Model 5b architecture (wide=24, deep=192→96→64→48) - enough capacity for 29 features
- MSE loss - simpler than weighted Huber
- LR warmup and scheduling - fine-tuning capability

**Key Learnings**:
1. **Architecture size must match input complexity**: Model 5a's small architecture (wide=16, deep=128→64→32) works for 20 features but is too small for Model 5b's 29 features
2. **Experiment 5's approach is best**: The full Model 5b architecture with moderate LR (2e-5) achieved the best results so far
3. **More features need more capacity**: Even if some features are redundant, the architecture needs enough capacity to process all input features effectively

**Next Steps**:
- Revert to Experiment 5's configuration (best performing: 0.0062 val_loss)
- OR try removing redundant lag extraction while keeping Experiment 5's larger architecture size
- OR try a hybrid: Remove lag extraction but increase architecture size moderately (e.g., wide=20, deep=160→80→48)

---

## Comparison with Model 5a

| Metric | Model 5a | Model 5b (Original) | Model 5b (Exp 4) | Model 5b (Exp 5) | Model 5b (Exp 6) | Model 5b (Exp 8) |
|--------|----------|---------------------|------------------|------------------|------------------|------------------|
| val_loss | 0.00068 | 0.0094 | 0.0071 | 0.0062 | 0.0077 | ⏳ Pending |
| val_mae | 0.00445 | 0.0156 | 0.0151 | 0.0143 | 0.0159 | ⏳ Pending |
| Best epoch | 97 | 17 | 99 | 99 | 73 | ⏳ Pending |
| Model size | 787.74 KB | 1356.52 KB | 1356.52 KB | 1356.52 KB | 844.02 KB | ⏳ Pending |
| Learning rate | 1e-5 | 5e-4 | 1e-5 | 2e-5 | 1e-5 | 1e-5 |
| Loss | MSE | Weighted Huber | MSE | MSE | MSE | MSE |
| Architecture | Simple dense | Lag extraction + dense | Lag extraction + dense | Lag extraction + dense | Simple dense (like 5a) | Simple dense (Model 5a) |
| Architecture size | wide=16, deep=128→64→32 | wide=24, deep=192→96→64→48 | wide=24, deep=192→96→64→48 | wide=24, deep=192→96→64→48 | wide=16, deep=128→64→32 | wide=16, deep=128→64→32 |
| Target scaling | Global | Per-horizon | Per-horizon | Per-horizon | Per-horizon | Global |
| Batch size | 256 | 64 | 512 | 512 | 512 | 512 |
| Input features | 20 | 29 | 29 | 29 | 29 (includes temp, temp_delta_1) | 29 (includes temp, temp_delta_1) |

---

## Next Steps

1. **Experiment 5 is currently the best** (val_loss=0.0062, val_mae=0.0143)
2. **Options to try**:
   - **Option A**: Revert to Experiment 5's exact configuration (full architecture + 2e-5 LR)
   - **Option B**: Remove lag extraction branch but keep Experiment 5's larger architecture size (wide=24, deep=192→96→64→48)
   - **Option C**: Hybrid approach - remove lag extraction but use moderate architecture size (wide=20, deep=160→80→48) with 2e-5 LR
3. **Key constraint**: Architecture size must be proportional to input feature count. Model 5b's 29 features need more capacity than Model 5a's 20 features.

---

## Notes

- Model 5a achieved excellent results with a simpler architecture and lower learning rate
- Model 5b's lag extraction branch was **redundant** - lag features are already explicitly in the input
- The key insight: Don't try to extract what's already provided. Use Model 5a's proven architecture with Model 5b's additional input features.
- GPU optimizations (batch size, CPU threads) are safe to keep as they only affect training efficiency
- Per-horizon target scaling is theoretically better than global scaling

---

## Key Learnings Summary

1. **Architecture size must match input complexity**: Model 5a's architecture (wide=16, deep=128→64→32) works for 20 features but is too small for Model 5b's 29 features
2. **More features need more capacity**: Even if some features are redundant, the architecture needs enough capacity to process all input features effectively
3. **Experiment 5 is best so far**: Full Model 5b architecture (wide=24, deep=192→96→64→48) with moderate LR (2e-5) achieved val_loss=0.0062
4. **Training config matters**: Moderate LR (2e-5) works better for larger architectures than very low LR (1e-5)
5. **Simplicity doesn't always win**: Model 5a's simple architecture works for 20 features, but Model 5b's 29 features need more capacity
6. **Redundancy might not be the issue**: The lag extraction branch might actually be helping, or the architecture size is more important than removing redundancy

---

---

## Experiment 7: Revert to Experiment 5 Configuration

**Date**: After Experiment 6  
**Goal**: Restore Experiment 5's best-performing configuration

**Changes Made**:
1. **Architecture**: Restored full Model 5b architecture
   - Wide: 16 → 24 (restored)
   - Deep: 128→64→32 → 192→96→64→48 (restored)
   - Interaction: 16→32 → 24→48 (restored)
   - Lag extraction: Restored (24 per lag, final 48)
   - Final merged layer: Restored (256 units)

2. **Training Configuration**:
   - Learning rate: 1e-5 → 2e-5 (restored Experiment 5's moderate LR)
   - LR warmup: 1e-5 → 2e-5 over 5 epochs (restored)
   - LR scheduling: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6) (restored)

**Rationale**: Experiment 5 achieved the best results so far (val_loss=0.0062, val_mae=0.0143). The full architecture with moderate LR works best for Model 5b's 29 input features.

**Results**: ⏳ **PENDING** - Awaiting training results

**Expected**: Should match or exceed Experiment 5's performance (val_loss=0.0062)

---

## Experiment 8: Option 1 - Model 5a Architecture + Model 5b Features + Global Scaling ⭐

**Date**: After architectural analysis  
**Goal**: Use Model 5a's proven architecture with Model 5b's additional features to beat Model 5a

**Strategy**: Model 5a achieved excellent results (val_loss=0.00068) with a simple architecture. By using Model 5a's exact architecture but with Model 5b's additional features (`temperature`, `temp_delta_1`), we should outperform Model 5a because we have more input information with the same proven architecture.

**Changes Made**:
1. **Architecture**: Use Model 5a's exact structure
   - Wide: 24 → 16 (Model 5a size)
   - Deep: 192→96→64→48 → 128→64→32 (Model 5a structure)
   - Interaction: 24→48 → 16→32 (Model 5a size)
   - **Removed**: Lag extraction branch (redundant - lag features already in input)
   - **Removed**: Final merged layer (Model 5a goes directly to output)

2. **Target Scaling**: Switch to Model 5a's global scaling
   - Changed from per-horizon scaling to global scaling (single min/max for all targets)
   - More stable than per-horizon scaling

3. **Training Configuration**: Use Model 5a's proven config
   - Learning rate: 2e-5 → 1e-5 (Model 5a's successful rate)
   - Early stopping: patience=10 → 5 (Model 5a's patience)
   - LR scheduling: min_lr=1e-6 → 1e-7 (Model 5a's setting)
   - LR warmup: Effectively no-op (1e-5 to 1e-5)

4. **Kept Model 5b's Advantages**:
   - Additional input features: `temperature` and `temp_delta_1` (Model 5a doesn't have these)
   - GPU optimizations: batch size 512, optimized CPU threads

**Expected Advantages**:
- Model 5a's proven architecture (val_loss=0.00068)
- More input information than Model 5a (temperature, temp_delta_1)
- Global scaling (more stable than per-horizon)
- No redundant lag extraction wasting capacity
- Simple architecture may generalize better

**Results**:
- val_loss: 0.00617 (~9× worse than Model 5a's 0.000682)
- val_mae: 0.01464 (worse than Model 5a's 0.00445)
- Best epoch: 52
- Model size: 844 KB

**Outcome**: ❌ **FAILED** - Did not beat Model 5a  
**Analysis**:
- Feature importance scores were nearly uniform across all 29 features (range 0.06–0.16), meaning the model found no dominant signal. In contrast, Model 5a shows `temp_lag120` clearly dominant at 0.093.
- Root cause: flattening 180 × 30 = 5,400 inputs means `temp_lag120` is buried at one specific position among thousands of equally-weighted inputs. Dense layers have no inductive bias to find it. Model 5a succeeds because `temp_lag120` is handed directly as position 1 of a 27-element feature vector with a strong direct weight.
- The flattened-window approach is fundamentally mismatched to the task. More features and better training config cannot fix it.

**Key insight**: The window-flattening approach cannot replicate Model 5a's performance. The architecture must have **temporal inductive bias** — i.e., something that naturally attends to specific time positions in the sequence. Given the Edge TPU constraint (no LSTM, GRU, or attention), the only viable path is **Dilated Conv1D**.

---

## Experiment 9: Dilated Conv1D — Edge TPU-Compatible Temporal Architecture

**Date**: After Experiment 8 analysis  
**Goal**: Match or beat Model 5a (val_loss=0.000682) using a raw 180-minute window with no pre-computed lag features, while remaining Edge TPU-compilable

**Strategic context**: The SPEC's north-star goal is to feed the model a raw 180-minute window of station readings and let the network learn temporal structure internally — no `temp_lag30/60/120`, no `humidity_lag*`, no pre-computed deltas. Cyclical time encodings (`sin`/`cos` of `time_of_day`, `day_of_year`) are acceptable minimal processing. The compiled model must run on the Coral Edge TPU (rules out LSTM, GRU, and attention).

**Architecture**:
- Input: `(180, n_features)` — one row per minute, raw station readings + cyclical time encodings per timestep
- No pre-computed lag features anywhere in the input
- `Conv1D(64, kernel=3, padding='causal', activation='relu')` — initial projection
- 5× Dilated residual blocks with dilation rates [1, 2, 4, 8, 16]:
  - `Conv1D(64, kernel=3, padding='causal', dilation_rate=d)` + `BatchNorm` + `ReLU`
  - Residual `Add` from block input
  - Combined receptive field: covers ~180 steps (1+2+4+8+16 × 2 × (3-1) = 62 steps of context per block, stacked)
- `GlobalAveragePooling1D()` — collapses time dimension, Edge TPU-compatible
- `Dense(64, relu)` → `Dropout(0.3)` → `Dense(32, relu)` → `BatchNorm`
- 3 output heads: `Dense(1, linear)` each for diff_1hr, diff_2hr, diff_3hr

**Per-timestep features (no lag columns)**:
- Raw station: `temperature`, `relative_humidity`, `station_pressure`, `solar_radiation`, `illuminance`, `uv`, `wind_avg`, `wind_gust`, `wind_lull`, `wind_direction_sin`, `wind_direction_cos`, `rain_accumulated`
- Cyclical time (acceptable minimal processing): `time_of_day_sin`, `time_of_day_cos`, `time_of_day_sin2`, `time_of_day_cos2`, `day_of_year_sin`, `day_of_year_cos`

**Training Configuration**:
- Learning rate: 1e-4 (higher than Model 5a's 1e-5 — Conv1D benefits from faster initial learning)
- Loss: MSE
- Batch size: 512
- Early stopping: patience=10 (Conv1D may need more epochs to converge than dense)
- ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-7
- Epochs: 100
- Target scaling: global min/max with ±2°C padding (Model 5a's approach)
- Gap-aware windowing: enabled (drop windows spanning data collection gaps)

**Edge TPU compliance checklist**:
- Causal padding (no future leakage, fully static shapes)
- BatchNorm fused at inference (supported)
- No dynamic shapes, no Lambda layers
- All ops in Edge TPU supported set: Conv1D, BatchNorm, ReLU, GlobalAveragePooling1D, Dense, Add, Dropout (removed at inference)

**Expected outcome**: The dilated stack's receptive field grows exponentially with dilation, so the model can learn "look 120 steps back" from data — achieving what `temp_lag120` does explicitly in Model 5a, but learned rather than hand-engineered.

**Results** (stopped early at epoch 17):
- val_loss: ~0.0165 (plateauing)
- val_1hr: 0.0025, val_2hr: 0.0053, val_3hr: 0.0086
- val_mae_1hr: 0.0334, val_mae_2hr: 0.0499, val_mae_3hr: 0.0644
- Epoch-by-epoch val_loss: 0.2615 → 0.0860 → 0.0390 → 0.0284 → 0.0242 → 0.0214 → ... → 0.0167 → 0.0165

**Outcome**: ❌ **INSUFFICIENT** - Converging ~24× short of Model 5a  
**Root cause identified**: The dilation stack [1, 2, 4, 8, 16] only provides a receptive field of **~65 steps** (cumulative: 3+2+4+8+16+32 = 65 with kernel=3). The model is literally blind to timesteps older than ~65 minutes. Since `temp_lag120` (120 steps back) was Model 5a's most important feature, the Conv1D cannot learn its equivalent — those timesteps are outside its receptive field. The model plateaued once it had learned everything visible within 65 steps.

The 1hr head (0.0025) is actually comparable to older dense models, confirming the architecture works for short-horizon context. The 2hr and 3hr heads lag because they need information from beyond the receptive field.

**Receptive field calculation**:
| Layer | Dilation | Adds | Cumulative reach |
|-------|---------|------|-----------------|
| conv_init | 1 | 2 | 3 |
| d=1 | 1 | 2 | 5 |
| d=2 | 2 | 4 | 9 |
| d=4 | 4 | 8 | 17 |
| d=8 | 8 | 16 | 33 |
| d=16 | 16 | 32 | **65** |

Fix: extend dilation stack to [1, 2, 4, 8, 16, 32, 64] → cumulative reach of **257 steps**, covering the full 180-minute window.

---

## Experiment 10: Extended Dilated Conv1D — Full 180-Step Receptive Field

**Date**: After Experiment 9 analysis  
**Goal**: Fix Experiment 9's receptive field limitation so the Conv1D can see the full 180-minute window

**Root cause of Exp 9 failure**: Dilation rates [1, 2, 4, 8, 16] only reach ~65 steps back. `temp_lag120` (the most important feature in Model 5a) lies at step 120, which is outside this receptive field. The model plateaued because it had extracted everything learnable within 65 minutes.

**Fix**: Add two more dilated blocks with rates [32, 64], extending the receptive field to 257 steps — well beyond the 180-minute window.

**Receptive field with new stack [1, 2, 4, 8, 16, 32, 64]**:
| Layer | Dilation | Adds | Cumulative reach |
|-------|---------|------|-----------------|
| conv_init | 1 | 2 | 3 |
| d=1 | 1 | 2 | 5 |
| d=2 | 2 | 4 | 9 |
| d=4 | 4 | 8 | 17 |
| d=8 | 8 | 16 | 33 |
| d=16 | 16 | 32 | 65 |
| d=32 | 32 | 64 | 129 |
| d=64 | 64 | 128 | **257** ✓ |

**Architecture changes from Exp 9**:
- Dilation rates: [1, 2, 4, 8, 16] → **[1, 2, 4, 8, 16, 32, 64]** (two extra blocks)
- Everything else identical: Conv1D(64), BatchNorm, ReLU, residual Add, GlobalAveragePooling1D, Dense(64)→Dropout→Dense(32)→BN, 3 output heads

**Training Configuration** (same as Exp 9):
- Learning rate: 1e-4
- Loss: MSE
- Batch size: 512
- Early stopping: patience=10
- ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-7
- Epochs: 100
- Target scaling: global min/max with ±2°C padding
- Gap-aware windowing: enabled

**Edge TPU compliance**: Same as Exp 9 — all ops remain Conv1D, BatchNorm, ReLU, GlobalAveragePooling1D, Dense, Add (fully supported).

**Expected outcome**: With the full 180-step window visible, the model can now learn the equivalent of `temp_lag120` and `temp_lag60` from the sequence, breaking through the ~0.0165 plateau of Exp 9 and approaching Model 5a's 0.000682.

**Results**:
- val_loss: 0.0039 (best at epoch 25, early stopped at epoch 35)
- val_mae: 0.0130
- Best epoch: 25
- Model size (quantized): 125.20 KB
- Float MAE (°C): 1hr=0.01, 2hr=0.02, 3hr=0.03
- **Quantized MAE (°C): 1hr=1.07, 2hr=0.99, 3hr=1.51** ← catastrophic degradation

**Outcome**: ⚠️ **PARTIAL** - Best float accuracy of any 5b experiment, but quantization failed  
**Analysis**:
- Extended receptive field worked: ~4× improvement over Exp 9 (0.0039 vs 0.0165), confirming the full 180-step window matters
- Still 5.7× short of Model 5a's 0.000682 in float
- **Two problems identified**:
  1. **Early stopping fired before LR reduction could help**: ReduceLROnPlateau fired at epoch 35 (same epoch as early stop, patience=10 from best epoch 25). The LR reduction never had epochs to act. Fix: reduce ReduceLROnPlateau patience to 3, increase early stopping patience to 20.
  2. **Catastrophic INT8 quantization degradation (~100×)**: BatchNorm layers across the dilated stack produce activation ranges that INT8 cannot represent faithfully. The representative dataset calibration is insufficient to cover the dynamic range of intermediate conv activations. Fix: replace BatchNorm with ReLU6 (which clips activations to [0,6], giving INT8 a bounded range to quantize), or use QAT (Quantization-Aware Training).

**Feature importance** (top features):
- `time_of_day_cos` (0.030), `solar_radiation` (0.024), `time_of_day_sin2` (0.023), `uv` (0.019)
- `temperature` was lowest at 0.009 — the Conv1D is learning temporal context rather than relying on the current reading

---

## Experiment 11: Fix LR Scheduling + Fix Quantization (ReLU6 + QAT)

**Date**: After Experiment 10 analysis  
**Goal**: Fix the two blocking issues from Exp 10: premature convergence and catastrophic quantization degradation

**Problem 1 — LR scheduling never activated**:
In Exp 10, best epoch was 25, early stopping patience was 10, so training ended at epoch 35 — the same epoch ReduceLROnPlateau fired. The LR reduction had no epochs to improve the model. Fix: reduce ReduceLROnPlateau patience to 3 and increase early stopping patience to 20, giving the model at least 10+ epochs at the reduced LR.

**Problem 2 — INT8 quantization degrades ~100×**:
BatchNorm produces unbounded activations that INT8 quantization cannot represent faithfully across the full dilated stack. Two complementary fixes:
- Replace `ReLU` → `ReLU6` in conv blocks: clips activations to [0, 6], giving INT8 a well-bounded range to work with (same approach used by MobileNet for Edge TPU)
- Enable **Quantization-Aware Training (QAT)**: inserts fake quantization nodes during training so the model learns to be robust to INT8 rounding. QAT is the standard solution for models that degrade under post-training quantization.

**Architecture** (same as Exp 10 — dilation rates [1, 2, 4, 8, 16, 32, 64]):
- All `ReLU` activations in conv blocks → **`ReLU6`** (bounds activations for INT8)
- Dense head ReLU activations → **`ReLU6`** for consistency
- Everything else identical

**Training Configuration**:
- Learning rate: 1e-4
- Loss: MSE
- Batch size: 512
- **Early stopping: patience=20** (up from 10 — gives LR reductions room to act)
- **ReduceLROnPlateau: patience=3** (down from 5 — fires faster, before early stopping)
- ReduceLROnPlateau: factor=0.5, min_lr=1e-7
- Epochs: 100
- **QAT**: Apply `tf.quantization.experimental.quantize_model()` after initial training converges, then fine-tune for additional epochs with fake quantization nodes active

**Expected outcome**:
- Float accuracy: ReduceLROnPlateau fires at epoch ~28 (25+3), giving ~17 more epochs at 5e-5 before early stopping at epoch ~45. Should push val_loss below 0.002.
- Quantization: ReLU6 + QAT should bring quantized MAE within 2–3× of float (vs the current 100×), making Edge TPU deployment viable.

**Results** (stopped manually at epoch ~92):
- Best val_loss: **0.0042** (epoch 83)
- val_mae: 0.0173 (1hr), 0.0251 (2hr), 0.0323 (3hr)
- LR cascade: 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6 → 3.125e-6 → 1.56e-6 → 7.81e-7
- Val_loss range throughout training: 0.0042–0.0053 — completely flat despite LR cascade

**Outcome**: ❌ **FAILED** — Slightly worse than Exp 10's 0.0039, and far short of Model 5a's 0.000682  
**Analysis**:
- Two bugs consumed most of the training run:
  1. `LearningRateScheduler(lambda epoch, lr: lr)` was overriding `ReduceLROnPlateau` every epoch, resetting LR to 1e-4 after each reduction — epochs 1–57 wasted
  2. After removing `LearningRateScheduler`, `ReduceLROnPlateau` was still not sticking due to Keras 3 API incompatibility (`optimizer.lr` vs `optimizer.learning_rate`). Fixed by replacing with a custom `ReduceLRCallback` using a lambda callable pattern accepted by Keras 3 Adam
- Even with working LR scheduling (epochs 66–92), val_loss was completely flat. The LR cascade from 5e-5 all the way down to ~8e-7 produced zero improvement in val_loss
- **Root cause: capacity bottleneck.** 64 filters throughout the dilated stack is insufficient representational capacity for 18 input features × 180 timesteps. The model has converged to a local optimum it cannot escape regardless of LR
- ReLU6 quantization fix was not tested (val_loss never got close enough to justify quantizing)

**Key discovery**: The LR scheduling bugs in Exp 10 and 11 were not the primary bottleneck — even with correct scheduling, the 64-filter architecture plateaus at ~0.004. The architecture itself needs more capacity.

---

## Experiment 12: Wider Filters (128 channels) — Capacity Increase

**Date**: After Experiment 11 analysis  
**Goal**: Break through the ~0.004 val_loss plateau by doubling filter count throughout the dilated stack

**Root cause of Exp 11 failure**: 64 filters in the dilated Conv1D stack is insufficient capacity for 18 features × 180 timesteps. The model converges hard at ~0.004 regardless of LR schedule. Doubling to 128 filters quadruples the parameter count in the conv layers, giving the model more representational power to learn the temporal structure of the 180-minute window.

**Architecture (first attempt — 128 filters, aborted)**:
- Conv filters: 64 → 128 throughout, dense head 64 → 128
- Batch size: 1024

**Results after 2 epochs (aborted — Metal GPU OOM)**:
- Epoch 1 val_loss: 0.0725
- Epoch 2 val_loss: **0.0155** (3× better than Exp 11 at same epoch — very promising trajectory)
- Aborted: 16 GB shared RAM exhausted (28 GB swap, Metal GPU stalled then hung)

**Outcome**: ❌ **ABORTED** — hardware OOM, not a model failure  
**Analysis**: The 128-filter trajectory was the most promising yet — epoch 2 val_loss of 0.0155 vs Exp 11's 0.0487. Wider filters are clearly helping. The failure was purely a memory constraint: 128 filters + batch 1024 exceeds what 16 GB shared CPU/GPU RAM can sustain. Reduce to 96 filters + batch 512 and retry.

---

## Experiment 12b: 96 Filters — Hardware-Constrained Capacity Increase

**Date**: After Exp 12 OOM  
**Goal**: Same as Exp 12 — break through the 0.004 plateau — but sized to fit within 16 GB RAM

**Architecture changes from Exp 11**:
- Conv filters: **64 → 96** throughout (initial projection + all 7 dilated blocks)
- Dense head: **64 → 96** (first dense layer, scaled to match conv width)
- Everything else identical: dilation rates [1, 2, 4, 8, 16, 32, 64], ReLU6, residual Add, GlobalAveragePooling1D, Dropout(0.3), Dense(32), 3 output heads

**Training Configuration**:
- Learning rate: 1e-4 (fresh training)
- Loss: MSE
- Batch size: **512** (reduced from 1024 — halves per-step memory)
- Early stopping: patience=20
- ReduceLRCallback: factor=0.5, patience=3, min_lr=1e-7
- Epochs: 100
- Target scaling: global min/max with ±2°C padding

**Expected outcome**:
- 96 filters gives ~2.25× the conv parameters of 64 — should meaningfully exceed 64's 0.004 ceiling
- Trajectory from Exp 12 (128 filters) suggests wider filters help significantly; 96 should capture most of that gain
- Realistic target: val_loss in the 0.001–0.003 range; matching Model 5a's 0.000682 possible but uncertain

**Results** (completed epoch 55, early stopped):
- Best val_loss: **0.0031** (epoch 35) — best float result yet, beats Exp 10/11
- val_mae (float): 1hr=0.01°C, 2hr=0.02°C, 3hr=0.03°C
- LR cascade: 1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6 → 1e-7 (hit min_lr by ep55)
- Quantized model size: 245 KB
- **Quantized MAE: 1hr=0.50°C, 2hr=1.11°C, 3hr=2.07°C** ← catastrophic again

**Outcome**: ⚠️ **PARTIAL** — Best float accuracy yet (0.0031), but quantization failed again  
**Analysis**:
- 96 filters broke through the 64-filter ceiling (0.0042 → 0.0031) — capacity increase confirmed as helpful
- Still 4.5× short of Model 5a's 0.000682 in float
- **Root cause of quantization failure identified**: ReLU6 was placed before `Add` in each residual block, but the residual path bypasses ReLU6 entirely. The `Add` output = unclipped_residual + ReLU6(conv), which is unbounded. INT8 quantization fails because every block output has an unbounded activation range.
  - Fix: add `ReLU6` **after** each `Add` to clip the combined output
- **LR cascaded too fast**: patience=3 with noisy val_loss fired every 3-4 epochs, exhausting the LR budget (1e-4 → 1e-7) by epoch 55. Need patience=5 to give each LR level more time to converge.
- Feature importance still very uniform (0.0099–0.0291) — no dominant feature, model is distributing attention across the sequence

---

## Experiment 13: Fix Residual Quantization + Slower LR Decay

**Date**: After Exp 12b analysis  
**Goal**: Fix the quantization failure by bounding residual block outputs, and slow LR decay to allow better convergence

**Problem 1 — Unbounded residual Add outputs**:
```
residual = x                          # unbounded
x = Conv → BN → ReLU6(x)             # clipped to [0, 6]
x = Add(residual, x)                  # = unbounded + [0,6] = UNBOUNDED ← INT8 fails here
```
Fix: add `ReLU6` after `Add`:
```
x = Add(residual, x)
x = ReLU6(x)                          # clips block output to [0, 6] ← INT8 safe
```

**Problem 2 — LR patience too low**:
patience=3 with noisy val_loss (oscillating ±0.001) triggered reductions every 3-4 epochs, cascading from 1e-4 to 1e-7 in 55 epochs without giving each LR level enough time. Fix: patience=5.

**Architecture** (same as Exp 12b — 96 filters, except):
- **ReLU6 added after every residual Add** (7 new activations in the dilated stack)

**Training Configuration** (same as Exp 12b, except):
- **ReduceLRCallback patience: 3 → 5**

**Expected outcome**:
- Quantized MAE should drop from ~1°C toward float MAE (~0.01°C) — all activations now bounded
- Slower LR decay gives model more gradient steps at each scale, potentially improving float accuracy below 0.0031

**Results** (training stopped at epoch 62, best at epoch 42):
- Best val_loss: **0.0037** (epoch 42)
- val_mae (float): 1hr=0.01°C, 2hr=0.02°C, 3hr=0.03°C
- LR cascade: ended at 7.81e-07 (ReduceLRCallback fired at epoch 62)
- Quantized model size: 245.51 KB
- **Quantized MAE: 1hr=1.14°C, 2hr=0.97°C, 3hr=1.25°C** ← still catastrophic

Feature importance (top/bottom):
- Top: `time_of_day_cos` (0.025), `illuminance` (0.021), `time_of_day_sin2` (0.017), `wind_direction_cos` (0.015)
- Bottom: `temperature` (0.0095) — Conv1D is learning temporal context, not current reading

**Outcome**: ❌ **FAILED** — Float accuracy slightly regressed (0.0037 vs Exp 12b's 0.0031), and quantization still catastrophic  
**Analysis**:
- Float accuracy: slightly worse than Exp 12b despite patience=5 fix — the slower LR decay didn't help convergence
- **Quantization still catastrophic**: ReLU6 after `Add` did not fix INT8 degradation. The ~100× gap (0.01°C float → 1.14°C quantized) persists
- Root cause of quantization failure is deeper than residual Add bounding: INT8 quantization of the Conv1D activations themselves is failing, not just the residual paths. The representative dataset calibration cannot cover the dynamic range of 96-filter intermediate activations across 7 dilated blocks
- **QAT (Quantization-Aware Training)** was listed as a fix in Exp 11 planning but not yet implemented — this is the next logical step. QAT inserts fake-quantization nodes during training so the model learns INT8-robust weights, rather than trying to post-hoc calibrate a float model
- Alternatively: the model size/complexity is making PTQ infeasible — may need to reduce to fewer layers or filters to make INT8 work without QAT

---

## Experiment 14: Multi-Point Temporal Extraction

**Date**: 2026-04-12  
**Goal**: Close the val_loss gap with Model 5a by fixing GlobalAveragePooling's positional information loss

**Root cause of Exp 10–13 plateau**: `GlobalAveragePooling1D` averages all 180 timesteps equally. Model 5a's dominant feature is `temp_lag120` — it succeeds because that signal has a direct weight path in a dense layer. GAP dilutes the equivalent signal from the dilated stack across all 180 steps, weakening it ~60×. All the LR tuning, filter count increases, and quantization fixes in Exp 10–13 could not overcome this fundamental information bottleneck.

**Fix**: Replace GAP with explicit temporal extraction at 4 key lag positions:
- `t=0` — current state (index -1)
- `t=-30` — 30 min ago (index -31, equiv. to `temp_lag30`)
- `t=-60` — 60 min ago (index -61, equiv. to `temp_lag60`)
- `t=-120` — 120 min ago (index -121, equiv. to `temp_lag120`, Model 5a's dominant feature)

Concatenate the 4 slices: `4 × 96 = 384-dim` input to the dense head.

**Architecture changes from Exp 13**:
- **Removed**: `GlobalAveragePooling1D`
- **Removed**: `ReLU6` after residual `Add` (was for quantization; not needed while chasing accuracy)
- **Added**: 4× `Lambda` layers extracting `conv_output[:, offset, :]` at t=0, -30, -60, -120
- **Added**: `Concatenate` to merge the 4 temporal slices (384-dim)
- **Dense head enlarged**: `Dense(96)→Dense(32)` → `Dense(128)→Dense(64)` (handles larger input)
- ReLU6 → standard ReLU throughout (quantization not a concern for now)

**Training Configuration** (same as Exp 13):
- Learning rate: 1e-4
- Loss: MSE
- Batch size: 512
- Early stopping: patience=20
- ReduceLRCallback: factor=0.5, patience=5, min_lr=1e-7
- Epochs: 100

**Expected outcome**: Direct access to the t=-120 representation from the dilated stack should give the model the same positional advantage Model 5a gets from `temp_lag120`, breaking through the ~0.003 plateau.

**Results** (best epoch 38, early stopped at epoch 58):
- Best val_loss: **0.0024** (epoch 38) — best Conv1D result yet
- val_mae: 0.0098 (1hr ~0.01°C, 2hr ~0.01°C, 3hr ~0.02°C)
- LR cascade: ended at 7.81e-07 by epoch 58
- Quantized model size: 291.80 KB
- **Quantized MAE: 1hr=0.52°C, 2hr=1.44°C, 3hr=1.14°C** ← still catastrophic

Feature importance (top/bottom):
- Top: `time_of_day_cos` (0.035), `illuminance` (0.027), `time_of_day_sin` (0.023)
- Bottom: `temperature` (0.0108) — raw temperature ranked last

**Outcome**: ⚠️ **PARTIAL** — Best Conv1D float accuracy yet (0.0024 vs 0.0031 in Exp 12b), but still 3.5× short of Model 5a's 0.000682. Quantization still failing.  
**Analysis**:
- Multi-point temporal extraction confirmed as the right direction — 0.0024 vs 0.0031 (Exp 12b)
- **Critical finding: `temperature` ranked dead last in feature importance.** In Model 5a, `temp_lag120` was the dominant feature by a wide margin. Here, the model is relying almost entirely on time-of-day features (time_of_day_cos/sin dominates) to predict temperature change — it has learned "what typically happens at this time of day" rather than "what is the actual temperature trajectory."
- Root cause: extracting the 96-dim conv representation at t=-120 gives the dense head a *mixed* vector encoding all 18 features. The temperature signal at that offset is diluted across 96 filters shared among all features. The model finds it easier to just key off time_of_day than to disentangle temperature from the blended representation.
- **Fix**: add raw feature values at the 4 key lag positions as explicit skip connections into the dense head. Concatenating `[raw_temp_t0, raw_temp_t30, raw_temp_t60, raw_temp_t120]` gives the model the same direct "temperature was X at t-120" signal that drove Model 5a's accuracy — without pre-computing lag features at data-prep time.
- Training loss (epoch 58: 0.0010) vs val_loss (0.0027): train/val gap suggests some overfitting; Dropout(0.3) may need adjustment

---

## Experiment 15: Multi-Point Temporal Extraction + Raw Temperature Skip Connections

**Date**: 2026-04-13  
**Goal**: Fix Exp 14's core failure — temperature ranked last in feature importance — by giving the dense head an explicit direct path to raw temperature at key lag positions

**Root cause of Exp 14 failure**: Feature importance showed `temperature` dead last (0.0108) with `time_of_day_cos` dominant (0.035). The model learned seasonal/diurnal patterns rather than the actual temperature trajectory. A 96-dim conv slice at t=-120 blends all 18 features — the dense head finds it easier to weight time encodings than to disentangle temperature from the mixed representation.

**Fix**: Add raw (scaled) temperature at 4 lag positions as explicit skip connections:
- `input[:, -1,   0:1]` → raw temp at t=0
- `input[:, -31,  0:1]` → raw temp at t=-30 (equiv. `temp_lag30`)
- `input[:, -61,  0:1]` → raw temp at t=-60 (equiv. `temp_lag60`)
- `input[:, -121, 0:1]` → raw temp at t=-120 (equiv. `temp_lag120`, Model 5a's dominant feature)

Temperature is feature index 0. Slicing `0:1` preserves the (batch, 1) shape for Concatenate.

**Dense head input**: `[conv_t0(96), conv_t30(96), conv_t60(96), conv_t120(96), raw_temp×4(4)]` = 388-dim  
**Architecture**: otherwise identical to Exp 14 (96 filters, dilation [1..64], Dense(128)→Dense(64))  
**Training config**: identical to Exp 14 (lr=1e-4, patience=20/5, batch=512)

**Expected outcome**: With a direct weight path from raw_temp_t120 to each output head, the model should replicate Model 5a's dominant `temp_lag120` signal without pre-computing lag columns. Feature importance should shift toward temperature features.

**Results** (best epoch 49, early stopped at epoch 69):
- Best val_loss: **0.0018** (epoch 49) — best Conv1D result yet, improving on Exp 14's 0.0024
- val_mae: 0.0081 (1hr ~0.01°C, 2hr ~0.01°C, 3hr ~0.01°C)
- Quantized model size: 293.48 KB
- **Quantized output: CONSTANT** — model outputs identical predictions for all 5 samples ← completely broken

Feature importance (top/bottom):
- Top: `time_of_day_cos` (0.039), `time_of_day_sin` (0.036), `solar_radiation` (0.023)
- Bottom: `temperature` (0.0137) — improved slightly from Exp 14's 0.0108 but still dead last

**Outcome**: ⚠️ **PARTIAL** — Float accuracy improved (0.0018 vs 0.0024), ~2.6× short of Model 5a's 0.000682. Quantization completely broken.

**Analysis**:
- Raw temperature skip connections helped float accuracy (0.0018 vs 0.0024 in Exp 14) — the direct temperature signal is useful
- **Temperature still dead last**: the 4 raw temperature scalars helped but the model still relies heavily on time-of-day. The conv representations dominate the 388-dim input (384 of 388 dims), so the 4 scalar temperature values get relatively small gradient weight
- **Quantization failure root cause identified**: Input quantization scale=0.12787, mapping float [0,1] to INT8 [-128, -120] — only 8 out of 256 INT8 levels used. The TFLite calibration computed a scale that covers the full INT8 range including large values that never appear, leaving the actual [0,1] input range crushed into 8 levels. Model outputs a constant because all inputs look the same after quantization.
- **Fix for float accuracy**: extend the skip connections from just temperature (1 feature) to ALL n_features at the 4 lag positions. That adds 4×18=72 explicit scalars alongside the 4×96=384 conv slices. The dense head gets direct access to humidity, pressure, wind, etc. at each lag — the same information Model 5a has via explicit lag columns.

---

## Experiment 16: Multi-Point Temporal Extraction + All-Feature Skip Connections

**Date**: 2026-04-15  
**Goal**: Extend Exp 15's temperature-only skips to all features, giving every sensor reading a direct weight path to the output heads

**Root cause of Exp 15 shortfall**: 4 temperature scalars are only ~1% of the 388-dim dense head input (4 of 388). The 384 conv dims still dominate gradient flow, so the model leans on time-of-day from the conv representations. Temperature still ranked last despite the skip.

**Fix**: Replace 4 temperature scalars with all `n_features` values at each of the 4 lag positions:
- `input[:, -1,   :]` — all features at t=0
- `input[:, -31,  :]` — all features at t=-30
- `input[:, -61,  :]` — all features at t=-60
- `input[:, -121, :]` — all features at t=-120

**Dense head input**: `[conv_t0(96), conv_t30(96), conv_t60(96), conv_t120(96), raw_t0(n_features), raw_t30(n_features), raw_t60(n_features), raw_t120(n_features)]`  
With n_features=18: `384 + 72 = 456-dim` (~16% of input is direct raw readings vs ~1% in Exp 15)

This mirrors exactly what Model 5a provides: the model sees actual sensor readings at t=0, -30, -60, -120 with direct weight paths — plus the learned conv representations for pattern context. The raw readings at t=-120 include temperature, humidity, pressure, wind — every feature that was pre-computed as `*_lag120` columns in Model 5a.

**Architecture**: identical to Exp 15 except raw skip extended from temp-only to all features  
**Training config**: identical (lr=1e-4, patience=20/5, batch=512)

**Expected outcome**: With all features at the 4 lag offsets having direct paths to the output, the model should replicate the full information Model 5a has from its explicit lag columns. Feature importance should show a more balanced distribution including temperature and other sensors.

**Results** (best epoch 26, early stopped at epoch 46):
- Best val_loss: **0.0024** (epoch 26) — regression from Exp 15's 0.0018, matched Exp 14
- val_mae: 0.0095 (1hr ~0.01°C, 2hr ~0.01°C, 3hr ~0.02°C)
- Quantized model size: 301.60 KB
- **Quantized MAE: 1.82°C / 0.92°C / 3.71°C** — non-constant this time (improvement over Exp 15) but still unusable

Feature importance (all features shown):
- Top: `time_of_day_cos` (0.0323), `solar_radiation` (0.0220), `time_of_day_sin2` (0.0188), `time_of_day_cos2` (0.0187), `time_of_day_sin` (0.0180)
- Bottom: `temperature` (0.0095) — dead last again despite all-feature skip connections

**Outcome**: ❌ **FAILED** — float accuracy regressed from Exp 15; temperature still ranks last; quantization still broken

**Analysis**:
- Extending skip connections from temperature-only to all features **did not help** float accuracy — it made it worse (0.0024 vs 0.0018). Adding 72 more scalars to the dense head may have introduced noise or conflicting gradient signals.
- **Temperature dead last is a persistent pattern**: every Conv1D experiment produces this. The model learns time-of-day diurnal patterns instead of the actual temperature trajectory. Skip connections haven't changed that.
- **Quantization failure root cause confirmed**: Input scale = 0.0912, mapping float [0,1] to INT8 [-128, −117] — only ~11 out of 256 INT8 levels used. The `representative_data_gen` uses `X_train_flat` without clipping; any outlier values > 1.0 inflate the scale, crushing actual inputs into a tiny band of INT8 levels. Exp 15 had 8 levels, Exp 16 has 11 — same root cause.
- **Fix is one line in `representative_data_gen`**: add `np.clip(window, 0.0, 1.0)` before yielding. This is the next experiment.

---

## Experiment 17: Dedicated Temperature Branch

**Date**: 2026-04-16  
**Goal**: Break the persistent pattern of `time_of_day` dominating feature importance and `temperature` ranking last — without adding any pre-computed lag features — by giving temperature its own dedicated Conv1D branch with isolated gradient flow

**Why previous approaches failed**:
- Exp 14–16 all added skip connections from the input to the dense head, but the skip values are raw scalars at 4 discrete offsets — the model still learns diurnal patterns from the 384-dim conv representation which dominates the concatenated input
- Adding `temp_delta_30/60/120` as features would be the obvious fix but **violates the no-pre-computed-lags constraint** in the spec — those are lag columns by another name
- The root problem is gradient competition: 18 features share the same 96 Conv1D filters. Time-of-day encodings have a strong, consistent diurnal signal that is easy for shared filters to latch onto. Temperature's multi-step trend is a weaker, noisier signal competing for the same filter capacity

**Fix**: Split the single Conv1D stack into two parallel branches:
1. **Main branch** — all 18 features, Conv1D(64 filters, dilation [1..64]): learns multi-feature context (humidity, pressure, solar, wind patterns)
2. **Temperature branch** — `temperature` + `temp_delta_1` only (2 channels), Conv1D(32 filters, dilation [1..64]): dedicated capacity for the temperature trajectory — these filters cannot be "hijacked" by time-of-day because those features are not present in this branch's input

Each branch gets its own temporal extraction at t=0/-30/-60/-120, then both are concatenated into the dense head:
```
Dense head input: [main_t0(64), main_t30(64), main_t60(64), main_t120(64),
                   temp_t0(32), temp_t30(32), temp_t60(32), temp_t120(32)]
                 = 256 + 128 = 384-dim
```

This stays fully within spec: no pre-computed lags, no lag columns — just 2 raw features (temperature, temp_delta_1) fed through their own conv stack. The model learns the temperature trajectory from the raw 180-step signal, it just has dedicated capacity to do so.

**Architecture changes from Exp 15**:
- **Removed**: all-temperature raw skip connections (Exp 15's 4 scalar skips — replaced by a full dedicated branch)
- **Removed**: 96-filter single stack → split into main(64) + temp(32)
- **Added**: parallel `Conv1D(32, dilation=[1..64])` stack on `input[:, :, 0:2]` (temperature + temp_delta_1 only)
- **Added**: temporal extraction at 4 offsets from temperature branch (temp_t0/30/60/120)
- **Dense head**: `Dense(128)` → `Dense(64)` → 3 output heads (same as Exp 15)

**Why 64+32 filters instead of 96+96**:
- Keeps model size comparable to Exp 15 (~300 KB target)
- Main branch still has more capacity than temperature branch — appropriate since it processes more features
- Total filter count (64+32=96) equals the single-branch filter count of Exp 14–16

**Training config**: identical to Exp 15 (lr=1e-4, patience=20/5, batch=512, MSE loss)

**Expected outcome**:
- `temperature` should rank higher in feature importance — dedicated filters cannot be co-opted by time-of-day
- val_loss target: improve on Exp 15's 0.0018, pushing toward Model 5a's 0.000682
- Feature importance should show a more balanced distribution

**Success criteria**: val_loss < 0.0015 AND temperature not dead last in feature importance

**Results** (best epoch 6, early stopped at epoch 51):
- Best val_loss: **0.0039** (epoch 6) — regression from Exp 15's 0.0018; worse than Exp 14's 0.0024
- val_mae: 0.0133 (1hr ~0.01°C, 2hr ~0.02°C, 3hr ~0.04°C)
- Quantized model size: 221.48 KB
- **Quantized MAE: 1.90°C / 5.14°C / 3.33°C** — still broken despite correct input quantization

Feature importance (all features shown, ranked):
- **temperature: 0.0228 ← #1 for the first time in any Conv1D experiment**
- illuminance: 0.0194, time_of_day_cos: 0.0183, uv: 0.0163, time_of_day_sin2: 0.0160
- time_of_day_cos2: 0.0156, wind_avg: 0.0153, day_of_year_sin: 0.0153
- rain_accumulated: 0.0153, wind_direction_cos: 0.0153, day_of_year_cos: 0.0153
- station_pressure: 0.0152, wind_direction_sin: 0.0152, wind_gust: 0.0152, wind_lull: 0.0152
- relative_humidity: 0.0147, time_of_day_sin: 0.0144, solar_radiation: 0.0138
- **temp_delta_1: 0.0108 ← last** (the dedicated branch captures temperature trajectory via raw temp; the 1-step delta adds less additional signal)

**Outcome**: ⚠️ **PARTIAL** — Feature importance fixed (temperature #1 ✅), float accuracy regressed (0.0039 ❌), quantization still broken ❌

**Analysis**:
- **Dedicated temperature branch confirmed working**: temperature jumped from dead last to #1. `time_of_day_cos` dropped from 0.0323 → 0.0183. The gradient isolation hypothesis is correct.
- **Input quantization fixed**: scale=0.00392, min=-128, max=127 — all 256 INT8 levels now used. The tighter `temp_delta_1` domain bounds (+/- 5°C) likely helped correct the scale. However quantized MAE is still 1.9–5°C — the intermediate Conv1D activations are the remaining source of INT8 degradation.
- **Float accuracy regressed due to capacity reduction**: reducing the main branch from 96 → 64 filters was too aggressive. Total filter count (64+32=96) is the same as Exp 14–16's single branch, but 64 filters is insufficient for the 19-feature multi-context task. The model also overfit badly — training loss ~0.001 vs val_loss ~0.004 (4× gap), and val_loss never improved past epoch 6, triggering LR reductions at epochs 9, 15, 20, 25, 30, 51 and collapsing to 6.25e-6 by epoch 51.
- **Fix**: restore main branch to 96 filters (full capacity, same as Exp 14–16) while keeping the 32-filter temp branch. This gives full context capacity + dedicated temperature capacity. Increase ReduceLRCallback patience from 5 → 8 to reduce premature LR cascades on noisy val_loss.

---

## Experiment 18: Full-Capacity Dual Branch (96 main + 32 temp)

**Date**: 2026-04-18  
**Goal**: Combine Exp 17's proven feature importance fix (dedicated temperature branch) with Exp 15's float accuracy (val_loss 0.0018) by restoring main branch to full 96-filter capacity

**Root cause of Exp 17 float regression**: Main branch was reduced to 64 filters to keep model size comparable to prior experiments. But 64 filters is insufficient for learning multi-feature context across 19 features — the single-branch experiments all used 96 filters and Exp 14–16 needed that capacity to reach 0.0018–0.0024. Reducing it caused underfitting and an early best epoch (6) with persistent val_loss oscillation.

**Architecture**:
- **Main branch**: `Conv1D(96 filters, dilation=[1..64])` on all 19 features — full capacity restored, identical to Exp 14–16
- **Temperature branch**: `Conv1D(32 filters, dilation=[1..64])` on `temperature` + `temp_delta_1` only — kept from Exp 17
- **Temporal extraction**: 4 offsets (t=0/-30/-60/-120) from each branch
- **Dense head input**: `(4×96) + (4×32) = 384 + 128 = 512-dim`
- **Dense head**: `Dense(192)` → `Dense(96)` → 3 output heads (scaled up proportionally from Exp 17's 128→64 to handle the larger 512-dim input)

**Training config changes from Exp 17**:
- **ReduceLRCallback patience: 5 → 8** — reduces premature LR cascade on noisy val_loss oscillations
- Everything else identical (lr=1e-4, early stopping patience=20, batch=512, MSE loss)

**Expected outcome**:
- Temperature should remain #1 in feature importance (isolated branch still present)
- val_loss target: match or beat Exp 15's 0.0018, ideally approach Model 5a's 0.000682
- Model size: ~350–400 KB (larger than Exp 17's 221 KB due to 96-filter main branch)

**Success criteria**: val_loss < 0.0018 AND temperature remains in top 3 feature importance

**Results** (best epoch 53, early stopped at epoch 73):
- Best val_loss: **0.0014** (epoch 53) — best float accuracy of any Conv1D experiment
- val_mae: 0.0060 (avg); 1hr ~0.00°C, 2hr ~0.01°C, 3hr ~0.01°C (in scaled units)
- Quantized model size: 394.54 KB
- **Quantized MAE: 0.55°C / 1.09°C / 1.90°C** — still broken

Feature importance (ranked):
- Top: `time_of_day_cos` (0.0282), `illuminance` (0.0250), `solar_radiation` (0.0236), `time_of_day_cos2` (0.0233), `uv` (0.0232)
- **temperature: 0.0185 (18th/19)** — slipped back from Exp 17's #1 ranking
- **temp_delta_1: 0.0170 (19th/last)**

**Outcome**: ⚠️ **PARTIAL** — Best float accuracy yet (0.0014 vs Exp 15's 0.0018), still ~2× short of Model 5a's 0.000682. Temperature ranking regressed. Quantization still broken.

**Analysis**:
- **Float accuracy improved**: 96-filter main branch restored full capacity — confirmed as the right call. 0.0014 is the best Conv1D result so far.
- **Temperature ranking regressed**: Exp 17 had temperature #1 (main=64 filters), Exp 18 has temperature 18th (main=96 filters). The 96-filter main branch dominates gradient flow over the 32-filter temp branch. The gradient isolation is diluted by the 3:1 filter ratio. To keep temperature's dedicated signal competitive, the temp branch needs more filters relative to the main branch.
- **Overfitting is the plateau cause**: training loss at epoch 73 was ~6.2e-4 while val_loss was 1.4e-3 — a 2.3× gap. The model converged on training data but stopped generalizing after epoch 53. Dropout(0.3) is insufficient for the 512-dim concatenated representation.
- **Quantization unchanged**: input scale=0.00392 (all 256 INT8 levels used ✓), but intermediate Conv1D activations still produce ~1–2°C quantized error. The intermediate activation range problem was never solved — requires QAT or fundamentally different quantization approach.
- **Note**: results file saved as `conv1d_exp16_run1` (code name not updated from Exp 16). The old Exp 16 results file was overwritten. Update experiment name in code for next run.

**Fix for Exp 19**:
1. **Temperature ranking**: increase temp branch from 32 → 64 filters (2:1 ratio main:temp vs current 3:1). Should restore temperature's competitive gradient signal while keeping main branch at full capacity.
2. **Overfitting**: increase Dropout from 0.3 → 0.4 in the dense head.
3. **Experiment name**: update code to `conv1d_exp19_run1` to avoid overwriting results.

---

## Experiment 19: Rebalanced Dual Branch (96 main + 64 temp) + Dropout Fix

**Date**: 2026-04-19  
**Goal**: Recover temperature's #1 feature importance ranking (lost when main branch was scaled to 96 filters) while maintaining Exp 18's best float accuracy (0.0014), and address overfitting.

**Root causes of Exp 18 shortfalls**:
1. **Temperature ranking slipped to 18th**: 96-filter main vs 32-filter temp = 3:1 ratio. The main branch's 3× more filters dominate the concatenated 512-dim representation, drowning out the temperature branch's gradient signal. Exp 17 had 2:1 ratio (64:32) and temperature was #1.
2. **Overfitting plateau at epoch 53**: train loss 6.2e-4 vs val_loss 1.4e-3 (2.3× gap). Dropout(0.3) on the 512-dim dense head is insufficient — the model memorized training patterns after epoch 53.

**Architecture changes from Exp 18**:
- **Temp branch filters: 32 → 64** (2:1 main:temp ratio, matching Exp 17's ratio)
- Dense head input: `(4×96) + (4×64) = 384 + 256 = 640-dim` (up from 512-dim)
- Dense head: `Dense(192)` → `Dense(96)` (same — handles larger input)
- **Dropout: 0.3 → 0.4** — more regularization to reduce overfitting

**Training config**: identical to Exp 18 (lr=1e-4, patience=20/8, batch=512, MSE loss)

**Expected outcome**:
- Temperature should return to top features (2:1 ratio proved sufficient in Exp 17)
- val_loss target: match or improve on Exp 18's 0.0014 — reduced overfitting should push best epoch later
- Ideally approach Model 5a's 0.000682

**Success criteria**: val_loss < 0.0014 AND temperature in top 5 feature importance

**Results** (aborted at epoch ~64, best at epoch 31):
- Best val_loss: **0.0015** (epoch 31) — regression from Exp 18's 0.0014
- val_loss range epochs 31–63: 0.0015–0.0068 — severe oscillation throughout
- LR reduction bug: `ReduceLRCallback` fired at epochs 39, 47, and 60 but reductions never stuck (logged as `learning_rate: 1.0000e-04` every epoch). All ~64 epochs effectively trained at 1e-4 with no scheduling.
- Epoch times: 1700–4400s (2–3× slower than Exp 18's ~1350s) due to larger temp branch
- Run aborted — not expected to improve with remaining epochs

**Outcome**: ❌ **FAILED** — Worse than Exp 18 on every metric

**Analysis**:
- **64-filter temp branch causes gradient interference**: val_loss oscillation was present from epoch 31 (before any LR change), not caused by the LR bug. The 640-dim concatenated head with near-equal main/temp capacity destabilizes gradient flow compared to Exp 18's 3:1 ratio.
- **LR bug persisted**: Three different approaches tried (`tf.Variable`, property setter, `_learning_rate.assign`) — none worked because the optimizer's internal variable requires direct access via `opt._learning_rate.assign()`. Fixed in the code but not active during this run.
- **Epoch 2× slower**: 640-dim head + 64-filter temp branch costs significantly more compute per step with no accuracy benefit.
- **Conclusion**: The Exp 18 architecture (96 main + 32 temp, 512-dim head) was the right balance. Increasing temp branch capacity hurts rather than helps.

---

## Experiment 20: Exp 18 Architecture + Working LR Scheduling

**Date**: 2026-04-21  
**Goal**: Re-run Exp 18's best architecture with correct LR scheduling to see how far it can go without the `tf.Variable` bug

**Key insight**: Exp 18 produced the best result so far (val_loss 0.0014) but the LR reductions were implemented via `tf.Variable` passed to Adam — which Keras 3 accepts but Keras 3's property getter returns a copy, so `.assign()` on the return value didn't update the actual optimizer state. The LR in Exp 18 may have effectively been stuck at 1e-4 throughout just like Exp 19.

**Root cause of LR bug (all experiments)**: Keras 3 Adam stores the LR in `optimizer._learning_rate` (a `tf.Variable`). Accessing it via the `optimizer.learning_rate` property getter returns a copy — calling `.assign()` on that copy modifies the copy, not the stored variable. Fix: call `optimizer._learning_rate.assign(new_lr)` directly.

**Architecture**: identical to Exp 18 — no changes
- Main branch: `Conv1D(96 filters, dilation=[1..64])` on all 19 features
- Temp branch: `Conv1D(32 filters, dilation=[1..64])` on temperature + temp_delta_1 only
- Temporal extraction at t=0/-30/-60/-120 from each branch
- Dense head input: `(4×96) + (4×32) = 512-dim`
- Dense head: `Dense(192)` → `Dropout(0.3)` → `Dense(96)` → 3 outputs

**Changes from Exp 19**:
- `TEMP_FILTERS`: 64 → **32** (back to Exp 18 architecture)
- `Dropout`: 0.4 → **0.3** (back to Exp 18)
- LR callback: **`opt._learning_rate.assign(new_lr)`** — correct fix, verified to update internal variable
- Experiment name: `conv1d_exp20_run1`

**Training config**: identical to Exp 18 (lr=1e-4, patience=20/8, batch=512, MSE loss)

**Expected outcome**:
- Stable training (no oscillation) like Exp 18
- Working LR reductions should push val_loss below Exp 18's 0.0014
- Temperature should remain at or near top of feature importance (32-filter temp branch proved sufficient in Exp 17)

**Success criteria**: val_loss < 0.0014

**Results**: Cancelled

---

## Experiment 21: Multi-Scale Temporal Basis Expansion

**Date**: 2026-04-21 (planned — implement after Exp 20 completes)
**Goal**: Close the remaining 2× gap to Model 5a by replacing hard single-point temporal extraction with learned multi-scale temporal neighborhoods

**Core insight**: The current architecture's `SliceTimestep(-121)` takes exactly one 96-dim vector from t=-120. If the dilated stack is even slightly misaligned, the extraction misses. Model 5a wins because `temp_lag120` is an exact, pre-aligned signal with a direct weight path. The solution is to give each extraction point a **learned neighborhood** — instead of "what is at exactly t=-120?", ask "what temporal pattern exists around t=-90 to t=-150?". This is the closest approximation to soft temporal selection under the Edge TPU op constraint.

**Why this may close the 5a gap**:
- Model 5a advantage: exact temporal alignment via pre-computed lag features
- Current model weakness: must infer temporal alignment indirectly through the dilated stack
- Multi-scale bank adds: learned temporal basis functions that approximate alignment without explicit lags or forbidden ops (no attention, no softmax, no sigmoid gating)

**Architecture** (changes from Exp 20 in **bold**):

```
Input (180, n_features)
  ↓
Dilated stack [1,2,4,8,16,32,64] → (180, 96)    ← UNCHANGED from Exp 20
  ↓
Multi-scale temporal bank (main branch):          ← NEW: inserted before extraction
  Branch A: Conv1D(96, kernel=3, dilation=1,  causal) → (180, 96)  covers ±2 steps
  Branch B: Conv1D(96, kernel=3, dilation=4,  causal) → (180, 96)  covers ±8 steps
  Branch C: Conv1D(96, kernel=3, dilation=8,  causal) → (180, 96)  covers ±16 steps
  Branch D: Conv1D(96, kernel=3, dilation=16, causal) → (180, 96)  covers ±32 steps
  Concatenate → (180, 384)
  Project: Conv1D(96, kernel=1, causal) → (180, 96)   ← mixes scales, preserves shape
  ↓
SliceTimestep at t=0/-30/-60/-120 → 4×96 = 384-dim    ← UNCHANGED from Exp 20
  ↓
Temperature branch (parallel, same structure):
  Dilated stack [1..64] → (180, 32)               ← UNCHANGED from Exp 20
  Multi-scale bank (scaled down):                  ← NEW
    4 branches × Conv1D(32, kernel=3, dilation=1/4/8/16)
    Concatenate → (180, 128)
    Project: Conv1D(32, kernel=1) → (180, 32)
  SliceTimestep at t=0/-30/-60/-120 → 4×32 = 128-dim
  ↓
Concatenate both branches: 384 + 128 = 512-dim    ← same as Exp 20, controlled comparison
  ↓
Dense(192) → Dropout(0.3) → Dense(96) → 3 outputs ← UNCHANGED from Exp 20
```

**Key design decisions**:
- **k=3 throughout, vary dilation** — dilation controls neighborhood span; varying kernel size adds parameter cost without extra receptive field benefit
- **Project back to 96/32** before extraction — keeps dense head input dimension identical to Exp 20 (512-dim). This is the critical controlled variable: if val_loss improves, the multi-scale bank is the cause
- **Dilated stack retained** — the [1..64] stack provides the full 257-step receptive field; the multi-scale bank enriches the representations at each timestep before extraction, it does not replace the stack
- **Temperature branch gets its own bank** — keeps gradient isolation intact; 32-filter branches maintain the 3:1 capacity ratio proven in Exp 18/20
- **All ops TPU-compatible**: Conv1D, ReLU, Concatenate, BatchNorm, Add — no attention, no softmax, no sigmoid

**Training config**: identical to Exp 20 (lr=1e-4, patience=20/8, batch=512, MSE, working LR scheduling)

**Expected model size**: ~500–600 KB quantized (Exp 20 ~395 KB + 4 extra Conv1D layers per branch)

**Success criteria**: val_loss < Exp 20's result AND temporal extraction neighborhoods produce measurably different representations at each offset (verifiable via feature importance)

**Risks**:
- Memory: 4 parallel Conv1D(96) on (180, 96) inputs may stress 16GB shared RAM — monitor epoch 1 closely; reduce to 3 branches if needed
- If val_loss doesn't improve vs Exp 20, the multi-scale bank adds no value and the extraction pinpoint accuracy is not the bottleneck — conclusion: the gap to Model 5a is elsewhere

**Actual model stats (from training run)**:
- Total params: 508,608 (1.94 MB float32 unquantized)
- Trainable params: 505,088 / Non-trainable: 3,520
- Training batches/epoch: 1454 | Validation batches: 715
- Pre-training validation: ✅ all 6 checks passed

**Results** (stopped at epoch 81, best at epoch 60):
- Best val_loss: **0.0017** (epoch 60) — regression from Exp 18's 0.0014
- val_loss at stopping (epoch 81): 0.0023
- Train loss at stopping: 2.49e-4 → **train/val gap: ~9×** (severe overfitting)
- LR: stuck at 1e-4 throughout — `ReduceLRCallback` never fired despite 20+ epochs past best
- TFLite model size (INT8 quantized, epoch 60): **586.68 KB**
- Quantized accuracy: not tested (overfitting ruled out continued run)

**Outcome**: ❌ **FAILED** — Worse than Exp 18 (0.0017 vs 0.0014), still 2.5× short of Model 5a's 0.000682

**Analysis**:
- **Multi-scale temporal bank did not help**: Exp 21 regressed vs Exp 18 despite the additional architecture. The multi-scale bank adds parameters and compute but doesn't improve the fundamental information bottleneck.
- **LR scheduling never fired**: `ReduceLRCallback` with patience=8 should have reduced LR around epoch 68. LR remained at 1e-4 through epoch 81. Root cause: mixed precision training wraps Adam in `LossScaleOptimizer` — the `_set_lr()` fix (`opt._learning_rate.assign()`) targets the inner optimizer, but Keras/TF may be re-creating the inner optimizer variable when mixed precision loss scaling adjusts, losing the assigned value.
- **Overfitting worsened**: train/val gap at stopping was ~9× (vs Exp 18's ~2.3× at epoch 73). The additional parameters from the multi-scale bank increased overfitting without improving generalization.
- **Conclusion**: The multi-scale temporal bank is not the path forward. The two blocking issues — LR scheduling unreliable with mixed precision, and overfitting — need to be addressed before adding more architecture complexity.

**Fix for next experiment**:
1. **Disable mixed precision** — removes `LossScaleOptimizer` wrapper entirely, makes `ReduceLRCallback` work reliably again. Training speed difference is negligible on M1 Pro (bottleneck is data pipeline, not compute).
2. **Revert to Exp 18 architecture** (96 main + 32 temp, no multi-scale bank) — Exp 18 achieved the best result (0.0014) and the multi-scale bank added no benefit.
3. **With working LR scheduling**, give the proven architecture a full run with proper LR reductions to see if it can push below 0.0014.

---

## Experiment 22: Exp 18 Architecture + Disabled Mixed Precision + Working LR Scheduling

**Date**: 2026-04-24 (planned)
**Goal**: Re-run Exp 18's best architecture with mixed precision disabled so `ReduceLRCallback` works reliably, and determine the true accuracy ceiling of the proven dual-branch architecture.

**Root cause of LR bug (persistent across Exp 19–21)**: Mixed precision wraps Adam in `LossScaleOptimizer`. All attempts to assign the inner optimizer's LR have failed — the wrapper interferes with the assignment or re-initializes the inner variable after loss scale adjustments. Disabling mixed precision removes the wrapper entirely and eliminates this entire problem class.

**Changes from Exp 21**:
1. **Disable mixed precision** — remove `tf.keras.mixed_precision.set_global_policy('mixed_float16')`. Training will run in float32 throughout. Speed impact is minimal on M1 Pro Metal (bottleneck is CPU data pipeline, not GPU tensor math).
2. **Revert to Exp 18 architecture** — remove multi-scale temporal bank. Main(96) + temp(32), 512-dim dense head, Dense(192)→Dropout(0.3)→Dense(96)→3 outputs.
3. **Verify LR is actually reducing** — add an explicit `print` in `_set_lr` to confirm the assignment takes effect.

**Training config**: lr=1e-4, patience=20/8, batch=512, MSE loss — identical to Exp 18/21

**Success criteria**: val_loss < 0.0014 (Exp 18's result) with confirmed LR reductions visible in training log

**Results** (best epoch 33, interrupted at epoch 40, resumed from epoch 35, early stopped at epoch 57):
- Best val_loss: **0.0020** (epoch 33) — regression from Exp 18's 0.0014
- Final train_loss: 6.6e-4 vs val_loss 0.0020 — train/val gap ~3× (overfitting)
- LR reductions confirmed working: 1e-4 → 5e-5 (epoch 21) → 2.5e-5 (epoch 53)
- LR display bug fixed: epoch summary now shows the new LR after reduction (not the pre-reduction value)
- LR revert bug fixed: checkpoint validator was silently restoring LR to 1e-4 after each epoch via `model.load_weights()` — fixed by saving/restoring optimizer LR around validation
- Val_loss very noisy throughout (oscillating 0.0020–0.0052), never improving past epoch 33
- Quantized MAE: 2.38°C / 3.85°C / 4.19°C — broken (not a focus per updated goal priority)

Feature importance (ranked):
- Top: `time_of_day_cos` (0.0208), `solar_radiation` (0.0190), `time_of_day_sin` (0.0189), `illuminance` (0.0185)
- Bottom: `temp_delta_1` (0.0145), `temperature` (0.0144) — temperature near last again

**Outcome**: ❌ **FAILED** — Worse than Exp 18 despite working LR scheduling. Overfitting (3× train/val gap) is the primary bottleneck.

**Analysis**:
- Working LR scheduling did not recover Exp 18's 0.0014 — the architecture is overfitting, not underfitting. LR reductions alone cannot fix overfitting.
- The 3× train/val gap (6.6e-4 train vs 0.0020 val) indicates the model memorises training sequences but does not generalise. Dropout(0.3) is insufficient for the 512-dim concatenated head.
- Val_loss oscillation is larger than in Exp 18 — suggests the 512-dim head has too many parameters relative to the effective size of the validation distribution.
- Temperature still near last in feature importance — the 3:1 filter ratio (96:32) continues to dilute the temperature branch signal. Overfitting may be amplifying this: the main branch memorises diurnal patterns on training data rather than learning generalisable temperature dynamics.

---

## Experiment 23: Address Overfitting — Stronger Regularisation

**Date**: 2026-04-26
**Goal**: Close the 3× train/val gap that prevented Exp 22 from matching Exp 18's 0.0014, and push float val_loss toward Model 5a's 0.000682.

**Focus**: Float accuracy only. Per the updated project goal, quantization is Phase 2 — do not run the TFLite conversion step or report quantized metrics until float val_loss ≤ 0.000682.

**Root cause of Exp 22 failure**: Overfitting. train_loss 6.6e-4 vs val_loss 0.0020 (3× gap). Dropout(0.3) on the 512-dim concatenated head is too weak. The model memorises training sequences (strong diurnal + solar patterns in training split) but fails to generalise. LR scheduling is now correct, so the remaining levers are regularisation and architecture capacity.

**Changes from Exp 22**:
1. **Dropout: 0.3 → 0.5** — primary fix for the train/val gap. Applied to the 512-dim concatenated representation before the first dense layer.
2. **L2 regularisation on dense layers** — `kernel_regularizer=l2(1e-4)` on `Dense(192)` and `Dense(96)`. Penalises large weights in the head, complementing dropout.
3. **Early stopping patience: 20 → 25** — gives the model more time after LR reductions to find a lower val_loss minimum. With patience=20 in Exp 22, the run stopped at epoch 57 (best epoch 33 + 20 = 53, plus a few more). Patience=25 extends the window.
4. **Skip quantization step** — do not run TFLite conversion or report quantized MAE. Float val_loss is the only success criterion.

**Architecture**: identical to Exp 18/22 — no changes
- Main branch: `Conv1D(96 filters, dilation=[1..64])` on all features → `(180, 96)`
- Temp branch: `Conv1D(32 filters, dilation=[1..64])` on `temperature` + `temp_delta_1` → `(180, 32)`
- Temporal extraction at t=0/−30/−60/−120 from each branch
- Dense head input: `(4×96) + (4×32) = 512-dim`
- Dense head: `Dense(192, l2)` → `Dropout(0.5)` → `Dense(96, l2)` → 3 output heads

**Training config**:
- Learning rate: 1e-4 (same starting point)
- Loss: MSE
- Batch size: 512
- Early stopping: patience=**25** (up from 20)
- ReduceLRCallback: factor=0.5, patience=8, min_lr=1e-7 (unchanged)
- Epochs: 100
- Mixed precision: disabled (unchanged from Exp 22)
- LR scheduling: working correctly (unchanged from Exp 22)

**Expected outcome**:
- Train/val gap should close from 3× to ≤ 1.5× with Dropout(0.5) + L2
- val_loss target: ≤ 0.0014 (match Exp 18), ideally push below toward 0.000682
- Temperature feature importance should remain competitive (architecture unchanged)
- No quantization results reported

**Success criteria**: float val_loss ≤ 0.0014 with train/val gap ≤ 2×

**Results** (stopped by training watchdog at epoch 76):
- Best val_loss: **0.0027** — regression from Exp 18's 0.0014 and Exp 22's 0.0020
- val_mae: **0.0104**
- LR schedule confirmed: 1e-4 → 5e-5 → 2.5e-5 (reduced at epoch 72 via ReduceLRCallback)
- Val_loss in final epochs (72–76): 0.0046 → 0.0043 → **0.0039** → 0.0043 → 0.0044 — oscillating, not improving
- Train loss at epoch 76: ~5.15e-4 vs val_loss ~0.0044 → **train/val gap ~8.5×** (worsened from Exp 22's 3×)
- Quantized MAE (ran despite Phase 2 skip plan): 2.20°C / 2.16°C / 1.46°C — broken quantization
- Quantized model size: 394.54 KB

Feature importance (ranked):
- Top: `time_of_day_cos` (0.0190), `illuminance` (0.0182), `solar_radiation` (0.0177), `relative_humidity` (0.0173)
- Bottom: `temperature` (0.0149, 18th/19), `temp_delta_1` (0.0147, 19th/19) — temperature still near last

**Outcome**: ❌ **FAILED** — val_loss 0.0027 vs target 0.0014; train/val gap 8.5× vs target ≤ 2× (both criteria missed, gap worsened despite stronger regularization)

**Analysis**:
- **Dropout(0.5) + L2 made overfitting worse, not better**: The train/val gap increased from 3× (Exp 22) to 8.5× (Exp 23). Stronger dropout in the dense head is causing the model to underfit during training (lower train loss) while the validation curve diverges further — the opposite of what was intended.
- **Best val_loss of 0.0027 was achieved early in training** (before the logged epochs 72–76), meaning the model never meaningfully improved over its initial fit before oscillation set in.
- **Temperature continues to rank last**: 18th–19th out of 19 features. The 3:1 filter ratio (96 main : 32 temp branch) is systematically diluting the temperature signal. This is now a persistent failure across Exp 18, 22, and 23.
- **Val_loss oscillation persists**: High variance in val_loss (range 0.0039–0.0046 in epochs 72–76) suggests the learning rate and/or architecture are fundamentally mismatched with the validation distribution.
- **Watchdog stop vs early stopping**: Training was stopped by the run_with_restart.py watchdog (not the Keras EarlyStopping callback), suggesting training hung or exceeded a time limit rather than converging.
- **Core conclusion**: Regularisation strength is not the root problem. The architecture itself — specifically the extreme imbalance between the main and temp branches — is preventing the model from learning generalizable temperature dynamics. Exp 18–23 have all failed with this same 96:32 filter ratio.

**Implications for next experiment**:
- Rebalance branch capacity: reduce main branch (96 → 64 filters) and increase temp branch (32 → 64 filters) to give temperature equal representation
- Or: simplify to a single shared branch and rely on the sequence to learn temporal patterns without the explicit branch split
- Revisit the dense head size: 512-dim concatenated head may be too large regardless of regularization strength — consider reducing to 256 or 384
- Consider longer patience for ReduceLRCallback (8 → 12) to allow val_loss to stabilize before reducing

---

## Comparison with Model 5a

| Metric | Model 5a | Exp 17 | Exp 18 | Exp 19 | Exp 20 | Exp 21 | Exp 22 | Exp 23 |
|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| val_loss | 0.000682 | 0.0039 | **0.0014** | 0.0015 ❌ | Cancelled | 0.0017 ❌ | 0.0020 ❌ | 0.0027 ❌ |
| val_mae | 0.00445 | 0.0133 | 0.0060 | — | — | — | — | 0.0104 |
| Best epoch | 97 | 6 | 53 | 31 | — | 60 | 33 | ~early |
| Model size (quant) | 788 KB | 221 KB | 395 KB | — | — | 587 KB | — | 395 KB |
| Quantized MAE 1hr (°C) | — | 1.90 ❌ | 0.55 ❌ | not tested | — | not tested | — | 2.20 ❌ |
| Top feature importance | temp_lag120 | **temperature ✅** | time_of_day_cos | — | — | — | time_of_day_cos | time_of_day_cos |
| temperature rank | #1 | **#1 ✅** | 18th/19 | — | — | — | last | 18th/19 |
| Architecture | Dense | Dual: main(64)+temp(32) | Dual: main(96)+temp(32) | Dual: main(96)+temp(64) | Dual: main(96)+temp(32) | Dual: main(96)+temp(32) + MSB | Dual: main(96)+temp(32) | Dual: main(96)+temp(32) |
| Dropout | — | 0.3 | 0.3 | 0.4 | 0.3 | 0.3 | 0.3 | **0.5** |
| L2 regularization | — | No | No | No | No | No | No | **Yes (1e-4)** |
| Dense head input dim | — | 384 | 512 | 640 | 512 | 512 | 512 | 512 |
| LR scheduling working | Yes | No ❌ | No ❌ | No ❌ | — | No ❌ | Yes ✅ | Yes ✅ |
| train/val gap | Low | Low | Low | High ❌ | — | ~9× ❌ | ~3× ❌ | **~8.5× ❌** |
| Pre-computed lags | Yes | No | No | No | No | No | No | No |
| Edge TPU viable (quant) | Yes | No ❌ | No ❌ | No ❌ | — | No ❌ | No ❌ | No ❌ |

---

## Experiment 24: Rebalance Branch Capacity — Equal main/temp Filters + Smaller Dense Head

**Date**: 2026-04-28
**Goal**: Fix the persistent temperature-last-in-feature-importance failure by giving the temperature branch equal representation in the concatenated head. Reduce the dense head to lower overfitting without relying on aggressive dropout or L2.

**Root cause of Exp 18–23 failure (structural, not tunable)**: The 96:32 main/temp filter ratio means the concatenated temporal representation is 384 main features vs 128 temp features (4 taps × each branch). Temperature only contributes 25% of the 512-dim input to the dense head. No amount of dropout, L2, or LR tuning can overcome a structural signal imbalance — the dense head can only work with what the branches provide. Temperature will keep ranking last until the branches are balanced.

Additionally, Exp 22–23 showed a persistent train/val gap (3× → 8.5×) that worsened with stronger regularization. The dense head at 512→192→96 has too many parameters relative to the information content. A smaller head with moderate dropout is a more principled fix than extreme dropout on an oversized head.

**Changes from Exp 23**:
1. **Rebalance branch filters: main(96→64), temp(32→64)** — equal 64-filter capacity for both branches. Concatenated head input: (4 taps × 64 main) + (4 taps × 64 temp) = 256 + 256 = 512-dim. Temperature now contributes 50% of the representation instead of 25%.
2. **Reduce dense head: Dense(192)→Dense(96) → Dense(128)→Dense(64)** — smaller head with fewer parameters to reduce structural overfitting. Total dense params roughly halved.
3. **Revert dropout: 0.5 → 0.3** — Exp 23 proved 0.5 worsened the gap; 0.3 is the correct level for this architecture.
4. **Remove L2 regularization** — Exp 23 showed L2 on the dense layers added no benefit and contributed to the gap widening.
5. **Skip quantization step** — Phase 2 only after float val_loss ≤ 0.000682.

**Architecture**:
- Main branch: `Conv1D(64 filters, dilation=[1..64])` on all features → GlobalAvgPool at t=0/−30/−60/−120 → 256-dim
- Temp branch: `Conv1D(64 filters, dilation=[1..64])` on `temperature` + `temp_delta_1` → GlobalAvgPool at t=0/−30/−60/−120 → 256-dim
- Dense head input: `(4×64) + (4×64) = 512-dim` (same total, but 50/50 split)
- Dense head: `Dense(128, relu)` → `Dropout(0.3)` → `Dense(64, relu)` → 3 output heads

**Training config**: identical to Exp 22/23 — no changes
- Learning rate: 1e-4
- Loss: MSE
- Batch size: 512
- Early stopping: patience=25
- ReduceLRCallback: factor=0.5, patience=8, min_lr=1e-7
- Epochs: 100
- Mixed precision: disabled
- LR scheduling: confirmed working (float32)

**Expected outcome**:
- Temperature should rise from 18th/19 toward the top 5 in feature importance — equal branch capacity removes the structural suppression
- Train/val gap should narrow: smaller dense head + Dropout(0.3) has fewer parameters to memorize
- val_loss target: ≤ 0.0014 (match Exp 18), pushing toward 0.000682

**Success criteria**: float val_loss ≤ 0.0014 with temperature in top 10 feature importance

**Results** (best epoch 46, watchdog stopped at epoch 66):
- Best val_loss: **0.001343** — new Conv1D record, beats Exp 18's 0.0014
- val_mae: 0.006052 — matches Exp 18's 0.0060
- Model size: **306.89 KB** (quantized) — smaller than Exp 18's 394.54 KB (64-filter main vs 96)
- LR at stop: reduced to 1.25e-5 at epoch 62 (1e-4 → 5e-5 → 2.5e-5 → 1.25e-5); val_loss oscillating 0.0015–0.0017 in epochs 62–66 with no improvement toward best
- Watchdog stopped at epoch 66; early stopping with patience=25 would have triggered at epoch 71 (46+25)
- **Quantized MAE: 1hr=0.61°C, 2hr=1.16°C, 3hr=2.63°C** — broken (ran despite Phase 2 skip plan)
- Quantized outputs near-constant (same semi-constant pattern as Exp 15)
- Input quantization: scale=0.003921 (≈ 1/255) — full INT8 range used ✅ (same as Exp 17+)

Feature importance (ranked):
- Top: `time_of_day_sin2` (0.0231), `time_of_day_cos` (0.0216), `solar_radiation` (0.0214), `illuminance` (0.0207), `wind_direction_cos` (0.0202)
- Bottom: `temperature` (0.0179, 17th/19), `time_of_day_sin` (0.0170, 18th), `temp_delta_1` (0.0147, 19th/last)
- Distribution very uniform: range 0.0147–0.0231 (~1.6× spread), tightest across all experiments

**Success criteria assessment**:
- ✅ float val_loss ≤ 0.0014: **0.001343** passes
- ❌ temperature in top 10: **17th/19** — still near last

**Outcome**: ⚠️ **PARTIAL** — Best float accuracy of any Conv1D experiment (0.001343), but temperature importance target missed. Still 2× short of Model 5a's 0.000682.

**Analysis**:
- Equal branch capacity (64:64) achieved the best float result yet — the reduced main branch (96→64) cut overfitting while the equal temp branch maintained representation. The smaller dense head (192→128, 96→64) reduced parameters without harming accuracy.
- **Temperature still ranks 17th/19** — the equal-capacity rebalance made the overall distribution more uniform (tightest spread yet, only 1.6× from top to bottom), but this appears to reflect the model distributing attention evenly rather than the temp branch becoming dominant. Time-of-day encodings continue to rank highest because they provide a strong, consistent proxy for the diurnal temperature cycle.
- **Feature importance is now so uniform** (0.0147–0.0231) that the ranking itself may be less meaningful — the model is using all features roughly equally, which is arguably correct behavior.
- **The 2× gap to Model 5a (0.001343 vs 0.000682)** likely reflects the fundamental limitation of inferring temperature dynamics purely from the sequence vs Model 5a's explicit lag columns (`temp_lag120`). No amount of branch rebalancing recovers that direct positional signal.
- **Quantized model near-constant**: outputs are semi-constant (1hr: all samples output 0.098, 2hr: only 2 distinct values). Input quantization is correct (scale=0.003921, full INT8 range), confirming the root cause is intermediate Conv1D activation quantization, not input scaling. This has been the failure mode since Exp 12 and requires QAT to fix.
- **Next direction**: With float accuracy plateauing around 0.0013–0.0014, the question is whether any Conv1D architecture without pre-computed lags can close the remaining 2× gap, or whether a different approach (e.g., Conv2D, explicit skip connections from the raw input at key offsets, or reintroducing limited lag features) is needed.

---

## Experiment 25: Quantization-Aware Training (QAT) Fine-Tuning

**Date**: 2026-04-29
**Goal**: Fix the persistent INT8 quantization failure (near-constant outputs, Exp 12–24) by inserting fake-quantization nodes during training so the model learns weights that are inherently INT8-robust.

**Root cause of all prior quantization failures (Exp 12–24)**:
Post-training quantization (PTQ) calibration fails because intermediate Conv1D activations have large, unbounded dynamic ranges. The representative dataset cannot cover the full range, so TFLite computes wrong calibration scales and the quantized model collapses to near-constant outputs. ReLU6 attempts (Exp 13) only bounded the residual Add path, not the Conv activations themselves. The input quantization was fixed in Exp 17 (scale=0.003921, full INT8 range) but intermediate activations remain the root cause.

**QAT approach (tensorflow-model-optimization 0.8.0)**:
- `tfmot.quantization.keras.quantize_model` inserts fake-quantization nodes (simulated INT8 rounding via straight-through estimator) into the forward pass during training
- Model learns INT8-robust weights: activations stay within a range that INT8 can represent accurately
- TFLite conversion uses the learned per-layer quantization scales — **no representative dataset needed**
- The fake-quant nodes simulate both the round-to-INT8 and the clamp-to-[min,max] operation during backprop

**Implementation**:
- `quantize_annotate_model` with a clone function to skip `SliceTimestep`/`SliceFeatures` (custom tensor-slicing ops with no weights; QuantizeWrapper not needed)
- Float weights from Exp 24's `best_model.weights.h5` (epoch 46, val_loss=0.001343) are loaded before QAT wrapping and copied to the QAT model via `annotated.set_weights(float_model.get_weights())`
- QAT compilation + fine-tuning at LR=1e-5 (10× lower than float training to preserve accuracy while adapting for quantization)

**Architecture**: Identical to Exp 24 — 64:64 equal branches, Dense(128→64), 512-dim head
**Training config**:
- Learning rate: **1e-5** (QAT fine-tuning)
- Loss: MSE
- Batch size: 512
- Early stopping: patience=**15** (shorter, fine-tuning run)
- ReduceLRCallback: factor=0.5, patience=8, min_lr=1e-8
- Max epochs: **50** (fine-tuning from epoch 0)
- Starting weights: Exp 24 best checkpoint (val_loss=0.001343)

**TFLite conversion**: `TFLiteConverter.from_keras_model(qat_model)` — no representative dataset; quantization scales are embedded in the QAT model's fake-quant variables

**Success criteria**: Quantized MAE ≤ 0.5°C / 1.0°C / 2.0°C (meaningful improvement over Exp 24's 0.61/1.16/2.63) while keeping float val_loss ≤ 0.001343

**Results** (best epoch 17, watchdog stopped at epoch 100):
- Best val_loss: **0.0015** — regression from Exp 24's 0.001343; QAT fine-tuning degraded float accuracy
- val_mae: 0.0066 — slightly worse than Exp 24's 0.006052
- Best epoch: **17** — model adapted to fake-quant constraints very quickly, then val_loss oscillated at 0.0016–0.0017 for the remaining 83 epochs without recovering
- LR at stop: reduced to 1.25e-5 by epoch 98 (1e-4 → 5e-5 → 2.5e-5 → 1.25e-5); val_loss oscillating ~0.0016–0.0017 in final epochs
- Model size: **306.89 KB** (identical to Exp 24 — QAT doesn't change architecture)
- **Quantized MAE: 1hr=0.73°C, 2hr=1.12°C, 3hr=1.88°C**
  - 1hr: 0.61 → 0.73°C — worse than Exp 24 PTQ ❌
  - 2hr: 1.16 → 1.12°C — marginal improvement ✅
  - 3hr: 2.63 → 1.88°C — meaningful improvement ✅
- **Quantized outputs still near-constant**: 5-sample probe shows only 2 distinct values per output head (diff_1hr: [0.092,0.092,0.092,0.092,0.088]; diff_2hr: [0.100,0.100,0.100,0.100,0.093]; diff_3hr: [0.085,0.085,0.085,0.085,0.073]). QAT did NOT fix the activation collapse.

Feature importance (ranked):
- Top: `time_of_day_cos` (0.0211), `time_of_day_sin2` (0.0210), `solar_radiation` (0.0204), `illuminance` (0.0191), `wind_direction_cos` (0.0189)
- Bottom: `temperature` (0.0173, 17th/19), `time_of_day_sin` (0.0160, 18th), `temp_delta_1` (0.0151, 19th/last)
- Distribution virtually identical to Exp 24 — QAT fine-tuning did not shift feature importance

**Success criteria assessment**:
- ❌ Quantized MAE ≤ 0.5°C / 1.0°C / 2.0°C: **0.73 / 1.12 / 1.88°C** — all three thresholds missed
- ❌ float val_loss ≤ 0.001343: **0.0015** — degraded from starting point

**Outcome**: ❌ **FAILED** — QAT did not fix quantization. Near-constant outputs persist in the INT8 model despite fake-quant training. Float accuracy regressed.

**Analysis**:
- **Root cause of QAT failure**: The `SliceTimestep`/`SliceFeatures` custom ops were excluded from QAT wrapping (no weights, so QuantizeWrapper skipped). These ops extract the temporal tap representations and feed them into the dense head — their output activations have unbounded range that PTQ can't calibrate and QAT doesn't constrain. The fake-quant nodes in the Conv1D layers are not sufficient; the bottleneck is the tensor-slicing path.
- **Best epoch 17 then plateau**: The model adapted quickly to the fake-quant constraints at LR=1e-5 but had no room to improve further. The LR schedule reduced every 8 epochs of no improvement, eventually stalling at 1.25e-5 for the last ~40 epochs with no recovery.
- **Feature importance unchanged**: The diurnal/solar dominance and temperature-last ranking are structural — QAT fine-tuning cannot alter what signals the Conv1D branch has learned to extract.
- **Quantization failure is architectural**: The Conv1D + SliceTimestep architecture creates activation ranges that neither PTQ nor QAT can tame without wrapping the slice ops. Fixing this would require a fundamentally different temporal extraction mechanism.
- **Path forward**: Abandon QAT for this architecture. The quantization failure is not solvable with the current Conv1D + SliceTimestep approach. The more impactful problem is the 2× float accuracy gap vs Model 5a, caused by the absence of an explicit temperature anchor. **Experiment 26 will add `temp_lag60` and `temp_lag120` as explicit input features**, restoring the direct temperature signal that Model 5a's #1 feature provides.

---

## Experiment 26: Explicit Lag Features (temp_lag60 + temp_lag120)

**Date**: 2026-04-30
**Goal**: Eliminate the phase lag / time-delay observed in Exp 24/25 predictions by restoring explicit temperature anchor features, and close the 2× float accuracy gap vs Model 5a (0.001343 → 0.000682).

**Root cause of phase lag (Exp 17–25)**:
All Conv1D experiments (Exp 17–25) have shown `temperature` ranking 17th–19th in feature importance, while `time_of_day` and `solar_radiation` dominate. The model predicts from the *expected* diurnal curve rather than from actual observed temperatures, producing predictions that visually lag real temperature transitions. Model 5a's #1 feature (`temp_lag120`) provides an explicit anchor to actual temperature 2 hours ago — the Conv1D experiments proved empirically that this cannot be learned implicitly from the raw sequence when diurnal signals are present.

**Changes from Exp 24** (base architecture — skip Exp 25 QAT weights, they degraded accuracy):
1. **Add `temp_lag60` and `temp_lag120` as explicit input features** — direct temperature anchors at 1hr and 2hr prior, matching Model 5a's key signals
2. **Return to float training** — PTQ or QAT is a Phase 2 concern; fix float accuracy first
3. **Starting weights**: fresh random init (not Exp 24/25 checkpoints — lag features change the input dimension)
4. **SPEC update**: the no-precomputed-lags goal is retired; 25 experiments proved it is not achievable at Model 5a accuracy with Conv1D + Edge TPU constraint

**Architecture**: Identical to Exp 24 — Dual: main(64)+temp(64), Dense(128→64), 512-dim head
- Input features: 19 → **21** (adds `temp_lag60`, `temp_lag120`)

**Training config**: identical to Exp 22/24
- Learning rate: 1e-4
- Loss: MSE
- Batch size: 512
- Early stopping: patience=25
- ReduceLRCallback: factor=0.5, patience=8, min_lr=1e-7
- Epochs: 100

**Expected outcome**:
- `temp_lag120` and `temp_lag60` should rank near the top of feature importance (as in Model 5a)
- val_loss should drop substantially toward or below 0.000682
- Phase lag in predictions should disappear — model anchors on actual temperatures, not diurnal curve

**Success criteria**: float val_loss ≤ 0.000682 (match or beat Model 5a) with `temperature`/`temp_lag120` in top 5 feature importance

**Results** (best epoch 25, watchdog stopped at epoch 50):
- Best val_loss: **0.0039** — no improvement over Exp 24's 0.001343; adding lag features degraded combined float accuracy
- val_mae: **0.0121** — worse than Exp 24's 0.00605
- Best epoch: **25** — peaked very early, then degraded steadily despite 4 LR reductions (1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6)
- Model size: **307.64 KB** (quantized) — identical to Exp 24/25
- Note: run was named `conv1d_exp25_run1` (naming bug in script — should be exp26)

Validation MAE in °C (best float model, epoch 25):
- diff_1hr: **0.01 °C**
- diff_2hr: **0.02 °C**
- diff_3hr: **0.03 °C**

Feature importance (all 21 features, ranked):
- **#1: `temp_lag60`** (0.0791) ✅ — explicit lag now dominates as intended
- #2: `illuminance` (0.0677), #3: `time_of_day_sin2` (0.0665), #4: `wind_lull` (0.0643), #5: `time_of_day_cos2` (0.0642)
- #19: `temp_lag120` (0.0586) ❌ — expected near top; 2hr anchor not being used effectively
- #20: `temp_delta_1` (0.0535), #21: `temperature` (0.0520, dead last) ❌
- Distribution: 0.0520–0.0791 (~1.5× spread); `temp_lag60` stands out clearly at top, but `temp_lag120` falls to near the bottom

Quantized TFLite validation (500 samples):
- diff_1hr: **2.14 °C** MAE — outputs constant (all samples → -0.231)
- diff_2hr: **6.74 °C** MAE — outputs constant (all samples → 0.38)
- diff_3hr: **6.91 °C** MAE — near-constant (slight variation)
- Same SliceTimestep activation collapse as Exp 12–25; quantization completely broken

**Success criteria assessment**:
- ❌ float val_loss ≤ 0.000682: **0.0039** — 5.7× short of target; regressed from Exp 24
- ❌ `temp_lag120` in top 5: **19th/21** — not used effectively by the model
- ✅ `temp_lag60` is #1 feature — explicit 1hr anchor working; phase lag likely improved

**Outcome**: ❌ **FAILED** — Float accuracy regressed vs Exp 24 (0.0039 vs 0.001343). The SliceTimestep quantization failure is unchanged. `temp_lag60` correctly emerged as the top feature (confirming explicit lags help), but `temp_lag120` ranked 19th — the architecture is not exploiting both anchors.

**Analysis**:
- **Float regression vs Exp 24**: The same architecture with 2 extra input features produced a worse combined val_loss (0.0039 vs 0.001343). The most likely explanation is that Exp 24's model was learning the diurnal curve (a well-structured, easy signal) — adding explicit lag features forces the model to reason about actual temperature dynamics, which is a harder problem that the same architecture and capacity cannot solve as effectively at this training budget.
- **Early peak at epoch 25**: The model found its best at epoch 25 and degraded for the remaining 25 epochs, triggering early stopping. Four consecutive LR reductions failed to recover, suggesting the architecture had saturated in this formulation, not just the optimizer.
- **`temp_lag120` ranked 19th**: The 1hr lag (`temp_lag60`) and the 2hr lag (`temp_lag120`) provide redundant temperature anchoring — the model may have learned to rely entirely on `temp_lag60` and treat `temp_lag120` as noise given both encode similar information. Alternatively, the dual-branch architecture routes both lags through the shared `temp` branch where one dominates.
- **Quantization**: Same near-constant collapse as Exp 12–25. The root cause (unbounded SliceTimestep activations) is architectural — no training approach fixes it within this architecture.
- **Path forward**: The Conv1D + SliceTimestep architecture has hit a ceiling. Both the float accuracy problem (Exp 24 with diurnal cheating was the actual best at 0.001343, still 2× from target) and the quantization failure are structural. **Experiment 27 will switch to Conv2D** — reshape input to (180, 21, 1), use Conv2D + GlobalAveragePooling2D for temporal aggregation, eliminating SliceTimestep entirely. Conv2D + GlobalAveragePooling is the standard Edge TPU-proven pattern (MobileNet/EfficientNet) with no custom ops and well-characterized INT8 quantization behavior.

---

## Comparison with Model 5a

| Metric | Model 5a | Exp 17 | Exp 18 | Exp 19 | Exp 20 | Exp 21 | Exp 22 | Exp 23 | Exp 24 | Exp 25 (QAT) | Exp 26 | Exp 27 | Exp 28 |
|--------|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------------|--------|--------|--------|
| val_loss | 0.000682 | 0.0039 | **0.0014** | 0.0015 ❌ | Cancelled | 0.0017 ❌ | 0.0020 ❌ | 0.0027 ❌ | **0.001343** ✅ | 0.0015 ❌ | 0.0039 ❌ | 0.0027 ❌ | 0.0028 ❌ |
| val_mae | 0.00445 | 0.0133 | 0.0060 | — | — | — | — | 0.0104 | **0.00605** | 0.0066 | 0.0121 ❌ | 0.013 avg | 0.0096 |
| Best epoch | 97 | 6 | 53 | 31 | — | 60 | 33 | ~early | 46 | **17** | **25** | **89** | ~61 |
| Model size (quant) | 788 KB | 221 KB | 395 KB | — | — | 587 KB | — | 395 KB | **307 KB** | 307 KB | 307 KB | **188 KB** | **193 KB** |
| Quantized MAE 1hr (°C) | — | 1.90 ❌ | 0.55 ❌ | not tested | — | not tested | — | 2.20 ❌ | 0.61 ❌ | **0.73 ❌** | 2.14 ❌ | 1.57 ⚠️ | **1.12 ⚠️** |
| Quantized MAE 2hr (°C) | — | — | — | — | — | — | — | — | — | — | — | 2.21 ❌ | **1.63 ⚠️** |
| Quantized MAE 3hr (°C) | — | — | — | — | — | — | — | — | — | — | — | 2.63 ❌ | **2.01 ⚠️** |
| Top feature importance | temp_lag120 | **temperature ✅** | time_of_day_cos | — | — | — | time_of_day_cos | time_of_day_cos | time_of_day_sin2 | time_of_day_cos | **temp_lag60 ✅** | time_of_day_sin2 | **temperature ✅** |
| temperature rank | #1 | **#1 ✅** | 18th/19 | — | — | — | last | 18th/19 | 17th/19 | 17th/19 | 21st/21 ❌ | #2 ✅ | **#1 ✅** |
| temp_lag60 rank | — | — | — | — | — | — | — | — | — | — | #1 ✅ | #19 ❌ | **#6 ✅** |
| temp_lag120 rank | — | — | — | — | — | — | — | — | — | — | — | #18 ❌ | **#8 ✅** |
| Architecture | Dense | Dual: main(64)+temp(32) | Dual: main(96)+temp(32) | Dual: main(96)+temp(64) | Dual: main(96)+temp(32) | Dual: main(96)+temp(32) + MSB | Dual: main(96)+temp(32) | Dual: main(96)+temp(32) | **Dual: main(64)+temp(64)** | QAT on Exp 24 | Dual: main(64)+temp(64) | **Conv2D+GAP** | **Conv2D+GAP+skip** |
| Input features | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | **21** | **21** | **21** |
| Pre-computed lags | Yes | No | No | No | No | No | No | No | No | No | **Yes (lag60, lag120)** | **Yes (lag60, lag120)** | **Yes (lag60, lag120)** |
| Edge TPU viable (quant) | Yes | No ❌ | No ❌ | No ❌ | — | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | **Yes ✅** | **Yes ✅** |

---

## Experiment 27: Conv2D Architecture

**Date**: 2026-05-01
**Goal**: Eliminate the SliceTimestep quantization failure by switching to a Conv2D architecture with no custom ops, while retaining the explicit lag features (temp_lag60, temp_lag120) that proved effective in Exp 26.

**Root cause of all Conv1D quantization failures (Exp 12–26)**:
`SliceTimestep`/`SliceFeatures` custom ops extract temporal tap representations from the Conv1D output and feed them into the dense head. These ops have unbounded output activation ranges that PTQ calibration cannot estimate and that QAT fake-quant nodes cannot constrain (they were excluded from QAT wrapping in Exp 25 because they have no weights). The failure is architectural — no training strategy fixes it within the Conv1D + SliceTimestep formulation.

**Changes from Exp 26**:
1. **Replace Conv1D + SliceTimestep with Conv2D + GlobalAveragePooling2D** — all supported Edge TPU ops, no custom ops, proven INT8 quantization pattern (MobileNet/EfficientNet family)
2. **Retain explicit lag features** — `temp_lag60` and `temp_lag120` stay as inputs (21 features total); Exp 26 confirmed `temp_lag60` correctly anchors predictions when explicit
3. **Retain 3-output multi-task head** — diff_1hr, diff_2hr, diff_3hr

**Architecture**:
- Input: `(180, 21)` → Reshape to `(180, 21, 1)` — treat as single-channel "image" (time × features)
- Conv2D blocks with `kernel=(3,1)` to slide along the time axis
- BatchNorm + ReLU6 after each conv (bounded activations, Edge TPU-compatible)
- GlobalAveragePooling2D for temporal aggregation (replaces SliceTimestep entirely)
- Dense head → 3 output heads (diff_1hr, diff_2hr, diff_3hr)

**Training config**:
- Learning rate: 1e-4
- Loss: MSE
- Batch size: 512
- Early stopping: patience=25
- ReduceLRCallback: factor=0.5, patience=8, min_lr=1e-7
- Epochs: 100

**Success criteria**: float val_loss ≤ 0.000682 (match or beat Model 5a) **and** quantized MAE ≤ 0.5°C / 1.0°C / 2.0°C (meaningful, not collapsed)

**Success criteria assessment**:
- ❌ float val_loss ≤ 0.000682: **0.0027** — exceeds target, but loss scales differ across architectures (Conv2D trains 3 separate MSE heads in °C-difference space vs Model 5a's single normalized output)
- ✅ Quantized outputs meaningful (not collapsed): **1.57°C / 2.21°C / 2.63°C** — all three outputs vary per sample
- ✅ **First successful PTQ** in any Model 5b experiment (Exp 12–27)
- ❌ Quantized MAE ≤ 0.5°C / 1.0°C / 2.0°C: actual 1.57 / 2.21 / 2.63 — meaningful but all exceed target

**Results**:
- Best val_loss: **0.0027** (epoch ~89; EarlyStopping with patience=25 restored best weights)
- Val MAE (float): diff_1hr: **0.01°C**, diff_2hr: **0.01°C**, diff_3hr: **0.02°C**
- Quantized model size: **188.58 KB** (smallest yet; Exp 24–26 were all 307 KB)
- Quantized MAE (TFLite INT8, 500 samples): diff_1hr: **1.57°C**, diff_2hr: **2.21°C**, diff_3hr: **2.63°C** — outputs vary per sample ✅
- Training epoch times: stable **~922–940s** throughout 100 epochs after training stability fix

**Feature importance** (all 21 features, ranked):
- **#1: `time_of_day_sin2`** (0.093)
- **#2: `temperature`** (0.088) ✅ — dramatic improvement from 17th–21st in all Conv1D experiments
- #3–17: mid-tier features (range ~0.065–0.085)
- **#18: `temp_lag120`** (0.063) ❌ — expected near top; diluted 180× by GlobalAveragePooling2D
- **#19: `temp_lag60`** (0.063) ❌ — expected near top; diluted 180× by GlobalAveragePooling2D
- **#20: `temp_delta_1`** (0.056), **#21: `time_of_day_cos`** (0.053)
- Distribution: 0.053–0.093 (~1.75× spread); notably flatter than Conv1D experiments

**Training stability fix**:
Epoch-boundary hang on macOS Metal: `CheckpointValidationCallback.on_epoch_end` called `model.load_weights()` on the live training model between epochs, writing GPU variables during Metal's command buffer lifecycle and corrupting state before the next epoch's `tf.while_loop` (with `steps_per_execution=20`). Conv2D + BatchNorm has additional GPU state (running mean/variance) making it more sensitive than Conv1D. Fix: replaced all GPU operations in checkpoint validation with a pure CPU h5py file-readability check (zero GPU operations). All hangs eliminated; epoch times stabilized to ~922–940s.

**Outcome**: ⚠️ **PARTIAL SUCCESS** — First successful PTQ across all Model 5b experiments. Conv2D + GlobalAveragePooling2D quantizes correctly with meaningful, non-collapsed outputs. Float accuracy excellent (0.01–0.02°C MAE on temperature differences). Quantized accuracy (1.57–2.63°C) above the success-criteria targets. The Edge TPU deployment path is architecturally confirmed; closing the float→quantized gap and fixing the GAP/lag-feature dilution problem are the remaining challenges.

**Analysis**:
- **PTQ success**: Conv2D + GlobalAveragePooling2D uses only standard Edge TPU ops with bounded activations (ReLU6 + BatchNorm). No custom ops, no unbounded activation ranges. Same standard pattern as MobileNet/EfficientNet with well-characterized INT8 calibration behavior.
- **Temperature ranked #2**: Improvement from 17th–21st in all Conv1D experiments. Conv2D with `kernel=(3,1)` learns local temporal gradients per feature channel independently before pooling — current temperature receives direct gradient signal. In Conv1D, the shared temporal representation with SliceTimestep extraction suppressed the current-timestep feature.
- **Lag features ranked 18th–19th (GAP dilution — the key problem)**: `GlobalAveragePooling2D` averages all 180 timesteps equally. `temp_lag60` and `temp_lag120` occupy one position each in a 180-timestep channel; their anchor value is diluted 180× by surrounding timesteps that carry no lag information. The dense head receives only a time-averaged representation, never the actual lag value. This is the fundamental GAP/explicit-anchor incompatibility.
- **Val_loss oscillation**: Train loss stable ~0.0022–0.0023 throughout; val_loss swings 0.0042–0.0374 in consecutive epochs. Generalization gap — model fits training distribution tightly but generalizes noisily to validation windows.
- **Float→quantized accuracy gap**: Float 0.01–0.02°C vs quantized 1.57–2.63°C. INT8 quantization error accumulates across 4 Conv2D blocks with compounding rounding errors per layer. QAT or per-channel quantization could reduce this gap.

---

## Experiment 28: Conv2D + Last-Timestep Skip Connection

**Date**: 2026-05-01
**Goal**: Fix the GAP-dilution problem from Exp 27: `temp_lag60` and `temp_lag120` ranked 18th–19th because GlobalAveragePooling2D dilutes single-timestep anchor values 180×. Add a skip connection that extracts `temperature`, `temp_lag60`, and `temp_lag120` from the last timestep (t=0, the most recent reading) and routes them directly to the dense head alongside the GAP context vector.

**Root cause from Exp 27**:
`GlobalAveragePooling2D` is incompatible with features that carry meaningful value only at a single timestep. The explicit lag values `temp_lag60[t=0]` and `temp_lag120[t=0]` (the actual 1hr and 2hr anchor temperatures) are diluted 180× when averaged with 179 other timesteps where those columns carry no anchor information. The model correctly learned that current temperature is useful (#2 via Conv2D temporal gradients) but cannot exploit explicit lags through GAP averaging.

**Changes from Exp 27**:
1. **Add last-timestep skip connection** — extract `temperature`, `temp_lag60`, `temp_lag120` from `input[:, -1, [idx_temp, idx_lag60, idx_lag120]]` and route through a small Dense(16) sub-network directly to the concatenation layer, bypassing GlobalAveragePooling2D entirely
2. **Two-path architecture**: temporal context path (Conv2D → GAP → Dense(64)) + anchor path (last-timestep slice → Dense(16)) → Concatenate(80) → dense head
3. **Cosine decay LR schedule** to reduce val_loss oscillation (smooth decay replaces step-function ReduceLROnPlateau)
4. Retain all other Exp 27 settings: 21 features, same Conv2D block structure, BatchNorm + ReLU6, 3-output head, batch size 512

**Architecture**:
```
Input: (180, 21) → Reshape to (180, 21, 1)
  ├─ Conv2D path:  [Conv2D(32)→BN→ReLU6 → Conv2D(64)→BN→ReLU6 → ...] → GAP → Dense(64) → context
  └─ Skip path:   input[:, -1, [temp, lag60, lag120]] → Dense(16) → anchors
Concatenate([context, anchors])  →  Dense(32) → ReLU6 → Dense(3) outputs
```
All ops Edge TPU compatible: Slice/Gather on input tensor is a standard TFLite op.

**Training config**:
- Learning rate: cosine decay from 1e-4 → 1e-6 over 100 epochs
- Loss: MSE (per-head)
- Batch size: 512
- Early stopping: patience=25
- Epochs: 100

**Success criteria**:
- `temp_lag60` and `temp_lag120` both in top 5 feature importance (confirming skip connection is actively used)
- float val_loss ≤ 0.000682 (match or beat Model 5a)
- Quantized MAE improves on Exp 27: ≤ 1.0°C / 2.0°C / 2.5°C
- Edge TPU viable (quantized outputs vary per sample)

**Results**:
- Best val_loss: **0.0028** — plateaued epochs 91–100 with no movement; still 4.1× short of Model 5a
- Val MAE (float): diff_1hr: **0.01°C**, diff_2hr: **0.01°C**, diff_3hr: **0.02°C**
- Val MAE (normalized): **0.0096** combined
- Quantized model size: **193.26 KB**
- Quantized MAE (TFLite INT8, 500 samples): diff_1hr: **1.12°C**, diff_2hr: **1.63°C**, diff_3hr: **2.01°C**
- Training loss at epoch 100: ~3.53e-4; train/val gap: **~8×**
- Converged fully: val_loss unchanged from epoch ~91 onward; cosine LR reached ~1e-6

**Feature importance** (all 21 features, ranked):
- **#1: `temperature`** (0.3450) ✅ — jumped from #2 (0.088 in Exp 27); skip connection gave it dominant gradient signal
- **#2: `time_of_day_sin2`** (0.2247)
- **#3: `time_of_day_cos`** (0.2150)
- **#4: `time_of_day_sin`** (0.1290)
- **#5: `time_of_day_cos2`** (0.1235)
- **#6: `temp_lag60`** (0.1212) ✅ — up from #19 (0.063 in Exp 27); skip connection working
- **#7: `uv`** (0.1096)
- **#8: `temp_lag120`** (0.1016) ✅ — up from #18 (0.063 in Exp 27); skip connection working
- #9: `solar_radiation` (0.1005), #10: `relative_humidity` (0.0911)
- #11–21: illuminance, station_pressure, wind features, day_of_year, rain, temp_delta_1 (0.0869–0.0660)

**Success criteria assessment**:
- ⚠️ `temp_lag60` and `temp_lag120` in top 5: **missed** (#6 and #8), but dramatically improved from dead last (Exp 27: #18/#19); skip connection confirmed active
- ❌ float val_loss ≤ 0.000682: **0.0028** — 4.1× short; float accuracy identical to Exp 27 despite better feature routing
- ✅ Quantized MAE improves on Exp 27 (1.57/2.21/2.63): **1.12/1.63/2.01°C** — all three improved
- ✅ Edge TPU viable: outputs vary per sample

**Outcome**: ⚠️ **PARTIAL SUCCESS** — Skip connection fully resolved the GAP-dilution problem: temperature jumped to #1 importance (0.345 vs 0.088) and lag features moved from bottom-2 to top-10. Quantized accuracy improved on all three horizons vs Exp 27. However, float val_loss did not improve (0.0028 vs 0.0027) — the architecture correctly routes anchor features now, but the train/val gap (~8×) is structural and was not addressed. Float→quantized gap remains large (0.01°C float vs 1.12°C quantized).

**Analysis**:
- **Feature routing solved, accuracy wall unchanged**: The skip connection did exactly what it was designed to do — `temp_lag60`/`temp_lag120` are now exploited. Yet val_loss is identical to Exp 27. This suggests the GAP-dilution of lag features was not the primary bottleneck for float accuracy; the train/val generalization gap is.
- **Train/val gap of 8×**: Training loss ~3.5e-4, val_loss ~0.0028. The model fits training sequences well but generalizes noisily to validation. This is a regularization/capacity problem, not a feature-routing problem.
- **Float→quantized gap**: 0.01°C float vs 1.12°C quantized is ~100× degradation. INT8 rounding accumulates across 4 Conv2D blocks + dense head. Per-channel quantization or QAT could reduce this, but quantization is Phase 2.
- **Path forward**: The float accuracy ceiling (~0.0027–0.0028) has been hit by two consecutive Conv2D experiments with different architectures. To close the remaining 4× gap to Model 5a, the next lever is the train/val generalization gap — stronger regularization (higher dropout, L2), data augmentation, or a fundamentally different approach to reducing overfitting.

---

## Experiment 29: Slope Features in Skip Path + Input Tensor

**Date**: 2026-05-04
**Goal**: Close the remaining 4× float accuracy gap to Model 5a (0.000682) by adding explicit rolling-regression slope features — the single change responsible for Model 5a's 15× improvement over Model 5. Exp 28 proved the skip path correctly routes single-timestep anchor values; Exp 29 uses the same mechanism to deliver explicit trend signals that the Conv2D blocks are currently trying (and failing) to learn implicitly.

**Root cause of Exp 28 accuracy wall**:
The Conv2D path uses kernels of size 3, 7, and 15 timesteps to capture temporal trends, but those learned representations are (a) averaged away by GlobalAveragePooling2D and (b) shared across all 21 features with no temperature-specific gradient. The 8× train/val gap indicates the model is overfitting to implicit trend patterns in training data rather than learning stable, generalizable signals. Model 5a's breakthrough was replacing 1-step raw deltas with Numba-computed linear-regression slopes over 15–60 sample windows — stable, low-noise trend signals that generalize well.

**Changes from Exp 28**:
1. **Pre-compute 6 slope features** using the Numba rolling-regression function already in the script (currently commented out at lines 333–360), applied to both train and val data:
   - `temp_slope_15` — temperature slope over 15-minute window
   - `temp_slope_30` — temperature slope over 30-minute window
   - `temp_slope_60` — temperature slope over 60-minute window
   - `solar_slope_30` — solar radiation slope over 30-minute window
   - `humidity_slope_30` — relative humidity slope over 30-minute window
   - `pressure_slope_60` — station pressure slope over 60-minute window
2. **Add all 6 slope features to the input tensor** (n_features: 21 → 27) so the Conv2D path sees them for temporal context
3. **Expand the skip/anchor path** to extract t=0 values for all 9 anchor features (temp, lag60, lag120 + 6 slopes); expand anchor Dense from 16 → 32 units to accommodate the wider input
4. Retain all other Exp 28 settings: cosine decay LR (1e-4 → 1e-6 over 100 epochs), same Conv2D block structure (FILTERS=64, kernels 3/7/15/feat), BatchNorm + ReLU6, batch size 512, 3-output head

**Architecture**:
```
Input: (180, 27) → Reshape to (180, 27, 1)
  ├─ Conv2D path:  [Conv2D(64,k=3)→BN→ReLU6 → Conv2D(64,k=7)→BN→ReLU6
  │                 → Conv2D(64,k=15)→BN→ReLU6 → Conv2D(64,k=27)→BN→ReLU6]
  │                → GAP → Dense(64) → ReLU6 → context
  └─ Skip path:   input[:, -1, [temp, lag60, lag120,
                                temp_slope_15, temp_slope_30, temp_slope_60,
                                solar_slope_30, humidity_slope_30, pressure_slope_60]]
                  → Dense(32) → ReLU6 → anchors
Concatenate([context(64), anchors(32)])  →  Dense(32) → ReLU6 → Dense(3) outputs
```

**Training config**:
- Learning rate: cosine decay from 1e-4 → 1e-6 over 100 epochs (same as Exp 28)
- Loss: MSE (per-head)
- Batch size: 512
- Early stopping: patience=25
- Epochs: 100

**Success criteria**:
- `temp_slope_60`, `temp_slope_30`, or `temp_slope_15` appear in top 5 feature importance (confirming explicit slopes are actively used)
- float val_loss ≤ 0.001343 (beat the all-time Conv best from Exp 24)
- Ideally float val_loss approaches 0.000682 (Model 5a parity)
- Quantized MAE maintains or improves on Exp 28 (≤ 1.12°C / 1.63°C / 2.01°C)
- Edge TPU viable (quantized outputs vary per sample)

**Intermediate results (epochs 81–84, 2026-05-06)**:

| Epoch | train_loss | val_loss | val_1hr_mae | val_2hr_mae | val_3hr_mae | LR |
|-------|------------|----------|-------------|-------------|-------------|-----|
| 81 | 2.6457e-04 | 0.0028 | 0.0154 | 0.0159 | 0.0210 | 1.05e-5 |
| 82 | 2.6372e-04 | 0.0027 | 0.0153 | 0.0155 | 0.0182 | 9.56e-6 |
| 83 | 2.6385e-04 | 0.0026 | 0.0154 | 0.0157 | 0.0172 | 8.71e-6 |
| 84 | 2.5856e-04 | 0.0026 | 0.0154 | 0.0160 | 0.0172 | 7.89e-6 |

**Results (final, epoch 101, 2026-05-07)**:
- Best val_loss: **0.0026** — best epoch was 15; cosine LR reached ~1e-6 at epoch 100 with no further improvement
- Val MAE (float): diff_1hr: **0.01°C**, diff_2hr: **0.01°C**, diff_3hr: **0.02°C**
- Val MAE (normalized): **0.0084** combined
- Final train_loss: 2.4413e-04; train/val gap: **~10×**
- Quantized model size: **218.20 KB**
- Quantized MAE (TFLite INT8, 500 samples): diff_1hr: **0.82°C**, diff_2hr: **1.49°C**, diff_3hr: **1.63°C**

**Feature importance** (all 27 features, ranked):
- **#1: `time_of_day_sin2`** (0.1376)
- **#2: `time_of_day_cos`** (0.1154)
- **#3: `temp_lag60`** (0.1115) ✅ skip path working
- **#4: `uv`** (0.0982)
- **#5: `illuminance`** (0.0947)
- **#6: `temperature`** (0.0822) — dropped from #1 (0.345) in Exp 28; diluted by slope features
- **#7: `time_of_day_sin`** (0.0803)
- **#8: `wind_gust`** (0.0798)
- **#9: `relative_humidity`** (0.0783), **#10: `station_pressure`** (0.0779)
- **#11: `temp_lag120`** (0.0774), **#12: `temp_slope_60`** (0.0773), **#13: `solar_radiation`** (0.0770)
- **#14: `wind_direction_sin`** (0.0767), **#15: `wind_lull`** (0.0755)
- **#16–22**: `day_of_year_cos` (0.0749), `rain_accumulated` (0.0746), `day_of_year_sin` (0.0746), `humidity_slope_30` (0.0745), `solar_slope_30` (0.0743), `wind_direction_cos` (0.0739), `pressure_slope_60` (0.0734)
- **#23–26**: `temp_delta_1` (0.0727), `wind_avg` (0.0720), `temp_slope_30` (0.0715), `temp_slope_15` (0.0707)
- **#27: `time_of_day_cos2`** (0.0550)

**Success criteria assessment**:
- ❌ Slope features in top 5: `temp_slope_60` is #12 (0.077); all six slopes rank 12th–26th
- ❌ float val_loss ≤ 0.001343: **0.0026** — 1.9× short of Exp 24's Conv best; 3.8× short of Model 5a
- ✅ Quantized MAE ≤ Exp 28 thresholds (1.12/1.63/2.01): **0.82/1.49/1.63°C** — all three improved
- ⚠️ Edge TPU viable: diff_3hr outputs near-constant (-0.133 to -0.127) on 500 validation samples

**Outcome**: ⚠️ **PARTIAL SUCCESS** — Quantization improved on all three horizons vs Exp 28, and float accuracy improved marginally (0.0026 vs 0.0028). However, slope features did not land in the top 5 and did not break the float accuracy wall. The `temperature` feature importance collapsed from #1 (0.345 in Exp 28) to #6 (0.082) — the expanded skip-path anchor set and additional Conv2D feature channels are crowding out the raw temperature gradient that drove Exp 28's strongest signal.

**Analysis**:
- **Slopes are used, but not dominant**: All six slope features rank in the lower half of the 27-feature set (~0.07–0.08 importance each). The model uses them but treats them as weak corroborating signals rather than primary predictors. This contrasts with Model 5a where slope features were decisive.
- **Time-of-day still dominates**: `time_of_day_sin2` (#1, 0.138) and `time_of_day_cos` (#2, 0.115) lead the ranking. The diurnal signal is the strongest generalizable pattern in the validation set, which the model correctly exploits but which limits actual temperature-change accuracy.
- **Overfitting worsened**: Train/val gap widened from ~8× (Exp 28) to ~10× (Exp 29). Adding 6 features increased effective capacity without improving generalization. Best epoch at 15 confirms the model overfits almost immediately and the remaining 85 epochs only recover marginally.
- **`temperature` signal diluted**: In Exp 28, routing just three anchor features (temp, lag60, lag120) through the skip Dense(16) gave `temperature` a dominant gradient. In Exp 29, nine features compete through Dense(32), and the six new slope features introduce correlated temperature-trend signals that split the gradient, reducing each feature's individual importance score.
- **Path forward**: The float accuracy ceiling (~0.0026) has now been confirmed across three consecutive Conv2D experiments (Exp 27–29). The primary bottleneck is the ~10× train/val generalization gap, not feature engineering. The next lever must be regularization: L2 weight decay, higher dropout, or reduced filter count to force better generalization. Adding more features has now been tried twice (Exp 26 added lags, Exp 29 added slopes) without breaking the wall.

---

## Experiment 30: Dropout(0.3) Regularization on Context Vector

**Date**: 2026-05-07
**Goal**: Break the ~10× train/val generalization gap that has persisted across Exp 27–29 by adding explicit dropout regularization to the Conv2D temporal path.

**Root cause of the Exp 27–29 accuracy wall**:
Three consecutive Conv2D experiments have plateaued at val_loss ~0.0026–0.0028 with train loss ~2.5–2.6e-4 — a 10× gap. Feature engineering (lag features in Exp 26, slope features in Exp 29) improved quantized MAE marginally but did not move the float ceiling. The bottleneck is overfitting, not insufficient features. Exp 29's best epoch was epoch 15 of 100; the remaining 85 epochs provide no generalization improvement.

**Key insight from Model 5a**:
Model 5a (val_loss=0.000682) uses `Dropout(0.3)` after its first Dense(128) layer and achieves a train/val gap of ~2× rather than 10×. It achieves this **without** slope features — just lag features, the flatten→Dense architecture, and dropout. Dropout is the single largest architectural difference between Model 5a and the current Conv2D stack.

**Change from Exp 29**:
Add a single `Dropout(0.3)` layer after `relu_context` (the Dense(64) + ReLU6 temporal representation, before the Concatenate with the anchor path). This is the direct equivalent of Model 5a's dropout placement — applied to the richest intermediate representation before the final prediction head.

```
# Before (Exp 29):
GAP → Dense(64) → ReLU6 → context
# After (Exp 30):
GAP → Dense(64) → ReLU6 → Dropout(0.3) → context
```

All other settings unchanged from Exp 29: slope features (n_features=27), anchor path with 9 features → Dense(32) → ReLU6, cosine LR decay (1e-4 → 1e-6 over 100 epochs), batch size 512, patience=25.

**Architecture**:
```
Input: (180, 27) → Reshape to (180, 27, 1)
  ├─ Conv2D path:  [Conv2D(64,k=3)→BN→ReLU6 → Conv2D(64,k=7)→BN→ReLU6
  │                 → Conv2D(64,k=15)→BN→ReLU6 → Conv2D(64,k=27)→BN→ReLU6]
  │                → GAP → Dense(64) → ReLU6 → Dropout(0.3) → context  ← NEW
  └─ Skip path:   input[:, -1, [temp, lag60, lag120,
                                temp_slope_15, temp_slope_30, temp_slope_60,
                                solar_slope_30, humidity_slope_30, pressure_slope_60]]
                  → Dense(32) → ReLU6 → anchors
Concatenate([context(64), anchors(32)])  →  Dense(32) → ReLU6 → Dense(3) outputs
```

**Hypothesis**: Dropout forces the 64 context neurons to learn redundant representations, preventing individual neurons from memorising training-set-specific patterns. This should:
1. Close the train/val gap from ~10× toward ~2–3×
2. Push val_loss below the 0.001343 threshold (Exp 24 Conv best) and toward the 0.000682 Model 5a target
3. Potentially delay the best epoch from epoch 15 to a later epoch where the model has learned more robust representations

**Training config**:
- Learning rate: cosine decay from 1e-4 → 1e-6 over 100 epochs (unchanged)
- Loss: MSE (per-head, unchanged)
- Batch size: 512 (unchanged)
- Early stopping: patience=25 (unchanged)
- Epochs: 100

**Success criteria**:
- Train/val gap narrows from ~10× to ≤ 3× (primary signal that dropout is working)
- float val_loss ≤ 0.001343 (beat all-time Conv best from Exp 24)
- Ideally float val_loss approaches 0.000682 (Model 5a parity)
- Best epoch shifts later than epoch 15 (confirms regularization is taking effect)

**Results (final, epoch 101, 2026-05-10)**:
- Best val_loss: **0.0032** — best epoch was **3**; dropout did not delay convergence, model peaked even earlier than Exp 29
- Val MAE (float): diff_1hr: **0.01°C**, diff_2hr: **0.02°C**, diff_3hr: **0.02°C**
- Val MAE (normalized combined): **0.0108**
- Final train_loss: 7.8044e-04 (elevated by dropout during training); train/val gap: **~4.1×**
- Quantized model size: **218.20 KB**
- Quantized MAE (TFLite INT8, 500 samples): diff_1hr: **0.67°C**, diff_2hr: **1.39°C**, diff_3hr: **1.71°C**

**Feature importance** (all 27 features, ranked):
- **#1: `time_of_day_sin2`** (0.1011)
- **#2: `time_of_day_sin`** (0.0882)
- **#3: `solar_radiation`** (0.0824)
- **#4: `time_of_day_cos2`** (0.0738), **#5: `temp_lag120`** (0.0738) ✅ skip path working
- **#6: `time_of_day_cos`** (0.0727), **#7: `uv`** (0.0711)
- **#8: `solar_slope_30`** (0.0679), **#9: `illuminance`** (0.0673)
- **#10–15**: `wind_direction_sin` (0.0660), `wind_avg` (0.0656), `humidity_slope_30` (0.0655), `temp_slope_60` (0.0649), `wind_gust` (0.0634), `wind_direction_cos` (0.0634)
- **#16–21**: `temp_lag60` (0.0632), `station_pressure` (0.0619), `rain_accumulated` (0.0614), `day_of_year_sin` (0.0614), `temp_slope_30` (0.0613), `wind_lull` (0.0613)
- **#22–26**: `day_of_year_cos` (0.0612), `temp_delta_1` (0.0612), `pressure_slope_60` (0.0610), `relative_humidity` (0.0605), `temp_slope_15` (0.0594)
- **#27: `temperature`** (0.0563)

**Edge TPU compilation**:
- Compiled successfully; 14/15 ops on Edge TPU, 1 op (GATHER) on CPU
- 2 Edge TPU subgraphs; quantized size: 218.20 KB → 309.37 KB (compiled)
- On-chip memory: 229.75 KB used, 6.61 MB remaining

**Success criteria assessment**:
- ⚠️ Train/val gap ≤ 3×: gap narrowed to **~4.1×** (from ~10×) — meaningful improvement but short of target; note train loss is inflated by dropout during training
- ❌ float val_loss ≤ 0.001343: **0.0032** — regressed vs Exp 29 (0.0026); absolute accuracy got worse
- ❌ float val_loss approaches 0.000682: **0.0032** — 4.7× away from Model 5a parity
- ❌ Best epoch later than 15: **epoch 3** — peaked even earlier than Exp 29's epoch 15

**Outcome**: ❌ **REGRESSION** — Dropout(0.3) reduced the train/val ratio gap (~10× → ~4.1×) but absolute val_loss regressed from 0.0026 to 0.0032. Best epoch at 3 is the earliest seen across all Conv2D experiments, suggesting the dropout rate is too aggressive: the model is underfitting rather than just being regularized. Quantized 1hr and 2hr MAE improved slightly (0.67°C / 1.39°C vs 0.82°C / 1.49°C) but 3hr regressed (1.71°C vs 1.63°C). Feature importance is extremely flat — all 27 features cluster between 0.056–0.101, compared to Exp 29's wider spread. Time-of-day features dominate (#1–#4) while `temperature` collapsed to last place (#27, 0.056), suggesting the skip path's anchor gradient is being overwhelmed by the regularized context path.

**Analysis**:
- **Dropout narrowed the ratio gap but hurt absolute val_loss**: The train/val ratio fell from ~10× to ~4.1×, but this is partly because dropout inflates training loss, making the ratio look better than it is. The true signal is that val_loss got worse (0.0032 vs 0.0026), meaning the model is now underfitting — Dropout(0.3) is too strong for a 64-unit context vector.
- **Best epoch at 3 is a regression signal**: Across Exp 27 (epoch 3), Exp 28 (not stated), Exp 29 (epoch 15), Exp 30 (epoch 3). Dropout did not delay the best epoch — it collapsed it. The model is learning its best representations in the first few epochs and then degrading, which is consistent with underfitting rather than regularization.
- **Feature importance compression**: All 27 features now score between 0.056–0.101. Dropout on the context path forces the model to spread gradient across all features uniformly (each context neuron gets randomly masked), effectively averaging out feature-specific signals. The anchor path (skip connection) is not regularized and should retain specificity, but `temperature` at #27 (0.056) suggests the concat head is dominated by the (now noisy) context path.
- **Path forward**: Dropout(0.3) is too aggressive. Options: (a) reduce dropout rate to 0.1–0.15 and try again; (b) try L2 weight decay instead of dropout on the context Dense(64); (c) apply dropout to the final Dense(32) head instead of the context vector; (d) accept the ~0.0026 float ceiling and focus on improving quantized accuracy through QAT or output-head calibration.

---

## Experiment 31: Dropout(0.1) — Lighter Regularization on Context Vector

**Date**: 2026-05-10
**Goal**: Find the correct dropout rate for the context vector by stepping down from the over-aggressive Dropout(0.3) used in Exp 30. Exp 30 confirmed dropout is the right tool (train/val gap narrowed from ~10× to ~4.1×) but the rate was too high — best epoch collapsed to 3 and val_loss regressed from 0.0026 to 0.0032.

**Root cause of Exp 30 regression**:
Dropout(0.3) randomly silences 30% of the 64 context neurons per step. For a 64-unit vector this is ~19 neurons masked per batch, forcing the remaining ~45 to cover the entire temporal representation. The model cannot maintain feature-specific gradients at that noise level — all 27 features compressed into a flat 0.056–0.101 importance band, `temperature` fell to dead last (#27), and the model peaked at epoch 3 (earlier than even Exp 29's epoch 15). Dropout(0.3) is too wide a filter.

**Change from Exp 30**:
Reduce `Dropout(0.3)` → `Dropout(0.1)` on the context vector. Everything else is identical to Exp 30 (and Exp 29 before it).

```
# Before (Exp 30):
GAP → Dense(64) → ReLU6 → Dropout(0.3) → context
# After (Exp 31):
GAP → Dense(64) → ReLU6 → Dropout(0.1) → context
```

Dropout(0.1) masks only ~6–7 of the 64 context neurons per step — enough to prevent individual neurons from memorising training-set patterns, but light enough to preserve the gradient signal that keeps feature importances differentiated.

**Architecture** (identical to Exp 30 except dropout rate):
```
Input: (180, 27) → Reshape to (180, 27, 1)
  ├─ Conv2D path:  [Conv2D(64,k=3)→BN→ReLU6 → Conv2D(64,k=7)→BN→ReLU6
  │                 → Conv2D(64,k=15)→BN→ReLU6 → Conv2D(64,k=27)→BN→ReLU6]
  │                → GAP → Dense(64) → ReLU6 → Dropout(0.1) → context  ← CHANGED
  └─ Skip path:   input[:, -1, [temp, lag60, lag120,
                                temp_slope_15, temp_slope_30, temp_slope_60,
                                solar_slope_30, humidity_slope_30, pressure_slope_60]]
                  → Dense(32) → ReLU6 → anchors
Concatenate([context(64), anchors(32)])  →  Dense(32) → ReLU6 → Dense(3) outputs
```

**Training config** (unchanged from Exp 29/30):
- Learning rate: cosine decay from 1e-4 → 1e-6 over 100 epochs
- Loss: MSE (per-head)
- Batch size: 512
- Early stopping: patience=25
- Epochs: 100

**Success criteria**:
- float val_loss ≤ 0.0026 (beat Exp 29; Exp 30 regressed to 0.0032)
- Train/val gap ≤ 4× (vs ~4.1× in Exp 30, ~10× in Exp 29 — need real val_loss improvement, not just ratio)
- Best epoch later than epoch 3 (confirms lighter dropout preserves the learning trajectory)
- Feature importance spread wider than Exp 30's flat 0.056–0.101 band (confirms context gradient is preserved)

---

## Infrastructure: macOS Training Hang — Root Cause and Fix

**Date**: 2026-05-12

### Problem

Training hangs indefinitely when the Mac is in active use (browsing, typing, etc.), but runs cleanly when left idle. The hang is always in the same place:

```
ProcessFunctionLibraryRuntime::RunSync
  → absl::Notification::WaitForNotification()
    → pthread_cond_wait  (blocked forever)
```

`RunSync` dispatches a compiled `tf.function` to the Metal GPU asynchronously and waits for a completion notification that never arrives.

### Root Cause

**`steps_per_execution=20` sustains ~90% GPU utilisation, leaving Metal no idle time to serve the display without preempting TF.**

With `steps_per_execution=20`, each `RunSync` call bundles 20 mini-batches into a single Python→GPU dispatch. The GPU runs continuously for the full duration (~27 s at the time of the last recorded hang) with no opportunity for Metal's scheduler to service WindowServer (the display compositor) between mini-batches. When the Mac is in interactive use, Metal must preempt TF's in-flight compute shader. Under CUDA/Linux, NVIDIA guarantees forward progress after preemption. Metal's guarantee is weaker — under sustained display pressure a preempted command buffer can be delayed indefinitely. TF's `absl::Notification` wait has no timeout, so the main thread blocks forever.

**What the earlier non-hanging experiments had in common**: the Conv1D experiments ran with `mixed_float16` precision enabled. On Apple Silicon's Metal backend, not all ops have float16 kernels, so TF silently falls back to CPU float32 for some operations. Those CPU fallbacks naturally broke up sustained GPU usage — the GPU got implicit breathing room between Metal dispatches. When mixed precision was disabled (Exp 22, for unrelated LR scheduling reasons — see that entry), all ops moved to the GPU in float32 with no CPU interleaving, which is what drove utilisation to ~90% and made the hangs appear.

**Attempted fix that did not work**: a `GPUThrottleCallback` that slept between `on_train_batch_end` calls. With `steps_per_execution=20`, `on_train_batch_end` fires *after* each 20-batch `RunSync` completes. Sleeping there gives Metal gaps between dispatches, but the hang occurs *within* a single dispatch — so the sleep was never reachable when a hang was in progress.

### Fix

**Set `steps_per_execution=1`.** Each `RunSync` call now covers exactly one mini-batch, then control returns to Python. The Python-side callback overhead between successive `RunSync` calls acts as the scheduling gap that Metal needs to service display requests without preempting TF — functionally the same role the CPU-op fallbacks played when mixed precision was enabled, but without re-enabling mixed precision (which breaks LR scheduling).

```python
# Before (sustained GPU, hangs during interactive use):
steps_per_execution=20

# After (natural Python gaps between every mini-batch):
steps_per_execution=1
```

The overhead cost is negligible: epoch time is ~940 s, so even if Python overhead per batch is 5–10 ms, across ~1450 batches that is 7–15 s — under 2% of epoch time.

Mixed precision remains disabled. Re-enabling it would fix the GPU utilisation problem via CPU fallbacks, but silently breaks the LR scheduler (Adam gets wrapped in `LossScaleOptimizer`; LR assignments target the inner optimizer and are silently ignored).

### Last hang recorded

Epoch 85, batch 940/1147 (82%), stuck for 30 minutes (written to `training_hang_detected.json`). Average batch time at hang: 27.5 s per `steps_per_execution=20` group — already elevated, indicating Metal was already struggling before the full stall.

---

## Experiment 31: ⛔ CANCELLED

**Date**: 2026-05-14
**Status**: Cancelled before training — superseded by Exp 32 based on cross-model learnings from Model 5a Clean.

**Reason**: Tuning Dropout(0.3) → Dropout(0.1) addresses regularization strength but does not touch the underlying training dynamics problem. Model 5a Clean (Exp 1–8) identified that **training schedule — not architecture or regularization rate — determines whether the optimizer escapes local minima**. Model 5b plateaus at epoch 15 (Exp 29) with a ~10× train/val gap; this is the same symptom as Model 5a Clean's early-stop at epoch 27 under fixed LR, which ReduceLROnPlateau resolved in a single change. Applying a dropout rate search while leaving the cosine decay schedule in place is a lower-leverage path than directly importing the training dynamics fix that produced Model 5a Clean's best results.

---

## Data Quality Fix: Gap Detection Threshold and Target Clipping

**Date**: 2026-05-13

### Background

While investigating the quantized model noise floor, training was run on a "clean data" variant of Model 5a that added `_invalidate_targets_crossing_gaps()`. Results were dramatically worse than the original despite the intent being to improve data quality. This triggered an analysis of what the cleaning was actually doing.

### Problem: Gap detection threshold too aggressive

All training scripts used `tol_s=90` (Model 5b, clean_data 5a) or `tol_s=300` (Model 5a Pi) as the gap detection threshold. Analysis of the training data revealed:

| Threshold | Gaps triggered | Est. rows invalidated | % of training data |
|-----------|---------------|----------------------|-------------------|
| 90 s | 9,042 | ~1,627,560 | **160%** (massive overlap) |
| 300 s | 215 | ~38,700 | 3.8% |
| 600 s | 53 | ~9,540 | 0.9% |

The dataset contains ~9,000 gaps of 90 s–5 min — individual missed 1-minute readings from Wi-Fi glitches. These are not sensor restart events; the sensor kept running and its forward-looking targets are valid. The `_invalidate_targets_crossing_gaps()` function blankets a 180-row window before every such gap, effectively removing the majority of training data when `tol_s=90`.

Only 53 gaps exceed 10 minutes; these are real outages where a post-restart sensor glitch could corrupt the temperature reading used as a forward target.

### Evidence of impact

| Run | val_loss (norm) | val_mae (norm) | Actual MAE (°C) |
|-----|----------------|----------------|----------------|
| Original 5a (no cleaning) | 0.000724 | 0.00460 | 0.083°C |
| Clean data (tol_s=90, broken) | 0.003926 | 0.01165 | 0.166°C |
| Clean data (tol_s=600, fixed) | 0.001069 | 0.00544 | 0.076°C |

> Note: normalized metrics are not directly comparable across runs because the target scaler range changed (outlier-inflated range in original: ±18°C; clipped range: ±14°C). Convert to actual °C using `actual_MAE = val_mae × (target_range / 2)` for a fair comparison.

The fixed cleaning produces an 8% improvement in actual °C MAE over the original, while the broken cleaning was 2× worse.

### Secondary fix: Target clipping

Even with `tol_s=600`, a small number of extreme target values survive gap detection. In the training dataset:

- `|diff_1hr| > 10°C`: 12 rows (0.001%) — physically impossible for a 1-hour outdoor temperature change
- `|diff_3hr| > 10°C`: 1,208 rows (0.12%) — p99.9 is ~10°C; values beyond this are sensor artefacts
- `|diff_3hr| > 13°C`: 11 rows (0.001%)

These outliers inflate the target scaler range, compressing the normalized representation of all real temperature changes. A clip at ±12°C removes the artefacts without affecting genuine weather signal.

```python
DIFF_CLIP = 12.0
diff_targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']
train_df[diff_targets] = train_df[diff_targets].clip(-DIFF_CLIP, DIFF_CLIP)
val_df[diff_targets]   = val_df[diff_targets].clip(-DIFF_CLIP, DIFF_CLIP)
```

### Fix applied

Updated `tol_s=600` and added `DIFF_CLIP=12.0` in all three training scripts:

- `workspace/Model 5 new arch. slope calc clean_data/train_model.py` (was `tol_s=90`)
- `workspace/Model 5a pi/train_model.py` (was `tol_s=300`)
- `workspace/Model 5b Conv2D/train_model_conv2D.py` (was `tol_s=90`)

---

---

## Experiment 32: ReduceLROnPlateau + L2 Regularization — Training Dynamics Transfer from Model 5a Clean

**Date**: 2026-05-14

### Context — Cross-model learning from Model 5a Clean

Model 5a Clean explored the identical prediction task (3-horizon temperature difference) with a different architecture (AveragePooling1D + Dense bottleneck). Its most impactful finding was that **training dynamics — not architecture or feature set — controlled whether the optimizer escaped a local minimum**:

| Model 5a Clean experiment | val_loss | Key observation |
|--------------------------|----------|----------------|
| Exp 1–2: fixed LR 1e-5, patience=5 | 0.001069 | Stopped at epoch 27; model snapped to time-of-day minimum |
| Exp 3: ReduceLROnPlateau + lr=1e-4 + patience=20 | 0.000440 | Ran to epoch 95; escaped minimum; 58% improvement |
| Exp 4–7: same schedule + architecture refinements | 0.0002–0.0005 | Further improvements all built on the Exp 3 training config |

Model 5b shows the same early-plateau symptom: Exp 29 (best non-dropout result) peaked at **epoch 15** with a **~10× train/val gap**. The cosine decay schedule (1e-4 → 1e-6 over 100 epochs) is a fixed schedule — it continues stepping down LR according to a predetermined curve regardless of whether the val_loss is improving or stalled. When the model hits a plateau, the decaying LR only weakens the gradient signal rather than redirecting it. ReduceLROnPlateau waits for a stall and then halves the LR, giving the optimizer a new descent direction each time — this is what drove Model 5a Clean from epoch 27 to epoch 95.

L2 regularization addresses the ~10× train/val overfitting gap directly through weight norm constraints. Dropout(0.3) collapsed feature importance to a flat 0.056–0.101 band and caused underfitting (epoch 3); Dropout(0.1) is an incremental fix. L2 constrains weights without masking gradient signals per batch, so feature differentiation is preserved.

### Hypothesis

Replacing cosine decay with ReduceLROnPlateau and adding L2 kernel regularization will break through the ~0.0026 float ceiling by:
1. **Responding adaptively to val_loss plateaus** — when the Conv2D model stalls (as at epoch 15 in Exp 29), LR halving creates a new gradient descent direction rather than continuing the predetermined decay
2. **Constraining weight magnitudes** — L2 on Conv2D and Dense kernels reduces the overfitting gap without the feature gradient collapse caused by Dropout(0.3)

### Changes from Exp 29 (best non-dropout baseline, float val_loss=0.0026)

1. **Replace cosine LR decay with ReduceLROnPlateau**: `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)` — same config as Model 5a Clean Exp 3
2. **Add L2 regularization**: `kernel_regularizer=l2(1e-4)` on all Conv2D and Dense(64, 32) layers
3. **Increase EarlyStopping patience**: 25 → 30 (allows ReduceLROnPlateau to fire ~3 times before early stopping)
4. **Remove Dropout**: no dropout in this experiment — L2 regularization is the sole regularizer

```python
# Before (Exp 29 — cosine schedule, no regularization):
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
lr_schedule = LearningRateScheduler(
    lambda epoch: cosine_decay(epoch, initial_lr=1e-4, final_lr=1e-6, total_epochs=100)
)
callbacks = [lr_schedule, EarlyStopping(monitor='val_loss', patience=25)]
# No kernel_regularizer on any layer

# After (Exp 32 — adaptive schedule + L2):
from tensorflow.keras.regularizers import l2
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
]
# All Conv2D and Dense(64, 32) layers: kernel_regularizer=l2(1e-4)
```

### Architecture (identical to Exp 29 except regularization; no Dropout)

```
Input: (180, 27) → Reshape to (180, 27, 1)
  ├─ Conv2D path:  [Conv2D(64,k=3,l2=1e-4)→BN→ReLU6
  │                 → Conv2D(64,k=7,l2=1e-4)→BN→ReLU6
  │                 → Conv2D(64,k=15,l2=1e-4)→BN→ReLU6
  │                 → Conv2D(64,k=27,l2=1e-4)→BN→ReLU6]
  │                → GAP → Dense(64,l2=1e-4) → ReLU6 → context   ← no Dropout
  └─ Skip path:   input[:, -1, [temp_delta_1, temp_lag60, temp_lag120,
                                temp_slope_15, temp_slope_30, temp_slope_60,
                                solar_slope_30, humidity_slope_30, pressure_slope_60]]
                  → Dense(32,l2=1e-4) → ReLU6 → anchors
Concatenate([context(64), anchors(32)]) → Dense(32,l2=1e-4) → ReLU6 → Dense(3) outputs
```

### Training config

| Parameter | Exp 29 | Exp 32 |
|-----------|--------|--------|
| LR schedule | cosine decay 1e-4→1e-6, 100 epochs | **ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7)** |
| Starting LR | 1e-4 | 1e-4 (same) |
| Kernel regularizer | none | **l2(1e-4) on all Conv2D + Dense** |
| Dropout | none | none |
| EarlyStopping patience | 25 | **30** |
| Loss / batch / max epochs | MSE, 512, 100 | MSE, 512, 100 (same) |

### Success criteria

- float val_loss < 0.0026 (beat Exp 29 — any improvement confirms ReduceLROnPlateau is the lever)
- best epoch > 20 (confirms ReduceLROnPlateau extended training past the Exp 29 plateau at epoch 15)
- train/val gap < 10× (L2 should close the overfitting gap vs Exp 29's ~10×)
- feature importance spread ≥ Exp 29's range (L2 preserves gradient differentiation unlike Dropout(0.3))

### Results

Training ran to epoch 100 (not early stopped); best epoch was **46**.  
LR decayed twice via ReduceLROnPlateau: `1e-4 → 5e-5 → 2.5e-5` (at epoch 100).

| Metric | Value | vs Success Criteria |
|--------|-------|---------------------|
| float val_loss | **0.0024** | ✅ < 0.0026 (beat Exp 29) |
| float val_mae (normalized) | 0.0069 | — |
| float MAE (°C) | 1hr=0.01, 2hr=0.01, 3hr=0.01 | — |
| Best epoch | **46** | ✅ > 20 (vs Exp 29 plateau at epoch 15) |
| Train/val gap | ~3.1× (train_loss=7.70e-4, val_loss=0.0024) | ✅ < 10× (L2 is working) |
| Quantized model size | 218.20 KB | — |
| Quantized MAE (°C) | 1hr=0.67, 2hr=1.39, 3hr=1.71 | ❌ catastrophic |

**Permutation Feature Importance (by increase in val_loss)**:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | temp_lag120 | **0.1025** |
| 2 | temp_lag60 | 0.0997 |
| 3 | time_of_day_sin2 | 0.0854 |
| 4 | uv | 0.0849 |
| 5 | humidity_slope_30 | 0.0841 |
| … | (mid-range) | 0.079–0.083 |
| 27 | time_of_day_cos | **0.0719** |

Feature importance range: **0.0719–0.1025** — clear spread vs Exp 29/30's flat 0.056–0.101 band. ✅

**Qualitative shift**: `temp_lag120` is now the **#1 feature** (0.1025) and `temp_lag60` is #2 (0.0997). This mirrors Model 5a's dominant `temp_lag120` (0.093) — the skip path's direct anchor to lag features is working as designed. Previous experiments always had `time_of_day_cos` at the top; that is now ranked last (#27). L2 regularization preserved gradient differentiation where Dropout(0.3) had collapsed it.

**Note on best-run tracker**: The script reported `conv2d_exp27_run1` (val_loss=0.0032) as the best run rather than Exp 32 (val_loss=0.0024). This appears to be a bug in the cross-run comparison logic — 0.0024 < 0.0032 so Exp 32 is the actual float accuracy leader.

**Live deployment result (4.5 years of InfluxDB data, 2026-05-20)**:

| Metric | Value | Model 5ac (reference) |
|--------|-------|----------------------|
| Mean bias (Actual − Predicted) | **+0.182°C** | −0.297°C |
| StdDev (Actual − Predicted) | **0.335°C** | 0.147°C |
| INT8 steps equivalent | 3.0 | 1.0 |

Exp 32 is the closest Model 5b has come to Model 5ac, but still **2.3× wider** than Model 5ac's 0.147°C StdDev. The +0.182°C mean bias indicates the model runs slightly warm vs reality — a systematic offset that L2 regularization did not eliminate. The float val_loss improvement (Exp 29: 0.0026 → Exp 32: 0.0024) produced a proportional but small improvement in live StdDev.

**Outcome**: ⚠️ **PARTIAL SUCCESS** — New float accuracy record (0.0024, beats Exp 29's 0.0026 and all prior experiments). All success criteria met for float accuracy and training dynamics. Quantization still catastrophic.

**Analysis**:

1. **ReduceLROnPlateau confirmed as the correct lever**: best epoch moved from 15 (Exp 29) to 46 — the optimizer escaped the early plateau and ran 3× longer. The 3.1× train/val gap (vs ~10× in Exp 29) confirms L2 is closing the overfitting gap without collapsing feature gradients.

2. **Still 3.5× short of Model 5a's 0.000682**: The Conv2D architecture is improving experiment-over-experiment but has not broken through to Model 5a territory. The gap is structural — GAP compresses all temporal context into a single 64-dim vector, then the skip path provides 9 scalar anchors. The network may need either more Conv2D capacity or a richer skip path to fully replicate Model 5a's performance.

3. **Quantization failure persists** (0.67–1.71°C vs 0.01°C float): same root cause as prior experiments — INT8 cannot faithfully represent the dynamic range of BatchNorm outputs across the Conv2D stack. PTQ has never worked for this architecture. QAT (Quantization-Aware Training) remains the only untried fix.

4. **Next direction**: The training schedule (ReduceLROnPlateau) and regularization (L2) are now dialled in. The remaining float accuracy gap likely requires either (a) deeper/wider Conv2D (e.g., 128 filters), (b) longer patience allowing more ReduceLROnPlateau cycles, or (c) a richer skip path. Quantization must eventually move to QAT.

5. **Training hit `max_epochs=100` cap — early stopping never fired**: With best epoch=46 and patience=30, EarlyStopping should have fired at epoch 76. It did not, because each LR reduction (at ~epoch 54 and ~epoch 62) produced a tiny val_loss improvement that reset EarlyStopping's `wait` counter to 0. The two callbacks — `ReduceLRCallback` and `EarlyStopping` — maintain **independent** `wait` and `best` trackers; reducing the LR does not reset EarlyStopping's counter, but the improved val_loss that follows a LR cut does. The final improvement reset EarlyStopping around epoch ~70, so patience=30 would have fired at ~100 — but `max_epochs=100` ended training first. Evidence: the LR ended at 2.5e-5 (two reductions), and training ran to exactly the epoch cap. `restore_best_weights=True` correctly recovered epoch 46 weights, so accuracy is unaffected; the only waste was ~54 extra epochs of compute. **Fix: raise `max_epochs` from 100 to 150** — EarlyStopping will terminate training on its own schedule without hitting the cap.

---

---

## Experiment 33: Fix GATHER Op — Full Edge TPU Mapping

**Date**: 2026-05-20

### Problem

Exp 32's compiled model (`weather_model_5b_best_edgetpu.tflite`) runs with 2 Edge TPU subgraphs — a sign that one op forces a CPU fallback mid-graph. The compiler log confirms:

```
Number of operations that will run on Edge TPU: 14
Number of operations that will run on CPU: 1

Operator                       Count      Status
CONV_2D                        4          Mapped to Edge TPU
GATHER                         1          Operation not supported   ← CPU fallback
STRIDED_SLICE                  1          Mapped to Edge TPU
RESHAPE                        1          Mapped to Edge TPU
MEAN                           1          Mapped to Edge TPU
CONCATENATION                  1          Mapped to Edge TPU
FULLY_CONNECTED                6          Mapped to Edge TPU
```

The `GATHER` op is generated by `LastTimestepAnchorFeatures.call()`:

```python
def call(self, inputs):
    row = inputs[:, -1, :]                               # STRIDED_SLICE ✅ Edge TPU
    return tf.gather(row, self.anchor_indices, axis=1)   # GATHER        ❌ CPU only
```

`tf.gather` with a non-contiguous index list is not in the Edge TPU supported op set. The STRIDED_SLICE immediately before it is supported, but the GATHER forces a CPU round-trip that splits the graph into 2 subgraphs, adding PCIe/USB transfer latency between them.

### Fix

Replace `tf.gather` (non-contiguous index selection → GATHER) with a full last-timestep slice (→ STRIDED_SLICE, Edge TPU-supported). `Dense(32)` downstream learns its own feature weights from all `n_features` inputs rather than receiving a hardcoded 9-feature subset.

**Before (Exp 29–32)**:
```python
# anchor_indices = [idx(temperature), idx(temp_lag60), idx(temp_lag120),
#                   idx(temp_slope_15), idx(temp_slope_30), idx(temp_slope_60),
#                   idx(solar_slope_30), idx(humidity_slope_30), idx(pressure_slope_60)]
row = inputs[:, -1, :]                              # STRIDED_SLICE (9 chosen features)
anchor_vec = tf.gather(row, anchor_indices, axis=1) # GATHER — NOT Edge TPU
# → Dense(9 → 32)
```

**After (Exp 33)**:
```python
anchor_vec = inputs[:, -1, :]   # STRIDED_SLICE — Edge TPU ✅  (all n_features)
# → Dense(n_features → 32)      # Dense learns feature selection
```

**Weight count change**: Dense anchor layer goes from 9 × 32 = 288 → 27 × 32 = 864 weights (still negligible relative to Conv2D blocks). Dense will learn to down-weight irrelevant features; the important lag and slope features from Exp 32's importance ranking should still dominate.

**Backward compatibility**: `LastTimestepAnchorFeatures.__init__` accepts and silently discards a legacy `anchor_indices` kwarg so Exp 29–32 saved models can still be deserialized without error.

### Changes from Exp 32

1. **`LastTimestepAnchorFeatures.call()`**: `tf.gather` removed; returns `inputs[:, -1, :]` directly
2. **`LastTimestepAnchorFeatures.__init__()`**: `anchor_indices` parameter removed (legacy kwarg discarded for compat)
3. **`LastTimestepAnchorFeatures.get_config()`**: `anchor_indices` field removed
4. **Model builder**: `anchor_indices` list construction removed; call site updated to `LastTimestepAnchorFeatures(name="last_ts_anchors")(input_layer)`
5. **`max_epochs`: 100 → 150** — Exp 32 hit the epoch cap before EarlyStopping could fire (see analysis in Exp 32 results item 5). Raising the cap ensures EarlyStopping is always the termination condition rather than the hard limit.
6. **Everything else** (architecture, training config, L2, ReduceLROnPlateau, patience=30) identical to Exp 32

### Expected Edge TPU output

```
Number of Edge TPU subgraphs: 1        ← single contiguous subgraph
Number of operations that will run on Edge TPU: 15   ← all 15 ops
Number of operations that will run on CPU: 0
```

### Success criteria

- All 15 ops mapped to Edge TPU (0 CPU fallbacks)
- Single Edge TPU subgraph
- float val_loss ≤ 0.0024 (match or beat Exp 32 — architecture change is minor)
- EarlyStopping fires before epoch 150 (confirms the cap is no longer the bottleneck)
- Quantized MAE: any improvement over Exp 32's 0.67/1.39/1.71°C is a bonus

### Will Exp 33 get closer to Model 5ac?

Exp 33 makes two changes that could improve live accuracy:

1. **Longer training (max_epochs 100 → 150)**: Exp 32 hit the 100-epoch cap with only 2 ReduceLROnPlateau cycles (1e-4 → 5e-5 → 2.5e-5). With the cap raised, EarlyStopping (patience=30) will terminate training naturally — allowing 1–2 more LR halving cycles before early stopping. This is the same mechanism that drove Model 5a Clean from 0.001069 → 0.000440 (58% improvement). If float val_loss drops from 0.0024 toward ~0.0015, the live StdDev should scale roughly proportionally: **estimated ~0.25–0.27°C** (vs 0.335°C today).

2. **Full anchor path (27 features → Dense(32) vs 9 hardcoded features)**: Giving the Dense anchor layer all 27 features lets it learn its own importance weighting rather than receiving a fixed 9-feature subset. This may help or be neutral — the Exp 32 feature importance already shows temp_lag120 (#1, 0.1025) and temp_lag60 (#2, 0.0997) dominating, which are both in the hardcoded set.

**Bottom line**: Exp 33 is likely to close some of the gap with Model 5ac purely through longer training, not from the architecture change. The 2.3× gap (0.335°C vs 0.147°C) is too large to close in one experiment — reaching Model 5ac parity in float requires val_loss to fall below 0.000682 (3.5× further improvement), which will need additional experiments after Exp 33. Quantization (QAT) remains the path to matching Model 5ac's INT8 precision floor.

### Results

Training ran past the epoch 150 cap (watchdog stopped); best epoch was **19**.
LR cascaded all the way to min_lr=1e-7 via multiple ReduceLROnPlateau reductions — same pattern as Exp 32 where tiny post-reduction improvements reset EarlyStopping's counter, allowing many LR cycles before the cap was hit.

| Metric | Value | vs Success Criteria |
|--------|-------|---------------------|
| float val_loss | **0.0025** | ❌ > 0.0024 (minor regression from Exp 32) |
| float val_mae (normalized) | 0.0067 | — |
| float MAE (°C) | 1hr=0.01, 2hr=0.01, 3hr=0.01 | — |
| Best epoch | **19** | ❌ regression from Exp 32's epoch 46 |
| Quantized model size | 218.45 KB | — |
| Quantized MAE (°C) | 1hr=0.67, 2hr=1.39, 3hr=1.71 | ❌ unchanged (catastrophic) |
| Edge TPU: all 15 ops mapped | Not confirmed in output | — |

**Permutation Feature Importance (conv2d_exp32_run1 = Exp 33 naming)**:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | temperature | **0.1000** |
| 2 | temp_lag60 | 0.0937 |
| 3 | temp_lag120 | 0.0853 |
| 4 | time_of_day_sin2 | 0.0839 |
| 5 | relative_humidity | 0.0828 |
| 6 | pressure_slope_60 | 0.0811 |
| 7 | wind_lull | 0.0797 |
| … | (mid-range) | 0.074–0.079 |
| 27 | time_of_day_cos2 | **0.0718** |

Importance range: **0.0718–0.1000** — wider spread than Exp 30 (0.056–0.101), comparable to Exp 32.

**Key qualitative shift vs Exp 32**: `temperature` is now #1 (0.100) instead of `temp_lag120` (#1 in Exp 32, 0.1025). Giving the anchor Dense all 27 features provides a direct gradient path to current temperature, displacing the lag anchors as the dominant signal. Exp 32's 9-feature anchor forced the model to rely on `temp_lag120` and `temp_lag60` because those were the only lag signals in the anchor — widening to 27 features diluted that specificity.

**Note on run naming and tracker bug**: The script names this run `conv2d_exp32_run1` (a naming artefact from Exp 32's continuation). The tracker comparison logic still reports `conv2d_exp27_run1` (val_loss=0.0032) as "best run" — the same inverted-comparison bug from Exp 32. As a result, **`weather_model_5b_best.tflite` was copied from the exp27 checkpoint (old GATHER architecture), NOT from Exp 33**. This was confirmed when Edge TPU compilation after training showed GATHER still present:

```
GATHER  1  Operation not supported   ← from exp27 model, not Exp 33
```

Exp 33 (val_loss=0.0025) is the actual float accuracy leader, and its architecture does not contain GATHER — but that model was never promoted to `weather_model_5b_best.tflite` due to the tracker bug. The GATHER fix in Exp 33 is confirmed in the code; it has not yet been validated against the compiled TFLite output.

**Outcome**: ❌ **MINOR REGRESSION** — val_loss regressed from Exp 32's 0.0024 to 0.0025; best epoch regressed from 46 to 19. The full-27-feature anchor caused earlier overfitting. The GATHER fix (STRIDED_SLICE) maintains Edge TPU compatibility but accuracy cost was real. Quantized MAE unchanged.

**Analysis**:

1. **Full anchor caused earlier overfitting**: Giving Dense(32) all 27 features provides such a rich direct path from current-timestep sensors to the output that the model reaches its best generalisation at epoch 19 rather than epoch 46. With more direct "shortcuts", the Conv2D path has less gradient pressure to extract useful temporal representations. The 27-feature anchor is simultaneously better (richer signal) and worse (enables lazy early convergence).

2. **Best epoch regression (46 → 19) is a structural problem**: The LR cascaded all the way to min_lr=1e-7, meaning many LR reductions fired, but none produced a val_loss improvement beyond epoch 19. This is a capacity/information ceiling — not a training dynamics issue. ReduceLROnPlateau alone cannot break through if the model's maximum generalisation capacity has been reached.

3. **Quantized MAE unchanged at 0.67/1.39/1.71°C**: QAT remains the only untested path to closing the float→quantized gap. PTQ cannot handle the dynamic range of BatchNorm activations across the Conv2D stack.

4. **Path forward**: The 0.0024–0.0025 ceiling has now been confirmed across Exp 32 and Exp 33. Breaking through requires giving the model more useful information — specifically, direct access to sensor readings at *multiple* historical timesteps (t=-60, t=-120), not just the current timestep. The skip path currently only anchors at t=-1; adding a second anchor at t=-121 gives the model the same information advantage that drove Model 5a's `temp_lag120` to be its single most important feature.

---

## Experiment 34: Replace GAP with Multi-Point Temporal Extraction

**Date**: 2026-05-23

### Hypothesis

The float val_loss ceiling (~0.0024–0.0025 across Exp 32–33) is structural: `GlobalAveragePooling2D` averages all 180 timesteps equally, diluting the signal from any single historical position by ~180×. Model 5a never has this bottleneck — every lag feature has its own weight column with a direct gradient path to the output.

Extracting the Conv2D feature map at three specific temporal positions — t=−1 (current), t=−61 (60 min ago), t=−121 (120 min ago) — and concatenating them gives the dense head positionally-specific learned representations:
- `conv_out[:, −1, :]` — what the Conv2D learned about the current moment
- `conv_out[:, −61, :]` — what it learned about 60 min ago (equiv. to temp_lag60 position in Model 5a)
- `conv_out[:, −121, :]` — what it learned about 120 min ago (equiv. to temp_lag120 position, Model 5a's #1 feature)

Each of these is a 64-dim vector encoding all 27 features at that timestep as seen through the Conv2D stack. Concatenated: 192-dim → Dense(64). GAP's 64-dim average is replaced by three specific snapshots.

### Changes from Exp 33

1. **Fix tracker comparison bug** (prerequisite): Results reader used a hardcoded stale filename (`conv2d_exp27_run*.json`), causing the wrong model to be promoted as "best". Fixed by introducing `EXP_NAME = "conv2d_exp34"` so writer and reader always share the same source.
2. **Remove `GlobalAveragePooling2D`**: replaced entirely; no other GAP-related changes.
3. **Add `Reshape((SEQ_LEN, FILTERS))`** after the final conv block: collapses the trivial size-1 feature dimension from `(batch, 180, 1, 64)` to `(batch, 180, 64)` before slicing.
4. **Add `MultiPointTemporalExtraction`**: new custom layer that extracts conv representations at t=−1, t=−61, t=−121 via three STRIDED_SLICE ops and concatenates them (CONCATENATION) → `(batch, 192)`. All four ops are Edge TPU-confirmed.
5. **Anchor path unchanged from Exp 33**: `inputs[:, −1, :]` (27 features) → `Dense(32, L2=1e-4)` → `ReLU6`. No t=−121 in the anchor — the conv path now handles that position.
6. **Everything else unchanged** from Exp 32: Conv2D(64) × 4 blocks, L2=1e-4 throughout, ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-7), EarlyStopping(patience=30), max_epochs=150, MSE, batch=512.

### Architecture

```
Input: (180, 27) → Reshape to (180, 27, 1)
  ├─ Conv2D path:  [Conv2D(64,k=3,l2=1e-4)→BN→ReLU6
  │                 → Conv2D(64,k=7,l2=1e-4)→BN→ReLU6
  │                 → Conv2D(64,k=15,l2=1e-4)→BN→ReLU6
  │                 → Conv2D(64,k=27,l2=1e-4)→BN→ReLU6]  ← shape: (180, 1, 64)
  │                → Reshape(180, 64)                     ← collapses size-1 dim
  │                → MultiPointTemporalExtraction          ← STRIDED_SLICE × 3
  │                  [t=−1, t=−61, t=−121] → Concat(192)  ← CONCATENATION
  │                → Dense(64,l2=1e-4) → ReLU6 → context(64)
  └─ Anchor path:  input[:, -1, :] → STRIDED_SLICE ✅ (27 features)
                   → Dense(32,l2=1e-4) → ReLU6 → anchors(32)
Concatenate([context(64), anchors(32)]) → Dense(32,l2=1e-4) → ReLU6 → Dense(3)
```

### Edge TPU compliance

All new ops are from the confirmed Edge TPU-compatible set (Exp 30 compiler log):
- `RESHAPE` ✅ — confirmed mapped in Exp 30
- `STRIDED_SLICE` × 3 ✅ — confirmed mapped in Exp 30/33
- `CONCATENATION` ✅ — confirmed mapped in Exp 30

No new op types introduced. Expected: single subgraph, all 16 ops on Edge TPU.

### Training config (unchanged from Exp 32)

| Parameter | Value |
|-----------|-------|
| LR schedule | ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7) |
| Starting LR | 1e-4 |
| L2 (all layers) | 1e-4 |
| Dropout | none |
| EarlyStopping patience | 30 |
| max_epochs | 150 |
| Loss / batch | MSE, 512 |

### Success criteria

- float val_loss < 0.0024 (beat both Exp 32 and Exp 33 — GAP removal should be impactful)
- best epoch > 19 (no shortcutting through anchor; conv path must contribute)
- `temp_lag120` or `temp_lag60` in top 5 feature importance (confirms positional extraction is used)
- All ops on Edge TPU, single subgraph
- EarlyStopping fires before epoch 150

### Results

Training ran to epoch 82 before the watchdog stopped it; best epoch was **21**.  
LR cascaded to 6.25e-6 (4 ReduceLROnPlateau reductions: 1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6) — same pattern as Exp 32/33 where tiny post-reduction improvements reset EarlyStopping's counter.

| Metric | Value | vs Success Criteria |
|--------|-------|---------------------|
| float val_loss | **0.0104** | ❌ massive regression from Exp 32's 0.0024 (~4.3×) |
| float val_mae (normalized) | 0.0158 | — |
| float MAE (°C) | 1hr=0.01, 2hr=0.04, 3hr=0.08 | ❌ 2hr and 3hr much worse (Exp 32: all 0.01°C) |
| Best epoch | **21** | ❌ barely above Exp 33's 19; conv path not contributing |
| Quantized model size | 227.60 KB | — |
| Quantized MAE (°C) | 1hr=0.78, 2hr=1.35, 3hr=1.83 | ❌ worse than Exp 32 (0.67/1.39/1.71) |
| Edge TPU: all ops mapped | **✅ single subgraph, all 18 ops** | ✅ first clean single-subgraph result |
| EarlyStopping fires before epoch 150 | watchdog fired at epoch 82 | — |

**Edge TPU compiler output**:
```
Number of Edge TPU subgraphs: 1     ← single contiguous subgraph ✅
Total number of operations: 18      ← all on Edge TPU
Operator                       Count      Status
FULLY_CONNECTED                6          Mapped to Edge TPU
RESHAPE                        2          Mapped to Edge TPU
CONV_2D                        4          Mapped to Edge TPU
CONCATENATION                  2          Mapped to Edge TPU
STRIDED_SLICE                  4          Mapped to Edge TPU
On-chip memory: 236.75 KB used, 6.59 MB remaining
```

**Permutation Feature Importance** (ranked by increase in val_loss):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | time_of_day_cos | **0.0676** |
| 2 | humidity_slope_30 | 0.0642 |
| 3 | temp_lag120 | 0.0628 |
| 4 | wind_lull | 0.0622 |
| 5 | uv | 0.0617 |
| … | (mid-range) | 0.058–0.062 |
| 25 | temp_lag60 | **0.0551** |
| 26 | time_of_day_sin | 0.0490 |
| 27 | temp_slope_60 | **0.0405** |

Importance range: **0.0405–0.0676** — the flattest distribution seen across all experiments. `time_of_day_cos` returned to #1; `temp_lag60` collapsed to #25 (was #2 at 0.0997 in Exp 32); `temp_slope_60` is the weakest feature at #27.

**Outcome**: ❌ **SEVERE REGRESSION** — Float val_loss regressed from 0.0024 (Exp 32) to 0.0104 (~4.3× worse). Single subgraph Edge TPU mapping is a win, but all accuracy metrics degraded substantially. The multi-point temporal extraction hypothesis failed.

**Analysis**:

1. **Why MultiPointTemporalExtraction failed**: The four Conv2D blocks in this architecture process features along the TEMPORAL dimension with kernels of size (3,1), (7,1), (15,1), and (1, n_features). The stacked temporal receptive field is at most 3+7+15 = 25 consecutive timesteps (not dilated). Extracting at t=−61 and t=−121 from this feature map gives the dense head a representation of local feature interactions within ~25 steps of those positions — **not** a 60- or 120-minute historical context. The model could not replicate the explicit `temp_lag60`/`temp_lag120` signal through the Conv2D extraction because the convolutions' receptive field never reaches those timesteps.

2. **GAP was doing real work**: GlobalAveragePooling2D averaged the local-neighborhood conv output across all 180 timesteps. This gave the context vector a rough temporal summary — imperfect but effective. Replacing GAP with point extractions from a shallow-receptive-field conv stack removed even that averaging signal, leaving only 3 isolated 25-step windows. The context path now contributes less temporal information than before, forcing the model to rely entirely on the anchor path (current-timestep features).

3. **Early convergence (best epoch 21)**: With a weakened context path, the model converges early using only the anchor path's current-timestep features (including the precomputed `temp_lag60`/`temp_lag120`). There is no meaningful gradient pressure from the conv path to keep training beneficial past epoch 21. Same symptom as Exp 33's epoch-19 shortcut.

4. **Feature importance compression (0.0405–0.0676)**: The 192-dim MultiPointExtraction → Dense(64) projection blends three ~25-step neighborhood representations into a context that carries redundant or conflicting signals. The gradient distributes nearly uniformly across all features because the context path cannot differentiate meaningful temporal patterns over 60–120 min horizons.

5. **EXP_NAME bug fix confirmed working**: The tracker correctly identified `conv2d_exp34_run1` as the best run and copied the right model. The Exp 33 bug (stale filename causing wrong model to be promoted) is resolved.

6. **Path forward**: Revert to GAP (which proved effective for temporal summarization in Exp 32/33) and address the real bottleneck — insufficient Conv2D capacity. Wider filters (64 → 96) give the conv blocks more representational power to distinguish temporal patterns and should prevent the easy early-epoch shortcut through the anchor path.

---

## Experiment 35: Revert GAP + Wider Conv2D Filters (96) + Longer ReduceLR Patience

**Date**: 2026-05-26

### Context — Recovering from Exp 34 regression

Exp 34 proved that `MultiPointTemporalExtraction` is structurally incompatible with this Conv2D architecture. The four conv blocks (kernels 3/7/15/n_features) have a stacked temporal receptive field of only ~25 steps. Extracting the feature map at t=−61 and t=−121 gives local-neighborhood context around those positions, not 60/120-minute history. `GlobalAveragePooling2D` was doing genuine temporal summarization across all 180 timesteps; removing it caused the 0.0024 → 0.0104 regression. Exp 35 reverts GAP and attacks the real bottleneck: insufficient Conv2D representational capacity.

### What changed and why

**1. Revert to GlobalAveragePooling2D (from Exp 34's MultiPointTemporalExtraction)**

GAP averages all 180 positions of the Conv2D output, giving the context vector a global temporal summary. Imperfect — but necessary given the conv blocks' limited 25-step receptive field. The Exp 34 regression confirmed that point extraction from a shallow-receptive-field stack is worse than full averaging.

**2. Widen Conv2D filters: 64 → 96**

The Exp 32 analysis identified "more Conv2D capacity" as the primary lever for breaking through the 0.0024 ceiling. With 64 filters, the four conv blocks compress 27 features × 180 timesteps into a 64-dim GAP vector — a severe bottleneck. Widening to 96 filters (+50%) increases the GAP output to 96-dim before the Dense(64) projection, giving the optimizer more room to encode distinct temporal patterns and raising the gradient pressure on the conv path relative to the anchor path. Higher conv path gradient pressure means the model is less likely to shortcut through the anchor alone, which is what caused the epoch-19/21 early convergence in Exp 33/34.

**3. ReduceLROnPlateau patience: 8 → 12**

In Exp 32, the LR fired only twice (patience=8, LR ended at 2.5e-5) before hitting the 100-epoch cap. Raising max_epochs to 150 (Exp 33) helped, but patience=8 still fires quickly on noisy val_loss curves, exhausting the LR budget before the model has converged. Patience=12 gives the optimizer 12 stable epochs at each LR level before cutting — the same extended-patience strategy that drove Model 5a Clean from 0.001069 → 0.000440 (58% improvement) by allowing the optimizer to fully exploit each learning rate before stepping down.

### Architecture

```
Input: (180, 27) → Reshape to (180, 27, 1)
  ├─ Conv2D path:  [Conv2D(96, k=(3,1),  L2=1e-4) → BN → ReLU6
  │                 → Conv2D(96, k=(7,1),  L2=1e-4) → BN → ReLU6
  │                 → Conv2D(96, k=(15,1), L2=1e-4) → BN → ReLU6
  │                 → Conv2D(96, k=(1,27), L2=1e-4) → BN → ReLU6]  ← (180, 1, 96)
  │                → GlobalAveragePooling2D → (96,)                 ← reverted from Exp 34
  │                → Dense(64, L2=1e-4) → ReLU6 → context(64)
  └─ Anchor path:  input[:, -1, :] → STRIDED_SLICE ✅ (27 features)
                   → Dense(32, L2=1e-4) → ReLU6 → anchors(32)
Concatenate([context(64), anchors(32)]) → Dense(32, L2=1e-4) → ReLU6 → Dense(3) outputs
```

### Changes from Exp 34

| Parameter | Exp 34 | Exp 35 |
|-----------|--------|--------|
| FILTERS | 64 | **96** |
| Temporal context method | MultiPointTemporalExtraction (t=−1/−61/−121) | **GlobalAveragePooling2D** |
| ReduceLR patience | 8 | **12** |
| Starting LR | 1e-4 | 1e-4 (same) |
| L2 | 1e-4 | 1e-4 (same) |
| EarlyStopping patience | 30 | 30 (same) |
| max_epochs | 150 | 150 (same) |
| Anchor path | input[:, −1, :] → Dense(32) | input[:, −1, :] → Dense(32) (same) |

### Expected Edge TPU ops

Back to the GAP-based op set (no RESHAPE×2 or STRIDED_SLICE×4 from Exp 34):

| Op | Count | Status |
|----|-------|--------|
| CONV_2D | 4 | Edge TPU ✅ |
| MEAN (GAP2D) | 1 | Edge TPU ✅ |
| RESHAPE | 1 | Edge TPU ✅ |
| FULLY_CONNECTED | 6 | Edge TPU ✅ |
| CONCATENATION | 1 | Edge TPU ✅ |
| STRIDED_SLICE | 1 | Edge TPU ✅ |

Expected: **single subgraph, all 14 ops on Edge TPU** (same as Exp 33's expected result; confirmed single-subgraph mapping now that GATHER is gone).

### Success criteria

- float val_loss < 0.0024 (beat Exp 32 — GAP revert + wider filters should recover and improve)
- best epoch > 30 (wider filters slow the optimizer; more gradient pressure on conv path)
- `temp_lag120` back in top 3 feature importance (was #1 in Exp 32 at 0.1025; GAP + anchor combination should restore this)
- All ops on Edge TPU, single subgraph

### Training Hang Investigation (2026-05-27 – 2026-05-28)

Exp 35 could not complete training. Six distinct bugs were found and fixed across two days. The unifying theme: every GPU tensor read or write that occurs at an epoch boundary (in a callback's `on_epoch_end`) accumulates state in the macOS Metal command queue. Beyond a threshold, the first GPU op of the next epoch deadlocks.

**Metal GPU pipeline corruption — background**

Apple Developer Forums (thread/803658, thread/713944) confirm a known unfixed tensorflow-metal bug affecting all versions through v1.2.0 (2025): the Metal GPU command queue accumulates state across epochs and never resets within a process. `model.save()` (full H5) is the most destructive trigger — it forces a full GPU→CPU flush that permanently corrupts the pipeline. But `model.save_weights()` and `model.get_weights()` also read GPU tensors and contribute to the accumulation. Multiple users on Apple Developer Forums report CPU training is faster than Metal GPU for small-to-mid Conv2D models on M1 Pro. `FORCE_CPU=1` environment-variable path added for comparison.

---

**Bug 1 — Glob matches `.weights.h5` as full-model checkpoint (prevents reliable resume)**

`glob.glob("model_epoch_*.h5")` also matches `model_epoch_40_batch_1124.weights.h5` (a weights-only file) because `*.h5` is a trailing suffix of `.weights.h5`. The checkpoint loader then tried `tf.keras.models.load_model()` on a weights-only file, received "No model config found", and fell back to training from epoch 0.

Fix: filter the glob result to exclude `.weights.h5` files:
```python
full_model_checkpoints = [
    f for f in glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.h5"))
    if not f.endswith('.weights.h5')
]
```

---

**Bug 2 — `PeriodicModelCheckpoint.on_train_batch_end` calls `model.save()` mid-epoch**

At batch N−20 of every epoch, `model.save()` (full H5) was called. On macOS Metal this corrupts the GPU command queue for all remaining batches in that epoch and accumulates across epochs, producing the observed geometric slowdown (≈1.78× per epoch: 205ms → 366ms → 652ms → 2000ms+/step).

Initial attempted fix was `save_near_epoch_end=False`. However that left the `on_epoch_end` path active (see Bug 3), so the class was ultimately deleted entirely.

---

**Bug 3 — `PeriodicModelCheckpoint.on_epoch_end` calls `model.save()` every 5 epochs**

Even with mid-epoch saves disabled, `PeriodicModelCheckpoint.on_epoch_end` called `model.save()` when `(epoch + 1) % 5 == 0`. This fired at the end of epoch 4 (0-indexed), corrupting Metal. Batch 0 of epoch 5 deadlocked.

Symptom:
```
Epoch 4/150 — 1144/1144 — 249s 218ms/step   ← completes fine
Epoch 5/150 —    0/1144   ← hangs, watchdog kills after 30 min
```

Fix: `PeriodicModelCheckpoint` class deleted entirely (~95 lines). `LatestEpochSaver` covers every epoch with `save_weights()` only.

---

**Bug 4 — Pre-training validation step 5 calls `model.save()` + `load_model()` before training**

Step 5 of the pre-training validation routine called `model.save()` (full H5) to verify serialization, then `tf.keras.models.load_model()` to verify reload. Both corrupt Metal. Observed as epoch 1 starting at 2s/step on restarts where the GPU was already in a degraded state from a prior hang.

Fix first attempted: replace with `model.save_weights()` + h5py readability check. This still read GPU tensors, causing epoch 5 to start at 787ms/step instead of 218ms/step.

Final fix: step 5 removed entirely. Training-loop checkpoint saves have already proven reliable; a pre-flight write test adds no safety benefit and costs a GPU read.

---

**Bug 5 — "Your input ran out of data" — Keras 3 EpochIterator with finite dataset**

After resuming from checkpoint at epoch 39, training stopped after ~10 batches/epoch:
```
111/1144 [...] UserWarning: Your input ran out of data; interrupting training.
```

Keras 3's `EpochIterator` creates one iterator for ALL remaining epochs at `model.fit()` start. With a finite dataset (1144 batches) and `initial_epoch=39`, it allocated 1144 ÷ 111 remaining epochs ≈ 10 batches/epoch.

Fix: capture cardinality before `.repeat()`, pass `steps_per_epoch` to `model.fit()`:
```python
train_steps = int(train_ds.cardinality().numpy())   # 1144 — before .repeat()
train_ds = train_ds.repeat()                         # cardinality → -1 (infinite)
train_ds = train_ds.prefetch(buffer_size=4)
history = model.fit(train_ds, steps_per_epoch=train_steps, ...)
```

Note: Keras 3 still emits the "ran out of data" warning even with `.repeat()` — spurious diagnostic fired at iterator creation. Confirmed benign: epochs complete all 1144 batches after this fix.

---

**Bug 6 — Two `ModelCheckpoint` callbacks + `CheckpointValidationCallback` accumulate GPU reads (epoch 6 batch-0 hang)**

After removing `PeriodicModelCheckpoint`, epoch 5 completed but epoch 6 hung at batch 0. Three remaining GPU interactions at every epoch-end:

1. `checkpoint_cb` (`ModelCheckpoint`, `save_weights_only=True`) — fires `save_weights()` when `val_loss` improves. On every process restart, Keras resets `.best` to `float('inf')`, so the first epoch after any restart always triggers this save unconditionally.
2. `best_full_model_cb` (identical, different filename) — same unconditional save on restart.
3. `CheckpointValidationCallback.on_epoch_end` — its preamble called `model.get_weights()` on every epoch to save weights for potential restoration, even for the weights-only h5py path that never modifies model weights.

On the first epoch after each restart: 3 extra GPU tensor reads in addition to the 1 from `LatestEpochSaver` = 4 total. This accumulated enough Metal command-queue state to deadlock batch 0 of the next epoch.

Observed symptom:
```
Epoch 5/150 — 1144/1144 — 910s 787ms/step   ← slow start from pre-training GPU ops
  ✅ All checkpoints validated successfully   ← CheckpointValidationCallback fires
Epoch 6/150 —    0/1144   ← hangs, watchdog kills after 30 min
```

Fix:
- `checkpoint_cb` and `best_full_model_cb` removed from `active_callbacks`. Best-model tracking absorbed into `LatestEpochSaver`: after the single `save_weights()` call, if `val_loss < self.best_val_loss`, copies the file to `best_model.weights.h5` via `shutil.copy2` (pure disk I/O, no GPU). `best_val_loss` seeded from `early_stopping.best` at instantiation so the threshold persists across restarts.
- `CheckpointValidationCallback` class deleted. `model.get_weights()` in `validate_checkpoint_loading` moved inside the `is_full_model` branch only; the weights-only path is now pure h5py with zero GPU ops.

**Result: epoch-end GPU interactions reduced from 4 to exactly 1** — the single `save_weights()` in `LatestEpochSaver`.

After this fix, training survived 2 epochs (6 and 7) before hanging at epoch 8 batch 0 — up from 1 epoch. This confirmed that fewer GPU reads per epoch delays but does not eliminate the hang. Even 1 `save_weights()` per epoch is enough to eventually deadlock on Metal. See Bug 7.

---

**Fix — `LatestEpochSaver` callback (per-epoch resume + best-model tracking)**

`ModelCheckpoint(save_best_only=True)` only updates when `val_loss` improves. After hanging at epoch 4 with best epoch = 1, resume restarted from epoch 1, discarding epochs 2–3. `LatestEpochSaver` saves `model_latest.weights.h5` + `model_latest_epoch.json` after every epoch. Inserted as Priority 2 in the checkpoint loading chain (after any surviving full-model `.h5`, before best-weights).

After Bug 6 was identified, `LatestEpochSaver` was also expanded to replace the two `ModelCheckpoint` callbacks: it copies `model_latest.weights.h5` → `best_model.weights.h5` when `val_loss` improves, keeping best-model tracking as pure file I/O with no second GPU call.

**Final GPU interaction count per epoch-end: 1.**

---

---

**Bug 7 — Metal command-queue accumulation is unavoidable within a single process**

Even with epoch-end GPU interactions reduced to 1 (`save_weights()` in `LatestEpochSaver`), training survived epochs 6 and 7 before hanging at epoch 8 batch 0. The `save_weights()` is not optional — it is how the checkpoint is written. The Metal GPU bug therefore cannot be worked around purely within the epoch-end callback layer.

Root cause: every `save_weights()` call reads GPU tensor data, which interacts with the Metal command queue. Over multiple epochs, this interaction accumulates enough state to deadlock the first GPU op of the subsequent epoch. Fewer reads per epoch delay the deadlock but don't prevent it.

Fix: **exit the process cleanly after each epoch** (`MaxEpochsPerRun` callback, exit code 42). The OS gives the next process a completely fresh Metal GPU context, so the accumulation resets to zero every epoch.

`MaxEpochsPerRun(max_epochs_per_run=1)` is added as the last training callback (after `reduce_lr` and `lr_state_saver`). When it fires, it sets `model.stop_training = True` and marks `triggered = True`. After `model.fit()` returns, the script checks `max_epochs_per_run.triggered` and calls `sys.exit(42)`. `train_loop.sh` treats exit code 42 as "restart immediately" and relaunches the script, which loads from `model_latest.weights.h5` (Priority 2) and trains the next epoch.

Additional fix: `lr_state_saver` reordered to run **after** `reduce_lr` in the callback list. Previously it ran before `reduce_lr`, so the saved `best` and `wait` values were always one epoch stale — observed as `best=inf` every restart regardless of actual training history. Now the saved state correctly reflects the current epoch's ReduceLR decision.

**train_loop.sh** — restart loop script created:
```
exit 0  → training complete, stop
exit 42 → per-epoch Metal reset, restart immediately
exit 99 → watchdog hang, pause 10s then restart
other   → unexpected error, stop
```

**Complete history of GPU-touching call sites, in order of discovery:**

| Call site | GPU op | When | Observed effect | Fix |
|-----------|--------|------|-----------------|-----|
| `PeriodicModelCheckpoint.on_train_batch_end` | `model.save()` | Batch N−20, every epoch | Geometric slowdown 1.78×/epoch | Deleted class |
| `PeriodicModelCheckpoint.on_epoch_end` | `model.save()` | Epochs 5, 10, 15, … | Epoch N+1 batch-0 deadlock | Deleted class |
| Pre-training validation step 5 (original) | `model.save()` + `load_model()` | Every restart | First epoch at 2s/step | Removed |
| Pre-training validation step 5 (revised) | `save_weights()` | Every restart | First epoch at 787ms/step | Removed entirely |
| `ModelCheckpoint checkpoint_cb` | `save_weights()` | Epoch end, always fires on restart (best resets to inf) | +1 GPU read per epoch-end | Merged into `LatestEpochSaver` |
| `ModelCheckpoint best_full_model_cb` | `save_weights()` | Same as above | +1 GPU read per epoch-end | Merged into `LatestEpochSaver` |
| `CheckpointValidationCallback` preamble | `model.get_weights()` | Every epoch end | +1 GPU read per epoch-end | Deleted class |

**Why Exp 35 hangs but earlier experiments did not**

Exp 35 uses FILTERS=96 (vs 64 in Exp 32–34). Each `save_weights()` call copies proportionally more GPU tensor data, accelerating Metal command-queue accumulation and reaching the deadlock threshold in fewer epochs. Earlier experiments with FILTERS=64 either fell below the threshold within their training window, or were run at different times when the GPU thermal state was more favourable.

A secondary factor: macOS reduces GPU clock speed when the display sleeps (screen lock). Combined with TF-Metal's command-queue accumulation, operations that complete normally with an active display can deadlock with a locked screen. Using `caffeinate -d -i` prevents both display sleep and system idle sleep while allowing manual screen lock, which may allow multi-epoch runs. The `max_epochs_per_run` parameter in the code is the tuning knob: 1 is safe for any screen-lock scenario; higher values can be tried when running with `caffeinate`.

**Actual overhead of the 1-epoch-per-run approach (observed Exp 35, epoch 14):**

| Phase | Duration |
|-------|----------|
| Metal JIT shader warmup (~74 batches at 3s/step) | ~4 min |
| Steady-state training (1070 batches at ~280ms/step) | ~5 min |
| Checkpoint load + validation suite (per restart) | ~2 min |
| **Total per epoch** | **~11 min** |

Compared to the previous hang-and-restart pattern (2 epochs + 30 min watchdog wait = 47 min per 2 epochs = 23.5 min/epoch), the new approach is roughly **50% faster** in wall-clock time, fully automated, and never loses an epoch to a hang.

**Bug 8 — Overnight GPU degradation: Metal JIT warmup never completes (epoch 32, runs #46–#49)**

After ~31 hours of continuous overnight training (epochs 1–31, ending around 3–5 AM), training entered an unrecoverable loop at epoch 32. Runs #46, #47, #48, and #49 all hung in mid-epoch at batches 78, 18, 30, and 22 respectively. Batch speed was permanently stuck at 2–2.5s/step — exactly the JIT shader warmup rate — and never transitioned to normal training speed (250–450ms/step).

This is a distinct failure pattern from the epoch-boundary hangs (Bugs 1–7):

| Pattern | Location | Speed at hang | Cause |
|---------|----------|---------------|-------|
| Bugs 1–7 (epoch-boundary) | Batch 0 of next epoch | High latency or deadlock | Metal command-queue accumulation |
| Bug 8 (mid-epoch degradation) | Batch N (N = 18–78) of current epoch | 2–2.5s/step stuck | GPU hardware clock throttled |

Root cause: macOS aggressively throttles the GPU clock during extended locked-screen operation. After several hours at reduced clock speed, the Metal pipeline enters a state where JIT shader compilation for new processes never completes normally — each new process gets a fresh Metal *context* (command-queue accumulation resets to zero) but the GPU hardware itself remains throttled. Every new `train_loop.sh` restart triggers another failed warmup rather than fixing the issue.

The per-epoch restart strategy (MaxEpochsPerRun + exit code 42) **cannot fix this** — it resets Metal process state, not GPU hardware thermal/clock state.

Fixes (in order of preference):
1. **`FORCE_CPU=1 ./train_loop.sh`** — bypass Metal entirely; CPU training is slower (~15–20 min/epoch vs ~11 min/epoch on GPU) but unaffected by GPU degradation. Resumes from `model_latest.weights.h5` at epoch 31, `best_val_loss=0.025739`.
2. **Unlock screen + wait** — unlock macOS display lock, run `caffeinate -d -i &`, wait 5–10 minutes for GPU clock to recover, then restart `train_loop.sh` without `FORCE_CPU`.
3. **Reboot** — most reliable full GPU reset; adds ~2 min delay.

Prevention: always run `caffeinate -d -i` in a terminal before starting an overnight training session. This prevents display sleep (the macOS setting `System Settings → Displays → Prevent automatic sleeping on power adapter when the display is off` also helps). Manual screen lock is still possible without triggering GPU throttling if `caffeinate` is running.

---

### Results (Kaggle, `conv2d_exp35_run1`)

**Training outcome**:
- Ran to epoch 124, early stopped
- Best epoch: **8** — success criterion was > 30 ❌
- val_loss: **0.002368** — success criterion was < 0.0024 ❌ (essentially tied with Exp 32)
- val_mae: 0.006582 (normalized)
- Model size: 469.06 KB

**Edge TPU compilation** (compiled locally from `Kaggle/results_finished/`):
- Compiled successfully in 346 ms ✅
- Single subgraph, **all 14 ops on Edge TPU** ✅
- On-chip memory used: 626.50 KiB (well within 8 MB SRAM)
- Off-chip streaming: 0 B
- Output model: 672.70 KiB

| Op | Count | Status |
|----|-------|--------|
| FULLY_CONNECTED | 6 | Edge TPU ✅ |
| CONV_2D | 4 | Edge TPU ✅ |
| MEAN (GAP) | 1 | Edge TPU ✅ |
| RESHAPE | 1 | Edge TPU ✅ |
| STRIDED_SLICE | 1 | Edge TPU ✅ |
| CONCATENATION | 1 | Edge TPU ✅ |

**Feature importance** (top 5 and bottom):

| Feature | Importance |
|---------|-----------|
| `temp_lag60` | 0.0909 |
| `temp_lag120` | 0.0894 |
| `temperature` | 0.0877 |
| `temp_slope_60` | 0.0833 |
| `time_of_day_sin2` | 0.0826 |
| … | … |
| `wind_direction_sin` | 0.0778 |
| `uv` | 0.0738 |
| `time_of_day_cos2` | 0.0710 |

Significant positive shift: temperature features now occupy the top 3 positions (vs `time_of_day` dominating in all prior Conv1D experiments, Exp 9–25). However the overall spread is extremely flat (0.071–0.091 across 27 features); Model 5a shows sharp dominance of `temp_lag120` at 0.093.

**Outcome**: ❌ **FAILED** — neither success criterion met. val_loss (0.002368) is essentially identical to Exp 32 (0.0024), confirming the Conv2D + GAP architecture has a hard ceiling. best_epoch=8 despite patience=12 and 96 filters — the optimizer finds a local optimum via the anchor path and the Conv2D path adds nothing further.

**Analysis**:

1. **GAP ceiling confirmed**: Widening filters 64 → 96 and raising ReduceLR patience 8 → 12 produced zero measurable improvement. The bottleneck is not capacity or LR schedule — it is the information loss from GlobalAveragePooling2D averaging all 180 positions into a single 96-dim vector with no positional identity. The model cannot learn "temp at t−120 was X" via GAP; it can only learn "some pattern occurred somewhere in the 3-hour window."

2. **best_epoch=8 reveals anchor shortcutting**: The anchor path (Dense(32) on current-state features) converges within a few epochs because it has direct access to `temp_lag60` and `temp_lag120`. The Conv2D path receives weak gradient pressure once the anchor "solves" the problem. It never develops into a genuine temporal feature extractor.

3. **Core structural redundancy**: The Conv2D path sees `temp_lag60` and `temp_lag120` as explicit columns in every one of the 180 timesteps of the `(180, 27)` input. There is no incentive for the Conv2D to learn temporal dynamics from first principles — the answer is already encoded in the input. The Conv2D essentially re-extracts information the anchor already has more efficiently.

4. **Temporal receptive field**: The four Conv2D kernels `(3,1)→(7,1)→(15,1)→(1,27)` give a combined temporal reach of only ~25 steps (~25 minutes). Patterns at t−60 or t−120 are outside the Conv2D's visible window unless they happen to appear in the explicit lag columns of nearby timesteps.

---

## Kaggle Migration — Platform Context

**Date:** 2026-06-01  
**Context:** Training infrastructure shift for Experiment 35 (and future experiments)

### Why Kaggle

Mac training with `FILTERS=96` (Exp 35) ran at ~940 s/epoch (~11 min with JIT warmup overhead). The per-epoch Metal restart workaround (Bug 7) plus the GPU throttling failure after ~31 hours (Bug 8) made multi-day overnight runs unreliable. The Coral USB TPU and Hailo-8 HAT are inference-only accelerators; neither can run TF training. Kaggle's free GPU tier provides T4 ×2 with CUDA — no Metal quirks, no per-epoch restart needed, and 12 GPU-hours per session (30 GPU-hours/week). At ~18.4 min/epoch on Kaggle, ~38 epochs fit in a single 12-hour session vs 12–15 on Mac.

### Changes to `train_model_conv2D.py`

**1. `KAGGLE_MODE` flag and dataset path (top of file)**
```python
KAGGLE_MODE = True   # set False for local Mac training
KAGGLE_DATASET = "datasets/dacarson/weatherml-training-data"
```
Controls all Kaggle-specific branches below.

**2. Data directory**
```python
if KAGGLE_MODE:
    data_dir = f"/kaggle/input/{KAGGLE_DATASET}"
else:
    data_dir = ".."
```
Training data is mounted read-only under `/kaggle/input/` as a Kaggle Dataset.

**3. `steps_per_execution=1` (unconditional — both compile calls)**

Initially set to `50 if KAGGLE_MODE else 1` to amortise Python↔GPU dispatch overhead. On T4/CUDA, TF compiles the 50-step dispatch as an unrolled XLA while-loop graph before any step executes. This compilation never completed in reasonable time — training appeared to hang for 20+ minutes after "Epoch 1/150" was printed, with no batches completing.

Fix: `steps_per_execution=1` unconditionally in both `model.compile()` calls (main compile ~line 1384; QAT compile ~line 1761). On CUDA, Python↔GPU dispatch overhead is negligible at N=1, so there is no throughput cost. On Metal, this was already 1 for stability.

**4. Prefetch buffer**
```python
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE if KAGGLE_MODE else 4)
val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE if KAGGLE_MODE else 4)
```
`AUTOTUNE` allows TF to determine optimal prefetch depth for T4 I/O. On Mac, `4` was chosen empirically to avoid Metal pipeline stalls.

**5. `MaxEpochsPerRun` — disabled on Kaggle**
```python
max_epochs_per_run = MaxEpochsPerRun(max_epochs_per_run=1 if not KAGGLE_MODE else 9999)
```
The per-epoch OS restart (exit code 42 + `train_loop.sh`) was the Metal command-queue accumulation workaround (Bug 7). CUDA/T4 has no equivalent issue; training runs continuously for the full session.

**6. Conditional `tfmot` import**
```python
ENABLE_QAT = False
if ENABLE_QAT:
    import tensorflow_model_optimization as tfmot
```
`tensorflow-model-optimization` is not installed in the default Kaggle TF environment. Moving the import inside `if ENABLE_QAT:` prevents an `ImportError` at notebook start when QAT is disabled.

**7. `MirroredStrategy` for both T4s**
```python
_n_gpus = len(tf.config.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy() if _n_gpus > 1 else tf.distribute.get_strategy()
if _n_gpus > 1:
    print(f"Using MirroredStrategy across {_n_gpus} GPUs")

with strategy.scope():
    # all Input, Conv2D, BatchNorm, Dense, model = Model(...) definitions
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(...)
```
Without `MirroredStrategy`, TF defaulted to GPU 0 only (13.6 GiB used, GPU 1 idle at 105 MiB). Wrapping all layer construction and `model.compile()` inside `with strategy.scope():` enables NCCL AllReduce across both T4s. Observed speedup: ~20% (not 2×) — gradient sync overhead limits linear scaling for this model size.

### Errors Encountered and Fixed

| Error | Cause | Fix |
|-------|-------|-----|
| `ResourceExhaustedError` OOM (15 GB) | `TRAIN_BATCH_SIZE=1024` + `steps_per_execution=50` — Conv2D backprop tensor `f32[96,1024,180,27]` ≈ 1.9 GB × 50-step XLA graph exceeded 16 GB VRAM | Reverted to `TRAIN_BATCH_SIZE=512`, then set `steps_per_execution=1` |
| 20+ min hang after "Epoch 1/150" | `steps_per_execution=50` triggered XLA 50-step while-loop graph compilation before first step | `steps_per_execution=1` unconditionally |
| Only GPU 0 used (GPU 1 idle) | No `MirroredStrategy` — TF defaulted to first GPU | Added `MirroredStrategy` + `with strategy.scope():` |
| `ConcurrencyViolation` (ExpectedSequence vs ActualSequence) | Auto-save sequence desync after force-stopping a session | Hard refresh (Cmd+Shift+R), close duplicate tabs |
| Overnight results lost | Training run via "Execute Cell" (interactive) — `/kaggle/working/` outputs are ephemeral in interactive sessions | Use "Save Version → Save & Run All" for any run whose outputs must persist; monitor via Logs tab |

### Kaggle-Specific Workflow Notes

- **Interactive "Execute Cell" sessions**: outputs written to `/kaggle/working/` are discarded when the session ends. Nothing persists.
- **"Save Version → Save & Run All"**: runs the notebook in a clean VM, writes outputs to the version's permanent storage. These survive session end and can be downloaded from the Output tab.
- **Monitoring a Save & Run All run**: open the notebook version → Logs tab. The log auto-updates in real time. GPU panel shows live utilisation.
- **Session limit**: 12 hours per Save & Run All run; 30 GPU-hours/week total (free tier).

### Confirmed Training Performance (both T4s, NCCL, `FILTERS=96`, `TRAIN_BATCH_SIZE=512`)

| Metric | Value |
|--------|-------|
| Step time (steady state) | ~831 ms/step |
| Steps per epoch | 1,144 |
| Training time/epoch | ~950 s |
| Validation time/epoch | ~228 s |
| **Total per epoch** | **~18.4 min** |
| Epochs in 12-hour session | ~38 |
| GPU utilisation | Both T4s at 100% |
| Gradient sync | NCCL (peer-to-peer) |

Epoch 12 loss was ~0.0052; val_loss trajectory consistent with previous Mac runs before GPU throttling forced a platform change.

---

## Experiment 36: Conv2D Pattern Analysis — What Has the Model Actually Learned?

**Date**: 2026-06-07  
**Goal**: Understand what temporal and cross-feature patterns the Exp 35 Conv2D filters have learned, and what they are structurally capable of learning, before committing to the next architecture change.

### Motivation

After 35 experiments, the Conv2D + GAP architecture has a hard ceiling at ~0.0024 val_loss — 3.5× short of Model 5a's 0.000682. Before abandoning or redesigning the Conv2D path, it is worth empirically verifying what the filters have converged to. Two competing hypotheses need to be distinguished:

- **Hypothesis A**: The Conv2D filters have learned meaningful local temporal patterns (rate-of-change detection, cross-feature correlations, threshold crossings) but GAP is discarding the positional information needed to make them useful for output prediction. The Conv2D is doing real work; the problem is downstream.
- **Hypothesis B**: The Conv2D filters have collapsed to generic smoothing kernels. Because `temp_lag60`, `temp_lag120`, and slope features are already explicit in every timestep of the input, the network has no incentive to develop independent temporal representations. The Conv2D is redundant.

The two-stream architecture proposed for Experiment 37 is only worth building if Hypothesis A is at least partially true. If the filters are generic noise, the architecture needs a different inductive bias entirely.

### Analysis Script: `analyze_conv2d_patterns.py`

A standalone script that loads the Exp 35 weights and runs three analyses:

**1. Filter weight visualisation** — what does each layer respond to?

Each Conv2D layer's learned filters are 1-D temporal kernels (applied feature-by-feature):
- `conv2d_1` (k=3,1): 96 filters of shape (3, 1, 1, 96) — 3-timestep temporal patterns per input channel
- `conv2d_2` (k=7,1): 96 filters of shape (7, 1, 96, 96) — 7-timestep patterns on 96 input channels
- `conv2d_3` (k=15,1): 96 filters of shape (15, 1, 96, 96) — 15-timestep patterns on 96 input channels
- `conv2d_4` (k=1,27): 96 filters of shape (1, 27, 96, 96) — cross-feature mixing at a single timestep

For `conv2d_1`, plot the 96 temporal weight vectors as a heatmap (time × filter_index). Clusters of similar filters indicate learned pattern families (rising edge, falling edge, oscillation, plateau). Uniformly flat or random filters indicate the layer has not learned structured patterns.

For `conv2d_4`, plot the 96 weight matrices (27 features × 96 output channels) as a heatmap. Bright rows indicate features the network mixes heavily into the learned representation; uniform brightness indicates no feature selection.

**2. Activation map visualisation** — which time positions fire on real data?

Load 3–5 representative 3-hour windows from the San Francisco training data:
- Clear sunny day (strong diurnal signal)
- Foggy morning burn-off (temperature step-change)
- Frontal passage (rapid multi-sensor changes)

For each window, extract the output of `conv2d_3` (the last temporal Conv2D) before GAP, shape `(180, 1, 96)`. Reshape to `(180, 96)` and plot as a time × channel heatmap. This reveals whether the model activates strongly at specific time positions (evidence of temporal localisation) or uniformly across all 180 timesteps (evidence that GAP is working as intended — or that no position matters more than another).

If activations are concentrated near t−60 and t−120, the model is using the explicit lag columns to re-create lag-detection filters. If activations are concentrated near t=0 (the most recent timestep), the anchor path has suppressed gradient flow to the conv path entirely. If activations are broadly uniform, GAP is discarding the temporal structure without loss.

**3. Input saliency maps** — which (time, feature) cells drive predictions?

Compute `∂output_1hr / ∂input` via `tf.GradientTape` for each representative window. The gradient has shape `(180, 27)` — one value per (timestep, feature) cell. Plot as a heatmap. The time axis reveals which minutes the model actually attends to; the feature axis reveals which sensors are driving each prediction horizon.

Expected outcomes under each hypothesis:
- **Hypothesis A** (Conv2D doing real work): saliency spreads across multiple time positions and sensors, including positions not covered by explicit lag features (e.g., t−30, t−45, t−90)
- **Hypothesis B** (Conv2D redundant): saliency is concentrated entirely at t=−1 (anchor features) and at the specific timesteps corresponding to t−60 and t−120 (the explicit lag columns) — everywhere else is near zero

### What the Results Will Determine

| Finding | Implication |
|---------|-------------|
| Filters show structured temporal patterns (edges, oscillations) | Hypothesis A — Conv2D architecture is worth extending |
| Filters are uniform / random | Hypothesis B — consider replacing Conv2D path with Dense |
| Activations concentrated at t−60, t−120 positions | Conv2D is re-learning the explicit lag features; separate the streams |
| Saliency spreads across many timesteps | Conv2D is capturing additional temporal context beyond lags |
| Saliency concentrated at t=0 only | Anchor path is dominating; Conv2D path is not contributing |

### Results (2026-06-07)

**Filter weights** (`01_filter_weights.png`):

| Layer | Dead filters (L2 < 0.01) | % dead |
|-------|--------------------------|--------|
| conv2d_t1 (3-step) | 56 / 96 | **58%** |
| conv2d_t2 (7-step) | 61 / 96 | **64%** |
| conv2d_t3 (15-step) | 30 / 96 | 31% |
| conv2d_feat (cross-feature) | 0 / 96 | 0% |

conv2d_t1 kernel stats: mean=0.0004, std=0.0303, range [−0.12, +0.12] — weights are tiny compared to a well-trained conv. The temporal layers have largely collapsed. `conv2d_feat` is fully alive because the anchor path's gradient flows directly through the current-timestep features at that layer.

**Activation maps** (`02_activation_maps.png`):

Mean activation at key time positions in the `relu_feat` layer:

| Scenario | t=−180 | t=0 | t=−60 | t=−120 |
|----------|--------|-----|-------|--------|
| warm_up | **0.992** | 0.782 | 0.399 | 0.174 |
| cool_down | **0.902** | 1.210 | 0.267 | 0.281 |
| sunny | **0.802** | 0.746 | 0.251 | 0.196 |
| foggy | **0.975** | 0.694 | 0.162 | 0.151 |
| windy | **1.331** | 0.652 | 0.391 | 0.602 |

The model activates most strongly at the **oldest timestep (t=−180)** in all scenarios. Activations at t=−60 and t=−120 are low despite those being the positions of the explicit lag features. The middle of the window is essentially silent. The Conv2D has learned to attend to the **endpoints** of the sequence (oldest + newest) and ignore the 178 positions in between.

**Input saliency** (`03_saliency_maps.png`, `04_mean_saliency.png`):

Top-5 saliency cells per horizon:

| Horizon | #1 | #2 | #3 |
|---------|-----|-----|-----|
| Δ1hr | `temp_lag120` t=0 (3.16) | `temperature` t=0 (1.55) | `temperature` t=−179 (1.40) |
| Δ2hr | `temp_lag60` t=0 (1.48) | `temperature` t=−179 (1.33) | `temp_lag60` t=−1 (0.70) |
| Δ3hr | `temperature` t=−179 (1.48) | `temp_delta_1` t=0 (0.67) | `temperature` t=0 (0.59) |

**Verdict — Hypothesis B confirmed:**
- Δ1hr and Δ2hr are entirely driven by the anchor path at t=0 (`temp_lag120` and `temp_lag60` respectively). The Conv2D contributes nothing for these horizons.
- Δ3hr is the exception: the dominant saliency is `temperature` at t=−179 (the **oldest row in the window** = temperature from ~3 hours ago). The model has learned an implicit `temp_lag180` feature by reading the raw temperature value at the start of the sequence — information the anchor path does not have. This is the **one genuine Conv2D contribution**.

**Implication**: The Conv2D is doing exactly one useful thing: providing a 180-minute temperature anchor that the anchor path lacks. The fix is explicit: add `temp_lag180` as an input feature, giving the anchor a direct weight path to this value. This makes the Conv2D's only unique contribution redundant and removes the incentive for the model to keep reading the oldest sequence row.

---

## Experiment 37: Add `temp_lag180` Explicit Feature

**Date**: 2026-06-07  
**Goal**: Capture the one useful thing the Exp 35 Conv2D was doing (implicit 3-hr temperature anchor) as an explicit input feature, giving the anchor path a direct weight to it.

### Motivation (from Exp 36 analysis)

Exp 36 saliency showed:
- Δ1hr dominated by `temp_lag120` at t=0 — already explicit ✅
- Δ2hr dominated by `temp_lag60` at t=0 — already explicit ✅
- Δ3hr dominated by `temperature` at t=−179 (oldest sequence row) — **not explicit**

The Conv2D was implicitly constructing `temp_lag180` by attending to the oldest row of the `(180, 27)` input. Making it an explicit anchor feature:
1. Gives the anchor Dense(32) a direct weight path: `temp_lag180 → output_3hr`
2. Mirrors what Model 5a does with `temp_lag60`/`temp_lag120` but for the 3hr horizon
3. If Δ3hr accuracy improves, confirms the Conv2D's contribution was captured
4. Removes the one reason the model needed to read the oldest sequence row; Conv2D path now has no privileged information the anchor lacks

### Changes from Exp 35

**`train_model_conv2D.py`**:
1. Compute `temp_lag180` alongside `temp_lag60`/`temp_lag120` (same `_past_temp_at` helper, offset=180 min)
2. Add `temp_lag180` to `features` list (position 4, after `temp_lag120`)
3. Add `temp_lag180` domain bounds: `(-10, 55)` (same as `temperature`)
4. `EXP_NAME = "conv2d_exp37"`

No architecture changes. `n_features` goes from 27 → 28. The anchor `LastTimestepAnchorFeatures` automatically includes the new feature (returns all `n_features` at `input[:, -1, :]`). The Conv2D path also receives it in the sequence but is not expected to exploit it given the dead temporal filters.

### Success Criteria

- val_loss < 0.002368 (beat Exp 35)
- Δ3hr improvement: the 3hr head should benefit most from the explicit 180-min anchor
- best_epoch > 8 (if the new feature lets the model continue refining after initial convergence)
- `temp_lag180` ranks in top 5 feature importance (confirms the anchor is using it)

### Expected outcome

The Δ3hr output head should see the most improvement — it now has a direct `temp_lag180 → diff_3hr` weight path that previously required the Conv2D to extract it indirectly. Δ1hr and Δ2hr are unlikely to change significantly (their dominant signals were already explicit). Overall val_loss improvement depends on how much the Δ3hr head was being bottlenecked.

If val_loss does not improve, it indicates the Conv2D's implicit temp_lag180 extraction was not actually helping accuracy — likely because GAP was diluting it. In that case, proceed to Exp 38 (two-stream architecture).

### Kaggle T4 Infrastructure Tuning (applied before first Exp 37 run)

Three `KAGGLE_MODE`-gated performance changes made to `train_model_conv2D.py` to take advantage of the Tesla T4's Tensor Cores and CUDA backend. No architecture, feature, or hyperparameter changes.

| Change | Local (Metal) | Kaggle (T4/CUDA) | Reason |
|--------|--------------|-----------------|--------|
| Mixed precision | float32 | mixed_float16 | T4 Tensor Cores give ~2× FP16 speedup; `_unwrap_optimizer` already handles `LossScaleOptimizer`; output heads already `dtype="float32"` |
| XLA JIT | Off | On | Metal doesn't support XLA; CUDA does — fuses Conv2D+BN+ReLU6 into single kernels (~10–30% speedup) |
| Batch size | 512 | 1024 | Better T4 utilization; `steps_per_execution` stays 1 (XLA+steps>1 caused 20min CUDA compilation per earlier runs) |

LR unchanged at 1e-4: accepting the reduced gradient noise of a 2× batch as potentially stabilising rather than applying the linear scaling rule (which would muddle comparison with Exp 35 convergence curves).

### Kaggle Run 1 — NaN crash at epoch 38

Training progressed well through epoch 37 (val_diff_1hr_mae ≈ 0.0198 normalized ≈ 0.28°C at epoch 24, still improving). At epoch 38 all losses went NaN.

**Root cause:** FP16 gradient overflow. `mixed_float16` wraps Adam in `LossScaleOptimizer`, which grows its loss scale over epochs. Around epoch 38 the scale became large enough that a batch's gradients exceeded FP16's max (~65504), went Inf → NaN, and corrupted all weights.

`LatestEpochSaver` then wrote the NaN-corrupted weights to `model_latest.weights.h5`, which would be loaded on restart (Priority 2 in checkpoint loading). Recovery required deleting `model_latest.weights.h5` and `model_latest_epoch.json` so the loader falls back to `best_model.weights.h5` (last saved at the best pre-NaN epoch).

**Fix applied:** Added `global_clipnorm=1.0` to Adam when `KAGGLE_MODE`. Global gradient norm clipping caps the magnitude of the combined gradient vector before the optimizer update step, preventing FP16 overflow. No change to LR, architecture, or local training behaviour.

### Kaggle Run 2 — Completed (with gradient clipping + warm start)

**Setup**: `global_clipnorm=1.0` applied. `best_model.weights.h5` from Run 1 (pre-NaN, best epoch ~24) was published to the checkpoint dataset. Because `model_latest_epoch.json` was deleted (NaN recovery), the loader fell through to Priority 3 (`best_model.weights.h5`) and loaded it as a warm start with `initial_epoch=0`. ES and LR state were cleared (cold restart from epoch 1, warm weights).

**Training progress** (partial — epochs observed before completion):
| Epoch | val_loss | Notes |
|-------|----------|-------|
| 22 | 0.0103 | Early (warm start already better than cold init) |
| 29 | 0.0039 | Best observed mid-run; LR still at 1e-4 |
| 30 | 0.0041 | Slight regression (normal noise) |

**Final results** (training completed, EarlyStopping fired):
- Validation MAE (from `model.evaluate()` after `restore_best_weights`):
  - diff_1hr: 0.01°C, diff_2hr: 0.01°C, diff_3hr: 0.01°C
- Two T4 GPUs active via MirroredStrategy throughout
- Gradient clipping prevented another NaN crash ✅

**Feature importance** (top/bottom):
- **#1: `temp_lag180` (0.0810)** ← Exp 37 hypothesis confirmed: explicit lag180 is the strongest signal
- #2: `time_of_day_cos2` (0.0803), #3: `humidity_slope_30` (0.0770), #4: `temp_delta_1` (0.0754)
- Bottom: `temp_slope_60` (0.0693), `time_of_day_sin` (0.0721), `pressure_slope_60` (0.0722)
- Distribution very flat (range 0.069–0.081) — model uses all features roughly equally; no single dominant feature

**⚠️ NEW ISSUE: TFLite conversion failed — FP16 ops not supported by legacy converter**

```
error: 'tf.Conv2D' op is neither a custom op nor a flex op
TF Select ops: ConcatV2, Conv2D, MatMul, Relu6, StridedSlice
tf.Conv2D(tensor<1x180x28x1xf16>...) ← FP16 tensors
```

**Root cause**: `mixed_float16` training causes Keras to insert internal float16 casts. The traced `concrete_function` contains `tf.*` FP16 ops (`Conv2D`, `MatMul`, `Relu6`, `StridedSlice`, `ConcatV2`). The legacy TFLite converter (`experimental_new_converter=False`) cannot lower these to native TFLite ops.

**This is distinct from the Exp 19–21 `mixed_float16` issue** (which was `LossScaleOptimizer` breaking `ReduceLRCallback`). The TFLite FP16 conversion failure is a new failure mode, first encountered here because all previous experiments used float32 locally (mixed precision was disabled from Exp 22 onwards for the LR reason), and Exp 37 Run 2 is the first run to attempt TFLite conversion with `mixed_float16` active.

**Fix applied to `train_model_conv2D.py`**: Before TFLite conversion, save weights, reset global policy to float32, rebuild model via `tf.keras.models.clone_model`, load weights, convert from the float32 clone. The `export_model` variable is now always defined before the SavedModel fallback path uses it.

```python
if KAGGLE_MODE:
    _tmp_w = os.path.join(checkpoint_dir, "tmp_export.weights.h5")
    model.save_weights(_tmp_w)
    tf.keras.mixed_precision.set_global_policy('float32')
    export_model = tf.keras.models.clone_model(model)
    export_model.load_weights(_tmp_w)
    os.remove(_tmp_w)
else:
    export_model = model
```

**Outcome**: Training ✅ successful. TFLite conversion ❌ failed (fix ready for next run). Weights saved in checkpoint dataset for Run 3.

**For next run**: Publish `best_model.weights.h5` + `model_latest.weights.h5` + `model_latest_epoch.json` to enable full resume (not just warm start). Run 3 will test the TFLite fix.

---

## Experiment 38: Two-Stream Architecture — Separating Conv2D from Lag Features (Proposed)

**Status**: In progress — Kaggle Run 1 started 2026-06-15 (2× T4 GPU, MirroredStrategy, float32)  
**Goal**: Give the Conv2D path a unique, non-redundant role by removing explicit lag and slope features from its input stream, forcing it to learn temporal dynamics from the raw sensor trace independently of the anchor path.

### The Core Problem with Exp 29–35

From Exp 29 onward, `temp_lag60`, `temp_lag120`, and 6 rolling-regression slopes are explicit columns in the `(180, 27)` input. Every timestep of the Conv2D's input already contains a complete summary of the recent temperature history. The Conv2D has no incentive to learn independent temporal representations because the anchor path (which has direct Dense connections to the same lag features) already provides that information more efficiently.

Result: best_epoch=8 across Exp 33–35, anchor path solves the problem, Conv2D adds marginal refinement, GAP ceiling ~0.0024.

### Proposed Architecture

```
Input: (180, 27)
   │
   ├─ Stream A — Conv2D path (raw sensors ONLY, no lag/slope features):
   │   input[:, :, :n_raw]    ← only temperature, humidity, pressure, solar_radiation,
   │                              illuminance, uv, wind_avg, wind_gust, wind_lull,
   │                              wind_direction_sin, wind_direction_cos, rain_accumulated,
   │                              time_of_day_sin/cos/sin2/cos2, day_of_year_sin/cos  (18 features)
   │   Reshape to (180, 18, 1)
   │   → Conv2D(96, k=(3,1))  → BN → ReLU6
   │   → Conv2D(96, k=(7,1))  → BN → ReLU6
   │   → Conv2D(96, k=(15,1)) → BN → ReLU6
   │   → Conv2D(96, k=(1,18)) → BN → ReLU6   ← cross-feature on 18 raw sensors
   │   → GlobalAveragePooling2D → (96,)
   │   → Dense(64) → ReLU6 → conv_context(64)
   │   Role: learn temporal dynamics and cross-sensor patterns that lag/slope features do not capture
   │
   └─ Stream B — Anchor path (current timestep, FULL 27 features):
       input[:, -1, :]         ← current state + temp_lag60 + temp_lag120 + all 6 slopes + temp_delta_1
       → Dense(32) → ReLU6 → anchor(32)
       Role: direct access to all engineered temporal features; the "known-good" baseline

Concatenate([conv_context(64), anchor(32)]) → Dense(32) → ReLU6 → Dense(3) outputs
```

### Why This Should Help

1. **Conv2D has a unique role**: It now sees only the raw 18-sensor time series without pre-computed temporal summaries. It must learn representations the anchor cannot provide — multi-sensor co-activation patterns, local dynamics, threshold crossings over 3–15 minute windows.

2. **Anchor path unchanged**: The anchor still has all 27 features including `temp_lag60`, `temp_lag120`, slopes, and `temp_delta_1`. No information is lost. The val_loss floor is at worst the same as Exp 35's anchor-only contribution.

3. **No structural redundancy**: The gradient cannot shortcut to a "re-learn the lag feature" solution in the Conv2D path because those features are absent. The optimizer is forced to use the Conv2D path for genuinely complementary information or zero it out (in which case we learn the Conv2D adds nothing on raw sensors and the dense path is the right architecture).

4. **Feature importance should separate**: If the architecture works, feature importance from the Conv2D path should cluster on raw sensor features (solar_radiation, wind patterns, humidity dynamics) while the anchor path holds `temp_lag120` dominance. A continued flat distribution across all 27 features would indicate the streams are not learning complementary representations.

### Success Criteria

- val_loss < 0.002368 (beat Exp 35 — baseline comparison)
- best_epoch > 20 (Conv2D path is contributing beyond epoch 8 anchor convergence)
- Feature importance split: Conv2D stream filters activate on raw sensor dynamics; anchor path retains `temp_lag60`/`temp_lag120` dominance
- Edge TPU: single subgraph, all ops mapped

### Architecture Changes from Exp 35

| Parameter | Exp 35 | Exp 38 |
|-----------|--------|--------|
| Conv2D input | All 28 features | Raw 18 sensors only (no lag/slope) |
| Conv2D cross-feature kernel | k=(1,28) | k=(1,18) |
| Anchor input | current 28 features | current 28 features (unchanged) |
| FILTERS | 96 | 96 (unchanged) |
| ReduceLR patience | 12 | 12 (unchanged) |
| EarlyStopping patience | 30 | 30 (unchanged) |

### Note on Exp 36 + 37 Findings

Exp 36 saliency confirmed the Conv2D fires on the oldest sequence row (t=−179) for Δ3hr predictions, effectively learning an implicit `temp_lag180`. Exp 37 makes this explicit. If Exp 37 fully captures the benefit, the two-stream architecture may still be needed to give the Conv2D a genuinely complementary role — but the feature split is now informed: the Conv2D stream should receive raw physical sensors without any pre-computed lag or slope features, and the anchor gets all 28 engineered features including `temp_lag180`.

The Exp 36 saliency showed activations at t=−60 and t=−120 were low (not the dominant positions), and the Conv2D was not re-learning `temp_lag60`/`temp_lag120` — it was reading t=−179 instead. The two-stream should therefore focus on teaching the Conv2D to detect multi-sensor co-activation patterns (solar + humidity + wind dynamics) rather than additional temperature lags.

---

---

### Kaggle Run 3 — NaN crash at epoch 73 + TFLite still failing

**Status**: Failed — NaN from epoch 73 onward; TFLite conversion still erroring  
**Resumed from**: epoch 56 (Priority 2 checkpoint: `model_latest.weights.h5` + `model_latest_epoch.json`)  
**LR at start**: 1e-4 (lr_state.json not published — optimizer restarted fresh)

#### NaN Crash

Training produced NaN from epoch 73 onward. LR had reduced to 5e-5 by the time of the crash.

**Root cause**: `global_clipnorm=1.0` does not prevent FP16 gradient overflow. The `LossScaleOptimizer` multiplies the loss by a dynamic scale (default initial 32768, doubles every 2000 steps). After 37k+ steps, the scale grows enough that `real_gradient × scale` overflows FP16 max (~65504) *during backprop* — before the scale division and before `global_clipnorm` ever runs. `global_clipnorm` operates only on the already-divided final gradients; it cannot intercept mid-backprop overflow.

Run 1 crashed at epoch 38 (~19k steps); Run 3 crashed at epoch 73 (accumulated ~37k steps). Both were LossScaleOptimizer scale overflow, just at different scale magnitudes.

#### TFLite Conversion — clone_model Fix Did Not Work

The `clone_model` + `set_global_policy('float32')` fix from Run 2 still failed. Error output shows `f16` tensors (e.g., `tensor<1x180x28x1xf16>`) even after the rebuild. Root cause: `clone_model` serializes each layer's dtype via `get_config()`, which returns `"mixed_float16"` for layers trained under that policy. The global policy change does not override per-layer configs already baked into the model's config JSON. The cloned model is still `mixed_float16`.

#### Fix Applied for Run 4

**Disabled `mixed_float16` entirely** — both on Kaggle and locally:

1. Removed `set_global_policy('mixed_float16')` for KAGGLE_MODE
2. Removed `global_clipnorm=1.0` from Adam (no longer needed — float32 backprop cannot overflow)
3. Simplified TFLite export to `export_model = model` (no float32 rebuild needed)

Float32 on T4 is ~50–60% slower (~250–280s/epoch vs 175s), but eliminates NaN risk and the TFLite fp16 conversion path entirely.

#### Checkpoint Status

`best_model.weights.h5` in the Kaggle output contains good weights from the last epoch where val_loss improved (before epoch 73 NaN). `model_latest.weights.h5` contains NaN weights (saved at epoch 86). For Run 4, publish only `best_model.weights.h5` — do **not** publish `model_latest.weights.h5` or `model_latest_epoch.json` (would trigger Priority 2 with NaN weights). The Priority 3 path will load `best_model.weights.h5` and start fresh from those weights.

---

### Local Mac Test — Float32 Verification (between Run 3 and Run 4)

**Status**: Abandoned — hang at epoch 3 batch 77  
**Purpose**: Verify float32 fix before next Kaggle run. 2 epochs completed; Metal GPU stall at batch 77/1025 (same intermittent macOS scheduling issue as prior experiments). Confirmed float32 runs without NaN locally; no useful training progress.

---

### Kaggle Run 4 — Timeout at epoch 68 ⏱️

**Status**: Timed out — Kaggle 12-hour (43200s) wall hit during epoch 68  
**Resumed from**: `best_model.weights.h5` from Run 3 (before NaN; Priority 3 warm start, initial_epoch=0)  
**LR at start**: 1e-4 (fresh optimizer state — `lr_state.json` not published)  
**Results folder**: `Kaggle/results_4_exp37/`

**Training progress** (epochs near timeout):
| Epoch | train_loss | val_loss | LR | Notes |
|-------|-----------|----------|-----|-------|
| 64 | 7.81e-4 | 0.0022 | 5e-5 | Near best |
| 65 | 7.79e-4 | 0.0032 | 2.5e-5 | LR reduced (wait=12); val_loss spike |
| 66 | 7.32e-4 | 0.0029 | 2.5e-5 | Recovering |
| 67 | 7.24e-4 | 0.0024 | 2.5e-5 | Recovering toward best |
| 68 | — | — | — | Timeout at 43208.8s (mid-epoch) |

**Best result**: val_loss = **0.002192** (epoch ~53, inferred from ReduceLR: `wait=12, best=0.002192` at ep65)

**Exp 37 success criteria check**:
- val_loss < 0.002368 ✅ (0.002192 < 0.002368)
- best_epoch > 20 ✅ (~53)
- `temp_lag180` top 5 importance ✅ (#1 at 0.0810, confirmed in Run 2)

**Checkpoint state** (`Kaggle/results_4_exp37/checkpoints/`):
| File | Status |
|------|--------|
| `best_model.weights.h5` | ✅ Epoch ~53 weights (val_loss=0.002192) |
| `model_latest.weights.h5` | ✅ Epoch 67 weights |
| `model_latest_epoch.json` | ✅ `{"epoch": 67}` |
| `lr_state.json` | ✅ `{"lr": 2.5e-5, "best": 0.002192, "wait": 2}` |
| `early_stopping_state.json` | ✅ `{"best": 0.002192, "wait": 14}` — 16 epochs remain before ES fires |

**Resume options**:
1. **Full resume from epoch 67** — publish all 5 checkpoint files; full LR + ES state restored; 16 epochs left before ES fires at 2.5e-5
2. **Move to Exp 38** — 0.002192 plateau with same anchor-ceiling pattern as Exp 29–35 suggests structural limitation; proceed to two-stream architecture

---

### Kaggle Run 5 — EarlyStopping fired, training complete ✅

**Status**: Complete — EarlyStopping fired at epoch 131  
**Resumed from**: Run 4 checkpoint (Priority 2: `model_latest.weights.h5` + `model_latest_epoch.json`, epoch 67)  
**LR at start**: 2.5e-5 (restored from `lr_state.json`: best=0.002192, wait=2)  
**Results folder**: `Kaggle/results_5_exp37/`

**Training**: float32 throughout (mixed_float16 disabled since Run 4 fix). EarlyStopping patience=30 exhausted. ES and LR state files deleted by the training script on clean completion.

| Metric | Value |
|--------|-------|
| val_loss | **0.002117** (new Exp 37 best; Run 4 best was 0.002192) |
| val_mae (normalized) | 0.006122 |
| Best session epoch | 9 → actual epoch 76 (initial_epoch=67; session index 8 = epoch 67+8) |
| Final epoch | 131 |
| Model size (INT8 TFLite) | **478 KB** |

**TFLite conversion**: ✅ Succeeded — float32-throughout model converts cleanly with no fp16 op errors. File: `weather_model_5b_quant_conv2d_exp37_run1.tflite`

**INT8 validation results** (PTQ, 500 samples):
| Output | INT8 scale | MAE (float) | MAE (INT8) | Degradation |
|--------|-----------|-------------|------------|-------------|
| diff_1hr | 0.002508/step → ~0.036°C | ~0.09°C | **0.68°C** | ~7.5× |
| diff_2hr | 0.003258/step → ~0.046°C | ~0.09°C | **1.20°C** | ~13× |
| diff_3hr | 0.004972/step → ~0.071°C | ~0.09°C | **1.63°C** | ~18× |

**⚠️ PTQ output collapse confirmed**: Sample dequantized outputs across 5 consecutive windows are nearly identical (diff_1hr: −0.083/−0.078/−0.078/−0.083/−0.080; range = 0.005 normalized ≈ 0.07°C). The float model varies much more between windows; the INT8 model is converging to near-constant predictions, the same collapse pattern seen in Exp 12–24. PTQ does not work for this architecture — QAT will be required after the float accuracy target is met.

**Feature importance** (Run 5, flat distribution 0.066–0.081):
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `temperature` | 0.0810 |
| 2 | `temp_lag180` | 0.0777 |
| 3 | `time_of_day_cos2` | 0.0769 |
| 4 | `humidity_slope_30` | 0.0756 |
| 5 | `temp_slope_30` | 0.0749 |
| … | … | … |
| 26 | `temp_slope_60` | 0.0685 |
| 27 | `time_of_day_sin` | 0.0658 |

`temperature` displaced `temp_lag180` as #1 (was #1 in Run 2 at 0.0810; now #2 at 0.0777). Both are still effectively tied. The 28-feature range of 0.066–0.081 is extremely flat — no dominant signal; the anchor Dense(32) is distributing gradient evenly across all inputs rather than specialising.

**Checkpoint state** (`Kaggle/results_5_exp37/checkpoints/`):
| File | Status |
|------|--------|
| `best_model.weights.h5` | ✅ Best epoch (~76) weights |
| `model_latest.weights.h5` | ✅ Epoch 131 weights |
| `model_latest_epoch.json` | ✅ `{"epoch": 131}` |
| `early_stopping_state.json` | 🗑️ Deleted (training complete) |
| `lr_state.json` | 🗑️ Deleted (training complete) |

**Exp 37 final success criteria check**:
- val_loss < 0.002368 ✅ (0.002117)
- best_epoch > 20 ✅ (actual epoch ~76)
- `temp_lag180` top 5 importance ✅ (#2 at 0.0777)
- TFLite conversion ✅ (after disabling mixed_float16 in Run 4)

**Exp 37 conclusion**: All success criteria met. `temp_lag180` confirmed as a top-2 feature. However, the anchor-ceiling persists: best val_loss 0.002117 is 3× short of Model 5a (0.000682). The flat feature importance distribution across all 28 features is the diagnostic: the anchor Dense(32) is doing most of the predictive work and the Conv2D is adding marginal refinement. This motivates **Exp 38** — the two-stream architecture that removes lag/slope features from the Conv2D path, forcing it to learn genuinely complementary temporal dynamics.

---

### Exp 38 Kaggle Run 1 — ✅ COMPLETE (2026-06-16)

**Status**: Complete — trained to max_epochs=150 (early stopping never fired); final outputs in `Kaggle/results_2_exp38/`  
**Infrastructure**: 2× T4 GPU, MirroredStrategy, batch=1024, float32, XLA JIT enabled  
**Speed**: ~826–921 ms/step, ~423–472 s/epoch across the full run  
**Results folder**: `Kaggle/results_2_exp38/`

**Epoch observations**:

| Epoch | train_loss | val_loss | val_1hr_mae | val_2hr_mae | val_3hr_mae | LR | Notes |
|-------|-----------|----------|------------|------------|------------|-----|-------|
| 1 | 0.0626 | 0.1617 | 0.1188 | 0.1045 | 0.2567 | 1e-4 | XLA JIT warmup artifact — val_3hr anomalously high during first-pass algorithm selection |
| 2 | 0.0460 | 0.0480 | 0.0383 | 0.0443 | 0.0567 | 1e-4 | Healthy convergence; val ≈ train (no overfitting); all three heads balanced |
| 3–104 | — | — | — | — | — | — | Not captured in this log (earlier Kaggle session); checkpoint at epoch 104 restored with ReduceLR best=0.004292, wait=9/12 |
| 105 | 0.0018 | 0.0071 | 0.0262 | 0.0349 | 0.0444 | 5e-5 | Resumed; val_loss above restored best |
| 106 | 0.0018 | 0.0114 | 0.0273 | 0.0435 | 0.0670 | 5e-5 | Worse; wait hits 12 → triggers LR reduction |
| 107 | 0.0018 | 0.0102 | 0.0335 | 0.0472 | 0.0579 | 2.5e-5 | ReduceLR fired (wait=12, best=0.004292) |
| 108 | 0.0017 | **0.0040** | 0.0224 | 0.0176 | 0.0195 | 2.5e-5 | New best (beats restored 0.004292) |
| 109–117 | ~0.0016–0.0017 | 0.0039–0.0092 | — | — | — | 2.5e-5 | Noisy oscillation, no LR change; wait cycles up to 9 then resets on incidental dips |
| 118 | 0.0016 | 0.0039 | 0.0220 | 0.0181 | 0.0183 | 2.5e-5 | Ties best |
| 119–124 | ~0.0015–0.0016 | 0.0039–0.0051 | — | — | — | 2.5e-5 | Same oscillation pattern continues |
| 125 | 0.0015 | **0.0038** | 0.0221 | 0.0173 | 0.0184 | 2.5e-5 | New best so far this run |
| 126–127 | 0.0015–0.0016 | 0.0043–0.0055 | — | — | — | 2.5e-5 | Still oscillating; train_loss continues slow monotonic decline |

**Observations (epochs 105–150)**:
- Train loss declined smoothly throughout (0.0018 → 0.0014); no instability.
- Val loss oscillated between ~0.0038 and ~0.011 for the entire resumed portion — never produced a meaningful sustained downward trend.
- LR reduced a second time (2.5e-5 → 1.25e-5) during the 141–150 stretch; val_loss at epoch 150 was 0.0050, above the best.
- **Early stopping (patience=30) never fired** — val loss oscillated enough to keep resetting the wait counter, but no real improvement occurred after epoch ~32.

**Final Results (complete)**:
- **val_loss: 0.003779** (best)
- **val_mae: 0.0101** (normalized, corresponds to ~0.01°C-scale diffs in target space)
- **Best epoch: 32** — the model's best checkpoint was epoch 32 out of 150; the remaining 118 epochs produced noisy oscillation around but not below this floor. This is a variant of the same early-plateau pattern seen in Exp 33–35 (best_epoch ~8–10), just delayed slightly.
- **Quantized TFLite MAE: diff_1hr=1.07°C, diff_2hr=2.15°C, diff_3hr=2.24°C** — catastrophic PTQ failure; INT8 collapsed the model's dynamic range to near-constant predictions (sample outputs: −0.121/−0.124/−0.121/−0.117/−0.119 — all nearly identical). Consistent with all prior PTQ failures; QAT remains the only untried path.

**Feature importance (full list)**:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | temperature | 0.0790 |
| 2 | time_of_day_cos | 0.0779 |
| 3 | uv | 0.0757 |
| 4 | wind_lull | 0.0720 |
| 5 | temp_lag120 | 0.0718 |
| 6 | temp_lag180 | 0.0713 |
| 7 | day_of_year_cos | 0.0713 |
| 8 | temp_slope_30 | 0.0712 |
| 9 | temp_slope_15 | 0.0711 |
| 10 | temp_delta_1 | 0.0711 |
| … | rain_accumulated … pressure_slope_60 … | 0.0711 |
| 26 | time_of_day_cos2 | 0.0685 |
| 27 | time_of_day_sin2 | 0.0669 |
| 28 | time_of_day_sin | 0.0661 |

**Feature importance range: 0.0661–0.0790 (spread = 0.013)** — still very flat.

**Conclusions:**

❌ **Two-stream hypothesis NOT confirmed.** Success criteria all missed:

| Criterion | Target | Actual |
|-----------|--------|--------|
| val_loss | < 0.001 (gate) or < 0.002368 (Exp 35 baseline) | 0.003779 — WORSE than Exp 37's 0.002117 |
| best_epoch | > 20 (technically met) | 32, but model stalled for 118 more epochs with no improvement |
| Feature importance | Clear stream split | Still flat (0.0661–0.0790), no specialisation |
| TFLite INT8 | Working PTQ | Catastrophic failure; PTQ not viable |
| val_loss gate | < 0.001 to proceed to Exp 39 | Not met |

**Positive signal (small)**: `uv` (3rd, Conv2D-stream only) and `wind_lull` (4th, Conv2D-stream only) appear higher in the rankings than in Exp 37 — the two-stream split may have nudged the Conv2D toward raw physical sensors rather than re-learning temperature lags. But this signal is weak given the overall flat distribution and regression in val_loss vs Exp 37.

**Root cause of regression vs Exp 37**: The two-stream split reduced the anchor path's available information (Conv2D now sees only 18 raw features instead of 28, so the model has less information flow for the first ~32 epochs before the Conv2D path contributes anything). Anchor-ceiling convergence at epoch 32 with a smaller effective input budget produced a weaker result. After that, the Conv2D path's raw-sensor representations weren't sufficiently complementary to recover.

**Next steps per documented gate:**
- Exp 39 (dilated convolutions) is already running locally. Per the gate, it was conditional on Exp 38 val_loss < 0.001; that gate was NOT met. Continue monitoring Exp 39 but treat its results with the caveat that the prerequisite (confirmed Conv2D contribution) was not established.
- **Anchor-only baseline**: deserves a run — if anchor Dense alone matches Exp 38/Exp 37 performance, Conv2D has never contributed, and architecture direction should shift to a wider anchor head rather than increasingly complex Conv2D paths.

---

## Experiment 39: Dilated Convolutions — Expanding Temporal Receptive Field

**Status**: ✅ COMPLETE (2026-06-17) — 150 epochs, Mac Metal GPU  
**Goal**: Extend the Conv2D temporal receptive field from 23 minutes to ~4 hours using exponentially dilated convolutions, giving the Conv2D path the ability to detect multi-hour weather patterns rather than only sub-30-minute local dynamics.

### The Core Receptive Field Problem

The stacked Conv2D layers in Exp 38 (and all prior experiments) have a combined temporal receptive field of only **23 timesteps = 23 minutes**:

| Layer | Kernel | Dilation | Cumulative receptive field |
|-------|--------|----------|---------------------------|
| conv2d_t1 | (3, 1) | 1 | 3 min |
| conv2d_t2 | (7, 1) | 1 | 3 + 6 = **9 min** |
| conv2d_t3 | (15, 1) | 1 | 9 + 14 = **23 min** |
| conv2d_feat | (1, n_raw) | — | no temporal change |

`GlobalAveragePooling2D` averages the filter responses over all 180 positions in the window, so the model can detect a 23-minute pattern *anywhere* in the 3-hour window — but it cannot detect any pattern that spans more than 23 minutes. A pressure drop developing over 2 hours, a marine layer building over 90 minutes, or a solar ramp-down signalling an approaching cloud deck are invisible to the current Conv2D path.

**Window size is not the fix**: expanding the input from 180 → 360 timesteps would not increase the receptive field — it would give the same 23-minute pattern detector more positions to fire at. The fundamental limit is filter span, not window span.

### Proposed Architecture Change (Stream A only)

Replace the three uniform Conv2D temporal layers with an exponentially dilated pyramid:

```
Current (Exp 38):                           Proposed (Exp 39):
Conv2D(96, k=(3,1),  d=1)  → RF: 3 min    Conv2D(96, k=(3,1),  d=1)   → RF: 3 min
Conv2D(96, k=(7,1),  d=1)  → RF: 9 min    Conv2D(96, k=(7,1),  d=4)   → RF: 3 + 4×6  = 27 min
Conv2D(96, k=(15,1), d=1)  → RF: 23 min   Conv2D(96, k=(15,1), d=16)  → RF: 27 + 16×14 = 251 min (~4 hrs)
Conv2D(96, k=(1,n_raw))    → cross-feat    Conv2D(96, k=(1,n_raw))     → cross-feat (unchanged)
GlobalAveragePooling2D                      GlobalAveragePooling2D
```

Dilated receptive field calculation:
- Layer 1: k=3, d=1 → RF = 3
- Layer 2: k=7, d=4 → RF = 3 + (7−1)×4 = 27
- Layer 3: k=15, d=16 → RF = 27 + (15−1)×16 = 251 timesteps ≈ **4 hours 11 minutes**

Same parameter count, same compute budget — exponentially larger temporal context.

### Why This Could Break the Ceiling

The patterns we hypothesize the Conv2D should learn (from the user's hypothesis motivating Exp 38) are multi-hour phenomena:
- Solar radiation ramping down over 60–90 min while humidity climbs → approaching marine layer
- Pressure dropping steadily over 2–3 hours while wind direction shifts → frontal passage
- UV declining + wind lull → cloud deck formation in progress

These patterns are too slow to be visible in a 23-minute window. With a 4-hour receptive field, the Conv2D can detect the **trajectory** of multi-sensor interactions over the full 180-step window — which is exactly the complementary role the anchor Dense path (current-timestep features) cannot provide.

### Architecture Changes from Exp 38

| Parameter | Exp 38 | Exp 39 |
|-----------|--------|--------|
| Conv2D input | Raw 18 sensors (no lag/slope) | Raw 18 sensors (unchanged) |
| conv2d_t1 | k=(3,1), d=1 → RF 3 min | k=(3,1), d=1 → RF 3 min (unchanged) |
| conv2d_t2 | k=(7,1), d=1 → RF 9 min | k=(7,1), d=4 → RF 27 min |
| conv2d_t3 | k=(15,1), d=1 → RF 23 min | k=(15,1), d=16 → RF ~251 min |
| Anchor path | current 28 features → Dense(32) | unchanged |
| FILTERS | 96 | 96 (unchanged) |
| SEQ_LEN | 180 | 180 (unchanged — window size is not the bottleneck) |

### Dependency

Do not implement until Exp 38 demonstrates that the two-stream separation helps (val_loss < 0.001 sustained past epoch 20). If Exp 38 plateaus at the same ~0.002 ceiling as Exp 29–35, dilated convolutions alone are unlikely to fix a problem rooted in the anchor-ceiling architecture. Exp 38 is the necessary prior: confirm the Conv2D path can contribute something, *then* expand its temporal reach.

### Success Criteria

- val_loss < 0.001 (meaningful improvement over Exp 37/38 anchor ceiling)
- best_epoch > 30 (Conv2D temporal patterns require more gradient steps to learn than lag-extraction shortcuts)
- 3hr head disproportionate improvement over Exp 38 (longer receptive field should help the hardest prediction horizon most)
- Feature importance: Conv2D stream filters should activate on multi-sensor dynamics, not concentrate on oldest timestep (as seen in Exp 36 saliency)

### Local Run — In Progress (2026-06-16)

**Infrastructure**: Mac Metal GPU, single device, float32, `steps_per_execution=1`, `MaxEpochsPerRun`-based periodic process restart (clean stop + relaunch every ~10 epochs) to avoid the Metal scheduling hang ([[feedback_metal_gpu_hang]] — confirms the GPU-only hang risk; the restart workaround is doing its job, no hang through at least 6 restart cycles / 61 epochs).

| Epoch | val_diff_1hr_loss | val_diff_2hr_loss | val_diff_3hr_loss | val_loss | LR | Notes |
|-------|-------------------|--------------------|--------------------|----------|-----|-------|
| 52 | 0.0013 | 0.0030 | 0.0038 | 0.0416 | 6.25e-6 | |
| 53 | 0.0013 | 0.0029 | 0.0037 | 0.0413 | 6.25e-6 | |
| 54 | 0.0013 | 0.0027 | 0.0035 | 0.0409 | 6.25e-6 | |
| 55 | 0.0012 | 0.0025 | 0.0034 | 0.0406 | 6.25e-6 | |
| 56 | 0.0012 | 0.0024 | 0.0032 | 0.0403 | 6.25e-6 | |
| 57 | 0.0012 | 0.0022 | 0.0031 | 0.0400 | 6.25e-6 | |
| 58 | 0.0012 | 0.0021 | 0.0030 | 0.0398 | 6.25e-6 | |
| 59 | 0.0012 | 0.0021 | 0.0030 | 0.0397 | 6.25e-6 | |
| 60 | 0.0012 | 0.0020 | 0.0029 | 0.0395 | 6.25e-6 | |
| 61 | 0.0012 | 0.0019 | 0.0028 | 0.0394 | 6.25e-6 | MaxEpochsPerRun clean stop + restart after this epoch |

**Observations**:
- No hang observed through 61 epochs / 6+ restart cycles — the periodic Metal context reset is working as intended.
- LR is already down to 6.25e-6 by epoch 52 (4 ReduceLR halvings from the 1e-4 start), so improvement per epoch is necessarily small from here; the per-head losses are still trending down slowly and smoothly (no oscillation, unlike Exp 38's noisy val_loss pattern), but at a low LR this could just be fine settling rather than a real breakout.
- **`val_loss` (0.039–0.042) is ~6–7× larger than the sum of the three head losses** (e.g. epoch 61: 0.0012+0.0019+0.0028 = 0.0059 vs val_loss 0.0394). For Exp 38 this ratio was much smaller (~1.4× at epoch 125: sum 0.0027 vs val_loss 0.0038). Both experiments use the same `l2(1e-4)` regularizer on the same-shaped layers (dilation doesn't change parameter count), so this gap is most likely the L2 term reflecting larger weight magnitudes in the dilated path, not a bug — but it makes the raw `val_loss` numbers **not directly comparable** between Exp 38 and Exp 39. Use the per-head `val_diff_*hr_loss`/`mae` for cross-experiment comparison instead. Worth a quick sanity check (e.g. `model.evaluate` decomposition or weight-norm logging) if this hasn't been confirmed already.
- On that comparable basis, Exp 39 at epoch 61 (1hr/2hr/3hr = 0.0012/0.0019/0.0028) is currently **behind** Exp 38's best epoch 125 (0.00097/0.00077/0.0010), including on the 3hr head that the dilated receptive field specifically targets. Not conclusive — Exp 39 is mid-run with LR already near its floor — but no sign yet of the hoped-for 3hr breakout.
- The printed lines for this stretch show only `val_*` metrics with no train-side `loss`/`diff_*hr_loss` fields — possibly just how this excerpt was captured/pasted, but worth double-checking the raw log has train metrics too (the earlier one-time "Your input ran out of data; interrupting training" warning is a separate, apparently self-resolving issue — training continued at 513/513 steps/epoch afterward).

### Kaggle Run 1 — In Progress (2026-06-16)

**Infrastructure**: 2× T4 GPU, MirroredStrategy, batch=1024, float32, XLA JIT enabled  
**Resumed from**: epoch 70 checkpoint (published from local run above)  
**Checkpoint dataset**: `datasets/dacarson/weatherml-5b-checkpoints-exp39-2`

**Restored state at resume:**
- LR: 6.25e-6 (4 halvings from 1e-4 initial — already near minimum useful LR)
- ReduceLR: best=0.039381, wait=8/12 — 4 more epochs without improvement triggers another halving (→ 3.125e-6)
- EarlyStopping: best=0.039431, wait=3/30 — plenty of patience left

**Optimizer cold-restart warning** (appears every session):
```
Skipping variable loading for optimizer 'adam', because it has 2 variables 
whereas the saved optimizer has 38 variables.
```
Adam's 38 momentum/velocity buffers are NOT restored from checkpoint — optimizer starts cold every session. At LR 6.25e-6 the effect is small (weights well-established, gradients tiny) but the first 1–2 epochs post-resume may show slightly noisier val_loss as Adam rebuilds its moment estimates.

**Progress (epochs 72–86):**

| Epoch | val_1hr_loss | val_2hr_loss | val_3hr_loss | val_loss | LR |
|-------|-------------|-------------|-------------|----------|-----|
| 72 | 9.3806e-04 | 0.0013 | 0.0019 | 0.0354 | 6.25e-6 |
| 73 | 8.8516e-04 | 0.0013 | 0.0019 | 0.0346 | 6.25e-6 |
| 74 | 8.6761e-04 | 0.0013 | 0.0018 | 0.0338 | 6.25e-6 |
| 75 | 8.4824e-04 | 0.0012 | 0.0018 | 0.0330 | 6.25e-6 |
| 76 | 8.1074e-04 | 0.0012 | 0.0017 | 0.0324 | 6.25e-6 |
| 77 | 7.8995e-04 | 0.0011 | 0.0017 | 0.0317 | 6.25e-6 |
| 78 | 7.7091e-04 | 0.0011 | 0.0017 | 0.0310 | 6.25e-6 |
| 79 | 7.4505e-04 | 0.0011 | 0.0016 | 0.0304 | 6.25e-6 |
| 80 | 7.2561e-04 | 0.0011 | 0.0016 | 0.0299 | 6.25e-6 |
| 81 | 7.1461e-04 | 0.0011 | 0.0016 | 0.0293 | 6.25e-6 |
| 82 | 6.9440e-04 | 0.0011 | 0.0016 | 0.0288 | 6.25e-6 |
| 83 | 6.8020e-04 | 0.0010 | 0.0015 | 0.0282 | 6.25e-6 |
| 84 | 6.6903e-04 | 0.0010 | 0.0016 | 0.0278 | 6.25e-6 |
| 85 | 6.4568e-04 | 0.0010 | 0.0015 | 0.0273 | 6.25e-6 |
| 86 | 6.3301e-04 | 0.0010 | 0.0015 | 0.0268 | 6.25e-6 |
| 87 | — | — | — | — | running |

**Observations (Kaggle run 1, ep72–86)**:
- All three heads still descending smoothly with no stall through 16 epochs at LR=6.25e-6. Val_loss improving every epoch keeps the ReduceLR counter resetting — no halving imminent.
- **1hr head (6.33e-4 at ep86) has now beaten Exp 38's best 1hr (9.7e-4) by a wide margin** — 35% lower.
- **2hr head (1.0e-3) has now matched Exp 38's best 2hr (1.0e-3)** and is still declining. Exp 38's best was 7.7e-4; at current rate (~2e-5/epoch) it should pass that within the next ~15 epochs.
- **3hr head (1.5e-3) still trails Exp 38's best (1.0e-3)** — the disproportionate 3hr improvement the dilated-RF hypothesis predicted has not appeared; 3hr is improving in proportion with the other heads.
- Train-val gap widening slightly (ep76: val/train ratio for 1hr ≈ 1.43×; ep86 ≈ 1.65×) — mild onset of overfitting, but not yet alarming. Worth watching if the gap accelerates.
- **1hr head (6.33e-4) is now approaching Model 5a's val_loss (6.82e-4)** — though these are not directly comparable (diff target vs absolute temperature), it is a useful calibration point.
- Timing: session started at epoch 70, epoch 86 at ~11.9k seconds = 3.3 hours elapsed. With ~12.5k seconds remaining in a 9-hour session, expect to reach approximately **epoch 116** before session 2 ends.
- **After session 2 times out (~ep116): switch to Mac Metal GPU** for the remaining ~34 epochs (116→150 or EarlyStopping). No third Kaggle session needed.

### Mac Final Run — ✅ COMPLETE (2026-06-17)

**Infrastructure**: Mac Metal GPU, float32, periodic MaxEpochsPerRun restarts  
**Final LR**: 3.1250e-06 (5 halvings from 1e-4 initial)  
**Total epochs**: 150 (full training run)

| Metric | Epoch 149 | Epoch 150 |
|--------|-----------|-----------|
| val_diff_1hr_loss | 4.4156e-04 | **4.4525e-04** |
| val_diff_2hr_loss | 8.1360e-04 | **8.1921e-04** |
| val_diff_3hr_loss | 1.3e-03 | **1.3e-03** |
| val_loss | 0.0164 | **0.0163** |
| LR | 3.125e-06 | 3.125e-06 |

**Final Metrics (from script summary):**

| Metric | Value |
|--------|-------|
| val_loss | **0.0163** |
| val_mae (normalized) | **0.0091** |
| Best epoch | 2 (of final Mac session = epoch 150 overall) |
| Quantized model size | 388.37 KB |
| Per-head float MAE (normalized, script labels as "°C") | 0.0131 / 0.0173 / 0.0226 |

> **"Best epoch: 2" explained**: The MaxEpochsPerRun restart mechanism resets Keras callback epoch counters each session. "Best epoch: 2" is the best epoch within the final Mac session (2 epochs: 149 and 150 overall), not epoch 2 of training. The saved checkpoint corresponds to epoch 150 overall.

> **"val_loss not comparable"**: Val_loss (0.0163) is dominated by L2 regularization from the dilated kernels' larger weight magnitudes (~6–7× the sum of per-head losses). This inflated L2 means the raw val_loss is **not comparable** between Exp 39 and Exp 37/38. Use per-head losses for cross-experiment comparison.

**Quantized TFLite Validation (500 samples, PTQ INT8):**

| Output | Scale | ZP | Quantized MAE |
|--------|-------|----|---------------|
| diff_1hr | 0.00206 | 17 | **0.52°C** |
| diff_2hr | 0.00386 | 4 | **1.06°C** |
| diff_3hr | 0.00487 | 14 | **1.38°C** |

PTQ failed catastrophically — consistent with all prior experiments. QAT required.

**Permutation Feature Importance (top / bottom):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | time_of_day_sin2 | 0.0754 |
| 2 | time_of_day_sin | 0.0738 |
| 3 | time_of_day_cos | 0.0729 |
| 4 | solar_radiation | 0.0718 |
| 5 | relative_humidity | 0.0703 |
| 6 | temp_lag180 | 0.0693 |
| 7 | temp_lag60 | 0.0688 |
| — | … | … |
| 24 | temp_lag120 | 0.0677 |
| — | … | … |
| 28 | temperature | **0.0446** |

Full importance range: 0.0446–0.0754. Diurnal features (time_of_day_sin2/sin/cos) dominate; temperature (current, t=0) is last. **Note**: `temperature` being last does NOT indicate the early-experiment diurnal-anchoring failure — the lag features (temp_lag60/120/180) are in the anchor path and the model correctly weights them over the current-timestep value. The diurnal dominance at the TOP is more concerning.

### Exp 39 vs Exp 38: Per-Head Loss Comparison

| Head | Exp 38 best | Exp 39 final (ep150) | Change |
|------|-------------|----------------------|--------|
| 1hr | 9.7e-04 | **4.45e-04** | **−54%** ✅ |
| 2hr | 7.7e-04 | 8.19e-04 | +6% ≈ tied |
| 3hr | 1.0e-03 | 1.3e-03 | **+30%** ❌ |

### Success Criteria Assessment

| Criterion | Target | Actual | Met? |
|-----------|--------|--------|------|
| val_loss < 0.001 | < 0.001 | 0.0163 (L2-inflated) | ❌ |
| best_epoch > 30 | > 30 | 150 overall | ✅ (in spirit) |
| 3hr head disproportionate improvement over Exp 38 | 3hr best improvement | 3hr +30% WORSE | ❌ |
| Feature importance: Conv2D multi-sensor dynamics | time_of_day not dominant | time_of_day_sin2 #1 | ❌ |

**Conclusion**: ❌ **Dilated RF hypothesis not confirmed.** The 1hr head achieved a dramatic 54% improvement over Exp 38, which is encouraging, but the 3hr head regressed 30% — the opposite of what the dilated receptive field hypothesis predicted (longer RF should help the longest horizon most). The diurnal signals still dominate feature importance. The anchor-ceiling pattern persists: the model relies on anchor-path lag features and time-of-day encodings rather than Conv2D-extracted multi-hour weather dynamics.

The 1hr improvement may reflect the dilated Conv2D providing richer recent-history context (27–251 min RF) that complements the anchor's scalar lag snapshots, but the 3hr regression suggests the wider RF isn't helping — or is interfering with — the longest-horizon prediction.

**Exp 40 dependency met?** Weakly — the 1hr improvement suggests some Conv2D contribution, but the 3hr regression and unmet success criteria mean the dilated architecture's value is unconfirmed for the multi-horizon task. Proceed with the **anchor-only baseline first** before Exp 40.

---

## Experiment 40: Remove Diurnal Features from Conv2D Path — Force Physical Sensor Learning

**Status**: ✅ Complete — best_epoch=56, 86 epochs trained (training watchdog stopped)  
**Goal**: Prevent the dilated Conv2D from shortcutting on time-of-day encodings by routing all four diurnal features (`time_of_day_sin`, `time_of_day_cos`, `time_of_day_sin2`, `time_of_day_cos2`) to the anchor path only. Forces Conv2D to learn temperature dynamics from physical sensor readings across the 180-min window.

### Motivation: Exp 39 Diurnal Dominance

Exp 39's feature importance ranked `time_of_day_sin2` as the #1 feature — the same "Diurnal Dominance" pattern seen in pre-Conv2D experiments. The dilated receptive field (~251 min) was not sufficient to overcome the shortcut: with diurnal encodings visible at every timestep across 180 steps, the Conv2D learned "what time of day is it?" rather than "what is the temperature trajectory doing?"

The fix mirrors what was already done for `temp_lag60/120/180`: those features are excluded from the Conv2D input (they live only in `engineered_features`, which sits after the `n_conv` slice boundary). We apply the same treatment to the diurnal encodings.

`day_of_year_sin/cos` remain in the Conv2D path — they are a slow annual signal, not the diurnal cycle causing the problem.

### Architecture Changes from Exp 39

| Parameter | Exp 39 | Exp 40 |
|-----------|--------|--------|
| Conv2D input | 18 sensors incl. `time_of_day_sin/cos/sin2/cos2` | 14 physical sensors only — diurnal removed (`n_conv` = 14 + optional wind/lull/rain) |
| Anchor path | All features at t=−1 (unchanged) | All features at t=−1 — diurnal encodings still reach anchor via full-feature slice |
| Reshape kernel | `(SEQ_LEN, n_raw, 1)` | `(SEQ_LEN, n_conv, 1)` |
| Feature-mixing Conv2D | `kernel_size=(1, n_raw)` | `kernel_size=(1, n_conv)` |
| Dilated Conv2D RF | k=(3,1)d=1 / k=(7,1)d=4 / k=(15,1)d=16 | Unchanged |
| Lag features | `temp_lag60/120/180`, `temp_delta_1` in anchor | Unchanged (kept in anchor) |

Feature vector ordering: `conv_features` + `diurnal_features` + `engineered_features`  
Conv2D slice: `input[:, :, :n_conv]` — StridedSlice, Edge TPU compatible ✅

### Why This Is Edge TPU Compatible

The Conv2D input is still extracted as a contiguous prefix of the feature vector (`input[:, :, :n_conv]`). Diurnal features are placed immediately after, followed by engineered features — the ordering changes but the slice op remains a StridedSlice with no gather required.

### Success Criteria

- `time_of_day_sin2` no longer ranks as top feature — physical sensors (temperature, solar_radiation) should rise in importance
- Per-head val losses ≥ Exp 39's best (goal: match or improve)
- Temperature feature importance rises from its Exp 39 ranking
- No diurnal phase-lag artefact in qualitative prediction plots
- best_epoch > 30

### Note: Lag Removal Already Included

`temp_lag60`, `temp_lag120`, `temp_lag180`, and `temp_delta_1` are also removed from the anchor path in this experiment — both fixes are combined in Exp 40. No separate follow-on experiment is needed for lag removal.

### Training Progress

| Epoch | val_loss | val_task_loss | val_1hr_loss | val_2hr_loss | val_3hr_loss | LR |
|-------|----------|---------------|--------------|--------------|--------------|-----|
| 52 | 0.1704 | 0.1264 | 0.0118 | 0.0113 | 0.1033 | 5.0e-5 |
| 53 | 0.1702 | 0.1262 | 0.0118 | 0.0113 | 0.1032 | 5.0e-5 |
| 55 | 0.1701 | 0.1261 | 0.0118 | 0.0111 | 0.1032 | 5.0e-5 |
| **56** | **0.1701** | **0.1261** | **0.0118** | **0.0111** | **0.1032** | 5.0e-5 |
| 67 | 0.1715 | 0.1276 | 0.0126 | 0.0106 | 0.1043 | 5.0e-5 |
| 68 | 0.1719 | 0.1280 | 0.0127 | 0.0106 | 0.1047 | 2.5e-5 ← LR drop |
| 73 | 0.1735 | 0.1296 | 0.0133 | 0.0105 | 0.1058 | 2.5e-5 |
| 74 | 0.1736 | 0.1297 | 0.0134 | 0.0105 | 0.1058 | 2.5e-5 |
| 75 | 0.1736 | 0.1297 | 0.0134 | 0.0104 | 0.1058 | 2.5e-5 |
| 76 | 0.1737 | 0.1298 | 0.0135 | 0.0104 | 0.1059 | 2.5e-5 |
| 77 | 0.1738 | 0.1300 | 0.0136 | 0.0104 | 0.1060 | 2.5e-5 |
| 78 | 0.1740 | 0.1301 | 0.0137 | 0.0103 | 0.1061 | 2.5e-5 |
| 79 | 0.1742 | 0.1303 | 0.0138 | 0.0103 | 0.1062 | 2.5e-5 |
| 80 | 0.1747 | 0.1308 | 0.0140 | 0.0103 | 0.1066 | 1.25e-5 ← LR drop |
| 81 | 0.1749 | 0.1311 | 0.0141 | 0.0102 | 0.1068 | 1.25e-5 |
| 82 | 0.1754 | 0.1316 | 0.0143 | 0.0102 | 0.1071 | 1.25e-5 |
| 83 | 0.1756 | 0.1318 | 0.0144 | 0.0102 | 0.1072 | 1.25e-5 |
| 84 | 0.1757 | 0.1319 | 0.0144 | 0.0102 | 0.1073 | 1.25e-5 |
| 85 | 0.1760 | 0.1322 | 0.0145 | 0.0101 | 0.1075 | 1.25e-5 |
| 86 | — | — | 0.0146 | 0.0101 | 0.1077 | 1.25e-5 ← watchdog stopped |

Best checkpoint: **epoch 56** (val_task_loss=0.1261, val_1hr=0.0118, val_2hr=0.0111, val_3hr=0.1032)

### Final Results

| Metric | Value |
|--------|-------|
| val_loss | **0.1701** |
| val_mae (normalized) | **0.0971** |
| Best epoch | 56 |
| Quantized model size | 352.24 KB |

**Post-training validation MAE (float, °C):**

| Head | Float MAE |
|------|-----------|
| diff_1hr | 0.20°C |
| diff_2hr | 0.19°C |
| diff_3hr | **1.73°C** |

**PTQ quantized validation MAE (°C):**

| Head | Quantized MAE |
|------|--------------|
| diff_1hr | 3.04°C |
| diff_2hr | 1.09°C |
| diff_3hr | **7.26°C** |

**Permutation Feature Importance:**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | time_of_day_cos2 | 0.0449 |
| 2 | time_of_day_sin2 | 0.0432 |
| 3 | temperature | 0.0425 |
| 4 | solar_slope_30 | 0.0423 |
| 5 | temp_slope_60 | 0.0422 |
| … | … | … |
| 23 | solar_radiation | 0.0361 |
| 24 | time_of_day_sin | 0.0338 |

Full importance range: **0.0338–0.0449** — extremely flat (24 features only). No `temp_lag*` features appear — confirmed removed from the anchor path entirely.

### Success Criteria Assessment

| Criterion | Target | Actual | Met? |
|-----------|--------|--------|------|
| time_of_day_sin2 no longer #1 | Not #1 | #2 (0.0432) — diurnal still dominates top 2 | ⚠️ Partial |
| Physical sensors rise (temperature) | temperature rank improves | #3 (0.0425) — up from #28 in Exp 39 | ✅ |
| Per-head val losses ≥ Exp 39 | 1hr ≤ 4.45e-4, 2hr ≤ 8.19e-4, 3hr ≤ 1.3e-3 | 1hr=0.0118 (26× worse), 2hr=0.0111 (14× worse), 3hr=0.1032 (79× worse) | ❌ |
| best_epoch > 30 | > 30 | 56 | ✅ |

### Conclusion

❌ **FAILED — lag feature removal is the dominant cause, not diurnal routing.**

Exp 40 combined two changes: (1) diurnal out of Conv2D path, (2) `temp_lag60/120/180` and `temp_delta_1` removed from anchor. The lag removal was catastrophic:

| Head | Exp 39 val_loss | Exp 40 val_loss | Factor |
|------|-----------------|-----------------|--------|
| 1hr  | 4.45e-4 | 0.0118 | **26× worse** |
| 2hr  | 8.19e-4 | 0.0111 | **14× worse** |
| 3hr  | 1.3e-3  | 0.1032 | **79× worse** |

In Exp 39, `temp_lag180` (#6, 0.0693) and `temp_lag60` (#7, 0.0688) were the model's primary temperature trajectory anchors. Without them, the anchor Dense(32) has no multi-hour temperature reference and the dilated Conv2D (RF ~251 min) cannot learn equivalent implicit representations — it falls back to diurnal signals instead. The extremely flat importance distribution (0.0338–0.0449, only 24 features) is the anchor-collapse signature: no dominant signal, equally mediocre performance across all inputs.

**Partial positive signal from diurnal routing**: `time_of_day_sin2` dropped from #1 (0.0754 in Exp 39) to #2 (0.0432), and `temperature` rose from #28 (0.0446) to #3 (0.0425). These signal the diurnal routing change had real effect — but buried under the lag regression. **Exp 41 tests the diurnal routing hypothesis properly by restoring lag features.**

---

## Experiment 41: Isolate Diurnal Routing — Restore Lag Features, Keep Diurnal Out of Conv2D

**Status**: Proposed  
**Goal**: Properly isolate the diurnal routing hypothesis from Exp 40 by restoring `temp_lag60`, `temp_lag120`, `temp_lag180`, and `temp_delta_1` to the anchor path. Exp 40 combined two changes (diurnal-out-of-Conv2D + lag-removed-from-anchor), making the dominant cause undiagnosable. Exp 41 tests only the diurnal routing change against an Exp 39 baseline.

### Architecture Changes from Exp 39

| Parameter | Exp 39 | Exp 40 | Exp 41 |
|-----------|--------|--------|--------|
| Conv2D input | 18 sensors incl. diurnal | 14 physical sensors — diurnal removed | **14 physical sensors — diurnal removed** (same as Exp 40) |
| Anchor path | All 28 features incl. `temp_lag60/120/180`, `temp_delta_1` | 24 features — lag/delta removed ❌ | **All features incl. lag/delta** — lag restored ✅ |
| Lag features | In anchor path | ❌ Removed entirely | ✅ Restored to `engineered_features` |
| Diurnal in Conv2D | ✅ In `conv_features` | ❌ Moved to `diurnal_features` after `n_conv` boundary | ❌ Same as Exp 40 (kept out of Conv2D) |
| Reshape kernel | `(SEQ_LEN, 18, 1)` | `(SEQ_LEN, 14, 1)` | `(SEQ_LEN, 14, 1)` |
| Feature-mixing Conv2D | `kernel_size=(1, 18)` | `kernel_size=(1, 14)` | `kernel_size=(1, 14)` |

Feature vector ordering: `conv_features (14)` + `diurnal_features (4: time_of_day_sin/cos/sin2/cos2)` + `engineered_features (lag/slope)`.  
Conv2D slice: `input[:, :, :14]` — StridedSlice, Edge TPU compatible ✅  
Diurnal features still reach anchor via `input[:, -1, :]` (full last-timestep slice).

### Code Change Required

In `train_model_conv2D.py`, confirm:
- `time_of_day_sin`, `time_of_day_cos`, `time_of_day_sin2`, `time_of_day_cos2` are in `diurnal_features` (not `conv_features`)
- `temp_lag60`, `temp_lag120`, `temp_lag180`, `temp_delta_1` are present in `engineered_features`
- `n_conv = len(conv_features)` = 14

### Success Criteria

- Per-head val losses match or beat Exp 39 (1hr ≤ 4.45e-4, 2hr ≤ 8.19e-4, 3hr ≤ 1.3e-3)
- `time_of_day_sin2` no longer #1 feature — should fall behind lag features or solar/temperature
- `temp_lag60/120/180` remain in top 10 feature importance
- `temperature` rises from #28 (Exp 39) — diurnal removal from Conv2D should reduce shortcutting
- best_epoch > 30

---

*Last updated: 2026-06-18 (Exp 40 ✅ complete — best_epoch=56, val_loss=0.1701; lag removal caused 26×/14×/79× per-head regression; Exp 41 proposed to isolate diurnal routing with lag features restored)*

---

## Architecture Research Notes — Beyond Conv2D (2026-06-18)

### Why Conv2D Cannot Learn Implicit Lag Features

The fundamental mismatch: Conv2D+GAP is a **translation-invariant pattern detector** ("what patterns exist *somewhere* in this window?"). Lag features are **position-specific scalar extractions** ("what is the value at *this exact* time offset?").

GlobalAveragePooling destroys positional information by uniformly averaging all 180 timesteps. A filter that activates at exactly t=−60 survives GAP at 1/180th strength. A diurnal signal that fires consistently across all 180 positions survives at full strength. This is why diurnal features always dominate: GAP architecturally selects for translation-invariant patterns over position-specific anchors, regardless of how wide the temporal receptive field is.

Exp 36's saliency analysis confirmed this directly: the Conv2D found exactly one positional shortcut (raw temperature at t=−179 ≈ implicit temp_lag180) and could sustain nothing else through GAP averaging.

### The Architecture That Enables Lag Discovery: Self-Attention

In a self-attention layer:
```
attention_weight(t) = softmax( Q · K(t)ᵀ / √d )
output = Σ_t  attention_weight(t) · V(t)
```
The model learns non-uniform weights over all timesteps — high weight on specific (feature, timestep) pairs, near-zero on everything else. Multi-head attention lets different heads specialise: head 1 might attend to pressure at t=−180, head 2 to temperature at t=−60. The output at t=0 is a **position-specific weighted sum**, not a uniform average, so a single-position signal is not diluted.

The **Temporal Fusion Transformer (TFT)** (Lim et al., 2019) is the most directly applicable architecture:
- **Variable Selection Network**: learns per-feature importance weights at each timestep
- **Multi-head temporal self-attention**: learns which past timesteps matter per head
- Multi-horizon output heads (1hr/2hr/3hr natively)
- Interpretable attention maps and variable importance scores you can inspect post-training

### The Zambretti Forecaster Insight: A Direct Feature Gap

The Zambretti Forecaster (1915) uses **3-hour pressure tendency** (pressure_now − pressure_3hr_ago) as its primary signal. This encodes whether a pressure system is approaching or retreating over a meteorologically meaningful timescale. Current feature set vs. what's missing:

| Current features | Missing |
|---|---|
| `station_pressure` (current) | `pressure_lag_120`, `pressure_lag_180` |
| `pressure_slope_60` (60-min rate) | `pressure_diff_180` = pressure_now − pressure_3hr_ago |

`pressure_slope_60` is a local derivative over 60 minutes. Zambretti's signal is a coarser, longer-window difference — fronts move over hours, not minutes. A TFT would likely rediscover this automatically: attention weights on the pressure column would spike at t≈−150 to t≈−180.

### What Model 5b Was Actually Trying to Achieve

**Correction to earlier framing**: Model 5a (788 KB, val_loss=0.000682) runs fine on Coral TPU — it is the deployed production model. The SRAM overflow in the tracker refers specifically to `Model 5a clean dense_wide_run1`, a wider variant. The standard deployed Model 5a has no deployment problem.

The original motivation for Model 5b (stated in the "Retired goal" section) was: **learn engineered features automatically from raw time series, with no pre-computed lag features**. Not Edge TPU compatibility — Model 5a already solved that. The bet was that a convolutional or recurrent architecture could discover temporal structure (which lags matter, which cross-sensor interactions matter) without a human pre-specifying them.

That original goal was retired at Exp 26 when explicit lag features were added after 25 failed experiments. Since Exp 26, Model 5b has been trying to beat Model 5a accuracy using Conv2D with explicit lags — but Model 5a already runs on Coral and already has explicit lags (hand-engineered). The architectures are converging toward the same solution, and Model 5a wins on every val_loss comparison.

### Do We Still Need the Conv2D?

The Conv2D's unique value proposition was always: learn temporal feature representations that hand-engineering cannot express (trajectory shapes, cross-sensor interactions at multiple timescales). The only thing it offers over explicit scalars is translation-invariant pattern detection. That potential has not materialised meaningfully in 40 experiments — the Dense anchor path consistently does the heavy lifting.

**The honest strategic position**: if the path forward is "use TFT to discover optimal lags → encode as explicit features," that produces an improved Model 5a, not a Model 5b. Model 5a already runs on Coral, and in float terms Exp37 (Dense anchor + explicit lags) was already comparable to it — the Conv2D added nothing beyond what the Dense anchor achieved alone.

The Conv2D remains worth pursuing only if there is evidence it can learn something that explicit features cannot — trajectory shapes, multi-sensor interactions, weather pattern fingerprints. No experiment has confirmed this yet.

### Running TFT on Coral Edge TPU with CPU Fallback

Technically possible: `edgetpu_compiler` partitions the graph automatically — supported ops (Dense, BN, ReLU6, Add, Reshape) go to TPU at INT8; unsupported ops (dynamic MatMul, Softmax over variable dims, LayerNorm) fall back to host CPU at float32. Data transfers at the TPU/CPU boundary are handled automatically.

However, **attention is most of the compute**. The Q·Kᵀ + Softmax + ·V chain is the dominant operation in TFT. If that runs on CPU and only the Dense heads run on TPU, the PCIe/USB transfer overhead makes the effective TPU speedup near zero. The 2-subgraph models in earlier experiments showed the performance cost of TPU/CPU splits even for small unsupported op islands. If most of the model runs on CPU, the TPU delegation buys almost nothing.

### Recommended Path

**Track A (immediate, low cost):** Add Zambretti-motivated explicit pressure lag features to the current architecture. `pressure_lag_180` and `pressure_diff_180` (= station_pressure − pressure_lag_180) are motivated by a century of meteorological practice. `pressure_lag_120` covers the 2-hour window. One training run; could immediately improve the 3hr head, which is the worst-performing head throughout.

**Track B (research, no deployment constraint):** Train a TFT on the same dataset as a pure discovery tool. Inspect attention maps and variable selection weights to find which (feature, lag) pairs drive accuracy beyond what current explicit features capture. Encode the findings — same loop as Exp 36→37 but systematic. The destination is an improved Model 5a (Dense, fully on TPU, no CPU islands), not a new Conv2D architecture.

**The strategic question this raises**: given that Model 5a already runs on Coral and beats every Conv2D result, the primary remaining motivation for Model 5b is the original one — auto-learned features without hand-engineering. If that goal requires TFT (which can't deploy on Coral efficiently), the project may be better served by improving Model 5a's feature engineering directly.

*Last updated: 2026-06-18 (architecture research notes added — corrected Model 5a SRAM claim; restated original 5b motivation as auto-learned features not Edge TPU compat; TFT as discovery tool; strategic direction open question)*
