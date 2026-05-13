# Model 5b Experiment Log

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

### StdDev Benchmarks (measured over 4.5 years of live InfluxDB data)

| Scenario | StdDev | INT8 steps |
|----------|--------|-----------|
| Model 5a deployed (INT8 TPU) | **0.145°C** | 1.0 |
| Model 5b current (float, unfinished) | **0.333°C** | 3.0 |
| Model 5b best PTQ INT8 (Exp 30, 1hr) | ~0.8–1.0°C est. | ~7–9 |
| Model 5b with QAT, current float ceiling (~0.0026 val_loss) | ~0.15–0.25°C est. | ~1.4–2.3 |
| Model 5b with QAT, if float target met (≤ 0.000682 val_loss) | **~0.10–0.14°C est.** | ~0.9–1.3 |
| Theoretical INT8 floor (perfect model) | 0.032°C | 0.3 |

**Conclusion**: if Model 5b meets the float accuracy target and QAT is applied, it should reach **~0.10–0.14°C StdDev** — marginally better than Model 5a's 0.145°C because of the finer INT8 step size. The practical floor is approximately 1 INT8 step (0.111°C). The current PTQ gap of 3–4× in MAE (float→INT8) must be closed by QAT; PTQ alone has never produced a deployable result for this architecture.

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

*Last updated: 2026-05-12*
