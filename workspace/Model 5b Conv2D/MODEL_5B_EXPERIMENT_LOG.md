# Model 5b Experiment Log

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

**Results**: ⏳ **PENDING**

---

## Comparison with Model 5a

| Metric | Model 5a | Exp 8 | Exp 9 | Exp 10 | Exp 11 | Exp 12 | Exp 12b |
|--------|----------|-------|-------|--------|--------|--------|---------|
| val_loss | 0.000682 | 0.00617 | ~0.0165 | 0.0039 | 0.0042 | OOM@ep2 | ⏳ |
| val_mae | 0.00445 | 0.01464 | ~0.049 | 0.0130 | ~0.017 | — | ⏳ |
| Best epoch | 97 | 52 | stopped@17 | 25 | ~83 | aborted | ⏳ |
| Model size | 788 KB | 844 KB | — | 125 KB | 1.28 MB | — | ⏳ |
| Quantized MAE 1hr (°C) | — | — | — | **1.07** ❌ | not tested | — | ⏳ |
| Architecture | Dense f=— | Dense f=— | Conv1D d=[1..16] f=64 | Conv1D d=[1..64] f=64 | Conv1D d=[1..64] f=64 | Conv1D d=[1..64] f=128 | Conv1D d=[1..64] f=96 |
| Batch size | 256 | 512 | 512 | 512 | 1024 | 1024 ❌ | 512 |
| Receptive field | N/A | N/A | ~65 steps | ~257 steps | ~257 steps | ~257 steps | ~257 steps |
| Pre-computed lags | Yes | Yes | No | No | No | No | No |
| Edge TPU viable (quant) | Yes | Yes | — | **No** ❌ | not tested | — | ⏳ |

---

*Last updated: 2026-04-10*
*Status: Experiment 12b in progress — 96-filter architecture, batch 512, hardware-safe capacity increase*
