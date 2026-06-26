# Model 5c — Track B Experiment Log

**Script**: `train_model_track_b.py`  
**Target**: val_loss < 0.000373 (beat Model 5a clean dense_wide_run1), INT8 on Coral Edge TPU, 3hr INT8 MAE < 0.930°C (Model 5b deployed bar)

---

## Summary — State After Runs 1–13

### Baselines

| Model | val_loss | INT8 3hr StdDev | Coral TPU |
|-------|----------|----------------|-----------|
| Model 5a deployed | 0.000682 | 0.988°C | ✅ |
| Model 5a clean dense_wide_run1 | 0.000373 | — | ❌ SRAM overflow |
| Model 5b Exp37 deployed | ~0.002 | 0.930°C | ✅ |
| **Track B Run 11/13 (FP32)** | **0.000300** | **0.898–0.907°C** | ⚠️ INT8 degradation |

**FP32 target beaten by 24%. 3hr INT8 beats Model 5b bar. 1hr INT8 remains too degraded for production.**

### Architecture (settled, Runs 10–13)

```
Input(180, 11) → AvgPool(6) → flat(330)
  → Bottleneck(64, ReLU6)
  → [Wide(16, ReLU6)  +  Deep(128→ReLU6 → 64→ReLU6 → 32→ReLU6)]
  → Merge(48) → 3 output heads (Dense(1))
```

No BatchNorm. No residual Add. No interaction path. L2=1e-6. SEQ_LEN=180, AveragePooling(6).

### Feature set (11 features)

| Category | Features |
|----------|----------|
| Temporal | time_of_day_sin/cos/sin2/cos2, day_of_year_sin/cos |
| Temperature | temperature |
| Pressure | pressure_slope_60 |
| Solar | solar_radiation, illuminance, uv |

### Key lessons from Runs 1–13

**SEQ_LEN=1 ceiling (Runs 1–5):** 22-scalar summaries of 3 hours of data hit a hard information ceiling at ~0.48–0.53°C 1hr MAE regardless of feature engineering, architecture size, or regularization. Architectural changes (multi-path, residual, BN, dropout) gave 3–6% improvement; the real bottleneck was the single-timestep input.

**Breakthrough (Run 6):** SEQ_LEN=180 + AveragePooling(6) gave 10–16× FP32 improvement. But INT8 was catastrophically degraded (+810%) due to the interaction path's element-wise square producing an unbounded [0, k²] range.

**Progressive INT8 fixes (Runs 7–13):**
- Remove interaction path → INT8 fixed when no temperature; FP32 catastrophic (Run 7)
- Restore temperature + ReLU6 throughout → FP32 great, INT8 still catastrophic (Run 8)
- Add ReLU6 to Wide path → no INT8 change (Run 9); Wide path was not the cause
- Remove residual Add from Deep path → first to beat FP32 target; INT8 improved but +271% at 1hr (Run 10)
- Remove BatchNorm → best FP32 (0.000300); 2hr/3hr INT8 improved, 1hr INT8 worsened (Run 11)
- Calibration source: train→val → <4% INT8 change; calibration distribution not the cause (Run 13)

**INT8 degradation root cause — remaining hypothesis:** TFLite may not be fusing separate `Dense(n) → Activation("relu6")` Keras layers — the intermediate pre-activation tensor may be quantized with an unbounded scale. Fix: `Dense(n, activation="relu6")` in the constructor, which is more reliably emitted as a single fused FULLY_CONNECTED op.

### Track B INT8 MAE history

| Run | FP32 1hr | INT8 1hr | INT8 2hr | INT8 3hr | Key change |
|-----|---------|---------|---------|---------|------------|
| 6   | 0.041 | 0.373 (+810%) | 0.726 | 0.933 | SEQ_LEN=180 breakthrough |
| 7   | 0.318 | 0.428 (+35%) ✅ | 0.615 | 0.710 | No temperature — FP32 broken |
| 8   | 0.088 | 0.448 (+409%) | 0.861 | 1.421 | Temperature restored |
| 9   | 0.093 | 0.602 (+547%) | 0.949 | 1.249 | Wide path ReLU6 — no change |
| 10  | 0.094 | **0.349** (+271%) | 0.702 | 0.989 | Residual Add removed |
| 11  | **0.091** | 0.522 (+474%) | **0.588** | **0.898** ✅ | BN removed |
| 13  | 0.091 | 0.521 (+474%) | 0.567 | 0.907 ✅ | Val calibration — no change |

---

## Run 14 — Fused Dense+ReLU6 Op (INT8 Fix Attempt), Fresh Training

**Date**: TBD  
**Script**: `train_model_track_b.py`  
**Platform**: Mac Metal (M-series)  
**Results stored in**: `results_5c_trackb_dense_b_run14/`

**Hypothesis**: TFLite's per-tensor INT8 quantization degrades when the pre-activation tensor (between `Dense` and `Activation("relu6")`) is quantized separately as an unbounded intermediate. Using `Dense(n, activation="relu6")` in the Keras constructor tells Keras to represent the activation as part of the layer definition, which should enable the TFLite converter to emit a single fused FULLY_CONNECTED op and quantize only the post-activation [0, 6]-bounded output.

**Configuration changes from Run 13**:
1. **Fuse Dense + activation into single layer**: replace all `Dense(n, ...) → Activation("relu6")` pairs with `Dense(n, activation="relu6", ...)` in both the training model and the fp32 export model. Applied at: Bottleneck, Wide path, and all three Deep path layers.
2. **Requires fresh training**: fused architecture changes layer names → current Run 11/13 checkpoint weights are incompatible. Training from scratch.
3. **Architecture otherwise unchanged**: same 11 features, same layer sizes, L2=1e-6, SEQ_LEN=180, AveragePooling(6), no BN, no residual Add, no interaction path.
4. **SKIP_TRAINING = False**: full 300-epoch training run.

**Architecture** (fused activations):
```
Input(180, 11) → AvgPool(6) → flat(330)
  → Dense(64, activation="relu6", use_bias=False)           ← was Dense(64)→Activation("relu6")
  → [Dense(16, activation="relu6", use_bias=False)          ← was Dense(16)→Activation("relu6")
     + Dense(128, activation="relu6", use_bias=False)        ← was Dense(128)→Activation("relu6")
       → Dense(64, activation="relu6", use_bias=False)
       → Dense(32, activation="relu6", use_bias=False)]
  → Merge(48) → 3 × Dense(1) output heads
```

**Expected outcomes**:
- **INT8**: if op fusion is the root cause, expect recovery toward Run 7-level degradation (+35%) — the hypothesis predicts 1hr INT8 ≈ 0.09–0.15°C (from Run 7 proportions), vs current 0.521°C.
- **FP32**: expect similar to Run 11/13 (0.091/0.092/0.121°C). The fused `Dense(n, activation="relu6")` is mathematically identical to `Dense(n) → relu6` — no change in training dynamics or representational capacity.
- **val_loss**: expect ≈ 0.000300 (same as Run 11/13 given identical architecture and features). Fresh training introduces variance; range 0.000280–0.000350 is plausible.

**Results**:
- val_loss (includes L2): *TBD*
- val_task_loss: *TBD*
- diff_1hr MAE (FP32): *TBD*
- diff_2hr MAE (FP32): *TBD*
- diff_3hr MAE (FP32): *TBD*
- diff_1hr MAE (INT8, n=2000): *TBD*
- diff_2hr MAE (INT8, n=2000): *TBD*
- diff_3hr MAE (INT8, n=2000): *TBD*
- Best epoch: *TBD*
- Final LR: *TBD*
- FP32 TFLite: *TBD* | INT8 TFLite: *TBD*

**Permutation feature importance**: *TBD*

**Key findings**: *TBD*

**Outcome**: *TBD*

**Changes for Run 15 (if Run 14 fixes INT8)**:
- Deploy INT8 TFLite to Coral TPU on Pi and measure 30-day live StdDev
- If live performance matches INT8 validation MAE, Track B is production-ready

**Changes for Run 15 (if Run 14 does NOT fix INT8)**:
- The remaining structural hypothesis is exhausted (interaction path → Wide bounds → residual Add → BN → calibration → op fusion all tested)
- Consider quantization-aware training (QAT): brief fine-tuning of Run 11/13 weights using `tf.quantization.experimental.quantize_model()` at very low LR (1e-6) to allow the model to adapt its weights to quantization noise. Risk: may regress FP32; use a separate QAT checkpoint. This was rejected at Run 8 (Model 5b Exp 25 QAT caused regression) but should be retried with the now-stable architecture.
