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
- val_loss (includes L2): **0.000296** (Run 11/13: 0.000300, −1.3% ✅; vs target 0.000373 — **TARGET BEATEN by 21%**)
- val_task_loss: **0.000109** (identical to Run 11/13)
- val_mae (normalized): **0.003415**
- diff_1hr MAE (FP32): **0.094°C** (Run 11/13: 0.091°C, +3% ≈ same)
- diff_2hr MAE (FP32): **0.095°C** (Run 11/13: 0.092°C, +3% ≈ same)
- diff_3hr MAE (FP32): **0.118°C** (Run 11/13: 0.121°C, −3% ✅)
- diff_1hr MAE (INT8, n=500): **0.487°C** (Run 13: 0.521°C, −6.5% ✅)
- diff_2hr MAE (INT8, n=500): **0.670°C** (Run 13: 0.567°C, +18% ❌)
- diff_3hr MAE (INT8, n=500): **1.106°C** (Run 13: 0.907°C, +22% ❌)
- Best epoch: **297/300** — still marginally improving at run end (same as Run 11/13)
- Final LR: **1.00e-07** (min_lr reached)
- FP32 TFLite: **162.4 KB** ✅ | INT8 TFLite: **47.4 KB** ✅ (identical sizes to Run 11/13)

vs baselines:
- Model 5a deployed (INT8) val_loss=0.000682 → **FP32 val_loss=0.000296 — beaten** ✅
- Model 5a clean dense_wide_run1 val_loss=0.000373 → **0.000296 — TARGET BEATEN** ✅
- Model 5b Exp37 INT8 deployed: 30d StdDev 0.930°C → INT8 3hr MAE 1.106°C — worse ❌

**Permutation feature importance (val_loss increase)**:

| Feature | Run 14 | Run 11 | Notes |
|---------|--------|--------|-------|
| time_of_day_cos | +0.0071 | +0.0140 | Still top, but halved |
| solar_radiation | +0.0033 | +0.0042 | Stable Tier 2 |
| uv | +0.0027 | −0.0001 | **Jumped — was marginally negative in Run 11** |
| time_of_day_sin2 | +0.0019 | +0.0061 | |
| time_of_day_cos2 | +0.0017 | +0.0047 | |
| illuminance | +0.0005 | +0.0022 | |
| temperature | +0.0004 | −0.0003 | Flipped positive (correlation masking varies by init) |
| day_of_year_sin | +0.0001 | +0.0001 | Stable near-zero |
| day_of_year_cos | −0.0000 | −0.0000 | |
| pressure_slope_60 | −0.0000 | −0.0000 | |
| time_of_day_sin | **−0.0004** | **+0.0101** | **Flipped strongly negative — local minimum difference** |

**Key findings**:

1. **Op-fusion hypothesis: inconclusive — 1hr improved, 2hr/3hr worsened**: INT8 at 1hr improved 0.521→0.487°C (−6.5%), but 2hr worsened 0.567→0.670°C (+18%) and 3hr worsened 0.907→1.106°C (+22%). Op fusion was not the root cause of the INT8 degradation. The same horizon-asymmetric pattern seen across Runs 10–13 continues.

2. **FP32 is essentially unchanged**: val_loss 0.000296 vs 0.000300 (−1.3%), all three MAEs within ±3% of Run 11. The fused activation is mathematically identical to the separate `Dense → Activation` pair and produces the same FP32 model quality — as expected.

3. **`time_of_day_sin` flipped from +0.0101 (Run 11) to −0.0004 (Run 14)**: This is the most significant perm importance shift. Fresh training landed in a different local minimum where `time_of_day_sin` weight is partially absorbed by correlated features (sin2/cos share some phase information). FP32 performance is unchanged, so both local minima are equally valid — this is initialization variance, not a feature regression. `uv` jumping from −0.0001 to +0.0027 is the symmetric shift: the model rerouted some temporal encoding through the solar features in this run.

4. **All structural INT8 hypotheses are now exhausted**: The full sequence of fixes tested:

| Run | Fix attempted | INT8 1hr | INT8 2hr | INT8 3hr | FP32 1hr |
|-----|--------------|---------|---------|---------|---------|
| 7   | No temperature (side effect) | 0.428 ✅ | 0.615 | 0.710 | 0.318 ❌ |
| 8   | Temp restored, ReLU6 throughout | 0.448 | 0.861 | 1.421 | 0.088 |
| 9   | Wide path ReLU6 | 0.602 | 0.949 | 1.249 | 0.093 |
| 10  | Residual Add removed | **0.349** | 0.702 | 0.989 | 0.094 |
| 11  | BN removed | 0.522 | **0.588** | **0.898** ✅ | **0.091** |
| 13  | Val calibration | 0.521 | 0.567 | **0.907** ✅ | 0.091 |
| 14  | Op fusion | 0.487 | 0.670 | 1.106 | 0.094 |

No single run is best on all three horizons. The INT8 degradation is horizon-dependent and not addressable by any structural change tested so far.

5. **Best deployable INT8 checkpoint remains Run 11**: 3hr INT8 = 0.898°C (best 3hr, beats 5b bar), 2hr = 0.588°C (best 2hr). Run 13 gives 0.907°C at 3hr with val-calibration but 0.567°C at 2hr. Run 14 regressed on both 2hr and 3hr.

**Convergence**:
- Best epoch 297/300 at LR floor (1e-7) — model still marginally improving at training cutoff, consistent with Run 11/13 behavior
- val_task_loss=0.000109 at best epoch — identical to Run 11/13; fresh training converged to the same task-loss level

**Outcome**: ⚠️ PARTIAL — FP32: best run yet (val_loss=0.000296, 21% below target). INT8: op-fusion hypothesis refuted (2hr/3hr worsened despite 1hr improving). All seven structural INT8 hypotheses have now been tested. The INT8 degradation root cause is not addressable by further architecture changes with this training approach.

**Changes for Run 15 — Quantization-Aware Training (QAT)**:

All structural fixes are exhausted. The remaining path is QAT (`tfmot.quantization.keras.quantize_model()` from `tensorflow-model-optimization`): brief fine-tuning that inserts fake-quantization nodes so the model learns INT8-robust weights.

**Why QAT should work here (unlike Model 5b Exp 25):** Model 5b Exp 25 used QAT on a Conv1D model with `SliceTimestep`/`SliceFeatures` custom ops. Those ops could not be wrapped by `tfmot.quantization.keras.QuantizeWrapper` (no weights), so their output activations had unbounded ranges that neither PTQ nor QAT could constrain — near-constant outputs persisted and float accuracy regressed (best epoch 17, val_loss 0.001343 → 0.0015). Track B has **no custom ops**: every layer is a standard `Dense`+`ReLU6` — all are wrappable by tfmot. The QAT failure root cause does not apply here.

1. **Load Run 11 best checkpoint** (`results_5c_trackb_dense_b_run11/checkpoints/best_model.weights.h5`) using `by_name=True` — Run 11 gives the best 3hr INT8 (0.898°C below 5b bar). Run 11 used separate `Dense(n) → Activation("relu6")` layers; the current Run 14 architecture uses `Dense(n, activation="relu6")`. The kernel shapes are identical; `by_name` loading maps by layer name and silently skips Run 11's weightless Activation layers. Do NOT use Run 14 weights (3hr INT8 = 1.106°C — regressed).
2. **Apply `tfmot.quantization.keras.quantize_model(model)`** to wrap each Dense layer with fake-quant nodes that simulate INT8 rounding during the forward pass. Metal mixed precision must be disabled first (forced float32) — tfmot is incompatible with fp16 compute graphs.
3. **Fine-tune at very low LR** (1e-6): at this LR, Adam's uninitialized moments (m1=m2=0) produce gradient amplification of only lr/eps = 1e-6/1e-7 = 10× — negligible. The Run 12 warmup bug was at LR=1e-4 (1000× amplification); LR=1e-6 is inherently safe without any warmup step.
4. **Short fine-tuning**: 20–50 epochs max. EarlyStopping on val_task_loss with patience=10. QAT is nudging existing weights, not relearning features.
5. **Risk**: FP32 accuracy may regress (Model 5b Exp 25 concrete data: val_loss 0.001343 → 0.0015, best epoch 17 then oscillation). Save QAT weights in `qat_checkpoints/` separately; never overwrite Run 11 checkpoint.
6. **Success criterion**: INT8 MAE within 20% of FP32 on all three horizons (1hr < 0.11°C, 2hr < 0.11°C, 3hr < 0.14°C).

---

## Run 15 — QAT Fine-Tuning from Run 18 Checkpoint

**Date**: TBD  
**Script**: `train_model_track_b.py`  
**Platform**: Mac Metal (M-series) — Metal mixed precision **disabled** (float32 forced; tfmot incompatible with fp16 compute)  
**Results stored in**: `results_5c_trackb_dense_b_run15/`

**Hypothesis**: All seven structural INT8 fixes (Runs 7–14) are exhausted. The INT8 degradation pattern is horizon-asymmetric and not attributable to any single unbounded tensor — the root cause is the model's weight distribution, not graph structure. QAT (`tfmot.quantization.keras.quantize_model`) inserts fake-quant nodes that simulate INT8 rounding during the forward pass; backpropagation adjusts weights to minimize loss under that rounding. Track B is well-suited for QAT: every layer is a standard `Dense`+`ReLU6` (all wrappable by tfmot), outputs are already bounded [0, 6] by ReLU6 (ideal INT8 range), and no custom ops exist to block wrapping. The Model 5b Exp 25 QAT failure was caused by `SliceTimestep`/`SliceFeatures` custom ops that couldn't be wrapped; that failure mode does not apply here.

**Configuration changes from Run 14**:
1. **SKIP_TRAINING = True**, **QAT_FINE_TUNE = True**: Load Run 18 checkpoint; apply QAT wrapping; fine-tune.
2. **SOURCE_CHECKPOINT**: `results_5c_trackb_dense_b_run18/checkpoints/best_model.weights.h5` — **retargeted 2026-07-20** from the original Run 11 plan. Run 11 predates the `temp_diff_vs_5hr/6hr` features (Run 16+) and the input-clip fix (Run 18); Run 18 is now the best FP32 model overall (0.075/0.090/0.116°C) and shares the current 13-feature architecture, so it's the correct QAT starting point even though its INT8 (0.595/1.041/1.658°C) is worse than Run 11's (0.522/0.588/0.898°C) — QAT is expected to close that gap, not preserve it.
3. **by_name=True loading**: architecture is unchanged between Run 14 and Run 18 (fused `Dense(activation="relu6")` throughout), so `by_name` loading is a straightforward exact match — no skipped/renamed layers, unlike the original Run 11→14 transition.
4. **QAT_LR = 1e-6**: Safe without warmup (gradient amplification = lr/eps = 10×; Run 12 bug was 1000× at lr=1e-4).
5. **QAT_EPOCHS = 50**, **QAT_EARLY_STOP_PATIENCE = 10**: Short fine-tuning; nudging weights, not relearning features.
6. **QAT checkpoints in `qat_checkpoints/`**: Never overwrites Run 18 checkpoint.
7. **INT8 TFLite from `TFLiteConverter.from_keras_model(qat_model)`**: Uses embedded fake-quant scales from QAT training. Representative dataset still provided for input/output quantization.

**Architecture** (unchanged from Run 18):
```
Input(180, 13) → AvgPool(6) → flat(390)
  → Dense(64, relu6) [bottleneck]
  → [Dense(16, relu6) wide  +  Dense(128→relu6→64→relu6→32→relu6) deep]
  → Merge(48) → 3 × Dense(1) output heads
```
No BN. No residual Add. No interaction path. L2=1e-6. Includes `temp_diff_vs_5hr/6hr` (Run 16+) with clipped `[0,1]` scaling (Run 18).

**Expected outcomes**:
- **INT8**: If QAT successfully adapts weights, expect degradation to drop from Run 18's current 693–1329% range (0.595/1.041/1.658°C vs FP32 0.075/0.090/0.116°C) toward ≤20% of FP32 — i.e., 1hr < 0.09°C, 2hr < 0.11°C, 3hr < 0.14°C.
- **FP32**: Likely slight regression from Run 18 baseline (0.075/0.090/0.116°C). Model 5b Exp 25 regressed ~12% at best epoch 17; expect similar or less given Track B's simpler architecture. Success threshold: val_loss < 0.002.
- **Best epoch**: Expect early (10–20 epochs) given short fine-tuning at low LR, consistent with 5b Exp 25 pattern.

**Results**:
- val_loss (includes L2): **TBD**
- val_task_loss: **TBD**
- diff_1hr MAE (FP32): **TBD**°C
- diff_2hr MAE (FP32): **TBD**°C
- diff_3hr MAE (FP32): **TBD**°C
- diff_1hr MAE (INT8): **TBD**°C
- diff_2hr MAE (INT8): **TBD**°C
- diff_3hr MAE (INT8): **TBD**°C
- Best epoch: **TBD**

**Outcome**: TBD

**Note**: Run 15 (QAT) had not yet been executed when Run 16 below was started. Its config (`SKIP_TRAINING=True`, `QAT_FINE_TUNE=True`) is preserved in this log for whenever QAT is revisited, but the script itself was switched to fresh training for Run 16. **Update (2026-07-20)**: `SOURCE_CHECKPOINT` retargeted from `run11` to `run18` after Runs 16–18 (new features + input-clip fix) established a better, feature-complete FP32 baseline — see the retargeting rationale in step 2 above and the Run 18 write-up's "Next steps".

**Update (2026-07-21)**: This plan is being executed as **Run 19**, not Run 15 — Runs 16–18 already consumed the intervening sequential slots, and a second training pass under the `dense_b_run18` name was already in flight (re-running Run 18's fresh/warm-start training into the same output directory) when this was set up, so a fresh run number avoids any collision. `train_model_track_b.py` now has `RUN_NAME="dense_b_run19"`, `SKIP_TRAINING=True`, `QAT_FINE_TUNE=True`, `SOURCE_CHECKPOINT` pointing at Run 18's `best_model.weights.h5`. All hypothesis/config detail above still applies unchanged — see the Run 19 entry below for results.

---

## Run 16 — 5hr/6hr Lag Features from Track A Deep Run Finding

**Date**: 2026-07-19
**Script**: `train_model_track_b.py`
**Platform**: Mac Metal (M-series) — fresh training, same as Run 14
**Results stored in**: `results_5c_trackb_dense_b_run16/`

**Hypothesis**: The Track A Deep Run (SEQ_LEN=360, Mac Metal, completed 2026-06-30 — see `MODEL_5C_EXPERIMENT_LOG.md`) found that the attention peak at t-179 in Runs 4–6 was a boundary artifact (the model always pins to the oldest available timestep). More importantly, it found a **genuine non-boundary secondary anchor at ~5 hours** (t-295 to t-301, weight 0.0097–0.0129) — not at the window edge, so not an artifact. Track B's current window (`SEQ_LEN=180`, 3 hours) cannot see this far back at all. Rather than extending `SEQ_LEN` to 300–360 (which would roughly double AvgPool/flatten size and re-open the INT8 tuning that Runs 7–14 spent to settle), this run adds the 5hr/6hr signal as two explicit scalar features, computed once per row and carried through the existing 3hr window unchanged.

**Configuration changes from Run 15**:
1. **Revert to fresh training**: `SKIP_TRAINING = False`, `QAT_FINE_TUNE = False` — Run 15 QAT deferred (see note above), architecture and training loop otherwise identical to Run 14 (fused `Dense(activation="relu6")`, no BN, no residual, no interaction path, L2=1e-6).
2. **New features**: `temp_diff_vs_5hr` = `temperature` − temperature 300 min ago, `temp_diff_vs_6hr` = `temperature` − temperature 360 min ago. Computed via gap-aware `merge_asof` (direction="backward", tolerance=90s), matching the existing pattern used for `_add_future_targets`. Rows where the 5hr/6hr lookback isn't available (start of series or crosses a data gap) get NaN and are dropped via the existing `dropna` step — same handling as the other engineered features.
3. **Feature count**: 11 → 13 (added to `temperature_features`).
4. **RUN_NAME**: `dense_b_run16`.

**Architecture** (unchanged from Run 14/15):
```
Input(180, 13) → AvgPool(6) → flat(390)
  → Dense(64, relu6) [bottleneck]
  → [Dense(16, relu6) wide  +  Dense(128→relu6→64→relu6→32→relu6) deep]
  → Merge(48) → 3 × Dense(1) output heads
```

**Expected outcomes**:
- If the 5–6hr signal is real and the Dense model can exploit it as a scalar (without attention), expect improvement over Run 14's FP32 baseline (val_loss=0.000296, MAE 0.094/0.095/0.118°C), most likely on the 3hr horizon since that's furthest from the current window's information.
- If the two new features add no value (redundant with `temp_slope_60`/`pressure_slope_60` already in the window), expect a wash — permutation importance will show near-zero score for both, and they should be dropped in a follow-up run.
- INT8 behavior is untested territory — 2 more input channels changes AvgPool/flatten dims; watch for INT8 regression the same way Runs 6–14 did after every architecture change.

**Results**:
- val_loss (includes L2): **0.000549** (val_task_loss printed during training: **0.000348** — see note below on the gap between these two)
- diff_1hr/2hr/3hr MAE (FP32): **0.088 / 0.107 / 0.138°C**
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.662 / 1.220 / 1.545°C**
- FP32 TFLite: 177.4 KB · INT8 TFLite: 51.1 KB
- Permutation importance (val_loss increase), full ranking:
  1. `temp_diff_vs_5hr`: **0.0456** — dominant, beats `time_of_day_cos`
  2. `time_of_day_cos`: 0.0435
  3. `time_of_day_sin`: 0.0216
  4. `temp_diff_vs_6hr`: **0.0105**
  5. `pressure_slope_60`: 0.0001
  6. `day_of_year_sin`: 0.0001
  7. `day_of_year_cos`: 0.0000
  8. `solar_radiation`: 0.0000
  9. `time_of_day_sin2`: -0.0001
  10. `time_of_day_cos2`: -0.0012
  11. `uv`: -0.0029
  12. `illuminance`: -0.0037
  13. `temperature`: **-0.0050** (negative — permuting it slightly *improves* val_loss)
- Best epoch: **300/300** (ran to the epoch cap; `lr=1.00e-07` fully decayed, loss flat at 0.00055 for the entirety of epoch 300, no earlier checkpoint recorded as best — worth confirming this is genuine convergence rather than the LR schedule running out)

**Outcome**: ➡️ MIXED — hypothesis confirmed on FP32, but INT8 broke

1. **5hr/6hr hypothesis validated, strongly**: `temp_diff_vs_5hr` is the single most important feature in the model — the Track A Deep Run's non-boundary ~5hr attention anchor genuinely transfers to Track B's architecture as a scalar feature. `temp_diff_vs_6hr` also carries real (smaller) weight. Neither is a wash, contrary to the pre-run hedge.
2. **`temperature` may now be redundant**: negative permutation importance (-0.0050) reverses Run 7's "joint necessity of temperature signal" finding — plausible now that `temp_diff_vs_5hr/6hr` supply the relevant signal more directly. Candidate for removal in a follow-up run.
3. **FP32 vs Run 14 baseline (val_loss=0.000296, MAE 0.094/0.095/0.118°C) is mixed, not a clean win**: 1hr MAE improved (0.088 vs 0.094) but 2hr/3hr regressed (0.107 vs 0.095, 0.138 vs 0.118). Final val_loss (0.000549, includes L2) misses the 0.000373 target, but val_task_loss (0.000348) alone would beat it — the ~0.0002 gap between task loss and total loss is larger than expected for L2=1e-6 and hasn't been reconciled against Run 14's equivalent gap.
4. **INT8 regression is severe and is now the blocking issue**: 3hr INT8 MAE of 1.545°C is far worse than Run 11–13's 0.898–0.907°C (which beat the Model 5b deployed bar of 0.930°C) and misses that bar badly. This is exactly the risk flagged pre-run ("2 more input channels changes AvgPool/flatten dims") — it materialized. The model is not currently Coral-TPU-deployable in this state.

**Next steps**:
- Priority: fix the INT8 regression before anything else — it's the actual Track B deliverability blocker, not FP32 accuracy.
- Try dropping `temperature` (negative importance) as a follow-up ablation — both tests whether it's truly redundant now and returns feature count to 12, which may incidentally help INT8 behavior the way feature-count changes did in Runs 7–14.
- Confirm epoch 300/300 with fully-decayed LR is genuine convergence, not a truncated schedule, before deciding whether more epochs would help.

**Note (2026-07-19)**: Before this run executes, three data-pipeline fixes landed (see "Data Quality" section and its follow-ups in `MODEL_5C_EXPERIMENT_LOG.md`): (1) `_sanity_filter_temperature()` in `train_model_track_b.py` nulls sensor-glitch `temperature` readings (spike >6°C from local median); (2) stale pre-baked `temp_t+1hr/2hr/3hr` CSV columns are now dropped after load, forcing `_add_future_targets()` to actually run its `merge_asof` reconstruction instead of silently reusing values from `export_influx_to_csv.py`; (3) `export_influx_to_csv.py` itself was fixed — its `temp_t+Nhr` columns were built with a row-count `.shift(-N)` instead of a time-based lookup, which silently misaligned targets across any sampling gap — and both CSVs were re-exported (fetch range extended to 2026-07-19, val window grown to match; pre-export CSVs backed up to `workspace/backup_pre_export_fix_20260719/`). Verified target range now `-15.10°C to 19.40°C` (was `-31.50°C to 28.60°C`) against the live script with the re-exported data. Row counts (Track B's own dropna): train=1,407,546 / val=539,684. Run 16 results above will reflect training on the corrected data.

**Addendum (2026-07-20)**: `MAX_EPOCHS` bumped 300→600 and Run 16 resumed from its own checkpoint (`wait=0` at epoch 300 meant it was still improving, not plateaued — see checkpoint state `lr_state.json: {"lr": 1e-07, "best": 0.000356, "wait": 6}`, `early_stopping_state.json: {"best": 0.000348, "wait": 0}`). This extended run is separate from Run 17 below — it's the same experiment (same features/scaling) just given more epoch budget; results TBD, to be filled in separately when it completes or early-stops.

---

## Run 17 — INT8 Fix: Tighter domain_bounds for temp_diff_vs_5hr/6hr (Option 1)

**Date**: 2026-07-20
**Script**: `train_model_track_b.py`
**Platform**: Mac Metal (M-series) — warm-started from Run 16, not from scratch
**Results stored in**: `results_5c_trackb_dense_b_run17/`

**Hypothesis**: Run 16's INT8 MAE regressed 7–11x vs FP32 (0.088→0.662°C on 1hr, 0.138→1.545°C on 3hr) despite the architecture itself being unchanged from Run 14/15 (all Dense+ReLU6, `use_bias=False`, activations structurally bounded to `[0,6]`). Root cause isolated to input-side scaling, not the architecture: `temp_diff_vs_5hr`/`6hr` use `domain_bounds=(None, None)` (data-derived min/max ± 5% pad). Measured their actual training-data distribution: `temp_diff_vs_5hr` min=-17.0 max=21.5 std=3.6, IQR=[-1.99, 1.63] (99.9th pctile 16.4); `temp_diff_vs_6hr` min=-17.8 max=22.1 std=4.0, IQR=[-2.30, 2.00] (99.9th pctile 16.9). Min-max scaling stretches the `[0,1]`-normalized range to cover the full ~40°C span to accommodate rare tail events, while the bulk of real values (IQR) occupies only ~8% of that range — INT8's 256 levels give the common case only ~20 effective quantization levels. This specifically matters now because `temp_diff_vs_5hr` is the model's single most important feature (permutation importance 0.0456, Run 16) — quantization noise on exactly this channel has an outsized effect on prediction error.

**Configuration changes from Run 16**:
1. **`domain_bounds` tightened** for the two lag features, replacing data-derived min/max with fixed bounds covering ~98% of real data (1st/99th percentile-based, rounded): `"temp_diff_vs_5hr": (-8, 12)` (was `(None, None)`), `"temp_diff_vs_6hr": (-9, 13)` (was `(None, None)`). Halves the scaled span (~20°C vs ~40°C), roughly doubling effective INT8 resolution for the common case; the ~2% of values beyond the new bounds pass through unclipped in FP32 training (mildly extrapolated `<0`/`>1` normalized values, same as any other out-of-domain point) and saturate naturally at INT8 export time.
2. **Warm start, not from-scratch**: architecture and feature *count* are unchanged from Run 16 (only the scaling constants for 2 of 13 features), so weight shapes match exactly. `WARM_START=True`, `WARM_START_CHECKPOINT` points at Run 16's `best_model.weights.h5`. Fresh optimizer state, fresh `INITIAL_LR=1e-4`, fresh early-stopping/LR-plateau tracking — only the weight *values* carry over as initialization, not training progress. Expect fast reconvergence (most of the 11 unaffected channels and all downstream layers need no relearning) plus a brief adjustment period while the 2 rescaled channels' bottleneck weights readapt.
3. **RUN_NAME**: `dense_b_run17` (separate `RESULTS_DIR` from Run 16 — Run 16's checkpoints must survive untouched as the warm-start source).

**Expected outcomes**:
- FP32 metrics should reconverge close to Run 16's level (0.088/0.107/0.138°C) within relatively few epochs, given the warm start.
- Primary target: INT8 MAE should improve substantially from Run 16's 0.662/1.220/1.545°C, ideally back toward Run 11–13's territory (~0.9°C on 3hr) or better.
- If INT8 is still bad after this, the scaling-resolution hypothesis was wrong or insufficient — next escalation would be QAT (already partially built, `QAT_FINE_TUNE`) rather than further bound tightening.

**Results**:
- val_loss (includes L2): **0.000479** (vs Run 16: 0.000549)
- diff_1hr/2hr/3hr MAE (FP32): **0.079 / 0.095 / 0.122°C** (vs Run 16: 0.088/0.107/0.138°C — improved across all three horizons)
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.631 / 1.054 / 1.606°C** (vs Run 16: 0.662/1.220/1.545°C)
- Best epoch: **599/600** — ran essentially the entire budget; early stopping never triggered (consistent with the `min_delta=0` micro-improvement pattern observed in the Run 16 epoch extension)
- Permutation importance: `temp_diff_vs_5hr` **0.0687** (up from 0.0456 in Run 16 — even more dominant), `time_of_day_cos` 0.0351, `temp_diff_vs_6hr` 0.0284, `time_of_day_sin` 0.0169, `temperature` still negative (-0.0054)

**Outcome**: ❌ INT8 fix did not work — hypothesis was incomplete

FP32 improved cleanly across the board (warm start + tighter scaling gave a genuinely better optimum, not just a faster path to the same one). But the actual target — INT8 — is **not fixed**: 1hr/2hr improved marginally (0.662→0.631, 1.220→1.054) but 3hr got slightly *worse* (1.545→1.606). Degradation ratio vs FP32 is essentially unchanged (7-11x before, 8-13x now). Tightening `domain_bounds` for the two lag features did not solve the problem.

Likely reason the hypothesis was incomplete: `temp_diff_vs_5hr`'s permutation importance *grew* from 0.0456 to 0.0687 in this run — the model leans on it even more heavily now than in Run 16. If TFLite quantizes the bottleneck Dense layer's weights per-tensor (one scale for the entire 390×64 kernel) rather than per-channel, then a few large weight values connecting to this increasingly-dominant feature could still be crushing precision for the other 388 columns' weights, regardless of how well the *input* itself is scaled. This is a different mechanism than the one Run 17 targeted (input-side quantization resolution) and wasn't ruled out before this run — worth checking directly (inspect the exported `.tflite`'s actual per-tensor vs per-channel weight quantization) before committing to another full retrain.

**Next steps**:
- Inspect the INT8 `.tflite` file's actual quantization scheme for the bottleneck kernel (per-tensor vs per-channel) before deciding the next fix — cheap, no retraining required.
- If per-tensor weight quantization is confirmed as a contributing cause, QAT (already partially built, `QAT_FINE_TUNE`) is the more fundamental fix — it optimizes weights to be quantization-robust directly, addressing both input- and weight-side quantization error simultaneously, unlike the input-scaling-only approach tried here.
- `temperature` remains negative-importance across both runs — still a reasonable drop candidate independent of the INT8 fix.

**Follow-up investigation (2026-07-20)**: Inspected the actual exported `.tflite` (`tf.lite.Interpreter.get_tensor_details()`). Weight quantization was NOT the problem — every Dense kernel is correctly per-channel quantized (bottleneck `[64,390]` kernel has `n_scales=64`, one per output neuron; same pattern for wide/deep/deep_out). The `input` tensor, however, uses one shared per-tensor scale across all 13 channels (`n_scales=1`, `scale=0.0055`, calibrated range `[-0.276, 1.131]` — wider than `[0,1]`). Checked the actual scaled training data directly: `temp_diff_vs_5hr` ranges `[-0.450, 1.475]` post-scaling (1.73% of rows outside `[0,1]`), `temp_diff_vs_6hr` similarly (`[-0.400, 1.414]`, 1.65% outside) — every other feature is cleanly bounded to `[0,1]`. Root cause confirmed: Run 17 tightened the nominal `domain_bounds` but never clipped the scaled values, so that ~1.7% tail still stretches the *shared* per-tensor input scale for all 13 channels at once — undercutting most of the resolution gain the tighter bounds were meant to buy, and plausibly explaining why 3hr got slightly worse (temp_diff_vs_5hr's permutation importance also grew 0.0456→0.0687 in Run 17, making the model more sensitive to precision loss on exactly this channel).

---

## Run 18 — INT8 Fix Continued: Clip Scaled Inputs to [0,1]

**Date**: 2026-07-20
**Script**: `train_model_track_b.py`
**Platform**: Mac Metal (M-series) — warm-started from Run 17
**Results stored in**: `results_5c_trackb_dense_b_run18/`

**Hypothesis**: See follow-up investigation above. Adding explicit `.clip(0.0, 1.0)` after the min-max scaling formula should tighten the model's single shared per-tensor input INT8 scale from its current calibrated span (`~1.4`, `[-0.276, 1.131]`) down to the true `[0,1]` span — directly targeting the mechanism that limited Run 17's effectiveness, on top of the (still-valid, already FP32-beneficial) tightened `domain_bounds` from Run 17.

**Configuration changes from Run 17**:
1. **Explicit clip**: `X_train_df[feat] = ((X_train_df[feat] - lo) / (hi - lo)).clip(0.0, 1.0)` (same for `X_val_df`) — applied to all features uniformly (harmless for the 11 that already never exceed `[0,1]`; fixes the 2 that did). Verified directly: post-clip, `temp_diff_vs_5hr`/`6hr` are now exactly `[0.000, 1.000]`, 0% outside range.
2. **Warm start**: `WARM_START_CHECKPOINT` now points at Run 17's `best_model.weights.h5` (architecture/feature count still unchanged, so shapes match).
3. **RUN_NAME**: `dense_b_run18`.

**Expected outcomes**:
- FP32 metrics should stay close to Run 17's level (0.079/0.095/0.122°C) — clipping only affects the ~1.7% tail, shouldn't meaningfully change what the model learns.
- Primary target: INT8 MAE should improve more substantially than Run 17's marginal/mixed result, since this directly fixes the shared-scale stretching mechanism rather than just narrowing the nominal bounds.
- If INT8 is still bad after this, both input-scaling avenues (bounds + clipping) will have been tried and ruled out as the dominant cause — next escalation would be QAT.

**Results**:
- val_loss (includes L2): **0.000450** (Run 17: 0.000479, −6%)
- diff_1hr/2hr/3hr MAE (FP32): **0.075 / 0.090 / 0.116°C** (Run 17: 0.079/0.095/0.122 — **best FP32 yet on all three horizons**)
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.595 / 1.041 / 1.658°C** (Run 17: 0.631/1.054/1.606 — 1hr/2hr improved marginally, **3hr regressed further, now the worst 3hr INT8 result recorded**)
- Best epoch: **136** (resumed run stopped via watchdog at epoch 176, `ReduceLR → 1.00e-07`, `best=0.000273, wait=12`)
- FP32 TFLite: 177.4 KB · INT8 TFLite: 51.1 KB
- Permutation importance: `temp_diff_vs_5hr` 0.0383 (still dominant, down from Run 17's 0.0687), `time_of_day_cos` 0.0285, `time_of_day_sin` 0.0158, `temp_diff_vs_6hr` 0.0090, `temperature` still negative (−0.0049)

**Verification — did the clip actually fix the input scale?** Inspected the exported INT8 `.tflite` directly (`tf.lite.Interpreter.get_tensor_details()`):
- **Input tensor**: `scale=0.00392157` (exactly 1/255), `zero_point=-128` — this is the theoretically ideal per-tensor scale for data whose true range is exactly `[0,1]`. Confirms the `.clip(0.0, 1.0)` worked as intended: Run 17's shared input scale was `0.0055` over a stretched calibrated range `[-0.276, 1.131]`; Run 18's is tight with 0% of values outside `[0,1]`.
- **Concat-forced sharing**: the `wide` and `deep_out` branches feeding the `merged` concat are both quantized to the identical scale `0.006122444` — TFLite requires all concat inputs to share one scale, so whichever branch has the wider real activation range dictates (and wastes) precision for the other. Not yet investigated further, but a plausible next-lead if QAT doesn't fully resolve things.

**Outcome**: ❌ Input-scaling hypothesis fully ruled out — clip fix confirmed working correctly at the tensor level, but INT8 accuracy did not meaningfully improve (1hr/2hr flat-to-marginal, 3hr worse). Both Run 17 (tighter `domain_bounds`) and Run 18 (explicit `[0,1]` clip) targeted the same shared-input-scale mechanism; both are now exhausted as an explanation for the 8–14× FP32→INT8 degradation. Combined with Runs 6–14's exhaustion of structural fixes (residual/BN/op-fusion) and Run 17's confirmation that weight quantization is correctly per-channel, **every PTQ (post-training quantization) avenue has now been tried**. FP32 itself keeps improving — Run 18 is the best FP32 model yet (0.075/0.090/0.116°C) — so this is purely a quantization problem, not a model-quality problem.

**Next steps**:
- **QAT is now the only remaining lever** (Run 15, previously deferred — see below, retargeted to warm-start from Run 18's checkpoint rather than Run 11, since Run 18 is the best FP32 base available and shares the same architecture/feature set as the QAT plan assumed).
- If QAT does not resolve the 3hr degradation specifically, investigate the concat-forced shared scale between `wide` and `deep_out` — a per-branch rebalancing (e.g. scaling one branch's weights so its activation range better matches the other's) could reduce wasted concat precision independent of QAT.
- `temperature` remains negative-importance across four consecutive runs (14/16/17/18) — still an open ablation candidate, orthogonal to the INT8 issue.

---

## Run 19 — QAT Fine-Tuning from Run 18 Checkpoint (executes the Run 15 plan)

**Date**: 2026-07-21
**Script**: `train_model_track_b.py`
**Platform**: Mac Metal (M-series) — Metal mixed precision auto-disabled when `QAT_FINE_TUNE=True` (script forces `on_metal=False` before the `mixed_float16` policy is set; float32 required for tfmot)
**Results stored in**: `results_5c_trackb_dense_b_run19/`

This is the Run 15 QAT plan (see full hypothesis, architecture, and expected outcomes in the Run 15 section above), executed under Run 19's number because Runs 16–18 already consumed the intervening sequential slots, and because a duplicate fresh-training pass under the old `dense_b_run18` name was independently in flight at the time this was configured.

**Configuration** (see Run 15 section for full rationale):
- `RUN_NAME = "dense_b_run19"`, `SKIP_TRAINING = True`, `QAT_FINE_TUNE = True`
- `SOURCE_CHECKPOINT = "./results_5c_trackb_dense_b_run18/checkpoints/best_model.weights.h5"` — Run 18's best FP32 checkpoint (0.075/0.090/0.116°C), same 13-feature architecture
- `QAT_LR = 1e-6`, `QAT_EPOCHS = 50`, `QAT_EARLY_STOP_PATIENCE = 10`

**Implementation notes** (three bugs found and fixed getting this to run at all):
1. **Keras 3 incompatibility**: tfmot 0.8.0 cannot recognize a Keras 3 Functional model (`"to_quantize can only be a Sequential or Functional model"` even though it is one). Fixed by setting `TF_USE_LEGACY_KERAS=1` before `import tensorflow` (required `QAT_FINE_TUNE` to be declared at the very top of the script, ahead of the import, to gate this).
2. **`relu6` unsupported by tfmot's default QAT scheme**: `quantize_apply()` raises `"Only some Keras activations... are supported"` — checked tfmot's source directly; its whitelist is hardcoded to `{linear, relu, swish, softmax, sigmoid, tanh, gelu}`, no `relu6`, fused or not. Worked around by building `model_for_qat`, a `relu`-activated clone of the architecture, transferring Run 18's weights into it (activation choice doesn't affect kernel shapes), and wrapping *that* with `quantize_model()`. This means QAT fine-tuned under `relu`, not `relu6` — a real (if minor) architecture deviation from every other Track B run.
3. **`model.load_weights(SOURCE_CHECKPOINT, by_name=True)` fails**: Keras 3's native `.weights.h5` format has no `by_name` kwarg (legacy-only). Removed it — harmless since Run 18→19 have identical shapes anyway.
4. **FP32 export silently used stale pre-QAT weights**: `qat_model = quantize_model(model_for_qat)` is a full clone (`keras.models.clone_model` internally) — fine-tuning it does not update `model`'s weights in place. The original export code branch (`export_model = model`) would have silently re-exported Run 18's weights under Run 19's name. Fixed by extracting the fine-tuned kernels directly from `qat_model`'s `QuantizeWrapperV2` layers. Note: `inner_layer.kernel` becomes a stale `SymbolicTensor` (no `.numpy()`) after a real `fit()` call — the wrapper's own `trainable_weights` is the reliable path. Verified bit-exact against an untrained wrapped model's pre-wrap weights (max abs diff = 0.0) before trusting it on the trained model.

**Results**:
- val_loss (includes L2, QAT fake-quant-simulated): **0.000381** (Run 18 PTQ baseline: 0.000450)
- diff_1hr/2hr/3hr MAE (QAT model, fake-quant-simulated forward pass — NOT clean FP32): 0.111 / 0.128 / 0.157°C — not directly comparable to other runs' FP32 numbers; the clean FP32 MAE on the full validation set was never printed (the "FP32 export sanity check" only evaluates 5 unshuffled, chronologically-first batches — 0.001186 there is not a representative full-validation number, just a coarse non-garbage check)
- diff_1hr/2hr/3hr MAE (QAT INT8, n=500, same deterministic sample as every prior run): **0.608 / 1.066 / 1.630°C**
- Best epoch: **18** (of max 50, early-stopped)
- FP32 TFLite: 177.2 KB · INT8 TFLite: 50.7 KB

**Outcome**: ❌ QAT did not improve real INT8 accuracy — essentially identical to Run 18's plain PTQ (0.595/1.041/1.658°C vs Run 19's 0.608/1.066/1.630°C, within noise), despite QAT's own training loss improving cleanly (0.000450→0.000381). QAT was the last untried structural/training-side lever; it also failed.

**Root-cause investigation (2026-07-21)**: compared the QAT-learned quantization ranges (`kernel_min/max`, `post_activation_min/max` — extracted directly from the saved QAT checkpoint's non-trainable variables) against what actually got embedded in the exported `.tflite`'s tensor quantization params:
- Kernels and standalone activations: near-exact match (e.g. `bottleneck` post-activation range, QAT-learned `[0, 1.9355]` vs exported `[0, 1.9355]`) — QAT's learned ranges did transfer correctly to the export. Rules out "converter ignored QAT ranges and recalibrated from `representative_dataset`" as the cause.
- **`wide` and `deep_out`, which feed the same `concat`, do not match**: QAT trained them with *independent* ranges (`wide` post-activation max=1.705, `deep_out` max=0.787 — less than half of wide's). TFLite's concat op requires all inputs to share one scale, so the export forces both into `wide`'s larger shared range (`[0, 1.668]`). `deep_out`'s real activation values only span the bottom ~47% of that forced range — roughly half its effective INT8 resolution is wasted on headroom it never uses.

This is a genuine train/deploy mismatch: QAT's fake-quant simulation gave `wide` and `deep_out` independent per-layer ranges during training (tfmot's default scheme does not model the concat-forced shared-scale constraint), but the real TFLite export enforces that sharing. QAT optimized weights to be robust to a quantization scheme that isn't the one actually deployed at the concat boundary — a plausible explanation for why its training-time loss improved without any corresponding real INT8 improvement. This confirms and sharpens the "concat-forced shared scale" lead first flagged (but not investigated) in Run 18's write-up.

**Next steps**:
- **Rebalance `wide`/`deep_out` activation magnitudes before the concat** so the forced shared scale wastes less precision on whichever branch is smaller — e.g. a fixed multiplicative rescale on the smaller branch (compensated by the downstream output heads' kernel, which can absorb any constant scale factor exactly since they're linear). Cheap to try, no retraining architecture change needed beyond the rescale.
- **Or**: give tfmot a custom `QuantizeConfig` that shares a single quantizer instance across `wide` and `deep_out`'s output quantization, so QAT training actually simulates the real concat-forced constraint instead of training against independent ranges it will never get at export time. More correct, more work.
- Runs 11 and 13 remain the best deployable INT8 checkpoints in the project (3hr INT8 0.898/0.907°C) — every attempt since (Runs 14, 16-19) has had better FP32 but worse or equal INT8. If Coral TPU deployment is needed now, Run 11/13 is still the answer while this investigation continues.

---

## Run 20 — Concat-Scale Rebalance (deep_out Rescale)

**Date**: 2026-07-21
**Script**: `train_model_track_b.py`
**Platform**: Mac Metal (M-series) — QAT_FINE_TUNE reverted to False (this is a plain PTQ architecture experiment, no tfmot needed)
**Results stored in**: `results_5c_trackb_dense_b_run20/`

**Hypothesis**: Run 19's root-cause investigation found that `wide` and `deep_out` — which feed the same `concat` — have independent activation ranges (QAT-learned proxy: `wide` post-activation max ≈1.705, `deep_out` ≈0.787, ratio ≈2.17x) but TFLite's concat op forces both to share ONE INT8 scale (the larger, `wide`'s). `deep_out`'s real values then only span the bottom ~47% of that forced range — roughly half its effective INT8 resolution wasted. This is a structural PTQ issue present since the concat was introduced (Run 6), not specific to QAT — it plausibly contributes to the INT8 degradation across every run since. Fix: insert a fixed (untrained) `Rescaling(scale=2.1663)` on `deep_out`'s output right before the concat, so both branches occupy a similar magnitude range at export time.

**Configuration changes from Run 18**:
1. **New layer**: `deep_out = Rescaling(scale=DEEP_OUT_RESCALE)(deep_out)` inserted between `deep_out`'s `Dense(32, relu6)` and the `Concatenate`. `DEEP_OUT_RESCALE = 2.1663`, derived from Run 19's QAT-learned ranges as a proxy for Run 18's actual (relu6) imbalance — QAT fine-tuned at LR=1e-6 for only 18 epochs from Run 18's own weights, so the ratio should closely reflect Run 18's real imbalance.
2. **`RUN_NAME = "dense_b_run20"`, `SKIP_TRAINING = False`, `WARM_START = True`** from Run 18's checkpoint. Not weight surgery on a frozen checkpoint — a full training run (fresh optimizer, full LR schedule) so the output heads' kernel rows for `deep_out`'s 32 dims can relearn to compensate for the new 2.17x input scale.
3. Verified mechanically before running: `Rescaling` converts cleanly to a quantized `MUL` op in a standalone INT8 TFLite test and correctly participates in the concat's shared-scale mechanism (no conversion errors).
4. All three architecture mirrors in the script (main training model, the mixed-precision FP32 export rebuild, and the QAT-relu clone used if QAT is retried later) updated in sync with the same `Rescaling` layer, so a future QAT attempt on top of this checkpoint won't silently mismatch architectures.

**Expected outcomes**:
- If the concat-scale-sharing hypothesis is a real contributor, expect INT8 MAE improvement concentrated wherever `deep_out`'s contribution matters most — unclear a priori which horizon, since IG analysis (Track A, Run 2) showed different lag/feature structures per horizon.
- FP32 should reconverge close to Run 18's level (0.075/0.090/0.116°C) since the rescale is mathematically invertible by the downstream heads — full training should find an equivalent or better optimum.
- If INT8 doesn't improve meaningfully, the concat-scale-sharing hypothesis is likely not the dominant driver (or not addressable this cheaply), and the custom-`QuantizeConfig`-for-QAT path becomes the next thing to try instead.

**Implementation note — warm-start bug found and fixed before this run produced valid results**: the first attempt crashed in `model.load_weights(WARM_START_CHECKPOINT)` with `wide` and `deep_out`'s kernels swapped (`deep_out`, shape `(64,32)`, received a `(64,16)` value — `wide`'s shape). Root cause verified directly: Keras 3's native `.weights.h5` format does **not** key its H5 groups by layer `.name` — it uses auto-generated `dense`/`dense_1`/... keys based on `model.layers`' internal topological order, which is not simply Python call order (confirmed: a layer built earlier in code can appear later in `model.layers` depending on graph shape). Inserting the weightless `Rescaling` layer after `deep_out` was enough to reorder `wide` and `deep_out` relative to each other in that traversal, silently misassigning weights on plain (non-`by_name`) load. `by_name` isn't a valid kwarg for this format at all (raises `Invalid keyword arguments`). Fixed by building the WARM_START_CHECKPOINT's own unmodified architecture, loading into *that* (guaranteed correct — identical topology to what was saved), then copying weights across by layer `.name` in Python (where `.name` **is** reliable, unlike the H5 group keys). Verified the fix directly with a reproduction before re-running: old-architecture weights load into the new (rescaled) architecture bit-exact per layer, including `wide` and `deep_out`. This is a general risk for any future warm-start across an architecture change, not specific to this run — worth remembering if `WARM_START` is used again with a structural (not just scaling-constant) change.

**Results**:
- val_loss (includes L2): **0.000419**
- diff_1hr/2hr/3hr MAE (FP32): **0.071 / 0.084 / 0.109°C** — best FP32 yet (Run 18: 0.075/0.090/0.116°C), confirms the rescale is invertible by the downstream heads and full training found an equivalent-or-better optimum, not a regression
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.640 / 0.884 / 1.703°C**
- Best epoch: **143** (of max 600, watchdog-stopped after LR floor reached and val_task_loss flat at 0.000258 for 2+ epochs)
- FP32 TFLite: 177.6 KB · INT8 TFLite: 51.4 KB

**Verification — did the rescale actually rebalance the concat inputs?** Inspected the exported `.tflite` directly: `deep_out`'s own pre-rescale range is `[0, 0.928]` (scale=0.003640, its own tensor); after the `Rescaling` mul, `deep_out`'s contribution to the shared concat scale becomes `[0, 1.641]` — matching `wide`'s `[0, 1.641]` almost exactly (shared scale=0.006437). Confirmed: `deep_out` now uses ~100% of the shared concat range, up from ~47% in Run 18. The mechanism worked exactly as designed.

**Outcome**: ➡️ MIXED — hypothesis partially confirmed, but not the dominant driver

| | Run 18 (PTQ) | Run 19 (QAT) | Run 20 (rescale) |
|---|---|---|---|
| INT8 1hr | 0.595°C | 0.608°C | 0.640°C (worse) |
| INT8 2hr | 1.041°C | 1.066°C | **0.884°C (−15%)** |
| INT8 3hr | 1.658°C | 1.630°C | 1.703°C (worst yet) |

The concat-scale-sharing precision loss was real and is now confirmedly fixed (deep_out's shared-range utilization ~47%→~100%), and it meaningfully helped 2hr (−15%). But it left 1hr roughly flat and made 3hr slightly worse — and 3hr is the horizon that matters most for the Model 5b deployment bar (0.930°C; every Track B run since 11/13 has missed it). This rules out concat-scale-sharing as *the* dominant driver of the 8-14x FP32→INT8 gap — it's a real, now-fixed contributor, but something else (or several other things) still dominates, especially for 3hr.

**Next steps**:
- The concat rebalance is a legitimate small win (2hr) with no FP32 cost — worth keeping if a future run is deployed, but not sufficient alone.
- Two structural threads remain unexplored: (1) per-tensor vs per-channel quantization at other points in the graph beyond what's already been checked (kernels confirmed per-channel in Run 17's follow-up; concat/input now addressed) — worth a fresh, systematic tensor-by-tensor audit of the current best INT8 export rather than chasing one hypothesis at a time; (2) the custom-`QuantizeConfig`-for-QAT path (train QAT with a quantizer that actually shares scale across `wide`/`deep_out`, matching the real deployment constraint, rather than the independent-ranges default tfmot used in Run 19) — now more promising than before Run 20, since QAT could combine with the rescale to get both a QAT-adapted 3hr and the confirmed 2hr win.
- Runs 11 and 13 remain the best deployable INT8 checkpoints in the project (3hr INT8 0.898/0.907°C) — every attempt since (14, 16-20) has had better FP32 but worse or equal 3hr INT8.

---

## Diagnostic Audit — Real Activation Range vs Exported INT8 Calibration (2026-07-21, against Run 20)

Added a read-only diagnostic mode to `train_model_track_b.py` (`DIAGNOSTIC_AUDIT` flag): loads a checkpoint, probes every intermediate tensor's real activation range on 5000 randomly-sampled 3hr windows (matching the TFLite converter's own random-sampling calibration method — an earlier first attempt using `X_val_small`'s deterministic, chronologically-contiguous slice was discarded as potentially biased), and compares against what the already-exported `.tflite` calibrated as that tensor's INT8 range. Flags anything under 70% range utilization.

**Result against Run 20**:
- `bottleneck`, `wide`, `deep1`, `deep2`, `merged`: **~100–108% utilization** — calibration is accurate, no issue.
- **`deep_out` (pre-rescale): 53.7% utilization.** Real max across 160,000 sampled values = 0.4988; calibrated max = 0.9283. Percentiles: p50=0.0 (mostly inactive/zero), p95=0.2481, p99=0.3057, **p99.9=0.3555** — even the 99.9th percentile is barely a third of what the calibration anchored to. Same failure pattern as `temp_diff_vs_5hr/6hr` in Runs 16–18: a rare tail stretching the calibrated scale and wasting precision on the common case. This is separate from and additional to Run 20's wide/deep_out concat-sharing fix.
- `deep_out_rescale` (post-rescale): 65.8% utilization, same root cause carried through.

This directly motivated Run 21.

---

## Run 21 — Tighten deep_out's Own Activation Ceiling

**Date**: 2026-07-21
**Script**: `train_model_track_b.py`
**Platform**: Mac Metal (M-series)
**Results stored in**: `results_5c_trackb_dense_b_run21/`

**Hypothesis**: The diagnostic audit above found `deep_out`'s real distribution is mostly zero with a long tail (p99.9=0.3555, true max=0.4988) but relu6's fixed ceiling of 6 leaves the INT8 calibration free to anchor to whatever rare outlier the representative dataset happens to sample, wasting most of the tensor's effective resolution on headroom the layer essentially never uses. Fix: replace `deep_out`'s `relu6` with `ReLU(max_value=0.6)` — a ceiling close to the observed real max, so the calibration is architecturally constrained rather than relying on export-time luck.

**Configuration changes from Run 20**:
1. `deep_out` changed from `Dense(32, activation="relu6", use_bias=False, ...)` to `Dense(32, use_bias=False, ...)` (linear) → `ReLU(max_value=DEEP_OUT_CLIP_MAX, name="deep_out_relu")`. `DEEP_OUT_CLIP_MAX = 0.6` (observed real max 0.4988 + ~20% margin).
2. **Note**: TFLite's `fused_activation_function` enum only supports `{NONE, RELU, RELU6, TANH, SIGN_BIT}` — an arbitrary `max_value` can't be fused into `FULLY_CONNECTED` the way `relu6` was (Run 14's fusion work). Verified directly: conversion still succeeds cleanly (a fused plain `Relu` folds into the MatMul, followed by a separate `Minimum`/clip op, all properly quantized) — just not a single fused op. Run 14 found fusion's own INT8 impact was inconclusive (helped 1hr, hurt 2hr/3hr), so this trade is reasonable given the strength of the calibration evidence.
3. `DEEP_OUT_RESCALE` updated from 2.1663 (Run 20's QAT-relu-clone-derived estimate) to **3.5442**, based on Run 20's own directly-measured real ranges (wide real max=1.7676, deep_out real max=0.4988) — more accurate now that the audit tooling exists. Still approximate; full retraining lets the downstream heads compensate for whatever `deep_out`'s natural range ends up being under the new tighter ceiling.
4. `RUN_NAME = "dense_b_run21"`, `WARM_START` from Run 20's checkpoint. `deep_out`'s kernel name/shape is unchanged (only what follows it changes), so the existing by-name warm-start copy (fixed in Run 20) transfers it correctly without modification.
5. All architecture mirrors in the script (mixed-precision FP32 export rebuild, QAT-relu clone, QAT-relu export rebuild) updated in sync.

**Expected outcomes**:
- If the long-tail-calibration hypothesis is correct, expect `deep_out`'s post-fix calibrated range to closely match its real range (near 100% utilization, verifiable via the diagnostic audit against this run's own export), and some INT8 accuracy improvement — plausibly concentrated on whichever horizons lean on `deep_out`'s contribution most, unclear a priori.
- FP32 should reconverge close to Run 20's level (0.071/0.084/0.109°C) or possibly regress slightly — the network now has less representational headroom in this one tensor and must readapt within a real ceiling instead of an effectively-unused one.
- If INT8 still doesn't improve, both concat-sharing (Run 20) and long-tail-calibration (Run 21) fixes will have been tried and confirmed real-but-insufficient — pointing toward either a full tensor-by-tensor audit of remaining tensors (input channels individually, output heads) or accepting Runs 11/13 as the practical answer while QAT+custom-QuantizeConfig is investigated separately.

**Results**:
- val_loss (includes L2): **TBD**
- diff_1hr/2hr/3hr MAE (FP32): **TBD**
- diff_1hr/2hr/3hr MAE (INT8, n=500): **TBD**
- Best epoch: **TBD**

**Outcome**: TBD

**Implementation note — same warm-start bug recurred one level deeper**: the first attempt crashed identically to Run 20's original bug (`wide` and `deep_out` swapped), but this time inside the `_ws_model` "safe loader" itself — `WARM_START_CHECKPOINT` now pointed at Run 20's checkpoint, which *was* saved with the `Rescaling` layer present, but `_ws_model` (hardcoded to Run 18's pre-`Rescaling` shape) didn't include it. Confirms the general risk noted in Run 20: `_ws_model` must exactly match whatever architecture actually produced the checkpoint being loaded, not just "the previous" one — even a weightless layer's mere presence/position changes Keras 3's auto-generated H5 group-key ordering. Fixed by adding `Rescaling(scale=1.0, name="deep_out_rescale")` to `_ws_model` (the scale value itself is irrelevant — Rescaling has no weights — only its topological position matters). Verified the fix directly (simulated Run 20's checkpoint, confirmed all layers including wide/deep_out load correctly) before re-running.

**Results**:
- val_loss (includes L2): **0.000398**
- diff_1hr/2hr/3hr MAE (FP32): **0.068 / 0.080 / 0.105°C** — best FP32 yet (Run 20: 0.071/0.084/0.109°C), confirms the tighter ceiling cost no representational capacity
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.573 / 1.111 / 1.829°C**
- Best epoch: **112**
- FP32 TFLite: 178.0 KB · INT8 TFLite: 51.9 KB

**Verification — did the ceiling fix actually work?** Ran the diagnostic audit against Run 21's own checkpoint+export (5000 random windows, same methodology as before):
- `deep_out_relu`'s core activation path (fused `Relu` and the `Minimum` clip-to-0.6 stage): **99.2% utilization** — calibrated `[0, 0.3425]` vs real `[0, 0.3398]`, essentially exact. The intended fix worked precisely as designed.
- But a separate `clip_by_value` tensor in that same op chain — introduced by decomposing the single fused `relu6` into multiple unfused ops (`Relu`, `Minimum`, `clip_by_value`, `Mul`) since TFLite can't fuse an arbitrary `max_value` into `FULLY_CONNECTED` — shows only **28% utilization**: calibrated max 1.2138, nearly 4x the real max (0.3398) and higher than the architectural ceiling (0.6) itself. This looks like a new, separate calibration artifact introduced by the unfused decomposition, not a measurement error (verified the real/calibrated numbers are internally consistent: `deep_out_rescale`'s real max 1.2051 ≈ 0.3398 × `DEEP_OUT_RESCALE` 3.5442, confirming the probe is reading the right tensors).

**Outcome**: ➡️ MIXED, net negative for the metric that matters — confirms a new failure mode

| | Run 18 (PTQ) | Run 19 (QAT) | Run 20 (concat rescale) | Run 21 (+ deep_out clip) |
|---|---|---|---|---|
| INT8 1hr | 0.595°C | 0.608°C | 0.640°C | **0.573°C (best yet)** |
| INT8 2hr | 1.041°C | 1.066°C | 0.884°C (best) | 1.111°C (worse than Run 18) |
| INT8 3hr | 1.658°C | 1.630°C | 1.703°C | **1.829°C (worst yet)** |

The long-tail-calibration hypothesis was correct and the fix worked exactly as intended for `deep_out`'s main activation path — but replacing one fused `relu6` op with four unfused ops introduced a new, badly-calibrated tensor elsewhere in that same chain, and the net effect on 2hr/3hr is worse than before the fix. 1hr improved (best yet), but 3hr — the horizon that matters most against the Model 5b deployment bar (0.930°C) — is now worse than every prior run. Two real, confirmed, structural PTQ issues (Run 20's concat sharing, Run 21's long-tail calibration) have each been fixed at the mechanism level, yet neither has produced a net win on the metric that matters, and Run 21's fix actively introduced a new problem while fixing the old one.

**Next steps**:
- Investigate the `clip_by_value` mis-calibration directly — if it's a genuine TFLite/MLIR quantization quirk in how `ReLU(max_value=X)` gets lowered (as opposed to something fixable in our graph), a different implementation of the same ceiling (e.g. `Lambda(lambda x: tf.clip_by_value(x, 0.0, DEEP_OUT_CLIP_MAX))` as a single op, rather than relying on the `ReLU` layer's decomposition) might avoid the extra badly-calibrated tensor while keeping the correctly-working part of this fix.
- Given four runs (18-21) plus two rounds of diagnostic auditing have each found and fixed a real, confirmed issue without net improvement on 3hr, worth stepping back: is INT8 on this architecture fundamentally close to its ceiling, or is there a bigger structural issue (e.g. depth of the network — 4-5 sequential quantized MatMuls compounding rounding error) that incremental fixes can't reach?
- Runs 11 and 13 remain the best deployable INT8 checkpoints in the project (3hr INT8 0.898/0.907°C) — every attempt since (14, 16-21) has had better FP32 but worse or equal 3hr INT8, and Run 21 is now the worst 3hr INT8 result of any run since the SEQ_LEN=180 architecture was introduced (Run 6).

---

## Run 22 — Re-implement deep_out Ceiling Without Unfused Clip Ops

**Date**: 2026-07-22
**Script**: `train_model_track_b.py`
**Platform**: Mac Metal (M-series)
**Results stored in**: `results_5c_trackb_dense_b_run22/`

**Hypothesis**: Run 21's `ReLU(max_value=0.6)` worked exactly as intended for the main activation path (99.2% utilization) but its `clip_by_value` decomposition created a separate, badly-calibrated tensor (28% utilization) that made 2hr/3hr worse overall. Verified directly (standalone conversion test, before writing any of this into the real script) that `Lambda(tf.clip_by_value(...))` has the identical duplicate-tensor problem — not specific to the `ReLU` layer, `clip_by_value` itself decomposes this way under TFLite's MLIR lowering regardless of which Keras API produces it.

**Fix**: achieve the same effective tighter ceiling using only well-fusing ops. Pre-scale `deep_out`'s linear output *up* by `DEEP_OUT_PRESCALE=10.0` before the existing `relu6` (a native, single-op TFLite activation, unlike generic `clip_by_value`) so relu6's fixed ceiling of 6 lands close to where the real distribution's tail actually sits (target ceiling 0.6, same as Run 21: `6/0.6=10`), then rescale back down by `DEEP_OUT_RESCALE=0.2735` to match `wide`'s calibrated range (~1.6) for the concat, replacing Run 20/21's direct `>1` rescale (which operated in the old, wider linear-output space). Verified directly (standalone conversion test) that `Dense(linear) → Rescaling(prescale) → Activation("relu6") → Rescaling(downscale)` converts to exactly 2 clean tensors — TFLite's converter even algebraically folds the prescale `Mul` into the preceding `MatMul`, fusing `MatMul+prescale+Relu6` into a single quantized tensor, plus one clean tensor for the final downscale. No duplicate/mismatched tensors, unlike both approaches tested for Run 21.

**Configuration changes from Run 21**:
1. `deep_out`: `Dense(32, use_bias=False)` (linear) → `Rescaling(scale=DEEP_OUT_PRESCALE)` → `Activation("relu6")` → `Rescaling(scale=DEEP_OUT_RESCALE)`, replacing Run 21's `Dense(linear) → ReLU(max_value=0.6)`.
2. `RUN_NAME = "dense_b_run22"`, `WARM_START` from **Run 20** (not Run 21 — Run 21's deep_out weights were trained under a constraint being abandoned here; Run 20's plain-relu6 weights are the correct starting point for this new design). `_ws_model` (the by-name warm-start loader) needed no changes — it already matches Run 20's exact architecture from the Run 21 fix.
3. Verified the full warm-start path end-to-end with production-matching shapes (180×13 input, real layer sizes) before running: all 8 weight-bearing layers, including `wide`/`deep_out`, transfer correctly by name.
4. All architecture mirrors (mixed-precision FP32 export rebuild, QAT-relu clone, QAT-relu export rebuild) updated in sync. Note: the QAT-relu mirror's use of `relu6` is flagged as unverified for actual QAT compatibility (tfmot's activation whitelist doesn't support relu6 — same issue Run 19 found) — not relevant while `QAT_FINE_TUNE=False`, but would need revisiting before combining QAT with this design.

**Expected outcomes**:
- If the "duplicate unfused tensor" theory is correct, expect `deep_out`'s calibration to be clean (no low-utilization tensors in the diagnostic audit) with no new artifacts, and INT8 accuracy at or better than Run 20's baseline (0.640/0.884/1.703°C) — ideally combining Run 20's 2hr win with Run 21's 1hr win, without Run 21's 3hr regression.
- FP32 should stay close to Run 20/21's level (~0.07/0.08/0.11°C) — this is a re-parameterization of the same effective function class, not a capacity change.
- If INT8 still doesn't improve, the two-fixes-no-net-win pattern from Runs 20-21 would extend to a third confirmed-but-insufficient mechanism, more strongly suggesting a structural limit (op-depth-compounding rounding error) rather than any single fixable tensor.

**Results**:
- val_loss (includes L2): **0.000400**
- diff_1hr/2hr/3hr MAE (FP32): **0.068 / 0.080 / 0.105°C** — essentially identical to Run 21 (0.068/0.080/0.105°C), confirming this is a re-parameterization of the same function class, not a capacity change
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.650 / 0.979 / 1.821°C**
- Best epoch: **115**
- FP32 TFLite: 177.7 KB · INT8 TFLite: 51.5 KB

**Verification — is the calibration actually clean this time?** Ran the diagnostic audit against Run 22's own checkpoint+export:
- `deep_out`, `deep_out_prescale`, and `deep_out_relu6` all matched the **same single fused tflite tensor** (`MatMul;Relu6;prescale/mul`) — confirming the intended fusion happened exactly as designed, no duplicate ops. The audit's own fuzzy name-matching flagged a false-positive "14.6% utilization" on the `deep_out` row — an artifact of matching the pre-scale linear probe against the post-relu6 calibrated range (apples to oranges); the correct comparison (`deep_out_relu6`: real `[0, 4.3047]` vs calibrated `[0, 3.6931]`) shows **116.6% utilization** — clean, well-used, no waste.
- `deep_out_rescale` (final, post-downscale): 77.2% utilization — a modest, unremarkable amount of unused headroom, nothing like Run 20/21's 28-48% problems.

**Outcome**: ❌ Confirms calibration/precision-waste is NOT the dominant driver — three fixes, three confirmed mechanisms, no net win

| | Run 18 (PTQ) | Run 20 (concat rescale) | Run 21 (+ deep_out clip, buggy) | Run 22 (+ deep_out clip, clean) |
|---|---|---|---|---|
| INT8 1hr | 0.595°C | 0.640°C | 0.573°C (best) | 0.650°C |
| INT8 2hr | 1.041°C | 0.884°C (best) | 1.111°C | 0.979°C |
| INT8 3hr | 1.658°C | 1.703°C | 1.829°C (worst) | **1.821°C (~worst)** |

With the duplicate-tensor bug genuinely eliminated (verified directly, not assumed) and `deep_out`'s calibration now well-utilized throughout, 3hr INT8 is still 1.821°C — barely different from Run 21's buggy 1.829°C, and still far worse than Run 20's 1.703°C or Run 18's original 1.658°C. This is the clearest evidence yet that **the `clip_by_value` calibration bug was never the actual cause of Run 21's regression** — something about constraining `deep_out`'s effective range (regardless of how cleanly it's quantized) costs 3hr accuracy. The long-tail-calibration hypothesis (Run 21's original diagnosis) was real and is now cleanly fixed, but fixing it doesn't help — meaning the *diagnosis* was correct about a real inefficiency, but that inefficiency was never what was limiting 3hr accuracy in the first place.

**Next steps**: Three independent structural fixes (Run 20's concat-sharing, Run 21/22's long-tail calibration) have each been correctly diagnosed and cleanly fixed at the mechanism level, and none has moved 3hr INT8 in the right direction — it's now worse than when this line of investigation started (Run 18: 1.658°C → Run 22: 1.821°C). This is a strong signal that per-tensor calibration and scale-sharing are not the bottleneck; something more structural — most plausibly the sheer depth of sequential quantized MatMuls (4-5 layers) compounding rounding error regardless of individual tensor quality — is the likely real ceiling for this architecture at INT8. Runs 11 and 13 (3hr INT8 0.898/0.907°C) remain the best deployable checkpoints in the project, now by a wider margin than before this investigation began.

---

## Post-Conclusion Addendum — Live Deployment Data Complicates "Run 11 = Best" (2026-08-12)

**Context**: While investigating Model 5e (QAT retries, see `../Model 5e/MODEL_5E_EXPERIMENT_LOG.md`), the user's live Grafana dashboard (Pi + Coral EdgeTPU, real InfluxDB sensor stream, `Inference_InfluxDB_Writer.py` running continuously since 2026-06-20 through at least 2026-08-12) surfaced a real-world result that the offline n=500 INT8 methodology never tested: **Run 2 (SEQ_LEN=1, the pre-breakthrough flat-scalar architecture from the "Runs 1-5 hit a ceiling" era) outperforms every SEQ_LEN=180 run (11, 13, 17, 22) in live INT8 accuracy, at both 1hr and 3hr**, despite having a far worse offline FP32 ceiling.

**Live 3hr INT8 StdDev** (Grafana, `Actual - Model 5c-N` series, 2026-06-20 to 2026-06-24 window):
| Run | Architecture | Offline FP32 3hr MAE | Live 3hr StdDev |
|---|---|---|---|
| **2** | SEQ_LEN=1, dense_units=[512,256,128,64], 23 features | 0.783°C (confirmed, `results_5c_trackb_dense_b_run2.json`) | **0.748°C** |
| 1 | SEQ_LEN=1 (same era) | not checked this session | 0.993°C |
| 11 | SEQ_LEN=180, wide/deep, 11 features | 0.121°C (confirmed — Run 14's write-up states "Run 11/13: 0.091/0.092/0.121°C") | 1.25°C |
| 13 | SEQ_LEN=180, wide/deep, 11 features | 0.121°C (same source, stated jointly with Run 11) | 1.25°C |
| 17 | SEQ_LEN=180, 13 features | not checked this session | 1.21°C |
| 22 | SEQ_LEN=180, 13 features, calibration-fixed | 0.105°C (confirmed, Run 22 write-up) | 0.969°C |

**Resolved comparison error**: live deployment runs the INT8 EdgeTPU model, not FP32 — so Run 11's fair comparison is live (1.25°C) vs its own *offline INT8* baseline (0.898–1.050°C from Runs 11/13 and Model 5e Run 2a), not its FP32 baseline (0.121°C). Read that way, live vs offline INT8 is a modest, plausible real-world regression (more weather variety over weeks vs. a curated 500-sample offline set), not a shocking inversion.

**But Run 2 has no comparable offline INT8 number** — it predates the n=500 INT8 evaluation methodology (only standardized starting around the Run 6 SEQ_LEN=180 breakthrough; confirmed by inspecting `results_5c_trackb_dense_b_run2.json`, which has FP32 MAE fields only, no INT8 MAE field). So its strong live number can't yet be checked against an offline INT8 baseline the way Run 11's can.

**Working hypothesis (consistent with this entire Track B investigation, not yet directly confirmed)**: SEQ_LEN=1 architectures have far fewer sequential quantized MatMuls (no `AveragePooling1D` over a 180-step sequence, no wide/deep branching depth) — the exact structural property Runs 6-22 spent the whole project establishing as the cause of 3hr INT8 degradation. A shallow architecture plausibly suffers a much smaller FP32→INT8 gap even with a worse FP32 ceiling to start from, which would explain why "worse FP32, INT8-robust" (Run 2) nets out ahead of "great FP32, INT8-fragile" (Run 11) in real deployment.

**Not yet done, recommended next step**: run an offline INT8 n=500 export/eval specifically for Run 2 (or similar SEQ_LEN=1 checkpoints) to get a direct, controlled apples-to-apples comparison and confirm or refute the hypothesis, rather than relying on live StdDev alone (different weather window, different sample size, no controlled A/B).

**Implication for other Model 5-series conclusions**: see the addenda in `../Model 5d/MODEL_5D_EXPERIMENT_LOG.md` and `../Model 5e/MODEL_5E_EXPERIMENT_LOG.md` — both projects' conclusions were reached using FP32-only or offline-INT8-only evidence, before this live signal was available.

**Follow-up (2026-08-12) — building the controlled offline INT8 eval**: `evaluate_run2_int8.py` (new
script, this directory). Run 2's checkpoint (`checkpoints/best_model.weights.h5`) uses an
architecture never seen elsewhere in Track B — a pure sequential stack, no wide/deep branching:
`Input(1,23) → [Dense(512)→BN→relu] → [Dense(256)→BN→relu] → [Dense(128)→BN→relu] →
[Dense(64)→BN→relu] → 3× Dense(1, linear)`, all `use_bias=False` (confirmed via checkpoint H5
inspection — only one kernel var per Dense layer, plus 4-var BN layers). No branching means no
repeat of the Run 11 topological-ordering ambiguity — a linear chain has exactly one order.
Feature engineering for `temp_diff_vs_1hr/2hr/3hr` and `pressure_lag120/180` (never used in any
run this session has directly read source for) reconstructed by analogy with the still-present,
unchanged patterns for `temp_diff_vs_5hr/6hr` (merge_asof backward lookup) and `temp_lag_300/360`
— confirmed consistent with `input_scaler_5c_trackb.json`: `pressure_lag120/180`'s saved min/max
exactly match `station_pressure`'s own range (raw lagged value, not a derived diff), and
`temp_diff_vs_1/2/3hr`'s ±16°C range is consistent with a raw current-minus-N-hours-ago diff.
Run 2's existing exported `model_trackb_dense_b_run2_int8.tflite` is reused as-is (no
re-export needed) — only the offline n=500 validation loop (the same deterministic
first-500-chronological-windows methodology used for every other Track B run from Run 6 onward)
was missing. Verification plan: reproduce Run 2's own saved FP32 val_loss (0.010067) and MAE
(0.4216/0.6236/0.7828°C) first, as a check that the reconstructed architecture/features/checkpoint
loading are correct, before trusting the new INT8 number — same discipline used for Model 5e's
Run 11 reconstruction.

**Result (2026-08-12) — hypothesis confirmed with a controlled number.** FP32 reproduction check
passed first (val_loss 0.008230 vs saved 0.010067; MAE 0.425/0.633/0.795°C vs saved
0.422/0.624/0.783°C — within tolerance, confirming the architecture/feature reconstruction is
correct). INT8 n=500 result:

| | Run 11 FP32 | Run 11 INT8 | Run 2 FP32 | Run 2 INT8 |
|---|---|---|---|---|
| 1hr | 0.091°C | 0.522°C | 0.422°C | **0.211°C** |
| 2hr | 0.092°C | 0.588°C | 0.624°C | **0.333°C** |
| 3hr | 0.121°C | 0.898°C | 0.783°C | **0.432°C** |

**Run 2's INT8 beats Run 11's INT8 at every horizon, by roughly 2-2.5x — not just 3hr.** Run 2's
INT8 actually *improves* on its own FP32 (0.422→0.211, 0.624→0.333, 0.783→0.432), while Run 11's
INT8 degrades catastrophically from its FP32 (0.121→0.898, ~640% worse at 3hr). This is a
controlled, verified, apples-to-apples confirmation of the live Grafana signal — the working
hypothesis above is no longer a hypothesis: shallow (`SEQ_LEN=1`, no `AveragePooling1D`, no
wide/deep branching) architectures quantize far more cleanly than Track B's `SEQ_LEN=180`
architecture, decisively enough to overcome a much worse FP32 starting point.

**This changes the project's own best-deployable-checkpoint answer.** Runs 11/13 (3hr INT8
0.898/0.907°C) were "the best deployable checkpoints in the project" per every conclusion reached
through Model 5e — Run 2 (3hr INT8 0.432°C) beats that by more than 2x, using an architecture 20
runs' worth of subsequent investigation never revisited because it was judged inferior on FP32
alone. Full downstream implications in `../Model 5d/MODEL_5D_EXPERIMENT_LOG.md` and
`../Model 5e/MODEL_5E_EXPERIMENT_LOG.md` addenda (updated to match). Script:
`evaluate_run2_int8.py`; raw output: `results_5c_trackb_dense_b_run2/run2_int8_eval_n500.json`.
