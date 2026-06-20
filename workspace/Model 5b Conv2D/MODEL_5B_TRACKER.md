# Model 5b Experiment Tracker — ⛔ CONCLUDED 2026-06-18

## Goal

The **original goal** of Model 5b was to learn engineered temporal features automatically from a raw 180-minute sensor window, without hand-pre-computing lag features like `temp_lag60/120/180`. Model 5a (val_loss=0.000682, 788 KB) already runs on Coral TPU — Edge TPU compatibility was a constraint, not the motivation. The bet was that a convolutional architecture could discover which lags and cross-sensor interactions matter, removing the need for human feature engineering.

That original goal was retired at Exp 26 after 25 failed experiments. Exp 26 onwards added explicit lag features, shifting the goal to: match or beat **Model 5a clean** accuracy (val_loss=0.000373) using a Conv2D architecture, while remaining Edge TPU-compilable.

| Model 5a clean variant | val_loss | MAE | Edge TPU |
|----------------------|----------|-----|---------|
| dense_wide_run1 (primary accuracy target) | **0.000373** | ~0.100°C | ❌ (SRAM overflow — wider model only) |
| avgpool_run1 | 0.000508 | ~0.120°C | ✅ |
| **Deployed Model 5a pi** | **0.000682** | **~0.145°C** | **✅ (788 KB, production)** |

## Target
**Phase 1 (current):** float val_loss ≤ 0.000373 (Model 5a clean `dense_wide_run1`)  
**Phase 2 (after Phase 1):** QAT to close the float→quantized accuracy gap (float ~0.01°C vs quantized ~0.67–1.71°C)

## Architecture (Exp 39/41 — two-stream with dilated Conv2D, diurnal routing)
```
Input: (180, 28)
  ├─ Stream A — Conv2D path (14 physical sensors ONLY, no diurnal/lag/slope):
  │   input[:, :, :14]  ← temperature, humidity, pressure, solar, illuminance, uv,
  │                         wind_avg, wind_gust, wind_lull, wind_dir_sin/cos, rain,
  │                         day_of_year_sin/cos
  │   Reshape to (180, 14, 1)
  │   → Conv2D(96, k=(3,1))→BN→ReLU6 → Conv2D(96, k=(7,1))→BN→ReLU6
  │   → Conv2D(96, k=(15,1))→BN→ReLU6 → Conv2D(96, k=(1,14))→BN→ReLU6
  │   → GlobalAveragePooling2D → (96,)
  │   → Dense(64, L2=1e-4) → ReLU6 → conv_context(64)
  │   Role: learn cross-sensor temporal dynamics; no diurnal shortcut available
  └─ Stream B — Anchor path (current timestep, all 28 features):
      input[:, -1, :]  ← STRIDED_SLICE ✅
      → Dense(32, L2=1e-4) → ReLU6 → anchor(32)
      Role: direct access to all engineered temporal summaries incl. diurnal + lag features
Concatenate([conv_context(64), anchor(32)]) → Dense(32, L2=1e-4) → ReLU6 → Dense(3) outputs
```

Feature vector ordering: `conv_features (14)` + `diurnal_features (4: time_of_day_sin/cos/sin2/cos2)` + `engineered_features (temp_lag60/120/180, temp_delta_1, 6 slopes)`

**Temporal receptive field (Exp 39+):** k=(3,1)d=1 / k=(7,1)d=4 / k=(15,1)d=16 → combined RF = **251 timesteps ≈ 4 hours 11 minutes**. GAP pools over all 180 positions, so patterns up to ~4hr span can be detected anywhere in the window. Prior experiments (Exp 27–38) used dilation=1, giving only 23-minute RF.

**Fixed inputs (Exp 41):** 28 features total; Conv2D stream sees first 14 (physical sensors + seasonal cycle, no diurnal); anchor stream sees all 28 (adds `temp_lag60/120/180`, `temp_delta_1`, diurnal, 6 slope features)  
**Training:** ReduceLROnPlateau (factor=0.5, patience=12, min_lr=1e-7), initial LR=1e-4, MSE, batch=1024, EarlyStopping patience=30, L2=1e-4 on all layers, FILTERS=96  
**Training platform:** Kaggle T4 ×2 GPU, MirroredStrategy, float32, XLA JIT enabled, ~863ms/step, ~443s/epoch

## Key Learnings — What Doesn't Work
| Approach | Outcome | Reason |
|---|---|---|
| Flatten → Dense (Exp 1–8) | val_loss ~0.006–0.009 | Lag value buried at one position in 5,400-dim input; no temporal inductive bias |
| Conv1D + SliceTimestep (Exp 9–26) | Best float 0.001343; quantization always broken | SliceTimestep/SliceFeatures ops excluded from QAT; activation range unbounded; never Edge TPU-viable |
| No explicit lag features (Exp 9–25) | temperature always ranked dead last | Diurnal signal (time_of_day) too dominant; Conv1D cannot learn implicit temperature anchoring |
| Conv2D + GAP without skip path (Exp 27) | lag features ranked 18th–19th | GAP averages 180 timesteps, diluting single-timestep anchor values 180× |
| Dropout(0.3) on context vector (Exp 30) | val_loss regressed 0.0026→0.0032; best epoch=3 | Too aggressive for 64-unit vector; model underfits; all features compress to flat 0.056–0.101 band |
| Adding more features (Exp 29 slopes, Exp 26 lags) | No float accuracy improvement | Float ceiling was overfitting (train/val ~10×), not insufficient features |
| PTQ post-training quantization (all Conv1D) | Constant output collapse | Unbounded intermediate activations; can't be fixed with representative dataset calibration |
| Cosine LR decay (Exp 27–31) | Best epoch 15, train/val ~10× | Fixed schedule keeps decaying LR even during plateaus; ReduceLROnPlateau fires on stall instead |
| `tf.gather` for non-contiguous feature selection (Exp 29–32) | GATHER op not Edge TPU-compatible; forces CPU split | Only STRIDED_SLICE is supported; use full last-timestep slice + Dense to learn selection |
| Full 27-feature anchor at t=−1 only (Exp 33) | val_loss regressed 0.0024→0.0025; best epoch 46→19 | Richer anchor creates shortcut to early convergence; model stops learning temporal representations |
| Hardcoded results filename in best-run collector (Exp 27–33 bug) | Old exp27 model copied as "best"; GATHER persisted in deployed model | Results reader used stale `conv2d_exp27_run*.json` name instead of current run's filename |

## Key Learnings — What Works
| Approach | Result |
|---|---|
| Conv2D + GlobalAveragePooling2D (no custom ops) | First successful PTQ; all ops Edge TPU-compatible |
| Skip path for anchor features | Fixes GAP dilution; `temp_lag120`→#1, `temp_lag60`→#2 (Exp 32) |
| Explicit `temp_lag60` + `temp_lag120` features | Direct anchor values dominate feature importance when skip path is used |
| Slope features in skip path | Quantized MAE improved all 3 horizons vs Exp 28 |
| ReduceLROnPlateau (Exp 32) | Best epoch moved from 15 → 46; train/val gap narrowed from ~10× → ~3.1× |
| L2 regularization on Conv2D + Dense (Exp 32) | Closes overfitting gap without flattening feature importance (unlike Dropout) |
| Single `EXP_NAME` constant for run naming (Exp 34 fix) | Results writer and reader always in sync; stale filename bug impossible |
| `temp_lag180` as explicit anchor feature (Exp 37) | Confirmed as #1 feature importance in Exp 37 Run 2 (0.0810); Exp 36 saliency predicted this |

---

## Experiment Results Summary

| Exp | Key Change | float val_loss | Best Epoch | train/val gap | Quant MAE (1/2/3hr °C) | Edge TPU |
|-----|-----------|---------------|-----------|--------------|------------------------|---------|
| 24 | Conv1D dual branch 64:64, best Conv1D result | 0.001343 | 46 | ~2.3× | 0.61/1.16/2.63 | ❌ |
| 27 | Switch to Conv2D + GAP (no skip path) | 0.0027 | ~89 | oscillating | 1.57/2.21/2.63 | ✅ first PTQ success |
| 28 | + skip path (temp, lag60, lag120 → Dense(16)) | 0.0028 | ~91 | ~8× | 1.12/1.63/2.01 | ✅ |
| 29 | + 6 slope features in input + skip expanded to Dense(32) | 0.0026 | 15 | ~10× | 0.82/1.49/1.63 | ⚠️ diff_3hr near-constant |
| 30 | + Dropout(0.3) on context vector | 0.0032 | 3 | ~4.1× | 0.67/1.39/1.71 | ✅ |
| 31 | ⛔ **CANCELLED** — superseded by Exp 32 | — | — | — | — | — |
| **32** | **ReduceLROnPlateau + L2(1e-4) + no Dropout** | **0.0024** ⭐ | **46** | **~3.1×** | 0.67/1.39/1.71 | ✅ (2 subgraphs; GATHER) |
| 33 | GATHER fix: full 27-feat slice, no tf.gather | 0.0025 | 19 | — | 0.67/1.39/1.71 | ⚠️ tracker bug copied exp27 as best; GATHER still in deployed model |
| **34** | **Replace GAP with multi-point extraction (t=−1, t=−61, t=−121) + tracker bug fix** | **0.0104** ❌ | 21 | ~flat | 0.78/1.35/1.83 | ✅ 1 subgraph, all 18 ops |
| **35** | **Revert GAP + wider Conv2D (64→96) + patience 8→12 (Kaggle T4 ×2)** | **0.002368** ❌ | **8** | anchor-only | not quantized | ✅ 1 subgraph, 14 ops |
| 36 | Conv2D pattern analysis — **Hypothesis B confirmed**: 58–64% dead temporal filters; Conv2D attends only t=−179 for Δ3hr | N/A (analysis) | — | — | — | — |
| **37** | **Add `temp_lag180` (28th feature); test whether Conv2D adds value beyond anchor** | **0.002117** ✅ (ES fired ep.131) | 76 | flat | not quantized | ✅ TFLite 478 KB (PTQ collapse) |
| **38** | **Two-stream: Conv2D raw sensors only; anchor all 28 features** | **0.003779** ❌ | **32** | — | 1.07/2.15/2.24 | ✅ |
| **39** | **Dilated Conv2D: k=(3,1)d=1 / k=(7,1)d=4 / k=(15,1)d=16 → RF ~251 min** | **0.0163** (L2-inflated, not comparable) | **150** (session ep=2) | — | 0.52/1.06/1.38 | ✅ |
| **40** | **Diurnal out of Conv2D + lag features removed from anchor (two changes at once)** | **0.1701** ❌ | **56** | — | 3.04/1.09/7.26 | ✅ |

---

## Key Learnings — What Doesn't Work (updated)
| Approach | Outcome | Reason |
|---|---|---|
| Flatten → Dense (Exp 1–8) | val_loss ~0.006–0.009 | Lag value buried at one position in 5,400-dim input; no temporal inductive bias |
| Conv1D + SliceTimestep (Exp 9–26) | Best float 0.001343; quantization always broken | SliceTimestep/SliceFeatures ops excluded from QAT; activation range unbounded; never Edge TPU-viable |
| No explicit lag features (Exp 9–25) | temperature always ranked dead last | Diurnal signal (time_of_day) too dominant; Conv1D cannot learn implicit temperature anchoring |
| Conv2D + GAP without skip path (Exp 27) | lag features ranked 18th–19th | GAP averages 180 timesteps, diluting single-timestep anchor values 180× |
| Dropout(0.3) on context vector (Exp 30) | val_loss regressed 0.0026→0.0032; best epoch=3 | Too aggressive for 64-unit vector; model underfits; all features compress to flat 0.056–0.101 band |
| Adding more features (Exp 29 slopes, Exp 26 lags) | No float accuracy improvement | Float ceiling was overfitting (train/val ~10×), not insufficient features |
| PTQ post-training quantization (all Conv1D) | Constant output collapse | Unbounded intermediate activations; can't be fixed with representative dataset calibration |
| Cosine LR decay (Exp 27–31) | Best epoch 15, train/val ~10× | Fixed schedule keeps decaying LR even during plateaus; ReduceLROnPlateau fires on stall instead |
| `tf.gather` for non-contiguous feature selection (Exp 29–32) | GATHER op not Edge TPU-compatible; forces CPU split | Only STRIDED_SLICE is supported; use full last-timestep slice + Dense to learn selection |
| Full 27-feature anchor at t=−1 only (Exp 33) | val_loss regressed 0.0024→0.0025; best epoch 46→19 | Richer anchor creates shortcut to early convergence; model stops learning temporal representations |
| Hardcoded results filename in best-run collector (Exp 27–33 bug) | Old exp27 model copied as "best"; GATHER persisted in deployed model | Results reader used stale `conv2d_exp27_run*.json` name instead of current run's filename |
| MultiPointTemporalExtraction replacing GAP (Exp 34) | val_loss regressed 0.0024→0.0104 (~4.3×); best epoch=21; quantized MAE worse | Conv2D temporal kernels (3/7/15 steps) have ~25-step receptive field; extracting at t=−61/−121 gives local neighborhood at those positions, not 60/120-min history; GAP temporal averaging was doing real work |
| Wider Conv2D filters + longer patience with lag features in all 180 Conv2D input timesteps (Exp 35) | val_loss 0.002368 ≈ Exp 32 (0.0024); best_epoch=8; no improvement | Explicit `temp_lag60`/`temp_lag120` in every Conv2D timestep creates structural redundancy; anchor path solves at epoch 8, Conv2D receives no gradient pressure to develop further |
| Stacked uniform Conv2D kernels k=(3,1)→(7,1)→(15,1), dilation=1 (Exp 29–38) | Combined temporal receptive field only 23 timesteps = 23 minutes | GAP pools over 180 positions but each filter only sees 23-min local slice; multi-hour weather patterns (fronts, marine layer, diurnal pressure trends) are invisible |
| Dilated Conv2D (k=(3,1)d=1 / k=(7,1)d=4 / k=(15,1)d=16, RF ~251 min) (Exp 39) | 1hr improved 54%, but 3hr regressed 30% vs Exp 38; hypothesis not confirmed | Longer RF did NOT produce disproportionate 3hr improvement as predicted; diurnal features still dominate importance; Conv2D multi-hour pattern detection unconfirmed |
| Removing explicit lag features from anchor path (Exp 40) | Per-head losses 26×/14×/79× worse than Exp 39; flat feature importance (0.034–0.045, only 24 features) | `temp_lag60/120/180` are the primary temperature trajectory anchors; dilated Conv2D (RF ~251 min) cannot learn equivalent implicit representations from raw sensor sequence |

---

## Exp 35 Results: GAP Ceiling Confirmed

**Outcome:** ❌ FAILED — neither success criterion met.

| Criterion | Target | Actual |
|-----------|--------|--------|
| float val_loss | < 0.0024 | **0.002368** (essentially tied with Exp 32) |
| best epoch | > 30 | **8** |
| `temp_lag120` in top 5 | yes | #2 (0.0894) — but spread flat (0.071–0.091 across all 27) |
| Edge TPU single subgraph | yes | ✅ all 14 ops on TPU |

**Feature importance (top 3):** `temp_lag60` 0.0909 → `temp_lag120` 0.0894 → `temperature` 0.0877. Distribution is extremely flat — no clear dominant feature vs. Model 5a's sharp `temp_lag120` at 0.093.

**Root cause — structural redundancy:** `temp_lag60` and `temp_lag120` are explicit columns in every one of the 180 timesteps of the Conv2D input. The anchor path (Dense on current-state features) converges at epoch 8 using those same values. The Conv2D path receives no gradient pressure to develop independent temporal representations. Wider filters and longer patience cannot fix a structural information problem.

**Conv2D temporal limitation:** The four Conv2D kernels `(3,1)→(7,1)→(15,1)→(1,27)` give a combined temporal reach of only ~25 steps. Any pattern beyond 25 minutes is outside the Conv2D's visible window unless it appears in the explicit lag columns.

**Training platform note:** Exp 35 was migrated from Mac to Kaggle (T4 ×2, MirroredStrategy, ~18.4 min/epoch) after Mac Metal GPU bugs made multi-day training unreliable at FILTERS=96. See experiment log for detailed bug list (Bugs 1–8).

---

## ⛔ Project Concluded — Succeeded by Model 5c TFT

**Conclusion**: Model 5b (Exp 37, INT8 on Coral) beat Model 5a clean (5ac) in live deployment at all timeframes < 1 year. Float MAE ~0.174°C vs 5a ~0.161°C — comparable, not 3× worse as raw val_loss suggested. PTQ caused some accuracy degradation vs. float but the deployed INT8 model still outperformed 5ac. The Conv2D layer itself adds no value; Dense anchor path does all the work.

Note: the raw val_loss comparison (5b 0.002117 vs 5a 0.000682) used throughout this project is misleading — the two models use different target scaler ranges (28.45°C vs 36.07°C) so the numbers are not directly comparable.

**Succeeded by**: [Model 5c TFT](../Model%205c%20TFT/MODEL_5C_PLAN.md)
- Track A: TFT as full prediction model (no TPU constraint)
- Track B: TFT as discovery tool → explicit features → lean Dense model on Coral TPU

Exp 41 (diurnal routing with lag features restored) is **cancelled** — no longer relevant given the project conclusion.

---

## Final Experiment Status

### Exp 36 — Conv2D Pattern Analysis (2026-06-07) ✅ Completed

**Goal:** Empirically verify what the Exp 35 Conv2D filters learned before committing to the next architecture change.

**Key findings (`analyze_conv2d_patterns.py` on Exp 35 weights):**
- **Dead filters:** 58% of temporal filters in conv2d_t1 and 64% in conv2d_t2 have near-zero weights. The temporal Conv2D blocks have largely collapsed.
- **Activation maps:** The model activates most strongly at **t=−180 (oldest timestep)** in all scenarios, and at t=0 (current). The middle 178 positions are essentially silent.
- **Saliency maps — Hypothesis B confirmed:**
  - Δ1hr: dominated by `temp_lag120` at t=0 (anchor path)
  - Δ2hr: dominated by `temp_lag60` at t=0 (anchor path)
  - Δ3hr: dominated by `temperature` at **t=−179** (oldest sequence row — the Conv2D's one unique contribution)

**Verdict:** The Conv2D does exactly one useful thing: provide an implicit `temp_lag180` for the Δ3hr head by reading raw temperature at the start of the 3-hour window. The fix is to make `temp_lag180` explicit.

---

### Exp 37 — Add `temp_lag180` Explicit Feature (2026-06-07) ✅ Complete

**Core change:** Added `temp_lag180` as the 28th input feature. This removes the Conv2D's incentive to attend to the oldest sequence row and tests whether the Conv2D contributes *anything* when the anchor path has all engineered temporal summaries including 180-min lookback.

**Run history:**
| Run | Platform | Outcome | Notes |
|-----|----------|---------|-------|
| 1 | Kaggle T4 ×2 | NaN at epoch 38 | `mixed_float16` LossScaleOptimizer overflow |
| 2 | Kaggle T4 ×2 | NaN at epoch 56 | `global_clipnorm=1.0` insufficient fix |
| 3 | Kaggle T4 ×2 | NaN at epoch 73 | Accumulated scale growth; fix: disable `mixed_float16` entirely |
| 4 | Kaggle T4 ×2 | Timeout at epoch 68 | float32 (post-fix); best val_loss=0.002192 at ep.~53 |
| 5 | Kaggle T4 ×2 | ✅ ES fired at epoch 131 | Resumed from ep.67; best val_loss=**0.002117** at ep.76 |

**Final result:** val_loss=0.002117, val_mae=0.006122 (normalized), best_epoch=76. `temp_lag180` confirmed as top-2 feature (0.0777). Feature distribution extremely flat (range 0.066–0.081) — anchor Dense(32) distributing gradient evenly, no Conv2D specialisation. Anchor-ceiling persists.

**Conclusion:** All Exp 37 success criteria met (val_loss < 0.002368 ✅, best_epoch > 20 ✅, temp_lag180 top 5 ✅, TFLite conversion ✅). However, best val_loss 0.002117 is still ~5.7× short of the Model 5a clean target (0.000373). The flat feature importance confirms the Conv2D is not learning complementary patterns — structural redundancy remains. → motivates **Exp 38**.

---

## Exp 32 Notable Result: temp_lag120 Finally #1

For the first time across all Model 5b experiments, `temp_lag120` is the top feature (importance 0.1025), with `temp_lag60` at #2 (0.0997). Previous experiments always had `time_of_day_cos` dominant, meaning the model was predicting based on diurnal patterns rather than the actual temperature trajectory. The ReduceLROnPlateau + L2 combination gave the skip-path anchors the gradient weight they need — now matching Model 5a's behavior where `temp_lag120` was also dominant (0.093).

---

## Next Experiments

### Exp 38 — Two-Stream Architecture ✅ Complete (2026-06-16)

Remove explicit lag/slope features from the Conv2D input stream so it must learn from raw sensors only, giving it a non-redundant role.

| Stream | Input | Role |
|--------|-------|------|
| Conv2D | Raw 18 sensors (no `temp_lag*`, no slopes) | Learn cross-sensor temporal dynamics; cannot shortcut to lag re-extraction |
| Anchor | Full 28 features incl. all lag + slope | Direct access to all engineered temporal summaries |

**Final result:** val_loss=**0.003779**, val_mae=0.0101 (normalized), best_epoch=**32**. TFLite PTQ: catastrophic failure (diff_1hr=1.07°C, diff_2hr=2.15°C, diff_3hr=2.24°C). Results in `Kaggle/results_2_exp38/`.

**Feature importance range: 0.0661–0.0790** — still very flat. `uv` (3rd) and `wind_lull` (4th) rank higher than in Exp 37 (weak signal of Conv2D raw-sensor contribution), but distribution unchanged structurally. `temperature` (current) is top feature, not lag features. Model peaked at epoch 32; 118 subsequent epochs produced only noisy oscillation — no further improvement. Early stopping (patience=30) never fired.

**Conclusion:** ❌ Two-stream hypothesis NOT confirmed. Exp 38 is **1.79× WORSE than Exp 37** (0.003779 vs 0.002117). The gate (val_loss < 0.001) was not met. Removing lag/slope from the Conv2D stream didn't force useful specialisation — it reduced effective input information during the anchor-ceiling phase (epoch 1–32), yielding a weaker baseline that the Conv2D path never recovered. Anchor-ceiling persists.

---

### Exp 39 — Dilated Convolutions ✅ Complete (2026-06-17)

Expanded the Conv2D temporal receptive field from 23 minutes to ~4 hours using exponentially dilated kernels. 150 epochs, Mac Metal GPU. LR decayed to 3.125e-6 (5 halvings).

| Layer | Exp 38 kernel/dilation | Exp 39 kernel/dilation | RF change |
|-------|----------------------|----------------------|-----------|
| conv2d_t1 | k=(3,1), d=1 | k=(3,1), d=1 | 3 min (unchanged) |
| conv2d_t2 | k=(7,1), d=1 → RF 9 min | k=(7,1), d=4 | 27 min |
| conv2d_t3 | k=(15,1), d=1 → RF 23 min | k=(15,1), d=16 | **~251 min (~4 hours)** |

**Final per-head results vs Exp 38:**

| Head | Exp 38 best | Exp 39 final (ep150) | Change |
|------|-------------|----------------------|--------|
| 1hr | 9.7e-04 | **4.45e-04** | **−54%** ✅ |
| 2hr | 7.7e-04 | 8.19e-04 | +6% ≈ tied |
| 3hr | 1.0e-03 | 1.3e-03 | **+30%** ❌ |

**Key findings:**
- 1hr head dramatically improved (54% better than Exp 38)
- 3hr head regressed 30% vs Exp 38 — the opposite of what the dilated-RF hypothesis predicted (3hr should have benefited most from longer temporal context)
- val_loss (0.0163) is dominated by L2 from dilated weight magnitudes; not comparable to Exp 37/38
- PTQ quantization failed: 0.52/1.06/1.38°C
- Feature importance: time_of_day_sin2 #1 (0.0754), temperature last (0.0446) — lag features in anchor path (temp_lag60/180 rank 6th–7th), but diurnal signals still dominate the top

**Conclusion:** ❌ Dilated RF hypothesis not confirmed. 3hr regression and diurnal feature dominance indicate the Conv2D is not learning multi-hour weather pattern dynamics as hypothesized. The 1hr improvement is real but unexplained by the hypothesis — it may reflect richer recent-history context rather than true multi-hour pattern detection.

**Gate for Exp 40:** Weakly unmet (mixed results, 3hr regression). Exp 40 was run anyway (2026-06-18) — see below.

---

### Anchor-Only Baseline (open — deferred)

With Exp 38 WORSE than Exp 37 and Exp 39 showing mixed results, the key open question is: does removing the Conv2D path entirely produce the same or better result? Deferred while diurnal routing hypothesis (Exp 41) is tested — if Exp 41 shows no improvement over Exp 39, the anchor-only baseline becomes the next priority.

---

### Exp 40 — Diurnal Out of Conv2D + Lag Features Removed ✅ Complete (2026-06-18)

**Status**: ✅ Complete — best_epoch=56, val_loss=0.1701, quantized MAE 3.04/1.09/7.26°C  
**Outcome**: ❌ FAILED — lag feature removal caused catastrophic regression (26×/14×/79× worse per-head vs Exp 39). Lag features (`temp_lag60/120/180`, `temp_delta_1`) are the model's essential temperature trajectory anchors; the dilated Conv2D cannot replace them implicitly. Diurnal routing showed partial effect (time_of_day_sin2 dropped from #1 to #2; temperature rose from #28 to #3) but is uninterpretable alongside the lag regression.

**Key lesson**: Never combine two major feature changes in one experiment — confounds the diagnosis when one change is beneficial and the other catastrophic.

---

### Exp 41 — Isolate Diurnal Routing: Restore Lag Features (⛔ CANCELLED — project concluded)

**Status**: Cancelled — project concluded before this experiment ran.  
**Goal**: Test only the diurnal routing hypothesis from Exp 40 — remove `time_of_day_sin/cos/sin2/cos2` from Conv2D input, keep them in anchor path only — while restoring `temp_lag60`, `temp_lag120`, `temp_lag180`, `temp_delta_1` to the anchor path.

| Parameter | Exp 39 | Exp 41 |
|-----------|--------|--------|
| Conv2D input | 18 sensors incl. diurnal | **14 physical sensors — diurnal removed** |
| Anchor path | All 28 features incl. lag | **All 28 features incl. lag** (lag restored vs Exp 40) |
| n_conv | 18 | **14** |
| Reshape kernel | `(SEQ_LEN, 18, 1)` | `(SEQ_LEN, 14, 1)` |
| Feature-mixing Conv2D | `kernel_size=(1, 18)` | `kernel_size=(1, 14)` |
| Dilated kernels / RF | k=(3,1)d=1/k=(7,1)d=4/k=(15,1)d=16, RF ~251 min | Unchanged |

**Code change**: Confirm `time_of_day_sin/cos/sin2/cos2` are in `diurnal_features` (not `conv_features`), and `temp_lag60`, `temp_lag120`, `temp_lag180`, `temp_delta_1` are present in `engineered_features`.

**Success criteria:**
- Per-head val losses match or beat Exp 39 (1hr ≤ 4.45e-4, 2hr ≤ 8.19e-4, 3hr ≤ 1.3e-3)
- `time_of_day_sin2` no longer #1 feature — should fall behind lag features or solar/temperature
- `temp_lag60/120/180` remain in top 10 feature importance
- `temperature` rises from #28 (Exp 39)
- best_epoch > 30

---

### Quantization (Phase 2)

PTQ has never worked for this architecture; QAT is the only untried fix. Do not pursue until float val_loss ≤ 0.000373 is achieved.
