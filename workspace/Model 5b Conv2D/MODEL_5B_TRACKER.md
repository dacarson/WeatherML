# Model 5b Experiment Tracker

## Goal
Match or beat **Model 5a** (val_loss=0.000682, val_mae=0.00445) using a Conv2D architecture with explicit temperature lag features, while remaining Edge TPU-compilable (all ops on TPU, fits in 8 MB SRAM, INT8 quantized).

## Target
**Phase 1 (current):** float val_loss ≤ 0.000682  
**Phase 2 (after Phase 1):** QAT to close the float→quantized accuracy gap (float ~0.01°C vs quantized ~0.82–1.12°C)

## Architecture (Exp 27+ baseline)
```
Input: (180, n_features) → Reshape to (180, n_features, 1)
  ├─ Conv2D path: [Conv2D(64, k=3)→BN→ReLU6 → ... → Conv2D(64, k=feat)→BN→ReLU6]
  │               → GlobalAveragePooling2D → Dense(64) → ReLU6 → [optional Dropout] → context
  └─ Skip path:   input[:, -1, anchor_features] → Dense(32) → ReLU6 → anchors
Concatenate([context, anchors]) → Dense(32) → ReLU6 → Dense(3) outputs
```

**Fixed inputs (Exp 27+):** 27 features = 12 raw station + 6 cyclical time encodings + `temp_delta_1` + `temp_lag60` + `temp_lag120` + 6 slope features (`temp_slope_15/30/60`, `solar_slope_30`, `humidity_slope_30`, `pressure_slope_60`)  
**Fixed training:** cosine LR decay 1e-4 → 1e-6 over 100 epochs, MSE loss, batch=512, patience=25

## Key Learnings — What Doesn't Work
| Approach | Outcome | Reason |
|---|---|---|
| Flatten → Dense (Exp 1–8) | val_loss ~0.006–0.009 | Lag value buried at one position in 5,400-dim input; no temporal inductive bias |
| Conv1D + SliceTimestep (Exp 9–26) | Best float 0.001343; quantization always broken | SliceTimestep/SliceFeatures ops excluded from QAT; activation range unbounded; never Edge TPU-viable |
| No explicit lag features (Exp 9–25) | temperature always ranked dead last | Diurnal signal (time_of_day) too dominant; Conv1D cannot learn implicit temperature anchoring |
| Conv2D + GAP without skip path (Exp 27) | lag features ranked 18th–19th | GAP averages 180 timesteps, diluting single-timestep anchor values 180× |
| Exp 30: Dropout(0.3) on context vector | val_loss regressed 0.0026→0.0032; best epoch=3 | Too aggressive for 64-unit vector; model underfits; all features compress to flat 0.056–0.101 band |
| Adding more features (Exp 29 slopes, Exp 26 lags) | No float accuracy improvement | Float ceiling is overfitting (train/val ~10×), not insufficient features |
| PTQ post-training quantization (all Conv1D) | Constant output collapse | Unbounded intermediate activations; can't be fixed with representative dataset calibration |

## Key Learnings — What Works
| Approach | Result |
|---|---|
| Conv2D + GlobalAveragePooling2D (no custom ops) | First successful PTQ; all ops Edge TPU-compatible |
| Skip path for anchor features | Fixes GAP dilution; temperature→#1, lag60/lag120 rise to top-10 |
| Explicit `temp_lag60` + `temp_lag120` features | temp_lag60 correctly anchors predictions (Exp 26+) |
| Slope features in skip path | Quantized MAE improved all 3 horizons vs Exp 28 |
| Cosine LR decay | Smoother convergence vs ReduceLROnPlateau |
| Dropout(0.3) — narrows train/val *ratio* | Gap went ~10× → ~4.1× but absolute val_loss got worse (underfitting) |

---

## Experiment Results Summary

| Exp | Key Change | float val_loss | Best Epoch | train/val gap | Quant MAE (1/2/3hr °C) | Edge TPU |
|-----|-----------|---------------|-----------|--------------|------------------------|---------|
| 24 | Conv1D dual branch 64:64, best Conv1D | 0.001343 | 46 | ~2.3× | 0.61/1.16/2.63 | ❌ |
| 27 | Switch to Conv2D + GAP (no skip) | 0.0027 | ~89 | oscillating | 1.57/2.21/2.63 | ✅ first PTQ success |
| 28 | + skip path (temp, lag60, lag120 → Dense(16)) | 0.0028 | ~91 | ~8× | 1.12/1.63/2.01 | ✅ |
| 29 | + 6 slope features in input + skip expanded to Dense(32) | 0.0026 | 15 | ~10× | 0.82/1.49/1.63 | ⚠️ diff_3hr near-constant |
| 30 | + Dropout(0.3) on context vector | 0.0032 | 3 | ~4.1× | 0.67/1.39/1.71 | ✅ |
| **31** | Dropout(0.3) → **Dropout(0.1)** | ⏳ | — | — | — | — |

---

## Current Experiment: Exp 31

**Hypothesis:** Dropout(0.3) was too aggressive — it narrowed the train/val ratio but caused underfitting (best epoch=3, val_loss regressed). Dropout(0.1) masks only ~6–7 of 64 neurons per step — enough to prevent memorization, light enough to preserve feature gradient differentiation.

**Single change from Exp 30:**
```
GAP → Dense(64) → ReLU6 → Dropout(0.1) → context   # was Dropout(0.3)
```

**Success criteria:**
- float val_loss ≤ 0.0026 (beat Exp 29)
- best epoch later than epoch 3 (confirms regularization, not underfitting)
- train/val gap ≤ 4× (keep ratio improvement without sacrificing absolute accuracy)
- feature importance spread wider than Exp 30's flat 0.056–0.101 band

---

## Next Experiments (Planned — pending Exp 31 results)

**If Exp 31 val_loss improves (< 0.0026):**
- Exp 32: Try Dropout(0.2) — compare 0.1 vs 0.2 to find optimal rate
- Exp 33: Add L2 weight decay (1e-4) to Dense(64) alongside Dropout(0.1) — compound regularization

**If Exp 31 still underfits (val_loss ≥ 0.0026, best epoch still ≤ 5):**
- Exp 32: Move dropout to the final Dense(32) head instead of context vector — smaller representation may tolerate higher rates
- Or: Try L2 regularization on Conv2D filters instead of dropout

**If regularization ceiling ~0.0026 is confirmed (Exp 31–32 both fail to improve):**
- Accept ~0.0026 float ceiling and pivot to Phase 2: QAT to close float→quantized gap (current ~0.82°C 1hr, target ≤0.5°C)
