# Model 5c: TFT-Based Feature Discovery

## Project Goal

Use a **Temporal Fusion Transformer (TFT)** to empirically discover which sensor features and lag timeframes drive multi-horizon temperature prediction accuracy. Then use those findings in one of two ways:

- **Track A — Everything Model**: deploy the TFT directly for predictions (no Coral TPU constraint; runs on Pi CPU or Mac)
- **Track B — Optimized TPU Model**: encode the TFT-discovered lag features as explicit inputs to a lean Dense model, deployed on Coral Edge TPU at INT8

---

## What Model 5b Established (Key Inputs)

Model 5b ran 40 experiments across Conv1D and Conv2D architectures. The definitive findings:

| Finding | Evidence |
|---------|---------|
| Conv2D adds no value over Dense-only | Anchor Dense path dominates every experiment once explicit lag features are present |
| Explicit temperature lag features are essential | `temp_lag60/120/180` consistently top-ranked; removing them causes 26×/79× per-head regression (Exp 40) |
| GAP destroys positional information | Single-position signals diluted 180×; diurnal (fires everywhere) always survives over lags (fire at one position) |
| PTQ causes significant accuracy degradation in offline validation | Offline validation showed near-constant output collapse (0.68/1.20/1.63°C MAE per head for Exp37); model does compile and run on Coral — live accuracy impact may differ from offline validation |
| Float accuracy matched Model 5a by Exp 37 | Exp37 MAE ~0.174°C vs 5a ~0.161°C in actual °C — marginal difference; raw val_loss comparison (0.002117 vs 0.000682) was misleading due to different scaler ranges (28.45 vs 36.07°C) |

**Feature gap identified (Zambretti Forecaster insight):** Barometric pressure *tendency* over 3 hours is a primary meteorological signal for weather system approach/departure. The current feature set has `pressure_slope_60` (60-min local derivative) but is missing `pressure_lag_120`, `pressure_lag_180`, and `pressure_diff_180` (= station_pressure − pressure_lag_180). A TFT would likely rediscover this; it is also directly actionable as a Track B feature candidate.

---

## Why TFT

The TFT (Lim et al., 2019) is the right tool because it was designed for exactly this problem:

- **Variable Selection Network (VSN)**: learns per-feature importance weights at each timestep — answers "which features matter at which point in the window?"
- **Multi-head temporal self-attention**: learns which past timesteps to attend to per head — answers "which lag timeframes drive accuracy?"
- **Multi-horizon output**: predicts 1hr/2hr/3hr natively, with per-horizon attention patterns
- **Interpretable**: attention maps and variable importance scores are outputs, not side-effects
- **No GAP**: attention produces position-specific weighted sums — a signal at t=−180 is not diluted by the other 179 positions

Attention is what Conv2D+GAP could not do: **non-uniform, learned weighting over specific past positions**.

---

## Track A — Everything Model (TFT, no TPU constraint)

**Goal**: Deploy TFT directly for predictions. No Edge TPU constraint.

**Inputs**: same raw 180-minute sensor window as Model 5b, plus extended lag candidates:
- All current features (temperature, humidity, pressure, solar, wind, illuminance, uv, rain)
- Cyclical encodings (time_of_day, day_of_year)
- Existing explicit lags: `temp_lag60`, `temp_lag120`, `temp_lag180`, `temp_delta_1`
- New candidates: `pressure_lag_60`, `pressure_lag_120`, `pressure_lag_180`, `pressure_diff_180`
- Slope features: `temp_slope_15/30/60`, `solar_slope_30`, `humidity_slope_30`, `pressure_slope_60`

**Outputs**: `temp_diff_1hr`, `temp_diff_2hr`, `temp_diff_3hr` (same targets as 5a/5b)

**Deployment**: TFLite float32 or float16 on Pi CPU. The Coral TPU delegate can be omitted entirely — inference runs on ARM CPU.

**Success criteria**:
- val_loss < 0.000373 (beat Model 5a clean dense_wide_run1)
- Per-horizon MAE < 0.10°C normalized
- Deployed 30-day StdDev < 0.607°C (beat Model 5b Exp32 float, the current best)

---

## Track B — Optimized TPU Model (TFT → explicit features → Dense)

**Goal**: Use TFT as a discovery tool only. Extract the highest-importance (feature, lag) pairs from its attention maps and variable selection weights, encode those as explicit features, then train a lean Dense model (Model 5a style) that fits on the Coral TPU at INT8.

**Process**:
1. Train Track A TFT
2. Inspect attention maps: which timesteps get highest attention per head, per output horizon?
3. Inspect VSN weights: which features contribute at which timeframes?
4. Select top N (feature, lag) pairs as new explicit features (e.g., `pressure_lag_180`, `humidity_lag_60`)
5. Add those features to Model 5a's input — train Dense model
6. Quantize to INT8, compile with `edgetpu_compiler`, deploy on Coral

This is the same loop as the Exp 36→37 discovery (saliency map revealed `temp_lag180` was implicit → added it explicitly → improved results), but systematic and across all features.

**Success criteria**:
- Deployed on Coral Edge TPU at INT8 with all ops on TPU (no CPU fallback)
- val_loss < 0.000373 (beat Model 5a clean)
- 30-day deployed StdDev < 0.988°C (current Model 5a)

---

## Feature Space to Explore

Beyond the features already established as useful, the TFT should be given the opportunity to discover:

| Candidate | Motivation |
|-----------|-----------|
| `pressure_lag_120`, `pressure_lag_180` | Zambretti: 3-hr pressure tendency is the primary frontal signal |
| `pressure_diff_180` = pressure_now − pressure_lag_180 | Zambretti tendency directly encoded |
| `humidity_lag_60`, `humidity_lag_120` | Marine layer / fog onset has humidity trajectory signature |
| `wind_dir_lag_60` sin/cos | Wind direction *change* (backing vs. veering) indicates system rotation |
| Extended temp lags: `temp_lag_240`, `temp_lag_300` | May matter for 3hr head — current window only reaches 180 min |

If the TFT's attention on temperature spikes at t>180 minutes, the input window itself should be extended.

---

## Implementation Notes

**Framework**: TensorFlow/Keras (consistent with 5a/5b codebase) or `pytorch-forecasting` (battle-tested TFT reference implementation). The `pytorch-forecasting` TFT is the canonical implementation and has the best interpretability tooling.

**Training platform**: Mac Metal GPU or Kaggle T4. No Edge TPU quantization constraint during discovery.

**Input window**: start with 180 minutes (matches 5a/5b). If attention maps show significant weight beyond t=−180, extend to 240 or 360 minutes.

**Sequence format**: TFT expects known time-varying inputs (sensors), static covariates (none here unless station metadata is added), and future known inputs (time-of-day, day-of-year cyclicals that are always known ahead).

---

## Reference Baselines

| Model | val_loss | Deployed StdDev (30 days) | Edge TPU |
|-------|---------|--------------------------|---------|
| Model 5a (deployed) | 0.000682 | 0.988°C | ✅ |
| Model 5a clean avgpool_run1 | 0.000508 | — | ✅ |
| Model 5a clean dense_wide_run1 | **0.000373** | — | ❌ (SRAM overflow) |
| Model 5b Exp37 (INT8 on Coral, deployed) | ~0.002 | 0.720°C (6-month) / 0.930°C (30-day) / 0.922°C (7-day); beats 5a/5ac by 19–26% | ✅ |
| Model 5b Exp32 (float, deployed) | ~0.002 | 0.607°C (30-day May 2026) | ❌ float only |

**Live bias (June 2026):** 5b-37 mean error −0.063°C at 30 days — nearly unbiased vs 5a (+0.157°C) and 5ac (−0.298°C).

Model 5c Track A target: beat Exp32 float 30-day StdDev of 0.607°C.  
Model 5c Track B target: beat Exp37 INT8 6-month StdDev of 0.721°C, fully on TPU with TFT-discovered features.

---

*Created: 2026-06-18. Motivated by Model 5b conclusion (40 experiments) that Conv2D adds no value over Dense + explicit lag features.*
