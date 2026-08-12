# Model 5d: Flat-Feature Dense Model for Coral Edge TPU

## Project Goal

Supersede Model 5c Track B's `SEQ_LEN=180 + AvgPool + wide/deep + concat` architecture with a
single-path, flat-feature Dense model that eliminates the structural sources of Track B's INT8
problems by construction, rather than patching them one tensor at a time.

**Predecessor**: Model 5c (Track A: TFT feature discovery; Track B: TFT-informed Dense model).
Track B ran 22 experiments; the last 5 (Runs 18-22) each diagnosed and cleanly fixed a real,
confirmed INT8 precision-loss mechanism (input-scale stretching, concat forced-shared-scale,
long-tail activation calibration) without ever recovering 3hr INT8 accuracy — see
`../Model 5c TFT/MODEL_5C_TRACK_B_EXPERIMENT_LOG.md` Runs 18-22 for the full investigation.
The conclusion driving this pivot: the bottleneck is structural (4-5 sequential quantized
MatMuls, branching, concat), not any single fixable tensor.

---

## Why This Architecture

Two findings from Model 5c make a flat, sequence-free architecture viable now in a way it
wasn't in Track B's own Runs 1-5 (which used SEQ_LEN=1 and hit a hard ~0.48-0.53°C 1hr MAE
ceiling):

1. **Track A's TFT (Integrated Gradients, Run 2)** found `temp_slope_15/30/60` dominate raw
   temperature attribution — the model is primarily computing a slope-weighted prediction, not
   reading raw levels. These are precomputed scalars (local linear regression over a window),
   not something requiring attention over the raw sequence.
2. **Track B's own Run 16 discovery**: `temp_diff_vs_5hr/6hr` (a non-boundary ~5hr attention
   anchor Track A found in a SEQ_LEN=360 deep run) became the single most important feature by
   permutation importance across every subsequent Track B run — again, a precomputed scalar via
   `merge_asof` lookback, not a property of the raw window.

Between these, the temporal information the model actually uses is already distilled into
explicit scalar features. The raw 180-step sequence's remaining marginal value (whatever the
`AvgPool` was still contributing beyond these lags/slopes) is traded for a dramatically simpler,
shallower, more Edge-TPU-friendly graph.

**Architecture**: `Input(N_features) → Dense(64, relu6) → Dense(32, relu6) → 3 output heads`.
No `AvgPool`, no wide/deep branching, no `Concatenate`. Verified directly (standalone conversion
test) that this converts to a minimal, clean INT8 graph: 2 fused `MatMul+Relu6` tensors plus 3
independent linear output heads — no forced-shared-scale tensors, no long op chains.

---

## Feature Set (20 features)

Track B's surviving features (temporal, `temp_diff_vs_5hr/6hr`, `pressure_slope_60`, solar)
plus features Track A's TFT found essential (stable across Track A Runs 2-5) that Track B's
`SEQ_LEN=180` architecture rejected as redundant with the pooled raw sequence:
`temp_slope_15/30/60`, `relative_humidity`, `humidity_slope_30`, `station_pressure`. That
redundancy rationale doesn't apply here — there is no sequence for them to be redundant with.

| Category | Features |
|----------|----------|
| Temporal | `time_of_day_sin/cos/sin2/cos2`, `day_of_year_sin/cos` |
| Temperature | `temperature`, `temp_diff_vs_5hr`, `temp_diff_vs_6hr`, `temp_slope_15/30/60` |
| Humidity | `relative_humidity`, `humidity_slope_30` |
| Pressure | `station_pressure`, `pressure_slope_60` |
| Solar | `solar_radiation`, `solar_slope_30`, `illuminance`, `uv` |

Wind/rain remain excluded — confirmed floor-level importance (VSN < 0.01, permutation
importance ~0) across every Track A and Track B run to date.

---

## Targets

Same as every prior model in this project: `temp_diff_1hr`, `temp_diff_2hr`, `temp_diff_3hr`.

## Success Criteria

- val_loss < 0.000373 (Model 5a clean `dense_wide_run1` target, carried forward from Track B)
- INT8 3hr MAE < 0.898°C (beat Track B's best deployable checkpoint, Run 11)
- Clean, minimal TFLite INT8 graph with no forced-shared-scale or long-tail-calibration tensors
  (verifiable via the diagnostic-audit technique developed in Track B Runs 20-22, portable to
  this architecture if needed)

## Open Questions For Run 1+

- Does dropping the raw sequence cost meaningful FP32 accuracy versus Track B's best runs
  (0.068-0.075°C 1hr MAE), given the new feature set is richer than Track B's own 13 features?
- Do `relative_humidity`/`station_pressure`/`humidity_slope_30` — confirmed harmful in Track
  B's SEQ_LEN=180 architecture — turn out useful or still harmful here? Permutation importance
  in Run 1 should tell us; drop candidates for Run 2 if still negative.
- Is `HIDDEN_1=64, HIDDEN_2=32` the right capacity, or does this need to be wider/narrower/
  deeper once real numbers come in?
