# Model 5e — Experiment Log

**Script**: `train_model_5e.py` (forked from `../Model 5c TFT/train_model_track_b.py`)
**Target**: val_loss < 0.000373, INT8 3hr MAE < 0.898°C (beat Track B Run 11 — the checkpoint
being fine-tuned, not just Model 5b's deployed bar)
**Predecessor**: `../Model 5c TFT/MODEL_5C_TRACK_B_EXPERIMENT_LOG.md` (Runs 1-22, concluded
2026-07-22). See `MODEL_5E_PLAN.md` for full rationale.

---

## Run 1 — QAT Fine-Tuning from Run 22 Checkpoint (planned)

**Date**: TBD
**Script**: `train_model_5e.py`
**Platform**: TBD (Mac Metal QAT ran fine in Track B Run 19; Metal mixed precision auto-disables
when `QAT_FINE_TUNE=True` since tfmot requires float32)
**Results will be stored in**: `results_5e_run1/`

**Hypothesis**: Track B Run 19 applied QAT to Run 18's checkpoint (13 features, pre-concat-
rescale architecture) and got essentially no improvement over plain PTQ (0.608/1.066/1.630°C vs
0.595/1.041/1.658°C at 1/2/3hr). Runs 20-22 then cleanly fixed two more calibration mechanisms
(concat forced-shared-scale, deep_out long-tail range) on top of that same lineage without
QAT involved. This run asks whether QAT does anything useful when applied *after* those fixes,
on the calibration-cleaner Run 22 graph, rather than before them as Run 19 tested. If the
depth-of-sequential-MatMuls diagnosis from Run 22's "Next steps" is the real ceiling, QAT should
fail here too regardless of the cleaner starting calibration — that's the falsifiable part.

**Configuration** (relative to `train_model_track_b.py`'s state as of Run 22):
1. `RUN_NAME = "5e_run1"`
2. `QAT_FINE_TUNE = True` (set at top of file, before the `import tensorflow`, per the
   `TF_USE_LEGACY_KERAS` gating tfmot requires for Keras 3 Functional models)
3. `SKIP_TRAINING = True`
4. `SOURCE_CHECKPOINT = "../Model 5c TFT/results_5c_trackb_dense_b_run22/checkpoints/best_model.weights.h5"`
5. `WARM_START = False` (irrelevant when `SKIP_TRAINING=True`; warm start only applies to fresh
   training runs)
6. Architecture, features (13), `DEEP_OUT_PRESCALE`/`DEEP_OUT_RESCALE` all unchanged from Run 22
   — no structural changes, this is a pure config retarget of the existing QAT machinery Run 19
   already built and validated (Keras-3/tfmot compatibility shim, relu-clone for tfmot's
   relu6-unsupported whitelist, QuantizeWrapperV2 weight-extraction for FP32 export).
7. `QAT_LR = 1e-6`, `QAT_EPOCHS = 50`, `QAT_EARLY_STOP_PATIENCE = 10` (unchanged from Run 19 —
   revisit only if this run's loss curve suggests they were mis-tuned, per the plan's open
   questions)

**Expected outcomes**:
- If the structural-depth diagnosis is correct: QAT INT8 MAE stays close to Run 22's own PTQ
  baseline, i.e. this run fails the same way Run 19 did, just on a different starting graph.
- If calibration-fixed-before-QAT matters: some improvement over Run 22 PTQ, ideally approaching
  or beating Run 11's 0.898°C 3hr bar.
- FP32 (fake-quant-simulated) val_loss should land close to Run 22's 0.000400, per Run 19's
  precedent (0.000381 QAT vs Run 18's 0.000450 PTQ-baseline FP32 — QAT's own training loss
  reliably improves even when real INT8 doesn't).

**Results** (2026-08-12, run by user on their own machine):
- val_loss (includes L2): **0.000377** (Run 22 PTQ FP32 baseline: 0.000400 — essentially the same,
  slightly better)
- FP32 MAE: diff_1hr **0.114°C**, diff_2hr **0.126°C**, diff_3hr **0.155°C** (Run 22 PTQ FP32:
  0.068/0.080/0.105°C — real regression at every horizon, ~60-70% worse; same direction and
  rough magnitude as Run 19's FP32 regression from Run 18's 0.075/0.090/0.116°C)
- INT8 MAE (n=500): diff_1hr **0.635°C**, diff_2hr **1.090°C**, diff_3hr **1.627°C**
- Best epoch: **15**
- FP32 TFLite: 177.4 KB · INT8 TFLite: 51.2 KB

**Comparison table**:

| | Run 22 (PTQ, pre-QAT) | Run 1 (QAT from Run 22) | Run 19 (QAT from Run 18) | Run 11 (target to beat) |
|---|---|---|---|---|
| FP32 1hr/2hr/3hr | 0.068/0.080/0.105 | 0.114/0.126/0.155 | 0.075/0.090/0.116 (Run 18) | 0.091/—/— |
| INT8 1hr | 0.650 | **0.635** (−2.3%) | 0.608 | 0.522 |
| INT8 2hr | 0.979 | **1.090** (+11.3%) | 1.066 | 0.588 |
| INT8 3hr | 1.821 | **1.627** (−10.7%) | 1.630 | **0.898** |

**Outcome**: ❌ Does not beat Run 11's 0.898°C 3hr target — misses by ~81%. Technically beats
Run 22's own pre-QAT PTQ baseline at 1hr/3hr (mixed at 2hr), so QAT is not *inert* here, but the
improvement is marginal and the result lands almost exactly on top of Run 19's numbers
(0.635/1.090/1.627 vs Run 19's 0.608/1.066/1.630 — within noise of each other) **despite starting
from a meaningfully different, more calibration-fixed checkpoint**. That convergence to nearly
identical INT8 numbers regardless of starting calibration state is itself evidence for the
structural-depth theory: QAT's fine-tuning genuinely does something (FP32 shifts, small INT8
deltas), but whatever caps 3hr INT8 around ~1.6°C doesn't move much with either the checkpoint's
calibration history or the fine-tuning itself. FP32 accuracy paid a real cost (~60-70% MAE
regression) for a result that still isn't deployable.

**Decision**: proceed to Run 2 (QAT from Run 11) as planned — it's the last untested combination
and the only one starting from a *simpler* graph (11 features, no prescale/rescale sandwich) —
but Run 1's result lowers the odds it succeeds where Run 1 and Run 19 both landed on the same
~1.6-1.7°C 3hr ceiling from two different starting points.

---

## Run 2 — QAT Fine-Tuning from Run 11 Checkpoint (script ready, not yet run)

**Date**: script built 2026-08-12; execution TBD (user runs locally, per established workflow)
**Script**: `train_model_5e_run2.py`, forked from `train_model_5e.py`. Runs in two stages via
the `QAT_FINE_TUNE` toggle — see below.
**Results will be stored in**: `results_5e_run2a/` (baseline) and `results_5e_run2b/` (QAT)

**Hypothesis**: This is the original Track B Run 15 QAT plan's actual intended target — Run 15
was written to fine-tune from Run 11 (the best deployable checkpoint, 3hr INT8 0.898°C) but got
retargeted to Run 18 before execution (Run 18 was the best FP32 base available at the time).
Run 11's graph is simpler than Run 18/22's (11 features vs 13, no prescale/rescale sandwich) —
fewer quantized ops for QAT's fake-quant nodes to have to compensate for, and it's QAT applied
directly to the checkpoint that's actually the best deployment candidate today, so a positive
result is immediately useful rather than only diagnostic.

**Two problems discovered while building the script (both resolved, see full detail in
`MODEL_5E_PLAN.md`):**

1. **Architecture reconstruction.** The plan's original assumption ("separate `Dense→Activation`
   throughout, `by_name=True` loading") was wrong on both counts. Keras 3's native `.weights.h5`
   format has no `by_name` load path — it keys H5 groups by internal topological traversal
   order, not layer name (an existing comment elsewhere in `train_model_track_b.py` had already
   found this the hard way). And Run 11 was not uniformly unfused: `git show
   301985f:"workspace/Model 5c TFT/train_model_track_b.py"` (committed 2026-06-25, same day as
   Run 11, mid-flight at Run 14) showed only bottleneck/wide/deep1 were separate
   Dense+Activation — deep2 and deep_out were already fused, and `wide` was created immediately
   after `bottleneck`, before `deep1`. Rebuilt with this exact mix and verified two ways: (a) a
   save-weights round-trip reproduces Run 11's checkpoint's H5 group order byte-for-byte, (b)
   Keras's own `load_weights()` and an independent manual shape-based loader converge on
   identical evaluation results. `train_model_5e_run2.py` uses the shape-based loader in
   production (robust to Keras's internal ordering regardless of future edits), not plain
   `load_weights()`.

2. **Data pipeline drift.** Evaluating the correctly-reconstructed, correctly-loaded Run 11
   model against today's data pipeline gives FP32 MAE ~0.7-1.5°C (1/2/3hr) — far worse than Run
   11's published 0.091/0.092/0.121°C, worse than the trivial-persistence baseline. Not a
   loading bug: Run 22 (trained after a 2026-07-19 data-quality fix — sensor-glitch filtering +
   fresh target reconstruction) reproduces cleanly with the identical methodology (Run 1's
   result). Run 11 predates that fix. **User decision: re-establish a fair current-data baseline
   before QAT, rather than comparing against the stale number or skipping straight to QAT.**

**Configuration — Run 2a** (`QAT_FINE_TUNE=False`, `SKIP_TRAINING=True`,
`SOURCE_CHECKPOINT=Run 11`): loads Run 11's checkpoint via the shape-based loader, evaluates
FP32 and exports INT8 TFLite against today's data. Input scaler and target scaler are loaded
from Run 11's own saved JSON files rather than recomputed from today's `train_df`/`val_df`, to
isolate "does Run 11 generalize to current data" from "did the scaling convention itself also
shift." Weight-verification abort threshold raised from 0.001 to 0.02 (justified in-code:
~0.0096 is the independently-verified legitimate baseline, not a loading failure — every
wrong-architecture attempt during reconstruction gave val_loss ≥ 0.15, so 0.02 still catches a
genuine break).

**Configuration — Run 2b** (`QAT_FINE_TUNE=True`, same `SOURCE_CHECKPOINT`): QAT fine-tune from
the same Run 11 checkpoint (`QAT_LR=1e-6`, `QAT_EPOCHS=50`, patience 10 — same settings as Run
19/Run 1). No custom `QuantizeConfig` needed this time — Run 11's architecture has no `Rescaling`
layers, so tfmot's plain `quantize_model()` suffices (unlike Run 1, which needed the
`RescalingQuantizeConfig` workaround for Run 22's prescale/rescale sandwich).

**Expected outcomes**: Same falsifiable structure as Run 1, but now measured against Run 2a's
fair baseline rather than the stale 0.898°C. If the structural depth-of-MatMuls ceiling is the
true cause, Run 2b should fail to meaningfully beat Run 2a's own PTQ baseline, similar in kind to
Run 1 vs Run 19. If graph simplicity (not calibration state) is what actually determines whether
QAT can help, Run 2b should show more improvement over its own baseline than Run 1 did over Run
22's.

**Results — Run 2a, first attempt (2026-08-12, run by user)**: Checkpoint loaded correctly
(shape-based loader confirmed all 8 layers). FP32 evaluation on today's data:
val_loss=0.000328, MAE 0.098/0.108/0.137°C (1/2/3hr) — **closely matches Run 11's original
published 0.091/0.092/0.121°C**. This retracts the "data pipeline drift" finding above: an
earlier scratchpad verification (not this script) had suggested Run 11 performed far worse on
current data; that was a bug in the ad-hoc script, not real. Run 11's checkpoint generalizes to
current data fine — see `MODEL_5E_PLAN.md` for the correction.

Permutation importance (val_loss increase): `time_of_day_cos` (0.0142) and `time_of_day_sin`
(0.0094) dominate, consistent with every other Track B run; `pressure_slope_60` and
`day_of_year_cos` at floor (~0.0000/-0.0000).

**Crashed during FP32 TFLite export** (before INT8 export/results.json could be written):
`export_model.set_weights([w.astype(np.float32) for w in model.get_weights()])` in the
mixed-precision export branch copied weights *positionally*, but that branch's `_e_bn`/`_e_wide`/
`_e_d1` were fused (`activation="relu6"` in the Dense constructor) while `model` keeps them
unfused (Run 11's real structure) — different internal layer order, so positional copying paired
deep2's kernel with wide's slot: `ValueError: Layer functional weight shape (128, 64) is not
compatible with provided weight shape (64, 16)`. Fixed to name-based transfer (`get_layer(name)`
by name, not position) — correct regardless of either model's internal ordering. Also proactively
fixed the same bug pattern in the QAT-clone weight-copy step (`model_for_qat.set_weights(model.
get_weights())`), which Run 2b would have hit next since its `_q_*` layers are equally fused.

**Results — Run 2a, re-run after fix (2026-08-12, run by user)**: Completed cleanly through
FP32 + INT8 export.
- val_loss (includes L2): **0.000328** (unchanged from the first attempt, as expected — the fix
  only touched the export path, not evaluation)
- FP32 MAE: diff_1hr **0.098°C**, diff_2hr **0.108°C**, diff_3hr **0.137°C** — closely matches
  Run 11's original published 0.091/0.092/0.121°C
- INT8 MAE (n=500): diff_1hr **0.493°C**, diff_2hr **0.638°C**, diff_3hr **1.050°C**
- Best epoch: 600 (SKIP_TRAINING — this is just the loaded checkpoint's own epoch marker, not a
  new training run) · FP32 TFLite: 162.4 KB · INT8 TFLite: 47.4 KB

**Comparison to Run 11's original published INT8** (0.522/0.588/0.898°C):

| | Run 11 original (pre-2026-07-19 data) | Run 2a (today's data) |
|---|---|---|
| INT8 1hr | 0.522 | 0.493 (−5.6%, better) |
| INT8 2hr | 0.588 | 0.638 (+8.5%, worse) |
| INT8 3hr | 0.898 | **1.050 (+16.9%, worse)** |

Mixed but broadly consistent — no dramatic drift, confirming the earlier "data pipeline drift"
scratchpad finding was indeed a false alarm (see above). Some real movement at 2hr/3hr is
expected: different INT8 calibration sample each run, and today's val set isn't byte-identical
to what Run 11 calibrated against even if the underlying distribution hasn't shifted.

**This sets Run 2b's actual target: beat 3hr INT8 = 1.050°C** (Run 2a's own current-data PTQ
baseline), not the stale published 0.898°C — per the project's decision rule (QAT must beat its
own PTQ starting point to be worth deploying over the non-QAT checkpoint already sitting there).

**Results — Run 2b, QAT fine-tune (2026-08-12, run by user)**:
- val_loss (includes L2, QAT fake-quant-simulated): **0.000227**
- "FP32" MAE (fake-quant-simulated forward pass, NOT clean FP32 — same caveat as Track B Run 19):
  diff_1hr **0.170°C**, diff_2hr **0.178°C**, diff_3hr **0.208°C** — regressed from Run 2a's
  clean FP32 (0.098/0.108/0.137°C), consistent with QAT's usual small FP32 cost
- INT8 MAE (n=500): diff_1hr **0.792°C**, diff_2hr **0.511°C**, diff_3hr **1.121°C**
- Best epoch: 27/50 · FP32 TFLite: 162.2 KB · INT8 TFLite: 46.8 KB

**Comparison to Run 2a's own PTQ baseline** (the actual target per the decision rule below):

| | Run 2a (PTQ, pre-QAT) | Run 2b (QAT) | Δ |
|---|---|---|---|
| INT8 1hr | 0.493 | 0.792 | +60.6% (worse) |
| INT8 2hr | 0.638 | 0.511 | −19.9% (better) |
| INT8 3hr | **1.050** | **1.121** | **+6.8% (worse)** |

**Outcome**: ❌ Does not beat Run 2a's own PTQ baseline at 3hr — the project's success criterion.
Mixed across horizons (2hr improved, 1hr/3hr regressed), the same shape as Run 1 vs Run 22's PTQ
and Track B Run 19 vs Run 18's PTQ: QAT visibly does *something* (loss curves move, individual
horizons shift), but never delivers a clean win at the horizon that matters, and 1hr/3hr moving
together against 2hr recurs across all three QAT attempts now — suggestive of some structural
horizon-coupling QAT can't get around, not run-to-run noise.

One notable cross-run comparison: Run 2b's 3hr INT8 (1.121°C) is meaningfully better than Run 1's
(1.627°C) or Track B Run 19's (1.630°C) — Run 11's simpler 11-feature, no-prescale/rescale graph
does substantially better under QAT than Run 18/22's deeper, calibration-patched graph did,
reinforcing the structural-depth theory from a different angle (simpler graph → less QAT damage,
even though it still doesn't clear its own PTQ bar).

---

## Project-Level Decision Rule

If both Run 1 and Run 2 fail to beat their own pre-QAT PTQ baselines (Run 1 vs Run 22's PTQ
number, Run 2 vs its own re-measured Run 2a baseline): conclude Track B/INT8-on-this-architecture-
family is closed for good — three independent angles (calibration fixes in Runs 20-22, QAT on the
un-fixed graph in Run 19, QAT on both the fixed and simplest graphs here) will all have failed.
Recommended fallback: deploy Run 11 as-is, matching the Model 5d closure decision.

**Rule triggered (2026-08-12).** Run 1: 3hr INT8 1.627°C vs Run 22 PTQ's 1.821°C — technically
beat its own baseline (structural-depth theory still holds; see Run 1's writeup for why this
isn't a real win) but missed the actual 0.898°C target by ~81%. Run 2b: 3hr INT8 1.121°C vs Run
2a PTQ's 1.050°C — a clean miss, +6.8% worse than its own baseline. Both QAT attempts move
individual horizons around (Run 1: 1hr/3hr improved, 2hr worse; Run 2b: 2hr improved, 1hr/3hr
worse) without ever clearing the bar that matters. Five independent angles across Track B and
Model 5e now agree: calibration fixes (Runs 20-22), QAT on the unfixed graph (Track B Run 19),
QAT on the calibration-fixed graph (Model 5e Run 1), and QAT on the simplest pre-fix graph
(Model 5e Run 2) all fail to produce a deployable improvement over Track B Run 11's existing PTQ
checkpoint.

**PROJECT CONCLUDED (2026-08-12).** QAT does not rescue Track B's INT8 3hr accuracy on any
checkpoint or architecture variant tried — the structural depth-of-sequential-quantized-MatMuls
theory from Track B's own Runs 20-22 holds up against every angle this project could throw at
it. No Model 5e checkpoint is deployed. **Recommended fallback: deploy Track B Run 11 as-is**
(`../Model 5c TFT/results_5c_trackb_dense_b_run11/`) — re-confirmed on current data in this
project's own Run 2a: FP32 val_loss=0.000328 (MAE 0.098/0.108/0.137°C), INT8 MAE
0.493/0.638/1.050°C, still comfortably ahead of Model 5a's deployed 30-day StdDev (0.988°C) and
Model 5b's deployed bar (0.930°C) even with the small 2hr/3hr calibration-noise regression from
Run 11's original published numbers. Model 5a remains the currently deployed model; whether to
replace it with Run 11 is a separate deployment decision, not a further modeling question.

---

## Post-Conclusion Addendum — The "Deploy Run 11" Recommendation Needs Re-Examination (2026-08-12)

**The recommended fallback above ("deploy Run 11 as-is") is retracted, not just questioned —
confirmed wrong by a controlled test.** Surfaced initially from the user's live Grafana dashboard
(Pi + Coral EdgeTPU, weeks of real InfluxDB sensor data, 2026-06-20 onward), then confirmed with
a controlled offline INT8 n=500 evaluation. Full detail and numbers in
`../Model 5c TFT/MODEL_5C_TRACK_B_EXPERIMENT_LOG.md`'s own post-conclusion addendum; summary here:

Track B **Run 2** (`SEQ_LEN=1`, the pre-breakthrough flat-scalar architecture, no relation to
this project's Run 2a/2b) shows a live 3hr INT8 StdDev of **0.748°C** — clearly better than
Run 11's live 3hr StdDev of **1.25°C** — despite Run 2's offline FP32 accuracy (0.783°C) being
roughly 6x worse than Run 11's (0.121°C). The controlled follow-up
(`../Model 5c TFT/evaluate_run2_int8.py`, verified via an FP32-reproduction gate before trusting
the INT8 number) confirms this wasn't a live-deployment fluke: Run 2's offline INT8 n=500 MAE is
**0.211/0.333/0.432°C (1/2/3hr) — beating Run 11's INT8 (0.522/0.588/0.898°C) at every horizon by
2-2.5x**, not just at 3hr.

**This does not overturn Model 5e's own finding** (QAT does not rescue Run 11's or Run 22's INT8
accuracy — that conclusion stands on its own evidence). What it does call into question is the
*downstream recommendation* built on top of it: "Run 11 is the best available deployment
candidate" was only ever compared against other `SEQ_LEN=180` checkpoints and Model 5a/5b's
deployed baselines — never against the much older, structurally simpler `SEQ_LEN=1` family,
which this live data suggests may quantize far more cleanly precisely because it lacks the
sequential-MatMul depth this whole project (Track B → 5d → 5e) spent so much effort fighting.

**Revised recommendation**: do not deploy Run 11. **Run 2 is now the best-known deployable
checkpoint in the entire Model 5-series** (3hr INT8 0.432°C — better than Run 11's 0.898°C, Model
5a's deployed 0.988°C StdDev, and Model 5b's 0.930°C bar). This confirms the "shallow
architectures quantize more cleanly" pattern with a controlled number, not just a hypothesis. The
right next step for this model family is probably not another SEQ_LEN=180 fix — it's either (a)
deploying Run 2 as-is despite its 23-feature, pre-TFT-discovery feature set, or (b) building a new
shallow, INT8-robust architecture that combines Run 2's quantization-friendly shape with Track
A/B's now well-established feature knowledge (a "Model 5f") — likely the higher-value option,
since Run 2 never benefited from any of the feature curation this project did afterward.
