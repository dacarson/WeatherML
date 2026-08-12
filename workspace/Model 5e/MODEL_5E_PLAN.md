# Model 5e: QAT Retry on Untested Track B Checkpoints

## Project Goal

Determine whether Quantization-Aware Training (QAT) can close Track B's INT8 gap when applied
to the two checkpoint/architecture combinations it was never actually tried against, before
concluding INT8 deployment on Track B's architecture family is a dead end.

**Predecessor**: Model 5c Track B (concluded 2026-07-22, 22 experiments). Model 5d then tried
dropping the raw sequence entirely (concluded 2026-08-11, premise didn't hold — see
`../Model 5d/MODEL_5D_PLAN.md`). Model 5a remains the deployed model throughout.

---

## Why This Project (and why it's narrower than it looks)

Track B already tried QAT once, as Run 19: `tfmot.quantization.keras.quantize_model()`,
fine-tuned from **Run 18's** checkpoint (13 features, pre-concat-rescale architecture). It
failed — INT8 MAE came back essentially identical to plain post-training quantization
(0.608/1.066/1.630°C vs Run 18 PTQ's 0.595/1.041/1.658°C at 1/2/3hr). Runs 20-22 then each
diagnosed and cleanly fixed a real, confirmed INT8 precision-loss mechanism (concat
forced-shared-scale, long-tail deep_out calibration) without ever recovering 3hr INT8 accuracy.
Track B's own conclusion: the bottleneck is likely the sheer depth of sequential quantized
MatMuls (4-5 layers compounding rounding error), not any single fixable tensor or weight
distribution — see `../Model 5c TFT/MODEL_5C_TRACK_B_EXPERIMENT_LOG.md` Runs 19-22.

That conclusion means a plain repeat of Run 19 is not expected to work. What's actually
untested:

1. **QAT was never run against Run 11's checkpoint** — the *best deployable* Track B model
   (3hr INT8 0.898°C, still the project's best result) and the original Run 15 plan's intended
   target, before it got retargeted to Run 18 pre-execution. Run 11 predates the 13-feature set
   and predates the concat-rescale fixes; it's a materially different (simpler) graph than what
   Run 19 actually tested QAT against.
2. **QAT was never run against Run 22's checkpoint** — the architecture *after* all three
   calibration fixes (concat rescale, deep_out prescale/relu6/rescale) were cleanly verified.
   If QAT's fine-tuning can do anything useful, doing it on top of a graph that's already been
   calibration-optimized (rather than Run 18's un-fixed graph) is the more favorable case.

Both are genuinely new data points. Neither is guaranteed to beat Run 19 — the structural
depth-of-MatMuls diagnosis applies equally to both — but "QAT failed on the one checkpoint we
tried it on" is weaker evidence than "QAT failed on the best checkpoint and the most-fixed
checkpoint," and costs two short fine-tuning runs (20-50 epochs each, not a fresh 300-epoch
train) to find out.

**If both fail**: treat Track B/INT8-on-this-architecture-family as conclusively closed and
fall back to deploying Run 11 as-is (3hr INT8 0.898°C already beats Track B's own deployment
target), same as the Model 5d closure decision already flagged.

---

## Two Planned Experiments

### Run 1 — QAT from Run 22 Checkpoint (calibration-fixed architecture)

- **Source**: `../Model 5c TFT/results_5c_trackb_dense_b_run22/checkpoints/best_model.weights.h5`
- **Architecture**: unchanged from Run 22 — 13 features, `Input(180,13) → AvgPool(6) →
  flat(390) → Bottleneck(64,relu6) → [Wide(16,relu6) + Deep(128→relu6→64→relu6→32→linear→
  prescale(10)→relu6→rescale(0.2735))] → Merge(48) → 3 heads`.
- **Why start here**: zero architecture changes needed — `train_model_track_b.py`'s current
  state already matches Run 22 exactly, so this is a direct config retarget (`SOURCE_CHECKPOINT`
  + `QAT_FINE_TUNE=True`), same low-risk shape as Run 19.
- **Script**: `train_model_5e.py` (copied from `train_model_track_b.py` as of Run 22, QAT
  flags flipped on, `SOURCE_CHECKPOINT` retargeted).

### Run 2 — QAT from Run 11 Checkpoint (best deployable, pre-fix architecture)

- **Source**: `../Model 5c TFT/results_5c_trackb_dense_b_run11/checkpoints/best_model.weights.h5`
- **Architecture**: reverted to Run 11's actual shape (11 features, no `temp_diff_vs_5hr/6hr`)
  — see "Architecture reconstruction" below for how this was determined; it is NOT simply
  "everything unfused" as originally assumed here.
- **Why this matters**: this is the original Run 15 QAT plan's actual intended target,
  never executed — Run 19 substituted Run 18 instead. It's also QAT applied to the model that's
  actually the best deployment candidate today, so a positive result here is directly useful,
  not just diagnostic.
- **Script**: `train_model_5e_run2.py`, built and verified — see experiment log for the two-stage
  (2a/2b) approach this now requires.

**Architecture reconstruction (2026-08-12)**: The original assumption above — "separate
`Dense(n) → Activation('relu6')` throughout, `by_name=True` loading" — was wrong on both counts,
discovered while building the script:
1. Keras 3's native `.weights.h5` format has no `by_name` load path at all (confirmed by an
   existing comment in `train_model_track_b.py`'s WARM_START block, from an earlier Track B
   run that hit the same issue); it keys H5 groups by internal topological traversal order, not
   layer `.name`.
2. Run 11 was NOT uniformly unfused. Checking out `git show
   301985f:"workspace/Model 5c TFT/train_model_track_b.py"` (committed 2026-06-25, the same day
   as Run 11, mid-flight at Run 14) showed only bottleneck/wide/deep1 were separate
   Dense+Activation in Run 11 — deep2 and deep_out were **already fused**
   (`Dense(n, activation="relu6")`) by that point, and `wide` was created immediately after
   `bottleneck`, before `deep1`, in Python source order.

Rebuilding with this exact mix was verified — via a save-weights round-trip test — to reproduce
Run 11's checkpoint's H5 group order byte-for-byte, and two independent loading methods (Keras's
own `load_weights()` and a manual shape-based loader keyed on each Dense layer's unique kernel
shape) converge on identical evaluation results. The architecture in `train_model_5e_run2.py` is
confirmed correct, not a best-effort guess.

**Data pipeline drift — investigated, then RETRACTED (2026-08-12)**: an early scratchpad
verification (standalone script, not `train_model_5e_run2.py` itself) suggested Run 11's
checkpoint performed far worse on current data (~0.7-1.5°C MAE) than its published numbers. This
turned out to be wrong — a bug in that ad-hoc script (most likely its improvised, non-gap-aware
sequence windowing), not a real finding. **The actual `train_model_5e_run2.py` Run 2a execution
(2026-08-12) shows val_loss=0.000328, MAE 0.098/0.108/0.137°C (1/2/3hr) — closely matching Run
11's original published 0.091/0.092/0.121°C.** Run 11's checkpoint generalizes to current data
just fine; there is no meaningful drift. Left in place: loading Run 11's own saved input/target
scalers (rather than recomputing from today's `train_df`) is still the more correct choice
methodologically, even though it turns out not to matter much in practice here, and the
weight-verification threshold (raised from 0.001 to 0.02) is harmless since real val_loss clears
the original threshold too — but the "0.0096 is a known legitimate baseline" reasoning attached
to that threshold was based on the retracted finding and should be read as historical, not
current.

**Approach**: `train_model_5e_run2.py` runs in two stages via the `QAT_FINE_TUNE` toggle:
- **Run 2a** (`QAT_FINE_TUNE=False`, default): load Run 11's checkpoint, evaluate FP32 + export
  INT8 on today's data. Confirmed to closely reproduce Run 11's original numbers (see above) —
  this stage is now mainly a sanity re-confirmation plus a fresh INT8 export, not a materially
  different baseline.
- **Run 2b** (`QAT_FINE_TUNE=True`): QAT fine-tune from the same Run 11 checkpoint, compared
  against Run 2a's baseline (which is now expected to closely track the original 0.898°C 3hr
  INT8 bar rather than diverge from it).

**Bug found and fixed while running Run 2a (2026-08-12)**: the FP32 export step (mixed-precision
branch) and the QAT-clone weight-copy step both used *positional* `model.get_weights()` →
`other_model.set_weights(list)`. Those helper models fuse bottleneck/wide/deep1
(`activation="relu6"`/`"relu"` in the Dense constructor) while `model` keeps them unfused
(separate Dense+Activation, Run 11's real structure) — different internal layer traversal order,
so positional copying silently paired the wrong shapes (hit directly: "weight shape (128, 64) is
not compatible with provided weight shape (64, 16)"). Fixed both to use name-based transfer
(`get_layer(name).get_weights()` → `set_weights()`), which is correct regardless of either
model's internal ordering. The QAT-clone instance was caught proactively before Run 2b hit it.

---

## Targets

Unchanged from Track B: val_loss < 0.000373, INT8 3hr MAE < 0.898°C (beat Track B's own best,
Run 11 — i.e., QAT needs to beat the checkpoint it started from to be worth deploying over the
plain PTQ version already sitting in `results_5c_trackb_dense_b_run11/` /
`results_5c_trackb_dense_b_run22/`).

## Success Criteria

- Either run's QAT INT8 MAE at 3hr meaningfully beats its own pre-QAT PTQ baseline (Run 22:
  1.821°C; Run 11: re-measured on current data at 1.050°C, see Run 2a) — not just "close to
  FP32," but an actual improvement over the non-QAT deployable checkpoint, since that's the only
  reason to prefer a QAT model over what's already deployable.
- If neither run improves over its own PTQ baseline: conclude Track B/INT8 is closed, no
  further QAT or calibration work planned, recommend deploying Run 11 as-is.

**Outcome (2026-08-12): neither run cleared its own bar.** Run 1: 3hr INT8 1.627°C, technically
below Run 22's 1.821°C PTQ but still missing the real 0.898°C target by ~81%. Run 2b: 3hr INT8
1.121°C vs Run 2a's own 1.050°C PTQ baseline — a clean miss (+6.8%). **Project concluded — see
`MODEL_5E_EXPERIMENT_LOG.md` "Project-Level Decision Rule" for the full closure writeup.
Recommended fallback: deploy Track B Run 11 as-is.**

## Open Questions For Run 1+

- Does QAT on the *calibration-fixed* Run 22 graph behave differently from Run 19's QAT on the
  *un-fixed* Run 18 graph, or does the structural depth argument dominate regardless of prior
  calibration state?
- Is Run 11's simpler (11-feature, no-prescale) graph inherently more QAT-friendly than Run 18/22's
  13-feature, prescale/rescale graph — fewer quantized ops for fake-quant nodes to compensate for?
- Should `QAT_LR`/`QAT_EPOCHS` be revisited from Run 19's settings (1e-6, 50 epochs, patience 10),
  or is that config itself validated and reusable as-is?
