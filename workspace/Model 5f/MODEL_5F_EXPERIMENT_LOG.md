# Model 5f — Experiment Log

**Target**: INT8 3hr MAE ≤ 0.432°C (Track B Run 2's confirmed best-known result), via the same
n=500 offline methodology used throughout Track B / Model 5d / Model 5e for a fair comparison.
See `MODEL_5F_PLAN.md` for full rationale and the open design-variable table (features,
BatchNorm, activation, width) that Run 2 vs. Model 5d Run 3 left unresolved.

---

## Run 1 — Run 2's Architecture + Track A/B's Curated Features (planned)

**Date**: TBD
**Hypothesis**: Isolate the features variable while holding Run 2's proven-INT8-friendly
architecture fixed (BatchNorm after every layer, plain `relu`, 512→256→128→64, `use_bias=False`).
If curated features (Model 5d Run 3's 20-feature set: `temp_slope_15/30/60`,
`temp_diff_vs_5hr/6hr`, drops raw `relative_humidity`/`station_pressure` etc.) improve on Run 2's
0.432°C 3hr INT8, features were the missing piece and BatchNorm/relu/width weren't the
bottleneck differentiating Run 2 from Run 3. If not, follow-up runs will isolate BatchNorm and
activation choice individually rather than assuming.

**Script**: `train_model_5f_run1.py`, forked from `../Model 5d/train_model_5d.py` (data pipeline,
feature engineering, TFLite export/eval infrastructure reused unchanged — verified working via
`evaluate_run3_int8.py`'s FP32 reproduction check). Compiles; not yet executed.

**Configuration**:
1. `RUN_NAME = "5f_run1"`, fresh training (no `SKIP_TRAINING`/warm-start — Model 5d's script
   family doesn't have that mechanism, and this is a genuinely new architecture with no prior
   checkpoint to start from).
2. **Architecture** — Track B Run 2's exact stack, confirmed via checkpoint H5 inspection:
   `Input(20) → [Dense(512,use_bias=False)→BatchNorm→relu] → [Dense(256)→BN→relu] →
   [Dense(128)→BN→relu] → [Dense(64)→BN→relu] → 3× Dense(1,linear,use_bias=False)`. No wide/deep
   branching, no concat — a pure sequential chain, unlike every other architecture in this
   project family since Track B Run 6.
3. **Features** — Model 5d Run 3's 20-feature set (all four time-of-day harmonics restored,
   `temp_slope_15/30/60`, `temp_diff_vs_5hr/6hr`, `relative_humidity`, `humidity_slope_30`,
   `station_pressure`, `pressure_slope_60`, solar/illuminance/uv) — NOT Model 5d Run 5's
   16-feature diurnal-ablated set, and NOT Track B Run 2's original 23 features.
4. **Training recipe** — Track B Run 2's own hyperparams (`L2_REG=1e-5`, `MAX_EPOCHS=300`,
   `INITIAL_LR=1e-4`, `TRAIN_BATCH_SIZE=2048`), not Model 5d's, to avoid introducing a second
   uncontrolled change alongside the features swap.
5. Two latent positional-weight-copy bugs (same class fixed twice already this session for Track
   B Run 1 and Model 5e Run 2 — `model.get_weights()` → `other.set_weights(list)` between
   differently-structured models) were pre-empted here: the mixed-precision FP32 export rebuild
   uses name-based transfer from the start, even though this architecture's linear-chain
   structure makes positional copying theoretically safe this time too.

**Expected outcomes**: if Run 1's INT8 3hr MAE beats Run 2's 0.432°C, curated features were the
(or a major) missing piece, and BatchNorm/relu/width weren't differentiating Run 2 from Run 3 —
project can move toward finalizing a deployment candidate. If Run 1 doesn't clearly beat 0.432°C,
follow-up runs should isolate BatchNorm-vs-none and `relu`-vs-`relu6` individually on top of this
same curated feature set, rather than assuming which variable matters.

**Results (2026-08-12, run by user)**:
- FP32: val_loss=0.012822, MAE 0.472/0.687/0.832°C (1/2/3hr) — worse than BOTH Track B Run 2
  (0.422/0.624/0.783°C) and Model 5d Run 3 (0.429/0.634/0.790°C) at every horizon.
- INT8 (n=500): 0.293/0.523/**0.570**°C — better than Model 5d Run 3 (0.634°C at 3hr) but worse
  than Track B Run 2 (0.432°C at 3hr).

**Correction to this run's own header/plan framing**: described the feature swap as "curated vs
uncurated" (implying Track A/B dropped harmful raw features like `relative_humidity`/
`station_pressure`). Checked both saved feature lists directly — that's wrong. 18 of 20-23
features are IDENTICAL between Run 2's and Run 3's sets, including `relative_humidity`,
`station_pressure`, `humidity_slope_30`, `pressure_slope_60`. The actual difference is narrow:
Run 2 has `temp_diff_vs_1hr/2hr/3hr` + `pressure_lag120/180` (5 features) that this run's set
lacks; this run's set has `temp_diff_vs_5hr/6hr` (2 features) instead.

**Reading the result with the corrected framing**: two separate effects are now visible cleanly
across the three data points (Run 2, Run 3, Run 1):
- **Architecture helps**: holding features fixed at Run 3's set, swapping in Run 2's architecture
  improved INT8 3hr from 0.634°C (Run 3) to 0.570°C (Run 1) — BatchNorm+relu+wide genuinely beats
  relu6+no-BN+narrow.
- **This specific feature swap hurts**: holding architecture fixed at Run 2's, swapping Run 2's
  own `temp_diff_vs_1/2/3hr` + `pressure_lag120/180` for `temp_diff_vs_5hr/6hr` made INT8 worse
  (0.432°C → 0.570°C), not better. Dropping the near-term diffs/pressure lags cost more than the
  5hr/6hr addition gained.

**Decision for Run 2**: test a union feature set (Run 2's original 23 + `temp_diff_vs_5hr/6hr` =
25 features) through Run 2's architecture — checks whether the 5hr/6hr signal adds on top of the
near-term diffs/pressure lags rather than needing to replace them.

---

## Run 2 — Union Feature Set (Run 2's 23 + temp_diff_vs_5hr/6hr) Through Run 2's Architecture (planned)

**Date**: TBD
**Hypothesis**: Run 1 showed that dropping `temp_diff_vs_1/2/3hr` + `pressure_lag120/180` in
favor of `temp_diff_vs_5hr/6hr` cost more than it gained (INT8 3hr 0.432°C → 0.570°C). This run
tests whether `temp_diff_vs_5hr/6hr` has any positive value *additively* — added on top of Run
2's full original feature set rather than substituted in — rather than concluding it's
categorically unhelpful for this architecture. Architecture held fixed at Run 2's (BatchNorm,
relu, 512→256→128→64) throughout, so `n_features` changes from 23→25 but nothing else does.

**Configuration**: `train_model_5f_run2.py`, forked from `train_model_5f_run1.py`. Feature list =
Run 2's original 23 (`time_of_day_*`, `temperature`, `temp_slope_15/30/60`,
`temp_diff_vs_1hr/2hr/3hr`, `relative_humidity`, `humidity_slope_30`, `station_pressure`,
`pressure_slope_60`, `pressure_lag120/180`, `solar_radiation`, `solar_slope_30`, `illuminance`,
`uv`) plus `temp_diff_vs_5hr/6hr` = 25 total. Lag-feature computation extended to compute both
the 1/2/3hr (via `_add_backward_lag`-style merge_asof, same technique as
`evaluate_run2_int8.py`) and 5/6hr (existing `_add_past_lags`) lags in the same pass, plus
`pressure_lag120/180` as raw (non-diffed) features.

**Expected outcomes**: if the union beats Run 2's own 0.432°C, `temp_diff_vs_5hr/6hr` carries
real incremental signal once the near-term features are still present, and this becomes the new
best-known checkpoint. If it lands close to or worse than Run 2's 0.432°C (with FP32 possibly
similar to Run 2's 0.783°C, given `temp_diff_vs_1/2/3hr` already showed near-zero permutation
importance in Run 2's own model), `temp_diff_vs_5hr/6hr` is genuinely not adding value for this
architecture/task, distinct from Model 5d's Run 3 conclusion where it was Track B's most
important feature — a real architecture-dependence finding either way.

**Script**: `train_model_5f_run2.py`, forked from `train_model_5f_run1.py`. Architecture
unchanged; `_add_past_lags` extended to compute the union set (temp_diff_vs_1/2/3hr via the same
`_add_backward_lag` technique already verified in `evaluate_run2_int8.py`, plus the existing
temp_diff_vs_5hr/6hr, plus pressure_lag120/180 as raw features). Feature list verified to total
25 before compiling. Compiles; not yet executed.

**Results (2026-08-12, run by user)**:
- FP32: val_loss=0.016360 (val_mae normalized 0.040575), MAE 0.475/0.705/0.919°C (1/2/3hr) — worse
  than Track B Run 2 (0.422/0.624/0.783°C) AND worse than this project's own Run 1
  (0.472/0.687/0.832°C) at every horizon, despite Run 1 having *dropped* the near-term
  `temp_diff_vs_1/2/3hr`/`pressure_lag120/180` features this run restores.
- INT8 (n=500): 0.350/0.590/**0.672**°C — worse than Track B Run 2 (0.432°C at 3hr) AND worse than
  Run 1 (0.570°C at 3hr). Best epoch 9 (early relative to Run 1/Track B Run 2, worth noting).
- FP32: 729.3 KB, INT8: 210.3 KB.

**Reading the result**: the union set (25 features = Run 2's 23 + `temp_diff_vs_5hr/6hr`) is worse
than *both* inputs it was combined from, not just worse than the better one. This rules out the
"additive, no harm" framing in the hypothesis — adding `temp_diff_vs_5hr/6hr` on top of the
near-term diffs measurably hurt, it didn't just fail to help. Combined with Run 1 (near-term
features swapped OUT for 5hr/6hr → also worse than Track B Run 2), `temp_diff_vs_5hr/6hr` looks
actively harmful for this architecture in both a substitutive and additive role — the opposite of
its role in Model 5d Run 3, where it was the most important feature. Track B Run 2's original
23-feature set (0.432°C, unmodified) remains the best-known checkpoint in this family; neither
Run 1 nor Run 2 has beaten it. Best epoch 9 is also unusually early relative to other runs and
may be worth a quick check (early stopping / LR schedule interaction) before drawing a firm
architecture conclusion from this run alone.

**Open question for follow-up**: with the features axis now twice unhelpful, the remaining
untested variables from the original design table (BatchNorm on/off, `relu` vs `relu6`, width)
are back in focus as originally planned in `MODEL_5F_PLAN.md`, rather than continuing to vary
features.

---

## Run 3 — Narrowest Isolation: Swap ONLY temp_diff_vs_1/2/3hr for temp_diff_vs_5hr/6hr (planned)

**Date**: TBD
**Hypothesis**: Run 1 and Run 2 both changed multiple features at once relative to Run 2's own
0.432°C baseline — Run 1 dropped `temp_diff_vs_1/2/3hr` *and* `pressure_lag120/180` together (5
features out, 2 in); Run 2 added `temp_diff_vs_5hr/6hr` on top of an otherwise-untouched 23,
making the set larger and more redundant rather than isolating anything. Neither run isolated the
near-vs-far `temp_diff` variable on its own. This run does: keep Run 2's 23 features exactly as
they are — including `pressure_lag120/180`, `temp_slope_15/30/60`, and raw `temperature` — and
swap out only `temp_diff_vs_1hr/2hr/3hr` for `temp_diff_vs_5hr/6hr` (22 features total, one
variable changed). If this beats 0.432°C, the far-horizon diffs are a strict improvement over the
near-horizon ones in isolation, and Run 1/Run 2's worse results trace to their *other*
simultaneous changes (pressure_lag drop in Run 1; redundant feature-count growth in Run 2), not to
`temp_diff_vs_5hr/6hr` itself. If it still doesn't beat 0.432°C, `temp_diff_vs_5hr/6hr` looks
unhelpful for this architecture regardless of how it's combined with the rest of Run 2's set —
closing out the features axis more conclusively than Run 1/Run 2 did individually.

**Configuration**: `train_model_5f_run3.py`, forked from `train_model_5f_run2.py`. Architecture,
training recipe, and data pipeline unchanged. Feature list = Run 2's original 23 minus
`temp_diff_vs_1hr/2hr/3hr`, plus `temp_diff_vs_5hr/6hr` = 22 total (`temp_diff_vs_1/2/3hr` are
still computed by `_add_past_lags` but simply excluded from the `features` list, so no pipeline
changes needed beyond the feature list itself).

**Script**: `train_model_5f_run3.py`. Compiles; not yet executed.

**Results (2026-08-12, run by user)**:
- FP32: val_loss=0.014740 (val_mae normalized 0.040700), MAE 0.501/0.719/0.887°C (1/2/3hr) —
  worse than Track B Run 2's baseline (0.422/0.624/0.783°C) and worse than Run 1
  (0.472/0.687/0.832°C), but better than Run 2's union set (0.475/0.705/0.919°C) at every horizon.
- INT8 (n=500): 0.266/0.448/**0.643**°C — worse than Track B Run 2's baseline (0.432°C at 3hr) and
  worse than Run 1 (0.570°C), but better than Run 2's union set (0.672°C). Best epoch 18 (not
  early like Run 2's epoch 9 — normal convergence this time).
- FP32: 723.3 KB, INT8: 208.8 KB.

**Reading the result**: this closes out the features axis more conclusively than Run 1/Run 2
individually managed to. With every other feature held fixed at Run 2's own set — including
`pressure_lag120/180`, which Run 1 had confounded by dropping simultaneously — swapping only
`temp_diff_vs_1/2/3hr` for `temp_diff_vs_5hr/6hr` still made INT8 3hr worse (0.432°C → 0.643°C).
The near-term diffs are not an artifact that happened to correlate with `pressure_lag120/180` or
some other co-dropped feature in Run 1; they're independently more valuable than the far-term
diffs for this specific architecture (BatchNorm, `relu`, 512→256→128→64, `SEQ_LEN=1`). This is a
clean contradiction of Model 5d Run 3, where `temp_diff_vs_5hr/6hr` was reported as the *most*
important feature — the same feature is helpful for one architecture and harmful for another,
a real architecture-dependence finding, not a data or methodology artifact.

**Decision**: the features axis is now closed for this project — three runs (1, 2, 3), three ways
of introducing `temp_diff_vs_5hr/6hr` (substitution bundled with other drops, pure addition, and
isolated substitution), three losses relative to Track B Run 2's untouched 23-feature baseline
(0.432°C). No further feature-set experiments are planned. Per `MODEL_5F_PLAN.md`'s original
design table, the next axis to isolate is architecture: BatchNorm on/off, `relu` vs `relu6`, and
width, holding Run 2's original 23 features fixed throughout.

---

## Run 4 — Architecture Axis Begins: Remove BatchNorm (planned)

**Date**: TBD
**Hypothesis**: Features are ruled out as the driver of Run 2 vs. Model 5d Run 3's INT8 gap (Run
1/2/3, closed above). Architecture is the remaining candidate — Run 2 (BatchNorm, `relu`, wide
512→256→128→64) and Run 3 (no BatchNorm, `relu6`, narrow 128→64→32) differ on three architecture
variables simultaneously, and no experiment has isolated which one(s) matter. BatchNorm is the
most notable: Track B's *deep* `SEQ_LEN=180` architecture explicitly **removed** BatchNorm as an
INT8 fix (Run 11 — "BN γ unconstrained by L2, produces wide pre-clip activation range"). Whether
that finding reverses for this shallow `SEQ_LEN=1` architecture, or BatchNorm doesn't matter much
here either way, is unknown and worth testing directly rather than assuming either direction.

**Configuration**: `train_model_5f_run4.py`, forked from `train_model_5f_run2.py`. Feature set
reverted to Track B Run 2's original 23 (the confirmed-best set — features axis is closed, so no
reason to build on top of a worse set). Architecture: Run 2's exact stack (`relu`, `use_bias=False`,
512→256→128→64) with the four `BatchNormalization` layers removed — nothing else changes.
`use_bias` stays `False` (not switched to `True`) so this isolates BatchNorm specifically, rather
than conflating its removal with reintroducing bias terms; this also matches Model 5d Run 3's own
convention of no-bias/no-BN Dense layers, keeping the comparison apples-to-apples. Training recipe
(L2, LR schedule, batch size) unchanged from Run 2.

**Expected outcomes**: if INT8 3hr MAE stays close to Run 2's 0.432°C, BatchNorm isn't what drives
Run 2's edge over Run 3, and the gap must come from activation and/or width instead — next run
should isolate `relu` vs `relu6`. If it degrades sharply toward Run 3's 0.634°C or worse,
BatchNorm is confirmed as a major factor — and notably the *opposite* factor from what Track B's
deep architecture found, a genuine architecture-depth-dependent reversal rather than a universal
INT8 rule.

**Script**: `train_model_5f_run4.py`. Compiles; not yet executed.

**Results (2026-08-13, run by user)**:
- FP32: val_loss=0.012097 (val_mae normalized 0.036066), MAE 0.430/0.636/0.800°C (1/2/3hr) —
  essentially tied with Track B Run 2's FP32 (0.422/0.624/0.783°C), marginally worse (~2-3%) at
  every horizon.
- INT8 (n=500): 0.218/0.324/**0.396**°C — **beats Track B Run 2's 0.432°C at 3hr** (~8.3%
  improvement), mixed at shorter horizons (1hr: 0.218 vs Run 2's 0.211, slightly worse; 2hr: 0.324
  vs 0.333, better). Best epoch 9. FP32: 720.9 KB, INT8: 193.7 KB (smallest INT8 file of any Model
  5f run so far).

**🏆 New best-known deployable checkpoint in the Model 5-series** — first result across Model 5d,
5e, and 5f to beat Track B Run 2's 0.432°C 3hr INT8 bar.

**Reading the result**: removing BatchNorm from the shallow architecture *improved* INT8
robustness despite FP32 accuracy being essentially unchanged (slightly worse, not better) —
quantization got measurably easier with almost no accuracy trade-off pre-quantization. This
**confirms, rather than reverses**, Track B's deep-architecture finding (Run 11: "BN γ
unconstrained by L2, produces wide pre-clip activation range" hurts INT8) — the direction was the
same in both the deep and shallow architecture after all. What's architecture-dependent is the
*magnitude*: shallow+BN (Run 2, 0.432°C) already dramatically outperformed deep+no-BN (Run 11,
0.898°C), so shallow-vs-deep is still the dominant factor; removing BN from the shallow
architecture adds a further, smaller improvement on top (0.432°C → 0.396°C) rather than being the
main story on its own. The original plan's framing — "this may be a reversal, not an assumption to
build on" — was appropriately cautious but the caution wasn't needed here: same direction, smaller
effect size.

**Decision**: BatchNorm is confirmed as a real (helpful-to-remove) factor. Two variables remain
unexplored in the design table: `relu` vs `relu6`, and width. Next run should test one of those
on top of Run 4's winning no-BatchNorm configuration (Run 2's 23 features, `relu`,
512→256→128→64, BatchNorm removed) rather than reverting to BatchNorm-on as the base. Given the
size of this jump, also worth flagging: prior Model 5-series history (Track B Run 2 itself) showed
that offline n=500 eval rankings don't always survive live deployment — this checkpoint is a
strong offline candidate but not yet validated end-to-end the way Track B Run 2 eventually was.

---

## Run 5 — relu vs relu6, on Top of Run 4's Winning No-BatchNorm Base (planned)

**Date**: TBD
**Hypothesis**: Run 4 (BatchNorm removed) beat Track B Run 2's 0.432°C bar (0.396°C), confirming
BatchNorm hurts INT8 in this architecture family, same direction as Track B's deep-architecture
finding. Two design-table variables remain untested: `relu` vs `relu6`, and width. This run tests
activation, building on Run 4's winning base (not reverting to BatchNorm-on) — swap plain `relu`
for `relu6` (bounded [0,6]) in all four hidden layers, everything else unchanged. `relu6` is the
more common INT8-friendly choice in this project's own history (Track B's deep architecture uses
it throughout, fused into the Dense constructor for a single `FULLY_CONNECTED` op). Two
hypotheses: (a) `relu6` helps further on top of BatchNorm removal — bounding activations reduces
per-tensor INT8 scale stretch from rare large activations, a mechanism independent of BatchNorm;
or (b) `relu6` doesn't matter much (or even hurts) now that BatchNorm is gone, since BN was
arguably the larger source of unbounded pre-clip range in Run 2, and its removal may have already
captured most of the available gain. Either outcome narrows down which specific mechanism drives
INT8 robustness in this architecture family, rather than leaving "BatchNorm and/or relu6" as an
undifferentiated bundle.

**Configuration**: `train_model_5f_run5.py`, forked from `train_model_5f_run4.py`. Feature set and
width unchanged (Run 2's 23 features, 512→256→128→64, `use_bias=False`, BatchNorm removed). Only
change: all four `Activation("relu")` calls (training model and the fp32 export-model rebuild)
swapped to `Activation("relu6")`. Training recipe unchanged.

**Expected outcomes**: if INT8 3hr MAE improves on Run 4's 0.396°C, `relu6`'s bounded range adds
real value independent of BatchNorm, and width becomes the last variable to test. If it's flat or
worse, `relu6` isn't the active ingredient here — BatchNorm removal was the whole story, and width
is the remaining candidate to explain any further gap to a theoretical ceiling.

**Script**: `train_model_5f_run5.py`. Compiles; not yet executed.

**Results (2026-08-13, run by user)**:
- FP32: val_loss=0.011846 (val_mae normalized 0.035948), MAE 0.427/0.636/0.797°C (1/2/3hr) —
  essentially identical to Run 4 (0.430/0.636/0.800°C), negligible change.
- INT8 (n=500): 0.206/0.298/**0.308**°C — **beats Run 4 at every horizon**, and by a wide margin
  at 3hr (0.396°C → 0.308°C, ~22% further improvement; ~29% total vs Track B Run 2's original
  0.432°C). Best epoch 8. FP32: 720.9 KB, INT8: 193.7 KB (same file sizes as Run 4, as expected —
  activation choice doesn't change parameter count).

**🏆 New best-known deployable checkpoint in the Model 5-series (again)** — INT8 3hr MAE 0.308°C,
beating Run 4's 0.396°C and Track B Run 2's original 0.432°C.

**Reading the result**: hypothesis (a) confirmed decisively — `relu6`'s bounded activation range
is a real, independent contributor to INT8 robustness, not redundant with BatchNorm removal.
FP32 barely moved (same pattern as Run 4: the gain is entirely a quantization-robustness effect,
not a better-trained model), but INT8 improved dramatically at every horizon, most at 3hr. The two
fixes stack: BatchNorm removal (Run 2 → Run 4: 0.432°C → 0.396°C) and `relu6` (Run 4 → Run 5:
0.396°C → 0.308°C) are separate, additive mechanisms — both reduce per-tensor INT8 scale stretch
from unbounded/wide pre-clip activation ranges, from different sources (BN's unconstrained γ vs.
plain relu's unbounded upper range), and removing both compounds rather than one subsuming the
other.

**Decision**: width remains the last untested variable in the original design table. Given how
much the first two variables mattered, it's still worth isolating rather than assuming it's
irrelevant — but the marginal upside is unclear given how large a jump `relu6` already delivered.
Track B Run 2's checkpoint is now decisively superseded; Run 5 is the strongest offline candidate
in the Model 5-series to date, still pending live-deployment validation (see Run 4's note above —
same caveat applies, more strongly given the size of the gain).

---

## Run 6 — Width/Depth: Narrow to Model 5d Run 3's 128→64→32 (planned)

**Date**: TBD
**Hypothesis**: BatchNorm (Run 4) and activation (Run 5) are both confirmed real, additive
contributors to INT8 robustness on top of Track B Run 2's baseline. Width is the last untested
variable in the original design table. This run narrows Run 5's winning stack (no BatchNorm,
`relu6`, Run 2's 23 features) from 512→256→128→64 (4 hidden layers) to exactly Model 5d Run 3's
128→64→32 (3 hidden layers) — closing the original three-way comparison cleanly: Run 2
(wide+BN+relu, 0.432°C) vs. Run 3 (narrow+no-BN+relu6, 0.634°C) vs. Run 5 (wide+no-BN+relu6,
0.308°C) vs. this run (narrow+no-BN+relu6). **Note this is not a pure width isolation** — Run 3's
shape also drops a hidden layer (4→3 layers), conflating width and depth. This matches the
original design table's own framing (`MODEL_5F_PLAN.md` listed "512→256→128→64" vs "128→64→32" as
a single "Width" row), so it's a known, accepted imprecision rather than a new one introduced
here — but it means a positive or negative result here won't cleanly separate "fewer parameters"
from "fewer layers" as the cause. If capacity turns out to matter, a follow-up could still isolate
those two conflated changes if warranted.

**Configuration**: `train_model_5f_run6.py`, forked from `train_model_5f_run5.py`. Feature set,
BatchNorm-removed status, `relu6` activation, `use_bias=False`, and training recipe all unchanged
from Run 5. Only change: `HIDDEN_1/2/3/4 = 512/256/128/64` (4 layers) → `HIDDEN_1/2/3 = 128/64/32`
(3 layers) in both the training model and the fp32 export-model rebuild.

**Expected outcomes**: if INT8 3hr MAE stays close to Run 5's 0.308°C, capacity above
~128→64→32 isn't buying anything at this task's information ceiling, and Run 5's win is fully
attributable to BatchNorm removal + `relu6` — Run 3's underperformance was about its BatchNorm/
`relu6` choices, not its narrow width. If it degrades meaningfully toward Run 3's 0.634°C, capacity
matters too, and closes out the design table with all three variables confirmed as real
contributors. Given how large the `relu6` jump already was, this is the least likely of the three
variables to move the needle further, but it's the last one — worth closing out rather than
assuming.

**Script**: `train_model_5f_run6.py`. Compiles.

**Results (2026-08-13, run by user; this entry backfilled later — see note)**: FP32 MAE
0.430/0.635/0.798°C (1/2/3hr), val_loss=0.010197, best_epoch=44 — close to Run 5's FP32
(0.427/0.636/0.797°C), consistent with the established pattern. Original INT8 (n=500):
**1.282/0.308/0.360°C** — 3hr (0.360°C) is worse than Run 5's 0.308°C but still beats Run 4's
0.396°C, and 1hr (1.282°C) was a severe, isolated outlier (2hr/3hr look normal) — exactly the same
symptom later diagnosed in Run 8 (see that entry) and confirmed via re-quantization below to be the
same unseeded-calibration bug, not a real property of the narrower architecture.

**Note on this entry**: originally logged as "planned"/"not yet executed" and never backfilled with
results at the time the run was actually completed — the mismatch was caught while re-quantizing
Run 6 alongside Run 5/7/8. Numbers above are reconstructed from `results_5f_run6/results_5f_run6.json`
and a direct re-evaluation of the original `model_5f_5f_run6_int8.tflite` against the same n=500
eval slice used throughout this log, so they're exact, not estimated.

**Reading the result**: with the 1hr outlier explained away as a calibration artifact, the
underlying conclusion holds: narrowing from 512→256→128→64 to 128→64→32 hurts 3hr accuracy
(0.308°C → 0.360°C, ~17% worse), closing out the width/depth axis of the original design table —
capacity above 128→64→32 does matter, unlike the (initially more surprising) hypothesis that Run 5's
win was fully attributable to BatchNorm removal + `relu6` alone.

**Decision**: Run 6 does not beat Run 5 — narrowing hurts. Confirmed unaffected by the calibration
bug (see "Calibration Fix" entry below): re-quantized 3hr MAE is 0.360°C, identical to the original
number, only 1hr changed (1.282°C → 0.210°C).

---

## Run 7 — Solar-Feature Tail Fix + Diff Features (planned)

**Date**: TBD
**Hypothesis**: Two independent observations, both surfaced while inspecting Run 5's
`input_scaler_5f.json`, motivate this run — bundled together per user request rather than isolated,
so the result won't cleanly separate the two causes if it moves (see Configuration for the known
confound this creates).

1. **Tail-stretched INT8 scale for `illuminance`/`solar_radiation`/`uv`.** These three have
   `domain_bounds` ceiling `None` (see line ~404), so their INT8 scale is set by the raw training-set
   max, not a percentile. Checked against `train_data_sf.csv` (1.487M rows): `illuminance`'s max
   (184,938) is a near-never-visited tail — only 42 rows (0.003%) exceed 150,000, while the 90th
   percentile is 83,714 and the 99th is 117,536. That single tail stretches the INT8 scale so most
   of the 256 available levels are spent on values almost never seen, leaving coarse resolution for
   the typical daytime range. `solar_radiation` (90th pct 689 vs max 1417) and `uv` (90th pct 5.37
   vs max 13.21) show the same pattern. This is the identical failure mode already fixed for
   `temp_diff_vs_5hr/6hr` in Model 5c Track B Run 17/18 (see the `domain_bounds` comment at line
   ~395) — that fix was never extended to these three. Given `illuminance` is Run 5's 2nd-highest
   permutation-importance feature (0.0042, behind only `time_of_day_cos`), wasted INT8 resolution
   here is a plausible contributor to the FP32→INT8 gap this project exists to close. Note
   permutation importance is measured on the FP32 Keras model, so it would not have surfaced this —
   it's an INT8-only effect.
2. **Diff features for the same three variables, requested alongside the bound fix.** Run 5's
   `temp_diff_vs_1/2/3hr` are the only "current-minus-past" style features in the set; `illuminance`,
   `solar_radiation`, and `uv` are fed as raw levels only (solar has `solar_slope_30`, a regression
   slope, but no simple backward diff). Adding 1hr backward diffs for these three tests whether the
   *trend* carries more signal than the *absolute reading*, mirroring the reasoning that already
   motivated `temp_diff_vs_*` for temperature.

**Configuration**: `train_model_5f_run7.py`, forked from `train_model_5f_run5.py` (Run 5's
architecture — no BatchNorm, `relu6`, `use_bias=False`, 512→256→128→64 — is left unchanged; only
the feature/scaling axis moves). Two changes, bundled per user request (known confound — a result
here won't disentangle which of the two mattered without a follow-up ablation):
  - `domain_bounds` ceilings for `illuminance`, `solar_radiation`, `uv` tightened from `None` (raw
    max) to their 99th-percentile values from `train_data_sf.csv`: `illuminance` → 120,000 (99th
    pct 117,536), `solar_radiation` → 970 (99th pct 969), `uv` → 9.5 (99th pct 9.49). Floor stays 0.
    Values above the new ceiling are clipped to 1.0 by the existing per-feature clip, same handling
    already used for `temp_diff_vs_5hr/6hr`.
  - New features `illuminance_diff_1hr`, `solar_radiation_diff_1hr`, `uv_diff_1hr` added via the
    same `_backward_lag` 60-minute merge_asof pattern already used for `temp_diff_vs_1hr` —
    current value minus the value ~60 minutes ago (90s tolerance). Added *alongside* the existing
    absolute features, not replacing them (Run 5's 23 features → 26). These new diff distributions
    were checked for the same tail problem before deciding their bounds: `illuminance_diff_1hr`
    99.9th pct is 94,920 vs max 179,719 (~53% of range used below 99.9th pct) — noticeably less
    skewed than the raw level was, so left on natural min/max+5% padding (no `domain_bounds` entry),
    the same treatment `temp_diff_vs_1/2/3hr` already get. Worth revisiting with explicit tight
    bounds in a follow-up if this run's INT8 numbers suggest otherwise.

**Expected outcomes**: if INT8 3hr MAE improves over Run 5's 0.308°C, at least one of the two
changes helped — permutation importance on the new diff features (low vs. high relative to their
absolute counterparts) will indicate whether the diffs specifically are pulling weight, though it
won't isolate the bound fix from the diff-feature addition without a follow-up run that applies
only one of the two. If flat or worse, either the tail wasn't actually costing accuracy at this
task's precision floor, or the three new features are adding noise the model has to route around
(unlikely to be the deciding factor at 26 features on a wide 512-256-128-64 stack, but worth
checking permutation importance either way).

**Script**: `train_model_5f_run7.py`. Compiles; not yet executed.

**Results (2026-08-13, run by user)**:
- FP32: MAE 0.429/0.629/0.791°C (1/2/3hr), val_loss=0.011898 — essentially identical to Run 5
  (0.427/0.636/0.797°C, val_loss=0.011846). Matches the established pattern: feature/scaling
  changes alone don't move FP32 accuracy.
- INT8 (n=500): 0.302/0.344/**0.381**°C — **worse than Run 5 at every horizon**, most severely at
  1hr (0.206°C → 0.302°C, +47%) and still meaningfully worse at 3hr (0.308°C → 0.381°C, +24%).
  Run 7 does **not** beat Run 5; Run 5 remains the best-known deployable checkpoint.
- Permutation importance (FP32 model): `solar_radiation` 0.0027→0.0151 (~5.6x), `illuminance`
  0.0042→0.0113 (~2.7x), `uv` 0.0013→0.0069 (~5.3x) — all three absolute-level features became
  substantially *more* important after bound tightening, consistent with the tail-waste hypothesis
  (more usable resolution → model leans on them harder). The three new diff features are at the
  noise floor: `illuminance_diff_1hr` 0.0001, `uv_diff_1hr` 0.0001, `solar_radiation_diff_1hr`
  -0.0000 (negative — pure noise). **Directly answers the motivating question: diffs do not carry
  more signal than absolute values for these three variables — the opposite, if anything.**

**Reading the result**: the bundle was net negative for the actual deployment target (INT8), even
though its FP32-importance side effect (bound tightening making the levels more informative) reads
as a partial win in isolation. Because the two changes were bundled, this run can't say which one
caused the INT8 regression — but the diff features carrying ~zero learned weight makes them an
unlikely direct cause; they're inert extra channels, not adding wrong signal, and they share the
same [0,1] post-clip range as every other channel (per-tensor INT8 quantization is bounded to
[0,1] before it ever sees any single feature's raw scale, so extra channels within that range
shouldn't by themselves stretch the shared scale). More plausible: clipping the top ~1% of
solar/illuminance/uv readings to a single saturated value removed exactly the discriminative
information among bright/high-solar samples that the model had been relying on for near-term
(1hr-dominant) prediction — consistent with the INT8 degradation being worst at 1hr and easing by
3hr. Also plausible: this is ordinary run-to-run stochastic variance from a single training run,
which this project's methodology (single seed per run) can't currently distinguish from a real
effect.

**Decision**: Run 5 remains the best-known deployable checkpoint (INT8 0.308°C @ 3hr) — Run 7 is
superseded, not adopted. The diff-feature half of this bundle looks safe to drop (near-zero
permutation importance, no evidence they're worth their added complexity). If the bound-tightening
hypothesis is still worth testing cleanly, a follow-up should isolate it alone (tightened bounds,
no new diff features) against Run 5, rather than bundling further changes.

---

## Run 8 — Isolate the Bound-Tightening Half of Run 7 (planned)

**Date**: TBD
**Hypothesis**: Run 7 bundled two changes (tightened illuminance/solar_radiation/uv INT8 bounds +
three new diff features) and came back worse than Run 5 at every INT8 horizon, even though FP32
was flat and permutation importance showed the tightened-bound features became substantially more
informative (`solar_radiation` 5.6x, `illuminance` 2.7x, `uv` 5.3x). The diff features were at the
noise floor (two near-zero, one negative), making them an unlikely direct cause of the INT8
regression — but Run 7 can't prove that; the two changes were never tested apart. This run removes
the diff features entirely and tests the bound-tightening alone against Run 5, to find out whether
the tightening itself is responsible for the regression, or whether Run 7's result was driven by
the (now-removed) diff features, or was just run-to-run stochastic variance from a single seed.

**Configuration**: `train_model_5f_run8.py`, forked from `train_model_5f_run5.py` (not Run 7) —
Run 5's exact 23 features, unchanged. Only change: `domain_bounds` ceilings for `illuminance`
(`None` → 120,000), `solar_radiation` (`None` → 970), `uv` (`None` → 9.5), identical to the values
Run 7 used, carried over unchanged. Architecture (no BatchNorm, `relu6`, `use_bias=False`,
512→256→128→64), training recipe, and hyperparameters are all identical to Run 5.

**Expected outcomes**: if INT8 3hr MAE comes back close to or better than Run 5's 0.308°C, the
bound tightening itself is fine (or helps) and Run 7's regression was caused by the diff features
or by seed variance — the tightening should be kept and a follow-up can retest with a different
seed to separate those two remaining explanations. If INT8 MAE is still meaningfully worse than
Run 5's 0.308°C (closer to Run 7's 0.381°C), the tightening/clipping itself is the problem — most
likely the saturation of the top ~1% of bright-sky readings destroying discriminative signal the
model relies on for near-term prediction — and the `domain_bounds` ceilings for these three
features should revert to `None` (raw max), leaving the tail-waste question unresolved for a
future run with a less aggressive percentile choice (e.g. 99.9th instead of 99th).

**Script**: `train_model_5f_run8.py`. Compiles; not yet executed.

**Results (2026-08-13, run by user)**:
- FP32: MAE 0.421/0.634/0.797°C (1/2/3hr), val_loss=0.012128 (val_mae normalized 0.035798) — essentially identical to Run 5
  (0.427/0.636/0.797°C). Consistent with the established pattern.
- INT8 (n=500): **2.022/0.280/0.303°C**. 2hr and 3hr both *improved* over Run 5 (0.298→0.280,
  0.308→0.303) — but 1hr is wildly worse (0.206°C → 2.022°C, ~10x), an outlier severe enough to
  investigate rather than accept at face value.

**Root cause investigation**: inspected the exported INT8 `.tflite` files' output-tensor
quantization parameters directly (`scale`/`zero_point` per head) for Run 5, 7, and 8. Run 8's
`diff_1hr` output has `zero_point=127` — pinned at the extreme edge of the int8 range — meaning
the representative-dataset calibration pass never observed a positive `diff_1hr` prediction above
the tensor's max representable value, +2.15°C (vs. Run 5's +3.35°C, Run 7's +2.92°C). Any real
validation sample where temperature genuinely rises >~2°C in the next hour (an ordinary morning
warm-up) gets hard-clipped to +2.15°C at inference — a large, systematic error, sufficient on its
own to explain the MAE blowup. `diff_2hr`/`diff_3hr` show the same narrowing pattern (max
representable value dropped in all three runs) but weren't catastrophic, plausibly because their
larger natural error baseline dilutes a few clipped-outlier samples while `diff_1hr`'s small
baseline (~0.42°C) does not.

**This traces to a pre-existing pipeline bug, not this run's feature change**:
`representative_data_gen()` (script line ~1088) draws an **unseeded** random 2000-row sample from
`X_val` for INT8 calibration, with no stratification guaranteeing coverage of rare-but-important
extremes (fast warming/cooling events). This makes calibration — and therefore every reported INT8
number in the Model 5f series — sensitive to which 2000 rows happen to get drawn, run to run. Run
8 most likely just drew an unlucky sample that missed large positive `diff_1hr` swings almost
entirely; this is not evidence that bound-tightening itself hurts 1hr prediction, and it also casts
some doubt on how much weight to put on small INT8 deltas between prior runs (Run 4/5/6's
comparisons all used this same unseeded calibration).

**Decision**: inconclusive on the bound-tightening hypothesis — Run 8's 1hr result cannot be
trusted as-is. Before drawing any conclusion from Run 8 (or re-litigating Run 7), fix
`representative_data_gen()` to use a fixed seed (for reproducibility) and/or stratify the sample
across the target distribution (e.g., by `diff_1hr` percentile bins) so calibration reliably covers
the full range rather than a random contiguous mass. Then re-run Run 8 (or a fresh Run 9) to get a
trustworthy comparison against Run 5.

---

## Calibration Fix + Retroactive Re-quantization of Run 5/7/8 (planned)

**Date**: TBD
**Why**: Run 8's investigation found the INT8 export pipeline's `representative_data_gen()` (used
identically in Run 5/7/8, and every prior 5f/5d/5c run) draws an unseeded random 2000-row sample
from `X_val` with no stratification, so calibration quality — and therefore every reported INT8
number in this series — depends on random luck in which rows get drawn. Rather than only fixing it
forward, this also re-quantizes Run 5, Run 7, and Run 8 retroactively from their saved
`checkpoints/best_model.weights.h5` (all three exist, confirmed) using one fixed, stratified
calibration set, so their INT8 numbers become directly comparable — no retraining needed, since
only the post-training quantization step changes.

**Approach**:
- New standalone script `requantize_int8.py` (not a training run — an offline re-quantization
  tool). Re-derives `train_df`/`val_df` via the same feature-engineering pipeline used across
  Run 5/7/8 (superset of columns, since Run 7 added the three solar diff features).
- Calibration set: rows stratified by *true* `temp_diff_1hr` decile (bin edges from the train
  distribution, so they're fixed independent of any particular run), capped at ~200 rows/bin, from
  a `numpy.random.default_rng(seed=42)` draw — guarantees calibration sees the full range of real
  1hr swings (including the large-positive tail that broke Run 8), not just whatever a random
  unseeded draw happened to include. The same physical timestamps are used across all three runs;
  each run applies its own saved `features` list and `input_scaler_5f.json` bounds to those rows.
- For each run: rebuild the (identical family) architecture from `results_5f_{run}.json`'s
  `hyperparams`/`n_features`, load `best_model.weights.h5`, rebuild the concrete function, re-run
  TFLite INT8 conversion with the new representative dataset, save as
  `model_5f_{run}_int8_requant.tflite`, and re-evaluate on the same first-500-rows eval slice each
  run's original script used (val set ordering is deterministic/time-sorted, so this slice was
  already comparable across runs — only calibration was the uncontrolled variable).
- The same stratified/seeded representative-dataset logic is also carried into the training-script
  template so future runs (Run 9+) don't reintroduce this bug.

**Results (2026-08-13, run in-session, CPU inference only — no retraining)**:

First attempt stratified the calibration set by the **true** `temp_diff_1hr` target (deciles from
the train distribution). This *failed* — Run 5 got dramatically worse (0.206°C → 1.288°C at 1hr)
instead of staying stable. Root cause: TFLite's calibration range is set by what the model actually
**outputs** for the representative inputs, not by the true label of those inputs. This MSE-trained
family hedges toward smaller-magnitude predictions even for inputs whose true target is extreme, so
stratifying by true target doesn't reliably pull in extreme *predictions* — a real negative result,
kept here so it isn't retried.

Fixed by stratifying on each model's own FP32 predicted values instead (`export_model.predict()`
over the full val set), with the top/bottom 25 most extreme predictions per head force-included
(can't be missed regardless of sampling luck) plus a seeded (42) stratified sample across predicted
`diff_1hr` deciles for realistic mid-range coverage (~2,000–2,100 rows/run). Re-quantized Run 5,
Run 7, and Run 8 from their saved `checkpoints/best_model.weights.h5` — no retraining:

| | 1hr | 2hr | 3hr |
|---|---|---|---|
| Run 5 (re-quantized) | 0.207°C | 0.301°C | 0.310°C |
| Run 7 (re-quantized) | 0.196°C | 0.344°C | 0.382°C |
| Run 8 (re-quantized) | 0.219°C | **0.277°C** | **0.303°C** |

Run 5's re-quantized number (0.207/0.301/0.310°C) lands almost exactly on its original reported
number (0.206/0.298/0.308°C) — strong evidence the new calibration methodology is sound and Run 5's
original result wasn't itself a lucky calibration draw. Run 8's 1hr number is now sane (2.022°C →
0.219°C), confirming the original blowup was purely a calibration artifact, not a real consequence
of the bound-tightening.

**Run 6 re-quantized too (2026-08-13, same session)** — added after noticing Run 6's own 1hr number
(1.282°C, see that entry above) showed the identical symptom. `build_model()`/`requantize_int8.py`
generalized to accept a variable-length hidden-layer list (Run 6 is a 3-layer 128→64→32 stack, not
Run 5/7/8's 4-layer 512→256→128→64):

| | 1hr | 2hr | 3hr |
|---|---|---|---|
| Run 6 original | 1.282°C | 0.308°C | 0.360°C |
| Run 6 (re-quantized) | 0.210°C | 0.306°C | **0.360°C** |

3hr is identical before and after (0.360°C both times) and 2hr barely moves (0.308°C → 0.306°C) —
only 1hr was affected, the cleanest confirmation yet that this calibration bug is isolated to
whichever head/run happens to draw an unlucky sample, not a systemic quality issue with the
affected runs' underlying models. Run 6's original conclusion (narrowing to 128→64→32 hurts 3hr
accuracy vs. Run 5/8's wider stack) is unchanged by the fix — 0.360°C remains clearly worse than
Run 5's 0.310°C and Run 8's 0.303°C.

**Reading the result — conclusions now reverse from both Run 7 and Run 8's original write-ups**:
- **Run 8 (bound-tightening alone, no diff features) is the new best-known deployable checkpoint**,
  beating Run 5 at 2hr (0.301→0.277°C, 8%) and 3hr (0.310→0.303°C, 2%), with 1hr statistically
  indistinguishable (0.207 vs 0.219°C). The illuminance/solar_radiation/uv bound-tightening
  hypothesis is confirmed: it was Run 7's bundled diff features masking a real, if modest, win.
- **Run 7's diff features are a net negative once calibration is trustworthy**, not merely inert.
  Direct Run 7 vs. Run 8 comparison (identical architecture and bound-tightening; only difference
  is the three diff features) shows diff features trade a small 1hr gain (0.219→0.196°C) for
  larger 2hr (0.277→0.344°C, 24% worse) and 3hr (0.303→0.382°C, 26% worse) losses. Combined with
  their near-zero permutation importance, they should be dropped from any future feature set.

**Decision**: adopt **Run 8** as the new best-known deployable checkpoint in the Model 5f series
(INT8 0.303°C @ 3hr, re-quantized with the fixed calibration methodology), superseding Run 5 —
pending live-deployment validation, per this project's standing caution about offline-only results
(see Run 4/5's notes). Re-quantized artifacts: `results_5f_{run}/model_5f_{run}_int8_requant.tflite`
for run5/6/7/8; full numeric results in `requant_comparison.json`. The fixed, prediction-stratified
`representative_data_gen()` is now in `train_model_5f_run8.py` (serving as the template for Run 9+)
and in the standalone `requantize_int8.py` tool, which can re-quantize any future run from its
saved checkpoint without retraining if this question comes up again.

---

## Live-Deployment Validation (2026-08-13) — the "pending" caveat resolved

**Why**: every "best checkpoint" conclusion in this project (Run 4, Run 5, Run 8) has carried the
same standing caveat — offline n=500 ranking didn't always survive live deployment previously (see
[[project_model5c]]). User's Grafana dashboard (Pi + Coral EdgeTPU, live InfluxDB data, `Actual −
Predicted` error per model, Mean/StdDev) now has backfilled inference for Run 5/6/7/8 alongside the
existing Track B/Model 5a/5b models. A single 7-day window initially looked alarming — Run 8 was
the *worst* of the 5f cluster at 3hr (0.856°C StdDev vs. Run 7's 0.809°C) — but per this session's
own methodology advice (compare over the longest common window, 30 days minimum, matching this
project's established convention, not a short snapshot dominated by one or two events), three
independent 30-day windows were pulled instead: 2026-06-14 to 07-14, 06-29 to 07-29, and 07-14 to
08-13.

**3hr StdDev, average across all three windows** (Run 5/6/7/8, Track B Run 1 = "Model 5c-1", Track
B Run 2 = "Model 5c-2"):

| | Run 5 | Run 6 | Run 7 | Run 8 | TrackB Run1 | TrackB Run2 |
|---|---|---|---|---|---|---|
| avg 3hr | 0.907°C | 0.903°C | **0.894°C** | **0.893°C** | 0.949°C | 1.034°C |

**1hr StdDev, average across all three windows**:

| | Run 5 | Run 6 | Run 7 | Run 8 | TrackB Run1 | TrackB Run2 |
|---|---|---|---|---|---|---|
| avg 1hr | 0.536°C | 0.531°C | 0.535°C | **0.530°C** | 0.560°C | 0.567°C |

**Reading the result**: the single 7-day snapshot was noise, exactly as expected — one window
showed Model 5c-1 (Track B Run 1, never given an offline n=500 number in this entire investigation)
as the clear best performer at both horizons; averaged over three 30-day windows it's clearly
behind the whole Model 5f cluster at 3hr (0.949°C vs. ~0.89-0.91°C) and only competitive with it in
one of the three windows. **Run 7 and Run 8 are statistically tied for best at 3hr** (0.894 vs.
0.893°C, 0.001°C apart — not a real difference) and **Run 8 is best-or-tied-best at 1hr** (0.530°C,
edging Run 6's 0.531°C). Track B Run 2 — the checkpoint every Model 5f run was originally
benchmarked against — is decisively the worst of this group at 3hr in all three windows (1.034°C
avg), confirming Model 5f's architecture work (BatchNorm removal, `relu6`, bound-tightening) is a
real, live-validated improvement, not an offline-only artifact.

**Decision**: the "pending live-deployment validation" caveat on Run 8 is resolved — it holds up.
Precise framing for any future reference: **Run 8 is a well-supported, best-or-tied-best live
performer at both horizons**, not decisively better than Run 7 (they're inseparable at 3hr across
three independent windows), but both clearly and consistently ahead of Track B Run 1/Run 2. Given
Run 7 and Run 8 are statistically indistinguishable live despite Run 8's cleaner offline story
(Run 7's diff features showed near-zero permutation importance and hurt 2hr/3hr in the offline
re-quantized comparison — see the "Calibration Fix" entry above), Run 8 remains the recommended
deployment candidate — simpler feature set, equal live performance, better-understood offline
behavior. No further live-validation work needed to justify this choice.
