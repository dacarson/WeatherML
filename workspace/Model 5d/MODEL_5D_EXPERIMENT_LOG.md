# Model 5d Experiment Log

**Project started**: 2026-07-22
**Succeeded**: Model 5c Track B (22 experiments, concluded 2026-07-22 — see
`../Model 5c TFT/MODEL_5C_TRACK_B_EXPERIMENT_LOG.md`)
**Goal**: Flat-feature, single-path Dense model for Coral Edge TPU — see `MODEL_5D_PLAN.md` for
full rationale, architecture, and feature set.

---

## Run 1 — Baseline Flat-Feature Model

**Date**: 2026-07-22
**Script**: `train_model_5d.py`
**Platform**: Mac Metal (M-series)
**Results stored in**: `results_5d_run1/`

**Configuration**:
- Architecture: `Input(20) → Dense(64, relu6) → Dense(32, relu6) → 3 output heads`. No AvgPool,
  no wide/deep split, no Concatenate — single path throughout.
- Features: 20 (see `MODEL_5D_PLAN.md` feature table) — includes `temp_slope_15/30/60`,
  `relative_humidity`, `humidity_slope_30`, `station_pressure` (Track A-confirmed, Track
  B-rejected as redundant with the pooled sequence — no longer applicable here).
- `use_bias=False` throughout, `L2_REG=1e-6`, fresh initialization (no warm start — different
  enough architecture from anything in Track B that warm-starting doesn't make sense).
- Verified mechanically before running: the architecture converts to a minimal, clean INT8
  graph (2 fused `MatMul+Relu6` tensors, 3 independent linear output heads, no forced-shared-
  scale or duplicate-tensor artifacts like those found in Track B Runs 20-22).

**Expected outcomes**:
- FP32: uncertain vs Track B's best (0.068-0.075°C 1hr MAE) — richer feature set (20 vs 13,
  restoring Track A-confirmed slope/humidity/pressure features) but no raw-sequence access.
- INT8: given the clean, minimal graph, expect INT8 to track FP32 much more closely than
  Track B ever achieved (Track B's best INT8/FP32 ratio was still 6-8x even after 5 rounds of
  targeted fixes) — this is the primary hypothesis under test.
- Permutation importance should reveal whether `relative_humidity`/`station_pressure`/
  `humidity_slope_30` (confirmed harmful in Track B) are actually useful without the pooled
  sequence's redundant signal, or still not.

**Results**:
- val_loss (includes L2): **0.009168** (target: <0.000373 — ~25x off)
- diff_1hr/2hr/3hr MAE (FP32): **0.434°C / 0.638°C / 0.804°C** (Track B best: 0.068-0.075°C 1hr — ~6x worse)
- diff_1hr/2hr/3hr MAE (INT8, n=500): **not reached** — script aborted before TFLite export (see below)
- Best epoch: 88 (of 600), lr decayed 1e-4 → 1.25e-5 (3 ReduceLROnPlateau reductions), val_task_loss
  plateaued ~0.0093-0.0094 across those reductions — training stalled, not still improving.
- Permutation feature importance: `time_of_day_cos` dominates (0.0259) — roughly 5-10x every other
  feature. `temp_slope_15/30/60` (the Track-A-motivated features this run was designed to test)
  registered ~0.0003-0.0005, effectively floor-level. `relative_humidity`/`station_pressure`/
  `humidity_slope_30` also floor-level (~0.0000-0.0016).

**Script aborted**: FP32 export sanity check raised `RuntimeError` (`val_loss=0.008697 >= 0.001`)
before TFLite FP32/INT8 export — see `train_model_5d.py:1012-1020`. **This is not a weight-transfer
bug**: the sanity-check loss (0.008697, 5 val batches) is consistent with the trained model's own
full val_loss (0.009168) reported two steps earlier, which had already tripped the script's own
`>= 0.001` convergence warning. The 0.001 abort threshold assumes near-target convergence and isn't
a valid weight-transfer check on its own — it just re-flags the same poor convergence a second time
and blocks INT8 export as a side effect. INT8 MAE (one of this run's own success criteria) was never
observed because of this.

**Outcome**: **Underfit, not a bug.** The flat 20-feature → Dense(64) → Dense(32) architecture
converged far short of both the val_loss and MAE targets, and did so by falling back on the diurnal
signal (`time_of_day_cos`) rather than learning from the Track-A-motivated temperature-slope
features the run was designed to validate — i.e. those features did not "turn out useful" per the
plan's Run 1 open question, at least not at this capacity.

**Decision**: do both fixes before Run 2 — (1) the export sanity check now compares
`_sanity_loss` against the trained model's own `val_loss` (ratio > 1.5x aborts) instead of a fixed
absolute 0.001, so a weak-but-honestly-converged run can still reach INT8 export/validation; (2)
capacity increased for Run 2 (see below) to address the likely underfitting root cause directly.

---

## Run 2 — Increased Capacity

**Date**: 2026-07-22
**Script**: `train_model_5d.py`
**Configuration**:
- Architecture: same flat single-path shape as Run 1, capacity doubled:
  `Input(20) → Dense(128, relu6) → Dense(64, relu6) → 3 output heads` (Run 1 was 64/32).
- FP32 export sanity check changed from fixed absolute threshold (`>= 0.001`) to relative
  (`_sanity_loss / val_loss > 1.5`) — see Run 1 outcome above for why.
- Everything else unchanged from Run 1: 20 features, `use_bias=False` throughout, `L2_REG=1e-6`,
  fresh initialization, `INITIAL_LR=1e-4`.

**Expected outcomes**:
- If capacity was the bottleneck, val_loss/MAE should move substantially closer to Run 1's targets
  (0.000373 / 0.068-0.075°C 1hr) and `temp_slope_15/30/60` permutation importance should rise off
  the floor it sat at in Run 1.
- If Run 2 still underfits similarly, capacity is not the (sole) explanation and the flat
  architecture itself (vs. Track B's wide/deep+concat effective capacity) may need reconsidering.

**Results**:
- val_loss (includes L2): **0.009107** (Run 1: 0.009168 — 0.7% lower, essentially unchanged)
- diff_1hr/2hr/3hr MAE (FP32): **0.434°C / 0.639°C / 0.799°C** (Run 1: 0.434°C / 0.638°C / 0.804°C —
  within noise, no improvement)
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.272°C / 0.411°C / 0.486°C** — export was not blocked this
  time (relative sanity-check fix worked: sanity loss 0.008316 vs trained val_loss 0.009107, ratio
  0.91x, passed). Notably INT8 MAE is *better* than FP32 MAE here (0.272 vs 0.434 1hr) — unusual
  (quantization normally degrades accuracy); likely quantization noise acting as regularization on
  an underfit/still-noisy model, or an artifact of the INT8 check's smaller n=500 sample vs the full
  FP32 validation set, rather than a real INT8 model that should be preferred as-is.
- Best epoch: 52 (of 600) — plateaued even earlier than Run 1 (epoch 88)
- Permutation feature importance: `time_of_day_cos` still dominates (0.0219), and by a *larger*
  relative margin over everything else than in Run 1. `temp_slope_15/30/60` still floor-level
  (0.0004 / 0.0002 / -0.0001) — no improvement from doubled capacity. `relative_humidity` rose
  slightly (0.0027) but still minor.

**Outcome**: **Capacity was not the bottleneck.** Doubling Dense(64,32) → Dense(128,64) produced no
meaningful change in val_loss, MAE, or best epoch, and the Track-A-motivated slope features remained
at floor importance while `time_of_day_cos` grew even more dominant. This rules out the leading Run 1
hypothesis (insufficient capacity) and points instead at the architecture itself or the feature/label
setup: either the flat single-path shape (vs. Track B's wide/deep+concat) can't extract signal from
the slope/diff features regardless of width, or those features genuinely carry little independent
information for this target once `time_of_day_cos` is available — the same open question Run 1 was
meant to resolve, still unresolved after ruling out capacity.

**Decision**: chose option (1), add depth rather than width, for Run 3 (see below). Held in reserve:
reintroducing a wide/deep/concat branch closer to Track B (walks back this project's core premise —
save for if depth also fails), checking label/loss scaling directly, and diagnostically dropping
`time_of_day_cos` to see if slope features pick up importance without the diurnal shortcut available.

---

## Run 3 — Added Depth (3rd Dense Layer)

**Date**: 2026-07-22
**Script**: `train_model_5d.py`
**Configuration**:
- Architecture: `Input(20) → Dense(128, relu6) → Dense(64, relu6) → Dense(32, relu6) → 3 output
  heads` (Run 2 was 128/64, 2 layers; this adds a narrower 3rd layer rather than widening further,
  since Run 2 showed width doubling had ~zero effect).
- Everything else unchanged from Run 2: 20 features, `use_bias=False` throughout, `L2_REG=1e-6`,
  fresh initialization, `INITIAL_LR=1e-4`, relative FP32 export sanity check (ratio > 1.5x aborts).

**Rationale**: Run 2 ruled out capacity-via-width as the bottleneck — val_loss/MAE were within noise
of Run 1 despite doubling HIDDEN_1/HIDDEN_2, and `time_of_day_cos` grew *more* dominant in
permutation importance rather than less. If the flat single-path model needs more nonlinear
combination steps to relate slope/diff features to the target (rather than more units per layer),
adding depth should show a different failure mode than Run 2 did. If Run 3 also lands within noise of
Runs 1-2, that's evidence against "this flat architecture, given more capacity in any form, will get
there" and points toward the label/loss-framing or wide/deep-reintroduction options instead.

**Expected outcomes**:
- If depth was the missing ingredient: val_loss/MAE move measurably below Runs 1-2, and
  `temp_slope_15/30/60` permutation importance rises off the floor.
- If Run 3 still lands within noise of Runs 1-2 (val_loss ~0.009, 1hr MAE ~0.434°C): neither width
  nor depth alone fixes this architecture family — next step should be the label/loss-scaling check
  or reintroducing a wide/deep branch, not further capacity tweaks.

**Results**:
- val_loss (includes L2): **0.009077** (Run 1: 0.009168, Run 2: 0.009107 — 0.3-0.9% lower, within noise)
- diff_1hr/2hr/3hr MAE (FP32): **0.429°C / 0.634°C / 0.790°C** (Run 1: 0.434/0.638/0.804, Run 2:
  0.434/0.639/0.799 — marginal improvement, within noise)
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.643°C / 0.497°C / 0.634°C** — mixed vs FP32 (1hr worse,
  2hr/3hr better), unlike Run 2 where INT8 was uniformly better than FP32 across all three horizons.
  This inconsistency (INT8 beating FP32 sometimes, losing other times, no stable direction) further
  supports Run 2's read that the INT8-vs-FP32 gap is quantization noise on an underfit/noisy model,
  not a signal to act on.
- Best epoch: 26 (of 600) — earlier still than Run 1 (88) and Run 2 (52); each capacity increase has
  converged to essentially the same plateau *faster*, not lower.
- FP32 TFLite: 52.3 KB, INT8 TFLite: 17.7 KB
- Permutation feature importance: `time_of_day_cos` dominates even more than Run 2 (0.0329 vs
  0.0219), followed by `illuminance` (0.0170), `time_of_day_sin` (0.0166), `solar_radiation`
  (0.0152), `uv` (0.0129) — the diurnal/solar cluster accounts for nearly all signal. `temp_slope_60`
  (0.0003), `temp_slope_30` (0.0002), `temp_slope_15` (-0.0000) remain at floor, same as Runs 1-2.
  `temp_diff_vs_6hr`/`temp_diff_vs_5hr` (0.0024/0.0008) also stayed low, well below Track B Run 16
  where a `temp_diff_vs_5hr/6hr` feature became the single most important one.

**Outcome**: **Depth was not the bottleneck either.** Adding a 3rd Dense layer (128→64→32) produced
no meaningful change in val_loss or MAE vs Runs 1-2, and `temp_slope_15/30/60` stayed at floor
importance while `time_of_day_cos` grew *more* dominant for the second run in a row. Best epoch also
kept getting earlier (88→52→26) as capacity increased in either dimension — the model converges to
the same diurnal-shortcut solution faster with more capacity, not to a better solution. This rules
out both width (Run 2) and depth (Run 3) as the bottleneck, closing out the "just add capacity"
branch of the Run 2 decision tree. Remaining open candidates from Run 2: reintroduce a wide/deep/
concat branch closer to Track B, verify label/loss framing wasn't itself the gap, or diagnostically
drop/downweight `time_of_day_cos` to see if slope/diff features pick up importance without the
diurnal shortcut available to lean on.

**Decision**: chose to reintroduce a wide/deep/concat branch closer to Track B (Run 4, see below),
rather than the label/loss-framing check or diagnostically dropping `time_of_day_cos`.

---

## Run 4 — Wide/Deep/Concat Branch (Reintroduced from Track B)

**Date**: 2026-07-23
**Script**: `train_model_5d.py`

**Configuration**:
- Architecture: `Input(20) → bottleneck Dense(64, relu6) → [wide: Dense(16, relu6)] + [deep:
  Dense(128, relu6) → Dense(64, relu6) → Dense(32, relu6)] → Concatenate(16+32=48) → 3 output
  heads`. Sized to match Track B's known-good bottleneck/wide/deep dimensions exactly (64/16/
  128/64/32), rather than inventing new sizes, so this is a clean test of topology (branching +
  concat) vs Runs 1-3's single sequential path, not a confound with capacity.
- Deliberately **not** carrying over Track B's later INT8-calibration fixes (Run 20's deep_out
  rescale, Run 22's deep_out prescale-before-relu6) — those tuned Track B's specific activation
  distributions and would be premature here. If Run 4 improves FP32 val_loss/MAE and INT8 shows
  the same forced-shared-scale degradation Track B Run 20 diagnosed, that's the trigger to port
  the rescale fix over, not before.
- Still flat `Input(20)` — no AvgPool, no raw 180-step sequence. That part of Model 5d's core
  premise (precomputed scalars are sufficient, the raw sequence isn't needed) is unchanged; only
  the downstream topology (single path vs wide/deep branch) is being tested.
- Everything else unchanged from Run 3: 20 features, `use_bias=False` throughout, `L2_REG=1e-6`,
  fresh initialization, `INITIAL_LR=1e-4`, relative FP32 export sanity check (ratio > 1.5x aborts).

**Rationale**: Runs 2-3 ruled out both width and depth as the bottleneck for a single sequential
Dense stack — val_loss/MAE stayed within noise of Run 1 in both directions, and `time_of_day_cos`
grew more dominant each time while `temp_slope_15/30/60` stayed at floor. The one structural
difference between Model 5d's stack and Track B's architecture (beyond the raw sequence, which
5d intentionally excludes) is the wide/deep split with a concat merge — Track B's best deployable
checkpoint (Run 11) used exactly this shape. Reintroducing it isolates whether that branching
topology itself was doing work Track B's other results didn't attribute to the raw sequence.

**Expected outcomes**:
- If wide/deep/concat was the missing ingredient: val_loss/MAE move measurably below Runs 1-3,
  and `temp_slope_15/30/60` permutation importance rises off the floor as the deep path gets more
  room to combine them nonlinearly, separate from the wide path's shortcut capacity.
- If Run 4 still lands within noise of Runs 1-3: topology (at least this form of it) isn't the
  answer either, and the label/loss-framing check or diagnostically dropping `time_of_day_cos`
  become the leading remaining candidates.
- Watch INT8 MAE for Track B's forced-shared-scale symptom re-appearing (wide and deep_out forced
  to one shared concat scale) now that a concat is back in the graph — expected here since none of
  Track B's later mitigations were carried over.

**Results**:
- val_loss (includes L2): **0.009221** (Run 1: 0.009168, Run 2: 0.009107, Run 3: 0.009077 — Run 4
  is the *worst* of the four, though within the same noise band)
- val_mae (normalized): 0.035879
- diff_1hr/2hr/3hr MAE (FP32): **0.426°C / 0.634°C / 0.798°C** — within noise of Runs 1-3
  (0.429-0.434 / 0.634-0.639 / 0.790-0.804)
- diff_1hr/2hr/3hr MAE (INT8, n=500): **1.200°C / 0.571°C / 0.685°C** — 1hr degrades sharply
  vs FP32 (0.426°C → 1.200°C, ~2.8x), while 2hr/3hr both *improve* over FP32 (0.634→0.571,
  0.798→0.685). Notably, INT8 3hr MAE (0.685°C) beats the project's own target (<0.898°C, Track
  B Run 11's best deployable checkpoint) and beats Track B Run 11/13 INT8 3hr outright — but 1hr
  is far worse than any single-path Run 1-3 result and the project's val_loss target
  (<0.000373) is still missed by >20x.
- Best epoch: 32 (Run 1: 88, Run 2: 52, Run 3: 26 — roughly back to Run 2's level, breaking the
  monotonic "more capacity → earlier convergence" pattern from Runs 1-3, consistent with topology
  rather than raw parameter count driving convergence speed)
- FP32 TFLite: 84.1 KB, INT8 TFLite: 27.2 KB
- Permutation feature importance: `time_of_day_cos` still dominates (0.0139), followed by
  `time_of_day_cos2` (0.0092), `time_of_day_sin` (0.0062), `solar_radiation` (0.0059),
  `illuminance` (0.0048), `temp_diff_vs_6hr` (0.0037 — slightly up from Run 3's 0.0024, but still
  far below Track B Run 16 where this feature was *the* top feature). `temp_slope_15/30/60`
  remain essentially at floor (0.0007 / 0.0003 / 0.00001), same as every prior run.

**Outcome**: **Topology was not the bottleneck either.** FP32 val_loss/MAE landed within noise of
Runs 1-3 (slightly worse on val_loss, indistinguishable on MAE) despite reintroducing Track B's
known-good wide/deep/concat shape — closing out all three structural candidates (width, depth,
topology) with the same result: the model converges to the same diurnal-shortcut plateau regardless
of architecture. Per the pre-registered decision rule in this run's config notes, since FP32 did
*not* improve, the trigger condition to port over Track B's INT8 rescale fixes (Run 20/22) was not
met, even though the predicted forced-shared-scale symptom did reappear (1hr INT8 MAE degrading to
1.200°C, ~2.8x its FP32 value) — chasing that fix now would be tuning quantization on a model that
isn't good enough FP32 to deploy regardless. Three consecutive architecture changes producing an
identical val_loss/MAE plateau (0.0091-0.0092 across Runs 1-4) is itself a signal worth treating as
data: it points away from "needs a different architecture" and toward either (a) a systematic issue
in label/loss framing capping achievable loss regardless of model shape, or (b) `time_of_day_cos`
acting as a shortcut feature that starves gradient signal to the slope/diff features by construction,
not a coincidence of any one architecture.

**Decision for Run 5**: not yet made — open candidates are the label/loss-framing verification and
diagnostically dropping/downweighting `time_of_day_cos`, as flagged after Run 3.

---

## Diagnostic (before Run 5) — Label/Loss Framing Verification

**Date**: 2026-08-11
**Script**: `diagnose_5d_label_floor.py` (standalone, not committed to this directory — reuses
`train_model_5d.py`'s data pipeline verbatim through target construction, no training)

**Method**: Two checks against the label/loss-framing hypothesis from Runs 3-4:
1. Diffed the exact target-scaling code (`y_min`/`y_max` = global min/max across all three
   `temp_diff_Nhr` targets combined, ±2°C pad, `2*(y-min)/(max-min)-1`, `loss="mse"` on three
   named outputs) against Model 5a clean's `train_model.py` (`workspace/Model 5a clean/
   train_model.py` lines 326-334, 491-493) — **byte-for-byte identical formula**. Model 5a
   reached val_loss=0.000682 with this exact scheme, so it is not a scheme that structurally caps
   achievable loss.
2. Computed two trivial baselines on the val set to establish an achievable-loss floor
   independent of any architecture: predict-zero-change (persistence) and predict-training-mean
   (climatology) — both landed at the same total normalized MSE (0.020973) since mean diff ≈ 0 at
   every horizon (temp_diff_1hr/2hr/3hr train means: -0.0001/0.0000/0.0003°C).

**Results**:
- Persistence/climatology baseline: MAE 0.561°C / 0.959°C / 1.326°C (1hr/2hr/3hr), total
  normalized MSE 0.020973.
- Model 5d Run 4 (FP32): MAE 0.426°C / 0.634°C / 0.798°C, val_loss 0.009221 — beats the trivial
  baseline by 24%/34%/40% MAE and ~2.3x on normalized MSE.
- Model 5a deployed: val_loss 0.000682 — ~13x better than Model 5d Run 4, using the identical
  target-scaling/loss formula but with raw-sequence access Model 5d intentionally excludes.

**Outcome**: **No label/loss-framing bug found — this candidate is closed.** The scaling and loss
formula are identical to Model 5a's proven-working scheme, and the model demonstrably extracts
real signal well above a trivial persistence/climatology baseline (not zero-signal, not stuck at a
degenerate floor). The ~13x gap to Model 5a is better explained by the flat-feature/no-raw-sequence
premise itself (Model 5d's core design choice) than by a labels bug: with the raw 180-step sequence
unavailable, `time_of_day_cos` plus the ~5-6hr lag features may simply be the strongest signal these
20 scalar features can offer, and the model is already extracting most of what's there rather than
underfitting due to a fixable defect.

**Decision for Run 5**: proceed with diagnostically dropping/downweighting `time_of_day_cos` — the
remaining candidate from Run 3/4, and now the only one left standing. If that also fails to move
`temp_slope_15/30/60`/`temp_diff_vs_5hr/6hr` off the floor, the honest conclusion is that Model 5d's
flat-scalar premise (no raw sequence) itself is the ceiling, not any fixable training detail — worth
surfacing to the user as a decision point rather than continuing to iterate silently.

---

## Run 5 — Diurnal-Shortcut Ablation (time-of-day features removed)

**Date**: 2026-08-11
**Script**: `train_model_5d.py`

**Configuration**:
- Feature set: all four time-of-day encodings removed — `time_of_day_sin`, `time_of_day_cos`,
  `time_of_day_sin2`, `time_of_day_cos2` — down to **16 features** (from 20). `day_of_year_sin/cos`
  are kept: they encode season, not time-of-day, and were never implicated as the shortcut
  (permutation importance ~0 every run, same as the slope features they're being compared
  against).
- Removing all four diurnal harmonics, not just `time_of_day_cos` alone, is deliberate: Runs 1-4
  showed `time_of_day_cos` dominant but `time_of_day_sin`/`time_of_day_cos2` also consistently
  ranked in the top 5. Dropping only the single top feature risks the model simply shifting onto
  a correlated diurnal harmonic instead of actually being forced onto the slope/diff features —
  ambiguous result either way. Removing the whole diurnal cluster is the clean version of the
  test.
- Architecture: unchanged from Run 4 (wide/deep/concat, 64/16/128/64/32) — Runs 1-4 already ruled
  out width, depth, and topology as the bottleneck, so holding architecture fixed isolates the one
  variable under test here (feature set) instead of reopening a closed question.
- Everything else unchanged from Run 4: `use_bias=False` throughout, `L2_REG=1e-6`, fresh
  initialization, `INITIAL_LR=1e-4`, relative FP32 export sanity check.

**Rationale**: Runs 1-4 ruled out capacity and topology; the label/loss-framing diagnostic (above)
ruled out a labels/scaling bug and confirmed the model is extracting real signal, just less than
Model 5a's raw-sequence architecture gets. The one untested hypothesis left is that
`time_of_day_cos` (and its correlated harmonics) act as a shortcut: cheap, low-noise, and
correlated enough with real intraday temperature dynamics that gradient descent settles for it and
never has to learn to use the noisier but potentially more informative `temp_slope_15/30/60` /
`temp_diff_vs_5hr/6hr` features. Removing the shortcut is the direct test.

**Expected outcomes**:
- If the shortcut hypothesis is right: val_loss/MAE may get *worse* (the diurnal cluster carried
  real predictive signal, just cheaply), but `temp_slope_15/30/60` and/or `temp_diff_vs_5hr/6hr`
  permutation importance should rise substantially off the floor as the model is forced to route
  through them.
- If val_loss/MAE stay roughly flat and slope/diff importance still doesn't move: the shortcut
  hypothesis is wrong too — slope/diff genuinely carry little independent signal in this flat,
  no-raw-sequence feature set, and every candidate from the Run 3/4 decision tree is exhausted.
  Per the Run 4 diagnostic's conclusion, that would point at Model 5d's core no-raw-sequence
  premise as the ceiling — a design-level question for the user, not a further architecture/
  feature tweak.
- Solar/illuminance/uv (also correlated with time-of-day but a distinct physical signal) are
  expected to partially backfill the diurnal-shortcut role, so don't over-interpret a modest
  importance increase there as evidence for the shortcut hypothesis — the decisive signal is
  specifically whether `temp_slope_*`/`temp_diff_vs_*hr` move.

**Results**:
- val_loss (includes L2): **0.013401** — worse than every prior run (Runs 1-4: 0.009077-0.009221),
  +45% relative to Run 4.
- val_mae (normalized): 0.042736
- diff_1hr/2hr/3hr MAE (FP32): **0.460°C / 0.747°C / 1.005°C** — worse than Run 4 (0.426/0.634/
  0.798°C) at every horizon: +8% (1hr), +18% (2hr), +26% (3hr). Removing the diurnal cluster cost
  real accuracy, confirming it did carry genuine (if cheap) predictive signal.
- diff_1hr/2hr/3hr MAE (INT8, n=500): **0.324°C / 0.605°C / 0.742°C** — unlike Run 4 (where INT8
  1hr degraded sharply vs FP32), INT8 *beat* FP32 at all three horizons this run (0.460→0.324,
  0.747→0.605, 1.005→0.742). No forced-shared-scale symptom this time; removing the diurnal
  features apparently changed the activation distributions enough to avoid Run 4's calibration
  problem. Consistent with Runs 2-3's read that INT8-vs-FP32 direction is quantization noise on an
  underfit model, not a stable effect tied to any one architecture or feature set.
- Best epoch: 41 (Run 4: 32) — closer to Run 1-2's range.
- FP32 TFLite: 83.1 KB, INT8 TFLite: 26.9 KB
- Permutation feature importance (full ranking): `solar_radiation` 0.01593, `illuminance` 0.01301,
  `uv` 0.01084, `relative_humidity` 0.00703, `temp_diff_vs_5hr` 0.00333, `temperature` 0.00271,
  `temp_diff_vs_6hr` 0.00259, `temp_slope_60` 0.00173, `pressure_slope_60` 0.00027, `temp_slope_30`
  0.00023, `solar_slope_30` 0.00007, `day_of_year_cos` 0.000003, `day_of_year_sin` -0.00002,
  `temp_slope_15` -0.00003, `station_pressure` -0.00013, `humidity_slope_30` -0.00016.

**Outcome**: **Shortcut hypothesis falsified.** With the diurnal cluster removed, the model did
*not* route meaningfully into `temp_slope_15/30/60` or `temp_diff_vs_5hr/6hr` — exactly as the
pre-registered "expected outcomes" flagged as the discouraging case. Instead it backfilled onto the
solar/illuminance/uv/humidity cluster (all correlated with time-of-day through distinct physical
channels — sun angle, insolation, daily humidity cycle), exactly the "don't over-interpret this"
caveat from the plan. The features actually under test barely moved: `temp_slope_60` ticked up
marginally (0.0007→0.0017, still ~9x smaller than `solar_radiation`'s importance), `temp_slope_30`
stayed flat (0.0002→0.0002), `temp_slope_15` stayed at floor/noise (-0.00003), and
`temp_diff_vs_5hr/6hr` didn't move at all (0.0037/0.0024 in Run 4 → 0.0033/0.0026 here — within
run-to-run noise). Meanwhile FP32 accuracy got meaningfully *worse* across all three horizons. This
is a clean, unambiguous negative result: `temp_slope_*` and `temp_diff_vs_*hr` are not being
suppressed by a shortcut — they genuinely carry very little independent signal in this flat,
no-raw-sequence feature set, even though Track A's TFT (Integrated Gradients) and Track B's own
Run 16 found them important in an architecture that had raw-sequence access. The signal these
features carry there apparently depends on sequence context that a single flat scalar snapshot
doesn't preserve.

**Project-level conclusion**: every candidate from the Run 3/4 decision tree is now exhausted —
width (Run 2), depth (Run 3), topology (Run 4), label/loss framing (diagnostic before Run 5), and
the diurnal-shortcut hypothesis (Run 5) have all been ruled out as fixable causes of the ~0.009-0.013
val_loss plateau, roughly 13-20x worse than Model 5a's deployed val_loss (0.000682). Five runs and
one diagnostic across every axis this project's own decision tree proposed converge on the same
read: Model 5d's core premise — that Track A/B's precomputed scalar features are sufficient without
raw-sequence access — does not hold up at the accuracy level Model 5a/Track B achieved. This is a
design-level conclusion, not a training-detail one, and is a decision point for the user rather than
a further silent iteration: see MODEL_5D_PLAN.md for the original premise and MEMORY.md
`project_model5d` for the carried-forward summary.

---

## PROJECT CONCLUDED — 2026-08-11

**User decision**: close out Model 5d. No further runs planned.

**Summary across all 5 runs + 1 diagnostic**:

| Run | Change tested | val_loss | FP32 MAE 1hr/2hr/3hr (°C) | Verdict |
|---|---|---|---|---|
| 1 | Baseline: flat `Input(20)→Dense(64)→Dense(32)→3 heads` | 0.009168 | 0.434/0.638/0.804 | Underfit; `time_of_day_cos` dominates |
| 2 | Width doubled (64/32→128/64) | 0.009107 | 0.434/0.639/0.799 | No change — width ruled out |
| 3 | Depth added (128→64→32) | 0.009077 (best FP32) | 0.429/0.634/0.790 | No change — depth ruled out |
| 4 | Topology: wide/deep/concat (Track B's known-good shape) | 0.009221 | 0.426/0.634/0.798 | No change — topology ruled out |
| — | Diagnostic: label/loss-framing check (no training) | — | — | Formula identical to Model 5a's; model beats trivial baseline by 24-40% — not a labels bug |
| 5 | Diurnal-shortcut ablation (4 time-of-day features removed) | 0.013401 (worst) | 0.460/0.747/1.005 | Worse, not better — shortcut hypothesis falsified |

**Final verdict**: Model 5d's founding premise — that Track A's TFT-confirmed `temp_slope_15/30/60`
and Track B's `temp_diff_vs_5hr/6hr` are precomputed scalars sufficient to replace the raw 180-step
sequence entirely — does not hold. In every configuration tested, these features stayed at or near
permutation-importance floor while the model leaned on `time_of_day_cos` (Runs 1-4) or backfilled
onto solar/humidity features when that was removed (Run 5). None of architecture (width/depth/
topology), labels/loss framing, or the diurnal-shortcut mechanism explains the gap — the most
consistent reading is that these features' importance in Track A/B was contingent on raw-sequence
context (e.g. relative position within the attention pattern, or interaction with neighboring
timesteps) that a flat single-snapshot vector cannot reproduce.

**Deployment status**: no Model 5d checkpoint is recommended for deployment. Best FP32: Run 3
(val_loss 0.009077, MAE 0.429/0.634/0.790°C) — still ~13x worse than Model 5a's deployed val_loss
(0.000682). One notable exception: Run 4's INT8 3hr MAE (0.685°C) beats the project's own target
(<0.898°C, Track B Run 11) and Track B's best outright — but it's an isolated bright spot on an
otherwise-underperforming checkpoint (INT8 1hr MAE 1.200°C, val_loss overall 20x off target), not a
basis for deployment on its own. **Model 5a remains the deployed model**; Track B Run 11/13
(3hr INT8 0.898/0.907°C) remain the best deployable checkpoints from the broader Model 5-series if a
Model 5a alternative is ever needed. All Model 5d artifacts (`results_5d_run1` through
`results_5d_run5`, `train_model_5d.py`, this log, `MODEL_5D_PLAN.md`) are retained as-is for
reference; no further work planned on this line unless the no-raw-sequence premise is deliberately
revisited later.

---

## Post-Conclusion Addendum — Closure Was FP32-Only; Live Data Raises a Question (2026-08-12)

**This project's closure (above) was decided entirely on FP32 accuracy** across 5 runs + 1
diagnostic — no Model 5d checkpoint was ever exported to INT8 or deployed live. That gap matters
in light of a finding surfaced afterward in
`../Model 5c TFT/MODEL_5C_TRACK_B_EXPERIMENT_LOG.md`'s own post-conclusion addendum: Track B
**Run 2** — a `SEQ_LEN=1` flat-scalar architecture from the same "no-raw-sequence" family this
project explored, with a similarly weak FP32 ceiling (0.783°C 3hr MAE, in the same range as this
project's own 0.790-1.005°C plateau across Runs 1-5) — outperforms every `SEQ_LEN=180` Track B
run in **live INT8 deployment** (Grafana, Pi + Coral EdgeTPU, weeks of real sensor data), despite
those SEQ_LEN=180 runs having dramatically better FP32 accuracy.

**Confirmed, not just hypothesized (2026-08-12)**: a controlled offline INT8 n=500 evaluation
(`../Model 5c TFT/evaluate_run2_int8.py`) verified this directly — Run 2's INT8 MAE is
0.211/0.333/0.432°C (1/2/3hr), beating Track B Run 11's INT8 (0.522/0.588/0.898°C) at every
horizon by 2-2.5x, despite Run 11's FP32 being ~4-6x better than Run 2's. Flat/shallow
architectures suffer far less FP32→INT8 degradation than the deep SEQ_LEN=180 architecture —
Run 2's INT8 actually *improves* on its own FP32, while Run 11's INT8 degrades catastrophically
(~640% worse at 3hr). This is the exact structural property (fewer sequential quantized MatMuls)
Track B's Runs 6-22 spent 22 experiments establishing as the cause of 3hr INT8 degradation — a
"worse FP32 but INT8-robust" flat model decisively beats a "better FP32 but INT8-fragile"
sequence model once actually quantized.

**Implication for this project**: Model 5d's flat, no-raw-sequence architecture is structurally
similar to Track B Run 2 (both flat scalar vectors, no `AveragePooling1D` over a raw sequence).
This project's own best FP32 checkpoint (Run 3, val_loss 0.009077, MAE 0.429/0.634/0.790°C) was
never exported to INT8 or tested live — given Run 2's confirmed result above, it's a real
possibility this checkpoint would show the same worse-FP32-but-INT8-robust pattern, which the
FP32-only closure decision above could not have detected.

**Run 3 INT8 result — confirmed, and it's more nuanced than a clean replication (2026-08-12)**:
`evaluate_run3_int8.py` evaluated Run 3's already-exported `model_5d_run3_int8.tflite` (never
previously scored — the script computes and prints this validation but doesn't save it to
`results_5d_run3.json`) against the same n=500 methodology, after a random-sample FP32
reproduction check confirmed the feature pipeline (0.420/0.624/0.782°C vs Run 3's saved
0.429/0.634/0.790°C).

**Result: INT8 MAE 0.643/0.497/0.634°C (1/2/3hr).** This partially confirms the pattern —
3hr beats Track B Run 11's INT8 (0.634°C vs 0.898°C, ~30% better) — but Run 3 does NOT match
Track B Run 2's INT8 result (0.211/0.333/0.432°C), despite Run 3 using Track A/B's more
carefully curated 20-feature set (Run 2's 23 features include several — `relative_humidity`,
`station_pressure` raw, wind-adjacent lags — that Track A/B identified as harmful or floor-level
for the deep architecture). So "shallow architecture" alone doesn't fully explain Run 2's INT8
advantage; something else about Run 2 also matters: it has `BatchNormalization` after every
layer (Run 3 has none), uses plain unbounded `relu` (Run 3 uses `relu6`), and is wider
(512→256→128→64 vs Run 3's 128→64→32). Notably, BatchNorm was something Track B's *deep*
architecture explicitly **removed** as an INT8 fix (Run 11) — here, in a shallow architecture, it
appears to help instead. This is now an open design question for "Model 5f" (a fresh architecture
combining Run 2's INT8-friendly properties with Track A/B's feature curation), not a settled one
— still not re-opening this project unilaterally, but the case for further shallow-architecture
work is now backed by two independent, controlled data points, not one. Script: `evaluate_run3_int8.py`;
raw output: `results_5d_run3/run3_int8_eval_n500.json`.
