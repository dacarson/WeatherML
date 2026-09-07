# SolarChargeML — Experiment Log

**Target**: Beat the heuristic's own historical `excess_solar_watts` predictions (logged in
`pvs6.solar_charge_control`) on 5-min-ahead excess-solar MAE, evaluated on `val_data.csv`
(2026-07-01 → present) restricted to rows where the heuristic's prediction is also logged (~77%
coverage — see `SOLARCHARGE_PLAN.md` §5). Persistence (`excess_future_w = excess_now_w`) is a
secondary, weaker baseline — `corr(excess_now_w, excess_future_w) = 0.915` at this horizon means
persistence alone is already fairly strong, so beating the heuristic specifically is the real bar.

---

## Run 1 — HistGradientBoostingRegressor baseline

**Date**: 2026-08-19
**Hypothesis**: instantaneous state (production/load/weather) plus short-term slope features
(10/30-min trends in `pv_p`, `solar_radiation`, `excess_now_w`) predicts 5-min-ahead excess solar
power better than the heuristic's single linear `pv_p`-slope extrapolation, because it can use
measured irradiance trend (a more direct cloud-transient signal than `pv_p`'s own lagged slope)
and non-linear interactions a plain linear extrapolation can't capture.

**Script**: `train_run1.py`

**Features**:
- state: `pv_p`, `net_p`, `site_load_p`, `baseline_house_load_w`, `excess_now_w`
- weather: `solar_radiation`, `illuminance`, `uv`, `wind_avg`, `wind_gust`, `wind_lull`,
  `wind_direction`, `relative_humidity`, `station_pressure`, `temperature`, `rain_accumulated`
- cyclic time: `time_of_day_sin/cos`, `day_of_year_sin/cos`
- engineered slopes (10/30 min): `pv_p_slope_10`, `pv_p_slope_30`, `solar_radiation_slope_10`,
  `excess_now_slope_10` — computed after reindexing `train_data.csv`/`val_data.csv` onto a
  regular 1-minute grid inside the training script, so shift-based lags stay time-correct across
  the export's small gaps (`export_and_join.py` dropped 2.5% of rows for missing fields).

Target: `excess_future_w` (5-min ahead) — the same quantity the heuristic's `excess_solar_watts`
estimates.

**Model**: sklearn `HistGradientBoostingRegressor`. Chosen for RPi deployment simplicity — pure
sklearn dependency, joblib-picklable (this repo's Forecaster_1/2 projects already deploy
sklearn-style pickled models this way), no ONNX/TFLite export step needed, and CPU inference is
comfortably fast enough for a 5-minute control cycle. Also matches `SOLARCHARGE_PLAN.md` §3's
"start simple" note — no evidence yet this problem needs a sequence model.

**Training/eval split**: `train_data.csv` (2025-04-29 → 2026-06-30) / `val_data.csv` (2026-07-01
→ 2026-08-20), chronological, from `export_and_join.py`.

**Evaluation**: MAE on `val_data.csv`, two ways:
1. Full val set — model vs. naive persistence (`excess_future_w = excess_now_w`).
2. Rows where the heuristic's `excess_solar_watts` is also logged — model vs. the heuristic's
   actual historical prediction. This is the number that matters for the go/no-go decision.

**Expected outcomes**: beating the heuristic on the shared-row backtest validates the
weather-informed short-horizon approach; beating persistence but not the heuristic means the
heuristic's slope extrapolation is already capturing something this feature set misses (window
length, or the raw-`pv_p`-vs-irradiance distinction) and feature/window tuning is the next step;
not even beating persistence means the model/feature set is under-fit and needs revisiting before
any further architecture comparison is meaningful.

**Results (2026-08-19)**:

| Eval set (backtest subset, has logged `excess_solar_watts`) | n | Model MAE | Heuristic MAE | Persistence MAE |
|---|---|---|---|---|
| All val rows | 65,778 | 244.9 W | 252.1 W | **242.3 W** |
| Daytime only (`pv_p >= 500W`, the only regime `predicted_excess` actually gates charging in) | 31,214 | 392.5 W | **390.3 W** | 389.6 W |

The "all val rows" number is misleading: it's dominated by nighttime rows where
`solar_charge_controller.py` never uses `predicted_excess` at all (its `production < 500W` branch
bypasses it entirely — see `SOLARCHARGE_PLAN.md` §1 step 5), so a large fraction of that MAE is
free credit for correctly predicting "still near zero." Restricting to `pv_p >= 500W` — the actual
operating regime — all three numbers are within ~1% of each other (**389.6–392.5 W**). Run 1 does
**not** beat the heuristic, and doesn't beat naive persistence either, in the regime that matters.

**Diagnosis**: at a 5-minute horizon, `excess_now_w` and `excess_future_w` are already highly
correlated (0.915 overall, per `SOLARCHARGE_PLAN.md` §5) — persistence is a very strong baseline,
and both the heuristic's linear slope term and this run's slope/weather features are only
providing a small correction on top of it. Predicting the raw target directly means the model has
to reproduce that entire dominant persistence signal through splits before any of the harder,
actually-informative structure (the correction itself) shows up in the loss. This is a case where
predicting the *delta* (`excess_future_w - excess_now_w`) rather than the raw value is the
standard fix — it isolates the low-variance, actually-hard-to-predict quantity instead of asking
the model to re-derive persistence from scratch.

**Decision for Run 2**: reformulate the target as `excess_delta_w = excess_future_w - excess_now_w`
(prediction = `excess_now_w + model.predict(...)`), same features and model family, to test
whether that reformulation — not a bigger/different model — is what's needed to beat the heuristic
in the daytime regime.

---

## Run 2 — Delta target, daytime-restricted training

**Date**: 2026-08-19
**Hypothesis**: per Run 1's diagnosis, predicting `excess_delta_w = excess_future_w -
excess_now_w` (reconstructing `excess_future_w = excess_now_w + model.predict(...)`) isolates the
actually-hard-to-predict correction instead of making the model re-derive the dominant persistence
signal from scratch, and should beat both persistence and the heuristic in the daytime regime
where Run 1 was statistically tied with both. Also restrict training (not just evaluation) to
`pv_p >= 500W` rows, since nighttime rows are near-zero-delta and dilute training signal for a
regime the model will never actually be invoked in (`solar_charge_controller.py`'s
`production < 500W` branch bypasses `predicted_excess` entirely).

**Script**: `train_run2.py`, forked from `train_run1.py` — same features, same
`HistGradientBoostingRegressor` config, only the target and the train/val filtering change, to
isolate the reformulation's effect rather than conflating it with an architecture change.

**Expected outcomes**: if Run 2 beats the heuristic on the daytime backtest, the delta
reformulation (± daytime-only training) was the fix and this becomes the deployment candidate to
live-validate. If it beats persistence but not the heuristic, the heuristic's 30-min slope window
is doing something this run's 10/30-min slope features still don't capture and window-length
tuning is the next step. If it doesn't even beat persistence, the delta reformulation alone isn't
sufficient and the feature set itself needs revisiting (e.g. longer lookback windows, or
lower-level per-second `sunpower_power` volatility features lost in the 1-minute resampling).

**Results (2026-08-19)**:

| Eval set | n | Model MAE | Heuristic MAE | Persistence MAE |
|---|---|---|---|---|
| Daytime val (no heuristic-logged filter) | 32,734 | 407.3 W | — | 424.3 W |
| Daytime backtest subset (heuristic logged) | 31,214 | 400.6 W | **389.6 W*** | 389.6 W |

\* Heuristic and persistence MAE are effectively identical here (390.3 vs 389.6 W); the
apparent 3-way tie from Run 1 persists.

**Hypothesis falsified.** The delta reformulation did not help — it's slightly *worse* than
Run 1's direct-target model on the same backtest subset (400.6 W vs 392.5 W), and both remain
worse than persistence/heuristic. Daytime-restricted training (211k vs Run 1's 582k rows) is
confounded with the target-reformulation change here, so it's not possible to attribute the
regression to one or the other in isolation — a design gap in this run worth avoiding next time.

**Reassessment**: two runs now show the same pattern — persistence and the heuristic are
statistically indistinguishable (~390 W MAE) in the regime that matters, and neither
reformulation attempted here has beaten that. The likely explanation is that 1-minute-mean
resampling in `export_and_join.py` throws away exactly the sub-minute volatility (cloud-edge
transients) that would differentiate a smarter model from "assume the next 5 minutes look like
now" — the current feature set literally cannot see faster structure than the persistence
baseline already captures. Continuing to iterate on model architecture or target reformulation
without addressing that is unlikely to move the number. **Paused here to report findings and
decide direction with the user** rather than continuing to fish for a positive result — candidate
next directions: (a) re-export with sub-minute volatility stats (std/min/max of `pv_p` and
`solar_radiation` within each 1-min bin, not just the mean) to give the model access to transient
structure the heuristic can't see either, or (b) accept that 5-min-ahead point prediction may be
fundamentally close to persistence-limited at this horizon and reconsider what "better than the
heuristic" should even mean for this project.

**Decision (2026-08-19, user)**: proceed with (a) — re-export with sub-minute volatility features.

---

## Run 3 — Sub-minute volatility features (planned)

**Date**: 2026-08-19
**Hypothesis**: `export_and_join.py`'s 1-minute `MEAN()`-only resampling discards the sub-minute
transient structure (cloud-edge passage, load spikes from unscheduled appliances — the exact
nuisances named in the original project ask) that both persistence and the heuristic are already
blind to. Adding `STDDEV`/`MIN`/`MAX` per 1-min bin for `pv_p`, `site_load_p` (unscheduled house
loads), and `solar_radiation` gives the model information neither baseline has access to, which is
necessary for it to differentiate itself at all — Runs 1 and 2 couldn't have beaten the baselines
even with a perfect model, because the input data was information-equivalent to what persistence
already uses.

**Script changes**:
- `export_and_join.py`: `fetch_1min_means` generalized to `fetch_1min_stats`, taking a
  `{field: [agg_fns]}` map instead of a flat field list. `MEAN` keeps existing bare column names
  (no breaking change to Runs 1/2's columns); `STDDEV`/`MIN`/`MAX` add `_std`/`_min`/`_max`
  suffixed columns. All aggregates for a given field come from a single InfluxDB query (one
  raw-data pass), not one query per function. `train_data.csv`/`val_data.csv` regenerated in
  place (gitignored, no run-specific filename per repo convention).
- **Correction during implementation (2026-08-19)**: originally requested `STDDEV`/`MIN`/`MAX`
  for `solar_radiation` too, from `wf/obs_st`. That measurement's native `report_interval` is 1
  minute (confirmed via direct query — exactly 1 raw sample per 1-min bin), so `STDDEV` of a
  single sample is undefined (NULL) and `MIN`/`MAX` are identical to `MEAN`. Requiring those
  columns non-null in the dropna step discarded **99.6% of rows** (688,408 → 2,652) on the first
  attempt. Fixed by dropping `solar_radiation` back to `MEAN`-only — there is no sub-minute
  weather signal available from this station at all, only from `sunpower_power`'s ~60
  samples/minute (`pv_p`, `site_load_p`). Re-ran clean.
- `train_run3.py`: forked from `train_run2.py` (delta target — Run 2's reformulation itself
  wasn't clearly harmful in isolation, it was confounded with the daytime-only-training change;
  keeping it here since it's still the theoretically better-motivated target). Adds the new
  volatility columns (raw + `_range = _max - _min` engineered in the training script, matching
  the repo's convention of keeping the export script's derived features light) to `FEATURES`.

**Expected outcomes**: if Run 3 beats the heuristic on the daytime backtest, the missing
transient-volatility signal was the actual bottleneck, not model architecture or target framing —
proceed to live-validation. If it still doesn't, the 5-minute horizon may genuinely be
persistence-limited at 1-minute-and-finer resolution too, and the framing question from the
"pause and reconsider" option becomes the live one.

**Results (2026-08-19)**, adding `pv_p_std`, `pv_p_range` (=`pv_p_max - pv_p_min`),
`site_load_p_std`, `site_load_p_range` to Run 2's feature set (28 features total):

| Eval set | n | Model MAE | Heuristic MAE | Persistence MAE |
|---|---|---|---|---|
| Daytime val (no heuristic-logged filter) | 32,753 | 401.7 W | — | 424.1 W |
| Daytime backtest subset (heuristic logged) | 31,233 | 395.2 W | **390.0 W** | 389.4 W |

**Hypothesis not confirmed, but directionally right.** Sub-minute volatility features moved the
model closer to the baselines (395.2 W, vs. Run 2's 400.6 W on the same comparable subset) but
still didn't cross them (390.0/389.4 W). Persistence and the heuristic remain statistically tied
with each other, and three runs now — direct target, delta target, delta + volatility — have all
landed in the same ~389–407 W band without separating from either baseline.

**Synthesis across Runs 1–3**: the consistent pattern (persistence ≈ heuristic ≈ every model
variant tried, all within a few percent) is stronger evidence than any single run that 5-minute-
ahead excess solar is close to its persistence ceiling with the data sources tried so far —
1-minute-resampled PVS6 + co-located weather station. Model architecture and target framing
changes haven't moved the needle; only richer input signal (Run 3's volatility features) moved it
at all, and only partially. Paused here rather than continuing to iterate on architecture — this
is a decision point for the user: keep pushing on richer/lower-level signal (e.g. skip the 1-min
resample entirely and feed raw ~1-5s `sunpower_power` samples directly, or bring in
`wf/rapid_wind`/`wf/evt/precip` for faster-than-1-min weather signal), or treat "beat the
heuristic's MAE" as not the right bar for this project and reconsider the goal.

**Decision (2026-08-20, user)**: proceed with pushing further on raw signal.

---

## Run 4 — Finer time resolution (30s grid, planned)

**Date**: 2026-08-20
**Hypothesis**: Run 3's `pv_p`/`site_load_p` volatility features (`STDDEV`/range within each
1-minute bin) helped partially — the first movement toward the baselines across four runs — which
points at temporal resolution, not architecture, as the remaining lever. Going all the way to raw
~1-5s `sunpower_power` samples (~40.6M rows) is likely overkill for a tree model and mostly
redundant information; a 30-second grid is a middle ground: still ~15-30 raw `pv_p` samples per
bin (enough for meaningful `STDDEV`/range), 2x the row count of the 1-min grid (~1.3M rows, still
small), and — the actual point — a fresher `excess_now_w` "current state" (≤30s stale instead of
up to 60s) plus room for a genuinely fast slope feature (e.g. 2-min trend) the 1-min grid couldn't
resolve at all.

**Script changes**:
- `export_and_join.py`: bin width parameterized (`BIN = "30s"`, used in both the InfluxDB
  `GROUP BY time()` clause and the pandas resample grid — previously hardcoded to 1 minute in
  both places). `wf/obs_st` reports once every 60s (confirmed in Run 3's correction), so at a
  30s grid every other row is null immediately after reindexing — forward-filled with
  `limit=1` (≤30s staleness) rather than left null, since the station's own read is already up to
  60s stale at its native cadence and this doesn't make that meaningfully worse.
  `HORIZON_MIN=5` stays a wall-clock constant; the row-shift for the target is now computed from
  it (`horizon_steps = 5min / 30s = 10`) rather than hardcoded to match a 1-minute grid.
- **Correction during implementation (2026-08-20)**: `solar_charge_control` is written once per
  control cycle (~60s) and holds a step-function value in between — at the finer 30s grid, every
  other bin came back null for `charging_power_watts`/`excess_solar_watts`/`solar_slope_w_per_s`.
  The existing `charging_power_watts.fillna(0.0)` (for the EV-subtraction step) would have
  silently misread "not freshly logged this bin" as "car not charging" on every other row.
  Forward-filled `ctrl` with `limit=20` (~10 min, well past one control cycle) before the `fillna`
  step, so a genuine value carries forward correctly and a real service outage still surfaces as
  NaN rather than being carried forward indefinitely.
- `train_run4.py`: forked from `train_run3.py`. Slope windows recomputed in row-steps for the new
  bin width (10min/30min slopes = 20/60 steps, not 10/30) plus a new 2-minute slope (4 steps) —
  the fast-transient signal the 1-min grid physically couldn't represent.

**Expected outcomes**: if Run 4 beats the heuristic on the daytime backtest, temporal resolution
was the bottleneck across Runs 1-3, not features or architecture — proceed to live-validation. If
it's directionally better than Run 3 but still short (mirroring Run 3's partial-but-incomplete
movement), that's further evidence for resolution-as-bottleneck and going to raw per-sample data
may be warranted despite the cost. If it doesn't move at all from Run 3, resolution likely isn't
the answer and the "reconsider the goal" framing question from Run 3's synthesis should be
revisited directly rather than continuing to spend runs on this axis.

**Results (2026-08-20)**, 30s grid, adding `pv_p_slope_2min`/`excess_now_slope_2min` (30 features
total; also benefits from the ctrl-ffill correctness fix above, which raised backtest coverage
from ~77-83% in Runs 1-3 to 99.9% here):

| Eval set | n | Model MAE | Heuristic MAE | Persistence MAE |
|---|---|---|---|---|
| Daytime val (no heuristic-logged filter) | 65,341 | 363.3 W | — | 408.5 W |
| Daytime backtest subset (heuristic logged, 99.9% of daytime val) | 65,305 | **363.4 W** | 384.9 W | 408.6 W |

**Hypothesis confirmed — first run to beat both baselines.** Model MAE is 5.6% better than the
heuristic and 11.1% better than naive persistence, on a backtest now covering essentially all of
the daytime validation window (not a small/biased subset). Temporal resolution was the actual
bottleneck across Runs 1-3, not target framing or the volatility features themselves in isolation
— they only paid off once the underlying grid was fine enough to carry a fresher `excess_now_w`
and a fast (2-min) slope alongside them.

**Decision**: `model_run4.joblib` is the first viable deployment candidate. Per
`SOLARCHARGE_PLAN.md` §4/§5 and the `[[feedback_live_validation_window]]` project convention, an
offline backtest win is not sufficient on its own — next step is live validation across 3+
independent windows before any change to `solar_charge_controller.py` itself. Not yet started.

---

## Live shadow validation results (2026-08-20 → 2026-09-06, ~17 days, reported by user)

`model_run4.joblib` beat the heuristic on watts-level MAE by ~9-17% depending on TOU period,
consistent with the 5.6% offline backtest and stable across four check-ins. **But this did not
translate 1:1 into better amp decisions**: `determine_target_amperage()` rounds up to the nearest
allowed amp (~240W-wide buckets), so small forecast differences often land on the same integer
amp. Measured against a perfect-hindsight "ideal amp," the model's edge shrank to roughly +3
percentage points of exact-amp-match and ~13% relative reduction in mean amp error — real, but
much smaller than the watts MAE number suggested. Converted to PG&E NEM 3.0 cost, the model is
saving on the order of **$4.60/month**, concentrated almost entirely in the `off_peak` TOU period.

**This is the key methodological finding driving Runs 5+: watts MAE is not the right optimization
target.** From here on, amp-decision accuracy (vs. a perfect-hindsight ideal amp, by TOU period)
and simulated $ impact are the metrics that matter; watts MAE is reported only for continuity
with Runs 1-4.

New shared modules added for this: `feature_engineering.py` (Runs 1-4's feature logic, factored
out of `train_run4.py` so Run 5+ scripts share one definition instead of drifting copies —
`train_run1-4.py` are left untouched as historical records) and `decision_policy.py` (TOU
threshold / amp rounding / PG&E rate tables / cost simulation, implementing the brief's given
`get_tou_excess_threshold`/`determine_target_amperage` logic exactly, plus a
`add_tou_and_season()` helper that derives TOU period and season from each row's own
Pacific-local timestamp rather than `solar_charge_controller.py`'s real-time-only
`datetime.now()`-based version).

---

## Run 5a — Retrain through today, same architecture (planned)

**Date**: 2026-09-06
**Hypothesis**: `model_run4.joblib` was trained on data only through 2026-08-20; simply
extending the training window through today (more data, same architecture/features/hyperparams)
might improve amp/dollar performance even before touching the loss function or feature set —
cheapest thing to try first, and a necessary control before attributing any Run 5b result to the
loss-function change rather than just "more data."

**Split-date change**: `export_and_join.py`'s `TRAIN_END`/`VAL_START` move from
2026-06-30/07-01 to **2026-07-16T23:59:59Z / 2026-07-17T00:00:00Z** — chosen to keep the
validation window roughly the same length as Run 4's (51 days: Jul17→Sep6, vs. Run 4's 50 days:
Jul1→Aug20) for comparability, while extending training by the ~2.5 weeks of new data between
the two cutoffs. `FETCH_START` unchanged (2025-04-29); `FETCH_END` was already
`datetime.now(timezone.utc)` (dynamic), so it picks up today's data automatically on re-run. This
new val window overlaps the live shadow-validation period (started 2026-08-20), which lets the
offline $ simulation below be sanity-checked against the user's live-measured ~$4.60/month figure
for the overlapping sub-period.

**Script**: `train_run5a.py`, using the new shared `feature_engineering.py` — same features,
same `HistGradientBoostingRegressor` hyperparameters as `train_run4.py`
(`max_iter=300, learning_rate=0.05, max_depth=8, random_state=42, validation_fraction=0.1,
early_stopping=True, n_iter_no_change=15`), same delta-target reconstruction. Only the
underlying data changes.

**Evaluation**: `evaluate_candidates.py` (new, shared across Run 5a/5b) computes, per candidate
(heuristic, `model_run4` unchanged, `model_run5a`, ideal/perfect-hindsight), on the *same* new
val set:
1. Watts MAE (continuity with Runs 1-4).
2. Amp-decision accuracy vs. the perfect-hindsight ideal amp — % exact match and mean |amp
   error|, broken out by `tou_period` (peak/part_peak/off_peak).
3. Simulated $ cost (PG&E NEM 3.0 marginal rates, seasonal) — total and by TOU period, comparing
   each candidate's amp decisions (applied against the *real* realized excess) against the
   heuristic's and the ideal's.

Evaluating `model_run4` unchanged on the *new* val set (not its original Jul1-Aug20 one) isolates
"did retraining help" from "did the eval window change" — both vary between Run 4's original
report and Run 5a otherwise.

**Expected outcomes**: if Run 5a's amp-accuracy/$ numbers beat `model_run4`'s (on the same new
val set), more data alone helps and Run 5b's quantile-loss experiment starts from this as the new
baseline rather than Run 4. If not, more data alone isn't sufficient and Run 5b's result will
need to be judged against Run 4 directly instead.

---

## Run 5b — Asymmetric (quantile) loss sweep (planned)

**Date**: 2026-09-06
**Hypothesis**: under NEM 3.0, over-predicting excess (charging too much) pulls the shortfall
from the grid at the full import rate; under-predicting (charging too little) only forgoes the
much smaller export credit (e.g. Summer off-peak: 0.2649 import vs. 0.0531 export — importing is
~5x more expensive than the credit forgone). These are not symmetric costs, so the
squared-error loss every prior run has used is the wrong objective for this problem — a model
that's unbiased in watts MAE is *not* unbiased in dollar terms, since over- and under-shoots cost
differently. `HistGradientBoostingRegressor(loss="quantile", quantile=q)` (available since
sklearn 1.1; the deployed pin is exactly 1.7.0) lets the model target a specific quantile of the
conditional distribution instead of the mean — `q<0.5` biases predictions below the median (i.e.
below the typical realized value), which is exactly the "prefer to under-predict excess" bias
NEM 3.0's asymmetric rates call for. Since the reconstructed prediction is
`excess_now_w + model.predict(delta)`, a quantile bias on the delta target shifts the
reconstructed excess prediction by the same rank, so this works on the delta target unchanged.

**Script**: `train_run5b.py`, forked from `train_run5a.py` (same extended dataset, same
features/hyperparams) — only `loss`/`quantile` change. Sweeps `quantile ∈ {0.3, 0.4, 0.5}`
(`0.5` is also a genuinely different loss from Run 4/5a's default `squared_error` — pinball loss
at the median approximates L1/MAE-optimal, not mean-optimal — so it's an informative point in the
sweep on its own, independent of the asymmetry). Saves all three as
`model_run5b_q{30,40,50}.joblib`; only the winner (if any beats Run 5a on the $ metric) gets
promoted to `model_run5b.joblib`, per the "don't promote on watts MAE alone" rule below.

**Evaluation**: same `evaluate_candidates.py` harness as Run 5a, all three quantile variants
added as candidates evaluated on the same val set alongside heuristic/ideal/Run 4/Run 5a.

**Promotion rule (both runs)**: per the shadow-mode findings above, a lower watts MAE alone is
not sufficient evidence of improvement. A candidate only gets promoted to a new
`model_runN.joblib` — and only that specific candidate — if it beats the current best (Run 4, or
Run 5a if that wins its own comparison) on the **simulated $ metric**, with amp-decision accuracy
as supporting evidence. `sklearn` version used to train must remain exactly `1.7.0`
(deployed pin in the `chargepoint-sunpower-chargemanager` repo's `requirements.txt` — a version
mismatch breaks unpickling on the Pi, as happened once already during Run 4's live deployment).
Any promoted model's feature list is unchanged from Run 4 (same `FEATURES` order, same joblib
bundle shape `{"model", "features"}`), so no change is needed to
`model_shadow_logger.py`'s `build_feature_frame()` in the other repo — only if a future run
changes the feature set would that file need updating too.

**Expected outcomes**: if a `q<0.5` variant wins on $ despite a possibly worse (or unchanged)
watts MAE, that directly confirms the asymmetric-loss hypothesis and becomes the new deployment
candidate. If `q=0.5` alone (symmetric, but MAE-optimal rather than MSE-optimal) already wins,
the loss *shape* (robustness to outliers) mattered more than the asymmetry. If no quantile
variant beats Run 5a/Run 4 on $, asymmetric loss doesn't help at this amp-bucket granularity —
plausible given how coarse the ~240W buckets are — and the amp/dollar gap identified in shadow
mode may need a different lever entirely (e.g. a classification-style objective directly on amp
buckets, rather than a regression + post-hoc rounding).

**Results (2026-09-06)**, all candidates evaluated on the same extended val set
(`evaluate_candidates.py`, new shared harness — amp accuracy vs. perfect-hindsight ideal amp by
TOU period, plus simulated $/month on decision-cadence-matched rows, 65,271-row backtest,
2026-07-17 → 2026-09-07):

| Candidate | Watts MAE | ALL exact-match % | ALL mean \|amp err\| | $/month | $ vs. heuristic |
|---|---|---|---|---|---|
| heuristic | 387.0 W | 66.9% | 1.13 | 50.93 | — |
| **ideal** (perfect hindsight, same policy) | 0.0 W | 100.0% | 0.00 | 46.98 | +3.96 |
| model_run4 (unchanged, new val window) | 362.3 W | 64.6% | 1.09 | 49.04 | +1.90 |
| model_run5a (retrained, more data) | 356.1 W | 65.6% | 1.06 | 49.46 | +1.47 |
| model_run5b_q30 | 383.4 W | 66.9% | 1.11 | 44.60 | **+6.33** |
| model_run5b_q40 | 354.9 W | 68.0% | 1.05 | 48.26 | +2.67 |
| model_run5b_q50 | 340.0 W | 68.7% | 1.00 | 52.54 | −1.61 |

**Headline surprise, investigated and confirmed real, not a bug**: `model_run5b_q30` saves
**more** per month ($6.33) than the theoretical **ideal** ($3.96) — despite having *worse* watts
MAE than every other candidate (383.4 W) and roughly the *same* amp exact-match rate as the
heuristic. Root cause, confirmed by direct inspection: `determine_target_amperage()` always
rounds **up**, and for any true excess in `(-500W, 1800W)` — 40.6% of all off-peak rows in this
backtest — the existing policy (fed even a perfect forecast) floors to a flat 8A regardless of
how close to zero the real excess is, forcing up to 1920W of grid import by design (this
matches the off-peak policy's documented intent: tolerate grid draw rather than stop/start on
brief dips). `model_run5b_q30`'s systematic ~163W low bias pushes 8.8% of that 8A-floor band
down across the off-peak TOU gate (`-500W` threshold) to a 0A "don't charge" decision instead,
avoiding that floor-import cost. **The "ideal" baseline is not the dollar-optimal achievable
outcome — it's only "the best the existing round-up policy can do with perfect information."** A
deliberately biased-low forecast can structurally beat it by partially correcting for the
policy's own round-up inefficiency, a mechanism distinct from (and compounding with) the
originally-hypothesized NEM 3.0 rate-asymmetry effect.

**Oscillation check (necessary before trusting this)**: a policy that charges less often could
just be trading dollar savings for more stop/start cycling — exactly what the off-peak
threshold's lax `-500W` tolerance was designed to prevent, and a cost this simulation doesn't
model. Checked directly (0→nonzero amp transitions/day on decision-cadence rows):

| Candidate | off_peak flips/day | ALL flips/day |
|---|---|---|
| heuristic | 4.46 | 7.11 |
| ideal | 6.51 | 9.45 |
| model_run4 | 3.15 | 5.42 |
| model_run5a | 3.06 | 5.46 |
| **model_run5b_q30** | **4.36** | **6.61** |
| model_run5b_q40 | 3.61 | 6.07 |
| model_run5b_q50 | 3.54 | 6.15 |

`model_run5b_q30`'s oscillation rate is *below* the currently-deployed heuristic's, not above
it — the $ win isn't bought with more cycling. (Note "ideal" has the *highest* oscillation rate
of all: perfect knowledge chases every real transient at the threshold boundary, while every
trained model's smoother/noisier predictions act as an implicit low-pass filter — an
unadvertised side benefit of imperfect forecasts here.)

**Other findings**:
- Run 5a (more data alone) beat `model_run4` on watts MAE (356.1 vs 362.3W) but is essentially a
  wash on $ (+1.47 vs +1.90/month) — more data alone did not close the amp/dollar gap.
- `q=0.5` (median/pinball loss, still symmetric) has the *best* watts MAE of any candidate
  (340.0W) but is the only candidate that **loses money vs. the heuristic** (−1.61/month) — the
  clearest demonstration in this project that watts MAE and dollar impact are different
  objectives, exactly the premise motivating this whole run.
- `q=0.3` was the best of the three brief-specified quantiles and was not yet a local optimum
  candidate by construction (only 3 points tested) — see Run 5c.

**Sanity cross-check against live shadow data**: this offline ideal-vs-heuristic gap ($3.96/mo)
is in the same ballpark as, but smaller than, the user's live-measured `model_run4`-vs-heuristic
gap (~$4.60/mo over the 2026-08-20→09-06 shadow window). Expected, not concerning: different
windows (this backtest spans Jul17-Sep6, mostly *before* live shadow deployment), and the live
figure reflects the heuristic's actual applied amperage (with its real start/stop hysteresis)
rather than this harness's simplified formula-only reconstruction of the heuristic's decisions
(see `decision_policy.py`'s docstring) — a deliberate simplification for cross-candidate
consistency, not a discrepancy to chase down further here.

**Decision**: do not promote Run 5a or Run 5b as-is yet — `q=0.3`'s result motivates
characterizing the quantile-vs-$ curve further before picking a value to deploy, since only
three points were tested and the mechanism (rounding-policy correction) suggests there may be a
better quantile nearby, or a point where it turns over. See Run 5c.

---

## Run 5c — Bracketing the quantile sweep below 0.3 (planned)

**Date**: 2026-09-06
**Hypothesis**: Run 5b tested only `{0.3, 0.4, 0.5}` (the brief's specified sweep) and found
`q=0.3` best by a wide margin, with a well-understood mechanism (correcting for
`determine_target_amperage`'s round-up-always convention in the off-peak 8A-floor band) that
doesn't obviously saturate at 0.3 — there could be a better value nearby, or the $ benefit could
turn over (start losing real charging opportunities) below some point. Sweep `{0.10, 0.15, 0.20,
0.25}` to bracket the space between "no bias" (0.5) and "0.3" and find where $/month peaks,
using the exact-match/mean-amp-error and oscillation-rate diagnostics from Run 5b's results as
guardrails against a value that's cheap only by refusing to charge when it shouldn't.

**Script**: `train_run5c.py`, identical to `train_run5b.py` except the quantile list.

**Promotion rule**: same as Run 5b — only promote if a candidate beats the current best
(`model_run5b_q30`, $6.33/month) on the $ metric, without materially worse amp exact-match or a
higher off-peak oscillation rate than the currently-deployed heuristic (4.46 flips/day) — that
guardrail is new here specifically because a low enough quantile could otherwise "win" on $ by
refusing to charge so often it effectively abandons the off-peak minimum-floor behavior the
heuristic's design intentionally trades a little cost for.

**Results (2026-09-06)**, same val set, evaluated with the guardrails added to
`evaluate_candidates.py` (off-peak/overall oscillation rate now computed for every candidate):

| Candidate | Watts MAE | ALL exact% | ALL \|err\| | $/month | $ vs heuristic | off_pk flips/day |
|---|---|---|---|---|---|---|
| heuristic | 387.0 W | 66.9% | 1.13 | 50.93 | — | **4.46** |
| model_run5b_q30 | 383.4 W | 66.9% | 1.11 | 44.60 | +6.33 | **4.36** |
| model_run5c_q25 | 413.3 W | 65.6% | 1.19 | 42.88 | +8.05 | 4.52 |
| model_run5c_q20 | 452.3 W | 63.0% | 1.31 | 40.65 | +10.28 | 5.15 |
| model_run5c_q15 | 525.2 W | 60.4% | 1.51 | 37.83 | +13.10 | 5.73 |
| model_run5c_q10 | 615.5 W | 57.7% | 1.76 | 34.38 | +16.55 | 6.63 |

**The guardrail caught exactly the failure mode it was designed for.** $/month keeps improving
monotonically all the way down to `q=0.10` (+$16.55/month, more than 4x `q30`'s already-surprising
number) — a naive "just optimize the $ metric" search would keep pushing the quantile lower
without limit. But watts MAE, amp exact-match, and mean amp error all degrade monotonically in
the same direction, and **off-peak oscillation exceeds the heuristic's own 4.46 flips/day at
every tested point except `q30` itself** (`q25` already crosses over to 4.52). This confirms the
$ simulation doesn't fully price in the value of actually using available solar — an
aggressive-enough bias "wins" partly by refusing to charge so often it stops mattering whether a
forecast is good, which is not a real improvement, just an unmodeled cost showing up as a free
lunch. **`q=0.3` is the only point across the full 0.10-0.50 sweep that both beats the heuristic
substantially on $ and stays at or below its real-world oscillation rate** — not a coincidence
that it's also close to the crossover point between `q25` (4.52, just above heuristic) and where
oscillation starts climbing sharply.

**Decision: stop the sweep here.** Chasing a finer optimum between 0.25-0.30 risks overfitting
hyperparameter choice to this specific 52-day validation window's idiosyncrasies rather than
finding a genuinely better setting — `q=0.3` already satisfies every guardrail cleanly and by a
comfortable margin. **`model_run5b_q30` is promoted to `model_run5.joblib`** as this project's
new best deployment candidate, superseding `model_run4.joblib`.

**Final comparison, all Run 5 work** (best-of-run 5a and 5b/5c vs. Run 4 and the heuristic):

| | Watts MAE | $/month vs. heuristic | off_pk flips/day (heuristic: 4.46) |
|---|---|---|---|
| model_run4 | 362.3 W | +1.90 | 3.15 (quieter, but leaves most of the $ opportunity on the table) |
| model_run5a (more data alone) | 356.1 W | +1.47 | 3.06 |
| **model_run5 = model_run5b_q30 (promoted)** | 383.4 W | **+6.33** | 4.36 |

Model_run5 has a *worse* watts MAE than model_run4 or model_run5a — the headline finding of this
entire Run 5 investigation: **once amp-bucket rounding and asymmetric NEM 3.0 rates are in the
loop, watts MAE and dollar impact are not just imperfectly correlated, they can point in opposite
directions.** More training data alone (Run 5a) did not close the amp/dollar gap found in shadow
mode; an asymmetric loss tuned against the real decision policy did, substantially.

**Not yet done**: `model_run5.joblib` needs the same deployment treatment `model_run4.joblib`
got (copy into `chargepoint-sunpower-chargemanager`, update `model_shadow_logger.py`'s default
model path or its systemd config, live-validate) before this becomes more than an offline
result — offline $ simulation is not a substitute for live validation, per this project's own
established convention (`[[feedback_live_validation_window]]`). Feature list/order and joblib
bundle shape are unchanged from Run 4, so `model_shadow_logger.py`'s `build_feature_frame()`
needs no changes — only the model file itself and which path the service loads.

---

## model_run5 deployment (2026-09-06/07)

`model_run5.joblib` copied into `chargepoint-sunpower-chargemanager`,
`model_shadow_logger.py`'s `--model-path` default changed to `model_run5.joblib`,
`MODEL_RUN5_README.md` written (full methodology/results summary for that repo),
`MODEL_RUN4_README.md` marked superseded with a pointer forward. Verified structurally
end-to-end against the live Pi InfluxDB before handoff (loads cleanly, all 30 features present,
correctly gated off during nighttime). User committed (`80a416c`) and deployed
(`model_shadow_logger.service` restarted 2026-09-06 20:20:11 PDT).

**Post-deploy verification — false alarm, corrected.** Reconstructed the exact feature vector
for a logged shadow prediction (2026-09-06 19:05:59 UTC, `model_excess_watts=2815.17W`) and
compared against both models: `model_run4.joblib` reproduced the logged value almost exactly
(2815.17W) while `model_run5.joblib` did not (2834.37W) — appeared to show the deployed service
was still running the old model. **Root cause: that test point was from 12:05 PDT, over 8 hours
*before* the actual service restart (20:20:11 PDT)** — at that timestamp `model_run4` genuinely
was what was running, so the match was expected, not a deployment failure. Confirmed via
`git log -1` on the Pi (commit `80a416c`, the model_run5 swap), `grep` on
`model_shadow_logger.py` (`--model-path` defaults to `model_run5.joblib`), and
`systemctl status` (service active, restarted recently) that the deployment itself is correct.

**Outstanding**: no `solar_charge_shadow` points have been written since the restart yet
(nighttime — the daytime-only gate correctly suppresses logging, nothing to log until sunrise).
A proper post-restart live cross-check (reconstruct a daytime tick logged *after* 2026-09-06
20:20:11 PDT and confirm it matches `model_run5`'s predictions, not `model_run4`'s) is still
needed once real daylight data accumulates — not yet done as of this entry. Once confirmed, the
live-validation clock for `model_run5` (per `[[feedback_live_validation_window]]`, want 3+
independent windows before drawing conclusions) starts from the first genuine post-restart
daytime prediction, not from the commit/restart time itself.
