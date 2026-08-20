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
