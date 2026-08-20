# SolarChargeML — Project Plan

**Goal**: Replace the excess-solar prediction step inside
[ChargePoint-SunPower-ChargeManager](https://github.com/dacarson/ChargePoint-SunPower-ChargeManager)'s
`solar_charge_controller.py` with a trained model, so the EV charger tracks true excess solar more
accurately than the current average+slope heuristic — biasing toward grid draw off-peak and toward
export on-peak, through unscheduled house loads and passing clouds/fog.

---

## 1. Current system (baseline to beat)

`solar_charge_controller.py` runs a control loop every `--control-interval` minutes (default 5):

1. Pulls `MEAN(pv_p)` and `MEAN(net_p)` from InfluxDB over the control window, plus a linear
   `DERIVATIVE(MEAN(pv_p))` ("slope") over a longer `--slope-window` (default 30 min).
2. `average_excess = pv_production - non_EV_house_load` (derived from `net_p` and the currently
   known EV charging watts).
3. `predicted_excess = average_excess + solar_slope * control_interval * 60` — i.e. naive linear
   extrapolation of the recent trend.
4. `predicted_excess` is compared against a time-of-use threshold (peak/part-peak/off-peak have
   different bias — see `get_tou_excess_threshold`) and converted to a target amperage via
   `determine_target_amperage` (round up to nearest allowed ChargePoint amperage step).
5. Everything downstream of `predicted_excess` (TOU thresholds, hysteresis, session
   start/stop/amperage-change logic, manual-override detection) is hand-tuned and works — **not**
   in scope to replace.

**This project's scope is narrow and deliberate: produce a better `predicted_excess_watts` for a
fixed horizon, as a drop-in replacement for step 3 above.** Nothing else in the control loop changes.

Useful side effect: the controller has been logging its own `excess_solar_watts` (its
`predicted_excess`) to InfluxDB (`solar_charge_control` measurement) since 2025-04-30. That's a
ready-made historical baseline — the ML model can be backtested against the heuristic's actual
past predictions, not just against a re-derived formula.

---

## 2. Data sources (confirmed live, 2026-08-19)

Both InfluxDB databases live on the Pi at `10.0.1.188:8086`, reachable from this Mac (same as
WeatherML's existing `export_influx_to_csv.py` pattern).

### `pvs6` database

| Measurement | Fields | Range | Notes |
|---|---|---|---|
| `sunpower_power` | `pv_p`, `pv_en`, `net_p`, `net_en`, `site_load_p`, `site_load_en` (all float, kW/kWh) | 2025-04-29 → present, ~40.6M points | Published on every PVS6 websocket message (~1–5s cadence). `site_load_p` is *total* site consumption incl. any EV charging — must subtract EV load to get baseline house load (see §4). |
| `solar_charge_control` | `excess_solar_watts`, `solar_slope_w_per_s`, `charging_power_watts`, `target_amperage`, `current_amperage` | 2025-04-30 → present, ~528k points | Written by the controller once per control cycle. `charging_power_watts` covers the *entire* history — this is exactly what's needed to subtract EV load from `site_load_p` (see §4). `excess_solar_watts` is the heuristic's own historical prediction — the baseline to beat. |

### `weather` database (WeatherFlow Tempest, same property)

| Measurement | Fields | Range | Notes |
|---|---|---|---|
| `wf/obs_st` | `solar_radiation`, `illuminance`, `uv`, `wind_avg/gust/lull`, `wind_direction`, `humidity`, `relative_humidity`, `station_pressure`, `temperature`, `rain_accumulated`, `precipitation_type`, lightning fields | 2022-06 → present (co-located with the ~40.6M-point PV history from 2025-04-29 on) | ~60s report interval. `solar_radiation`/`illuminance`/`uv` are direct measured-irradiance signals — expected to be strong leading indicators of `pv_p`, and far more informative about clouds/fog than extrapolating `pv_p`'s own recent slope. Wind fields may help characterize cloud-transit speed; humidity/pressure trends may help flag fog. |
| `wf/obs_sky` | Same solar fields, older/retired station (data stops ~when `obs_st` — the combo Tempest — took over) | Legacy only; not needed given `obs_st` covers the full PV history window |

This directly answers open question #2 from the WeatherML memory pattern: this project **does**
incorporate WeatherFlow station data, not just PVS6 history, since decided with the user
2026-08-19.

---

## 3. Modeling framing (decided 2026-08-19)

- **Predict**: excess solar power (production − non-EV house load), N minutes ahead. A direct,
  minimal-blast-radius replacement for `predicted_excess` in `get_solar_power_status` /
  `main()`'s control loop — not an end-to-end amperage or TOU model.
- **Horizon**: 5 minutes, matching the controller's actual deployed `--control-interval` (confirmed
  2026-08-19 — the "next 15 min" framing in the original ask was approximate, not the real cycle
  time). Train and evaluate for this horizon; no need for a multi-horizon model given a single,
  fixed deployment cadence.
- **Inputs**: recent history of `pv_p`, `site_load_p` (EV-subtracted), `net_p`, plus WeatherFlow
  `solar_radiation`/`illuminance`/`uv`/`wind_*`/`humidity`/`station_pressure`/`temperature`, plus
  cyclic time-of-day / day-of-year features (sun position is deterministic and a strong prior on
  top of which weather perturbs).
- **Architecture**: TBD after EDA — likely start simple (gradient-boosted trees or a small dense
  net on engineered lag/slope features, in the spirit of Model 5d's flat-feature approach) before
  reaching for anything sequence-based like the Model 5c/5f TFT family. No evidence yet that this
  problem needs that complexity.

---

## 4. Known nuances / open problems to solve before training

1. **Isolating non-EV house load.** `site_load_p` includes EV charging power whenever the car was
   charging historically. `solar_charge_control.charging_power_watts` covers the full history and
   should be subtractable: `baseline_house_load = site_load_p - charging_power_watts` (aligned by
   timestamp, nearest-match since the two measurements are on different write cadences). Needs
   verification that `charging_power_watts` is accurate enough (it's itself sometimes an
   amperage-based *estimate* in the source script, not always a direct meter reading — see
   `get_current_charging_watts` in `solar_charge_controller.py`).
2. **Join strategy / resampling.** `sunpower_power` (~1-5s) and `wf/obs_st` (~60s) need a common
   resolution. Resample both to 1-minute bins (mean) before building lag features, matching the
   controller's own use of 5/30-min windows.
3. **Train/val/test split.** Must be chronological (no shuffling across time) to avoid leakage,
   consistent with every other WeatherML project.
4. **Evaluation baseline.** Backtest against the heuristic's own logged `excess_solar_watts`
   (same historical timestamps, same ground truth) — a fair, apples-to-apples MAE comparison, not
   just "beat a re-derived formula."
5. **Deployment path.** Once a model beats the heuristic offline, it needs a live-validation phase
   (per `[[feedback_live_validation_window]]` project convention: 3+ independent windows, not one)
   before being wired into `solar_charge_controller.py` for real. Deployment target is the same
   Raspberry Pi that already runs `solar_charge_controller.py`/`pvs6_ws_logger.py` as systemd
   services (confirmed 2026-08-19) — so the model needs to run inference on RPi CPU with low
   latency/footprint (a control cycle happens every 5 min). This favors a small model (gradient-
   boosted trees or a small dense net) over anything GPU-sized, and likely a TFLite/ONNX export
   step mirroring the Model 5-series INT8 export pipeline, rather than a raw framework checkpoint.

---

## 5. Next steps

1. ~~Build a data export/join script~~ — done (`export_and_join.py`). Pulls 1-min `MEAN()`
   aggregates of `sunpower_power` + `solar_charge_control` (`pvs6` db) and `wf/obs_st` (`weather`
   db) from `10.0.1.188:8086`, joins on a shared regular 1-minute UTC grid, computes
   `baseline_house_load_w = site_load_p - charging_power_watts`, `excess_now_w = pv_p -
   baseline_house_load_w`, and the `excess_future_w` target (5-min-ahead `excess_now_w`, plain
   positional shift since the grid is guaranteed regular). Chronological split at
   2026-06-30/07-01. Run 2026-08-19:
   - 671,433 clean rows after dropping 2.5% with missing required fields (688,392 → 671,433).
   - 600,714 train rows (2025-04-29 → 2026-06-30), 70,719 val rows (2026-07-01 → 2026-08-20).
   - 518,845 rows (77%) also carry the heuristic's own logged `excess_solar_watts` for backtest
     comparison — gaps are from periods the controller service wasn't running/logging.
   - `train_data.csv` / `val_data.csv` are gitignored (145MB/17MB) — re-run `export_and_join.py`
     to regenerate; not committed.
2. EDA sanity check (`sanity_check.py`, run 2026-08-19):
   - `corr(pv_p, solar_radiation) = 0.958` — confirms measured irradiance is a strong leading
     signal, as hypothesized in §2/§3.
   - `corr(excess_now_w, excess_future_w) = 0.915` at the 5-min horizon — persistence alone is
     already a strong baseline at this short a horizon. This raises the bar for what counts as a
     win: the real test is beating the heuristic's slope-extrapolation specifically (via the
     logged `excess_solar_watts` backtest), not just beating naive persistence.
   - `charging_power_watts` maxes out at exactly 9600W (40A × 240V, the charger's max amperage)
     and is nonzero on 6.1% of rows — sane distribution, confirms the EV-subtraction logic is
     being meaningfully exercised rather than silently zero.
3. Baseline model (Run 1) + backtest — done, see `SOLARCHARGE_EXPERIMENT_LOG.md` for the full
   run-by-run history. Runs 1-3 (1-minute grid, various target/feature reformulations) all landed
   within a few percent of the heuristic and of naive persistence — no clear win. Run 4
   (2026-08-20, finer 30-second grid + a 2-minute slope feature) broke out of that band:
   **`model_run4.joblib` beats the heuristic by 5.6% MAE and persistence by 11.1%** on a daytime
   backtest covering 99.9% of the validation window. First viable deployment candidate.
4. Live validation: **running** as of 2026-08-20 10:32 PDT. `model_shadow_logger.service` is
   enabled and stable on the Pi (`castropi`), writing fresh `solar_charge_shadow` points to
   InfluxDB (`pvs6` db) roughly every 60s during daytime. Deployment hit two issues, both fixed:
   (1) the Pi's venv initially lacked the new Python deps (`numpy` missing — `pip install` had
   been run outside the venv the first time); (2) `scikit-learn>=1.7.0` let pip install a newer
   version on the Pi than trained the model, breaking unpickling
   (`ModuleNotFoundError: No module named '_loss'` — pickled sklearn models are only
   version-portable to the exact training version) — fixed by pinning `scikit-learn==1.7.0`
   exactly in `requirements.txt` (committed 2026-08-20). Once 3+ independent windows have
   accumulated, per `[[feedback_live_validation_window]]`, compare `model_excess_watts` vs.
   `heuristic_excess_watts` in `solar_charge_shadow` before any change to
   `solar_charge_controller.py` itself.
