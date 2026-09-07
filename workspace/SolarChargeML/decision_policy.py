"""
Shared TOU/amperage decision policy + PG&E cost model for evaluating SolarChargeML candidates
on the metric that actually matters — amp decisions and dollars, not just watts MAE.

The threshold/rounding logic mirrors solar_charge_controller.py exactly (given verbatim in the
Run 5 brief); it deliberately omits that script's additional start/stop hysteresis (the
"don't start a new session below minimum, but let an already-charging one continue" rule),
since that depends on session state this offline harness doesn't have — same simplification
already documented in model_shadow_logger.py's compute_model_target_amps(). This only affects
the absolute amp-accuracy numbers at the margin, not the relative comparison between candidates,
since every candidate (ideal/heuristic/model) is scored through the identical simplified policy.
"""
import numpy as np
import pandas as pd
import pytz

_PACIFIC = pytz.timezone("America/Los_Angeles")

MIN_AMPERAGE = 8
MAX_AMPERAGE = 40
ALLOWED_AMPS = list(range(MIN_AMPERAGE, MAX_AMPERAGE + 1))
VOLTAGE = 240
MINIMUM_WATTS_REQUIRED = (MIN_AMPERAGE - 0.5) * VOLTAGE  # 1800W

# PG&E-only marginal rates ($/kWh), derived from the user's real bill (~/pge/estimate_bill.py
# has the itemized source components; these are the already-combined marginal $/kWh figures).
RATES = {
    "Summer": {"peak": 0.3402, "part_peak": 0.2767, "off_peak": 0.2649, "export_credit": 0.0531},
    "Winter": {"peak": 0.2700, "part_peak": 0.2678, "off_peak": 0.2673, "export_credit": 0.0239},
}
SUMMER_MONTHS = {6, 7, 8, 9}


def get_tou_period(hour):
    """Vectorizable version of solar_charge_controller.py's get_tou_period(), taking an
    already-computed Pacific-local hour instead of calling datetime.now() itself."""
    if 16 <= hour < 21:
        return "peak"
    if hour < 15:
        return "off_peak"
    return "part_peak"


def get_tou_excess_threshold(base_minimum_watts, tou_period):
    if tou_period == "peak":
        return base_minimum_watts * 2.0
    if tou_period == "off_peak":
        return -500.0
    return base_minimum_watts


def determine_target_amperage(excess_w, allowed_amps=ALLOWED_AMPS, voltage=VOLTAGE):
    if excess_w <= 0:
        return 0
    ideal = max(excess_w / voltage, min(allowed_amps) - 0.5)
    possible = [a for a in allowed_amps if a >= ideal]
    return min(possible) if possible else max(allowed_amps)


def decide_amp(excess_w, tou_period):
    """Full policy: TOU-threshold gate, then amp rounding. Matches the Run 5 brief's given
    pseudocode exactly (no start/stop hysteresis — see module docstring)."""
    threshold = get_tou_excess_threshold(MINIMUM_WATTS_REQUIRED, tou_period)
    if excess_w < threshold:
        return 0
    return determine_target_amperage(excess_w)


def add_tou_and_season(df):
    """Add tou_period/season columns computed from each row's own Pacific-local timestamp —
    NOT solar_charge_controller.py's get_tou_period(), which calls datetime.now() and is only
    meaningful for live use. df.index must be a UTC DatetimeIndex."""
    local = df.index.tz_convert(_PACIFIC)
    df = df.copy()
    df["tou_period"] = [get_tou_period(h) for h in local.hour]
    df["season"] = np.where(local.month.isin(SUMMER_MONTHS), "Summer", "Winter")
    return df


def amp_decisions(excess_w, tou_period):
    """Vectorized decide_amp over arrays/Series."""
    excess_w = np.asarray(excess_w, dtype=float)
    tou_period = np.asarray(tou_period)
    return np.array([decide_amp(e, t) for e, t in zip(excess_w, tou_period)])


def simulate_cost(amp, actual_excess_w, tou_period, season, dt_hours):
    """Per-row grid/solar energy split and $ cost for a chosen amp against the REAL excess
    that occurred (not whatever was predicted) — matches the Run 5 brief's formula exactly.
    Vectorized; all args are same-length arrays/Series except dt_hours (scalar)."""
    amp = np.asarray(amp, dtype=float)
    actual_excess_w = np.asarray(actual_excess_w, dtype=float)
    amp_w = amp * VOLTAGE
    grid_kwh = np.maximum(0, amp_w - np.maximum(actual_excess_w, 0)) / 1000.0 * dt_hours
    solar_kwh = np.minimum(amp_w, np.maximum(actual_excess_w, 0)) / 1000.0 * dt_hours

    import_rate = np.array([RATES[s][t] for s, t in zip(season, tou_period)])
    export_credit = np.array([RATES[s]["export_credit"] for s in season])
    cost = grid_kwh * import_rate + solar_kwh * export_credit
    return cost, grid_kwh, solar_kwh
