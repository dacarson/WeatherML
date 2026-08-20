from influxdb import InfluxDBClient
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

HOST = "10.0.1.188"
PORT = 8086
USER = "admin"
PASS = "24planet"

# sunpower_power's first recorded sample (confirmed via FIRST(pv_p) on 2026-08-19)
FETCH_START = datetime(2025, 4, 29, tzinfo=timezone.utc)
FETCH_END = datetime.now(timezone.utc)

# Fetch in day-aligned chunks to stay under InfluxDB's max-select-point limit, matching
# the convention in ../export_influx_to_csv.py. Chunk boundaries land exactly on BIN-width
# GROUP BY bin edges (chunk_days is always a whole number of days, and BIN divides evenly into
# a day), so concatenating chunks produces no duplicate or missing bins at the seams.
CHUNK_DAYS = 60

# Bin width for both the InfluxDB GROUP BY time() clause and the pandas resample grid. 30s
# (not the original 1m, and not raw ~1-5s samples) is a middle ground: still enough raw pv_p
# samples per bin (~15-30) for meaningful STDDEV/range, doubles row count vs. 1m (still small),
# and gives a fresher "current state" + room for a genuinely fast slope feature that a 1-minute
# grid can't resolve at all — see SOLARCHARGE_EXPERIMENT_LOG.md Run 4.
BIN = "30s"

HORIZON_MIN = 5  # matches solar_charge_controller.py's deployed --control-interval
HORIZON_STEPS = int(pd.Timedelta(minutes=HORIZON_MIN) / pd.Timedelta(BIN))

TRAIN_END = pd.Timestamp("2026-06-30T23:59:59Z")
VAL_START = pd.Timestamp("2026-07-01T00:00:00Z")


def fetch_binned_stats(db, measurement, field_aggs, start, end, chunk_days=CHUNK_DAYS, bin_width=BIN):
    """Query BIN-width aggregates from `measurement`, chunked over time.

    `field_aggs` maps each field to a list of InfluxDB aggregate functions, e.g.
    {"pv_p": ["MEAN", "STDDEV", "MIN", "MAX"]}. All aggregates for a field come from one
    query (one raw-data scan), not one query per function. Output columns are named after
    the bare field for MEAN (backwards compatible with the original mean-only export) and
    "{field}_{agg}" (lowercased) for every other aggregate, e.g. "pv_p_std", "pv_p_min".
    """
    client = InfluxDBClient(host=HOST, port=PORT, username=USER, password=PASS, database=db)

    suffixes = {"MEAN": "", "STDDEV": "_std", "MIN": "_min", "MAX": "_max"}
    out_cols = []
    select_parts = []
    for field, aggs in field_aggs.items():
        for agg in aggs:
            col = f"{field}{suffixes[agg.upper()]}"
            select_parts.append(f'{agg.upper()}("{field}") AS "{col}"')
            out_cols.append(col)
    select_clause = ", ".join(select_parts)

    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        start_str = chunk_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = chunk_end.strftime('%Y-%m-%dT%H:%M:%SZ')
        query = (
            f'SELECT {select_clause} FROM "{measurement}" '
            f"WHERE time >= '{start_str}' AND time < '{end_str}' "
            f"GROUP BY time({bin_width}) fill(null)"
        )
        print(f"  [{db}.{measurement}] {start_str} -> {end_str}...")
        result = client.query(query)
        points = list(result.get_points())
        if points:
            chunks.append(pd.DataFrame(points))
        chunk_start = chunk_end
    client.close()

    if not chunks:
        return pd.DataFrame(columns=out_cols, index=pd.DatetimeIndex([], name="time", tz="UTC"))

    df = pd.concat(chunks, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])
    df = df.drop_duplicates(subset='time').set_index('time').sort_index()
    return df[out_cols]


print(f"Fetching PVS6 solar production / grid / site load ({BIN} stats)...")
# pv_p and site_load_p get STDDEV/MIN/MAX in addition to MEAN: sub-minute volatility in
# production (cloud-edge transients) and load (unscheduled appliances switching on/off) is
# exactly the structure a 1-min MEAN alone throws away — see SOLARCHARGE_EXPERIMENT_LOG.md Run 3.
pv = fetch_binned_stats(
    "pvs6", "sunpower_power",
    {"pv_p": ["MEAN", "STDDEV", "MIN", "MAX"],
     "net_p": ["MEAN"],
     "site_load_p": ["MEAN", "STDDEV", "MIN", "MAX"]},
    FETCH_START, FETCH_END,
)
pv *= 1000.0  # kW -> W (all columns here are power values or linear functions of them)

print("Fetching solar_charge_control (heuristic's own predictions + EV charging power)...")
ctrl = fetch_binned_stats(
    "pvs6", "solar_charge_control",
    {"excess_solar_watts": ["MEAN"],
     "charging_power_watts": ["MEAN"],
     "solar_slope_w_per_s": ["MEAN"]},
    FETCH_START, FETCH_END,
)

print(f"Fetching WeatherFlow Tempest observations ({BIN} means)...")
# wf/obs_st's native report_interval is 1 minute (confirmed 2026-08-19: exactly 1 raw sample
# per 1-min bin), so STDDEV/MIN/MAX would be undefined/redundant here — no sub-minute signal
# to extract from this source. That signal instead comes from pv_p/site_load_p above, which
# are sampled ~60x/minute. At BIN=30s (< the station's 60s cadence) every other bin is null
# right after the fetch — forward-filled below after reindexing.
wx = fetch_binned_stats(
    "weather", "wf/obs_st",
    {"solar_radiation": ["MEAN"],
     "illuminance": ["MEAN"], "uv": ["MEAN"], "wind_avg": ["MEAN"], "wind_gust": ["MEAN"],
     "wind_lull": ["MEAN"], "wind_direction": ["MEAN"], "relative_humidity": ["MEAN"],
     "station_pressure": ["MEAN"], "temperature": ["MEAN"], "rain_accumulated": ["MEAN"]},
    FETCH_START, FETCH_END,
)

# Reindex every series onto the same regular BIN-width grid before joining. GROUP BY time(BIN)
# fill(null) already does this per-measurement, but each measurement can have missing bins at
# different points (device offline, service restart, etc.) — reindexing onto one shared grid
# guarantees the join lines up bin-for-bin across all three sources.
full_index = pd.date_range(FETCH_START, FETCH_END, freq=BIN, tz="UTC", inclusive="left")
pv = pv.reindex(full_index)
# solar_charge_control is written once per control cycle (~60s) and holds a step-function value
# in between (the controller doesn't recompute until its next iteration) — at a 30s grid that
# means forward-filling, not leaving null. Without this, charging_power_watts.fillna(0.0) below
# would wrongly read "not freshly logged this bin" as "car not charging" on every other row.
# limit=20 (~10 min, well over one control cycle) still lets a genuine service outage surface
# as NaN rather than being carried forward indefinitely.
ctrl = ctrl.reindex(full_index).ffill(limit=20)
# wf/obs_st reports once every ~60s; at a 30s grid, forward-fill the alternating empty bin
# (limit=1, so a real gap in the station's own data still surfaces as NaN rather than being
# carried forward indefinitely) rather than losing every other row to dropna.
wx = wx.reindex(full_index).ffill(limit=1)

df = pv.join(ctrl, how="left").join(wx, how="left")
df.index.name = "time"

# EV charging load is included in site_load_p whenever the car was charging historically.
# Subtract it out to isolate the baseline (non-EV) house load the model needs to predict
# around — this is the quantity determine_target_amperage ultimately needs excess solar for.
df["baseline_house_load_w"] = df["site_load_p"] - df["charging_power_watts"].fillna(0.0)
df["excess_now_w"] = df["pv_p"] - df["baseline_house_load_w"]

# Prediction target: excess HORIZON_MIN minutes (HORIZON_STEPS bins) ahead. Safe to use a plain
# positional shift (not merge_asof) because full_index is a perfectly regular BIN-width grid by
# construction — unlike ../export_influx_to_csv.py's wf/obs_st join, there's no irregular
# sampling to guard against here.
df["excess_future_w"] = df["excess_now_w"].shift(-HORIZON_STEPS)

df["day_of_year"] = df.index.dayofyear
df["time_of_day"] = df.index.hour + df.index.minute / 60.0

# Required for every training row. excess_solar_watts (the heuristic's own historical
# prediction) is deliberately NOT required here — it's only used for backtest comparison on
# whatever subset of rows has it, and gaps in the controller's own logging (service restarts,
# etc.) shouldn't discard otherwise-good training rows.
required_fields = [
    "pv_p", "pv_p_std", "pv_p_min", "pv_p_max",
    "net_p", "site_load_p", "site_load_p_std", "site_load_p_min", "site_load_p_max",
    "baseline_house_load_w", "excess_now_w",
    "solar_radiation", "illuminance", "uv", "wind_avg", "wind_gust", "wind_lull",
    "wind_direction", "relative_humidity", "station_pressure", "temperature",
    "rain_accumulated", "day_of_year", "time_of_day", "excess_future_w",
]
before = len(df)
df_clean = df.dropna(subset=required_fields)
print(f"Dropped {before - len(df_clean)} / {before} rows with missing required fields "
      f"({(before - len(df_clean)) / before:.1%}).")

n_with_baseline = df_clean["excess_solar_watts"].notna().sum()
print(f"{n_with_baseline} / {len(df_clean)} rows also have the heuristic's own "
      f"excess_solar_watts logged (usable for backtest comparison).")

train_df = df_clean[df_clean.index <= TRAIN_END]
val_df = df_clean[df_clean.index >= VAL_START]

train_df.to_csv("train_data.csv")
val_df.to_csv("val_data.csv")

print(f"Wrote {len(train_df)} training rows to train_data.csv "
      f"({train_df.index.min()} -> {train_df.index.max()})")
print(f"Wrote {len(val_df)} validation rows to val_data.csv "
      f"({val_df.index.min()} -> {val_df.index.max()})")
