# Inference_InfluxDB_Writer_track_a.py — Model 5c Track A (TFT), FP32 only
#
# Track A's Variable Selection Network uses a 4D tf.BatchMatMulV2 einsum that the TFLite
# converter cannot handle — this is a permanent, established limitation (see MODEL_5C
# project notes), not a bug to work around. There is no .tflite export for any Track A
# checkpoint and there never can be one, so unlike Inference_InfluxDB_Writer.py (Track B),
# this script loads the raw Keras model directly (architecture reconstructed from
# train_model_tft_track_a.py's module-level layer classes + best_model.weights.h5) and
# runs FP32 inference via plain TF/Keras — no TFLite, no INT8, no Edge TPU, ever.
#
# Architecture and hyperparameters (D_MODEL, N_HEADS, DROPOUT_RATE) are imported directly
# from train_model_tft_track_a.py so they can never drift out of sync with training. SEQ_LEN
# is NOT imported from that module — its module-level SEQ_LEN constant was edited in place
# for a later "Track A Deep" run (180→360) and no longer reflects Run 6's checkpoint. This
# script hardcodes SEQ_LEN=180, verified correct for Run 6 by reproducing its claimed offline
# MAE (0.0028/0.0050/0.0075 C) on the full validation population before this script was written
# (see verify_track_a_fp32.py). If a future Track A run uses a different SEQ_LEN, this constant
# must be updated to match, and the same verification should be re-run first.
#
# Track A's 24-feature set has no lag/diff scalars at all (temp_lag*, temp_diff_vs_*,
# pressure_lag*, humidity_lag*) — TFT attends to the raw sequence directly and was never given
# those as explicit inputs, unlike Track B/Model 5f. It DOES use six features Track B dropped
# for floor importance: wind_avg, wind_gust, wind_direction_sin/cos, wind_lull, rain_accumulated
# — the InfluxDB query below fetches all of them.
#
# IMPORTANT: humidity_slope_30 uses rolling window=30 here, matching
# train_model_tft_track_a.py line 524 exactly. Track B's Inference_InfluxDB_Writer.py uses
# window=60 for the same-named feature — a real, confirmed discrepancy between the two tracks,
# not a typo. Do not "fix" this to match Track B.

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
import argparse

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model_tft_track_a import (
    GatedResidualNetwork, SinusoidalPositionalEncoding, VariableSelectionNetwork,
    D_MODEL, N_HEADS, DROPOUT_RATE,
)

SEQ_LEN = 180  # Run 6's SEQ_LEN — see header comment. Verify before pointing at a different run.

QUERY_BATCH_SIZE = 250_000   # minutes of weather data per batch (~55 days)
WARM_UP_MINUTES  = 240       # SEQ_LEN(180) + longest slope window(60) + buffer
BACKFILL_HOURS   = 3         # re-run last N hours to add newly-available actuals
EXTRA_SAMPLES    = 250

WRITE_EVERY   = 2_000    # flush to InfluxDB every N predictions
RESTART_EVERY = 100_000  # full process restart (keeps memory bounded over long backfills)
PREDICT_CHUNK = 64       # windows per model() call — attention memory scales as
                          # batch*heads*seq_len^2, so this matters more than it would for a
                          # Dense model. 512 caused ~1.06GB single allocations (batch=512,
                          # 8 heads, 180x180 attention scores) that tripped TF's "exceeds 10%
                          # of free system memory" warning on a Pi; 64 keeps that under ~137MB.
                          # Raise this only if the target machine has RAM headroom to spare.


def _exit_restart_for_wrapper():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(88)


def rolling_slope(data, window):
    """Vectorised linear-regression slope over a sliding window. Identical to training."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    slopes = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    x_c = x - x.mean()
    denom = np.sum(x_c ** 2)
    shape = (n - window + 1, window)
    strides = (data.strides[0], data.strides[0])
    wins = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    has_nan = np.any(np.isnan(wins), axis=1)
    y_c = wins - np.nanmean(wins, axis=1, keepdims=True)
    s = np.sum(x_c * y_c, axis=1) / denom
    s[has_nan] = np.nan
    slopes[window - 1:] = s
    return slopes


def sanity_filter_temperature(df, label, window="31min", threshold_c=6.0):
    """Nulls sensor-glitch temperature spikes — matches the training pipeline exactly.
    Not present in Track B's live script (an established gap there, not copied here)."""
    local_median = df["temperature"].rolling(window, center=True, min_periods=3).median()
    spike = (df["temperature"] - local_median).abs() > threshold_c
    n_spikes = int(spike.sum())
    if n_spikes:
        df.loc[spike, "temperature"] = np.nan
        print(f"warn {label}: nulled {n_spikes} temperature sensor-glitch rows "
              f"(>{threshold_c}C from local {window} median)")
    return df


def _future_temp_at(df_index, temperatures, offset, tol_s=90):
    """Future temperature at (t + offset) via forward merge_asof — for backfilling actuals."""
    tol = pd.Timedelta(seconds=tol_s)
    base = pd.DataFrame({"base_time": df_index})
    base["lookup_time"] = base["base_time"] + offset
    src = pd.DataFrame({"src_time": df_index, "temperature": temperatures})
    merged = pd.merge_asof(base, src, left_on="lookup_time", right_on="src_time",
                           direction="forward", tolerance=tol)
    return merged["temperature"].to_numpy(dtype=np.float32)


def create_influx_points(base_timestamp, actuals, diffs_c, current_temp):
    points = []
    for hrs, diff_c, ai in [(1, diffs_c[0], 0), (2, diffs_c[1], 1), (3, diffs_c[2], 2)]:
        future_ts = base_timestamp + pd.Timedelta(hours=hrs)
        pred_temp = float(current_temp) + float(diff_c)
        fields = {f"pred_{hrs}hr_temperature": pred_temp}
        if actuals is not None and not np.isnan(actuals[ai]):
            fields[f"actual_{hrs}hr_temperature"] = float(actuals[ai])
        points.append({"measurement": MEASUREMENT, "time": future_ts.isoformat(), "fields": fields})
    return points


def build_model(n_features):
    input_layer = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")
    vsn_layer = VariableSelectionNetwork(n_features, D_MODEL, dropout=DROPOUT_RATE, name="vsn")
    vsn_out, _vsn_weights = vsn_layer(input_layer)
    enc = SinusoidalPositionalEncoding(D_MODEL, max_len=SEQ_LEN + 1, name="pos_enc")(vsn_out)
    mha_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=N_HEADS, key_dim=D_MODEL // N_HEADS, dropout=DROPOUT_RATE, name="temporal_attention")
    attn_out, _attn_scores = mha_layer(query=enc, key=enc, value=enc, return_attention_scores=True)
    grn_attn = GatedResidualNetwork(D_MODEL, dropout=DROPOUT_RATE, name="grn_post_attn")
    attn_out = grn_attn(enc + attn_out)
    grn_ff = GatedResidualNetwork(D_MODEL, dropout=DROPOUT_RATE, name="grn_feedforward")
    ff_out = grn_ff(attn_out)
    last_ts = ff_out[:, -1, :]
    out_1 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_1hr")(last_ts)
    out_2 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_2hr")(last_ts)
    out_3 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_3hr")(last_ts)
    return tf.keras.Model(inputs=input_layer, outputs=[out_1, out_2, out_3], name="tft_live")


# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Model 5c Track A (TFT) FP32 inference and write to InfluxDB.")
parser.add_argument("--run", default="run6",
    help="Trained run to serve, e.g. run6 (selects ./results_5c_<run>). Default: run6.")
parser.add_argument("--location", choices=["sf", "ps"], default="sf",
    help="Weather data source: sf (San Francisco, default) or ps (Palm Springs).")
args = parser.parse_args()

RUN_NAME    = args.run
RESULTS_DIR = f"./results_5c_{RUN_NAME}"
WEIGHTS_PATH       = f"{RESULTS_DIR}/checkpoints/best_model.weights.h5"
INPUT_SCALER_PATH  = f"{RESULTS_DIR}/input_scaler_5c.json"
TARGET_SCALER_PATH = f"{RESULTS_DIR}/target_scaler_5c.json"

LOCATIONS = {
    "sf": {"db": "weather",         "measurement": f"model_5c_tft_{RUN_NAME}",    "label": "San Francisco"},
    "ps": {"db": "ps_smartweather", "measurement": f"model_5c_tft_{RUN_NAME}_ps", "label": "Palm Springs"},
}
_loc = LOCATIONS[args.location]
DB, MEASUREMENT = _loc["db"], _loc["measurement"]
print(f"Location: {_loc['label']} | DB: {DB} | Measurement: {MEASUREMENT}")

# ---------------------------------------------------------------------------
# Load scalers + build/load model
# ---------------------------------------------------------------------------
with open(INPUT_SCALER_PATH) as f:
    input_scaler = json.load(f)
with open(TARGET_SCALER_PATH) as f:
    target_scaler = json.load(f)
y_min, y_max = float(target_scaler["min"]), float(target_scaler["max"])
FEATURE_ORDER = list(input_scaler.keys())
FEATURE_COUNT = len(FEATURE_ORDER)
print(f"Target diff range: {y_min:.2f}C to {y_max:.2f}C")
print(f"Features ({FEATURE_COUNT}): {', '.join(FEATURE_ORDER)}")

print(f"Building TFT (SEQ_LEN={SEQ_LEN}, D_MODEL={D_MODEL}, N_HEADS={N_HEADS}) and loading weights...")
model = build_model(FEATURE_COUNT)
model.load_weights(WEIGHTS_PATH)
print("Weights loaded.")

# ---------------------------------------------------------------------------
# Connect to InfluxDB
# ---------------------------------------------------------------------------
reader_client = InfluxDBClient(host="10.0.1.188", port=8086,
                               username="admin", password="24planet", database=DB)
writer_client = InfluxDBClient(host="localhost", port=8086,
                               username="admin", password="24planet", database=DB)

print("Checking InfluxDB for last prediction timestamp...")
last_ts = None
try:
    res = writer_client.query(f'SELECT LAST("pred_1hr_temperature") FROM "{MEASUREMENT}"')
    pts = list(res.get_points())
    if pts:
        last_ts = pts[0]["time"]
        print(f"  Resuming; last pred_1hr_temperature at {last_ts}")
    else:
        print("  No previous predictions; starting from the beginning.")
except Exception as e:
    print(f"  Could not query InfluxDB: {e}; starting from beginning.")

# ---------------------------------------------------------------------------
# Fetch data from InfluxDB — includes wind_lull/rain_accumulated, which
# Track B's script never needed (dropped for floor importance there).
# ---------------------------------------------------------------------------
raw_fields = ("temperature, relative_humidity, station_pressure, "
              "solar_radiation, illuminance, uv, wind_avg, wind_gust, wind_direction, "
              "wind_lull, rain_accumulated")

if last_ts:
    last_input_ts = pd.to_datetime(last_ts) - pd.Timedelta(hours=1)
    query_from = last_input_ts - pd.Timedelta(minutes=WARM_UP_MINUTES + BACKFILL_HOURS * 60)
    query_to   = last_input_ts + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
    print(f"  Query window: {query_from.isoformat()} to {query_to.isoformat()}")
    query = (f'SELECT {raw_fields} FROM "wf/obs_st" '
             f"WHERE time >= '{query_from.isoformat()}' AND time <= '{query_to.isoformat()}'")
else:
    first_res = reader_client.query('SELECT FIRST(temperature) FROM "wf/obs_st"')
    first_pts = list(first_res.get_points())
    if not first_pts:
        raise RuntimeError("No data in 'wf/obs_st'.")
    start_ts_dt = pd.to_datetime(first_pts[0]["time"])
    query_to    = start_ts_dt + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
    query_from  = start_ts_dt
    last_input_ts = None
    print(f"  Full backfill from {start_ts_dt.isoformat()}")
    query = (f'SELECT {raw_fields} FROM "wf/obs_st" '
             f"WHERE time >= '{start_ts_dt.isoformat()}' AND time <= '{query_to.isoformat()}'")

end_ts = query_to
result = reader_client.query(query)
points = list(result.get_points())
print(f"Fetched {len(points)} points from InfluxDB")
if not points:
    print("No data in query window - nothing to do.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Build DataFrame and derived features
# ---------------------------------------------------------------------------
df = pd.DataFrame(points)
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)
df.sort_index(inplace=True)

df = sanity_filter_temperature(df, "live")

tod = df.index.hour + df.index.minute / 60.0
doy = df.index.dayofyear.astype(float)
df["time_of_day_sin"]  = np.sin(2 * np.pi * tod / 24.0)
df["time_of_day_cos"]  = np.cos(2 * np.pi * tod / 24.0)
df["time_of_day_sin2"] = np.sin(4 * np.pi * tod / 24.0)
df["time_of_day_cos2"] = np.cos(4 * np.pi * tod / 24.0)
df["day_of_year_sin"]  = np.sin(2 * np.pi * doy / 365.25)
df["day_of_year_cos"]  = np.cos(2 * np.pi * doy / 365.25)
if "wind_direction" in df.columns:
    df["wind_direction_sin"] = np.sin(2 * np.pi * df["wind_direction"] / 360.0)
    df["wind_direction_cos"] = np.cos(2 * np.pi * df["wind_direction"] / 360.0)

print("Computing rolling slope features...")
df["temp_slope_15"]     = rolling_slope(df["temperature"].values, 15)
df["temp_slope_30"]     = rolling_slope(df["temperature"].values, 30)
df["temp_slope_60"]     = rolling_slope(df["temperature"].values, 60)
df["solar_slope_30"]    = rolling_slope(df["solar_radiation"].values, 30)
df["humidity_slope_30"] = rolling_slope(df["relative_humidity"].values, 30)  # window=30, Track A's own value
df["pressure_slope_60"] = rolling_slope(df["station_pressure"].values, 60)
print("  Slopes computed")

temperatures_arr = df["temperature"].values.astype(np.float32)
print("Building future temperature targets...")
df["temp_t+1hr"] = _future_temp_at(df.index, temperatures_arr, pd.Timedelta(hours=1))
df["temp_t+2hr"] = _future_temp_at(df.index, temperatures_arr, pd.Timedelta(hours=2))
df["temp_t+3hr"] = _future_temp_at(df.index, temperatures_arr, pd.Timedelta(hours=3))
print(f"  Missing targets: {df[['temp_t+1hr','temp_t+2hr','temp_t+3hr']].isna().sum().to_dict()}")

print(f"\nRows before dropna: {len(df)}")
missing_feats = [f for f in FEATURE_ORDER if f not in df.columns]
if missing_feats:
    raise ValueError(f"Missing features: {missing_feats}\nAvailable: {sorted(df.columns.tolist())}")

df.dropna(subset=FEATURE_ORDER, inplace=True)
float_cols = df.select_dtypes(include=["float64"]).columns
if len(float_cols):
    df[float_cols] = df[float_cols].astype(np.float32)
print(f"Rows after dropna: {len(df)}")
print(f"Data range: {df.index.min()} to {df.index.max()}")

# ---------------------------------------------------------------------------
# Pre-normalise — no clip, matching train_model_tft_track_a.py exactly (that
# script does not clip inputs to [0,1], unlike Track B's pipeline).
# ---------------------------------------------------------------------------
_f_mins = np.array([float(input_scaler[f]["min"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_maxs = np.array([float(input_scaler[f]["max"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_denoms = _f_maxs - _f_mins
_f_denoms[_f_denoms == 0.0] = 1.0
scaled_data = (df[FEATURE_ORDER].values.astype(np.float32) - _f_mins) / _f_denoms

targets_arr = df[["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]].values.astype(np.float32)
temp_arr    = df["temperature"].values.astype(np.float32)
timestamps  = df.index

# ---------------------------------------------------------------------------
# Determine start index
# ---------------------------------------------------------------------------
start_index = 0
if last_input_ts is not None:
    try:
        backfill_start = last_input_ts - pd.Timedelta(hours=BACKFILL_HOURS)
        start_index = int(df.index.searchsorted(backfill_start))
    except Exception:
        print("Could not compute backfill start - starting from row 0.")
        start_index = 0

if start_index >= len(df):
    print("No new data - predictions are up to date.")
    sys.exit(0)

print(f"\nStarting inference at index {start_index} / {len(df)-1} "
      f"({timestamps[start_index] if start_index < len(df) else 'N/A'})")
print(f"CPU/GPU FP32 (Keras, no TFLite) inference, {FEATURE_COUNT} features, SEQ_LEN={SEQ_LEN}")

# ---------------------------------------------------------------------------
# Inference loop — chunked/batched (unlike the TFLite scripts' one-row-at-a-time
# loop) since a real attention model has enough per-call overhead that batching
# meaningfully matters for backfill throughput.
# ---------------------------------------------------------------------------
print("Running inference...")
run_count = 0
prediction_points = []
_timing_reported = False

valid_start = max(start_index, SEQ_LEN - 1)
for chunk_start in range(valid_start, len(df), PREDICT_CHUNK):
    chunk_end = min(chunk_start + PREDICT_CHUNK, len(df))
    idxs = list(range(chunk_start, chunk_end))
    windows = np.stack([scaled_data[i - SEQ_LEN + 1:i + 1] for i in idxs]).astype(np.float32)

    t0 = time.perf_counter()
    try:
        out_1, out_2, out_3 = model(windows, training=False)
        out_1 = out_1.numpy().reshape(-1)
        out_2 = out_2.numpy().reshape(-1)
        out_3 = out_3.numpy().reshape(-1)
    except Exception as e:
        print(f"Inference error on chunk starting at row {chunk_start}: {e}")
        continue
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not _timing_reported:
        _timing_reported = True
        print(f"  First chunk: {len(idxs)} windows in {elapsed_ms:.1f}ms "
              f"({elapsed_ms/len(idxs):.2f}ms/window)")

    diffs_norm = np.stack([out_1, out_2, out_3], axis=1)  # (n, 3)
    diffs_c_all = 0.5 * (diffs_norm + 1.0) * (y_max - y_min) + y_min

    for j, i in enumerate(idxs):
        targets_row = targets_arr[i]
        actuals = None if np.any(np.isnan(targets_row)) else targets_row
        current_temp = float(temp_arr[i])
        prediction_points.extend(
            create_influx_points(timestamps[i], actuals, diffs_c_all[j], current_temp))
        run_count += 1

    if len(prediction_points) >= WRITE_EVERY:
        print(f"  {run_count:,} predictions | last chunk {elapsed_ms/len(idxs):.2f}ms/window")
        try:
            writer_client.write_points(prediction_points, time_precision="ms")
        except Exception as e:
            print(f"  InfluxDB write error: {e}")
        prediction_points = []

    if run_count % RESTART_EVERY < PREDICT_CHUNK and run_count >= RESTART_EVERY:
        print("Restarting to bound memory growth over long backfills...")
        if prediction_points:
            writer_client.write_points(prediction_points, time_precision="ms")
        _exit_restart_for_wrapper()

if prediction_points:
    try:
        writer_client.write_points(prediction_points, time_precision="ms")
    except Exception as e:
        print(f"Final InfluxDB write error: {e}")

print(f"Inference complete. {run_count:,} predictions written to '{MEASUREMENT}'.")

end_ts_dt = pd.to_datetime(end_ts)
if end_ts_dt.tzinfo is None:
    end_ts_dt = end_ts_dt.tz_localize("UTC")
now_utc = pd.Timestamp.now(tz="UTC")
if end_ts_dt < now_utc - pd.Timedelta(hours=1):
    print(f"Query window ended at {end_ts_dt} (past) - restarting for next batch...")
    _exit_restart_for_wrapper()

sys.stdout.flush()
sys.stderr.flush()
