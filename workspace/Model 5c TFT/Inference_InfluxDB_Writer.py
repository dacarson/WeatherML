# Inference_InfluxDB_Writer.py — Model 5c Track B Dense
#
# Runs inference using the INT8 TFLite model on Coral Edge TPU (falls back to FP32
# TFLite on CPU when no TPU delegate is available, e.g. Mac development).
#
# Compatible with Runs 1–5 (SEQ_LEN=1, flat feature vector), Run 6-15 (SEQ_LEN=180,
# 3-hour sliding window + AveragePooling), and Run 16+ (adds temp_diff_vs_5hr/6hr
# scalar lag features). The sequence length and feature set are both detected
# automatically (SEQ_LEN from the model's input shape, features from input_scaler.json)
# — no code change needed when switching between runs, as long as every feature a
# given run's input_scaler.json references has a corresponding column computed below.
#
# Predictions are stored at the TARGET timestamp (T+1hr, T+2hr, T+3hr) so that
# Grafana shows them as a forward-looking fan from "now".

import time
import numpy as np
import json
import pandas as pd
from influxdb import InfluxDBClient
import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Configuration — update RUN_NAME to switch between trained runs
# ---------------------------------------------------------------------------
RUN_NAME    = "dense_b_run17"
RESULTS_DIR = f"./results_5c_trackb_{RUN_NAME}"

MODEL_EDGETPU_PATH = f"{RESULTS_DIR}/model_trackb_{RUN_NAME}_int8_edgetpu.tflite"
MODEL_INT8_PATH    = f"{RESULTS_DIR}/model_trackb_{RUN_NAME}_int8.tflite"
MODEL_FP32_PATH  = f"{RESULTS_DIR}/model_trackb_{RUN_NAME}_fp32.tflite"
INPUT_SCALER_PATH  = f"{RESULTS_DIR}/input_scaler_5c_trackb.json"
TARGET_SCALER_PATH = f"{RESULTS_DIR}/target_scaler_5c_trackb.json"

# ---------------------------------------------------------------------------
# Location profiles — controls which InfluxDB database to read/write
# ---------------------------------------------------------------------------
LOCATIONS = {
    "sf": {
        "db":          "weather",
        "measurement": f"model_5c_trackb_{RUN_NAME}",
        "label":       "San Francisco",
    },
    "ps": {
        "db":          "ps_smartweather",
        "measurement": f"model_5c_trackb_ps_{RUN_NAME}",
        "label":       "Palm Springs",
    },
}

QUERY_BATCH_SIZE  = 250_000   # minutes of weather data per batch (≈55 days)
WARM_UP_MINUTES   = 420      # lookback for 360-min temp_diff_vs_6hr lag (Run 16+) + 60-min buffer
BACKFILL_HOURS    = 3        # re-run last N hours to add newly-available actuals
EXTRA_SAMPLES     = 250      # extra buffer past batch window

WRITE_EVERY   = 12_500   # flush to InfluxDB every N predictions
RESTART_EVERY = 500_000  # full process restart (TPU state reset)

# ---------------------------------------------------------------------------
# TFLite runtime — prefer tflite_runtime (Pi/Coral), fall back to tensorflow
# ---------------------------------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    _TFLITE_RUNTIME = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exit_restart_for_wrapper():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(88)


def _build_delegate_options(tpu_arg: str):
    if not tpu_arg:
        return None
    tpu_arg = str(tpu_arg).strip()
    if not tpu_arg:
        return None
    device = tpu_arg if ":" in tpu_arg else f":{tpu_arg}"
    return {"device": device}


def _extract_interpreter_constants(in_det, out_det):
    idx      = in_det[0]["index"]
    i_scale, i_zp = in_det[0]["quantization"]
    o_indices = [out_det[k]["index"]             for k in range(3)]
    o_scales  = [out_det[k]["quantization"][0]   for k in range(3)]
    o_zps     = [out_det[k]["quantization"][1]   for k in range(3)]
    return idx, i_scale, i_zp, o_indices, o_scales, o_zps


def gap_aware_lag(series, lag_minutes, tol_s=90):
    """Time-based lag with NaN where the lag crosses a data gap.
    Identical to the training pipeline's gap_aware_lag."""
    lag_td = pd.Timedelta(minutes=lag_minutes)
    tol    = pd.Timedelta(seconds=tol_s)
    shifted_idx = series.index - lag_td
    lagged = series.reindex(shifted_idx, method="nearest", tolerance=tol)
    lagged.index = series.index
    return lagged


def rolling_slope(data, window):
    """Vectorised linear-regression slope over a sliding window.
    Identical to the training pipeline's rolling_slope."""
    data   = np.asarray(data, dtype=np.float64)
    n      = len(data)
    slopes = np.full(n, np.nan)
    x      = np.arange(window, dtype=np.float64)
    x_c    = x - x.mean()
    denom  = np.sum(x_c ** 2)
    shape  = (n - window + 1, window)
    strides = (data.strides[0], data.strides[0])
    wins   = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    has_nan = np.any(np.isnan(wins), axis=1)
    y_c    = wins - np.nanmean(wins, axis=1, keepdims=True)
    s      = np.sum(x_c * y_c, axis=1) / denom
    s[has_nan] = np.nan
    slopes[window - 1:] = s
    return slopes


def _future_temp_at(df_index, temperatures, offset, tol_s=90):
    """Return future temperature at (t + offset) via forward merge_asof.
    Matches the training pipeline's _add_future_targets."""
    tol = pd.Timedelta(seconds=tol_s)
    base = pd.DataFrame({"base_time": df_index})
    base["lookup_time"] = base["base_time"] + offset
    src  = pd.DataFrame({"src_time": df_index, "temperature": temperatures})
    merged = pd.merge_asof(base, src, left_on="lookup_time", right_on="src_time",
                           direction="forward", tolerance=tol)
    return merged["temperature"].to_numpy(dtype=np.float32)


def _past_temp_at(df_index, temperatures, offset, tol_s=90):
    """Return past temperature at (t - offset) via backward merge_asof.
    Matches the training pipeline's _add_past_lags (temp_diff_vs_5hr/6hr, Run 16+) —
    deliberately NOT gap_aware_lag's reindex(method="nearest"), which can match a
    reading slightly AFTER the lookup time; training only ever looks backward."""
    tol = pd.Timedelta(seconds=tol_s)
    base = pd.DataFrame({"base_time": df_index})
    base["lookup_time"] = base["base_time"] - offset
    src  = pd.DataFrame({"src_time": df_index, "temperature": temperatures})
    merged = pd.merge_asof(base, src, left_on="lookup_time", right_on="src_time",
                           direction="backward", tolerance=tol)
    return merged["temperature"].to_numpy(dtype=np.float32)


def predict_on_window(window_norm, interpreter, input_index, input_scale, input_zero_point,
                      output_indices, output_scales, output_zero_points,
                      input_buffer, use_int8, y_min, y_max):
    """Run one inference step.

    window_norm: (SEQ_LEN, FEATURE_COUNT) float32 in [0, 1].
    For SEQ_LEN=1 (Runs 1-5) this is a single-row window (1, FEATURE_COUNT).
    For SEQ_LEN=180 (Run 6+) this is a 3-hour rolling window (180, FEATURE_COUNT).
    input_buffer must have shape (1, SEQ_LEN, FEATURE_COUNT).
    """
    start = time.perf_counter()

    if use_int8:
        np.clip(
            np.round(window_norm / input_scale + input_zero_point),
            -128, 127,
            out=input_buffer[0],
        )
        interpreter.set_tensor(input_index, input_buffer)
        interpreter.invoke()
        output_norm = np.array([
            (np.squeeze(interpreter.get_tensor(output_indices[k])) - output_zero_points[k])
            * output_scales[k]
            for k in range(3)
        ], dtype=np.float32)
    else:
        # FP32 path (CPU / Mac dev) — input_buffer is already float32
        input_buffer[0] = window_norm
        interpreter.set_tensor(input_index, input_buffer)
        interpreter.invoke()
        output_norm = np.array([
            float(np.squeeze(interpreter.get_tensor(output_indices[k])))
            for k in range(3)
        ], dtype=np.float32)

    # Rescale from [-1, 1] normalized diff → °C diff
    diffs_c = 0.5 * (output_norm + 1.0) * (y_max - y_min) + y_min
    inference_time_ms = (time.perf_counter() - start) * 1000
    return diffs_c, inference_time_ms


def create_influx_points(base_timestamp, actuals, diffs_c, current_temp):
    """Create one InfluxDB point per prediction horizon, stored at the target time."""
    points = []
    for hrs, diff_c, ai in [(1, diffs_c[0], 0), (2, diffs_c[1], 1), (3, diffs_c[2], 2)]:
        future_ts = base_timestamp + pd.Timedelta(hours=hrs)
        pred_temp = float(current_temp) + float(diff_c)
        fields = {f"pred_{hrs}hr_temperature": pred_temp}
        if actuals is not None and not np.isnan(actuals[ai]):
            fields[f"actual_{hrs}hr_temperature"] = float(actuals[ai])
        points.append({
            "measurement": MEASUREMENT,
            "time": future_ts.isoformat(),
            "fields": fields,
        })
    return points


# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Model 5c Track B inference and write to InfluxDB.")
parser.add_argument("--tpu", default="",
    help="EdgeTPU device selector (e.g. 0, :0, usb:0). Leave empty to auto-select.")
parser.add_argument("--no-tpu", action="store_true",
    help="Disable EdgeTPU delegate; use FP32 model on CPU (Mac dev / no TPU).")
parser.add_argument("--cpu-int8", action="store_true",
    help="Run INT8 model on CPU without EdgeTPU delegate (faster than EdgeTPU for tiny models).")
parser.add_argument("--location", choices=list(LOCATIONS.keys()), default="sf",
    help="Weather data source: sf (San Francisco, default) or ps (Palm Springs).")
args = parser.parse_args()

# Resolve location — sets DB and MEASUREMENT used throughout the rest of the script
_loc      = LOCATIONS[args.location]
DB          = _loc["db"]
MEASUREMENT = _loc["measurement"]
print(f"📍 Location: {_loc['label']} | DB: {DB} | Measurement: {MEASUREMENT}")

# ---------------------------------------------------------------------------
# Load scalers
# ---------------------------------------------------------------------------
print(f"Loading scalers from {RESULTS_DIR}...")
with open(INPUT_SCALER_PATH) as f:
    input_scaler = json.load(f)
with open(TARGET_SCALER_PATH) as f:
    target_scaler = json.load(f)

y_min = float(target_scaler["min"])
y_max = float(target_scaler["max"])
print(f"  Target diff range: {y_min:.2f}°C to {y_max:.2f}°C")

FEATURE_ORDER = list(input_scaler.keys())
FEATURE_COUNT = len(FEATURE_ORDER)
print(f"  Features ({FEATURE_COUNT}): {', '.join(FEATURE_ORDER)}")

# ---------------------------------------------------------------------------
# Connect to InfluxDB
# ---------------------------------------------------------------------------
print(f"Connecting to InfluxDB (db={DB})...")
reader_client = InfluxDBClient(host="10.0.1.188", port=8086,
                               username="admin", password="24planet",
                               database=DB)
writer_client = InfluxDBClient(host="localhost", port=8086,
                               username="admin", password="24planet",
                               database=DB)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
use_int8 = not args.no_tpu
if args.cpu_int8:
    model_path = MODEL_INT8_PATH
    use_int8 = True
    print(f"Loading INT8 TFLite model on CPU (no delegate): {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
elif use_int8:
    model_path = MODEL_EDGETPU_PATH
    print(f"Loading INT8 TFLite model with EdgeTPU delegate: {model_path}")
    try:
        delegate_opts = _build_delegate_options(args.tpu)
        if delegate_opts:
            print(f"  Using EdgeTPU device: {delegate_opts['device']}")
            edgetpu_delegate = tflite.load_delegate("libedgetpu.so.1", delegate_opts)
        else:
            edgetpu_delegate = tflite.load_delegate("libedgetpu.so.1")
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[edgetpu_delegate])
        print(f"  ✅ EdgeTPU delegate loaded — ops will run on TPU hardware")
    except Exception as e:
        print(f"  ⚠️  EdgeTPU delegate failed ({e}); falling back to FP32 CPU.")
        use_int8 = False
        model_path = MODEL_FP32_PATH

if not use_int8:
    model_path = MODEL_FP32_PATH
    print(f"Loading FP32 TFLite model (CPU): {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"  Input shape:  {input_details[0]['shape']}")
print(f"  Output shapes: {[output_details[k]['shape'] for k in range(3)]}")

input_index, input_scale, input_zero_point, output_indices, output_scales, output_zero_points = \
    _extract_interpreter_constants(input_details, output_details)

# Detect SEQ_LEN from the model's input shape — (1, SEQ_LEN, FEATURE_COUNT).
# Runs 1-5: SEQ_LEN=1 (flat feature vector). Run 6+: SEQ_LEN=180 (3-hour window).
SEQ_LEN = int(input_details[0]["shape"][1])
print(f"  SEQ_LEN: {SEQ_LEN} (detected from model input shape)")

if use_int8:
    input_buffer = np.empty((1, SEQ_LEN, FEATURE_COUNT), dtype=np.int8)
else:
    input_buffer = np.empty((1, SEQ_LEN, FEATURE_COUNT), dtype=np.float32)

# ---------------------------------------------------------------------------
# Determine resume point
# ---------------------------------------------------------------------------
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
    print(f"  ⚠️  Could not query InfluxDB: {e}; starting from beginning.")

# ---------------------------------------------------------------------------
# Fetch data from InfluxDB
# ---------------------------------------------------------------------------
raw_fields = ("temperature, relative_humidity, station_pressure, "
              "solar_radiation, illuminance, uv, wind_avg, wind_gust, wind_direction")

if last_ts:
    # Predictions stored at T+1hr; convert back to the input time T.
    last_input_ts = pd.to_datetime(last_ts) - pd.Timedelta(hours=1)
    query_from = last_input_ts - pd.Timedelta(minutes=WARM_UP_MINUTES + BACKFILL_HOURS * 60)
    query_to   = last_input_ts + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
    print(f"  Query window: {query_from.isoformat()} → {query_to.isoformat()}")
    query = (f'SELECT {raw_fields} FROM "wf/obs_st" '
             f"WHERE time >= '{query_from.isoformat()}' AND time <= '{query_to.isoformat()}'")
else:
    first_res = reader_client.query('SELECT FIRST(temperature) FROM "wf/obs_st"')
    first_pts = list(first_res.get_points())
    if not first_pts:
        raise RuntimeError("❌ No data in 'wf/obs_st'.")
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
print(f"📥 Fetched {len(points)} points from InfluxDB")
if not points:
    print("⚠️  No data in query window — nothing to do.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Build DataFrame and derived features
# ---------------------------------------------------------------------------
df = pd.DataFrame(points)
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)
df.sort_index(inplace=True)

# Cyclical encodings
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

# Rolling slopes (must match training exactly — note: humidity_slope_30 uses window=60)
print("⚙️  Computing rolling slope features...")
df["temp_slope_15"]     = rolling_slope(df["temperature"].values, 15)
df["temp_slope_30"]     = rolling_slope(df["temperature"].values, 30)
df["temp_slope_60"]     = rolling_slope(df["temperature"].values, 60)
df["solar_slope_30"]    = rolling_slope(df["solar_radiation"].values, 30)
df["humidity_slope_30"] = rolling_slope(df["relative_humidity"].values, 60)  # window=60, matches training
df["pressure_slope_60"] = rolling_slope(df["station_pressure"].values, 60)
print("   ✅ Slopes computed")

# Gap-aware lags (Run 1: raw lags; Run 2: signed diffs — auto-selected via FEATURE_ORDER)
print("⚙️  Computing gap-aware lag features...")
_temp_lag60  = gap_aware_lag(df["temperature"], 60)
_temp_lag120 = gap_aware_lag(df["temperature"], 120)
_temp_lag180 = gap_aware_lag(df["temperature"], 180)

# Raw lags (Run 1 features)
df["temp_lag60"]  = _temp_lag60
df["temp_lag120"] = _temp_lag120
df["temp_lag180"] = _temp_lag180
# Signed diffs (Run 2 features)
df["temp_diff_vs_1hr"] = df["temperature"] - _temp_lag60
df["temp_diff_vs_2hr"] = df["temperature"] - _temp_lag120
df["temp_diff_vs_3hr"] = df["temperature"] - _temp_lag180

df["pressure_lag120"] = gap_aware_lag(df["station_pressure"], 120)
df["pressure_lag180"] = gap_aware_lag(df["station_pressure"], 180)
df["humidity_lag60"]  = gap_aware_lag(df["relative_humidity"], 60)

# Run 16+ features: temp_diff_vs_5hr/6hr (Track A Deep Run non-boundary attention anchor).
temperatures_arr = df["temperature"].values.astype(np.float32)
_temp_lag300 = _past_temp_at(df.index, temperatures_arr, pd.Timedelta(minutes=300))
_temp_lag360 = _past_temp_at(df.index, temperatures_arr, pd.Timedelta(minutes=360))
df["temp_diff_vs_5hr"] = temperatures_arr - _temp_lag300
df["temp_diff_vs_6hr"] = temperatures_arr - _temp_lag360
print("   ✅ Lags computed")

# Future targets for backfilling actuals
print("⚙️  Building future temperature targets...")
df["temp_t+1hr"] = _future_temp_at(df.index, temperatures_arr, pd.Timedelta(hours=1))
df["temp_t+2hr"] = _future_temp_at(df.index, temperatures_arr, pd.Timedelta(hours=2))
df["temp_t+3hr"] = _future_temp_at(df.index, temperatures_arr, pd.Timedelta(hours=3))
print(f"   Missing targets: {df[['temp_t+1hr','temp_t+2hr','temp_t+3hr']].isna().sum().to_dict()}")

print(f"\n🧪 Rows before dropna: {len(df)}")

# Verify all model features can be computed
missing_feats = [f for f in FEATURE_ORDER if f not in df.columns]
if missing_feats:
    raise ValueError(f"❌ Missing features: {missing_feats}\n"
                     f"   Available: {sorted(df.columns.tolist())}")

df.dropna(subset=FEATURE_ORDER, inplace=True)
float_cols = df.select_dtypes(include=["float64"]).columns
if len(float_cols):
    df[float_cols] = df[float_cols].astype(np.float32)
print(f"✅ Rows after dropna: {len(df)}")
print(f"📊 Data range: {df.index.min()} → {df.index.max()}")

# ---------------------------------------------------------------------------
# Pre-normalise entire feature matrix (vectorised — avoids per-row Python overhead)
# ---------------------------------------------------------------------------
_f_mins   = np.array([float(input_scaler[f]["min"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_maxs   = np.array([float(input_scaler[f]["max"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_denoms = _f_maxs - _f_mins
_f_denoms[_f_denoms == 0.0] = 1.0

scaled_data = np.clip(
    (df[FEATURE_ORDER].values.astype(np.float32) - _f_mins) / _f_denoms,
    0.0, 1.0,
)

targets_arr   = df[["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]].values.astype(np.float32)
temp_arr      = df["temperature"].values.astype(np.float32)
timestamps    = df.index

# ---------------------------------------------------------------------------
# Determine start index (skip warm-up rows; backfill if resuming)
# ---------------------------------------------------------------------------
start_index = 0
if last_input_ts is not None:
    try:
        backfill_start = last_input_ts - pd.Timedelta(hours=BACKFILL_HOURS)
        start_index = int(df.index.searchsorted(backfill_start))
    except Exception:
        print("⚠️  Could not compute backfill start — starting from row 0.")
        start_index = 0

if start_index >= len(df):
    print("⚠️  No new data — predictions are up to date.")
    sys.exit(0)

print(f"\n🔁 Starting inference at index {start_index} / {len(df)-1} "
      f"({timestamps[start_index] if start_index < len(df) else 'N/A'})")
_mode_label = ("💻 CPU INT8" if args.cpu_int8 else
               "📡 EdgeTPU INT8" if use_int8 else "💻 CPU FP32")
print(f"{_mode_label} inference, {FEATURE_COUNT} features, SEQ_LEN={SEQ_LEN}")

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
print("Running inference...")
run_count = 0
run_sum   = 0.0
run_min   = float("inf")
run_max   = float("-inf")
prediction_points = []
_timing_reported = False  # print one-shot timing after first prediction for sanity-check

for i in range(start_index, len(df)):
    # For SEQ_LEN=1 (Runs 1-5): window is a single row (1, FEATURE_COUNT).
    # For SEQ_LEN=180 (Run 6+): window is a 3-hour rolling slice (180, FEATURE_COUNT).
    if i < SEQ_LEN - 1:
        continue  # not enough history to fill the window yet

    window_norm = scaled_data[i - SEQ_LEN + 1:i + 1]  # (SEQ_LEN, FEATURE_COUNT)

    targets_row = targets_arr[i]
    actuals = None if np.any(np.isnan(targets_row)) else targets_row

    try:
        diffs_c, t_inf = predict_on_window(
            window_norm, interpreter,
            input_index, input_scale, input_zero_point,
            output_indices, output_scales, output_zero_points,
            input_buffer, use_int8, y_min, y_max,
        )
    except Exception as e:
        print(f"⚠️  Inference error at row {i} ({timestamps[i]}): {e}")
        continue

    run_count += 1
    run_sum += t_inf
    if t_inf < run_min: run_min = t_inf
    if t_inf > run_max: run_max = t_inf

    if not _timing_reported:
        _timing_reported = True
        # TPU: typically <5ms per inference. CPU INT8: 10-50ms. CPU FP32: 10-100ms.
        tpu_likely = use_int8 and not args.cpu_int8 and t_inf < 10.0
        hint = "✅ looks like TPU" if tpu_likely else "⚠️  looks like CPU (>10ms suggests no TPU)"
        print(f"  First inference: {t_inf:.3f}ms — {hint}")

    current_temp = float(temp_arr[i])
    prediction_points.extend(
        create_influx_points(timestamps[i], actuals, diffs_c, current_temp))

    if run_count % WRITE_EVERY == 0:
        print(f"  {run_count:,} predictions | "
              f"avg {run_sum/run_count:.3f}ms | "
              f"min {run_min:.3f}ms | max {run_max:.3f}ms")
        try:
            writer_client.write_points(prediction_points, time_precision="ms")
        except Exception as e:
            print(f"  ⚠️  InfluxDB write error: {e}")
        prediction_points = []

    if run_count % RESTART_EVERY == 0:
        print("🧼 Restarting to reset TPU state...")
        if prediction_points:
            writer_client.write_points(prediction_points, time_precision="ms")
        _exit_restart_for_wrapper()

# Final flush
if prediction_points:
    try:
        writer_client.write_points(prediction_points, time_precision="ms")
    except Exception as e:
        print(f"⚠️  Final InfluxDB write error: {e}")

print(f"✅ Inference complete. {run_count:,} predictions written to '{MEASUREMENT}'.")

# Restart to fetch next batch if this window is in the past
end_ts_dt = pd.to_datetime(end_ts)
if end_ts_dt.tzinfo is None:
    end_ts_dt = end_ts_dt.tz_localize("UTC")
now_utc = pd.Timestamp.now(tz="UTC")
if end_ts_dt < now_utc - pd.Timedelta(hours=1):
    print(f"📦 Query window ended at {end_ts_dt} (past) — restarting for next batch...")
    _exit_restart_for_wrapper()

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
