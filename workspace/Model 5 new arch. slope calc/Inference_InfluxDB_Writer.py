# Inference_InfluxDB_Writer.py

import time
import numpy as np
import json
import pandas as pd
from influxdb import InfluxDBClient
import datetime
import os
import tflite_runtime.interpreter as tflite
import sys
import atexit
import pprint
import argparse

# --- Configuration ---
MODEL_PATH = "weather_model_5a_best_edgetpu.tflite"
SEQ_LEN = 180  # Sequence length matching training (3 hours of history)
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 80000  # InfluxDB server max-select-points limit is 100k; effective window = this + 2*SEQ_LEN(360) + EXTRA_SAMPLES(250)
EXTRA_SAMPLES = 250  # Extra buffer beyond sequence and furthest target
# FEATURE_COUNT will be determined dynamically based on available features
# Base: 24 features + optional wind_direction (2), wind_lull (1), rain_accumulated (1)

FEATURE_ORDER_BASE = [
    'uv', 'wind_avg', 'wind_gust',
    'solar_radiation', 'illuminance',
    'relative_humidity', 'station_pressure',
    'day_of_year_sin', 'day_of_year_cos',
    'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
    'temp_lag30', 'humidity_lag30', 'temp_lag60', 'humidity_lag60', 'temp_lag120', 'humidity_lag120',
    'wind_avg_lag30', 'wind_gust_lag30', 'uv_lag30', 'pressure_lag30'
]

# --- Functions ---

def _exit_restart_for_wrapper():
    """Tell run_with_restart.py to launch another batch (exit 88).

    Use os._exit instead of sys.exit: after a long Edge TPU run, interpreter
    shutdown can segfault in native delegate teardown (parent would see -11).
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(88)


def _build_delegate_options(tpu_arg: str):
    """Return EdgeTPU delegate options for an optional TPU selector."""
    if not tpu_arg:
        return None

    tpu_arg = str(tpu_arg).strip()
    if not tpu_arg:
        return None

    # Accept either explicit forms (usb:0, pci:1, :0) or plain numeric ids (0, 1, ...).
    if ":" in tpu_arg:
        device = tpu_arg
    else:
        device = f":{tpu_arg}"
    return {"device": device}


def _extract_interpreter_constants(interp_input_details, interp_output_details):
    """Pre-extract all constant values from interpreter details to avoid per-call dict lookups."""
    idx = interp_input_details[0]['index']
    i_scale, i_zp = interp_input_details[0]['quantization']
    o_indices = [interp_output_details[k]['index'] for k in range(3)]
    o_scales   = [interp_output_details[k]['quantization'][0] for k in range(3)]
    o_zps      = [interp_output_details[k]['quantization'][1] for k in range(3)]
    return idx, i_scale, i_zp, o_indices, o_scales, o_zps


def predict_on_window(window, interpreter, input_index, input_scale, input_zero_point,
                      output_indices, output_scales, output_zero_points, input_buffer):
    start = time.perf_counter()
    # window is already float32 and clipped to [0,1] by the pre-normalization step.
    # Quantize directly into the pre-allocated int8 buffer — no intermediate allocations.
    np.clip(
        np.round(window / input_scale + input_zero_point),
        -128, 127,
        out=input_buffer[0],
    )
    interpreter.set_tensor(input_index, input_buffer)
    interpreter.invoke()

    # Dequantize all 3 outputs using pre-extracted constants (no dict lookups).
    output_scaled = np.array([
        (interpreter.get_tensor(output_indices[k])[0][0] - output_zero_points[k]) * output_scales[k]
        for k in range(3)
    ], dtype=np.float32)

    output_rescaled = 0.5 * (output_scaled + 1.0) * (y_max - y_min) + y_min
    inference_time_ms = (time.perf_counter() - start) * 1000
    return output_rescaled, inference_time_ms

def create_influx_points(base_timestamp, actuals, preds, current_temp, temp_lag30):
    # preds are temperature differences, so add current temperature to get actual predicted temperatures
    # Each prediction is stored at its TARGET timestamp (base_timestamp + horizon) so that
    # the graph shows predictions extending into the future rather than ending at "now".
    pred_1hr_temp = float(preds[0]) + float(current_temp)
    pred_2hr_temp = float(preds[1]) + float(current_temp)
    pred_3hr_temp = float(preds[2]) + float(current_temp)

    # actuals is a numpy array [temp_t+1hr, temp_t+2hr, temp_t+3hr]
    horizons = [
        (1, pred_1hr_temp, 0),
        (2, pred_2hr_temp, 1),
        (3, pred_3hr_temp, 2),
    ]

    points = []
    for hrs, pred_temp, actual_idx in horizons:
        future_ts = base_timestamp + pd.Timedelta(hours=hrs)
        fields = {f"pred_{hrs}hr_temperature": pred_temp}
        if actuals is not None and not np.isnan(actuals[actual_idx]):
            fields[f"actual_{hrs}hr_temperature"] = float(actuals[actual_idx])
        points.append({
            "measurement": "model_5a",
            "time": future_ts.isoformat(),
            "fields": fields
        })
    return points

# --- Load Scaler Parameters ---
parser = argparse.ArgumentParser(description="Run Model 5 inference and write results to InfluxDB.")
parser.add_argument(
    "--tpu",
    default="",
    help="EdgeTPU device selector (examples: 0, :0, usb:0, pci:0).",
)
args = parser.parse_args()

print("Loading input scaler...")
with open("input_scaler_5a.json", "r") as f:
    input_scaler = json.load(f)

# --- Load Target Scaler ---
print("Loading target scaler...")
with open("target_scaler_5a.json", "r") as f:
    target_scaler = json.load(f)
y_min = target_scaler["min"]
y_max = target_scaler["max"]

# --- Connect to InfluxDB ---
print("Connecting to InfluxDB...")
# Reader client for weather data from remote host
reader_client = InfluxDBClient(
    host="10.0.1.188",
    port=8086,
    username="admin",
    password="24planet",
    database="weather"
)

# Writer client for predictions to localhost
writer_client = InfluxDBClient(
    host="localhost",
    port=8086,
    username="admin",
    password="24planet",
    database="weather"
)

# --- Load Model ---
print("Loading TFLite model with EdgeTPU delegate...")
delegate_options = _build_delegate_options(args.tpu)
if delegate_options:
    print(f"Using EdgeTPU device: {delegate_options['device']}")
    edgetpu_delegate = tflite.load_delegate('libedgetpu.so.1', delegate_options)
else:
    edgetpu_delegate = tflite.load_delegate('libedgetpu.so.1')

interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[edgetpu_delegate]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index, input_scale, input_zero_point, output_indices, output_scales, output_zero_points = \
    _extract_interpreter_constants(input_details, output_details)

# --- Determine Resume Point from InfluxDB ---
print("Checking InfluxDB for last prediction timestamp...")
last_ts = None
try:
    last_pred_result = writer_client.query('SELECT LAST("pred_1hr_temperature") FROM "model_5a"')
    last_points = list(last_pred_result.get_points())
    if last_points:
        last_ts = last_points[0]['time']
        print(f"📁 Resuming from InfluxDB; last_timestamp = {last_ts}")
    else:
        print("📄 No previous predictions in InfluxDB; starting from the beginning.")
except Exception as e:
    print(f"⚠️ Could not query InfluxDB for last prediction: {e}")
    print("📄 Starting from the beginning.")

# Only select the fields needed for feature generation and inference
fields = "temperature, relative_humidity, station_pressure, solar_radiation, illuminance, uv, wind_avg, wind_gust, wind_direction, wind_lull, rain_accumulated"

if last_ts:
    last_ts_dt = pd.to_datetime(last_ts)
    # Predictions are stored at T+1hr (the target time), so convert back to input time T
    last_input_ts_dt = last_ts_dt - pd.Timedelta(hours=1)
    # Look back far enough to:
    # 1. Re-process the last SEQ_LEN rows to backfill actuals now available
    # 2. Provide a full SEQ_LEN input window for each of those backfill rows
    # 3. Allow lag features (up to 120 min) to be valid for the oldest backfill row
    backfill_lookback = pd.Timedelta(minutes=2 * SEQ_LEN + 120)
    resume_from_dt = last_input_ts_dt - backfill_lookback
    resume_from = resume_from_dt.isoformat()
    end_dt = last_input_ts_dt + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
    end_ts = end_dt.isoformat()
    print(f"Resuming from {last_input_ts_dt} (pred stored at {last_ts_dt}) with backfill lookback to {resume_from_dt}")
    query = f'SELECT {fields} FROM "wf/obs_st" WHERE time >= \'{resume_from}\' AND time <= \'{end_ts}\''
else:
    # No usable timestamp -> full backfill; get earliest timestamp first (safe 1-point query)
    print("🚀 No usable progress timestamp; starting from the beginning of the dataset.")
    first_result = reader_client.query('SELECT FIRST(temperature) FROM "wf/obs_st"')
    first_points = list(first_result.get_points())
    if not first_points:
        raise RuntimeError("❌ No data found in InfluxDB measurement 'wf/obs_st'.")
    start_ts_dt = pd.to_datetime(first_points[0]['time'])
    end_dt = start_ts_dt + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
    start_ts = start_ts_dt.isoformat()
    end_ts = end_dt.isoformat()
    query = f'SELECT {fields} FROM "wf/obs_st" WHERE time >= \'{start_ts}\' AND time <= \'{end_ts}\''

result = reader_client.query(query)
points = list(result.get_points())

# --- Load DataFrame ---
df = pd.DataFrame(points)

# Validate that 'time' column comes directly from InfluxDB
if 'time' not in df.columns:
    raise ValueError(
        "❌ InfluxDB query did not return a 'time' field. "
        "The time column is required for accurate day_of_year and time_of_day feature creation. "
        "Please verify that your InfluxDB query includes time values."
    )

df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Add derived features (added to training data when exported from InfluxDB)
df['day_of_year'] = df.index.dayofyear
df['time_of_day'] = df.index.hour + df.index.minute / 60.0

# Enhanced cyclical encoding for time_of_day to better capture daily patterns
# Convert time_of_day (0-24 hours) to sine and cosine components
df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)

# Add higher-order cyclical components for more complex daily patterns
df['time_of_day_sin2'] = np.sin(4 * np.pi * df['time_of_day'] / 24.0)
df['time_of_day_cos2'] = np.cos(4 * np.pi * df['time_of_day'] / 24.0)

# Add cyclical encoding for day_of_year to better capture seasonal patterns
# Convert day_of_year (1-365) to sine and cosine components
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# Add cyclical encoding for wind_direction (0-360 degrees) - very useful for local weather patterns
if 'wind_direction' in df.columns:
    df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360.0)
    df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360.0)

# --- Feature Engineering ---
# Note: Delta features (temperature_delta, pressure_delta, etc.) are not used in this model
# They were removed from training to simplify the model architecture
df['temp_lag30'] = df['temperature'].shift(30)
df['humidity_lag30'] = df['relative_humidity'].shift(30)

# Additional lag features for multiple time horizons
df['temp_lag60'] = df['temperature'].shift(60)   # 1 hour ago
df['temp_lag120'] = df['temperature'].shift(120) # 2 hours ago
df['humidity_lag60'] = df['relative_humidity'].shift(60)   # 1 hour ago
df['humidity_lag120'] = df['relative_humidity'].shift(120) # 2 hours ago

# Manual interaction features - COMMENTED OUT: Using learned FeatureInteraction layer instead
# Time-of-day interactions with environmental factors
# df['time_sin_solar'] = df['time_of_day_sin'] * df['solar_radiation_delta']
# df['time_cos_solar'] = df['time_of_day_cos'] * df['solar_radiation_delta']
# df['time_sin_uv'] = df['time_of_day_sin'] * df['uv']
# df['time_cos_uv'] = df['time_of_day_cos'] * df['uv']

# Additional interaction features - wind and temperature lag interactions
# df['time_sin_wind'] = df['time_of_day_sin'] * df['wind_avg']
# df['time_cos_wind'] = df['time_of_day_cos'] * df['wind_avg']
# df['time_sin_temp_lag'] = df['time_of_day_sin'] * df['temp_lag30']
# df['time_cos_temp_lag'] = df['time_of_day_cos'] * df['temp_lag30']

# Add seasonal-time interactions (how environmental effects vary by season)
# df['season_sin_temp_lag'] = df['day_of_year_sin'] * df['temp_lag30']
# df['season_cos_temp_lag'] = df['day_of_year_cos'] * df['temp_lag30']
# df['season_sin_humidity'] = df['day_of_year_sin'] * df['humidity_lag30']
# df['season_cos_humidity'] = df['day_of_year_cos'] * df['humidity_lag30']

# Add more environmental lag features (wind, UV, pressure)
df['wind_avg_lag30'] = df['wind_avg'].shift(30)
df['wind_gust_lag30'] = df['wind_gust'].shift(30)
df['uv_lag30'] = df['uv'].shift(30)
df['pressure_lag30'] = df['station_pressure'].shift(30)

# Build FEATURE_ORDER dynamically based on available features
FEATURE_ORDER = FEATURE_ORDER_BASE.copy()

# Add wind_direction features if available
if 'wind_direction_sin' in df.columns:
    FEATURE_ORDER.extend(['wind_direction_sin', 'wind_direction_cos'])

# Add wind_lull if available (contrast to wind_gust, indicates wind variability)
if 'wind_lull' in df.columns:
    FEATURE_ORDER.append('wind_lull')

# Add rain_accumulated if available (precipitation can cause cooling)
if 'rain_accumulated' in df.columns:
    FEATURE_ORDER.append('rain_accumulated')

# Set FEATURE_COUNT dynamically based on available features
FEATURE_COUNT = len(FEATURE_ORDER)
print(f"📊 Using {FEATURE_COUNT} features: {', '.join(FEATURE_ORDER)}")

# Targets
print("Creating shifted targets...")
df['temp_t+1hr'] = df['temperature'].shift(-60)
df['temp_t+2hr'] = df['temperature'].shift(-120)
df['temp_t+3hr'] = df['temperature'].shift(-180)

print("❓ Missing target counts:")
print(df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].isna().sum())

print("\n🧪 Total rows before processing:", len(df))

# Process the data but keep the extra rows for target validation
required_fields = FEATURE_ORDER

# Drop rows where input features are incomplete (e.g. early rows with NaN lags).
# Rows with NaN targets (last 3 hours) are kept — predictions are made without actuals.
df.dropna(subset=required_fields, inplace=True)
# Optimize memory usage: convert float64 to float32
float_cols = df.select_dtypes(include=['float64']).columns
if len(float_cols) > 0:
    df[float_cols] = df[float_cols].astype(np.float32)
    print(f"🔧 Converted {len(float_cols)} float64 columns to float32 for efficiency.")
print("✅ Total rows after dropping NaNs:", len(df))
print(f"📊 Data range: {df.index.min()} to {df.index.max()}")

# Pre-normalize the entire feature matrix once — avoids per-row Python loop inside the inference loop
_f_mins = np.array([float(input_scaler[f]["min"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_maxs = np.array([float(input_scaler[f]["max"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_denoms = _f_maxs - _f_mins
_f_denoms[_f_denoms == 0.0] = 1.0  # avoid div-by-zero; those columns will be 0 after subtraction
_zero_cols = (_f_maxs - _f_mins) == 0.0

scaled_data = np.clip(
    (df[FEATURE_ORDER].values.astype(np.float32) - _f_mins) / _f_denoms,
    0.0, 1.0
)
scaled_data[:, _zero_cols] = 0.0

# Pre-extract per-row arrays to avoid pandas .iloc overhead in the inference loop
targets_arr = df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].values.astype(np.float32)
temperatures = df['temperature'].values.astype(np.float32)
temp_lag30_arr = df['temp_lag30'].values.astype(np.float32)
timestamps = df.index

# Pre-allocate the INT8 input buffer once; reused every call to avoid per-prediction allocation.
input_buffer = np.empty((1, SEQ_LEN, FEATURE_COUNT), dtype=np.int8)

# Normalize and run inference
print("Running inference...")
# Running stats replace a growing list — O(1) update, no full-list scan at flush time.
run_count = 0
run_sum = 0.0
run_min = float('inf')
run_max = float('-inf')
prediction_points = []

# Integer thresholds avoid float division in the hot loop.
WRITE_EVERY   = BATCH_SIZE // 4   # flush to InfluxDB every N predictions
RESTART_EVERY = 500_000           # full process restart every N predictions

start_index = 0
if last_ts:
    try:
        # Predictions are stored at T+1hr; convert back to input time T before backfilling
        last_input_ts = pd.to_datetime(last_ts) - pd.Timedelta(hours=1)
        backfill_from = last_input_ts - pd.Timedelta(minutes=SEQ_LEN)
        start_index = df.index.searchsorted(backfill_from)
    except Exception:
        print(f"⚠️  Could not locate backfill position for {last_ts}, starting from beginning")
        start_index = 0

min_start = max(start_index, SEQ_LEN - 1)

# Clamp min_start to valid range - ensure we can actually make predictions
if min_start >= len(df):
    print(f"⚠️  No new data to process (calculated start_index={start_index}, min_start={min_start}, df length={len(df)})")
    print(f"    All predictions are up to date!")
    min_start = len(df)  # Set to len(df) so the loop won't execute
    resume_time = "N/A"
else:
    resume_time = df.index[min_start] if min_start < len(df) else "N/A"

print(f"🔁 Resuming from index {min_start} / {len(df) - 1} at {resume_time} (backfilling 3 hrs of actuals)")
print(f"📊 Using sequence length of {SEQ_LEN} timesteps (3 hours of history)")

for i in range(min_start, len(df)):
    window_start = i - SEQ_LEN + 1

    if window_start < 0:
        print(f"⚠️ Skipping row {i}: insufficient data for sequence")
        continue

    scaled_window = scaled_data[window_start:i+1]  # shape (SEQ_LEN, FEATURE_COUNT)

    if len(scaled_window) != SEQ_LEN:
        print(f"⚠️ Skipping row {i}: insufficient data for sequence (have {len(scaled_window)}, need {SEQ_LEN})")
        continue

    targets = targets_arr[i]
    actuals = targets  # each horizon's actual is written independently if available

    try:
        preds, t_inf = predict_on_window(
            scaled_window, interpreter, input_index,
            input_scale, input_zero_point,
            output_indices, output_scales, output_zero_points,
            input_buffer,
        )
    except Exception as e:
        print(f"⚠️ Exception in prediction at row {i}")
        print(f"  Error: {e}")
        continue

    try:
        run_count += 1
        run_sum += t_inf
        if t_inf < run_min: run_min = t_inf
        if t_inf > run_max: run_max = t_inf

        timestamp = timestamps[i]
        current_temp = float(temperatures[i])
        temp_lag30 = float(temp_lag30_arr[i])
        prediction_points.extend(create_influx_points(timestamp, actuals, preds, current_temp, temp_lag30))

        if run_count % WRITE_EVERY == 0:
            print(f"After {run_count} runs:")
            print(f"  Avg Inference Time: {run_sum / run_count:.2f} ms")
            print(f"  Min Inference Time: {run_min:.2f} ms")
            print(f"  Max Inference Time: {run_max:.2f} ms")

            writer_client.write_points(prediction_points, time_precision='ms')
            prediction_points = []

        # Restart process fully to reset TPU state.
        if run_count % RESTART_EVERY == 0:
            print("🧼 Restarting process to fully reset TPU state...")
            print("🧼 Exiting to allow external restart...")
            _exit_restart_for_wrapper()

    except Exception as e:
        print(f"⚠️ Exception during Influx write prep at row {i}")
        print(f"  preds: shape = {preds.shape}, values = {preds}")
        print(f"  targets: {targets}")
        print(f"  Error: {e}")
        continue

# Final flush
if prediction_points:
    writer_client.write_points(prediction_points, time_precision='ms')

print("✅ Inference complete.")

# If the query window ended in the past, there is likely more data in InfluxDB.
# Checking the time window (not point count) handles sparse/gappy data correctly —
# a gap-heavy batch may return fewer than QUERY_BATCH_SIZE points even mid-dataset.
end_ts_dt = pd.to_datetime(end_ts)
if end_ts_dt.tzinfo is None:
    end_ts_dt = end_ts_dt.tz_localize('UTC')
now_utc = pd.Timestamp.now(tz='UTC')
if end_ts_dt < now_utc - pd.Timedelta(hours=1):
    print(f"📦 Query window ended at {end_ts_dt} (past) — more data may exist. Restarting to fetch next batch...")
    _exit_restart_for_wrapper()

# Use os._exit to bypass Python interpreter shutdown — the EdgeTPU native delegate
# segfaults during normal teardown.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
