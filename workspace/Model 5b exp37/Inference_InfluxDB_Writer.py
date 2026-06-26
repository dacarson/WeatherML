# Inference_InfluxDB_Writer.py — Model 5b Experiment 37
# INT8-quantized Conv2D model compiled for Edge TPU.
# New vs previous inference scripts: adds temp_lag180 feature (Exp 37).

import time
import numpy as np
import json
import pandas as pd
import datetime
import os
import sys
import argparse
import tflite_runtime.interpreter as tflite
from influxdb import InfluxDBClient

# --- Configuration ---
MODEL_PATH = "weather_model_5b_exp37_edgetpu.tflite"
MEASUREMENT = "model_5b_exp37"
SEQ_LEN = 180  # 3 hours of history
GAP_STEP_TOLERANCE_S = 180
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 80000
EXTRA_SAMPLES = 250

def _exit_restart_for_wrapper():
    """Tell run_with_restart.py to launch another batch (exit 88)."""
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
    device = tpu_arg if ":" in tpu_arg else f":{tpu_arg}"
    return {"device": device}

# --- Functions ---

def apply_gap_policy(df: pd.DataFrame, seq_len: int, gap_step_tolerance_s: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_index()
    dt_s = df.index.to_series().diff().dt.total_seconds()
    gap_mask = dt_s > float(gap_step_tolerance_s)
    gap_positions = np.flatnonzero(gap_mask.to_numpy())

    if gap_positions.size == 0:
        return df

    print(f"🔍 Gap detection: found {gap_positions.size} gap(s) > {gap_step_tolerance_s}s")
    if gap_positions.size <= 10:
        for pos in gap_positions[:10]:
            print(f"   Gap at {df.index[pos]}: {dt_s.iloc[pos]:.1f}s")

    idx_after_gap = df.index[gap_positions]
    _zero_at_gap = ['temp_delta_1',
                    'temp_slope_15', 'temp_slope_30', 'temp_slope_60',
                    'solar_slope_30', 'humidity_slope_30', 'pressure_slope_60']
    for col in _zero_at_gap:
        if col in df.columns:
            df.loc[idx_after_gap, col] = 0.0

    print(f"ℹ️  Gap policy: zeroed temp_delta_1 and slope features after gaps; not dropping rows.")
    return df


def predict_on_window(window, interpreter, input_index, input_scale, input_zero_point,
                      output_indices, output_scales, output_zero_points, input_buffer):
    start = time.perf_counter()
    np.clip(
        np.round(window / input_scale + input_zero_point),
        -128, 127,
        out=input_buffer[0],
    )
    interpreter.set_tensor(input_index, input_buffer)
    interpreter.invoke()

    output_scaled = np.array([
        (interpreter.get_tensor(output_indices[k])[0][0] - output_zero_points[k]) * output_scales[k]
        for k in range(3)
    ], dtype=np.float32)

    output_rescaled = 0.5 * (output_scaled + 1.0) * (y_maxs - y_mins) + y_mins
    inference_time_ms = (time.perf_counter() - start) * 1000
    return output_rescaled, inference_time_ms


def create_influx_point(timestamp, actuals, preds, current_temp):
    pred_1hr_temp = float(current_temp) + float(preds[0])
    pred_2hr_temp = float(current_temp) + float(preds[1])
    pred_3hr_temp = float(current_temp) + float(preds[2])

    fields = {
        "pred_1hr_temperature": pred_1hr_temp,
        "pred_2hr_temperature": pred_2hr_temp,
        "pred_3hr_temperature": pred_3hr_temp,
    }

    if actuals is not None:
        fields["actual_1hr_temperature"] = float(actuals['temp_t+1hr'])
        fields["actual_2hr_temperature"] = float(actuals['temp_t+2hr'])
        fields["actual_3hr_temperature"] = float(actuals['temp_t+3hr'])

    return {
        "measurement": MEASUREMENT,
        "time": timestamp.isoformat(),
        "fields": fields,
    }

# --- Load Scaler Parameters ---
parser = argparse.ArgumentParser(description="Run Model 5b Exp 37 inference and write results to InfluxDB.")
parser.add_argument(
    "--tpu",
    default="",
    help="EdgeTPU device selector (examples: 0, :0, usb:0, pci:0).",
)
args = parser.parse_args()

print("Loading input scaler...")
with open("input_scaler_5b.json", "r") as f:
    input_scaler = json.load(f)

print("Loading target scaler...")
with open("target_scaler_5b.json", "r") as f:
    target_scaler = json.load(f)

def _as_float(x) -> float:
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError(f"Expected a number, got {type(x)}: {x}")

if "mins" in target_scaler and "maxs" in target_scaler:
    def _extract_triplet(v):
        if isinstance(v, dict):
            for k0, k1, k2 in [("diff_1hr", "diff_2hr", "diff_3hr"), ("1hr", "2hr", "3hr")]:
                if k0 in v and k1 in v and k2 in v:
                    return np.array([_as_float(v[k0]), _as_float(v[k1]), _as_float(v[k2])], dtype=np.float32)
            ks = sorted(v.keys())
            return np.array([_as_float(v[k]) for k in ks], dtype=np.float32)
        return np.array([_as_float(x) for x in v], dtype=np.float32)
    y_mins = _extract_triplet(target_scaler["mins"])
    y_maxs = _extract_triplet(target_scaler["maxs"])
elif "min" in target_scaler and "max" in target_scaler:
    y_mins = np.array([_as_float(target_scaler["min"])] * 3, dtype=np.float32)
    y_maxs = np.array([_as_float(target_scaler["max"])] * 3, dtype=np.float32)
else:
    raise ValueError(f"Unsupported target_scaler_5b.json format. Got keys={list(target_scaler.keys())}")

print("🎯 Target scaler bounds (diff) per-horizon:")
print(f"  1hr: min={y_mins[0]:.4f}°C, max={y_maxs[0]:.4f}°C")
print(f"  2hr: min={y_mins[1]:.4f}°C, max={y_maxs[1]:.4f}°C")
print(f"  3hr: min={y_mins[2]:.4f}°C, max={y_maxs[2]:.4f}°C")

# --- Connect to InfluxDB ---
print("Connecting to InfluxDB...")
reader_client = InfluxDBClient(
    host="10.0.1.188",
    port=8086,
    username="admin",
    password="24planet",
    database="weather"
)
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
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index    = input_details[0]['index']
input_scale, input_zero_point = input_details[0]['quantization']
output_indices     = [output_details[k]['index'] for k in range(3)]
output_scales      = [output_details[k]['quantization'][0] for k in range(3)]
output_zero_points = [output_details[k]['quantization'][1] for k in range(3)]

print(f"Input quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")

# --- Determine Resume Point ---
print(f"Checking InfluxDB for last prediction in '{MEASUREMENT}'...")
last_ts = None
try:
    last_pred_result = writer_client.query(f'SELECT LAST("pred_1hr_temperature") FROM "{MEASUREMENT}"')
    last_points = list(last_pred_result.get_points())
    if last_points:
        last_ts = last_points[0]['time']
        print(f"📁 Resuming from InfluxDB; last_timestamp = {last_ts}")
    else:
        print("📄 No previous predictions in InfluxDB; starting from the beginning.")
except Exception as e:
    print(f"⚠️ Could not query InfluxDB for last prediction: {e}")
    print("📄 Starting from the beginning.")

fields = "temperature, relative_humidity, station_pressure, solar_radiation, illuminance, uv, wind_avg, wind_gust, wind_direction, wind_lull, rain_accumulated"

if last_ts:
    last_ts_dt = pd.to_datetime(last_ts)
    backfill_lookback = pd.Timedelta(minutes=2 * SEQ_LEN)
    resume_from_dt = last_ts_dt - backfill_lookback
    resume_from = resume_from_dt.isoformat()
    end_dt = last_ts_dt + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
    end_ts = end_dt.isoformat()
    print(f"Resuming from {last_ts_dt} with backfill lookback to {resume_from_dt}")
    query = f'SELECT {fields} FROM "wf/obs_st" WHERE time >= \'{resume_from}\' AND time <= \'{end_ts}\''
else:
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
print(f"📥 Fetched {len(points)} points from InfluxDB")

# --- Build DataFrame ---
df = pd.DataFrame(points)

if 'time' not in df.columns:
    raise ValueError("❌ InfluxDB query did not return a 'time' field.")

df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Cyclical time encodings
df['day_of_year'] = df.index.dayofyear
df['time_of_day'] = df.index.hour + df.index.minute / 60.0
df['time_of_day_sin']  = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
df['time_of_day_cos']  = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
df['time_of_day_sin2'] = np.sin(4 * np.pi * df['time_of_day'] / 24.0)
df['time_of_day_cos2'] = np.cos(4 * np.pi * df['time_of_day'] / 24.0)
df['day_of_year_sin']  = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos']  = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# 1-step temperature delta (zeroed at gap boundaries by apply_gap_policy)
df['temp_delta_1'] = df['temperature'].diff().fillna(0.0)

# Explicit temperature lag features via time-based lookup (matching training pipeline)
def _past_temp_at(src_df: pd.DataFrame, offset: pd.Timedelta, tol: pd.Timedelta) -> np.ndarray:
    base = pd.DataFrame({"base_time": src_df.index, "lookup_time": src_df.index - offset})
    src  = pd.DataFrame({"src_time": src_df.index,
                          "temperature": src_df['temperature'].to_numpy(dtype=np.float32)})
    merged = pd.merge_asof(base, src, left_on="lookup_time", right_on="src_time",
                           direction="backward", tolerance=tol)
    return merged["temperature"].to_numpy(dtype=np.float32)

_lag_tol = pd.Timedelta(seconds=90)
df['temp_lag60']  = _past_temp_at(df, pd.Timedelta(minutes=60),  _lag_tol)
df['temp_lag120'] = _past_temp_at(df, pd.Timedelta(minutes=120), _lag_tol)
df['temp_lag180'] = _past_temp_at(df, pd.Timedelta(minutes=180), _lag_tol)  # Exp 37

# Rolling linear-regression slopes (matching training)
def rolling_slope(data, window):
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

print("⚙️  Computing rolling slope features...")
df['temp_slope_15']     = rolling_slope(df['temperature'].values,       15)
df['temp_slope_30']     = rolling_slope(df['temperature'].values,       30)
df['temp_slope_60']     = rolling_slope(df['temperature'].values,       60)
df['solar_slope_30']    = rolling_slope(df['solar_radiation'].values,   30)
df['humidity_slope_30'] = rolling_slope(df['relative_humidity'].values, 30)
df['pressure_slope_60'] = rolling_slope(df['station_pressure'].values,  60)
print("   ✅ Slope features computed")

# Gap policy: zero delta/slope at gap boundaries
df = apply_gap_policy(df, SEQ_LEN, GAP_STEP_TOLERANCE_S)

# Wind direction cyclical encoding
if 'wind_direction' in df.columns:
    df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360.0)
    df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360.0)

# Feature order from training (authoritative — don't reorder manually)
FEATURE_ORDER = list(input_scaler.keys())
FEATURE_COUNT = len(FEATURE_ORDER)
print(f"📊 Using {FEATURE_COUNT} features: {', '.join(FEATURE_ORDER)}")

missing_features = [f for f in FEATURE_ORDER if f not in df.columns]
if missing_features:
    raise ValueError(
        f"❌ Missing required features: {missing_features}\n"
        f"Available columns: {sorted(df.columns.tolist())}"
    )

# Build future-temperature targets for backfilling actuals into InfluxDB
print("Creating future temperature targets...")
_temp = df['temperature'].astype(np.float32)

def _future_temp_at(offset: pd.Timedelta, tolerance: pd.Timedelta) -> np.ndarray:
    base = pd.DataFrame({"base_time": df.index, "lookup_time": df.index + offset})
    src  = pd.DataFrame({"src_time": df.index, "temperature": _temp.to_numpy(dtype=np.float32)})
    merged = pd.merge_asof(base, src, left_on="lookup_time", right_on="src_time",
                           direction="forward", tolerance=tolerance)
    return merged["temperature"].to_numpy(dtype=np.float32)

df['temp_t+1hr'] = _future_temp_at(pd.Timedelta(hours=1), pd.Timedelta(seconds=90))
df['temp_t+2hr'] = _future_temp_at(pd.Timedelta(hours=2), pd.Timedelta(seconds=90))
df['temp_t+3hr'] = _future_temp_at(pd.Timedelta(hours=3), pd.Timedelta(seconds=90))

print("❓ Missing target counts:")
print(df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].isna().sum())
print(f"\n🧪 Total rows before NaN drop: {len(df)}")

df.dropna(subset=FEATURE_ORDER, inplace=True)
float_cols = df.select_dtypes(include=['float64']).columns
if len(float_cols) > 0:
    df[float_cols] = df[float_cols].astype(np.float32)
print(f"✅ Total rows after dropping NaNs: {len(df)}")
print(f"📊 Data range: {df.index.min()} to {df.index.max()}")

# --- Pre-normalize entire feature matrix ---
_f_mins   = np.array([float(input_scaler[f]["min"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_maxs   = np.array([float(input_scaler[f]["max"]) for f in FEATURE_ORDER], dtype=np.float32)
_f_denoms = _f_maxs - _f_mins
_f_denoms[_f_denoms == 0.0] = 1.0
_zero_cols = (_f_maxs - _f_mins) == 0.0

scaled_data = np.clip(
    (df[FEATURE_ORDER].values.astype(np.float32) - _f_mins) / _f_denoms,
    0.0, 1.0
)
scaled_data[:, _zero_cols] = 0.0

targets_arr = df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].values.astype(np.float32)
temp_arr    = df['temperature'].values.astype(np.float32)
timestamps  = df.index

input_buffer = np.empty((1, SEQ_LEN, FEATURE_COUNT), dtype=np.int8)

# --- Determine start index ---
start_index = 0
if last_ts:
    try:
        backfill_from = pd.to_datetime(last_ts) - pd.Timedelta(minutes=SEQ_LEN)
        start_index = df.index.searchsorted(backfill_from)
    except Exception:
        start_index = 0

min_start = max(start_index, SEQ_LEN - 1)

if min_start >= len(df):
    print(f"⚠️  No new data to process. All predictions are up to date!")
    min_start = len(df)
    resume_time = "N/A"
else:
    resume_time = df.index[min_start]

print(f"🔁 Resuming from index {min_start} / {len(df) - 1} at {resume_time}")
print(f"📊 Using sequence length of {SEQ_LEN} timesteps (3 hours of history)")

# --- Inference loop ---
print("Running inference...")
run_count = 0
run_sum   = 0.0
run_min   = float('inf')
run_max   = float('-inf')
prediction_points = []

WRITE_EVERY   = BATCH_SIZE // 4
RESTART_EVERY = 500_000

for i in range(min_start, len(df)):
    window_start = i - SEQ_LEN + 1
    if window_start < 0:
        continue

    scaled_window = scaled_data[window_start:i+1]
    if len(scaled_window) != SEQ_LEN:
        continue

    targets_row = targets_arr[i]
    actuals = None if np.any(np.isnan(targets_row)) else {
        'temp_t+1hr': float(targets_row[0]),
        'temp_t+2hr': float(targets_row[1]),
        'temp_t+3hr': float(targets_row[2]),
    }

    try:
        preds, t_inf = predict_on_window(
            scaled_window, interpreter, input_index,
            input_scale, input_zero_point,
            output_indices, output_scales, output_zero_points,
            input_buffer,
        )
    except Exception as e:
        print(f"⚠️ Exception in prediction at row {i}: {e}")
        continue

    try:
        run_count += 1
        run_sum += t_inf
        if t_inf < run_min: run_min = t_inf
        if t_inf > run_max: run_max = t_inf

        timestamp    = timestamps[i]
        current_temp = float(temp_arr[i])
        prediction_points.append(create_influx_point(timestamp, actuals, preds, current_temp))

        if run_count % WRITE_EVERY == 0:
            print(f"After {run_count} runs:")
            print(f"  Avg Inference Time: {run_sum / run_count:.2f} ms")
            print(f"  Min Inference Time: {run_min:.2f} ms")
            print(f"  Max Inference Time: {run_max:.2f} ms")
            writer_client.write_points(prediction_points, time_precision='ms')
            prediction_points = []

        if run_count % RESTART_EVERY == 0:
            print("🧼 Restarting process to fully reset TPU state...")
            _exit_restart_for_wrapper()

    except Exception as e:
        print(f"⚠️ Exception during Influx write prep at row {i}: {e}")
        continue

# Final flush
if prediction_points:
    writer_client.write_points(prediction_points, time_precision='ms')

print("✅ Inference complete.")

end_ts_dt = pd.to_datetime(end_ts)
if end_ts_dt.tzinfo is None:
    end_ts_dt = end_ts_dt.tz_localize('UTC')
now_utc = pd.Timestamp.now(tz='UTC')
if end_ts_dt < now_utc - pd.Timedelta(hours=1):
    print(f"📦 Query window ended at {end_ts_dt} — more data may exist. Restarting...")
    _exit_restart_for_wrapper()

# Use os._exit to bypass interpreter shutdown — EdgeTPU native delegate can segfault on teardown.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
