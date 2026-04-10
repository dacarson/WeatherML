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
# --- Configuration ---
MODEL_PATH = "weather_model_5b_best_edgetpu.tflite"
SEQ_LEN = 180  # Sequence length matching training (3 hours of history)
GAP_STEP_TOLERANCE_S = 180  # Gap policy tolerance - increased to be less aggressive (was 90)
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 200000
EXTRA_SAMPLES = 250  # Extra buffer beyond sequence and furthest target
# FEATURE_COUNT will be determined dynamically based on available features
# Base: temperature, temp_delta_1, + 22 base features + optional wind_direction (2), wind_lull (1), rain_accumulated (1)

PROGRESS_PATH = "progress_diff.json"

def save_progress(last_timestamp):
    """Persist the last *committed* prediction timestamp."""
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_timestamp": last_timestamp}, f)

def load_progress():
    """Return the last committed timestamp from progress file, or None."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f).get("last_timestamp")
    return None

# Base feature order (Model 5b starts with temperature and temp_delta_1)
# Note: Final FEATURE_ORDER will be read from input_scaler_5b.json to match training exactly
FEATURE_ORDER_BASE = [
    'temperature',
    'temp_delta_1',
    'uv', 'wind_avg', 'wind_gust',
    'solar_radiation', 'illuminance',
    'relative_humidity', 'station_pressure',
    'day_of_year_sin', 'day_of_year_cos',
    'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
    # NO lag features - Conv2D learns temporal patterns from the sequence
]

# --- Functions ---

def apply_gap_policy(df: pd.DataFrame, seq_len: int, gap_step_tolerance_s: int) -> pd.DataFrame:
    """Apply the same gap policy used during training.

    - Detect gaps where consecutive timestamps differ by more than gap_step_tolerance_s.
    - Set temp_delta_1 = 0.0 on the first row after a gap.
    - Drop the first seq_len rows after each gap so no seq_len window spans the gap.

    Returns a new DataFrame (copy) with rows removed.
    """
    if df.empty:
        return df

    # Ensure sorted index
    df = df.sort_index()

    dt_s = df.index.to_series().diff().dt.total_seconds()
    gap_mask = dt_s > float(gap_step_tolerance_s)
    gap_positions = np.flatnonzero(gap_mask.to_numpy())  # row positions where a gap starts (row after the gap)

    if gap_positions.size == 0:
        return df
    
    # Diagnostic: show gap details
    print(f"🔍 Gap detection: found {gap_positions.size} gap(s) > {gap_step_tolerance_s}s")
    if gap_positions.size <= 10:  # Only show details if not too many gaps
        for pos in gap_positions[:10]:
            gap_time = df.index[pos]
            gap_size = dt_s.iloc[pos]
            print(f"   Gap at {gap_time}: {gap_size:.1f}s")

    # Zero temp_delta_1 at the first row after each gap (prevents huge delta caused by missing minutes)
    if 'temp_delta_1' in df.columns:
        idx_after_gap = df.index[gap_positions]
        df.loc[idx_after_gap, 'temp_delta_1'] = 0.0

    # GAP POLICY: Row dropping DISABLED
    # Since you have continuous data and use timestamp-based targets (merge_asof), we don't need to drop rows.
    # The model can handle small gaps in sequence windows, and timestamp-based targets ensure correctness.
    # If accuracy degrades with large gaps, we can re-enable selective row dropping, but for now we want
    # continuous predictions without artificial gaps.
    # 
    # The gap detection above is kept for diagnostics only (to understand data quality).
    print(f"ℹ️  Gap policy: Detected {gap_positions.size} gap(s) > {gap_step_tolerance_s}s. Zeroed temp_delta_1 after gaps, but NOT dropping rows (enables continuous predictions).")

    return df

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

    output_rescaled = 0.5 * (output_scaled + 1.0) * (y_maxs - y_mins) + y_mins
    inference_time_ms = (time.perf_counter() - start) * 1000
    return output_rescaled, inference_time_ms

def create_influx_point(timestamp, actuals, preds, current_temp):
    """Create a single InfluxDB point.

    Model outputs `preds` are temperature *diffs* in °C in the convention used at training.
    This script assumes diffs are (future - current), so predicted future temps are:
      pred_temp = current_temp + pred_diff
    """
    # Predicted future temperatures from predicted diffs
    pred_1hr_temp = float(current_temp) + float(preds[0])
    pred_2hr_temp = float(current_temp) + float(preds[1])
    pred_3hr_temp = float(current_temp) + float(preds[2])

    return {
        "measurement": "model_5b",
        "time": timestamp.isoformat(),
        "fields": {
            "actual_1hr_temperature": float(actuals['temp_t+1hr']),
            "actual_2hr_temperature": float(actuals['temp_t+2hr']),
            "actual_3hr_temperature": float(actuals['temp_t+3hr']),
            "pred_1hr_temperature": pred_1hr_temp,
            "pred_2hr_temperature": pred_2hr_temp,
            "pred_3hr_temperature": pred_3hr_temp,
        },
    }

# --- Load Scaler Parameters ---
print("Loading input scaler...")
with open("input_scaler_5b.json", "r") as f:
    input_scaler = json.load(f)

# --- Load Target Scaler ---
print("Loading target scaler...")
with open("target_scaler_5b.json", "r") as f:
    target_scaler = json.load(f)

# Backward compatible target scaler parsing:
# Supported formats:
#  - legacy scalar: {"min": <float>, "max": <float>}
#  - per-horizon lists: {"mins": [m1, m2, m3], "maxs": [M1, M2, M3]}
#  - per-horizon dicts: {"mins": {"diff_1hr": m1, ...}, "maxs": {"diff_1hr": M1, ...}}

_HORIZON_KEYS = [
    "diff_1hr", "diff_2hr", "diff_3hr",
    "1hr", "2hr", "3hr",
    "h1", "h2", "h3",
]

def _as_float(x) -> float:
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError(f"Expected a number, got {type(x)}: {x}")

def _extract_triplet(v, label: str) -> np.ndarray:
    """Extract [1hr,2hr,3hr] floats from dict/list formats."""
    # Dict: try known keys first
    if isinstance(v, dict):
        for k0, k1, k2 in [("diff_1hr", "diff_2hr", "diff_3hr"), ("1hr", "2hr", "3hr"), ("h1", "h2", "h3")]:
            if k0 in v and k1 in v and k2 in v:
                return np.array([_as_float(v[k0]), _as_float(v[k1]), _as_float(v[k2])], dtype=np.float32)
        # If keys are unexpected but exactly 3 items, fall back to sorted keys
        if len(v) == 3:
            ks = sorted(list(v.keys()))
            return np.array([_as_float(v[ks[0]]), _as_float(v[ks[1]]), _as_float(v[ks[2]])], dtype=np.float32)
        raise ValueError(f"Could not parse {label}: dict keys={list(v.keys())}")

    # List/tuple: numbers
    if isinstance(v, (list, tuple)):
        if len(v) != 3:
            raise ValueError(f"Expected 3 values for {label}, got {len(v)}")
        return np.array([_as_float(v[0]), _as_float(v[1]), _as_float(v[2])], dtype=np.float32)

    # Scalar
    return np.array([_as_float(v)] * 3, dtype=np.float32)

if "mins" in target_scaler and "maxs" in target_scaler:
    y_mins = _extract_triplet(target_scaler["mins"], "target_scaler.mins")
    y_maxs = _extract_triplet(target_scaler["maxs"], "target_scaler.maxs")
elif "min" in target_scaler and "max" in target_scaler:
    y_mins = np.array([_as_float(target_scaler["min"])] * 3, dtype=np.float32)
    y_maxs = np.array([_as_float(target_scaler["max"])] * 3, dtype=np.float32)
else:
    raise ValueError(
        "Unsupported target_scaler_5b.json format. Expected keys {mins,maxs} or {min,max}. "
        f"Got keys={list(target_scaler.keys())}"
    )

# Keep legacy scalar names for printing/debugging only
y_min = float(np.min(y_mins))
y_max = float(np.max(y_maxs))

print("🎯 Target scaler bounds (diff) per-horizon:")
print(f"  1hr: min={y_mins[0]:.6f}, max={y_maxs[0]:.6f}")
print(f"  2hr: min={y_mins[1]:.6f}, max={y_maxs[1]:.6f}")
print(f"  3hr: min={y_mins[2]:.6f}, max={y_maxs[2]:.6f}")

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
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def _extract_interpreter_constants(interp_input_details, interp_output_details):
    """Pre-extract all constant values from interpreter details to avoid per-call dict lookups."""
    idx = interp_input_details[0]['index']
    i_scale, i_zp = interp_input_details[0]['quantization']
    o_indices = [interp_output_details[k]['index'] for k in range(3)]
    o_scales   = [interp_output_details[k]['quantization'][0] for k in range(3)]
    o_zps      = [interp_output_details[k]['quantization'][1] for k in range(3)]
    return idx, i_scale, i_zp, o_indices, o_scales, o_zps

input_index, input_scale, input_zero_point, output_indices, output_scales, output_zero_points = \
    _extract_interpreter_constants(input_details, output_details)

# --- Query Historic Data ---
print("Querying data from InfluxDB...")
drop_measurements = False
last_ts = None
progress_exists = os.path.exists(PROGRESS_PATH)

# 1) Primary source of truth: progress_diff.json (if present)
if progress_exists:
    try:
        with open(PROGRESS_PATH) as f:
            progress_data = json.load(f)
        last_ts = progress_data.get("last_timestamp")
        if last_ts:
            print(f"📁 Resuming from progress file; last_timestamp = {last_ts}")
        else:
            print("⚠️ Progress file found but last_timestamp is missing or empty.")
    except Exception as e:
        print(f"⚠️ Could not read progress file {PROGRESS_PATH}: {e}")
        last_ts = None
else:
    # No progress file -> user wants to start from scratch
    print("📄 progress_diff.json not found; starting full recompute from the beginning.")
    drop_measurements = True

# 2) Optional fallback to InfluxDB *only if* a progress file exists but has no usable timestamp
if progress_exists and last_ts is None:
    try:
        print("🔎 Progress file unusable; checking last prediction time from InfluxDB (model_5b)...")
        last_pred_result = writer_client.query('SELECT LAST("pred_1hr_temperature") FROM "model_5b"')
        last_points = list(last_pred_result.get_points())
        if last_points:
            last_ts = last_points[0]['time']
            drop_measurements = False  # we are continuing from existing data
            print(f"  ✔ Found last prediction at {last_ts} (from InfluxDB)")
        else:
            print("  ℹ️ No previous predictions found in InfluxDB (model_5b). Will recompute from scratch.")
            drop_measurements = True
    except Exception as e:
        print(f"  ⚠️ Could not determine last prediction from InfluxDB: {e}")
        # If we got here via a bad progress file, safest is to recompute and drop old predictions
        drop_measurements = True
        last_ts = None

# Only select the fields needed for feature generation and inference
fields = "temperature, relative_humidity, station_pressure, solar_radiation, illuminance, uv, wind_avg, wind_gust, wind_direction, wind_lull, rain_accumulated"

if last_ts:
    # When resuming, pull a lookback window BEFORE last_ts so that:
    # - We can construct a full SEQ_LEN-long sequence for the very next timestep
    # - There are no gaps in predictions between runs.
    last_ts_dt = pd.to_datetime(last_ts)
    resume_from_dt = last_ts_dt - pd.Timedelta(minutes=SEQ_LEN)
    end_dt = resume_from_dt + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
    resume_from = resume_from_dt.isoformat()
    end_ts = end_dt.isoformat()
    print(f"Resuming from {last_ts_dt} with lookback starting at {resume_from_dt}")
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
print(f"📥 Fetched {len(points)} points from InfluxDB")

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

# Temperature difference feature: current timestep temperature minus previous timestep temperature,
# matching the training pipeline's temp_delta_1 feature.
df['temp_delta_1'] = df['temperature'].diff().fillna(0.0)

# ---------------------------------------------------------------------
# REMOVED LAG FEATURES: Conv2D learns temporal patterns from sequence
#
# Model 5b's Conv2D architecture learns temporal dependencies directly
# from the 180-timestep sequence, so explicit lag features are not needed.
# The multi-scale temporal processing and difference computation in the
# Conv2D branch will discover these patterns automatically.
# ---------------------------------------------------------------------

# Apply training-time gap policy so inference windows never span missing-data gaps.
df = apply_gap_policy(df, SEQ_LEN, GAP_STEP_TOLERANCE_S)

# Add cyclical encoding for wind_direction (0-360 degrees) - very useful for local weather patterns
if 'wind_direction' in df.columns:
    df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360.0)
    df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360.0)

# IMPORTANT: Use the exact feature order from training.
# The training script writes `input_scaler_5b.json` in the feature order used to build model inputs.
FEATURE_ORDER = list(input_scaler.keys())

FEATURE_COUNT = len(FEATURE_ORDER)
print(f"📊 Using {FEATURE_COUNT} features (from input_scaler_5b.json): {', '.join(FEATURE_ORDER)}")

# Verify all required features exist in the dataframe
missing_features = [f for f in FEATURE_ORDER if f not in df.columns]
if missing_features:
    raise ValueError(
        f"❌ Missing required features in dataframe: {missing_features}\n"
        f"Available columns: {sorted(df.columns.tolist())}"
    )

# Targets
print("Creating shifted targets...")
# IMPORTANT: Build targets by time offset (future temperature at +1h/+2h/+3h),
# not by row shift, to stay correct even when data has missing/irregular samples.
_temp = df['temperature'].astype(np.float32)

def _future_temp_at(offset: pd.Timedelta, tolerance: pd.Timedelta) -> np.ndarray:
    """Return temperature at (t + offset) using the SAME rule as training.

    Training uses merge_asof(..., direction='forward', tolerance=...) so the target is the
    first sample AT OR AFTER the desired horizon time (within tolerance).
    """
    base = pd.DataFrame({
        "base_time": df.index,
    })
    base["lookup_time"] = base["base_time"] + offset

    src = pd.DataFrame({
        "src_time": df.index,
        "temperature": _temp.to_numpy(dtype=np.float32),
    })

    # df.index is already sorted/monotonic; lookup_time is also monotonic.
    merged = pd.merge_asof(
        base,
        src,
        left_on="lookup_time",
        right_on="src_time",
        direction="forward",
        tolerance=tolerance,
    )

    return merged["temperature"].to_numpy(dtype=np.float32)

# Assuming ~1-minute cadence; allow up to 90s tolerance to accommodate small gaps/jitter.
df['temp_t+1hr'] = _future_temp_at(pd.Timedelta(hours=1), pd.Timedelta(seconds=90))
df['temp_t+2hr'] = _future_temp_at(pd.Timedelta(hours=2), pd.Timedelta(seconds=90))
df['temp_t+3hr'] = _future_temp_at(pd.Timedelta(hours=3), pd.Timedelta(seconds=90))

print("❓ Missing target counts:")
print(df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].isna().sum())

print("\n🧪 Total rows before processing:", len(df))

# Process the data but keep the extra rows for target validation
required_fields = FEATURE_ORDER + ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

# For inference, we only process up to the point where we have complete targets
# This ensures we can validate predictions against actual future values
df.dropna(subset=required_fields, inplace=True)
# Optimize memory usage: convert float64 to float32
float_cols = df.select_dtypes(include=['float64']).columns
if len(float_cols) > 0:
    df[float_cols] = df[float_cols].astype(np.float32)
    print(f"🔧 Converted {len(float_cols)} float64 columns to float32 for efficiency.")
print("✅ Total rows after dropping NaNs:", len(df))
print(f"📊 Data range: {df.index.min()} to {df.index.max()}")

# Normalize and run inference
print("Running inference...")
# Running stats replace a growing list — O(1) update, no full-list scan at flush time.
run_count = 0
run_sum = 0.0
run_min = float('inf')
run_max = float('-inf')
prediction_points = []

# Integer thresholds avoid float division in the hot loop.
# WRITE_EVERY stays coupled to BATCH_SIZE for data-safety (frequent InfluxDB flushes).
# RESET_EVERY / RESTART_EVERY are decoupled so they can be tuned independently to find
# the true HIB error threshold. Increase RESET_EVERY first; if stable, increase RESTART_EVERY.
WRITE_EVERY   = BATCH_SIZE // 4   # flush to InfluxDB every N predictions
# RESET_EVERY removed: interpreter reset at 100k had zero measurable effect across 187k+ inferences
# with no HIB errors. The TPU firmware self-manages via its own ~25-50k flush event (~58-88ms spike).
RESTART_EVERY = 500_000           # full process restart every N predictions (was 200_000)

if drop_measurements:
    print("🧹 Dropping old prediction measurements...")
    try:
        writer_client.query('DROP MEASUREMENT "model_5b"')
        print('  ✔ Dropped measurement: model_5b')
    except Exception as e:
        print(f"  ⚠️ Could not drop model_5b: {e}")

# Calculate start_index AFTER dropping NaNs to ensure valid indices
start_index = 0
if last_ts:
    try:
        last_ts_dt = pd.to_datetime(last_ts)
        # Find the position of the last committed timestamp in the cleaned dataframe
        pos = df.index.get_loc(last_ts_dt)
        # Handle case where get_loc returns a slice or mask (shouldn't happen with unique index, but be safe)
        if isinstance(pos, (slice, pd.Series, np.ndarray)):
            if isinstance(pos, slice):
                pos = pos.start if pos.start is not None else 0
            else:
                pos = int(pos[0]) if len(pos) > 0 else 0
        else:
            pos = int(pos)
        # Start from the next row after the last committed prediction
        start_index = pos + 1
    except (KeyError, IndexError, TypeError):
        # Timestamp not found in dataframe (might have been dropped due to NaNs)
        print(f"⚠️  Last timestamp {last_ts} not found in cleaned dataframe, starting from beginning")
        start_index = 0

# Need SEQ_LEN timesteps for the sequence, plus we need to be able to access targets
# Target at index i requires data from i-SEQ_LEN+1 to i (inclusive), so we need i >= SEQ_LEN-1
min_start = max(start_index, SEQ_LEN - 1)

# Clamp min_start to valid range - ensure we can actually make predictions
if min_start >= len(df):
    print(f"⚠️  No new data to process (calculated start_index={start_index}, min_start={min_start}, df length={len(df)})")
    print(f"    All predictions are up to date!")
    min_start = len(df)  # Set to len(df) so the loop won't execute
    resume_time = "N/A"
else:
    resume_time = df.index[min_start] if min_start < len(df) else "N/A"

print(f"🔁 Resuming from index {min_start} / {len(df) - 1} at {resume_time}")
print(f"📊 Using sequence length of {SEQ_LEN} timesteps (3 hours of history)")

# Pre-normalize the entire feature matrix once to avoid per-window Python overhead.
# This converts ~18 Python-loop iterations × SEQ_LEN rows into a single vectorized op.
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

# Pre-extract targets and temperature as plain numpy arrays for fast indexing.
targets_arr = df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].values.astype(np.float32)
temp_arr = df['temperature'].values.astype(np.float32)
timestamps = df.index

# Pre-allocate the INT8 input buffer once; reused every call to avoid per-prediction allocation.
input_buffer = np.empty((1, SEQ_LEN, FEATURE_COUNT), dtype=np.int8)

for i in range(min_start, len(df)):
    window_start = i - SEQ_LEN + 1

    if window_start < 0:
        print(f"⚠️ Skipping row {i}: insufficient data for sequence")
        continue

    scaled_window = scaled_data[window_start:i+1]  # shape (SEQ_LEN, FEATURE_COUNT)

    if len(scaled_window) != SEQ_LEN:
        print(f"⚠️ Skipping row {i}: insufficient data for sequence (have {len(scaled_window)}, need {SEQ_LEN})")
        continue

    targets_row = targets_arr[i]
    if np.any(np.isnan(targets_row)):
        print(f"⚠️ Skipping row {i} due to NaNs in targets")
        continue

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
        current_temp = float(temp_arr[i])
        targets = {
            'temp_t+1hr': float(targets_arr[i, 0]),
            'temp_t+2hr': float(targets_arr[i, 1]),
            'temp_t+3hr': float(targets_arr[i, 2]),
        }
        prediction_points.append(create_influx_point(timestamp, targets, preds, current_temp))

        if run_count % WRITE_EVERY == 0:
            print(f"After {run_count} runs:")
            print(f"  Avg Inference Time: {run_sum / run_count:.2f} ms")
            print(f"  Min Inference Time: {run_min:.2f} ms")
            print(f"  Max Inference Time: {run_max:.2f} ms")

            writer_client.write_points(prediction_points, time_precision='ms')

            last_flushed_ts = prediction_points[-1]["time"]
            try:
                save_progress(last_flushed_ts)
            except Exception as e:
                print(f"  ⚠️ Could not update progress file {PROGRESS_PATH}: {e}")

            prediction_points = []

        # Restart process fully to reset TPU state.
        if run_count % RESTART_EVERY == 0:
            print("🧼 Restarting process to fully reset TPU state...")
            print("🧼 Exiting to allow external restart...")
            sys.exit(88)  # Special exit code recognized by wrapper

    except Exception as e:
        print(f"⚠️ Exception during Influx write prep at row {i}")
        print(f"  preds: shape = {preds.shape}, values = {preds}")
        print(f"  Error: {e}")
        continue

# Final flush
if prediction_points:
    writer_client.write_points(prediction_points, time_precision='ms')
    # Ensure progress file reflects the last committed prediction
    last_flushed_ts = prediction_points[-1]["time"]
    try:
        save_progress(last_flushed_ts)
    except Exception as e:
        print(f"⚠️ Could not update progress file {PROGRESS_PATH} on final flush: {e}")

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
    sys.exit(88)
