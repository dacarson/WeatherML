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
MODEL_PATH = "weather_model_5a_best_edgetpu.tflite"
SEQ_LEN = 180  # Sequence length matching training (3 hours of history)
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 100000
EXTRA_SAMPLES = 250  # Extra buffer beyond sequence and furthest target
# FEATURE_COUNT will be determined dynamically based on available features
# Base: 24 features + optional wind_direction (2), wind_lull (1), rain_accumulated (1)

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

def predict_on_window(window, interpreter, input_index, output_index):
    start = time.perf_counter()
    # window should be shape (SEQ_LEN, FEATURE_COUNT)
    window_features = window[:, :FEATURE_COUNT]
    input_scale, input_zero_point = input_details[0]['quantization']
    # Reshape to (1, SEQ_LEN, FEATURE_COUNT) for batch inference
    input_tensor = np.clip(np.round(window_features / input_scale + input_zero_point), -128, 127).astype(np.int8).reshape(1, SEQ_LEN, FEATURE_COUNT)
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    # print("🔎 Output Tensors:")
    # for i in range(3):
        # tensor = interpreter.get_tensor(output_details[i]['index'])
        # scale, zero_point = output_details[i]['quantization']
        # print(f"Output {i} (raw): {tensor[0][0]}, scale: {scale}, zero_point: {zero_point}")
        # dequant = (tensor[0][0] - zero_point) * scale
        # print(f"Output {i} (dequantized): {dequant}")
    outputs = []
    for i in range(3):
        tensor = interpreter.get_tensor(output_details[i]['index'])
        q_value = tensor[0][0]
        scale, zero_point = output_details[i]['quantization']
        dequant = (q_value - zero_point) * scale
        outputs.append(dequant)

    output = np.array(outputs, dtype=np.float32)
    # Rescale model output from normalized [-1, 1] to real temperature using correct inverse scaling
    output = np.clip(output, -1.0, 1.0)
    output_rescaled = 0.5 * (output + 1.0) * (y_max - y_min) + y_min

    inference_time_ms = (time.perf_counter() - start) * 1000
    return output_rescaled, inference_time_ms

def create_influx_point(timestamp, actuals, preds, current_temp, temp_lag30):
    # preds are temperature differences, so add current temperature to get actual predicted temperatures
    pred_1hr_temp = float(preds[0]) + float(current_temp)
    pred_2hr_temp = float(preds[1]) + float(current_temp)
    pred_3hr_temp = float(preds[2]) + float(current_temp)
    
    return {
        "measurement": "model_5a",
        "time": timestamp.isoformat(),
        "fields": {
            "actual_1hr_temperature": float(actuals['temp_t+1hr']),
            "actual_2hr_temperature": float(actuals['temp_t+2hr']),
            "actual_3hr_temperature": float(actuals['temp_t+3hr']),
            "pred_1hr_temperature": pred_1hr_temp,
            "pred_2hr_temperature": pred_2hr_temp,
            "pred_3hr_temperature": pred_3hr_temp
        }
    }

# --- Load Scaler Parameters ---
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
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']
output_scale = output_details[0]['quantization'][0]
output_zero_point = int(output_details[0]['quantization'][1])

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
        print("🔎 Progress file unusable; checking last prediction time from InfluxDB (model_5a)...")
        last_pred_result = writer_client.query('SELECT LAST("pred_1hr_temperature") FROM "model_5a"')
        last_points = list(last_pred_result.get_points())
        if last_points:
            last_ts = last_points[0]['time']
            drop_measurements = False  # we are continuing from existing data
            print(f"  ✔ Found last prediction at {last_ts} (from InfluxDB)")
        else:
            print("  ℹ️ No previous predictions found in InfluxDB (model_5a). Will recompute from scratch.")
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
    resume_from = resume_from_dt.isoformat()
    end_dt = resume_from_dt + pd.Timedelta(minutes=QUERY_BATCH_SIZE + EXTRA_SAMPLES)
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
inference_times = []
prediction_points = []

if drop_measurements:
    print("🧹 Dropping old prediction measurements...")
    try:
        writer_client.query('DROP MEASUREMENT "model_5a"')
        print('  ✔ Dropped measurement: model_5a')
    except Exception as e:
        print(f"  ⚠️ Could not drop model_5a: {e}")

start_index = 0
if last_ts:
    try:
        start_index = df.index.get_loc(pd.to_datetime(last_ts)) + 1
    except KeyError:
        start_index = 0

resume_time = df.index[start_index] if start_index < len(df) else "N/A"
# Need SEQ_LEN timesteps for the sequence, plus we need to be able to access targets
# Target at index i requires data from i-SEQ_LEN+1 to i (inclusive), so we need i >= SEQ_LEN-1
min_start = max(start_index, SEQ_LEN - 1)
print(f"🔁 Resuming from index {min_start} / {len(df) - 1} at {resume_time}")
print(f"📊 Using sequence length of {SEQ_LEN} timesteps (3 hours of history)")

for i in range(min_start, len(df)):
    # Build sequence: use timesteps from i-SEQ_LEN+1 to i (inclusive) = SEQ_LEN timesteps
    # This gives us the most recent SEQ_LEN timesteps ending at index i
    window_start = i - SEQ_LEN + 1
    window_df = df.iloc[window_start:i+1]  # i+1 because iloc is exclusive on the end
    target_idx = i
    targets = df.iloc[target_idx][['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']]

    # Normalize - window_df should have exactly SEQ_LEN rows
    if len(window_df) != SEQ_LEN:
        print(f"⚠️ Skipping row {i}: insufficient data for sequence (have {len(window_df)}, need {SEQ_LEN})")
        continue
    
    window = window_df[FEATURE_ORDER].values  # Shape: (SEQ_LEN, FEATURE_COUNT)
    scaled_window = np.empty_like(window)
    for j, feature in enumerate(FEATURE_ORDER):
        f_min = input_scaler[feature]["min"]
        f_max = input_scaler[feature]["max"]
        scaled_window[:, j] = (window[:, j] - f_min) / (f_max - f_min)

    if targets.isnull().any():
        print(f"⚠️ Skipping row {i} due to NaNs in targets: {targets}")
        continue

    try:
        preds, t_inf = predict_on_window(scaled_window, interpreter, input_index, output_index)
    except Exception as e:
        print(f"⚠️ Exception in prediction at row {i}")
        print(f"  Error: {e}")
        continue

    try:
        inference_times.append(t_inf)

        timestamp = df.index[target_idx]
        current_temp = df.iloc[target_idx]['temperature']
        temp_lag30 = df.iloc[target_idx]['temp_lag30']
        prediction_points.append(create_influx_point(timestamp, targets, preds, current_temp, temp_lag30))

        # Write every (BATCH_SIZE / 4) predictions
        if len(inference_times) % (BATCH_SIZE / 4) == 0:
            print(f"After {len(inference_times)} runs:")
            print(f"  Avg Inference Time: {np.mean(inference_times):.2f} ms")
            print(f"  Min Inference Time: {np.min(inference_times):.2f} ms")
            print(f"  Max Inference Time: {np.max(inference_times):.2f} ms")

            writer_client.write_points(prediction_points, time_precision='ms')

            # Update progress file to reflect the last *flushed* prediction timestamp
            last_flushed_ts = prediction_points[-1]["time"]
            try:
                save_progress(last_flushed_ts)
            except Exception as e:
                print(f"  ⚠️ Could not update progress file {PROGRESS_PATH}: {e}")

            prediction_points = []

        # Reset interpreter every 10k runs to avoid TPU HIB errors
        if len(inference_times) % (BATCH_SIZE / 2) == 0:
            print("🔁 Resetting interpreter after 10k runs to avoid TPU HIB errors...")
            interpreter = tflite.Interpreter(
                model_path=MODEL_PATH,
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
            )
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_index = input_details[0]['index']
            output_index = output_details[0]['index']

        # Restart process every 100k runs to fully reset TPU state
        if len(inference_times) % BATCH_SIZE == 0:
            print("🧼 Restarting process after 100k runs to fully reset TPU state...")
            print("🧼 Exiting to allow external restart...")
            sys.exit(88)  # Special exit code recognized by wrapper

    except Exception as e:
        print(f"⚠️ Exception during Influx write prep at row {i}")
        print(f"  preds: shape = {preds.shape}, values = {preds}")
        print(f"  targets: shape = {targets.shape}, values = {targets}")
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
