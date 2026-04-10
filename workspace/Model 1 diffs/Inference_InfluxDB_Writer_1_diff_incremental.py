# Inference_InfluxDB_Writer_incremental.py
# Incremental inference script that processes only new data since last prediction

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
MODEL_PATH = "weather_model_1_diff_best_edgetpu.tflite"
SCALER_PATH = "scaler_params.json"
WINDOW_SIZE = 1
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 100000
FEATURE_COUNT = 36

# Maximum lag needed for features (temp_lag120 = 120 samples = 240 minutes = 4 hours)
MAX_LAG_SAMPLES = 120
# Delta window size
DELTA_WINDOW = 15
# Future targets (temp_t+3hr = 180 samples = 360 minutes = 6 hours)
FUTURE_TARGET_SAMPLES = 180
# Buffer for safety (extra samples to ensure we have complete data)
HISTORICAL_BUFFER = 50  # ~100 minutes buffer

FEATURE_ORDER = [
    'illuminance_delta', 'solar_radiation_delta', 'uv', 'wind_avg', 'wind_gust',
    'day_of_year_sin', 'day_of_year_cos',
    'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
    'time_sin_solar', 'time_cos_solar', 'time_sin_uv', 'time_cos_uv',
    'time_sin_wind', 'time_cos_wind', 'time_sin_temp_lag', 'time_cos_temp_lag',
    'season_sin_temp_lag', 'season_cos_temp_lag', 'season_sin_humidity', 'season_cos_humidity',
    'temperature_delta', 'pressure_delta', 'humidity_delta',
    'temp_lag30', 'humidity_lag30', 'temp_lag60', 'humidity_lag60', 'temp_lag120', 'humidity_lag120',
    'wind_avg_lag30', 'wind_gust_lag30', 'uv_lag30', 'pressure_lag30'
]

# --- Functions ---

def predict_on_window(window, interpreter, input_index, output_index):
    start = time.perf_counter()
    window_36 = window[:, :FEATURE_COUNT]
    input_scale, input_zero_point = input_details[0]['quantization']
    input_tensor = np.clip(np.round(window_36[0] / input_scale + input_zero_point), -128, 127).astype(np.int8).reshape(1, FEATURE_COUNT)
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
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
        "measurement": "model_1_diff",
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

def get_last_prediction_timestamp(writer_client):
    """Query writer_client to get the timestamp of the last written prediction."""
    try:
        query = 'SELECT time FROM "model_1_diff" ORDER BY time DESC LIMIT 1'
        result = writer_client.query(query)
        points = list(result.get_points())
        if points:
            last_time = points[0]['time']
            print(f"📅 Last prediction found at: {last_time}")
            return pd.to_datetime(last_time)
        else:
            print("📅 No previous predictions found. Will process from beginning.")
            return None
    except Exception as e:
        print(f"⚠️ Error querying last prediction: {e}")
        print("📅 Assuming no previous predictions. Will process from beginning.")
        return None

def compute_temperature_delta(series, window=15):
    """Vectorized slope computation over a rolling window for performance on Raspberry Pi 5."""
    values = series.to_numpy(dtype=np.float32)
    n = len(values)
    if n < window:
        return pd.Series(np.full(n, np.nan, dtype=np.float32), index=series.index)

    # Create overlapping windows view of shape (n - window + 1, window)
    y_rolling = np.lib.stride_tricks.sliding_window_view(values, window)
    x = np.arange(window, dtype=np.float32)
    x_demean = x - x.mean()
    denom = np.dot(x_demean, x_demean)

    # Compute mean-centered slopes using einsum for efficiency
    y_mean = y_rolling.mean(axis=1)
    slopes = np.einsum('ij,j->i', (y_rolling - y_mean[:, None]), x_demean) / denom

    # Pad with NaNs at the start to preserve alignment
    slopes = np.concatenate([np.full(window - 1, np.nan, dtype=np.float32), slopes])
    return pd.Series(slopes, index=series.index)

# --- Load Scaler Parameters ---
print("Loading input scaler...")
with open("input_scaler_diff.json", "r") as f:
    input_scaler = json.load(f)

# --- Load Target Scaler ---
print("Loading target scaler...")
with open("target_scaler_diff.json", "r") as f:
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

# --- Get Last Prediction Timestamp ---
print("🔍 Checking for last written prediction...")
last_prediction_time = get_last_prediction_timestamp(writer_client)

# --- Query Data from Reader Client ---
print("📥 Querying weather data from reader_client...")
fields = "temperature, relative_humidity, station_pressure, uv, solar_radiation, illuminance, wind_avg, wind_gust"

if last_prediction_time is None:
    # First run: process from beginning (no previous predictions found)
    print("🆕 First run: processing from beginning")
    # Need historical buffer + future targets
    total_samples_needed = MAX_LAG_SAMPLES + HISTORICAL_BUFFER + FUTURE_TARGET_SAMPLES + QUERY_BATCH_SIZE
    query = f'SELECT {fields} FROM "wf/obs_st" ORDER BY time ASC LIMIT {total_samples_needed}'
    processing_start_time = None
else:
    # Incremental run: start from (last_prediction_time - historical buffer)
    # We need historical data to compute features for the first new prediction
    historical_samples = MAX_LAG_SAMPLES + HISTORICAL_BUFFER
    # Estimate time needed: each sample is ~2 minutes
    historical_minutes = historical_samples * 2
    query_start_time = last_prediction_time - pd.Timedelta(minutes=historical_minutes)
    
    # Also need future targets, so query enough new data
    # Query a reasonable batch size plus future targets
    future_samples = FUTURE_TARGET_SAMPLES + QUERY_BATCH_SIZE
    query = f'SELECT {fields} FROM "wf/obs_st" WHERE time >= \'{query_start_time.isoformat()}\' ORDER BY time ASC LIMIT {historical_samples + future_samples}'
    print(f"📅 Querying from {query_start_time} (to get historical context)")
    processing_start_time = last_prediction_time

result = reader_client.query(query)
points = list(result.get_points())

if not points:
    print("❌ No data found. Exiting.")
    sys.exit(0)

print(f"✅ Retrieved {len(points)} data points")

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
df.sort_index(inplace=True)  # Ensure chronological order

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

# --- Feature Engineering ---
print("🔧 Computing features...")
df['temperature_delta'] = compute_temperature_delta(df['temperature'], window=DELTA_WINDOW)
df['pressure_delta'] = compute_temperature_delta(df['station_pressure'], window=DELTA_WINDOW)
df['humidity_delta'] = compute_temperature_delta(df['relative_humidity'], window=DELTA_WINDOW)
df['illuminance_delta'] = compute_temperature_delta(df['illuminance'], window=DELTA_WINDOW)
df['solar_radiation_delta'] = compute_temperature_delta(df['solar_radiation'], window=DELTA_WINDOW)
df['temp_lag30'] = df['temperature'].shift(30)
df['humidity_lag30'] = df['relative_humidity'].shift(30)

# Additional lag features for multiple time horizons
df['temp_lag60'] = df['temperature'].shift(60)   # 1 hour ago
df['temp_lag120'] = df['temperature'].shift(120) # 2 hours ago
df['humidity_lag60'] = df['relative_humidity'].shift(60)   # 1 hour ago
df['humidity_lag120'] = df['relative_humidity'].shift(120) # 2 hours ago

# Add interaction features to capture complex relationships
# Time-of-day interactions with environmental factors
df['time_sin_solar'] = df['time_of_day_sin'] * df['solar_radiation_delta']
df['time_cos_solar'] = df['time_of_day_cos'] * df['solar_radiation_delta']
df['time_sin_uv'] = df['time_of_day_sin'] * df['uv']
df['time_cos_uv'] = df['time_of_day_cos'] * df['uv']

# Additional interaction features - wind and temperature lag interactions
df['time_sin_wind'] = df['time_of_day_sin'] * df['wind_avg']
df['time_cos_wind'] = df['time_of_day_cos'] * df['wind_avg']
df['time_sin_temp_lag'] = df['time_of_day_sin'] * df['temp_lag30']
df['time_cos_temp_lag'] = df['time_of_day_cos'] * df['temp_lag30']

# Add seasonal-time interactions (how environmental effects vary by season)
df['season_sin_temp_lag'] = df['day_of_year_sin'] * df['temp_lag30']
df['season_cos_temp_lag'] = df['day_of_year_cos'] * df['temp_lag30']
df['season_sin_humidity'] = df['day_of_year_sin'] * df['humidity_lag30']
df['season_cos_humidity'] = df['day_of_year_cos'] * df['humidity_lag30']

# Add more environmental lag features (wind, UV, pressure)
df['wind_avg_lag30'] = df['wind_avg'].shift(30)
df['wind_gust_lag30'] = df['wind_gust'].shift(30)
df['uv_lag30'] = df['uv'].shift(30)
df['pressure_lag30'] = df['station_pressure'].shift(30)

# Targets
print("🎯 Creating shifted targets...")
df['temp_t+1hr'] = df['temperature'].shift(-60)
df['temp_t+2hr'] = df['temperature'].shift(-120)
df['temp_t+3hr'] = df['temperature'].shift(-180)

print("❓ Missing target counts:")
print(df[['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']].isna().sum())

print("\n🧪 Total rows before processing:", len(df))
print(f"📊 Data range: {df.index.min()} to {df.index.max()}")

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
if len(df) > 0:
    print(f"📊 Processable data range: {df.index.min()} to {df.index.max()}")

# --- Determine Processing Start Index ---
start_index = 0
if processing_start_time is not None:
    # Find the first index AFTER last_prediction_time
    # We've already processed up to and including last_prediction_time
    try:
        # Find indices where time > processing_start_time (strictly greater)
        mask = df.index > processing_start_time
        if mask.any():
            first_new_idx = df.index[mask][0]
            start_index = df.index.get_loc(first_new_idx)
            print(f"🔁 Starting processing from index {start_index} (timestamp: {df.index[start_index]})")
            print(f"   Previous last prediction was at: {processing_start_time}")
        else:
            print(f"⚠️ No new data found after {processing_start_time}. Exiting.")
            sys.exit(0)
    except Exception as e:
        print(f"⚠️ Error finding start index: {e}")
        print("Starting from beginning...")
        start_index = 0
else:
    print("🆕 First run: processing from beginning")

# Ensure we have enough historical context for feature engineering
# We need at least MAX_LAG_SAMPLES rows before start_index to compute all features
if start_index < MAX_LAG_SAMPLES:
    print(f"⚠️ Warning: start_index ({start_index}) is less than MAX_LAG_SAMPLES ({MAX_LAG_SAMPLES})")
    print("   Some features may have NaNs. Adjusting start_index...")
    start_index = MAX_LAG_SAMPLES

# Check if there's anything to process
if start_index >= len(df) - WINDOW_SIZE:
    print("✅ No new data to process. Up to date!")
    sys.exit(0)

# Normalize and run inference
print("🚀 Running inference...")
print("📝 Note: This script will append to existing predictions (incremental mode)")
inference_times = []
prediction_points = []

resume_time = df.index[start_index] if start_index < len(df) else "N/A"
print(f"🔁 Processing from index {start_index} / {len(df) - WINDOW_SIZE} at {resume_time}")
print(f"📈 Will process {len(df) - WINDOW_SIZE - start_index} predictions")

for i in range(start_index, len(df) - WINDOW_SIZE):
    window_df = df.iloc[i:i+WINDOW_SIZE]
    target_idx = i + WINDOW_SIZE
    targets = df.iloc[target_idx][['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']]

    # Normalize
    window = window_df[FEATURE_ORDER].values
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

        # Write every 1000 predictions
        if len(inference_times) % (BATCH_SIZE / 4) == 0:
            print(f"After {len(inference_times)} runs:")
            print(f"  Avg Inference Time: {np.mean(inference_times):.2f} ms")
            print(f"  Min Inference Time: {np.min(inference_times):.2f} ms")
            print(f"  Max Inference Time: {np.max(inference_times):.2f} ms")
            print(f"  Last timestamp: {timestamp}")

            writer_client.write_points(prediction_points, time_precision='ms')
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
    print(f"✅ Wrote final batch of {len(prediction_points)} predictions")

print(f"✅ Inference complete. Processed {len(inference_times)} new predictions.")
if inference_times:
    print(f"📊 Inference statistics:")
    print(f"  Avg Inference Time: {np.mean(inference_times):.2f} ms")
    print(f"  Min Inference Time: {np.min(inference_times):.2f} ms")
    print(f"  Max Inference Time: {np.max(inference_times):.2f} ms")

