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
MODEL_PATH = "weather_model_1_diff_best_edgetpu.tflite"
SCALER_PATH = "scaler_params.json"
WINDOW_SIZE = 1
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 100000
FEATURE_COUNT = 36

PROGRESS_PATH = "progress_diff.json"

def save_progress(index):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_index": index}, f)

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f).get("last_index", 0)
    return 0

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

# --- Query Historic Data ---
print("Querying data from InfluxDB...")
drop_measurements = False
last_ts = None
# Only select the fields needed for feature generation and inference
fields = "temperature, relative_humidity, station_pressure, uv, solar_radiation, illuminance, wind_avg, wind_gust"
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as f:
        last_ts = json.load(f).get("last_timestamp")
        resume_datetime = pd.to_datetime(last_ts)
        print(f"Resuming from {resume_datetime}")
    if last_ts:
        # When resuming, query significantly more data to account for:
        # - Feature engineering needs (temperature_delta window: 15 samples)
        # - Future targets (temp_t+3hr needs 180 samples)
        # - Processing buffer to ensure we have complete data
        # Query enough extra data to ensure we have complete targets after shifting
        # Need 180 samples for temp_t+3hr target, plus buffer for other features
        extra_samples = 250  # Extra buffer beyond the 180 needed for furthest target
        query = f'SELECT {fields} FROM "wf/obs_st" WHERE time > \'{last_ts}\' LIMIT {QUERY_BATCH_SIZE + extra_samples}'
    else:
        query = f'SELECT {fields} FROM "wf/obs_st" LIMIT {QUERY_BATCH_SIZE + 250}'
        drop_measurements = True
else:
    query = f'SELECT {fields} FROM "wf/obs_st" LIMIT {QUERY_BATCH_SIZE + 250}'
    drop_measurements = True
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

# --- Feature Engineering ---
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

df['temperature_delta'] = compute_temperature_delta(df['temperature'], window=15)
df['pressure_delta'] = compute_temperature_delta(df['station_pressure'], window=15)
df['humidity_delta'] = compute_temperature_delta(df['relative_humidity'], window=15)
df['illuminance_delta'] = compute_temperature_delta(df['illuminance'], window=15)
df['solar_radiation_delta'] = compute_temperature_delta(df['solar_radiation'], window=15)
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
        writer_client.query('DROP MEASUREMENT "model_1_diff"')
        print('  ✔ Dropped measurement: model_1_diff')
    except Exception as e:
        print(f"  ⚠️ Could not drop model_1_diff: {e}")

start_index = 0
if last_ts:
    try:
        start_index = df.index.get_loc(pd.to_datetime(last_ts)) + 1
    except KeyError:
        start_index = 0

resume_time = df.index[start_index] if start_index < len(df) else "N/A"
print(f"🔁 Resuming from index {start_index} / {len(df) - WINDOW_SIZE} at {resume_time}")

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
        current_ts = df.index[target_idx].isoformat()
        with open(PROGRESS_PATH, "w") as f:
            json.dump({"last_index": i, "last_timestamp": current_ts}, f)
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

print("✅ Inference complete.")
