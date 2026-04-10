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
MODEL_PATH = "weather_model_1_cyclic_best_edgetpu.tflite"
SCALER_PATH = "scaler_params.json"
WINDOW_SIZE = 1
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 100000
FEATURE_COUNT = 14

PROGRESS_PATH = "progress.json"

def save_progress(index):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_index": index}, f)

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f).get("last_index", 0)
    return 0

FEATURE_ORDER = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 
    'time_of_day_sin', 'time_of_day_cos', 'day_of_year_sin', 'day_of_year_cos',
    'temperature_delta', 'temp_lag1', 'humidity_lag1'
]

# --- Functions ---

def predict_on_window(window, interpreter, input_index, output_index):
    start = time.perf_counter()
    window_12 = window[:, :FEATURE_COUNT]
    input_scale, input_zero_point = input_details[0]['quantization']
    input_tensor = np.clip(np.round(window_12[0] / input_scale + input_zero_point), -128, 127).astype(np.int8).reshape(1, FEATURE_COUNT)
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

def create_influx_point(timestamp, actuals, preds, temp_lag1):
    return {
        "measurement": "model_1_periodic",
        "time": timestamp.isoformat(),
        "fields": {
            "actual_1hr_temperature": float(actuals['temp_t+1hr']),
            "actual_2hr_temperature": float(actuals['temp_t+2hr']),
            "actual_3hr_temperature": float(actuals['temp_t+3hr']),
            "actual_temp_lag_1": float(temp_lag1),
            "pred_1hr_temperature": float(preds[0]),
            "pred_2hr_temperature": float(preds[1]),
            "pred_3hr_temperature": float(preds[2])
        }
    }

# --- Load Scaler Parameters ---
print("Loading input scaler...")
with open("input_scaler_cyclic.json", "r") as f:
    input_scaler = json.load(f)

# --- Load Target Scaler ---
print("Loading target scaler...")
with open("target_scaler_cyclic.json", "r") as f:
    target_scaler = json.load(f)
y_min = target_scaler["min"]
y_max = target_scaler["max"]

# --- Connect to InfluxDB ---
print("Connecting to InfluxDB...")
client = InfluxDBClient(
    host="10.0.1.141",
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
        query = f'SELECT * FROM "wf/obs_st" WHERE time > \'{last_ts}\' LIMIT {QUERY_BATCH_SIZE + extra_samples}'
    else:
        query = f'SELECT * FROM "wf/obs_st" LIMIT {QUERY_BATCH_SIZE + 250}'
        drop_measurements = True
else:
    query = f'SELECT * FROM "wf/obs_st" LIMIT {QUERY_BATCH_SIZE + 250}'
    drop_measurements = True
result = client.query(query)
points = list(result.get_points())

# --- Load DataFrame ---
df = pd.DataFrame(points)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Add derived features (added to training data when exported from InfluxDB)
df['day_of_year'] = df.index.dayofyear
df['time_of_day'] = df.index.hour + df.index.minute / 60.0

# Add cyclic features for time_of_day and day_of_year (matching training script)
# Convert time_of_day to radians (0-24 hours -> 0-2π)
df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)

# Convert day_of_year to radians (1-366 days -> 0-2π)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)

# --- Feature Engineering ---
# Compute temperature_delta as the slope over a backward-looking rolling window (like training script)
def compute_temperature_delta(series, window=15):
    values = series.values
    slopes = np.full_like(values, fill_value=np.nan, dtype=np.float32)
    x = np.arange(window)

    # Start from window-1 to ensure we have enough historical data
    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1]  # Backward-looking window
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        slope = np.dot(x - x.mean(), y - y_mean) / np.dot(x - x.mean(), x - x.mean())
        slopes[i] = slope

    return pd.Series(slopes, index=series.index)

df['temperature_delta'] = compute_temperature_delta(df['temperature'], window=15)
df['temp_lag1'] = df['temperature'].shift(1)
df['humidity_lag1'] = df['relative_humidity'].shift(1)


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
print("✅ Total rows after dropping NaNs:", len(df))
print(f"📊 Data range: {df.index.min()} to {df.index.max()}")

# Normalize and run inference
print("Running inference...")
inference_times = []
prediction_points = []

if drop_measurements:
    print("🧹 Dropping old prediction measurements...")
    try:
        client.query('DROP MEASUREMENT "model_1_periodic"')
        print('  ✔ Dropped measurement: model_1_periodic')
    except Exception as e:
        print(f"  ⚠️ Could not drop model_1_periodic: {e}")

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
        temp_lag1 = df.iloc[target_idx]['temp_lag1']
        prediction_points.append(create_influx_point(timestamp, targets, preds, temp_lag1))

        # Write every 1000 predictions
        if len(inference_times) % (BATCH_SIZE / 4) == 0:
            print(f"After {len(inference_times)} runs:")
            print(f"  Avg Inference Time: {np.mean(inference_times):.2f} ms")
            print(f"  Min Inference Time: {np.min(inference_times):.2f} ms")
            print(f"  Max Inference Time: {np.max(inference_times):.2f} ms")

            client.write_points(prediction_points, time_precision='ms')
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
    client.write_points(prediction_points, time_precision='ms')

print("✅ Inference complete.")
