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

# --- Configuration ---
MODEL_PATH = "weather_model_conv1d_quant_best.tflite"
SCALER_PATH = "scaler_params.json"
WINDOW_SIZE = 90
FEATURE_COUNT = 8

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
    'temp_avg_15min',
    'temperature_delta',
    'sin_time_of_day',
    'cos_time_of_day',
    'illuminance',
    'solar_radiation',
    'station_pressure',
    'relative_humidity'
]

# --- Functions ---

def load_manual_scaler(path):
    with open(path, "r") as f:
        obj = json.load(f)
    return {k: (v["min"], v["max"]) for k, v in obj.items()}

input_scaler = load_manual_scaler("input_scaler.json")

# Load target scaler values
with open("target_scaler.json") as f:
    target_scaler = json.load(f)
Y_mean = np.array(target_scaler["mean"])
Y_scale = np.array(target_scaler["scale"])

def predict_on_window(window, interpreter, input_index, output_index):
    start = time.perf_counter()
    # Ensure shape and quantize
    window_8 = window[:, :FEATURE_COUNT]  # select only 8 features
    scale, zero_point = input_details[0]['quantization']
    input_tensor = np.clip(np.round(window_8 / scale + zero_point), -128, 127).astype(np.int8).reshape(1, WINDOW_SIZE, FEATURE_COUNT)
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)[0]
    inference_time_ms = (time.perf_counter() - start) * 1000
    return output.astype(np.int8), inference_time_ms

def create_influx_point(timestamp, actual, preds):
    return {
        "measurement": "model_1",
        "time": timestamp.isoformat(),
        "fields": {
            "actual_temperature": float(actual),
            "pred_1hr_temperature": float(preds[0]),
            "pred_2hr_temperature": float(preds[1]),
            "pred_3hr_temperature": float(preds[2])
        }
    }

# --- Load Scaler Parameters ---
print("Loading scaler parameters...")

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

# --- Query Historic Data ---
print("Querying data from InfluxDB...")
drop_measurements = False
last_ts = None
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as f:
        last_ts = json.load(f).get("last_timestamp")
    if last_ts:
        query = f'SELECT * FROM "wf/obs_st" WHERE time > \'{last_ts}\''
    else:
        query = 'SELECT * FROM "wf/obs_st"'
        drop_measurements = True
else:
    query = 'SELECT * FROM "wf/obs_st"'
    drop_measurements = True
result = client.query(query)
points = list(result.get_points())

# --- Load DataFrame ---
df = pd.DataFrame(points)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# --- Feature Engineering ---
df['day_of_year'] = df.index.dayofyear
df['time_of_day'] = df.index.hour + df.index.minute / 60.0
df['time_rad'] = 2 * np.pi * df['time_of_day'] / 24.0
df['sin_time_of_day'] = np.sin(df['time_rad'])
df['cos_time_of_day'] = np.cos(df['time_rad'])
df['day_rad'] = 2 * np.pi * df['day_of_year'] / 365.0
df['sin_day_of_year'] = np.sin(df['day_rad'])
df['cos_day_of_year'] = np.cos(df['day_rad'])

# Compute temp_avg_15min as a 15-min rolling average of 'temperature'
df['temp_avg_15min'] = df['temperature'].rolling(window=15, min_periods=1).mean().shift(1)

# Compute temperature_delta as the slope over a rolling window (like training script)
def compute_temperature_delta(series, window=15):
    values = series.values
    slopes = np.full_like(values, fill_value=np.nan, dtype=np.float32)
    x = np.arange(window)

    half_w = window // 2
    for i in range(half_w, len(values) - half_w):
        y = values[i - half_w:i + half_w + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        slope = np.dot(x - x.mean(), y - y_mean) / np.dot(x - x.mean(), x - x.mean())
        slopes[i] = slope

    return pd.Series(slopes, index=series.index)

df['temperature_delta'] = compute_temperature_delta(df['temperature'], window=15)

# Targets
print("Creating shifted targets...")
df['temp_t+1hr'] = df['temperature'].shift(-60)
df['temp_t+2hr'] = df['temperature'].shift(-120)
df['temp_t+3hr'] = df['temperature'].shift(-180)

# Drop missing
required_fields = FEATURE_ORDER + ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']
df.dropna(subset=required_fields, inplace=True)

# Normalize and run inference
print("Running inference...")
inference_times = []
prediction_points = []

start_index = 0
if last_ts:
    try:
        start_index = df.index.get_loc(pd.to_datetime(last_ts)) + 1
    except KeyError:
        start_index = 0

if drop_measurements:
    print("🧹 Starting from index 0, dropping old prediction measurements...")
    for measurement in ["temperature_pred_1hr", "temperature_pred_2hr", "temperature_pred_3hr"]:
        try:
            client.query(f'DROP MEASUREMENT "{measurement}"')
            print(f"  ✔ Dropped measurement: {measurement}")
        except Exception as e:
            print(f"  ⚠️ Could not drop {measurement}: {e}")
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
        f_min, f_max = input_scaler[feature]
        scaled_window[:, j] = (window[:, j] - f_min) / (f_max - f_min)

    try:
        preds, t_inf = predict_on_window(scaled_window, interpreter, input_index, output_index)
        # Dequantize and rescale predictions
        output_scale, output_zero_point = output_details[0]['quantization']
        preds = (preds.astype(np.float32) - output_zero_point) * output_scale
        preds = preds * Y_scale + Y_mean
        current_ts = df.index[target_idx].isoformat()
        with open(PROGRESS_PATH, "w") as f:
            json.dump({"last_index": i, "last_timestamp": current_ts}, f)
        inference_times.append(t_inf)

        timestamp = df.index[target_idx]
        prediction_points.append(create_influx_point(timestamp, targets['temp_t+1hr'], preds))

        # Write every 1000 predictions
        if len(inference_times) % 1000 == 0:
            print(f"After {len(inference_times)} runs:")
            print(f"  Avg Inference Time: {np.mean(inference_times):.2f} ms")
            print(f"  Min Inference Time: {np.min(inference_times):.2f} ms")
            print(f"  Max Inference Time: {np.max(inference_times):.2f} ms")

            client.write_points(prediction_points, time_precision='ms')
            prediction_points = []

        # Reset interpreter every 10k runs to avoid TPU HIB errors
        if len(inference_times) % 10000 == 0:
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
        if len(inference_times) % 100000 == 0:
            print("🧼 Restarting process after 100k runs to fully reset TPU state...")
            print("🧼 Exiting to allow external restart...")
            sys.exit(88)  # Special exit code recognized by wrapper

    except Exception as e:
        print(f"Skipping row {i} due to error: {e}")

# Final flush
if prediction_points:
    client.write_points(prediction_points, time_precision='ms')

print("✅ Inference complete.")
