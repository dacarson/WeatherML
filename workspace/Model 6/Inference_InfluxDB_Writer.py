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
import gc

# --- Configuration ---
MODEL_PATH = "weather_model_6_best_edgetpu.tflite"
SCALER_PATH = "scaler_params.json"
WINDOW_SIZE = 1
BATCH_SIZE = 50000
QUERY_BATCH_SIZE = 100000
# FEATURE_COUNT will be determined dynamically based on available features
# Base: 23 features (removed solar_radiation_delta) + optional wind_direction (2), wind_lull (1), rain_accumulated (1)

PROGRESS_PATH = "progress_diff.json"

def save_progress(index):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_index": index}, f)

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f).get("last_index", 0)
    return 0

FEATURE_ORDER_BASE = [
    'illuminance_delta', 'uv', 'wind_avg', 'wind_gust', 
    'day_of_year_sin', 'day_of_year_cos',
    'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
    'temperature_delta', 'pressure_delta', 'humidity_delta', 
    'temp_lag30', 'humidity_lag30', 'temp_lag60', 'humidity_lag60', 'temp_lag120', 'humidity_lag120',
    'wind_avg_lag30', 'wind_gust_lag30', 'uv_lag30', 'pressure_lag30',
    # Context features - help model understand state without giving absolute current value
    'solar_radiation_deviation', 'solar_clear_sky_ratio', 'clear_sky_deficit',
    'solar_radiation_variance_30min', 'solar_radiation_change_30min', 'solar_radiation_change_60min',
    'solar_radiation_mean_30min', 'solar_radiation_std_30min', 'solar_radiation_mean_60min',
    'solar_clear_sky_ratio_mean_30min', 'sky_clarity_trend_30min',
    'uv_mean_30min', 'uv_std_30min',
    'humidity_mean_30min', 'humidity_std_30min',
    'temperature_mean_30min', 'temperature_std_30min',
    'temperature_change_30min', 'humidity_change_30min',
    'solar_illuminance_ratio', 'fog_likelihood', 'fog_indicator',
    'marine_push_score', 'marine_push_flag'
]

# --- Functions ---

def predict_on_window(window, interpreter, input_details, output_details, input_index):
    """Run inference on a window of features.
    
    Args:
        window: Feature window array
        interpreter: TFLite interpreter instance
        input_details: Input tensor details (passed explicitly to avoid global dependency)
        output_details: Output tensor details (passed explicitly to avoid global dependency)
        input_index: Input tensor index
    
    Returns:
        tuple: (output_rescaled, inference_time_ms)
    """
    start = time.perf_counter()
    window_features = window[:, :FEATURE_COUNT]
    input_scale, input_zero_point = input_details[0]['quantization']
    input_tensor = np.clip(np.round(window_features[0] / input_scale + input_zero_point), -128, 127).astype(np.int8).reshape(1, FEATURE_COUNT)
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
    # Rescale model output from normalized [-1, 1] to real solar radiation difference using correct inverse scaling
    output = np.clip(output, -1.0, 1.0)
    output_rescaled = 0.5 * (output + 1.0) * (y_max - y_min) + y_min

    inference_time_ms = (time.perf_counter() - start) * 1000
    return output_rescaled, inference_time_ms

def create_influx_point(timestamp, actuals, preds, current_solar_radiation):
    # preds are solar radiation differences, so add current solar radiation to get actual predicted solar radiation
    pred_30min_solar = float(preds[0]) + float(current_solar_radiation)
    pred_60min_solar = float(preds[1]) + float(current_solar_radiation)
    pred_90min_solar = float(preds[2]) + float(current_solar_radiation)
    
    return {
        "measurement": "model_6",
        "time": timestamp.isoformat(),
        "fields": {
            "actual_30min_solar_radiation": float(actuals['solar_radiation_t+30min']),
            "actual_60min_solar_radiation": float(actuals['solar_radiation_t+60min']),
            "actual_90min_solar_radiation": float(actuals['solar_radiation_t+90min']),
            "pred_30min_solar_radiation": pred_30min_solar,
            "pred_60min_solar_radiation": pred_60min_solar,
            "pred_90min_solar_radiation": pred_90min_solar
        }
    }

# --- Load Scaler Parameters ---
print("Loading input scaler...")
with open("input_scaler_6.json", "r") as f:
    input_scaler = json.load(f)

# --- Load Target Scaler ---
print("Loading target scaler...")
with open("target_scaler_6.json", "r") as f:
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
fields = "temperature, relative_humidity, station_pressure, uv, solar_radiation, illuminance, wind_avg, wind_gust, wind_direction, wind_lull, rain_accumulated"
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as f:
        last_ts = json.load(f).get("last_timestamp")
        resume_datetime = pd.to_datetime(last_ts)
        print(f"Resuming from {resume_datetime}")
    if last_ts:
        # When resuming, query significantly more data to account for:
        # - Feature engineering needs (temperature_delta window: 15 samples)
        # - Future targets (solar_radiation_t+90min needs 90 samples)
        # - Processing buffer to ensure we have complete data
        # Query enough extra data to ensure we have complete targets after shifting
        # Need 90 samples for solar_radiation_t+90min target, plus buffer for other features
        extra_samples = 250  # Extra buffer beyond the 90 needed for furthest target
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

# Add cyclical encoding for wind_direction (0-360 degrees) - very useful for local weather patterns
if 'wind_direction' in df.columns:
    df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360.0)
    df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360.0)

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

# Add features to help model understand current solar radiation state without giving absolute value
# Compute expected solar radiation based on time-of-day and season
time_from_noon = (df['time_of_day'].values - 12) / 12.0
time_from_noon = np.clip(time_from_noon, -1, 1)
daylight_factor = np.maximum(0, np.cos(np.pi * time_from_noon)) ** 2
seasonal_factor = (df['day_of_year_cos'].values + 1) / 2

df['time_bin'] = pd.cut(df['time_of_day'], bins=48, labels=False)
df['season_bin'] = pd.cut(df['day_of_year'], bins=12, labels=False)

# Compute expected solar radiation from historical averages per time/season bin
expected_solar = df.groupby(['time_bin', 'season_bin'])['solar_radiation'].transform('mean')
df['expected_solar_radiation'] = expected_solar.fillna(df['solar_radiation'].mean())

# Deviation from expected - tells model if above/below expected without giving absolute value
df['solar_radiation_deviation'] = df['solar_radiation'] - df['expected_solar_radiation']

eps = 1e-6
expected_safe = np.maximum(df['expected_solar_radiation'], 50.0)
df['solar_clear_sky_ratio'] = df['solar_radiation'] / (expected_safe + eps)
df['solar_clear_sky_ratio'] = df['solar_clear_sky_ratio'].clip(lower=0.0, upper=5.0)
df['clear_sky_deficit'] = np.maximum(df['expected_solar_radiation'] - df['solar_radiation'], 0.0)
df['clear_sky_deficit'] = df['clear_sky_deficit'].clip(upper=800.0)

def add_rolling_stats(df_obj):
    df_obj['solar_radiation_mean_30min'] = df_obj['solar_radiation'].rolling(window=30, min_periods=1).mean()
    df_obj['solar_radiation_std_30min'] = df_obj['solar_radiation'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=200.0)
    df_obj['solar_radiation_mean_60min'] = df_obj['solar_radiation'].rolling(window=60, min_periods=1).mean()

    df_obj['uv_mean_30min'] = df_obj['uv'].rolling(window=30, min_periods=1).mean()
    df_obj['uv_std_30min'] = df_obj['uv'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=5.0)

    df_obj['humidity_mean_30min'] = df_obj['relative_humidity'].rolling(window=30, min_periods=1).mean()
    df_obj['humidity_std_30min'] = df_obj['relative_humidity'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=20.0)

    df_obj['temperature_mean_30min'] = df_obj['temperature'].rolling(window=30, min_periods=1).mean()
    df_obj['temperature_std_30min'] = df_obj['temperature'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=10.0)

    df_obj['solar_clear_sky_ratio_mean_30min'] = df_obj['solar_clear_sky_ratio'].rolling(window=30, min_periods=1).mean()
    df_obj['solar_clear_sky_ratio_mean_30min'] = df_obj['solar_clear_sky_ratio_mean_30min'].clip(lower=0.0, upper=5.0)
    df_obj['sky_clarity_trend_30min'] = (df_obj['solar_clear_sky_ratio'] - df_obj['solar_clear_sky_ratio_mean_30min']).clip(lower=-3.0, upper=3.0)

add_rolling_stats(df)

# Solar radiation to illuminance ratio - helps detect clear vs hazy/cloudy conditions
df['solar_illuminance_ratio'] = df['solar_radiation'] / (df['illuminance'] + 1e-6)

# Rolling variance in solar radiation - low variance indicates stable/clear conditions
def compute_rolling_variance(series, window=30):
    """Compute rolling variance for solar radiation stability detection."""
    values = series.to_numpy(dtype=np.float32)
    n = len(values)
    if n < window:
        return pd.Series(np.full(n, np.nan, dtype=np.float32), index=series.index)
    
    variances = []
    for i in range(n):
        if i < window - 1:
            variances.append(np.nan)
        else:
            window_data = values[i - window + 1:i + 1]
            if np.any(np.isnan(window_data)):
                variances.append(np.nan)
            else:
                variances.append(np.var(window_data))
    
    return pd.Series(variances, index=series.index, dtype=np.float32)

df['solar_radiation_variance_30min'] = compute_rolling_variance(df['solar_radiation'], window=30)

# Solar radiation change features - how much has it changed recently (gives trend context)
df['solar_radiation_change_30min'] = df['solar_radiation'] - df['solar_radiation'].shift(30)
df['solar_radiation_change_60min'] = df['solar_radiation'] - df['solar_radiation'].shift(60)

df['temperature_change_30min'] = df['temperature'] - df['temp_lag30']
df['humidity_change_30min'] = df['relative_humidity'] - df['humidity_lag30']

def compute_marine_features(df_obj):
    if 'wind_direction' in df_obj.columns:
        direction_rad = np.deg2rad(df_obj['wind_direction'].values)
    elif 'wind_direction_sin' in df_obj.columns and 'wind_direction_cos' in df_obj.columns:
        direction_rad = np.arctan2(df_obj['wind_direction_sin'].values, df_obj['wind_direction_cos'].values)
        direction_rad = (direction_rad + 2 * np.pi) % (2 * np.pi)
    else:
        direction_rad = None

    if direction_rad is not None:
        onshore_component = np.cos(direction_rad - (3 * np.pi / 2))
        onshore_component = np.clip(onshore_component, -1.0, 1.0)
    else:
        onshore_component = np.zeros(len(df_obj))

    marine_push = np.clip(onshore_component, 0.0, 1.0) * df_obj['wind_avg'].values
    humidity_norm = np.clip((df_obj['relative_humidity'].values - 80.0) / 20.0, 0.0, 1.0)

    df_obj['marine_push_score'] = marine_push * (0.5 + 0.5 * humidity_norm)
    df_obj['marine_push_flag'] = ((np.clip(onshore_component, 0.0, 1.0) > 0.5) & (df_obj['wind_avg'].values > 2.0) & (df_obj['relative_humidity'].values > 85.0)).astype(np.float32)

    expected_safe_vals = np.maximum(df_obj['expected_solar_radiation'].values, 50.0)
    ratio_deficit = np.clip((0.7 - df_obj['solar_clear_sky_ratio'].values) / 0.7, 0.0, 1.0)
    deficit_norm = np.clip(df_obj['clear_sky_deficit'].values / (expected_safe_vals + eps), 0.0, 1.0)
    dimming = np.clip(-df_obj['solar_radiation_change_30min'].fillna(0.0).values / 50.0, 0.0, 1.0)
    fog_likelihood = 0.3 * humidity_norm + 0.35 * ratio_deficit + 0.25 * deficit_norm + 0.10 * dimming
    df_obj['fog_likelihood'] = np.clip(fog_likelihood, 0.0, 1.0)
    df_obj['fog_indicator'] = (df_obj['fog_likelihood'] >= 0.6).astype(np.float32)

compute_marine_features(df)
# Clean up temporary columns
df.drop(columns=['time_bin', 'season_bin', 'expected_solar_radiation'], inplace=True, errors='ignore')

# Additional lag features for multiple time horizons
df['temp_lag60'] = df['temperature'].shift(60)   # 1 hour ago
df['temp_lag120'] = df['temperature'].shift(120) # 2 hours ago
df['humidity_lag60'] = df['relative_humidity'].shift(60)   # 1 hour ago
df['humidity_lag120'] = df['relative_humidity'].shift(120) # 2 hours ago

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
df['solar_radiation_t+30min'] = df['solar_radiation'].shift(-30)
df['solar_radiation_t+60min'] = df['solar_radiation'].shift(-60)
df['solar_radiation_t+90min'] = df['solar_radiation'].shift(-90)

# Calculate solar radiation differences
df['solar_radiation_diff_30min'] = df['solar_radiation_t+30min'] - df['solar_radiation']
df['solar_radiation_diff_60min'] = df['solar_radiation_t+60min'] - df['solar_radiation']
df['solar_radiation_diff_90min'] = df['solar_radiation_t+90min'] - df['solar_radiation']

print("❓ Missing target counts:")
print(df[['solar_radiation_t+30min', 'solar_radiation_t+60min', 'solar_radiation_t+90min']].isna().sum())

print("\n🧪 Total rows before processing:", len(df))

# Process the data but keep the extra rows for target validation
required_fields = FEATURE_ORDER + ['solar_radiation_t+30min', 'solar_radiation_t+60min', 'solar_radiation_t+90min', 'solar_radiation_diff_30min', 'solar_radiation_diff_60min', 'solar_radiation_diff_90min']

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
# Use a rolling window for inference times to prevent unbounded memory growth
# Keep only recent statistics matching batch size for accurate recent stats
MAX_INFERENCE_TIME_SAMPLES = BATCH_SIZE
inference_times = []
prediction_points = []
total_inferences = 0  # Track total number of inferences for reset logic

if drop_measurements:
    print("🧹 Dropping old prediction measurements...")
    try:
        writer_client.query('DROP MEASUREMENT "model_6"')
        print('  ✔ Dropped measurement: model_6')
    except Exception as e:
        print(f"  ⚠️ Could not drop model_6: {e}")

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
    targets = df.iloc[target_idx][['solar_radiation_t+30min', 'solar_radiation_t+60min', 'solar_radiation_t+90min', 'solar_radiation_diff_30min', 'solar_radiation_diff_60min', 'solar_radiation_diff_90min']]

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
        preds, t_inf = predict_on_window(scaled_window, interpreter, input_details, output_details, input_index)
    except Exception as e:
        print(f"⚠️ Exception in prediction at row {i}")
        print(f"  Error: {e}")
        continue

    try:
        current_ts = df.index[target_idx].isoformat()
        with open(PROGRESS_PATH, "w") as f:
            json.dump({"last_index": i, "last_timestamp": current_ts}, f)
        inference_times.append(t_inf)
        total_inferences += 1
        
        # Prevent unbounded memory growth by keeping only recent samples
        if len(inference_times) > MAX_INFERENCE_TIME_SAMPLES:
            inference_times.pop(0)

        timestamp = df.index[target_idx]
        current_solar_radiation = df.iloc[target_idx]['solar_radiation']
        prediction_points.append(create_influx_point(timestamp, targets, preds, current_solar_radiation))

        # Write every 12500 predictions (BATCH_SIZE / 4)
        if total_inferences % (BATCH_SIZE / 4) == 0:
            print(f"After {total_inferences} runs:")
            print(f"  Avg Inference Time: {np.mean(inference_times):.2f} ms")
            print(f"  Min Inference Time: {np.min(inference_times):.2f} ms")
            print(f"  Max Inference Time: {np.max(inference_times):.2f} ms")

            writer_client.write_points(prediction_points, time_precision='ms')
            prediction_points = []

        # Reset interpreter every 25k runs (BATCH_SIZE / 2) to avoid TPU HIB errors
        if total_inferences % (BATCH_SIZE / 2) == 0:
            print(f"🔁 Resetting interpreter after {total_inferences} runs to avoid TPU HIB errors...")
            # Explicitly delete old interpreter to free resources
            old_interpreter = interpreter
            del old_interpreter
            # Force garbage collection to ensure EdgeTPU resources are released
            gc.collect()
            # Small delay to allow TPU to fully release resources
            time.sleep(0.1)
            
            # Create new interpreter
            interpreter = tflite.Interpreter(
                model_path=MODEL_PATH,
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
            )
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_index = input_details[0]['index']
            output_index = output_details[0]['index']

        # Restart process every 50k runs (BATCH_SIZE) to fully reset TPU state
        if total_inferences % BATCH_SIZE == 0:
            print(f"🧼 Restarting process after {total_inferences} runs to fully reset TPU state...")
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
