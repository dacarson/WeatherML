import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide GPU devices
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import joblib
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import copy
import subprocess
import json
# Using post-training quantization instead of QAT for better compatibility

TEMP_MIN = 0.0
TEMP_MAX = 50.0

# Load preprocessed data
train_df = pd.read_csv("../train_data.csv")
print("Loaded train_data.csv:", train_df.shape)
val_df = pd.read_csv("../val_data.csv")
print("Loaded val_data.csv:", val_df.shape)

# Parse timestamp as datetime
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])

# Compute delta_minutes and is_gap features
train_df['delta_minutes'] = train_df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
val_df['delta_minutes'] = val_df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
train_df['is_gap'] = (train_df['delta_minutes'] > 1.5).astype(int)
val_df['is_gap'] = (val_df['delta_minutes'] > 1.5).astype(int)

# Add lag features
train_df['temp_lag1'] = train_df['temperature'].shift(1)
train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
val_df['temp_lag1'] = val_df['temperature'].shift(1)
val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)

# Compute temperature_delta only (backward-looking window for real-time inference)
def rolling_slope(series, window):
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
        raw=True
    )

train_df['temperature_delta'] = rolling_slope(train_df['temperature'], window=15)
val_df['temperature_delta'] = rolling_slope(val_df['temperature'], window=15)

print("Added lag and temperature_delta features.")

# ---- Time-of-day and day-of-year features ----
# Assumes 'time_of_day' is already a float in hours
train_df['time_rad'] = 2 * np.pi * train_df['time_of_day'] / 24.0
val_df['time_rad'] = 2 * np.pi * val_df['time_of_day'] / 24.0
train_df['sin_time_of_day'] = np.sin(train_df['time_rad'])
train_df['cos_time_of_day'] = np.cos(train_df['time_rad'])
val_df['sin_time_of_day'] = np.sin(val_df['time_rad'])
val_df['cos_time_of_day'] = np.cos(val_df['time_rad'])

# Drop rows with NaNs
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
print("Dropped NaNs from datasets.")


# Define feature and target columns
features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 'temperature_delta',
    'temp_lag1', 'humidity_lag1', 'sin_time_of_day', 'cos_time_of_day', 'day_of_year',
    'delta_minutes', 'is_gap'
]
targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

baseline_preds = val_df['temp_lag1'].values
baseline_true = val_df['temp_t+1hr'].values
baseline_mae = np.mean(np.abs(baseline_preds - baseline_true))
print(f"Baseline MAE using temp_lag1 for temp_t+1hr: {baseline_mae:.4f}")

# Set window size
window_size = 180  # 180 minutes

# Define domain bounds for clamped min/max
domain_bounds = {
    "wind_gust": (0, None),
    "wind_avg": (0, None),
    "day_of_year": (1, 366),
    "time_of_day": (0, 24),
    "uv": (0, None),
    "relative_humidity": (0, 100),
    "humidity_lag1": (0, 100),
    "illuminance": (0, None),
    "solar_radiation": (0, None),
    "station_pressure": (None, None),
    "temperature_delta": (None, None),
    "temp_lag1": (None, None),
    "sin_time_of_day": (-1, 1),
    "cos_time_of_day": (-1, 1),
    "delta_minutes": (0, 90),
    "is_gap": (0, 1)
}

# Apply clamped min/max scaling with ±5% padding
input_scaler = {}
for feature in features:
    f_min = train_df[feature].min()
    f_max = train_df[feature].max()
    range_pad = 0.05 * (f_max - f_min)
    floor, ceiling = domain_bounds.get(feature, (None, None))
    f_min_adj = max(f_min - range_pad, floor) if floor is not None else f_min - range_pad
    f_max_adj = min(f_max + range_pad, ceiling) if ceiling is not None else f_max + range_pad
    input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
    # Insert check for zero variance before normalization
    if f_max_adj == f_min_adj:
        print(f"⚠️ Zero variance in feature '{feature}', skipping normalization.")
        train_df[feature] = 0.0
        val_df[feature] = 0.0
        continue
    train_df[feature] = (train_df[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    val_df[feature] = (val_df[feature] - f_min_adj) / (f_max_adj - f_min_adj)

# Drop any NaNs that were introduced during normalization
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

# Diagnostic loop to identify features still containing NaNs
for feature in features:
    if train_df[feature].isna().any():
        print(f"🚨 Still NaNs in feature: {feature}")

# Save input scaling ranges
with open("input_scaler.json", "w") as f:
    json.dump(input_scaler, f, indent=2)

def scale_temp(t):
    return (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)

# Inverse scaling for temperature
def inverse_scale_temp(t_scaled):
    return t_scaled * (TEMP_MAX - TEMP_MIN) + TEMP_MIN

for t in targets:
    train_df[t] = train_df[t].map(scale_temp)
    val_df[t] = val_df[t].map(scale_temp)

# Diagnostics for NaN/Inf after normalization
print("Any NaNs in train features:", np.isnan(train_df[features]).values.any())
print("Any Infs in train features:", np.isinf(train_df[features]).values.any())
print("Any NaNs in targets:", np.isnan(train_df[targets]).values.any())
print("Any Infs in targets:", np.isinf(train_df[targets]).values.any())

print("Features normalized with StandardScaler.")

def dump_scaler(min_val, max_val, filename):
    with open(filename, "w") as f:
        json.dump({"min": min_val, "max": max_val}, f)

dump_scaler(TEMP_MIN, TEMP_MAX, "target_scaler_params.json")


# Use a NumPy-based generator approach for windowed dataset creation
def numpy_window_generator(X, y, window_size):
    for i in range(len(X) - window_size):
        X_window = X[i:i+window_size]
        y_target = y[i+window_size]
        yield X_window, y_target

X_train_np = train_df[features].values.astype(np.float32)
y_train_np = train_df[targets].values.astype(np.float32)
X_val_np = val_df[features].values.astype(np.float32)
y_val_np = val_df[targets].values.astype(np.float32)

train_dataset = tf.data.Dataset.from_generator(
    lambda: numpy_window_generator(X_train_np, y_train_np, window_size),
    output_signature=(
        tf.TensorSpec(shape=(window_size, len(features)), dtype=tf.float32),
        tf.TensorSpec(shape=(len(targets),), dtype=tf.float32),
    )
).repeat().batch(32).prefetch(tf.data.AUTOTUNE)

# Function to create validation dataset from generator
def get_val_dataset():
    return tf.data.Dataset.from_generator(
        lambda: numpy_window_generator(X_val_np, y_val_np, window_size),
        output_signature=(
            tf.TensorSpec(shape=(window_size, len(features)), dtype=tf.float32),
            tf.TensorSpec(shape=(len(targets),), dtype=tf.float32),
        )
    ).batch(32).prefetch(tf.data.AUTOTUNE)


def build_and_train_model(name):
    print(f"\n--- Running: {name} ---\n")
    # Residual stack of dilated convolutions (without QAT for compatibility)
    input_layer = tf.keras.layers.Input(shape=(window_size, len(features)))
    def residual_dilated_block(x, filters, kernel_size, dilation_rate):
        shortcut = x
        x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    for dilation in [1, 2, 4, 8]:
        x = residual_dilated_block(x, filters=32, kernel_size=3, dilation_rate=dilation)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu',
                               kernel_regularizer=regularizers.l1(1e-5))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu',
                               kernel_regularizer=regularizers.l1(1e-5))(x)
    x_norm = tf.keras.layers.BatchNormalization()(x)
    output_1 = tf.keras.layers.Dense(1, activation=None, name='t1hr')(x_norm)
    output_2 = tf.keras.layers.Dense(1, activation=None, name='t2hr')(x_norm)
    output_3 = tf.keras.layers.Dense(1, activation=None, name='t3hr')(x_norm)
    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

    print("✅ Built Conv1D model (using post-training quantization)")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss={'t1hr': 'mse', 't2hr': 'mse', 't3hr': 'mse'},
        metrics={'t1hr': 'mae', 't2hr': 'mae', 't3hr': 'mae'}
    )
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting model training...")
    history = model.fit(
        train_dataset.map(lambda x, y: (x, {'t1hr': y[:, 0:1], 't2hr': y[:, 1:2], 't3hr': y[:, 2:3]})),
        validation_data=get_val_dataset().map(lambda x, y: (x, {'t1hr': y[:, 0:1], 't2hr': y[:, 1:2], 't3hr': y[:, 2:3]})),
        epochs=100,
        steps_per_epoch = (len(X_train_np) - window_size) // 32,
        callbacks=[early_stopping]
    )
    print("Model training complete.")

    print("Evaluating model on validation data...")
    mae_sum = np.zeros(len(targets))
    rmse_sum = 0
    count = 0
    total_samples = 0

    for x_batch, y_batch in get_val_dataset():
        preds_raw = model.predict(x_batch, verbose=0)
        y_true = y_batch.numpy()
        y_pred = np.concatenate(preds_raw, axis=1)
        mae_sum += np.sum(np.abs(y_true - y_pred), axis=0)
        rmse_sum += np.sum((y_true - y_pred) ** 2)
        total_samples += y_true.shape[0]
        count += 1

    mae_per_target = mae_sum / total_samples
    val_rmse = np.sqrt(rmse_sum / (total_samples * len(targets)))
    for i, t in enumerate(targets):
        print(f"Validation MAE for {t} (normalized): {mae_per_target[i]:.4f}")
    print(f"Average Validation MAE across all targets: {np.mean(mae_per_target):.4f}")
    print("Validation MAE (normalized):")
    print(f"  t+1hr: {mae_per_target[0]:.2f}")
    print(f"  t+2hr: {mae_per_target[1]:.2f}")
    print(f"  t+3hr: {mae_per_target[2]:.2f}")
    print(f"Validation RMSE (normalized): {val_rmse:.4f}")

    # Compute MAE in Celsius using inverse scaling
    y_true_c = inverse_scale_temp(y_true)
    y_pred_c = inverse_scale_temp(y_pred)
    mae_celsius = np.mean(np.abs(y_true_c - y_pred_c), axis=0)
    print("Validation MAE (in °C):")
    print(f"  t+1hr: {mae_celsius[0]:.2f} °C")
    print(f"  t+2hr: {mae_celsius[1]:.2f} °C")
    print(f"  t+3hr: {mae_celsius[2]:.2f} °C")

    eval_results = model.evaluate(get_val_dataset().map(lambda x, y: (x, {'t1hr': y[:, 0:1], 't2hr': y[:, 1:2], 't3hr': y[:, 2:3]})), verbose=0)
    val_loss = eval_results[0]
    val_mae = np.mean(eval_results[1:])  # average MAE across 3 outputs
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\nFinal Metrics [{name}]:")
    print(f"  val_loss (normalized): {val_loss:.4f}")
    print(f"  val_mae (normalized): {val_mae:.4f}")
    print(f"  Best epoch: {best_epoch}")

    try:
        print("Calculating permutation feature importance...")
        X_val_sample, y_val_sample = [], []
        for x, y in get_val_dataset().unbatch().take(100):
            X_val_sample.append(x.numpy())
            # Safe extraction for 1D tensors
            if isinstance(y, dict):
                y_t1 = y['t1hr'].numpy().squeeze()
                y_t2 = y['t2hr'].numpy().squeeze()
                y_t3 = y['t3hr'].numpy().squeeze()
            else:
                y_np = y.numpy()
                y_t1 = y_np[..., 0].squeeze()
                y_t2 = y_np[..., 1].squeeze()
                y_t3 = y_np[..., 2].squeeze()
            y_val_sample.append([y_t1, y_t2, y_t3])
        y_val_sample = np.array(y_val_sample).reshape((100, 3))

        feature_importance = {}
        for i, feature in enumerate(features):
            X_val_permuted = X_val_sample.copy()
            flat = X_val_permuted[:, :, i].flatten()
            np.random.shuffle(flat)
            X_val_permuted[:, :, i] = flat.reshape(X_val_permuted[:, :, i].shape)
            # Get model prediction on permuted X
            y_pred = model.predict(X_val_permuted, verbose=0)
            # If model outputs a list of heads, stack them
            if isinstance(y_pred, list):
                y_pred = np.concatenate(y_pred, axis=1)
            # Flatten if needed
            if y_pred.ndim == 2 and y_pred.shape[1] == 1 and y_pred.shape[0] == 300:
                y_pred = y_pred.reshape((100, 3))
            if isinstance(y_pred, list):
                y_pred = np.concatenate(y_pred, axis=1)
            # y_val for this sample
            y_val = y_val_sample
            # Compute MSE loss
            permuted_loss = np.mean((y_val - y_pred) ** 2)
            feature_importance[feature] = permuted_loss - val_loss

        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")
    except Exception as e:
        print(f"Skipping feature importance due to error: {e}")
        sorted_importance = []

    # Save results as JSON
    metrics = {
        "name": name,
        "val_loss": float(val_loss),
        "val_mae": float(val_mae),
        "best_epoch": int(best_epoch),
        "average_mae_denormalized": float(np.mean(mae_per_target)),
        "val_rmse_denormalized": float(val_rmse),
        "average_mae_celsius": float(np.mean(mae_celsius)),
        "feature_importance": [(f, float(i)) for f, i in sorted_importance] if sorted_importance else [],
    }
    
    with open(f"results_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Return the trained model for quantization
    return model


def quantize_and_run_inference(model, val_array, window_size):
    # Convert Keras model to TFLite with post-training quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization calibration
    def representative_data_gen():
        # Sample evenly across the entire year of data for better representation
        # Take every 500th sample to get ~1000 samples spread across the year
        num_samples = 1000
        step = max((len(val_array) - window_size) // num_samples, 1)
        for i in range(0, len(val_array) - window_size, step):
            if len(val_array) - i >= window_size:  # Ensure we have enough samples
                window = val_array[i : i + window_size]
                yield [np.expand_dims(window.astype(np.float32), axis=0)]
    
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_new_quantizer = True
    
    tflite_model = converter.convert()
    tflite_filename = "conv1d_quantized.tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved as {tflite_filename}")
    print("Running inference on quantized model for verification...")

    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n🔍 Quantization Parameters:")
    for i, detail in enumerate(output_details):
        print(f"Output {i}: scale = {detail['quantization'][0]}, zero_point = {detail['quantization'][1]}")

    # Use a sample from the validation set
    x_sample = val_array[:window_size]
    x_sample = np.expand_dims(x_sample, axis=0)

    # Quantize input
    input_scale, input_zero_point = input_details[0]['quantization']
    x_quantized = np.round(x_sample / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], x_quantized)
    interpreter.invoke()

    y_pred = []
    for i in range(3):  # t1hr, t2hr, t3hr
        output_scale, output_zero_point = output_details[i]['quantization']
        y = interpreter.get_tensor(output_details[i]['index'])
        y = output_scale * (y.astype(np.float32) - output_zero_point)
        y_pred.append(y[0][0])

    y_pred_c = inverse_scale_temp(np.array(y_pred))
    print("Sample quantized model prediction (in °C):")
    print(f"  t+1hr: {y_pred_c[0]:.2f} °C")
    print(f"  t+2hr: {y_pred_c[1]:.2f} °C")
    print(f"  t+3hr: {y_pred_c[2]:.2f} °C")

    # Copy the best model's .tflite file to weather_model_2.tflite
    import shutil
    shutil.copyfile(tflite_filename, "weather_model_2.tflite")
    print("✅ Copied best model to weather_model_2.tflite")
    
    return tflite_filename


# Clean up old models and checkpoints
print("Cleaning up old models and checkpoints...")
import glob
import shutil

# Remove old model directories
old_models = glob.glob("*_qat_saved_model")
for model_dir in old_models:
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print(f"Removed old model directory: {model_dir}")

# Remove old TFLite files
old_tflite = glob.glob("*.tflite")
for tflite_file in old_tflite:
    if os.path.exists(tflite_file):
        os.remove(tflite_file)
        print(f"Removed old TFLite file: {tflite_file}")

# Remove old results files
old_results = glob.glob("results_*.json")
for results_file in old_results:
    if os.path.exists(results_file):
        os.remove(results_file)
        print(f"Removed old results file: {results_file}")

print("Cleanup complete.\n")

# Run multiple times and select the best model
best_model_metrics = None
best_model_name = None
num_runs = 1

best_model = None
for i in range(num_runs):
    run_name = f"conv1d_run{i+1}"
    print(f"\nTraining {run_name}...")
    model = build_and_train_model(run_name)
    
    with open(f"results_{run_name}.json") as f:
        metrics = json.load(f)

    if (best_model_metrics is None) or (metrics["average_mae_denormalized"] < best_model_metrics["average_mae_denormalized"]):
        best_model_metrics = metrics
        best_model_name = run_name
        best_model = model

print(f"\n✅ Best model: {best_model_name}")
print(f"Average MAE: {best_model_metrics['average_mae_denormalized']:.4f}")

tflite_filename = quantize_and_run_inference(best_model, val_df[features].values.astype(np.float32), window_size)

# --- Validate quantized TFLite model on validation data ---
def validate_quantized_model(tflite_model_path, X_val, y_val, window_size, num_samples=500):
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    print(f"\nValidating TFLite model on {num_samples} samples...")

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scales = [d['quantization'][0] for d in output_details]
    output_zero_points = [d['quantization'][1] for d in output_details]

    print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output scales: {output_scales}")
    print(f"Output zero points: {output_zero_points}")

    # Create validation windows
    X_val_windows = []
    y_val_targets = []
    for i in range(min(num_samples, len(X_val) - window_size)):
        X_val_windows.append(X_val[i:i+window_size])
        y_val_targets.append(y_val[i+window_size])

    X_val_windows = np.array(X_val_windows)
    y_val_targets = np.array(y_val_targets)

    print(f"Input range: min={X_val_windows.min():.4f}, max={X_val_windows.max():.4f}")

    # Quantize input
    input_quantized = np.round(X_val_windows / input_scale + input_zero_point).astype(input_details[0]['dtype'])
    print(f"Quantized input range: min={input_quantized.min()}, max={input_quantized.max()}")

    y_preds_dequant = [[] for _ in range(3)]

    for i in range(len(input_quantized)):
        input_data = np.expand_dims(input_quantized[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        for j in range(3):
            output_data = interpreter.get_tensor(output_details[j]['index'])
            output_float = (output_data.astype(np.float32) - output_zero_points[j]) * output_scales[j]
            y_preds_dequant[j].append(output_float[0])

    # Debug: show example raw output before rescaling
    print("\nSample dequantized model outputs (before rescaling):")
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        print(f"  {name}: {np.array(y_preds_dequant[j][:5]).round(3)}")

    print("\nValidation MAE (in °C):")
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        y_preds_rescaled = inverse_scale_temp(np.array(y_preds_dequant[j]))
        y_val_rescaled = inverse_scale_temp(y_val_targets[:, j])
        mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
        print(f"  {name}: {mae:.2f} °C")

# Run validation on the quantized model
validate_quantized_model(tflite_filename, val_df[features].values.astype(np.float32), 
                        val_df[targets].values.astype(np.float32), window_size)

# Final cleanup - remove intermediate files, keep only the best model
print("\nPerforming final cleanup...")

# Remove all model directories except the best one
all_models = glob.glob("*_qat_saved_model")
for model_dir in all_models:
    if model_dir != f"{best_model_name}_qat_saved_model":
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print(f"Removed intermediate model: {model_dir}")

# Remove all TFLite files except the best one
all_tflite = glob.glob("*.tflite")
for tflite_file in all_tflite:
    if tflite_file not in ["weather_model_2.tflite", tflite_filename]:
        if os.path.exists(tflite_file):
            os.remove(tflite_file)
            print(f"Removed intermediate TFLite: {tflite_file}")

# Remove all results files except the best one
all_results = glob.glob("results_*.json")
for results_file in all_results:
    if results_file != f"results_{best_model_name}.json":
        if os.path.exists(results_file):
            os.remove(results_file)
            print(f"Removed intermediate results: {results_file}")

print("Final cleanup complete.")
print(f"Kept best model: {best_model_name}")
print(f"Final TFLite model: weather_model_2.tflite")
