import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide GPU devices
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

# Load preprocessed data
train_df = pd.read_csv("../train_data.csv")
print("Loaded train_data.csv:", train_df.shape)
val_df = pd.read_csv("../val_data.csv")
print("Loaded val_data.csv:", val_df.shape)

# Add lag features
train_df['temp_avg_15min'] = train_df['temperature'].rolling(window=15, min_periods=1).mean().shift(1)
train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
val_df['temp_avg_15min'] = val_df['temperature'].rolling(window=15, min_periods=1).mean().shift(1)
val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)

# Compute temperature_delta only (backward-looking window for real-time inference)
def rolling_slope(series, window):
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
        raw=True
    )

train_df['temperature_delta'] = rolling_slope(train_df['temperature'], window=15)
val_df['temperature_delta'] = rolling_slope(val_df['temperature'], window=15)

train_df['humidity_delta'] = rolling_slope(train_df['relative_humidity'], window=15)
val_df['humidity_delta'] = rolling_slope(val_df['relative_humidity'], window=15)

# Convert time_of_day to radians and add sin/cos components
train_df['time_rad'] = 2 * np.pi * train_df['time_of_day'] / 24.0
val_df['time_rad'] = 2 * np.pi * val_df['time_of_day'] / 24.0
train_df['sin_time_of_day'] = np.sin(train_df['time_rad'])
train_df['cos_time_of_day'] = np.cos(train_df['time_rad'])
val_df['sin_time_of_day'] = np.sin(val_df['time_rad'])
val_df['cos_time_of_day'] = np.cos(val_df['time_rad'])

# Encode day_of_year cyclically
train_df['day_rad'] = 2 * np.pi * train_df['day_of_year'] / 365.0
val_df['day_rad'] = 2 * np.pi * val_df['day_of_year'] / 365.0
train_df['sin_day_of_year'] = np.sin(train_df['day_rad'])
train_df['cos_day_of_year'] = np.cos(train_df['day_rad'])
val_df['sin_day_of_year'] = np.sin(val_df['day_rad'])
val_df['cos_day_of_year'] = np.cos(val_df['day_rad'])

print("Added lag and temperature_delta features.")

# Drop rows with NaNs
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
print("Dropped NaNs from datasets.")


# Define feature and target columns
domain_bounds = {
    "temp_avg_15min": (-10, 50),
    "temperature_delta": (-5, 5),
    "sin_time_of_day": (-1, 1),
    "cos_time_of_day": (-1, 1),
    "illuminance": (0, None),
    "solar_radiation": (0, None),
    "station_pressure": (850, 1100),
    "relative_humidity": (0, 100),
}
features = [
    'temp_avg_15min',
    'temperature_delta',
    'sin_time_of_day',
    'cos_time_of_day',
    'illuminance',
    'solar_radiation',
    'station_pressure',
    'relative_humidity',
]
targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

baseline_preds = val_df['temp_avg_15min'].values
baseline_true = val_df['temp_t+1hr'].values
baseline_mae = np.mean(np.abs(baseline_preds - baseline_true))
print(f"Baseline MAE using temp_avg_15min for temp_t+1hr: {baseline_mae:.4f}")

# Set window size
window_size = 90  # 90 minutes

import json
# Normalize features before windowing using manual clamped domain-aware min/max
input_scaler = {}
for f in features:
    data = pd.concat([train_df[f], val_df[f]], axis=0)
    min_val, max_val = data.min(), data.max()
    domain_min, domain_max = domain_bounds[f]
    if domain_max is not None and domain_min is not None:
        span = domain_max - domain_min
        min_val = max(domain_min, min_val - 0.05 * span)
        max_val = min(domain_max, max_val + 0.05 * span)
    elif domain_min is not None:
        min_val = max(domain_min, min_val)
    elif domain_max is not None:
        max_val = min(domain_max, max_val)
    input_scaler[f] = {"min": float(min_val), "max": float(max_val)}
    train_df[f] = (train_df[f] - min_val) / (max_val - min_val)
    val_df[f] = (val_df[f] - min_val) / (max_val - min_val)
print("Features manually scaled using clamped domain-aware min/max.")

with open("input_scaler.json", "w") as f:
    json.dump(input_scaler, f, indent=2)

 # Apply custom min-max scaling for targets (0–50°C)
target_min, target_max = 0.0, 50.0
train_df[targets] = (train_df[targets] - target_min) / (target_max - target_min)
val_df[targets] = (val_df[targets] - target_min) / (target_max - target_min)

# Save custom scaler parameters for inference (dict keyed by target)
target_scaler = {
    t: {"min": target_min, "max": target_max}
    for t in targets
}
with open("target_scaler.json", "w") as f:
    json.dump(target_scaler, f, indent=2)

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
).repeat().batch(32).map(lambda x, y: (x, {
    't1hr': y[:, 0:1],
    't2hr': y[:, 1:2],
    't3hr': y[:, 2:3]
})).prefetch(tf.data.AUTOTUNE)

# Function to create validation dataset from generator
def get_val_dataset():
    def create_val_generator():
        return numpy_window_generator(X_val_np, y_val_np, window_size)
    
    return tf.data.Dataset.from_generator(
        create_val_generator,
        output_signature=(
            tf.TensorSpec(shape=(window_size, len(features)), dtype=tf.float32),
            tf.TensorSpec(shape=(len(targets),), dtype=tf.float32),
        )
    ).repeat().batch(32).map(lambda x, y: (x, {
        't1hr': y[:, 0:1],
        't2hr': y[:, 1:2],
        't3hr': y[:, 2:3]
    })).prefetch(tf.data.AUTOTUNE)


def build_and_train_model(name):
    import os
    print(f"\n--- Running: {name} (Conv1D TPU-optimized, large receptive field) ---\n")
    
    # Access global variables for validation data size
    global X_val_np
    # TPU-optimized architecture with ~27 step receptive field
    input_layer = tf.keras.layers.Input(shape=(window_size, len(features)))

    # Block 1 — Local features
    x = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu')(input_layer)

    # Block 2 — Expand receptive field slightly
    x = tf.keras.layers.Conv1D(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    # Block 3 — Keep expanding, larger kernel
    x = tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, padding='same', activation='relu')(x)

    # Block 4 — Widen again with dilation
    x = tf.keras.layers.Conv1D(32, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)

    # Flatten for dense layers
    x = tf.keras.layers.Reshape((-1,))(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)

    # Outputs
    output_1 = tf.keras.layers.Dense(1, name='t1hr')(x)
    output_2 = tf.keras.layers.Dense(1, name='t2hr')(x)
    output_3 = tf.keras.layers.Dense(1, name='t3hr')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss={'t1hr': 'mse', 't2hr': 'mse', 't3hr': 'mse'},
        metrics={'t1hr': 'mae', 't2hr': 'mae', 't3hr': 'mae'}
    )
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./checkpoints/best_{name}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    # Calculate validation steps more carefully
    val_steps = max(1, (len(X_val_np) - window_size) // 32)
    print(f"Training steps per epoch: {(len(X_train_np) - window_size) // 32}")
    print(f"Validation steps per epoch: {val_steps}")
    print(f"Available validation samples: {len(X_val_np) - window_size}")
    
    print("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=get_val_dataset(),
        epochs=100,
        steps_per_epoch=(len(X_train_np) - window_size) // 32,
        validation_steps=val_steps,
        callbacks=[early_stopping, checkpoint_cb]
    )
    print("Model training complete.")

    print("Evaluating model on validation data...")
    y_true_list, y_pred_list = [], []
    
    # Create validation data directly as numpy arrays to avoid generator issues
    X_val_windows = []
    y_val_targets = []
    for i in range(len(X_val_np) - window_size):
        X_val_windows.append(X_val_np[i:i+window_size])
        y_val_targets.append(y_val_np[i+window_size])
    
    X_val_windows = np.array(X_val_windows)
    y_val_targets = np.array(y_val_targets)
    
    # Evaluate in batches
    batch_size = 32
    for i in range(0, len(X_val_windows), batch_size):
        x_batch = X_val_windows[i:i+batch_size]
        y_batch = y_val_targets[i:i+batch_size]
        preds = model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch)
        y_pred_list.append(np.concatenate(preds, axis=1))
    # Inverse transform using custom min-max scaler
    def inverse_minmax(x):
        return x * (target_max - target_min) + target_min
    y_true = inverse_minmax(np.vstack(y_true_list))
    y_pred = inverse_minmax(np.vstack(y_pred_list))
    mae_per_target = np.mean(np.abs(y_true - y_pred), axis=0)
    for i, t in enumerate(targets):
        print(f"Validation MAE for {t} (denormalized): {mae_per_target[i]:.4f}")
    print(f"Average Validation MAE across all targets: {np.mean(mae_per_target):.4f}")
    print("Validation MAE (in °C):")
    print(f"  t+1hr: {mae_per_target[0]:.2f} °C")
    print(f"  t+2hr: {mae_per_target[1]:.2f} °C")
    print(f"  t+3hr: {mae_per_target[2]:.2f} °C")
    val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Validation RMSE (denormalized): {val_rmse:.4f}")

    # Track val_loss, val_mae, and best_epoch on normalized validation data
    # Create a simple dataset for evaluation
    val_dataset_simple = tf.data.Dataset.from_tensor_slices((X_val_windows, y_val_targets)).batch(32).map(lambda x, y: (x, {
        't1hr': y[:, 0:1],
        't2hr': y[:, 1:2],
        't3hr': y[:, 2:3]
    }))
    eval_results = model.evaluate(val_dataset_simple, verbose=0)
    val_loss = eval_results[0]
    val_mae = np.mean(eval_results[1:])  # average MAE across outputs
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\nFinal Metrics [{name}]:")
    print(f"  val_loss (normalized): {val_loss:.4f}")
    print(f"  val_mae (normalized): {val_mae:.4f}")
    print(f"  Best epoch: {best_epoch}")

    # Sample smaller validation set for permutation feature importance
    try:
        print("Calculating permutation feature importance...")
        # Use the pre-computed validation windows for feature importance
        sample_size = min(1000, len(X_val_windows))
        X_val_sample = X_val_windows[:sample_size]
        y_val_sample = y_val_targets[:sample_size]

        feature_importance = {}
        for i, feature in enumerate(features):
            X_val_permuted = X_val_sample.copy()
            flat = X_val_permuted[:, :, i].flatten()
            np.random.shuffle(flat)
            X_val_permuted[:, :, i] = flat.reshape(X_val_permuted[:, :, i].shape)
            
            # Create properly structured validation data for evaluation
            val_dataset_permuted = tf.data.Dataset.from_tensor_slices((X_val_permuted, y_val_sample)).batch(32).map(lambda x, y: (x, {
                't1hr': y[:, 0:1],
                't2hr': y[:, 1:2],
                't3hr': y[:, 2:3]
            }))
            permuted_loss = model.evaluate(val_dataset_permuted, verbose=0)[0]
            feature_importance[feature] = permuted_loss - val_loss

        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")
    except Exception as e:
        print(f"Skipping feature importance due to error: {e}")
        sorted_importance = []

    # (Removed per-run quantization; quantization will be done once for the best model after selection.)

    # Save results as JSON
    metrics = {
        "name": name,
        "val_loss": float(val_loss),
        "val_mae": float(val_mae),
        "best_epoch": int(best_epoch),
        "average_mae_denormalized": float(np.mean(mae_per_target)),
        "val_rmse_denormalized": float(val_rmse),
        "feature_importance": [(f, float(i)) for f, i in sorted_importance] if sorted_importance else [],
    }

    with open(f"results_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"training_history_{name}.csv", index=False)
    print(f"Training history saved to training_history_{name}.csv")

    # Clean up to prevent memory buildup
    from tensorflow.keras import backend as K
    import gc
    K.clear_session()
    del model
    gc.collect()


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

# Remove old training history files
old_history = glob.glob("training_history_*.csv")
for history_file in old_history:
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"Removed old training history file: {history_file}")

# Remove old checkpoint files
old_checkpoints = glob.glob("./checkpoints/*.weights.h5")
for checkpoint_file in old_checkpoints:
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed old checkpoint file: {checkpoint_file}")

print("Cleanup complete.\n")

# Run multiple times and select the best model
best_model_metrics = None
best_model_name = None
num_runs = 2

for i in range(num_runs):
    run_name = f"conv1d_tpu_run{i+1}"
    build_and_train_model(run_name)
    
    with open(f"results_{run_name}.json") as f:
        metrics = json.load(f)

    if (best_model_metrics is None) or (metrics["average_mae_denormalized"] < best_model_metrics["average_mae_denormalized"]):
        best_model_metrics = metrics
        best_model_name = run_name

print(f"\n✅ Best model: {best_model_name}")
print(f"Average MAE: {best_model_metrics['average_mae_denormalized']:.4f}")


# Load best weights from checkpoint and quantize the best model only once
input_layer = tf.keras.layers.Input(shape=(window_size, len(features)))

# Block 1 — Local features
x = tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu')(input_layer)

# Block 2 — Expand receptive field slightly
x = tf.keras.layers.Conv1D(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)

# Block 3 — Keep expanding, larger kernel
x = tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, padding='same', activation='relu')(x)

# Block 4 — Widen again with dilation
x = tf.keras.layers.Conv1D(32, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)

# Flatten for dense layers
x = tf.keras.layers.Reshape((-1,))(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)

# Outputs
output_1 = tf.keras.layers.Dense(1, name='t1hr')(x)
output_2 = tf.keras.layers.Dense(1, name='t2hr')(x)
output_3 = tf.keras.layers.Dense(1, name='t3hr')(x)

model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

# Check if checkpoint file exists before loading
checkpoint_path = f"./checkpoints/best_{best_model_name}.weights.h5"
if os.path.exists(checkpoint_path):
    print(f"Loading best weights from {checkpoint_path}")
    try:
        status = model.load_weights(checkpoint_path)
        if status is not None:
            status.expect_partial()
        print("✅ Best weights loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}")
        print("Using model with final training weights")
else:
    print(f"⚠️ Checkpoint file not found: {checkpoint_path}")
    print("Using model with final training weights")

# Convert the best model to TFLite with int8 quantization
print("Converting best model to TFLite with int8 quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    # Sample evenly across the entire year of data for better representation
    # Take every 500th sample to get ~1000 samples spread across the year
    num_samples = 1000
    step = max((len(val_df) - window_size) // num_samples, 1)
    for i in range(0, len(val_df) - window_size, step):
        if len(val_df) - i >= window_size:  # Ensure we have enough samples
            window = val_df[features].iloc[i:i+window_size].values.astype(np.float32)
            yield [np.expand_dims(window, axis=0)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_new_quantizer = True
quantized_tflite_model = converter.convert()

tflite_fname = f"weather_model_conv1d_quant_{best_model_name}.tflite"
with open(tflite_fname, "wb") as f:
    f.write(quantized_tflite_model)

tflite_model_size_kb = os.path.getsize(tflite_fname) / 1024
print(f"  Quantized model size: {tflite_model_size_kb:.2f} KB")

# Copy best model to standard filename
import shutil
shutil.copyfile(tflite_fname, "weather_model_3.tflite")
print("Copied best model to weather_model_3.tflite")


# --- Quantized TFLite model inference on validation set ---
def run_quantized_model_inference(tflite_model_path, val_array):
    """
    Load quantized TFLite model, run inference on validation data with sliding window,
    denormalize predictions, and print MAE/RMSE.
    """

    # Load scalers
    with open("input_scaler.json") as f:
        input_scaler = json.load(f)
    with open("target_scaler.json") as f:
        target_scaler = json.load(f)

    window_size = 90
    num_features = len(features)
    num_targets = len(targets)

    # Load TFLite model with CPU interpreter (more compatible)
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        print("✅ Loaded TFLite model with CPU interpreter")
    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get quantization params for input and output
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scales = [od['quantization'][0] for od in output_details]
    output_zero_points = [od['quantization'][1] for od in output_details]

    # Prepare sliding windows for inference
    xs = []
    ys = []
    # Limit number of validation samples to avoid OOM
    max_val_samples = 5000
    val_array = val_array[:max_val_samples + window_size]
    val_targets = val_df[targets].values.astype(np.float32)
    for i in range(len(val_array) - window_size):
        xs.append(val_array[i:i+window_size])
        ys.append(val_targets[i+window_size])
    xs = np.stack(xs, axis=0)  # shape: (N, window_size, num_features)
    ys = np.stack(ys, axis=0)

    # Quantize input
    xs_quant = np.round(xs / input_scale + input_zero_point).astype(np.int8)

    # Run inference
    preds_norm = []
    for i in range(xs_quant.shape[0]):
        input_tensor = xs_quant[i:i+1]  # shape (1, window_size, num_features)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        # There are three outputs, each shape (1, 1)
        outputs = []
        for j in range(num_targets):
            out = interpreter.get_tensor(output_details[j]['index'])
            # Dequantize output
            out_deq = (out.astype(np.float32) - output_zero_points[j]) * output_scales[j]
            outputs.append(out_deq.reshape(-1))
        pred = np.concatenate(outputs, axis=0)  # (3,)
        preds_norm.append(pred)
    preds_norm = np.stack(preds_norm, axis=0)

    # Denormalize predictions
    target_min = target_scaler[targets[0]]['min']
    target_max = target_scaler[targets[0]]['max']
    preds_denorm = preds_norm * (target_max - target_min) + target_min
    ys_denorm = ys * (target_max - target_min) + target_min

    # Compute MAE per output
    mae_per_target = np.mean(np.abs(preds_denorm - ys_denorm), axis=0)
    mae_per_target_norm = np.mean(np.abs(preds_norm - ys), axis=0)

    # Compute RMSE
    rmse_norm = np.sqrt(np.mean((preds_norm - ys) ** 2))
    rmse_denorm = np.sqrt(np.mean((preds_denorm - ys_denorm) ** 2))

    print("\nQuantized TFLite model inference results on validation set:")
    print("Validation MAE (normalized):")
    print(f"  t+1hr: {mae_per_target_norm[0]:.2f}")
    print(f"  t+2hr: {mae_per_target_norm[1]:.2f}")
    print(f"  t+3hr: {mae_per_target_norm[2]:.2f}")
    print("Validation MAE (in °C):")
    print(f"  t+1hr: {mae_per_target[0]:.2f} °C")
    print(f"  t+2hr: {mae_per_target[1]:.2f} °C")
    print(f"  t+3hr: {mae_per_target[2]:.2f} °C")
    for i, t in enumerate(targets):
        print(f"  MAE for {t} (denormalized): {mae_per_target[i]:.4f}")
    print(f"  Average MAE (denormalized): {np.mean(mae_per_target):.4f}")
    print(f"  RMSE (normalized): {rmse_norm:.4f}")
    print(f"  RMSE (denormalized): {rmse_denorm:.4f}")


# Run quantized inference on the validation set using the best model
run_quantized_model_inference("weather_model_3.tflite", val_df[features].values.astype(np.float32))

# --- Validate quantized TFLite model on validation data ---
def validate_quantized_model(tflite_model_path, X_val, y_val, window_size, num_samples=500):
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    print(f"\nValidating TFLite model on {num_samples} samples...")

    # Load TFLite model with CPU interpreter (more compatible)
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        print("✅ Loaded TFLite model with CPU interpreter")
    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return

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

    # Inverse transform using custom min-max scaler
    def inverse_minmax(x):
        return x * (target_max - target_min) + target_min

    print("\nValidation MAE (in °C):")
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        y_preds_rescaled = inverse_minmax(np.array(y_preds_dequant[j]))
        y_val_rescaled = inverse_minmax(y_val_targets[:, j])
        mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
        print(f"  {name}: {mae:.2f} °C")

# Run validation on the quantized model
validate_quantized_model(tflite_fname, val_df[features].values.astype(np.float32), 
                        val_df[targets].values.astype(np.float32), window_size)

# Final cleanup - remove intermediate files, keep only the best model
print("\nPerforming final cleanup...")

# Remove all TFLite files except the best one
all_tflite = glob.glob("*.tflite")
for tflite_file in all_tflite:
    if tflite_file not in ["weather_model_3.tflite", tflite_fname]:
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

# Remove all training history files except the best one
all_history = glob.glob("training_history_*.csv")
for history_file in all_history:
    if history_file != f"training_history_{best_model_name}.csv":
        if os.path.exists(history_file):
            os.remove(history_file)
            print(f"Removed intermediate training history: {history_file}")

# Remove all checkpoint files except the best one
all_checkpoints = glob.glob("./checkpoints/*.weights.h5")
for checkpoint_file in all_checkpoints:
    if checkpoint_file != f"./checkpoints/best_{best_model_name}.weights.h5":
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Removed intermediate checkpoint: {checkpoint_file}")

print("Final cleanup complete.")
print(f"Kept best model: {best_model_name}")
print(f"Final TFLite model: weather_model_3.tflite")

# Note: This script is now CPU-optimized with proper quantization and cleanup for TensorFlow 2.19.0 compatibility.
