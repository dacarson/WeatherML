import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# !pip install -U tf_keras
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
import os
import json
import glob
import shutil

# Clean up from previous runs
for f in glob.glob("results_dense_wide_run*.json"):
    os.remove(f)
for f in glob.glob("weather_model_4_quant_dense_wide_run*.tflite"):
    os.remove(f)
if os.path.exists("weather_model_4_best.tflite"):
    os.remove("weather_model_4_best.tflite")
if os.path.exists("./checkpoints"):
    shutil.rmtree("./checkpoints")
os.makedirs("./checkpoints", exist_ok=True)

# Load preprocessed data
train_df = pd.read_csv("../train_data.csv")
val_df = pd.read_csv("../val_data.csv")

# Add lag features
train_df['temp_lag1'] = train_df['temperature'].shift(1)
train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
val_df['temp_lag1'] = val_df['temperature'].shift(1)
val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)

# Compute temperature_delta only
def rolling_slope(series, window):
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
        raw=True
    )

train_df['temperature_delta'] = rolling_slope(train_df['temperature'], window=15)
val_df['temperature_delta'] = rolling_slope(val_df['temperature'], window=15)

# Drop rows with NaNs
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

# Define base feature and target columns
base_features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
    'temperature_delta', 'temp_lag1', 'humidity_lag1'
]
targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

# Create MSB/LSB feature names
msb_lsb_features = []
for feature in base_features:
    msb_lsb_features.append(f"{feature}_LSB")
    msb_lsb_features.append(f"{feature}_MSB")

## Per-feature min/max scaling with ±5% padding
# Domain bounds for select features
domain_bounds = {
    "wind_gust": (0, None),
    "wind_avg": (0, None),
    "day_of_year": (1, 366),
    "time_of_day": (0, 24),
    "uv": (0, None),
    "relative_humidity": (0, 100),
    "humidity_lag1": (0, 100),
    "illuminance": (0, None),
    "solar_radiation": (0, None)
}

# Process base features first
X_train_base = train_df[base_features].copy()
X_val_base = val_df[base_features].copy()
input_scaler = {}
for feature in base_features:
    f_min = train_df[feature].min()
    f_max = train_df[feature].max()
    range_pad = 0.05 * (f_max - f_min)

    floor, ceiling = domain_bounds.get(feature, (None, None))
    f_min_adj = max(f_min - range_pad, floor) if floor is not None else f_min - range_pad
    f_max_adj = min(f_max + range_pad, ceiling) if ceiling is not None else f_max + range_pad

    input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
    X_train_base[feature] = (X_train_base[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    X_val_base[feature] = (X_val_base[feature] - f_min_adj) / (f_max_adj - f_min_adj)

# Convert to MSB/LSB as separate features
def convert_to_msb_lsb_separate(data):
    """Convert normalized data to MSB/LSB as separate features"""
    # Scale to [0, 65535] range for 16-bit representation
    data_scaled = (data * 65535).astype(np.uint16)
    
    # Extract MSB and LSB
    lsb = (data_scaled & 0xFF).astype(np.int8)
    msb = (data_scaled >> 8).astype(np.int8)
    
    # Convert back to float32 for model input
    lsb_float = lsb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    msb_float = msb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Combine MSB and LSB as separate features
    # Order: [feature1_LSB, feature1_MSB, feature2_LSB, feature2_MSB, ...]
    msb_lsb_data = np.empty((data.shape[0], data.shape[1] * 2))
    msb_lsb_data[:, 0::2] = lsb_float  # LSB features (even indices)
    msb_lsb_data[:, 1::2] = msb_float  # MSB features (odd indices)
    
    return msb_lsb_data

# Convert to MSB/LSB separate features
X_train = convert_to_msb_lsb_separate(X_train_base.values)
X_val = convert_to_msb_lsb_separate(X_val_base.values)

print(f"MSB/LSB training data shape: {X_train.shape}")
print(f"MSB/LSB validation data shape: {X_val.shape}")
print(f"Feature names: {msb_lsb_features}")
with open("input_scaler.json", "w") as f:
    json.dump(input_scaler, f, indent=2)

# Normalize target values
y_min = train_df[targets].min().min()
y_max = train_df[targets].max().max()
# Save original target range
target_range = (y_min, y_max)
train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

with open("target_scaler.json", "w") as f:
    json.dump({"min": y_min, "max": y_max, "range": target_range}, f)


def build_and_train_model(name):
    print(f"\n--- Running: {name} ---\n")

    y_train = train_df[targets].values
    y_val = val_df[targets].values

    # Input layer for MSB/LSB separate features (24 features total)
    input_layer = tf.keras.layers.Input(shape=(len(base_features) * 2,), name="input")
    input_normalized = tf.keras.layers.LayerNormalization()(input_layer)

    wide = tf.keras.layers.Dense(16)(input_normalized)
    deep = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_normalized)
    deep = tf.keras.layers.Dropout(0.2)(deep)

    res = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(deep)
    shortcut = tf.keras.layers.Dense(32)(deep)
    res = tf.keras.layers.Add()([shortcut, res])
    res = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(res)

    merged = tf.keras.layers.Concatenate()([wide, res])
    output_1 = tf.keras.layers.Dense(1, activation='linear', name='t1hr')(merged)
    output_2 = tf.keras.layers.Dense(1, activation='linear', name='t2hr')(merged)
    output_3 = tf.keras.layers.Dense(1, activation='linear', name='t3hr')(merged)
    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics={
            't1hr': 'mae',
            't2hr': 'mae',
            't3hr': 'mae'
        }
    )
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/model_{epoch:02d}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    # Use the MSB/LSB separate features directly (already converted)
    X_train_input = X_train
    X_val_input = X_val

    history = model.fit(
        X_train_input, [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
        validation_data=(X_val_input, [y_val[:, 0], y_val[:, 1], y_val[:, 2]]),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, checkpoint_cb]
    )

    # The early stopping callback with restore_best_weights=True already restored the best weights
    eval_results = model.evaluate(X_val_input, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)
    val_loss = eval_results[0]
    val_mae = np.mean(eval_results[1:])  # average MAE across 3 targets
    # Report MAEs in original units (°C)
    t1_mae_c = eval_results[1] * (y_max - y_min)
    t2_mae_c = eval_results[2] * (y_max - y_min)
    t3_mae_c = eval_results[3] * (y_max - y_min)
    print(f"\nValidation MAE (in °C):")
    print(f"  t+1hr: {t1_mae_c:.2f} °C")
    print(f"  t+2hr: {t2_mae_c:.2f} °C")
    print(f"  t+3hr: {t3_mae_c:.2f} °C")
    baseline_loss = val_loss
    feature_importance = {}
    for i, feature in enumerate(base_features):
        # For MSB/LSB separate features, we need to permute both LSB and MSB
        X_val_permuted = copy.deepcopy(X_val)
        lsb_idx = i * 2      # LSB feature index
        msb_idx = i * 2 + 1  # MSB feature index
        np.random.shuffle(X_val_permuted[:, lsb_idx])
        np.random.shuffle(X_val_permuted[:, msb_idx])
        permuted_loss = model.evaluate(X_val_permuted, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)[0]
        feature_importance[feature] = permuted_loss - baseline_loss

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance:.4f}")

    # The early stopping callback with restore_best_weights=True already restored the best weights

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # For Model 4's custom int16 approach, use full INT8 quantization
    # The separate MSB/LSB features should work well with standard quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_disable_per_channel = False
    converter._experimental_new_quantizer = True

    def representative_data_gen():
        # Sample evenly across the entire year of data for better representation
        # Convert MSB/LSB features back to INT8 for proper quantization
        step = max(1, len(X_train) // 1000)
        for i in range(0, len(X_train), step):
            if len(X_train) - i >= 1:  # Ensure we have at least 1 sample
                x = X_train[i]
                # Convert from [0,1] back to INT8 range [-128, 127]
                x_int8 = (x * 255 - 128).astype(np.int8)
                x_combined = np.expand_dims(x_int8, axis=0)
                yield [x_combined]

    converter.representative_dataset = representative_data_gen

    quantized_tflite_model = converter.convert()

    tflite_fname = f"weather_model_4_quant_{name}.tflite"
    with open(tflite_fname, "wb") as f:
        f.write(quantized_tflite_model)

    tflite_model_size_kb = os.path.getsize(tflite_fname) / 1024

    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\nFinal Metrics [{name}]:")
    print(f"  val_loss: {val_loss:.4f}")
    print(f"  val_mae: {val_mae:.4f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Quantized model size: {tflite_model_size_kb:.2f} KB")

    metrics = {
        "name": name,
        "val_loss": float(val_loss),
        "val_mae": float(val_mae),
        "best_epoch": int(best_epoch),
        "feature_importance": [(f, float(i)) for f, i in sorted_importance],
        "model_size_kb": float(tflite_model_size_kb)
    }

    with open(f"results_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


for run_id in range(1):
    run_name = f"dense_wide_run{run_id+1}"
    build_and_train_model(run_name)

results = []
for json_file in glob.glob("results_dense_wide_run*.json"):
    with open(json_file, "r") as f:
        metrics = json.load(f)
        results.append(metrics)

best = min(results, key=lambda x: x["val_loss"])
print(f"\nBest run: {best['name']} with val_loss: {best['val_loss']:.4f} and val_mae: {best['val_mae']:.4f}")

# Copy best model to canonical filename
import shutil
best_model_file = f"weather_model_4_quant_{best['name']}.tflite"
shutil.copy(best_model_file, "weather_model_4_best.tflite")
print(f"Best model copied to: weather_model_4_best.tflite")

# --- Validate quantized TFLite model on validation data ---
def validate_quantized_model(tflite_model_path, X_val, y_val, y_min, y_max, num_samples=500):
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    print(f"\nValidating TFLite model on {num_samples} samples...")

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Model 4 uses custom int16 quantization with dynamic range
    # Check if quantization parameters exist
    if 'quantization' in input_details[0] and input_details[0]['quantization'] is not None:
        input_scale, input_zero_point = input_details[0]['quantization']
        print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    else:
        print("No input quantization (dynamic range)")
        input_scale, input_zero_point = 1.0, 0.0
    
    if 'quantization' in output_details[0] and output_details[0]['quantization'] is not None:
        output_scales = [d['quantization'][0] for d in output_details]
        output_zero_points = [d['quantization'][1] for d in output_details]
        print(f"Output scales: {output_scales}")
        print(f"Output zero points: {output_zero_points}")
    else:
        print("No output quantization (dynamic range)")
        output_scales = [1.0] * len(output_details)
        output_zero_points = [0.0] * len(output_details)

    X_val_subset = X_val[:num_samples]
    y_val_subset = y_val[:num_samples]

    print(f"Input range: min={X_val_subset.min():.4f}, max={X_val_subset.max():.4f}")
    print(f"Input shape: {X_val_subset.shape}")
    print(f"Model expects input type: {input_details[0]['dtype']}")
    print(f"Model expects input shape: {input_details[0]['shape']}")

    y_preds_dequant = [[] for _ in range(3)]

    for i in range(len(X_val_subset)):
        input_data = np.expand_dims(X_val_subset[i], axis=0)
        # Convert to the expected input type
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32)
        elif input_details[0]['dtype'] == np.int8:
            # Convert from [0,1] range to INT8 range [-128, 127]
            input_data = (input_data * 255 - 128).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        for j in range(3):
            output_data = interpreter.get_tensor(output_details[j]['index'])
            if output_scales[j] != 1.0 or output_zero_points[j] != 0.0:
                # Quantized output - dequantize
                output_float = (output_data.astype(np.float32) - output_zero_points[j]) * output_scales[j]
            else:
                # Non-quantized output - use as is
                output_float = output_data.astype(np.float32)
            y_preds_dequant[j].append(output_float[0])

    # Debug: show example raw output before rescaling
    print("\nSample dequantized model outputs (before rescaling):")
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        print(f"  {name}: {np.array(y_preds_dequant[j][:5]).round(3)}")

    print("\nValidation MAE (in °C):")
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        y_preds_rescaled = 0.5 * (np.array(y_preds_dequant[j]) + 1) * (y_max - y_min) + y_min
        y_val_rescaled = 0.5 * (y_val_subset[:, j] + 1) * (y_max - y_min) + y_min
        mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
        print(f"  {name}: {mae:.2f} °C")

y_val_array = val_df[targets].values
validate_quantized_model("weather_model_4_best.tflite", X_val, y_val_array, y_min, y_max)
