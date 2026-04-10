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
for f in glob.glob("weather_model_4_hybrid_quant_dense_wide_run*.tflite"):
    os.remove(f)
if os.path.exists("weather_model_4_hybrid_best.tflite"):
    os.remove("weather_model_4_hybrid_best.tflite")
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

# Precision allocation based on analysis results
int32_features = ['illuminance']  # Needs 17 bits
int16_features = [f for f in base_features if f not in int32_features]  # All others need 9-14 bits

print(f"INT32 features (1): {int32_features}")
print(f"INT16 features (11): {int16_features}")

# Create hybrid feature names
hybrid_features = []
for feature in int16_features:
    hybrid_features.append(f"{feature}_LSB")
    hybrid_features.append(f"{feature}_MSB")
for feature in int32_features:
    hybrid_features.append(f"{feature}_B0")  # bits 0-7
    hybrid_features.append(f"{feature}_B1")  # bits 8-15
    hybrid_features.append(f"{feature}_B2")  # bits 16-23
    hybrid_features.append(f"{feature}_B3")  # bits 24-31

print(f"Total hybrid features: {len(hybrid_features)}")
print(f"Feature names: {hybrid_features}")

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

# Convert to hybrid precision features
def convert_to_hybrid_precision(data):
    """Convert normalized data to hybrid precision features"""
    hybrid_data = []
    
    for i, feature in enumerate(base_features):
        feature_data = data[:, i]
        
        if feature in int32_features:
            # Convert to INT32 (4 bytes)
            data_scaled = (feature_data * (2**32 - 1)).astype(np.uint32)
            b0 = (data_scaled & 0xFF).astype(np.uint8)           # bits 0-7
            b1 = ((data_scaled >> 8) & 0xFF).astype(np.uint8)    # bits 8-15
            b2 = ((data_scaled >> 16) & 0xFF).astype(np.uint8)   # bits 16-23
            b3 = ((data_scaled >> 24) & 0xFF).astype(np.uint8)   # bits 24-31
            
            # Normalize to [0, 1]
            b0_float = b0.astype(np.float32) / 255.0
            b1_float = b1.astype(np.float32) / 255.0
            b2_float = b2.astype(np.float32) / 255.0
            b3_float = b3.astype(np.float32) / 255.0
            
            hybrid_data.extend([b0_float, b1_float, b2_float, b3_float])
            
        else:  # int16_features
            # Convert to INT16 (2 bytes: MSB/LSB)
            data_scaled = (feature_data * 65535).astype(np.uint16)
            lsb = (data_scaled & 0xFF).astype(np.uint8)
            msb = (data_scaled >> 8).astype(np.uint8)
            
            # Normalize to [0, 1]
            lsb_float = lsb.astype(np.float32) / 255.0
            msb_float = msb.astype(np.float32) / 255.0
            
            hybrid_data.extend([lsb_float, msb_float])
    
    return np.column_stack(hybrid_data)

# Convert to hybrid precision features
X_train = convert_to_hybrid_precision(X_train_base.values)
X_val = convert_to_hybrid_precision(X_val_base.values)

print(f"Hybrid training data shape: {X_train.shape}")
print(f"Hybrid validation data shape: {X_val.shape}")

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

    # Input layer for hybrid precision features (26 features total)
    input_layer = tf.keras.layers.Input(shape=(len(hybrid_features),), name="input")
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

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./checkpoints/model_{name}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    # Use the hybrid precision features directly (already converted)
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
        # For hybrid precision features, we need to permute the appropriate bytes
        X_val_permuted = copy.deepcopy(X_val)
        
        if feature in int32_features:
            # Permute all 4 bytes for INT32 features
            b0_idx = len(int16_features) * 2 + int32_features.index(feature) * 4
            b1_idx = b0_idx + 1
            b2_idx = b0_idx + 2
            b3_idx = b0_idx + 3
            np.random.shuffle(X_val_permuted[:, b0_idx])
            np.random.shuffle(X_val_permuted[:, b1_idx])
            np.random.shuffle(X_val_permuted[:, b2_idx])
            np.random.shuffle(X_val_permuted[:, b3_idx])
        else:
            # Permute MSB/LSB for INT16 features
            lsb_idx = int16_features.index(feature) * 2
            msb_idx = lsb_idx + 1
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
    
    # For Model 4's hybrid precision approach, use full INT8 quantization
    # The separate byte features should work well with standard quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_disable_per_channel = False
    converter._experimental_new_quantizer = True

    def representative_data_gen():
        # Sample evenly across the entire year of data for better representation
        # Convert hybrid precision features back to INT8 for proper quantization
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

    tflite_fname = f"weather_model_4_hybrid_quant_{name}.tflite"
    with open(tflite_fname, "wb") as f:
        f.write(quantized_tflite_model)

    model_size_kb = len(quantized_tflite_model) / 1024
    print(f"Quantized model size: {model_size_kb:.2f} KB")

    return {
        "name": name,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "t1_mae_c": t1_mae_c,
        "t2_mae_c": t2_mae_c,
        "t3_mae_c": t3_mae_c,
        "model_size_kb": model_size_kb,
        "best_epoch": len(history.history['loss']),
        "feature_importance": feature_importance
    }


# Run training
results = []
for run in range(1, 2):  # Single run for now
    result = build_and_train_model(f"dense_wide_run{run}")
    results.append(result)

# Save results
with open("results_dense_wide_run1.json", "w") as f:
    json.dump(results[0], f, indent=2)

# Find best model
best = min(results, key=lambda x: x["val_loss"])
print(f"\nBest run: {best['name']} with val_loss: {best['val_loss']:.4f} and val_mae: {best['val_mae']:.4f}")

# Copy best model to canonical filename
import shutil
best_model_file = f"weather_model_4_hybrid_quant_{best['name']}.tflite"
shutil.copy(best_model_file, "weather_model_4_hybrid_best.tflite")
print(f"Best model copied to: weather_model_4_hybrid_best.tflite")

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

    # Model 4 uses hybrid precision quantization with INT8
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
                dequantized = (output_data.astype(np.float32) - output_zero_points[j]) * output_scales[j]
            else:
                dequantized = output_data.astype(np.float32)
            y_preds_dequant[j].append(dequantized[0][0])

    # Convert to numpy arrays
    y_preds_dequant = [np.array(preds) for preds in y_preds_dequant]
    
    # Rescale predictions back to original temperature units
    y_preds_rescaled = []
    for preds in y_preds_dequant:
        # Convert from [-1, 1] back to original range
        rescaled = (preds + 1) / 2 * (y_max - y_min) + y_min
        y_preds_rescaled.append(rescaled)
    
    # Calculate MAE for each target
    mae_t1 = mean_absolute_error(y_val_subset[:, 0], y_preds_rescaled[0])
    mae_t2 = mean_absolute_error(y_val_subset[:, 1], y_preds_rescaled[1])
    mae_t3 = mean_absolute_error(y_val_subset[:, 2], y_preds_rescaled[2])
    
    print(f"Sample dequantized model outputs (before rescaling):")
    print(f"t+1hr: {y_preds_dequant[0][:5].reshape(-1, 1)}")
    print(f"t+2hr: {y_preds_dequant[1][:5].reshape(-1, 1)}")
    print(f"t+3hr: {y_preds_dequant[2][:5].reshape(-1, 1)}")
    
    print(f"Validation MAE (in °C):")
    print(f"t+1hr: {mae_t1:.2f} °C")
    print(f"t+2hr: {mae_t2:.2f} °C")
    print(f"t+3hr: {mae_t3:.2f} °C")

# Validate the quantized model
y_val_array = val_df[targets].values
validate_quantized_model("weather_model_4_hybrid_best.tflite", X_val, y_val_array, y_min, y_max)
