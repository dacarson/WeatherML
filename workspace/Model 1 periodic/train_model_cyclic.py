# Configuration
NUM_RUNS = 10  # Number of training runs to perform

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
import os
import json
import glob

# Load preprocessed data
train_df = pd.read_csv("../train_data.csv")
val_df = pd.read_csv("../val_data.csv")

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

# Add cyclic features for time_of_day and day_of_year
# Convert time_of_day to radians (0-24 hours -> 0-2π)
train_df['time_of_day_sin'] = np.sin(2 * np.pi * train_df['time_of_day'] / 24)
train_df['time_of_day_cos'] = np.cos(2 * np.pi * train_df['time_of_day'] / 24)
val_df['time_of_day_sin'] = np.sin(2 * np.pi * val_df['time_of_day'] / 24)
val_df['time_of_day_cos'] = np.cos(2 * np.pi * val_df['time_of_day'] / 24)

# Convert day_of_year to radians (1-366 days -> 0-2π)
train_df['day_of_year_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 366)
train_df['day_of_year_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 366)
val_df['day_of_year_sin'] = np.sin(2 * np.pi * val_df['day_of_year'] / 366)
val_df['day_of_year_cos'] = np.cos(2 * np.pi * val_df['day_of_year'] / 366)

# Drop rows with NaNs
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

# Define feature and target columns - replace time_of_day and day_of_year with their cyclic components
features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 
    'time_of_day_sin', 'time_of_day_cos', 'day_of_year_sin', 'day_of_year_cos',
    'temperature_delta', 'temp_lag1', 'humidity_lag1'
]
targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

## Per-feature min/max scaling with ±5% padding
# Domain bounds for select features - updated for cyclic features
domain_bounds = {
    "wind_gust": (0, None),
    "wind_avg": (0, None),
    "time_of_day_sin": (-1, 1),
    "time_of_day_cos": (-1, 1),
    "day_of_year_sin": (-1, 1),
    "day_of_year_cos": (-1, 1),
    "uv": (0, None),
    "relative_humidity": (0, 100),
    "humidity_lag1": (0, 100),
    "illuminance": (0, None),
    "solar_radiation": (0, None),
    "temp_lag1": (-10, 55),
    "station_pressure": (None, None),  # Allow dynamic bounds for expanded pressure range
    "temperature_delta": (None, None)  # Allow dynamic bounds for expanded delta range
}

X_train = train_df[features].copy()
X_val = val_df[features].copy()
input_scaler = {}
for feature in features:
    f_min = train_df[feature].min()
    f_max = train_df[feature].max()
    range_pad = 0.05 * (f_max - f_min)

    floor, ceiling = domain_bounds.get(feature, (None, None))
    f_min_adj = floor if floor is not None else f_min - range_pad
    f_max_adj = ceiling if ceiling is not None else f_max + range_pad

    input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
    X_train[feature] = (X_train[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    X_val[feature] = (X_val[feature] - f_min_adj) / (f_max_adj - f_min_adj)
X_train = X_train.values
X_val = X_val.values
with open("input_scaler_cyclic.json", "w") as f:
    json.dump(input_scaler, f, indent=2)

# Normalize target values - Updated to accommodate expanded dataset
# Calculate bounds from actual data with padding for better generalization
y_min = min(train_df[targets].min().min(), train_df['temperature'].min()) - 5
y_max = max(train_df[targets].max().max(), train_df['temperature'].max()) + 15
# Save original target range
target_range = (y_min, y_max)
train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

with open("target_scaler_cyclic.json", "w") as f:
    json.dump({"min": y_min, "max": y_max, "range": target_range}, f)

print(f"Updated scaling bounds for expanded dataset (cyclic model):")
print(f"Temperature range: {y_min:.2f}°C to {y_max:.2f}°C")


def build_and_train_model(name):
    print(f"\n--- Running: {name} ---\n")

    y_train = train_df[targets].values
    y_val = val_df[targets].values

    input_layer = tf.keras.layers.Input(shape=(len(features),), name="input")

    wide = tf.keras.layers.Dense(16)(input_layer)
    deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    deep = tf.keras.layers.Dropout(0.3)(deep)

    res = tf.keras.layers.Dense(64, activation='relu')(deep)
    shortcut = tf.keras.layers.Dense(64)(deep)
    res = tf.keras.layers.Add()([shortcut, res])
    res = tf.keras.layers.Dense(32, activation='relu')(res)

    merged = tf.keras.layers.Concatenate()([wide, res])
    output_1 = tf.keras.layers.Dense(1, activation='linear', name='t1hr')(merged)
    output_2 = tf.keras.layers.Dense(1, activation='linear', name='t2hr')(merged)
    output_3 = tf.keras.layers.Dense(1, activation='linear', name='t3hr')(merged)
    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)  # Reverted to original for SF-only dataset
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

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Reverted to original for SF-only dataset
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/model_{epoch:02d}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    history = model.fit(
        X_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
        validation_data=(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]]),
        epochs=100,  # Reverted to original for SF-only dataset
        batch_size=32,  # Reverted to original for SF-only dataset
        callbacks=[early_stopping, checkpoint_cb]
    )

    # The early stopping callback with restore_best_weights=True already restored the best weights

    eval_results = model.evaluate(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)
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
    for i, feature in enumerate(features):
        X_val_permuted = copy.deepcopy(X_val)
        np.random.shuffle(X_val_permuted[:, i])
        permuted_loss = model.evaluate(X_val_permuted, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)[0]
        feature_importance[feature] = permuted_loss - baseline_loss

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance:.4f}")

    # The early stopping callback with restore_best_weights=True already restored the best weights

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        # Sample evenly across the entire year of data for better representation
        # Take every 500th sample to get ~1000 samples spread across the year
        step = max(1, len(X_train) // 1000)
        for i in range(0, len(X_train), step):
            if len(X_train) - i >= 1:  # Ensure we have at least 1 sample
                yield [X_train[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_disable_per_channel = False
    converter._experimental_new_quantizer = True  # Use MLIR quantizer (recommended)

    quantized_tflite_model = converter.convert()

    tflite_fname = f"weather_model_1_cyclic_quant_{name}.tflite"
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

    with open(f"results_cyclic_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


for run_id in range(NUM_RUNS):
    run_name = f"dense_wide_run{run_id+1}"
    build_and_train_model(run_name)

results = []
# Only collect results from the current run session
for run_id in range(NUM_RUNS):
    json_file = f"results_cyclic_dense_wide_run{run_id+1}.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            metrics = json.load(f)
            results.append(metrics)

if results:
    best = min(results, key=lambda x: x["val_loss"])
    print(f"\nBest run: {best['name']} with val_loss: {best['val_loss']:.4f} and val_mae: {best['val_mae']:.4f}")
else:
    print("\nNo results found!")
    exit(1)

# Copy best model to canonical filename
import shutil
best_model_file = f"weather_model_1_cyclic_quant_{best['name']}.tflite"
shutil.copy(best_model_file, "weather_model_1_cyclic_best.tflite")
print(f"Best model copied to: weather_model_1_cyclic_best.tflite")

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

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scales = [d['quantization'][0] for d in output_details]
    output_zero_points = [d['quantization'][1] for d in output_details]

    print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output scales: {output_scales}")
    print(f"Output zero points: {output_zero_points}")

    X_val_subset = X_val[:num_samples]
    y_val_subset = y_val[:num_samples]

    print(f"Input range: min={X_val_subset.min():.4f}, max={X_val_subset.max():.4f}")

    input_quantized = np.round(X_val_subset / input_scale + input_zero_point).astype(input_details[0]['dtype'])
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
        y_preds_rescaled = 0.5 * (np.array(y_preds_dequant[j]) + 1) * (y_max - y_min) + y_min
        y_val_rescaled = 0.5 * (y_val_subset[:, j] + 1) * (y_max - y_min) + y_min
        mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
        print(f"  {name}: {mae:.2f} °C")

y_val_array = val_df[targets].values
validate_quantized_model("weather_model_1_cyclic_best.tflite", X_val, y_val_array, y_min, y_max)
