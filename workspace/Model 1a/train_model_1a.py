import tensorflow as tf
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

# Define feature and target columns
features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
    'temperature_delta', 'temp_lag1', 'humidity_lag1'
]
targets = ['temp_t+1hr']

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

X_train = train_df[features].copy()
X_val = val_df[features].copy()
input_scaler = {}
for feature in features:
    f_min = train_df[feature].min()
    f_max = train_df[feature].max()
    range_pad = 0.05 * (f_max - f_min)

    floor, ceiling = domain_bounds.get(feature, (None, None))
    f_min_adj = max(f_min - range_pad, floor) if floor is not None else f_min - range_pad
    f_max_adj = min(f_max + range_pad, ceiling) if ceiling is not None else f_max + range_pad

    input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
    X_train[feature] = (X_train[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    X_val[feature] = (X_val[feature] - f_min_adj) / (f_max_adj - f_min_adj)
X_train = X_train.values
X_val = X_val.values
with open("input_scaler.json", "w") as f:
    json.dump(input_scaler, f, indent=2)

# Normalize target values
y_min = train_df[targets].min().min()
y_max = train_df[targets].max().max()
# Save original target range
target_range = (y_min, y_max)
train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

print("Training target range:", train_df[targets].min().min(), train_df[targets].max().max())

with open("target_scaler.json", "w") as f:
    json.dump({"min": y_min, "max": y_max, "range": target_range}, f)


def build_and_train_model(name):
    print(f"\n--- Running: {name} ---\n")

    y_train = train_df[targets].values
    y_val = val_df[targets].values

    input_layer = tf.keras.layers.Input(shape=(len(features),), name="input")

    wide = tf.keras.layers.Dense(16, name='wide')(input_layer)
    deep = tf.keras.layers.Dense(128, activation='relu', name='deep')(input_layer)
    deep = tf.keras.layers.Dropout(0.3, name='dropout')(deep)

    res = tf.keras.layers.Dense(64, activation='relu', name='res1')(deep)
    shortcut = tf.keras.layers.Dense(64, name='shortcut')(deep)
    res = tf.keras.layers.Add(name='res_add')([shortcut, res])
    res = tf.keras.layers.Dense(32, activation='relu', name='res2')(res)

    merged = tf.keras.layers.Concatenate()([wide, res])
    output = tf.keras.layers.Dense(1, activation='linear', name='t1hr')(merged)
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    model.summary()

    # Create a debug model to inspect internal activations
    debug_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer('wide').output,
            model.get_layer('deep').output,
            model.get_layer('res1').output,
            model.get_layer('res_add').output,
            model.get_layer('res2').output,
            model.output
        ]
    )

    # Evaluate a small batch to check activation ranges
    print("\nInspecting intermediate activations on validation samples:")
    sample_batch = tf.convert_to_tensor(X_val[:10], dtype=tf.float32)
    activations = debug_model(sample_batch)

    for i, act in enumerate(activations):
        arr = act.numpy()
        print(f"  Layer {i}: min={arr.min():.4f}, max={arr.max():.4f}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/model_{epoch:02d}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, checkpoint_cb]
    )


    eval_results = model.evaluate(X_val, y_val, verbose=0)
    val_loss = eval_results[0]
    t1_mae_c = eval_results[1] * (y_max - y_min)
    print(f"\nValidation MAE (in °C):")
    print(f"  t+1hr: {t1_mae_c:.2f} °C")
    val_mae = eval_results[1]
    baseline_loss = val_loss
    feature_importance = {}
    for i, feature in enumerate(features):
        X_val_permuted = copy.deepcopy(X_val)
        np.random.shuffle(X_val_permuted[:, i])
        permuted_eval = model.evaluate(X_val_permuted, y_val, verbose=0)
        feature_importance[feature] = permuted_eval[0] - baseline_loss

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance:.4f}")


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for i in range(min(1000, len(X_train))):
            sample = X_train[i:i+1].astype(np.float32)
            sample = np.clip(sample, 0.0, 1.0)
            # print(f"Rep sample {i}: min={sample.min():.4f}, max={sample.max():.4f}")
            yield [sample]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter._experimental_disable_per_channel = False
    converter._experimental_new_quantizer = True
    converter.experimental_new_converter = True

    quantized_tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    print("Quantization OK. Input type:", interpreter.get_input_details()[0]['dtype'])
    print("Quantization OK. Output type:", interpreter.get_output_details()[0]['dtype'])

    tflite_fname = f"weather_model_1_quant_{name}.tflite"
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


for run_id in range(2):
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
best_model_file = f"weather_model_1_quant_{best['name']}.tflite"
shutil.copy(best_model_file, "weather_model_1_best.tflite")
print(f"Best model copied to: weather_model_1_best.tflite")

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
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    print(f"Input quant scale: {input_scale}, zero point: {input_zero_point}")
    print(f"Output quant scale: {output_scale}, zero point: {output_zero_point}")

    print("Input dtype from TFLite model:", input_details[0]['dtype'])
    print("Output dtype from TFLite model:", output_details[0]['dtype'])

    X_val_subset = X_val[:num_samples]
    y_val_subset = y_val[:num_samples]

    input_quantized = np.round(X_val_subset / input_scale + input_zero_point).astype(input_details[0]['dtype'])

    y_preds_dequant = []

    for i in range(len(input_quantized)):
        input_data = np.expand_dims(input_quantized[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if output_details[0]['dtype'] == np.int8:
            output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
        else:
            output_float = output_data.astype(np.float32)
        y_preds_dequant.append(output_float[0])

    # Debug: show example raw output before rescaling
    print("\nSample dequantized model outputs (before rescaling):")
    print(f"  t+1hr: {np.array(y_preds_dequant[:5]).round(3)}")

    print(f"\nRaw prediction stats — min: {np.min(y_preds_dequant):.4f}, max: {np.max(y_preds_dequant):.4f}")

    y_preds_rescaled = 0.5 * (np.array(y_preds_dequant) + 1) * (y_max - y_min) + y_min
    y_val_rescaled = 0.5 * (y_val_subset + 1) * (y_max - y_min) + y_min
    mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
    print(f"\nValidation MAE (in °C): {mae:.2f} °C")

y_val_array = val_df[targets].values.reshape(-1)
validate_quantized_model("weather_model_1_best.tflite", X_val, y_val_array, y_min, y_max)
