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

# Load preprocessed data
train_df = pd.read_csv("train_data.csv")
print("Loaded train_data.csv:", train_df.shape)
val_df = pd.read_csv("val_data.csv")
print("Loaded val_data.csv:", val_df.shape)

# Add lag features
train_df['temp_lag1'] = train_df['temperature'].shift(1)
train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
val_df['temp_lag1'] = val_df['temperature'].shift(1)
val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)

# Compute temperature_delta only
def rolling_slope(series, window):
    return series.rolling(window=window, center=True).apply(
        lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
        raw=True
    )

train_df['temperature_delta'] = rolling_slope(train_df['temperature'], window=15)
val_df['temperature_delta'] = rolling_slope(val_df['temperature'], window=15)

print("Added lag and temperature_delta features.")

# Drop rows with NaNs
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
print("Dropped NaNs from datasets.")


# Define feature and target columns
features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 'temperature_delta',
    'temp_lag1', 'humidity_lag1', 'time_of_day', 'day_of_year'
]
targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

baseline_preds = val_df['temp_lag1'].values
baseline_true = val_df['temp_t+1hr'].values
baseline_mae = np.mean(np.abs(baseline_preds - baseline_true))
print(f"Baseline MAE using temp_lag1 for temp_t+1hr: {baseline_mae:.4f}")

# Set window size
window_size = 60  # 90 minutes

# Normalize features before windowing
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
print("Features normalized with StandardScaler.")

target_scaler = StandardScaler()
train_df[targets] = target_scaler.fit_transform(train_df[targets])
val_df[targets] = target_scaler.transform(val_df[targets])
joblib.dump(target_scaler, "target_scaler.joblib")

# Save the scaler
joblib.dump(scaler, "scaler.joblib")


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
).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: numpy_window_generator(X_val_np, y_val_np, window_size),
    output_signature=(
        tf.TensorSpec(shape=(window_size, len(features)), dtype=tf.float32),
        tf.TensorSpec(shape=(len(targets),), dtype=tf.float32),
    )
).batch(32).prefetch(tf.data.AUTOTUNE)


def build_and_train_model(name):
    print(f"\n--- Running: {name} ---\n")
    # Simpler Conv1D model that converges well and compiles on EdgeTPU
    input_layer = tf.keras.layers.Input(shape=(window_size, len(features)))
    conv = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='causal')(input_layer)
    conv = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='causal')(conv)
    conv = tf.keras.layers.GlobalAveragePooling1D()(conv)
    dense = tf.keras.layers.Dense(64, activation='relu')(conv)
    dense = tf.keras.layers.Dense(32, activation='relu')(dense)
    output_layer = tf.keras.layers.Dense(3)(dense)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[early_stopping]
    )
    print("Model training complete.")

    print("Evaluating model on validation data...")
    y_true_list, y_pred_list = [], []
    for x_batch, y_batch in val_dataset:
        preds = model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(preds)
    y_true = target_scaler.inverse_transform(np.vstack(y_true_list))
    y_pred = target_scaler.inverse_transform(np.vstack(y_pred_list))
    mae_per_target = np.mean(np.abs(y_true - y_pred), axis=0)
    for i, t in enumerate(targets):
        print(f"Validation MAE for {t} (denormalized): {mae_per_target[i]:.4f}")
    print(f"Average Validation MAE across all targets: {np.mean(mae_per_target):.4f}")
    val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Validation RMSE (denormalized): {val_rmse:.4f}")

    feature_importance = {}
    print("Calculating permutation feature importance...")

    X_val_np, y_val_np = [], []
    for x, y in val_dataset.unbatch().take(5000):  # Sample size for permutation
        X_val_np.append(x.numpy())
        y_val_np.append(y.numpy())
    X_val_np = np.array(X_val_np)
    y_val_np = np.array(y_val_np)

    for i, feature in enumerate(features):
        X_val_permuted = X_val_np.copy()
        flat = X_val_permuted[:, :, i].flatten()
        np.random.shuffle(flat)
        X_val_permuted[:, :, i] = flat.reshape(X_val_permuted[:, :, i].shape)
        permuted_loss = model.evaluate(X_val_permuted, y_val_np, verbose=0)[0]
        feature_importance[feature] = permuted_loss - model.evaluate(val_dataset, verbose=0)[0]

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance:.4f}")

    print("Converting model to TFLite with int8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for x, _ in train_dataset.unbatch().take(100):
            yield [np.expand_dims(x.numpy(), axis=0)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS  # fallback to float for unsupported ops
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quantized_tflite_model = converter.convert()

    tflite_fname = f"weather_model_conv1d_quant_{name}.tflite"
    with open(tflite_fname, "wb") as f:
        f.write(quantized_tflite_model)

    print(f"Compiling TFLite model with EdgeTPU compiler: {tflite_fname}")
    result = subprocess.run(
        ["edgetpu_compiler", tflite_fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(result.stdout)
    print(result.stderr)


# Run default architecture
build_and_train_model("default")
