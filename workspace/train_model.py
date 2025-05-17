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

# Load preprocessed data
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

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

# Drop rows with NaNs
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

# Define feature and target columns
features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
    'temperature_delta', 'temp_lag1', 'humidity_lag1'
]
targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

# Normalize inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_val = scaler.transform(val_df[features])

# Save the scaler
joblib.dump(scaler, "scaler.joblib")

y_train = train_df[targets].values
y_val = val_df[targets].values


def build_and_train_model(name):
    print(f"\n--- Running: {name} ---\n")
    input_layer = tf.keras.layers.Input(shape=(len(features),))
    wide = tf.keras.layers.Dense(16)(input_layer)
    deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    deep = tf.keras.layers.Dropout(0.3)(deep)

    res = tf.keras.layers.Dense(64, activation='relu')(deep)
    shortcut = tf.keras.layers.Dense(64)(deep)
    res = tf.keras.layers.Add()([shortcut, res])
    res = tf.keras.layers.Dense(32, activation='relu')(res)

    merged = tf.keras.layers.Concatenate()([wide, res])
    output_layer = tf.keras.layers.Dense(3)(merged)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )

    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    baseline_loss = val_loss
    feature_importance = {}
    for i, feature in enumerate(features):
        X_val_permuted = copy.deepcopy(X_val)
        np.random.shuffle(X_val_permuted[:, i])
        permuted_loss = model.evaluate(X_val_permuted, y_val, verbose=0)[0]
        feature_importance[feature] = permuted_loss - baseline_loss

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance:.4f}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for i in range(0, len(X_train), 1):
            yield [X_train[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quantized_tflite_model = converter.convert()

    tflite_fname = f"weather_model_1_quant_{name}.tflite"
    with open(tflite_fname, "wb") as f:
        f.write(quantized_tflite_model)

    result = subprocess.run(
        ["edgetpu_compiler", tflite_fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(result.stdout)
    print(result.stderr)

    tflite_model_size_kb = os.path.getsize(tflite_fname) / 1024

    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\nFinal Metrics [{name}]:")
    print(f"  val_loss: {val_loss:.4f}")
    print(f"  val_mae: {val_mae:.4f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Quantized model size: {tflite_model_size_kb:.2f} KB")

    metrics = {
        "name": name,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "best_epoch": best_epoch,
        "feature_importance": sorted_importance,
        "model_size_kb": tflite_model_size_kb
    }

    with open(f"results_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


# Run default architecture
build_and_train_model("default")
