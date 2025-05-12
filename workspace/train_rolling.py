
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from tensorflow.keras.callbacks import EarlyStopping
import copy
import subprocess
import os
import csv

# ========== CONFIG ==========
CSV_PATH = "combined_data.csv"
TIMESTAMP_COL = "timestamp"
TARGET_COLS = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']
FEATURE_COLS = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
    'temperature_delta', 'temp_lag1', 'humidity_lag1'
]
N_SPLITS = 10
TRAIN_WINDOW = 10080  # 1 week
VAL_WINDOW = 10080    # 1 week

# ========== SPLITTER ==========
def rolling_expanding_split(df, timestamp_col, n_splits, train_window, val_window):
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    total_len = len(df)
    for i in range(n_splits):
        if train_window is None:
            train_end = i * val_window + val_window
            train_start = 0
        else:
            train_end = i * val_window + train_window
            train_start = max(0, train_end - train_window)
        val_start = train_end
        val_end = val_start + val_window
        if val_end > total_len:
            break
        yield df.iloc[train_start:train_end].copy(), df.iloc[val_start:val_end].copy()

# ========== FEATURE ENGINEERING ==========
def add_features(df):
    df['temp_lag1'] = df['temperature'].shift(1)
    df['humidity_lag1'] = df['relative_humidity'].shift(1)
    def rolling_slope(series, window):
        return series.rolling(window=window, center=True).apply(
            lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
            raw=True
        )
    df['temperature_delta'] = rolling_slope(df['temperature'], window=15)
    df.dropna(inplace=True)
    return df

# ========== MODEL ==========
def build_model(input_dim, output_dim):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    wide = tf.keras.layers.Dense(16)(input_layer)
    deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    deep = tf.keras.layers.Dropout(0.3)(deep)
    res = tf.keras.layers.Dense(64, activation='relu')(deep)
    shortcut = tf.keras.layers.Dense(64)(deep)
    res = tf.keras.layers.Add()([shortcut, res])
    res = tf.keras.layers.Dense(32, activation='relu')(res)
    merged = tf.keras.layers.Concatenate()([wide, res])
    output_layer = tf.keras.layers.Dense(output_dim)(merged)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# ========== TFLITE CONVERSION ==========
def convert_to_tflite(model, X_train, fold_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data_gen():
        for i in range(min(len(X_train), 500)):
            yield [X_train[i:i+1].astype(np.float32)]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(f"🔁 Converting model for fold {fold_name}...")
    tflite_model = converter.convert()
    tflite_fname = f"weather_model_1_quant_fold{fold_name}.tflite"
    with open(tflite_fname, "wb") as f:
        f.write(tflite_model)

    print(f"🚀 Compiling with EdgeTPU Compiler...")
    result = subprocess.run(
        ["edgetpu_compiler", tflite_fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(result.stdout)
    print(result.stderr)

# ========== MAIN ==========
def main():
    df = pd.read_csv(CSV_PATH, parse_dates=[TIMESTAMP_COL])

    metrics_file = "fold_metrics.csv"
    write_header = not os.path.exists(metrics_file)

    with open(metrics_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "fold", "val_loss", "val_mae", "baseline_loss"
            ] + [f"importance_{col}" for col in FEATURE_COLS])

    for fold, (train_df, val_df) in enumerate(rolling_expanding_split(df, TIMESTAMP_COL, N_SPLITS, TRAIN_WINDOW, VAL_WINDOW)):
        fold_name = f"{fold + 1}"
        print(f"\n🌀 Fold {fold_name}")

        train_df = add_features(train_df)
        val_df = add_features(val_df)

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[TARGET_COLS].values
        X_val = val_df[FEATURE_COLS].values
        y_val = val_df[TARGET_COLS].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = build_model(input_dim=X_train.shape[1], output_dim=len(TARGET_COLS))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping]
        )

        val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
        baseline_loss = val_loss

        feature_importance = {}
        for i, feature in enumerate(FEATURE_COLS):
            X_val_permuted = copy.deepcopy(X_val_scaled)
            np.random.shuffle(X_val_permuted[:, i])
            permuted_loss = model.evaluate(X_val_permuted, y_val, verbose=0)[0]
            feature_importance[feature] = permuted_loss - baseline_loss

        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\n📊 Permutation Feature Importance (Fold {fold_name}):")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")

        with open(metrics_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                fold + 1, val_loss, val_mae, baseline_loss
            ] + [feature_importance[col] for col in FEATURE_COLS])

        convert_to_tflite(model, X_train_scaled, fold_name)

if __name__ == "__main__":
    main()
