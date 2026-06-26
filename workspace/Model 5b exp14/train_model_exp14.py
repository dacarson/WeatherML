"""
Experiment 14: Multi-Point Temporal Extraction (Conv1D, 18 features)

Architecture: single-branch dilated Conv1D, 96 filters,
dilation stack [1, 2, 4, 8, 16, 32, 64], kernel=3.
GlobalAveragePooling replaced by explicit temporal extraction
at t=0, t=-30, t=-60, t=-120 (strided slice).

18 input features (no temp_delta_1, no lag columns, no slope features):
  temperature, uv, wind_avg, wind_gust, solar_radiation, illuminance,
  relative_humidity, station_pressure,
  day_of_year_sin, day_of_year_cos,
  time_of_day_sin, time_of_day_cos, time_of_day_sin2, time_of_day_cos2,
  wind_direction_sin, wind_direction_cos, wind_lull, rain_accumulated

Results (actual run, best epoch 38, early stopped epoch 58):
  val_loss = 0.002351
  val_mae  = 0.009763
  model_size = 291.80 KB
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "../.."          # local: workspace/ parent contains train_data_sf.csv
SCALER_INPUT_PATH  = "input_scaler_exp14.json"
SCALER_OUTPUT_PATH = "output_scaler_exp14.json"
RESULTS_PATH       = "results_exp14.json"
MODEL_FLOAT_PATH   = "weather_model_5b_conv1d_exp14_float.keras"
MODEL_TFLITE_PATH  = "weather_model_5b_conv1d_exp14.tflite"
CHECKPOINT_DIR     = "checkpoints"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN       = 180      # 3 hours of 1-min observations
N_FEATURES    = 18
FILTERS       = 96
KERNEL_SIZE   = 3
DILATION_RATES = [1, 2, 4, 8, 16, 32, 64]
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64
DROPOUT_RATE  = 0.3

LR            = 1e-4
BATCH_SIZE    = 512
MAX_EPOCHS    = 100
ES_PATIENCE   = 20       # early-stopping patience
LR_PATIENCE   = 5        # ReduceLR patience
LR_FACTOR     = 0.5
LR_MIN        = 1e-7
TARGET_PAD_C  = 2.0      # ±2°C padding on global target scaler

GAP_STEP_TOLERANCE_S = 90

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data_sf.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val_data_sf.csv"))


def _prepare_time_index(df, label):
    for c in ("time", "timestamp", "ts", "datetime", "date"):
        if c in df.columns:
            time_col = c
            break
    else:
        return df
    df = df.copy()
    s = df[time_col]
    if np.issubdtype(s.dtype, np.number):
        v = float(np.nanmax(s.to_numpy(dtype=np.float64)))
        unit = "ns" if v >= 1e17 else "us" if v >= 1e14 else "ms" if v >= 1e11 else "s"
        df[time_col] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    else:
        df[time_col] = pd.to_datetime(s, utc=True, errors="coerce")
    df = df.set_index(time_col).sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _add_future_targets(df, label, tolerance_s=90):
    if all(c in df.columns for c in ("temp_t+1hr", "temp_t+2hr", "temp_t+3hr")):
        return df
    base = df.reset_index()
    if "time" not in base.columns:
        base = base.rename(columns={base.columns[0]: "time"})
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
    base = base.sort_values("time").reset_index(drop=True)
    base["row_id"] = np.arange(len(base), dtype=np.int64)
    src = base[["time", "temperature"]].rename(columns={"temperature": "temperature_future"})
    tol = pd.Timedelta(seconds=int(tolerance_s))
    for mins, col in ((60, "temp_t+1hr"), (120, "temp_t+2hr"), (180, "temp_t+3hr")):
        want = base[["row_id", "time"]].copy()
        want["t_query"] = want["time"] + pd.Timedelta(minutes=int(mins))
        merged = pd.merge_asof(
            want.sort_values("t_query"), src,
            left_on="t_query", right_on="time",
            direction="forward", tolerance=tol,
        ).sort_values("row_id")
        base[col] = merged["temperature_future"].to_numpy()
    return base.drop(columns=["row_id"]).set_index("time")


def _invalidate_cross_gap_targets(df, label, tol_s=600):
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    target_horizons = {60: "temp_t+1hr", 120: "temp_t+2hr", 180: "temp_t+3hr"}
    present = {h: c for h, c in target_horizons.items() if c in df.columns}
    if not present:
        return df
    dt_s = df.index.to_series().diff().dt.total_seconds()
    gap_positions = np.flatnonzero((dt_s > float(tol_s)).to_numpy())
    if gap_positions.size == 0:
        return df
    df = df.copy()
    n_nulled = 0
    for pos in gap_positions:
        if pos == 0:
            continue
        gap_boundary = df.index[pos - 1]
        for h_min, col in present.items():
            cutoff = gap_boundary - pd.Timedelta(minutes=h_min)
            mask = (df.index > cutoff) & (df.index <= gap_boundary)
            n = int(mask.sum())
            if n > 0:
                df.loc[mask, col] = np.nan
                n_nulled += n
    if n_nulled:
        print(f"⚠️  {label}: nulled {n_nulled} cross-gap target lookups")
    return df


def _apply_gap_safety(df, label, seq_len, max_step_s):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    dt_s = df.index.to_series().diff().dt.total_seconds()
    gap_positions = np.flatnonzero((dt_s > float(max_step_s)).to_numpy())
    if gap_positions.size == 0:
        return df
    print(f"🔍 {label}: {gap_positions.size} gap(s) > {max_step_s}s")
    keep = np.ones(len(df), dtype=bool)
    drop_span = max(int(seq_len // 2), 0)
    for pos in gap_positions:
        keep[int(pos):min(int(pos) + drop_span, len(df))] = False
    dropped = int((~keep).sum())
    if dropped:
        print(f"🧩 {label}: dropping {dropped} rows around gaps")
        df = df.iloc[keep]
    return df


# ---------------------------------------------------------------------------
# Prepare dataframes
# ---------------------------------------------------------------------------
for df_ref, df_label in ((train_df, "train"), (val_df, "val")):
    pass  # will overwrite below

train_df = _prepare_time_index(train_df, "train")
val_df   = _prepare_time_index(val_df,   "val")
train_df = _add_future_targets(train_df, "train")
val_df   = _add_future_targets(val_df,   "val")
train_df = _invalidate_cross_gap_targets(train_df, "train", tol_s=600)
val_df   = _invalidate_cross_gap_targets(val_df,   "val",   tol_s=600)

# Cyclical time encodings
for df in (train_df, val_df):
    df['time_of_day_sin']  = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_cos']  = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_sin2'] = np.sin(4 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_cos2'] = np.cos(4 * np.pi * df['time_of_day'] / 24.0)
    df['day_of_year_sin']  = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos']  = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    if 'wind_direction' in df.columns:
        df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360.0)
        df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360.0)

# Targets
train_df['temp_diff_1hr'] = train_df['temp_t+1hr'] - train_df['temperature']
train_df['temp_diff_2hr'] = train_df['temp_t+2hr'] - train_df['temperature']
train_df['temp_diff_3hr'] = train_df['temp_t+3hr'] - train_df['temperature']
val_df['temp_diff_1hr']   = val_df['temp_t+1hr']   - val_df['temperature']
val_df['temp_diff_2hr']   = val_df['temp_t+2hr']   - val_df['temperature']
val_df['temp_diff_3hr']   = val_df['temp_t+3hr']   - val_df['temperature']

train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

train_df = _apply_gap_safety(train_df, "train", SEQ_LEN, GAP_STEP_TOLERANCE_S)
val_df   = _apply_gap_safety(val_df,   "val",   SEQ_LEN, GAP_STEP_TOLERANCE_S)

# ---------------------------------------------------------------------------
# Feature list — Exp 14 exact 18-feature order
# ---------------------------------------------------------------------------
features = [
    'temperature',
    'uv', 'wind_avg', 'wind_gust',
    'solar_radiation', 'illuminance',
    'relative_humidity', 'station_pressure',
    'day_of_year_sin', 'day_of_year_cos',
    'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
]
if 'wind_direction_sin' in train_df.columns:
    features.extend(['wind_direction_sin', 'wind_direction_cos'])
if 'wind_lull' in train_df.columns:
    features.append('wind_lull')
if 'rain_accumulated' in train_df.columns:
    features.append('rain_accumulated')

targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']
n_features = len(features)
print(f"Using {n_features} features: {features}")
assert n_features == N_FEATURES, f"Expected {N_FEATURES} features, got {n_features}"

# ---------------------------------------------------------------------------
# Input scaling — per-feature min/max with domain bounds
# ---------------------------------------------------------------------------
domain_bounds = {
    "temperature":      (-10, 55),
    "uv":               (0, None),
    "wind_avg":         (0, 30),
    "wind_gust":        (0, 40),
    "solar_radiation":  (0, None),
    "illuminance":      (0, None),
    "relative_humidity":(0, 100),
    "station_pressure": (None, None),
    "day_of_year_sin":  (-1, 1),
    "day_of_year_cos":  (-1, 1),
    "time_of_day_sin":  (-1, 1),
    "time_of_day_cos":  (-1, 1),
    "time_of_day_sin2": (-1, 1),
    "time_of_day_cos2": (-1, 1),
    "wind_direction_sin": (-1, 1),
    "wind_direction_cos": (-1, 1),
    "wind_lull":        (0, None),
    "rain_accumulated": (0, None),
}

X_train_df = train_df[features].copy()
X_val_df   = val_df[features].copy()
input_scaler = {}
for feat in features:
    f_min = train_df[feat].min()
    f_max = train_df[feat].max()
    pad   = 0.05 * (f_max - f_min)
    floor, ceiling = domain_bounds.get(feat, (None, None))
    adj_min = floor    if floor    is not None else f_min - pad
    adj_max = ceiling  if ceiling  is not None else f_max + pad
    input_scaler[feat] = {"min": adj_min, "max": adj_max}
    X_train_df[feat] = (X_train_df[feat] - adj_min) / (adj_max - adj_min)
    X_val_df[feat]   = (X_val_df[feat]   - adj_min) / (adj_max - adj_min)

X_train_flat = X_train_df.values
X_val_flat   = X_val_df.values

with open(SCALER_INPUT_PATH, "w") as f:
    json.dump(input_scaler, f, indent=2)
print(f"Saved {SCALER_INPUT_PATH}")

# ---------------------------------------------------------------------------
# Target scaling — global min/max with ±2°C padding (Model 5a approach)
# ---------------------------------------------------------------------------
raw_train_targets = train_df[targets].copy()
raw_val_targets   = val_df[targets].copy()

y_min = float(raw_train_targets.min().min()) - TARGET_PAD_C
y_max = float(raw_train_targets.max().max()) + TARGET_PAD_C

for t in targets:
    train_df[t] = 2.0 * (raw_train_targets[t] - y_min) / (y_max - y_min) - 1.0
    val_df[t]   = 2.0 * (raw_val_targets[t]   - y_min) / (y_max - y_min) - 1.0

with open(SCALER_OUTPUT_PATH, "w") as f:
    json.dump({"min": y_min, "max": y_max, "range": [y_min, y_max]}, f, indent=2)
print(f"Saved {SCALER_OUTPUT_PATH}  (y_min={y_min:.4f}, y_max={y_max:.4f})")

# ---------------------------------------------------------------------------
# Sequence datasets
# ---------------------------------------------------------------------------
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

y_all_train = train_df[targets].values
y_all_val   = val_df[targets].values

train_ds = timeseries_dataset_from_array(
    data=X_train_flat, targets=y_all_train,
    sequence_length=SEQ_LEN, sequence_stride=1,
    sampling_rate=1, batch_size=BATCH_SIZE, shuffle=True,
)
val_ds = timeseries_dataset_from_array(
    data=X_val_flat, targets=y_all_val,
    sequence_length=SEQ_LEN, sequence_stride=1,
    sampling_rate=1, batch_size=BATCH_SIZE, shuffle=False,
)

def split_targets(x, y):
    return x, (y[:, 0], y[:, 1], y[:, 2])

AUTOTUNE = tf.data.AUTOTUNE
train_steps = int(train_ds.cardinality().numpy())
train_ds = train_ds.map(split_targets, num_parallel_calls=AUTOTUNE).repeat().prefetch(4)
val_ds   = val_ds.map(split_targets,   num_parallel_calls=AUTOTUNE).prefetch(4)

print(f"train_steps={train_steps}  val_batches={val_ds.cardinality().numpy()}")

# ---------------------------------------------------------------------------
# Model — Exp 14 architecture
# ---------------------------------------------------------------------------
def build_exp14_model(seq_len, n_feat, filters, kernel, dilation_rates,
                      dense1, dense2, dropout):
    inp = tf.keras.layers.Input(shape=(seq_len, n_feat), name="input")

    # Initial projection (no BatchNorm, activation built in)
    x = tf.keras.layers.Conv1D(
        filters, kernel, padding='causal', activation='relu',
        name="conv_init"
    )(inp)

    # Dilated residual blocks — Conv1D + BatchNorm + ReLU + residual Add
    # (no ReLU6, no activation after Add — Exp 14 reverted Exp 11-13 quantization changes)
    for i, d in enumerate(dilation_rates):
        shortcut = x
        x = tf.keras.layers.Conv1D(
            filters, kernel, padding='causal', dilation_rate=d,
            use_bias=False, name=f"dconv_{d}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{d}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{d}")(x)
        x = tf.keras.layers.Add(name=f"add_{d}")([shortcut, x])

    # Multi-point temporal extraction via strided slice at 4 lag positions.
    # Lambda → STRIDED_SLICE in TFLite (Edge TPU compatible).
    t0   = tf.keras.layers.Lambda(lambda z: z[:, -1,   :], name="t0")(x)
    t30  = tf.keras.layers.Lambda(lambda z: z[:, -31,  :], name="t30")(x)
    t60  = tf.keras.layers.Lambda(lambda z: z[:, -61,  :], name="t60")(x)
    t120 = tf.keras.layers.Lambda(lambda z: z[:, -121, :], name="t120")(x)

    merged = tf.keras.layers.Concatenate(name="temporal_concat")([t0, t30, t60, t120])
    # merged: 4 × filters = 384-dim

    x = tf.keras.layers.Dense(dense1, activation='relu', name="dense1")(merged)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    x = tf.keras.layers.Dense(dense2, activation='relu', name="dense2")(x)

    out_1hr = tf.keras.layers.Dense(1, activation='linear', name='diff_1hr')(x)
    out_2hr = tf.keras.layers.Dense(1, activation='linear', name='diff_2hr')(x)
    out_3hr = tf.keras.layers.Dense(1, activation='linear', name='diff_3hr')(x)

    return tf.keras.Model(inp, [out_1hr, out_2hr, out_3hr], name="conv1d_exp14")


model = build_exp14_model(
    SEQ_LEN, n_features, FILTERS, KERNEL_SIZE, DILATION_RATES,
    DENSE_UNITS_1, DENSE_UNITS_2, DROPOUT_RATE,
)
model.summary()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

model.compile(
    optimizer=optimizer,
    loss={'diff_1hr': 'mse', 'diff_2hr': 'mse', 'diff_3hr': 'mse'},
    metrics={'diff_1hr': 'mae', 'diff_2hr': 'mae', 'diff_3hr': 'mae'},
)

# Custom ReduceLR callback (Keras 3 compatible — uses lambda to set lr)
class ReduceLRCallback(tf.keras.callbacks.Callback):
    def __init__(self, factor, patience, min_lr, monitor='val_loss', verbose=1):
        super().__init__()
        self.factor   = factor
        self.patience = patience
        self.min_lr   = min_lr
        self.monitor  = monitor
        self.verbose  = verbose
        self._best    = float('inf')
        self._wait    = 0

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor, float('inf'))
        if current < self._best:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate.assign(new_lr)
                if self.verbose:
                    print(f"\nEpoch {epoch+1}: ReduceLR — lr {old_lr:.3e} → {new_lr:.3e}")
                self._wait = 0


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=ES_PATIENCE, restore_best_weights=True, verbose=1,
    ),
    ReduceLRCallback(
        factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN, monitor='val_loss',
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best.weights.h5"),
        monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0,
    ),
]

history = model.fit(
    train_ds,
    epochs=MAX_EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)

# ---------------------------------------------------------------------------
# Save float model and results
# ---------------------------------------------------------------------------
model.save(MODEL_FLOAT_PATH)
print(f"Float model saved: {MODEL_FLOAT_PATH}")

best_epoch = int(np.argmin(history.history['val_loss']))
best_val_loss = float(history.history['val_loss'][best_epoch])
best_val_mae  = float(np.mean([
    history.history['val_diff_1hr_mae'][best_epoch],
    history.history['val_diff_2hr_mae'][best_epoch],
    history.history['val_diff_3hr_mae'][best_epoch],
]))

results = {
    "name": "conv1d_exp14_run1",
    "val_loss": best_val_loss,
    "val_mae":  best_val_mae,
    "best_epoch": best_epoch,
    "n_features": n_features,
    "features": features,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results: val_loss={best_val_loss:.6f}  val_mae={best_val_mae:.6f}  epoch={best_epoch}")

# ---------------------------------------------------------------------------
# TFLite quantization (representative dataset for INT8 calibration)
# ---------------------------------------------------------------------------
def build_sequence_data(X_flat, y, seq_len, max_samples=500):
    rows, tgts = [], []
    for i in range(seq_len - 1, min(seq_len - 1 + max_samples, len(X_flat))):
        rows.append(X_flat[i - seq_len + 1:i + 1, :])
        tgts.append(y[i])
    return np.array(rows, dtype=np.float32), np.array(tgts, dtype=np.float32)

X_rep, _ = build_sequence_data(X_val_flat, y_all_val, SEQ_LEN, max_samples=500)

def representative_dataset():
    for i in range(len(X_rep)):
        yield [X_rep[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open(MODEL_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
size_kb = os.path.getsize(MODEL_TFLITE_PATH) / 1024
print(f"TFLite model: {MODEL_TFLITE_PATH}  ({size_kb:.2f} KB)")
print("Done. Compile for Edge TPU with: edgetpu_compiler", MODEL_TFLITE_PATH)
