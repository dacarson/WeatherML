"""
Experiment 16 directory — contains the Exp 18 model (dual-branch Conv1D, 19 features).

Background: Exp 18 training code reused the Exp 16 filename (code name not updated),
so weather_model_5b_quant_conv1d_exp16_run1.tflite (394.54 KB) is the Exp 18 model.
The original Exp 16 (all-feature skips, 18 features, 301.60 KB) was overwritten.

Exp 18 Architecture:
  - Main branch: Conv1D(96 filters, dilation=[1,2,4,8,16,32,64]) on all 19 features
  - Temperature branch: Conv1D(32 filters, same dilation) on temperature + temp_delta_1 only
    (input[:, :, 0:2] — temperature at index 0, temp_delta_1 at index 1)
  - Temporal extraction: t=0/-30/-60/-120 from each branch
  - Merged dim: 4×96 (main) + 4×32 (temp) = 512-dim
  - Dense head: Dense(192) -> Dropout(0.3) -> Dense(96) -> 3x Dense(1)

19 input features (Exp 14/15 18 features + temp_delta_1 inserted at index 1):
  temperature (0), temp_delta_1 (1), uv, wind_avg, wind_gust, solar_radiation,
  illuminance, relative_humidity, station_pressure,
  day_of_year_sin, day_of_year_cos,
  time_of_day_sin, time_of_day_cos, time_of_day_sin2, time_of_day_cos2,
  wind_direction_sin, wind_direction_cos, wind_lull, rain_accumulated

  temp_delta_1 = temperature.diff() (1-minute rate of change, zeroed at gaps)

Results (actual run, best epoch 53, early stopped epoch 73):
  val_loss = 0.001362
  val_mae  = 0.005980
  model_size = 394.54 KB
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "../.."
SCALER_INPUT_PATH  = "input_scaler_exp16.json"
SCALER_OUTPUT_PATH = "output_scaler_exp16.json"
RESULTS_PATH       = "results_exp16.json"
MODEL_FLOAT_PATH   = "weather_model_5b_conv1d_exp16_float.keras"
MODEL_TFLITE_PATH  = "weather_model_5b_conv1d_exp16.tflite"
CHECKPOINT_DIR     = "checkpoints"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN          = 180
N_FEATURES       = 19
MAIN_FILTERS     = 96
TEMP_FILTERS     = 32
KERNEL_SIZE      = 3
DILATION_RATES   = [1, 2, 4, 8, 16, 32, 64]
DENSE_UNITS_1    = 192
DENSE_UNITS_2    = 96
DROPOUT_RATE     = 0.3

LR            = 1e-4
BATCH_SIZE    = 512
MAX_EPOCHS    = 100
ES_PATIENCE   = 20
LR_PATIENCE   = 8    # increased from 5 in Exp 17/18
LR_FACTOR     = 0.5
LR_MIN        = 1e-7
TARGET_PAD_C  = 2.0

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
train_df = _prepare_time_index(train_df, "train")
val_df   = _prepare_time_index(val_df,   "val")
train_df = _add_future_targets(train_df, "train")
val_df   = _add_future_targets(val_df,   "val")
train_df = _invalidate_cross_gap_targets(train_df, "train", tol_s=600)
val_df   = _invalidate_cross_gap_targets(val_df,   "val",   tol_s=600)

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

# temp_delta_1: 1-minute rate of change of temperature
# Zeroed at first row of each split (no prior row for diff)
for df in (train_df, val_df):
    df['temp_delta_1'] = df['temperature'].diff().fillna(0.0)

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
# Feature list — 19 features: temperature, temp_delta_1, then the 17 others
# temperature at index 0 and temp_delta_1 at index 1 are required for temp branch
# ---------------------------------------------------------------------------
features = [
    'temperature',
    'temp_delta_1',
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
assert features[0] == 'temperature', "temperature must be at index 0"
assert features[1] == 'temp_delta_1', "temp_delta_1 must be at index 1"

# ---------------------------------------------------------------------------
# Input scaling
# ---------------------------------------------------------------------------
domain_bounds = {
    "temperature":      (-10, 55),
    "temp_delta_1":     (-5, 5),
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
# Target scaling
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
# Model — Exp 18 / exp16_run1 dual-branch architecture
# ---------------------------------------------------------------------------
def build_exp16_model(seq_len, n_feat, main_filters, temp_filters, kernel,
                      dilation_rates, dense1, dense2, dropout):
    inp = tf.keras.layers.Input(shape=(seq_len, n_feat), name="input")

    # ---- Main branch: all features, 96 filters ----
    mx = tf.keras.layers.Conv1D(
        main_filters, kernel, padding='causal', activation='relu',
        name="main_conv_init"
    )(inp)
    for d in dilation_rates:
        shortcut = mx
        mx = tf.keras.layers.Conv1D(
            main_filters, kernel, padding='causal', dilation_rate=d,
            use_bias=False, name=f"main_dconv_{d}"
        )(mx)
        mx = tf.keras.layers.BatchNormalization(name=f"main_bn_{d}")(mx)
        mx = tf.keras.layers.ReLU(name=f"main_relu_{d}")(mx)
        mx = tf.keras.layers.Add(name=f"main_add_{d}")([shortcut, mx])

    main_t0   = tf.keras.layers.Lambda(lambda z: z[:, -1,   :], name="main_t0")(mx)
    main_t30  = tf.keras.layers.Lambda(lambda z: z[:, -31,  :], name="main_t30")(mx)
    main_t60  = tf.keras.layers.Lambda(lambda z: z[:, -61,  :], name="main_t60")(mx)
    main_t120 = tf.keras.layers.Lambda(lambda z: z[:, -121, :], name="main_t120")(mx)

    # ---- Temperature branch: temperature + temp_delta_1 only (indices 0:2), 32 filters ----
    temp_inp = tf.keras.layers.Lambda(
        lambda z: z[:, :, 0:2], name="temp_branch_input"
    )(inp)
    tx = tf.keras.layers.Conv1D(
        temp_filters, kernel, padding='causal', activation='relu',
        name="temp_conv_init"
    )(temp_inp)
    for d in dilation_rates:
        shortcut = tx
        tx = tf.keras.layers.Conv1D(
            temp_filters, kernel, padding='causal', dilation_rate=d,
            use_bias=False, name=f"temp_dconv_{d}"
        )(tx)
        tx = tf.keras.layers.BatchNormalization(name=f"temp_bn_{d}")(tx)
        tx = tf.keras.layers.ReLU(name=f"temp_relu_{d}")(tx)
        tx = tf.keras.layers.Add(name=f"temp_add_{d}")([shortcut, tx])

    temp_t0   = tf.keras.layers.Lambda(lambda z: z[:, -1,   :], name="temp_t0")(tx)
    temp_t30  = tf.keras.layers.Lambda(lambda z: z[:, -31,  :], name="temp_t30")(tx)
    temp_t60  = tf.keras.layers.Lambda(lambda z: z[:, -61,  :], name="temp_t60")(tx)
    temp_t120 = tf.keras.layers.Lambda(lambda z: z[:, -121, :], name="temp_t120")(tx)

    # Concatenate: 4×96 (main) + 4×32 (temp) = 512-dim
    merged = tf.keras.layers.Concatenate(name="temporal_concat")(
        [main_t0, main_t30, main_t60, main_t120,
         temp_t0, temp_t30, temp_t60, temp_t120]
    )

    x = tf.keras.layers.Dense(dense1, activation='relu', name="dense1")(merged)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    x = tf.keras.layers.Dense(dense2, activation='relu', name="dense2")(x)

    out_1hr = tf.keras.layers.Dense(1, activation='linear', name='diff_1hr')(x)
    out_2hr = tf.keras.layers.Dense(1, activation='linear', name='diff_2hr')(x)
    out_3hr = tf.keras.layers.Dense(1, activation='linear', name='diff_3hr')(x)

    return tf.keras.Model(inp, [out_1hr, out_2hr, out_3hr], name="conv1d_exp16")


model = build_exp16_model(
    SEQ_LEN, n_features, MAIN_FILTERS, TEMP_FILTERS, KERNEL_SIZE, DILATION_RATES,
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
    "name": "conv1d_exp16_run1",
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
# TFLite quantization
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
