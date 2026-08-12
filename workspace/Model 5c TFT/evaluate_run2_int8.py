import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

# Offline INT8 n=500 evaluation for Track B Run 2 — this run predates the n=500 INT8
# methodology (only standardized starting around the Run 6 SEQ_LEN=180 breakthrough), so it
# was never given a fair, controlled INT8 number to compare against the live Grafana result
# that started this investigation (see MODEL_5C_TRACK_B_EXPERIMENT_LOG.md's post-conclusion
# addendum). Run 2's own INT8 TFLite export (model_trackb_dense_b_run2_int8.tflite) already
# exists — this script only adds the missing evaluation step, reusing the same deterministic
# first-500-chronological-windows methodology every other Track B run's n=500 number uses.
#
# Architecture and feature engineering below were reconstructed from the checkpoint file and
# input_scaler_5c_trackb.json (not from an original Run 2 script, which no longer exists) — see
# the file header comment in MODEL_5C_TRACK_B_EXPERIMENT_LOG.md's "Follow-up" entry for how each
# piece was verified. A pre-flight FP32 reproduction check (below) confirms the reconstruction
# before trusting the INT8 number.

RUN_DIR = "./results_5c_trackb_dense_b_run2"
CKPT = f"{RUN_DIR}/checkpoints/best_model.weights.h5"
INT8_TFLITE = f"{RUN_DIR}/model_trackb_dense_b_run2_int8.tflite"

with open(f"{RUN_DIR}/input_scaler_5c_trackb.json") as f:
    input_scaler = json.load(f)
with open(f"{RUN_DIR}/target_scaler_5c_trackb.json") as f:
    target_scaler = json.load(f)
y_min, y_max = target_scaler["min"], target_scaler["max"]

FEATURES = list(input_scaler.keys())
n_features = len(FEATURES)
print(f"Features ({n_features}): {FEATURES}")

# -------------------------------------------------------------------------
# Data pipeline — generic pieces (time index, sanity filter, future targets, gap
# invalidation) reused verbatim from the current script family; they're not tied to any
# particular feature set.
# -------------------------------------------------------------------------

def _prepare_time_index(df, label):
    time_col = next((c for c in ["time", "Time", "timestamp", "datetime"] if c in df.columns),
                     df.columns[0])
    df = df.copy()
    s = df[time_col]
    if np.issubdtype(s.dtype, np.number):
        v = float(np.nanmax(s.to_numpy(dtype=np.float64)))
        unit = "ns" if v >= 1e17 else "us" if v >= 1e14 else "ms" if v >= 1e11 else "s"
        df[time_col] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    else:
        df[time_col] = pd.to_datetime(s, utc=True, errors="coerce")
    if df[time_col].isna().any():
        n_bad = int(df[time_col].isna().sum())
        df = df.dropna(subset=[time_col])
        print(f"⚠️  {label}: dropped {n_bad} rows with unparseable/missing timestamps")
    df = df.set_index(time_col).sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _sanity_filter_temperature(df, label, window="31min", threshold_c=6.0):
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    local_median = df["temperature"].rolling(window, center=True, min_periods=3).median()
    spike = (df["temperature"] - local_median).abs() > threshold_c
    n_spikes = int(spike.sum())
    if n_spikes:
        df.loc[spike, "temperature"] = np.nan
        print(f"⚠️  {label}: nulled {n_spikes} temperature sensor-glitch rows")
    return df


def _add_future_targets(df, label, tolerance_s=90):
    if all(c in df.columns for c in ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]):
        return df
    base = df.reset_index()
    base = base.rename(columns={base.columns[0]: "time"})
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
    base = base.sort_values("time").reset_index(drop=True)
    base["row_id"] = np.arange(len(base), dtype=np.int64)
    src = base[["time", "temperature"]].rename(columns={"temperature": "temperature_future"})
    tol = pd.Timedelta(seconds=int(tolerance_s))
    for mins, col in ((60, "temp_t+1hr"), (120, "temp_t+2hr"), (180, "temp_t+3hr")):
        want = base[["row_id", "time"]].copy()
        want["t_query"] = want["time"] + pd.Timedelta(minutes=int(mins))
        merged = pd.merge_asof(want.sort_values("t_query"), src, left_on="t_query",
                               right_on="time", direction="forward", tolerance=tol)
        base[col] = merged.sort_values("row_id")["temperature_future"].to_numpy()
    return base.drop(columns=["row_id"]).set_index("time")


def _invalidate_targets_crossing_gaps(df, label, tol_s=600):
    present = {h: c for h, c in {60: "temp_t+1hr", 120: "temp_t+2hr", 180: "temp_t+3hr"}.items()
               if c in df.columns}
    dt_s = df.index.to_series().diff().dt.total_seconds()
    gap_positions = np.flatnonzero((dt_s > float(tol_s)).to_numpy())
    if gap_positions.size == 0:
        print(f"✅ {label}: no cross-gap target contamination")
        return df
    df = df.copy()
    n_nulled = 0
    for pos in gap_positions:
        if pos == 0:
            continue
        boundary = df.index[pos - 1]
        for h, col in present.items():
            mask = (df.index > boundary - pd.Timedelta(minutes=h)) & (df.index <= boundary)
            n = int(mask.sum())
            if n:
                df.loc[mask, col] = np.nan
                n_nulled += n
    if n_nulled:
        print(f"⚠️  {label}: nulled {n_nulled} cross-gap targets across {gap_positions.size} gap(s)")
    return df


def _add_backward_lag(df, source_col, minutes, out_col, tolerance_s=90):
    """current-row-aligned lookup of source_col's value `minutes` ago, via merge_asof —
    same pattern as _add_past_lags' temp_lag_300/360 (Run 16+), generalized to any column/lag."""
    base = df.reset_index().rename(columns={df.reset_index().columns[0]: "time"})
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
    base = base.sort_values("time").reset_index(drop=True)
    base["row_id"] = np.arange(len(base), dtype=np.int64)
    src = base[["time", source_col]].rename(columns={source_col: "_past"})
    tol = pd.Timedelta(seconds=int(tolerance_s))
    want = base[["row_id", "time"]].copy()
    want["t_query"] = want["time"] - pd.Timedelta(minutes=int(minutes))
    merged = pd.merge_asof(want.sort_values("t_query"), src, left_on="t_query",
                           right_on="time", direction="backward", tolerance=tol)
    base[out_col] = merged.sort_values("row_id")["_past"].to_numpy()
    return base.drop(columns=["row_id"]).set_index("time")


def rolling_slope(data, window):
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    slopes = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    x_c = x - x.mean()
    denom = np.sum(x_c ** 2)
    shape = (n - window + 1, window)
    strides = (data.strides[0], data.strides[0])
    wins = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    has_nan = np.any(np.isnan(wins), axis=1)
    y_c = wins - np.nanmean(wins, axis=1, keepdims=True)
    s = np.sum(x_c * y_c, axis=1) / denom
    s[has_nan] = np.nan
    slopes[window - 1:] = s
    return slopes


WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
val_df = pd.read_csv(f"{WORKSPACE}/val_data_sf.csv")
val_df = val_df.drop(columns=[c for c in ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]
                              if c in val_df.columns])
val_df = _prepare_time_index(val_df, "val_df")
val_df = _sanity_filter_temperature(val_df, "val_df")
val_df = _add_future_targets(val_df, "val_df")
val_df = _invalidate_targets_crossing_gaps(val_df, "val_df", tol_s=600)

val_df["time_of_day_sin"] = np.sin(2 * np.pi * val_df["time_of_day"] / 24.0)
val_df["time_of_day_cos"] = np.cos(2 * np.pi * val_df["time_of_day"] / 24.0)
val_df["time_of_day_sin2"] = np.sin(4 * np.pi * val_df["time_of_day"] / 24.0)
val_df["time_of_day_cos2"] = np.cos(4 * np.pi * val_df["time_of_day"] / 24.0)
val_df["day_of_year_sin"] = np.sin(2 * np.pi * val_df["day_of_year"] / 365.25)
val_df["day_of_year_cos"] = np.cos(2 * np.pi * val_df["day_of_year"] / 365.25)

print("⚙️  Computing rolling slope features...")
val_df["temp_slope_15"] = rolling_slope(val_df["temperature"].values, 15)
val_df["temp_slope_30"] = rolling_slope(val_df["temperature"].values, 30)
val_df["temp_slope_60"] = rolling_slope(val_df["temperature"].values, 60)
val_df["solar_slope_30"] = rolling_slope(val_df["solar_radiation"].values, 30)
val_df["humidity_slope_30"] = rolling_slope(val_df["relative_humidity"].values, 60)  # window=60, matches training
val_df["pressure_slope_60"] = rolling_slope(val_df["station_pressure"].values, 60)

print("⚙️  Computing backward-lag features (temp_diff_vs_1/2/3hr, pressure_lag120/180)...")
val_df = _add_backward_lag(val_df, "temperature", 60, "_temp_lag_60")
val_df = _add_backward_lag(val_df, "temperature", 120, "_temp_lag_120")
val_df = _add_backward_lag(val_df, "temperature", 180, "_temp_lag_180")
val_df["temp_diff_vs_1hr"] = val_df["temperature"] - val_df["_temp_lag_60"]
val_df["temp_diff_vs_2hr"] = val_df["temperature"] - val_df["_temp_lag_120"]
val_df["temp_diff_vs_3hr"] = val_df["temperature"] - val_df["_temp_lag_180"]
val_df = _add_backward_lag(val_df, "station_pressure", 120, "pressure_lag120")
val_df = _add_backward_lag(val_df, "station_pressure", 180, "pressure_lag180")

val_df["temp_diff_1hr"] = val_df["temp_t+1hr"] - val_df["temperature"]
val_df["temp_diff_2hr"] = val_df["temp_t+2hr"] - val_df["temperature"]
val_df["temp_diff_3hr"] = val_df["temp_t+3hr"] - val_df["temperature"]

targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]
dropna_cols = FEATURES + targets
val_df = val_df.dropna(subset=dropna_cols).reset_index(drop=True)
print(f"val rows after dropna: {len(val_df):,}")

X_val_df = val_df[FEATURES].copy()
for feat in FEATURES:
    lo = input_scaler[feat]["min"]
    hi = input_scaler[feat]["max"]
    X_val_df[feat] = ((X_val_df[feat] - lo) / (hi - lo)).clip(0.0, 1.0)
X_val_flat = X_val_df.values.astype(np.float32)

y_raw = val_df[targets].values.astype(np.float32)
y_scaled = 2.0 * (y_raw - y_min) / (y_max - y_min) - 1.0

# -------------------------------------------------------------------------
# Architecture — reconstructed from checkpoint H5 inspection: pure sequential stack
# (Input -> [Dense->BN->relu] x4 -> 3 linear heads), no branching, use_bias=False throughout.
# -------------------------------------------------------------------------
inp = tf.keras.layers.Input(shape=(1, n_features), name="input")
x = tf.keras.layers.Reshape((n_features,), name="flatten")(inp)
x = tf.keras.layers.Dense(512, use_bias=False, name="dense0")(x)
x = tf.keras.layers.BatchNormalization(name="bn0")(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dense(256, use_bias=False, name="dense1")(x)
x = tf.keras.layers.BatchNormalization(name="bn1")(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dense(128, use_bias=False, name="dense2")(x)
x = tf.keras.layers.BatchNormalization(name="bn2")(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dense(64, use_bias=False, name="dense3")(x)
x = tf.keras.layers.BatchNormalization(name="bn3")(x)
x = tf.keras.layers.Activation("relu")(x)
o1 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_1hr")(x)
o2 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_2hr")(x)
o3 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_3hr")(x)
model = tf.keras.Model(inp, [o1, o2, o3])

# Shape-based direct load — a pure linear chain has no branching ambiguity, but this stays
# robust regardless (same technique used for Run 11, see MODEL_5E project).
import h5py
f = h5py.File(CKPT, "r")
dense_shape_to_name = {
    (23, 512): "dense0", (512, 256): "dense1", (256, 128): "dense2", (128, 64): "dense3",
}
head_arrs = []
by_name = {l.name: l for l in model.layers}
for i in range(7):
    key = "dense" if i == 0 else f"dense_{i}"
    arr = f[f"layers/{key}/vars/0"][:]
    if arr.shape in dense_shape_to_name:
        by_name[dense_shape_to_name[arr.shape]].set_weights([arr])
    elif arr.shape == (64, 1):
        head_arrs.append(arr)
assert len(head_arrs) == 3, f"expected 3 head arrays, got {len(head_arrs)}"
for head_name, arr in zip(["diff_1hr", "diff_2hr", "diff_3hr"], head_arrs):
    by_name[head_name].set_weights([arr])

for i in range(4):
    bn_key = "batch_normalization" if i == 0 else f"batch_normalization_{i}"
    gamma, beta, moving_mean, moving_var = [f[f"layers/{bn_key}/vars/{j}"][:] for j in range(4)]
    by_name[f"bn{i}"].set_weights([gamma, beta, moving_mean, moving_var])
f.close()
print("✅ Checkpoint loaded (shape/name-matched)")

model.compile(optimizer="sgd", loss="mse",
             metrics={"diff_1hr": "mae", "diff_2hr": "mae", "diff_3hr": "mae"})

# -------------------------------------------------------------------------
# Verification gate: reproduce Run 2's own saved FP32 numbers before trusting anything else.
# -------------------------------------------------------------------------
X_seq = X_val_flat[:, np.newaxis, :]  # (N, 1, n_features)
res = model.evaluate(X_seq, {"diff_1hr": y_scaled[:, 0], "diff_2hr": y_scaled[:, 1],
                             "diff_3hr": y_scaled[:, 2]}, verbose=0, batch_size=2048)
names = model.metrics_names
scale = (y_max - y_min) / 2.0
mae_c = {h: res[names.index(f"{h}_mae")] * scale for h in ["diff_1hr", "diff_2hr", "diff_3hr"]}
print(f"\nFP32 reproduction check:")
print(f"  val_loss: {res[0]:.6f}  (Run 2's saved: 0.010067)")
print(f"  MAE: 1hr={mae_c['diff_1hr']:.4f}  2hr={mae_c['diff_2hr']:.4f}  3hr={mae_c['diff_3hr']:.4f}")
print(f"  Run 2's saved MAE: 1hr=0.4216  2hr=0.6236  3hr=0.7828")

if abs(mae_c["diff_3hr"] - 0.7828) > 0.05:
    raise RuntimeError(
        "FP32 reproduction did not match Run 2's saved numbers closely enough — architecture "
        "or feature reconstruction is wrong. Not proceeding to INT8 eval on an unverified model.")
print("✅ FP32 reproduction matches — architecture and features confirmed correct.\n")

# -------------------------------------------------------------------------
# INT8 evaluation — same deterministic first-500-chronological-windows methodology used
# throughout Track B (see train_model_track_b.py's INT8 validation block).
# -------------------------------------------------------------------------
interp = tf.lite.Interpreter(model_path=INT8_TFLITE)
interp.allocate_tensors()
in_det = interp.get_input_details()
out_det = interp.get_output_details()
in_scale, in_zp = in_det[0]["quantization"]
print(f"INT8 model input shape: {in_det[0]['shape']}, scale={in_scale}, zp={in_zp}")

n_val = min(500, X_seq.shape[0])
preds_int8 = [[] for _ in range(3)]
for i in range(n_val):
    sample = X_seq[i:i + 1]
    q_in = np.round(sample / in_scale + in_zp).astype(in_det[0]["dtype"])
    interp.set_tensor(in_det[0]["index"], q_in)
    interp.invoke()
    for j in range(3):
        out_s, out_zp = out_det[j]["quantization"]
        raw = interp.get_tensor(out_det[j]["index"])
        preds_int8[j].append(float(np.squeeze(raw - out_zp) * out_s))

print(f"\nValidation MAE — INT8 (°C), n={n_val}:")
int8_mae = {}
for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
    pred_c = (np.array(preds_int8[j]) + 1) * 0.5 * (y_max - y_min) + y_min
    true_c = (y_scaled[:n_val, j] + 1) * 0.5 * (y_max - y_min) + y_min
    mae = float(np.mean(np.abs(pred_c - true_c)))
    int8_mae[name] = mae
    print(f"  {name}: {mae:.3f}°C")

out_path = f"{RUN_DIR}/run2_int8_eval_n500.json"
with open(out_path, "w") as f:
    json.dump({
        "n": n_val,
        "int8_mae_c": int8_mae,
        "fp32_reproduction_check": {"val_loss": res[0], "mae_c": mae_c},
    }, f, indent=2)
print(f"\n✅ Results saved → {out_path}")
