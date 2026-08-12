import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

# Offline INT8 n=500 evaluation for Model 5d Run 3 — this project's closure was decided on FP32
# accuracy alone; the INT8 TFLite export already exists (model_5d_run3_int8.tflite) but was never
# actually evaluated and saved. This is step one of "Model 5f": before designing anything new,
# check whether Model 5d's own best FP32 checkpoint already shows the same
# worse-FP32-but-INT8-robust pattern confirmed for Track B Run 2 (see
# ../Model 5c TFT/MODEL_5C_TRACK_B_EXPERIMENT_LOG.md's post-conclusion addendum and
# ../Model 5c TFT/evaluate_run2_int8.py for the precedent this script follows).
#
# Both FP32 and INT8 TFLite files already exist for Run 3 — evaluated directly via the TFLite
# interpreter below, no Keras/checkpoint reconstruction needed (unlike Track B Run 2, which had
# no surviving original script). Data pipeline functions copied verbatim from train_model_5d.py
# (unchanged there since Run 3, confirmed by inspection) rather than reconstructed.

RUN_DIR = "./results_5d_run3"
FP32_TFLITE = f"{RUN_DIR}/model_5d_run3_fp32.tflite"
INT8_TFLITE = f"{RUN_DIR}/model_5d_run3_int8.tflite"

with open(f"{RUN_DIR}/input_scaler_5d.json") as f:
    input_scaler = json.load(f)
with open(f"{RUN_DIR}/target_scaler_5d.json") as f:
    target_scaler = json.load(f)
y_min, y_max = target_scaler["min"], target_scaler["max"]

FEATURES = list(input_scaler.keys())
n_features = len(FEATURES)
print(f"Features ({n_features}): {FEATURES}")


# -------------------------------------------------------------------------
# Data pipeline — copied verbatim from train_model_5d.py (unchanged there since Run 3).
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


def _add_past_lags(df, label, tolerance_s=90):
    if all(c in df.columns for c in ["temp_diff_vs_5hr", "temp_diff_vs_6hr"]):
        return df
    base = df.reset_index()
    if "time" not in base.columns:
        base = base.rename(columns={base.columns[0]: "time"})
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
    base = base.sort_values("time").reset_index(drop=True)
    base["row_id"] = np.arange(len(base), dtype=np.int64)
    src = base[["time", "temperature"]].copy().rename(columns={"temperature": "temperature_past"})
    tol = pd.Timedelta(seconds=int(tolerance_s))
    for mins, col in ((300, "temp_lag_300"), (360, "temp_lag_360")):
        want = base[["row_id", "time"]].copy()
        want["t_query"] = want["time"] - pd.Timedelta(minutes=int(mins))
        merged = pd.merge_asof(want.sort_values("t_query"), src,
                               left_on="t_query", right_on="time",
                               direction="backward", tolerance=tol)
        base[col] = merged.sort_values("row_id")["temperature_past"].to_numpy()
    base["temp_diff_vs_5hr"] = base["temperature"] - base["temp_lag_300"]
    base["temp_diff_vs_6hr"] = base["temperature"] - base["temp_lag_360"]
    return base.drop(columns=["row_id", "temp_lag_300", "temp_lag_360"]).set_index("time")


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
val_df = _add_past_lags(val_df, "val_df")

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
val_df["humidity_slope_30"] = rolling_slope(val_df["relative_humidity"].values, 60)
val_df["pressure_slope_60"] = rolling_slope(val_df["station_pressure"].values, 60)

val_df["temp_diff_1hr"] = val_df["temp_t+1hr"] - val_df["temperature"]
val_df["temp_diff_2hr"] = val_df["temp_t+2hr"] - val_df["temperature"]
val_df["temp_diff_3hr"] = val_df["temp_t+3hr"] - val_df["temperature"]

targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]
val_df = val_df.dropna(subset=FEATURES + targets).reset_index(drop=True)
print(f"val rows after dropna: {len(val_df):,}")

X_val_df = val_df[FEATURES].copy()
for feat in FEATURES:
    lo = input_scaler[feat]["min"]
    hi = input_scaler[feat]["max"]
    X_val_df[feat] = ((X_val_df[feat] - lo) / (hi - lo)).clip(0.0, 1.0)
X_val = X_val_df.values.astype(np.float32)  # flat (N, 20) — Model 5d has no sequence dim at all

y_raw = val_df[targets].values.astype(np.float32)
y_val = 2.0 * (y_raw - y_min) / (y_max - y_min) - 1.0

n_check = min(2000, X_val.shape[0])
X_val_small = X_val[:n_check]
y_val_small = y_val[:n_check]

# -------------------------------------------------------------------------
# FP32 reproduction check — evaluate the ALREADY-EXPORTED FP32 TFLite directly (no Keras
# reconstruction risk at all) against Run 3's own saved numbers before trusting anything further.
#
# Run 3's saved diff_Nhr_mae_c is a FULL-validation-set metric (Keras model.evaluate over all
# ~540K rows), not a 500-sample subsample — so checking against the first 500 CHRONOLOGICAL rows
# (a specific few-hours weather window, potentially easier or harder than the year-round average)
# isn't a fair comparison and did in fact diverge (0.564 vs 0.790 on the first attempt). Use a
# large RANDOM sample instead for a statistically representative comparison, fast enough to stay
# a real "check" rather than a full 540K-row loop.
# -------------------------------------------------------------------------
interp_fp32 = tf.lite.Interpreter(model_path=FP32_TFLITE)
interp_fp32.allocate_tensors()
in_det_fp32 = interp_fp32.get_input_details()
out_det_fp32 = interp_fp32.get_output_details()
print(f"FP32 model input shape: {in_det_fp32[0]['shape']}")

rng = np.random.default_rng(0)
n_fp32_check = min(5000, X_val.shape[0])
check_idxs = rng.choice(X_val.shape[0], size=n_fp32_check, replace=False)
X_check = X_val[check_idxs]
y_check = y_val[check_idxs]

preds_fp32 = [[] for _ in range(3)]
for i in range(n_fp32_check):
    sample = X_check[i:i + 1]
    interp_fp32.set_tensor(in_det_fp32[0]["index"], sample)
    interp_fp32.invoke()
    for j in range(3):
        preds_fp32[j].append(float(np.squeeze(interp_fp32.get_tensor(out_det_fp32[j]["index"]))))

print(f"\nFP32 reproduction check (n={n_fp32_check}, random sample):")
scale = (y_max - y_min) / 2.0
fp32_mae = {}
for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
    pred_c = (np.array(preds_fp32[j]) + 1) * 0.5 * (y_max - y_min) + y_min
    true_c = (y_check[:, j] + 1) * 0.5 * (y_max - y_min) + y_min
    mae = float(np.mean(np.abs(pred_c - true_c)))
    fp32_mae[name] = mae
    print(f"  {name}: {mae:.4f}°C")
print("  Run 3's saved MAE (full val set): 1hr=0.4287  2hr=0.6344  3hr=0.7904")

if abs(fp32_mae["diff_3hr"] - 0.7904) > 0.08:
    raise RuntimeError(
        "FP32 reproduction did not match Run 3's saved numbers closely enough — feature "
        "reconstruction is wrong. Not proceeding to INT8 eval on an unverified pipeline.")
print("✅ FP32 reproduction matches — feature pipeline confirmed correct.\n")

# -------------------------------------------------------------------------
# INT8 evaluation — same deterministic first-500-chronological-windows methodology used
# throughout Track B and Model 5d's own (unsaved) validation block.
# -------------------------------------------------------------------------
interp = tf.lite.Interpreter(model_path=INT8_TFLITE)
interp.allocate_tensors()
in_det = interp.get_input_details()
out_det = interp.get_output_details()
in_scale, in_zp = in_det[0]["quantization"]
print(f"INT8 model input shape: {in_det[0]['shape']}, scale={in_scale}, zp={in_zp}")

n_val = min(500, X_val_small.shape[0])
preds_int8 = [[] for _ in range(3)]
for i in range(n_val):
    sample = X_val_small[i:i + 1]
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
    true_c = (y_val_small[:n_val, j] + 1) * 0.5 * (y_max - y_min) + y_min
    mae = float(np.mean(np.abs(pred_c - true_c)))
    int8_mae[name] = mae
    print(f"  {name}: {mae:.3f}°C")

out_path = f"{RUN_DIR}/run3_int8_eval_n500.json"
with open(out_path, "w") as f:
    json.dump({
        "n": n_val,
        "int8_mae_c": int8_mae,
        "fp32_reproduction_check": fp32_mae,
    }, f, indent=2)
print(f"\n✅ Results saved → {out_path}")
