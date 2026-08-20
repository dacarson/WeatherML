import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Offline re-quantization tool — NOT a training run. Rebuilds INT8 exports for Run 5, Run 7, and
# Run 8 from their saved `checkpoints/best_model.weights.h5` using one fixed, stratified
# representative dataset, so their INT8 numbers become directly comparable.
#
# Why: `representative_data_gen()` in every 5f training script (Run 5/7/8 and earlier) draws an
# unseeded random 2000-row sample from val data for INT8 calibration, with no guarantee it covers
# rare-but-important extremes. This is what caused Run 8's diff_1hr INT8 MAE to blow up to 2.02°C —
# the calibration draw missed positive diff_1hr swings above ~2.15°C almost entirely, so any real
# validation sample with a genuine >2°C 1hr warming got hard-clipped at inference. See
# MODEL_5F_EXPERIMENT_LOG.md's "Calibration Fix + Retroactive Re-quantization" entry.
#
# No retraining needed — only the post-training INT8 quantization step changes, and that only
# needs the already-trained FP32 weights.

RUNS = ["run5", "run6", "run7", "run8"]
SEED = 42
N_BINS = 10
CAP_PER_BIN = 200
N_EVAL = 500


# -----------------------------------------------------------------------------
# Feature engineering — identical to train_model_5f_run7.py (the superset: Run 7 added
# illuminance/solar_radiation/uv 1hr diffs on top of Run 5's columns; Run 5/8 simply don't select
# those extra columns via their own saved `features` list).
# -----------------------------------------------------------------------------
def _prepare_time_index(df, label):
    time_col = next((c for c in ("time", "timestamp", "ts", "datetime", "date")
                     if c in df.columns), None)
    if time_col is None:
        return df
    df = df.copy()
    s = df[time_col]
    if np.issubdtype(s.dtype, np.number):
        v = float(np.nanmax(s.to_numpy(dtype=np.float64)))
        unit = "ns" if v >= 1e17 else "us" if v >= 1e14 else "ms" if v >= 1e11 else "s"
        df[time_col] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    else:
        df[time_col] = pd.to_datetime(s, utc=True, errors="coerce")
    if df[time_col].isna().any():
        df = df.dropna(subset=[time_col])
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
    if spike.any():
        df.loc[spike, "temperature"] = np.nan
    return df


def _add_future_targets(df, label, tolerance_s=90):
    if all(c in df.columns for c in ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]):
        return df
    base = df.reset_index()
    if "time" not in base.columns:
        base = base.rename(columns={base.columns[0]: "time"})
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
    base = base.sort_values("time").reset_index(drop=True)
    base["row_id"] = np.arange(len(base), dtype=np.int64)
    src = base[["time", "temperature"]].copy().rename(columns={"temperature": "temperature_future"})
    tol = pd.Timedelta(seconds=int(tolerance_s))
    for mins, col in ((60, "temp_t+1hr"), (120, "temp_t+2hr"), (180, "temp_t+3hr")):
        want = base[["row_id", "time"]].copy()
        want["t_query"] = want["time"] + pd.Timedelta(minutes=int(mins))
        merged = pd.merge_asof(want.sort_values("t_query"), src,
                               left_on="t_query", right_on="time",
                               direction="forward", tolerance=tol)
        base[col] = merged.sort_values("row_id")["temperature_future"].to_numpy()
    return base.drop(columns=["row_id"]).set_index("time")


def _invalidate_targets_crossing_gaps(df, label, tol_s=90):
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    present = {h: c for h, c in {60: "temp_t+1hr", 120: "temp_t+2hr", 180: "temp_t+3hr"}.items()
               if c in df.columns}
    if not present:
        return df
    dt_s = df.index.to_series().diff().dt.total_seconds()
    gap_positions = np.flatnonzero((dt_s > float(tol_s)).to_numpy())
    if gap_positions.size == 0:
        return df
    df = df.copy()
    for pos in gap_positions:
        if pos == 0:
            continue
        boundary = df.index[pos - 1]
        for h, col in present.items():
            mask = (df.index > boundary - pd.Timedelta(minutes=h)) & (df.index <= boundary)
            if mask.any():
                df.loc[mask, col] = np.nan
    return df


def _add_past_lags(df, label, tolerance_s=90):
    _target_cols = ["temp_diff_vs_1hr", "temp_diff_vs_2hr", "temp_diff_vs_3hr",
                    "temp_diff_vs_5hr", "temp_diff_vs_6hr",
                    "pressure_lag120", "pressure_lag180",
                    "illuminance_diff_1hr", "solar_radiation_diff_1hr", "uv_diff_1hr"]
    if all(c in df.columns for c in _target_cols):
        return df

    def _backward_lag(base, source_col, minutes, out_col):
        src = base[["time", source_col]].rename(columns={source_col: "_past"})
        tol = pd.Timedelta(seconds=int(tolerance_s))
        want = base[["row_id", "time"]].copy()
        want["t_query"] = want["time"] - pd.Timedelta(minutes=int(minutes))
        merged = pd.merge_asof(want.sort_values("t_query"), src, left_on="t_query",
                               right_on="time", direction="backward", tolerance=tol)
        base[out_col] = merged.sort_values("row_id")["_past"].to_numpy()
        return base

    base = df.reset_index()
    if "time" not in base.columns:
        base = base.rename(columns={base.columns[0]: "time"})
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
    base = base.sort_values("time").reset_index(drop=True)
    base["row_id"] = np.arange(len(base), dtype=np.int64)

    for mins, tmp_col in ((60, "_temp_lag_60"), (120, "_temp_lag_120"), (180, "_temp_lag_180"),
                          (300, "_temp_lag_300"), (360, "_temp_lag_360")):
        base = _backward_lag(base, "temperature", mins, tmp_col)
    base["temp_diff_vs_1hr"] = base["temperature"] - base["_temp_lag_60"]
    base["temp_diff_vs_2hr"] = base["temperature"] - base["_temp_lag_120"]
    base["temp_diff_vs_3hr"] = base["temperature"] - base["_temp_lag_180"]
    base["temp_diff_vs_5hr"] = base["temperature"] - base["_temp_lag_300"]
    base["temp_diff_vs_6hr"] = base["temperature"] - base["_temp_lag_360"]

    base = _backward_lag(base, "station_pressure", 120, "pressure_lag120")
    base = _backward_lag(base, "station_pressure", 180, "pressure_lag180")

    base = _backward_lag(base, "illuminance", 60, "_illuminance_lag_60")
    base = _backward_lag(base, "solar_radiation", 60, "_solar_radiation_lag_60")
    base = _backward_lag(base, "uv", 60, "_uv_lag_60")
    base["illuminance_diff_1hr"] = base["illuminance"] - base["_illuminance_lag_60"]
    base["solar_radiation_diff_1hr"] = base["solar_radiation"] - base["_solar_radiation_lag_60"]
    base["uv_diff_1hr"] = base["uv"] - base["_uv_lag_60"]

    _drop_cols = ["row_id", "_temp_lag_60", "_temp_lag_120", "_temp_lag_180",
                 "_temp_lag_300", "_temp_lag_360",
                 "_illuminance_lag_60", "_solar_radiation_lag_60", "_uv_lag_60"]
    return base.drop(columns=_drop_cols).set_index("time")


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


def load_and_engineer(data_dir):
    train_df = pd.read_csv(f"{data_dir}/train_data_sf.csv")
    val_df = pd.read_csv(f"{data_dir}/val_data_sf.csv")

    _stale_target_cols = ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]
    train_df = train_df.drop(columns=[c for c in _stale_target_cols if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in _stale_target_cols if c in val_df.columns])

    train_df = _prepare_time_index(train_df, "train_df")
    val_df = _prepare_time_index(val_df, "val_df")
    train_df = _sanity_filter_temperature(train_df, "train_df")
    val_df = _sanity_filter_temperature(val_df, "val_df")
    train_df = _add_future_targets(train_df, "train_df")
    val_df = _add_future_targets(val_df, "val_df")
    train_df = _invalidate_targets_crossing_gaps(train_df, "train_df", tol_s=600)
    val_df = _invalidate_targets_crossing_gaps(val_df, "val_df", tol_s=600)
    train_df = _add_past_lags(train_df, "train_df")
    val_df = _add_past_lags(val_df, "val_df")

    for df in (train_df, val_df):
        df["time_of_day_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_sin2"] = np.sin(4 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_cos2"] = np.cos(4 * np.pi * df["time_of_day"] / 24.0)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    for df in (train_df, val_df):
        df["temp_slope_15"] = rolling_slope(df["temperature"].values, 15)
        df["temp_slope_30"] = rolling_slope(df["temperature"].values, 30)
        df["temp_slope_60"] = rolling_slope(df["temperature"].values, 60)
        df["solar_slope_30"] = rolling_slope(df["solar_radiation"].values, 30)
        df["humidity_slope_30"] = rolling_slope(df["relative_humidity"].values, 60)
        df["pressure_slope_60"] = rolling_slope(df["station_pressure"].values, 60)

    for df in (train_df, val_df):
        df["temp_diff_1hr"] = df["temp_t+1hr"] - df["temperature"]
        df["temp_diff_2hr"] = df["temp_t+2hr"] - df["temperature"]
        df["temp_diff_3hr"] = df["temp_t+3hr"] - df["temperature"]

    return train_df, val_df


# -----------------------------------------------------------------------------
# Stratified, seeded calibration set — fixes the root cause found in Run 8.
#
# First attempt stratified by the *true* temp_diff_1hr target and got a worse result for Run 5
# (0.206C -> 1.288C) — proof the true-target approach doesn't work: TFLite calibration range is
# set by what the model actually OUTPUTS for the representative inputs, not by the true label of
# those inputs. An MSE-trained model can (and evidently does) hedge toward smaller-magnitude
# predictions even for inputs whose true target is extreme, so stratifying inputs by true target
# doesn't reliably pull in extreme *predictions*. Fixed by stratifying on the model's own FP32
# predicted values instead, with hard-forced inclusion of the most extreme predictions per head so
# calibration can't miss them regardless of sampling luck.
# -----------------------------------------------------------------------------
def build_prediction_stratified_calibration(model, X_pool, n_bins=N_BINS, cap_per_bin=CAP_PER_BIN,
                                             n_extreme=25, seed=SEED, batch_size=8192):
    preds = model.predict(X_pool, batch_size=batch_size, verbose=0)
    preds = np.stack([np.squeeze(p, axis=-1) for p in preds], axis=1)  # (N, 3): diff_1hr/2hr/3hr

    rng = np.random.default_rng(seed)
    chosen = set()

    # Force in the most extreme predicted values for every head — calibration must see these
    # regardless of which rows a stratified/random draw would otherwise pick.
    for h in range(preds.shape[1]):
        order = np.argsort(preds[:, h])
        chosen.update(order[:n_extreme].tolist())
        chosen.update(order[-n_extreme:].tolist())

    # Stratify the rest by predicted diff_1hr decile (the head that broke) so the bulk of the
    # calibration set still reflects the model's typical output distribution, not just the tails.
    edges = np.quantile(preds[:, 0], np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bins = np.digitize(preds[:, 0], edges[1:-1], right=True)
    for b in range(n_bins):
        idx = np.flatnonzero(bins == b)
        if idx.size == 0:
            continue
        take = min(cap_per_bin, idx.size)
        pick = rng.choice(idx, size=take, replace=False)
        chosen.update(pick.tolist())

    return np.array(sorted(chosen))


def build_X(df, features, input_scaler):
    Xdf = df[features].copy()
    for feat in features:
        lo = input_scaler[feat]["min"]
        hi = input_scaler[feat]["max"]
        Xdf[feat] = ((Xdf[feat] - lo) / (hi - lo)).clip(0.0, 1.0)
    return Xdf.values.astype(np.float32)


def build_model(n_features, hidden):
    # `hidden` is an ordered list of hidden-layer widths — variable length, since Run 6 uses a
    # narrower 3-layer stack (128/64/32) vs. Run 5/7/8's 4-layer stack (512/256/128/64).
    def _reg():
        return tf.keras.regularizers.l2(1e-5)

    input_layer = tf.keras.layers.Input(shape=(n_features,), name="input")
    x = input_layer
    for i, h in enumerate(hidden):
        x = tf.keras.layers.Dense(h, use_bias=False, name=f"dense{i}", kernel_regularizer=_reg())(x)
        x = tf.keras.layers.Activation("relu6")(x)
    out_1 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_1hr")(x)
    out_2 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_2hr")(x)
    out_3 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_3hr")(x)
    return tf.keras.Model(inputs=input_layer, outputs=[out_1, out_2, out_3])


def requantize_run(run, train_df, val_df):
    results_dir = f"results_5f_{run}"
    with open(f"{results_dir}/results_5f_{run}.json") as f:
        results = json.load(f)
    features = results["features"]
    hp = results["hyperparams"]
    n_features = results["n_features"]
    with open(f"{results_dir}/input_scaler_5f.json") as f:
        input_scaler = json.load(f)
    with open(f"{results_dir}/target_scaler_5f.json") as f:
        target_scaler = json.load(f)
    y_min, y_max = target_scaler["min"], target_scaler["max"]

    targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]
    vdf = val_df.dropna(subset=features + targets).copy()

    X_val = build_X(vdf, features, input_scaler)
    y_val = np.stack([2.0 * (vdf[t] - y_min) / (y_max - y_min) - 1.0
                      for t in targets], axis=1).astype(np.float32)

    n_eval = min(N_EVAL, len(vdf))
    X_eval = X_val[:n_eval]
    y_eval = y_val[:n_eval]

    hidden = [hp[k] for k in ("hidden_1", "hidden_2", "hidden_3", "hidden_4") if k in hp]
    model = build_model(n_features, hidden)
    model.load_weights(f"{results_dir}/checkpoints/best_model.weights.h5")

    calib_idx = build_prediction_stratified_calibration(model, X_val)
    X_calib = X_val[calib_idx]
    print(f"[{run}] eval rows={n_eval}  calibration rows={X_calib.shape[0]} "
          f"(prediction-stratified from {X_val.shape[0]} val rows)")

    run_model = tf.function(model)
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, n_features], tf.float32, name="input"))

    def representative_data_gen():
        for i in range(X_calib.shape[0]):
            yield [X_calib[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_model = converter.convert()

    out_path = f"{results_dir}/model_5f_{run}_int8_requant.tflite"
    with open(out_path, "wb") as f:
        f.write(int8_model)

    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_scale, in_zp = in_det[0]["quantization"]

    preds = [[] for _ in range(3)]
    for i in range(n_eval):
        sample = X_eval[i:i + 1]
        q_in = np.round(sample / in_scale + in_zp).astype(in_det[0]["dtype"])
        interp.set_tensor(in_det[0]["index"], q_in)
        interp.invoke()
        for j in range(3):
            out_s, out_zp = out_det[j]["quantization"]
            raw = interp.get_tensor(out_det[j]["index"])
            preds[j].append(float(np.squeeze(raw - out_zp) * out_s))

    maes = {}
    for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        pred_c = (np.array(preds[j]) + 1) * 0.5 * (y_max - y_min) + y_min
        true_c = (y_eval[:n_eval, j] + 1) * 0.5 * (y_max - y_min) + y_min
        mae = float(np.mean(np.abs(pred_c - true_c)))
        maes[name] = mae
        print(f"  {name}: {mae:.3f}°C")

    for j, o in enumerate(out_det):
        scale, zp = o["quantization"]
        lo = scale * (-128 - zp)
        hi = scale * (127 - zp)
        lo_c = (lo + 1) * 0.5 * (y_max - y_min) + y_min
        hi_c = (hi + 1) * 0.5 * (y_max - y_min) + y_min
        print(f"  head[{j}] representable C_range=[{lo_c:+.2f},{hi_c:+.2f}]")

    return maes


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    print("Loading + engineering train/val data...")
    train_df, val_df = load_and_engineer(data_dir)
    print(f"Calibration methodology: prediction-stratified per model "
          f"({N_BINS} bins x up to {CAP_PER_BIN} on predicted diff_1hr, "
          f"plus top/bottom 25 extreme predictions per head, seed={SEED})\n")

    all_results = {}
    for run in RUNS:
        print(f"=== Re-quantizing {run} ===")
        all_results[run] = requantize_run(run, train_df, val_df)
        print()

    print("=== Summary (re-quantized, prediction-stratified calibration) ===")
    print(f"{'run':8s} {'1hr':>8s} {'2hr':>8s} {'3hr':>8s}")
    for run, maes in all_results.items():
        print(f"{run:8s} {maes['diff_1hr']:8.3f} {maes['diff_2hr']:8.3f} {maes['diff_3hr']:8.3f}")

    with open("requant_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved -> requant_comparison.json")


if __name__ == "__main__":
    main()
