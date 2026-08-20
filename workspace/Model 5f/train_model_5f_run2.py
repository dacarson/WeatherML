import tensorflow as tf

# Model 5f Run 2: Run 2's architecture + Run 2's original 23 features + temp_diff_vs_5hr/6hr
# (union, not substitution)
#
# Forked from train_model_5f_run1.py. See MODEL_5F_PLAN.md and MODEL_5F_EXPERIMENT_LOG.md's Run 1
# writeup for full rationale — corrects a mischaracterization in Run 1's own header: the feature
# swap there was NOT "curated vs uncurated" (18 of 20-23 features are identical between Track B
# Run 2's set and Model 5d Run 3's set, including relative_humidity/station_pressure raw — those
# were never dropped). The actual difference was narrow: Run 2 has temp_diff_vs_1hr/2hr/3hr +
# pressure_lag120/180 (5 features) that Run 3's set lacks; Run 3's set has temp_diff_vs_5hr/6hr
# (2 features) instead. Run 1 (Run 3's feature set + Run 2's architecture) landed at INT8 3hr
# 0.570°C — worse than Run 2's own 0.432°C. Reading across all three runs: architecture helps
# (Run 3's features through Run 2's arch: 0.634°C -> 0.570°C) but the SUBSTITUTION of
# temp_diff_vs_1/2/3hr + pressure_lag120/180 for temp_diff_vs_5hr/6hr cost more than it gained
# (Run 2: 0.432°C -> Run 1: 0.570°C, same architecture).
#
# Run 2 design: test whether temp_diff_vs_5hr/6hr adds value ADDITIVELY rather than as a
# replacement — Run 2's original 23 features + temp_diff_vs_5hr/6hr = 25 total, same architecture
# (BatchNorm, relu, 512->256->128->64) as Run 1. If this beats Run 2's 0.432°C, the 5hr/6hr signal
# is real but was lost by dropping the near-term diffs/pressure lags in Run 1, not by the
# 5hr/6hr features themselves being unhelpful.

RUN_NAME = "5f_run2"
RESULTS_DIR = f"./results_{RUN_NAME}"

# Track B Run 2's own L2 (not Model 5d's) — holding Run 2's training recipe fixed along with its
# architecture, since the goal is to isolate the features variable specifically, not introduce a
# second uncontrolled change.
L2_REG = 1e-5
# Track B Run 2's exact stack shape (BatchNorm + relu after each layer) — see checkpoint
# inspection in evaluate_run2_int8.py's header comment for how this was confirmed.
HIDDEN_1 = 512
HIDDEN_2 = 256
HIDDEN_3 = 128
HIDDEN_4 = 64

# Training — Track B Run 2's own hyperparams (hyperparams field in
# results_5c_trackb_dense_b_run2.json), not Model 5d's, for the same reason as L2_REG above.
MAX_EPOCHS = 300
INITIAL_LR = 1e-4
REDUCE_LR_PATIENCE = 12
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7
EARLY_STOP_PATIENCE = 40
TRAIN_BATCH_SIZE = 2048
VAL_BATCH_SIZE = 2048


def main():
    import multiprocessing as mp
    import multiprocessing

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    import os
    import json
    import shutil
    import time
    import threading
    import numpy as np
    import pandas as pd

    from tensorflow.keras.callbacks import EarlyStopping, Callback

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"ℹ️  Results directory: {RESULTS_DIR}")
    print(f"ℹ️  Run name: {RUN_NAME}")

    # -------------------------------------------------------------------------
    # GPU / threading configuration
    # -------------------------------------------------------------------------
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    if force_cpu:
        tf.config.set_visible_devices([], "GPU")
        print("ℹ️  FORCE_CPU=1: GPU disabled")

    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices and not force_cpu:
        print(f"✅ GPU detected: {len(physical_devices)} device(s)")
        try:
            for dev in physical_devices:
                tf.config.experimental.set_memory_growth(dev, True)
        except RuntimeError as e:
            print(f"⚠️  GPU config: {e}")
        cores = multiprocessing.cpu_count()
        tf.config.threading.set_intra_op_parallelism_threads(max(4, cores // 2))
        tf.config.threading.set_inter_op_parallelism_threads(2)
    else:
        print("⚠️  No GPU — using CPU")
        cores = multiprocessing.cpu_count()
        tf.config.threading.set_intra_op_parallelism_threads(max(4, cores - 4))
        tf.config.threading.set_inter_op_parallelism_threads(2)

    tf.config.set_soft_device_placement(True)

    # on_metal drives both the XLA JIT decision (Metal's PluggableDevice backend can't do XLA
    # JIT) and the mixed-precision decision (fp16 compute is safe here — no attention/softmax
    # numerical issues, and there's no QAT in this script to conflict with it).
    on_metal = (bool(physical_devices) and not force_cpu)
    if force_cpu:
        tf.config.optimizer.set_jit(True)
        print("ℹ️  XLA JIT enabled (CPU)")
    elif on_metal:
        tf.config.optimizer.set_jit(False)
        print("ℹ️  Metal GPU detected — XLA JIT disabled (Metal scheduler incompatibility)")
    else:
        tf.config.optimizer.set_jit(True)
        print("ℹ️  XLA JIT enabled (CPU)")

    if on_metal:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("ℹ️  Mixed precision enabled: fp16 compute / fp32 master weights (Metal GPU)")

    # -------------------------------------------------------------------------
    # Data loading helpers (identical to Model 5c Track A/B)
    # -------------------------------------------------------------------------
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
            n_bad = int(df[time_col].isna().sum())
            df = df.dropna(subset=[time_col])
            print(f"⚠️  {label}: dropped {n_bad} rows with unparseable/missing timestamps")
        df = df.set_index(time_col).sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        return df

    def _sanity_filter_temperature(df, label, window="31min", threshold_c=6.0):
        # Nulls rare sensor-glitch rows (brief dropout/self-heating, not real weather) where
        # temperature jumps implausibly fast and reverts a few minutes later. See
        # Model 5c Track B's MODEL_5C_EXPERIMENT_LOG.md "Data Quality" section for the original
        # investigation (2026-07-19).
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        df = df.copy()
        local_median = df["temperature"].rolling(window, center=True, min_periods=3).median()
        spike = (df["temperature"] - local_median).abs() > threshold_c
        n_spikes = int(spike.sum())
        if n_spikes:
            df.loc[spike, "temperature"] = np.nan
            print(f"⚠️  {label}: nulled {n_spikes} temperature sensor-glitch rows "
                  f"(>{threshold_c}°C from local {window} median)")
        return df

    def _add_future_targets(df, label, tolerance_s=90):
        if all(c in df.columns for c in ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]):
            return df
        if "temperature" not in df.columns:
            raise ValueError(f"{label}: missing 'temperature'")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{label}: need DatetimeIndex for target construction")
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
        print(f"\n❓ Missing target counts for {label}:")
        print(base[["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]].isna().sum())
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
        else:
            print(f"✅ {label}: no cross-gap target contamination")
        return df

    def _add_past_lags(df, label, tolerance_s=90):
        # Union of Track B Run 2's near-term lags (temp_diff_vs_1/2/3hr, pressure_lag120/180)
        # and Track A Deep Run's non-boundary ~5hr anchor (temp_diff_vs_5hr/6hr) — see Run 2's
        # own writeup in MODEL_5F_EXPERIMENT_LOG.md for why both sets are kept this time rather
        # than one substituting for the other. temp_diff_vs_1/2/3hr/pressure_lag construction
        # copied from evaluate_run2_int8.py's _add_backward_lag (already verified there via an
        # FP32-reproduction check against Run 2's own saved numbers).
        _target_cols = ["temp_diff_vs_1hr", "temp_diff_vs_2hr", "temp_diff_vs_3hr",
                        "temp_diff_vs_5hr", "temp_diff_vs_6hr",
                        "pressure_lag120", "pressure_lag180"]
        if all(c in df.columns for c in _target_cols):
            return df
        if "temperature" not in df.columns:
            raise ValueError(f"{label}: missing 'temperature'")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{label}: need DatetimeIndex for lag construction")

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

        print(f"\n❓ Missing lag counts for {label}:")
        print(base[_target_cols].isna().sum())
        _drop_cols = ["row_id", "_temp_lag_60", "_temp_lag_120", "_temp_lag_180",
                     "_temp_lag_300", "_temp_lag_360"]
        return base.drop(columns=_drop_cols).set_index("time")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    train_df = pd.read_csv(f"{data_dir}/train_data_sf.csv")
    val_df = pd.read_csv(f"{data_dir}/val_data_sf.csv")

    # Drop pre-baked temp_t+Nhr columns from the external export pipeline — see Model 5c Track
    # B's MODEL_5C_EXPERIMENT_LOG.md for why these can't be trusted (row-count .shift(), not
    # time-based; silently misaligns targets across sampling gaps).
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

    # -------------------------------------------------------------------------
    # Cyclical encodings
    # -------------------------------------------------------------------------
    for df in (train_df, val_df):
        df["time_of_day_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_sin2"] = np.sin(4 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_cos2"] = np.cos(4 * np.pi * df["time_of_day"] / 24.0)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # -------------------------------------------------------------------------
    # Rolling slope features
    # -------------------------------------------------------------------------
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

    print("⚙️  Computing rolling slope features...")
    for df in (train_df, val_df):
        df["temp_slope_15"] = rolling_slope(df["temperature"].values, 15)
        df["temp_slope_30"] = rolling_slope(df["temperature"].values, 30)
        df["temp_slope_60"] = rolling_slope(df["temperature"].values, 60)
        df["solar_slope_30"] = rolling_slope(df["solar_radiation"].values, 30)
        df["humidity_slope_30"] = rolling_slope(df["relative_humidity"].values, 60)
        df["pressure_slope_60"] = rolling_slope(df["station_pressure"].values, 60)
    print("   ✅ Slope features computed")

    # -------------------------------------------------------------------------
    # Targets
    # -------------------------------------------------------------------------
    for df in (train_df, val_df):
        df["temp_diff_1hr"] = df["temp_t+1hr"] - df["temperature"]
        df["temp_diff_2hr"] = df["temp_t+2hr"] - df["temperature"]
        df["temp_diff_3hr"] = df["temp_t+3hr"] - df["temperature"]

    # -------------------------------------------------------------------------
    # Feature list — union of Track B Run 2's original 23 features and Model 5d Run 3's
    # temp_diff_vs_5hr/6hr addition = 25 total. Run 1 substituted 5hr/6hr in place of
    # temp_diff_vs_1/2/3hr + pressure_lag120/180 and lost accuracy; this run adds 5hr/6hr on top
    # instead of replacing anything, to test whether the 5hr/6hr signal is additive.
    # -------------------------------------------------------------------------
    temporal_features = [
        "time_of_day_sin", "time_of_day_cos", "time_of_day_sin2", "time_of_day_cos2",
        "day_of_year_sin", "day_of_year_cos",
    ]
    temperature_features = [
        "temperature",
        "temp_slope_15",
        "temp_slope_30",
        "temp_slope_60",
        "temp_diff_vs_1hr",
        "temp_diff_vs_2hr",
        "temp_diff_vs_3hr",
        "temp_diff_vs_5hr",
        "temp_diff_vs_6hr",
    ]
    humidity_features = [
        "relative_humidity",
        "humidity_slope_30",
    ]
    pressure_features = [
        "station_pressure",
        "pressure_slope_60",
        "pressure_lag120",
        "pressure_lag180",
    ]
    solar_features = [
        "solar_radiation",
        "solar_slope_30",
        "illuminance",
        "uv",
    ]

    features = (temporal_features + temperature_features + humidity_features +
                pressure_features + solar_features)
    targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]

    print(f"\nFeature set: {len(features)} features")
    for f in features:
        print(f"  {f}")

    _dropna_cols = features + targets
    train_df.dropna(subset=_dropna_cols, inplace=True)
    val_df.dropna(subset=_dropna_cols, inplace=True)
    print(f"\nAfter dropna: train={len(train_df):,} rows, val={len(val_df):,} rows")

    # -------------------------------------------------------------------------
    # Per-feature min/max scaling with domain bounds and ±5% padding
    # -------------------------------------------------------------------------
    domain_bounds = {
        "temperature":        (-10, 55),
        "temp_slope_15":      (None, None),
        "temp_slope_30":      (None, None),
        "temp_slope_60":      (None, None),
        "solar_slope_30":     (None, None),
        # Tight, ~1st/99th-percentile-informed bounds for the two dominant lag features — see
        # Model 5c Track B's Run 17/18 investigation: raw min-max wastes most of INT8's 256
        # levels on rarely-visited tails for these specific features.
        "temp_diff_vs_5hr":   (-8, 12),
        "temp_diff_vs_6hr":   (-9, 13),
        "relative_humidity":  (0, 100),
        "humidity_slope_30":  (None, None),
        "pressure_slope_60":  (None, None),
        "uv":                 (0, None),
        "solar_radiation":    (0, None),
        "illuminance":        (0, None),
        "station_pressure":   (None, None),
        "day_of_year_sin":    (-1, 1),
        "day_of_year_cos":    (-1, 1),
    }

    X_train_df = train_df[features].copy()
    X_val_df = val_df[features].copy()
    input_scaler = {}
    for feat in features:
        f_min = float(train_df[feat].min())
        f_max = float(train_df[feat].max())
        pad = 0.05 * (f_max - f_min)
        floor, ceiling = domain_bounds.get(feat, (None, None))
        lo = floor if floor is not None else f_min - pad
        hi = ceiling if ceiling is not None else f_max + pad
        input_scaler[feat] = {"min": lo, "max": hi}
        # Explicit clip to [0,1] — the exported model's 'input' tensor uses one shared
        # per-tensor INT8 scale across all channels, so any unclipped excursion stretches that
        # shared scale for every channel, not just its own. See Track B Run 18.
        X_train_df[feat] = ((X_train_df[feat] - lo) / (hi - lo)).clip(0.0, 1.0)
        X_val_df[feat] = ((X_val_df[feat] - lo) / (hi - lo)).clip(0.0, 1.0)

    X_train = X_train_df.values.astype(np.float32)
    X_val = X_val_df.values.astype(np.float32)

    scaler_path = os.path.join(RESULTS_DIR, "input_scaler_5f.json")
    with open(scaler_path, "w") as f:
        json.dump(input_scaler, f, indent=2)
    print(f"✅ Input scaler saved → {scaler_path}")

    # Global target scaling (same approach as Model 5a/5b/5c)
    raw_train_tgts = train_df[targets].copy()
    raw_val_tgts = val_df[targets].copy()
    TARGET_PAD_C = 2.0
    y_min = float(raw_train_tgts.min().min()) - TARGET_PAD_C
    y_max = float(raw_train_tgts.max().max()) + TARGET_PAD_C
    y_train = np.stack([2.0 * (raw_train_tgts[t] - y_min) / (y_max - y_min) - 1.0
                        for t in targets], axis=1).astype(np.float32)
    y_val = np.stack([2.0 * (raw_val_tgts[t] - y_min) / (y_max - y_min) - 1.0
                      for t in targets], axis=1).astype(np.float32)

    target_scaler_path = os.path.join(RESULTS_DIR, "target_scaler_5f.json")
    with open(target_scaler_path, "w") as f:
        json.dump({"min": y_min, "max": y_max}, f, indent=2)
    print(f"✅ Target scaler saved → {target_scaler_path}")

    print(f"\n=== SCALING BOUNDS ===")
    print(f"Global target range: {y_min:.2f}°C to {y_max:.2f}°C")
    for t in targets:
        print(f"  {t}: raw [{float(raw_train_tgts[t].min()):.2f}, {float(raw_train_tgts[t].max()):.2f}]°C")

    n_features = len(features)

    # -------------------------------------------------------------------------
    # tf.data datasets — SEQ_LEN=1: each row is already a complete flat feature vector, no
    # windowing needed (unlike Track B's timeseries_dataset_from_array over SEQ_LEN=180).
    # -------------------------------------------------------------------------
    def split_targets(x, y):
        return x, (y[:, 0], y[:, 1], y[:, 2])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=min(100_000, len(X_train)), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(TRAIN_BATCH_SIZE)
    train_ds = train_ds.map(lambda x, y: split_targets(x, y), num_parallel_calls=AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(VAL_BATCH_SIZE)
    val_ds = val_ds.map(lambda x, y: split_targets(x, y), num_parallel_calls=AUTOTUNE)

    train_steps = int(train_ds.cardinality().numpy())
    train_ds = train_ds.repeat()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    print(f"\nTraining: {train_steps} batches/epoch (batch={TRAIN_BATCH_SIZE})")
    print(f"Validation: {val_ds.cardinality().numpy()} batches")

    # Small in-memory set for permutation importance and TFLite validation.
    _n_small = min(2000, len(X_val))
    X_val_small = X_val[:_n_small]
    y_val_small = y_val[:_n_small]
    print(f"Small val set: X_val_small={X_val_small.shape}, y_val_small={y_val_small.shape}")

    # -------------------------------------------------------------------------
    # Training watchdog (identical to Track A/B)
    # -------------------------------------------------------------------------
    class TrainingWatchdog(Callback):
        def __init__(self, timeout_minutes=5, check_interval_seconds=30,
                     train_ds=None, total_batches_hint=None):
            super().__init__()
            self.base_timeout_seconds = timeout_minutes * 60
            self.timeout_seconds = self.base_timeout_seconds
            self.check_interval = check_interval_seconds
            self.train_ds = train_ds
            self._total_batches_hint = total_batches_hint
            self.last_activity_time = None
            self.current_epoch = None
            self.current_batch = None
            self.total_batches = None
            self.epoch_start_time = None
            self.last_batch_time = None
            self.batch_durations = []
            self.batch_update_frequency = 20
            self.watchdog_thread = None
            self.stop_watchdog = threading.Event()

        def on_train_begin(self, logs=None):
            self.last_activity_time = time.time()
            self.stop_watchdog.clear()
            try:
                c = self.train_ds.cardinality().numpy()
                self.total_batches = int(c) if c >= 0 else self._total_batches_hint
            except Exception:
                self.total_batches = self._total_batches_hint
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="Watchdog")
            self.watchdog_thread.start()
            print(f"🛡️  Watchdog started (timeout: {self.base_timeout_seconds/60:.1f} min)")

        def on_train_end(self, logs=None):
            if self.watchdog_thread:
                self.stop_watchdog.set()
                self.watchdog_thread.join(timeout=5.0)
                print("🛡️  Training watchdog stopped")

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch
            self.current_batch = 0
            self.epoch_start_time = time.time()
            self.last_activity_time = time.time()
            self.batch_durations = []

        def on_epoch_end(self, epoch, logs=None):
            self.last_activity_time = time.time()

        def on_train_batch_begin(self, batch, logs=None):
            self.current_batch = batch
            self.last_batch_time = time.time()
            if batch % self.batch_update_frequency == 0:
                self.last_activity_time = time.time()

        def on_train_batch_end(self, batch, logs=None):
            dur = time.time() - self.last_batch_time if self.last_batch_time else 0
            self.batch_durations.append(dur)
            if len(self.batch_durations) > 100:
                self.batch_durations = self.batch_durations[-100:]
            if len(self.batch_durations) >= 10:
                avg = sum(self.batch_durations) / len(self.batch_durations)
                self.timeout_seconds = max(avg * self.batch_update_frequency * 5,
                                           self.base_timeout_seconds)
            if batch % self.batch_update_frequency == 0:
                self.last_activity_time = time.time()

        def _watchdog_loop(self):
            while not self.stop_watchdog.is_set():
                if self.stop_watchdog.wait(self.check_interval):
                    break
                if self.last_activity_time is None:
                    continue
                elapsed = time.time() - self.last_activity_time
                if elapsed > self.timeout_seconds:
                    print(f"\n🚨 HANG DETECTED: {elapsed/60:.1f} min idle — forcing exit")
                    try:
                        with open(os.path.join(RESULTS_DIR, "training_hang.json"), "w") as f:
                            json.dump({"epoch": self.current_epoch, "batch": self.current_batch}, f)
                    except Exception:
                        pass
                    os._exit(99)

    # -------------------------------------------------------------------------
    # Callbacks (identical to Track A/B)
    # -------------------------------------------------------------------------
    class TaskLossLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            keys = tuple(f"val_diff_{h}hr_loss" for h in (1, 2, 3))
            if all(k in logs for k in keys):
                logs["val_task_loss"] = sum(logs[k] for k in keys)
                return
            logs["val_task_loss"] = logs.get("val_loss", float("inf"))

    class NaNLossTerminator(Callback):
        def on_train_batch_end(self, batch, logs=None):
            loss = (logs or {}).get("loss", 0.0)
            if np.isnan(loss) or np.isinf(loss):
                print(f"\n🚨 NaN/Inf loss at batch {batch + 1} — stopping")
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            loss = (logs or {}).get("loss", 0.0)
            if np.isnan(loss) or np.isinf(loss):
                print(f"\n🚨 NaN/Inf loss at epoch {epoch + 1} — stopping")
                self.model.stop_training = True

    class ReduceLRCallback(Callback):
        def __init__(self, initial_lr, monitor="val_task_loss", factor=0.5,
                     patience=12, min_lr=1e-7, min_delta=0.0, verbose=1):
            super().__init__()
            self.initial_lr = initial_lr
            self.monitor = monitor
            self.factor = factor
            self.patience = patience
            self.min_lr = min_lr
            self.min_delta = min_delta
            self.verbose = verbose
            self.best = float("inf")
            self.wait = 0

        def _unwrap(self, opt):
            for attr in ("_optimizer", "inner_optimizer"):
                inner = getattr(opt, attr, None)
                if inner is not None:
                    return inner
            return opt

        def _get_lr(self):
            inner = self._unwrap(self.model.optimizer)
            lr_var = getattr(inner, "_learning_rate", None)
            if lr_var is not None and hasattr(lr_var, "numpy"):
                return float(lr_var.numpy())
            try:
                lr = inner.learning_rate
                return float(lr.numpy()) if hasattr(lr, "numpy") else float(lr)
            except Exception:
                return float(self.initial_lr)

        def _set_lr(self, new_lr):
            inner = self._unwrap(self.model.optimizer)
            lr_var = getattr(inner, "_learning_rate", None)
            if lr_var is not None and hasattr(lr_var, "assign"):
                lr_var.assign(float(new_lr))
            else:
                inner.learning_rate = float(new_lr)
            confirmed = self._get_lr()
            if abs(confirmed - float(new_lr)) > 1e-10:
                print(f"\n⚠️  LR set {new_lr:.2e}, readback {confirmed:.2e}")
            elif self.verbose:
                print(f"\n✅ LR → {confirmed:.2e}")

        def on_epoch_end(self, epoch, logs=None):
            current = (logs or {}).get(self.monitor)
            if current is None:
                return
            current_lr = self._get_lr()
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    if current_lr > self.min_lr:
                        new_lr = max(current_lr * self.factor, self.min_lr)
                        self._set_lr(new_lr)
                        if self.verbose:
                            print(f"\nEpoch {epoch+1}: ReduceLR → {new_lr:.2e} "
                                  f"(best={self.best:.6f}, wait={self.wait})")
                    self.wait = 0
            if logs is not None:
                logs["learning_rate"] = current_lr

    class EpochProgressCallback(Callback):
        def __init__(self, log_every=50, total_steps=None, max_epochs=MAX_EPOCHS):
            super().__init__()
            self.log_every = log_every
            self.total_steps = total_steps
            self.max_epochs = max_epochs
            self._epoch_start = None
            self._loss_window = []

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_start = time.time()
            self._loss_window = []
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}", flush=True)

        def on_train_batch_end(self, batch, logs=None):
            loss = (logs or {}).get("loss", 0.0)
            self._loss_window.append(float(loss))
            if len(self._loss_window) > self.log_every:
                self._loss_window.pop(0)
            if (batch + 1) % self.log_every != 0:
                return
            elapsed = time.time() - self._epoch_start
            avg_loss = sum(self._loss_window) / len(self._loss_window)
            if self.total_steps:
                pct = (batch + 1) / self.total_steps * 100
                eta = elapsed / (batch + 1) * (self.total_steps - batch - 1)
                print(f"  step {batch+1}/{self.total_steps} ({pct:.0f}%) "
                      f"— loss {avg_loss:.5f} — {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                      flush=True)
            else:
                print(f"  step {batch+1} — loss {avg_loss:.5f} — {elapsed:.0f}s elapsed",
                      flush=True)

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self._epoch_start
            vtl = (logs or {}).get("val_task_loss", (logs or {}).get("val_loss", float("nan")))
            try:
                opt = self.model.optimizer
                for attr in ("_optimizer", "inner_optimizer"):
                    inner = getattr(opt, attr, None)
                    if inner is not None:
                        opt = inner
                        break
                lr_var = getattr(opt, "_learning_rate", None)
                lr = float(lr_var.numpy()) if lr_var is not None and hasattr(lr_var, "numpy") \
                    else float(opt.learning_rate)
            except Exception:
                lr = float("nan")
            print(f"  → val_task_loss={vtl:.6f}  lr={lr:.2e}  epoch={elapsed:.0f}s", flush=True)

    class LatestEpochSaver(Callback):
        def __init__(self, checkpoint_dir, initial_best=float("inf")):
            super().__init__()
            self.weights_path = os.path.join(checkpoint_dir, "model_latest.weights.h5")
            self.meta_path = os.path.join(checkpoint_dir, "model_latest_epoch.json")
            self.best_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
            self.best_task_loss = initial_best

        def on_epoch_end(self, epoch, logs=None):
            task_loss = (logs or {}).get("val_task_loss", float("inf"))
            if np.isnan(task_loss) or np.isinf(task_loss):
                return
            try:
                self.model.save_weights(self.weights_path)
                with open(self.meta_path, "w") as f:
                    json.dump({"epoch": epoch + 1}, f)
            except Exception as e:
                print(f"\n⚠️  Could not save checkpoint: {e}")
                return
            if task_loss < self.best_task_loss:
                self.best_task_loss = task_loss
                try:
                    shutil.copy2(self.weights_path, self.best_path)
                except Exception as e:
                    print(f"\n⚠️  Could not update best checkpoint: {e}")

    class LRStateSaver(Callback):
        def __init__(self, reduce_lr_cb, state_path):
            super().__init__()
            self.reduce_lr = reduce_lr_cb
            self.state_path = state_path

        def on_epoch_end(self, epoch, logs=None):
            try:
                with open(self.state_path, "w") as f:
                    json.dump({"lr": float(self.reduce_lr._get_lr()),
                               "best": float(self.reduce_lr.best),
                               "wait": int(self.reduce_lr.wait)}, f)
            except Exception:
                pass

    class EarlyStoppingStateSaver(Callback):
        def __init__(self, es_cb, state_path):
            super().__init__()
            self.es = es_cb
            self.state_path = state_path

        def on_epoch_end(self, epoch, logs=None):
            try:
                with open(self.state_path, "w") as f:
                    json.dump({"best": float(self.es.best), "wait": int(self.es.wait)}, f)
            except Exception:
                pass

    # =========================================================================
    # Model build + training
    # =========================================================================
    print(f"\n--- Building Model 5d: {RUN_NAME} ---\n")

    n_gpus = len(tf.config.list_physical_devices("GPU"))
    if n_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy across {n_gpus} GPUs")
    elif n_gpus == 1 and not force_cpu:
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        print("Using OneDeviceStrategy: /GPU:0 (explicit Metal placement)")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        print("Using OneDeviceStrategy: /CPU:0")

    def _reg():
        return tf.keras.regularizers.l2(L2_REG)

    with strategy.scope():
        # Track B Run 2's exact stack shape (confirmed via checkpoint H5 inspection — see
        # evaluate_run2_int8.py's header comment): a pure sequential chain, no wide/deep
        # branching, no concat. Every layer is Dense(use_bias=False) -> BatchNormalization ->
        # relu (plain, unbounded — NOT relu6). Unchanged from Run 1 — this run only changes the
        # feature set (union of Run 2's 23 + temp_diff_vs_5hr/6hr = 25, see file header).
        input_layer = tf.keras.layers.Input(shape=(n_features,), name="input")
        x = tf.keras.layers.Dense(HIDDEN_1, use_bias=False,
                                  name="dense0", kernel_regularizer=_reg())(input_layer)
        x = tf.keras.layers.BatchNormalization(name="bn0")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dense(HIDDEN_2, use_bias=False,
                                  name="dense1", kernel_regularizer=_reg())(x)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dense(HIDDEN_3, use_bias=False,
                                  name="dense2", kernel_regularizer=_reg())(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dense(HIDDEN_4, use_bias=False,
                                  name="dense3", kernel_regularizer=_reg())(x)
        x = tf.keras.layers.BatchNormalization(name="bn3")(x)
        x = tf.keras.layers.Activation("relu")(x)

        out_1 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_1hr")(x)
        out_2 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_2hr")(x)
        out_3 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_3hr")(x)

        model = tf.keras.Model(inputs=input_layer, outputs=[out_1, out_2, out_3],
                               name=f"model_5f_{RUN_NAME}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics={"diff_1hr": "mae", "diff_2hr": "mae", "diff_3hr": "mae"},
        )

    model.summary()
    print(f"\nArchitecture: input({n_features}) → "
          f"[dense({HIDDEN_1})→BN→relu] → [dense({HIDDEN_2})→BN→relu] → "
          f"[dense({HIDDEN_3})→BN→relu] → [dense({HIDDEN_4})→BN→relu] → 3 output heads "
          f"(Track B Run 2's architecture and features, + temp_diff_vs_5hr/6hr)")
    print(f"n_features: {n_features}, L2_REG: {L2_REG}")

    # -------------------------------------------------------------------------
    # Checkpoint loading (resume support)
    # -------------------------------------------------------------------------
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    es_state_path = os.path.join(checkpoint_dir, "early_stopping_state.json")
    lr_state_path = os.path.join(checkpoint_dir, "lr_state.json")
    initial_epoch = 0

    _latest_weights = os.path.join(checkpoint_dir, "model_latest.weights.h5")
    _latest_meta = os.path.join(checkpoint_dir, "model_latest_epoch.json")

    if os.path.exists(_latest_weights) and os.path.exists(_latest_meta):
        try:
            with open(_latest_meta) as f:
                _epoch = int(json.load(f).get("epoch", 0))
            model.load_weights(_latest_weights)
            initial_epoch = _epoch
            print(f"✅ Resumed from epoch {initial_epoch}")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e} — starting fresh")
            initial_epoch = 0

    # -------------------------------------------------------------------------
    # Pre-training validation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔍 PRE-TRAINING VALIDATION")
    print("=" * 60)
    errors = []

    print("\n1️⃣  Forward pass...")
    try:
        _probe = model.predict(X_val_small[:4], verbose=0)
        print(f"   ✅ Output shapes: {[o.shape for o in _probe]}")
    except Exception as e:
        errors.append(f"Forward pass: {str(e)[:200]}")

    print("\n2️⃣  Batch check...")
    try:
        xb, yb = next(iter(train_ds))
        assert xb.shape[1] == n_features, f"feat mismatch: {xb.shape[1]} != {n_features}"
        x_nan = int(tf.reduce_sum(tf.cast(tf.math.is_nan(xb), tf.int32)).numpy())
        if x_nan:
            errors.append(f"Batch has {x_nan} NaN values")
        print(f"   ✅ Batch shape: x={xb.shape}, y={[t.shape for t in yb]}")
    except Exception as e:
        errors.append(f"Dataset: {str(e)[:200]}")

    print("\n3️⃣  Warmup step...")
    try:
        t0 = time.time()
        wb, wy = next(iter(train_ds.take(1)))
        wloss = model.train_on_batch(wb, wy)
        ws = time.time() - t0
        wloss_val = float(wloss[0] if isinstance(wloss, (list, tuple)) else wloss)
        est = ws * train_steps / 60.0
        print(f"   ✅ Warmup: {ws:.1f}s  loss={wloss_val:.6f}  "
              f"est epoch ~{est:.1f} min ({train_steps} steps)")
        if np.isnan(wloss_val) or np.isinf(wloss_val):
            errors.append(f"Warmup loss is {wloss_val}")
    except Exception as e:
        errors.append(f"Warmup: {str(e)[:200]}")

    print("\n" + "=" * 60)
    if errors:
        print("❌ VALIDATION FAILED")
        for e in errors:
            print(f"  • {e}")
        raise RuntimeError("; ".join(errors[:3]))
    print("✅ PRE-TRAINING VALIDATION PASSED")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Build callbacks
    # -------------------------------------------------------------------------
    early_stopping = EarlyStopping(monitor="val_task_loss", patience=EARLY_STOP_PATIENCE,
                                   restore_best_weights=True, mode="min")
    reduce_lr = ReduceLRCallback(initial_lr=INITIAL_LR, monitor="val_task_loss",
                                 factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE,
                                 min_lr=REDUCE_LR_MIN, min_delta=1e-5)

    callbacks = [
        TaskLossLogger(),
        NaNLossTerminator(),
        reduce_lr,
        early_stopping,
        LatestEpochSaver(checkpoint_dir),
        LRStateSaver(reduce_lr, lr_state_path),
        EarlyStoppingStateSaver(early_stopping, es_state_path),
        EpochProgressCallback(log_every=50, total_steps=train_steps, max_epochs=MAX_EPOCHS),
        TrainingWatchdog(timeout_minutes=10, check_interval_seconds=30,
                         train_ds=train_ds, total_batches_hint=train_steps),
    ]

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    if initial_epoch >= MAX_EPOCHS:
        print(f"\n✅ Already complete (epoch {initial_epoch} >= {MAX_EPOCHS})")
        history = None
    else:
        print(f"\n🚀 Training from epoch {initial_epoch + 1} to {MAX_EPOCHS}")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MAX_EPOCHS,
            initial_epoch=initial_epoch,
            steps_per_epoch=train_steps,
            callbacks=callbacks,
            verbose=0,
        )

    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION")
    print("=" * 60)
    eval_results = model.evaluate(val_ds, verbose=0)
    # eval_results layout: [0] total_loss  [1] d1_mse  [2] d2_mse  [3] d3_mse
    #                      [4] d1_mae  [5] d2_mae  [6] d3_mae
    val_loss = float(eval_results[0])
    scale = (y_max - y_min) / 2.0
    if len(eval_results) >= 7:
        val_mae = float(np.mean(eval_results[4:7]))
        diff_1hr_mae_c = float(eval_results[4]) * scale
        diff_2hr_mae_c = float(eval_results[5]) * scale
        diff_3hr_mae_c = float(eval_results[6]) * scale
    else:
        val_mae = float(np.mean([np.sqrt(eval_results[i]) for i in range(1, 4)]))
        diff_1hr_mae_c = np.sqrt(float(eval_results[1])) * scale
        diff_2hr_mae_c = np.sqrt(float(eval_results[2])) * scale
        diff_3hr_mae_c = np.sqrt(float(eval_results[3])) * scale
        print(f"  ℹ️  eval_results has {len(eval_results)} values — reporting RMSE instead of MAE")

    print(f"\nValidation MAE (°C):")
    print(f"  diff_1hr: {diff_1hr_mae_c:.3f}°C")
    print(f"  diff_2hr: {diff_2hr_mae_c:.3f}°C")
    print(f"  diff_3hr: {diff_3hr_mae_c:.3f}°C")
    print(f"\nval_loss (includes L2): {val_loss:.6f}")

    if val_loss >= 0.001:
        print(f"⚠️  val_loss={val_loss:.6f} higher than expected (>0.001) — check convergence")

    best_epoch = (initial_epoch + int(np.argmin(history.history["val_task_loss"]) + 1)
                  ) if history else initial_epoch

    # -------------------------------------------------------------------------
    # Permutation feature importance
    # -------------------------------------------------------------------------
    print("\n📈 Permutation Feature Importance (val_loss increase):")
    X_perm = X_val_small  # (N, n_features)
    y_perm = (y_val_small[:, 0], y_val_small[:, 1], y_val_small[:, 2])
    baseline_ds = tf.data.Dataset.from_tensor_slices(
        (X_perm, y_perm)).batch(VAL_BATCH_SIZE)
    baseline_loss = float(model.evaluate(baseline_ds, verbose=0)[0])

    feature_importance = {}
    for fi, feat in enumerate(features):
        Xp = X_perm.copy()
        col = Xp[:, fi].copy()
        np.random.shuffle(col)
        Xp[:, fi] = col
        perm_ds = tf.data.Dataset.from_tensor_slices(
            (Xp, y_perm)).batch(VAL_BATCH_SIZE)
        perm_loss = float(model.evaluate(perm_ds, verbose=0)[0])
        feature_importance[feat] = perm_loss - baseline_loss

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance:
        print(f"  {feat}: {imp:.4f}")

    # -------------------------------------------------------------------------
    # TFLite export — FP32 (validation) then INT8 (Coral deployment)
    # mixed_float16 stores hidden layer kernels as fp16. TFLite's native converter only
    # supports fp32 MatMul → FULLY_CONNECTED; it cannot lower fp16 MatMul and raises
    # ERROR_NEEDS_FLEX_OPS. Fix: rebuild the model under the float32 policy, copy weights
    # (get_weights() returns fp32 master copies even when the compute policy is fp16).
    # -------------------------------------------------------------------------
    if on_metal:
        print("\n🔧 Building fp32 export model (casting mixed-precision weights)...")
        _orig_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("float32")
        _e_inp = tf.keras.layers.Input(shape=(n_features,), name="input")
        _e = tf.keras.layers.Dense(HIDDEN_1, use_bias=False, name="dense0")(_e_inp)
        _e = tf.keras.layers.BatchNormalization(name="bn0")(_e)
        _e = tf.keras.layers.Activation("relu")(_e)
        _e = tf.keras.layers.Dense(HIDDEN_2, use_bias=False, name="dense1")(_e)
        _e = tf.keras.layers.BatchNormalization(name="bn1")(_e)
        _e = tf.keras.layers.Activation("relu")(_e)
        _e = tf.keras.layers.Dense(HIDDEN_3, use_bias=False, name="dense2")(_e)
        _e = tf.keras.layers.BatchNormalization(name="bn2")(_e)
        _e = tf.keras.layers.Activation("relu")(_e)
        _e = tf.keras.layers.Dense(HIDDEN_4, use_bias=False, name="dense3")(_e)
        _e = tf.keras.layers.BatchNormalization(name="bn3")(_e)
        _e = tf.keras.layers.Activation("relu")(_e)
        _e_o1 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_1hr")(_e)
        _e_o2 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_2hr")(_e)
        _e_o3 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_3hr")(_e)
        export_model = tf.keras.Model(inputs=_e_inp, outputs=[_e_o1, _e_o2, _e_o3])
        # Name-based transfer, not positional model.get_weights()/set_weights(list) — this is a
        # linear chain so there's no branching-order ambiguity like Track B's wide/deep case, but
        # name-based costs nothing extra and removes the risk entirely (see MODEL_5C_TRACK_B and
        # MODEL_5E logs for two prior incidents of the positional version silently pairing wrong
        # shapes when the two models' internal layer graphs weren't identical).
        _model_layers_by_name = {l.name: l for l in model.layers}
        for _l in export_model.layers:
            if _l.name in _model_layers_by_name and _l.get_weights():
                _l.set_weights([w.astype(np.float32)
                               for w in _model_layers_by_name[_l.name].get_weights()])
        export_model.compile(optimizer="sgd", loss="mse")
        tf.keras.mixed_precision.set_global_policy(_orig_policy)
        print("   ✅ fp32 export model ready")
    else:
        export_model = model

    print("\n🔍 FP32 export model sanity check...")
    _sanity = export_model.evaluate(val_ds.take(5), verbose=0)
    _sanity_loss = float(_sanity[0])
    # Compare against the trained model's own val_loss (relative), not a fixed absolute threshold.
    # A poorly-converged-but-correctly-copied model reproduces its own (bad) val_loss; a real
    # weight-transfer bug (wrong shapes/order) produces a much larger divergence than sampling
    # noise from the 5-batch subset can explain.
    _sanity_ratio = _sanity_loss / max(val_loss, 1e-9)
    if _sanity_ratio > 1.5:
        raise RuntimeError(
            f"FP32 export model sanity check FAILED: val_loss={_sanity_loss:.6f} vs trained "
            f"model's val_loss={val_loss:.6f} (ratio {_sanity_ratio:.2f}x). Weight transfer to the "
            f"export model produced invalid results. Aborting before INT8 export.")
    print(f"   ✅ FP32 export sanity check passed: val_loss={_sanity_loss:.6f} "
          f"(trained model val_loss={val_loss:.6f}, ratio {_sanity_ratio:.2f}x)")

    run_model = tf.function(export_model)
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, n_features], tf.float32, name="input"))

    tflite_fp32_path = os.path.join(RESULTS_DIR, f"model_5f_{RUN_NAME}_fp32.tflite")
    tflite_int8_path = os.path.join(RESULTS_DIR, f"model_5f_{RUN_NAME}_int8.tflite")
    tflite_fp32_kb = 0.0
    tflite_int8_kb = 0.0

    print("\n🔧 Exporting FP32 TFLite model...")
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        fp32_model = converter.convert()
        with open(tflite_fp32_path, "wb") as f:
            f.write(fp32_model)
        tflite_fp32_kb = os.path.getsize(tflite_fp32_path) / 1024
        print(f"   ✅ FP32 TFLite: {tflite_fp32_kb:.1f} KB → {tflite_fp32_path}")
    except Exception as e:
        print(f"   ⚠️  FP32 export failed: {e}")

    print("🔧 Exporting INT8 TFLite model (Coral Edge TPU)...")
    try:
        def representative_data_gen():
            n = X_val.shape[0]
            idxs = np.random.choice(n, size=min(2000, n), replace=False)
            for idx in idxs:
                yield [X_val[idx:idx + 1].astype(np.float32)]

        converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_data_gen
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.int8
        converter_int8.inference_output_type = tf.int8
        int8_model = converter_int8.convert()
        with open(tflite_int8_path, "wb") as f:
            f.write(int8_model)
        tflite_int8_kb = os.path.getsize(tflite_int8_path) / 1024
        print(f"   ✅ INT8 TFLite: {tflite_int8_kb:.1f} KB → {tflite_int8_path}")
    except Exception as e:
        print(f"   ⚠️  INT8 export failed: {e}")

    # -------------------------------------------------------------------------
    # INT8 model validation
    # -------------------------------------------------------------------------
    if os.path.exists(tflite_int8_path):
        print("\n🔍 Validating INT8 model...")
        try:
            interp = tf.lite.Interpreter(model_path=tflite_int8_path)
            interp.allocate_tensors()
            in_det = interp.get_input_details()
            out_det = interp.get_output_details()
            in_scale, in_zp = in_det[0]["quantization"]

            n_val = min(500, X_val_small.shape[0])
            preds_int8 = [[] for _ in range(3)]
            for i in range(n_val):
                sample = X_val_small[i:i+1]
                q_in = np.round(sample / in_scale + in_zp).astype(in_det[0]["dtype"])
                interp.set_tensor(in_det[0]["index"], q_in)
                interp.invoke()
                for j in range(3):
                    out_s, out_zp = out_det[j]["quantization"]
                    raw = interp.get_tensor(out_det[j]["index"])
                    preds_int8[j].append(float(np.squeeze(raw - out_zp) * out_s))

            print(f"Validation MAE — INT8 (°C), n={n_val}:")
            for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
                pred_c = (np.array(preds_int8[j]) + 1) * 0.5 * (y_max - y_min) + y_min
                true_c = (y_val_small[:n_val, j] + 1) * 0.5 * (y_max - y_min) + y_min
                mae = float(np.mean(np.abs(pred_c - true_c)))
                print(f"  {name}: {mae:.3f}°C")
        except Exception as e:
            print(f"   ⚠️  INT8 validation failed: {e}")

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    results = {
        "name": RUN_NAME,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "diff_1hr_mae_c": diff_1hr_mae_c,
        "diff_2hr_mae_c": diff_2hr_mae_c,
        "diff_3hr_mae_c": diff_3hr_mae_c,
        "best_epoch": best_epoch,
        "tflite_fp32_kb": tflite_fp32_kb,
        "tflite_int8_kb": tflite_int8_kb,
        "n_features": n_features,
        "architecture": f"input({n_features})->[dense({HIDDEN_1})->bn->relu]->[dense({HIDDEN_2})->bn->relu]->[dense({HIDDEN_3})->bn->relu]->[dense({HIDDEN_4})->bn->relu]->3heads;flat_seq_len_1;track_b_run2_arch_and_features_plus_5hr6hr",
        "l2_reg": L2_REG,
        "feature_importance_permutation": sorted_importance,
        "features": features,
        "hyperparams": {
            "architecture": "flat_seq_len_1_sequential_batchnorm_relu_track_b_run2_shape",
            "hidden_1": HIDDEN_1,
            "hidden_2": HIDDEN_2,
            "hidden_3": HIDDEN_3,
            "hidden_4": HIDDEN_4,
            "l2_reg": L2_REG,
            "initial_lr": INITIAL_LR,
            "max_epochs": MAX_EPOCHS,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
            "reduce_lr_factor": REDUCE_LR_FACTOR,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "train_batch_size": TRAIN_BATCH_SIZE,
        },
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{RUN_NAME}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved → {results_path}")

    print(f"\nFinal Metrics [{RUN_NAME}]:")
    print(f"  val_loss (includes L2): {val_loss:.6f}")
    print(f"  val_mae (normalized):   {val_mae:.6f}")
    print(f"  diff_1hr MAE:           {diff_1hr_mae_c:.3f}°C")
    print(f"  diff_2hr MAE:           {diff_2hr_mae_c:.3f}°C")
    print(f"  diff_3hr MAE:           {diff_3hr_mae_c:.3f}°C")
    print(f"  Best epoch:             {best_epoch}")
    print(f"  FP32 TFLite:            {tflite_fp32_kb:.1f} KB")
    print(f"  INT8 TFLite:            {tflite_int8_kb:.1f} KB")
    print(f"\nBaseline reference:")
    print(f"  Model 5a deployed (INT8):        val_loss=0.000682 | 30d StdDev 0.988°C")
    print(f"  Model 5b Exp37 INT8 deployed:    30d StdDev 0.930°C")
    print(f"  Track B Run 11/13 INT8 (best):   3hr MAE 0.898/0.907°C")


if __name__ == "__main__":
    main()
