import os

# QAT fine-tuning flag — declared here, before the TensorFlow import, because tfmot's default
# quantization scheme cannot recognize Keras 3 Functional models at all ("to_quantize can only
# be a Sequential or Functional model" even though it is one). TF_USE_LEGACY_KERAS must be set
# before `import tensorflow` for tf.keras to resolve to the legacy tf_keras engine tfmot
# expects; setting it after import has no effect. This mirrors QAT_FINE_TUNE in the config
# block below (kept in sync manually — that block is the source of truth for QAT_LR/EPOCHS/etc,
# this line only exists early enough to gate the env var).
QAT_FINE_TUNE = False
if QAT_FINE_TUNE:
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

# Model 5c Track B: TFT-Informed Dense Model for Coral Edge TPU
#
# Encodes TFT-discovered lag anchors (t-60, t-120, t-180) as explicit scalar features
# in a lean Dense model. Architecture is Model-5a-style: flat feature vector → Dense
# layers → 3 output heads. Target: INT8 on Coral TPU, beating Model 5a's val_loss=0.000682.
#
# Key differences from Track A (TFT):
#   - No sequence window: each sample is a single flat feature vector (SEQ_LEN=1)
#   - Lag features are pre-computed scalars, not learned by attention
#   - Full INT8 quantization → Coral TPU deployment
#   - ~10× smaller model, ~100× faster inference
#
# TFT discoveries encoded here (from 6 Track A runs):
#   - Lag anchors: t-60 (Head 0 specialist), t-120 (Head 6 specialist), t-180 (dominant)
#   - Drop: wind_lull, rain_accumulated (floor in all 6 runs, VSN < 0.002)
#   - Raw temperature: perm importance last (0.013) — slopes fully substitute; kept for run 1
#   - New: pressure_lag120/180 for Zambretti 3-hour pressure tendency signal

KAGGLE_MODE = False
KAGGLE_DATASET = "datasets/dacarson/weatherml-training-data"
KAGGLE_CHECKPOINT_DATASET = ""
KAGGLE_CHECKPOINT_SUBDIR = ""

# Diagnostic audit (not a training run): loads a checkpoint read-only, compares each
# intermediate tensor's REAL activation range (from a forward pass on real validation data)
# against what its already-exported .tflite calibrated as that tensor's INT8 range, then exits
# before touching training/export. Used to hunt for other forced/mismatched-range precision
# losses like the wide/deep_out concat one found and fixed in Run 20. Set RUN_NAME/
# SOURCE_CHECKPOINT below to point at whichever run's checkpoint+tflite pair to audit.
DIAGNOSTIC_AUDIT = False

RUN_NAME = "dense_b_run22"
RESULTS_DIR = f"/kaggle/working" if KAGGLE_MODE else f"./results_5c_trackb_{RUN_NAME}"
AUDIT_SOURCE_RUN_DIR = "./results_5c_trackb_dense_b_run22"

# Run 22: re-implement Run 21's deep_out ceiling tightening WITHOUT introducing new unfused
# ops. Run 21's post-hoc audit found ReLU(max_value=0.6) worked exactly as intended for the
# main activation path (99.2% utilization) but its clip_by_value decomposition created a
# SEPARATE, badly-calibrated tensor (28% utilization, calibrated max 4x the real max) — net
# result was WORSE INT8 on 2hr/3hr despite fixing the originally-diagnosed problem. Verified
# directly (standalone conversion test) that a Lambda(tf.clip_by_value(...)) has the exact same
# duplicate-tensor issue — it's not specific to the ReLU layer, clip_by_value itself decomposes
# this way under TFLite's MLIR lowering.
#
# Fix: achieve the same effective tighter ceiling using only well-fusing ops. Pre-scale UP
# before the EXISTING relu6 (native TFLite op, unlike generic clip_by_value) so relu6's fixed
# ceiling of 6 lands close to where the real distribution's tail actually sits, then rescale
# back down afterward to match wide's range for the concat (as Run 20 already did). Verified
# directly: Dense(linear) -> Rescaling(prescale) -> Activation("relu6") -> Rescaling(downscale)
# converts to exactly 2 clean tensors — TFLite's converter even algebraically folds the
# Rescaling multiply INTO the preceding MatMul, fusing MatMul+prescale+Relu6 into a single
# quantized tensor, plus one clean tensor for the final downscale. No duplicate/mismatched
# tensors, unlike both Run 21 approaches tested.
SKIP_TRAINING = True if DIAGNOSTIC_AUDIT else False
SOURCE_CHECKPOINT = (f"{AUDIT_SOURCE_RUN_DIR}/checkpoints/best_model.weights.h5" if DIAGNOSTIC_AUDIT
                     else "./results_5c_trackb_dense_b_run18/checkpoints/best_model.weights.h5")

# Warm start: initialize weights from a prior run's checkpoint instead of random init, while
# keeping training otherwise fresh (optimizer state, LR schedule, early-stopping all reset).
# Only valid when architecture/feature shapes are unchanged from the source run. Applies only
# on this run's own first invocation (no checkpoint of its own yet) — a genuine resume of THIS
# run (model_latest.weights.h5 in this run's own RESULTS_DIR) always takes priority.
# Run 22: warm start from Run 20, NOT Run 21 — Run 21's deep_out weights were trained under a
# constraint (ReLU max_value=0.6) we're now abandoning; Run 20's plain-relu6 weights are the
# right starting point for this new pre/post-rescale design, and _ws_model below already
# matches Run 20's exact architecture unmodified (reused as-is, no changes needed this time).
WARM_START = True
WARM_START_CHECKPOINT = "./results_5c_trackb_dense_b_run20/checkpoints/best_model.weights.h5"

# Pre-scale applied to deep_out's linear output BEFORE relu6 (Run 22). Chosen so relu6's fixed
# ceiling of 6 corresponds to the same effective 0.6 target Run 21 used directly: 6 / 0.6 = 10.
DEEP_OUT_PRESCALE = 10.0

# Rescale applied to deep_out's output AFTER relu6, before the concat (Run 20's original
# purpose: match wide's calibrated range, ~1.6-1.8 per the Run 20/21 audits). Since relu6's
# ceiling is now effectively 6 (post-prescale), target ~1.6/6 ≈ 0.27 instead of Run 20/21's
# >1 multiplier (which rescaled directly from the old ~0.5-max linear space).
DEEP_OUT_RESCALE = 0.2735

# QAT fine-tuning — active when SKIP_TRAINING=True + QAT_FINE_TUNE=True.
# QAT_FINE_TUNE itself is declared at the top of the file (before `import tensorflow`) since
# TF_USE_LEGACY_KERAS must be set before that import to take effect — see comment there.
QAT_LR = 1e-6       # safe without warmup: gradient amplification = lr/eps = 10× (vs 1000× at Run 12's 1e-4)
QAT_EPOCHS = 50
QAT_EARLY_STOP_PATIENCE = 10

# Architecture — Run 16: same as Run 14/15 (fused Dense+ReLU6), fresh training with 2 new
#   features (temp_diff_vs_5hr, temp_diff_vs_6hr — see MODEL_5C_TRACK_B_EXPERIMENT_LOG.md).
#   Input:      (180, 13) — 3hr temporal window, 13 features
#   AvgPool:    AveragePooling1D(pool_size=6) → (30, 13) → Reshape(390)
#   Bottleneck: Dense(64, activation="relu6")
#   Wide path:  Dense(16, activation="relu6")
#   Deep path:  Dense(128, activation="relu6") → Dense(64, activation="relu6")
#               → Dense(32, activation="relu6")
#   Merge:      Concat(16+32=48) → 3 output heads
#   No BN, No residual, No interaction path — same constraints as Runs 11–14
L2_REG = 1e-6

# Training
MAX_EPOCHS = 600
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
    import sys
    import json
    import glob
    import re
    import shutil
    import time
    import copy
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
            gpu_mb = os.environ.get("GPU_MEMORY_MB")
            if gpu_mb:
                for dev in physical_devices:
                    tf.config.experimental.set_virtual_device_configuration(
                        dev, [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=int(gpu_mb))])
            else:
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
        tf.config.threading.set_intra_op_parallelism_threads(
            int(os.environ.get("TF_NUM_INTRAOP_THREADS", max(4, cores - 4))))
        tf.config.threading.set_inter_op_parallelism_threads(
            int(os.environ.get("TF_NUM_INTEROP_THREADS", 2)))

    tf.config.set_soft_device_placement(True)

    # on_metal is a pure hardware-detection flag (physically running on Metal GPU) — it drives
    # the XLA JIT decision below and must stay accurate regardless of QAT, since Metal's
    # PluggableDevice backend can't do XLA JIT compilation no matter what precision is used.
    on_metal = (bool(physical_devices) and not force_cpu and not KAGGLE_MODE)
    if KAGGLE_MODE or force_cpu:
        tf.config.optimizer.set_jit(True)
        print("ℹ️  XLA JIT enabled")
    elif on_metal:
        tf.config.optimizer.set_jit(False)
        print("ℹ️  Metal GPU detected — XLA JIT disabled (Metal scheduler incompatibility)")
    else:
        tf.config.optimizer.set_jit(True)
        print("ℹ️  XLA JIT enabled (CPU)")

    # use_mixed_precision is the separate "should this run use fp16 compute" decision — distinct
    # from on_metal because QAT requires float32 (tfmot fake-quant nodes are incompatible with
    # fp16 compute graphs) even though the hardware is still physically Metal.
    use_mixed_precision = on_metal
    if QAT_FINE_TUNE and use_mixed_precision:
        use_mixed_precision = False
        print("ℹ️  QAT_FINE_TUNE=True: mixed precision disabled (float32 required for tfmot)")

    # Mixed precision: fp16 compute on Metal GPU.
    # Safe for Dense models (no attention/softmax numerical issues that affect TFT).
    # Output layers use dtype='float32' explicitly so targets stay FP32.
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("ℹ️  Mixed precision enabled: fp16 compute / fp32 master weights (Metal GPU)")

    # -------------------------------------------------------------------------
    # Checkpoint restore (Kaggle: copy from published dataset to working dir)
    # -------------------------------------------------------------------------
    if KAGGLE_MODE and KAGGLE_CHECKPOINT_DATASET:
        _root = f"/kaggle/input/{KAGGLE_CHECKPOINT_DATASET}"
        _src = os.path.join(_root, KAGGLE_CHECKPOINT_SUBDIR, "checkpoints") \
               if KAGGLE_CHECKPOINT_SUBDIR else os.path.join(_root, "checkpoints")
        if os.path.exists(_src):
            _dst = os.path.join(RESULTS_DIR, "checkpoints")
            os.makedirs(_dst, exist_ok=True)
            for _f in os.listdir(_src):
                shutil.copy(os.path.join(_src, _f), _dst)
            print(f"✅ Checkpoints restored from {_src}")
        else:
            print(f"⚠️  Checkpoint path not found: {_src}")

    # -------------------------------------------------------------------------
    # Data loading helpers (identical to Track A)
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
        # Raw station data contains rare sensor-glitch rows (brief dropout/self-heating,
        # not real weather) where temperature jumps implausibly fast and reverts a few
        # minutes later — e.g. 2023-01-23 09:18 UTC: 11.1°C -> -7.5°C -> 9.9°C with
        # relative_humidity simultaneously collapsing to 0%. These poisoned the
        # temp_diff_Nhr targets (e.g. -31.5°C global min) since diffs are computed via
        # merge_asof against raw 'temperature'. Null any reading that deviates from its
        # local (time-centered) median by more than threshold_c; downstream dropna
        # removes the row entirely. Found via `git log` discussion 2026-07-19.
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
        # temp_diff_vs_5hr / temp_diff_vs_6hr: from the Track A Deep Run (SEQ_LEN=360) finding
        # of a genuine non-boundary attention anchor at ~5 hours (t-295 to t-301). Track B's
        # SEQ_LEN=180 window can't see that far back, so this is carried in as a scalar feature.
        if all(c in df.columns for c in ["temp_diff_vs_5hr", "temp_diff_vs_6hr"]):
            return df
        if "temperature" not in df.columns:
            raise ValueError(f"{label}: missing 'temperature'")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{label}: need DatetimeIndex for lag construction")
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
        print(f"\n❓ Missing 5hr/6hr lag counts for {label}:")
        print(base[["temp_diff_vs_5hr", "temp_diff_vs_6hr"]].isna().sum())
        return base.drop(columns=["row_id", "temp_lag_300", "temp_lag_360"]).set_index("time")


    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    if KAGGLE_MODE:
        data_dir = f"/kaggle/input/{KAGGLE_DATASET}"
    else:
        # Robust regardless of working directory — CSVs live in workspace/ (one level up)
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    train_df = pd.read_csv(f"{data_dir}/train_data_sf.csv")
    val_df = pd.read_csv(f"{data_dir}/val_data_sf.csv")

    # The raw CSVs ship with pre-baked temp_t+1hr/2hr/3hr columns from an earlier export
    # pipeline. _add_future_targets() short-circuits (returns df unchanged) whenever those
    # columns already exist, so keeping them means the sanity filter below (and the
    # gap-aware merge_asof reconstruction in _add_future_targets/_invalidate_targets_
    # crossing_gaps) never actually run — targets stay locked to whatever that external
    # pipeline computed, including any of its own bad readings. Drop them so targets are
    # always freshly derived from (now sanity-filtered) 'temperature'.
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
    # Cyclical encodings (identical to Track A)
    # -------------------------------------------------------------------------
    for df in (train_df, val_df):
        df["time_of_day_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_sin2"] = np.sin(4 * np.pi * df["time_of_day"] / 24.0)
        df["time_of_day_cos2"] = np.cos(4 * np.pi * df["time_of_day"] / 24.0)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        if "wind_direction" in df.columns:
            df["wind_direction_sin"] = np.sin(2 * np.pi * df["wind_direction"] / 360.0)
            df["wind_direction_cos"] = np.cos(2 * np.pi * df["wind_direction"] / 360.0)

    # -------------------------------------------------------------------------
    # Rolling slope features (identical to Track A)
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

    # Only drop rows where the actual model inputs or targets contain NaN.
    # Raw CSV columns like 'humidity' may be almost entirely NaN and must not
    # cause valid training rows to be excluded.
    _dropna_cols = [
        "time_of_day_sin", "time_of_day_cos", "time_of_day_sin2", "time_of_day_cos2",
        "day_of_year_sin", "day_of_year_cos",
        "temperature",           # feature (restored Run 8) + required for target diff computation
        # relative_humidity dropped — confirmed harmful in Runs 2, 4, 5, 6, 7 (5 consecutive runs)
        # humidity_slope_30 dropped (Run 11): negative perm importance in Runs 7, 8, 9, 10 (4 consecutive)
        "pressure_slope_60",     # station_pressure dropped (Run 10): negative perm in 5 consecutive runs
        "solar_radiation", "illuminance", "uv",
        # solar_slope_30 removed from features (Run 9); also redundant in dropna since
        # pressure_slope_60's 60-row window already excludes all rows solar_slope_30 would.
        "temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr",
        # temp_slope_15/30/60 not features; first 60 rows still excluded by pressure_slope_60
        "temp_diff_vs_5hr", "temp_diff_vs_6hr",  # Run 16: drop rows without 5hr/6hr history
    ]
    train_df.dropna(subset=_dropna_cols, inplace=True)
    val_df.dropna(subset=_dropna_cols, inplace=True)
    print(f"\nAfter dropna: train={len(train_df):,} rows, val={len(val_df):,} rows")

    # -------------------------------------------------------------------------
    # Feature list — Track B Run 16
    #
    # Run 16: 13 features. Added temp_diff_vs_5hr/6hr from Track A Deep Run non-boundary
    #         attention anchor (~5hr) — see MODEL_5C_TRACK_B_EXPERIMENT_LOG.md.
    # Run 11: 11 features. Removed BatchNorm (INT8 fix — BN γ unconstrained by L2).
    #         Dropped humidity_slope_30 (negative perm in Runs 7, 8, 9, 10 — 4 consecutive).
    # Run 10: 12 features. Removed residual Add from Deep path (INT8 fix).
    #         Dropped station_pressure (negative perm importance in 5 consecutive runs).
    # Run 9:  13 features. Wide path ReLU6 added; dropped solar_slope_30.
    # AveragePooling1D(pool_size=6) compresses (180, n_features) → (30, n_features) → Reshape.
    #
    # Run 10 perm importance (val_loss increase):
    #   time_of_day_cos (+0.0124), sin (+0.0104) — temporal dominant
    #   illuminance (+0.0097), solar_radiation (+0.0088), cos2 (+0.0064), uv (+0.0062), sin2 (+0.0042)
    #   temperature (+0.0006), day_of_year_sin (+0.0001)
    #   pressure_slope_60 (−0.0000), day_of_year_cos (−0.0000), humidity_slope_30 (−0.0002)
    # -------------------------------------------------------------------------
    temporal_features = [
        "time_of_day_sin", "time_of_day_cos",
        "time_of_day_sin2", "time_of_day_cos2",
        "day_of_year_sin", "day_of_year_cos",
    ]
    temperature_features = [
        "temperature",           # essential — Run 7 proved joint necessity of temperature signal
        "temp_diff_vs_5hr",      # Run 16: Track A Deep Run non-boundary attention anchor (~5hr)
        "temp_diff_vs_6hr",      # Run 16: Track A Deep Run window-boundary anchor (~6hr)
    ]
    humidity_features = [
        # relative_humidity: dropped — confirmed harmful in Runs 2, 4, 5, 6, 7 (5 consecutive runs)
        # humidity_slope_30: dropped (Run 11) — negative perm in Runs 7, 8, 9, 10 (4 consecutive)
    ]
    pressure_features = [
        # station_pressure dropped (Run 10): negative perm importance in Runs 5, 6, 7, 8, 9 (5 consecutive)
        "pressure_slope_60",     # VSN #14 — Zambretti tendency; remained borderline positive across runs
    ]
    solar_features = [
        "solar_radiation",       # VSN #15
        # solar_slope_30 dropped: negative perm importance in Runs 6/7/8 (−0.0001/−0.0002/−0.0003)
        "illuminance",           # VSN #16
        "uv",                    # VSN #20
    ]

    features = (temporal_features + temperature_features + humidity_features +
                pressure_features + solar_features)
    targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]

    print(f"\nFeature set: {len(features)} features")
    for f in features:
        print(f"  {f}")

    # -------------------------------------------------------------------------
    # Per-feature min/max scaling with domain bounds and ±5% padding
    # -------------------------------------------------------------------------
    domain_bounds = {
        "temperature":        (-10, 55),
        "temp_slope_15":      (None, None),
        "temp_slope_30":      (None, None),
        "temp_slope_60":      (None, None),
        "solar_slope_30":     (None, None),
        # Run 17: tightened from (None, None) (data-derived min/max ± 5% pad, ~-19/+23) to fixed
        # ~1st/99th-percentile bounds — INT8 fix (Option 1). Run 16 measured these features'
        # actual distribution: std ~3.6-4.0°C, IQR ~[-2, 2]°C, but min/max span ~40°C — min-max
        # scaling wasted most of INT8's 256 levels on rarely-visited tails, and temp_diff_vs_5hr
        # is the model's single most important feature (perm importance 0.0456), so quantization
        # noise there had an outsized effect (INT8 MAE 7-11x worse than FP32).
        # Run 17 correction: initially left the ~2% of out-of-bounds values unclipped, assuming
        # they'd "saturate naturally at INT8 export" — wrong. The exported model's single
        # 'input' tensor uses ONE shared per-tensor INT8 scale across all 13 channels, so any
        # unclipped excursion (measured: temp_diff_vs_5hr reached -0.45/+1.475 in scaled units,
        # 1.7% of rows) stretches that shared scale for every channel, not just its own —
        # undercutting most of the resolution gain the tighter bounds were meant to buy. Run 18
        # adds explicit .clip(0,1) below to fix this. See MODEL_5C_TRACK_B_EXPERIMENT_LOG.md Run 18.
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
        "time_of_day_sin":    (-1, 1),
        "time_of_day_cos":    (-1, 1),
        "time_of_day_sin2":   (-1, 1),
        "time_of_day_cos2":   (-1, 1),
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
        # Run 18: explicit clip to [0,1] — the exported model's 'input' tensor uses one shared
        # per-tensor INT8 scale across all 13 channels, so any unclipped excursion (previously:
        # temp_diff_vs_5hr/6hr reaching -0.45/+1.475, ~1.7% of rows) stretches that shared scale
        # for every channel, not just its own. See MODEL_5C_TRACK_B_EXPERIMENT_LOG.md Run 18.
        X_train_df[feat] = ((X_train_df[feat] - lo) / (hi - lo)).clip(0.0, 1.0)
        X_val_df[feat] = ((X_val_df[feat] - lo) / (hi - lo)).clip(0.0, 1.0)

    X_train_flat = X_train_df.values.astype(np.float32)
    X_val_flat = X_val_df.values.astype(np.float32)

    scaler_path = os.path.join(RESULTS_DIR, f"input_scaler_5c_trackb.json")
    with open(scaler_path, "w") as f:
        json.dump(input_scaler, f, indent=2)
    print(f"✅ Input scaler saved → {scaler_path}")

    # Global target scaling (same approach as Track A / 5a / 5b)
    raw_train_tgts = train_df[targets].copy()
    raw_val_tgts = val_df[targets].copy()
    TARGET_PAD_C = 2.0
    y_min = float(raw_train_tgts.min().min()) - TARGET_PAD_C
    y_max = float(raw_train_tgts.max().max()) + TARGET_PAD_C
    for t in targets:
        train_df[t] = 2.0 * (raw_train_tgts[t] - y_min) / (y_max - y_min) - 1.0
        val_df[t] = 2.0 * (raw_val_tgts[t] - y_min) / (y_max - y_min) - 1.0

    target_scaler_path = os.path.join(RESULTS_DIR, f"target_scaler_5c_trackb.json")
    with open(target_scaler_path, "w") as f:
        json.dump({"min": y_min, "max": y_max}, f, indent=2)
    print(f"✅ Target scaler saved → {target_scaler_path}")

    print(f"\n=== SCALING BOUNDS ===")
    print(f"Global target range: {y_min:.2f}°C to {y_max:.2f}°C")
    for t in targets:
        print(f"  {t}: raw [{float(raw_train_tgts[t].min()):.2f}, {float(raw_train_tgts[t].max()):.2f}]°C")

    n_features = len(features)
    # SEQ_LEN=180: full 3hr temporal window. AveragePooling1D(pool_size=6) compresses
    # (180, n_features) → (30, n_features) → flatten before the Dense bottleneck.
    SEQ_LEN = 180

    y_all_train = train_df[targets].values.astype(np.float32)
    y_all_val = val_df[targets].values.astype(np.float32)

    # -------------------------------------------------------------------------
    # tf.data datasets (SEQ_LEN=180 — 3hr temporal window)
    # timeseries_dataset_from_array aligns targets to the END of each window:
    # window X[i:i+180] uses target y[i+179] (the current timestep).
    # -------------------------------------------------------------------------
    from tensorflow.keras.preprocessing import timeseries_dataset_from_array

    train_ds = timeseries_dataset_from_array(
        data=X_train_flat, targets=y_all_train,
        sequence_length=SEQ_LEN, sequence_stride=1, sampling_rate=1,
        batch_size=TRAIN_BATCH_SIZE, shuffle=True,
    )
    val_ds = timeseries_dataset_from_array(
        data=X_val_flat, targets=y_all_val,
        sequence_length=SEQ_LEN, sequence_stride=1, sampling_rate=1,
        batch_size=VAL_BATCH_SIZE, shuffle=False,
    )

    def split_targets(x, y):
        return x, (y[:, 0], y[:, 1], y[:, 2])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(split_targets, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(split_targets, num_parallel_calls=AUTOTUNE)

    train_steps = int(train_ds.cardinality().numpy())
    train_ds = train_ds.repeat()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    print(f"\nTraining: {train_steps} batches/epoch (batch={TRAIN_BATCH_SIZE})")
    print(f"Validation: {val_ds.cardinality().numpy()} batches")

    # Small in-memory sequence set for permutation importance and TFLite validation.
    # Window j: X_val_flat[j:j+SEQ_LEN] → target y_all_val[j+SEQ_LEN-1] (end-aligned).
    _n_small = min(2000, len(X_val_flat) - SEQ_LEN + 1)
    X_val_small = np.stack([X_val_flat[j:j + SEQ_LEN] for j in range(_n_small)])
    y_val_small = y_all_val[SEQ_LEN - 1: SEQ_LEN - 1 + _n_small]
    print(f"Small val set: X_val_small={X_val_small.shape}, y_val_small={y_val_small.shape}")

    # -------------------------------------------------------------------------
    # Training watchdog (identical to Track A)
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
    # Callbacks
    # -------------------------------------------------------------------------
    class TaskLossLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            # Support normal ("val_diff_Xhr_loss") and QAT ("val_quant_diff_Xhr_loss") output names.
            for prefix in ("", "quant_"):
                keys = tuple(f"val_{prefix}diff_{h}hr_loss" for h in (1, 2, 3))
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
    print(f"\n--- Building Track B Dense model: {RUN_NAME} ---\n")

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
        # Two-path model with shared bottleneck: Wide + Deep (no BN, no residual).
        # Interaction path removed (Run 7): element-wise square hostile to INT8 (+810% Run 6).
        # Wide path has ReLU6 (Run 9): bounds Wide to [0, 6].
        # Residual block removed (Run 10): Add layer → unbounded INT8 tensors (+547/+909/+1036% Run 9).
        # BatchNorm removed (Run 11): BN γ unconstrained by L2=1e-6, produces wide pre-clip activation
        # range even with ReLU6 downstream; INT8 calibration scale covers the γ-amplified extremes,
        # leaving typical activations with insufficient precision (seen as +271/+602/+724% in Run 10).
        # All intermediate tensors are now strictly Dense(W@x) → ReLU6: bounded to [0,6] by construction.
        # All Dense layers use use_bias=False — Coral Edge TPU best practice.
        #
        # Merge: Concat(16+32 = 48) → 3 output heads
        input_layer = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")
        pooled = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, name="avgpool")(input_layer)
        flat = tf.keras.layers.Reshape((SEQ_LEN // 6 * n_features,), name="flatten")(pooled)

        # Shared bottleneck — fused activation (Run 14): Dense(activation="relu6") emits a single
        # FULLY_CONNECTED op; TFLite quantizes only the post-activation [0,6]-bounded output.
        bottleneck = tf.keras.layers.Dense(64, activation="relu6", use_bias=False,
                                           name="bottleneck", kernel_regularizer=_reg())(flat)

        # Wide path — fused ReLU6; both paths bounded [0,6] for uniform INT8 per-tensor scale.
        wide = tf.keras.layers.Dense(16, activation="relu6", use_bias=False,
                                     name="wide", kernel_regularizer=_reg())(bottleneck)

        # Deep path — no residual (Run 10), no BN (Run 11), fused activations (Run 14).
        # All intermediate tensors bounded to [0, 6]; no unbounded pre-activation tensor.
        deep = tf.keras.layers.Dense(128, activation="relu6", use_bias=False,
                                     name="deep1", kernel_regularizer=_reg())(bottleneck)
        deep = tf.keras.layers.Dense(64, activation="relu6", use_bias=False,
                                     name="deep2", kernel_regularizer=_reg())(deep)
        # Run 22: deep_out's own ceiling tightened WITHOUT unfused custom-clip ops (Run 21's
        # ReLU(max_value=X) fixed the main activation path but its clip_by_value decomposition
        # introduced a separate badly-calibrated tensor — net worse on 2hr/3hr). Instead:
        # pre-scale up before relu6 (a native, single-op TFLite activation) so its fixed ceiling
        # of 6 lands close to the real distribution's tail (Run 20 audit: real max 0.4988,
        # p99.9=0.3555), then rescale back down to match wide's range for the concat. Verified
        # directly: this converts to 2 clean tensors, no duplicates — TFLite even folds the
        # prescale Mul into the preceding MatMul, fusing MatMul+prescale+Relu6 into one tensor.
        deep_out = tf.keras.layers.Dense(32, use_bias=False,
                                         name="deep_out", kernel_regularizer=_reg())(deep)
        deep_out = tf.keras.layers.Rescaling(scale=DEEP_OUT_PRESCALE, name="deep_out_prescale")(deep_out)
        deep_out = tf.keras.layers.Activation("relu6", name="deep_out_relu6")(deep_out)

        # Run 20: rescale deep_out before the concat. TFLite's concat op forces both concat
        # inputs to share ONE INT8 scale; deep_out's activations run at roughly a third of
        # wide's magnitude (Run 20 audit: wide real max 1.7676 vs deep_out real max 0.4988), so
        # the forced shared scale was wasting most of deep_out's effective resolution. This
        # fixed, untrained multiply brings both branches to a similar range; no weights, so it's
        # free at inference and a single MUL op that TFLite quantizes cleanly.
        deep_out = tf.keras.layers.Rescaling(scale=DEEP_OUT_RESCALE, name="deep_out_rescale")(deep_out)

        # Merge: 16 + 32 = 48 dims
        merged = tf.keras.layers.Concatenate(name="merged")([wide, deep_out])

        out_1 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_1hr")(merged)
        out_2 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_2hr")(merged)
        out_3 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_3hr")(merged)

        model = tf.keras.Model(inputs=input_layer, outputs=[out_1, out_2, out_3],
                               name=f"track_b_{RUN_NAME}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics={"diff_1hr": "mae", "diff_2hr": "mae", "diff_3hr": "mae"},
        )

    model.summary()
    print(f"\nArchitecture: avgpool(6)→flat({SEQ_LEN // 6 * n_features}) → bottleneck(64,ReLU6) → wide(16,ReLU6) + deep(128→ReLU6→64→ReLU6→32→ReLU6) → merge(48)")
    print(f"SEQ_LEN: {SEQ_LEN}, n_features: {n_features}, L2_REG: {L2_REG}")
    if QAT_FINE_TUNE and SKIP_TRAINING:
        print(f"QAT fine-tuning from {SOURCE_CHECKPOINT} — LR={QAT_LR:.0e}, max {QAT_EPOCHS} epochs, patience {QAT_EARLY_STOP_PATIENCE}")
    else:
        print(f"Run 18: clip scaled inputs to [0,1] (INT8 fix continued) — max {MAX_EPOCHS} epochs")

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    es_state_path = os.path.join(checkpoint_dir, "early_stopping_state.json")
    lr_state_path = os.path.join(checkpoint_dir, "lr_state.json")
    initial_epoch = 0

    # Warm start: only on this run's own first invocation (no checkpoint of its own yet).
    # Weight shapes must match exactly (architecture/feature count unchanged from the source
    # run) — deliberately NOT using skip_mismatch here, so a real shape mismatch fails loudly
    # instead of silently warm-starting a subset of layers.
    #
    # NOT using plain model.load_weights(WARM_START_CHECKPOINT) directly into `model`: verified
    # directly that Keras 3's native .weights.h5 format does NOT key its H5 groups by layer
    # .name at all — it uses auto-generated dense/dense_1/... keys based on model.layers'
    # internal topological order, which is NOT simply Python call order (confirmed: a layer
    # built earlier in code can appear later in model.layers depending on graph shape). Run 20
    # inserted `Rescaling` after deep_out — a weightless layer — which was enough to reorder
    # `wide` and `deep_out` relative to each other in that traversal, silently swapping their
    # loaded weights when using plain load_weights(). by_name isn't even a valid kwarg for this
    # format. Fix: build the WARM_START_CHECKPOINT's own (unmodified) architecture, load into
    # THAT (guaranteed correct — identical topology to what was saved), then copy weights
    # across by layer .name in Python, which IS reliable (unlike the H5 group keys).
    #
    # IMPORTANT — this block must be kept in exact topological sync with whatever architecture
    # actually PRODUCED WARM_START_CHECKPOINT, not the current run's architecture: even a
    # weightless layer (Rescaling, ReLU) changes the auto-generated H5 group-key ordering (Run
    # 21 hit this directly — WARM_START_CHECKPOINT=Run 20, which included Rescaling; this block
    # had been left at Run 18's pre-Rescaling shape and failed the same way Run 20's first
    # attempt did). The Rescaling `scale=` value below is irrelevant to correctness (the layer
    # has no weights either way) — only its topological presence/position matters.
    if not SKIP_TRAINING and WARM_START:
        _own_latest = os.path.join(checkpoint_dir, "model_latest.weights.h5")
        _own_best = os.path.join(checkpoint_dir, "best_model.weights.h5")
        if not os.path.exists(_own_latest) and not os.path.exists(_own_best):
            if os.path.exists(WARM_START_CHECKPOINT):
                _ws_inp  = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")
                _ws_pool = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, name="avgpool")(_ws_inp)
                _ws_flat = tf.keras.layers.Reshape((SEQ_LEN // 6 * n_features,), name="flatten")(_ws_pool)
                _ws_bn   = tf.keras.layers.Dense(64, activation="relu6", use_bias=False, name="bottleneck")(_ws_flat)
                _ws_wide = tf.keras.layers.Dense(16, activation="relu6", use_bias=False, name="wide")(_ws_bn)
                _ws_d    = tf.keras.layers.Dense(128, activation="relu6", use_bias=False, name="deep1")(_ws_bn)
                _ws_d    = tf.keras.layers.Dense(64, activation="relu6", use_bias=False, name="deep2")(_ws_d)
                _ws_dout = tf.keras.layers.Dense(32, activation="relu6", use_bias=False, name="deep_out")(_ws_d)
                _ws_dout = tf.keras.layers.Rescaling(scale=1.0, name="deep_out_rescale")(_ws_dout)
                _ws_mg   = tf.keras.layers.Concatenate(name="merged")([_ws_wide, _ws_dout])
                _ws_o1 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_1hr")(_ws_mg)
                _ws_o2 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_2hr")(_ws_mg)
                _ws_o3 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_3hr")(_ws_mg)
                _ws_model = tf.keras.Model(inputs=_ws_inp, outputs=[_ws_o1, _ws_o2, _ws_o3])
                _ws_model.load_weights(WARM_START_CHECKPOINT)
                _ws_names = {l.name for l in _ws_model.layers}
                _copied = []
                for _l in model.layers:
                    if _l.name in _ws_names and _l.get_weights():
                        _l.set_weights(_ws_model.get_layer(_l.name).get_weights())
                        _copied.append(_l.name)
                print(f"✅ Warm-started {len(_copied)} layers by name from {WARM_START_CHECKPOINT}: "
                      f"{_copied} (fresh optimizer/LR/early-stopping — not a resume)")
            else:
                print(f"⚠️  WARM_START_CHECKPOINT not found: {WARM_START_CHECKPOINT} — using random init")

    # SKIP_TRAINING path: load weights from SOURCE_CHECKPOINT.
    # Keras 3 native ".weights.h5" format: load_weights() has no by_name kwarg (that's a
    # legacy TF1/Keras2 HDF5-only argument). skip_mismatch=True is still valid and kept as a
    # safety net in case SOURCE_CHECKPOINT is ever retargeted to a run with different shapes.
    if SKIP_TRAINING:
        if os.path.exists(SOURCE_CHECKPOINT):
            model.load_weights(SOURCE_CHECKPOINT, skip_mismatch=True)
            print(f"✅ Loaded source checkpoint: {SOURCE_CHECKPOINT}")
            if QAT_FINE_TUNE:
                print(f"   QAT fine-tuning will run (initial_epoch stays 0)")
            else:
                initial_epoch = MAX_EPOCHS  # forces the training block to be skipped
                print(f"   Training skipped (SKIP_TRAINING=True, QAT_FINE_TUNE=False)")
        else:
            raise FileNotFoundError(
                f"SKIP_TRAINING=True but SOURCE_CHECKPOINT not found: {SOURCE_CHECKPOINT}")

    if DIAGNOSTIC_AUDIT:
        print("\n" + "=" * 60)
        print("🔬 DIAGNOSTIC AUDIT — real activation range vs exported INT8 calibration")
        print("=" * 60)
        # Use the SAME random-sampling methodology as representative_data_gen (not X_val_small's
        # deterministic, chronologically-contiguous first-2000-windows slice) — a biased sample
        # could itself produce a spuriously low utilization reading, and the whole point of this
        # audit is to tell that apart from a genuine long-tail calibration mismatch.
        _n_audit = X_val_flat.shape[0] - SEQ_LEN + 1
        _audit_idxs = np.random.choice(_n_audit, size=min(5000, _n_audit), replace=False)
        X_val_random = np.stack([X_val_flat[j:j + SEQ_LEN] for j in _audit_idxs])

        _probe_names = ["bottleneck", "wide", "deep1", "deep2", "deep_out", "deep_out_prescale",
                         "deep_out_relu6", "deep_out_rescale", "merged",
                         "diff_1hr", "diff_2hr", "diff_3hr"]
        _probe_names = [n for n in _probe_names if n in [l.name for l in model.layers]]
        probe_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(n).output for n in _probe_names])
        _probe_out = probe_model.predict(X_val_random, batch_size=VAL_BATCH_SIZE, verbose=0)
        _real_ranges = {n: (float(np.min(o)), float(np.max(o)))
                         for n, o in zip(_probe_names, _probe_out)}
        _percentiles = {n: np.percentile(o, [50, 95, 99, 99.9, 100])
                        for n, o in zip(_probe_names, _probe_out)}

        _audit_tflite_path = f"{AUDIT_SOURCE_RUN_DIR}/model_trackb_{os.path.basename(AUDIT_SOURCE_RUN_DIR).replace('results_5c_trackb_', '')}_int8.tflite"
        if not os.path.exists(_audit_tflite_path):
            print(f"⚠️  No exported INT8 tflite found at {_audit_tflite_path} — "
                  f"printing real ranges only, no calibration comparison.")
            for n, (lo, hi) in _real_ranges.items():
                print(f"  {n:20s} real=[{lo:8.4f}, {hi:8.4f}]")
        else:
            _interp = tf.lite.Interpreter(model_path=_audit_tflite_path)
            _interp.allocate_tensors()
            _tflite_ranges = {}
            for d in _interp.get_tensor_details():
                scale, zp = d["quantization"]
                if scale == 0.0:
                    continue
                _tflite_ranges[d["name"]] = (scale * (-128 - zp), scale * (127 - zp))

            print(f"{'tensor':20s} {'real_min':>10s} {'real_max':>10s}  "
                  f"{'calib_min':>10s} {'calib_max':>10s}  {'util%':>7s}  matched_tflite_tensor")
            for n, (rlo, rhi) in _real_ranges.items():
                # Fuzzy match: find tflite tensor names containing this layer name.
                _candidates = [k for k in _tflite_ranges if f"/{n}/" in k or k.endswith(f"/{n}")
                               or f"{n}_1/" in k or f"{n}/" in k]
                if not _candidates:
                    print(f"  {n:20s} {rlo:10.4f} {rhi:10.4f}  {'?':>10s} {'?':>10s}  {'?':>7s}  (no match found)")
                    continue
                for c in _candidates:
                    clo, chi = _tflite_ranges[c]
                    span_r = rhi - rlo
                    span_c = chi - clo
                    util = 100.0 * span_r / span_c if span_c > 1e-9 else float("nan")
                    flag = "  <-- LOW UTILIZATION" if util < 70.0 else ""
                    print(f"  {n:20s} {rlo:10.4f} {rhi:10.4f}  {clo:10.4f} {chi:10.4f}  {util:6.1f}%  {c}{flag}")
                    if util < 70.0:
                        p50, p95, p99, p999, p100 = _percentiles[n]
                        print(f"    {'':20s} percentiles: p50={p50:.4f} p95={p95:.4f} "
                              f"p99={p99:.4f} p99.9={p999:.4f} p100(max)={p100:.4f}  "
                              f"(n={_probe_out[_probe_names.index(n)].size} values, "
                              f"{min(5000, _n_audit)} random 3hr windows)")
        print("\n✅ Audit complete — exiting before training/export (DIAGNOSTIC_AUDIT=True)")
        return

    _latest_weights = os.path.join(checkpoint_dir, "model_latest.weights.h5")
    _latest_meta = os.path.join(checkpoint_dir, "model_latest_epoch.json")
    _best_weights = os.path.join(checkpoint_dir, "best_model.weights.h5")

    if not SKIP_TRAINING:
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
        elif os.path.exists(_best_weights):
            try:
                model.load_weights(_best_weights)
                print("✅ Loaded best-model checkpoint (epoch unknown)")
            except Exception as e:
                print(f"⚠️  Could not load best checkpoint: {e}")

    # Restore optimizer state (only when resuming a normal training run)
    if not SKIP_TRAINING:
        if initial_epoch > 0:
            if os.path.exists(lr_state_path):
                try:
                    with open(lr_state_path) as f:
                        lr_state = json.load(f)
                    lr_var = getattr(getattr(optimizer, "_learning_rate", None), "assign", None)
                    if lr_var is not None:
                        optimizer._learning_rate.assign(float(lr_state["lr"]))
                    else:
                        optimizer.learning_rate = float(lr_state["lr"])
                    print(f"   ✅ Restored LR: {lr_state['lr']:.2e}")
                except Exception as e:
                    print(f"   ⚠️  Could not restore LR: {e}")
        else:
            for p in (es_state_path, lr_state_path, _latest_weights, _latest_meta):
                if os.path.exists(p):
                    os.remove(p)

    # -------------------------------------------------------------------------
    # Pre-training validation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔍 PRE-TRAINING VALIDATION")
    print("=" * 60)
    errors = []

    print("\n1️⃣  Forward pass...")
    try:
        dummy = tf.zeros((1, SEQ_LEN, n_features), dtype=tf.float32)
        out = model(dummy, training=False)
        assert len(out) == 3, f"Expected 3 outputs, got {len(out)}"
        print(f"   ✅ Output shapes: {[o.shape for o in out]}")
    except Exception as e:
        errors.append(f"Forward pass: {str(e)[:200]}")

    print("\n2️⃣  Batch check...")
    try:
        xb, yb = next(iter(train_ds))
        assert xb.shape[1] == SEQ_LEN, f"seq mismatch: {xb.shape[1]} != {SEQ_LEN}"
        assert xb.shape[2] == n_features, f"feat mismatch: {xb.shape[2]} != {n_features}"
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
        if SKIP_TRAINING:
            # When SKIP_TRAINING=True the Adam optimizer has fresh (zero) second moments.
            # train_on_batch with m2≈0 amplifies gradients by lr/eps = 1e-4/1e-7 = 1000×,
            # completely scrambling the loaded checkpoint weights in a single step.
            # Use a pure eval pass instead — confirms the model is live without touching weights.
            _wds = tf.data.Dataset.from_tensors((wb, wy))
            wloss_result = model.evaluate(_wds, verbose=0)
            wloss_val = float(wloss_result[0] if isinstance(wloss_result, (list, tuple))
                              else wloss_result)
            print(f"   ✅ Forward eval (train_on_batch skipped — optimizer moments uninitialized): "
                  f"loss={wloss_val:.6f}")
        else:
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
    # QAT — wrap base model with fake-quant nodes, prepare QAT fine-tuning
    # -------------------------------------------------------------------------
    qat_model = model  # default: base model (non-QAT training path uses this)
    if QAT_FINE_TUNE and SKIP_TRAINING:
        print("\n" + "=" * 60)
        print("🔧 QAT WRAPPING")
        print("=" * 60)
        try:
            import tensorflow_model_optimization as tfmot
        except ImportError:
            raise ImportError(
                "tensorflow-model-optimization required for QAT: "
                "pip install tensorflow-model-optimization")

        print("\n🔍 Pre-QAT FP32 baseline (Run 18 weights):")
        _pre_eval = model.evaluate(val_ds.take(20), verbose=0)
        _pre_loss = float(_pre_eval[0])
        scale_pre = (y_max - y_min) / 2.0
        if len(_pre_eval) >= 7:
            _pre_1hr = float(_pre_eval[4]) * scale_pre
            _pre_2hr = float(_pre_eval[5]) * scale_pre
            _pre_3hr = float(_pre_eval[6]) * scale_pre
            print(f"   val_loss (pre-QAT FP32): {_pre_loss:.6f}")
            print(f"   FP32 MAE: 1hr={_pre_1hr:.3f}°C  2hr={_pre_2hr:.3f}°C  3hr={_pre_3hr:.3f}°C")
        else:
            print(f"   val_loss (pre-QAT FP32): {_pre_loss:.6f}")
        if _pre_loss >= 0.001:
            raise RuntimeError(
                f"Pre-QAT val_loss={_pre_loss:.6f} — Run 18 weights not loaded correctly "
                f"(expected < 0.001). Check SOURCE_CHECKPOINT path.")

        # tfmot's default QAT scheme has a hardcoded whitelist of supported activations
        # ({linear, relu, swish, softmax, sigmoid, tanh, gelu}) — relu6 is not on it, fused
        # or not (verified directly: quantize_apply raises "Only some Keras activations under
        # `keras.activations` are supported" on this architecture as-is). Build a parallel
        # relu-activated clone purely for QAT wrapping; weights transfer via get/set_weights
        # since activation choice doesn't change kernel shapes (same technique already used
        # for the FP32 export model below). QAT learns explicit per-tensor quantization ranges
        # from calibration during fine-tuning, so relu6's hard clip-at-6 (which PTQ relied on)
        # is not required the same way — untested assumption, but the only way to use tfmot's
        # default (non-custom-QuantizeConfig) path with this architecture.
        _q_inp  = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")
        _q_pool = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, name="avgpool")(_q_inp)
        _q_flat = tf.keras.layers.Reshape((SEQ_LEN // 6 * n_features,), name="flatten")(_q_pool)
        _q_bn   = tf.keras.layers.Dense(64, activation="relu", use_bias=False, name="bottleneck")(_q_flat)
        _q_wide = tf.keras.layers.Dense(16, activation="relu", use_bias=False, name="wide")(_q_bn)
        _q_d    = tf.keras.layers.Dense(128, activation="relu", use_bias=False, name="deep1")(_q_bn)
        _q_d    = tf.keras.layers.Dense(64, activation="relu", use_bias=False, name="deep2")(_q_d)
        _q_dout = tf.keras.layers.Dense(32, use_bias=False, name="deep_out")(_q_d)
        _q_dout = tf.keras.layers.Rescaling(scale=DEEP_OUT_PRESCALE, name="deep_out_prescale")(_q_dout)
        # NOTE: relu6 here would hit the same tfmot activation-whitelist issue Run 19 found for
        # deep_out's original relu6 (tfmot's default scheme doesn't support relu6 at all) — this
        # mirror is not verified QAT-compatible as-is; revisit if QAT is combined with Run 22.
        _q_dout = tf.keras.layers.Activation("relu6", name="deep_out_relu6")(_q_dout)
        # Run 20+: mirror the deep_out rescale from the training architecture (see model build
        # above) — must match whatever DEEP_OUT_RESCALE the SOURCE_CHECKPOINT was trained with,
        # or the weight transfer below will silently carry over a scale-mismatched deep_out.
        _q_dout = tf.keras.layers.Rescaling(scale=DEEP_OUT_RESCALE, name="deep_out_rescale")(_q_dout)
        _q_mg   = tf.keras.layers.Concatenate(name="merged")([_q_wide, _q_dout])
        _q_o1 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_1hr")(_q_mg)
        _q_o2 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_2hr")(_q_mg)
        _q_o3 = tf.keras.layers.Dense(
            1, activation="linear", use_bias=False, dtype="float32", name="diff_3hr")(_q_mg)
        model_for_qat = tf.keras.Model(inputs=_q_inp, outputs=[_q_o1, _q_o2, _q_o3],
                                       name=f"track_b_{RUN_NAME}_relu_for_qat")
        model_for_qat.set_weights(model.get_weights())
        print("ℹ️  Built relu-activated clone for QAT wrapping (tfmot does not support relu6); "
              "weights copied from the relu6 model loaded from Run 18.")

        qat_model = tfmot.quantization.keras.quantize_model(model_for_qat)
        # Determine actual output names after QAT wrapping for metrics dict
        _qat_out_names = [out.name.split("/")[0] for out in qat_model.outputs]
        _qat_metrics = {name: "mae" for name in _qat_out_names}
        qat_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LR, clipnorm=1.0),
            loss="mse",
            metrics=_qat_metrics,
        )
        print(f"\n✅ QAT model: {len(qat_model.layers)} layers (QuantizeWrapper applied)")
        print(f"   Output names: {_qat_out_names}")
        print(f"   Fine-tuning: LR={QAT_LR:.0e}, max={QAT_EPOCHS} epochs, "
              f"patience={QAT_EARLY_STOP_PATIENCE}")

    # -------------------------------------------------------------------------
    # Build callbacks
    # -------------------------------------------------------------------------
    early_stopping = EarlyStopping(monitor="val_task_loss", patience=EARLY_STOP_PATIENCE,
                                   restore_best_weights=True, mode="min")

    if initial_epoch > 0 and os.path.exists(es_state_path):
        try:
            with open(es_state_path) as f:
                es_state = json.load(f)
            early_stopping.best = float(es_state["best"])
            early_stopping.wait = int(es_state["wait"])
            print(f"✅ Restored ES: best={early_stopping.best:.6f}, wait={early_stopping.wait}")
        except Exception as e:
            print(f"⚠️  Could not restore ES state: {e}")

    reduce_lr = ReduceLRCallback(
        initial_lr=INITIAL_LR, monitor="val_task_loss",
        factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE,
        min_lr=REDUCE_LR_MIN, min_delta=1e-5, verbose=1)

    if initial_epoch > 0 and os.path.exists(lr_state_path):
        try:
            with open(lr_state_path) as f:
                lr_state = json.load(f)
            reduce_lr.best = float(lr_state["best"])
            reduce_lr.wait = int(lr_state["wait"])
            print(f"✅ Restored ReduceLR: best={reduce_lr.best:.6f}, wait={reduce_lr.wait}")
        except Exception as e:
            print(f"⚠️  Could not restore ReduceLR state: {e}")

    _initial_best = getattr(early_stopping, "best", float("inf"))
    if not isinstance(_initial_best, float) or np.isnan(_initial_best):
        _initial_best = float("inf")

    callbacks = [
        TaskLossLogger(),
        NaNLossTerminator(),
        reduce_lr,
        LRStateSaver(reduce_lr, lr_state_path),
        early_stopping,
        EarlyStoppingStateSaver(early_stopping, es_state_path),
        LatestEpochSaver(checkpoint_dir, initial_best=_initial_best),
        EpochProgressCallback(log_every=50, total_steps=train_steps, max_epochs=MAX_EPOCHS),
        TrainingWatchdog(timeout_minutes=10, check_interval_seconds=30,
                         train_ds=train_ds, total_batches_hint=train_steps),
    ]

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    if QAT_FINE_TUNE and SKIP_TRAINING:
        # -------------------------------------------------------------------------
        # QAT fine-tuning (short, low-LR, early-stop patience=10)
        # -------------------------------------------------------------------------
        qat_checkpoint_dir = os.path.join(RESULTS_DIR, "qat_checkpoints")
        os.makedirs(qat_checkpoint_dir, exist_ok=True)

        qat_early_stopping = EarlyStopping(
            monitor="val_task_loss", patience=QAT_EARLY_STOP_PATIENCE,
            restore_best_weights=True, mode="min")

        qat_callbacks = [
            TaskLossLogger(),
            NaNLossTerminator(),
            qat_early_stopping,
            LatestEpochSaver(qat_checkpoint_dir, initial_best=float("inf")),
            EpochProgressCallback(log_every=10, total_steps=train_steps, max_epochs=QAT_EPOCHS),
            TrainingWatchdog(timeout_minutes=10, check_interval_seconds=30,
                             train_ds=train_ds, total_batches_hint=train_steps),
        ]

        print(f"\n🚀 QAT fine-tuning: up to {QAT_EPOCHS} epochs at LR={QAT_LR:.0e}, "
              f"patience={QAT_EARLY_STOP_PATIENCE}")
        history = qat_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=QAT_EPOCHS,
            steps_per_epoch=train_steps,
            callbacks=qat_callbacks,
            verbose=0,
        )
    elif initial_epoch >= MAX_EPOCHS:
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
    # QAT: evaluate the fine-tuned QAT model (FP32 accuracy may regress slightly vs Run 18 baseline).
    # Non-QAT: evaluate the base model as usual.
    eval_model = qat_model if (QAT_FINE_TUNE and SKIP_TRAINING) else model
    eval_results = eval_model.evaluate(val_ds, verbose=0)
    # eval_results layout (Keras multi-output with per-output loss + MAE metric):
    #   [0] total_loss  [1] d1_mse  [2] d2_mse  [3] d3_mse  [4] d1_mae  [5] d2_mae  [6] d3_mae
    # QAT: same layout; output names are quant_diff_Xhr but index positions are identical.
    val_loss = float(eval_results[0])
    scale = (y_max - y_min) / 2.0
    if len(eval_results) >= 7:
        val_mae = float(np.mean(eval_results[4:7]))
        diff_1hr_mae_c = float(eval_results[4]) * scale
        diff_2hr_mae_c = float(eval_results[5]) * scale
        diff_3hr_mae_c = float(eval_results[6]) * scale
    else:
        # Fallback: only per-output MSE returned — report as RMSE
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

    # Weight verification: confirm weights are sane before TFLite export.
    # QAT: FP32 may regress slightly from the Run 18 baseline (0.000450); threshold is loosened
    # to 0.002 to allow for expected fine-tuning regression without blocking export.
    if SKIP_TRAINING:
        if QAT_FINE_TUNE:
            if val_loss >= 0.002:
                print(f"⚠️  Post-QAT val_loss={val_loss:.6f} — significant FP32 regression "
                      f"(Run 18 baseline ~0.000450, threshold 0.002). Proceeding to export.")
            else:
                print(f"✅ Post-QAT weight check: val_loss={val_loss:.6f} "
                      f"(Run 18 baseline ~0.000450)")
        else:
            if val_loss >= 0.001:
                raise RuntimeError(
                    f"\n❌ WEIGHT VERIFICATION FAILED: val_loss={val_loss:.6f} "
                    f"(expected < 0.001 from {SOURCE_CHECKPOINT}).\n"
                    f"The checkpoint weights were not properly applied. Aborting before TFLite export.")
            print(f"✅ Weight verification passed: val_loss={val_loss:.6f} (< 0.001 threshold)")

    best_epoch = (initial_epoch + int(np.argmin(history.history["val_task_loss"]) + 1)
                  ) if history else initial_epoch

    # -------------------------------------------------------------------------
    # Permutation feature importance
    # -------------------------------------------------------------------------
    print("\n📈 Permutation Feature Importance (val_loss increase):")
    X_perm = X_val_small  # (N, SEQ_LEN, n_features)
    y_perm = (y_val_small[:, 0], y_val_small[:, 1], y_val_small[:, 2])
    baseline_ds = tf.data.Dataset.from_tensor_slices(
        (X_perm, y_perm)).batch(VAL_BATCH_SIZE)
    _fi_model = qat_model if (QAT_FINE_TUNE and SKIP_TRAINING) else model
    baseline_loss = float(_fi_model.evaluate(baseline_ds, verbose=0)[0])

    feature_importance = {}
    for fi, feat in enumerate(features):
        Xp = X_perm.copy()
        col = Xp[:, :, fi].copy()  # (N, SEQ_LEN) — shuffle sequences across samples
        np.random.shuffle(col)
        Xp[:, :, fi] = col
        perm_ds = tf.data.Dataset.from_tensor_slices(
            (Xp, y_perm)).batch(VAL_BATCH_SIZE)
        perm_loss = float(_fi_model.evaluate(perm_ds, verbose=0)[0])
        feature_importance[feat] = perm_loss - baseline_loss

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance:
        print(f"  {feat}: {imp:.4f}")

    # -------------------------------------------------------------------------
    # TFLite export — FP32 (validation) then INT8 (Coral deployment)
    #
    # mixed_float16 stores hidden layer kernels as fp16. TFLite's native
    # converter only supports fp32 MatMul → FULLY_CONNECTED; it cannot lower
    # fp16 MatMul and raises ERROR_NEEDS_FLEX_OPS. Fix: rebuild the model under
    # the float32 policy, copy weights (get_weights() returns fp32 master copies
    # even when the compute policy is fp16), then export that model.
    # -------------------------------------------------------------------------
    if use_mixed_precision:
        print("\n🔧 Building fp32 export model (casting mixed-precision weights)...")
        _orig_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("float32")
        # Mirror the training architecture exactly — layer names and weight shapes must match.
        _e_inp  = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")
        _e_pool = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, name="avgpool")(_e_inp)
        _e_flat = tf.keras.layers.Reshape((SEQ_LEN // 6 * n_features,), name="flatten")(_e_pool)
        _e_bn   = tf.keras.layers.Dense(64, activation="relu6", use_bias=False, name="bottleneck")(_e_flat)
        _e_wide = tf.keras.layers.Dense(16, activation="relu6", use_bias=False, name="wide")(_e_bn)
        _e_d    = tf.keras.layers.Dense(128, activation="relu6", use_bias=False, name="deep1")(_e_bn)
        _e_d    = tf.keras.layers.Dense(64, activation="relu6", use_bias=False, name="deep2")(_e_d)
        _e_dout = tf.keras.layers.Dense(32, use_bias=False, name="deep_out")(_e_d)
        _e_dout = tf.keras.layers.Rescaling(scale=DEEP_OUT_PRESCALE, name="deep_out_prescale")(_e_dout)
        _e_dout = tf.keras.layers.Activation("relu6", name="deep_out_relu6")(_e_dout)
        _e_dout = tf.keras.layers.Rescaling(scale=DEEP_OUT_RESCALE, name="deep_out_rescale")(_e_dout)
        _e_mg   = tf.keras.layers.Concatenate(name="merged")([_e_wide, _e_dout])
        _e_o1 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_1hr")(_e_mg)
        _e_o2 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_2hr")(_e_mg)
        _e_o3 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_3hr")(_e_mg)
        export_model = tf.keras.Model(inputs=_e_inp, outputs=[_e_o1, _e_o2, _e_o3])
        export_model.set_weights([w.astype(np.float32) for w in model.get_weights()])
        export_model.compile(optimizer="sgd", loss="mse")
        tf.keras.mixed_precision.set_global_policy(_orig_policy)
        print("   ✅ fp32 export model ready")
    elif QAT_FINE_TUNE and SKIP_TRAINING:
        # qat_model is a full clone (tfmot's quantize_model clones via keras.models.clone_model)
        # — fine-tuning it does NOT update model's weights in place. Using `model` here would
        # silently re-export pre-QAT Run 18 weights under this run's name. Extract the actual
        # QAT fine-tuned kernels from each QuantizeWrapperV2's wrapped inner layer instead.
        # Note: inner_layer.get_weights() does not reflect the live variable after wrapping
        # (verified directly). inner_layer.kernel also fails after a real fit() call — it
        # becomes a stale SymbolicTensor left over from QuantizeWrapperV2's fake-quant tracing
        # (has no .numpy()), reproduced directly against a trained QAT model. The reliable path
        # is the *wrapper's* own trainable_weights, which still holds the real ResourceVariable
        # (backing store is unaffected by the inner layer's .kernel attribute being shadowed).
        # Architecture must be relu, not relu6: QAT fine-tuned model_for_qat's relu graph
        # (tfmot doesn't support relu6 — see quantize_model call above), so that's what these
        # weights were actually optimized under.
        print("\n🔧 Building fp32 export model from QAT fine-tuned weights "
              "(stripping QuantizeWrapper, relu architecture)...")
        _qat_kernels = {}
        for _l in qat_model.layers:
            _inner = getattr(_l, "layer", None)
            if _inner is not None and hasattr(_inner, "kernel"):
                _kernel_var = next(
                    w for w in _l.trainable_weights
                    if w.name.startswith(f"{_inner.name}/kernel"))
                _qat_kernels[_inner.name] = _kernel_var.numpy()
        _e_inp  = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")
        _e_pool = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, name="avgpool")(_e_inp)
        _e_flat = tf.keras.layers.Reshape((SEQ_LEN // 6 * n_features,), name="flatten")(_e_pool)
        _e_bn   = tf.keras.layers.Dense(64, activation="relu", use_bias=False, name="bottleneck")(_e_flat)
        _e_wide = tf.keras.layers.Dense(16, activation="relu", use_bias=False, name="wide")(_e_bn)
        _e_d    = tf.keras.layers.Dense(128, activation="relu", use_bias=False, name="deep1")(_e_bn)
        _e_d    = tf.keras.layers.Dense(64, activation="relu", use_bias=False, name="deep2")(_e_d)
        _e_dout = tf.keras.layers.Dense(32, use_bias=False, name="deep_out")(_e_d)
        _e_dout = tf.keras.layers.Rescaling(scale=DEEP_OUT_PRESCALE, name="deep_out_prescale")(_e_dout)
        _e_dout = tf.keras.layers.Activation("relu6", name="deep_out_relu6")(_e_dout)
        _e_dout = tf.keras.layers.Rescaling(scale=DEEP_OUT_RESCALE, name="deep_out_rescale")(_e_dout)
        _e_mg   = tf.keras.layers.Concatenate(name="merged")([_e_wide, _e_dout])
        _e_o1 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_1hr")(_e_mg)
        _e_o2 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_2hr")(_e_mg)
        _e_o3 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_3hr")(_e_mg)
        export_model = tf.keras.Model(inputs=_e_inp, outputs=[_e_o1, _e_o2, _e_o3])
        for _l in export_model.layers:
            if _l.name in _qat_kernels:
                _l.set_weights([_qat_kernels[_l.name]])
            elif hasattr(_l, "kernel"):
                raise RuntimeError(
                    f"QAT export: no fine-tuned kernel found for layer '{_l.name}' — "
                    f"QuantizeWrapper layer-name extraction did not cover the full model.")
        export_model.compile(optimizer="sgd", loss="mse")
        print(f"   ✅ fp32 export model ready ({len(_qat_kernels)} kernels transferred from QAT)")
    else:
        export_model = model

    # FP32 export model sanity check: verify the weight copy produced a valid model.
    # If set_weights failed silently or the cast introduced errors, this catches it
    # before writing any TFLite file (avoiding a silently invalid INT8 export).
    print("\n🔍 FP32 export model sanity check...")
    _sanity = export_model.evaluate(val_ds.take(5), verbose=0)
    _sanity_loss = float(_sanity[0])
    # QAT export_model's loss reflects legitimate fine-tuned weights, which (per the post-QAT
    # weight-verification check above) may genuinely regress up to ~0.002 — not a copy bug.
    # Non-QAT threshold stays tight at 0.001 since that path is purely a dtype-cast copy and
    # any deviation from the source model's already-verified loss indicates a real bug.
    _sanity_threshold = 0.002 if (QAT_FINE_TUNE and SKIP_TRAINING) else 0.001
    if _sanity_loss >= _sanity_threshold:
        raise RuntimeError(
            f"FP32 export model sanity check FAILED: val_loss={_sanity_loss:.6f} "
            f"(expected < {_sanity_threshold}). Weight transfer to the export model produced "
            f"invalid results. Aborting before INT8 export.")
    print(f"   ✅ FP32 export sanity check passed: val_loss={_sanity_loss:.6f}")

    run_model = tf.function(export_model)
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, SEQ_LEN, n_features], tf.float32, name="input"))

    tflite_fp32_path = os.path.join(RESULTS_DIR, f"model_trackb_{RUN_NAME}_fp32.tflite")
    tflite_int8_path = os.path.join(RESULTS_DIR, f"model_trackb_{RUN_NAME}_int8.tflite")
    tflite_fp32_kb = 0.0
    tflite_int8_kb = 0.0

    # FP32 export
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

    # INT8 export (full-integer quantization for Coral Edge TPU)
    # QAT path: TFLiteConverter.from_keras_model(qat_model) uses the fake-quant scales
    # embedded in the QAT model's QuantizeWrapper layers for internal op quantization.
    # representative_dataset is still provided for input/output tensor quantization.
    # PTQ path: from_concrete_functions with representative_dataset as before.
    print("🔧 Exporting INT8 TFLite model (Coral Edge TPU)...")
    try:
        def representative_data_gen():
            n = X_val_flat.shape[0] - SEQ_LEN + 1
            idxs = np.random.choice(n, size=min(2000, n), replace=False)
            for idx in idxs:
                window = X_val_flat[idx:idx + SEQ_LEN][np.newaxis, :]  # (1, SEQ_LEN, n_features)
                yield [window.astype(np.float32)]

        if QAT_FINE_TUNE and SKIP_TRAINING:
            converter_int8 = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        else:
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
        _int8_label = "QAT INT8" if (QAT_FINE_TUNE and SKIP_TRAINING) else "INT8"
        print(f"   ✅ {_int8_label} TFLite: {tflite_int8_kb:.1f} KB → {tflite_int8_path}")
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
                sample = X_val_small[i:i+1]  # (1, SEQ_LEN, n_features)
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
        "architecture": f"avgpool6→flat390+bottleneck(64,relu6)+wide(16,relu6)+deep(128→relu6→64→relu6→32→linear)→deep_out_prescale(x{DEEP_OUT_PRESCALE:.4f})→relu6→deep_out_rescale(x{DEEP_OUT_RESCALE:.4f})→merge(48);13feat:no_bn,no_residual;run22_warmstart_from_run20",
        "l2_reg": L2_REG,
        "feature_importance_permutation": sorted_importance,
        "features": features,
        "hyperparams": {
            "architecture": "avgpool6_two_path_bottleneck_relu6_no_bn_no_residual_deep_out_prescale_relu6_rescale_run22_from_run20",
            "seq_len": SEQ_LEN,
            "l2_reg": L2_REG,
            "initial_lr": INITIAL_LR,
            "max_epochs": MAX_EPOCHS,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "batch_size": TRAIN_BATCH_SIZE,
        },
        "baselines": {
            "model_5a_deployed_val_loss": 0.000682,
            "model_5a_clean_dense_wide_val_loss": 0.000373,
            "model_5b_exp37_30d_stddev": 0.930,
        },
    }
    if history:
        results["history"] = {
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            "val_task_loss": [float(v) for v in history.history.get("val_task_loss", [])],
            "loss": [float(v) for v in history.history.get("loss", [])],
            "learning_rate": [float(v) for v in history.history.get("learning_rate", [])],
        }

    results_path = os.path.join(RESULTS_DIR, f"results_5c_trackb_{RUN_NAME}.json")
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
    print(f"  Model 5a clean dense_wide_run1:  val_loss=0.000373 (Track B target)")
    print(f"  Model 5b Exp37 INT8 deployed:    30d StdDev 0.930°C")


if __name__ == "__main__":
    main()
