import tensorflow as tf

# Set to True to use Kaggle input paths instead of local relative paths
KAGGLE_MODE = False
KAGGLE_DATASET = "datasets/dacarson/weatherml-training-data"
# Set to the checkpoint dataset path to resume from a previous Kaggle run, or "" to start fresh.
# Checkpoints are read from /kaggle/input/{KAGGLE_CHECKPOINT_DATASET}/checkpoints/
# Example: "datasets/dacarson/weatherml-5b-checkpoints"
KAGGLE_CHECKPOINT_DATASET = ""  # Exp 40: new experiment — start fresh, no checkpoint resume

# All output files (scalers, TFLite, results JSON, checkpoints) go here.
# On Kaggle this is /kaggle/working (the standard Kaggle output path).
# Locally this is an experiment-scoped subdirectory so runs don't overwrite each other.
RESULTS_DIR = "/kaggle/working" if KAGGLE_MODE else "./results_exp40"

try:
    _register_keras_serializable = tf.keras.saving.register_keras_serializable
except AttributeError:
    _register_keras_serializable = tf.keras.utils.register_keras_serializable


@_register_keras_serializable(package="WeatherML")
class LastTimestepAnchorFeatures(tf.keras.layers.Layer):
    """Last timestep (t=-1) slice for anchor features (Exp 33+).

    Kept for deserialization of Exp 29–33 checkpoints.
    """

    def __init__(self, **kwargs):
        # Accept and discard legacy anchor_indices kwarg so saved Exp 29–32
        # models can still be deserialized without error.
        kwargs.pop("anchor_indices", None)
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, -1, :]

    def get_config(self):
        return super().get_config()


class MultiPointTemporalExtraction(tf.keras.layers.Layer):
    """Multi-point temporal extraction from Conv2D feature map (Exp 34 only).

    Used in Exp 34 as a replacement for GlobalAveragePooling2D. Exp 34 showed
    this approach severely regresses accuracy (val_loss 0.0024 → 0.0104) because
    the Conv2D blocks have only a ~25-step stacked temporal receptive field;
    extracting at t=−61/−121 gives local-neighbourhood context, not 60/120-min
    history. Exp 35 reverted to GlobalAveragePooling2D.

    Kept here so Exp 34 checkpoints can be deserialized for comparison/analysis.
    """

    def call(self, inputs):
        # inputs: (batch, SEQ_LEN, FILTERS) — after Reshape from (SEQ_LEN, 1, FILTERS)
        t0   = inputs[:, -1,   :]   # STRIDED_SLICE ✅
        t60  = inputs[:, -61,  :]   # STRIDED_SLICE ✅
        t120 = inputs[:, -121, :]   # STRIDED_SLICE ✅
        return tf.concat([t0, t60, t120], axis=-1)  # CONCATENATION ✅ → (batch, FILTERS*3)

    def get_config(self):
        return super().get_config()


def main():
    """
    Main training function with comprehensive fail-fast safeguards.
    
    Before making code changes, always:
    1. Run: python3 -m py_compile train_model_conv2D.py  (validate syntax)
    2. Test checkpoint loading if modifying model architecture
    3. Add new custom layers to CUSTOM_OBJECTS dictionary
    4. Run pre-training validation to catch issues early
    """
    # This version removes scipy.stats.linregress and uses Numba-accelerated NumPy for rolling slope computation.
    # This provides better multi-core performance on Raspberry Pi and simplifies dependencies.
    import multiprocessing as mp
    import multiprocessing

    # Use 'fork' because it has been more stable in this environment.
    # If the start method was already set by a parent process, ignore the error.
    try:
        mp.set_start_method("fork", force=True)
        print("ℹ️  using multiprocessing start method 'fork'")
    except RuntimeError as e:
        print(f"⚠️  Could not set multiprocessing start method to 'fork' (already set?): {e}")

    import os

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"ℹ️  Results directory: {RESULTS_DIR}")

    # Asynchronous execution: CPU can prepare next batch while GPU computes current one.
    # Previously forced synchronous for macOS stability, but powermetrics confirmed the GPU
    # was idle ~34% of the time — a clear starvation symptom of the synchronous ping-pong.
    # The watchdog + restart script handle any hangs that do occur.
    import sys
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"

    if force_cpu:
        tf.config.set_visible_devices([], 'GPU')
        print("ℹ️  FORCE_CPU=1: GPU disabled, using CPU only")
    else:
        print("ℹ️  macOS: using asynchronous execution (GPU pipeline enabled)")

    # Enable GPU if available, otherwise fall back to CPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0 and not force_cpu:
        print(f"✅ GPU detected: {len(physical_devices)} device(s)")
        for device in physical_devices:
            print(f"   - {device.name}")
        # GPU_MEMORY_MB caps GPU memory so WindowServer stays responsive while training.
        # Example: GPU_MEMORY_MB=8192 leaves ~4-8GB free for the display/UI on a 16-24GB Mac.
        # Without it, memory_growth=True lets TF expand to fill all VRAM, which starves WindowServer.
        gpu_memory_mb = os.environ.get("GPU_MEMORY_MB")
        try:
            if gpu_memory_mb:
                limit_mb = int(gpu_memory_mb)
                for device in physical_devices:
                    tf.config.experimental.set_virtual_device_configuration(
                        device,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit_mb)]
                    )
                print(f"ℹ️  GPU_MEMORY_MB={limit_mb}: GPU memory capped at {limit_mb} MB (leaves headroom for WindowServer)")
            else:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("ℹ️  GPU memory growth enabled (set GPU_MEMORY_MB=<MB> to cap usage for WindowServer)")
        except RuntimeError as e:
            print(f"⚠️  Could not configure GPU memory: {e}")
        # For Metal/GPU training:
        # intra_op: parallelism within CPU ops (data preprocessing) — more threads help
        #   keep the GPU fed since data loading is CPU-bound.
        # inter_op: parallelism between independent TF graph ops — Metal uses a single
        #   serialized command queue, so multiple inter_op threads compete rather than
        #   cooperate. Keep at 2 to avoid dispatch contention.
        cores = multiprocessing.cpu_count()
        intra_op_threads = max(4, cores // 2)  # data preprocessing — CPU-bound, benefits from parallelism
        inter_op_threads = 2                    # Metal: single command queue; >2 causes contention
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        print(f"Using {intra_op_threads} intra_op and {inter_op_threads} inter_op CPU threads (Metal: low inter_op avoids command queue contention)")
    else:
        print("⚠️  No GPU detected, using CPU")
        cores = multiprocessing.cpu_count()
        print(f"Detected {cores} CPU cores")
        # For CPU-only training, balance threads for optimal performance
        # intra_op: parallelism within operations (matrix ops, conv ops)
        # inter_op: parallelism between operations (pipeline efficiency)
        # With 10 cores, use 6 for intra_op and 2 for inter_op to allow
        # good parallelization within ops while still enabling operation pipelining
        intra_op_threads = int(os.environ.get("TF_NUM_INTRAOP_THREADS", max(4, cores - 4)))
        inter_op_threads = int(os.environ.get("TF_NUM_INTEROP_THREADS", 2))
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        print(f"Using {intra_op_threads} intra_op and {inter_op_threads} inter_op threads (optimized for CPU training)")
    
    # Allow operations to fall back to CPU if needed
    tf.config.set_soft_device_placement(True)
    
    # Optional confirmation
    print("TF intra_op threads:", tf.config.threading.get_intra_op_parallelism_threads())
    print("TF inter_op threads:", tf.config.threading.get_inter_op_parallelism_threads())
    
    # XLA JIT: enabled for CUDA/T4 and CPU; disabled for Metal (not supported)
    if KAGGLE_MODE or force_cpu:
        tf.config.optimizer.set_jit(True)
        backend = "CUDA/T4" if KAGGLE_MODE else "CPU"
        print(f"ℹ️  XLA JIT enabled ({backend} — fuses Conv2D+BN+ReLU6 kernels)")
    else:
        tf.config.optimizer.set_jit(False)  # XLA not supported on Metal backend
    from tensorflow.keras.callbacks import EarlyStopping, Callback, LearningRateScheduler
    from tensorflow.keras.regularizers import l2
    import numpy as np
    import pandas as pd
    import threading
    import time
    import traceback
    try:
        import signal
        HAS_SIGNAL = True
    except ImportError:
        HAS_SIGNAL = False  # Windows compatibility (not used, but kept for potential future use)
    # Removed unused imports that required scikit-learn to simplify Docker dependencies:
    # import joblib
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import mean_squared_error
    import copy
    import subprocess
    import json
    import glob
    import re
    import shutil

    # Restore checkpoints from a previous Kaggle run before training starts.
    # Set KAGGLE_CHECKPOINT_DATASET at the top of the file to resume; leave "" to start fresh.
    if KAGGLE_MODE and KAGGLE_CHECKPOINT_DATASET:
        _resume_src = f"/kaggle/input/{KAGGLE_CHECKPOINT_DATASET}/checkpoints"
        if os.path.exists(_resume_src):
            _restore_checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
            os.makedirs(_restore_checkpoint_dir, exist_ok=True)
            for _fname in os.listdir(_resume_src):
                shutil.copy(os.path.join(_resume_src, _fname), _restore_checkpoint_dir)
            _epoch_file = os.path.join(_restore_checkpoint_dir, "model_latest_epoch.json")
            if os.path.exists(_epoch_file):
                try:
                    with open(_epoch_file) as _f:
                        _saved_epoch = json.load(_f).get("epoch", 0)
                    print(f"✅ Checkpoints restored from {_resume_src} — will resume from epoch {_saved_epoch + 1}")
                except Exception as _e:
                    print(f"✅ Checkpoints restored from {_resume_src} (could not read epoch: {_e})")
            else:
                print(f"✅ Checkpoints restored from {_resume_src} — no epoch state found, starting fresh")
        else:
            print(f"⚠️  KAGGLE_CHECKPOINT_DATASET set but path not found: {_resume_src}")

    # Load preprocessed data
    if KAGGLE_MODE:
        data_dir = f"/kaggle/input/{KAGGLE_DATASET}"
    else:
        data_dir = ".."
    train_df = pd.read_csv(f"{data_dir}/train_data_sf.csv")
    val_df = pd.read_csv(f"{data_dir}/val_data_sf.csv")

    # ---------------------------------------------------------------------
    # Ensure deterministic time ordering and build +1h/+2h/+3h targets using
    # timestamp-based lookup (robust to missing minutes / irregular cadence).
    # This replaces any assumption that +60 rows == +60 minutes.
    # ---------------------------------------------------------------------

    def _prepare_time_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
        # Try common timestamp column names
        time_col = None
        for c in ("time", "timestamp", "ts", "datetime", "date"):
            if c in df.columns:
                time_col = c
                break

        if time_col is None:
            # If no timestamp column found, assume targets already exist and skip timestamp processing
            return df

        df = df.copy()
        # Parse timestamps.
        # The CSV sometimes stores time as epoch seconds (e.g., 1681014644). If we call
        # pd.to_datetime() without a unit, pandas interprets numeric values as *nanoseconds*,
        # which collapses everything into 1970 and breaks merge_asof.
        s = df[time_col]
        if np.issubdtype(s.dtype, np.number):
            # Heuristic based on magnitude of epoch values
            v = float(np.nanmax(s.to_numpy(dtype=np.float64)))
            # seconds ~ 1e9 (2023), ms ~ 1e12, us ~ 1e15, ns ~ 1e18
            if v >= 1e17:
                unit = "ns"
            elif v >= 1e14:
                unit = "us"
            elif v >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            df[time_col] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        else:
            df[time_col] = pd.to_datetime(s, utc=True, errors="coerce")

        if df[time_col].isna().any():
            bad = int(df[time_col].isna().sum())
            raise ValueError(f"{label}: failed to parse {bad} timestamps from column '{time_col}'.")

        # Set index and sort deterministically
        df = df.set_index(time_col)
        df = df.sort_index()

        # Drop any duplicate timestamps (keep last) to make merge_asof stable
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]

        return df

    def _add_future_targets(df: pd.DataFrame, label: str, tolerance_s: int = 90) -> pd.DataFrame:
        """Adds temp_t+1hr/2hr/3hr via timestamp lookup if they don't already exist.

        Uses merge_asof against (time + horizon) with a tolerance.
        This is robust to gaps and jitter in sampling.
        """
        # Check if targets already exist
        if all(col in df.columns for col in ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]):
            return df

        if "temperature" not in df.columns:
            raise ValueError(f"{label}: missing required column 'temperature'.")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{label}: index must be DatetimeIndex for timestamp-based target construction.")

        # Reset index to a column named 'time' and ensure it's datetime64[ns, UTC]
        base = df.reset_index()
        # Ensure the index column is named 'time'
        if "time" not in base.columns:
            base = base.rename(columns={base.columns[0]: "time"})
        base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
        if base["time"].isna().any():
            bad = int(base["time"].isna().sum())
            raise ValueError(f"{label}: failed to parse {bad} timestamps after reset_index().")

        # Deterministic ordering + stable row identity
        base = base.sort_values("time").reset_index(drop=True)
        base["row_id"] = np.arange(len(base), dtype=np.int64)

        # Source series of temperatures keyed by time
        src = base[["time", "temperature"]].copy()
        src = src.sort_values("time")
        src = src.rename(columns={"temperature": "temperature_future"})

        tol = pd.Timedelta(seconds=int(tolerance_s))

        for mins, col in ((60, "temp_t+1hr"), (120, "temp_t+2hr"), (180, "temp_t+3hr")):
            want = base[["row_id", "time"]].copy()
            want["t_query"] = want["time"] + pd.Timedelta(minutes=int(mins))

            merged = pd.merge_asof(
                want.sort_values("t_query"),
                src,
                left_on="t_query",
                right_on="time",
                direction="forward",          # ensure we only match future samples
                tolerance=tol,
            )

            # Re-align to original base row order
            merged = merged.sort_values("row_id")
            base[col] = merged["temperature_future"].to_numpy()

            # If we somehow got zero matches, fail fast with helpful diagnostics
            if base[col].isna().all():
                tmin = base["time"].min()
                tmax = base["time"].max()
                smin = src["time"].min()
                smax = src["time"].max()
                raise ValueError(
                    f"{label}: all values for {col} are NaN after merge_asof. "
                    f"Check timestamp dtype/units and tolerance. "
                    f"base_time=[{tmin}..{tmax}] src_time=[{smin}..{smax}] tol={tol}"
                )

        # Report missing target counts (usually near dataset end or around gaps)
        missing = base[["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]].isna().sum()
        print(f"\n❓ Missing target counts (timestamp lookup) for {label}:")
        print(missing)

        # Restore time index and drop helper
        base = base.drop(columns=["row_id"]).set_index("time")
        return base

    def _invalidate_targets_crossing_gaps(df: pd.DataFrame, label: str, tol_s: int = 90) -> pd.DataFrame:
        """Null pre-computed target values whose future lookup crosses a time gap.

        The CSV columns temp_t+1hr/2hr/3hr are computed from InfluxDB at export time.
        When a sensor glitch or outage occurs at time T, rows up to 3 hours before T
        have their target columns pointing into the corrupted post-gap window.  Those
        rows have normal timestamps and are not caught by _apply_gap_safety (which only
        drops rows *after* a gap).  Nulling the cross-gap targets here lets the
        subsequent dropna() remove those rows cleanly.

        Example: gap at 12:29→12:36 on 2024-03-25. A row at 11:38 has t+1hr=12:38,
        which lands in the post-gap glitch zone → t+1hr is nulled → row is dropped.
        """
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
            gap_boundary = df.index[pos - 1]  # last valid timestamp before the gap
            for h_min, col in present.items():
                # Rows whose t+h_min lookup lands after gap_boundary are invalid.
                # Window: (gap_boundary - h_min, gap_boundary]
                cutoff = gap_boundary - pd.Timedelta(minutes=h_min)
                mask = (df.index > cutoff) & (df.index <= gap_boundary)
                n = int(mask.sum())
                if n > 0:
                    df.loc[mask, col] = np.nan
                    n_nulled += n

        if n_nulled > 0:
            print(f"\n⚠️  {label}: nulled {n_nulled} cross-gap target lookups across "
                  f"{gap_positions.size} gap(s) — will be removed by dropna()")
        else:
            print(f"\n✅ {label}: no cross-gap target contamination detected")

        return df

    # Prepare time index + sort, then build timestamp-based targets if needed
    train_df = _prepare_time_index(train_df, "train_df")
    val_df = _prepare_time_index(val_df, "val_df")

    train_df = _add_future_targets(train_df, "train_df")
    val_df = _add_future_targets(val_df, "val_df")

    # Null target values that reference post-gap (potentially glitched) InfluxDB data.
    # Must run before diff computation so corrupted targets never enter training targets.
    # tol_s=600: only invalidate around real outages (10+ min gaps); the 9000+
    # single missed-minute readings do NOT corrupt forward-looking targets.
    train_df = _invalidate_targets_crossing_gaps(train_df, "train_df", tol_s=600)
    val_df   = _invalidate_targets_crossing_gaps(val_df,   "val_df",   tol_s=600)

    # Experiment 9: No pre-computed lag features or deltas.
    # The dilated Conv1D architecture learns temporal structure from the raw 180-min window.

    # Enhanced cyclical encoding for time_of_day to better capture daily patterns
    # Convert time_of_day (0-24 hours) to sine and cosine components
    train_df['time_of_day_sin'] = np.sin(2 * np.pi * train_df['time_of_day'] / 24.0)
    train_df['time_of_day_cos'] = np.cos(2 * np.pi * train_df['time_of_day'] / 24.0)
    val_df['time_of_day_sin'] = np.sin(2 * np.pi * val_df['time_of_day'] / 24.0)
    val_df['time_of_day_cos'] = np.cos(2 * np.pi * val_df['time_of_day'] / 24.0)
    
    # Add higher-order cyclical components for more complex daily patterns
    train_df['time_of_day_sin2'] = np.sin(4 * np.pi * train_df['time_of_day'] / 24.0)
    train_df['time_of_day_cos2'] = np.cos(4 * np.pi * train_df['time_of_day'] / 24.0)
    val_df['time_of_day_sin2'] = np.sin(4 * np.pi * val_df['time_of_day'] / 24.0)
    val_df['time_of_day_cos2'] = np.cos(4 * np.pi * val_df['time_of_day'] / 24.0)

    # Add cyclical encoding for day_of_year to better capture seasonal patterns
    # Convert day_of_year (1-365) to sine and cosine components
    train_df['day_of_year_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365.25)
    train_df['day_of_year_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365.25)
    val_df['day_of_year_sin'] = np.sin(2 * np.pi * val_df['day_of_year'] / 365.25)
    val_df['day_of_year_cos'] = np.cos(2 * np.pi * val_df['day_of_year'] / 365.25)
    
    # Add cyclical encoding for wind_direction (0-360 degrees) - very useful for local weather patterns
    # Wind direction affects temperature (e.g., onshore vs offshore winds, prevailing winds)
    if 'wind_direction' in train_df.columns:
        train_df['wind_direction_sin'] = np.sin(2 * np.pi * train_df['wind_direction'] / 360.0)
        train_df['wind_direction_cos'] = np.cos(2 * np.pi * train_df['wind_direction'] / 360.0)
        val_df['wind_direction_sin'] = np.sin(2 * np.pi * val_df['wind_direction'] / 360.0)
        val_df['wind_direction_cos'] = np.cos(2 * np.pi * val_df['wind_direction'] / 360.0)

    # Exp 40: all explicit lag features and temp_delta_1 removed.
    # The dilated Conv2D (RF ~251 min) can see t-60, t-120, and t-180 directly
    # in the raw sequence. The anchor path is now purely instantaneous:
    # current-timestep raw sensors + cyclical encodings + slope features only.
    

    # Exp 29: Numba-accelerated rolling linear-regression slopes.
    # Slope = °C (or raw unit) per sample over the rolling window. Rows where
    # any window sample is NaN (e.g. start of series) remain NaN and are dropped
    # by dropna() below. Rows whose window spans a data gap are handled by
    # _apply_gap_safety (drops seq_len//2 rows after each gap, which covers the
    # longest slope window of 60 samples).
    def rolling_slope_numba(data, window):
        """Vectorised rolling linear-regression slope (NumPy only — Numba removed for NumPy 2.4 compatibility)."""
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

    print("⚙️  Computing rolling slope features (Exp 29)...")
    for df in (train_df, val_df):
        df['temp_slope_15']      = rolling_slope_numba(df['temperature'].values,       15)
        df['temp_slope_30']      = rolling_slope_numba(df['temperature'].values,       30)
        df['temp_slope_60']      = rolling_slope_numba(df['temperature'].values,       60)
        df['solar_slope_30']     = rolling_slope_numba(df['solar_radiation'].values,   30)
        df['humidity_slope_30']  = rolling_slope_numba(df['relative_humidity'].values, 30)
        df['pressure_slope_60']  = rolling_slope_numba(df['station_pressure'].values,  60)
    print("   ✅ Slope features computed")

    # Manual interaction features - COMMENTED OUT: Using learned FeatureInteraction layer instead
    # Time-of-day interactions with environmental factors
    # train_df['time_sin_solar'] = train_df['time_of_day_sin'] * train_df['solar_radiation_delta']
    # train_df['time_cos_solar'] = train_df['time_of_day_cos'] * train_df['solar_radiation_delta']
    # train_df['time_sin_uv'] = train_df['time_of_day_sin'] * train_df['uv']
    # train_df['time_cos_uv'] = train_df['time_of_day_cos'] * train_df['uv']
    # val_df['time_sin_solar'] = val_df['time_of_day_sin'] * val_df['solar_radiation_delta']
    # val_df['time_cos_solar'] = val_df['time_of_day_cos'] * val_df['solar_radiation_delta']
    # val_df['time_sin_uv'] = val_df['time_of_day_sin'] * val_df['uv']
    # val_df['time_cos_uv'] = val_df['time_of_day_cos'] * val_df['uv']
    
    # Additional interaction features - wind and temperature lag interactions
    # train_df['time_sin_wind'] = train_df['time_of_day_sin'] * train_df['wind_avg']
    # train_df['time_cos_wind'] = train_df['time_of_day_cos'] * train_df['wind_avg']
    # train_df['time_sin_temp_lag'] = train_df['time_of_day_sin'] * train_df['temp_lag30']
    # train_df['time_cos_temp_lag'] = train_df['time_of_day_cos'] * train_df['temp_lag30']
    # val_df['time_sin_wind'] = val_df['time_of_day_sin'] * val_df['wind_avg']
    # val_df['time_cos_wind'] = val_df['time_of_day_cos'] * val_df['wind_avg']
    # val_df['time_sin_temp_lag'] = val_df['time_of_day_sin'] * val_df['temp_lag30']
    # val_df['time_cos_temp_lag'] = val_df['time_of_day_cos'] * val_df['temp_lag30']
    
    # Add seasonal-time interactions (how environmental effects vary by season)
    # train_df['season_sin_temp_lag'] = train_df['day_of_year_sin'] * train_df['temp_lag30']
    # train_df['season_cos_temp_lag'] = train_df['day_of_year_cos'] * train_df['temp_lag30']
    # train_df['season_sin_humidity'] = train_df['day_of_year_sin'] * train_df['humidity_lag30']
    # train_df['season_cos_humidity'] = train_df['day_of_year_cos'] * train_df['humidity_lag30']
    # val_df['season_sin_temp_lag'] = val_df['day_of_year_sin'] * val_df['temp_lag30']
    # val_df['season_cos_temp_lag'] = val_df['day_of_year_cos'] * val_df['temp_lag30']
    # val_df['season_sin_humidity'] = val_df['day_of_year_sin'] * val_df['humidity_lag30']
    # val_df['season_cos_humidity'] = val_df['day_of_year_cos'] * val_df['humidity_lag30']

    # Calculate temperature differences as targets
    # Instead of predicting absolute temperatures, predict the change from current temperature
    train_df['temp_diff_1hr'] = train_df['temp_t+1hr'] - train_df['temperature']
    train_df['temp_diff_2hr'] = train_df['temp_t+2hr'] - train_df['temperature']
    train_df['temp_diff_3hr'] = train_df['temp_t+3hr'] - train_df['temperature']

    val_df['temp_diff_1hr'] = val_df['temp_t+1hr'] - val_df['temperature']
    val_df['temp_diff_2hr'] = val_df['temp_t+2hr'] - val_df['temperature']
    val_df['temp_diff_3hr'] = val_df['temp_t+3hr'] - val_df['temperature']

    # Drop rows with NaNs
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    # ---------------------------------------------------------------------
    # GAP-AWARE WINDOW SAFETY
    #
    # Data has gaps/irregular cadence. Even with timestamp-based targets,
    # a raw sliding window can still "cross" a gap (e.g., last 3 hours includes
    # a missing 20 minutes). That produces physically inconsistent sequences.
    #
    # Strategy:
    # 1) Make temp_delta_1 time-aware: if the time step is larger than our
    #    expected cadence tolerance, zero out the delta (or it would represent
    #    a multi-minute jump).
    # 2) Drop rows immediately after any gap for `SEQ_LEN` timesteps, ensuring
    #    no window of length SEQ_LEN can span a gap boundary.
    # ---------------------------------------------------------------------

    # Sequence length needed for gap safety calculation
    SEQ_LEN = 180  # 3 hours of history

    GAP_STEP_TOLERANCE_S = 90  # expected ~60s cadence; allow some jitter

    def _apply_gap_safety(df: pd.DataFrame, label: str, seq_len: int, max_step_s: int) -> pd.DataFrame:
        df = df.copy()

        # Only apply gap safety if we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"\n⚠️  {label}: no DatetimeIndex found, skipping gap-aware window safety")
            return df

        # Compute time deltas (seconds) between consecutive samples
        dt_s = df.index.to_series().diff().dt.total_seconds()

        # 1) Make temp_delta_1 time-aware: if we crossed a gap, don't treat that
        #    as a one-step delta.
        if "temp_delta_1" in df.columns:
            gap_mask = dt_s > float(max_step_s)
            if gap_mask.any():
                df.loc[gap_mask, "temp_delta_1"] = 0.0

        # 2) Drop rows after gaps so no window can span the boundary.
        #    If a gap occurs at position i (i.e., dt_s[i] > tol), then any
        #    window ending at positions i..i+seq_len-1 would include pre-gap data.
        #    Mark those rows invalid.
        gap_positions = np.flatnonzero((dt_s > float(max_step_s)).to_numpy())
        if gap_positions.size == 0:
            return df
        
        # Diagnostic: show gap details
        print(f"\n🔍 Gap detection in {label}: found {gap_positions.size} gap(s) > {max_step_s}s")
        if gap_positions.size <= 10:  # Only show details if not too many gaps
            for pos in gap_positions[:10]:
                gap_time = df.index[pos]
                gap_size = dt_s.iloc[pos]
                print(f"   Gap at {gap_time}: {gap_size:.1f}s")

        keep = np.ones(len(df), dtype=bool)
        # Reduced from seq_len to seq_len//2 to be less aggressive and preserve more training data
        drop_span = max(int(seq_len // 2), 0)  # Drop half the sequence length instead of full length
        for pos in gap_positions:
            start = int(pos)
            end = min(int(pos) + drop_span, len(df))
            keep[start:end] = False

        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"🧩 Gap safety: dropping {dropped} rows in {label} to prevent windows crossing gaps (tol={max_step_s}s, drop_span={drop_span}).")
            df = df.iloc[keep]

        return df

    # Apply gap safety to train/val (only if datetime index exists)
    train_df = _apply_gap_safety(train_df, "train_df", SEQ_LEN, GAP_STEP_TOLERANCE_S)
    val_df = _apply_gap_safety(val_df, "val_df", SEQ_LEN, GAP_STEP_TOLERANCE_S)

    # Define feature and target columns.
    # Exp 40: Two-stream architecture — conv features first so Conv2D path can slice
    # input[:, :, :n_conv] without a gather op (StridedSlice is Edge TPU-compatible).
    #
    # Stream A (Conv2D): physical sensors + slow seasonal cycle — no diurnal, no lag/slope.
    #   Diurnal dominance (Exp 39): time_of_day_* visible across 180 timesteps let Conv2D
    #   shortcut on "what time is it?" instead of learning multi-sensor dynamics.
    #   Diurnal features are placed after n_conv so only the anchor path sees them.
    #   day_of_year_sin/cos remain — slow annual cycle, not the daily shortcut.
    # Stream B (anchor): last timestep across all n_features — sees diurnal features too.
    #
    # Exp 40 also removes temp_lag60, temp_lag120, temp_lag180, and temp_delta_1 from
    #   the anchor path. The dilated Conv2D (RF ~251 min) can see those timesteps in
    #   the raw sequence; it should learn implicit lag relationships without shortcuts.
    conv_features = [
        'temperature',
        'uv', 'wind_avg', 'wind_gust',
        'solar_radiation', 'illuminance',
        'relative_humidity', 'station_pressure',
        'day_of_year_sin', 'day_of_year_cos',
    ]
    if 'wind_direction_sin' in train_df.columns:
        conv_features.extend(['wind_direction_sin', 'wind_direction_cos'])
    if 'wind_lull' in train_df.columns:
        conv_features.append('wind_lull')
    if 'rain_accumulated' in train_df.columns:
        conv_features.append('rain_accumulated')

    # Diurnal features: routed to anchor path only (excluded from Conv2D slice).
    # Placed between conv_features and engineered_features so the prefix slice
    # input[:, :, :n_conv] naturally excludes them without a gather op.
    diurnal_features = [
        'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
    ]

    engineered_features = [
        'temp_slope_15', 'temp_slope_30', 'temp_slope_60',
        'solar_slope_30', 'humidity_slope_30', 'pressure_slope_60',
    ]

    features = conv_features + diurnal_features + engineered_features
    targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']

    ## Per-feature min/max scaling with ±5% padding
    # Domain bounds for raw station features + cyclical encodings
    domain_bounds = {
        "temperature": (-10, 55),
        "temp_slope_15":     (None, None),
        "temp_slope_30":     (None, None),
        "temp_slope_60":     (None, None),
        "solar_slope_30":    (None, None),
        "humidity_slope_30": (None, None),
        "pressure_slope_60": (None, None),
        "wind_gust": (0, 40),
        "wind_avg": (0, 30),
        "uv": (0, None),
        "solar_radiation": (0, None),
        "illuminance": (0, None),
        "relative_humidity": (0, 100),
        "station_pressure": (None, None),
        "day_of_year_sin": (-1, 1),
        "day_of_year_cos": (-1, 1),
        "time_of_day_sin": (-1, 1),
        "time_of_day_cos": (-1, 1),
        "time_of_day_sin2": (-1, 1),
        "time_of_day_cos2": (-1, 1),
        "wind_direction_sin": (-1, 1),
        "wind_direction_cos": (-1, 1),
        "wind_lull": (0, None),
        "rain_accumulated": (0, None),
    }

    X_train_df = train_df[features].copy()
    X_val_df = val_df[features].copy()
    input_scaler = {}
    for feature in features:
        f_min = train_df[feature].min()
        f_max = train_df[feature].max()
        range_pad = 0.05 * (f_max - f_min)

        floor, ceiling = domain_bounds.get(feature, (None, None))
        f_min_adj = floor if floor is not None else f_min - range_pad
        f_max_adj = ceiling if ceiling is not None else f_max + range_pad

        input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
        X_train_df[feature] = (X_train_df[feature] - f_min_adj) / (f_max_adj - f_min_adj)
        X_val_df[feature] = (X_val_df[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    X_train_flat = X_train_df.values
    X_val_flat = X_val_df.values
    with open(os.path.join(RESULTS_DIR, "input_scaler_5b.json"), "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Normalize target values (temperature differences)
    # IMPORTANT: Use PER-HORIZON scaling bounds.
    # The 1h/2h/3h diffs have different natural ranges; using a single global
    # min/max compresses the smaller horizons and makes learning harder.

    # Capture raw (unscaled) targets for correct stats/printing
    raw_train_targets = train_df[targets].copy()
    raw_val_targets = val_df[targets].copy()

    # Global target scaling (Model 5a's approach): Use same min/max for all targets
    # This is more stable than per-horizon scaling and worked excellently for Model 5a
    TARGET_PAD_C = 2.0
    y_min = float(raw_train_targets.min().min()) - TARGET_PAD_C
    y_max = float(raw_train_targets.max().max()) + TARGET_PAD_C
    
    # Scale all targets using the same global bounds
    for t in targets:
        train_df[t] = 2.0 * (raw_train_targets[t] - y_min) / (y_max - y_min) - 1.0
        val_df[t] = 2.0 * (raw_val_targets[t] - y_min) / (y_max - y_min) - 1.0

    # Persist global target scaler (Model 5a format)
    with open(os.path.join(RESULTS_DIR, "target_scaler_5b.json"), "w") as f:
        json.dump({"min": y_min, "max": y_max, "range": (y_min, y_max)}, f, indent=2)

    # Per-target keys for inverse scaling (same bounds for all horizons — global scaling above)
    y_mins = {t: y_min for t in targets}
    y_maxs = {t: y_max for t in targets}

    # Raw (unscaled) target stats for diagnostics
    raw_1_min = float(raw_train_targets["temp_diff_1hr"].min())
    raw_1_max = float(raw_train_targets["temp_diff_1hr"].max())
    raw_2_min = float(raw_train_targets["temp_diff_2hr"].min())
    raw_2_max = float(raw_train_targets["temp_diff_2hr"].max())
    raw_3_min = float(raw_train_targets["temp_diff_3hr"].min())
    raw_3_max = float(raw_train_targets["temp_diff_3hr"].max())

    # Print min/max values for all features
    print("\n=== FEATURE MIN/MAX VALUES ===")
    print("Input Features (Original Scale):")
    for feature in features:
        f_min = train_df[feature].min()
        f_max = train_df[feature].max()
        print(f"  {feature}: min={f_min:.4f}, max={f_max:.4f}")

    print("\nTarget Features (Temperature Differences):")
    for target in targets:
        t_min = train_df[target].min()
        t_max = train_df[target].max()
        print(f"  {target}: min={t_min:.4f}°C, max={t_max:.4f}°C")

    print("\n=== SCALING BOUNDS ===")
    print("Temperature difference scaling bounds (global - Model 5a approach):")
    print(f"  Global range: {y_min:.2f}°C to {y_max:.2f}°C")
    print(f"Original temperature difference stats:")
    print(f"  1hr: min={raw_1_min:.2f}°C, max={raw_1_max:.2f}°C")
    print(f"  2hr: min={raw_2_min:.2f}°C, max={raw_2_max:.2f}°C")
    print(f"  3hr: min={raw_3_min:.2f}°C, max={raw_3_max:.2f}°C")

    print("\nInput Feature Scaling Parameters:")
    for feature in features:
        scaler_params = input_scaler[feature]
        print(f"  {feature}: min={scaler_params['min']:.4f}, max={scaler_params['max']:.4f}")
    print("=" * 50)


    n_features = len(features)
    n_conv = len(conv_features)  # Exp 40: Conv2D path sees only the first n_conv features (no diurnal, no lag/slope)

    # Use the scaled targets (already in [-1, 1]) to align with sequence windows
    y_all_train = train_df[targets].values
    y_all_val = val_df[targets].values

    # Streaming time-series datasets (no giant in-memory window arrays)
    from tensorflow.keras.preprocessing import timeseries_dataset_from_array
    
    # Larger batches = more work per GPU call = better GPU utilization.
    # T4/Kaggle: 1024 fills more of the 13.7 GB VRAM per step.
    # Local Metal: 1024 on M1 Pro (16 GB unified) — GPU runs at ~80% at 512 (compute-bound,
    # not memory-bound), so doubling batch halves step count with minimal memory risk.
    TRAIN_BATCH_SIZE = 1024
    VAL_BATCH_SIZE = 1024

    
    train_ds = timeseries_dataset_from_array(
        data=X_train_flat,
        targets=y_all_train,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    val_ds = timeseries_dataset_from_array(
        data=X_val_flat,
        targets=y_all_val,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
    )

    # Split 3-target vector into a tuple matching the model's three outputs
    def split_targets(x, y):
        return x, (y[:, 0], y[:, 1], y[:, 2])

    # Optimize data pipeline for GPU training
    AUTOTUNE = tf.data.AUTOTUNE

    # Map operation should be parallelized
    train_ds = train_ds.map(split_targets, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(split_targets, num_parallel_calls=AUTOTUNE)

    # Capture steps_per_epoch BEFORE .repeat() makes cardinality infinite (-1).
    # Keras 3's EpochIterator creates one iterator for all epochs over a finite
    # dataset and divides the total batches across (max_epochs - initial_epoch)
    # remaining epochs.  With initial_epoch=39 and 1144 total batches, that gives
    # only ~10 batches/epoch — hence the "ran out of data" warning at batch ~111.
    # .repeat() + explicit steps_per_epoch is the correct fix.
    train_steps = int(train_ds.cardinality().numpy())

    # .repeat() makes the dataset infinite; model.fit(steps_per_epoch=train_steps)
    # bounds each epoch to exactly train_steps batches.
    train_ds = train_ds.repeat()

    # Fixed prefetch depth instead of AUTOTUNE: AUTOTUNE is adaptive and grows its buffer
    # over successive epochs. After 8-10 epochs it can reach hundreds of batches (~GBs),
    # exhausting Metal GPU memory. This first corrupts validation (causing the val_loss
    # spike seen before each hang), then blocks epoch N+1's first batch indefinitely.
    # A fixed depth keeps GPU fed (4 × 7.4 MB = ~30 MB) without the runaway growth.
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE if KAGGLE_MODE else 4)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE if KAGGLE_MODE else 4)

    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"Validation batch size: {VAL_BATCH_SIZE}")

    print(f"\nSequence training dataset: {train_steps} batches/epoch (infinite via .repeat())")
    print(f"Sequence validation dataset: {val_ds.cardinality().numpy()} batches")

    # Small in-memory validation window set for feature importance & TFLite validation
    def build_sequence_data(X_flat, y, seq_len, max_samples=None):
        n_samples = X_flat.shape[0]
        windows = []
        targets_seq = []
        start = seq_len - 1
        for i in range(start, n_samples):
            window = X_flat[i - seq_len + 1:i + 1, :]  # shape (seq_len, n_features)
            windows.append(window)
            targets_seq.append(y[i])
            if max_samples is not None and len(windows) >= max_samples:
                break
        return np.array(windows, dtype=np.float32), np.array(targets_seq, dtype=np.float32)

    X_val_small, y_val_small = build_sequence_data(X_val_flat, y_all_val, SEQ_LEN, max_samples=2000)
    print(f"Small validation window set for analysis: X_val_small={X_val_small.shape}, y_val_small={y_val_small.shape}")

    class TrainingWatchdog(Callback):
        """Callback that monitors training progress and detects hangs.
        
        This watchdog tracks training activity (epoch start/end, batch completion)
        and raises an exception if no progress is detected for a specified timeout.
        """
        def __init__(self, timeout_minutes=2, check_interval_seconds=30, verbose=1, adaptive_timeout=True, train_ds=None, total_batches_hint=None):
            """
            Args:
                timeout_minutes: Maximum time (in minutes) without progress before detecting a hang
                check_interval_seconds: How often (in seconds) to check for hangs
                verbose: Verbosity level (0=silent, 1=print warnings, 2=print all activity)
                adaptive_timeout: If True, base timeout on actual batch times (default: True)
                                  Uses 50x the average batch time, with a minimum of timeout_minutes
                total_batches_hint: Pre-computed steps per epoch (use when train_ds uses .repeat()
                                    and cardinality() returns -1)
            """
            super().__init__()
            self.base_timeout_seconds = timeout_minutes * 60
            self.timeout_seconds = self.base_timeout_seconds  # Will adapt based on batch times
            self.check_interval = check_interval_seconds
            self.verbose = verbose
            self.adaptive_timeout = adaptive_timeout
            self.train_ds = train_ds  # Store dataset reference to get cardinality
            self._total_batches_hint = total_batches_hint  # Fallback when cardinality is -1
            
            # Progress tracking
            self.last_activity_time = None
            self.current_epoch = None
            self.current_batch = None
            self.total_batches = None
            self.epoch_start_time = None
            self.last_batch_time = None
            self.last_activity_type = None  # 'epoch_start', 'epoch_end', 'batch_end'
            self.batch_durations = []  # Track batch processing times
            self.batch_update_frequency = 5   # Update activity time every N batch_end callbacks.
            # With steps_per_execution=1, on_batch_end fires every actual batch,
            # so this refreshes the watchdog every 5 actual batches.
            
            # Watchdog thread control
            self.watchdog_thread = None
            self.stop_watchdog = threading.Event()
            self.hang_detected = threading.Event()
            self.hang_info = None
            
        def on_train_begin(self, logs=None):
            """Start watchdog thread when training begins."""
            self.last_activity_time = time.time()
            self.epoch_start_time = time.time()
            self.stop_watchdog.clear()
            self.hang_detected.clear()
            self.hang_info = None
            
            # Try to determine total batches per epoch from dataset cardinality
            # This is the most reliable method
            self.total_batches = None
            try:
                if self.train_ds is not None:
                    cardinality_val = self.train_ds.cardinality().numpy()
                    # Check if cardinality is known (not UNKNOWN_CARDINALITY which is -1)
                    if cardinality_val >= 0:
                        self.total_batches = int(cardinality_val)
                        if self.verbose >= 1:
                            print(f"🛡️  Determined total batches per epoch from dataset: {self.total_batches}")
            except (AttributeError, TypeError, tf.errors.OutOfRangeError, Exception) as e:
                if self.verbose >= 2:
                    print(f"🛡️  Could not get dataset cardinality: {e}")
            
            # Fallback: use the pre-computed hint (set when .repeat() makes cardinality -1)
            if self.total_batches is None and self._total_batches_hint:
                self.total_batches = self._total_batches_hint
                if self.verbose >= 1:
                    print(f"🛡️  Determined total batches per epoch from hint: {self.total_batches}")
            
            # Start watchdog thread
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="TrainingWatchdog"
            )
            self.watchdog_thread.start()
            if self.verbose >= 1:
                timeout_str = f"{self.base_timeout_seconds/60:.1f} min"
                if self.adaptive_timeout:
                    timeout_str += " (adaptive based on batch times)"
                print(f"🛡️  Training watchdog started (timeout: {timeout_str}, check interval: {self.check_interval}s)")
                if self.total_batches:
                    print(f"🛡️  Expected batches per epoch: {self.total_batches}")
        
        def on_train_end(self, logs=None):
            """Stop watchdog thread when training ends."""
            if self.watchdog_thread is not None:
                self.stop_watchdog.set()
                self.watchdog_thread.join(timeout=5.0)
                if self.verbose >= 1:
                    print("🛡️  Training watchdog stopped")
        
        def on_epoch_begin(self, epoch, logs=None):
            """Track epoch start."""
            self.current_epoch = epoch
            self.current_batch = 0
            self.epoch_start_time = time.time()
            self.last_activity_time = time.time()
            self.last_activity_type = 'epoch_start'
            self.batch_durations = []  # Reset for new epoch
            # Try to determine total batches from logs if available
            if logs and 'size' in logs:
                # This is approximate, actual size comes later
                pass
            if self.verbose >= 2:
                print(f"🛡️  Watchdog: Epoch {epoch} started")
        
        def on_epoch_end(self, epoch, logs=None):
            """Track epoch end."""
            epoch_duration = time.time() - self.epoch_start_time
            self.last_activity_time = time.time()
            self.last_activity_type = 'epoch_end'
            
            # Only update total_batches if we don't already have it from dataset cardinality
            # This prevents incorrect values if epoch ends early due to hang
            if self.total_batches is None and self.current_batch is not None and self.current_batch >= 0:
                # Total batches is current_batch + 1 (since batch numbers are 0-indexed)
                # This is a fallback if we couldn't get it from dataset
                self.total_batches = self.current_batch + 1
                if self.verbose >= 2:
                    print(f"🛡️  Watchdog: Estimated total batches per epoch from epoch end: {self.total_batches}")
            
            if self.total_batches and self.current_batch is not None:
                avg_batch_time = sum(self.batch_durations[-100:]) / min(100, len(self.batch_durations)) if self.batch_durations else 0
                batches_completed = self.current_batch + 1
                if self.verbose >= 2:
                    print(f"🛡️  Watchdog: Epoch {epoch} ended (duration: {epoch_duration:.1f}s, "
                          f"{batches_completed}/{self.total_batches} batches, avg batch: {avg_batch_time*1000:.1f}ms)")
        
        def on_train_batch_begin(self, batch, logs=None):
            """Track batch start."""
            self.current_batch = batch
            self.last_batch_time = time.time()
            
            # Update activity time more frequently near epoch boundaries (last 100 batches)
            # or every 50 batches otherwise
            should_update = False
            if self.total_batches:
                remaining = self.total_batches - batch
                # More frequent updates in last 100 batches of epoch
                if remaining <= 100:
                    should_update = (remaining % 10 == 0) or (remaining <= 5)
                else:
                    should_update = (batch % self.batch_update_frequency == 0)
            else:
                should_update = (batch % self.batch_update_frequency == 0)
            
            if should_update:
                self.last_activity_time = time.time()
                self.last_activity_type = 'batch_start'
        
        def update_activity(self, activity_type='activity'):
            """Update activity time (can be called by other callbacks to prevent false hang detection)."""
            self.last_activity_time = time.time()
            self.last_activity_type = activity_type
            if self.verbose >= 2:
                print(f"🛡️  Watchdog: Activity updated ({activity_type})")
        
        def on_train_batch_end(self, batch, logs=None):
            """Track batch end."""
            batch_duration = time.time() - self.last_batch_time if self.last_batch_time else 0
            self.batch_durations.append(batch_duration)
            # Keep only last 200 batch durations to avoid memory issues
            if len(self.batch_durations) > 200:
                self.batch_durations = self.batch_durations[-200:]
            
            # Adapt timeout based on actual batch times (after we have enough samples)
            if self.adaptive_timeout and len(self.batch_durations) >= 10:
                avg_batch_time = sum(self.batch_durations) / len(self.batch_durations)
                # Calculate expected time for next update based on update frequency
                # We update every N batches, so expected time = N * avg_batch_time
                # Use 5x that as timeout to account for normal variation
                # But never less than base timeout to ensure we catch complete hangs
                batches_until_update = self.batch_update_frequency
                if self.total_batches:
                    remaining = self.total_batches - batch
                    if remaining <= 100:
                        batches_until_update = 10 if remaining > 5 else 1
                
                expected_update_time = batches_until_update * avg_batch_time
                adaptive_timeout = max(expected_update_time * 5, self.base_timeout_seconds)
                
                # Only update if it's significantly different to avoid spam
                if abs(adaptive_timeout - self.timeout_seconds) > 5:
                    self.timeout_seconds = adaptive_timeout
                    if self.verbose >= 2:
                        print(f"🛡️  Watchdog: Adaptive timeout updated to {self.timeout_seconds:.1f}s "
                              f"(avg batch: {avg_batch_time*1000:.1f}ms, next update in ~{expected_update_time:.1f}s)")
            
            # Try to determine total batches from logs
            if logs and 'size' in logs and self.total_batches is None:
                # This is approximate, but helps with progress tracking
                pass
            
            # Update activity time more frequently near epoch boundaries
            should_update = False
            if self.total_batches:
                remaining = self.total_batches - batch
                # More frequent updates in last 100 batches of epoch
                if remaining <= 100:
                    should_update = (remaining % 10 == 0) or (remaining <= 5)
                else:
                    should_update = (batch % self.batch_update_frequency == 0)
            else:
                should_update = (batch % self.batch_update_frequency == 0)
            
            if should_update:
                self.last_activity_time = time.time()
                self.last_activity_type = 'batch_end'
                if self.verbose >= 2 and batch > 0:
                    remaining_str = f" ({self.total_batches - batch} remaining)" if self.total_batches else ""
                    print(f"🛡️  Watchdog: Batch {batch}{remaining_str} completed (duration: {batch_duration*1000:.1f}ms)")
        
        def _watchdog_loop(self):
            """Main watchdog loop that runs in a separate thread."""
            while not self.stop_watchdog.is_set():
                # Wait for check interval
                if self.stop_watchdog.wait(self.check_interval):
                    break  # Stop requested
                
                if self.last_activity_time is None:
                    continue
                
                # Check if too much time has passed without activity
                time_since_activity = time.time() - self.last_activity_time
                
                if time_since_activity > self.timeout_seconds:
                    # Hang detected! Gather comprehensive diagnostic information
                    current_time = time.time()
                    epoch_duration = current_time - self.epoch_start_time if self.epoch_start_time else 0
                    
                    # Calculate batch statistics
                    avg_batch_time = 0
                    median_batch_time = 0
                    if self.batch_durations:
                        avg_batch_time = sum(self.batch_durations) / len(self.batch_durations)
                        sorted_durations = sorted(self.batch_durations)
                        median_batch_time = sorted_durations[len(sorted_durations) // 2]
                    
                    # Calculate progress percentage
                    # Note: current_batch is 0-indexed, so if we're at batch 6600, we've completed batches 0-6600 (6601 batches)
                    progress_pct = None
                    batches_remaining = None
                    batches_completed = None
                    if self.total_batches and self.current_batch is not None:
                        batches_completed = self.current_batch + 1  # +1 because batches are 0-indexed
                        progress_pct = (batches_completed / self.total_batches) * 100
                        batches_remaining = self.total_batches - batches_completed
                    
                    # Gather system information if available
                    memory_info = {}
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = {
                            'rss_mb': process.memory_info().rss / 1024 / 1024,
                            'vms_mb': process.memory_info().vms / 1024 / 1024,
                            'percent': process.memory_percent(),
                            'num_threads': process.num_threads()
                        }
                    except (ImportError, AttributeError, Exception):
                        # psutil not available or error accessing process info
                        pass
                    
                    self.hang_info = {
                        'timeout_seconds': self.timeout_seconds,
                        'time_since_activity': time_since_activity,
                        'current_epoch': self.current_epoch,
                        'current_batch': self.current_batch,
                        'total_batches': self.total_batches,
                        'batches_completed': batches_completed,
                        'batches_remaining': batches_remaining,
                        'progress_percent': progress_pct,
                        'epoch_start_time': self.epoch_start_time,
                        'epoch_duration_seconds': epoch_duration,
                        'last_activity_time': self.last_activity_time,
                        'last_activity_type': self.last_activity_type,
                        'timestamp': current_time,
                        'batch_statistics': {
                            'total_tracked': len(self.batch_durations),
                            'avg_batch_time_ms': avg_batch_time * 1000,
                            'median_batch_time_ms': median_batch_time * 1000,
                            'last_batch_time_ms': self.batch_durations[-1] * 1000 if self.batch_durations else 0
                        },
                        'memory_info': memory_info
                    }
                    self.hang_detected.set()
                    
                    # Print diagnostic information
                    print("\n" + "="*80)
                    print("🚨 TRAINING HANG DETECTED BY WATCHDOG")
                    print("="*80)
                    print(f"Timeout threshold: {self.timeout_seconds/60:.1f} minutes")
                    print(f"Time since last activity: {time_since_activity/60:.1f} minutes ({time_since_activity:.1f} seconds)")
                    print(f"\nTraining State:")
                    print(f"  Current epoch: {self.current_epoch}")
                    print(f"  Current batch (0-indexed): {self.current_batch}")
                    if self.total_batches and batches_completed is not None:
                        print(f"  Batches completed: {batches_completed} / {self.total_batches}")
                        print(f"  Batches remaining: {batches_remaining}")
                        print(f"  Progress: {progress_pct:.1f}%")
                    print(f"  Epoch duration: {epoch_duration/60:.1f} minutes ({epoch_duration:.1f} seconds)")
                    print(f"  Last activity: {self.last_activity_type} at {time.ctime(self.last_activity_time)}")
                    
                    if self.batch_durations:
                        print(f"\nBatch Performance:")
                        print(f"  Average batch time: {avg_batch_time*1000:.1f} ms")
                        print(f"  Median batch time: {median_batch_time*1000:.1f} ms")
                        if len(self.batch_durations) > 1:
                            print(f"  Last batch time: {self.batch_durations[-1]*1000:.1f} ms")
                    
                    if memory_info:
                        print(f"\nSystem Resources:")
                        print(f"  Memory RSS: {memory_info.get('rss_mb', 0):.1f} MB")
                        print(f"  Memory VMS: {memory_info.get('vms_mb', 0):.1f} MB")
                        print(f"  Memory %: {memory_info.get('percent', 0):.1f}%")
                        print(f"  Threads: {memory_info.get('num_threads', 0)}")
                    
                    print("\n⚠️  This appears to be the same hang pattern seen in the stacktrace.")
                    print("   Main thread is likely blocked in TensorFlow execution.")
                    current_batch_str = str(self.current_batch) if self.current_batch is not None else "unknown"
                    current_epoch_str = str(self.current_epoch) if self.current_epoch is not None else "unknown"
                    print(f"   Hang occurred at batch {current_batch_str} of epoch {current_epoch_str}")
                    if batches_remaining is not None and batches_remaining < 20:
                        print(f"   ⚠️  HANG OCCURRED NEAR END OF EPOCH ({batches_remaining} batches remaining)")
                    elif batches_remaining == 0 and self.current_batch is not None and self.total_batches:
                        print(f"   ⚠️  HANG OCCURRED AFTER ALL BATCHES COMPLETED - likely during validation or callback processing")
                    print("\n💡 Attempting to interrupt training...")
                    print("="*80 + "\n")
                    
                    # Write hang info to a file for external monitoring
                    try:
                        with open(os.path.join(RESULTS_DIR, "training_hang_detected.json"), "w") as f:
                            json.dump(self.hang_info, f, indent=2)
                    except Exception as e:
                        print(f"⚠️  Could not write hang info: {e}")
                    
                    # Forcefully exit the process when TensorFlow is hung
                    # When TensorFlow is blocked in C++ code, signals (SIGINT/SIGTERM) may not
                    # be processed. Using os._exit() immediately terminates the process,
                    # bypassing Python cleanup and any blocked operations.
                    # Exit code 99 indicates a hang was detected (for restart scripts to detect)
                    print("🛑 Forcefully exiting process due to detected hang...")
                    print("   (Training can be resumed from the last checkpoint)")
                    try:
                        import os
                        # Use _exit instead of exit to bypass cleanup and signal handlers
                        # This works even when the main thread is blocked in TensorFlow C++ code
                        os._exit(99)  # Exit code 99 = hang detected
                    except Exception as e:
                        # If _exit fails (shouldn't happen), try regular exit
                        print(f"⚠️  os._exit failed, trying sys.exit: {e}")
                        import sys
                        sys.exit(99)
                    
                    break  # Exit watchdog loop (shouldn't reach here)
        
        def check_hang(self):
            """Check if a hang was detected. Can be called from main thread."""
            if self.hang_detected.is_set():
                raise RuntimeError(
                    f"Training hang detected: No progress for {self.hang_info['time_since_activity']/60:.1f} minutes. "
                    f"Epoch: {self.hang_info['current_epoch']}, Batch: {self.hang_info['current_batch']}"
                )

    # All custom layer classes registered so any experiment's checkpoint can be deserialized.
    CUSTOM_OBJECTS = {
        "LastTimestepAnchorFeatures": LastTimestepAnchorFeatures,
        "MultiPointTemporalExtraction": MultiPointTemporalExtraction,
    }
    
    def validate_checkpoint_loading(checkpoint_path, model, custom_objects):
        """
        Validate that a checkpoint can be loaded before relying on it.
        Returns (success: bool, error_message: str or None)

        Weights-only (.weights.h5) checkpoints are validated with a pure h5py
        disk read — zero GPU operations.  Full-model checkpoints use load_model()
        and are only called at startup (GPU idle), never from epoch-end callbacks.
        """
        is_weights_only = checkpoint_path.endswith('.weights.h5')
        is_full_model = (os.path.isdir(checkpoint_path) or
                         (checkpoint_path.endswith('.h5') and not is_weights_only))

        if is_weights_only:
            # Pure disk I/O — no GPU operations whatsoever.
            #
            # Prior approaches (model.load_weights on live model; clone_model) both
            # caused epoch-boundary hangs on macOS Metal: the first by corrupting
            # the GPU command queue state, the second by leaking Metal buffers that
            # the Python GC doesn't immediately free, causing progressive slowdowns
            # (923s → 1033s per epoch) until the allocator hung.
            #
            # Architecture mismatch is impossible here: these files are saved moments
            # earlier by the same model in the same training run.  The only real risk
            # is disk corruption during write, which h5py open + non-empty check
            # catches without touching the GPU.
            import h5py
            try:
                with h5py.File(checkpoint_path, 'r') as f:
                    if not len(f.keys()):
                        return False, "Checkpoint HDF5 file has no content"
                return True, None
            except Exception as e:
                return False, f"HDF5 read error: {str(e)[:200]}"

        elif is_full_model:
            # Full model checkpoint — uses load_model() (GPU op).
            # Only safe to call at startup before training begins (GPU idle).
            # Never call this from an epoch-end callback.
            original_weights = None
            try:
                original_weights = [w.copy() for w in model.get_weights()]
            except Exception:
                pass
            try:
                test_model = tf.keras.models.load_model(
                    checkpoint_path,
                    compile=False,
                    custom_objects=custom_objects
                )
                test_weights = test_model.get_weights()
                del test_model

                if len(test_weights) != len(model.get_weights()):
                    return False, (f"Architecture mismatch: checkpoint has {len(test_weights)} "
                                   f"weight arrays, current model has {len(model.get_weights())}")
                for i, (test_w, model_w) in enumerate(zip(test_weights, model.get_weights())):
                    if test_w.shape != model_w.shape:
                        return False, (f"Architecture mismatch at layer {i}: "
                                       f"checkpoint shape {test_w.shape} != model shape {model_w.shape}")
                return True, None
            except Exception as e:
                if original_weights:
                    try:
                        model.set_weights(original_weights)
                    except Exception:
                        pass
                error_msg = str(e)
                if "expected" in error_msg.lower() and ("variables" in error_msg.lower() or "variable" in error_msg.lower()):
                    return False, f"Architecture mismatch: {error_msg[:200]}"
                elif "shape" in error_msg.lower() and "mismatch" in error_msg.lower():
                    return False, f"Shape mismatch: {error_msg[:200]}"
                elif "_extract" in error_msg or "function" in error_msg.lower() or "serializable" in error_msg.lower():
                    return False, f"Serialization issue (old Lambda layer): {error_msg[:200]}"
                else:
                    return False, f"Checkpoint loading error: {error_msg[:200]}"

        else:
            return False, f"Unknown checkpoint format: {checkpoint_path}"

    # Mixed precision: disabled everywhere (Exp 37 — NaN crashes at epochs 38 and 73 on T4).
    # Root cause: LossScaleOptimizer grows its dynamic scale unboundedly (~32768 × 2^N after N
    # doublings). During backprop, gradients are multiplied by the scale in fp16 space; once
    # scale × real_gradient > 65504 (fp16 max), the result overflows to inf/NaN *before*
    # global_clipnorm can act (clipnorm operates on the already-divided final gradients).
    # float32 on T4 is ~50-60% slower than fp16 but eliminates NaN risk and TFLite FP16 issues.
    print("ℹ️  Mixed precision disabled (float32 throughout — reliable on T4 and Metal)")

    def build_and_train_model(name):
        print(f"\n--- Running: {name} ---\n")

        # ---------------------------------------------------------------------
        # Experiment 40: Remove diurnal features from Conv2D path + remove explicit
        # lag features from anchor — let dilated Conv2D learn temporal structure.
        #
        # Builds on Exp 39's dilated pyramid (RF ~251 min). Two changes vs. Exp 39:
        #
        # 1) Diurnal dominance fix: time_of_day_sin/cos/sin2/cos2 removed from Conv2D
        #    input. Exp 39 feature importance: time_of_day_sin2 ranked #1 — the Conv2D
        #    was shortcutting on "what time of day is it?" across 180 timesteps instead
        #    of learning physical sensor dynamics. Diurnal encodings still reach the
        #    anchor path (input[:, -1, :] covers all n_features).
        #
        # 2) Lag removal: temp_lag60, temp_lag120, temp_lag180, temp_delta_1 removed
        #    from anchor. With 251-min RF the raw sequence already contains t-60/t-120/
        #    t-180; the Conv2D can learn implicit lag relationships rather than reading
        #    anchor shortcuts.
        #
        # Stream A — Conv2D path (physical sensors only, no diurnal/lag/slope):
        #   input[:, :, :n_conv] → Reshape(SEQ_LEN, n_conv, 1)
        #   → Conv2D(96) dilated blocks → GAP → Dense(64) → context(64)
        # Stream B — Anchor path (purely instantaneous: raw sensors + diurnal + slopes):
        #   input[:, -1, :] → Dense(32) → ReLU6 → anchor(32)
        # Concatenate([context(64), anchor(32)]) → Dense(32) → ReLU6 → 3 heads
        # ---------------------------------------------------------------------

        FILTERS = 96  # up from 64 — more capacity in Conv2D temporal path

        # Use MirroredStrategy when multiple GPUs are available (transparent no-op on single GPU)
        _n_gpus = len(tf.config.list_physical_devices('GPU'))
        strategy = tf.distribute.MirroredStrategy() if _n_gpus > 1 else tf.distribute.get_strategy()
        if _n_gpus > 1:
            print(f"Using MirroredStrategy across {_n_gpus} GPUs")

        with strategy.scope():
            # Input: (batch, SEQ_LEN, n_features)
            input_layer = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")

            # --- Anchor path: last timestep raw features (STRIDED_SLICE, Edge TPU ✅) ---
            anchor_vec = LastTimestepAnchorFeatures(name="last_ts_anchors")(input_layer)
            anchor_vec = tf.keras.layers.Dense(32, use_bias=False, kernel_regularizer=l2(1e-4), name="dense_anchor")(anchor_vec)
            anchor_vec = tf.keras.layers.ReLU(max_value=6.0, name="relu_anchor")(anchor_vec)

            # Slice conv features for Conv2D path (StridedSlice — Edge TPU ✅)
            # Diurnal and engineered features are excluded: Conv2D must learn temporal
            # dynamics from physical sensor readings without time-of-day shortcuts.
            raw_input = input_layer[:, :, :n_conv]

            # Reshape to 4D: (batch, SEQ_LEN, n_conv, 1) — single-channel "image"
            x = tf.keras.layers.Reshape((SEQ_LEN, n_conv, 1), name="reshape")(raw_input)

            # Block 1: short temporal patterns (3 timesteps, independent per feature)
            x = tf.keras.layers.Conv2D(FILTERS, kernel_size=(3, 1), padding='same', use_bias=False, kernel_regularizer=l2(1e-4), name="conv2d_t1")(x)
            x = tf.keras.layers.BatchNormalization(name="bn_t1")(x)
            x = tf.keras.layers.ReLU(max_value=6.0, name="relu_t1")(x)

            # Block 2: medium temporal patterns — dilation=4 expands RF from 9 → 27 min
            x = tf.keras.layers.Conv2D(FILTERS, kernel_size=(7, 1), dilation_rate=(4, 1), padding='same', use_bias=False, kernel_regularizer=l2(1e-4), name="conv2d_t2")(x)
            x = tf.keras.layers.BatchNormalization(name="bn_t2")(x)
            x = tf.keras.layers.ReLU(max_value=6.0, name="relu_t2")(x)

            # Block 3: long temporal patterns — dilation=16 expands RF from 23 → ~251 min (~4 hrs)
            x = tf.keras.layers.Conv2D(FILTERS, kernel_size=(15, 1), dilation_rate=(16, 1), padding='same', use_bias=False, kernel_regularizer=l2(1e-4), name="conv2d_t3")(x)
            x = tf.keras.layers.BatchNormalization(name="bn_t3")(x)
            x = tf.keras.layers.ReLU(max_value=6.0, name="relu_t3")(x)

            # Block 4: feature mixing — kernel spans all n_conv conv features, collapses feature dim
            # Output shape: (batch, SEQ_LEN, 1, FILTERS)
            x = tf.keras.layers.Conv2D(FILTERS, kernel_size=(1, n_conv), padding='valid', use_bias=False, kernel_regularizer=l2(1e-4), name="conv2d_feat")(x)
            x = tf.keras.layers.BatchNormalization(name="bn_feat")(x)
            x = tf.keras.layers.ReLU(max_value=6.0, name="relu_feat")(x)

            # GlobalAveragePooling2D — averages all 180 × 1 positions.
            # Reverted from Exp 34's multi-point extraction: GAP proved necessary because
            # the conv temporal receptive field (~25 steps) cannot capture 60/120-min context
            # from point extractions at t=−61/−121. MEAN op — Edge TPU-compatible ✅
            x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)  # (batch, FILTERS=96)

            # Context vector: project FILTERS-dim → 64-dim
            x = tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=l2(1e-4), name="dense_context")(x)
            x = tf.keras.layers.ReLU(max_value=6.0, name="relu_context")(x)

            merged = tf.keras.layers.Concatenate(name="context_anchor_concat")([x, anchor_vec])
            merged = tf.keras.layers.Dense(32, use_bias=False, kernel_regularizer=l2(1e-4), name="dense_head")(merged)
            merged = tf.keras.layers.ReLU(max_value=6.0, name="relu_head")(merged)

            output_1 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype="float32", name='diff_1hr')(merged)
            output_2 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype="float32", name='diff_2hr')(merged)
            output_3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype="float32", name='diff_3hr')(merged)
            model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

            # Pass initial LR as a plain float — Keras 3 Adam requires float, LearningRateSchedule,
            # or callable; tf.Variable is not accepted. ReduceLRCallback reads/writes the
            # optimizer's internal learning_rate variable directly via self.model.optimizer.learning_rate.
            initial_lr = 1e-4
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics={
                    'diff_1hr': 'mae',
                    'diff_2hr': 'mae',
                    'diff_3hr': 'mae'
                },
                # steps_per_execution=1: each RunSync covers one mini-batch, returning control
                # to Python between every batch. This gives Metal natural scheduling gaps
                # (equivalent to the CPU-op fallbacks that mixed_float16 incidentally provided)
                # so WindowServer can use the GPU without preempting TF mid-dispatch.
                # Previously set to 20 (bundling 20 batches per RunSync) but that sustained
                # ~90% GPU utilisation and caused indefinite hangs on macOS when the Mac was
                # in interactive use. Overhead cost is negligible vs the ~940s epoch time.
                # Note: steps_per_execution>1 on CUDA/T4 caused 20+ min XLA graph compilation
                # with no benefit — CUDA dispatch overhead is negligible unlike Metal.
                steps_per_execution=1,
            )
        model.summary()

        # Check for existing checkpoints to resume training
        initial_epoch = 0
        # On Kaggle the restore block copies into ./checkpoints/ before training;
        # locally we use a name-scoped directory so exp38 and exp39 don't collide.
        checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
        es_state_path = os.path.join(checkpoint_dir, "early_stopping_state.json")
        lr_state_path = os.path.join(checkpoint_dir, "lr_state.json")
        
        if os.path.exists(checkpoint_dir):
            # Prefer full model checkpoints (more reliable) over weights-only
            # Look for both periodic checkpoints and near-end-of-epoch checkpoints
            full_model_checkpoints = [
                f for f in glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.h5"))
                if not f.endswith('.weights.h5')
            ]
            weights_checkpoints = glob.glob(os.path.join(checkpoint_dir, "model_*.weights.h5"))
            best_weights_checkpoint = os.path.join(checkpoint_dir, "best_model.weights.h5")
            
            checkpoint_to_load = None
            latest_epoch = 0
            checkpoint_type = None
            
            # Priority 1: Full model checkpoints (most reliable)
            if full_model_checkpoints:
                def extract_epoch_full(fpath):
                    # Handle both "model_epoch_XX.h5" and "model_epoch_XX_batch_YY.h5" formats
                    match = re.search(r'model_epoch_(\d+)(?:_batch_\d+)?\.h5', os.path.basename(fpath))
                    return int(match.group(1)) if match else 0
                full_model_checkpoints.sort(key=extract_epoch_full, reverse=True)
                checkpoint_to_load = full_model_checkpoints[0]
                latest_epoch = extract_epoch_full(checkpoint_to_load)
                checkpoint_type = "full_model"
                print(f"\n🔍 Found {len(full_model_checkpoints)} full model checkpoint(s).")
                print(f"   Latest: {os.path.basename(checkpoint_to_load)} (epoch {latest_epoch})")
            
            # Priority 2: Latest-epoch weights (saved by LatestEpochSaver after every epoch).
            # Preferred over best_model.weights.h5 because it always reflects the most
            # recently completed epoch, not just the epoch with the best val_loss.
            elif (os.path.exists(os.path.join(checkpoint_dir, "model_latest.weights.h5")) and
                  os.path.exists(os.path.join(checkpoint_dir, "model_latest_epoch.json")) and
                  not full_model_checkpoints):
                checkpoint_to_load = os.path.join(checkpoint_dir, "model_latest.weights.h5")
                checkpoint_type = "latest_weights"
                try:
                    with open(os.path.join(checkpoint_dir, "model_latest_epoch.json")) as f:
                        latest_epoch = int(json.load(f).get("epoch", 0))
                except Exception:
                    latest_epoch = 0
                print(f"\n🔍 Found latest-epoch weights checkpoint (epoch {latest_epoch}).")

            # Priority 3: Best weights checkpoint (if exists and no full model found)
            elif os.path.exists(best_weights_checkpoint) and not full_model_checkpoints:
                checkpoint_to_load = best_weights_checkpoint
                checkpoint_type = "best_weights"
                # Try to get epoch from results file
                results_file = os.path.join(RESULTS_DIR, f"results_5b_{name}.json")
                if os.path.exists(results_file):
                    try:
                        with open(results_file, "r") as f:
                            results = json.load(f)
                            latest_epoch = results.get("best_epoch", 0)
                    except:
                        pass
                print(f"\n🔍 Found best weights checkpoint: {os.path.basename(checkpoint_to_load)}")
                if latest_epoch > 0:
                    print(f"   Best epoch from results: {latest_epoch}")

            # Priority 4: Legacy weights checkpoints (fallback)
            elif weights_checkpoints and not full_model_checkpoints:
                def extract_epoch_weights(fpath):
                    match = re.search(r'model_(\d+)\.weights\.h5', os.path.basename(fpath))
                    return int(match.group(1)) if match else 0
                weights_checkpoints.sort(key=extract_epoch_weights, reverse=True)
                checkpoint_to_load = weights_checkpoints[0]
                latest_epoch = extract_epoch_weights(checkpoint_to_load)
                checkpoint_type = "weights"
                print(f"\n🔍 Found {len(weights_checkpoints)} legacy weights checkpoint(s).")
                print(f"   Latest: {os.path.basename(checkpoint_to_load)} (epoch {latest_epoch})")
                print(f"   ⚠️  Note: These may not be compatible with current model architecture")
            
            if checkpoint_to_load:
                print(f"📥 Attempting to load {checkpoint_type} checkpoint to resume from epoch {latest_epoch + 1}...")

                try:
                    if checkpoint_type == "full_model":
                        # Try each checkpoint from newest to oldest until one matches the
                        # current architecture. This handles leftover checkpoints from a
                        # previous experiment (different architecture) sitting alongside
                        # checkpoints from the current run.
                        loaded_checkpoint = None
                        for candidate in full_model_checkpoints:  # already sorted newest→oldest
                            candidate_epoch = extract_epoch_full(candidate)
                            print(f"   🔍 Validating {os.path.basename(candidate)}...")
                            can_load, validation_error = validate_checkpoint_loading(
                                candidate, model, CUSTOM_OBJECTS
                            )
                            if can_load:
                                loaded_checkpoint = candidate
                                latest_epoch = candidate_epoch
                                break
                            print(f"   ⚠️  Skipping epoch {candidate_epoch}: {validation_error}")

                        if loaded_checkpoint is None:
                            print(f"   ⚠️  No compatible checkpoint found among {len(full_model_checkpoints)} candidate(s).")
                            print(f"   Starting training from scratch with current architecture...")
                            initial_epoch = 0
                        else:
                            print(f"   ✅ Loading: {os.path.basename(loaded_checkpoint)} (epoch {latest_epoch})")
                            loaded_model = tf.keras.models.load_model(
                                loaded_checkpoint,
                                compile=False,
                                custom_objects=CUSTOM_OBJECTS
                            )
                            model.set_weights(loaded_model.get_weights())
                            del loaded_model
                            initial_epoch = latest_epoch
                            print(f"   ✅ Resumed. Training will continue from epoch {initial_epoch + 1}.")
                    else:
                        # weights-only checkpoint
                        model.load_weights(checkpoint_to_load)
                        initial_epoch = latest_epoch
                        print(f"✅ Successfully loaded weights checkpoint from epoch {latest_epoch}")

                    if initial_epoch > 0:
                        if os.path.exists(lr_state_path):
                            try:
                                with open(lr_state_path) as f:
                                    lr_state = json.load(f)
                                restored_lr = float(lr_state["lr"])
                                # Apply immediately to the optimizer so epoch 1 of the resume
                                # trains at the correct LR, not the initial 1e-4.
                                lr_var = getattr(optimizer, '_learning_rate', None)
                                if lr_var is not None and hasattr(lr_var, 'assign'):
                                    lr_var.assign(restored_lr)
                                else:
                                    optimizer.learning_rate = restored_lr
                                print(f"   ✅ Restored learning rate: {restored_lr:.2e} "
                                      f"(ReduceLR best={lr_state['best']:.6f}, wait={lr_state['wait']})")
                            except Exception as e:
                                print(f"   ⚠️  Could not restore LR state: {e} — using initial LR {initial_lr}")
                        else:
                            print(f"   Note: No LR state found — optimizer LR starts at {initial_lr}")

                except (ValueError, RuntimeError, OSError, TypeError) as e:
                    error_msg = str(e)
                    print(f"⚠️  Warning: Could not load checkpoint: {error_msg}")
                    print("   Starting training from scratch...")
                    initial_epoch = 0

        # Clear stale early stopping, LR, and latest-epoch state when starting from scratch.
        if initial_epoch == 0 and os.path.exists(es_state_path):
            os.remove(es_state_path)
            print("   🗑️  Cleared stale early stopping state (starting fresh)")
        if initial_epoch == 0 and os.path.exists(lr_state_path):
            os.remove(lr_state_path)
            print("   🗑️  Cleared stale LR state (starting fresh)")
        _latest_w = os.path.join(checkpoint_dir, "model_latest.weights.h5")
        _latest_m = os.path.join(checkpoint_dir, "model_latest_epoch.json")
        if initial_epoch == 0:
            for _f in (_latest_w, _latest_m):
                if os.path.exists(_f):
                    os.remove(_f)
                    print(f"   🗑️  Cleared stale latest-epoch state (starting fresh)")

        # =====================================================================
        # PRE-TRAINING VALIDATION SUITE: Fail fast and early
        # =====================================================================
        # Comprehensive validation before training starts to catch issues
        # immediately. Since each epoch takes ~1200 seconds, we want to catch
        # problems BEFORE the first epoch, not after.
        # =====================================================================
        
        def pre_training_validation(model, train_ds, val_ds, checkpoint_dir, custom_objects):
            """
            Comprehensive pre-training validation to catch issues before training starts.
            Returns (all_passed: bool, errors: list of error messages)
            """
            errors = []
            warnings = []
            
            print("\n" + "="*80)
            print("🔍 PRE-TRAINING VALIDATION SUITE")
            print("="*80)
            
            # 1. Validate model can be built and compiled
            print("\n1️⃣  Validating model architecture...")
            try:
                # Test forward pass with dummy data
                dummy_input = tf.zeros((1, SEQ_LEN, n_features), dtype=tf.float32)
                dummy_output = model(dummy_input, training=False)
                
                # Verify output structure
                if not isinstance(dummy_output, (list, tuple)) or len(dummy_output) != 3:
                    errors.append(f"Model output structure invalid: expected 3 outputs, got {len(dummy_output) if isinstance(dummy_output, (list, tuple)) else 'non-list'}")
                else:
                    print("   ✅ Model forward pass successful")
                    print(f"   ✅ Output shapes: {[out.shape for out in dummy_output]}")
            except Exception as e:
                errors.append(f"Model forward pass failed: {str(e)[:200]}")
            
            # 2. Validate datasets
            print("\n2️⃣  Validating datasets...")
            try:
                train_cardinality = train_ds.cardinality().numpy()
                val_cardinality = val_ds.cardinality().numpy()
                
                if train_cardinality < 0:
                    # .repeat() makes cardinality infinite (-1); use the pre-computed count
                    print(f"   ✅ Training dataset: {train_steps} batches/epoch (infinite via .repeat())")
                else:
                    print(f"   ✅ Training dataset: {train_cardinality} batches")
                
                if val_cardinality < 0:
                    warnings.append("Validation dataset cardinality unknown (may be fine for streaming)")
                else:
                    print(f"   ✅ Validation dataset: {val_cardinality} batches")
                
                # Test dataset iteration
                try:
                    sample_batch = next(iter(train_ds))
                    if len(sample_batch) != 2:
                        errors.append(f"Training dataset batch structure invalid: expected (x, y), got {len(sample_batch)} elements")
                    else:
                        x_sample, y_sample = sample_batch
                        if x_sample.shape[1] != SEQ_LEN:
                            errors.append(f"Training batch sequence length mismatch: expected {SEQ_LEN}, got {x_sample.shape[1]}")
                        if x_sample.shape[2] != n_features:
                            errors.append(f"Training batch feature count mismatch: expected {n_features}, got {x_sample.shape[2]}")
                        if not isinstance(y_sample, (list, tuple)) or len(y_sample) != 3:
                            errors.append(f"Training batch target structure invalid: expected 3 targets, got {len(y_sample) if isinstance(y_sample, (list, tuple)) else 'non-list'}")
                        else:
                            print("   ✅ Training dataset iteration successful")
                except Exception as e:
                    errors.append(f"Training dataset iteration failed: {str(e)[:200]}")
                
            except Exception as e:
                errors.append(f"Dataset validation failed: {str(e)[:200]}")
            
            # 3. Validate checkpoint directory
            print("\n3️⃣  Validating checkpoint directory...")
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Test write permissions
                test_file = os.path.join(checkpoint_dir, ".write_test")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    print("   ✅ Checkpoint directory writable")
                except Exception as e:
                    errors.append(f"Checkpoint directory not writable: {str(e)[:200]}")
                
                # Check disk space (at least 1GB free)
                try:
                    import shutil
                    total, used, free = shutil.disk_usage(checkpoint_dir)
                    free_gb = free / (1024**3)
                    if free_gb < 1.0:
                        warnings.append(f"Low disk space: {free_gb:.2f} GB free (recommend at least 1 GB)")
                    else:
                        print(f"   ✅ Disk space: {free_gb:.2f} GB free")
                except Exception:
                    warnings.append("Could not check disk space")
                    
            except Exception as e:
                errors.append(f"Checkpoint directory validation failed: {str(e)[:200]}")
            
            # 4. Validate custom objects are properly registered
            print("\n4️⃣  Validating custom objects registration...")
            try:
                if not custom_objects:
                    warnings.append("No custom objects registered (this may be fine if model has no custom layers)")
                else:
                    for name, obj in custom_objects.items():
                        if not hasattr(obj, 'get_config'):
                            errors.append(f"Custom object '{name}' missing get_config() method (required for serialization)")
                        else:
                            print(f"   ✅ Custom object '{name}' properly registered")
            except Exception as e:
                errors.append(f"Custom objects validation failed: {str(e)[:200]}")
            
            # 5. Serialization test skipped.
            # save_weights() reads all GPU tensors even before training begins.
            # On macOS Metal this partial flush corrupts the command queue,
            # causing the first epoch to run at 787ms/step instead of 218ms/step.
            # Checkpoint integrity is proven by the training loop's own save/load
            # cycle; no pre-flight test needed.
            print("\n5️⃣  Model serialization test: skipped (save_weights() on Metal corrupts pipeline)")
            print("   ℹ️  Checkpoint integrity verified by the training loop directly.")
            
            # 6. Validate checkpoint loading if resuming
            print("\n6️⃣  Validating checkpoint resumability (if applicable)...")
            # This is already done in the checkpoint loading code, but we'll note it here
            print("   ℹ️  Checkpoint validation will occur during checkpoint loading")
            
            # Summary
            print("\n" + "="*80)
            if errors:
                print("❌ PRE-TRAINING VALIDATION FAILED")
                print("="*80)
                print("Errors found:")
                for i, error in enumerate(errors, 1):
                    print(f"  {i}. {error}")
                if warnings:
                    print("\nWarnings:")
                    for i, warning in enumerate(warnings, 1):
                        print(f"  {i}. ⚠️  {warning}")
                print("="*80)
                return False, errors
            else:
                print("✅ PRE-TRAINING VALIDATION PASSED")
                if warnings:
                    print("\nWarnings (non-critical):")
                    for i, warning in enumerate(warnings, 1):
                        print(f"  {i}. ⚠️  {warning}")
                print("="*80)
                return True, []
        
        # Run pre-training validation
        validation_passed, validation_errors = pre_training_validation(
            model, train_ds, val_ds, checkpoint_dir, CUSTOM_OBJECTS
        )
        
        if not validation_passed:
            print("\n" + "="*80)
            print("🛑 TRAINING ABORTED: Pre-training validation failed")
            print("="*80)
            print("Fix the errors above before attempting to train.")
            print("This prevents wasting time on a training run that will fail.")
            print("="*80 + "\n")
            raise RuntimeError(f"Pre-training validation failed: {'; '.join(validation_errors[:3])}")

        # =====================================================================
        # Exp 28: QAT disabled — Conv2D + skip path should survive PTQ with ReLU6.
        # Re-enable only if PTQ quantized accuracy is poor.
        ENABLE_QAT = False

        if ENABLE_QAT:
            import tensorflow_model_optimization as tfmot

        QAT_ACTIVE = False
        QAT_LR = 1e-5
        QAT_EPOCHS = 50

        if ENABLE_QAT:
            try:
                print("\n🔢 Applying Quantization-Aware Training (QAT) wrapping...")

                # Load the best float weights (lowest val_loss) before QAT wrapping.
                # The checkpoint loader above may have loaded the LATEST epoch (e.g., 65)
                # but we want the BEST epoch (e.g., 46, val_loss=0.001343) as the starting
                # point for QAT fine-tuning.
                _best_w = os.path.join(checkpoint_dir, "best_model.weights.h5")
                if os.path.exists(_best_w):
                    model.load_weights(_best_w)
                    print(f"✅ Loaded best float weights from {_best_w} for QAT fine-tuning")
                else:
                    print("⚠️  best_model.weights.h5 not found — using currently loaded checkpoint")

                def _annotate_fn(layer):
                    # Exp 27: Conv2D architecture has no custom ops — annotate all layers.
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)

                annotated = tf.keras.models.clone_model(model, clone_function=_annotate_fn)
                # Copy float weights before quantize_apply adds fake-quant variables
                annotated.set_weights(model.get_weights())

                with tfmot.quantization.keras.quantize_scope({}):
                    model = tfmot.quantization.keras.quantize_apply(annotated)

                # Recompile at QAT learning rate (lower than float training to preserve accuracy)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LR),
                    loss='mse',
                    metrics={'diff_1hr': 'mae', 'diff_2hr': 'mae', 'diff_3hr': 'mae'},
                    steps_per_execution=1,
                )
                model.summary()

                initial_lr = QAT_LR   # ReduceLRCallback reads this below
                initial_epoch = 0     # Always fine-tune from epoch 0 for QAT
                QAT_ACTIVE = True
                print(f"✅ QAT applied. Float weights preserved. Fine-tuning at LR={QAT_LR:.0e} for up to {QAT_EPOCHS} epochs.")

            except Exception as e:
                print(f"\n⚠️  QAT wrapping failed: {e}")
                print("   Falling back to PTQ (post-training quantization).")

        # patience=30 for Exp 33 float training; 15 for QAT fine-tuning
        early_stopping = EarlyStopping(monitor='val_task_loss', patience=15 if QAT_ACTIVE else 30, restore_best_weights=True, mode='min')

        # Restore early stopping state across restarts so patience is cumulative.
        # Without this, each restart resets wait=0 and the model gets a fresh
        # 25-epoch budget, preventing early stopping from ever firing.
        if initial_epoch > 0 and not QAT_ACTIVE and os.path.exists(es_state_path):
            try:
                with open(es_state_path) as f:
                    es_state = json.load(f)
                early_stopping.best = es_state["best"]
                early_stopping.wait = es_state["wait"]
                print(f"   ✅ Restored early stopping state: best={es_state['best']:.6f}, "
                      f"wait={es_state['wait']}/{early_stopping.patience}")
                if early_stopping.wait >= early_stopping.patience:
                    print(f"   ⚠️  wait ≥ patience — early stopping will fire after the first epoch")
            except Exception as e:
                print(f"   ⚠️  Could not restore early stopping state: {e}")

        # Save best weights checkpoint (for resuming - weights only for smaller size)
        # This is updated every epoch if validation loss improves
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_task_loss",
            mode="min",
            verbose=0
        )

        # ALSO save best full model as backup (dual backup strategy)
        # This ensures we always have both formats available
        best_full_model_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model_full.weights.h5"),
            save_weights_only=True,   # Weights-only avoids optimizer lambda serialization warning
            save_best_only=True,
            monitor="val_task_loss",
            mode="min",
            verbose=0
        )
        
        class EarlyStoppingStateSaver(tf.keras.callbacks.Callback):
            """Persists EarlyStopping .best and .wait after each epoch.

            Ensures that patience is cumulative across watchdog restarts rather
            than resetting to zero each time training resumes from a checkpoint.
            """
            def __init__(self, early_stopping_cb, state_path):
                super().__init__()
                self.es = early_stopping_cb
                self.state_path = state_path

            def on_epoch_end(self, epoch, logs=None):
                try:
                    with open(self.state_path, "w") as f:
                        json.dump({"best": float(self.es.best), "wait": int(self.es.wait)}, f)
                except Exception as e:
                    print(f"\n⚠️  Could not save early stopping state: {e}")

        class LRStateSaver(tf.keras.callbacks.Callback):
            """Persists ReduceLRCallback lr/best/wait after each epoch for resume."""
            def __init__(self, reduce_lr_cb, state_path):
                super().__init__()
                self.reduce_lr = reduce_lr_cb
                self.state_path = state_path

            def on_epoch_end(self, epoch, logs=None):
                try:
                    current_lr = self.reduce_lr._get_lr()
                    with open(self.state_path, "w") as f:
                        json.dump({
                            "lr": float(current_lr),
                            "best": float(self.reduce_lr.best),
                            "wait": int(self.reduce_lr.wait),
                        }, f)
                except Exception as e:
                    print(f"\n⚠️  Could not save LR state: {e}")

        class LatestEpochSaver(tf.keras.callbacks.Callback):
            """Single-GPU-call epoch checkpoint: saves weights + tracks best model.

            Replaces separate checkpoint_cb and best_full_model_cb ModelCheckpoint
            callbacks.  Those callbacks reset their internal 'best' to float('inf')
            on every process restart, causing two extra save_weights() calls on the
            first epoch.  Combined with the GPU read in CheckpointValidationCallback,
            that was 4 Metal GPU interactions per epoch-end — enough to corrupt the
            Metal command queue and deadlock batch 0 of the next epoch.

            This callback does exactly one save_weights() per epoch, then copies
            the file to best_model.weights.h5 if val_task_loss improved (shutil.copy2
            is pure I/O, no GPU).  'best_task_loss' is seeded from the restored
            EarlyStopping state so the threshold carries across restarts.

            Monitors val_task_loss (sum of per-head losses, no L2) rather than
            val_loss.  For dilated Conv2D architectures L2 grows with weight
            magnitudes, inflating val_loss throughout training and making it an
            unreliable checkpoint criterion.  TaskLossLogger must run before this
            callback in the callback list.
            """
            def __init__(self, checkpoint_dir, initial_best_task_loss=float('inf')):
                super().__init__()
                self.weights_path   = os.path.join(checkpoint_dir, "model_latest.weights.h5")
                self.meta_path      = os.path.join(checkpoint_dir, "model_latest_epoch.json")
                self.best_path      = os.path.join(checkpoint_dir, "best_model.weights.h5")
                self.best_task_loss = initial_best_task_loss

            def on_epoch_end(self, epoch, logs=None):
                try:
                    self.model.save_weights(self.weights_path)
                    with open(self.meta_path, "w") as f:
                        json.dump({"epoch": epoch + 1}, f)
                except Exception as e:
                    print(f"\n⚠️  Could not save latest-epoch checkpoint: {e}")
                    return
                task_loss = (logs or {}).get("val_task_loss", float('inf'))
                if task_loss < self.best_task_loss:
                    self.best_task_loss = task_loss
                    try:
                        import shutil
                        shutil.copy2(self.weights_path, self.best_path)
                    except Exception as e:
                        print(f"\n⚠️  Could not update best-model checkpoint: {e}")

        class MaxEpochsPerRun(tf.keras.callbacks.Callback):
            """Stop training voluntarily after N epochs for a clean process restart.

            On macOS Metal, every save_weights() call at epoch-end reads GPU tensors
            and accumulates state in the Metal command queue.  After 1-2 epochs the
            accumulated state deadlocks batch 0 of the next epoch.  The only reliable
            fix is to exit the process after each epoch so the OS gives the next run a
            completely fresh Metal GPU context.

            This callback sets model.stop_training = True after `max_epochs_per_run`
            epochs.  The training script then exits with code 42, which the restart
            loop (train_loop.sh) treats as "restart needed, not done" and immediately
            re-launches.  Code 0 means training is truly complete.
            """
            def __init__(self, max_epochs_per_run=1):
                super().__init__()
                self.max_epochs_per_run = max_epochs_per_run
                self.epochs_run = 0
                self.triggered = False

            def on_epoch_end(self, epoch, logs=None):
                self.epochs_run += 1
                if self.epochs_run >= self.max_epochs_per_run:
                    print(f"\n🔄 MaxEpochsPerRun: clean stop after {self.epochs_run} epoch(s).")
                    print(f"   Restarting process for a fresh Metal GPU context.")
                    self.triggered = True
                    self.model.stop_training = True

        class TaskLossLogger(tf.keras.callbacks.Callback):
            """Logs val_task_loss = sum of per-head task losses (no L2 regularization).

            val_loss includes L2 from all Dense and Conv2D layers, which grows as
            weights increase during training.  For dilated Conv2D kernels this
            inflates val_loss ~6-7x relative to the per-head MSE sum, making it an
            unreliable signal for checkpointing, early stopping, and LR scheduling.

            Must be listed FIRST in active_callbacks so val_task_loss is in logs
            when EarlyStopping, LatestEpochSaver, and ReduceLRCallback execute.
            Falls back to val_loss if the per-head metric keys are absent.
            """
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    return
                keys = ('val_diff_1hr_loss', 'val_diff_2hr_loss', 'val_diff_3hr_loss')
                if all(k in logs for k in keys):
                    logs['val_task_loss'] = sum(logs[k] for k in keys)
                else:
                    logs['val_task_loss'] = logs.get('val_loss', float('inf'))

        es_state_saver = EarlyStoppingStateSaver(early_stopping, es_state_path)

        # Adaptive learning rate: reduce LR when validation loss plateaus.
        # Reads/writes the optimizer's internal learning_rate variable directly —
        # no tf.Variable wrapper needed, no lambda, no serialization warning.
        class ReduceLRCallback(tf.keras.callbacks.Callback):
            def __init__(self, initial_lr, monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1):
                super().__init__()
                self.initial_lr = initial_lr
                self.monitor = monitor
                self.factor = factor
                self.patience = patience
                self.min_lr = min_lr
                self.verbose = verbose
                self.best = float('inf')
                self.wait = 0

            def _unwrap_optimizer(self, opt):
                """Unwrap LossScaleOptimizer to reach the base Adam.
                TF 2.x stores the inner optimizer as _optimizer;
                Keras 3 uses inner_optimizer. Fall back to opt itself."""
                for attr in ('_optimizer', 'inner_optimizer'):
                    inner = getattr(opt, attr, None)
                    if inner is not None:
                        return inner
                return opt

            def _get_lr(self):
                opt = self.model.optimizer
                inner = self._unwrap_optimizer(opt)
                lr_var = getattr(inner, '_learning_rate', None)
                if lr_var is not None and hasattr(lr_var, 'numpy'):
                    return float(lr_var.numpy())
                # Fallback: public learning_rate property
                try:
                    lr = inner.learning_rate
                    if hasattr(lr, 'numpy'):
                        return float(lr.numpy())
                    return float(lr)
                except Exception:
                    return float(self.initial_lr)

            def _set_lr(self, new_lr):
                opt = self.model.optimizer
                inner = self._unwrap_optimizer(opt)
                lr_var = getattr(inner, '_learning_rate', None)
                if lr_var is not None and hasattr(lr_var, 'assign'):
                    lr_var.assign(float(new_lr))
                else:
                    inner.learning_rate = float(new_lr)
                # Verify the assignment actually took effect
                confirmed = self._get_lr()
                if abs(confirmed - float(new_lr)) > 1e-10:
                    print(f'\n⚠️  LR assignment may not have stuck: set {new_lr:.2e}, read back {confirmed:.2e}', flush=True)
                else:
                    print(f'\n✅ LR confirmed: {confirmed:.2e}', flush=True)

            def on_epoch_end(self, epoch, logs=None):
                current = logs.get(self.monitor)
                if current is None:
                    return
                current_lr = self._get_lr()
                if current < self.best:
                    self.best = current
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        if current_lr > self.min_lr:
                            new_lr = max(current_lr * self.factor, self.min_lr)
                            self._set_lr(new_lr)
                            current_lr = new_lr  # reflect the new LR in the epoch summary
                            if self.verbose:
                                print(f'\nEpoch {epoch + 1}: ReduceLRCallback reducing learning rate to {new_lr:.2e}. (wait={self.wait}, best={self.best:.6f})', flush=True)
                        self.wait = 0
                # Log LR AFTER any reduction so the epoch summary shows the new value
                if logs is not None:
                    logs['learning_rate'] = current_lr

        # Exp 32: ReduceLROnPlateau for both float and QAT training.
        reduce_lr = ReduceLRCallback(
            initial_lr=initial_lr,
            monitor='val_task_loss',
            factor=0.5,
            patience=12,  # Exp 35: extended from 8 → 12 to allow more epochs per LR level
            min_lr=1e-7,
            verbose=1,
        )
        lr_state_saver = LRStateSaver(reduce_lr, lr_state_path)

        # Restore ReduceLR plateau state so patience is cumulative across restarts.
        if initial_epoch > 0 and os.path.exists(lr_state_path):
            try:
                with open(lr_state_path) as f:
                    lr_state = json.load(f)
                reduce_lr.best = float(lr_state["best"])
                reduce_lr.wait = int(lr_state["wait"])
                print(f"   ✅ Restored ReduceLR state: best={reduce_lr.best:.6f}, wait={reduce_lr.wait}/{reduce_lr.patience}")
            except Exception as e:
                print(f"   ⚠️  Could not restore ReduceLR plateau state: {e}")

        # Training watchdog: detects hangs during training
        # Timeout: 10 minutes without any progress (increased to account for checkpoint validation)
        # Check interval: 30 seconds (how often to check for hangs)
        watchdog = TrainingWatchdog(
            timeout_minutes=30,  # Base timeout — adaptive will scale up from here based on actual batch times
            check_interval_seconds=30,
            verbose=1,
            train_ds=train_ds,
            total_batches_hint=train_steps  # train_ds.cardinality() is -1 after .repeat()
        )
        
        # CheckpointValidationCallback removed from active_callbacks: its model.get_weights()
        # preamble added an extra Metal GPU read every epoch, contributing to the command-queue
        # corruption that hangs batch 0 of the next epoch on macOS Metal.

        # Seed from restored EarlyStopping.best (a val_task_loss value after change) so the
        # threshold carries across restarts without overwriting best_model.weights.h5 on epoch 1.
        _initial_best = getattr(early_stopping, 'best', float('inf'))
        if not isinstance(_initial_best, float) or _initial_best != _initial_best:  # nan check
            _initial_best = float('inf')
        latest_epoch_saver = LatestEpochSaver(checkpoint_dir, initial_best_task_loss=_initial_best)
        task_loss_logger = TaskLossLogger()
        # max_epochs_per_run: how many epochs to train before a clean process restart.
        # Each restart pays ~6 min overhead (JIT warmup + checkpoint load + validation).
        # Lower = safer (less Metal accumulation); higher = faster (overhead amortised).
        # With screen unlocked / caffeinate -d -i: try 5–10.
        # With screen locked (default macOS GPU power-down): keep at 1.
        max_epochs_per_run = MaxEpochsPerRun(max_epochs_per_run=9999 if (KAGGLE_MODE or force_cpu) else 99)

        # Match RPi5 configuration exactly
        # Note: The watchdog callback will monitor training and detect hangs
        # If a hang is detected, it will write diagnostic info to training_hang_detected.json
        # and attempt to interrupt training via SIGINT
        
        # Check if we need to train (if initial_epoch >= epochs, training is already complete)
        # Exp 33: raised from 100 → 150. Exp 32 hit the 100-epoch cap before
        # EarlyStopping(patience=30) could fire — each LR reduction produced a
        # tiny val_loss improvement that reset EarlyStopping's wait counter, pushing
        # the expected stop epoch past 100. 150 gives EarlyStopping room to fire
        # regardless of how many LR reductions occur.
        max_epochs = QAT_EPOCHS if QAT_ACTIVE else 150
        history = None
        
        if initial_epoch >= max_epochs:
            print(f"\n✅ Training already complete (checkpoint is from epoch {initial_epoch}, max is {max_epochs})")
            print(f"   Skipping training and using existing checkpoint for evaluation")
            # Clear early stopping and LR state — training is done, next run should start fresh
            if os.path.exists(es_state_path):
                os.remove(es_state_path)
            if os.path.exists(lr_state_path):
                os.remove(lr_state_path)
            # Create empty history dict to avoid KeyError later
            history = type('History', (), {'history': {}})()
        else:
            try:
                if initial_epoch > 0:
                    print(f"\n🔄 Resuming training from epoch {initial_epoch + 1} (max epochs: {max_epochs})")
                # QAT models cannot save in H5 full-model format (QuantizeWrapper
                # serialization requires SavedModel).  Use weights-only checkpoints
                # and skip the full-model periodic and validator callbacks.
                if QAT_ACTIVE:
                    active_callbacks = [
                        task_loss_logger,     # FIRST: logs val_task_loss before other callbacks read it
                        early_stopping,
                        checkpoint_cb,        # weights-only, always works for QAT
                        best_full_model_cb,   # weights-only, always works for QAT
                        latest_epoch_saver,   # weights-only: saves every epoch for reliable resume
                        reduce_lr,
                        watchdog,
                    ]
                else:
                    active_callbacks = [
                        task_loss_logger,     # FIRST: logs val_task_loss before other callbacks read it
                        early_stopping,
                        es_state_saver,
                        latest_epoch_saver,   # 1 save_weights() per epoch; best model via file copy
                        # checkpoint_cb removed: merged into LatestEpochSaver (1 GPU op vs 3).
                        # best_full_model_cb removed: same reason.
                        # checkpoint_validator removed: its model.get_weights() preamble added an
                        # extra Metal GPU read every epoch. Total epoch-end GPU interactions: 1.
                        reduce_lr,
                        lr_state_saver,       # after reduce_lr so saved best/wait reflect this epoch
                        max_epochs_per_run,   # stop after N epochs for clean Metal context on restart
                        watchdog,
                    ]
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=max_epochs,
                    initial_epoch=initial_epoch,
                    steps_per_epoch=train_steps,
                    callbacks=active_callbacks,
                    verbose=2,  # one line per epoch; verbose=1 wastes I/O on 1025 progress bar updates
                )
            except KeyboardInterrupt as e:
                # Check if this was triggered by the watchdog
                try:
                    if watchdog.hang_detected.is_set():
                        print("\n" + "="*80)
                        print("🛑 Training interrupted due to detected hang")
                        print("="*80)
                        print("\nHang diagnostic information saved to: training_hang_detected.json")
                        print("\nThis matches the hang pattern from the stacktrace:")
                        print("  - Main thread blocked in absl::Notification::WaitForNotification()")
                        print("  - Worker threads waiting for work in Eigen::ThreadPool")
                        print("\nSuggested actions:")
                        print("  1. Review training_hang_detected.json for details")
                        print("  2. Check system resources (memory, threads)")
                        print("  3. Try reducing batch size or dataset size")
                        print("  4. Check TensorFlow version compatibility")
                        print("="*80 + "\n")
                        raise RuntimeError("Training hang detected by watchdog") from None
                    else:
                        # User-initiated interrupt
                        raise
                except AttributeError:
                    # Watchdog not accessible, re-raise original interrupt
                    raise e
            except RuntimeError as e:
                # Re-raise RuntimeError from watchdog (or other sources)
                if "hang detected" in str(e).lower():
                    raise
                # For other RuntimeErrors, re-raise as-is
                raise

        # The early stopping callback with restore_best_weights=True already restored the best weights

        # If MaxEpochsPerRun stopped training for a per-epoch Metal reset, exit now
        # before post-training quantization — training is not done yet.
        if max_epochs_per_run.triggered:
            sys.exit(42)

        # Training complete — clear early stopping and LR state so next experiment starts fresh
        if os.path.exists(es_state_path):
            os.remove(es_state_path)
        if os.path.exists(lr_state_path):
            os.remove(lr_state_path)

        eval_results = model.evaluate(val_ds, verbose=0)
        val_loss = eval_results[0]
        val_mae = np.mean(eval_results[1:])  # average MAE across 3 targets
        # Report MAEs in original units (°C difference) using per-horizon scaling
        diff_1_mae_c = eval_results[1] * (y_maxs["temp_diff_1hr"] - y_mins["temp_diff_1hr"]) / 2.0
        diff_2_mae_c = eval_results[2] * (y_maxs["temp_diff_2hr"] - y_mins["temp_diff_2hr"]) / 2.0
        diff_3_mae_c = eval_results[3] * (y_maxs["temp_diff_3hr"] - y_mins["temp_diff_3hr"]) / 2.0
        print(f"\nValidation MAE (in °C difference):")
        print(f"  diff_1hr: {diff_1_mae_c:.2f} °C")
        print(f"  diff_2hr: {diff_2_mae_c:.2f} °C")
        print(f"  diff_3hr: {diff_3_mae_c:.2f} °C")
        baseline_loss = val_loss
        feature_importance = {}
        for feat_idx, feature in enumerate(features):
            # Work on a small in-memory validation window set: X_val_small (num_win, SEQ_LEN, n_features)
            X_val_permuted = copy.deepcopy(X_val_small)
            # Permute this feature across all windows and time steps
            vals = X_val_permuted[:, :, feat_idx]  # shape (num_win, SEQ_LEN)
            flattened = vals.reshape(-1)
            np.random.shuffle(flattened)
            X_val_permuted[:, :, feat_idx] = flattened.reshape(vals.shape)
            perm_ds = tf.data.Dataset.from_tensor_slices(
                (X_val_permuted, (y_val_small[:, 0], y_val_small[:, 1], y_val_small[:, 2]))
            ).batch(256)
            permuted_loss = model.evaluate(perm_ds, verbose=0)[0]
            feature_importance[feature] = permuted_loss - baseline_loss

        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")

        # The early stopping callback with restore_best_weights=True already restored the best weights

        if QAT_ACTIVE:
            # QAT model: quantization parameters (min/max per layer) were learned during
            # training — no representative dataset calibration needed.  Convert directly
            # from the QAT Keras model.
            print("🔧 Converting QAT model to TFLite INT8 (no calibration dataset needed)...")
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                quantized_tflite_model = converter.convert()
            except Exception as qat_conv_err:
                print(f"⚠️  Full INT8 QAT conversion failed: {qat_conv_err}")
                print("   Retrying with float fallback ops allowed...")
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                quantized_tflite_model = converter.convert()
        else:
            # PTQ path: build concrete function with static shape, then calibrate.
            # Output heads are float32 (dtype="float32" on each Dense output layer), so
            # the model produces float32 outputs regardless of mixed precision mode.
            print("🔧 Preparing float32 model for TFLite conversion...")
            # Training is float32 everywhere (mixed_float16 disabled after Exp 37 NaN crashes),
            # so the trained model is already in float32 — no rebuild needed.
            export_model = model

            export_model(tf.zeros((1, SEQ_LEN, n_features), dtype=tf.float32), training=False)

            @tf.function(input_signature=[tf.TensorSpec(shape=[1, SEQ_LEN, n_features], dtype=tf.float32)])
            def model_inference(x):
                return export_model(x, training=False)

            # Convert concrete function to TFLite (ensures static shapes for Edge TPU)
            concrete_func = model_inference.get_concrete_function()
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

            # Disable experimental features that might introduce dynamic shapes
            converter._experimental_lower_tensor_list_ops = False
            # Use legacy converter for better Edge TPU compatibility (avoids dynamic shape issues)
            try:
                converter.experimental_new_converter = False
            except AttributeError:
                pass

            # Use full-integer quantization path suitable for Edge TPU
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            def representative_data_gen():
                num_samples = 500
                n = X_train_flat.shape[0]
                for _ in range(num_samples):
                    idx = np.random.randint(SEQ_LEN, n)
                    window = X_train_flat[idx - SEQ_LEN:idx, :]
                    window = window[np.newaxis, ...].astype(np.float32)
                    yield [window]

            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            try:
                quantized_tflite_model = converter.convert()
            except (RuntimeError, ValueError) as e:
                error_str = str(e).lower()
                if "READ_VARIABLE" in str(e) or "variable" in error_str or "dynamic" in error_str:
                    print("⚠️  Direct conversion failed. Trying SavedModel approach with fixed input signature...")
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        saved_model_path = os.path.join(tmpdir, "saved_model")

                        @tf.function(input_signature=[tf.TensorSpec(shape=[1, SEQ_LEN, n_features], dtype=tf.float32)])
                        def serving_fn(x):
                            return export_model(x, training=False)

                        tf.saved_model.save(
                            export_model,
                            saved_model_path,
                            signatures={'serving_default': serving_fn.get_concrete_function()}
                        )

                        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        converter.representative_dataset = representative_data_gen
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.int8
                        converter.inference_output_type = tf.int8
                        converter._experimental_lower_tensor_list_ops = False
                        try:
                            converter.experimental_new_converter = False
                        except AttributeError:
                            pass
                        quantized_tflite_model = converter.convert()
                else:
                    raise

        tflite_fname = os.path.join(RESULTS_DIR, f"weather_model_5b_quant_{name}.tflite")
        with open(tflite_fname, "wb") as f:
            f.write(quantized_tflite_model)

        tflite_model_size_kb = os.path.getsize(tflite_fname) / 1024

        # Determine best epoch from history if available, otherwise use initial_epoch.
        # Add initial_epoch offset so the reported epoch is global (not session-local).
        if history and hasattr(history, 'history') and len(history.history.get('val_task_loss', [])) > 0:
            best_session_epoch = int(np.argmin(history.history['val_task_loss'])) + 1
            best_epoch = initial_epoch + best_session_epoch
        elif history and hasattr(history, 'history') and len(history.history.get('val_loss', [])) > 0:
            best_session_epoch = int(np.argmin(history.history['val_loss'])) + 1
            best_epoch = initial_epoch + best_session_epoch
        else:
            # No training occurred (already at max epoch), use the checkpoint epoch
            best_epoch = initial_epoch if initial_epoch > 0 else 1
            print(f"   Note: Using checkpoint epoch {best_epoch} as best epoch (no new training occurred)")

        print(f"\nFinal Metrics [{name}]:")
        print(f"  val_loss: {val_loss:.4f}")
        print(f"  val_mae: {val_mae:.4f}")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Quantized model size: {tflite_model_size_kb:.2f} KB")

        # Accumulate per-epoch history across restarts by merging with any existing JSON.
        session_history = {}
        if history and hasattr(history, 'history') and history.history:
            session_history = {k: [float(v) for v in vals]
                               for k, vals in history.history.items()}

        results_path = os.path.join(RESULTS_DIR, f"results_5b_{name}.json")
        accumulated_history = {}
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    accumulated_history = json.load(f).get("history", {})
            except Exception:
                pass
        for k, vals in session_history.items():
            accumulated_history.setdefault(k, []).extend(vals)

        metrics = {
            "name": name,
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "best_epoch": int(best_epoch),
            "feature_importance": [(f, float(i)) for f, i in sorted_importance],
            "model_size_kb": float(tflite_model_size_kb),
            "history": accumulated_history,
        }

        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Configuration
    NUM_RUNS = 1  # Number of training runs to perform
    EXP_NAME = "conv2d_exp40"
    for run_id in range(NUM_RUNS):
        run_name = f"{EXP_NAME}_run{run_id+1}"
        build_and_train_model(run_name)

    results = []
    # Only collect results from the current run session
    for run_id in range(NUM_RUNS):
        json_file = os.path.join(RESULTS_DIR, f"results_5b_{EXP_NAME}_run{run_id+1}.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                metrics = json.load(f)
                results.append(metrics)

    if results:
        best = min(results, key=lambda x: x["val_loss"])
        print(f"\nBest run: {best['name']} with val_loss: {best['val_loss']:.4f} and val_mae: {best['val_mae']:.4f}")
    else:
        print("\nNo results found!")
        exit(1)

    # Copy best model to canonical filename
    import shutil
    best_model_file = os.path.join(RESULTS_DIR, f"weather_model_5b_quant_{best['name']}.tflite")
    shutil.copy(best_model_file, os.path.join(RESULTS_DIR, "weather_model_5b_best.tflite"))
    print(f"Best model copied to: {os.path.join(RESULTS_DIR, 'weather_model_5b_best.tflite')}")
    
    # Validate the best quantized model
    print("\n" + "="*50)
    print("VALIDATING BEST QUANTIZED MODEL")
    print("="*50)
    y_val_array = y_val_small
    validate_quantized_model(os.path.join(RESULTS_DIR, "weather_model_5b_best.tflite"), X_val_small, y_val_array, y_mins, y_maxs)

# --- Validate quantized TFLite model on validation data ---
def validate_quantized_model(tflite_model_path, X_val, y_val, y_mins, y_maxs, num_samples=500):
    import tensorflow as tf
    import numpy as np

    print(f"\nValidating TFLite model on {num_samples} samples...")

    # Try to create interpreter - XNNPACK may fail with some quantized operations
    # If it fails, we'll catch the error and provide guidance
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
    except RuntimeError as e:
        if "XNNPACK" in str(e) or "delegate" in str(e).lower():
            print("Warning: XNNPACK delegate failed. This quantized model contains operations")
            print("that aren't compatible with XNNPACK. Attempting CPU-only fallback...")
            
            # Try loading model content and using experimental options to avoid XNNPACK
            with open(tflite_model_path, 'rb') as f:
                model_content = f.read()
            
            try:
                # Try with experimental op resolver that might avoid XNNPACK
                interpreter = tf.lite.Interpreter(
                    model_content=model_content,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
                )
                interpreter.allocate_tensors()
                print("Successfully loaded model in CPU-only mode.")
            except Exception as e2:
                # If that also fails, try the simplest approach
                print("First fallback failed, trying basic interpreter...")
                try:
                    interpreter = tf.lite.Interpreter(model_content=model_content)
                    interpreter.allocate_tensors()
                    print("Successfully loaded model with basic interpreter.")
                except Exception as e3:
                    print(f"\nError: Unable to load TFLite model.")
                    print(f"XNNPACK error: {e}")
                    print(f"Fallback 1 error: {e2}")
                    print(f"Fallback 2 error: {e3}")
                    print("\nThis model may need to be reconverted without XNNPACK optimization.")
                    print("Skipping quantized model validation.")
                    return
        else:
            raise

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scales = [d['quantization'][0] for d in output_details]
    output_zero_points = [d['quantization'][1] for d in output_details]

    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    for idx, d in enumerate(output_details):
        print(f"Output {idx} dtype: {d['dtype']}, scale={output_scales[idx]}, zero_point={output_zero_points[idx]}")

    X_val_subset = X_val[:num_samples]
    y_val_subset = y_val[:num_samples]

    print(f"Input range: min={X_val_subset.min():.4f}, max={X_val_subset.max():.4f}")

    input_quantized = np.round(X_val_subset / input_scale + input_zero_point).astype(input_details[0]['dtype'])
    print(f"Quantized input range: min={input_quantized.min()}, max={input_quantized.max()}")

    y_preds_dequant = [[] for _ in range(3)]

    for i in range(len(input_quantized)):
        input_data = np.expand_dims(input_quantized[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        for j in range(3):
            output_data = interpreter.get_tensor(output_details[j]['index'])
            output_float = (output_data.astype(np.float32) - output_zero_points[j]) * output_scales[j]
            y_preds_dequant[j].append(output_float[0])

    # Debug: show example raw output before rescaling
    print("\nSample dequantized model outputs (before rescaling):")
    for j, name in enumerate(['diff_1hr', 'diff_2hr', 'diff_3hr']):
        print(f"  {name}: {np.array(y_preds_dequant[j][:5]).round(3)}")

    print("\nValidation MAE (in °C difference):")
    for j, name in enumerate(['diff_1hr', 'diff_2hr', 'diff_3hr']):
        # Convert scaled MAE back to °C per head using per-horizon scaling
        if name == 'diff_1hr':
            scale_to_c = (y_maxs["temp_diff_1hr"] - y_mins["temp_diff_1hr"]) / 2.0
        elif name == 'diff_2hr':
            scale_to_c = (y_maxs["temp_diff_2hr"] - y_mins["temp_diff_2hr"]) / 2.0
        else:
            scale_to_c = (y_maxs["temp_diff_3hr"] - y_mins["temp_diff_3hr"]) / 2.0
        
        y_preds_scaled = np.array(y_preds_dequant[j]).reshape(-1)
        y_true_scaled = y_val_subset[:, j].reshape(-1)
        mae_scaled = np.mean(np.abs(y_true_scaled - y_preds_scaled))
        mae = mae_scaled * scale_to_c
        print(f"  {name}: {mae:.2f} °C")



# Entry point guard
if __name__ == "__main__":
    main()
