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
    import tensorflow as tf

    # Enable synchronous execution on macOS for stability
    import sys
    if sys.platform == "darwin":
        try:
            tf.config.experimental.set_synchronous_execution(True)
            print("✅ macOS: synchronous execution enabled (stability > throughput)")
        except Exception as e:
            print(f"ℹ️  Could not enable synchronous execution: {e}")
    
    # Enable GPU if available, otherwise fall back to CPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"✅ GPU detected: {len(physical_devices)} device(s)")
        for device in physical_devices:
            print(f"   - {device.name}")
        # Enable memory growth to avoid allocating all GPU memory at once
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(f"⚠️  Could not set memory growth: {e}")
        # For GPU training, balance CPU threads for data loading while GPU computes
        # GPU cores handle all compute, but CPU needs enough threads to keep GPU fed with data
        # intra_op: parallelism within CPU operations (data preprocessing, window creation)
        # inter_op: parallelism between operations (pipeline efficiency, overlap data loading with GPU)
        cores = multiprocessing.cpu_count()
        # Use more threads for data loading - this is critical for GPU utilization
        # Data loading/preprocessing is CPU-bound and needs parallelization
        intra_op_threads = max(4, cores // 2)  # Use half cores for data preprocessing
        inter_op_threads = max(4, cores // 4)     # Use quarter cores for operation pipelining
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        print(f"Using {intra_op_threads} intra_op and {inter_op_threads} inter_op CPU threads (optimized for GPU data loading)")
        print("Note: More CPU threads help keep GPU fed with data, improving GPU utilization")
    else:
        print("⚠️  No GPU detected, using CPU")
        cores = multiprocessing.cpu_count()
        print(f"Detected {cores} CPU cores")
        # For CPU-only training, balance threads for optimal performance
        # intra_op: parallelism within operations (matrix ops, conv ops)
        # inter_op: parallelism between operations (pipeline efficiency)
        # With 10 cores, use 6 for intra_op and 2 for inter_op to allow
        # good parallelization within ops while still enabling operation pipelining
        intra_op_threads = max(4, cores - 4)  # Use most cores for intra-op
        inter_op_threads = 2                  # Keep low to avoid overhead
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        print(f"Using {intra_op_threads} intra_op and {inter_op_threads} inter_op threads (optimized for CPU training)")
    
    # Allow operations to fall back to CPU if needed
    tf.config.set_soft_device_placement(True)
    
    # Optional confirmation
    print("TF intra_op threads:", tf.config.threading.get_intra_op_parallelism_threads())
    print("TF inter_op threads:", tf.config.threading.get_inter_op_parallelism_threads())
    
    # Disable TensorFlow optimizations that might slow down small models
    tf.config.optimizer.set_jit(False)  # Disable XLA JIT
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    from tensorflow.keras.callbacks import EarlyStopping, Callback
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
    from numba import njit, prange

    # Load preprocessed data
    train_df = pd.read_csv("../train_data_sf.csv")
    val_df = pd.read_csv("../val_data_sf.csv")

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

    # Prepare time index + sort, then build timestamp-based targets if needed
    train_df = _prepare_time_index(train_df, "train_df")
    val_df = _prepare_time_index(val_df, "val_df")

    train_df = _add_future_targets(train_df, "train_df")
    val_df = _add_future_targets(val_df, "val_df")

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
    

    # Numba-accelerated rolling slope for multi-core performance
    # @njit(parallel=True)
    # def rolling_slope_numba(data, window):
    #     n = len(data)
    #     slopes = np.full(n, np.nan)
    #     x = np.arange(window)
    #     x_mean = np.mean(x)
    #     denom = np.sum((x - x_mean) ** 2)
    #     for i in prange(window - 1, n):
    #         y = data[i - window + 1:i + 1]
    #         if np.any(np.isnan(y)):
    #             continue
    #         y_mean = np.mean(y)
    #         num = np.sum((x - x_mean) * (y - y_mean))
    #         slopes[i] = num / denom
    #     return slopes
    #
    # # Compute deltas using numba-accelerated rolling slope
    # train_df['temperature_delta'] = rolling_slope_numba(train_df['temperature'].values, 15)
    # val_df['temperature_delta'] = rolling_slope_numba(val_df['temperature'].values, 15)
    # train_df['pressure_delta'] = rolling_slope_numba(train_df['station_pressure'].values, 15)
    # train_df['humidity_delta'] = rolling_slope_numba(train_df['relative_humidity'].values, 15)
    # val_df['pressure_delta'] = rolling_slope_numba(val_df['station_pressure'].values, 15)
    # val_df['humidity_delta'] = rolling_slope_numba(val_df['relative_humidity'].values, 15)
    # train_df['illuminance_delta'] = rolling_slope_numba(train_df['illuminance'].values, 15)
    # train_df['solar_radiation_delta'] = rolling_slope_numba(train_df['solar_radiation'].values, 15)
    # val_df['illuminance_delta'] = rolling_slope_numba(val_df['illuminance'].values, 15)
    # val_df['solar_radiation_delta'] = rolling_slope_numba(val_df['solar_radiation'].values, 15)

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
    # Experiment 9: raw station readings + cyclical time encodings only.
    # No pre-computed lag features — the dilated Conv1D learns temporal structure from the window.
    features = [
        'temperature',
        'uv', 'wind_avg', 'wind_gust',
        'solar_radiation', 'illuminance',
        'relative_humidity', 'station_pressure',
        'day_of_year_sin', 'day_of_year_cos',
        'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
    ]
    # Add wind_direction features if available
    if 'wind_direction_sin' in train_df.columns:
        features.extend(['wind_direction_sin', 'wind_direction_cos'])
    # Add wind_lull if available (contrast to wind_gust, indicates wind variability)
    if 'wind_lull' in train_df.columns:
        features.append('wind_lull')
    # Add rain_accumulated if available (precipitation can cause cooling)
    if 'rain_accumulated' in train_df.columns:
        features.append('rain_accumulated')
    targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']

    ## Per-feature min/max scaling with ±5% padding
    # Domain bounds for raw station features + cyclical encodings
    domain_bounds = {
        "temperature": (-10, 55),
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
    with open("input_scaler_5b.json", "w") as f:
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
    with open("target_scaler_5b.json", "w") as f:
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

    # Use the scaled targets (already in [-1, 1]) to align with sequence windows
    y_all_train = train_df[targets].values
    y_all_val = val_df[targets].values

    # Streaming time-series datasets (no giant in-memory window arrays)
    from tensorflow.keras.preprocessing import timeseries_dataset_from_array
    
    # Increased batch size for better GPU utilization
    # Larger batches = more work per GPU call = better GPU utilization
    # 64 was too small, causing GPU to wait for data
    TRAIN_BATCH_SIZE = 512   # Reduced from 1024 — 128-filter model OOM'd at 1024 on 16GB
    VAL_BATCH_SIZE = 512     # Match training batch size for consistency
    
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

    # Prefetch multiple batches to keep GPU busy while CPU loads data
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"Validation batch size: {VAL_BATCH_SIZE}")

    print(f"\nSequence training dataset: {train_ds.cardinality().numpy()} batches")
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
        def __init__(self, timeout_minutes=2, check_interval_seconds=30, verbose=1, adaptive_timeout=True, train_ds=None):
            """
            Args:
                timeout_minutes: Maximum time (in minutes) without progress before detecting a hang
                check_interval_seconds: How often (in seconds) to check for hangs
                verbose: Verbosity level (0=silent, 1=print warnings, 2=print all activity)
                adaptive_timeout: If True, base timeout on actual batch times (default: True)
                                  Uses 50x the average batch time, with a minimum of timeout_minutes
            """
            super().__init__()
            self.base_timeout_seconds = timeout_minutes * 60
            self.timeout_seconds = self.base_timeout_seconds  # Will adapt based on batch times
            self.check_interval = check_interval_seconds
            self.verbose = verbose
            self.adaptive_timeout = adaptive_timeout
            self.train_ds = train_ds  # Store dataset reference to get cardinality
            
            # Progress tracking
            self.last_activity_time = None
            self.current_epoch = None
            self.current_batch = None
            self.total_batches = None
            self.epoch_start_time = None
            self.last_batch_time = None
            self.last_activity_type = None  # 'epoch_start', 'epoch_end', 'batch_end'
            self.batch_durations = []  # Track batch processing times
            self.batch_update_frequency = 50  # Update activity time every N batches
            
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
            
            # Fallback: try to get from model steps_per_epoch if available
            if self.total_batches is None:
                try:
                    if hasattr(self.model, 'steps_per_epoch') and self.model.steps_per_epoch:
                        self.total_batches = self.model.steps_per_epoch
                except (AttributeError, TypeError):
                    pass
            
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
                        with open("training_hang_detected.json", "w") as f:
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

    # Custom layer for extracting exact lag values - properly serializable
    @tf.keras.utils.register_keras_serializable(package="WeatherML")
    class ExtractLagLayer(tf.keras.layers.Layer):
        """Extract exact feature values at a specific lag position."""
        def __init__(self, lag_steps, seq_len, **kwargs):
            super().__init__(**kwargs)
            self.lag_steps = lag_steps
            self.seq_len = seq_len
        
        def call(self, inputs):
            idx = self.seq_len - 1 - self.lag_steps
            return tf.gather(inputs, idx, axis=1)
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'lag_steps': self.lag_steps,
                'seq_len': self.seq_len
            })
            return config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    # Custom weighted pooling layer that emphasizes recent timesteps to reduce lag
    @tf.keras.utils.register_keras_serializable(package="WeatherML")
    class RecentWeightedPooling2D(tf.keras.layers.Layer):
        """Weighted average pooling that gives exponentially more weight to recent timesteps.
        
        This helps reduce prediction lag by emphasizing the most recent information
        rather than treating all timesteps equally like GlobalAveragePooling2D.
        """
        def __init__(self, decay_factor=0.95, **kwargs):
            """
            Args:
                decay_factor: Exponential decay factor for weights (0 < decay_factor <= 1).
                             Higher values (closer to 1) give more uniform weighting.
                             Lower values (e.g., 0.9) give much more weight to recent timesteps.
            """
            super().__init__(**kwargs)
            self.decay_factor = decay_factor
        
        def call(self, inputs):
            # inputs shape: (batch, time_steps, features, channels)
            # We want to weight timesteps so that more recent ones have higher weights
            # TPU-compatible: prefer static shapes, fall back to dynamic if needed
            
            # Get time_steps dimension - handle both static and dynamic shapes
            input_shape = inputs.shape
            time_steps_dim = input_shape[1]
            
            # Check if we have a static shape value
            try:
                # Try to get static value (works if shape is known)
                if time_steps_dim is not None:
                    time_steps = int(time_steps_dim)
                    time_steps_tensor = tf.constant(time_steps, dtype=tf.int32)
                    use_static_shape = True
                else:
                    raise ValueError("None dimension")
            except (TypeError, ValueError, AttributeError):
                # Fall back to dynamic shape
                time_steps_tensor = tf.shape(inputs)[1]
                use_static_shape = False
            
            # Create exponential weights: [decay_factor^0, decay_factor^1, ..., decay_factor^(n-1)]
            # Reverse so most recent timestep (last) gets weight decay_factor^0 = 1.0
            indices = tf.range(time_steps_tensor, dtype=tf.float32)
            time_steps_f = tf.cast(time_steps_tensor, dtype=tf.float32)
            weights = self.decay_factor ** (time_steps_f - 1.0 - indices)  # Most recent gets highest weight
            
            # Reshape weights to (1, time_steps, 1, 1) for broadcasting
            if use_static_shape:
                # Static shape for TPU compatibility
                weights = tf.reshape(weights, [1, time_steps, 1, 1])
            else:
                # Dynamic shape fallback - use -1 to infer dimension
                weights = tf.reshape(weights, [1, -1, 1, 1])
            
            # Apply weights and sum over time and feature dimensions
            weighted = inputs * weights
            # Sum over time and feature dimensions: (batch, channels)
            pooled = tf.reduce_sum(weighted, axis=[1, 2])
            
            # Normalize by sum of weights
            weight_sum = tf.reduce_sum(weights)
            pooled = pooled / weight_sum
            
            return pooled
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'decay_factor': self.decay_factor
            })
            return config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    # CENTRALIZED: All custom objects for checkpoint loading
    # Add any new custom layers/functions here to ensure they're always available when loading
    CUSTOM_OBJECTS = {
        'ExtractLagLayer': ExtractLagLayer,
        'RecentWeightedPooling2D': RecentWeightedPooling2D
    }
    
    def validate_checkpoint_loading(checkpoint_path, model, custom_objects):
        """
        Validate that a checkpoint can be loaded before relying on it.
        Returns (success: bool, error_message: str or None)
        
        Note: This function may modify model weights temporarily during validation.
        The caller should ensure model weights are restored if needed.
        """
        # Save current model weights to restore later (for weights-only validation)
        original_weights = None
        try:
            original_weights = [w.copy() for w in model.get_weights()]
        except:
            pass  # If we can't save weights, we'll proceed anyway
        
        try:
            # Determine checkpoint type
            is_weights_only = checkpoint_path.endswith('.weights.h5')
            is_full_model = (os.path.isdir(checkpoint_path) or 
                           (checkpoint_path.endswith('.h5') and not is_weights_only))
            
            if is_full_model:
                # Full model checkpoint - load into a temporary model
                test_model = tf.keras.models.load_model(
                    checkpoint_path,
                    compile=False,
                    custom_objects=custom_objects
                )
                # Verify we can extract weights and they match architecture
                test_weights = test_model.get_weights()
                del test_model
                
                # Check architecture compatibility
                if len(test_weights) != len(model.get_weights()):
                    return False, f"Architecture mismatch: checkpoint has {len(test_weights)} weight arrays, current model has {len(model.get_weights())}"
                
                # Check shape compatibility for each layer
                for i, (test_w, model_w) in enumerate(zip(test_weights, model.get_weights())):
                    if test_w.shape != model_w.shape:
                        return False, f"Architecture mismatch at layer {i}: checkpoint shape {test_w.shape} != model shape {model_w.shape}"
                
                return True, None
            elif is_weights_only:
                # Weights-only checkpoint - try loading into current model
                # This will fail if architecture doesn't match
                model.load_weights(checkpoint_path)
                # If we get here, loading succeeded
                # Restore original weights
                if original_weights:
                    model.set_weights(original_weights)
                return True, None
            else:
                return False, f"Unknown checkpoint format: {checkpoint_path}"
                
        except Exception as e:
            # Restore original weights if we modified them
            if original_weights:
                try:
                    model.set_weights(original_weights)
                except:
                    pass  # If restoration fails, at least we tried
            
            error_msg = str(e)
            # Check if it's an architecture mismatch
            if "expected" in error_msg.lower() and ("variables" in error_msg.lower() or "variable" in error_msg.lower()):
                return False, f"Architecture mismatch: {error_msg[:200]}"
            elif "shape" in error_msg.lower() and "mismatch" in error_msg.lower():
                return False, f"Shape mismatch: {error_msg[:200]}"
            elif "_extract" in error_msg or "function" in error_msg.lower() or "serializable" in error_msg.lower():
                return False, f"Serialization issue (old Lambda layer): {error_msg[:200]}"
            else:
                return False, f"Checkpoint loading error: {error_msg[:200]}"

    def build_and_train_model(name):
        print(f"\n--- Running: {name} ---\n")

        # ---------------------------------------------------------------------
        # Experiment 9: Dilated Conv1D — Edge TPU-compatible temporal architecture
        #
        # Goal: learn temporal structure (equivalent to temp_lag30/60/120) from the
        # raw 180-minute window without any pre-computed lag features.
        #
        # Receptive field of the dilated stack:
        #   Each Conv1D(kernel=3, dilation=d) adds 2*d steps of context.
        #   Stacking dilation rates [1,2,4,8,16] gives cumulative reach of
        #   2*(1+2+4+8+16) = 62 per pass; combined with the initial projection
        #   the full 180-step window is within reach.
        #
        # All ops are Edge TPU-compilable: Conv1D, BatchNorm, ReLU,
        # GlobalAveragePooling1D, Dense, Add, Dropout (removed at inference).
        # ---------------------------------------------------------------------

        # Input: (batch, 180, n_features)
        input_layer = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")

        # Exp 12: 96 filters (middle ground between Exp 11's 64 and OOM'd 128).
        # 128 filters + batch 1024 caused Metal GPU OOM on 16GB shared RAM.
        # 96 filters gives ~2.25x the conv parameters of 64 without the memory blow-up.
        FILTERS = 96

        # Initial projection: lift feature dimension to FILTERS channels
        x = tf.keras.layers.Conv1D(
            FILTERS, kernel_size=3, padding='causal', use_bias=False, name="conv_init"
        )(input_layer)
        x = tf.keras.layers.BatchNormalization(name="bn_init")(x)
        # ReLU6: clips activations to [0,6], bounding range for INT8 quantization
        x = tf.keras.layers.ReLU(max_value=6.0, name="relu_init")(x)

        # Dilated residual blocks — dilation rates [1, 2, 4, 8, 16, 32, 64]
        # Receptive field: cumulative reach = 257 steps, covering the full 180-min window.
        for dilation in [1, 2, 4, 8, 16, 32, 64]:
            residual = x
            x = tf.keras.layers.Conv1D(
                FILTERS, kernel_size=3, padding='causal',
                dilation_rate=dilation, use_bias=False,
                name=f"conv_d{dilation}"
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"bn_d{dilation}")(x)
            x = tf.keras.layers.ReLU(max_value=6.0, name=f"relu_d{dilation}")(x)
            x = tf.keras.layers.Add(name=f"add_d{dilation}")([residual, x])

        # Collapse time dimension — Edge TPU-compatible
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

        # Dense head (scaled to match wider conv)
        x = tf.keras.layers.Dense(96, use_bias=False, name="dense1")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name="relu_dense1")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
        x = tf.keras.layers.Dense(32, use_bias=False, name="dense2")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name="relu_dense2")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_dense2")(x)

        # Three output heads
        output_1 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype="float32", name='diff_1hr')(x)
        output_2 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype="float32", name='diff_2hr')(x)
        output_3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype="float32", name='diff_3hr')(x)
        model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

        # Fresh training at 1e-4. Keras 3 doesn't accept a tf.Variable as learning_rate,
        # but does accept a callable. We use a mutable list container so the callback can
        # update it each epoch and Adam will pick up the new value via the lambda on the next step.
        initial_lr = 1e-4
        lr_container = [initial_lr]
        optimizer = tf.keras.optimizers.Adam(learning_rate=lambda: lr_container[0])

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics={
                'diff_1hr': 'mae',
                'diff_2hr': 'mae',
                'diff_3hr': 'mae'
            },
            # Run 20 batches per Python→GPU call, reducing overhead.
            # With ~727 batches/epoch this gives ~36 graph calls instead of 727.
            steps_per_execution=20,
        )
        model.summary()

        # Check for existing checkpoints to resume training
        initial_epoch = 0
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
        
        if os.path.exists(checkpoint_dir):
            # Prefer full model checkpoints (more reliable) over weights-only
            # Look for both periodic checkpoints and near-end-of-epoch checkpoints
            full_model_checkpoints = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.h5"))
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
            
            # Priority 2: Best weights checkpoint (if exists and no full model found)
            elif os.path.exists(best_weights_checkpoint) and not full_model_checkpoints:
                checkpoint_to_load = best_weights_checkpoint
                checkpoint_type = "best_weights"
                # Try to get epoch from results file
                results_file = f"results_5b_{name}.json"
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
            
            # Priority 3: Legacy weights checkpoints (fallback)
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
                        # Validate checkpoint can be loaded BEFORE using it
                        print(f"   🔍 Validating checkpoint compatibility...")
                        can_load, validation_error = validate_checkpoint_loading(
                            checkpoint_to_load, model, CUSTOM_OBJECTS
                        )
                        
                        if not can_load:
                            print(f"   ❌ Checkpoint validation failed: {validation_error}")
                            
                            # Check if it's an architecture mismatch (different model structure)
                            is_architecture_mismatch = (
                                "architecture mismatch" in validation_error.lower() or
                                "shape" in validation_error.lower() and "mismatch" in validation_error.lower() or
                                "expected" in validation_error.lower() and "variables" in validation_error.lower()
                            )
                            
                            if is_architecture_mismatch:
                                print(f"   ⚠️  Architecture mismatch detected - checkpoint was saved with different model structure")
                                print(f"   This usually means the model code changed (e.g., different layer sizes)")
                                print(f"   Starting training from scratch with new architecture...")
                                initial_epoch = 0
                            else:
                                # For other errors (corruption, etc.), try fallback options
                                fallback_checkpoints = [
                                    ("best_model_full.h5", "best full model"),
                                    ("best_model.weights.h5", "best weights-only")
                                ]
                                
                                loaded_from_fallback = False
                                for fallback_file, fallback_desc in fallback_checkpoints:
                                    fallback_path = os.path.join(checkpoint_dir, fallback_file)
                                    if os.path.exists(fallback_path):
                                        print(f"   💡 Trying {fallback_desc} checkpoint as fallback...")
                                        can_load_fallback, fallback_error = validate_checkpoint_loading(
                                            fallback_path, model, CUSTOM_OBJECTS
                                        )
                                        
                                        # Check if fallback has architecture mismatch - if so, don't use it
                                        fallback_is_arch_mismatch = (
                                            "architecture mismatch" in fallback_error.lower() or
                                            ("shape" in fallback_error.lower() and "mismatch" in fallback_error.lower()) or
                                            ("expected" in fallback_error.lower() and "variables" in fallback_error.lower())
                                        ) if not can_load_fallback else False
                                        
                                        if fallback_is_arch_mismatch:
                                            print(f"   ⚠️  {fallback_desc} also has architecture mismatch - skipping")
                                            print(f"   Starting training from scratch with new architecture...")
                                            initial_epoch = 0
                                            break
                                        
                                        if can_load_fallback:
                                            if fallback_file.endswith('.weights.h5'):
                                                model.load_weights(fallback_path)
                                            else:
                                                loaded_model = tf.keras.models.load_model(
                                                    fallback_path, compile=False, custom_objects=CUSTOM_OBJECTS
                                                )
                                                model.set_weights(loaded_model.get_weights())
                                                del loaded_model
                                            initial_epoch = latest_epoch
                                            print(f"✅ Successfully loaded from {fallback_desc} (epoch {latest_epoch})")
                                            loaded_from_fallback = True
                                            break
                                        else:
                                            print(f"   ❌ {fallback_desc} checkpoint also failed: {fallback_error}")
                                
                                if not loaded_from_fallback:
                                    print(f"   ❌ All checkpoint options failed.")
                                    print(f"   Starting training from scratch...")
                                    initial_epoch = 0
                        else:
                            # Checkpoint validated - load it
                            try:
                                loaded_model = tf.keras.models.load_model(
                                    checkpoint_to_load, 
                                    compile=False,
                                    custom_objects=CUSTOM_OBJECTS
                                )
                                # Copy weights from loaded model to current model
                                model.set_weights(loaded_model.get_weights())
                                del loaded_model  # Free memory
                                initial_epoch = latest_epoch
                                print(f"✅ Successfully loaded full model checkpoint from epoch {latest_epoch}")
                            except Exception as load_error:
                                print(f"   ❌ Checkpoint validation passed but loading failed: {str(load_error)[:200]}")
                                print(f"   This shouldn't happen - there may be a race condition or file corruption.")
                                print(f"   Starting training from scratch...")
                                initial_epoch = 0
                    else:
                        # Try loading weights (simple approach - no by_name or skip_mismatch)
                        # These parameters aren't always supported in all TF versions
                        model.load_weights(checkpoint_to_load)
                        initial_epoch = latest_epoch
                        print(f"✅ Successfully loaded weights checkpoint from epoch {latest_epoch}")
                    
                    if initial_epoch > 0:
                        print(f"   Note: Optimizer state is reset (learning rate will restart at {initial_lr})")
                        print(f"   ReduceLROnPlateau will adjust learning rate based on validation loss")
                    
                except (ValueError, RuntimeError, OSError, TypeError) as e:
                    error_msg = str(e)
                    if "expected" in error_msg.lower() and ("variables" in error_msg.lower() or "variable" in error_msg.lower()):
                        print(f"⚠️  Warning: Checkpoint architecture mismatch detected.")
                        print(f"   The checkpoint was saved with a different model structure.")
                        print(f"   This can happen if the model code changed between training sessions.")
                        print(f"   Starting training from scratch...")
                        initial_epoch = 0
                    elif "Invalid keyword" in error_msg:
                        print(f"⚠️  Warning: Checkpoint loading failed: {error_msg}")
                        print(f"   This TensorFlow version may not support this checkpoint format.")
                        print(f"   Starting training from scratch...")
                        initial_epoch = 0
                    else:
                        print(f"⚠️  Warning: Could not load checkpoint: {error_msg}")
                        print("   Starting training from scratch...")
                        initial_epoch = 0

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
                    warnings.append("Training dataset cardinality unknown (may be fine for streaming)")
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
            
            # 5. Validate model can be saved and loaded (critical for checkpoints)
            print("\n5️⃣  Validating model serialization...")
            try:
                test_checkpoint_path = os.path.join(checkpoint_dir, ".serialization_test.h5")
                # Save model
                model.save(test_checkpoint_path, save_format='tf')
                
                # Verify file exists and has reasonable size
                if not os.path.exists(test_checkpoint_path):
                    errors.append("Model save failed: checkpoint file not created")
                else:
                    file_size = os.path.getsize(test_checkpoint_path)
                    if file_size < 1000:  # Less than 1KB is suspicious
                        errors.append(f"Model checkpoint suspiciously small: {file_size} bytes")
                    else:
                        print(f"   ✅ Model save successful ({file_size / (1024**2):.2f} MB)")
                    
                    # Try loading it back
                    try:
                        loaded_test_model = tf.keras.models.load_model(
                            test_checkpoint_path,
                            compile=False,
                            custom_objects=custom_objects
                        )
                        
                        # Verify architecture matches
                        test_weights = loaded_test_model.get_weights()
                        model_weights = model.get_weights()
                        
                        if len(test_weights) != len(model_weights):
                            errors.append(f"Model load failed: weight count mismatch ({len(test_weights)} vs {len(model_weights)})")
                        else:
                            # Check shapes match
                            shape_mismatches = []
                            for i, (tw, mw) in enumerate(zip(test_weights, model_weights)):
                                if tw.shape != mw.shape:
                                    shape_mismatches.append(f"Layer {i}: {tw.shape} != {mw.shape}")
                            
                            if shape_mismatches:
                                errors.append(f"Model load failed: shape mismatches - {', '.join(shape_mismatches[:3])}")
                            else:
                                print("   ✅ Model load successful (architecture matches)")
                        
                        del loaded_test_model
                    except Exception as e:
                        errors.append(f"Model load failed: {str(e)[:200]}")
                    
                    # Clean up test file
                    try:
                        os.remove(test_checkpoint_path)
                    except:
                        pass
                        
            except Exception as e:
                errors.append(f"Model serialization validation failed: {str(e)[:200]}")
            
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
        
        # Early stopping: Match Model 5a's patience (5) but allow longer if needed
        # Model 5a trained to epoch 97, so we want similar opportunity for convergence
        # With lower learning rate (1e-5), model may need more epochs to converge
        # patience=20: gives ReduceLROnPlateau (patience=3) room to fire and act before stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        
        # Save best weights checkpoint (for resuming - weights only for smaller size)
        # This is updated every epoch if validation loss improves
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/best_model.weights.h5",  # Fixed filename since save_best_only=True
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=0
        )
        
        # ALSO save best full model as backup (dual backup strategy)
        # This ensures we always have both formats available
        best_full_model_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/best_model_full.h5",
            save_weights_only=False,  # Save full model
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=0
        )
        
        # Custom callback to save full model checkpoints periodically and near epoch boundaries
        # Full model format is more reliable for resuming than weights-only
        # Saves every 5 epochs, and also saves near the end of each epoch (where hangs occur)
        class PeriodicModelCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, filepath, period=5, save_near_epoch_end=True, total_batches=None):
                super().__init__()
                self.filepath = filepath
                self.period = period
                self.save_near_epoch_end = save_near_epoch_end
                self.last_saved_epoch = -1
                self.current_epoch = 0
                self.total_batches = total_batches  # Pass from outside if known
            
            def on_epoch_begin(self, epoch, logs=None):
                self.current_epoch = epoch
            
            def on_train_batch_end(self, batch, logs=None):
                # Save checkpoint when we're in the last 20 batches of an epoch (where hangs occur)
                if self.save_near_epoch_end and self.total_batches:
                    remaining = self.total_batches - batch - 1
                    if remaining == 20:  # Save at batch N-20 to have a checkpoint before potential hang
                        try:
                            # Only save once per epoch at this point
                            if self.current_epoch != self.last_saved_epoch:
                                filepath = f"./checkpoints/model_epoch_{self.current_epoch + 1:02d}_batch_{batch + 1}.h5"
                                print(f"\n💾 Saving checkpoint near epoch end (batch {batch + 1}/{self.total_batches}): {os.path.basename(filepath)}")
                                # Save full model
                                self.model.save(filepath, save_format='tf')
                                
                                # VERIFY checkpoint was saved correctly (fail fast if save failed)
                                if not os.path.exists(filepath):
                                    raise RuntimeError(f"Checkpoint save failed: file {filepath} not created")
                                file_size = os.path.getsize(filepath)
                                if file_size < 1000:  # Less than 1KB is suspicious
                                    raise RuntimeError(f"Checkpoint suspiciously small: {file_size} bytes")
                                
                                # ALSO save weights-only as backup (dual backup strategy)
                                weights_filepath = filepath.replace('.h5', '.weights.h5')
                                self.model.save_weights(weights_filepath)
                                
                                # Verify weights checkpoint
                                if not os.path.exists(weights_filepath):
                                    raise RuntimeError(f"Weights checkpoint save failed: file {weights_filepath} not created")
                                
                                self.last_saved_epoch = self.current_epoch
                                print(f"   ✅ Saved both full model ({file_size / (1024**2):.2f} MB) and weights-only backup")
                        except Exception as e:
                            # Checkpoint save failure is CRITICAL - stop training to prevent wasting time
                            print(f"\n" + "="*80)
                            print(f"🚨 CRITICAL: Checkpoint save failed at batch {batch + 1} of epoch {self.current_epoch + 1}")
                            print(f"="*80)
                            print(f"Error: {str(e)[:200]}")
                            print(f"\n⚠️  If checkpoints can't be saved, resuming will be impossible.")
                            print(f"   Stopping training NOW to prevent wasting more time.")
                            print(f"="*80 + "\n")
                            self.model.stop_training = True
                            raise RuntimeError(f"Checkpoint save failed: {str(e)}") from e
            
            def on_epoch_end(self, epoch, logs=None):
                # Save periodic checkpoint every N epochs
                if (epoch + 1) % self.period == 0:
                    filepath = self.filepath.format(epoch=epoch + 1)
                    print(f"\n💾 Saving periodic full model checkpoint: {os.path.basename(filepath)}")
                    try:
                        # Save full model
                        self.model.save(filepath, save_format='tf')
                        
                        # VERIFY checkpoint was saved correctly (fail fast if save failed)
                        if not os.path.exists(filepath):
                            raise RuntimeError(f"Checkpoint save failed: file {filepath} not created")
                        file_size = os.path.getsize(filepath)
                        if file_size < 1000:  # Less than 1KB is suspicious
                            raise RuntimeError(f"Checkpoint suspiciously small: {file_size} bytes")
                        
                        # ALSO save weights-only as backup (dual backup strategy)
                        weights_filepath = filepath.replace('.h5', '.weights.h5')
                        self.model.save_weights(weights_filepath)
                        
                        # Verify weights checkpoint
                        if not os.path.exists(weights_filepath):
                            raise RuntimeError(f"Weights checkpoint save failed: file {weights_filepath} not created")
                        
                        self.last_saved_epoch = epoch
                        print(f"   ✅ Saved both full model ({file_size / (1024**2):.2f} MB) and weights-only backup")
                    except Exception as e:
                        # Checkpoint save failure is CRITICAL - stop training to prevent wasting time
                        print(f"\n" + "="*80)
                        print(f"🚨 CRITICAL: Periodic checkpoint save failed at epoch {epoch + 1}")
                        print(f"="*80)
                        print(f"Error: {str(e)[:200]}")
                        print(f"\n⚠️  If checkpoints can't be saved, resuming will be impossible.")
                        print(f"   Stopping training NOW to prevent wasting more time.")
                        print(f"="*80 + "\n")
                        self.model.stop_training = True
                        raise RuntimeError(f"Periodic checkpoint save failed: {str(e)}") from e
        
        # Get total batches for the periodic checkpoint callback
        try:
            total_batches_for_cb = train_ds.cardinality().numpy()
            if total_batches_for_cb < 0:
                total_batches_for_cb = None
        except:
            total_batches_for_cb = None
        
        periodic_checkpoint_cb = PeriodicModelCheckpoint(
            filepath="./checkpoints/model_epoch_{epoch:02d}.h5",
            period=5,  # Save every 5 epochs
            save_near_epoch_end=True,  # Also save near end of each epoch
            total_batches=total_batches_for_cb
        )
        # Adaptive learning rate: reduce LR when validation loss plateaus
        # Custom LR reduction callback using an explicit external tf.Variable (lr_var).
        # Passing lr_var directly to Adam means any assign() on it is immediately
        # seen by the optimizer — no Keras 2/3 API mismatch possible.
        class ReduceLRCallback(tf.keras.callbacks.Callback):
            def __init__(self, lr_container, monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1):
                super().__init__()
                self._lr = lr_container  # mutable list [current_lr]
                self.monitor = monitor
                self.factor = factor
                self.patience = patience
                self.min_lr = min_lr
                self.verbose = verbose
                self.best = float('inf')
                self.wait = 0

            def on_epoch_end(self, epoch, logs=None):
                current = logs.get(self.monitor)
                if current is None:
                    return
                current_lr = self._lr[0]
                # Log LR as a metric so it appears in epoch output
                if logs is not None:
                    logs['learning_rate'] = current_lr
                if current < self.best:
                    self.best = current
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        if current_lr > self.min_lr:
                            new_lr = max(current_lr * self.factor, self.min_lr)
                            self._lr[0] = new_lr
                            if self.verbose:
                                print(f'\nEpoch {epoch + 1}: ReduceLRCallback reducing learning rate to {new_lr:.2e}. (wait={self.wait}, best={self.best:.6f})', flush=True)
                        self.wait = 0

        reduce_lr = ReduceLRCallback(
            lr_container=lr_container,
            monitor='val_loss',
            factor=0.5,           # Reduce LR by half
            patience=3,           # Fire quickly (before early stopping patience=20 triggers)
            min_lr=1e-7,          # Minimum learning rate
            verbose=1             # Print when LR is reduced
        )
        
        # Checkpoint validation callback: validates that checkpoints can be loaded after each epoch
        # This catches serialization issues, architecture problems, or corruption IMMEDIATELY
        # Since each epoch takes ~1200 seconds (~20 minutes), losing an epoch is VERY costly
        # This small validation overhead (~1-3 seconds per epoch) is worth it for reliability
        # 
        # What it validates:
        # - Best model checkpoints (full + weights-only) after each epoch
        # - Periodic checkpoints (every 5 epochs)
        # - Stops training IMMEDIATELY if validation fails (prevents wasting another 20 minutes)
        #
        # Performance impact: ~1-3 seconds per epoch (0.1% overhead) vs 1200 seconds per epoch
        # This is an excellent trade-off for catching issues early
        class CheckpointValidationCallback(tf.keras.callbacks.Callback):
            """Validates checkpoint loading after each epoch to catch issues early."""
            def __init__(self, checkpoint_dir, custom_objects, verbose=1, watchdog=None):
                super().__init__()
                self.checkpoint_dir = checkpoint_dir
                self.custom_objects = custom_objects
                self.verbose = verbose
                self.last_validated_epoch = -1
                self.watchdog = watchdog  # Reference to watchdog to update activity time
            
            def on_epoch_end(self, epoch, logs=None):
                """Validate checkpoints after each epoch."""
                # Notify watchdog that checkpoint validation is starting (legitimate activity)
                if self.watchdog:
                    self.watchdog.update_activity('checkpoint_validation_start')
                
                if self.verbose >= 1:
                    print(f"\n🔍 Validating checkpoint resumability after epoch {epoch + 1}...")
                
                validation_passed = True
                validation_errors = []
                
                # Validate all checkpoint types that should exist
                checkpoints_to_validate = []
                
                # 1. Best full model (if it exists and was updated this epoch)
                best_full_path = os.path.join(self.checkpoint_dir, "best_model_full.h5")
                if os.path.exists(best_full_path):
                    checkpoints_to_validate.append(("best_model_full.h5", best_full_path, "full model"))
                
                # 2. Best weights-only (always exists if training started)
                best_weights_path = os.path.join(self.checkpoint_dir, "best_model.weights.h5")
                if os.path.exists(best_weights_path):
                    checkpoints_to_validate.append(("best_model.weights.h5", best_weights_path, "weights-only"))
                
                # 3. Periodic checkpoint (if this was a periodic epoch)
                if (epoch + 1) % 5 == 0:  # Periodic checkpoints are saved every 5 epochs
                    periodic_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch + 1:02d}.h5")
                    if os.path.exists(periodic_path):
                        checkpoints_to_validate.append((f"model_epoch_{epoch + 1:02d}.h5", periodic_path, "periodic full model"))
                
                if not checkpoints_to_validate:
                    if self.verbose >= 1:
                        print(f"   ⚠️  No checkpoints found to validate (this is normal on first epoch)")
                    return
                
                # Validate each checkpoint
                for checkpoint_name, checkpoint_path, checkpoint_type in checkpoints_to_validate:
                    try:
                        # Use the validation function to test loading
                        can_load, error_msg = validate_checkpoint_loading(
                            checkpoint_path, self.model, self.custom_objects
                        )
                        
                        if can_load:
                            if self.verbose >= 2:
                                print(f"   ✅ {checkpoint_name} ({checkpoint_type}): Valid")
                        else:
                            validation_passed = False
                            validation_errors.append(f"{checkpoint_name} ({checkpoint_type}): {error_msg}")
                            if self.verbose >= 1:
                                print(f"   ❌ {checkpoint_name} ({checkpoint_type}): FAILED - {error_msg[:100]}")
                    except Exception as e:
                        validation_passed = False
                        error_msg = str(e)[:200]
                        validation_errors.append(f"{checkpoint_name} ({checkpoint_type}): {error_msg}")
                        if self.verbose >= 1:
                            print(f"   ❌ {checkpoint_name} ({checkpoint_type}): EXCEPTION - {error_msg}")
                
                # If validation failed, STOP TRAINING IMMEDIATELY
                # This prevents wasting another 1200 seconds on a broken checkpoint
                if not validation_passed:
                    print("\n" + "="*80)
                    print("🚨 CHECKPOINT VALIDATION FAILED - STOPPING TRAINING")
                    print("="*80)
                    print(f"Epoch: {epoch + 1}")
                    print(f"\nValidation Errors:")
                    for error in validation_errors:
                        print(f"  - {error}")
                    print(f"\n⚠️  CRITICAL: Checkpoints cannot be loaded!")
                    print(f"   This means resuming will FAIL.")
                    print(f"   Stopping training NOW to prevent wasting more time.")
                    print(f"\nPossible causes:")
                    print(f"  1. Model architecture changed during training (shouldn't happen)")
                    print(f"  2. Custom layer serialization issue")
                    print(f"  3. File corruption")
                    print(f"  4. TensorFlow version incompatibility")
                    print(f"\n💡 Action: Fix the issue before continuing training.")
                    print("="*80 + "\n")
                    
                    # Stop training by setting model.stop_training
                    self.model.stop_training = True
                    raise RuntimeError(
                        f"Checkpoint validation failed after epoch {epoch + 1}. "
                        f"Checkpoints cannot be loaded. Errors: {'; '.join(validation_errors)}"
                    )
                else:
                    if self.verbose >= 1:
                        print(f"   ✅ All checkpoints validated successfully - resuming will work!")
                
                # Notify watchdog that checkpoint validation completed
                if self.watchdog:
                    self.watchdog.update_activity('checkpoint_validation_end')
                
                self.last_validated_epoch = epoch
        
        # Training watchdog: detects hangs during training
        # Timeout: 10 minutes without any progress (increased to account for checkpoint validation)
        # Check interval: 30 seconds (how often to check for hangs)
        watchdog = TrainingWatchdog(
            timeout_minutes=10,  # Increased from 2 to 10 minutes to account for checkpoint validation
            check_interval_seconds=30,
            verbose=1,
            train_ds=train_ds  # Pass dataset to get accurate batch count
        )
        
        checkpoint_validator = CheckpointValidationCallback(
            checkpoint_dir=checkpoint_dir,
            custom_objects=CUSTOM_OBJECTS,
            verbose=1,
            watchdog=watchdog  # Pass watchdog reference so validation can update activity time
        )

        # Match RPi5 configuration exactly
        # Note: The watchdog callback will monitor training and detect hangs
        # If a hang is detected, it will write diagnostic info to training_hang_detected.json
        # and attempt to interrupt training via SIGINT
        
        # Check if we need to train (if initial_epoch >= epochs, training is already complete)
        max_epochs = 100
        history = None
        
        if initial_epoch >= max_epochs:
            print(f"\n✅ Training already complete (checkpoint is from epoch {initial_epoch}, max is {max_epochs})")
            print(f"   Skipping training and using existing checkpoint for evaluation")
            # Create empty history dict to avoid KeyError later
            history = type('History', (), {'history': {}})()
        else:
            try:
                if initial_epoch > 0:
                    print(f"\n🔄 Resuming training from epoch {initial_epoch + 1} (max epochs: {max_epochs})")
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=max_epochs,
                    initial_epoch=initial_epoch,
                    # Callback order matters: checkpoint saving callbacks must run BEFORE validation
                    # This ensures validation tests the checkpoints that were just saved
                    callbacks=[
                        early_stopping,           # Monitor for early stopping
                        checkpoint_cb,            # Save best weights-only (runs in on_epoch_end)
                        best_full_model_cb,       # Save best full model (runs in on_epoch_end)
                        periodic_checkpoint_cb,   # Save periodic checkpoints (runs in on_epoch_end)
                        reduce_lr,                # Adjust learning rate on plateau
                        checkpoint_validator,     # Validate checkpoints AFTER they're saved (runs in on_epoch_end)
                        watchdog                  # Monitor for hangs
                    ]
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

        # Ensure model is in inference mode and build it with a concrete input shape
        # This helps avoid issues with stateful operations during TFLite conversion
        # Build the model with a concrete batch size to ensure all operations are traced
        dummy_input = tf.zeros((1, SEQ_LEN, n_features), dtype=tf.float32)
        _ = model(dummy_input, training=False)  # Trace the model in inference mode
        
        # For Edge TPU compatibility, we need static tensor shapes (fixed batch size)
        # Create a concrete function with fixed input signature (batch_size=1)
        # This ensures all tensor shapes are static, which Edge TPU requires
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, SEQ_LEN, n_features], dtype=tf.float32)])
        def model_inference(x):
            return model(x, training=False)
        
        # Convert concrete function to TFLite (ensures static shapes for Edge TPU)
        concrete_func = model_inference.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Disable experimental features that might introduce dynamic shapes
        converter._experimental_lower_tensor_list_ops = False
        # Use legacy converter for better Edge TPU compatibility (avoids dynamic shape issues)
        try:
            converter.experimental_new_converter = False
        except AttributeError:
            # Newer TF versions might not have this attribute
            pass

        # Use full-integer quantization path suitable for Edge TPU
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_data_gen():
            # Use a small, random subset of windows for calibration
            num_samples = 500  # 100–1000 is usually plenty
            n = X_train_flat.shape[0]

            for _ in range(num_samples):
                # Random index with enough history for a full window
                idx = np.random.randint(SEQ_LEN, n)
                window = X_train_flat[idx - SEQ_LEN:idx, :]  # shape (SEQ_LEN, n_features)
                # Add batch dimension and cast to float32
                window = window[np.newaxis, ...].astype(np.float32)
                # TFLite expects a list/tuple of inputs
                yield [window]

        converter.representative_dataset = representative_data_gen

        # Require full integer quantization for all ops
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Edge TPU requires int8 for best compatibility; use signed int8 I/O
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert the model - full integer quantization where possible
        # (If some ops cannot be fully quantized, TensorFlow will fall back and report it.)
        try:
            quantized_tflite_model = converter.convert()
        except (RuntimeError, ValueError) as e:
            error_str = str(e).lower()
            if "READ_VARIABLE" in str(e) or "variable" in error_str or "dynamic" in error_str:
                # Fallback: Save to SavedModel first, then convert with fixed input signature
                # This often resolves issues with stateful operations and dynamic shapes
                print("⚠️  Direct conversion failed. Trying SavedModel approach with fixed input signature...")
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    saved_model_path = os.path.join(tmpdir, "saved_model")
                    # Save in TensorFlow SavedModel format (not H5) with fixed input signature
                    # Create a model with fixed batch size for SavedModel export
                    @tf.function(input_signature=[tf.TensorSpec(shape=[1, SEQ_LEN, n_features], dtype=tf.float32)])
                    def serving_fn(x):
                        return model(x, training=False)
                    
                    # Export as SavedModel with concrete function
                    tf.saved_model.save(
                        model,
                        saved_model_path,
                        signatures={'serving_default': serving_fn.get_concrete_function()}
                    )
                    
                    # Convert from SavedModel instead
                    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = representative_data_gen
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    converter._experimental_lower_tensor_list_ops = False
                    # Use legacy converter for better Edge TPU compatibility
                    try:
                        converter.experimental_new_converter = False
                    except AttributeError:
                        pass
                    quantized_tflite_model = converter.convert()
            else:
                raise

        tflite_fname = f"weather_model_5b_quant_{name}.tflite"
        with open(tflite_fname, "wb") as f:
            f.write(quantized_tflite_model)

        tflite_model_size_kb = os.path.getsize(tflite_fname) / 1024

        # Determine best epoch from history if available, otherwise use initial_epoch
        if history and hasattr(history, 'history') and 'val_loss' in history.history and len(history.history['val_loss']) > 0:
            best_epoch = np.argmin(history.history['val_loss']) + 1
        else:
            # No training occurred (already at max epoch), use the checkpoint epoch
            best_epoch = initial_epoch if initial_epoch > 0 else 1
            print(f"   Note: Using checkpoint epoch {best_epoch} as best epoch (no new training occurred)")
        
        print(f"\nFinal Metrics [{name}]:")
        print(f"  val_loss: {val_loss:.4f}")
        print(f"  val_mae: {val_mae:.4f}")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Quantized model size: {tflite_model_size_kb:.2f} KB")

        metrics = {
            "name": name,
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "best_epoch": int(best_epoch),
            "feature_importance": [(f, float(i)) for f, i in sorted_importance],
            "model_size_kb": float(tflite_model_size_kb)
        }

        with open(f"results_5b_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Configuration
    NUM_RUNS = 1  # Number of training runs to perform
    for run_id in range(NUM_RUNS):
        run_name = f"conv1d_exp12_run{run_id+1}"
        build_and_train_model(run_name)

    results = []
    # Only collect results from the current run session
    for run_id in range(NUM_RUNS):
        json_file = f"results_5b_conv1d_exp12_run{run_id+1}.json"
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
    best_model_file = f"weather_model_5b_quant_{best['name']}.tflite"
    shutil.copy(best_model_file, "weather_model_5b_best.tflite")
    print(f"Best model copied to: weather_model_5b_best.tflite")
    
    # Validate the best quantized model
    print("\n" + "="*50)
    print("VALIDATING BEST QUANTIZED MODEL")
    print("="*50)
    y_val_array = y_val_small
    validate_quantized_model("weather_model_5b_best.tflite", X_val_small, y_val_array, y_mins, y_maxs)

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
