def main():
    # This version removes scipy.stats.linregress and uses Numba-accelerated NumPy for rolling slope computation.
    # This provides better multi-core performance on Raspberry Pi and simplifies dependencies.
    import multiprocessing as mp
    import multiprocessing
    mp.set_start_method("fork", force=True)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Hide GPU devices

    cores = multiprocessing.cpu_count()
    print(f"Setting TensorFlow threads → {cores} cores")

    # Force TensorFlow to use all CPU threads
    tf.config.threading.set_intra_op_parallelism_threads(cores)
    tf.config.threading.set_inter_op_parallelism_threads(cores)
    tf.config.set_soft_device_placement(True)

    # Optional confirmation
    print("TF intra_op threads:", tf.config.threading.get_intra_op_parallelism_threads())
    print("TF inter_op threads:", tf.config.threading.get_inter_op_parallelism_threads())
    
    # Disable TensorFlow optimizations that might slow down small models
    tf.config.optimizer.set_jit(False)  # Disable XLA JIT
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    import pandas as pd
    # Removed unused imports that required scikit-learn to simplify Docker dependencies:
    # import joblib
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import mean_squared_error
    import copy
    import subprocess
    import json
    import glob
    from numba import njit, prange

    def _invalidate_targets_crossing_gaps(df, label, tol_s=90):
        """Null pre-computed target values whose future lookup crosses a time gap.

        CSV columns temp_t+1hr/2hr/3hr are populated from InfluxDB at export time.
        Rows before a sensor outage have targets pointing into the post-gap glitch
        window; those rows survive with normal timestamps and are not caught by any
        gap-row drop.  Nulling them here lets the subsequent dropna() remove them.
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
            print(f"\n✅ {label}: no cross-gap target contamination detected")
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

        if n_nulled > 0:
            print(f"\n⚠️  {label}: nulled {n_nulled} cross-gap target lookups across "
                  f"{gap_positions.size} gap(s) — will be removed by dropna()")
        else:
            print(f"\n✅ {label}: no cross-gap target contamination detected")

        return df

    # Load preprocessed data
    train_df = pd.read_csv("../train_data_sf.csv")
    val_df = pd.read_csv("../val_data_sf.csv")

    # Set DatetimeIndex from the exported Unix-second timestamp column so that
    # gap detection in _invalidate_targets_crossing_gaps works correctly.
    if 'timestamp' in train_df.columns:
        train_df.index = pd.to_datetime(train_df['timestamp'], unit='s', utc=True)
    if 'timestamp' in val_df.columns:
        val_df.index = pd.to_datetime(val_df['timestamp'], unit='s', utc=True)

    # Null target values that reference post-gap (potentially glitched) InfluxDB data.
    # Must run before diff computation so corrupted targets never enter training targets.
    # tol_s=600: only invalidate around real outages (10+ min gaps); the 9000+ single
    # missed-minute readings in the dataset do NOT corrupt forward-looking targets.
    train_df = _invalidate_targets_crossing_gaps(train_df, "train_df", tol_s=600)
    val_df   = _invalidate_targets_crossing_gaps(val_df,   "val_df",   tol_s=600)

    # Add lag features (30 minutes ago) using np.roll for lag, set first 30 entries to np.nan
    def lag_feature(arr, lag):
        arr = np.asarray(arr)
        lagged = np.roll(arr, lag)
        lagged[:lag] = np.nan
        return lagged
    train_df['temp_lag30'] = lag_feature(train_df['temperature'].values, 30)
    train_df['humidity_lag30'] = lag_feature(train_df['relative_humidity'].values, 30)
    val_df['temp_lag30'] = lag_feature(val_df['temperature'].values, 30)
    val_df['humidity_lag30'] = lag_feature(val_df['relative_humidity'].values, 30)
    
    # Additional lag features for multiple time horizons
    train_df['temp_lag60'] = lag_feature(train_df['temperature'].values, 60)   # 1 hour ago
    train_df['temp_lag120'] = lag_feature(train_df['temperature'].values, 120) # 2 hours ago
    train_df['humidity_lag60'] = lag_feature(train_df['relative_humidity'].values, 60)   # 1 hour ago
    train_df['humidity_lag120'] = lag_feature(train_df['relative_humidity'].values, 120) # 2 hours ago
    val_df['temp_lag60'] = lag_feature(val_df['temperature'].values, 60)
    val_df['temp_lag120'] = lag_feature(val_df['temperature'].values, 120)
    val_df['humidity_lag60'] = lag_feature(val_df['relative_humidity'].values, 60)
    val_df['humidity_lag120'] = lag_feature(val_df['relative_humidity'].values, 120)
    
    # Add more environmental lag features (wind, UV, pressure)
    train_df['wind_avg_lag30'] = lag_feature(train_df['wind_avg'].values, 30)
    train_df['wind_gust_lag30'] = lag_feature(train_df['wind_gust'].values, 30)
    train_df['uv_lag30'] = lag_feature(train_df['uv'].values, 30)
    train_df['pressure_lag30'] = lag_feature(train_df['station_pressure'].values, 30)
    val_df['wind_avg_lag30'] = lag_feature(val_df['wind_avg'].values, 30)
    val_df['wind_gust_lag30'] = lag_feature(val_df['wind_gust'].values, 30)
    val_df['uv_lag30'] = lag_feature(val_df['uv'].values, 30)
    val_df['pressure_lag30'] = lag_feature(val_df['station_pressure'].values, 30)

    # Exp 8: time_of_day features removed to force model to learn from physical lag features.
    # train_df['time_of_day_sin'] = np.sin(2 * np.pi * train_df['time_of_day'] / 24.0)
    # train_df['time_of_day_cos'] = np.cos(2 * np.pi * train_df['time_of_day'] / 24.0)
    # val_df['time_of_day_sin'] = np.sin(2 * np.pi * val_df['time_of_day'] / 24.0)
    # val_df['time_of_day_cos'] = np.cos(2 * np.pi * val_df['time_of_day'] / 24.0)
    # train_df['time_of_day_sin2'] = np.sin(4 * np.pi * train_df['time_of_day'] / 24.0)
    # train_df['time_of_day_cos2'] = np.cos(4 * np.pi * train_df['time_of_day'] / 24.0)
    # val_df['time_of_day_sin2'] = np.sin(4 * np.pi * val_df['time_of_day'] / 24.0)
    # val_df['time_of_day_cos2'] = np.cos(4 * np.pi * val_df['time_of_day'] / 24.0)

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

    # Define feature and target columns
    features = [
        'uv', 'wind_avg', 'wind_gust',
        'solar_radiation', 'illuminance',
        'relative_humidity', 'station_pressure',
        'day_of_year_sin', 'day_of_year_cos',
        'temp_lag30', 'humidity_lag30', 'temp_lag60', 'humidity_lag60', 'temp_lag120', 'humidity_lag120',
        'wind_avg_lag30', 'wind_gust_lag30', 'uv_lag30', 'pressure_lag30'
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

    # Sequence length in minutes for full time-series window
    SEQ_LEN = 180  # 3 hours of history

    ## Per-feature min/max scaling with ±5% padding
    # Domain bounds for select features
    domain_bounds = {
        "wind_gust": (0, None),
        "wind_avg": (0, None),
        "uv": (0, None),
        "solar_radiation": (0, None),
        "illuminance": (0, None),
        "humidity_lag30": (0, 100),
        "relative_humidity": (0, 100),
        "temp_lag30": (-10, 55),
        "day_of_year_sin": (-1, 1),         # Sine component naturally bounded [-1, 1]
        "day_of_year_cos": (-1, 1),         # Cosine component naturally bounded [-1, 1]
        "wind_direction_sin": (-1, 1),      # Wind direction sine component (cyclical 0-360°)
        "wind_direction_cos": (-1, 1),      # Wind direction cosine component (cyclical 0-360°)
        "wind_lull": (0, None),             # Minimum wind speed
        "rain_accumulated": (0, None),      # Accumulated rainfall
        # Manual interaction features - COMMENTED OUT: Using learned FeatureInteraction layer instead
        # "time_sin_solar": (None, None),     # Interaction feature - dynamic bounds
        # "time_cos_solar": (None, None),     # Interaction feature - dynamic bounds
        # "time_sin_uv": (None, None),        # Interaction feature - dynamic bounds
        # "time_cos_uv": (None, None),        # Interaction feature - dynamic bounds
        # "time_sin_wind": (None, None),      # Interaction feature - dynamic bounds
        # "time_cos_wind": (None, None),      # Interaction feature - dynamic bounds
        # "time_sin_temp_lag": (None, None),  # Interaction feature - dynamic bounds
        # "time_cos_temp_lag": (None, None),  # Interaction feature - dynamic bounds
        "temp_lag60": (-10, 55),            # 1-hour lag temperature
        "temp_lag120": (-10, 55),           # 2-hour lag temperature
        "humidity_lag60": (0, 100),         # 1-hour lag humidity
        "humidity_lag120": (0, 100),        # 2-hour lag humidity
        "wind_avg_lag30": (0, None),        # 30-min lag wind average
        "wind_gust_lag30": (0, None),       # 30-min lag wind gust
        "uv_lag30": (0, None),              # 30-min lag UV
        "pressure_lag30": (None, None),     # 30-min lag pressure
        "station_pressure": (None, None),
        # "season_sin_temp_lag": (None, None), # Seasonal-temp interaction
        # "season_cos_temp_lag": (None, None), # Seasonal-temp interaction
        # "season_sin_humidity": (None, None), # Seasonal-humidity interaction
        # "season_cos_humidity": (None, None), # Seasonal-humidity interaction
        "temperature_delta": (None, None),  # Allow dynamic bounds for expanded delta range
        "pressure_delta": (None, None),     # Allow dynamic bounds for pressure delta range
        "humidity_delta": (None, None),     # Allow dynamic bounds for humidity delta range
        "illuminance_delta": (None, None),  # Allow dynamic bounds for illuminance delta range
        "solar_radiation_delta": (None, None)  # Allow dynamic bounds for solar radiation delta range
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
    with open("input_scaler_5a.json", "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Normalize target values (temperature differences)
    # Calculate bounds from actual temperature difference data with padding
    y_min = train_df[targets].min().min() - 2  # Add padding for temperature differences
    y_max = train_df[targets].max().max() + 2  # Add padding for temperature differences
    # Save original target range
    target_range = (y_min, y_max)
    train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
    val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

    with open("target_scaler_5a.json", "w") as f:
        json.dump({"min": y_min, "max": y_max, "range": target_range}, f)

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
    print("Temperature difference scaling bounds:")
    print(f"Temperature difference range: {y_min:.2f}°C to {y_max:.2f}°C")
    print(f"Original temperature difference stats:")
    print(f"  1hr: min={train_df['temp_diff_1hr'].min():.2f}°C, max={train_df['temp_diff_1hr'].max():.2f}°C")
    print(f"  2hr: min={train_df['temp_diff_2hr'].min():.2f}°C, max={train_df['temp_diff_2hr'].max():.2f}°C")
    print(f"  3hr: min={train_df['temp_diff_3hr'].min():.2f}°C, max={train_df['temp_diff_3hr'].max():.2f}°C")

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
    train_ds = timeseries_dataset_from_array(
        data=X_train_flat,
        targets=y_all_train,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=256,
        shuffle=True,
    )

    val_ds = timeseries_dataset_from_array(
        data=X_val_flat,
        targets=y_all_val,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=256,
        shuffle=False,
    )

    # Split 3-target vector into a tuple matching the model's three outputs
    def split_targets(x, y):
        return x, (y[:, 0], y[:, 1], y[:, 2])

    train_ds = train_ds.map(split_targets)
    val_ds = val_ds.map(split_targets)

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

    # Edge TPU-compatible feature interaction layer
    # Uses only Dense layers and element-wise operations (fully supported by Edge TPU)
    # Dense layers can learn feature interactions naturally through their weight matrices
    def build_feature_interaction_path(inputs, embedding_dim=16, projection_dim=32):
        """
        Builds an Edge TPU-compatible feature interaction path.
        Uses Dense layers (fully supported) which can learn interactions through
        their learned weight matrices.
        
        Strategy: 
        1. Project features into interaction embedding space
        2. Apply element-wise square to capture non-linear interactions
        3. Combine original and squared projections
        4. Final Dense layer to learn interaction patterns
        """
        # First projection: learns linear combinations that can capture interactions
        projected = tf.keras.layers.Dense(embedding_dim, activation='relu', use_bias=False, name='interaction_embed')(inputs)
        
        # Element-wise square to capture non-linear/pairwise interactions
        # This is equivalent to capturing x_i * x_j terms
        # Using Multiply layer (more explicit than Lambda for Edge TPU compatibility)
        squared = tf.keras.layers.Multiply(name='interaction_square')([projected, projected])
        
        # Combine original and squared features to capture both linear and interaction terms
        combined = tf.keras.layers.Concatenate(name='interaction_combine')([projected, squared])
        
        # Final projection: learns which interactions are important
        interactions = tf.keras.layers.Dense(projection_dim, activation='relu', use_bias=False, name='interaction_output')(combined)
        
        return interactions

    def build_and_train_model(name):
        print(f"\n--- Running: {name} ---\n")

        # Input is a full (time, feature) sequence
        input_layer = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")

        # Stage 1: temporal pooling — 180 timesteps → 30 using 6-minute averaging.
        # AveragePooling1D generates AVERAGE_POOL_2D (EdgeTPU v1, fully supported).
        # Avoids Dense on 3D input which generates FULLY_CONNECTED version 9 (not supported).
        pooled = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, name='temporal_pool')(input_layer)

        # Stage 2: flatten pooled sequence — 30 × n_features ≈ 810 dims (below ~1660 SRAM threshold).
        # Explicit Reshape (not Flatten) required: Flatten emits two -1 dims which TF can't trace.
        flat = tf.keras.layers.Reshape(((SEQ_LEN // 6) * n_features,), name='flatten_sequence')(pooled)

        # Stage 3: single shared bottleneck on the now-small flat vector
        bottleneck = tf.keras.layers.Dense(64, activation='relu', use_bias=False, name='bottleneck')(flat)

        # All downstream branches use the 64-dim bottleneck
        interaction_projection = build_feature_interaction_path(bottleneck, embedding_dim=16, projection_dim=32)

        wide = tf.keras.layers.Dense(16, use_bias=False, name="wide_dense")(bottleneck)
        deep = tf.keras.layers.Dense(128, activation='relu', use_bias=False, name="deep_dense1")(bottleneck)
        deep = tf.keras.layers.Dropout(0.3, name="deep_dropout")(deep)

        res = tf.keras.layers.Dense(64, activation='relu', use_bias=False, name="deep_res_dense1")(deep)
        shortcut = tf.keras.layers.Dense(64, use_bias=False, name="deep_res_shortcut")(deep)
        res = tf.keras.layers.Add(name="deep_res_add")([shortcut, res])
        res = tf.keras.layers.Dense(32, activation='relu', use_bias=False, name="deep_res_dense2")(res)


        merged = tf.keras.layers.Concatenate(name="merged_features")(
            [wide, res, interaction_projection]
        )
        output_1 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, name='diff_1hr')(merged)
        output_2 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, name='diff_2hr')(merged)
        output_3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, name='diff_3hr')(merged)
        model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics={
                'diff_1hr': 'mae',
                'diff_2hr': 'mae',
                'diff_3hr': 'mae'
            }
        )
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/model_{epoch:02d}.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=[early_stopping, checkpoint_cb, reduce_lr]
        )

        # The early stopping callback with restore_best_weights=True already restored the best weights

        eval_results = model.evaluate(val_ds, verbose=0)
        val_loss = eval_results[0]
        val_mae = np.mean(eval_results[1:])  # average MAE across 3 targets
        # Report MAEs in original units (°C difference)
        diff_1_mae_c = eval_results[1] * (y_max - y_min) / 2  # Convert from [-1,1] to original scale
        diff_2_mae_c = eval_results[2] * (y_max - y_min) / 2
        diff_3_mae_c = eval_results[3] * (y_max - y_min) / 2
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

        # Build a concrete function with a fixed input shape so that all tensors
        # are statically sized (required by the Edge TPU compiler).
        # Input shape is fixed to batch_size=1, SEQ_LEN timesteps, and n_features features.
        run_model = tf.function(model)
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([1, SEQ_LEN, n_features], tf.float32, name="input")
        )
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

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

        quantized_tflite_model = converter.convert()

        tflite_fname = f"weather_model_5a_quant_{name}.tflite"
        with open(tflite_fname, "wb") as f:
            f.write(quantized_tflite_model)

        tflite_model_size_kb = os.path.getsize(tflite_fname) / 1024

        best_epoch = np.argmin(history.history['val_loss']) + 1
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

        with open(f"results_5a_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Configuration
    NUM_RUNS = 1  # Number of training runs to perform
    for run_id in range(NUM_RUNS):
        run_name = f"no_tod_run{run_id+1}"
        build_and_train_model(run_name)

    results = []
    # Only collect results from the current run session
    for run_id in range(NUM_RUNS):
        json_file = f"results_5a_no_tod_run{run_id+1}.json"
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
    best_model_file = f"weather_model_5a_quant_{best['name']}.tflite"
    shutil.copy(best_model_file, "weather_model_5a_best.tflite")
    print(f"Best model copied to: weather_model_5a_best.tflite")
    
    # Validate the best quantized model
    print("\n" + "="*50)
    print("VALIDATING BEST QUANTIZED MODEL")
    print("="*50)
    y_val_array = val_df[targets].values
    validate_quantized_model("weather_model_5a_best.tflite", X_val_small, y_val_array, y_min, y_max)

# --- Validate quantized TFLite model on validation data ---
def validate_quantized_model(tflite_model_path, X_val, y_val, y_min, y_max, num_samples=500):
    import tensorflow as tf
    import numpy as np

    print(f"\nValidating TFLite model on {num_samples} samples...")

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

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
        y_preds_rescaled = 0.5 * (np.array(y_preds_dequant[j]) + 1) * (y_max - y_min) + y_min
        y_val_rescaled = 0.5 * (y_val_subset[:, j] + 1) * (y_max - y_min) + y_min
        mae = np.mean(np.abs(y_val_rescaled - y_preds_rescaled))
        print(f"  {name}: {mae:.2f} °C")



# Entry point guard
if __name__ == "__main__":
    main()
