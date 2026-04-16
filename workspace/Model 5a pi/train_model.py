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

    # intra_op: threads used within a single op (matrix multiply, etc.) — use all cores
    # inter_op: ops allowed to run concurrently — keep at 1 for sequential batch training
    #   so all cores stay focused on one op at a time instead of splitting across ops.
    tf.config.threading.set_intra_op_parallelism_threads(cores)
    tf.config.threading.set_inter_op_parallelism_threads(1)
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

    import argparse
    import datetime
    import gc
    from influxdb import InfluxDBClient

    parser = argparse.ArgumentParser(description="Train weather model 5a")
    parser.add_argument(
        '--dataset',
        choices=['home', 'ps', 'combined'],
        default='home',
        help=(
            'Dataset to train on: '
            '"home" = home station (weather db), '
            '"ps" = Palm Springs station (ps_smartweather db), '
            '"combined" = both datasets merged'
        )
    )
    args = parser.parse_args()
    print(f"Dataset: {args.dataset}")

    # Model name suffix reflects the dataset so outputs don't overwrite each other
    MODEL_TAG = {'home': '5a', 'ps': '5a_ps', 'combined': '5a_combined'}[args.dataset]

    # Configuration
    TRAIN_VAL_SPLIT_RATIO = 0.5  # 0.5 = 50/50 split, 0.7 = 70/30 split
    MIN_SAMPLES_PER_SET = 1000   # Minimum samples required for each set

    INFLUX_CONFIGS = {
        'home': {'host': '10.0.1.188', 'port': 8086,
                 'username': 'admin', 'password': '24planet',
                 'database': 'weather'},
        'ps':   {'host': '10.0.1.188', 'port': 8086,
                 'username': 'admin', 'password': '24planet',
                 'database': 'ps_smartweather'},
    }
    MEASUREMENT = "wf/obs_st"
    NEEDED_FIELDS = {
        'temperature', 'relative_humidity', 'wind_avg', 'wind_gust',
        'uv', 'station_pressure', 'solar_radiation', 'illuminance',
        'wind_direction', 'wind_lull', 'rain_accumulated',
    }

    def load_influx_data(cfg, label):
        """Load all data from one InfluxDB database into a time-indexed DataFrame."""
        print(f"\nConnecting to InfluxDB [{label}] ({cfg['database']})...")
        client = InfluxDBClient(**cfg)

        # Verify measurement exists
        measurements = [m['name'] for m in client.query('SHOW MEASUREMENTS').get_points()]
        if MEASUREMENT not in measurements:
            raise ValueError(f"[{label}] Measurement '{MEASUREMENT}' not found. Available: {measurements}")

        # Get time range (MIN/MAX query fails on some InfluxDB versions; fall back to ASC/DESC LIMIT 1)
        print(f"  Getting time range...")
        time_points = list(client.query(
            f'SELECT MIN(time) as min_time, MAX(time) as max_time FROM "{MEASUREMENT}"'
        ).get_points())
        if time_points and time_points[0].get('min_time'):
            min_time = time_points[0]['min_time']
            max_time = time_points[0]['max_time']
        else:
            first = list(client.query(f'SELECT time, temperature FROM "{MEASUREMENT}" ORDER BY time ASC LIMIT 1').get_points())
            last  = list(client.query(f'SELECT time, temperature FROM "{MEASUREMENT}" ORDER BY time DESC LIMIT 1').get_points())
            if not first or not last:
                raise ValueError(f"[{label}] No data found in '{MEASUREMENT}'.")
            min_time, max_time = first[0]['time'], last[0]['time']
        print(f"  Time range: {min_time} → {max_time}")

        # Determine which needed fields are actually present
        available = [f['fieldKey'] for f in client.query(f'SHOW FIELD KEYS FROM "{MEASUREMENT}"').get_points()]
        select_fields = [f for f in available if f in NEEDED_FIELDS]
        if not select_fields:
            raise ValueError(f"[{label}] None of the required fields found. Available: {available}")
        print(f"  Selecting {len(select_fields)} fields: {select_fields}")

        # Time-window pagination: LIMIT-based queries trip the max-select-point limit
        # because InfluxDB counts scanned points per shard before applying LIMIT.
        # Bounding by time restricts the scan to only the relevant shards.
        points_per_day = 1440 * len(select_fields)
        window_days = max(1, int(90000 / points_per_day))
        print(f"  Using {window_days}-day windows (~{window_days * points_per_day:,} pts/window)")

        select_clause = ', '.join(f'"{f}"' for f in select_fields)
        window = datetime.timedelta(days=window_days)
        range_start = pd.to_datetime(min_time).to_pydatetime().replace(tzinfo=datetime.timezone.utc)
        range_end   = pd.to_datetime(max_time).to_pydatetime().replace(tzinfo=datetime.timezone.utc)

        batch_dfs = []
        win_start = range_start
        while win_start <= range_end:
            win_end = win_start + window
            start_str = win_start.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str   = win_end.strftime('%Y-%m-%dT%H:%M:%SZ')
            points = list(client.query(
                f'SELECT {select_clause} FROM "{MEASUREMENT}"'
                f' WHERE time >= \'{start_str}\' AND time < \'{end_str}\''
                f' ORDER BY time ASC'
            ).get_points())
            if points:
                bdf = pd.DataFrame(points)
                bdf['time'] = pd.to_datetime(bdf['time'])
                bdf.set_index('time', inplace=True)
                bdf = bdf.astype({c: np.float32 for c in bdf.select_dtypes('float64').columns})
                batch_dfs.append(bdf)
                del points, bdf
                print(f"  {start_str} → {end_str}: loaded (total: {sum(len(d) for d in batch_dfs):,})")
            win_start = win_end

        if not batch_dfs:
            raise ValueError(f"[{label}] No data loaded from InfluxDB.")

        result = pd.concat(batch_dfs)
        del batch_dfs
        gc.collect()
        print(f"  ✅ {label}: {len(result):,} rows loaded")
        return result

    # Load dataset(s)
    if args.dataset == 'combined':
        df_home = load_influx_data(INFLUX_CONFIGS['home'], 'home')
        df_ps   = load_influx_data(INFLUX_CONFIGS['ps'],   'ps')
        print(f"\nCombining datasets ({len(df_home):,} home + {len(df_ps):,} PS)...")
        df = pd.concat([df_home, df_ps]).sort_index()
        del df_home, df_ps
        gc.collect()
    else:
        df = load_influx_data(INFLUX_CONFIGS[args.dataset], args.dataset)

    print(f"✅ Successfully loaded {len(df):,} total data points")

    # Derived time features
    df['day_of_year'] = df.index.dayofyear
    df['time_of_day'] = df.index.hour + df.index.minute / 60.0

    # Lag features on the full dataset (before splitting avoids boundary artifacts)
    def lag_feature(arr, lag):
        arr = np.asarray(arr)
        lagged = np.roll(arr, lag)
        lagged[:lag] = np.nan
        return lagged

    df['temp_lag30'] = lag_feature(df['temperature'].values, 30)
    df['temp_lag60'] = lag_feature(df['temperature'].values, 60)
    df['temp_lag120'] = lag_feature(df['temperature'].values, 120)
    df['humidity_lag30'] = lag_feature(df['relative_humidity'].values, 30)
    df['humidity_lag60'] = lag_feature(df['relative_humidity'].values, 60)
    df['humidity_lag120'] = lag_feature(df['relative_humidity'].values, 120)
    df['wind_avg_lag30'] = lag_feature(df['wind_avg'].values, 30)
    df['wind_gust_lag30'] = lag_feature(df['wind_gust'].values, 30)
    df['uv_lag30'] = lag_feature(df['uv'].values, 30)
    df['pressure_lag30'] = lag_feature(df['station_pressure'].values, 30)

    # Cyclical encoding for time_of_day
    df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_sin2'] = np.sin(4 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_cos2'] = np.cos(4 * np.pi * df['time_of_day'] / 24.0)

    # Cyclical encoding for day_of_year
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Cyclical encoding for wind_direction if available
    if 'wind_direction' in df.columns:
        df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360.0)
        df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360.0)

    # Future temperature targets (1-minute interval data, so 60 rows = 1 hour)
    df['temp_t+1hr'] = df['temperature'].shift(-60)
    df['temp_t+2hr'] = df['temperature'].shift(-120)
    df['temp_t+3hr'] = df['temperature'].shift(-180)

    # Temperature difference targets
    df['temp_diff_1hr'] = df['temp_t+1hr'] - df['temperature']
    df['temp_diff_2hr'] = df['temp_t+2hr'] - df['temperature']
    df['temp_diff_3hr'] = df['temp_t+3hr'] - df['temperature']

    # Drop rows with NaNs (lag features + future targets)
    print(f"Data before dropping NaNs: {len(df)} rows")
    df.dropna(inplace=True)
    print(f"Data after dropping NaNs: {len(df)} rows")

    if len(df) == 0:
        raise ValueError("No valid data remaining after preprocessing.")

    # Temporal train/val split: older data = training, newer data = validation
    total_duration = df.index.max() - df.index.min()
    split_time = df.index.min() + total_duration * TRAIN_VAL_SPLIT_RATIO
    split_idx = df.index.get_indexer([split_time], method='nearest')[0]
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    print(f"Temporal split at {split_time}:")
    print(f"  Training:   {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Validation: {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")

    if len(train_df) < MIN_SAMPLES_PER_SET or len(val_df) < MIN_SAMPLES_PER_SET:
        print(f"⚠️  Warning: small dataset (train={len(train_df)}, val={len(val_df)}, min={MIN_SAMPLES_PER_SET})")

    import gc
    del df
    gc.collect()

    # Define feature and target columns
    features = [
        'uv', 'wind_avg', 'wind_gust',
        'solar_radiation', 'illuminance',
        'relative_humidity', 'station_pressure',
        'day_of_year_sin', 'day_of_year_cos',
        'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
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
        "time_of_day_sin": (-1, 1),         # Sine component naturally bounded [-1, 1]
        "time_of_day_cos": (-1, 1),         # Cosine component naturally bounded [-1, 1]
        "time_of_day_sin2": (-1, 1),        # Higher-order sine component
        "time_of_day_cos2": (-1, 1),        # Higher-order cosine component
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
    with open(f"input_scaler_{MODEL_TAG}.json", "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Normalize target values (temperature differences)
    # Calculate bounds from actual temperature difference data with padding
    y_min = train_df[targets].min().min() - 2  # Add padding for temperature differences
    y_max = train_df[targets].max().max() + 2  # Add padding for temperature differences
    # Save original target range
    target_range = (y_min, y_max)
    train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
    val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

    with open(f"target_scaler_{MODEL_TAG}.json", "w") as f:
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

        # Flatten the entire sequence (all timesteps and features) into one wide vector
        # This allows the dense layers to learn Conv2D-like patterns over the full window.
        flat = tf.keras.layers.Reshape((SEQ_LEN * n_features,), name="flatten_sequence")(input_layer)

        # Learn feature interactions automatically (Edge TPU compatible)
        interaction_projection = build_feature_interaction_path(flat, embedding_dim=16, projection_dim=32)
        
        wide = tf.keras.layers.Dense(16, use_bias=False, name="wide_dense")(flat)
        deep = tf.keras.layers.Dense(128, activation='relu', use_bias=False, name="deep_dense1")(flat)
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

        # Optimized optimizer for Mac M1
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
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

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/model_{epoch:02d}.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )

        # Match RPi5 configuration exactly
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=[early_stopping, checkpoint_cb]
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

        tflite_fname = f"weather_model_{MODEL_TAG}_quant_{name}.tflite"
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

        with open(f"results_{MODEL_TAG}_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Configuration
    NUM_RUNS = 1  # Number of training runs to perform
    for run_id in range(NUM_RUNS):
        run_name = f"dense_wide_run{run_id+1}"
        build_and_train_model(run_name)

    results = []
    # Only collect results from the current run session
    for run_id in range(NUM_RUNS):
        json_file = f"results_{MODEL_TAG}_dense_wide_run{run_id+1}.json"
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
    best_model_file = f"weather_model_{MODEL_TAG}_quant_{best['name']}.tflite"
    shutil.copy(best_model_file, f"weather_model_{MODEL_TAG}_best.tflite")
    print(f"Best model copied to: weather_model_{MODEL_TAG}_best.tflite")
    
    # Validate the best quantized model
    print("\n" + "="*50)
    print("VALIDATING BEST QUANTIZED MODEL")
    print("="*50)
    y_val_array = val_df[targets].values
    validate_quantized_model(f"weather_model_{MODEL_TAG}_best.tflite", X_val_small, y_val_array, y_min, y_max)

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
