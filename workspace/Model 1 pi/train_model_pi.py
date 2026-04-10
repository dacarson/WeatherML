def main():
    # This version removes Modin/Dask and uses Numba-accelerated NumPy for rolling slope computation.
    # This provides better multi-core performance on Raspberry Pi and simplifies dependencies.
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    
    # Configuration
    TRAIN_VAL_SPLIT_RATIO = 0.5  # 0.5 = 50/50 split, 0.7 = 70/30 split
    MIN_SAMPLES_PER_SET = 1000   # Minimum samples required for each set
    
    # Data loading configuration for full dataset
    BATCH_SIZE = 50000           # Load data in batches to manage memory
    MAX_MEMORY_MB = 2000         # Maximum memory usage (2GB)
    
    # For 1.5 years of minute data (~800K points), we'll load everything in batches
    # Memory estimate: ~800K points ≈ 800MB RAM (manageable)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Hide GPU devices

    import multiprocessing

    cores = multiprocessing.cpu_count()
    print(f"Setting TensorFlow threads → {cores} cores")

    # Force TensorFlow to use all CPU threads
    tf.config.threading.set_intra_op_parallelism_threads(cores)
    tf.config.threading.set_inter_op_parallelism_threads(cores)
    tf.config.set_soft_device_placement(True)

    # Optional confirmation
    print("TF intra_op threads:", tf.config.threading.get_intra_op_parallelism_threads())
    print("TF inter_op threads:", tf.config.threading.get_inter_op_parallelism_threads())
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import copy
    import subprocess
    import json
    import glob
    from numba import njit, prange
    import flatbuffers
    print(f"flatbuffers version: {getattr(flatbuffers, '__version__', 'unknown')}")

    # Connect to InfluxDB and load live data
    from influxdb import InfluxDBClient
    import datetime
    
    print("Connecting to InfluxDB...")
    client = InfluxDBClient(
        host="10.0.1.188",
        port=8086,
        username="admin",
        password="24planet",
        database="weather"
    )
    
    # Load full dataset with batched queries
    print("Loading full dataset from InfluxDB with batched queries...")
    
    # First, check if the measurement exists and get basic info
    print("Checking InfluxDB connection and data availability...")
    
    # List all measurements to see what's available
    measurements_query = 'SHOW MEASUREMENTS'
    measurements_result = client.query(measurements_query)
    measurements = list(measurements_result.get_points())
    print(f"Available measurements: {[m['name'] for m in measurements]}")
    
    # Check if our target measurement exists
    target_measurement = "wf/obs_st"
    measurement_names = [m['name'] for m in measurements]
    
    if target_measurement not in measurement_names:
        print(f"❌ Measurement '{target_measurement}' not found!")
        print("Available measurements:")
        for m in measurements:
            print(f"  - {m['name']}")
        raise ValueError(f"Target measurement '{target_measurement}' not found in database")
    
    # Get the time range of all available data
    print(f"Getting data time range from '{target_measurement}'...")
    time_range_query = f'SELECT MIN(time) as min_time, MAX(time) as max_time FROM "{target_measurement}"'
    time_result = client.query(time_range_query)
    time_points = list(time_result.get_points())
    
    if not time_points:
        print("❌ No time range data returned. Checking for any data...")
        # Try a simple query to see if there's any data at all
        test_query = f'SELECT * FROM "{target_measurement}" LIMIT 1'
        test_result = client.query(test_query)
        test_points = list(test_result.get_points())
        
        if not test_points:
            raise ValueError(f"No data found in measurement '{target_measurement}'. Please check your data.")
        else:
            print(f"✅ Found data in '{target_measurement}', but time range query failed")
            print("Trying alternative time range query...")
            
            # Try alternative approach: get first and last records with a non-time field
            try:
                first_query = f'SELECT time, temperature FROM "{target_measurement}" ORDER BY time ASC LIMIT 1'
                last_query = f'SELECT time, temperature FROM "{target_measurement}" ORDER BY time DESC LIMIT 1'
                
                first_result = client.query(first_query)
                last_result = client.query(last_query)
                
                first_points = list(first_result.get_points())
                last_points = list(last_result.get_points())
                
                if first_points and last_points:
                    min_time = first_points[0]['time']
                    max_time = last_points[0]['time']
                    print(f"✅ Time range (alternative method): {min_time} to {max_time}")
                    
                    # Convert timestamps to readable format for verification
                    from datetime import datetime
                    min_dt = pd.to_datetime(min_time)
                    max_dt = pd.to_datetime(max_time)
                    duration = max_dt - min_dt
                    print(f"  Duration: {duration.days} days, {duration.seconds//3600} hours")
                else:
                    # Fallback to single record
                    first_record = test_points[0]
                    min_time = first_record['time']
                    max_time = first_record['time']
                    print(f"Using single record time: {min_time}")
            except Exception as e:
                print(f"Alternative time range query also failed: {e}")
                # Fallback to single record
                first_record = test_points[0]
                min_time = first_record['time']
                max_time = first_record['time']
                print(f"Using single record time: {min_time}")
    else:
        time_data = time_points[0]
        min_time = time_data['min_time']
        max_time = time_data['max_time']
        print(f"✅ Time range: {min_time} to {max_time}")
    
    # Get total count in a separate query
    print("Getting total data count...")
    
    # For InfluxDB v1.11.8, try different count approaches
    count_queries = [
        f'SELECT COUNT(temperature) FROM "{target_measurement}"',  # Count a specific field
        f'SELECT COUNT(*) FROM "{target_measurement}"',            # Standard count
    ]
    
    total_count = None
    for i, count_query in enumerate(count_queries):
        try:
            print(f"  Trying count query {i+1}: {count_query}")
            count_result = client.query(count_query)
            count_points = list(count_result.get_points())
            
            if count_points:
                count_data = count_points[0]
                print(f"  Debug - Count query result keys: {list(count_data.keys())}")
                
                # Try to find the count value
                for key, value in count_data.items():
                    if isinstance(value, (int, float)) and value > 0:
                        total_count = int(value)
                        print(f"  ✅ Found count in key '{key}': {total_count:,}")
                        break
                
                if total_count:
                    break
        except Exception as e:
            print(f"  Count query {i+1} failed: {e}")
            continue
    
    if not total_count:
        print("❌ All count queries failed. Estimating from sample...")
        # Fallback: estimate count from a sample
        sample_query = f'SELECT * FROM "{target_measurement}" LIMIT 1000'
        sample_result = client.query(sample_query)
        sample_points = list(sample_result.get_points())
        total_count = len(sample_points)
        print(f"⚠️  Using sample-based estimate: {total_count} (may be inaccurate)")
    else:
        print(f"✅ Total data points: {total_count:,}")
    
    print(f"Data range: {min_time} to {max_time}")
    print(f"Total data points: {total_count:,}")
    
    if total_count == 0:
        raise ValueError("No data found in InfluxDB. Please check your connection and data availability.")
    
    # Calculate memory usage estimate
    estimated_mb = total_count * 0.001  # Rough estimate: 1KB per record
    print(f"Estimated memory usage: ~{estimated_mb:.1f} MB")
    
    if estimated_mb > MAX_MEMORY_MB:
        print(f"⚠️  Warning: Dataset may use {estimated_mb:.1f}MB (limit: {MAX_MEMORY_MB}MB)")
        print("Consider reducing BATCH_SIZE or increasing MAX_MEMORY_MB if you encounter memory issues.")
    
    # Load data in batches to manage memory
    print(f"Loading data in batches of {BATCH_SIZE:,} records...")
    all_points = []
    offset = 0
    
    while True:
        print(f"  Loading batch {offset//BATCH_SIZE + 1}... (offset: {offset:,})")
        
        # Query batch with OFFSET and LIMIT
        batch_query = f'SELECT * FROM "{target_measurement}" ORDER BY time ASC LIMIT {BATCH_SIZE} OFFSET {offset}'
        result = client.query(batch_query)
        batch_points = list(result.get_points())
        
        if not batch_points:
            break  # No more data
            
        all_points.extend(batch_points)
        offset += len(batch_points)
        
        print(f"    Loaded {len(batch_points):,} records (total: {len(all_points):,})")
        
        # Break if we got fewer records than requested (end of data)
        if len(batch_points) < BATCH_SIZE:
            break
    
    points = all_points
    print(f"✅ Successfully loaded {len(points):,} data points")
    
    if len(points) == 0:
        raise ValueError("No data loaded from InfluxDB. Please check your connection and data availability.")
    
    # Convert to DataFrame
    df = pd.DataFrame(points)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Add derived features
    df['day_of_year'] = df.index.dayofyear
    df['time_of_day'] = df.index.hour + df.index.minute / 60.0
    
    # Add cyclical encoding for time_of_day (k=1,2,3 harmonics)
    df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_sin2'] = np.sin(2 * 2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_cos2'] = np.cos(2 * 2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_sin3'] = np.sin(3 * 2 * np.pi * df['time_of_day'] / 24.0)
    df['time_of_day_cos3'] = np.cos(3 * 2 * np.pi * df['time_of_day'] / 24.0)

    # Add cyclical encoding for day_of_year (k=1..4 harmonics)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_sin2'] = np.sin(2 * 2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_cos2'] = np.cos(2 * 2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_sin3'] = np.sin(3 * 2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_cos3'] = np.cos(3 * 2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_sin4'] = np.sin(4 * 2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_cos4'] = np.cos(4 * 2 * np.pi * df['day_of_year'] / 365.0)
    
    # Add lag features (30 minutes ago) using np.roll for lag, set first 30 entries to np.nan
    def lag_feature(arr, lag):
        arr = np.asarray(arr)
        lagged = np.roll(arr, lag)
        lagged[:lag] = np.nan
        return lagged
    
    df['temp_lag30'] = lag_feature(df['temperature'].values, 30)
    df['temp_lag60'] = lag_feature(df['temperature'].values, 60)
    df['temp_lag120'] = lag_feature(df['temperature'].values, 120)
    df['temp_lag360'] = lag_feature(df['temperature'].values, 360)    # 6h
    df['temp_lag720'] = lag_feature(df['temperature'].values, 720)    # 12h
    df['temp_lag1440'] = lag_feature(df['temperature'].values, 1440)  # 24h

    df['humidity_lag30'] = lag_feature(df['relative_humidity'].values, 30)
    df['humidity_lag60'] = lag_feature(df['relative_humidity'].values, 60)
    df['humidity_lag120'] = lag_feature(df['relative_humidity'].values, 120)
    df['humidity_lag1440'] = lag_feature(df['relative_humidity'].values, 1440)

    # 24h rolling baseline and deviation for temperature
    df['temp_mean_24h'] = (
        pd.Series(df['temperature']).rolling(window=1440, min_periods=1440).mean().values
    )
    df['temp_minus_mean_24h'] = df['temperature'] - df['temp_mean_24h']
    
    # Numba-accelerated rolling slope for multi-core performance
    @njit(parallel=True)
    def rolling_slope_numba(data, window):
        n = len(data)
        slopes = np.full(n, np.nan)
        x = np.arange(window)
        x_mean = np.mean(x)
        denom = np.sum((x - x_mean) ** 2)
        for i in prange(window - 1, n):
            y = data[i - window + 1:i + 1]
            if np.any(np.isnan(y)):
                continue
            y_mean = np.mean(y)
            num = np.sum((x - x_mean) * (y - y_mean))
            slopes[i] = num / denom
        return slopes

    # Compute deltas using numba-accelerated rolling slope
    print("Computing feature deltas...")
    df['temperature_delta'] = rolling_slope_numba(df['temperature'].values, 15)
    df['pressure_delta'] = rolling_slope_numba(df['station_pressure'].values, 15)
    df['humidity_delta'] = rolling_slope_numba(df['relative_humidity'].values, 15)
    df['illuminance_delta'] = rolling_slope_numba(df['illuminance'].values, 15)
    df['solar_radiation_delta'] = rolling_slope_numba(df['solar_radiation'].values, 15)
    
    # Create targets (temperature at future time points)
    print("Creating future temperature targets...")
    df['temp_t+1hr'] = df['temperature'].shift(-60)   # 1 hour = 60 samples (1 min intervals)
    df['temp_t+2hr'] = df['temperature'].shift(-120)  # 2 hours = 120 samples
    df['temp_t+3hr'] = df['temperature'].shift(-180)  # 3 hours = 180 samples
    
    # Calculate temperature differences as targets
    df['temp_diff_1hr'] = df['temp_t+1hr'] - df['temperature']
    df['temp_diff_2hr'] = df['temp_t+2hr'] - df['temperature']
    df['temp_diff_3hr'] = df['temp_t+3hr'] - df['temperature']
    
    # Drop rows with NaNs (especially important for future targets)
    print(f"Data before dropping NaNs: {len(df)} rows")
    df.dropna(inplace=True)
    print(f"Data after dropping NaNs: {len(df)} rows")
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after preprocessing. Check for missing values in your InfluxDB data.")
    
    # Temporal split: older data for training, newer data for validation
    # Split at the temporal midpoint (50/50 by time, not by count)
    total_duration = df.index.max() - df.index.min()
    split_time = df.index.min() + total_duration * TRAIN_VAL_SPLIT_RATIO
    
    # Find the index closest to the split time
    split_idx = df.index.get_indexer([split_time], method='nearest')[0]
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Temporal split (by time, not count):")
    print(f"  Split time: {split_time}")
    print(f"  Training data: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Validation data: {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    
    # Calculate actual time split ratio
    train_duration = (train_df.index.max() - train_df.index.min()).total_seconds() / 3600
    val_duration = (val_df.index.max() - val_df.index.min()).total_seconds() / 3600
    total_duration_hours = train_duration + val_duration
    actual_train_ratio = train_duration / total_duration_hours
    actual_val_ratio = val_duration / total_duration_hours
    
    print(f"  Time split: {actual_train_ratio:.1%} training, {actual_val_ratio:.1%} validation")
    
    # Data quality validation
    print("\n=== DATA QUALITY CHECKS ===")
    
    # Check for missing values in key fields
    required_fields = ['temperature', 'station_pressure', 'relative_humidity', 'illuminance', 'solar_radiation', 'uv', 'wind_avg', 'wind_gust']
    missing_counts = df[required_fields].isnull().sum()
    if missing_counts.any():
        print("⚠️  Missing values found:")
        for field, count in missing_counts[missing_counts > 0].items():
            print(f"  {field}: {count} missing values")
    
    # Check data range and outliers
    print(f"\nData statistics:")
    print(f"  Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
    print(f"  Pressure range: {df['station_pressure'].min():.1f} to {df['station_pressure'].max():.1f} hPa")
    print(f"  Humidity range: {df['relative_humidity'].min():.1f}% to {df['relative_humidity'].max():.1f}%")
    
    # Check for extreme outliers (beyond 3 standard deviations)
    for field in ['temperature', 'station_pressure', 'relative_humidity']:
        mean_val = df[field].mean()
        std_val = df[field].std()
        outliers = df[(df[field] < mean_val - 3*std_val) | (df[field] > mean_val + 3*std_val)]
        if len(outliers) > 0:
            print(f"  ⚠️  {field}: {len(outliers)} extreme outliers detected")
    
    # Verify we have enough data for both sets
    if len(train_df) < MIN_SAMPLES_PER_SET or len(val_df) < MIN_SAMPLES_PER_SET:
        print(f"⚠️  Warning: Very small dataset. Consider collecting more data or adjusting split ratio.")
        print(f"  Current: train={len(train_df)}, val={len(val_df)} (minimum: {MIN_SAMPLES_PER_SET} each)")
    
    # Check temporal coverage
    train_duration = (train_df.index.max() - train_df.index.min()).total_seconds() / 3600  # hours
    val_duration = (val_df.index.max() - val_df.index.min()).total_seconds() / 3600  # hours
    print(f"  Training period: {train_duration:.1f} hours")
    print(f"  Validation period: {val_duration:.1f} hours")
    
    print("\n=== DATA LOADING SUMMARY ===")
    print(f"✅ Successfully loaded {len(df)} samples from InfluxDB")
    print(f"✅ Split into {len(train_df)} training samples and {len(val_df)} validation samples")
    print(f"✅ All feature engineering completed (deltas, cyclical encoding, lag features)")
    print(f"✅ Temperature difference targets created for 1hr, 2hr, 3hr horizons")
    print("=" * 50)

    # Free memory from full DataFrame now that we've summarized
    import gc
    del df
    gc.collect()

    # Define feature and target columns
    features = [
        'illuminance_delta', 'solar_radiation_delta', 'uv', 'wind_avg', 'wind_gust',
        # Cyclic time-of-day (k=1..3)
        'time_of_day_sin', 'time_of_day_cos',
        'time_of_day_sin2', 'time_of_day_cos2',
        'time_of_day_sin3', 'time_of_day_cos3',
        # Day-of-year harmonics (k=1..4)
        'day_of_year_sin', 'day_of_year_cos',
        'day_of_year_sin2', 'day_of_year_cos2',
        'day_of_year_sin3', 'day_of_year_cos3',
        'day_of_year_sin4', 'day_of_year_cos4',
        # Dynamics
        'temperature_delta', 'pressure_delta', 'humidity_delta',
        # Lags (temperature)
        'temp_lag30', 'temp_lag60', 'temp_lag120', 'temp_lag360', 'temp_lag720', 'temp_lag1440',
        # Lags (humidity)
        'humidity_lag30', 'humidity_lag60', 'humidity_lag120', 'humidity_lag1440',
        # Rolling baseline features
        'temp_mean_24h', 'temp_minus_mean_24h'
    ]
    targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']

    ## Per-feature min/max scaling with ±5% padding
    # Domain bounds for select features
    domain_bounds = {
        "wind_gust": (0, None),
        "wind_avg": (0, None),
        "uv": (0, None),
        # Time-of-day harmonics
        "time_of_day_sin": (-1, 1),
        "time_of_day_cos": (-1, 1),
        "time_of_day_sin2": (-1, 1),
        "time_of_day_cos2": (-1, 1),
        "time_of_day_sin3": (-1, 1),
        "time_of_day_cos3": (-1, 1),
        # Day-of-year harmonics
        "day_of_year_sin": (-1, 1),
        "day_of_year_cos": (-1, 1),
        "day_of_year_sin2": (-1, 1),
        "day_of_year_cos2": (-1, 1),
        "day_of_year_sin3": (-1, 1),
        "day_of_year_cos3": (-1, 1),
        "day_of_year_sin4": (-1, 1),
        "day_of_year_cos4": (-1, 1),
        # Temperature lags
        "temp_lag30": (-10, 55),
        "temp_lag60": (-10, 55),
        "temp_lag120": (-10, 55),
        "temp_lag360": (-10, 55),
        "temp_lag720": (-10, 55),
        "temp_lag1440": (-10, 55),
        # Humidity lags
        "humidity_lag30": (0, 100),
        "humidity_lag60": (0, 100),
        "humidity_lag120": (0, 100),
        "humidity_lag1440": (0, 100),
        # Rolling baseline features
        "temp_mean_24h": (-10, 55),
        # Dynamics (use dynamic bounds)
        "temperature_delta": (None, None),
        "pressure_delta": (None, None),
        "humidity_delta": (None, None),
        "illuminance_delta": (None, None),
        "solar_radiation_delta": (None, None),
        # Deviation can be dynamic
        "temp_minus_mean_24h": (None, None)
    }

    X_train = train_df[features].copy()
    X_val = val_df[features].copy()
    input_scaler = {}
    for feature in features:
        f_min = train_df[feature].min()
        f_max = train_df[feature].max()
        range_pad = 0.05 * (f_max - f_min)

        floor, ceiling = domain_bounds.get(feature, (None, None))
        f_min_adj = floor if floor is not None else f_min - range_pad
        f_max_adj = ceiling if ceiling is not None else f_max + range_pad

        input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
        X_train[feature] = (X_train[feature] - f_min_adj) / (f_max_adj - f_min_adj)
        X_val[feature] = (X_val[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    # Ensure float32 to halve memory vs float64
    X_train = X_train.values.astype(np.float32)
    X_val = X_val.values.astype(np.float32)
    with open("input_scaler_diff.json", "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Normalize target values (temperature differences)
    # Calculate bounds from actual temperature difference data with padding
    y_min = train_df[targets].min().min() - 2  # Add padding for temperature differences
    y_max = train_df[targets].max().max() + 2  # Add padding for temperature differences
    # Save original target range
    target_range = (y_min, y_max)
    train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
    val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

    with open("target_scaler_diff.json", "w") as f:
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


    def build_and_train_model(name):
        print(f"\n--- Running: {name} ---\n")

        y_train = train_df[targets].values.astype(np.float32)
        y_val = val_df[targets].values.astype(np.float32)

        input_layer = tf.keras.layers.Input(shape=(len(features),), name="input")

        wide = tf.keras.layers.Dense(16)(input_layer)
        deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        deep = tf.keras.layers.Dropout(0.3)(deep)

        res = tf.keras.layers.Dense(64, activation='relu')(deep)
        shortcut = tf.keras.layers.Dense(64)(deep)
        res = tf.keras.layers.Add()([shortcut, res])
        res = tf.keras.layers.Dense(32, activation='relu')(res)

        merged = tf.keras.layers.Concatenate()([wide, res])
        output_1 = tf.keras.layers.Dense(1, activation='linear', name='diff_1hr')(merged)
        output_2 = tf.keras.layers.Dense(1, activation='linear', name='diff_2hr')(merged)
        output_3 = tf.keras.layers.Dense(1, activation='linear', name='diff_3hr')(merged)
        model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics={
                'diff_1hr': 'mae',
                'diff_2hr': 'mae',
                'diff_3hr': 'mae'
            },
            jit_compile=False
        )
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/model_{epoch:02d}.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )

        history = model.fit(
            X_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
            validation_data=(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]]),
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping, checkpoint_cb]
        )

        # The early stopping callback with restore_best_weights=True already restored the best weights

        eval_results = model.evaluate(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)
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
        for i, feature in enumerate(features):
            X_val_permuted = copy.deepcopy(X_val)
            np.random.shuffle(X_val_permuted[:, i])
            permuted_loss = model.evaluate(X_val_permuted, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)[0]
            feature_importance[feature] = permuted_loss - baseline_loss

        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nPermutation Feature Importance (by increase in val_loss) [{name}]:")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")

        # The early stopping callback with restore_best_weights=True already restored the best weights

        best_epoch = np.argmin(history.history['val_loss']) + 1
        print(f"\nFinal Metrics [{name}]:")
        print(f"  val_loss: {val_loss:.4f}")
        print(f"  val_mae: {val_mae:.4f}")
        print(f"  Best epoch: {best_epoch}")

        metrics = {
            "name": name,
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "best_epoch": int(best_epoch),
            "feature_importance": [(f, float(i)) for f, i in sorted_importance]
        }

        return model, metrics

    # Configuration
    NUM_RUNS = 5  # Number of training runs to perform
    best_metrics = None
    best_model = None
    all_results = []
    for run_id in range(NUM_RUNS):
        run_name = f"dense_wide_run{run_id+1}"
        model, metrics = build_and_train_model(run_name)
        all_results.append(metrics)
        if best_metrics is None or metrics["val_loss"] < best_metrics["val_loss"]:
            best_metrics = metrics
            best_model = model

    if not best_metrics:
        print("\nNo results found!")
        exit(1)

    print(f"\nBest run: {best_metrics['name']} with val_loss: {best_metrics['val_loss']:.4f} and val_mae: {best_metrics['val_mae']:.4f}")

    # Perform a single TFLite conversion on the best model only
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        step = max(1, len(X_train) // 1000)
        for i in range(0, len(X_train), step):
            yield [X_train[i:i+1].astype(np.float32)]

    quantized_tflite_model = None
    try:
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter._experimental_new_quantizer = False
        print("TFLite convert (best): trying INT8 (old quantizer)...")
        quantized_tflite_model = converter.convert()
    except Exception as e1:
        print(f"INT8 (old quantizer) failed: {e1}")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter._experimental_new_quantizer = True
            print("TFLite convert (best): trying INT8 (new quantizer)...")
            quantized_tflite_model = converter.convert()
        except Exception as e2:
            print(f"INT8 (new quantizer) failed: {e2}")
            converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("TFLite convert (best): falling back to dynamic-range quantization...")
            quantized_tflite_model = converter.convert()

    best_name = best_metrics["name"]
    tflite_fname = f"weather_model_1_diff_quant_{best_name}.tflite"
    with open(tflite_fname, "wb") as f:
        f.write(quantized_tflite_model)

    tflite_model_size_kb = os.path.getsize(tflite_fname) / 1024

    # Save only the best run's results
    metrics_to_save = dict(best_metrics)
    metrics_to_save["model_size_kb"] = float(tflite_model_size_kb)
    with open(f"results_diff_{best_name}.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    # Copy best model to canonical filename
    import shutil
    shutil.copy(tflite_fname, "weather_model_1_diff_best.tflite")
    print(f"Best model copied to: weather_model_1_diff_best.tflite")

    # --- Validate quantized TFLite model on validation data ---
    def validate_quantized_model(tflite_model_path, X_val, y_val, y_min, y_max, num_samples=500):
        import tensorflow as tf
        import numpy as np
        from sklearn.metrics import mean_absolute_error

        print(f"\nValidating TFLite model on {num_samples} samples...")

        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_scale, input_zero_point = input_details[0]['quantization']
        output_scales = [d['quantization'][0] for d in output_details]
        output_zero_points = [d['quantization'][1] for d in output_details]

        print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
        print(f"Output scales: {output_scales}")
        print(f"Output zero points: {output_zero_points}")

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
            mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
            print(f"  {name}: {mae:.2f} °C")

    y_val_array = val_df[targets].values
    validate_quantized_model("weather_model_1_diff_best.tflite", X_val, y_val_array, y_min, y_max)


# Entry point guard
if __name__ == "__main__":
    main()