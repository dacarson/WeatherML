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
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import copy
    import subprocess
    import json
    import glob
    from numba import njit, prange

    # Load preprocessed data
    train_df = pd.read_csv("../train_data.csv")
    val_df = pd.read_csv("../val_data.csv")

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
    train_df['temperature_delta'] = rolling_slope_numba(train_df['temperature'].values, 15)
    val_df['temperature_delta'] = rolling_slope_numba(val_df['temperature'].values, 15)
    train_df['pressure_delta'] = rolling_slope_numba(train_df['station_pressure'].values, 15)
    train_df['humidity_delta'] = rolling_slope_numba(train_df['relative_humidity'].values, 15)
    val_df['pressure_delta'] = rolling_slope_numba(val_df['station_pressure'].values, 15)
    val_df['humidity_delta'] = rolling_slope_numba(val_df['relative_humidity'].values, 15)
    train_df['illuminance_delta'] = rolling_slope_numba(train_df['illuminance'].values, 15)
    train_df['solar_radiation_delta'] = rolling_slope_numba(train_df['solar_radiation'].values, 15)
    val_df['illuminance_delta'] = rolling_slope_numba(val_df['illuminance'].values, 15)
    val_df['solar_radiation_delta'] = rolling_slope_numba(val_df['solar_radiation'].values, 15)

    # Add features to help model understand current solar radiation state without giving it the absolute value
    # This helps it learn patterns rather than just snapping to current value
    
    # Compute expected solar radiation from training data (per time-of-day and season)
    # This gives us a baseline expectation without giving the model the absolute current value
    # Group by time-of-day bins (48 bins = 30-minute intervals) and season bins (12 bins = monthly)
    train_df['time_bin'] = pd.cut(train_df['time_of_day'], bins=48, labels=False)
    train_df['season_bin'] = pd.cut(train_df['day_of_year'], bins=12, labels=False)
    
    # Compute mean solar radiation for each time/season combination (expected value)
    expected_solar_by_time_season = train_df.groupby(['time_bin', 'season_bin'])['solar_radiation'].mean()
    
    # Map expected values back to dataframe
    train_df['expected_solar_radiation'] = train_df.set_index(['time_bin', 'season_bin']).index.map(expected_solar_by_time_season)
    train_df['expected_solar_radiation'] = train_df['expected_solar_radiation'].fillna(train_df['solar_radiation'].mean())
    
    # Deviation from expected - tells model if above/below expected WITHOUT giving absolute current value
    # This prevents model from "snapping" to current value while still providing context
    train_df['solar_radiation_deviation'] = train_df['solar_radiation'] - train_df['expected_solar_radiation']

    # Clear-sky ratio: actual irradiance relative to expected clear-sky value (with safe denominator)
    eps = 1e-6
    train_expected_safe = np.maximum(train_df['expected_solar_radiation'], 50.0)
    train_df['solar_clear_sky_ratio'] = train_df['solar_radiation'] / (train_expected_safe + eps)
    train_df['solar_clear_sky_ratio'] = train_df['solar_clear_sky_ratio'].clip(lower=0.0, upper=5.0)
    # Clear-sky deficit (only positive when we're below expected)
    train_df['clear_sky_deficit'] = np.maximum(train_df['expected_solar_radiation'] - train_df['solar_radiation'], 0.0)
    train_df['clear_sky_deficit'] = train_df['clear_sky_deficit'].clip(upper=800.0)
    
    # Do the same for validation set (use training expected values)
    val_df['time_bin'] = pd.cut(val_df['time_of_day'], bins=48, labels=False)
    val_df['season_bin'] = pd.cut(val_df['day_of_year'], bins=12, labels=False)
    val_df['expected_solar_radiation'] = val_df.set_index(['time_bin', 'season_bin']).index.map(expected_solar_by_time_season)
    val_df['expected_solar_radiation'] = val_df['expected_solar_radiation'].fillna(train_df['solar_radiation'].mean())
    val_df['solar_radiation_deviation'] = val_df['solar_radiation'] - val_df['expected_solar_radiation']
    val_expected_safe = np.maximum(val_df['expected_solar_radiation'], 50.0)
    val_df['solar_clear_sky_ratio'] = val_df['solar_radiation'] / (val_expected_safe + eps)
    val_df['solar_clear_sky_ratio'] = val_df['solar_clear_sky_ratio'].clip(lower=0.0, upper=5.0)
    val_df['clear_sky_deficit'] = np.maximum(val_df['expected_solar_radiation'] - val_df['solar_radiation'], 0.0)
    val_df['clear_sky_deficit'] = val_df['clear_sky_deficit'].clip(upper=800.0)
    
    # Rolling statistics to capture local trends and volatility (no future leakage)
    def add_rolling_stats(df):
        df['solar_radiation_mean_30min'] = df['solar_radiation'].rolling(window=30, min_periods=1).mean()
        df['solar_radiation_std_30min'] = df['solar_radiation'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=200.0)
        df['solar_radiation_mean_60min'] = df['solar_radiation'].rolling(window=60, min_periods=1).mean()

        df['uv_mean_30min'] = df['uv'].rolling(window=30, min_periods=1).mean()
        df['uv_std_30min'] = df['uv'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=5.0)

        df['humidity_mean_30min'] = df['relative_humidity'].rolling(window=30, min_periods=1).mean()
        df['humidity_std_30min'] = df['relative_humidity'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=20.0)

        df['temperature_mean_30min'] = df['temperature'].rolling(window=30, min_periods=1).mean()
        df['temperature_std_30min'] = df['temperature'].rolling(window=30, min_periods=1).std().fillna(0.0).clip(lower=0.0, upper=10.0)

        # Clear-sky ratio trend vs its recent average
        df['solar_clear_sky_ratio_mean_30min'] = df['solar_clear_sky_ratio'].rolling(window=30, min_periods=1).mean().clip(lower=0.0, upper=5.0)
        df['sky_clarity_trend_30min'] = (df['solar_clear_sky_ratio'] - df['solar_clear_sky_ratio_mean_30min']).clip(lower=-3.0, upper=3.0)

    add_rolling_stats(train_df)
    add_rolling_stats(val_df)

    # Solar radiation to illuminance/fog context ratios
    train_df['solar_illuminance_ratio'] = train_df['solar_radiation'] / (train_df['illuminance'] + 1e-6)
    val_df['solar_illuminance_ratio'] = val_df['solar_radiation'] / (val_df['illuminance'] + 1e-6)
    
    # Rolling variance in solar radiation - low variance indicates stable/clear conditions
    @njit(parallel=True)
    def rolling_variance_numba(data, window):
        n = len(data)
        variances = np.full(n, np.nan)
        for i in prange(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            if np.any(np.isnan(window_data)):
                continue
            variances[i] = np.var(window_data)
        return variances
    
    train_df['solar_radiation_variance_30min'] = rolling_variance_numba(train_df['solar_radiation'].values, 30)
    val_df['solar_radiation_variance_30min'] = rolling_variance_numba(val_df['solar_radiation'].values, 30)
    
    # Solar radiation delta (change rate) - already computed, but use it as a feature
    # This tells the model the current trend without giving absolute value
    # Actually, we removed this earlier - let's not add it back since we have deviation now
    
    # Solar radiation lag differences - how much has it changed recently (gives trend context)
    train_df['solar_radiation_change_30min'] = train_df['solar_radiation'] - train_df['solar_radiation'].shift(30)
    train_df['solar_radiation_change_60min'] = train_df['solar_radiation'] - train_df['solar_radiation'].shift(60)
    val_df['solar_radiation_change_30min'] = val_df['solar_radiation'] - val_df['solar_radiation'].shift(30)
    val_df['solar_radiation_change_60min'] = val_df['solar_radiation'] - val_df['solar_radiation'].shift(60)

    # Temperature and humidity change over 30 minutes (explicit difference)
    train_df['temperature_change_30min'] = train_df['temperature'] - train_df['temp_lag30']
    val_df['temperature_change_30min'] = val_df['temperature'] - val_df['temp_lag30']
    train_df['humidity_change_30min'] = train_df['relative_humidity'] - train_df['humidity_lag30']
    val_df['humidity_change_30min'] = val_df['relative_humidity'] - val_df['humidity_lag30']

    # Marine-layer heuristics & fog indicators using only local sensors
    def compute_marine_features(df):
        if 'wind_direction' in df.columns:
            direction_rad = np.deg2rad(df['wind_direction'].values)
        elif 'wind_direction_sin' in df.columns and 'wind_direction_cos' in df.columns:
            direction_rad = np.arctan2(df['wind_direction_sin'].values, df['wind_direction_cos'].values)
            direction_rad = (direction_rad + 2 * np.pi) % (2 * np.pi)
        else:
            direction_rad = None

        if direction_rad is not None:
            # Onshore (westerly) component peaks near 270°
            onshore_component = np.cos(direction_rad - (3 * np.pi / 2))
            onshore_component = np.clip(onshore_component, -1.0, 1.0)
        else:
            onshore_component = np.zeros(len(df))

        marine_push = np.clip(onshore_component, 0.0, 1.0) * df['wind_avg'].values
        humidity_norm = np.clip((df['relative_humidity'].values - 80.0) / 20.0, 0.0, 1.0)
        # Higher marine push with high humidity → stronger marine intrusion signal
        df['marine_push_score'] = marine_push * (0.5 + 0.5 * humidity_norm)
        df['marine_push_flag'] = ((np.clip(onshore_component, 0.0, 1.0) > 0.5) & (df['wind_avg'].values > 2.0) & (df['relative_humidity'].values > 85.0)).astype(np.float32)

        # Fog likelihood blends humidity, clear-sky deficit, and negative solar trends
        expected_safe_vals = np.maximum(df['expected_solar_radiation'].values, 50.0)
        deficit_norm = np.clip(df['clear_sky_deficit'].values / (expected_safe_vals + eps), 0.0, 1.0)
        ratio_deficit = np.clip((0.7 - df['solar_clear_sky_ratio'].values) / 0.7, 0.0, 1.0)
        dimming = np.clip(-df['solar_radiation_change_30min'].fillna(0.0).values / 50.0, 0.0, 1.0)
        fog_likelihood = 0.3 * humidity_norm + 0.35 * ratio_deficit + 0.25 * deficit_norm + 0.10 * dimming
        df['fog_likelihood'] = np.clip(fog_likelihood, 0.0, 1.0)
        df['fog_indicator'] = (df['fog_likelihood'] >= 0.6).astype(np.float32)

    compute_marine_features(train_df)
    compute_marine_features(val_df)

    # Calculate solar radiation differences as targets
    # Predict the change in solar radiation from current value (similar to Model 5 temperature differences)
    # Use negative shift to get future values (shift(-30) gets value 30 minutes ahead)
    train_df['solar_radiation_t+30min'] = train_df['solar_radiation'].shift(-30)
    train_df['solar_radiation_t+60min'] = train_df['solar_radiation'].shift(-60)
    train_df['solar_radiation_t+90min'] = train_df['solar_radiation'].shift(-90)
    
    train_df['solar_radiation_diff_30min'] = train_df['solar_radiation_t+30min'] - train_df['solar_radiation']
    train_df['solar_radiation_diff_60min'] = train_df['solar_radiation_t+60min'] - train_df['solar_radiation']
    train_df['solar_radiation_diff_90min'] = train_df['solar_radiation_t+90min'] - train_df['solar_radiation']

    val_df['solar_radiation_t+30min'] = val_df['solar_radiation'].shift(-30)
    val_df['solar_radiation_t+60min'] = val_df['solar_radiation'].shift(-60)
    val_df['solar_radiation_t+90min'] = val_df['solar_radiation'].shift(-90)
    
    val_df['solar_radiation_diff_30min'] = val_df['solar_radiation_t+30min'] - val_df['solar_radiation']
    val_df['solar_radiation_diff_60min'] = val_df['solar_radiation_t+60min'] - val_df['solar_radiation']
    val_df['solar_radiation_diff_90min'] = val_df['solar_radiation_t+90min'] - val_df['solar_radiation']

    # Drop temporary columns used for expected solar radiation calculation
    train_df.drop(columns=['time_bin', 'season_bin', 'expected_solar_radiation'], inplace=True, errors='ignore')
    val_df.drop(columns=['time_bin', 'season_bin', 'expected_solar_radiation'], inplace=True, errors='ignore')
    
    # Drop rows with NaNs
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    # Define feature and target columns
    # Note: Removed solar_radiation_delta from features (replaced with temperature_delta as requested)
    # Note: Removed explicit seasonal interaction features - the feature interaction layer will learn these automatically
    # Using deviation from expected instead of absolute current value to avoid model "snapping" to current value
    features = [
        'illuminance_delta', 'uv', 'wind_avg', 'wind_gust', 
        'day_of_year_sin', 'day_of_year_cos',
        'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
        'temperature_delta', 'pressure_delta', 'humidity_delta', 
        'temp_lag30', 'humidity_lag30', 'temp_lag60', 'humidity_lag60', 'temp_lag120', 'humidity_lag120',
        'wind_avg_lag30', 'wind_gust_lag30', 'uv_lag30', 'pressure_lag30',
        # Context features - help model understand state without giving absolute current value
        # Deviation from expected tells model if above/below expected for time/season
        'solar_radiation_deviation', 'solar_clear_sky_ratio', 'clear_sky_deficit',
        'solar_radiation_variance_30min', 'solar_radiation_change_30min', 'solar_radiation_change_60min',
        'solar_radiation_mean_30min', 'solar_radiation_std_30min', 'solar_radiation_mean_60min',
        'solar_clear_sky_ratio_mean_30min', 'sky_clarity_trend_30min',
        'uv_mean_30min', 'uv_std_30min',
        'humidity_mean_30min', 'humidity_std_30min',
        'temperature_mean_30min', 'temperature_std_30min',
        'temperature_change_30min', 'humidity_change_30min',
        'solar_illuminance_ratio', 'fog_likelihood', 'fog_indicator',
        'marine_push_score', 'marine_push_flag'
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

    targets = ['solar_radiation_diff_30min', 'solar_radiation_diff_60min', 'solar_radiation_diff_90min']

    ## Per-feature min/max scaling with ±5% padding
    # Domain bounds for select features
    domain_bounds = {
        "wind_gust": (0, None),
        "wind_avg": (0, None),
        "uv": (0, None),
        "humidity_lag30": (0, 100),
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
        "temp_lag60": (-10, 55),            # 1-hour lag temperature
        "temp_lag120": (-10, 55),           # 2-hour lag temperature
        "humidity_lag60": (0, 100),         # 1-hour lag humidity
        "humidity_lag120": (0, 100),        # 2-hour lag humidity
        "wind_avg_lag30": (0, None),        # 30-min lag wind average
        "wind_gust_lag30": (0, None),       # 30-min lag wind gust
        "uv_lag30": (0, None),              # 30-min lag UV
        "pressure_lag30": (None, None),     # 30-min lag pressure
        "temperature_delta": (None, None),  # Allow dynamic bounds for expanded delta range
        "pressure_delta": (None, None),     # Allow dynamic bounds for pressure delta range
        "humidity_delta": (None, None),     # Allow dynamic bounds for humidity delta range
        "illuminance_delta": (None, None),  # Allow dynamic bounds for illuminance delta range
        # Context features - help model understand state without snapping to current value
        "solar_radiation_deviation": (None, None),  # Deviation from expected (can be negative or positive)
        "solar_clear_sky_ratio": (0, 5),     # Ratio clipped to [0, 5]
        "clear_sky_deficit": (0, 800),       # Deficit clipped to [0, 800]
        "solar_illuminance_ratio": (0, None),  # Ratio helps detect clear vs hazy conditions
        "solar_radiation_variance_30min": (0, None),  # Variance indicates stability (low = clear, high = variable)
        "solar_radiation_change_30min": (None, None),  # How much solar radiation changed in last 30 min
        "solar_radiation_change_60min": (None, None),   # How much solar radiation changed in last 60 min
        "solar_radiation_mean_30min": (0, None),       # Rolling mean cannot be negative
        "solar_radiation_std_30min": (0, 200),         # Std clipped in feature generation
        "solar_radiation_mean_60min": (0, None),
        "solar_clear_sky_ratio_mean_30min": (0, 5),
        "sky_clarity_trend_30min": (-3, 3),
        "uv_mean_30min": (0, None),
        "uv_std_30min": (0, 5),
        "humidity_mean_30min": (0, None),
        "humidity_std_30min": (0, 20),
        "temperature_mean_30min": (None, None),
        "temperature_std_30min": (0, 10),
        "temperature_change_30min": (None, None),
        "humidity_change_30min": (None, None),
        "fog_likelihood": (0, 1),
        "fog_indicator": (0, 1),
        "marine_push_score": (0, None),
        "marine_push_flag": (0, 1)
        # Note: Using deviation from expected instead of absolute current value to avoid model "snapping"
        # Note: solar_radiation_delta removed from features (replaced with temperature_delta)
        # Note: Explicit seasonal interaction features removed - feature interaction layer learns these automatically
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
    X_train = X_train.values
    X_val = X_val.values
    with open("input_scaler_6.json", "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Save original unscaled targets for baseline and statistics calculation
    val_targets_orig = val_df[targets].copy().values
    
    # Normalize target values (solar radiation differences)
    # Calculate bounds from actual solar radiation difference data with padding
    y_min = train_df[targets].min().min() - 10  # Add padding for solar radiation differences
    y_max = train_df[targets].max().max() + 10  # Add padding for solar radiation differences
    # Save original target range
    target_range = (y_min, y_max)
    train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
    val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

    with open("target_scaler_6.json", "w") as f:
        json.dump({"min": y_min, "max": y_max, "range": target_range}, f)

    # Print min/max values for all features
    print("\n=== FEATURE MIN/MAX VALUES ===")
    print("Input Features (Original Scale):")
    for feature in features:
        f_min = train_df[feature].min()
        f_max = train_df[feature].max()
        print(f"  {feature}: min={f_min:.4f}, max={f_max:.4f}")

    print("\nTarget Features (Solar Radiation Differences):")
    for target in targets:
        t_min = train_df[target].min()
        t_max = train_df[target].max()
        t_mean = train_df[target].mean()
        t_std = train_df[target].std()
        t_range = t_max - t_min
        print(f"  {target}: min={t_min:.2f}, max={t_max:.2f}, mean={t_mean:.2f}, std={t_std:.2f}, range={t_range:.2f}")

    print("\n=== SCALING BOUNDS ===")
    print("Solar radiation difference scaling bounds:")
    print(f"Solar radiation difference range: {y_min:.2f} to {y_max:.2f}")
    print(f"Original solar radiation difference stats:")
    print(f"  30min: min={train_df['solar_radiation_diff_30min'].min():.2f}, max={train_df['solar_radiation_diff_30min'].max():.2f}")
    print(f"  60min: min={train_df['solar_radiation_diff_60min'].min():.2f}, max={train_df['solar_radiation_diff_60min'].max():.2f}")
    print(f"  90min: min={train_df['solar_radiation_diff_90min'].min():.2f}, max={train_df['solar_radiation_diff_90min'].max():.2f}")

    print("\nInput Feature Scaling Parameters:")
    for feature in features:
        scaler_params = input_scaler[feature]
        print(f"  {feature}: min={scaler_params['min']:.4f}, max={scaler_params['max']:.4f}")
    print("=" * 50)


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
        projected = tf.keras.layers.Dense(embedding_dim, activation='relu', name='interaction_embed')(inputs)
        
        # Element-wise square to capture non-linear/pairwise interactions
        # This is equivalent to capturing x_i * x_j terms
        # Using Multiply layer (more explicit than Lambda for Edge TPU compatibility)
        squared = tf.keras.layers.Multiply(name='interaction_square')([projected, projected])
        
        # Combine original and squared features to capture both linear and interaction terms
        combined = tf.keras.layers.Concatenate(name='interaction_combine')([projected, squared])
        
        # Final projection: learns which interactions are important
        interactions = tf.keras.layers.Dense(projection_dim, activation='relu', name='interaction_output')(combined)
        
        return interactions

    def build_and_train_model(name):
        print(f"\n--- Running: {name} ---\n")

        y_train = train_df[targets].values
        y_val = val_df[targets].values

        input_layer = tf.keras.layers.Input(shape=(len(features),), name="input")

        # Learn feature interactions FIRST - this enriches the input features
        # The interaction layer learns how features interact (especially seasonal interactions)
        interaction_projection = build_feature_interaction_path(input_layer, embedding_dim=16, projection_dim=32)
        
        # Concatenate original features with learned interactions
        # This gives the deep network both raw features and learned interactions to work with
        enriched_features = tf.keras.layers.Concatenate(name='enriched_features')([input_layer, interaction_projection])
        
        # Wide path: simple linear combinations of enriched features
        wide = tf.keras.layers.Dense(16, name='wide')(enriched_features)
        
        # Deep path: non-linear transformations of enriched features
        deep = tf.keras.layers.Dense(128, activation='relu', name='deep_1')(enriched_features)
        deep = tf.keras.layers.Dropout(0.3, name='dropout')(deep)

        # Residual block for better gradient flow
        res = tf.keras.layers.Dense(64, activation='relu', name='res_1')(deep)
        shortcut = tf.keras.layers.Dense(64, name='res_shortcut')(deep)
        res = tf.keras.layers.Add(name='res_add')([shortcut, res])
        res = tf.keras.layers.Dense(32, activation='relu', name='res_2')(res)

        # Merge wide and deep paths
        merged = tf.keras.layers.Concatenate(name='merged')([wide, res])

        # Output layers
        output_1 = tf.keras.layers.Dense(1, activation='linear', name='solar_diff_30min')(merged)
        output_2 = tf.keras.layers.Dense(1, activation='linear', name='solar_diff_60min')(merged)
        output_3 = tf.keras.layers.Dense(1, activation='linear', name='solar_diff_90min')(merged)
        model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

        # Optimized optimizer for Mac M1
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics={
                'solar_diff_30min': 'mae',
                'solar_diff_60min': 'mae',
                'solar_diff_90min': 'mae'
            },
            jit_compile=False  # Conservative: disable JIT initially
        )
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/model_{epoch:02d}.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )

        # Match RPi5 configuration exactly
        history = model.fit(
            X_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
            validation_data=(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]]),
            epochs=100,
            batch_size=256,  # Match RPi5 batch size
            callbacks=[early_stopping, checkpoint_cb]
        )

        # The early stopping callback with restore_best_weights=True already restored the best weights

        eval_results = model.evaluate(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)
        val_loss = eval_results[0]
        val_mae = np.mean(eval_results[1:])  # average MAE across 3 targets
        # Report MAEs in original units (solar radiation difference)
        diff_30_mae = eval_results[1] * (y_max - y_min) / 2  # Convert from [-1,1] to original scale
        diff_60_mae = eval_results[2] * (y_max - y_min) / 2
        diff_90_mae = eval_results[3] * (y_max - y_min) / 2
        
        # Calculate baseline (predicting zero difference) using ORIGINAL unscaled targets
        baseline_30_mae = np.mean(np.abs(val_targets_orig[:, 0] - 0))
        baseline_60_mae = np.mean(np.abs(val_targets_orig[:, 1] - 0))
        baseline_90_mae = np.mean(np.abs(val_targets_orig[:, 2] - 0))
        
        # Calculate target statistics for context using ORIGINAL unscaled targets
        target_30_range = val_targets_orig[:, 0].max() - val_targets_orig[:, 0].min()
        target_60_range = val_targets_orig[:, 1].max() - val_targets_orig[:, 1].min()
        target_90_range = val_targets_orig[:, 2].max() - val_targets_orig[:, 2].min()
        target_30_std = val_targets_orig[:, 0].std()
        target_60_std = val_targets_orig[:, 1].std()
        target_90_std = val_targets_orig[:, 2].std()
        
        print(f"\nValidation MAE (in solar radiation difference):")
        print(f"  30min: {diff_30_mae:.2f} (baseline: {baseline_30_mae:.2f}, improvement: {((baseline_30_mae - diff_30_mae) / baseline_30_mae * 100):.1f}%)")
        print(f"  60min: {diff_60_mae:.2f} (baseline: {baseline_60_mae:.2f}, improvement: {((baseline_60_mae - diff_60_mae) / baseline_60_mae * 100):.1f}%)")
        print(f"  90min: {diff_90_mae:.2f} (baseline: {baseline_90_mae:.2f}, improvement: {((baseline_90_mae - diff_90_mae) / baseline_90_mae * 100):.1f}%)")
        print(f"\nTarget Statistics (validation set):")
        print(f"  30min: std={target_30_std:.2f}, range={target_30_range:.2f}, MAE as % of std: {(diff_30_mae/target_30_std*100):.1f}%")
        print(f"  60min: std={target_60_std:.2f}, range={target_60_range:.2f}, MAE as % of std: {(diff_60_mae/target_60_std*100):.1f}%")
        print(f"  90min: std={target_90_std:.2f}, range={target_90_range:.2f}, MAE as % of std: {(diff_90_mae/target_90_std*100):.1f}%")
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

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_data_gen():
            # Sample evenly across the entire year of data for better representation
            # Take every 500th sample to get ~1000 samples spread across the year
            step = max(1, len(X_train) // 1000)
            for i in range(0, len(X_train), step):
                if len(X_train) - i >= 1:  # Ensure we have at least 1 sample
                    yield [X_train[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter._experimental_disable_per_channel = False
        converter._experimental_new_quantizer = True  # Use MLIR quantizer (recommended)

        quantized_tflite_model = converter.convert()

        tflite_fname = f"weather_model_6_quant_{name}.tflite"
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

        with open(f"results_6_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Configuration
    NUM_RUNS = 5  # Number of training runs to perform
    for run_id in range(NUM_RUNS):
        run_name = f"dense_wide_run{run_id+1}"
        build_and_train_model(run_name)

    results = []
    # Only collect results from the current run session
    for run_id in range(NUM_RUNS):
        json_file = f"results_6_dense_wide_run{run_id+1}.json"
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
    best_model_file = f"weather_model_6_quant_{best['name']}.tflite"
    shutil.copy(best_model_file, "weather_model_6_best.tflite")
    print(f"Best model copied to: weather_model_6_best.tflite")
    
    # Validate the best quantized model
    print("\n" + "="*50)
    print("VALIDATING BEST QUANTIZED MODEL")
    print("="*50)
    y_val_array = val_df[targets].values
    validate_quantized_model("weather_model_6_best.tflite", X_val, y_val_array, y_min, y_max)

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
    for j, name in enumerate(['solar_diff_30min', 'solar_diff_60min', 'solar_diff_90min']):
        print(f"  {name}: {np.array(y_preds_dequant[j][:5]).round(2)}")

    print("\nValidation MAE (in solar radiation difference):")
    for j, name in enumerate(['solar_diff_30min', 'solar_diff_60min', 'solar_diff_90min']):
        y_preds_rescaled = 0.5 * (np.array(y_preds_dequant[j]) + 1) * (y_max - y_min) + y_min
        y_val_rescaled = 0.5 * (y_val_subset[:, j] + 1) * (y_max - y_min) + y_min
        mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
        
        # Calculate baseline and statistics
        baseline_mae = np.mean(np.abs(y_val_rescaled - 0))
        target_std = np.std(y_val_rescaled)
        target_range = np.max(y_val_rescaled) - np.min(y_val_rescaled)
        
        print(f"  {name}: {mae:.2f} (baseline: {baseline_mae:.2f}, std: {target_std:.2f}, range: {target_range:.2f})")
        print(f"    MAE as % of std: {(mae/target_std*100):.1f}%, improvement over baseline: {((baseline_mae - mae) / baseline_mae * 100):.1f}%")



# Entry point guard
if __name__ == "__main__":
    main()
