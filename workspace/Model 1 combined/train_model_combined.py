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
    train_df_sf = pd.read_csv("../train_data.csv")
    val_df_sf = pd.read_csv("../val_data.csv")

    train_df_ps = pd.read_csv("../train_data_ps.csv")
    val_df_ps = pd.read_csv("../val_data_ps.csv")
    
    # Add coastal climate flag (1 for San Francisco/coastal, 0 for Palm Springs/desert)
    train_df_sf['is_coastal'] = 1
    val_df_sf['is_coastal'] = 1
    train_df_ps['is_coastal'] = 0
    val_df_ps['is_coastal'] = 0
    
    # Handle missing columns - Palm Springs has wet_bulb_temperature that SF doesn't
    # Add wet_bulb_temperature column to SF data with NaN values
    if 'wet_bulb_temperature' not in train_df_sf.columns:
        train_df_sf['wet_bulb_temperature'] = np.nan
        val_df_sf['wet_bulb_temperature'] = np.nan
    
    # Combine datasets
    train_df = pd.concat([train_df_sf, train_df_ps], ignore_index=True)
    val_df = pd.concat([val_df_sf, val_df_ps], ignore_index=True)
    
    print(f"Combined training data: {len(train_df)} samples")
    print(f"  - San Francisco (coastal): {len(train_df_sf)} samples")
    print(f"  - Palm Springs (desert): {len(train_df_ps)} samples")
    print(f"Combined validation data: {len(val_df)} samples")
    print(f"  - San Francisco (coastal): {len(val_df_sf)} samples")
    print(f"  - Palm Springs (desert): {len(val_df_ps)} samples")
    
    # Print climate statistics to understand the differences
    print("\n=== CLIMATE STATISTICS ===")
    print("San Francisco (Coastal) - Training Data:")
    print(f"  Temperature: {train_df_sf['temperature'].min():.1f}°C to {train_df_sf['temperature'].max():.1f}°C (mean: {train_df_sf['temperature'].mean():.1f}°C)")
    print(f"  Humidity: {train_df_sf['relative_humidity'].min():.1f}% to {train_df_sf['relative_humidity'].max():.1f}% (mean: {train_df_sf['relative_humidity'].mean():.1f}%)")
    print(f"  Solar Radiation: {train_df_sf['solar_radiation'].min():.1f} to {train_df_sf['solar_radiation'].max():.1f} (mean: {train_df_sf['solar_radiation'].mean():.1f})")
    
    print("\nPalm Springs (Desert) - Training Data:")
    print(f"  Temperature: {train_df_ps['temperature'].min():.1f}°C to {train_df_ps['temperature'].max():.1f}°C (mean: {train_df_ps['temperature'].mean():.1f}°C)")
    print(f"  Humidity: {train_df_ps['relative_humidity'].min():.1f}% to {train_df_ps['relative_humidity'].max():.1f}% (mean: {train_df_ps['relative_humidity'].mean():.1f}%)")
    print(f"  Solar Radiation: {train_df_ps['solar_radiation'].min():.1f} to {train_df_ps['solar_radiation'].max():.1f} (mean: {train_df_ps['solar_radiation'].mean():.1f})")
    
    # Calculate daily temperature ranges to highlight climate differences
    sf_daily_range = train_df_sf.groupby(train_df_sf.index // 24)['temperature'].apply(lambda x: x.max() - x.min()).mean()
    ps_daily_range = train_df_ps.groupby(train_df_ps.index // 24)['temperature'].apply(lambda x: x.max() - x.min()).mean()
    
    print(f"\nDaily Temperature Range Analysis:")
    print(f"  San Francisco (Coastal): {sf_daily_range:.1f}°C average daily range")
    print(f"  Palm Springs (Desert): {ps_daily_range:.1f}°C average daily range")
    print(f"  Difference: {ps_daily_range - sf_daily_range:.1f}°C (Desert has {ps_daily_range/sf_daily_range:.1f}x larger daily range)")
    
    # Calculate humidity stability
    sf_humidity_std = train_df_sf['relative_humidity'].std()
    ps_humidity_std = train_df_ps['relative_humidity'].std()
    
    print(f"\nHumidity Stability Analysis:")
    print(f"  San Francisco (Coastal): {sf_humidity_std:.1f}% humidity std dev")
    print(f"  Palm Springs (Desert): {ps_humidity_std:.1f}% humidity std dev")
    print(f"  Difference: {ps_humidity_std - sf_humidity_std:.1f}% (Desert has {ps_humidity_std/sf_humidity_std:.1f}x more variable humidity)")
    print("=" * 50)

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

    # Simplified climate-specific features to avoid NaN issues
    # Use safer feature engineering that won't create extreme values
    
    # Climate-specific temperature features (safer approach)
    train_df['coastal_temp'] = train_df['is_coastal'] * train_df['temp_lag30']
    train_df['desert_temp'] = (1 - train_df['is_coastal']) * train_df['temp_lag30']
    val_df['coastal_temp'] = val_df['is_coastal'] * val_df['temp_lag30']
    val_df['desert_temp'] = (1 - val_df['is_coastal']) * val_df['temp_lag30']
    
    # Climate-specific humidity features
    train_df['coastal_humidity'] = train_df['is_coastal'] * train_df['humidity_lag30']
    train_df['desert_humidity'] = (1 - train_df['is_coastal']) * train_df['humidity_lag30']
    val_df['coastal_humidity'] = val_df['is_coastal'] * val_df['humidity_lag30']
    val_df['desert_humidity'] = (1 - val_df['is_coastal']) * val_df['humidity_lag30']
    
    # Climate-specific wind features
    train_df['coastal_wind'] = train_df['is_coastal'] * train_df['wind_avg']
    train_df['desert_wind'] = (1 - train_df['is_coastal']) * train_df['wind_avg']
    val_df['coastal_wind'] = val_df['is_coastal'] * val_df['wind_avg']
    val_df['desert_wind'] = (1 - val_df['is_coastal']) * val_df['wind_avg']
    
    # Temperature range features - coastal climates have smaller daily ranges
    train_df['daily_temp_range'] = train_df['temp_lag30'] - train_df['temp_lag120']  # 2-hour range
    train_df['coastal_temp_range'] = train_df['is_coastal'] * train_df['daily_temp_range']
    train_df['desert_temp_range'] = (1 - train_df['is_coastal']) * train_df['daily_temp_range']
    val_df['daily_temp_range'] = val_df['temp_lag30'] - val_df['temp_lag120']
    val_df['coastal_temp_range'] = val_df['is_coastal'] * val_df['daily_temp_range']
    val_df['desert_temp_range'] = (1 - val_df['is_coastal']) * val_df['daily_temp_range']
    
    # Humidity stability features - coastal climates have more stable humidity
    train_df['humidity_stability'] = np.abs(train_df['humidity_lag30'] - train_df['humidity_lag60'])
    train_df['coastal_humidity_stability'] = train_df['is_coastal'] * train_df['humidity_stability']
    train_df['desert_humidity_stability'] = (1 - train_df['is_coastal']) * train_df['humidity_stability']
    val_df['humidity_stability'] = np.abs(val_df['humidity_lag30'] - val_df['humidity_lag60'])
    val_df['coastal_humidity_stability'] = val_df['is_coastal'] * val_df['humidity_stability']
    val_df['desert_humidity_stability'] = (1 - val_df['is_coastal']) * val_df['humidity_stability']
    
    # Keep some of the original interaction features but remove the zero-importance ones
    train_df['time_sin_uv'] = train_df['time_of_day_sin'] * train_df['uv']
    train_df['time_cos_uv'] = train_df['time_of_day_cos'] * train_df['uv']
    train_df['time_sin_temp_lag'] = train_df['time_of_day_sin'] * train_df['temp_lag30']
    train_df['time_cos_temp_lag'] = train_df['time_of_day_cos'] * train_df['temp_lag30']
    val_df['time_sin_uv'] = val_df['time_of_day_sin'] * val_df['uv']
    val_df['time_cos_uv'] = val_df['time_of_day_cos'] * val_df['uv']
    val_df['time_sin_temp_lag'] = val_df['time_of_day_sin'] * val_df['temp_lag30']
    val_df['time_cos_temp_lag'] = val_df['time_of_day_cos'] * val_df['temp_lag30']

    # Calculate temperature differences as targets
    # Instead of predicting absolute temperatures, predict the change from current temperature
    train_df['temp_diff_1hr'] = train_df['temp_t+1hr'] - train_df['temperature']
    train_df['temp_diff_2hr'] = train_df['temp_t+2hr'] - train_df['temperature']
    train_df['temp_diff_3hr'] = train_df['temp_t+3hr'] - train_df['temperature']

    val_df['temp_diff_1hr'] = val_df['temp_t+1hr'] - val_df['temperature']
    val_df['temp_diff_2hr'] = val_df['temp_t+2hr'] - val_df['temperature']
    val_df['temp_diff_3hr'] = val_df['temp_t+3hr'] - val_df['temperature']

    # Check for NaN values before dropping
    print("\n=== CHECKING FOR NaN VALUES ===")
    nan_features_train = train_df.isnull().sum()
    nan_features_val = val_df.isnull().sum()
    
    print("Training data NaN counts:")
    for feature, count in nan_features_train.items():
        if count > 0:
            print(f"  {feature}: {count} NaN values")
    
    print("\nValidation data NaN counts:")
    for feature, count in nan_features_val.items():
        if count > 0:
            print(f"  {feature}: {count} NaN values")
    
    # Check for infinite values
    print("\nChecking for infinite values in training data:")
    inf_features_train = np.isinf(train_df.select_dtypes(include=[np.number])).sum()
    for feature, count in inf_features_train.items():
        if count > 0:
            print(f"  {feature}: {count} infinite values")
    
    print("\nChecking for infinite values in validation data:")
    inf_features_val = np.isinf(val_df.select_dtypes(include=[np.number])).sum()
    for feature, count in inf_features_val.items():
        if count > 0:
            print(f"  {feature}: {count} infinite values")
    
    # Check data distribution before dropping NaNs
    print("\n=== DATA DISTRIBUTION BEFORE DROPPING NaNs ===")
    print(f"Training data - SF samples: {len(train_df_sf)}, PS samples: {len(train_df_ps)}")
    print(f"Validation data - SF samples: {len(val_df_sf)}, PS samples: {len(val_df_ps)}")
    
    # Check coastal flag distribution before dropping NaNs
    print(f"\nCoastal flag distribution in training data:")
    print(f"  SF (should be 1): {train_df['is_coastal'].sum()} samples")
    print(f"  PS (should be 0): {(train_df['is_coastal'] == 0).sum()} samples")
    
    # Drop rows with NaNs but preserve both datasets
    print(f"\nDropping NaN values...")
    train_df_before = len(train_df)
    val_df_before = len(val_df)
    
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)
    
    print(f"Training data: {train_df_before} → {len(train_df)} samples")
    print(f"Validation data: {val_df_before} → {len(val_df)} samples")
    
    # Check coastal flag distribution after dropping NaNs
    print(f"\nCoastal flag distribution after dropping NaNs:")
    print(f"  SF (should be 1): {train_df['is_coastal'].sum()} samples")
    print(f"  PS (should be 0): {(train_df['is_coastal'] == 0).sum()} samples")
    
    # If we lost all SF data, we need to handle this differently
    if train_df['is_coastal'].sum() == 0:
        print("\n⚠️  WARNING: All San Francisco data was lost after dropping NaNs!")
        print("This is likely because SF data has different column structure or more NaN values.")
        print("Let's check the column differences...")
        
        print(f"\nSF columns: {list(train_df_sf.columns)}")
        print(f"PS columns: {list(train_df_ps.columns)}")
        
        # Check which columns have NaNs in SF vs PS
        sf_nan_counts = train_df_sf.isnull().sum()
        ps_nan_counts = train_df_ps.isnull().sum()
        
        print(f"\nNaN counts comparison:")
        for col in sf_nan_counts.index:
            if col in ps_nan_counts.index:
                print(f"  {col}: SF={sf_nan_counts[col]}, PS={ps_nan_counts[col]}")
            else:
                print(f"  {col}: SF={sf_nan_counts[col]}, PS=missing")
        
        # Try a more selective NaN dropping strategy
        print(f"\nTrying selective NaN dropping...")
        
        # Drop only rows with NaN in essential features, not all features
        essential_features = ['temperature', 'relative_humidity', 'uv', 'wind_avg', 'wind_gust', 
                             'station_pressure', 'solar_radiation', 'illuminance', 'time_of_day', 'day_of_year']
        
        # Recreate the combined datasets
        train_df_sf_clean = train_df_sf.copy()
        val_df_sf_clean = val_df_sf.copy()
        train_df_ps_clean = train_df_ps.copy()
        val_df_ps_clean = val_df_ps.copy()
        
        # Add coastal flags
        train_df_sf_clean['is_coastal'] = 1
        val_df_sf_clean['is_coastal'] = 1
        train_df_ps_clean['is_coastal'] = 0
        val_df_ps_clean['is_coastal'] = 0
        
        # Handle missing columns
        if 'wet_bulb_temperature' not in train_df_sf_clean.columns:
            train_df_sf_clean['wet_bulb_temperature'] = np.nan
            val_df_sf_clean['wet_bulb_temperature'] = np.nan
        
        # Drop NaNs only from essential features
        essential_cols_sf = [col for col in essential_features if col in train_df_sf_clean.columns]
        essential_cols_ps = [col for col in essential_features if col in train_df_ps_clean.columns]
        
        train_df_sf_clean = train_df_sf_clean.dropna(subset=essential_cols_sf)
        val_df_sf_clean = val_df_sf_clean.dropna(subset=essential_cols_sf)
        train_df_ps_clean = train_df_ps_clean.dropna(subset=essential_cols_ps)
        val_df_ps_clean = val_df_ps_clean.dropna(subset=essential_cols_ps)
        
        print(f"After selective NaN dropping:")
        print(f"  SF training: {len(train_df_sf_clean)} samples")
        print(f"  SF validation: {len(val_df_sf_clean)} samples")
        print(f"  PS training: {len(train_df_ps_clean)} samples")
        print(f"  PS validation: {len(val_df_ps_clean)} samples")
        
        # Recreate combined datasets
        train_df = pd.concat([train_df_sf_clean, train_df_ps_clean], ignore_index=True)
        val_df = pd.concat([val_df_sf_clean, val_df_ps_clean], ignore_index=True)
        
        print(f"Combined datasets:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Coastal flag distribution: SF={train_df['is_coastal'].sum()}, PS={(train_df['is_coastal'] == 0).sum()}")
        
        # Regenerate all features for the cleaned datasets
        print(f"\nRegenerating features for cleaned datasets...")
        
        # Handle wet_bulb_temperature NaN issue - fill with 0 for SF data
        train_df['wet_bulb_temperature'] = train_df['wet_bulb_temperature'].fillna(0)
        val_df['wet_bulb_temperature'] = val_df['wet_bulb_temperature'].fillna(0)
        
        # Add lag features - but handle NaNs more carefully
        print("Adding lag features...")
        train_df['temp_lag30'] = lag_feature(train_df['temperature'].values, 30)
        train_df['humidity_lag30'] = lag_feature(train_df['relative_humidity'].values, 30)
        val_df['temp_lag30'] = lag_feature(val_df['temperature'].values, 30)
        val_df['humidity_lag30'] = lag_feature(val_df['relative_humidity'].values, 30)
        
        train_df['temp_lag60'] = lag_feature(train_df['temperature'].values, 60)
        train_df['temp_lag120'] = lag_feature(train_df['temperature'].values, 120)
        train_df['humidity_lag60'] = lag_feature(train_df['relative_humidity'].values, 60)
        train_df['humidity_lag120'] = lag_feature(train_df['relative_humidity'].values, 120)
        val_df['temp_lag60'] = lag_feature(val_df['temperature'].values, 60)
        val_df['temp_lag120'] = lag_feature(val_df['temperature'].values, 120)
        val_df['humidity_lag60'] = lag_feature(val_df['relative_humidity'].values, 60)
        val_df['humidity_lag120'] = lag_feature(val_df['relative_humidity'].values, 120)
        
        train_df['wind_avg_lag30'] = lag_feature(train_df['wind_avg'].values, 30)
        train_df['wind_gust_lag30'] = lag_feature(train_df['wind_gust'].values, 30)
        train_df['uv_lag30'] = lag_feature(train_df['uv'].values, 30)
        train_df['pressure_lag30'] = lag_feature(train_df['station_pressure'].values, 30)
        val_df['wind_avg_lag30'] = lag_feature(val_df['wind_avg'].values, 30)
        val_df['wind_gust_lag30'] = lag_feature(val_df['wind_gust'].values, 30)
        val_df['uv_lag30'] = lag_feature(val_df['uv'].values, 30)
        val_df['pressure_lag30'] = lag_feature(val_df['station_pressure'].values, 30)

        # Cyclical encoding
        print("Adding cyclical features...")
        train_df['time_of_day_sin'] = np.sin(2 * np.pi * train_df['time_of_day'] / 24.0)
        train_df['time_of_day_cos'] = np.cos(2 * np.pi * train_df['time_of_day'] / 24.0)
        val_df['time_of_day_sin'] = np.sin(2 * np.pi * val_df['time_of_day'] / 24.0)
        val_df['time_of_day_cos'] = np.cos(2 * np.pi * val_df['time_of_day'] / 24.0)
        
        train_df['time_of_day_sin2'] = np.sin(4 * np.pi * train_df['time_of_day'] / 24.0)
        train_df['time_of_day_cos2'] = np.cos(4 * np.pi * train_df['time_of_day'] / 24.0)
        val_df['time_of_day_sin2'] = np.sin(4 * np.pi * val_df['time_of_day'] / 24.0)
        val_df['time_of_day_cos2'] = np.cos(4 * np.pi * val_df['time_of_day'] / 24.0)

        train_df['day_of_year_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365.25)
        train_df['day_of_year_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365.25)
        val_df['day_of_year_sin'] = np.sin(2 * np.pi * val_df['day_of_year'] / 365.25)
        val_df['day_of_year_cos'] = np.cos(2 * np.pi * val_df['day_of_year'] / 365.25)

        # Delta features - handle NaNs more carefully
        print("Adding delta features...")
        train_df['temperature_delta'] = rolling_slope_numba(train_df['temperature'].values, 15)
        val_df['temperature_delta'] = rolling_slope_numba(val_df['temperature'].values, 15)
        train_df['pressure_delta'] = rolling_slope_numba(train_df['station_pressure'].values, 15)
        train_df['humidity_delta'] = rolling_slope_numba(train_df['relative_humidity'].values, 15)
        val_df['pressure_delta'] = rolling_slope_numba(val_df['station_pressure'].values, 15)
        val_df['humidity_delta'] = rolling_slope_numba(val_df['relative_humidity'].values, 15)
        
        # Check data distribution after lag and delta features
        print(f"After lag and delta features:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Coastal flag distribution: SF={train_df['is_coastal'].sum()}, PS={(train_df['is_coastal'] == 0).sum()}")
        
        # If we're losing SF data, let's check what's causing it
        if train_df['is_coastal'].sum() == 0:
            print("⚠️  SF data lost after lag/delta features!")
            # Check which features have NaNs
            nan_counts = train_df.isnull().sum()
            print("NaN counts after lag/delta features:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count} NaN values")
            return  # Exit early to debug

        # Climate-specific features
        train_df['coastal_temp'] = train_df['is_coastal'] * train_df['temp_lag30']
        train_df['desert_temp'] = (1 - train_df['is_coastal']) * train_df['temp_lag30']
        val_df['coastal_temp'] = val_df['is_coastal'] * val_df['temp_lag30']
        val_df['desert_temp'] = (1 - val_df['is_coastal']) * val_df['temp_lag30']
        
        train_df['coastal_humidity'] = train_df['is_coastal'] * train_df['humidity_lag30']
        train_df['desert_humidity'] = (1 - train_df['is_coastal']) * train_df['humidity_lag30']
        val_df['coastal_humidity'] = val_df['is_coastal'] * val_df['humidity_lag30']
        val_df['desert_humidity'] = (1 - val_df['is_coastal']) * val_df['humidity_lag30']
        
        train_df['coastal_wind'] = train_df['is_coastal'] * train_df['wind_avg']
        train_df['desert_wind'] = (1 - train_df['is_coastal']) * train_df['wind_avg']
        val_df['coastal_wind'] = val_df['is_coastal'] * val_df['wind_avg']
        val_df['desert_wind'] = (1 - val_df['is_coastal']) * val_df['wind_avg']
        
        train_df['daily_temp_range'] = train_df['temp_lag30'] - train_df['temp_lag120']
        train_df['coastal_temp_range'] = train_df['is_coastal'] * train_df['daily_temp_range']
        train_df['desert_temp_range'] = (1 - train_df['is_coastal']) * train_df['daily_temp_range']
        val_df['daily_temp_range'] = val_df['temp_lag30'] - val_df['temp_lag120']
        val_df['coastal_temp_range'] = val_df['is_coastal'] * val_df['daily_temp_range']
        val_df['desert_temp_range'] = (1 - val_df['is_coastal']) * val_df['daily_temp_range']
        
        train_df['humidity_stability'] = np.abs(train_df['humidity_lag30'] - train_df['humidity_lag60'])
        train_df['coastal_humidity_stability'] = train_df['is_coastal'] * train_df['humidity_stability']
        train_df['desert_humidity_stability'] = (1 - train_df['is_coastal']) * train_df['humidity_stability']
        val_df['humidity_stability'] = np.abs(val_df['humidity_lag30'] - val_df['humidity_lag60'])
        val_df['coastal_humidity_stability'] = val_df['is_coastal'] * val_df['humidity_stability']
        val_df['desert_humidity_stability'] = (1 - val_df['is_coastal']) * val_df['humidity_stability']
        
        train_df['time_sin_uv'] = train_df['time_of_day_sin'] * train_df['uv']
        train_df['time_cos_uv'] = train_df['time_of_day_cos'] * train_df['uv']
        train_df['time_sin_temp_lag'] = train_df['time_of_day_sin'] * train_df['temp_lag30']
        train_df['time_cos_temp_lag'] = train_df['time_of_day_cos'] * train_df['temp_lag30']
        val_df['time_sin_uv'] = val_df['time_of_day_sin'] * val_df['uv']
        val_df['time_cos_uv'] = val_df['time_of_day_cos'] * val_df['uv']
        val_df['time_sin_temp_lag'] = val_df['time_of_day_sin'] * val_df['temp_lag30']
        val_df['time_cos_temp_lag'] = val_df['time_of_day_cos'] * val_df['temp_lag30']

        # Calculate temperature differences as targets
        train_df['temp_diff_1hr'] = train_df['temp_t+1hr'] - train_df['temperature']
        train_df['temp_diff_2hr'] = train_df['temp_t+2hr'] - train_df['temperature']
        train_df['temp_diff_3hr'] = train_df['temp_t+3hr'] - train_df['temperature']
        val_df['temp_diff_1hr'] = val_df['temp_t+1hr'] - val_df['temperature']
        val_df['temp_diff_2hr'] = val_df['temp_t+2hr'] - val_df['temperature']
        val_df['temp_diff_3hr'] = val_df['temp_t+3hr'] - val_df['temperature']
        
        # Instead of dropping all NaNs, let's use a more selective approach
        print("Handling remaining NaNs...")
        
        # Check what NaNs we have
        nan_counts = train_df.isnull().sum()
        print("NaN counts before final cleanup:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count} NaN values")
        
        # For lag features, fill NaNs with forward fill or mean
        lag_features = ['temp_lag30', 'temp_lag60', 'temp_lag120', 'humidity_lag30', 'humidity_lag60', 'humidity_lag120',
                       'wind_avg_lag30', 'wind_gust_lag30', 'uv_lag30', 'pressure_lag30']
        
        for feature in lag_features:
            if feature in train_df.columns:
                # Fill NaNs with the mean of non-NaN values
                mean_val = train_df[feature].mean()
                train_df[feature] = train_df[feature].fillna(mean_val)
                val_df[feature] = val_df[feature].fillna(mean_val)
        
        # For delta features, fill NaNs with 0 (no change)
        delta_features = ['temperature_delta', 'pressure_delta', 'humidity_delta']
        for feature in delta_features:
            if feature in train_df.columns:
                train_df[feature] = train_df[feature].fillna(0)
                val_df[feature] = val_df[feature].fillna(0)
        
        # For climate-specific features, they should be fine now
        climate_features = ['coastal_temp', 'desert_temp', 'coastal_humidity', 'desert_humidity', 
                           'coastal_wind', 'desert_wind', 'coastal_temp_range', 'desert_temp_range',
                           'coastal_humidity_stability', 'desert_humidity_stability']
        
        for feature in climate_features:
            if feature in train_df.columns:
                train_df[feature] = train_df[feature].fillna(0)
                val_df[feature] = val_df[feature].fillna(0)
        
        # For interaction features
        interaction_features = ['time_sin_uv', 'time_cos_uv', 'time_sin_temp_lag', 'time_cos_temp_lag']
        for feature in interaction_features:
            if feature in train_df.columns:
                train_df[feature] = train_df[feature].fillna(0)
                val_df[feature] = val_df[feature].fillna(0)
        
        print(f"Final datasets after feature generation:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Coastal flag distribution: SF={train_df['is_coastal'].sum()}, PS={(train_df['is_coastal'] == 0).sum()}")
        
        # Final check for any remaining NaNs
        final_nan_counts = train_df.isnull().sum()
        remaining_nans = final_nan_counts.sum()
        if remaining_nans > 0:
            print(f"⚠️  Still have {remaining_nans} NaN values remaining!")
            for col, count in final_nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count} NaN values")
        else:
            print("✅ All NaN values handled successfully!")

    # Define feature and target columns - Simplified climate-specific features
    features = [
        # Core environmental features
        'uv', 'wind_avg', 'wind_gust', 'temperature_delta', 'pressure_delta', 'humidity_delta',
        
        # Cyclical time features
        'day_of_year_sin', 'day_of_year_cos',
        'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
        
        # Lag features
        'temp_lag30', 'humidity_lag30', 'temp_lag60', 'humidity_lag60', 'temp_lag120', 'humidity_lag120',
        'wind_avg_lag30', 'wind_gust_lag30', 'uv_lag30', 'pressure_lag30',
        
        # Climate-specific features (simplified to avoid NaN issues)
        'is_coastal',  # Coastal climate flag (1=coastal/SF, 0=desert/PS)
        'coastal_temp', 'desert_temp',
        'coastal_humidity', 'desert_humidity',
        'coastal_wind', 'desert_wind',
        
        # Climate-specific range and stability features
        'daily_temp_range', 'coastal_temp_range', 'desert_temp_range',
        'humidity_stability', 'coastal_humidity_stability', 'desert_humidity_stability',
        
        # High-importance interaction features (keeping only the useful ones)
        'time_sin_uv', 'time_cos_uv', 'time_sin_temp_lag', 'time_cos_temp_lag'
    ]
    targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']

    # Check feature value ranges after dropping NaNs
    print("\n=== FEATURE VALUE RANGES (after dropping NaNs) ===")
    for feature in features:
        if feature in train_df.columns:
            f_min = train_df[feature].min()
            f_max = train_df[feature].max()
            f_mean = train_df[feature].mean()
            f_std = train_df[feature].std()
            print(f"  {feature}: min={f_min:.4f}, max={f_max:.4f}, mean={f_mean:.4f}, std={f_std:.4f}")

    ## Per-feature min/max scaling with ±5% padding
    # Domain bounds for select features
    domain_bounds = {
        # Core environmental features
        "wind_gust": (0, None),
        "wind_avg": (0, None),
        "uv": (0, None),
        "humidity_lag30": (0, 100),
        "temp_lag30": (-10, 55),
        
        # Cyclical time features
        "day_of_year_sin": (-1, 1),         # Sine component naturally bounded [-1, 1]
        "day_of_year_cos": (-1, 1),         # Cosine component naturally bounded [-1, 1]
        "time_of_day_sin": (-1, 1),         # Sine component naturally bounded [-1, 1]
        "time_of_day_cos": (-1, 1),         # Cosine component naturally bounded [-1, 1]
        "time_of_day_sin2": (-1, 1),        # Higher-order sine component
        "time_of_day_cos2": (-1, 1),        # Higher-order cosine component
        
        # Lag features
        "temp_lag60": (-10, 55),            # 1-hour lag temperature
        "temp_lag120": (-10, 55),           # 2-hour lag temperature
        "humidity_lag60": (0, 100),         # 1-hour lag humidity
        "humidity_lag120": (0, 100),        # 2-hour lag humidity
        "wind_avg_lag30": (0, None),        # 30-min lag wind average
        "wind_gust_lag30": (0, None),       # 30-min lag wind gust
        "uv_lag30": (0, None),              # 30-min lag UV
        "pressure_lag30": (None, None),     # 30-min lag pressure
        
        # Climate-specific features (simplified)
        "is_coastal": (0, 1),  # Binary flag: 0=desert/PS, 1=coastal/SF
        "coastal_temp": (None, None),       # Coastal temperature
        "desert_temp": (None, None),        # Desert temperature
        "coastal_humidity": (None, None),  # Coastal humidity
        "desert_humidity": (None, None),    # Desert humidity
        "coastal_wind": (None, None),       # Coastal wind
        "desert_wind": (None, None),        # Desert wind
        
        # Climate-specific range and stability features
        "daily_temp_range": (None, None),           # Temperature range over 2 hours
        "coastal_temp_range": (None, None),         # Coastal temperature range
        "desert_temp_range": (None, None),          # Desert temperature range
        "humidity_stability": (0, None),             # Humidity stability (absolute difference)
        "coastal_humidity_stability": (None, None), # Coastal humidity stability
        "desert_humidity_stability": (None, None),   # Desert humidity stability
        
        # High-importance interaction features
        "time_sin_uv": (None, None),        # Time-UV interaction
        "time_cos_uv": (None, None),        # Time-UV interaction
        "time_sin_temp_lag": (None, None),  # Time-temp interaction
        "time_cos_temp_lag": (None, None),  # Time-temp interaction
        
        # Delta features
        "temperature_delta": (None, None),  # Allow dynamic bounds for expanded delta range
        "pressure_delta": (None, None),     # Allow dynamic bounds for pressure delta range
        "humidity_delta": (None, None),     # Allow dynamic bounds for humidity delta range
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
    with open("input_scaler_combined_diff.json", "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Normalize target values (temperature differences)
    # Calculate bounds from actual temperature difference data with padding
    y_min = train_df[targets].min().min() - 2  # Add padding for temperature differences
    y_max = train_df[targets].max().max() + 2  # Add padding for temperature differences
    # Save original target range
    target_range = (y_min, y_max)
    train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
    val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

    with open("target_scaler_combined_diff.json", "w") as f:
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

        y_train = train_df[targets].values
        y_val = val_df[targets].values

        input_layer = tf.keras.layers.Input(shape=(len(features),), name="input")

        # Simplified but effective architecture to avoid NaN issues
        # Wide component for linear relationships and climate-specific features
        wide = tf.keras.layers.Dense(16, activation='relu')(input_layer)
        wide = tf.keras.layers.Dropout(0.1)(wide)
        
        # Deep component for complex non-linear relationships
        deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        deep = tf.keras.layers.Dropout(0.2)(deep)
        
        # Climate-specific branch - separate processing for coastal vs desert patterns
        climate_branch = tf.keras.layers.Dense(64, activation='relu')(deep)
        climate_branch = tf.keras.layers.Dropout(0.1)(climate_branch)
        
        # Main deep branch
        deep = tf.keras.layers.Dense(64, activation='relu')(deep)
        deep = tf.keras.layers.Dropout(0.1)(deep)
        
        # Residual connection for better gradient flow
        res = tf.keras.layers.Dense(32, activation='relu')(deep)
        shortcut = tf.keras.layers.Dense(32)(deep)
        res = tf.keras.layers.Add()([shortcut, res])
        
        # Combine all components
        merged = tf.keras.layers.Concatenate()([wide, res, climate_branch])
        merged = tf.keras.layers.Dense(32, activation='relu')(merged)
        merged = tf.keras.layers.Dropout(0.1)(merged)
        
        # Output layers with shared representation
        shared = tf.keras.layers.Dense(16, activation='relu')(merged)
        output_1 = tf.keras.layers.Dense(1, activation='linear', name='diff_1hr')(shared)
        output_2 = tf.keras.layers.Dense(1, activation='linear', name='diff_2hr')(shared)
        output_3 = tf.keras.layers.Dense(1, activation='linear', name='diff_3hr')(shared)
        model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

        # Conservative optimizer configuration to avoid NaN issues
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics={
                'diff_1hr': 'mae',
                'diff_2hr': 'mae',
                'diff_3hr': 'mae'
            },
            jit_compile=False  # Conservative: disable JIT initially
        )
        model.summary()

        # Conservative callbacks for stable training
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-4)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/model_{epoch:02d}.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )
        
        # Learning rate reduction for better convergence
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6, verbose=1
        )

        # Conservative training configuration
        history = model.fit(
            X_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
            validation_data=(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]]),
            epochs=100,  # Conservative epoch count
            batch_size=256,  # Larger batch size for stability
            callbacks=[early_stopping, checkpoint_cb, lr_scheduler],
            verbose=1
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

        tflite_fname = f"weather_model_combined_diff_quant_{name}.tflite"
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

        with open(f"results_combined_diff_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Configuration
    NUM_RUNS = 5  # Number of training runs to perform
    for run_id in range(NUM_RUNS):
        run_name = f"dense_wide_run{run_id+1}"
        build_and_train_model(run_name)

    results = []
    # Only collect results from the current run session
    for run_id in range(NUM_RUNS):
        json_file = f"results_combined_diff_dense_wide_run{run_id+1}.json"
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
    best_model_file = f"weather_model_combined_diff_quant_{best['name']}.tflite"
    shutil.copy(best_model_file, "weather_model_combined_diff_best.tflite")
    print(f"Best combined model copied to: weather_model_combined_diff_best.tflite")
    
    # Validate the best quantized model
    print("\n" + "="*50)
    print("VALIDATING BEST QUANTIZED MODEL")
    print("="*50)
    y_val_array = val_df[targets].values
    validate_quantized_model("weather_model_combined_diff_best.tflite", X_val, y_val_array, y_min, y_max)

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



# Entry point guard
if __name__ == "__main__":
    main()
