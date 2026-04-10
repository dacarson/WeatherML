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
        'illuminance_delta', 'solar_radiation_delta', 'uv', 'wind_avg', 'wind_gust', 
        'day_of_year_sin', 'day_of_year_cos',
        'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
        # Manual interaction features - COMMENTED OUT: Using learned FeatureInteraction layer instead
        # 'time_sin_solar', 'time_cos_solar', 'time_sin_uv', 'time_cos_uv',
        # 'time_sin_wind', 'time_cos_wind', 'time_sin_temp_lag', 'time_cos_temp_lag',
        # 'season_sin_temp_lag', 'season_cos_temp_lag', 'season_sin_humidity', 'season_cos_humidity',
        'temperature_delta', 'pressure_delta', 'humidity_delta', 
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
    with open("input_scaler_5.json", "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Normalize target values (temperature differences)
    # Calculate bounds from actual temperature difference data with padding
    y_min = train_df[targets].min().min() - 2  # Add padding for temperature differences
    y_max = train_df[targets].max().max() + 2  # Add padding for temperature differences
    # Save original target range
    target_range = (y_min, y_max)
    train_df[targets] = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
    val_df[targets] = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1

    with open("target_scaler_5.json", "w") as f:
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

        # Learn feature interactions automatically (Edge TPU compatible)
        interaction_projection = build_feature_interaction_path(input_layer, embedding_dim=16, projection_dim=32)
        
        wide = tf.keras.layers.Dense(16)(input_layer)
        deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        deep = tf.keras.layers.Dropout(0.3)(deep)

        res = tf.keras.layers.Dense(64, activation='relu')(deep)
        shortcut = tf.keras.layers.Dense(64)(deep)
        res = tf.keras.layers.Add()([shortcut, res])
        res = tf.keras.layers.Dense(32, activation='relu')(res)

        merged = tf.keras.layers.Concatenate()([wide, res, interaction_projection])
        output_1 = tf.keras.layers.Dense(1, activation='linear', name='diff_1hr')(merged)
        output_2 = tf.keras.layers.Dense(1, activation='linear', name='diff_2hr')(merged)
        output_3 = tf.keras.layers.Dense(1, activation='linear', name='diff_3hr')(merged)
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
            },
            jit_compile=False  # Conservative: disable JIT initially
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

        tflite_fname = f"weather_model_5_quant_{name}.tflite"
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

        with open(f"results_5_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Configuration
    NUM_RUNS = 5  # Number of training runs to perform
    for run_id in range(NUM_RUNS):
        run_name = f"dense_wide_run{run_id+1}"
        build_and_train_model(run_name)

    results = []
    # Only collect results from the current run session
    for run_id in range(NUM_RUNS):
        json_file = f"results_5_dense_wide_run{run_id+1}.json"
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
    best_model_file = f"weather_model_5_quant_{best['name']}.tflite"
    shutil.copy(best_model_file, "weather_model_5_best.tflite")
    print(f"Best model copied to: weather_model_5_best.tflite")
    
    # Validate the best quantized model
    print("\n" + "="*50)
    print("VALIDATING BEST QUANTIZED MODEL")
    print("="*50)
    y_val_array = val_df[targets].values
    validate_quantized_model("weather_model_5_best.tflite", X_val, y_val_array, y_min, y_max)

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
