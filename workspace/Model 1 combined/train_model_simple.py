def main():
    import multiprocessing as mp
    import multiprocessing
    mp.set_start_method("fork", force=True)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    cores = multiprocessing.cpu_count()
    print(f"Setting TensorFlow threads → {cores} cores")
    tf.config.threading.set_intra_op_parallelism_threads(cores)
    tf.config.threading.set_inter_op_parallelism_threads(cores)
    tf.config.set_soft_device_placement(True)
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    import pandas as pd
    import json
    from sklearn.metrics import mean_absolute_error

    print("=== SIMPLIFIED COMBINED MODEL ===")
    print("Focus: Make coastal flag actually useful")
    print("Strategy: Minimal features, robust data handling")
    print("=" * 50)

    # Load data
    print("Loading data...")
    train_df_sf = pd.read_csv("../train_data.csv")
    val_df_sf = pd.read_csv("../val_data.csv")
    train_df_ps = pd.read_csv("../train_data_ps.csv")
    val_df_ps = pd.read_csv("../val_data_ps.csv")
    
    print(f"SF training: {len(train_df_sf)} samples")
    print(f"SF validation: {len(val_df_sf)} samples")
    print(f"PS training: {len(train_df_ps)} samples")
    print(f"PS validation: {len(val_df_ps)} samples")

    # Add coastal flags
    train_df_sf['is_coastal'] = 1
    val_df_sf['is_coastal'] = 1
    train_df_ps['is_coastal'] = 0
    val_df_ps['is_coastal'] = 0

    # Handle missing columns
    if 'wet_bulb_temperature' not in train_df_sf.columns:
        train_df_sf['wet_bulb_temperature'] = 0
        val_df_sf['wet_bulb_temperature'] = 0

    # Combine datasets
    train_df = pd.concat([train_df_sf, train_df_ps], ignore_index=True)
    val_df = pd.concat([val_df_sf, val_df_ps], ignore_index=True)
    
    print(f"Combined training: {len(train_df)} samples")
    print(f"Combined validation: {len(val_df)} samples")
    print(f"Coastal distribution: SF={train_df['is_coastal'].sum()}, PS={(train_df['is_coastal'] == 0).sum()}")

    # Simple feature engineering - only essential features
    print("\nCreating simple features...")
    
    # Basic lag features (only 30-min lag to minimize NaNs)
    def simple_lag(arr, lag):
        arr = np.asarray(arr)
        lagged = np.roll(arr, lag)
        lagged[:lag] = arr[lag]  # Fill first lag values with the lag-th value
        return lagged
    
    train_df['temp_lag30'] = simple_lag(train_df['temperature'].values, 30)
    train_df['humidity_lag30'] = simple_lag(train_df['relative_humidity'].values, 30)
    val_df['temp_lag30'] = simple_lag(val_df['temperature'].values, 30)
    val_df['humidity_lag30'] = simple_lag(val_df['relative_humidity'].values, 30)

    # Cyclical time features
    train_df['time_of_day_sin'] = np.sin(2 * np.pi * train_df['time_of_day'] / 24.0)
    train_df['time_of_day_cos'] = np.cos(2 * np.pi * train_df['time_of_day'] / 24.0)
    val_df['time_of_day_sin'] = np.sin(2 * np.pi * val_df['time_of_day'] / 24.0)
    val_df['time_of_day_cos'] = np.cos(2 * np.pi * val_df['time_of_day'] / 24.0)

    train_df['day_of_year_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365.25)
    train_df['day_of_year_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365.25)
    val_df['day_of_year_sin'] = np.sin(2 * np.pi * val_df['day_of_year'] / 365.25)
    val_df['day_of_year_cos'] = np.cos(2 * np.pi * val_df['day_of_year'] / 365.25)

    # Climate-specific features - these are KEY
    train_df['coastal_temp'] = train_df['is_coastal'] * train_df['temp_lag30']
    train_df['desert_temp'] = (1 - train_df['is_coastal']) * train_df['temp_lag30']
    train_df['coastal_humidity'] = train_df['is_coastal'] * train_df['humidity_lag30']
    train_df['desert_humidity'] = (1 - train_df['is_coastal']) * train_df['humidity_lag30']
    
    val_df['coastal_temp'] = val_df['is_coastal'] * val_df['temp_lag30']
    val_df['desert_temp'] = (1 - val_df['is_coastal']) * val_df['temp_lag30']
    val_df['coastal_humidity'] = val_df['is_coastal'] * val_df['humidity_lag30']
    val_df['desert_humidity'] = (1 - val_df['is_coastal']) * val_df['humidity_lag30']

    # Temperature differences as targets
    train_df['temp_diff_1hr'] = train_df['temp_t+1hr'] - train_df['temperature']
    train_df['temp_diff_2hr'] = train_df['temp_t+2hr'] - train_df['temperature']
    train_df['temp_diff_3hr'] = train_df['temp_t+3hr'] - train_df['temperature']
    val_df['temp_diff_1hr'] = val_df['temp_t+1hr'] - val_df['temperature']
    val_df['temp_diff_2hr'] = val_df['temp_t+2hr'] - val_df['temperature']
    val_df['temp_diff_3hr'] = val_df['temp_t+3hr'] - val_df['temperature']

    # Minimal feature set - only the most important ones
    features = [
        # Core environmental
        'uv', 'wind_avg', 'wind_gust', 'station_pressure', 'solar_radiation',
        
        # Time features
        'time_of_day_sin', 'time_of_day_cos', 'day_of_year_sin', 'day_of_year_cos',
        
        # Lag features
        'temp_lag30', 'humidity_lag30',
        
        # Climate-specific features (THE KEY ONES)
        'is_coastal',
        'coastal_temp', 'desert_temp',
        'coastal_humidity', 'desert_humidity'
    ]
    
    targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']

    # Check for any NaNs and handle them
    print("\nChecking for NaNs...")
    nan_counts = train_df[features + targets].isnull().sum()
    for col, count in nan_counts.items():
        if count > 0:
            print(f"  {col}: {count} NaNs - filling with mean")
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            val_df[col] = val_df[col].fillna(mean_val)

    # Prepare data
    X_train = train_df[features].values
    X_val = val_df[features].values
    y_train = train_df[targets].values
    y_val = val_df[targets].values

    print(f"\nFinal data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")

    # Simple scaling
    print("\nScaling data...")
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Save scalers
    import joblib
    joblib.dump(scaler_X, 'scaler_X_simple.pkl')
    joblib.dump(scaler_y, 'scaler_y_simple.pkl')

    # Simple model architecture
    print("\nBuilding simple model...")
    input_layer = tf.keras.layers.Input(shape=(len(features),), name="input")
    
    # Wide component for linear relationships
    wide = tf.keras.layers.Dense(16, activation='relu')(input_layer)
    wide = tf.keras.layers.Dropout(0.1)(wide)
    
    # Deep component
    deep = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    deep = tf.keras.layers.Dropout(0.2)(deep)
    deep = tf.keras.layers.Dense(32, activation='relu')(deep)
    
    # Combine
    merged = tf.keras.layers.Concatenate()([wide, deep])
    merged = tf.keras.layers.Dense(32, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(0.1)(merged)
    
    # Outputs
    output_1 = tf.keras.layers.Dense(1, activation='linear', name='diff_1hr')(merged)
    output_2 = tf.keras.layers.Dense(1, activation='linear', name='diff_2hr')(merged)
    output_3 = tf.keras.layers.Dense(1, activation='linear', name='diff_3hr')(merged)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

    # Conservative training
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
        jit_compile=False
    )
    
    model.summary()

    # Training
    print("\nTraining model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, [y_train_scaled[:, 0], y_train_scaled[:, 1], y_train_scaled[:, 2]],
        validation_data=(X_val_scaled, [y_val_scaled[:, 0], y_val_scaled[:, 1], y_val_scaled[:, 2]]),
        epochs=50,
        batch_size=512,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate
    print("\nEvaluating model...")
    eval_results = model.evaluate(X_val_scaled, [y_val_scaled[:, 0], y_val_scaled[:, 1], y_val_scaled[:, 2]], verbose=0)
    
    # Convert back to original scale for MAE calculation
    y_pred_scaled = model.predict(X_val_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(np.column_stack(y_pred_scaled))
    y_val_orig = scaler_y.inverse_transform(y_val_scaled)
    
    print(f"\nValidation MAE (in °C difference):")
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_val_orig[:, i], y_pred[:, i])
        print(f"  {target}: {mae:.3f} °C")

    # Feature importance analysis
    print(f"\nFeature Importance Analysis:")
    baseline_loss = eval_results[0]
    
    feature_importance = {}
    for i, feature in enumerate(features):
        X_val_permuted = X_val_scaled.copy()
        np.random.shuffle(X_val_permuted[:, i])
        permuted_loss = model.evaluate(X_val_permuted, [y_val_scaled[:, 0], y_val_scaled[:, 1], y_val_scaled[:, 2]], verbose=0)[0]
        feature_importance[feature] = permuted_loss - baseline_loss

    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("Permutation Feature Importance:")
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")

    # Save model
    model.save('weather_model_simple.h5')
    print(f"\nModel saved as 'weather_model_simple.h5'")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('weather_model_simple.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved as 'weather_model_simple.tflite'")
    
    # Save feature list
    with open('features_simple.json', 'w') as f:
        json.dump(features, f, indent=2)
    
    print(f"Features saved as 'features_simple.json'")

if __name__ == "__main__":
    main()
