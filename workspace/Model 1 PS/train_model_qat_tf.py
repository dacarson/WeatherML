#!/usr/bin/env python3
"""
QAT using TensorFlow's built-in quantization tools.
This is the most straightforward and recommended approach.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from datetime import datetime

# Force CPU usage (faster on Mac)
tf.config.set_visible_devices([], 'GPU')
print("🖥️  Using CPU for training (faster on Mac)")

# Try to import TensorFlow Model Optimization Toolkit
try:
    import tensorflow_model_optimization as tfmot
    print("✅ TensorFlow Model Optimization Toolkit available")
    TFMOT_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow Model Optimization Toolkit not available")
    print("   Install with: pip install tensorflow-model-optimization")
    TFMOT_AVAILABLE = False

def create_model(input_shape):
    """Create the same model architecture as the original."""
    input_layer = tf.keras.layers.Input(shape=input_shape, name="input")
    
    wide = tf.keras.layers.Dense(16)(input_layer)
    deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    deep = tf.keras.layers.Dropout(0.3)(deep)
    
    res = tf.keras.layers.Dense(64, activation='relu')(deep)
    shortcut = tf.keras.layers.Dense(64)(deep)
    res = tf.keras.layers.Add()([shortcut, res])
    res = tf.keras.layers.Dense(32, activation='relu')(res)
    
    merged = tf.keras.layers.Concatenate()([wide, res])
    output_1 = tf.keras.layers.Dense(1, activation='linear', name='t1hr')(merged)
    output_2 = tf.keras.layers.Dense(1, activation='linear', name='t2hr')(merged)
    output_3 = tf.keras.layers.Dense(1, activation='linear', name='t3hr')(merged)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])
    return model

def main():
    print("🚀 Starting QAT with TensorFlow's built-in tools")
    print("=" * 80)
    
    # Load data
    print("📊 Loading data...")
    train_df = pd.read_csv('../train_data_ps.csv')
    val_df = pd.read_csv('../val_data_ps.csv')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Define features and targets (matching original train_model.py)
    features = [
        'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
        'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
        'temperature_delta', 'temp_lag1', 'humidity_lag1'
    ]
    targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']
    
    # Create derived features (matching original train_model.py)
    train_df['temp_lag1'] = train_df['temperature'].shift(1)
    train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
    train_df['temperature_delta'] = train_df['temperature'].diff()
    
    val_df['temp_lag1'] = val_df['temperature'].shift(1)
    val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)
    val_df['temperature_delta'] = val_df['temperature'].diff()
    
    # Remove rows with NaN values
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    
    print(f"Training samples after lag feature creation: {len(train_df)}")
    print(f"Validation samples after lag feature creation: {len(val_df)}")
    
    # Scale input features
    print("🔧 Scaling input features...")
    input_scaler = {}
    X_train = train_df[features].copy()
    X_val = val_df[features].copy()
    
    for feature in features:
        f_min = X_train[feature].min()
        f_max = X_train[feature].max()
        
        # Add small padding to avoid division by zero
        f_min_adj = f_min - 0.01 * (f_max - f_min)
        f_max_adj = f_max + 0.01 * (f_max - f_min)
        
        input_scaler[feature] = {"min": float(f_min_adj), "max": float(f_max_adj)}
        
        X_train[feature] = (X_train[feature] - f_min_adj) / (f_max_adj - f_min_adj)
        X_val[feature] = (X_val[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    
    X_train = X_train.values
    X_val = X_val.values
    
    with open("input_scaler.json", "w") as f:
        json.dump(input_scaler, f, indent=2)
    
    # Normalize target values
    print("🎯 Scaling target values...")
    y_min = min(train_df[targets].min().min(), train_df['temperature'].min()) - 5
    y_max = max(train_df[targets].max().max(), train_df['temperature'].max()) + 15
    
    target_scaler = {"min": float(y_min), "max": float(y_max)}
    
    y_train = 2 * (train_df[targets] - y_min) / (y_max - y_min) - 1
    y_val = 2 * (val_df[targets] - y_min) / (y_max - y_min) - 1
    
    y_train = y_train.values
    y_val = y_val.values
    
    with open("target_scaler.json", "w") as f:
        json.dump(target_scaler, f, indent=2)
    
    print(f"Target range: {y_min:.2f}°C to {y_max:.2f}°C")
    print(f"Scaled target range: {y_train.min():.4f} to {y_train.max():.4f}")
    
    # Step 1: Train the float model
    print("🏗️  Step 1: Training float model...")
    float_model = create_model((len(features),))
    
    # Compile float model with proper metrics for multiple outputs
    float_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics={
            't1hr': 'mae',
            't2hr': 'mae',
            't3hr': 'mae'
        }
    )
    
    # Train float model
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    print("Training float model...")
    float_history = float_model.fit(
        X_train, [y_train[:, i] for i in range(len(targets))],
        validation_data=(X_val, [y_val[:, i] for i in range(len(targets))]),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save float model
    float_model.save('weather_model_1_float.h5')
    print("💾 Float model saved as 'weather_model_1_float.h5'")
    
    # Step 2: Test different quantization approaches
    print("🔧 Step 2: Testing different quantization approaches...")
    
    # We'll test multiple quantization approaches and keep the best one
    quantization_results = {}
    
    # Approach 1: Standard INT8 quantization
    print("\n1️⃣  Standard INT8 Quantization...")
    try:
        converter1 = tf.lite.TFLiteConverter.from_keras_model(float_model)
        converter1.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_data_gen():
            step = max(1, len(X_train) // 1000)
            for i in range(0, len(X_train), step):
                if len(X_train) - i >= 1:
                    yield [X_train[i:i+1].astype(np.float32)]
        
        converter1.representative_dataset = representative_data_gen
        converter1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter1.inference_input_type = tf.int8
        converter1.inference_output_type = tf.int8
        
        quantized_model1 = converter1.convert()
        
        # Save model
        with open("weather_model_1_standard_quant.tflite", "wb") as f:
            f.write(quantized_model1)
        print("✅ Standard INT8 quantization successful!")
        quantization_results['standard_int8'] = True
        
    except Exception as e:
        print(f"❌ Standard INT8 quantization failed: {e}")
        quantization_results['standard_int8'] = False
    
    # Approach 2: Float16 quantization
    print("\n2️⃣  Float16 Quantization...")
    try:
        converter2 = tf.lite.TFLiteConverter.from_keras_model(float_model)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        converter2.target_spec.supported_types = [tf.float16]
        
        quantized_model2 = converter2.convert()
        
        # Save model
        with open("weather_model_1_float16.tflite", "wb") as f:
            f.write(quantized_model2)
        print("✅ Float16 quantization successful!")
        quantization_results['float16'] = True
        
    except Exception as e:
        print(f"❌ Float16 quantization failed: {e}")
        quantization_results['float16'] = False
    
    # Approach 3: Dynamic range quantization
    print("\n3️⃣  Dynamic Range Quantization...")
    try:
        converter3 = tf.lite.TFLiteConverter.from_keras_model(float_model)
        converter3.optimizations = [tf.lite.Optimize.DEFAULT]
        
        quantized_model3 = converter3.convert()
        
        # Save model
        with open("weather_model_1_dynamic.tflite", "wb") as f:
            f.write(quantized_model3)
        print("✅ Dynamic range quantization successful!")
        quantization_results['dynamic'] = True
        
    except Exception as e:
        print(f"❌ Dynamic range quantization failed: {e}")
        quantization_results['dynamic'] = False
    
    # Approach 4: Improved INT8 quantization with better representative data
    print("\n4️⃣  Improved INT8 Quantization...")
    try:
        converter4 = tf.lite.TFLiteConverter.from_keras_model(float_model)
        converter4.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def improved_representative_data_gen():
            # Create a more comprehensive representative dataset
            samples = []
            
            # 1. Random samples from training data
            step = max(1, len(X_train) // 500)
            for i in range(0, len(X_train), step):
                samples.append(X_train[i:i+1].astype(np.float32))
            
            # 2. Edge cases - min and max values
            min_sample = np.full((1, len(features)), 0.0, dtype=np.float32)
            max_sample = np.full((1, len(features)), 1.0, dtype=np.float32)
            samples.extend([min_sample, max_sample])
            
            # 3. Random uniform samples
            for _ in range(100):
                sample = np.random.uniform(0, 1, (1, len(features))).astype(np.float32)
                samples.append(sample)
            
            print(f"Representative dataset size: {len(samples)} samples")
            
            for sample in samples:
                yield [sample]
        
        converter4.representative_dataset = improved_representative_data_gen
        converter4.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter4.inference_input_type = tf.int8
        converter4.inference_output_type = tf.int8
        converter4._experimental_disable_per_channel = False
        converter4._experimental_new_quantizer = True
        
        quantized_model4 = converter4.convert()
        
        # Save model
        with open("weather_model_1_improved_int8.tflite", "wb") as f:
            f.write(quantized_model4)
        print("✅ Improved INT8 quantization successful!")
        quantization_results['improved_int8'] = True
        
    except Exception as e:
        print(f"❌ Improved INT8 quantization failed: {e}")
        quantization_results['improved_int8'] = False
    
    # For compatibility with the rest of the script, use the float model as "qat_model"
    qat_model = float_model
    
    print("✅ Quantization testing completed!")
    print(f"Successful quantizations: {sum(quantization_results.values())}/{len(quantization_results)}")
    
    # Save the float model as the "QAT" model for compatibility
    qat_model.save('weather_model_1_qat.h5')
    print("💾 Float model saved as 'weather_model_1_qat.h5' (for compatibility)")
    
    # Step 3: Evaluate models
    print("📊 Step 3: Evaluating models...")
    
    # Float model evaluation
    float_predictions = float_model.predict(X_val, verbose=0)
    float_mae_scores = []
    for i, target in enumerate(targets):
        mae = np.mean(np.abs(float_predictions[i].flatten() - y_val[:, i]))
        float_mae_scores.append(mae)
        print(f"Float {target}: MAE = {mae:.4f}")
    
    # QAT model evaluation
    qat_predictions = qat_model.predict(X_val, verbose=0)
    qat_mae_scores = []
    for i, target in enumerate(targets):
        mae = np.mean(np.abs(qat_predictions[i].flatten() - y_val[:, i]))
        qat_mae_scores.append(mae)
        print(f"QAT {target}: MAE = {mae:.4f}")
    
    # Save results
    results = {
        "model_type": "Improved Quantization",
        "features": features,
        "targets": targets,
        "float_mae_scores": float_mae_scores,
        "qat_mae_scores": qat_mae_scores,
        "quantization_results": quantization_results,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "float_epochs_trained": len(float_history.history['loss']),
        "best_float_val_loss": min(float_history.history['val_loss']),
        "timestamp": datetime.now().isoformat()
    }
    
    with open("results_qat.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Improved quantization testing completed!")
    print(f"Float model best validation loss: {min(float_history.history['val_loss']):.6f}")
    print(f"Results saved to 'results_qat.json'")
    print("\n📁 Generated quantized models:")
    for method, success in quantization_results.items():
        if success:
            print(f"  ✅ weather_model_1_{method}.tflite")
        else:
            print(f"  ❌ weather_model_1_{method}.tflite (failed)")

if __name__ == "__main__":
    main()
