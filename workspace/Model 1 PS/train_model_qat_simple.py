#!/usr/bin/env python3
"""
Simple QAT (Quantization-Aware Training) using TensorFlow's built-in tools.
This is a more straightforward approach that should work better.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from datetime import datetime

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
    print("🚀 Starting Simple QAT (Quantization-Aware Training) for Palm Springs Weather Model")
    print("=" * 80)
    
    # Load data
    print("📊 Loading data...")
    train_df = pd.read_csv('train_data_ps.csv')
    val_df = pd.read_csv('val_data_ps.csv')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Define features and targets
    features = ['temperature', 'relative_humidity', 'pressure', 'wind_speed', 'wind_direction', 'temp_lag1', 'humidity_lag1']
    targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']
    
    # Create lag features
    train_df['temp_lag1'] = train_df['temperature'].shift(1)
    train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
    val_df['temp_lag1'] = val_df['temperature'].shift(1)
    val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)
    
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
    
    # Step 1: Train the float model first
    print("🏗️  Step 1: Training float model...")
    float_model = create_model((len(features),))
    float_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
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
        epochs=100,  # Shorter training for float model
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save float model
    float_model.save('weather_model_1_float.h5')
    print("💾 Float model saved as 'weather_model_1_float.h5'")
    
    # Step 2: Create QAT model by cloning and applying fake quantization
    print("🔧 Step 2: Creating QAT model...")
    
    # Clone the trained model
    qat_model = tf.keras.models.clone_model(float_model)
    qat_model.set_weights(float_model.get_weights())
    
    # Apply fake quantization to the model
    # This simulates quantization during training
    qat_model = tf.quantization.quantize_model(qat_model)
    
    # Compile QAT model
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='mse',
        metrics=['mae']
    )
    
    print("🚀 Step 3: Fine-tuning with QAT...")
    
    # Fine-tune with QAT (shorter training since we're starting from trained weights)
    qat_callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    qat_history = qat_model.fit(
        X_train, [y_train[:, i] for i in range(len(targets))],
        validation_data=(X_val, [y_val[:, i] for i in range(len(targets))]),
        epochs=50,  # Shorter fine-tuning
        batch_size=32,
        callbacks=qat_callbacks,
        verbose=1
    )
    
    # Save QAT model
    qat_model.save('weather_model_1_qat.h5')
    print("💾 QAT model saved as 'weather_model_1_qat.h5'")
    
    # Step 4: Convert to quantized TFLite
    print("🔄 Step 4: Converting to quantized TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_data_gen():
        step = max(1, len(X_train) // 1000)
        for i in range(0, len(X_train), step):
            if len(X_train) - i >= 1:
                yield [X_train[i:i+1].astype(np.float32)]
    
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_disable_per_channel = False
    converter._experimental_new_quantizer = True
    
    try:
        quantized_tflite_model = converter.convert()
        print("✅ Quantization successful!")
        
        # Save quantized model
        tflite_fname = "weather_model_1_qat_quantized.tflite"
        with open(tflite_fname, "wb") as f:
            f.write(quantized_tflite_model)
        print(f"💾 Quantized model saved as '{tflite_fname}'")
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        print("Using float model instead...")
        # Fallback to float model
        converter_float = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        quantized_tflite_model = converter_float.convert()
        tflite_fname = "weather_model_1_qat_float.tflite"
        with open(tflite_fname, "wb") as f:
            f.write(quantized_tflite_model)
        print(f"💾 Float model saved as '{tflite_fname}'")
    
    # Evaluate both models
    print("\n📊 Evaluating models...")
    
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
        "model_type": "QAT",
        "features": features,
        "targets": targets,
        "float_mae_scores": float_mae_scores,
        "qat_mae_scores": qat_mae_scores,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "float_epochs_trained": len(float_history.history['loss']),
        "qat_epochs_trained": len(qat_history.history['loss']),
        "best_float_val_loss": min(float_history.history['val_loss']),
        "best_qat_val_loss": min(qat_history.history['val_loss']),
        "timestamp": datetime.now().isoformat()
    }
    
    with open("results_qat.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ QAT training completed!")
    print(f"Float model best validation loss: {min(float_history.history['val_loss']):.6f}")
    print(f"QAT model best validation loss: {min(qat_history.history['val_loss']):.6f}")
    print(f"Results saved to 'results_qat.json'")

if __name__ == "__main__":
    main()
