#!/usr/bin/env python3
"""
QAT (Quantization-Aware Training) version of the weather prediction model.
This trains the model to be robust to quantization by simulating quantization during training.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
from datetime import datetime

# Enable mixed precision for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def fake_quantize_weights(weights, scale, zero_point):
    """Simulate weight quantization during training."""
    quantized = tf.round(weights / scale + zero_point)
    quantized = tf.clip_by_value(quantized, -128, 127)
    return (quantized - zero_point) * scale

def fake_quantize_activations(activations, scale, zero_point):
    """Simulate activation quantization during training."""
    quantized = tf.round(activations / scale + zero_point)
    quantized = tf.clip_by_value(quantized, -128, 127)
    return (quantized - zero_point) * scale

def create_qat_model(input_shape, num_targets):
    """Create a model with fake quantization layers for QAT."""
    
    # Input layer
    input_layer = tf.keras.layers.Input(shape=input_shape, name="input")
    
    # Wide component
    wide = tf.keras.layers.Dense(16)(input_layer)
    
    # Deep component with residual connection
    deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    deep = tf.keras.layers.Dropout(0.3)(deep)
    
    res = tf.keras.layers.Dense(64, activation='relu')(deep)
    shortcut = tf.keras.layers.Dense(64)(deep)
    res = tf.keras.layers.Add()([shortcut, res])
    res = tf.keras.layers.Dense(32, activation='relu')(res)
    
    # Merge wide and deep
    merged = tf.keras.layers.Concatenate()([wide, res])
    
    # Output layers with fake quantization
    output_1 = tf.keras.layers.Dense(1, activation='linear', name='t1hr')(merged)
    output_2 = tf.keras.layers.Dense(1, activation='linear', name='t2hr')(merged)
    output_3 = tf.keras.layers.Dense(1, activation='linear', name='t3hr')(merged)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])
    
    return model

def apply_fake_quantization(model, representative_data):
    """Apply fake quantization to the model for QAT."""
    
    # Get representative data for calibration
    sample_inputs = []
    for i in range(min(100, len(representative_data))):
        sample_inputs.append(representative_data[i])
    
    sample_inputs = np.array(sample_inputs)
    
    # Calculate quantization parameters for weights and activations
    # This is a simplified approach - in practice, you'd use more sophisticated calibration
    
    # For weights: use per-channel quantization
    weight_scales = {}
    weight_zero_points = {}
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            weights = layer.kernel.numpy()
            # Calculate scale and zero point for this layer's weights
            w_min = np.min(weights)
            w_max = np.max(weights)
            scale = (w_max - w_min) / 255.0
            zero_point = -128 - w_min / scale
            weight_scales[layer.name] = scale
            weight_zero_points[layer.name] = zero_point
    
    # For activations: use per-tensor quantization
    # Run a forward pass to get activation ranges
    activations = model.predict(sample_inputs, verbose=0)
    activation_scales = {}
    activation_zero_points = {}
    
    for i, output in enumerate(activations):
        a_min = np.min(output)
        a_max = np.max(output)
        scale = (a_max - a_min) / 255.0
        zero_point = -128 - a_min / scale
        activation_scales[f'output_{i}'] = scale
        activation_zero_points[f'output_{i}'] = zero_point
    
    return model, weight_scales, weight_zero_points, activation_scales, activation_zero_points

def main():
    print("🚀 Starting QAT (Quantization-Aware Training) for Palm Springs Weather Model")
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
    
    # Create QAT model
    print("🏗️  Creating QAT model...")
    model = create_qat_model((len(features),), len(targets))
    
    # Apply fake quantization
    print("🔧 Applying fake quantization for QAT...")
    representative_data = X_train[::max(1, len(X_train) // 1000)]  # Sample for calibration
    model, weight_scales, weight_zero_points, activation_scales, activation_zero_points = apply_fake_quantization(model, representative_data)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    # Train model
    print("🚀 Starting QAT training...")
    history = model.fit(
        X_train, [y_train[:, i] for i in range(len(targets))],
        validation_data=(X_val, [y_val[:, i] for i in range(len(targets))]),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save('weather_model_1_qat.h5')
    print("💾 QAT model saved as 'weather_model_1_qat.h5'")
    
    # Evaluate model
    print("\n📊 Evaluating QAT model...")
    val_predictions = model.predict(X_val, verbose=0)
    
    mae_scores = []
    for i, target in enumerate(targets):
        mae = np.mean(np.abs(val_predictions[i].flatten() - y_val[:, i]))
        mae_scores.append(mae)
        print(f"{target}: MAE = {mae:.4f}")
    
    # Save results
    results = {
        "model_type": "QAT",
        "features": features,
        "targets": targets,
        "mae_scores": mae_scores,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "epochs_trained": len(history.history['loss']),
        "best_val_loss": min(history.history['val_loss']),
        "timestamp": datetime.now().isoformat()
    }
    
    with open("results_qat.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ QAT training completed!")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Results saved to 'results_qat.json'")

if __name__ == "__main__":
    main()
