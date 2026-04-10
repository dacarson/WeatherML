#!/usr/bin/env python3
"""
Standalone validation script for Model 4 quantized model.
Run this to test the quantized model without retraining.
Supports both dynamic range and Coral TPU INT8 quantization.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide GPU devices
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error
import json
import glob

def rolling_slope(series, window):
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
        raw=True
    )

def build_model():
    """Build the Model 4 architecture"""
    base_features = [
        'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
        'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
        'temperature_delta', 'temp_lag1', 'humidity_lag1'
    ]
    
    input_layer = tf.keras.layers.Input(shape=(len(base_features) * 2,), name="input")
    input_normalized = tf.keras.layers.LayerNormalization()(input_layer)

    wide = tf.keras.layers.Dense(16)(input_normalized)
    deep = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_normalized)
    deep = tf.keras.layers.Dropout(0.2)(deep)

    res = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(deep)
    shortcut = tf.keras.layers.Dense(32)(deep)
    res = tf.keras.layers.Add()([shortcut, res])
    res = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(res)

    merged = tf.keras.layers.Concatenate()([wide, res])
    output_1 = tf.keras.layers.Dense(1, activation='linear', name='t1hr')(merged)
    output_2 = tf.keras.layers.Dense(1, activation='linear', name='t2hr')(merged)
    output_3 = tf.keras.layers.Dense(1, activation='linear', name='t3hr')(merged)
    model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics={
            't1hr': 'mae',
            't2hr': 'mae',
            't3hr': 'mae'
        }
    )
    return model

def convert_to_msb_lsb_separate(data):
    """Convert normalized data to MSB/LSB as separate features"""
    # Scale to [0, 65535] range for 16-bit representation
    data_scaled = (data * 65535).astype(np.uint16)
    
    # Extract MSB and LSB
    lsb = (data_scaled & 0xFF).astype(np.int8)
    msb = (data_scaled >> 8).astype(np.int8)
    
    # Convert back to float32 for model input
    lsb_float = lsb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    msb_float = msb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Combine MSB and LSB as separate features
    # Order: [feature1_LSB, feature1_MSB, feature2_LSB, feature2_MSB, ...]
    msb_lsb_data = np.empty((data.shape[0], data.shape[1] * 2))
    msb_lsb_data[:, 0::2] = lsb_float  # LSB features (even indices)
    msb_lsb_data[:, 1::2] = msb_float  # MSB features (odd indices)
    
    return msb_lsb_data

def quantize_model_for_coral_tpu(model, X_train, output_path="weather_model_4_coral_tpu.tflite"):
    """Quantize model specifically for Coral TPU deployment"""
    print("Creating Coral TPU compatible INT8 quantization...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Coral TPU requires full INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Representative dataset for calibration
    def representative_data_gen():
        step = max(1, len(X_train) // 1000)
        for i in range(0, len(X_train), step):
            if len(X_train) - i >= 1:
                x = X_train[i]
                # Provide FLOAT32 data for quantization calibration
                x_combined = np.expand_dims(x, axis=0)
                yield [x_combined.astype(np.float32)]
    
    converter.representative_dataset = representative_data_gen
    converter._experimental_disable_per_channel = False
    converter._experimental_new_quantizer = True
    
    try:
        coral_tpu_model = converter.convert()
        print("✅ Coral TPU quantization successful!")
        
        with open(output_path, "wb") as f:
            f.write(coral_tpu_model)
        
        model_size_kb = len(coral_tpu_model) / 1024
        print(f"Coral TPU model size: {model_size_kb:.2f} KB")
        return True
        
    except Exception as e:
        print(f"❌ Coral TPU quantization failed: {e}")
        return False

def validate_quantized_model(tflite_model_path, X_val, y_val, y_min, y_max, num_samples=500):
    """Validate the quantized TFLite model."""
    print(f"\nValidating TFLite model on {num_samples} samples...")

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if quantization parameters exist
    if 'quantization' in input_details[0] and input_details[0]['quantization'] is not None:
        input_scale, input_zero_point = input_details[0]['quantization']
        print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    else:
        print("No input quantization (dynamic range)")
        input_scale, input_zero_point = 1.0, 0.0
    
    # Check if this is a problematic dynamic range quantization
    if input_scale == 0.0 and input_zero_point == 0.0:
        print("⚠️  WARNING: This appears to be a problematic dynamic range quantization!")
        print("   The model may output zeros due to quantization issues.")
        print("   Skipping validation of this model.")
        return
    
    if 'quantization' in output_details[0] and output_details[0]['quantization'] is not None:
        output_scales = [d['quantization'][0] for d in output_details]
        output_zero_points = [d['quantization'][1] for d in output_details]
        print(f"Output scales: {output_scales}")
        print(f"Output zero points: {output_zero_points}")
    else:
        print("No output quantization (dynamic range)")
        output_scales = [1.0] * len(output_details)
        output_zero_points = [0.0] * len(output_details)

    X_val_subset = X_val[:num_samples]
    y_val_subset = y_val[:num_samples]

    print(f"Input range: min={X_val_subset.min():.4f}, max={X_val_subset.max():.4f}")

    # For Coral TPU INT8 quantization, convert input to INT8 range
    if input_details[0]['dtype'] == np.int8:
        # Convert from [0,1] range to INT8 range [-128, 127]
        input_data = (X_val_subset * 255 - 128).astype(np.int8)
        print(f"Input range: min={X_val_subset.min():.4f}, max={X_val_subset.max():.4f}")
        print(f"INT8 input range: min={input_data.min()}, max={input_data.max()}")
    else:
        # For float32 models, use data as-is
        input_data = X_val_subset.astype(np.float32)
        print(f"Input range: min={X_val_subset.min():.4f}, max={X_val_subset.max():.4f}")

    print(f"Model expects input type: {input_details[0]['dtype']}")
    print(f"Model expects input shape: {input_details[0]['shape']}")
    print(f"Input data shape: {input_data.shape}")

    y_preds_dequant = [[] for _ in range(3)]

    for i in range(len(input_data)):
        input_sample = np.expand_dims(input_data[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_sample)
        interpreter.invoke()
        for j in range(3):
            output_data = interpreter.get_tensor(output_details[j]['index'])
            if output_scales[j] != 1.0 or output_zero_points[j] != 0.0:
                # Quantized output - dequantize
                output_float = (output_data.astype(np.float32) - output_zero_points[j]) * output_scales[j]
            else:
                # Non-quantized output - use as is
                output_float = output_data.astype(np.float32)
            y_preds_dequant[j].append(output_float[0])

    # Debug: show example raw output before rescaling
    print("\nSample dequantized model outputs (before rescaling):")
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        print(f"  {name}: {np.array(y_preds_dequant[j][:5]).round(3)}")

    print("\nValidation MAE (in °C):")
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        y_preds_rescaled = 0.5 * (np.array(y_preds_dequant[j]) + 1) * (y_max - y_min) + y_min
        y_val_rescaled = 0.5 * (y_val_subset[:, j] + 1) * (y_max - y_min) + y_min
        mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
        print(f"  {name}: {mae:.2f} °C")

def main():
    """Main validation function."""
    print("Loading data and preprocessing...")
    
    # Load preprocessed data
    train_df = pd.read_csv("../train_data.csv")
    val_df = pd.read_csv("../val_data.csv")

    # Add lag features
    train_df['temp_lag1'] = train_df['temperature'].shift(1)
    train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
    val_df['temp_lag1'] = val_df['temperature'].shift(1)
    val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)

    # Compute temperature_delta only
    train_df['temperature_delta'] = rolling_slope(train_df['temperature'], window=15)
    val_df['temperature_delta'] = rolling_slope(val_df['temperature'], window=15)

    # Drop rows with NaNs
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    # Define feature and target columns
    base_features = [
        'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
        'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
        'temperature_delta', 'temp_lag1', 'humidity_lag1'
    ]
    targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

    # Load scaler parameters
    with open("input_scaler.json", "r") as f:
        input_scaler = json.load(f)
    
    with open("target_scaler.json", "r") as f:
        target_scaler = json.load(f)

    # Apply input scaling to both train and val data
    X_train_base = train_df[base_features].copy()
    X_val_base = val_df[base_features].copy()
    
    for feature in base_features:
        f_min = input_scaler[feature]["min"]
        f_max = input_scaler[feature]["max"]
        X_train_base[feature] = (X_train_base[feature] - f_min) / (f_max - f_min)
        X_val_base[feature] = (X_val_base[feature] - f_min) / (f_max - f_min)

    # Convert to MSB/LSB separate features
    X_train = convert_to_msb_lsb_separate(X_train_base.values)
    X_val = convert_to_msb_lsb_separate(X_val_base.values)

    print(f"MSB/LSB training data shape: {X_train.shape}")
    print(f"MSB/LSB validation data shape: {X_val.shape}")

    # Apply target scaling
    y_train = train_df[targets].copy()
    y_val = val_df[targets].copy()
    y_min = target_scaler["min"]
    y_max = target_scaler["max"]
    y_train = 2 * (y_train - y_min) / (y_max - y_min) - 1
    y_val = 2 * (y_val - y_min) / (y_max - y_min) - 1
    y_train = y_train.values
    y_val = y_val.values

    # Load and test the trained model
    print("\nLoading trained model...")
    model = build_model()
    
    # Find the best checkpoint
    checkpoint_files = glob.glob("./checkpoints/model_*.weights.h5")
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        best_checkpoint = checkpoint_files[-1]
        print(f"Loading weights from: {best_checkpoint}")
        model.load_weights(best_checkpoint)
        
        # Test float model
        print("\nTesting float model...")
        eval_results = model.evaluate(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]], verbose=0)
        t1_mae_c = eval_results[1] * (y_max - y_min)
        t2_mae_c = eval_results[2] * (y_max - y_min)
        t3_mae_c = eval_results[3] * (y_max - y_min)
        print(f"Float model validation MAE (in °C):")
        print(f"  t+1hr: {t1_mae_c:.2f} °C")
        print(f"  t+2hr: {t2_mae_c:.2f} °C")
        print(f"  t+3hr: {t3_mae_c:.2f} °C")
        
        # Create Coral TPU quantized model
        if quantize_model_for_coral_tpu(model, X_train):
            # Validate the Coral TPU model
            validate_quantized_model("weather_model_4_coral_tpu.tflite", X_val, y_val, y_min, y_max)
        
        # Fix and validate existing model if it exists
        if os.path.exists("weather_model_4_best.tflite"):
            print("\n" + "="*50)
            print("Fixing existing quantized model...")
            print("="*50)
            
            # Re-quantize the existing model with proper INT8 quantization
            if quantize_model_for_coral_tpu(model, X_train, "weather_model_4_best_fixed.tflite"):
                print("✅ Fixed model created: weather_model_4_best_fixed.tflite")
                validate_quantized_model("weather_model_4_best_fixed.tflite", X_val, y_val, y_min, y_max)
            else:
                print("❌ Failed to fix existing model")
    else:
        print("No checkpoint found. Cannot proceed without trained model.")

if __name__ == "__main__":
    main()
