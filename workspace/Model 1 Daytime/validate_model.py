#!/usr/bin/env python3
"""
Standalone validation script for Model 1 quantized TFLite model.
Loads existing model and validation data without retraining.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error
from scipy.stats import linregress

def load_scaler_params(scaler_file):
    """Load scaler parameters from JSON file."""
    with open(scaler_file, 'r') as f:
        return json.load(f)

def load_target_scaler(scaler_file):
    """Load target scaler parameters from JSON file."""
    with open(scaler_file, 'r') as f:
        return json.load(f)

def rolling_slope(series, window):
    """Compute rolling slope for temperature delta calculation."""
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
        raw=True
    )

def preprocess_data(df, input_scaler, features):
    """Preprocess data using the same scaling as training."""
    # Add lag features
    df['temp_lag1'] = df['temperature'].shift(1)
    df['humidity_lag1'] = df['relative_humidity'].shift(1)
    
    # Compute temperature_delta
    df['temperature_delta'] = rolling_slope(df['temperature'], window=15)
    
    # Drop rows with NaNs
    df.dropna(inplace=True)
    
    # Apply input scaling
    X = df[features].copy()
    for feature in features:
        f_min = input_scaler[feature]["min"]
        f_max = input_scaler[feature]["max"]
        X[feature] = (X[feature] - f_min) / (f_max - f_min)
    
    return X.values

def validate_quantized_model(tflite_model_path, X_val, y_val, y_min, y_max, num_samples=500):
    """Validate the quantized TFLite model."""
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
    
    # Debug: show what the quantization formula produces
    print(f"Raw quantization: min={np.min(X_val_subset / input_scale + input_zero_point):.4f}, max={np.max(X_val_subset / input_scale + input_zero_point):.4f}")
    
    # Try different quantization approaches
    print("\nTrying different quantization approaches:")
    
    # Approach 1: Standard quantization
    input_quantized_1 = np.round(X_val_subset / input_scale + input_zero_point).astype(np.int8)
    input_quantized_1 = np.clip(input_quantized_1, -128, 127)
    print(f"Approach 1 (standard): min={input_quantized_1.min()}, max={input_quantized_1.max()}")
    
    # Approach 2: Try without zero_point offset
    input_quantized_2 = np.round(X_val_subset / input_scale).astype(np.int8)
    input_quantized_2 = np.clip(input_quantized_2, -128, 127)
    print(f"Approach 2 (no zero_point): min={input_quantized_2.min()}, max={input_quantized_2.max()}")
    
    # Approach 3: Scale to full INT8 range (-128 to +127)
    # Map [0, 1] to [-128, 127] for maximum range
    input_quantized_3 = np.round(X_val_subset * 255 - 128).astype(np.int8)
    input_quantized_3 = np.clip(input_quantized_3, -128, 127)
    print(f"Approach 3 (full range -128 to +127): min={input_quantized_3.min()}, max={input_quantized_3.max()}")
    
    # Approach 4: Use the model's quantization parameters but adjust input range
    # The model expects inputs where 0 maps to -128, so we need to shift our inputs
    # If our inputs are [0, 0.8377], we need to map them to the expected range
    input_quantized_4 = np.round(X_val_subset / input_scale + input_zero_point).astype(np.int8)
    input_quantized_4 = np.clip(input_quantized_4, -128, 127)
    print(f"Approach 4 (model's params): min={input_quantized_4.min()}, max={input_quantized_4.max()}")
    
    # Approach 5: Try mapping [0, 1] to [0, 255] then subtracting 128
    input_quantized_5 = np.round(X_val_subset * 255).astype(np.int8) - 128
    input_quantized_5 = np.clip(input_quantized_5, -128, 127)
    print(f"Approach 5 (0-255 then -128): min={input_quantized_5.min()}, max={input_quantized_5.max()}")
    
    # Let's try to figure out what input range the model expects
    # Test with different input ranges to see which one gives us a proper quantized range
    print("\nTesting different input ranges:")
    
    # Test 1: [0, 1] range
    test_input_1 = np.linspace(0, 1, 10).reshape(-1, 1)
    test_quant_1 = np.round(test_input_1 / input_scale + input_zero_point).astype(np.int8)
    print(f"Test [0,1]: quantized range {test_quant_1.min()} to {test_quant_1.max()}")
    
    # Test 2: [0, 2] range  
    test_input_2 = np.linspace(0, 2, 10).reshape(-1, 1)
    test_quant_2 = np.round(test_input_2 / input_scale + input_zero_point).astype(np.int8)
    print(f"Test [0,2]: quantized range {test_quant_2.min()} to {test_quant_2.max()}")
    
    # Test 3: [0, 10] range
    test_input_3 = np.linspace(0, 10, 10).reshape(-1, 1)
    test_quant_3 = np.round(test_input_3 / input_scale + input_zero_point).astype(np.int8)
    print(f"Test [0,10]: quantized range {test_quant_3.min()} to {test_quant_3.max()}")
    
    # Test 4: [0, 100] range
    test_input_4 = np.linspace(0, 100, 10).reshape(-1, 1)
    test_quant_4 = np.round(test_input_4 / input_scale + input_zero_point).astype(np.int8)
    print(f"Test [0,100]: quantized range {test_quant_4.min()} to {test_quant_4.max()}")
    
    # Use approach 3 for now since it gives us some variation
    input_quantized = input_quantized_3
    print(f"Using approach 3 - Final quantized input range: min={input_quantized.min()}, max={input_quantized.max()}")

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
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        print(f"  {name}: {np.array(y_preds_dequant[j][:5]).round(3)}")

    print("\nValidation MAE (in °C):")
    print("\n🔍 Clipping Analysis:")
    
    for j, name in enumerate(['t+1hr', 't+2hr', 't+3hr']):
        y_preds_rescaled = 0.5 * (np.array(y_preds_dequant[j]) + 1) * (y_max - y_min) + y_min
        y_val_rescaled = 0.5 * (y_val_subset[:, j] + 1) * (y_max - y_min) + y_min
        mae = mean_absolute_error(y_val_rescaled, y_preds_rescaled)
        
        # Clipping detection
        pred_min, pred_max = y_preds_rescaled.min(), y_preds_rescaled.max()
        val_min, val_max = y_val_rescaled.min(), y_val_rescaled.max()
        
        # Check for potential clipping
        clipping_threshold = 0.1  # 0.1°C tolerance
        upper_clipping = (pred_max >= y_max - clipping_threshold)
        lower_clipping = (pred_min <= y_min + clipping_threshold)
        
        # Check if predictions are systematically lower than actuals at high temperatures
        high_temp_mask = y_val_rescaled > np.percentile(y_val_rescaled, 90)  # Top 10% of actual temps
        if np.sum(high_temp_mask) > 0:
            high_temp_bias = np.mean(y_preds_rescaled[high_temp_mask] - y_val_rescaled[high_temp_mask])
        else:
            high_temp_bias = 0
            
        print(f"  {name}: {mae:.2f} °C")
        print(f"    📊 Range - Predicted: {pred_min:.2f} to {pred_max:.2f}°C, Actual: {val_min:.2f} to {val_max:.2f}°C")
        print(f"    🎯 Training range: {y_min:.2f} to {y_max:.2f}°C")
        
        if upper_clipping:
            print(f"    ⚠️  UPPER CLIPPING DETECTED! Predictions hitting training max ({y_max:.2f}°C)")
        if lower_clipping:
            print(f"    ⚠️  LOWER CLIPPING DETECTED! Predictions hitting training min ({y_min:.2f}°C)")
        if high_temp_bias < -1.0:
            print(f"    ⚠️  HIGH TEMP BIAS: Model predicts {abs(high_temp_bias):.2f}°C lower than actual at high temperatures")
        
        # Show distribution of predictions vs actuals
        print(f"    📈 High temp bias: {high_temp_bias:.2f}°C (negative = model predicts lower)")
        
        # Additional analysis for clipping patterns
        if pred_max < val_max - 2.0:  # If predictions are significantly lower than actual max
            print(f"    🔍 CONCERN: Predictions max ({pred_max:.2f}°C) is {val_max - pred_max:.2f}°C lower than actual max")
        
        # Check for systematic underestimation at high temperatures
        high_temp_indices = y_val_rescaled > np.percentile(y_val_rescaled, 95)  # Top 5% of temps
        if np.sum(high_temp_indices) > 10:  # If we have enough high temp samples
            avg_underestimation = np.mean(y_val_rescaled[high_temp_indices] - y_preds_rescaled[high_temp_indices])
            if avg_underestimation > 2.0:
                print(f"    🚨 SIGNIFICANT UNDERESTIMATION: Model predicts {avg_underestimation:.2f}°C lower on hottest days")
        
        print()

def main():
    """Main validation function."""
    print("Loading validation data and scalers...")
    
    # Load validation data
    val_df = pd.read_csv("../val_data_ps.csv")
    print(f"Loaded validation data: {val_df.shape}")
    
    # Load scalers
    input_scaler = load_scaler_params("input_scaler.json")
    target_scaler = load_target_scaler("target_scaler.json")
    
    # Define features (same as training)
    features = [
        'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
        'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
        'temperature_delta', 'temp_lag1', 'humidity_lag1'
    ]
    
    # Define targets
    targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']
    
    # Preprocess validation data
    X_val = preprocess_data(val_df, input_scaler, features)
    
    # Get target values and apply same scaling as training
    y_min = target_scaler["min"]
    y_max = target_scaler["max"]
    y_val = val_df[targets].values
    y_val_scaled = 2 * (y_val - y_min) / (y_max - y_min) - 1
    
    print(f"Preprocessed validation data: X_val shape={X_val.shape}, y_val shape={y_val_scaled.shape}")
    
    # Validate the quantized model
    validate_quantized_model("weather_model_1_best.tflite", X_val, y_val_scaled, y_min, y_max)

if __name__ == "__main__":
    main()
