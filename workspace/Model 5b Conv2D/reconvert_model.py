#!/usr/bin/env python3
"""
Reconvert existing trained model to TFLite with fixed shapes for Edge TPU compatibility.

This script loads the best trained model and converts it to TFLite with:
- Fixed batch size (1) for static tensor shapes (Edge TPU requirement)
- Full integer quantization (int8)
- Same quantization settings as training script
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import glob

# Constants matching training script
SEQ_LEN = 180  # 3 hours of history
SHORT_SEQ_LEN = 60

# Custom layer definition (must match training script)
@tf.keras.utils.register_keras_serializable(package="WeatherML")
class ExtractLagLayer(tf.keras.layers.Layer):
    """Extract exact feature values at a specific lag position."""
    def __init__(self, lag_steps, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.lag_steps = lag_steps
        self.seq_len = seq_len
    
    def call(self, inputs):
        idx = self.seq_len - 1 - self.lag_steps
        return tf.gather(inputs, idx, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'lag_steps': self.lag_steps,
            'seq_len': self.seq_len
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SliceTimestep(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
    def call(self, x):
        return x[:, self.index, :]
    def get_config(self):
        return {**super().get_config(), "index": self.index}

class SliceFeatures(tf.keras.layers.Layer):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end
    def call(self, x):
        return x[:, :, self.start:self.end]
    def get_config(self):
        return {**super().get_config(), "start": self.start, "end": self.end}

# Custom objects for loading the model
CUSTOM_OBJECTS = {
    'ExtractLagLayer': ExtractLagLayer,
    'SliceTimestep': SliceTimestep,
    'SliceFeatures': SliceFeatures,
}

def load_training_data_for_quantization():
    """Load training data for representative dataset."""
    print("Loading training data for quantization calibration...")
    train_df = pd.read_csv("../train_data.csv")
    
    # Load scaler to get exact feature order used during training
    scaler_file = "input_scaler_5b.json"
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    
    with open(scaler_file, 'r') as f:
        scaler_params = json.load(f)
    
    # Get feature order from scaler (this is the exact order used during training)
    features = list(scaler_params.keys())
    print(f"   Features from scaler: {len(features)} features")
    
    # Compute derived features that might not be in the CSV
    # These match the preprocessing in the training script
    
    # temp_delta_1: first-order difference
    if 'temp_delta_1' not in train_df.columns and 'temperature' in train_df.columns:
        train_df['temp_delta_1'] = train_df['temperature'].diff().fillna(0.0)
    
    # time_of_day cyclical encoding
    if 'time_of_day' in train_df.columns:
        if 'time_of_day_sin' not in train_df.columns:
            train_df['time_of_day_sin'] = np.sin(2 * np.pi * train_df['time_of_day'] / 24.0)
        if 'time_of_day_cos' not in train_df.columns:
            train_df['time_of_day_cos'] = np.cos(2 * np.pi * train_df['time_of_day'] / 24.0)
        if 'time_of_day_sin2' not in train_df.columns:
            train_df['time_of_day_sin2'] = np.sin(4 * np.pi * train_df['time_of_day'] / 24.0)
        if 'time_of_day_cos2' not in train_df.columns:
            train_df['time_of_day_cos2'] = np.cos(4 * np.pi * train_df['time_of_day'] / 24.0)
    
    # day_of_year cyclical encoding
    if 'day_of_year' in train_df.columns:
        if 'day_of_year_sin' not in train_df.columns:
            train_df['day_of_year_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365.25)
        if 'day_of_year_cos' not in train_df.columns:
            train_df['day_of_year_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365.25)
    
    # wind_direction cyclical encoding
    if 'wind_direction' in train_df.columns:
        if 'wind_direction_sin' not in train_df.columns:
            train_df['wind_direction_sin'] = np.sin(2 * np.pi * train_df['wind_direction'] / 360.0)
        if 'wind_direction_cos' not in train_df.columns:
            train_df['wind_direction_cos'] = np.cos(2 * np.pi * train_df['wind_direction'] / 360.0)
    
    # Check which features are available after computation
    available_features = [f for f in features if f in train_df.columns]
    missing_features = [f for f in features if f not in train_df.columns]
    
    if missing_features:
        print(f"   ⚠️  Warning: {len(missing_features)} features not found after computation: {missing_features}")
        print(f"   Using {len(available_features)} available features")
        features = available_features
    else:
        print(f"   ✅ All {len(features)} features available")
    
    # Extract features in the correct order
    X_train = train_df[features].values.astype(np.float32)
    
    # Apply scaling (matching training script: scale to [0,1] then clip)
    for i, feat in enumerate(features):
        if feat in scaler_params:
            min_val = scaler_params[feat]['min']
            max_val = scaler_params[feat]['max']
            denom = max_val - min_val
            if denom == 0.0:
                X_train[:, i] = 0.0
            else:
                # Scale to [0,1] (matching training script)
                scaled = (X_train[:, i] - min_val) / denom
                X_train[:, i] = np.clip(scaled, 0.0, 1.0)
    
    return X_train, features

def main():
    print("="*80)
    print("RECONVERTING MODEL TO TFLITE WITH FIXED SHAPES FOR EDGE TPU")
    print("="*80)
    
    # Find the best model checkpoint
    checkpoint_dir = "./checkpoints"
    # Specify which checkpoint to convert. Change this to use a different epoch.
    target_checkpoint = os.environ.get("CHECKPOINT", "model_epoch_60.h5")
    best_model_path = os.path.join(checkpoint_dir, target_checkpoint)

    if not os.path.exists(best_model_path):
        print(f"❌ Error: Checkpoint not found at {best_model_path}")
        print("Available checkpoints:")
        for f in sorted(glob.glob(os.path.join(checkpoint_dir, "*.h5"))):
            print(f"  - {os.path.basename(f)}")
        sys.exit(1)
    
    print(f"✅ Found best model: {best_model_path}")
    
    # Load the model
    print("\nLoading model...")
    model = tf.keras.models.load_model(
        best_model_path,
        compile=False,
        custom_objects=CUSTOM_OBJECTS
    )
    
    # Get input shape from model
    input_shape = model.input_shape
    if len(input_shape) == 3:  # (batch, seq_len, features)
        _, seq_len, n_features = input_shape
        print(f"✅ Model input shape: {input_shape}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Features: {n_features}")
    else:
        print(f"❌ Unexpected input shape: {input_shape}")
        sys.exit(1)
    
    # Load training data for representative dataset
    X_train_flat, features = load_training_data_for_quantization()
    print(f"✅ Loaded training data: {X_train_flat.shape}")
    
    # Build a fresh float32 model and copy weights from the loaded (possibly mixed-precision)
    # checkpoint. This is required because mixed_float16 embeds Cast(f16) ops inside every
    # layer — converting that model directly to TFLite leaves all internal ops as f16, which
    # the TFLite converter rejects. A fresh float32 model has no Cast ops.
    print("\n🔧 Building float32 export model for TFLite conversion...")
    tf.keras.mixed_precision.set_global_policy('float32')

    def build_export_model():
        # Exp 24 architecture: equal 64-filter branches, no MSB blocks, smaller dense head
        inp = tf.keras.layers.Input(shape=(seq_len, n_features), name="input")
        MAIN_F = 64
        TEMP_F = 64

        m = tf.keras.layers.Conv1D(MAIN_F, 3, padding='causal', use_bias=False, name="main_conv_init")(inp)
        m = tf.keras.layers.BatchNormalization(name="main_bn_init")(m)
        m = tf.keras.layers.ReLU(name="main_relu_init")(m)
        for d in [1, 2, 4, 8, 16, 32, 64]:
            r = m
            m = tf.keras.layers.Conv1D(MAIN_F, 3, padding='causal', dilation_rate=d, use_bias=False, name=f"main_conv_d{d}")(m)
            m = tf.keras.layers.BatchNormalization(name=f"main_bn_d{d}")(m)
            m = tf.keras.layers.ReLU(name=f"main_relu_d{d}")(m)
            m = tf.keras.layers.Add(name=f"main_add_d{d}")([r, m])

        temp_input = SliceFeatures(0, 2, name="temp_branch_slice")(inp)
        t = tf.keras.layers.Conv1D(TEMP_F, 3, padding='causal', use_bias=False, name="temp_conv_init")(temp_input)
        t = tf.keras.layers.BatchNormalization(name="temp_bn_init")(t)
        t = tf.keras.layers.ReLU(name="temp_relu_init")(t)
        for d in [1, 2, 4, 8, 16, 32, 64]:
            r = t
            t = tf.keras.layers.Conv1D(TEMP_F, 3, padding='causal', dilation_rate=d, use_bias=False, name=f"temp_conv_d{d}")(t)
            t = tf.keras.layers.BatchNormalization(name=f"temp_bn_d{d}")(t)
            t = tf.keras.layers.ReLU(name=f"temp_relu_d{d}")(t)
            t = tf.keras.layers.Add(name=f"temp_add_d{d}")([r, t])

        x = tf.keras.layers.Concatenate(name="temporal_concat")([
            SliceTimestep(-1,   name="main_extract_t0")(m),
            SliceTimestep(-31,  name="main_extract_t30")(m),
            SliceTimestep(-61,  name="main_extract_t60")(m),
            SliceTimestep(-121, name="main_extract_t120")(m),
            SliceTimestep(-1,   name="temp_extract_t0")(t),
            SliceTimestep(-31,  name="temp_extract_t30")(t),
            SliceTimestep(-61,  name="temp_extract_t60")(t),
            SliceTimestep(-121, name="temp_extract_t120")(t),
        ])
        x = tf.keras.layers.Dense(128, use_bias=False, name="dense1")(x)
        x = tf.keras.layers.ReLU(name="relu_dense1")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
        x = tf.keras.layers.Dense(64, use_bias=False, name="dense2")(x)
        x = tf.keras.layers.ReLU(name="relu_dense2")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_dense2")(x)
        o1 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype='float32', name='diff_1hr')(x)
        o2 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype='float32', name='diff_2hr')(x)
        o3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False, dtype='float32', name='diff_3hr')(x)
        return tf.keras.Model(inputs=inp, outputs=[o1, o2, o3])

    export_model = build_export_model()
    export_model(tf.zeros((1, seq_len, n_features), dtype=tf.float32), training=False)
    export_model.set_weights(model.get_weights())
    print("✅ Float32 export model built and weights copied")

    # Create concrete function with fixed input signature (batch_size=1)
    # This ensures all tensor shapes are static, which Edge TPU requires
    print("Creating concrete function with fixed batch size...")
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, seq_len, n_features], dtype=tf.float32)])
    def model_inference(x):
        return export_model(x, training=False)

    concrete_func = model_inference.get_concrete_function()
    
    # Create TFLite converter
    print("Setting up TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Disable experimental features that might introduce dynamic shapes
    converter._experimental_lower_tensor_list_ops = False
    try:
        converter.experimental_new_converter = False
    except AttributeError:
        pass  # Newer TF versions might not have this attribute
    
    # Full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization calibration
    def representative_data_gen():
        num_samples = 500
        n = X_train_flat.shape[0]
        
        for _ in range(num_samples):
            idx = np.random.randint(seq_len, n)
            window = X_train_flat[idx - seq_len:idx, :]  # shape (seq_len, n_features)
            window = window[np.newaxis, ...].astype(np.float32)
            yield [window]
    
    converter.representative_dataset = representative_data_gen
    
    # Require full integer quantization for all ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Edge TPU requires int8 for best compatibility
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert the model
    print("\nConverting model to TFLite (this may take a few minutes)...")
    try:
        quantized_tflite_model = converter.convert()
        print("✅ Conversion successful!")
    except (RuntimeError, ValueError) as e:
        error_str = str(e).lower()
        if "dynamic" in error_str or "READ_VARIABLE" in str(e):
            print("⚠️  Direct conversion failed. Trying SavedModel approach...")
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                saved_model_path = os.path.join(tmpdir, "saved_model")
                
                # Export as SavedModel with concrete function
                @tf.function(input_signature=[tf.TensorSpec(shape=[1, seq_len, n_features], dtype=tf.float32)])
                def serving_fn(x):
                    return model(x, training=False)
                
                tf.saved_model.save(
                    model,
                    saved_model_path,
                    signatures={'serving_default': serving_fn.get_concrete_function()}
                )
                
                # Convert from SavedModel
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_data_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                converter._experimental_lower_tensor_list_ops = False
                try:
                    converter.experimental_new_converter = False
                except AttributeError:
                    pass
                
                quantized_tflite_model = converter.convert()
                print("✅ Conversion successful via SavedModel!")
        else:
            print(f"❌ Conversion failed: {e}")
            raise
    
    # Save the model
    output_file = "weather_model_5b_best.tflite"
    print(f"\nSaving quantized model to {output_file}...")
    with open(output_file, "wb") as f:
        f.write(quantized_tflite_model)
    
    model_size_kb = os.path.getsize(output_file) / 1024
    print(f"✅ Model saved successfully!")
    print(f"   File: {output_file}")
    print(f"   Size: {model_size_kb:.2f} KB")
    
    # Verify the model can be loaded
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_file)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Model verified!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output shapes: {[d['shape'] for d in output_details]}")
    print(f"   Output dtypes: {[d['dtype'] for d in output_details]}")
    
    print("\n" + "="*80)
    print("✅ RECONVERSION COMPLETE!")
    print("="*80)
    print(f"\nThe model is now ready for Edge TPU compilation:")
    print(f"  ../../edgetpu-x86-compiler.sh {output_file}")

if __name__ == "__main__":
    main()
