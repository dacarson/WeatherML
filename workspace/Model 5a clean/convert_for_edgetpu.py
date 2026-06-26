import numpy as np
import tensorflow as tf
import pandas as pd
import json

SEQ_LEN = 180

# Load SavedModel from training step
model = tf.keras.models.load_model("weather_model_5a_saved_dense_wide_run1")

# Rebuild X_train_flat exactly as in train_model.py (copy/paste your feature and scaling code),
# but you only need enough to get X_train_flat and features list.

# For example:
train_df = pd.read_csv("../train_data.csv")
# ... (same feature engineering and scaling as in train_model.py) ...
# X_train_df = train_df[features].copy()
# apply same scaling -> X_train_flat = X_train_df.values

def representative_data_gen():
    num_samples = 500
    n = X_train_flat.shape[0]
    for _ in range(num_samples):
        idx = np.random.randint(SEQ_LEN, n)
        window = X_train_flat[idx-SEQ_LEN:idx, :]
        window = window[np.newaxis, ...].astype(np.float32)
        yield [window]

converter = tf.lite.TFLiteConverter.from_saved_model("weather_model_5a_saved_dense_wide_run1")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open("weather_model_5a_edge_compat.tflite", "wb") as f:
    f.write(tflite_model)
