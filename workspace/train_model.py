import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import joblib

# Load preprocessed data
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

# Add lag features
train_df['temp_lag1'] = train_df['temperature'].shift(1)
train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
val_df['temp_lag1'] = val_df['temperature'].shift(1)
val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)

# Compute rate-of-change features
for col in ['temperature', 'relative_humidity', 'station_pressure', 'solar_radiation', 'illuminance']:
    train_df[f'{col}_delta'] = train_df[col].diff()
    val_df[f'{col}_delta'] = val_df[col].diff()

# Drop rows with NaNs introduced by diff()
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

# Define feature and target columns
features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg',
    'wind_gust', 'day_of_year', 'time_of_day',
    'temperature_delta', 'solar_radiation_delta', 'illuminance_delta',
    'temp_lag1', 'humidity_lag1'
]

targets = ['temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr']

# Normalize inputs
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_val = scaler.transform(val_df[features])

# Save the scaler to re-use at inference time
joblib.dump(scaler, "scaler.joblib")

y_train = train_df[targets].values
y_val = val_df[targets].values

# Wide & Deep model
input_layer = tf.keras.layers.Input(shape=(len(features),))
wide = tf.keras.layers.Dense(16)(input_layer)
deep = tf.keras.layers.Dense(128, activation='relu')(input_layer)
deep = tf.keras.layers.BatchNormalization()(deep)
deep = tf.keras.layers.Dropout(0.3)(deep)

res = tf.keras.layers.Dense(64, activation='relu')(deep)
res = tf.keras.layers.BatchNormalization()(res)
shortcut = tf.keras.layers.Dense(64)(deep)
res = tf.keras.layers.Add()([shortcut, res])  # residual connection with projection

res = tf.keras.layers.Dense(32, activation='relu')(res)
res = tf.keras.layers.BatchNormalization()(res)

merged = tf.keras.layers.Concatenate()([wide, res])
output_layer = tf.keras.layers.Dense(3)(merged)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, lr_schedule]
)

# Permutation Feature Importance
from sklearn.metrics import mean_squared_error
import copy

# Calculate baseline validation loss
baseline_loss = model.evaluate(X_val, y_val, verbose=0)[0]

feature_importance = {}
for i, feature in enumerate(features):
    X_val_permuted = copy.deepcopy(X_val)
    np.random.shuffle(X_val_permuted[:, i])
    permuted_loss = model.evaluate(X_val_permuted, y_val, verbose=0)[0]
    feature_importance[feature] = permuted_loss - baseline_loss

sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("\nPermutation Feature Importance (by increase in val_loss):")
for feature, importance in sorted_importance:
    print(f"{feature}: {importance:.4f}")

# Save the model in .keras format for reference
# model.save("weather_model_1", save_format="tf")

# Convert directly from model object
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# You must provide a representative dataset function
def representative_data_gen():
    for i in range(0, len(X_train), 1):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_tflite_model = converter.convert()


with open("weather_model_1_quant.tflite", "wb") as f:
    f.write(quantized_tflite_model)

# Compile the quantized model for the Edge TPU
import subprocess

result = subprocess.run(
    ["edgetpu_compiler", "weather_model_1_quant.tflite"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
print(result.stdout)
print(result.stderr)
