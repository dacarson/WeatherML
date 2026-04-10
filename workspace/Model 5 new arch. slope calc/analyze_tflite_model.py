import tensorflow as tf

tflite_path = "weather_model_5a_quant_dense_wide_run2_requant.tflite"

with open(tflite_path, "rb") as f:
    tflite_model = f.read()

print("=== TFLite Analyzer Output ===")
tf.lite.experimental.Analyzer.analyze(
    model_content=tflite_model,
    gpu_compatibility=False
)
