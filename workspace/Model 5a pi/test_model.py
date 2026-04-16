import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="weather_model_5a_quant_dense_wide_run2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input dtype:", input_details[0]["dtype"])
print("Input quantization:", input_details[0]["quantization"])
for i, d in enumerate(output_details):
    print(f"Output {i} dtype:", d["dtype"])
    print(f"Output {i} quantization:", d["quantization"])
