import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="weather_model_conv1d_quant_conv1d_edgetpu.tflite")
interpreter.allocate_tensors()
print(interpreter.get_input_details())
