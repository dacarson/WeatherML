import tensorflow as tf
import numpy as np

MODEL_PATH = "weather_model_5a_quant_dense_wide_run2_requant.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
tensor_by_index = {t["index"]: t for t in tensor_details}

print(f"Total tensors: {len(tensor_details)}")

# Show input / output quantization for sanity
print("\n=== Inputs ===")
for d in interpreter.get_input_details():
    print(d["index"], d["name"], d["dtype"], "q=", d["quantization"])

print("\n=== Outputs ===")
for d in interpreter.get_output_details():
    print(d["index"], d["name"], d["dtype"], "q=", d["quantization"])

# Dump ops and their tensors
print("\n=== Ops details ===")
ops = interpreter._get_ops_details()  # private API but works

for i, op in enumerate(ops):
    op_name = op.get("op_name", op.get("builtin_code", "UNKNOWN"))
    print(f"\nOp #{i}: {op_name}")
    print("  Inputs:")
    for idx in op["inputs"]:
        t = tensor_by_index[idx]
        print(f"    {idx:3d}: {t['name']}  dtype={t['dtype']} q={t['quantization']}")
    print("  Outputs:")
    for idx in op["outputs"]:
        t = tensor_by_index[idx]
        print(f"    {idx:3d}: {t['name']}  dtype={t['dtype']} q={t['quantization']}")
