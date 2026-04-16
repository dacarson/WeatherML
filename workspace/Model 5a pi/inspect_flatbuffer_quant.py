import sys
import flatbuffers
from tensorflow.lite.python.schema_py_generated import Model

if len(sys.argv) < 2:
    print("Usage: python inspect_flatbuffer_quant.py model.tflite")
    sys.exit(1)

fname = sys.argv[1]

with open(fname, "rb") as f:
    buf = f.read()

model = Model.GetRootAsModel(buf, 0)

print("=== MODEL-LEVEL FIELDS ===")
print("Model version:", model.Version())
print("Subgraphs:", model.SubgraphsLength())
print("Buffers:", model.BuffersLength())

# ------------------------------------------------------------------
# 1. Try printing model-level inference_type (if present)
# ------------------------------------------------------------------
try:
    inference_type = model.OperatorCodes(0).DeprecatedBuiltinCode()
    print("OperatorCodes[0].DeprecatedBuiltinCode:", inference_type)
except:
    print("Could not read DeprecatedBuiltinCode")

# ------------------------------------------------------------------
# 2. Check all input & output tensor dtypes
# ------------------------------------------------------------------

def tensor_type_name(t):
    # TensorType enum values from schema.fbs
    mapping = {
        0: "FLOAT32",
        1: "FLOAT16",
        2: "INT32",
        3: "UINT8",
        4: "INT64",
        5: "STRING",
        6: "BOOL",
        7: "INT16",
        8: "COMPLEX64",
        9: "INT8",
        10: "FLOAT64",
        11: "COMPLEX128",
        12: "UINT64",
        13: "RESOURCE",
        14: "VARIANT",
        15: "UINT32",
        16: "UINT16",
    }
    return mapping.get(t, f"UNKNOWN({t})")

sub = model.Subgraphs(0)

print("\n=== INPUT TENSORS ===")
for i in range(sub.InputsLength()):
    tid = sub.Inputs(i)
    t = sub.Tensors(tid)
    print(f"Input {i}: tensor #{tid}, dtype={tensor_type_name(t.Type())}")

print("\n=== OUTPUT TENSORS ===")
for i in range(sub.OutputsLength()):
    tid = sub.Outputs(i)
    t = sub.Tensors(tid)
    print(f"Output {i}: tensor #{tid}, dtype={tensor_type_name(t.Type())}")

# ------------------------------------------------------------------
# 3. Check if ANY FLOAT tensors exist (strong signal of not fully quantized)
# ------------------------------------------------------------------
print("\n=== SCAN FOR FLOAT TENSORS ===")
float_tensors = []
for i in range(sub.TensorsLength()):
    t = sub.Tensors(i)
    if t.Type() == 0:  # FLOAT32
        float_tensors.append(i)

if len(float_tensors) == 0:
    print("No FLOAT tensors found → graph is integer-only ✔")
else:
    print("Float tensors found:", float_tensors)

# ------------------------------------------------------------------
# 4. Try detecting full integer quantization mode
# ------------------------------------------------------------------
print("\n=== FULL INTEGER QUANTIZATION CHECK ===")
full_int = True

# (A) Check input type is int8
for i in range(sub.InputsLength()):
    t = sub.Tensors(sub.Inputs(i))
    if t.Type() != 9:
        full_int = False
        print("❌ Input not INT8:", tensor_type_name(t.Type()))

# (B) Check output type is int8
for i in range(sub.OutputsLength()):
    t = sub.Tensors(sub.Outputs(i))
    if t.Type() != 9:
        full_int = False
        print("❌ Output not INT8:", tensor_type_name(t.Type()))

# (C) Check EVERY tensor has quantization parameters
for i in range(sub.TensorsLength()):
    t = sub.Tensors(i)
    q = t.Quantization()
    if q is None or q.ScaleLength() == 0:
        # Allow int32 bias tensors
        if t.Type() != 2:
            full_int = False
            print(f"❌ Tensor {i} missing quant params, dtype={tensor_type_name(t.Type())}")

if full_int:
    print("✔ Model appears to be FULLY INTEGER QUANTIZED (int8 graph + quant params)")
else:
    print("❌ Model is NOT fully integer quantized")
