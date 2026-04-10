import joblib
import json

scaler = joblib.load("scaler.joblib")
target_scaler = joblib.load("target_scaler.joblib")

def dump_scaler(scaler, filename):
    with open(filename, "w") as f:
        json.dump({
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }, f)

dump_scaler(scaler, "scaler_params.json")
dump_scaler(target_scaler, "target_scaler_params.json")
