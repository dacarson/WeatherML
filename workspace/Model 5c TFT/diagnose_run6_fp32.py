import os
import sys
import json

import numpy as np
import pandas as pd
import tensorflow as tf

# Diagnostic: does Run 6's FP32 model reproduce its own claimed offline MAE
# (0.041/0.044/0.073 C @ 1/2/3hr) when fed windowed input built the same way the new
# Inference_InfluxDB_Writer.py --run dense_b_run6 --no-tpu live script builds it?
# Live data showed 1.11C StdDev at 1hr -- ~15-20x worse than claimed -- so this checks
# whether the live inference script's feature/windowing logic has a bug, before
# concluding anything about the model itself.

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Model 5f"))
from requantize_int8 import load_and_engineer  # noqa: E402  (reuses verified cyclical/slope/target pipeline)

RUN_DIR = "results_5c_trackb_dense_b_run6"
N_EVAL = 2000


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    print("Loading + engineering train/val data (shared pipeline)...")
    train_df, val_df = load_and_engineer(data_dir)

    with open(f"{RUN_DIR}/input_scaler_5c_trackb.json") as f:
        input_scaler = json.load(f)
    with open(f"{RUN_DIR}/target_scaler_5c_trackb.json") as f:
        target_scaler = json.load(f)
    y_min, y_max = target_scaler["min"], target_scaler["max"]
    features = list(input_scaler.keys())
    n_features = len(features)
    seq_len = 180
    print(f"Features ({n_features}): {features}")

    targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]
    vdf = val_df.dropna(subset=features + targets).copy()
    print(f"val rows after dropna: {len(vdf):,}")

    Xdf = vdf[features].copy()
    for feat in features:
        lo = input_scaler[feat]["min"]
        hi = input_scaler[feat]["max"]
        Xdf[feat] = ((Xdf[feat] - lo) / (hi - lo)).clip(0.0, 1.0)
    X_flat = Xdf.values.astype(np.float32)
    y_raw = vdf[targets].values.astype(np.float32)

    interp = tf.lite.Interpreter(model_path=f"{RUN_DIR}/model_trackb_dense_b_run6_fp32.tflite")
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    print(f"Model input shape: {in_det[0]['shape']}")

    n = min(N_EVAL, len(X_flat) - seq_len)
    # Sample evenly spaced windows across the val set rather than just the first N rows,
    # so this isn't accidentally checking only one narrow time slice.
    start_indices = np.linspace(seq_len - 1, len(X_flat) - 1, n).astype(int)

    preds = [[] for _ in range(3)]
    trues = [[] for _ in range(3)]
    for idx in start_indices:
        window = X_flat[idx - seq_len + 1: idx + 1][np.newaxis, :, :]  # (1, 180, n_features)
        interp.set_tensor(in_det[0]["index"], window)
        interp.invoke()
        for j in range(3):
            out = float(np.squeeze(interp.get_tensor(out_det[j]["index"])))
            preds[j].append(out)
            trues[j].append(y_raw[idx, j])

    print(f"\nFP32 MAE reproducing live-script feature/windowing logic (n={n}):")
    for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        pred_norm = np.array(preds[j])
        pred_c = (pred_norm + 1) * 0.5 * (y_max - y_min) + y_min
        true_c = np.array(trues[j])
        mae = float(np.mean(np.abs(pred_c - true_c)))
        print(f"  {name}: {mae:.4f}C")

    print(f"\nRun 6's claimed offline MAE (from results_5c_trackb_dense_b_run6.json): "
          f"1hr=0.0409  2hr=0.0442  3hr=0.0730")


if __name__ == "__main__":
    main()
