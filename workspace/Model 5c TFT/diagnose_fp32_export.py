import os
import sys
import json
import argparse

import numpy as np
import tensorflow as tf

# Diagnostic: does <run>'s FP32 .tflite export reproduce its own claimed offline MAE when fed
# windowed input built the same way Inference_InfluxDB_Writer.py builds it? Run this BEFORE
# deploying any run's FP32 export live, given Run 6's export was found not to reproduce its own
# claimed number (see MODEL_5C_TRACK_B_EXPERIMENT_LOG.md's "Run 6 FP32 export doesn't reproduce
# its own number" entry) — a suspected weight-transfer bug specific to branching architectures
# (bottleneck+wide+deep+merge), which Run 11/14 also use, so it's worth checking every time before
# trusting a new run's FP32 artifact rather than assuming it's fine.

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Model 5f"))
from requantize_int8 import load_and_engineer  # noqa: E402  (reuses verified cyclical/slope/target pipeline)

N_EVAL = 2000

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="e.g. dense_b_run11")
args = parser.parse_args()

RUN_NAME = args.run
RUN_DIR = f"results_5c_trackb_{RUN_NAME}"


def main():
    with open(f"{RUN_DIR}/results_5c_trackb_{RUN_NAME}.json") as f:
        results = json.load(f)
    claimed = (results["diff_1hr_mae_c"], results["diff_2hr_mae_c"], results["diff_3hr_mae_c"])

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
    print(f"Features ({n_features}): {features}")

    fp32_path = f"{RUN_DIR}/model_trackb_{RUN_NAME}_fp32.tflite"
    interp = tf.lite.Interpreter(model_path=fp32_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_shape = in_det[0]["shape"]
    seq_len = int(in_shape[1]) if len(in_shape) == 3 else 1
    print(f"Model input shape: {in_shape} (seq_len={seq_len})")

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

    n = min(N_EVAL, len(X_flat) - seq_len)
    start_indices = np.linspace(seq_len - 1, len(X_flat) - 1, n).astype(int)

    preds = [[] for _ in range(3)]
    trues = [[] for _ in range(3)]
    for idx in start_indices:
        if seq_len == 1:
            window = X_flat[idx][np.newaxis, :]
        else:
            window = X_flat[idx - seq_len + 1: idx + 1][np.newaxis, :, :]
        interp.set_tensor(in_det[0]["index"], window)
        interp.invoke()
        for j in range(3):
            out = float(np.squeeze(interp.get_tensor(out_det[j]["index"])))
            preds[j].append(out)
            trues[j].append(y_raw[idx, j])

    print(f"\nFP32 MAE reproducing live-script feature/windowing logic (n={n}):")
    measured = []
    for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        pred_norm = np.array(preds[j])
        pred_c = (pred_norm + 1) * 0.5 * (y_max - y_min) + y_min
        true_c = np.array(trues[j])
        mae = float(np.mean(np.abs(pred_c - true_c)))
        measured.append(mae)
        print(f"  {name}: {mae:.4f}C")

    print(f"\n{RUN_NAME}'s claimed offline MAE: "
          f"1hr={claimed[0]:.4f}  2hr={claimed[1]:.4f}  3hr={claimed[2]:.4f}")

    ratios = [m / c if c > 0 else float("inf") for m, c in zip(measured, claimed)]
    if max(ratios) > 2.0:
        print(f"\n⚠️  MISMATCH: measured MAE is {max(ratios):.1f}x the claimed number at worst — "
              f"this FP32 export likely does NOT reflect the claimed accuracy. Do not deploy it "
              f"live as a trustworthy reference without investigating further.")
    else:
        print(f"\n✅ Measured MAE is within 2x of claimed at every horizon — export looks "
              f"consistent with the claimed offline number.")


if __name__ == "__main__":
    main()
