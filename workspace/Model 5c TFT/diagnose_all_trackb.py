import os
import sys
import json

import numpy as np
import tensorflow as tf

# Verifies every Track B run's FP32 .tflite export against its own claimed offline MAE, in one
# process (load_and_engineer is expensive — ~1.5M rows — so it must run once, not once per run).
# See MODEL_5C_TRACK_B_EXPERIMENT_LOG.md "This is systemic, not Run-6-specific" for why this
# matters: Run 6 and Run 11 (different feature sets, same branching architecture family) both
# failed to reproduce their own claimed numbers.

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Model 5f"))
from requantize_int8 import load_and_engineer  # noqa: E402

N_EVAL = 2000
RUNS = ["run1", "run2", "run3", "run4", "run5", "run6", "run7", "run8", "run9", "run10",
        "run11", "run12", "run13", "run14", "run16", "run17", "run18", "run19", "run20",
        "run21", "run22"]


def diagnose_one(run, val_df):
    run_name = f"dense_b_{run}"
    run_dir = f"results_5c_trackb_{run_name}"
    results_path = f"{run_dir}/results_5c_trackb_{run_name}.json"
    if not os.path.exists(results_path):
        return {"run": run, "status": "no results.json"}

    with open(results_path) as f:
        results = json.load(f)
    claimed = (results.get("diff_1hr_mae_c"), results.get("diff_2hr_mae_c"),
               results.get("diff_3hr_mae_c"))
    if any(c is None for c in claimed):
        return {"run": run, "status": "no claimed MAE in results.json"}

    input_scaler_path = f"{run_dir}/input_scaler_5c_trackb.json"
    target_scaler_path = f"{run_dir}/target_scaler_5c_trackb.json"
    fp32_path = f"{run_dir}/model_trackb_{run_name}_fp32.tflite"
    if not (os.path.exists(input_scaler_path) and os.path.exists(target_scaler_path)
            and os.path.exists(fp32_path)):
        return {"run": run, "status": "missing scaler/model file"}

    with open(input_scaler_path) as f:
        input_scaler = json.load(f)
    with open(target_scaler_path) as f:
        target_scaler = json.load(f)
    y_min, y_max = target_scaler["min"], target_scaler["max"]
    features = list(input_scaler.keys())

    try:
        interp = tf.lite.Interpreter(model_path=fp32_path)
        interp.allocate_tensors()
    except Exception as e:
        return {"run": run, "status": f"failed to load model: {e}"}
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_shape = in_det[0]["shape"]
    in_rank = len(in_shape)
    seq_len = int(in_shape[1]) if in_rank == 3 else 1

    targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]
    missing = [f for f in features if f not in val_df.columns]
    if missing:
        return {"run": run, "status": f"missing feature columns: {missing}"}
    vdf = val_df.dropna(subset=features + targets).copy()
    if len(vdf) <= seq_len:
        return {"run": run, "status": "not enough val rows"}

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
        if in_rank == 3:
            window = X_flat[idx - seq_len + 1: idx + 1][np.newaxis, :, :]  # (1, seq_len, n_feat)
        else:
            window = X_flat[idx][np.newaxis, :]  # (1, n_feat)
        try:
            interp.set_tensor(in_det[0]["index"], window)
            interp.invoke()
        except Exception as e:
            return {"run": run, "status": f"inference error: {e}"}
        for j in range(3):
            out = float(np.squeeze(interp.get_tensor(out_det[j]["index"])))
            preds[j].append(out)
            trues[j].append(y_raw[idx, j])

    measured = []
    for j in range(3):
        pred_norm = np.array(preds[j])
        pred_c = (pred_norm + 1) * 0.5 * (y_max - y_min) + y_min
        true_c = np.array(trues[j])
        measured.append(float(np.mean(np.abs(pred_c - true_c))))

    ratios = [m / c if c and c > 0 else float("inf") for m, c in zip(measured, claimed)]
    return {
        "run": run, "status": "ok", "in_rank": in_rank, "seq_len": seq_len,
        "n_features": len(features), "measured": measured, "claimed": claimed,
        "max_ratio": max(ratios),
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    print("Loading + engineering train/val data (once, shared across all runs)...")
    train_df, val_df = load_and_engineer(data_dir)
    print(f"val rows: {len(val_df):,}\n")

    results = []
    for run in RUNS:
        print(f"=== {run} ===", flush=True)
        r = diagnose_one(run, val_df)
        results.append(r)
        if r["status"] != "ok":
            print(f"  SKIPPED: {r['status']}\n")
            continue
        m1, m2, m3 = r["measured"]
        c1, c2, c3 = r["claimed"]
        print(f"  rank={r['in_rank']} seq_len={r['seq_len']} n_features={r['n_features']}")
        print(f"  measured: {m1:.4f}/{m2:.4f}/{m3:.4f}C   claimed: {c1:.4f}/{c2:.4f}/{c3:.4f}C"
              f"   max_ratio={r['max_ratio']:.1f}x")
        flag = "⚠️  MISMATCH" if r["max_ratio"] > 2.0 else "✅ OK"
        print(f"  {flag}\n")

    print("\n=== SUMMARY ===")
    print(f"{'run':10s} {'status':10s} {'ratio':>8s}  {'measured (1/2/3hr)':30s} claimed (1/2/3hr)")
    for r in results:
        if r["status"] != "ok":
            print(f"{r['run']:10s} {r['status']}")
            continue
        m1, m2, m3 = r["measured"]
        c1, c2, c3 = r["claimed"]
        flag = "MISMATCH" if r["max_ratio"] > 2.0 else "ok"
        print(f"{r['run']:10s} {flag:10s} {r['max_ratio']:7.1f}x  "
              f"{m1:.3f}/{m2:.3f}/{m3:.3f}         {c1:.3f}/{c2:.3f}/{c3:.3f}")

    with open("diagnose_all_trackb_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> diagnose_all_trackb_results.json")


if __name__ == "__main__":
    main()
