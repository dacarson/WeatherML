import os
import sys
import json

import h5py
import numpy as np
import tensorflow as tf

# Re-quantizes Track B Run 2 (the previous best-known checkpoint across the whole Model 5-series,
# INT8 0.211/0.333/0.432°C) with the same fixed, prediction-stratified calibration methodology used
# to re-quantize Model 5f's Run 5/6/7/8 — see ../Model 5f/MODEL_5F_EXPERIMENT_LOG.md's "Calibration
# Fix" entry and ../Model 5f/requantize_int8.py. Run 2's original INT8 export predates that fix and
# used its own (unknown, since the original training script no longer exists) representative
# dataset, so it's an open question whether its number is trustworthy the same way Model 5f's
# Run 8/6 turned out not to be.
#
# Reuses the verified feature-engineering pipeline and prediction-stratified calibration logic
# from Model 5f's requantize_int8.py directly (architecture-agnostic — only needs a `model` and an
# `X_pool` array). Architecture/checkpoint-loading below is reconstructed from evaluate_run2_int8.py
# (same directory), which already verified this reconstruction via an FP32 reproduction check
# against Run 2's own saved numbers — this script repeats that same gate before trusting anything.

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Model 5f"))
from requantize_int8 import load_and_engineer, build_X, build_prediction_stratified_calibration  # noqa: E402

RUN_DIR = "results_5c_trackb_dense_b_run2"
N_EVAL = 500


def build_trackb_run2_model(n_features):
    inp = tf.keras.layers.Input(shape=(n_features,), name="input")
    x = tf.keras.layers.Dense(512, use_bias=False, name="dense0")(inp)
    x = tf.keras.layers.BatchNormalization(name="bn0")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(256, use_bias=False, name="dense1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(128, use_bias=False, name="dense2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(64, use_bias=False, name="dense3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu")(x)
    o1 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_1hr")(x)
    o2 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_2hr")(x)
    o3 = tf.keras.layers.Dense(1, use_bias=False, dtype="float32", name="diff_3hr")(x)
    return tf.keras.Model(inp, [o1, o2, o3])


def load_trackb_run2_weights(model, ckpt_path):
    f = h5py.File(ckpt_path, "r")
    dense_shape_to_name = {
        (23, 512): "dense0", (512, 256): "dense1", (256, 128): "dense2", (128, 64): "dense3",
    }
    head_arrs = []
    by_name = {l.name: l for l in model.layers}
    for i in range(7):
        key = "dense" if i == 0 else f"dense_{i}"
        arr = f[f"layers/{key}/vars/0"][:]
        if arr.shape in dense_shape_to_name:
            by_name[dense_shape_to_name[arr.shape]].set_weights([arr])
        elif arr.shape == (64, 1):
            head_arrs.append(arr)
    assert len(head_arrs) == 3, f"expected 3 head arrays, got {len(head_arrs)}"
    for head_name, arr in zip(["diff_1hr", "diff_2hr", "diff_3hr"], head_arrs):
        by_name[head_name].set_weights([arr])
    for i in range(4):
        bn_key = "batch_normalization" if i == 0 else f"batch_normalization_{i}"
        gamma, beta, moving_mean, moving_var = [f[f"layers/{bn_key}/vars/{j}"][:] for j in range(4)]
        by_name[f"bn{i}"].set_weights([gamma, beta, moving_mean, moving_var])
    f.close()


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    print("Loading + engineering train/val data...")
    train_df, val_df = load_and_engineer(data_dir)

    with open(f"{RUN_DIR}/input_scaler_5c_trackb.json") as f:
        input_scaler = json.load(f)
    with open(f"{RUN_DIR}/target_scaler_5c_trackb.json") as f:
        target_scaler = json.load(f)
    y_min, y_max = target_scaler["min"], target_scaler["max"]
    features = list(input_scaler.keys())
    n_features = len(features)
    print(f"Features ({n_features}): {features}")

    targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]
    vdf = val_df.dropna(subset=features + targets).copy()
    X_val = build_X(vdf, features, input_scaler)
    y_val = np.stack([2.0 * (vdf[t] - y_min) / (y_max - y_min) - 1.0
                      for t in targets], axis=1).astype(np.float32)
    n_eval = min(N_EVAL, len(vdf))
    X_eval = X_val[:n_eval]
    y_eval = y_val[:n_eval]

    model = build_trackb_run2_model(n_features)
    load_trackb_run2_weights(model, f"{RUN_DIR}/checkpoints/best_model.weights.h5")
    model.compile(optimizer="sgd", loss="mse",
                 metrics={"diff_1hr": "mae", "diff_2hr": "mae", "diff_3hr": "mae"})

    # Verification gate — same check evaluate_run2_int8.py uses: reproduce Run 2's own saved FP32
    # numbers before trusting the reconstructed architecture/weights at all. Computed manually via
    # model.predict() + numpy MAE rather than model.evaluate()'s metric-name indexing, since this
    # TF version collapses per-output metrics into a single 'compile_metrics' entry.
    res = model.evaluate(X_val, {"diff_1hr": y_val[:, 0], "diff_2hr": y_val[:, 1],
                                 "diff_3hr": y_val[:, 2]}, verbose=0, batch_size=2048)
    val_loss = res[0]
    scale = (y_max - y_min) / 2.0
    fp32_preds = model.predict(X_val, batch_size=8192, verbose=0)
    mae_c = {}
    for j, h in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        mae_c[h] = float(np.mean(np.abs(np.squeeze(fp32_preds[j], axis=-1) - y_val[:, j]))) * scale
    print(f"\nFP32 reproduction check:")
    print(f"  val_loss: {val_loss:.6f}  (Run 2's saved: 0.010067)")
    print(f"  MAE: 1hr={mae_c['diff_1hr']:.4f}  2hr={mae_c['diff_2hr']:.4f}  3hr={mae_c['diff_3hr']:.4f}")
    print(f"  Run 2's saved MAE: 1hr=0.4216  2hr=0.6236  3hr=0.7828")
    if abs(mae_c["diff_3hr"] - 0.7828) > 0.05:
        raise RuntimeError(
            "FP32 reproduction did not match Run 2's saved numbers closely enough — architecture "
            "or feature reconstruction is wrong. Not proceeding to INT8 re-quantization on an "
            "unverified model.")
    print("✅ FP32 reproduction matches — architecture and features confirmed correct.\n")

    calib_idx = build_prediction_stratified_calibration(model, X_val)
    X_calib = X_val[calib_idx]
    print(f"Calibration rows: {X_calib.shape[0]} (prediction-stratified from {X_val.shape[0]} val rows)")

    run_model = tf.function(model)
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, n_features], tf.float32, name="input"))

    def representative_data_gen():
        for i in range(X_calib.shape[0]):
            yield [X_calib[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_model = converter.convert()

    out_path = f"{RUN_DIR}/model_trackb_run2_int8_requant.tflite"
    with open(out_path, "wb") as f:
        f.write(int8_model)
    print(f"✅ Re-quantized INT8 model → {out_path}")

    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_scale, in_zp = in_det[0]["quantization"]

    preds = [[] for _ in range(3)]
    for i in range(n_eval):
        sample = X_eval[i:i + 1]
        q_in = np.round(sample / in_scale + in_zp).astype(in_det[0]["dtype"])
        interp.set_tensor(in_det[0]["index"], q_in)
        interp.invoke()
        for j in range(3):
            out_s, out_zp = out_det[j]["quantization"]
            raw = interp.get_tensor(out_det[j]["index"])
            preds[j].append(float(np.squeeze(raw - out_zp) * out_s))

    print(f"\nRe-quantized INT8 MAE (n={n_eval}):")
    results = {}
    for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        pred_c = (np.array(preds[j]) + 1) * 0.5 * (y_max - y_min) + y_min
        true_c = (y_eval[:n_eval, j] + 1) * 0.5 * (y_max - y_min) + y_min
        mae = float(np.mean(np.abs(pred_c - true_c)))
        results[name] = mae
        print(f"  {name}: {mae:.3f}°C")

    for j, o in enumerate(out_det):
        scale_o, zp_o = o["quantization"]
        lo = scale_o * (-128 - zp_o)
        hi = scale_o * (127 - zp_o)
        lo_c = (lo + 1) * 0.5 * (y_max - y_min) + y_min
        hi_c = (hi + 1) * 0.5 * (y_max - y_min) + y_min
        print(f"  head[{j}] representable C_range=[{lo_c:+.2f},{hi_c:+.2f}]")

    print(f"\nOriginal (pre-fix) INT8 n=500 numbers: 1hr=0.211  2hr=0.333  3hr=0.432 (°C)")

    out_json = f"{RUN_DIR}/run2_int8_requant_n500.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
