import os
import sys
import json

import numpy as np
import tensorflow as tf

# Verifies Run 22 by rebuilding its EXACT architecture (verbatim from train_model_track_b.py's
# current model-building code, which is literally what produced this checkpoint) and loading
# weights via the native model.load_weights() API -- the same mechanism this script already uses
# successfully for its own warm-start/resume paths -- rather than hand-reconstructing from raw H5
# shapes (which would have been WRONG here: dense_1's shape matches 'deep1', not 'wide', despite
# wide being instantiated first in source order -- see MODEL_5C_TRACK_B_EXPERIMENT_LOG.md).
#
# If this passes the FP32-reproduction gate, re-exports to TFLite WITHOUT the suspected buggy
# "rebuild a second export_model and positionally copy weights into it" step -- converting the
# original model directly instead -- and re-tests INT8 with the already-fixed prediction-stratified
# calibration from ../Model 5f/requantize_int8.py.

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Model 5f"))
from requantize_int8 import load_and_engineer, build_prediction_stratified_calibration  # noqa: E402

RUN_NAME = "dense_b_run22"
RUN_DIR = f"results_5c_trackb_{RUN_NAME}"
SEQ_LEN = 180
N_FEATURES = 13
DEEP_OUT_PRESCALE = 10.0
DEEP_OUT_RESCALE = 0.2735
L2_REG = 1e-6
N_EVAL = 500


def build_run22_model():
    def _reg():
        return tf.keras.regularizers.l2(L2_REG)

    input_layer = tf.keras.layers.Input(shape=(SEQ_LEN, N_FEATURES), name="input")
    pooled = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, name="avgpool")(input_layer)
    flat = tf.keras.layers.Reshape((SEQ_LEN // 6 * N_FEATURES,), name="flatten")(pooled)

    bottleneck = tf.keras.layers.Dense(64, activation="relu6", use_bias=False,
                                       name="bottleneck", kernel_regularizer=_reg())(flat)
    wide = tf.keras.layers.Dense(16, activation="relu6", use_bias=False,
                                 name="wide", kernel_regularizer=_reg())(bottleneck)
    deep = tf.keras.layers.Dense(128, activation="relu6", use_bias=False,
                                 name="deep1", kernel_regularizer=_reg())(bottleneck)
    deep = tf.keras.layers.Dense(64, activation="relu6", use_bias=False,
                                 name="deep2", kernel_regularizer=_reg())(deep)
    deep_out = tf.keras.layers.Dense(32, use_bias=False,
                                     name="deep_out", kernel_regularizer=_reg())(deep)
    deep_out = tf.keras.layers.Rescaling(scale=DEEP_OUT_PRESCALE, name="deep_out_prescale")(deep_out)
    deep_out = tf.keras.layers.Activation("relu6", name="deep_out_relu6")(deep_out)
    deep_out = tf.keras.layers.Rescaling(scale=DEEP_OUT_RESCALE, name="deep_out_rescale")(deep_out)

    merged = tf.keras.layers.Concatenate(name="merged")([wide, deep_out])

    out_1 = tf.keras.layers.Dense(
        1, activation="linear", use_bias=False, dtype="float32", name="diff_1hr")(merged)
    out_2 = tf.keras.layers.Dense(
        1, activation="linear", use_bias=False, dtype="float32", name="diff_2hr")(merged)
    out_3 = tf.keras.layers.Dense(
        1, activation="linear", use_bias=False, dtype="float32", name="diff_3hr")(merged)

    return tf.keras.Model(inputs=input_layer, outputs=[out_1, out_2, out_3],
                          name=f"track_b_{RUN_NAME}_reconstructed")


def main():
    with open(f"{RUN_DIR}/results_5c_trackb_{RUN_NAME}.json") as f:
        results = json.load(f)
    claimed = (results["diff_1hr_mae_c"], results["diff_2hr_mae_c"], results["diff_3hr_mae_c"])
    print(f"Claimed offline MAE: {claimed[0]:.4f}/{claimed[1]:.4f}/{claimed[2]:.4f}C")

    with open(f"{RUN_DIR}/input_scaler_5c_trackb.json") as f:
        input_scaler = json.load(f)
    with open(f"{RUN_DIR}/target_scaler_5c_trackb.json") as f:
        target_scaler = json.load(f)
    y_min, y_max = target_scaler["min"], target_scaler["max"]
    features = list(input_scaler.keys())
    assert len(features) == N_FEATURES, f"expected {N_FEATURES} features, got {len(features)}"
    assert features == results["features"], "feature order mismatch vs results.json"

    print("\nBuilding Run 22's exact architecture (verbatim from train_model_track_b.py)...")
    model = build_run22_model()
    model.summary()

    ckpt_path = f"{RUN_DIR}/checkpoints/best_model.weights.h5"
    print(f"\nLoading weights via model.load_weights({ckpt_path})...")
    model.load_weights(ckpt_path)
    print("✅ Weights loaded")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    print("\nLoading + engineering val data...")
    train_df, val_df = load_and_engineer(data_dir)

    targets = ["temp_diff_1hr", "temp_diff_2hr", "temp_diff_3hr"]
    vdf = val_df.dropna(subset=features + targets).copy()
    print(f"val rows: {len(vdf):,}")

    Xdf = vdf[features].copy()
    for feat in features:
        lo = input_scaler[feat]["min"]
        hi = input_scaler[feat]["max"]
        Xdf[feat] = ((Xdf[feat] - lo) / (hi - lo)).clip(0.0, 1.0)
    X_flat = Xdf.values.astype(np.float32)
    y_raw = vdf[targets].values.astype(np.float32)

    n = min(N_EVAL, len(X_flat) - SEQ_LEN)
    start_indices = np.linspace(SEQ_LEN - 1, len(X_flat) - 1, n).astype(int)
    X_windows = np.stack([X_flat[i - SEQ_LEN + 1: i + 1] for i in start_indices], axis=0)
    y_true = y_raw[start_indices]

    print(f"\nRunning FP32 Keras predict() on {n} windows...")
    preds = model.predict(X_windows, batch_size=64, verbose=0)
    preds = np.stack([np.squeeze(p, axis=-1) for p in preds], axis=1)  # (n, 3)
    pred_c = (preds + 1) * 0.5 * (y_max - y_min) + y_min

    print(f"\nFP32 reproduction (Keras, weights loaded via load_weights):")
    measured = []
    for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        mae = float(np.mean(np.abs(pred_c[:, j] - y_true[:, j])))
        measured.append(mae)
        print(f"  {name}: {mae:.4f}C  (claimed {claimed[j]:.4f}C)")

    ratios = [m / c for m, c in zip(measured, claimed)]
    if max(ratios) > 2.0:
        print(f"\n⚠️  STILL MISMATCHED even loaded natively via Keras (max ratio {max(ratios):.1f}x) "
              f"— the checkpoint itself may not reproduce the claimed number, independent of any "
              f"TFLite export step. Stopping here; not proceeding to re-export.")
        return False

    print(f"\n✅ FP32 reproduction PASSES (max ratio {max(ratios):.2f}x) — the checkpoint is good, "
          f"confirming the bug is specifically in the TFLite export step. Proceeding to correct "
          f"re-export and INT8 re-test.")

    # -------------------------------------------------------------------------
    # Re-export WITHOUT the suspected buggy step: convert the original model directly,
    # no second "export_model" rebuild + positional weight copy.
    # -------------------------------------------------------------------------
    print("\n🔧 Re-exporting to TFLite directly from the verified model (no weight-copy step)...")
    run_model = tf.function(model)
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, SEQ_LEN, N_FEATURES], tf.float32, name="input"))

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    fp32_tflite = converter.convert()
    fp32_out_path = f"{RUN_DIR}/model_trackb_{RUN_NAME}_fp32_reexport.tflite"
    with open(fp32_out_path, "wb") as f:
        f.write(fp32_tflite)
    print(f"✅ FP32 re-export saved → {fp32_out_path}")

    # Sanity-check the re-exported FP32 TFLite against the same eval set before trusting INT8.
    interp_fp32 = tf.lite.Interpreter(model_path=fp32_out_path)
    interp_fp32.allocate_tensors()
    in_det = interp_fp32.get_input_details()
    out_det = interp_fp32.get_output_details()
    preds_tflite = [[] for _ in range(3)]
    for i in range(n):
        interp_fp32.set_tensor(in_det[0]["index"], X_windows[i:i + 1])
        interp_fp32.invoke()
        for j in range(3):
            preds_tflite[j].append(float(np.squeeze(interp_fp32.get_tensor(out_det[j]["index"]))))
    print("\nRe-exported FP32 .tflite MAE (sanity check vs. Keras predict() above):")
    for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        pc = (np.array(preds_tflite[j]) + 1) * 0.5 * (y_max - y_min) + y_min
        mae = float(np.mean(np.abs(pc - y_true[:, j])))
        print(f"  {name}: {mae:.4f}C")

    # -------------------------------------------------------------------------
    # INT8 re-export with the already-fixed prediction-stratified calibration.
    # -------------------------------------------------------------------------
    print("\n🔧 Building INT8 calibration set (prediction-stratified) and re-exporting INT8...")
    X_val_full = X_flat  # full val set for calibration pool
    # Build a pool of windows for calibration (subsample positions across the val set)
    pool_n = min(20000, len(X_flat) - SEQ_LEN)
    pool_indices = np.linspace(SEQ_LEN - 1, len(X_flat) - 1, pool_n).astype(int)
    X_pool = np.stack([X_flat[i - SEQ_LEN + 1: i + 1] for i in pool_indices], axis=0)

    calib_idx = build_prediction_stratified_calibration(model, X_pool)
    X_calib = X_pool[calib_idx]
    print(f"Calibration set: {X_calib.shape[0]} windows (from {pool_n} pooled)")

    def representative_data_gen():
        for i in range(X_calib.shape[0]):
            yield [X_calib[i:i + 1].astype(np.float32)]

    converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    int8_tflite = converter_int8.convert()
    int8_out_path = f"{RUN_DIR}/model_trackb_{RUN_NAME}_int8_reexport.tflite"
    with open(int8_out_path, "wb") as f:
        f.write(int8_tflite)
    print(f"✅ INT8 re-export saved → {int8_out_path}")

    interp_int8 = tf.lite.Interpreter(model_path=int8_out_path)
    interp_int8.allocate_tensors()
    in_det8 = interp_int8.get_input_details()
    out_det8 = interp_int8.get_output_details()
    in_scale, in_zp = in_det8[0]["quantization"]
    preds_int8 = [[] for _ in range(3)]
    for i in range(n):
        q_in = np.round(X_windows[i:i + 1] / in_scale + in_zp).astype(in_det8[0]["dtype"])
        interp_int8.set_tensor(in_det8[0]["index"], q_in)
        interp_int8.invoke()
        for j in range(3):
            out_s, out_zp = out_det8[j]["quantization"]
            raw = interp_int8.get_tensor(out_det8[j]["index"])
            preds_int8[j].append(float(np.squeeze(raw - out_zp) * out_s))

    print(f"\n=== RE-EXPORTED INT8 MAE (n={n}) — THE KEY RESULT ===")
    int8_results = {}
    for j, name in enumerate(["diff_1hr", "diff_2hr", "diff_3hr"]):
        pc = (np.array(preds_int8[j]) + 1) * 0.5 * (y_max - y_min) + y_min
        mae = float(np.mean(np.abs(pc - y_true[:, j])))
        int8_results[name] = mae
        print(f"  {name}: {mae:.4f}C")

    print(f"\nFor comparison:")
    print(f"  Original claimed FP32:      {claimed[0]:.4f}/{claimed[1]:.4f}/{claimed[2]:.4f}C")
    print(f"  Reconstructed FP32 (Keras): {measured[0]:.4f}/{measured[1]:.4f}/{measured[2]:.4f}C")
    print(f"  Re-exported INT8:           {int8_results['diff_1hr']:.4f}/"
          f"{int8_results['diff_2hr']:.4f}/{int8_results['diff_3hr']:.4f}C")
    print(f"  Original (broken) export INT8/FP32 for comparison: see diagnose_all_trackb_results.json")

    with open(f"{RUN_DIR}/run22_reexport_results.json", "w") as f:
        json.dump({"claimed_fp32": claimed, "reconstructed_fp32": measured,
                   "reexported_int8": int8_results}, f, indent=2)
    print(f"\nSaved → {RUN_DIR}/run22_reexport_results.json")
    return True


if __name__ == "__main__":
    main()
