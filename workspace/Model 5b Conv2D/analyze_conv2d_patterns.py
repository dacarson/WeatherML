#!/usr/bin/env python3
"""
Model 5b Conv2D — Pattern Analysis (Exp 35–37 compatible)
==========================================================
Six analyses to understand what the Conv2D temporal path learned and
whether it contributes meaningfully to predictions.

  1. Filter weight heatmaps     — what patterns each filter detects
  2. Activation maps            — which time positions fire on representative windows
  3. Input saliency maps        — which (timestep, feature) cells drive each horizon
  4. Path zeroing               — val_loss when anchor OR conv path is zeroed out
  5. Temporal occlusion         — MSE vs. which 15-min region is blacked out
  6. GAP feature correlation    — are Conv2D activations redundant with explicit lags?

Run from workspace/Model 5b Conv2D/:
    python3 analyze_conv2d_patterns.py

Set KAGGLE_MODE = True to run directly on Kaggle (different data/checkpoint paths).
Feature list and n_features are read from input_scaler_5b.json automatically,
so the script works unchanged for Exp 35, 36, and 37.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
KAGGLE_MODE = False

# Local path to the experiment results directory (ignored when KAGGLE_MODE=True)
# Change this to target a different experiment:
#   "results_finished_exp36"  for Exp 35/36
#   "results_1_exp37"         for the first Exp 37 run
EXP_RESULTS_DIR = "results_1_exp37"

# Bulk window sample count for Analyses 4–6 (higher = more accurate, slower)
N_BULK_WINDOWS = 1000

# Temporal occlusion config (Analysis 5)
OCCLUSION_WINDOW = 15   # timesteps to black out per position
OCCLUSION_STRIDE = 10   # step between positions

# ─────────────────────────────────────────────────────────────────────────────

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

if KAGGLE_MODE:
    RESULTS_DIR  = Path("/kaggle/working")
    DATA_CSV     = Path("/kaggle/input/datasets/dacarson/weatherml-training-data/val_data_sf.csv")
else:
    RESULTS_DIR  = SCRIPT_DIR / "Kaggle" / EXP_RESULTS_DIR
    # Prefer plain CSV; fall back to zip
    _csv  = SCRIPT_DIR / ".." / "val_data_sf.csv"
    _zip  = SCRIPT_DIR / ".." / "val_data_sf.csv.zip"
    DATA_CSV = _csv if _csv.exists() else _zip

WEIGHTS_PATH       = RESULTS_DIR / "checkpoints" / "best_model.weights.h5"
SCALER_PATH        = RESULTS_DIR / "input_scaler_5b.json"
TARGET_SCALER_PATH = RESULTS_DIR / "target_scaler_5b.json"
OUTPUT_DIR         = RESULTS_DIR / "exp_analysis"

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN  = 180
FILTERS  = 96
HORIZONS = ["1hr", "2hr", "3hr"]

# FEATURES and N_FEATURES are set in main() after loading the scaler JSON.
# Declared here as module-level placeholders for functions that use them.
FEATURES   = []
N_FEATURES = 0


# ── Custom layer ──────────────────────────────────────────────────────────────
class LastTimestepAnchorFeatures(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop("anchor_indices", None)
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, -1, :]

    def get_config(self):
        return super().get_config()


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(n_features, seq_len=SEQ_LEN, filters=FILTERS):
    inp = tf.keras.layers.Input(shape=(seq_len, n_features), name="input")

    a = LastTimestepAnchorFeatures(name="last_ts_anchors")(inp)
    a = tf.keras.layers.Dense(32, use_bias=False, kernel_regularizer=l2(1e-4), name="dense_anchor")(a)
    a = tf.keras.layers.ReLU(max_value=6.0, name="relu_anchor")(a)

    x = tf.keras.layers.Reshape((seq_len, n_features, 1), name="reshape")(inp)
    x = tf.keras.layers.Conv2D(filters, (3,  1), padding="same",  use_bias=False,
                                kernel_regularizer=l2(1e-4), name="conv2d_t1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_t1")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu_t1")(x)
    x = tf.keras.layers.Conv2D(filters, (7,  1), padding="same",  use_bias=False,
                                kernel_regularizer=l2(1e-4), name="conv2d_t2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_t2")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu_t2")(x)
    x = tf.keras.layers.Conv2D(filters, (15, 1), padding="same",  use_bias=False,
                                kernel_regularizer=l2(1e-4), name="conv2d_t3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_t3")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu_t3")(x)
    x = tf.keras.layers.Conv2D(filters, (1, n_features), padding="valid", use_bias=False,
                                kernel_regularizer=l2(1e-4), name="conv2d_feat")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_feat")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu_feat")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=l2(1e-4), name="dense_context")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu_context")(x)

    m = tf.keras.layers.Concatenate(name="context_anchor_concat")([x, a])
    m = tf.keras.layers.Dense(32, use_bias=False, kernel_regularizer=l2(1e-4), name="dense_head")(m)
    m = tf.keras.layers.ReLU(max_value=6.0, name="relu_head")(m)

    o1 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_1hr")(m)
    o2 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_2hr")(m)
    o3 = tf.keras.layers.Dense(1, activation="linear", use_bias=False, dtype="float32", name="diff_3hr")(m)
    return tf.keras.Model(inputs=inp, outputs=[o1, o2, o3])


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _rolling_slope(data, window):
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    slopes = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    x_c = x - x.mean()
    denom = np.sum(x_c ** 2)
    shape   = (n - window + 1, window)
    strides = (data.strides[0], data.strides[0])
    wins = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    has_nan = np.any(np.isnan(wins), axis=1)
    y_c = wins - np.nanmean(wins, axis=1, keepdims=True)
    s = np.sum(x_c * y_c, axis=1) / denom
    s[has_nan] = np.nan
    slopes[window - 1:] = s
    return slopes


def load_and_prepare(data_path: Path, features: list) -> pd.DataFrame:
    print("Loading validation data …")
    p = str(data_path)
    if p.endswith(".zip"):
        with zipfile.ZipFile(data_path) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(data_path)

    # Parse timestamp → sorted DatetimeIndex
    ts_col = next((c for c in ("time", "timestamp", "ts") if c in df.columns), None)
    assert ts_col, "No timestamp column found in CSV"
    v = float(df[ts_col].max())
    unit = "ns" if v >= 1e17 else "us" if v >= 1e14 else "ms" if v >= 1e11 else "s"
    df[ts_col] = pd.to_datetime(df[ts_col], unit=unit, utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index()

    # Cyclical time encodings
    tod = df["time_of_day"]
    df["time_of_day_sin"]  = np.sin(2 * np.pi * tod / 24.0)
    df["time_of_day_cos"]  = np.cos(2 * np.pi * tod / 24.0)
    df["time_of_day_sin2"] = np.sin(4 * np.pi * tod / 24.0)
    df["time_of_day_cos2"] = np.cos(4 * np.pi * tod / 24.0)
    doy = df["day_of_year"]
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "wind_direction" in df.columns:
        df["wind_direction_sin"] = np.sin(2 * np.pi * df["wind_direction"] / 360.0)
        df["wind_direction_cos"] = np.cos(2 * np.pi * df["wind_direction"] / 360.0)

    df["temp_delta_1"] = df["temperature"].diff().fillna(0.0)

    # Time-based temperature lag features (handles 60/120/180 min dynamically)
    tol = pd.Timedelta(seconds=90)
    lag_cols = {60: "temp_lag60", 120: "temp_lag120", 180: "temp_lag180"}
    for mins, col in lag_cols.items():
        if col not in features:
            continue
        base = pd.DataFrame({"base_time": df.index,
                              "lookup_time": df.index - pd.Timedelta(minutes=mins)})
        src  = pd.DataFrame({"src_time": df.index,
                              "temperature": df["temperature"].to_numpy(dtype=np.float32)})
        merged = pd.merge_asof(base, src, left_on="lookup_time", right_on="src_time",
                               direction="backward", tolerance=tol)
        df[col] = merged["temperature"].to_numpy(dtype=np.float32)

    print("  Computing slope features …")
    df["temp_slope_15"]     = _rolling_slope(df["temperature"].values,       15)
    df["temp_slope_30"]     = _rolling_slope(df["temperature"].values,       30)
    df["temp_slope_60"]     = _rolling_slope(df["temperature"].values,       60)
    df["solar_slope_30"]    = _rolling_slope(df["solar_radiation"].values,   30)
    df["humidity_slope_30"] = _rolling_slope(df["relative_humidity"].values, 30)
    df["pressure_slope_60"] = _rolling_slope(df["station_pressure"].values,  60)

    # Zero-fill any features present in the scaler but not in this CSV
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠️  Zero-filling missing features: {missing}")
        for f in missing:
            df[f] = 0.0

    df = df.dropna(subset=[f for f in features if f in df.columns])

    # Build diff targets (needed for Analyses 4 & 5)
    if all(c in df.columns for c in ("temp_t+1hr", "temp_t+2hr", "temp_t+3hr")):
        df["diff_1hr"] = df["temp_t+1hr"] - df["temperature"]
        df["diff_2hr"] = df["temp_t+2hr"] - df["temperature"]
        df["diff_3hr"] = df["temp_t+3hr"] - df["temperature"]

    print(f"  ✅ {len(df):,} rows ready")
    return df


def scale_features(df: pd.DataFrame, scaler: dict, features: list) -> np.ndarray:
    f_mins = np.array([scaler[f]["min"] for f in features], dtype=np.float32)
    f_maxs = np.array([scaler[f]["max"] for f in features], dtype=np.float32)
    denom  = np.where((f_maxs - f_mins) == 0, 1.0, f_maxs - f_mins)
    vals   = df[features].values.astype(np.float32)
    return np.clip((vals - f_mins) / denom, 0.0, 1.0)


def scale_targets(df: pd.DataFrame, y_min: float, y_max: float) -> np.ndarray:
    """Return scaled targets in [-1, 1] to match model output space."""
    if not all(c in df.columns for c in ("diff_1hr", "diff_2hr", "diff_3hr")):
        return None
    raw = df[["diff_1hr", "diff_2hr", "diff_3hr"]].values.astype(np.float32)
    return 2.0 * (raw - y_min) / (y_max - y_min) - 1.0


# ── Bulk window builder ───────────────────────────────────────────────────────
def build_bulk_windows(scaled_flat: np.ndarray, y_scaled,
                       n_windows: int = N_BULK_WINDOWS):
    """
    Sample n_windows evenly-spaced windows from the full validation set.
    Returns X (N, SEQ_LEN, n_feat) and y (N, 3) or None.
    """
    n = scaled_flat.shape[0]
    max_end = n - 1
    min_end = SEQ_LEN - 1
    valid_ends = np.arange(min_end, max_end + 1)
    if len(valid_ends) > n_windows:
        idx = np.round(np.linspace(0, len(valid_ends) - 1, n_windows)).astype(int)
        valid_ends = valid_ends[idx]

    X = np.stack([scaled_flat[e - SEQ_LEN + 1:e + 1] for e in valid_ends])

    if y_scaled is not None:
        y = y_scaled[valid_ends]
    else:
        y = None

    return X.astype(np.float32), (y.astype(np.float32) if y is not None else None)


# ── Representative window selector ───────────────────────────────────────────
def select_windows(df: pd.DataFrame, scaled: np.ndarray):
    """Return a list of (label, window, metadata_dict) tuples for 5 scenarios."""
    if len(df) < SEQ_LEN + 1:
        raise RuntimeError("Not enough rows for even one window.")

    valid_end = np.arange(SEQ_LEN - 1, min(len(df), SEQ_LEN - 1 + 20_000))
    d3    = df["diff_3hr"].values  if "diff_3hr"        in df.columns else np.zeros(len(df))
    solar = df["solar_radiation"].values
    wind  = df["wind_gust"].values
    humid = df["relative_humidity"].values
    temp  = df["temperature"].values

    low_solar = solar[valid_end] < np.percentile(solar[valid_end], 20)
    foggy_idx = (valid_end[np.argsort(-humid[valid_end][low_solar])[:1]][0]
                 if low_solar.any()
                 else valid_end[np.argsort(-humid[valid_end])[:1]][0])

    scenarios = {
        "warm_up":   valid_end[np.argsort(-d3[valid_end])[:1]][0],
        "cool_down": valid_end[np.argsort( d3[valid_end])[:1]][0],
        "sunny":     valid_end[np.argsort(-solar[valid_end])[:1]][0],
        "foggy":     foggy_idx,
        "windy":     valid_end[np.argsort(-wind[valid_end])[:1]][0],
    }

    windows = []
    for label, end_idx in scenarios.items():
        w    = scaled[end_idx - SEQ_LEN + 1:end_idx + 1]
        meta = {
            "timestamp": str(df.index[end_idx])[:16],
            "temp_now":  float(temp[end_idx]),
            "diff_3hr":  float(d3[end_idx]),
        }
        windows.append((label, w, meta))
        print(f"  {label:12s}: ending {meta['timestamp']}  "
              f"temp={meta['temp_now']:.1f}°C  Δ3hr={meta['diff_3hr']:+.1f}°C")
    return windows


# ── Helper: batch MSE ─────────────────────────────────────────────────────────
def _batch_mse(m, X: np.ndarray, y_scaled: np.ndarray,
               batch_size: int = 256) -> np.ndarray:
    """MSE per horizon in scaled target space (matches val_loss metric)."""
    n = len(X)
    mse = np.zeros(3, dtype=np.float64)
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        preds = m(X[start:end], training=False)
        for j in range(3):
            p = preds[j].numpy().flatten()
            t = y_scaled[start:end, j]
            mse[j] += np.sum((p - t) ** 2)
    return mse / n


# ── Analysis 1: Filter Weights ────────────────────────────────────────────────
def analysis_filter_weights(model, features):
    print("\n── Analysis 1: Filter weights ──")
    n_features = len(features)

    w_t1      = model.get_layer("conv2d_t1").get_weights()[0]   # (3, 1, 1, 96)
    kernels_t1 = w_t1[:, 0, 0, :].T                             # (96, 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle("Conv2D Filter Weights", fontsize=13)

    ax = axes[0]
    norm = TwoSlopeNorm(vcenter=0, vmin=kernels_t1.min(), vmax=kernels_t1.max())
    im = ax.imshow(kernels_t1, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_title("conv2d_t1 kernels\n(3-step, 96 filters)")
    ax.set_xlabel("Temporal offset (steps)")
    ax.set_ylabel("Filter index")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["t−2", "t−1", "t"])
    plt.colorbar(im, ax=ax, fraction=0.046)

    norms = {}
    for lname, axes_to_reduce in [
        ("conv2d_t1",   (0, 1, 2)),
        ("conv2d_t2",   (0, 1, 2)),
        ("conv2d_t3",   (0, 1, 2)),
        ("conv2d_feat", (0, 1, 2)),
    ]:
        w = model.get_layer(lname).get_weights()[0]
        norms[lname] = np.sqrt(np.sum(w ** 2, axis=axes_to_reduce))

    ax = axes[1]
    for lname, n in norms.items():
        ax.plot(np.sort(n)[::-1], label=lname)
    ax.set_title("Filter L2 norms (sorted)\nFlat tail near zero = dead filters")
    ax.set_xlabel("Filter rank")
    ax.set_ylabel("L2 norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    w_feat      = model.get_layer("conv2d_feat").get_weights()[0]   # (1, n_feat, 96, 96)
    feat_contrib = np.mean(np.abs(w_feat[0, :, :, :]), axis=1)       # (n_feat, 96)
    ax = axes[2]
    im = ax.imshow(feat_contrib, aspect="auto", cmap="viridis")
    ax.set_title("conv2d_feat feature-mixing\n(mean |weight|: inputs → channels)")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(features, fontsize=6)
    ax.set_xlabel("Output channel")
    plt.colorbar(im, ax=ax, fraction=0.046)

    out = OUTPUT_DIR / "01_filter_weights.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    dead = {lname: int((n < 0.01).sum()) for lname, n in norms.items()}
    print(f"  Near-zero filters (L2 < 0.01): {dead}")


# ── Analysis 2: Activation Maps ───────────────────────────────────────────────
def analysis_activation_maps(model, windows):
    print("\n── Analysis 2: Activation maps ──")

    act_layers = ["relu_t1", "relu_t2", "relu_t3", "relu_feat"]
    extractors = {
        name: tf.keras.Model(inputs=model.input,
                             outputs=model.get_layer(name).output)
        for name in act_layers
    }

    n_scenarios = len(windows)
    n_layers    = len(act_layers)
    fig = plt.figure(figsize=(4 * n_scenarios, 3.5 * n_layers))
    gs  = gridspec.GridSpec(n_layers, n_scenarios, hspace=0.45, wspace=0.3)
    fig.suptitle("Conv2D Activation Maps\n"
                 "x: time in 3-hr window (left=oldest, right=newest)  "
                 "y: filter channel  brightness: activation magnitude",
                 fontsize=10)

    t_axis = np.arange(SEQ_LEN) - SEQ_LEN + 1

    for col, (label, window, meta) in enumerate(windows):
        inp_t = tf.constant(window[np.newaxis], dtype=tf.float32)
        for row, lname in enumerate(act_layers):
            act = extractors[lname](inp_t, training=False).numpy()
            if act.ndim == 4:
                act = act[0, :, 0, :]
            else:
                act = act[0]

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(act.T, aspect="auto", cmap="hot",
                      extent=[t_axis[0], t_axis[-1], act.shape[1], 0])
            ax.axvline(-120, color="cyan",  lw=0.8, linestyle="--")
            ax.axvline(-60,  color="lime",  lw=0.8, linestyle="--")
            ax.axvline(0,    color="white", lw=0.8, linestyle="--")
            if row == 0:
                ax.set_title(f"{label}\n{meta['timestamp']}\n"
                             f"T={meta['temp_now']:.1f}°C Δ3h={meta['diff_3hr']:+.1f}°C",
                             fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{lname}\n({act.shape[1]} ch)", fontsize=7)
            ax.set_xlabel("min before pred", fontsize=6)
            ax.tick_params(labelsize=5)

    out = OUTPUT_DIR / "02_activation_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # Mean activations at key time positions (relu_feat layer)
    extractor_feat = extractors["relu_feat"]
    print("  Mean activation at key time positions (relu_feat):")
    for label, window, meta in windows:
        inp_t = tf.constant(window[np.newaxis], dtype=tf.float32)
        act = extractor_feat(inp_t, training=False).numpy()[0, :, 0, :]  # (180, 96)
        print(f"    {label:12s}: t=0={act[-1].mean():.3f}  "
              f"t=−60={act[-61].mean():.3f}  "
              f"t=−120={act[-121].mean():.3f}  "
              f"t=−180={act[0].mean():.3f}")


# ── Analysis 3: Input Saliency ────────────────────────────────────────────────
def analysis_saliency(model, windows, features):
    print("\n── Analysis 3: Input saliency ──")
    n_features  = len(features)
    n_scenarios = len(windows)
    n_horizons  = len(HORIZONS)

    fig = plt.figure(figsize=(4 * n_scenarios, 3.5 * n_horizons))
    gs  = gridspec.GridSpec(n_horizons, n_scenarios, hspace=0.45, wspace=0.3)
    fig.suptitle("Input Saliency (|∂output/∂input|)\n"
                 "rows: prediction horizon  cols: weather scenarios\n"
                 "x: feature index  y: time (0=oldest, 179=newest)\n"
                 "Bright = high gradient → input cell drives this prediction",
                 fontsize=10)

    all_saliency = np.zeros((n_horizons, SEQ_LEN, n_features), dtype=np.float64)

    for col, (label, window, meta) in enumerate(windows):
        inp_t = tf.Variable(window[np.newaxis], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            preds = model(inp_t, training=False)

        for row, (horizon, pred) in enumerate(zip(HORIZONS, preds)):
            grad = tape.gradient(pred, inp_t)
            if grad is None:
                print(f"  ⚠️  No gradient for {horizon}")
                continue
            sal = np.abs(grad.numpy()[0])
            all_saliency[row] += sal

            ax = fig.add_subplot(gs[row, col])
            vmax = sal.max() if sal.max() > 0 else 1.0
            ax.imshow(sal, aspect="auto", cmap="inferno", vmin=0, vmax=vmax,
                      extent=[-0.5, n_features - 0.5, SEQ_LEN - 0.5, -0.5])
            ax.axhline(SEQ_LEN - 1,   color="white", lw=0.6, linestyle="--")
            ax.axhline(SEQ_LEN - 61,  color="cyan",  lw=0.6, linestyle="--")
            ax.axhline(SEQ_LEN - 121, color="lime",  lw=0.6, linestyle="--")
            if row == 0:
                ax.set_title(f"{label}\n{meta['timestamp']}", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"Δ{horizon}\n(time↓ old→new)", fontsize=7)
            ax.set_xlabel("feature index", fontsize=6)
            ax.set_xticks(range(n_features))
            ax.set_xticklabels([f[:6] for f in features], rotation=90, fontsize=4)
            ax.tick_params(axis="y", labelsize=5)
        del tape

    out = OUTPUT_DIR / "03_saliency_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # Mean saliency across scenarios
    fig2, axes = plt.subplots(1, n_horizons, figsize=(5 * n_horizons, 7))
    fig2.suptitle("Mean Saliency (averaged over all scenarios)\n"
                  "Bright = consistently high gradient across scenarios", fontsize=11)
    mean_sal = all_saliency / n_scenarios

    for row, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
        ms = mean_sal[row]
        vmax = ms.max() if ms.max() > 0 else 1.0
        im = ax.imshow(ms, aspect="auto", cmap="inferno", vmin=0, vmax=vmax,
                       extent=[-0.5, n_features - 0.5, SEQ_LEN - 0.5, -0.5])
        ax.axhline(SEQ_LEN - 1,   color="white", lw=0.8, linestyle="--", label="t=0")
        ax.axhline(SEQ_LEN - 61,  color="cyan",  lw=0.8, linestyle="--", label="t=−60")
        ax.axhline(SEQ_LEN - 121, color="lime",  lw=0.8, linestyle="--", label="t=−120")
        ax.set_title(f"Δ{horizon}")
        ax.set_xlabel("feature")
        ax.set_ylabel("time (0=oldest, 179=newest)")
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(features, rotation=90, fontsize=6)
        ax.legend(fontsize=7, loc="upper left")
        plt.colorbar(im, ax=ax, fraction=0.046)

        flat = ms.flatten()
        top5 = np.argsort(-flat)[:5]
        print(f"  Top-5 saliency cells for Δ{horizon}:")
        for idx in top5:
            t_pos = idx // n_features
            f_pos = idx % n_features
            t_rel = t_pos - SEQ_LEN + 1
            print(f"    t={t_rel:+4d} min  {features[f_pos]:30s}  sal={flat[idx]:.4f}")

    out2 = OUTPUT_DIR / "04_mean_saliency.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {out2}")


# ── Analysis 4: Path Zeroing ──────────────────────────────────────────────────
def analysis_path_zeroing(model, X_bulk: np.ndarray, y_scaled: np.ndarray):
    """
    Zero out the anchor path OR the Conv2D/context path and measure val_loss.

    If the anchor path is doing all the work, zeroing it collapses performance.
    If Conv2D adds nothing, zeroing it leaves performance unchanged.
    """
    print(f"\n── Analysis 4: Path zeroing ({len(X_bulk)} windows) ──")

    # ── Build two ablation models sharing all weights with `model` ──
    inp        = model.input
    anchor_out = model.get_layer("relu_anchor").output   # (None, 32)
    conv_out   = model.get_layer("relu_context").output  # (None, 64)

    dense_head = model.get_layer("dense_head")
    relu_head  = model.get_layer("relu_head")
    out_layers = [model.get_layer(n) for n in ("diff_1hr", "diff_2hr", "diff_3hr")]

    def _head(merged):
        x = dense_head(merged)
        x = relu_head(x)
        return [ol(x) for ol in out_layers]

    # Conv-only: anchor path output zeroed
    zero_anc    = tf.keras.layers.Lambda(lambda x: x * 0.0, name="zero_anchor")(anchor_out)
    co_merged   = tf.keras.layers.Concatenate(name="cat_conv_only")([conv_out, zero_anc])
    model_conv  = tf.keras.Model(inputs=inp, outputs=_head(co_merged), name="conv_only")

    # Anchor-only: conv path output zeroed
    zero_conv    = tf.keras.layers.Lambda(lambda x: x * 0.0, name="zero_conv")(conv_out)
    ao_merged    = tf.keras.layers.Concatenate(name="cat_anchor_only")([zero_conv, anchor_out])
    model_anchor = tf.keras.Model(inputs=inp, outputs=_head(ao_merged), name="anchor_only")

    # Evaluate all three models
    mse_full   = _batch_mse(model,        X_bulk, y_scaled)
    mse_conv   = _batch_mse(model_conv,   X_bulk, y_scaled)
    mse_anchor = _batch_mse(model_anchor, X_bulk, y_scaled)

    total_full   = float(mse_full.mean())
    total_conv   = float(mse_conv.mean())
    total_anchor = float(mse_anchor.mean())

    print(f"\n  MSE (scaled target space, lower = better):")
    print(f"  {'Model':<28} {'Δ1hr':>8} {'Δ2hr':>8} {'Δ3hr':>8} {'Mean':>8}")
    print(f"  {'-'*60}")
    for label, mse in [("Full model (baseline)", mse_full),
                        ("Conv path only (anchor=0)", mse_conv),
                        ("Anchor path only (conv=0)", mse_anchor)]:
        print(f"  {label:<28} {mse[0]:8.5f} {mse[1]:8.5f} {mse[2]:8.5f} {mse.mean():8.5f}")

    conv_pct   = 100 * (total_conv   - total_full) / total_full
    anchor_pct = 100 * (total_anchor - total_full) / total_full
    print(f"\n  Zeroing anchor path  → {conv_pct:+.1f}% worse   (= Conv2D's marginal value)")
    print(f"  Zeroing conv path    → {anchor_pct:+.1f}% worse   (= Anchor's marginal value)")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Path Zeroing — What happens when each path is disabled?", fontsize=12)

    # Panel A: mean MSE per model
    ax = axes[0]
    labels = ["Full\nModel", "Conv Path\nOnly\n(anchor=0)", "Anchor Path\nOnly\n(conv=0)"]
    values = [total_full, total_conv, total_anchor]
    colors = ["steelblue", "darkorange", "forestgreen"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(total_full, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + total_full * 0.01,
                f"{val:.5f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean MSE (scaled space)")
    ax.set_title("Mean MSE across all 3 horizons")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: per-horizon MSE for each model
    ax = axes[1]
    x = np.arange(3)
    width = 0.25
    for i, (label, mse, color) in enumerate([
        ("Full model", mse_full, "steelblue"),
        ("Conv only",  mse_conv, "darkorange"),
        ("Anchor only",mse_anchor,"forestgreen"),
    ]):
        ax.bar(x + i * width, mse, width, label=label, color=color,
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Δ1hr", "Δ2hr", "Δ3hr"])
    ax.set_ylabel("MSE (scaled space)")
    ax.set_title("MSE per horizon")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out = OUTPUT_DIR / "05_path_zeroing.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    summary = {
        "mse_full_by_horizon":   mse_full.tolist(),
        "mse_conv_only":         mse_conv.tolist(),
        "mse_anchor_only":       mse_anchor.tolist(),
        "conv_zeroing_pct_worse":   anchor_pct,
        "anchor_zeroing_pct_worse": conv_pct,
        "interpretation": {
            "conv2d_marginal_contribution_pct":  round(anchor_pct, 2),
            "anchor_marginal_contribution_pct":  round(conv_pct, 2),
        }
    }
    with open(OUTPUT_DIR / "path_zeroing_results.json", "w") as f:
        json.dump(summary, f, indent=2)


# ── Analysis 5: Temporal Occlusion ────────────────────────────────────────────
def analysis_temporal_occlusion(model, X_bulk: np.ndarray, y_scaled: np.ndarray,
                                  window: int = OCCLUSION_WINDOW,
                                  stride: int = OCCLUSION_STRIDE):
    """
    Slide a blackout window across the 180-step sequence and measure MSE at
    each position. Peaks reveal which time regions the model relies on.

    A spike near t=0 means the anchor path dominates (just needs the latest value).
    Spikes near t=−60/−120 confirm the model uses those explicit lag anchors.
    Flat response means the temporal context is broadly used (or not used at all).
    """
    print(f"\n── Analysis 5: Temporal occlusion "
          f"(window={window}, stride={stride}, n={len(X_bulk)}) ──")

    baseline = _batch_mse(model, X_bulk, y_scaled)
    print(f"  Baseline MSE: {baseline}")

    positions = list(range(0, SEQ_LEN - window + 1, stride))
    deltas = np.zeros((len(positions), 3), dtype=np.float64)

    for k, t_start in enumerate(positions):
        X_occ = X_bulk.copy()
        X_occ[:, t_start:t_start + window, :] = 0.0   # black out this window
        mse_k = _batch_mse(model, X_occ, y_scaled)
        deltas[k] = mse_k - baseline
        if k % 5 == 0:
            t_rel = t_start - SEQ_LEN + 1
            print(f"  t={t_rel:+4d}…{t_rel + window:+4d}: ΔMSE={deltas[k].mean():+.6f}")

    # Time axis in minutes before prediction
    t_positions = np.array(positions) - SEQ_LEN + 1   # negative = minutes before pred

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Temporal Occlusion — MSE increase when {window}-step window is blacked out\n"
                 f"Peaks = regions the model depends on  |  Flat = unused time range",
                 fontsize=11)

    ax = axes[0]
    for j, (h, color) in enumerate(zip(HORIZONS, ["royalblue", "darkorange", "forestgreen"])):
        ax.plot(t_positions, deltas[:, j], color=color, label=f"Δ{h}", linewidth=1.5)
    for t_ref, label, col in [(-120, "t=−120", "purple"), (-60, "t=−60", "red"),
                                (0, "t=0", "black")]:
        if t_ref >= t_positions.min():
            ax.axvline(t_ref, color=col, linestyle="--", linewidth=0.8, alpha=0.7, label=label)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_ylabel("ΔMSE (scaled)")
    ax.set_title("Per-horizon MSE increase")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_positions, deltas.mean(axis=1), color="darkred", linewidth=2, label="Mean Δ3 horizons")
    for t_ref, label, col in [(-120, "t=−120", "purple"), (-60, "t=−60", "red"),
                                (0, "t=0", "black")]:
        if t_ref >= t_positions.min():
            ax.axvline(t_ref, color=col, linestyle="--", linewidth=0.8, alpha=0.7, label=label)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Time relative to prediction (minutes)")
    ax.set_ylabel("Mean ΔMSE")
    ax.set_title("Mean MSE increase (all 3 horizons)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / "06_temporal_occlusion.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # Top-5 most sensitive windows
    mean_delta = deltas.mean(axis=1)
    top5 = np.argsort(-mean_delta)[:5]
    print("\n  Top-5 most sensitive time windows (mean ΔMSE):")
    for k in top5:
        t_rel = positions[k] - SEQ_LEN + 1
        print(f"    t={t_rel:+4d}…{t_rel + window:+4d}: ΔMSE={mean_delta[k]:+.6f}")


# ── Analysis 6: GAP Feature Correlation ──────────────────────────────────────
def analysis_gap_correlation(model, X_bulk: np.ndarray, features: list):
    """
    Check whether the Conv2D's GlobalAveragePooling output is largely redundant
    with the explicit lag and slope features that are already in the anchor path.

    High max-correlation for a feature (> ~0.7) means the Conv2D is mostly
    re-learning what you already gave it explicitly. Low correlation means it's
    capturing genuinely new temporal structure.
    """
    print(f"\n── Analysis 6: GAP feature correlation ({len(X_bulk)} windows) ──")

    gap_extractor = tf.keras.Model(inputs=model.input,
                                   outputs=model.get_layer("gap").output)
    gap_acts = []
    for start in range(0, len(X_bulk), 256):
        batch = X_bulk[start:start + 256]
        gap_acts.append(gap_extractor(batch, training=False).numpy())
    gap_acts = np.vstack(gap_acts).astype(np.float64)   # (N, 96)
    print(f"  GAP activations: {gap_acts.shape}  "
          f"mean={gap_acts.mean():.3f}  std={gap_acts.std():.3f}")

    # Last-timestep features (what the anchor path actually sees)
    last_ts = X_bulk[:, -1, :].astype(np.float64)   # (N, n_features)

    n_feat   = len(features)
    n_gap    = gap_acts.shape[1]
    all_corr = np.zeros((n_feat, n_gap), dtype=np.float64)

    # Normalise GAP neurons once
    gap_std  = gap_acts.std(axis=0)
    gap_live = gap_std > 1e-8
    gap_norm = np.where(gap_live,
                        (gap_acts - gap_acts.mean(axis=0)) / np.where(gap_live, gap_std, 1.0),
                        0.0)

    for fi, fname in enumerate(features):
        col      = last_ts[:, fi]
        col_std  = col.std()
        if col_std < 1e-8:
            continue
        col_norm = (col - col.mean()) / col_std
        # Pearson r(feature_i, gap_j) = dot(normalised) / N
        all_corr[fi] = (col_norm @ gap_norm) / len(col_norm)

    max_corr_per_feat = np.max(np.abs(all_corr), axis=1)   # (n_features,)
    sorted_idx        = np.argsort(-max_corr_per_feat)

    print("\n  Max |Pearson r| between each feature and any GAP neuron:")
    print(f"  {'Feature':<30} {'Max |r|':>8}  {'Interpretation'}")
    print(f"  {'-'*65}")
    for fi in sorted_idx:
        r = max_corr_per_feat[fi]
        interp = ("high — Conv2D mostly re-learning this" if r > 0.7
                  else "moderate" if r > 0.4
                  else "low — Conv2D sees something new here")
        print(f"  {features[fi]:<30} {r:8.3f}  {interp}")

    # Plot 1: bar chart of max correlation per feature
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("GAP Feature Correlation — Are Conv2D activations redundant\n"
                 "with the explicit features the anchor path already has?", fontsize=11)

    ax = axes[0]
    sorted_feat   = [features[i] for i in sorted_idx]
    sorted_corrs  = max_corr_per_feat[sorted_idx]
    colors = ["tomato" if r > 0.7 else "gold" if r > 0.4 else "seagreen"
              for r in sorted_corrs]
    bars = ax.barh(range(n_feat), sorted_corrs, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(sorted_feat, fontsize=8)
    ax.axvline(0.7, color="tomato",  linestyle="--", linewidth=1, alpha=0.7, label="r=0.7 (high)")
    ax.axvline(0.4, color="gold",    linestyle="--", linewidth=1, alpha=0.7, label="r=0.4 (moderate)")
    ax.set_xlabel("Max |Pearson r| with any GAP neuron")
    ax.set_title("Per-feature max correlation\nRed = Conv2D mostly re-learning; Green = new info")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Plot 2: full correlation heatmap (features × GAP neurons)
    ax = axes[1]
    im = ax.imshow(np.abs(all_corr[sorted_idx, :]), aspect="auto", cmap="Reds",
                   vmin=0, vmax=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(sorted_feat, fontsize=7)
    ax.set_xlabel("GAP neuron index (0–95)")
    ax.set_title("|r| heatmap: features × GAP neurons\n"
                 "Bright rows = feature is correlated with Conv2D output")
    plt.colorbar(im, ax=ax, fraction=0.046)

    out = OUTPUT_DIR / "07_gap_correlation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # Dead neurons: fraction of GAP neurons with near-zero variance
    n_dead = int((~gap_live).sum())
    print(f"\n  Dead GAP neurons (std < 1e-8): {n_dead}/{n_gap}")
    mean_max = float(max_corr_per_feat.mean())
    print(f"  Mean max |r| across all features: {mean_max:.3f}")
    if mean_max > 0.6:
        print("  ⚠️  High redundancy: Conv2D GAP is mostly recapturing explicit features")
    elif mean_max > 0.35:
        print("  ⚠️  Moderate redundancy: some new temporal info, some overlap")
    else:
        print("  ✅ Low redundancy: Conv2D is learning complementary temporal structure")

    corr_summary = {
        "max_corr_per_feature": {features[fi]: float(max_corr_per_feat[fi]) for fi in range(n_feat)},
        "dead_gap_neurons": n_dead,
        "mean_max_abs_r": mean_max,
    }
    with open(OUTPUT_DIR / "gap_correlation_results.json", "w") as f:
        json.dump(corr_summary, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global FEATURES, N_FEATURES

    print("=" * 60)
    print("Conv2D Pattern Analysis")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p, name in [(WEIGHTS_PATH,       "weights"),
                    (SCALER_PATH,        "input scaler"),
                    (TARGET_SCALER_PATH, "target scaler"),
                    (DATA_CSV,           "validation data")]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}  ({name})")

    with open(SCALER_PATH) as f:
        scaler = json.load(f)
    with open(TARGET_SCALER_PATH) as f:
        tgt = json.load(f)

    # Feature list and count come directly from the scaler — automatic Exp 35/36/37 compat.
    FEATURES   = list(scaler.keys())
    N_FEATURES = len(FEATURES)
    y_min = float(tgt["min"])
    y_max = float(tgt["max"])

    print(f"\nFeatures ({N_FEATURES}): {FEATURES}")
    print(f"Target range: [{y_min:.2f}, {y_max:.2f}]")

    # Build model and load weights
    print("\nBuilding model and loading weights …")
    model = build_model(N_FEATURES)
    model(tf.zeros((1, SEQ_LEN, N_FEATURES), dtype=tf.float32), training=False)
    model.load_weights(str(WEIGHTS_PATH))
    total_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Loaded: {WEIGHTS_PATH.name}   params: {total_params:,}")

    # Load and preprocess validation data
    df     = load_and_prepare(DATA_CSV, FEATURES)
    scaled = scale_features(df, scaler, FEATURES)
    y_sc   = scale_targets(df, y_min, y_max)

    if y_sc is None:
        print("⚠️  Target columns (diff_1hr/2hr/3hr) not found — "
              "Analyses 4 & 5 will be skipped.")

    # Representative windows for Analyses 1–3
    print("\nSelecting representative windows …")
    windows = select_windows(df, scaled)

    # Bulk window set for Analyses 4–6
    print(f"\nBuilding {N_BULK_WINDOWS} bulk windows …")
    X_bulk, y_bulk = build_bulk_windows(scaled, y_sc, N_BULK_WINDOWS)
    print(f"  X_bulk: {X_bulk.shape}  "
          + (f"y_bulk: {y_bulk.shape}" if y_bulk is not None else "y_bulk: N/A"))

    # ── Run all analyses ──────────────────────────────────────────────────────
    analysis_filter_weights(model, FEATURES)
    analysis_activation_maps(model, windows)
    analysis_saliency(model, windows, FEATURES)

    if y_bulk is not None:
        analysis_path_zeroing(model, X_bulk, y_bulk)
        analysis_temporal_occlusion(model, X_bulk, y_bulk)
    else:
        print("\n⚠️  Skipping Analyses 4 & 5 (no target columns in validation data)")

    analysis_gap_correlation(model, X_bulk, FEATURES)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ All outputs saved to {OUTPUT_DIR}/")
    print("  01_filter_weights.png   — Conv2D filter weight heatmaps")
    print("  02_activation_maps.png  — per-layer activations on 5 scenarios")
    print("  03_saliency_maps.png    — per-window input saliency |∂out/∂in|")
    print("  04_mean_saliency.png    — mean saliency across all scenarios")
    print("  05_path_zeroing.png     — MSE when each path is disabled")
    print("  06_temporal_occlusion.png — MSE vs blacked-out time region")
    print("  07_gap_correlation.png  — GAP redundancy with explicit features")
    print(f"\nKey things to look for:")
    print("  05: If anchor-only ≈ full model → Conv2D adds nothing")
    print("      If conv-only collapses → anchor path is essential")
    print("  06: Peaks near t=0 → anchor dominates; near t=−60/−120 → lag features matter")
    print("      Flat response → model ignores temporal context (Conv2D not contributing)")
    print("  07: Most features red (r > 0.7) → Conv2D re-learning explicit lags; no new info")
    print("      Most features green (r < 0.4) → Conv2D learning genuinely new structure")


if __name__ == "__main__":
    main()
