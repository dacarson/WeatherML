#!/usr/bin/env python3
"""
Integrated Gradients analysis for Model 5c TFT.

Computes per-(feature, timestep) attributions for each prediction head,
giving the (feature × lag) cross-term that VSN weights and attention
scores alone cannot provide.

Usage — Run 2 (D_MODEL=64, N_HEADS=4, incomplete checkpoint):
    python3 analyze_integrated_gradients.py --results_dir results_5c_run2 --d_model 64 --n_heads 4

Usage — Run 3 (D_MODEL=128, N_HEADS=8, defaults):
    python3 analyze_integrated_gradients.py --results_dir results_5c_run3
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================================================================
# Custom layer definitions — must match train_model_tft_track_a.py exactly
# =========================================================================
L2_REG = 1e-4

try:
    _register = tf.keras.saving.register_keras_serializable
except AttributeError:
    _register = tf.keras.utils.register_keras_serializable


@_register(package="WeatherML5C")
class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.dense_elu = tf.keras.layers.Dense(
            units, activation="elu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.dense_linear = tf.keras.layers.Dense(
            units, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.dense_gate = tf.keras.layers.Dense(
            units, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.skip_proj = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.skip_proj = tf.keras.layers.Dense(
                self.units, use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        super().build(input_shape)

    def call(self, x, training=None):
        skip = self.skip_proj(x) if self.skip_proj is not None else x
        h = self.dense_elu(x)
        h = self.dropout_layer(h, training=training)
        gated = self.dense_linear(h) * self.dense_gate(h)
        return self.layer_norm(skip + gated)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "dropout": self.dropout_rate})
        return cfg


@_register(package="WeatherML5C")
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pos = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
        self._pe = tf.constant(pe[np.newaxis], dtype=tf.float32)

    def call(self, x):
        return x + tf.cast(self._pe[:, :tf.shape(x)[1], :], x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "max_len": self.max_len})
        return cfg


@_register(package="WeatherML5C")
class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, n_features, d_model, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_model = d_model
        self.dropout_rate = dropout
        self.embedding_kernel = None
        self.embedding_bias = None
        self.weight_grn = GatedResidualNetwork(n_features, dropout=dropout, name="vsn_weight_grn")
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        self.embedding_kernel = self.add_weight(
            name="embedding_kernel", shape=(self.n_features, self.d_model),
            initializer="glorot_uniform", regularizer=tf.keras.regularizers.l2(L2_REG))
        self.embedding_bias = self.add_weight(
            name="embedding_bias", shape=(self.n_features, self.d_model),
            initializer="zeros")
        super().build(input_shape)

    def call(self, x, training=None):
        weights = self.softmax(self.weight_grn(x, training=training))
        embedded = tf.nn.elu(
            tf.einsum("bsf,fd->bsfd", x, self.embedding_kernel) + self.embedding_bias)
        selected = tf.reduce_sum(embedded * weights[..., tf.newaxis], axis=-2)
        return self.layer_norm(selected), weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features, "d_model": self.d_model,
                    "dropout": self.dropout_rate})
        return cfg


# =========================================================================
# Model construction — mirrors train_model_tft_track_a.py build block
# =========================================================================
def build_model(n_features, seq_len, d_model, n_heads, dropout_rate=0.1):
    input_layer = tf.keras.layers.Input(shape=(seq_len, n_features), name="input")
    vsn_layer = VariableSelectionNetwork(n_features, d_model, dropout=dropout_rate, name="vsn")
    vsn_out, _ = vsn_layer(input_layer)
    enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 1, name="pos_enc")(vsn_out)
    mha_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads,
        dropout=dropout_rate, name="temporal_attention")
    attn_out, _ = mha_layer(query=enc, key=enc, value=enc, return_attention_scores=True)
    grn_attn = GatedResidualNetwork(d_model, dropout=dropout_rate, name="grn_post_attn")
    attn_out = grn_attn(enc + attn_out)
    grn_ff = GatedResidualNetwork(d_model, dropout=dropout_rate, name="grn_feedforward")
    ff_out = grn_ff(attn_out)
    last_ts = ff_out[:, -1, :]
    out_1 = tf.keras.layers.Dense(1, activation='linear', use_bias=False,
                                   dtype="float32", name='diff_1hr')(last_ts)
    out_2 = tf.keras.layers.Dense(1, activation='linear', use_bias=False,
                                   dtype="float32", name='diff_2hr')(last_ts)
    out_3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False,
                                   dtype="float32", name='diff_3hr')(last_ts)
    return tf.keras.Model(inputs=input_layer, outputs=[out_1, out_2, out_3])


# =========================================================================
# Preprocessing — replicates training script val data pipeline
# =========================================================================
def _rolling_slope(data, window):
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    slopes = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    x_c = x - x.mean()
    denom = np.sum(x_c ** 2)
    shape = (n - window + 1, window)
    strides = (data.strides[0], data.strides[0])
    wins = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    has_nan = np.any(np.isnan(wins), axis=1)
    y_c = wins - np.nanmean(wins, axis=1, keepdims=True)
    s = np.sum(x_c * y_c, axis=1) / denom
    s[has_nan] = np.nan
    slopes[window - 1:] = s
    return slopes


def prepare_val_data(data_dir, results_dir, seq_len=180):
    """
    Loads and preprocesses val_data_sf.csv using the scalers from results_dir.
    Returns X_flat (N, n_features), y_flat (N, 3), feature_names.
    """
    val_df = pd.read_csv(os.path.join(data_dir, "val_data_sf.csv"))

    # Time index
    time_col = next(
        (c for c in ("time", "timestamp", "ts", "datetime", "date") if c in val_df.columns), None)
    if time_col:
        s = val_df[time_col]
        if np.issubdtype(s.dtype, np.number):
            v = float(np.nanmax(s.to_numpy(dtype=np.float64)))
            unit = "ns" if v >= 1e17 else "us" if v >= 1e14 else "ms" if v >= 1e11 else "s"
            val_df[time_col] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        else:
            val_df[time_col] = pd.to_datetime(s, utc=True, errors="coerce")
        val_df = val_df.set_index(time_col).sort_index()
        if val_df.index.has_duplicates:
            val_df = val_df[~val_df.index.duplicated(keep="last")]

    # Future targets via merge_asof if not already present
    if not all(c in val_df.columns for c in ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]):
        base = val_df.reset_index()
        if "time" not in base.columns:
            base = base.rename(columns={base.columns[0]: "time"})
        base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
        base = base.sort_values("time").reset_index(drop=True)
        base["row_id"] = np.arange(len(base))
        src = base[["time", "temperature"]].rename(columns={"temperature": "temperature_future"})
        tol = pd.Timedelta(seconds=90)
        for mins, col in ((60, "temp_t+1hr"), (120, "temp_t+2hr"), (180, "temp_t+3hr")):
            want = base[["row_id", "time"]].copy()
            want["t_query"] = want["time"] + pd.Timedelta(minutes=mins)
            merged = pd.merge_asof(
                want.sort_values("t_query"), src,
                left_on="t_query", right_on="time",
                direction="forward", tolerance=tol,
            ).sort_values("row_id")
            base[col] = merged["temperature_future"].to_numpy()
        val_df = base.drop(columns=["row_id"]).set_index("time")

    # Invalidate cross-gap targets (tol=600s matches training script)
    if isinstance(val_df.index, pd.DatetimeIndex):
        dt_s = val_df.index.to_series().diff().dt.total_seconds()
        gap_positions = np.flatnonzero((dt_s > 600).to_numpy())
        if gap_positions.size > 0:
            val_df = val_df.copy()
            for pos in gap_positions:
                if pos == 0:
                    continue
                gap_boundary = val_df.index[pos - 1]
                for h_min, col in ((60, "temp_t+1hr"), (120, "temp_t+2hr"), (180, "temp_t+3hr")):
                    cutoff = gap_boundary - pd.Timedelta(minutes=h_min)
                    mask = (val_df.index > cutoff) & (val_df.index <= gap_boundary)
                    val_df.loc[mask, col] = np.nan

    # Cyclical encodings
    val_df['time_of_day_sin']  = np.sin(2 * np.pi * val_df['time_of_day'] / 24.0)
    val_df['time_of_day_cos']  = np.cos(2 * np.pi * val_df['time_of_day'] / 24.0)
    val_df['time_of_day_sin2'] = np.sin(4 * np.pi * val_df['time_of_day'] / 24.0)
    val_df['time_of_day_cos2'] = np.cos(4 * np.pi * val_df['time_of_day'] / 24.0)
    val_df['day_of_year_sin']  = np.sin(2 * np.pi * val_df['day_of_year'] / 365.25)
    val_df['day_of_year_cos']  = np.cos(2 * np.pi * val_df['day_of_year'] / 365.25)
    if 'wind_direction' in val_df.columns:
        val_df['wind_direction_sin'] = np.sin(2 * np.pi * val_df['wind_direction'] / 360.0)
        val_df['wind_direction_cos'] = np.cos(2 * np.pi * val_df['wind_direction'] / 360.0)

    # Slope features
    val_df['temp_slope_15']    = _rolling_slope(val_df['temperature'].values, 15)
    val_df['temp_slope_30']    = _rolling_slope(val_df['temperature'].values, 30)
    val_df['temp_slope_60']    = _rolling_slope(val_df['temperature'].values, 60)
    val_df['solar_slope_30']   = _rolling_slope(val_df['solar_radiation'].values, 30)
    val_df['humidity_slope_30']= _rolling_slope(val_df['relative_humidity'].values, 30)
    val_df['pressure_slope_60']= _rolling_slope(val_df['station_pressure'].values, 60)

    # Targets
    val_df['temp_diff_1hr'] = val_df['temp_t+1hr'] - val_df['temperature']
    val_df['temp_diff_2hr'] = val_df['temp_t+2hr'] - val_df['temperature']
    val_df['temp_diff_3hr'] = val_df['temp_t+3hr'] - val_df['temperature']
    val_df.dropna(inplace=True)

    # Gap-aware window safety (drop_span = seq_len // 2, matches training script)
    if isinstance(val_df.index, pd.DatetimeIndex):
        dt_s = val_df.index.to_series().diff().dt.total_seconds()
        gap_positions = np.flatnonzero((dt_s > 90).to_numpy())
        if gap_positions.size > 0:
            keep = np.ones(len(val_df), dtype=bool)
            drop_span = seq_len // 2
            for pos in gap_positions:
                keep[int(pos):min(int(pos) + drop_span, len(val_df))] = False
            val_df = val_df.iloc[keep]

    # Load saved scalers
    with open(os.path.join(results_dir, "input_scaler_5c.json")) as f:
        input_scaler = json.load(f)
    with open(os.path.join(results_dir, "target_scaler_5c.json")) as f:
        target_scaler = json.load(f)

    features = list(input_scaler.keys())
    targets  = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']

    X_df = val_df[features].copy()
    for feat in features:
        f_min = input_scaler[feat]["min"]
        f_max = input_scaler[feat]["max"]
        X_df[feat] = (X_df[feat] - f_min) / (f_max - f_min)
    X_flat = X_df.values.astype(np.float32)

    y_min = target_scaler["min"]
    y_max = target_scaler["max"]
    y_raw = val_df[targets].values.astype(np.float32)
    y_flat = 2.0 * (y_raw - y_min) / (y_max - y_min) - 1.0

    return X_flat, y_flat, features


def build_windows(X_flat, y_flat, seq_len, n_samples, seed=42):
    """Randomly sample n_samples sliding windows from the validation flat array."""
    rng = np.random.default_rng(seed)
    n_total = X_flat.shape[0] - seq_len + 1
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    X = np.stack([X_flat[i:i + seq_len] for i in indices]).astype(np.float32)
    y = y_flat[indices + seq_len - 1].astype(np.float32)
    return X, y


# =========================================================================
# Integrated Gradients
# =========================================================================
def compute_integrated_gradients(model, inputs, baseline=None, steps=50):
    """
    Standard integrated gradients (Sundararajan et al. 2017) for all 3 heads.

    inputs:   (N, seq_len, n_features) float32
    baseline: same shape — defaults to all-zeros (inputs are min-max scaled to [0,1])
    steps:    number of Riemann steps along the interpolation path

    Returns list of 3 arrays, each (seq_len, n_features): one per prediction head.
    The sign of each cell indicates direction (positive = pushes prediction up).
    """
    if baseline is None:
        baseline = np.zeros_like(inputs)

    N, seq_len, n_features = inputs.shape
    alphas = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    # Accumulate per-sample gradients over interpolation steps: (N, seq_len, n_features)
    grads_sum = [np.zeros((N, seq_len, n_features), dtype=np.float64) for _ in range(3)]

    print(f"  {steps + 1} interpolation steps × {N} samples...", flush=True)
    for step_i, alpha in enumerate(alphas):
        if step_i % 10 == 0:
            print(f"    step {step_i}/{steps}", flush=True)
        interp = tf.constant(
            (baseline + alpha * (inputs - baseline)).astype(np.float32))
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(interp)
            outputs = model(interp, training=False)
            # Scalars must be computed INSIDE the tape so the reduce_sum is recorded
            scalars = [tf.reduce_sum(outputs[h]) for h in range(3)]
        for head_idx in range(3):
            grads = tape.gradient(scalars[head_idx], interp)
            if grads is None:
                raise RuntimeError(
                    f"Gradient is None at step {step_i} head {head_idx} — "
                    "check that the model output is connected to the input tensor.")
            grads_sum[head_idx] += grads.numpy().astype(np.float64)
        del tape

    # Average over steps, then multiply by (input − baseline) per sample
    delta = (inputs - baseline).astype(np.float64)          # (N, seq_len, n_features)
    ig = [
        ((g / (steps + 1)) * delta).mean(axis=0)            # mean over N → (seq_len, n_features)
        for g in grads_sum
    ]
    return ig


# =========================================================================
# Display helpers
# =========================================================================
TARGET_NAMES = ['diff_1hr', 'diff_2hr', 'diff_3hr']


def print_top_pairs(ig, features, seq_len, top_k=20):
    """Top (feature, timestep) pairs sorted by absolute attribution."""
    abs_ig = np.abs(ig)
    flat_idx = np.argsort(abs_ig.ravel())[::-1][:top_k]
    rows, cols = np.unravel_index(flat_idx, ig.shape)
    print(f"  {'Rank':<5} {'Feature':<25} {'Lag':>10}   {'Attribution':>12}")
    print(f"  {'-'*5} {'-'*25} {'-'*10}   {'-'*12}")
    for rank, (row, col) in enumerate(zip(rows, cols), 1):
        lag = seq_len - 1 - int(row)
        print(f"  {rank:<5} {features[col]:<25} t-{lag:>3d}min   {ig[row, col]:>+12.6f}")


def print_feature_summary(ig, features):
    """Per-feature attribution summed over all timesteps."""
    feat_attr = ig.sum(axis=0)                           # (n_features,)
    order = np.argsort(np.abs(feat_attr))[::-1]
    max_abs = np.abs(feat_attr).max()
    print(f"  {'Feature':<25} {'Attribution':>12}  Bar")
    print(f"  {'-'*25} {'-'*12}  ---")
    for i in order:
        bar = int(abs(feat_attr[i]) / max_abs * 30)
        sign = '+' if feat_attr[i] >= 0 else '-'
        print(f"  {features[i]:<25} {feat_attr[i]:>+12.6f}  {sign}{'█' * bar}")


def print_timestep_summary(ig, seq_len, bucket=10):
    """Per-timestep attribution summed over features, displayed in lag-minute buckets."""
    ts_attr = ig.sum(axis=1)                             # (seq_len,)
    max_abs = np.abs(ts_attr).max()
    print(f"  {'Lag range':>22}  {'Attribution':>12}  Bar")
    print(f"  {'-'*22}  {'-'*12}  ---")
    for start_lag in range(0, seq_len, bucket):
        end_lag = min(start_lag + bucket - 1, seq_len - 1)
        pos_new = seq_len - 1 - start_lag
        pos_old = seq_len - 1 - end_lag
        bucket_val = ts_attr[pos_old:pos_new + 1].sum()
        bar = int(abs(bucket_val) / (max_abs * bucket) * 30)
        sign = '+' if bucket_val >= 0 else '-'
        label = f"t-{start_lag:>3d} to t-{end_lag:>3d}min"
        print(f"  {label:>22}  {bucket_val:>+12.6f}  {sign}{'█' * bar}")


def compare_with_attention(ig_list, seq_len, attn_path):
    """IG timestep attributions (per head) vs saved attention weights, side by side."""
    if not os.path.exists(attn_path):
        print(f"  (attention maps not found at {attn_path} — skipping comparison)")
        return
    with open(attn_path) as f:
        attn = json.load(f)
    attn_mean = np.array(attn["attention_mean_over_heads"])  # (seq_len,) pos 0 = oldest

    ts_attrs = [ig.sum(axis=1) for ig in ig_list]           # 3 × (seq_len,)

    checkpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                   100, 110, 120, 130, 140, 150, 160, 170, 174, 177, 179]
    print(f"  {'Lag':>10}  {'IG 1hr':>10}  {'IG 2hr':>10}  {'IG 3hr':>10}  {'Attention':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for lag in checkpoints:
        if lag >= seq_len:
            continue
        pos = seq_len - 1 - lag
        ig_vals = [ts_attrs[h][pos] for h in range(3)]
        print(f"  t-{lag:>3d}min  "
              f"{ig_vals[0]:>+10.5f}  {ig_vals[1]:>+10.5f}  {ig_vals[2]:>+10.5f}  "
              f"{attn_mean[pos]:>10.5f}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Integrated Gradients (feature × timestep) analysis for Model 5c TFT")
    parser.add_argument('--results_dir', default='results_5c_run2',
                        help='Results directory (contains checkpoints/, input_scaler_5c.json, '
                             'target_scaler_5c.json)')
    parser.add_argument('--data_dir', default='..',
                        help='Directory containing val_data_sf.csv (default: .. from script location)')
    parser.add_argument('--d_model', type=int, default=128,
                        help='D_MODEL for this run (64=Run2, 128=Run3+)')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='N_HEADS for this run (4=Run2, 8=Run3+)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seq_len', type=int, default=180)
    parser.add_argument('--n_samples', type=int, default=300,
                        help='Validation windows to average over (more = slower but more stable)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Riemann interpolation steps (50 is standard; 100 for higher accuracy)')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top (feature, lag) pairs to print per head')
    parser.add_argument('--weights_file', default='best_model.weights.h5',
                        help='Weights filename within checkpoints/')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: <results_dir>/integrated_gradients.json)')
    args = parser.parse_args()

    print("=" * 70)
    print("Model 5c TFT — Integrated Gradients Analysis")
    print("=" * 70)
    print(f"  results_dir : {args.results_dir}")
    print(f"  data_dir    : {args.data_dir}")
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  seq_len={args.seq_len}")
    print(f"  n_samples={args.n_samples}  steps={args.steps}")

    # Run on CPU: GradientTape is memory-bound, not compute-bound, and avoids
    # Metal GPU hang issues on macOS. Kaggle T4 is fine either way.
    tf.config.set_visible_devices([], 'GPU')
    print("\nℹ️  GPU disabled — IG runs on CPU (GradientTape is memory-bound, not compute-bound)")

    # ---- Preprocess validation data ----------------------------------------
    print("\n📂 Preprocessing validation data...")
    X_flat, y_flat, features = prepare_val_data(args.data_dir, args.results_dir, args.seq_len)
    n_features = len(features)
    print(f"   {X_flat.shape[0]} rows, {n_features} features")

    X_windows, y_windows = build_windows(X_flat, y_flat, args.seq_len, args.n_samples)
    print(f"   Windows sampled: {X_windows.shape}")

    # ---- Build and load model ----------------------------------------------
    print("\n🔧 Building model...")
    model = build_model(n_features, args.seq_len, args.d_model, args.n_heads, args.dropout)
    _ = model(tf.zeros((1, args.seq_len, n_features), dtype=tf.float32), training=False)
    weights_path = os.path.join(args.results_dir, "checkpoints", args.weights_file)
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        return
    model.load_weights(weights_path)
    print(f"   ✅ Loaded {weights_path}  ({model.count_params():,} parameters)")

    # ---- Integrated gradients ----------------------------------------------
    print(f"\n🔍 Computing integrated gradients (zero baseline)...")
    ig_list = compute_integrated_gradients(
        model, X_windows, baseline=None, steps=args.steps)
    print("   ✅ Done")

    # ---- Per-head results --------------------------------------------------
    attn_path = os.path.join(args.results_dir, "attention_maps_tft_run1.json")
    for head_idx, (ig, tname) in enumerate(zip(ig_list, TARGET_NAMES)):
        print(f"\n{'=' * 70}")
        print(f"🎯 {tname} — top {args.top_k} (feature, timestep) pairs by |attribution|")
        print(f"{'=' * 70}")
        print_top_pairs(ig, features, args.seq_len, top_k=args.top_k)

        print(f"\n--- Feature attribution (summed over all timesteps) ---")
        print_feature_summary(ig, features)

        print(f"\n--- Timestep attribution (summed over all features, 10-min buckets) ---")
        print_timestep_summary(ig, args.seq_len, bucket=10)

    # ---- IG vs attention comparison ----------------------------------------
    print(f"\n{'=' * 70}")
    print("📊 IG timestep attribution vs attention weights (summed over features)")
    print(f"{'=' * 70}")
    compare_with_attention(ig_list, args.seq_len, attn_path)

    # ---- Save full attribution matrix --------------------------------------
    output_path = args.output or os.path.join(args.results_dir, "integrated_gradients.json")
    out = {
        "description": "Integrated gradients (feature × timestep) for Model 5c TFT",
        "results_dir": args.results_dir,
        "n_samples": int(X_windows.shape[0]),
        "steps": args.steps,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "features": features,
        "attributions": {
            TARGET_NAMES[h]: ig_list[h].tolist() for h in range(3)
        },
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Full attribution matrix saved → {output_path}")
    print(f"   Shape per head: ({args.seq_len}, {n_features}) — rows=timesteps, cols=features")


if __name__ == "__main__":
    main()
