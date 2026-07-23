import tensorflow as tf

# Model 5c Track A: TFT-Based Feature Discovery
# No Edge TPU constraint. Float32 (or FP16) TFLite for Pi CPU deployment.
# Target: beat Model 5b Exp32 float 30-day StdDev of 0.607°C
#         and val_loss < 0.000373 (beat Model 5a clean dense_wide_run1)

# Set to True to use Kaggle input paths instead of local relative paths
KAGGLE_MODE = False
KAGGLE_DATASET = "datasets/dacarson/weatherml-training-data"
# Set to the checkpoint dataset path to resume from a previous Kaggle run, or "" to start fresh.
# Checkpoint dataset published from a previous run's Output tab.
# Format: "datasets/dacarson/<dataset-slug>"  (same prefix as KAGGLE_DATASET above)
# This resolves to /kaggle/input/datasets/dacarson/<slug>/checkpoints/
KAGGLE_CHECKPOINT_DATASET = ""  # start fresh — SEQ_LEN changed from 180→360, incompatible with prior checkpoints
# Set to "" — checkpoints/ is at the root of the published dataset (no wrapping subfolder).
KAGGLE_CHECKPOINT_SUBDIR = ""

# All output files go here.
# Mac local run: use a dated/named directory so Kaggle results aren't overwritten.
# Kaggle: /kaggle/working
RESULTS_DIR = "/kaggle/working" if KAGGLE_MODE else "./results_5c_track_a_mac_run1"

# TFT hyperparameters — Track A deep: SEQ_LEN=360, fresh start
D_MODEL = 128       # embedding / hidden dimension throughout TFT
N_HEADS = 8         # temporal self-attention heads
DROPOUT_RATE = 0.1  # GRN / attention dropout during training
L2_REG = 1e-4       # L2 regularisation on Dense weights

# Track A: float32 TFLite (no INT8 quantization — no Edge TPU constraint)
# True = FP16 (~half size), False = FP32 (exact, larger)
TFLITE_FLOAT16 = False

# Mixed precision: fp16 compute with fp32 master weights.
# Enabled only on local macOS Metal GPU — unstable on Kaggle T4/CUDA for this TFT
# (LayerNorm + attention softmax → NaN). Kaggle always trains float32.
MIXED_PRECISION = True

try:
    _register_keras_serializable = tf.keras.saving.register_keras_serializable
except AttributeError:
    _register_keras_serializable = tf.keras.utils.register_keras_serializable


@_register_keras_serializable(package="WeatherML5C")
class GatedResidualNetwork(tf.keras.layers.Layer):
    """GRN from Lim et al. TFT 2019: ELU → Dense → GLU (sigmoid gate) → LayerNorm + skip.

    Core building block used in the VSN weight computation and post-LSTM processing.
    """

    def __init__(self, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.dense_elu = tf.keras.layers.Dense(units, activation="elu",
                                               kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.dense_linear = tf.keras.layers.Dense(units,
                                                   kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.dense_gate = tf.keras.layers.Dense(units, activation="sigmoid",
                                                kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.skip_proj = None  # built lazily in build() if input_dim != units

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.skip_proj = tf.keras.layers.Dense(self.units, use_bias=False,
                                                   kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        super().build(input_shape)

    def call(self, x, training=None):
        skip = self.skip_proj(x) if self.skip_proj is not None else x
        h = self.dense_elu(x)
        h = self.dropout_layer(h, training=training)
        gated = self.dense_linear(h) * self.dense_gate(h)
        return self.layer_norm(skip + gated)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout": self.dropout_rate})
        return config


@_register_keras_serializable(package="WeatherML5C")
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """Adds fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Injects position information into the VSN output so the attention heads can
    distinguish t=-1 from t=-360 without an LSTM. No trainable parameters.
    The encoding table is precomputed once at construction time.
    """

    def __init__(self, d_model, max_len=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        import numpy as _np
        pe = _np.zeros((max_len, d_model), dtype=_np.float32)
        pos = _np.arange(max_len, dtype=_np.float32)[:, _np.newaxis]
        div = _np.exp(_np.arange(0, d_model, 2, dtype=_np.float32) * -(_np.log(10000.0) / d_model))
        pe[:, 0::2] = _np.sin(pos * div)
        pe[:, 1::2] = _np.cos(pos * div[:d_model // 2])
        # (1, max_len, d_model) constant — broadcast over batch
        self._pe = tf.constant(pe[_np.newaxis], dtype=tf.float32)

    def call(self, x):
        return x + tf.cast(self._pe[:, :tf.shape(x)[1], :], x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config


@_register_keras_serializable(package="WeatherML5C")
class VariableSelectionNetwork(tf.keras.layers.Layer):
    """VSN: learned per-timestep variable importance weights (Lim et al. 2019, Sec. 4.2).

    Inputs:  x  (batch, seq, n_features) — one scaled scalar per feature per timestep
    Outputs: selected  (batch, seq, d_model) — variable-weighted embedding
             weights   (batch, seq, n_features) — softmax selection weights (interpretability)

    The selection weights, averaged over time and samples, are the primary feature
    importance signal for Track B feature extraction.

    Feature embeddings use a single einsum instead of n_features separate Dense calls.
    Both approaches are mathematically identical (each feature has its own kernel row),
    but the einsum dispatches one GPU op instead of n_features small ops, which prevents
    the GPU from stalling on Python dispatch overhead between each Dense call.
    """

    def __init__(self, n_features, d_model, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_model = d_model
        self.dropout_rate = dropout
        # embedding_kernel/bias registered via add_weight in build() so they are
        # tracked as this layer's own weights (not as a sub-layer).
        self.embedding_kernel = None
        self.embedding_bias = None
        self.weight_grn = GatedResidualNetwork(n_features, dropout=dropout, name="vsn_weight_grn")
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        # (n_features, d_model): one projection vector per feature — same parameter count
        # as n_features separate Dense(d_model) layers with kernel shape (1, d_model).
        self.embedding_kernel = self.add_weight(
            name="embedding_kernel",
            shape=(self.n_features, self.d_model),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.l2(L2_REG),
        )
        self.embedding_bias = self.add_weight(
            name="embedding_bias",
            shape=(self.n_features, self.d_model),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x, training=None):
        # Selection weights from raw input: (batch, seq, n_features)
        weights = self.softmax(self.weight_grn(x, training=training))

        # Batched per-feature embedding via einsum — one GPU op for all features.
        # einsum 'bsf,fd->bsfd': for each (b,s,f,d): x[b,s,f] * kernel[f,d]
        # Equivalent to n_features separate Dense(d_model) calls but dispatched as one op.
        embedded = tf.nn.elu(
            tf.einsum("bsf,fd->bsfd", x, self.embedding_kernel) + self.embedding_bias
        )  # (batch, seq, n_features, d_model)

        # Weighted sum over feature axis: (batch, seq, d_model)
        selected = tf.reduce_sum(embedded * weights[..., tf.newaxis], axis=-2)
        return self.layer_norm(selected), weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features,
            "d_model": self.d_model,
            "dropout": self.dropout_rate,
        })
        return config


def main():
    """
    Main training function for Model 5c Track A (TFT).

    Architecture overview:
        Input (batch, SEQ_LEN, n_features)
        → VariableSelectionNetwork  → (batch, SEQ_LEN, D_MODEL)
        → Positional encoding       → (batch, SEQ_LEN, D_MODEL)
        → Multi-head self-attention → (batch, SEQ_LEN, D_MODEL)
        → GRN (post-attn)           → (batch, SEQ_LEN, D_MODEL)
        → GRN (feedforward)         → (batch, SEQ_LEN, D_MODEL)
        → last timestep             → (batch, D_MODEL)
        → 3 × Dense(1)              → diff_1hr, diff_2hr, diff_3hr

    Interpretability outputs (analysis_model only):
        vsn_weights  (batch, SEQ_LEN, n_features) — per-timestep feature importance
        attn_scores  (batch, N_HEADS, SEQ_LEN, SEQ_LEN) — temporal attention patterns
    """
    import multiprocessing as mp
    import multiprocessing

    try:
        mp.set_start_method("fork", force=True)
        print("ℹ️  using multiprocessing start method 'fork'")
    except RuntimeError as e:
        print(f"⚠️  Could not set multiprocessing start method to 'fork' (already set?): {e}")

    import os

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"ℹ️  Results directory: {RESULTS_DIR}")

    import sys
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"

    if force_cpu:
        tf.config.set_visible_devices([], 'GPU')
        print("ℹ️  FORCE_CPU=1: GPU disabled, using CPU only")
    else:
        print("ℹ️  macOS: using asynchronous execution (GPU pipeline enabled)")

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0 and not force_cpu:
        print(f"✅ GPU detected: {len(physical_devices)} device(s)")
        for device in physical_devices:
            print(f"   - {device.name}")
        gpu_memory_mb = os.environ.get("GPU_MEMORY_MB")
        try:
            if gpu_memory_mb:
                limit_mb = int(gpu_memory_mb)
                for device in physical_devices:
                    tf.config.experimental.set_virtual_device_configuration(
                        device,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit_mb)]
                    )
                print(f"ℹ️  GPU_MEMORY_MB={limit_mb}: GPU memory capped at {limit_mb} MB")
            else:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("ℹ️  GPU memory growth enabled (set GPU_MEMORY_MB=<MB> to cap for WindowServer)")
        except RuntimeError as e:
            print(f"⚠️  Could not configure GPU memory: {e}")
        cores = multiprocessing.cpu_count()
        intra_op_threads = max(4, cores // 2)
        inter_op_threads = 2
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        print(f"Using {intra_op_threads} intra_op and {inter_op_threads} inter_op threads "
              f"(Metal: low inter_op avoids command queue contention)")
    else:
        print("⚠️  No GPU detected, using CPU")
        cores = multiprocessing.cpu_count()
        intra_op_threads = int(os.environ.get("TF_NUM_INTRAOP_THREADS", max(4, cores - 4)))
        inter_op_threads = int(os.environ.get("TF_NUM_INTEROP_THREADS", 2))
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        print(f"Using {intra_op_threads} intra_op and {inter_op_threads} inter_op threads")

    tf.config.set_soft_device_placement(True)

    if KAGGLE_MODE or force_cpu:
        tf.config.optimizer.set_jit(True)
        backend = "CUDA/T4" if KAGGLE_MODE else "CPU"
        print(f"ℹ️  XLA JIT enabled ({backend})")
    else:
        tf.config.optimizer.set_jit(False)  # XLA not supported on Metal backend

    from tensorflow.keras.callbacks import EarlyStopping, Callback
    import numpy as np
    import pandas as pd
    import threading
    import time
    import copy
    import json
    import glob
    import re
    import shutil

    # Mixed precision: Metal GPU only. Kaggle T4/CUDA and CPU use float32 (fp16 → NaN on TFT).
    _on_metal_gpu = (
        MIXED_PRECISION
        and not KAGGLE_MODE
        and sys.platform == "darwin"
        and len(physical_devices) > 0
        and not force_cpu
    )
    _use_mixed_precision = _on_metal_gpu
    if _use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("ℹ️  Mixed precision enabled: fp16 compute, fp32 master weights (Metal GPU)")
        print("   Fixed loss scale (2^12) + clipnorm=0.5 guards against NaN overflow")
    elif KAGGLE_MODE:
        print("ℹ️  Mixed precision disabled on Kaggle (float32 — fp16 unstable for TFT on T4)")
    else:
        print("ℹ️  Mixed precision disabled (float32 throughout)")

    if KAGGLE_MODE and KAGGLE_CHECKPOINT_DATASET:
        _dataset_root = f"/kaggle/input/{KAGGLE_CHECKPOINT_DATASET}"
        if KAGGLE_CHECKPOINT_SUBDIR:
            _resume_src = os.path.join(_dataset_root, KAGGLE_CHECKPOINT_SUBDIR, "checkpoints")
        else:
            _resume_src = os.path.join(_dataset_root, "checkpoints")
        if os.path.exists(_resume_src):
            _restore_checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
            os.makedirs(_restore_checkpoint_dir, exist_ok=True)
            for _fname in os.listdir(_resume_src):
                shutil.copy(os.path.join(_resume_src, _fname), _restore_checkpoint_dir)
            _epoch_file = os.path.join(_restore_checkpoint_dir, "model_latest_epoch.json")
            if os.path.exists(_epoch_file):
                try:
                    with open(_epoch_file) as _f:
                        _saved_epoch = json.load(_f).get("epoch", 0)
                    print(f"✅ Checkpoints restored from {_resume_src} — resume from epoch {_saved_epoch + 1}")
                except Exception as _e:
                    print(f"✅ Checkpoints restored from {_resume_src} (could not read epoch: {_e})")
            else:
                print(f"✅ Checkpoints restored from {_resume_src} — no epoch state found, starting fresh")
        else:
            print(f"⚠️  KAGGLE_CHECKPOINT_DATASET set but path not found: {_resume_src}")

    # -------------------------------------------------------------------------
    # Load preprocessed data
    # -------------------------------------------------------------------------
    if KAGGLE_MODE:
        data_dir = f"/kaggle/input/{KAGGLE_DATASET}"
    else:
        data_dir = ".."
    train_df = pd.read_csv(f"{data_dir}/train_data_sf.csv")
    val_df = pd.read_csv(f"{data_dir}/val_data_sf.csv")

    # The raw CSVs ship with pre-baked temp_t+1hr/2hr/3hr columns from an earlier export
    # pipeline. _add_future_targets() short-circuits (returns df unchanged) whenever those
    # columns already exist, so keeping them means the sanity filter below (and the
    # gap-aware merge_asof reconstruction in _add_future_targets/_invalidate_targets_
    # crossing_gaps) never actually run — targets stay locked to whatever that external
    # pipeline computed, including any of its own bad readings. Drop them so targets are
    # always freshly derived from (now sanity-filtered) 'temperature'.
    _stale_target_cols = ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]
    train_df = train_df.drop(columns=[c for c in _stale_target_cols if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in _stale_target_cols if c in val_df.columns])

    def _prepare_time_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
        time_col = None
        for c in ("time", "timestamp", "ts", "datetime", "date"):
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            return df
        df = df.copy()
        s = df[time_col]
        if np.issubdtype(s.dtype, np.number):
            v = float(np.nanmax(s.to_numpy(dtype=np.float64)))
            if v >= 1e17:
                unit = "ns"
            elif v >= 1e14:
                unit = "us"
            elif v >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            df[time_col] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        else:
            df[time_col] = pd.to_datetime(s, utc=True, errors="coerce")
        if df[time_col].isna().any():
            bad = int(df[time_col].isna().sum())
            raise ValueError(f"{label}: failed to parse {bad} timestamps from column '{time_col}'.")
        df = df.set_index(time_col).sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        return df

    def _sanity_filter_temperature(df: pd.DataFrame, label: str, window: str = "31min",
                                    threshold_c: float = 6.0) -> pd.DataFrame:
        # Raw station data contains rare sensor-glitch rows (brief dropout/self-heating,
        # not real weather) where temperature jumps implausibly fast and reverts a few
        # minutes later — e.g. 2023-01-23 09:18 UTC: 11.1°C -> -7.5°C -> 9.9°C with
        # relative_humidity simultaneously collapsing to 0%. These poisoned the
        # temp_diff_Nhr targets (e.g. -31.5°C global min) since diffs are computed via
        # merge_asof against raw 'temperature'. Null any reading that deviates from its
        # local (time-centered) median by more than threshold_c; downstream dropna
        # removes the row entirely. Found via `git log` discussion 2026-07-19.
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        df = df.copy()
        local_median = df["temperature"].rolling(window, center=True, min_periods=3).median()
        spike = (df["temperature"] - local_median).abs() > threshold_c
        n_spikes = int(spike.sum())
        if n_spikes:
            df.loc[spike, "temperature"] = np.nan
            print(f"⚠️  {label}: nulled {n_spikes} temperature sensor-glitch rows "
                  f"(>{threshold_c}°C from local {window} median)")
        return df

    def _add_future_targets(df: pd.DataFrame, label: str, tolerance_s: int = 90) -> pd.DataFrame:
        if all(col in df.columns for col in ["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]):
            return df
        if "temperature" not in df.columns:
            raise ValueError(f"{label}: missing required column 'temperature'.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{label}: index must be DatetimeIndex for timestamp-based target construction.")
        base = df.reset_index()
        if "time" not in base.columns:
            base = base.rename(columns={base.columns[0]: "time"})
        base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")
        if base["time"].isna().any():
            bad = int(base["time"].isna().sum())
            raise ValueError(f"{label}: failed to parse {bad} timestamps after reset_index().")
        base = base.sort_values("time").reset_index(drop=True)
        base["row_id"] = np.arange(len(base), dtype=np.int64)
        src = base[["time", "temperature"]].copy().sort_values("time")
        src = src.rename(columns={"temperature": "temperature_future"})
        tol = pd.Timedelta(seconds=int(tolerance_s))
        for mins, col in ((60, "temp_t+1hr"), (120, "temp_t+2hr"), (180, "temp_t+3hr")):
            want = base[["row_id", "time"]].copy()
            want["t_query"] = want["time"] + pd.Timedelta(minutes=int(mins))
            merged = pd.merge_asof(
                want.sort_values("t_query"), src,
                left_on="t_query", right_on="time",
                direction="forward", tolerance=tol,
            )
            merged = merged.sort_values("row_id")
            base[col] = merged["temperature_future"].to_numpy()
            if base[col].isna().all():
                tmin = base["time"].min()
                tmax = base["time"].max()
                raise ValueError(
                    f"{label}: all values for {col} are NaN after merge_asof. "
                    f"base_time=[{tmin}..{tmax}] tol={tol}"
                )
        missing = base[["temp_t+1hr", "temp_t+2hr", "temp_t+3hr"]].isna().sum()
        print(f"\n❓ Missing target counts (timestamp lookup) for {label}:")
        print(missing)
        base = base.drop(columns=["row_id"]).set_index("time")
        return base

    def _invalidate_targets_crossing_gaps(df: pd.DataFrame, label: str, tol_s: int = 90) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        target_horizons = {60: "temp_t+1hr", 120: "temp_t+2hr", 180: "temp_t+3hr"}
        present = {h: c for h, c in target_horizons.items() if c in df.columns}
        if not present:
            return df
        dt_s = df.index.to_series().diff().dt.total_seconds()
        gap_positions = np.flatnonzero((dt_s > float(tol_s)).to_numpy())
        if gap_positions.size == 0:
            return df
        df = df.copy()
        n_nulled = 0
        for pos in gap_positions:
            if pos == 0:
                continue
            gap_boundary = df.index[pos - 1]
            for h_min, col in present.items():
                cutoff = gap_boundary - pd.Timedelta(minutes=h_min)
                mask = (df.index > cutoff) & (df.index <= gap_boundary)
                n = int(mask.sum())
                if n > 0:
                    df.loc[mask, col] = np.nan
                    n_nulled += n
        if n_nulled > 0:
            print(f"\n⚠️  {label}: nulled {n_nulled} cross-gap target lookups across "
                  f"{gap_positions.size} gap(s)")
        else:
            print(f"\n✅ {label}: no cross-gap target contamination detected")
        return df

    train_df = _prepare_time_index(train_df, "train_df")
    val_df = _prepare_time_index(val_df, "val_df")
    train_df = _sanity_filter_temperature(train_df, "train_df")
    val_df = _sanity_filter_temperature(val_df, "val_df")
    train_df = _add_future_targets(train_df, "train_df")
    val_df = _add_future_targets(val_df, "val_df")
    train_df = _invalidate_targets_crossing_gaps(train_df, "train_df", tol_s=600)
    val_df = _invalidate_targets_crossing_gaps(val_df, "val_df", tol_s=600)

    # -------------------------------------------------------------------------
    # Cyclical encodings (identical to 5b)
    # -------------------------------------------------------------------------
    for df in (train_df, val_df):
        df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
        df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)
        df['time_of_day_sin2'] = np.sin(4 * np.pi * df['time_of_day'] / 24.0)
        df['time_of_day_cos2'] = np.cos(4 * np.pi * df['time_of_day'] / 24.0)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        if 'wind_direction' in df.columns:
            df['wind_direction_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360.0)
            df['wind_direction_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360.0)

    # -------------------------------------------------------------------------
    # Rolling slope features (identical to 5b)
    # -------------------------------------------------------------------------
    def rolling_slope_numba(data, window):
        """Vectorised rolling linear-regression slope (NumPy only)."""
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

    print("⚙️  Computing rolling slope features...")
    for df in (train_df, val_df):
        df['temp_slope_15'] = rolling_slope_numba(df['temperature'].values, 15)
        df['temp_slope_30'] = rolling_slope_numba(df['temperature'].values, 30)
        df['temp_slope_60'] = rolling_slope_numba(df['temperature'].values, 60)
        df['solar_slope_30'] = rolling_slope_numba(df['solar_radiation'].values, 30)
        df['humidity_slope_30'] = rolling_slope_numba(df['relative_humidity'].values, 30)
        df['pressure_slope_60'] = rolling_slope_numba(df['station_pressure'].values, 60)
    print("   ✅ Slope features computed")

    # -------------------------------------------------------------------------
    # Targets: predict temperature change (diff), not absolute temperature
    # -------------------------------------------------------------------------
    for df in (train_df, val_df):
        df['temp_diff_1hr'] = df['temp_t+1hr'] - df['temperature']
        df['temp_diff_2hr'] = df['temp_t+2hr'] - df['temperature']
        df['temp_diff_3hr'] = df['temp_t+3hr'] - df['temperature']

    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    # -------------------------------------------------------------------------
    # Gap-aware window safety (identical to 5b)
    # -------------------------------------------------------------------------
    SEQ_LEN = 360  # 6-hour history window (Track A deep — probing beyond t-179 boundary)
    GAP_STEP_TOLERANCE_S = 90
    # Kaggle keeps stride=1 (maximize samples for final training).
    # Mac local: stride=10 cuts epochs from ~106 min to ~10 min.
    # Consecutive windows at stride=1 overlap by 99.7% — negligible diversity loss.
    SEQUENCE_STRIDE = 1 if KAGGLE_MODE else 10

    def _apply_gap_safety(df: pd.DataFrame, label: str, seq_len: int, max_step_s: int) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"\n⚠️  {label}: no DatetimeIndex found, skipping gap-aware window safety")
            return df
        dt_s = df.index.to_series().diff().dt.total_seconds()
        gap_positions = np.flatnonzero((dt_s > float(max_step_s)).to_numpy())
        if gap_positions.size == 0:
            return df
        print(f"\n🔍 Gap detection in {label}: found {gap_positions.size} gap(s) > {max_step_s}s")
        if gap_positions.size <= 10:
            for pos in gap_positions[:10]:
                gap_time = df.index[pos]
                gap_size = dt_s.iloc[pos]
                print(f"   Gap at {gap_time}: {gap_size:.1f}s")
        keep = np.ones(len(df), dtype=bool)
        drop_span = max(int(seq_len // 2), 0)
        for pos in gap_positions:
            start = int(pos)
            end = min(int(pos) + drop_span, len(df))
            keep[start:end] = False
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"🧩 Gap safety: dropping {dropped} rows in {label} "
                  f"(tol={max_step_s}s, drop_span={drop_span}).")
            df = df.iloc[keep]
        return df

    train_df = _apply_gap_safety(train_df, "train_df", SEQ_LEN, GAP_STEP_TOLERANCE_S)
    val_df = _apply_gap_safety(val_df, "val_df", SEQ_LEN, GAP_STEP_TOLERANCE_S)

    # -------------------------------------------------------------------------
    # Feature lists — Track A
    # No explicit lag scalars: TFT attends to the raw 360-step sequence and
    # discovers which past positions matter. The attention maps and VSN weights
    # from this run feed directly into Track B feature engineering.
    # -------------------------------------------------------------------------
    core_features = [
        'temperature',
        'uv', 'wind_avg', 'wind_gust',
        'solar_radiation', 'illuminance',
        'relative_humidity', 'station_pressure',
        'day_of_year_sin', 'day_of_year_cos',
    ]
    if 'wind_direction_sin' in train_df.columns:
        core_features.extend(['wind_direction_sin', 'wind_direction_cos'])
    if 'wind_lull' in train_df.columns:
        core_features.append('wind_lull')
    if 'rain_accumulated' in train_df.columns:
        core_features.append('rain_accumulated')

    cyclical_features = [
        'time_of_day_sin', 'time_of_day_cos', 'time_of_day_sin2', 'time_of_day_cos2',
    ]

    # Slope features kept: they are compressed trend summaries (local linear regression)
    # rather than raw past values, and are harder for attention to compute implicitly.
    slope_features = [
        'temp_slope_15', 'temp_slope_30', 'temp_slope_60',
        'solar_slope_30', 'humidity_slope_30', 'pressure_slope_60',
    ]

    features = core_features + cyclical_features + slope_features
    targets = ['temp_diff_1hr', 'temp_diff_2hr', 'temp_diff_3hr']

    print(f"\nFeature set: {len(features)} features")
    for f in features:
        print(f"  {f}")

    # -------------------------------------------------------------------------
    # Per-feature min/max scaling with domain bounds and ±5% padding
    # -------------------------------------------------------------------------
    domain_bounds = {
        "temperature":        (-10, 55),
        "temp_slope_15":      (None, None),
        "temp_slope_30":      (None, None),
        "temp_slope_60":      (None, None),
        "solar_slope_30":     (None, None),
        "humidity_slope_30":  (None, None),
        "pressure_slope_60":  (None, None),
        "wind_gust":          (0, 40),
        "wind_avg":           (0, 30),
        "uv":                 (0, None),
        "solar_radiation":    (0, None),
        "illuminance":        (0, None),
        "relative_humidity":  (0, 100),
        "station_pressure":   (None, None),
        "day_of_year_sin":    (-1, 1),
        "day_of_year_cos":    (-1, 1),
        "time_of_day_sin":    (-1, 1),
        "time_of_day_cos":    (-1, 1),
        "time_of_day_sin2":   (-1, 1),
        "time_of_day_cos2":   (-1, 1),
        "wind_direction_sin": (-1, 1),
        "wind_direction_cos": (-1, 1),
        "wind_lull":          (0, None),
        "rain_accumulated":   (0, None),
    }

    X_train_df = train_df[features].copy()
    X_val_df = val_df[features].copy()
    input_scaler = {}
    for feature in features:
        f_min = train_df[feature].min()
        f_max = train_df[feature].max()
        range_pad = 0.05 * (f_max - f_min)
        floor, ceiling = domain_bounds.get(feature, (None, None))
        f_min_adj = floor if floor is not None else f_min - range_pad
        f_max_adj = ceiling if ceiling is not None else f_max + range_pad
        input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
        X_train_df[feature] = (X_train_df[feature] - f_min_adj) / (f_max_adj - f_min_adj)
        X_val_df[feature] = (X_val_df[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    X_train_flat = X_train_df.values
    X_val_flat = X_val_df.values
    with open(os.path.join(RESULTS_DIR, "input_scaler_5c.json"), "w") as f:
        json.dump(input_scaler, f, indent=2)

    # Global target scaling (Model 5a / 5b approach)
    raw_train_targets = train_df[targets].copy()
    raw_val_targets = val_df[targets].copy()
    TARGET_PAD_C = 2.0
    y_min = float(raw_train_targets.min().min()) - TARGET_PAD_C
    y_max = float(raw_train_targets.max().max()) + TARGET_PAD_C
    for t in targets:
        train_df[t] = 2.0 * (raw_train_targets[t] - y_min) / (y_max - y_min) - 1.0
        val_df[t] = 2.0 * (raw_val_targets[t] - y_min) / (y_max - y_min) - 1.0
    with open(os.path.join(RESULTS_DIR, "target_scaler_5c.json"), "w") as f:
        json.dump({"min": y_min, "max": y_max, "range": (y_min, y_max)}, f, indent=2)
    y_mins = {t: y_min for t in targets}
    y_maxs = {t: y_max for t in targets}

    print(f"\n=== SCALING BOUNDS ===")
    print(f"Global target range: {y_min:.2f}°C to {y_max:.2f}°C")
    for t in targets:
        raw_min = float(raw_train_targets[t].min())
        raw_max = float(raw_train_targets[t].max())
        print(f"  {t}: raw [{raw_min:.2f}, {raw_max:.2f}]°C")

    n_features = len(features)
    y_all_train = train_df[targets].values
    y_all_val = val_df[targets].values

    # -------------------------------------------------------------------------
    # Streaming tf.data datasets
    # -------------------------------------------------------------------------
    from tensorflow.keras.preprocessing import timeseries_dataset_from_array

    # Kaggle T4 (16GB VRAM): 512.  Mac Metal: 64 keeps GPU fed without OOM.
    # Increase to 128 on 32GB+ Macs if Activity Monitor shows GPU still under-utilized.
    # Set FORCE_CPU=1 if Metal hangs — CPU runs cleanly (hangs are Metal-specific).
    TRAIN_BATCH_SIZE = 512 if KAGGLE_MODE else 64
    VAL_BATCH_SIZE = 512 if KAGGLE_MODE else 64

    train_ds = timeseries_dataset_from_array(
        data=X_train_flat, targets=y_all_train,
        sequence_length=SEQ_LEN, sequence_stride=SEQUENCE_STRIDE, sampling_rate=1,
        batch_size=TRAIN_BATCH_SIZE, shuffle=True,
    )
    val_ds = timeseries_dataset_from_array(
        data=X_val_flat, targets=y_all_val,
        sequence_length=SEQ_LEN, sequence_stride=SEQUENCE_STRIDE, sampling_rate=1,
        batch_size=VAL_BATCH_SIZE, shuffle=False,
    )

    def split_targets(x, y):
        return x, (y[:, 0], y[:, 1], y[:, 2])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(split_targets, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(split_targets, num_parallel_calls=AUTOTUNE)

    train_steps = int(train_ds.cardinality().numpy())
    train_ds = train_ds.repeat()
    # AUTOTUNE everywhere: TF measures how fast the GPU drains the queue and
    # buffers just enough batches to keep it fed.  The fixed-16 cap we used
    # previously starved the GPU on Mac (small batches drain in <100 ms).
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    print(f"\nTraining: {train_steps} batches/epoch (batch_size={TRAIN_BATCH_SIZE})")
    print(f"Validation: {val_ds.cardinality().numpy()} batches")

    def build_sequence_data(X_flat, y, seq_len, max_samples=None):
        n_samples = X_flat.shape[0]
        windows, tgts = [], []
        start = seq_len - 1
        for i in range(start, n_samples):
            windows.append(X_flat[i - seq_len + 1:i + 1, :])
            tgts.append(y[i])
            if max_samples is not None and len(windows) >= max_samples:
                break
        return np.array(windows, dtype=np.float32), np.array(tgts, dtype=np.float32)

    X_val_small, y_val_small = build_sequence_data(X_val_flat, y_all_val, SEQ_LEN, max_samples=2000)
    print(f"Small validation set: X_val_small={X_val_small.shape}, y_val_small={y_val_small.shape}")

    # -------------------------------------------------------------------------
    # Training watchdog (identical to 5b)
    # -------------------------------------------------------------------------
    class TrainingWatchdog(Callback):
        def __init__(self, timeout_minutes=2, check_interval_seconds=30, verbose=1,
                     adaptive_timeout=True, train_ds=None, total_batches_hint=None):
            super().__init__()
            self.base_timeout_seconds = timeout_minutes * 60
            self.timeout_seconds = self.base_timeout_seconds
            self.check_interval = check_interval_seconds
            self.verbose = verbose
            self.adaptive_timeout = adaptive_timeout
            self.train_ds = train_ds
            self._total_batches_hint = total_batches_hint
            self.last_activity_time = None
            self.current_epoch = None
            self.current_batch = None
            self.total_batches = None
            self.epoch_start_time = None
            self.last_batch_time = None
            self.last_activity_type = None
            self.batch_durations = []
            self.batch_update_frequency = 5
            self.watchdog_thread = None
            self.stop_watchdog = threading.Event()
            self.hang_detected = threading.Event()
            self.hang_info = None

        def on_train_begin(self, logs=None):
            self.last_activity_time = time.time()
            self.epoch_start_time = time.time()
            self.stop_watchdog.clear()
            self.hang_detected.clear()
            self.hang_info = None
            self.total_batches = None
            try:
                if self.train_ds is not None:
                    cardinality_val = self.train_ds.cardinality().numpy()
                    if cardinality_val >= 0:
                        self.total_batches = int(cardinality_val)
            except Exception:
                pass
            if self.total_batches is None and self._total_batches_hint:
                self.total_batches = self._total_batches_hint
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="TrainingWatchdog")
            self.watchdog_thread.start()
            if self.verbose >= 1:
                print(f"🛡️  Training watchdog started (timeout: {self.base_timeout_seconds/60:.1f} min adaptive)")

        def on_train_end(self, logs=None):
            if self.watchdog_thread is not None:
                self.stop_watchdog.set()
                self.watchdog_thread.join(timeout=5.0)
                if self.verbose >= 1:
                    print("🛡️  Training watchdog stopped")

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch
            self.current_batch = 0
            self.epoch_start_time = time.time()
            self.last_activity_time = time.time()
            self.last_activity_type = 'epoch_start'
            self.batch_durations = []

        def on_epoch_end(self, epoch, logs=None):
            self.last_activity_time = time.time()
            self.last_activity_type = 'epoch_end'
            if self.total_batches is None and self.current_batch is not None:
                self.total_batches = self.current_batch + 1

        def on_train_batch_begin(self, batch, logs=None):
            self.current_batch = batch
            self.last_batch_time = time.time()
            should_update = False
            if self.total_batches:
                remaining = self.total_batches - batch
                should_update = (remaining <= 100 and remaining % 10 == 0) or \
                                (remaining > 100 and batch % self.batch_update_frequency == 0)
            else:
                should_update = (batch % self.batch_update_frequency == 0)
            if should_update:
                self.last_activity_time = time.time()
                self.last_activity_type = 'batch_start'

        def on_train_batch_end(self, batch, logs=None):
            batch_duration = time.time() - self.last_batch_time if self.last_batch_time else 0
            self.batch_durations.append(batch_duration)
            if len(self.batch_durations) > 200:
                self.batch_durations = self.batch_durations[-200:]
            if self.adaptive_timeout and len(self.batch_durations) >= 10:
                avg_batch_time = sum(self.batch_durations) / len(self.batch_durations)
                batches_until_update = self.batch_update_frequency
                if self.total_batches:
                    remaining = self.total_batches - batch
                    if remaining <= 100:
                        batches_until_update = 10 if remaining > 5 else 1
                expected_update_time = batches_until_update * avg_batch_time
                adaptive_timeout = max(expected_update_time * 5, self.base_timeout_seconds)
                if abs(adaptive_timeout - self.timeout_seconds) > 5:
                    self.timeout_seconds = adaptive_timeout
            should_update = False
            if self.total_batches:
                remaining = self.total_batches - batch
                should_update = (remaining <= 100 and remaining % 10 == 0) or \
                                (remaining > 100 and batch % self.batch_update_frequency == 0)
            else:
                should_update = (batch % self.batch_update_frequency == 0)
            if should_update:
                self.last_activity_time = time.time()
                self.last_activity_type = 'batch_end'

        def _watchdog_loop(self):
            while not self.stop_watchdog.is_set():
                if self.stop_watchdog.wait(self.check_interval):
                    break
                if self.last_activity_time is None:
                    continue
                time_since_activity = time.time() - self.last_activity_time
                if time_since_activity > self.timeout_seconds:
                    self.hang_info = {
                        'time_since_activity': time_since_activity,
                        'current_epoch': self.current_epoch,
                        'current_batch': self.current_batch,
                    }
                    self.hang_detected.set()
                    print(f"\n🚨 TRAINING HANG DETECTED: {time_since_activity/60:.1f} min since last activity")
                    try:
                        with open(os.path.join(RESULTS_DIR, "training_hang_detected.json"), "w") as f:
                            json.dump(self.hang_info, f, indent=2)
                    except Exception:
                        pass
                    print("🛑 Forcefully exiting process due to detected hang...")
                    os._exit(99)
                    break

    # -------------------------------------------------------------------------
    # Custom objects for checkpoint serialization
    # -------------------------------------------------------------------------
    CUSTOM_OBJECTS = {
        "GatedResidualNetwork": GatedResidualNetwork,
        "VariableSelectionNetwork": VariableSelectionNetwork,
    }

    def validate_checkpoint_loading(checkpoint_path, model, custom_objects):
        is_weights_only = checkpoint_path.endswith('.weights.h5')
        is_full_model = (os.path.isdir(checkpoint_path) or
                         (checkpoint_path.endswith('.h5') and not is_weights_only))
        if is_weights_only:
            import h5py
            try:
                with h5py.File(checkpoint_path, 'r') as f:
                    if not len(f.keys()):
                        return False, "Checkpoint HDF5 file has no content"
                return True, None
            except Exception as e:
                return False, f"HDF5 read error: {str(e)[:200]}"
        elif is_full_model:
            original_weights = None
            try:
                original_weights = [w.copy() for w in model.get_weights()]
            except Exception:
                pass
            try:
                test_model = tf.keras.models.load_model(
                    checkpoint_path, compile=False, custom_objects=custom_objects)
                test_weights = test_model.get_weights()
                del test_model
                if len(test_weights) != len(model.get_weights()):
                    return False, (f"Architecture mismatch: checkpoint has {len(test_weights)} "
                                   f"weight arrays, model has {len(model.get_weights())}")
                for i, (tw, mw) in enumerate(zip(test_weights, model.get_weights())):
                    if tw.shape != mw.shape:
                        return False, f"Architecture mismatch at layer {i}: {tw.shape} != {mw.shape}"
                return True, None
            except Exception as e:
                if original_weights:
                    try:
                        model.set_weights(original_weights)
                    except Exception:
                        pass
                return False, f"Checkpoint loading error: {str(e)[:200]}"
        else:
            return False, f"Unknown checkpoint format: {checkpoint_path}"

    # =========================================================================
    # Model build + training
    # =========================================================================
    def build_and_train_model(name):
        print(f"\n--- Running: {name} ---\n")

        # ---------------------------------------------------------------------
        # TFT model (functional API so attention weights are easily extractable)
        # ---------------------------------------------------------------------
        _n_gpus = len(tf.config.list_physical_devices('GPU'))
        strategy = tf.distribute.MirroredStrategy() if _n_gpus > 1 else tf.distribute.get_strategy()
        if _n_gpus > 1:
            print(f"Using MirroredStrategy across {_n_gpus} GPUs")

        with strategy.scope():
            input_layer = tf.keras.layers.Input(shape=(SEQ_LEN, n_features), name="input")

            # 1. Variable Selection Network — learns which features matter per timestep
            vsn_layer = VariableSelectionNetwork(
                n_features, D_MODEL, dropout=DROPOUT_RATE, name="vsn")
            vsn_out, vsn_weights = vsn_layer(input_layer)  # (batch, SEQ_LEN, D_MODEL)

            # 2. Sinusoidal positional encoding — tells attention heads which timestep is which.
            #    Replaces LSTM: LSTM processes 360 timesteps sequentially (360 serial GPU
            #    dispatches on Metal). Positional encoding is a single addition; the attention
            #    matrix then captures all temporal dependencies in parallel (one matmul).
            enc = SinusoidalPositionalEncoding(
                D_MODEL, max_len=SEQ_LEN + 1, name="pos_enc")(vsn_out)  # (batch, SEQ_LEN, D_MODEL)

            # 3. Temporal self-attention — learns non-uniform weighting over past timesteps.
            #    Unlike Conv2D+GAP, attention preserves positional information: a signal at
            #    t=−360 is not diluted by the other 359 positions.
            mha_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=N_HEADS, key_dim=D_MODEL // N_HEADS,
                dropout=DROPOUT_RATE, name="temporal_attention")
            attn_out, attn_scores = mha_layer(
                query=enc, key=enc, value=enc,
                return_attention_scores=True,
            )  # attn_out: (batch, SEQ_LEN, D_MODEL), attn_scores: (batch, N_HEADS, SEQ_LEN, SEQ_LEN)

            # 4. Gated residual around attention output
            grn_attn = GatedResidualNetwork(D_MODEL, dropout=DROPOUT_RATE, name="grn_post_attn")
            attn_out = grn_attn(enc + attn_out)  # residual before GRN

            # 5. Point-wise feedforward GRN
            grn_ff = GatedResidualNetwork(D_MODEL, dropout=DROPOUT_RATE, name="grn_feedforward")
            ff_out = grn_ff(attn_out)  # (batch, SEQ_LEN, D_MODEL)

            # 6. Extract final timestep for multi-horizon output
            #    ff_out[:, -1, :] is a STRIDED_SLICE — Edge TPU compatible if ever needed
            last_ts = ff_out[:, -1, :]  # (batch, D_MODEL)

            # 7. Three output heads (same targets as 5a/5b)
            output_1 = tf.keras.layers.Dense(
                1, activation='linear', use_bias=False, dtype="float32", name='diff_1hr')(last_ts)
            output_2 = tf.keras.layers.Dense(
                1, activation='linear', use_bias=False, dtype="float32", name='diff_2hr')(last_ts)
            output_3 = tf.keras.layers.Dense(
                1, activation='linear', use_bias=False, dtype="float32", name='diff_3hr')(last_ts)

            # Training model (predictions only)
            model = tf.keras.Model(
                inputs=input_layer,
                outputs=[output_1, output_2, output_3],
                name=f"tft_{name}",
            )

            initial_lr = 1e-4
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=initial_lr, clipnorm=0.5 if _use_mixed_precision else 1.0)
            if _use_mixed_precision:
                try:
                    # Fixed loss scale prevents the runaway dynamic scaling that caused
                    # NaN crashes in 5b Exp37 at epochs 38 and 73 on T4.
                    # dynamic=False: scale stays at 2^12=4096 throughout training.
                    # (Run 2 used 2^15=32768 + clipnorm=1.0 — still hit NaN at epoch 52.)
                    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                        optimizer, dynamic=False, initial_scale=2 ** 12)
                    print("   ✅ LossScaleOptimizer(dynamic=False, scale=2^12) applied")
                except Exception as e:
                    # Keras 3 integrates loss scaling internally — no wrapper needed
                    print(f"   ℹ️  LossScaleOptimizer not needed in this Keras version: {e}")
            # steps_per_execution>1 keeps the Metal command queue filled between
            # Python callbacks, reducing GPU idle time.  Batch-level callbacks
            # (watchdog, NaN terminator, progress) still fire — just every N steps.
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics={'diff_1hr': 'mae', 'diff_2hr': 'mae', 'diff_3hr': 'mae'},
                steps_per_execution=1 if KAGGLE_MODE else 10,
            )

        # Analysis model shares all weights with `model` — outputs attention maps for Track B
        try:
            analysis_model = tf.keras.Model(
                inputs=input_layer,
                outputs=[output_1, output_2, output_3, vsn_weights, attn_scores],
                name=f"tft_analysis_{name}",
            )
            print("✅ Analysis model created (outputs VSN weights + attention scores)")
        except Exception as e:
            print(f"⚠️  Could not create analysis model: {e}")
            analysis_model = None

        model.summary()

        # ---------------------------------------------------------------------
        # Checkpoint loading (identical to 5b)
        # ---------------------------------------------------------------------
        checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        es_state_path = os.path.join(checkpoint_dir, "early_stopping_state.json")
        lr_state_path = os.path.join(checkpoint_dir, "lr_state.json")
        initial_epoch = 0

        full_model_checkpoints = [
            f for f in glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.h5"))
            if not f.endswith('.weights.h5')
        ]
        weights_checkpoints = glob.glob(os.path.join(checkpoint_dir, "model_*.weights.h5"))
        best_weights_checkpoint = os.path.join(checkpoint_dir, "best_model.weights.h5")

        checkpoint_to_load = None
        latest_epoch = 0
        checkpoint_type = None

        if full_model_checkpoints:
            def extract_epoch_full(fpath):
                match = re.search(r'model_epoch_(\d+)(?:_batch_\d+)?\.h5', os.path.basename(fpath))
                return int(match.group(1)) if match else 0
            full_model_checkpoints.sort(key=extract_epoch_full, reverse=True)
            checkpoint_to_load = full_model_checkpoints[0]
            latest_epoch = extract_epoch_full(checkpoint_to_load)
            checkpoint_type = "full_model"
            print(f"\n🔍 Found {len(full_model_checkpoints)} full model checkpoint(s).")
        elif (os.path.exists(os.path.join(checkpoint_dir, "model_latest.weights.h5")) and
              os.path.exists(os.path.join(checkpoint_dir, "model_latest_epoch.json"))):
            checkpoint_to_load = os.path.join(checkpoint_dir, "model_latest.weights.h5")
            checkpoint_type = "latest_weights"
            try:
                with open(os.path.join(checkpoint_dir, "model_latest_epoch.json")) as f:
                    latest_epoch = int(json.load(f).get("epoch", 0))
            except Exception:
                latest_epoch = 0
            print(f"\n🔍 Found latest-epoch weights checkpoint (epoch {latest_epoch}).")
        elif os.path.exists(best_weights_checkpoint):
            checkpoint_to_load = best_weights_checkpoint
            checkpoint_type = "best_weights"
            print(f"\n🔍 Found best weights checkpoint.")

        if checkpoint_to_load:
            print(f"📥 Attempting to load {checkpoint_type} checkpoint from epoch {latest_epoch}...")
            try:
                if checkpoint_type == "full_model":
                    loaded_checkpoint = None
                    for candidate in full_model_checkpoints:
                        candidate_epoch = extract_epoch_full(candidate)
                        print(f"   🔍 Validating {os.path.basename(candidate)}...")
                        can_load, validation_error = validate_checkpoint_loading(
                            candidate, model, CUSTOM_OBJECTS)
                        if can_load:
                            loaded_checkpoint = candidate
                            latest_epoch = candidate_epoch
                            break
                        print(f"   ⚠️  Skipping epoch {candidate_epoch}: {validation_error}")
                    if loaded_checkpoint is None:
                        print(f"   ⚠️  No compatible checkpoint found. Starting from scratch.")
                        initial_epoch = 0
                    else:
                        loaded_model = tf.keras.models.load_model(
                            loaded_checkpoint, compile=False, custom_objects=CUSTOM_OBJECTS)
                        model.set_weights(loaded_model.get_weights())
                        del loaded_model
                        initial_epoch = latest_epoch
                        print(f"   ✅ Resumed from epoch {initial_epoch + 1}.")
                else:
                    model.load_weights(checkpoint_to_load)
                    initial_epoch = latest_epoch
                    print(f"✅ Loaded weights checkpoint from epoch {latest_epoch}")

                if initial_epoch > 0 and os.path.exists(lr_state_path):
                    try:
                        with open(lr_state_path) as f:
                            lr_state = json.load(f)
                        restored_lr = float(lr_state["lr"])
                        lr_var = getattr(optimizer, '_learning_rate', None)
                        if lr_var is not None and hasattr(lr_var, 'assign'):
                            lr_var.assign(restored_lr)
                        else:
                            optimizer.learning_rate = restored_lr
                        print(f"   ✅ Restored LR: {restored_lr:.2e}")
                    except Exception as e:
                        print(f"   ⚠️  Could not restore LR state: {e}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                print("   Starting from scratch...")
                initial_epoch = 0

        if initial_epoch == 0:
            for p in (es_state_path, lr_state_path,
                      os.path.join(checkpoint_dir, "model_latest.weights.h5"),
                      os.path.join(checkpoint_dir, "model_latest_epoch.json")):
                if os.path.exists(p):
                    os.remove(p)
                    print(f"   🗑️  Cleared stale state: {os.path.basename(p)}")

        # ---------------------------------------------------------------------
        # Pre-training validation
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("🔍 PRE-TRAINING VALIDATION")
        print("=" * 70)
        errors = []

        print("\n1️⃣  Model forward pass...")
        try:
            dummy = tf.zeros((1, SEQ_LEN, n_features), dtype=tf.float32)
            dummy_out = model(dummy, training=False)
            if not isinstance(dummy_out, (list, tuple)) or len(dummy_out) != 3:
                errors.append(f"Expected 3 outputs, got {len(dummy_out)}")
            else:
                print(f"   ✅ Output shapes: {[o.shape for o in dummy_out]}")
        except Exception as e:
            errors.append(f"Forward pass failed: {str(e)[:200]}")

        print("\n2️⃣  Dataset iteration...")
        try:
            sample_batch = next(iter(train_ds))
            x_s, y_s = sample_batch
            assert x_s.shape[1] == SEQ_LEN, f"seq_len mismatch: {x_s.shape[1]} != {SEQ_LEN}"
            assert x_s.shape[2] == n_features, f"n_features mismatch: {x_s.shape[2]} != {n_features}"
            x_nan = int(tf.reduce_sum(tf.cast(tf.math.is_nan(x_s), tf.int32)).numpy())
            x_inf = int(tf.reduce_sum(tf.cast(tf.math.is_inf(x_s), tf.int32)).numpy())
            if x_nan or x_inf:
                errors.append(f"Batch contains {x_nan} NaN and {x_inf} Inf feature values")
            print(f"   ✅ Batch shape: x={x_s.shape}, y={[t.shape for t in y_s]}")
        except Exception as e:
            errors.append(f"Dataset iteration failed: {str(e)[:200]}")

        print("\n3️⃣  Checkpoint directory writable...")
        try:
            test_file = os.path.join(checkpoint_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("   ✅ Checkpoint directory writable")
        except Exception as e:
            errors.append(f"Checkpoint directory not writable: {str(e)[:200]}")

        print("\n4️⃣  Analysis model check...")
        if analysis_model is not None:
            try:
                ana_out = analysis_model(dummy, training=False)
                print(f"   ✅ Analysis model outputs: {len(ana_out)} tensors")
                print(f"      vsn_weights: {ana_out[3].shape}")
                print(f"      attn_scores: {ana_out[4].shape}")
            except Exception as e:
                print(f"   ⚠️  Analysis model forward pass failed: {e} (non-critical)")
        else:
            print("   ⚠️  Analysis model unavailable (attention maps will be skipped)")

        # Forward pass alone does not compile the XLA training graph. One train step
        # here surfaces compile time and gives a per-step estimate before epoch 1.
        print("\n5️⃣  Warmup training step (XLA train-graph compile)...")
        try:
            warmup_t0 = time.time()
            warmup_batch = next(iter(train_ds.take(1)))
            warmup_x, warmup_y = warmup_batch
            warmup_loss = model.train_on_batch(warmup_x, warmup_y)
            warmup_s = time.time() - warmup_t0
            est_epoch_min = warmup_s * train_steps / 60.0
            warmup_loss_val = float(warmup_loss[0] if isinstance(warmup_loss, (list, tuple)) else warmup_loss)
            if np.isnan(warmup_loss_val) or np.isinf(warmup_loss_val):
                errors.append(f"Warmup training loss is {warmup_loss_val} — weights already corrupted")
            print(f"   ✅ Warmup step: {warmup_s:.1f}s  loss={warmup_loss_val:.6f}")
            print(f"   ℹ️  Estimated epoch: ~{est_epoch_min:.0f} min "
                  f"({train_steps} steps × {warmup_s:.1f}s/step)")
            if est_epoch_min > 120:
                print(f"   ⚠️  Long epoch expected — TFT attention at batch={TRAIN_BATCH_SIZE} "
                      f"is compute-heavy; progress logs appear every 50 steps.")
        except Exception as e:
            errors.append(f"Warmup training step failed: {str(e)[:200]}")

        print("\n" + "=" * 70)
        if errors:
            print("❌ PRE-TRAINING VALIDATION FAILED")
            for err in errors:
                print(f"  • {err}")
            print("=" * 70)
            raise RuntimeError(f"Pre-training validation failed: {'; '.join(errors[:3])}")
        else:
            print("✅ PRE-TRAINING VALIDATION PASSED")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Callbacks (identical pattern to 5b)
        # ---------------------------------------------------------------------
        early_stopping = EarlyStopping(
            monitor='val_task_loss', patience=30,
            restore_best_weights=True, mode='min')

        if initial_epoch > 0 and os.path.exists(es_state_path):
            try:
                with open(es_state_path) as f:
                    es_state = json.load(f)
                early_stopping.best = es_state["best"]
                early_stopping.wait = es_state["wait"]
                print(f"   ✅ Restored early stopping: best={es_state['best']:.6f}, "
                      f"wait={es_state['wait']}/{early_stopping.patience}")
            except Exception as e:
                print(f"   ⚠️  Could not restore early stopping state: {e}")

        class EarlyStoppingStateSaver(Callback):
            def __init__(self, es_cb, state_path):
                super().__init__()
                self.es = es_cb
                self.state_path = state_path

            def on_epoch_end(self, epoch, logs=None):
                try:
                    with open(self.state_path, "w") as f:
                        json.dump({"best": float(self.es.best), "wait": int(self.es.wait)}, f)
                except Exception as e:
                    print(f"\n⚠️  Could not save early stopping state: {e}")

        class LRStateSaver(Callback):
            def __init__(self, reduce_lr_cb, state_path):
                super().__init__()
                self.reduce_lr = reduce_lr_cb
                self.state_path = state_path

            def on_epoch_end(self, epoch, logs=None):
                try:
                    current_lr = self.reduce_lr._get_lr()
                    with open(self.state_path, "w") as f:
                        json.dump({
                            "lr": float(current_lr),
                            "best": float(self.reduce_lr.best),
                            "wait": int(self.reduce_lr.wait),
                        }, f)
                except Exception as e:
                    print(f"\n⚠️  Could not save LR state: {e}")

        class LatestEpochSaver(Callback):
            """One save_weights() per epoch; copies to best_model if val_task_loss improved."""
            def __init__(self, checkpoint_dir, initial_best_task_loss=float('inf')):
                super().__init__()
                self.weights_path = os.path.join(checkpoint_dir, "model_latest.weights.h5")
                self.meta_path = os.path.join(checkpoint_dir, "model_latest_epoch.json")
                self.best_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
                self.best_task_loss = initial_best_task_loss

            def on_epoch_end(self, epoch, logs=None):
                task_loss = (logs or {}).get("val_task_loss", float('inf'))
                if np.isnan(task_loss) or np.isinf(task_loss):
                    return  # Don't overwrite latest checkpoint with NaN weights
                try:
                    self.model.save_weights(self.weights_path)
                    with open(self.meta_path, "w") as f:
                        json.dump({"epoch": epoch + 1}, f)
                except Exception as e:
                    print(f"\n⚠️  Could not save latest-epoch checkpoint: {e}")
                    return
                if task_loss < self.best_task_loss:
                    self.best_task_loss = task_loss
                    try:
                        shutil.copy2(self.weights_path, self.best_path)
                    except Exception as e:
                        print(f"\n⚠️  Could not update best-model checkpoint: {e}")

        class MaxEpochsPerRun(Callback):
            """Clean process restart after N epochs to refresh Metal GPU context."""
            def __init__(self, max_epochs_per_run=1):
                super().__init__()
                self.max_epochs_per_run = max_epochs_per_run
                self.epochs_run = 0
                self.triggered = False

            def on_epoch_end(self, epoch, logs=None):
                self.epochs_run += 1
                if self.epochs_run >= self.max_epochs_per_run:
                    print(f"\n🔄 MaxEpochsPerRun: clean stop after {self.epochs_run} epoch(s).")
                    self.triggered = True
                    self.model.stop_training = True

        class EpochProgressCallback(Callback):
            """Prints a progress line every `log_every` steps within each epoch.

            verbose=2 gives one line per epoch with no within-epoch feedback.
            On long epochs this feels like a hang. This callback prints elapsed
            time, step count, and a moving-average loss every `log_every` steps
            without the I/O overhead of verbose=1's per-step progress bar.
            """
            def __init__(self, log_every=100, total_steps=None, max_epochs=450):
                super().__init__()
                self.log_every = log_every
                self.total_steps = total_steps
                self.max_epochs = max_epochs
                self._epoch_start = None
                self._loss_window = []

            def on_epoch_begin(self, epoch, logs=None):
                self._epoch_start = time.time()
                self._loss_window = []
                print(f"\nEpoch {epoch + 1}/{self.max_epochs}", flush=True)

            def on_train_batch_end(self, batch, logs=None):
                loss = (logs or {}).get('loss', 0.0)
                self._loss_window.append(float(loss))
                if len(self._loss_window) > self.log_every:
                    self._loss_window.pop(0)

                if (batch + 1) % self.log_every != 0:
                    return

                elapsed = time.time() - self._epoch_start
                avg_loss = sum(self._loss_window) / len(self._loss_window)
                step_str = f"{batch + 1}"
                if self.total_steps:
                    pct = (batch + 1) / self.total_steps * 100
                    step_str = f"{batch + 1}/{self.total_steps} ({pct:.0f}%)"
                    eta = elapsed / (batch + 1) * (self.total_steps - batch - 1)
                    time_str = f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
                else:
                    time_str = f"{elapsed:.0f}s elapsed"
                print(f"  step {step_str} — loss {avg_loss:.5f} — {time_str}", flush=True)

            def on_epoch_end(self, epoch, logs=None):
                elapsed = time.time() - self._epoch_start
                vtl = (logs or {}).get('val_task_loss', (logs or {}).get('val_loss', float('nan')))
                # MirroredStrategy wraps optimizer under '_optimizer'; LossScaleOptimizer
                # uses 'inner_optimizer'. Try both before falling back to logs.
                try:
                    opt = self.model.optimizer
                    for attr in ('_optimizer', 'inner_optimizer'):
                        inner = getattr(opt, attr, None)
                        if inner is not None:
                            opt = inner
                            break
                    lr_var = getattr(opt, '_learning_rate', None)
                    if lr_var is not None:
                        lr = float(lr_var.numpy()) if hasattr(lr_var, 'numpy') else float(lr_var)
                    else:
                        lr_attr = opt.learning_rate
                        lr = float(lr_attr.numpy()) if hasattr(lr_attr, 'numpy') else float(lr_attr)
                except Exception:
                    lr = (logs or {}).get('lr', (logs or {}).get('learning_rate', float('nan')))
                print(f"  → val_task_loss={vtl:.6f}  lr={lr:.2e}  epoch={elapsed:.0f}s",
                      flush=True)

        class TaskLossLogger(Callback):
            """val_task_loss = sum of per-head MSE (excludes L2 reg terms in val_loss)."""
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    return
                keys = ('val_diff_1hr_loss', 'val_diff_2hr_loss', 'val_diff_3hr_loss')
                if all(k in logs for k in keys):
                    logs['val_task_loss'] = sum(logs[k] for k in keys)
                else:
                    logs['val_task_loss'] = logs.get('val_loss', float('inf'))

        class NaNLossTerminator(Callback):
            """Stops training immediately when training loss is NaN/Inf."""
            def on_train_batch_end(self, batch, logs=None):
                loss = (logs or {}).get('loss', 0.0)
                if np.isnan(loss) or np.isinf(loss):
                    epoch = (self.params or {}).get('epoch', 0) + 1
                    print(f"\n🚨 NaN/Inf training loss at epoch {epoch} batch {batch + 1} — stopping")
                    self.model.stop_training = True

            def on_epoch_end(self, epoch, logs=None):
                loss = (logs or {}).get('loss', 0.0)
                if np.isnan(loss) or np.isinf(loss):
                    print(f"\n🚨 NaN/Inf training loss at epoch {epoch + 1} — stopping immediately")
                    self.model.stop_training = True

        class ReduceLRCallback(Callback):
            def __init__(self, initial_lr, monitor='val_task_loss', factor=0.5,
                         patience=3, min_lr=1e-7, min_delta=0.0, verbose=1):
                super().__init__()
                self.initial_lr = initial_lr
                self.monitor = monitor
                self.factor = factor
                self.patience = patience
                self.min_lr = min_lr
                self.min_delta = min_delta
                self.verbose = verbose
                self.best = float('inf')
                self.wait = 0

            def _unwrap_optimizer(self, opt):
                for attr in ('_optimizer', 'inner_optimizer'):
                    inner = getattr(opt, attr, None)
                    if inner is not None:
                        return inner
                return opt

            def _get_lr(self):
                opt = self.model.optimizer
                inner = self._unwrap_optimizer(opt)
                lr_var = getattr(inner, '_learning_rate', None)
                if lr_var is not None and hasattr(lr_var, 'numpy'):
                    return float(lr_var.numpy())
                try:
                    lr = inner.learning_rate
                    return float(lr.numpy()) if hasattr(lr, 'numpy') else float(lr)
                except Exception:
                    return float(self.initial_lr)

            def _set_lr(self, new_lr):
                opt = self.model.optimizer
                inner = self._unwrap_optimizer(opt)
                lr_var = getattr(inner, '_learning_rate', None)
                if lr_var is not None and hasattr(lr_var, 'assign'):
                    lr_var.assign(float(new_lr))
                else:
                    inner.learning_rate = float(new_lr)
                confirmed = self._get_lr()
                if abs(confirmed - float(new_lr)) > 1e-10:
                    print(f'\n⚠️  LR assignment may not have stuck: set {new_lr:.2e}, read back {confirmed:.2e}')
                else:
                    print(f'\n✅ LR confirmed: {confirmed:.2e}')

            def on_epoch_end(self, epoch, logs=None):
                current = logs.get(self.monitor)
                if current is None:
                    return
                current_lr = self._get_lr()
                if current < self.best - self.min_delta:
                    self.best = current
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        if current_lr > self.min_lr:
                            new_lr = max(current_lr * self.factor, self.min_lr)
                            self._set_lr(new_lr)
                            current_lr = new_lr
                            if self.verbose:
                                print(f'\nEpoch {epoch+1}: ReduceLR → {new_lr:.2e} '
                                      f'(wait={self.wait}, best={self.best:.6f})')
                        self.wait = 0
                if logs is not None:
                    logs['learning_rate'] = current_lr

        es_state_saver = EarlyStoppingStateSaver(early_stopping, es_state_path)
        nan_loss_terminator = NaNLossTerminator()
        reduce_lr = ReduceLRCallback(
            initial_lr=initial_lr, monitor='val_task_loss',
            factor=0.5, patience=12, min_lr=1e-7, min_delta=1e-5, verbose=1)
        lr_state_saver = LRStateSaver(reduce_lr, lr_state_path)

        if initial_epoch > 0 and os.path.exists(lr_state_path):
            try:
                with open(lr_state_path) as f:
                    lr_state = json.load(f)
                reduce_lr.best = float(lr_state["best"])
                reduce_lr.wait = int(lr_state["wait"])
                print(f"   ✅ Restored ReduceLR: best={reduce_lr.best:.6f}, wait={reduce_lr.wait}/{reduce_lr.patience}")
            except Exception as e:
                print(f"   ⚠️  Could not restore ReduceLR state: {e}")

        watchdog = TrainingWatchdog(
            timeout_minutes=30, check_interval_seconds=30, verbose=1,
            train_ds=train_ds, total_batches_hint=train_steps)

        _initial_best = getattr(early_stopping, 'best', float('inf'))
        if not isinstance(_initial_best, float) or _initial_best != _initial_best:
            _initial_best = float('inf')
        latest_epoch_saver = LatestEpochSaver(checkpoint_dir, initial_best_task_loss=_initial_best)
        task_loss_logger = TaskLossLogger()
        max_epochs_per_run = MaxEpochsPerRun(
            max_epochs_per_run=9999 if (KAGGLE_MODE or force_cpu) else 99)

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------
        max_epochs = 450
        history = None

        if initial_epoch >= max_epochs:
            print(f"\n✅ Training already complete (epoch {initial_epoch} >= max {max_epochs})")
            history = type('History', (), {'history': {}})()
        else:
            try:
                if initial_epoch > 0:
                    print(f"\n🔄 Resuming from epoch {initial_epoch + 1} (max: {max_epochs})")
                active_callbacks = [
                    task_loss_logger,     # FIRST: populates val_task_loss for other callbacks
                    EpochProgressCallback(
                        log_every=50 if KAGGLE_MODE else 100,
                        total_steps=train_steps,
                        max_epochs=max_epochs,
                    ),
                    nan_loss_terminator,  # stop immediately on NaN — don't burn patience epochs
                    early_stopping,
                    es_state_saver,
                    latest_epoch_saver,
                    reduce_lr,
                    lr_state_saver,
                    max_epochs_per_run,
                    watchdog,
                ]
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=max_epochs,
                    initial_epoch=initial_epoch,
                    steps_per_epoch=train_steps,
                    callbacks=active_callbacks,
                    verbose=0,  # EpochProgressCallback prints within-epoch progress
                )
            except KeyboardInterrupt as e:
                try:
                    if watchdog.hang_detected.is_set():
                        raise RuntimeError("Training hang detected by watchdog") from None
                    else:
                        raise
                except AttributeError:
                    raise e

        if max_epochs_per_run.triggered:
            sys.exit(42)

        for p in (es_state_path, lr_state_path):
            if os.path.exists(p):
                os.remove(p)

        # ---------------------------------------------------------------------
        # Post-training evaluation
        # ---------------------------------------------------------------------
        eval_results = model.evaluate(val_ds, verbose=0)
        val_loss = eval_results[0]
        val_mae = np.mean(eval_results[1:])
        scale_c = (y_max - y_min) / 2.0
        diff_1_mae_c = eval_results[1] * scale_c
        diff_2_mae_c = eval_results[2] * scale_c
        diff_3_mae_c = eval_results[3] * scale_c
        print(f"\nValidation MAE (°C):")
        print(f"  diff_1hr: {diff_1_mae_c:.3f}°C")
        print(f"  diff_2hr: {diff_2_mae_c:.3f}°C")
        print(f"  diff_3hr: {diff_3_mae_c:.3f}°C")

        # Permutation importance fallback (works even without analysis model)
        baseline_loss = val_loss
        feature_importance = {}
        for feat_idx, feature in enumerate(features):
            X_perm = copy.deepcopy(X_val_small)
            vals = X_perm[:, :, feat_idx]
            flattened = vals.reshape(-1)
            np.random.shuffle(flattened)
            X_perm[:, :, feat_idx] = flattened.reshape(vals.shape)
            perm_ds = tf.data.Dataset.from_tensor_slices(
                (X_perm, (y_val_small[:, 0], y_val_small[:, 1], y_val_small[:, 2]))
            ).batch(256)
            permuted_loss = model.evaluate(perm_ds, verbose=0)[0]
            feature_importance[feature] = permuted_loss - baseline_loss

        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nPermutation Feature Importance (val_loss increase):")
        for feat, imp in sorted_importance:
            print(f"  {feat}: {imp:.4f}")

        # ---------------------------------------------------------------------
        # Attention map + VSN weight extraction (interpretability — Track B input)
        # ---------------------------------------------------------------------
        attention_map = {}
        vsn_importance = {}

        if analysis_model is not None:
            print("\n🔍 Extracting attention maps and VSN weights for Track B analysis...")
            n_analysis = min(1000, len(X_val_small))
            X_analysis = X_val_small[:n_analysis]

            all_vsn_weights = []
            all_attn_scores = []

            for i in range(0, n_analysis, 32):
                batch = X_analysis[i:i + 32].astype(np.float32)
                try:
                    ana_out = analysis_model(batch, training=False)
                    all_vsn_weights.append(ana_out[3].numpy())  # (batch, SEQ_LEN, n_features)
                    all_attn_scores.append(ana_out[4].numpy())  # (batch, N_HEADS, SEQ_LEN, SEQ_LEN)
                except Exception as e:
                    print(f"   ⚠️  Analysis batch {i}: {e}")
                    break

            if all_vsn_weights:
                vsn_all = np.concatenate(all_vsn_weights, axis=0)   # (N, SEQ_LEN, n_features)
                attn_all = np.concatenate(all_attn_scores, axis=0)  # (N, N_HEADS, SEQ_LEN, SEQ_LEN)

                # Average VSN weights over samples and time → per-feature importance score
                avg_vsn = vsn_all.mean(axis=(0, 1))  # (n_features,)
                vsn_importance = {feat: float(avg_vsn[i]) for i, feat in enumerate(features)}
                vsn_sorted = sorted(vsn_importance.items(), key=lambda x: x[1], reverse=True)

                print("\nVSN Feature Importance (mean selection weight across time & samples):")
                for feat, imp in vsn_sorted:
                    print(f"  {feat}: {imp:.4f}")

                # Average attention at last query timestep (directly relevant to predictions)
                # attn_all[:, head, query_t, key_t] — we want query_t = -1 (current timestep)
                avg_attn_last = attn_all[:, :, -1, :].mean(axis=0)  # (N_HEADS, SEQ_LEN)
                avg_attn_mean_heads = avg_attn_last.mean(axis=0)     # (SEQ_LEN,)

                # Top attended timesteps (most important lag positions)
                top_k = 10
                top_positions = np.argsort(avg_attn_mean_heads)[::-1][:top_k]
                print(f"\nTop {top_k} attended timestep positions (minutes before current):")
                for pos in top_positions:
                    lag_min = SEQ_LEN - 1 - int(pos)
                    print(f"  t-{lag_min:3d}min (position {int(pos)}): {avg_attn_mean_heads[pos]:.4f}")

                attention_map = {
                    "description": "Track A TFT attention maps for Track B feature extraction",
                    "n_samples_analyzed": n_analysis,
                    "seq_len": SEQ_LEN,
                    "n_heads": N_HEADS,
                    "features": features,
                    "vsn_feature_importance": {feat: float(avg_vsn[i]) for i, feat in enumerate(features)},
                    "attention_at_last_timestep_per_head": {
                        f"head_{h}": [float(v) for v in avg_attn_last[h]]
                        for h in range(N_HEADS)
                    },
                    "attention_mean_over_heads": [float(v) for v in avg_attn_mean_heads],
                    "top_attended_timesteps": [
                        {"position": int(pos), "lag_minutes": SEQ_LEN - 1 - int(pos),
                         "attention_weight": float(avg_attn_mean_heads[pos])}
                        for pos in top_positions
                    ],
                }

                attn_path = os.path.join(RESULTS_DIR, f"attention_maps_{name}.json")
                with open(attn_path, "w") as f:
                    json.dump(attention_map, f, indent=2)
                print(f"✅ Attention maps saved → {attn_path}")
            else:
                print("   ⚠️  No attention data collected")
        else:
            print("⚠️  Analysis model unavailable — skipping attention map extraction")

        # ---------------------------------------------------------------------
        # Export float TFLite model for Pi CPU deployment (Track A — no INT8)
        # The VSN einsum ('bsf,fd->bsfd') is incompatible with the old TFLite
        # converter (BatchMatMulV2 with seq_len=180 in the row dimension).
        # Attempt order: new converter → old converter → SavedModel path.
        # All failures are non-fatal: feature discovery data is already saved.
        # ---------------------------------------------------------------------
        precision_str = "fp16" if TFLITE_FLOAT16 else "fp32"
        print(f"\n🔧 Exporting {precision_str.upper()} TFLite model (Pi CPU deployment)...")

        tflite_model = None
        tflite_fname = None
        tflite_size_kb = 0.0

        try:
            export_model = model
            export_model(tf.zeros((1, SEQ_LEN, n_features), dtype=tf.float32), training=False)

            @tf.function(input_signature=[tf.TensorSpec(shape=[1, SEQ_LEN, n_features], dtype=tf.float32)])
            def model_inference(x):
                return export_model(x, training=False)

            concrete_func = model_inference.get_concrete_function()
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter._experimental_lower_tensor_list_ops = False
            if TFLITE_FLOAT16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            # Attempt 1: new converter (handles einsum ops the old one rejects)
            try:
                converter.experimental_new_converter = True
            except AttributeError:
                pass
            try:
                tflite_model = converter.convert()
                print("   ✅ Converted with new converter")
            except Exception as e1:
                print(f"   ⚠️  New converter failed ({type(e1).__name__}), trying old converter...")

                # Attempt 2: old converter
                try:
                    converter.experimental_new_converter = False
                    tflite_model = converter.convert()
                    print("   ✅ Converted with old converter")
                except Exception as e2:
                    print(f"   ⚠️  Old converter failed ({type(e2).__name__}), trying SavedModel path...")

                    # Attempt 3: SavedModel path (different graph lowering pipeline)
                    try:
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            saved_model_path = os.path.join(tmpdir, "saved_model")

                            @tf.function(input_signature=[tf.TensorSpec(shape=[1, SEQ_LEN, n_features], dtype=tf.float32)])
                            def serving_fn(x):
                                return export_model(x, training=False)

                            tf.saved_model.save(
                                export_model, saved_model_path,
                                signatures={'serving_default': serving_fn.get_concrete_function()})
                            converter2 = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                            converter2._experimental_lower_tensor_list_ops = False
                            if TFLITE_FLOAT16:
                                converter2.optimizations = [tf.lite.Optimize.DEFAULT]
                                converter2.target_spec.supported_types = [tf.float16]
                            tflite_model = converter2.convert()
                            print("   ✅ Converted via SavedModel path")
                    except Exception as e3:
                        print(f"   ⚠️  All TFLite conversion attempts failed.")
                        print(f"   Final error ({type(e3).__name__}): {str(e3)[:300]}")
                        print("   Skipping TFLite export — feature discovery data already saved.")

            if tflite_model is not None:
                tflite_fname = os.path.join(RESULTS_DIR, f"weather_model_5c_{name}_{precision_str}.tflite")
                with open(tflite_fname, "wb") as f:
                    f.write(tflite_model)
                tflite_size_kb = os.path.getsize(tflite_fname) / 1024
                print(f"✅ TFLite model saved ({tflite_size_kb:.1f} KB): {tflite_fname}")
                validate_float_model(tflite_fname, X_val_small, y_val_small, y_mins, y_maxs)
            else:
                print("⚠️  TFLite model not produced — skipping validation")

        except Exception as e:
            print(f"⚠️  TFLite export section failed unexpectedly ({type(e).__name__}): {e}")
            print("   Continuing with results saving...")

        # ---------------------------------------------------------------------
        # Determine best epoch
        # ---------------------------------------------------------------------
        if history and hasattr(history, 'history') and len(history.history.get('val_task_loss', [])) > 0:
            best_session_epoch = int(np.argmin(history.history['val_task_loss'])) + 1
            best_epoch = initial_epoch + best_session_epoch
        elif history and hasattr(history, 'history') and len(history.history.get('val_loss', [])) > 0:
            best_session_epoch = int(np.argmin(history.history['val_loss'])) + 1
            best_epoch = initial_epoch + best_session_epoch
        else:
            best_epoch = initial_epoch if initial_epoch > 0 else 1

        print(f"\nFinal Metrics [{name}]:")
        print(f"  val_loss (includes L2): {val_loss:.6f}")
        print(f"  val_mae (normalized):   {val_mae:.6f}")
        print(f"  diff_1hr MAE:           {diff_1_mae_c:.3f}°C")
        print(f"  diff_2hr MAE:           {diff_2_mae_c:.3f}°C")
        print(f"  diff_3hr MAE:           {diff_3_mae_c:.3f}°C")
        print(f"  Best epoch:             {best_epoch}")
        if tflite_fname:
            print(f"  TFLite size:            {tflite_size_kb:.1f} KB ({precision_str.upper()})")
        else:
            print(f"  TFLite:                 ⚠️  conversion failed (feature discovery data saved)")

        # Accumulate history across restarts
        session_history = {}
        if history and hasattr(history, 'history') and history.history:
            session_history = {k: [float(v) for v in vals]
                               for k, vals in history.history.items()}

        results_path = os.path.join(RESULTS_DIR, f"results_5c_{name}.json")
        accumulated_history = {}
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    accumulated_history = json.load(f).get("history", {})
            except Exception:
                pass
        for k, vals in session_history.items():
            accumulated_history.setdefault(k, []).extend(vals)

        metrics = {
            "name": name,
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "diff_1hr_mae_c": float(diff_1_mae_c),
            "diff_2hr_mae_c": float(diff_2_mae_c),
            "diff_3hr_mae_c": float(diff_3_mae_c),
            "best_epoch": int(best_epoch),
            "tflite_size_kb": float(tflite_size_kb),
            "tflite_precision": precision_str,
            "feature_importance_permutation": [(f, float(i)) for f, i in sorted_importance],
            "vsn_feature_importance": [(f, float(vsn_importance.get(f, 0.0))) for f in features],
            "hyperparams": {
                "d_model": D_MODEL, "n_heads": N_HEADS,
                "dropout": DROPOUT_RATE,
                "l2_reg": L2_REG, "seq_len": SEQ_LEN,
                "n_features": n_features,
            },
            "history": accumulated_history,
        }

        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Results saved → {results_path}")

    # =========================================================================
    # Run
    # =========================================================================
    NUM_RUNS = 1
    EXP_NAME = "tft_run"
    for run_id in range(NUM_RUNS):
        run_name = f"{EXP_NAME}{run_id + 1}"
        build_and_train_model(run_name)

    results = []
    for run_id in range(NUM_RUNS):
        json_file = os.path.join(RESULTS_DIR, f"results_5c_{EXP_NAME}{run_id + 1}.json")
        if os.path.exists(json_file):
            with open(json_file) as f:
                results.append(json.load(f))

    if results:
        best = min(results, key=lambda x: x["val_loss"])
        print(f"\nBest run: {best['name']} — val_loss={best['val_loss']:.6f} "
              f"mae={best['val_mae']:.6f}")
        # Baselines for reference
        print("\nBaseline reference:")
        print(f"  Model 5a clean dense_wide_run1: val_loss=0.000373 (Track A target)")
        print(f"  Model 5b Exp32 float deployed:  30-day StdDev 0.607°C (Track A target)")
    else:
        print("\nNo results found!")

    import shutil
    best_run_name = best['name']
    precision_str = best['tflite_precision']
    best_tflite = os.path.join(RESULTS_DIR, f"weather_model_5c_{best_run_name}_{precision_str}.tflite")
    if os.path.exists(best_tflite):
        shutil.copy(best_tflite, os.path.join(RESULTS_DIR, "weather_model_5c_best.tflite"))
        print(f"Best model copied → {os.path.join(RESULTS_DIR, 'weather_model_5c_best.tflite')}")


def validate_float_model(tflite_model_path, X_val, y_val, y_mins, y_maxs, num_samples=500):
    """Validate the float32/float16 TFLite model on validation data."""
    import tensorflow as tf
    import numpy as np

    print(f"\nValidating TFLite float model on {num_samples} samples...")

    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"⚠️  Could not load TFLite model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input dtype: {input_details[0]['dtype']}, shape: {input_details[0]['shape']}")
    for idx, d in enumerate(output_details):
        print(f"Output {idx} dtype: {d['dtype']}, shape: {d['shape']}")

    X_subset = X_val[:num_samples].astype(np.float32)
    y_subset = y_val[:num_samples]

    y_preds = [[] for _ in range(3)]
    for i in range(len(X_subset)):
        inp = np.expand_dims(X_subset[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        for j in range(3):
            output_data = interpreter.get_tensor(output_details[j]['index'])
            y_preds[j].append(float(output_data[0, 0]))

    y_min = y_mins["temp_diff_1hr"]
    y_max = y_maxs["temp_diff_1hr"]
    scale_to_c = (y_max - y_min) / 2.0

    print("\nTFLite float model MAE (°C):")
    for j, target_name in enumerate(['diff_1hr', 'diff_2hr', 'diff_3hr']):
        preds = np.array(y_preds[j]).reshape(-1)
        true = y_subset[:, j].reshape(-1)
        mae_scaled = np.mean(np.abs(true - preds))
        mae_c = mae_scaled * scale_to_c
        print(f"  {target_name}: {mae_c:.3f}°C")


if __name__ == "__main__":
    main()
