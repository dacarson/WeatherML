# Conv1D TPU-Optimized Model Architecture

This model is designed to forecast temperature at 1, 2, and 3 hours into the future using a windowed sequence of 90 minutes of weather and time-based features. The model architecture balances expressiveness with efficiency for quantization and Edge TPU deployment.

---

## Interesting Notes

> **Note:** While weather is seasonal, the model shows that `sin_day_of_year` and `cos_day_of_year` have little impact. Possible reasons include:
> 
> 1. **Short prediction horizon** – The model predicts only 1–3 hours ahead, where immediate conditions (like `solar_radiation`, `temperature_delta`) matter more than slow seasonal shifts.
> 2. **Redundancy** – Features like `solar_radiation`, `illuminance`, and `uv` already encode time-of-year effects more directly and dynamically.
> 3. **Normalization** – All features are normalized with `StandardScaler`, which can suppress the magnitude and variance of seasonal signals.
> 4. **Model simplicity** – The compact Conv1D model may prioritize high-variance features over subtle, long-cycle indicators due to limited capacity.

---

## 🧱 Input Layer

-```python
Input(shape=(90, 13), batch_size=1)
```

- **Input shape**: 90 time steps × 13 features per step
- **Features per timestep**:
  1. `temp_avg_15min`
  2. `temperature_delta`
  3. `sin_time_of_day`
  4. `cos_time_of_day`
  5. `illuminance`
  6. `solar_radiation`
  7. `station_pressure`
  8. `relative_humidity`
- **Purpose**: Accepts a fixed-length sliding window of normalized weather and temporal data
- **Batch size = 1**: Optimized for inference on Edge TPU (requires static batch size)

---

## 🔄 Convolutional Layers

### 1. Conv1D Layer

```python
Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
```

- **Function**: Extracts local temporal patterns from input sequences
- **ReLU activation**: Introduces non-linearity
- **Padding = 'same'**: Maintains input length

### 2. Conv1D Layer (Downsampling)

```python
Conv1D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')
```

- **Purpose**: Downsamples sequence length (90 → 45)
- **Extracts broader temporal dependencies**

### 3. Conv1D Layer

```python
Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
```

- **Further processing of downsampled temporal features**

---


## 📦 Reshape Layer

```python
Reshape((1440,))
```

- **Purpose**: Converts the (45, 32) output into a flat 1D vector with a fixed shape
- **Rationale**: Ensures compatibility with TFLite and Edge TPU by avoiding dynamic or high-rank reshaping

---

## 🔢 Dense Layers

### 1. Dense(64)

```python
Dense(units=64, activation='relu')
```

- **Adds modeling capacity**, especially to capture interactions between time features

### 2. Dense(32)

```python
Dense(units=32, activation='relu')
```

- **Reduces dimensionality** while preserving learned representations

### 3. Dense(16)

```python
Dense(units=16, activation='relu')
```

- **Lightweight layer** before output, aiding smooth quantization

---

## 🎯 Output Layer

```python
Dense(units=3)
```

- **Linear output** with 3 values:
  - temp_t+1hr
  - temp_t+2hr
  - temp_t+3hr
- **No activation function**: Targets are continuous-valued temperatures


---

## 🧪 Performance

**Best Epoch:** 15  
**Validation MAE:** `0.4778`  
**Validation Loss (MSE):** `0.4558`  
**Quantized Model Size:** `107.09 KB`

This MAE indicates that, on average, the model is off by **~0.48°C** on its multi-step hourly predictions. Denormalized MAE values are:
- t+1hr: 1.45°C
- t+2hr: 1.73°C
- t+3hr: 2.01°C

The model achieves strong accuracy with a small input feature set and is well-suited for Edge TPU deployment.

---

## 🧮 Permutation Feature Importance

The following values indicate the increase in validation loss when each feature is shuffled:

| Feature              | Importance (Δ val_loss) |
|----------------------|-------------------------|
| `temp_avg_15min`     | **+0.4010**             |
| `sin_time_of_day`    | +0.1381                 |
| `temperature_delta`  | –0.1887                 |
| `station_pressure`   | –0.1921                 |
| `relative_humidity`  | –0.1934                 |
| `cos_time_of_day`    | –0.2257                 |
| `illuminance`        | –0.2325                 |
| `solar_radiation`    | –0.2474                 |

---

## ⚙️ Total Parameters

- **102,371 trainable parameters**
- **Designed for fast inference and quantization efficiency**

---

## 💡 Summary

This architecture leverages:
- Temporal pattern recognition from Conv1D
- Dimensionality reduction and feature abstraction in dense layers
- Fixed input shape and batch size for Edge TPU compatibility
