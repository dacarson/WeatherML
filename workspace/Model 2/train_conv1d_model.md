

# WeatherML Dense-Wide Model Overview

## 📊 Model Summary

This model, named **`dense_wide`**, is a hybrid architecture combining both wide and deep components. It is trained to predict **temperature forecasts** at +1 hour, +2 hours, and +3 hours using engineered weather features.

---

## 🔢 Input Features

The model uses **12 input features**, all normalized using `StandardScaler`:

1. `illuminance` – Light intensity (lux)
2. `solar_radiation` – Solar energy received (W/m²)
3. `uv` – Ultraviolet index
4. `relative_humidity` – Ambient humidity (%)
5. `station_pressure` – Barometric pressure at station level (hPa)
6. `wind_avg` – Average wind speed
7. `wind_gust` – Gust wind speed
8. `day_of_year` – Day number (1 to 365)
9. `time_of_day` – Fractional time (0.0 = midnight, 0.5 = noon)
10. `temperature_delta` – Slope over 15-sample rolling window of temperature
11. `temp_lag1` – Previous time step temperature
12. `humidity_lag1` – Previous time step relative humidity

---

## 🧠 Model Architecture

````markdown
```text
Input: (12,) vector of normalized features

→ wide branch:
    Dense(16)                            
    - Projects raw input features into a low-dimensional space to capture direct relationships.

→ deep branch:
    Dense(128, relu)                     
    - Learns complex non-linear combinations of input features.
    
    Dropout(0.3)                         
    - Prevents overfitting by randomly zeroing 30% of the activations during training.
    
    Dense(64, relu)                      
    - Further abstracts learned patterns and introduces more depth.
    
    Dense(64)                            
    - Linear projection for shortcut (residual) connection.
    
    Add()                                
    - Combines the shortcut and deep path output, enabling residual learning which improves gradient flow and model stability.
    
    Dense(32, relu)                      
    - Reduces dimensionality and prepares the deep branch output for merging.

→ merged:
    Concatenate([wide, deep])           
    - Combines both wide (memorization) and deep (generalization) features into a single vector.

    Dense(3)                             
    - Outputs 3 temperature predictions (for +1hr, +2hr, +3hr) from the merged representation.
```
````

---

## 🧪 Performance

**Best Epoch:** 2  
**Validation MAE:** `0.7120`  
**Validation Loss (MSE):** `1.1681`  
**Quantized Model Size:** `24.95 KB`

This MAE indicates that, on average, the model is off by **~0.71°C** on its multi-step hourly predictions, making it suitable for general short-term temperature forecasting.

---

## 🧮 Permutation Feature Importance

The following values indicate the increase in validation loss when each feature is shuffled:

| Feature              | Importance (Δ val_loss) |
|----------------------|-------------------------|
| `temp_lag1`          | **26.5402**             |
| `time_of_day`        | 1.8868                  |
| `day_of_year`        | 0.3489                  |
| `temperature_delta`  | 0.2068                  |
| `solar_radiation`    | 0.1834                  |
| `uv`                 | 0.1657                  |
| `illuminance`        | 0.1615                  |
| `relative_humidity`  | 0.1204                  |
| `humidity_lag1`      | 0.0878                  |
| `wind_gust`          | 0.0782                  |
| `wind_avg`           | 0.0457                  |
| `station_pressure`   | 0.0367                  |

---

## ⚙️ Notes

- `temp_lag1` is by far the most influential feature, highlighting the temporal continuity of temperature.
- Quantization and Edge TPU compilation succeeded, and the model uses minimal memory on-chip (27.00 KiB cached).