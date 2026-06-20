# Model 1 Combined — Multi-Location Training (SF + Palm Springs)

## Overview

Experiment combining weather data from two distinct climate regions — San Francisco (coastal) and Palm Springs (desert) — into a single model. The goal was to train a generalizable model that handles both coastal marine-layer climate and dry desert climate. Climate-specific interaction features were added to help the model differentiate between the two regions.

## Features (~47)

### Base Features
`uv`, `wind_avg`, `wind_gust`, `temperature_delta`, `pressure_delta`, `humidity_delta`, cyclical time encoding (8 features: `time_of_day_sin/cos/sin2/cos2`, `day_of_year_sin/cos`), lag features (`temp_lag30/60/120`, `humidity_lag30/60/120`, `wind_avg_lag30/gust_lag30`, `uv_lag30`, `pressure_lag30`)

### Climate/Location Features
- `is_coastal` — binary flag (1=SF, 0=Palm Springs)
- `coastal_temp_modulation`, `desert_temp_modulation` — climate-specific temperature signals
- `coastal_humidity_modulation`, `desert_humidity_modulation`
- `coastal_wind_modulation`, `desert_wind_modulation`
- `coastal_solar_modulation`, `desert_solar_modulation`
- `coastal_time_temp`, `coastal_time_humidity`, `desert_time_temp`, `desert_time_humidity`
- `daily_temp_range`, `coastal_temp_range`, `desert_temp_range`
- `humidity_stability`, `coastal_humidity_stability`, `desert_humidity_stability`
- `coastal_season_temp`, `desert_season_temp`, `coastal_season_humidity`, `desert_season_humidity`

### Manual Interaction Features
- `time_sin_uv`, `time_cos_uv`, `time_sin_temp_lag`, `time_cos_temp_lag`

## Architecture

Wide+deep+residual+interaction, targets are temperature differences (temp_diff_1hr/2hr/3hr):

```
Input (47) → interaction_embed(Dense(16)) → square → Concatenate → Dense(32)
           ─┬─ Dense(16) (wide) ───────────────────────────────────────────────┐
            └─ Dense(128, relu) → Dropout(0.3) → Dense(64, relu) ──────────────┤ Concatenate
                                             └─ Dense(64) (shortcut)            ┘
                                                  Add → Dense(32, relu)
                                                        ↓
                                               Dense(1) × 3 (diff_1hr, diff_2hr, diff_3hr)
```

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Targets: temperature differences (°C change), not absolute temperatures

## Results

The run 1 result file has NaN for all metrics, indicating the combined model **failed to train** — likely due to data mixing issues, scaling problems across the two very different climates, or numerical instability from the large number of climate interaction features.

Model size: 388.2 KB (much larger than other Model 1 variants due to the 47+ features)

## Key Notes

- **Training failed** — val_loss and all feature importances are NaN
- The Palm Springs daily temperature range (~15–20°C) is much larger than San Francisco (~6°C), creating distribution mismatch issues
- The complex climate interaction features (coastalXtemperature, desertXhumidity, etc.) likely caused optimization difficulties
- Approach was not continued — subsequent multi-location work would need careful normalization per-climate
- `train_model_simple.py` also exists in this directory as a simpler alternative attempt
