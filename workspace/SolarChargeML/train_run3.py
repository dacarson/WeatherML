import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

STATE_FEATURES = ["pv_p", "net_p", "site_load_p", "baseline_house_load_w", "excess_now_w"]
VOLATILITY_FEATURES = [
    "pv_p_std", "pv_p_range", "site_load_p_std", "site_load_p_range",
]
WEATHER_FEATURES = [
    "solar_radiation", "illuminance", "uv", "wind_avg", "wind_gust", "wind_lull",
    "wind_direction", "relative_humidity", "station_pressure", "temperature", "rain_accumulated",
]
SLOPE_FEATURES = ["pv_p_slope_10", "pv_p_slope_30", "solar_radiation_slope_10", "excess_now_slope_10"]
TIME_FEATURES = ["time_of_day_sin", "time_of_day_cos", "day_of_year_sin", "day_of_year_cos"]
FEATURES = STATE_FEATURES + VOLATILITY_FEATURES + WEATHER_FEATURES + SLOPE_FEATURES + TIME_FEATURES
RAW_TARGET = "excess_future_w"
DELTA_TARGET = "excess_delta_w"
DAYTIME_PV_W = 500.0  # matches solar_charge_controller.py's production < 500W branch cutoff


def load_and_engineer(path):
    """Same as train_run2.py, plus (new in Run 3) pv_p_range/site_load_p_range derived from
    the min/max columns export_and_join.py now provides — the sub-minute volatility signal a
    1-min MEAN alone can't see (see SOLARCHARGE_EXPERIMENT_LOG.md Run 3)."""
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full_idx)

    df["pv_p_range"] = df["pv_p_max"] - df["pv_p_min"]
    df["site_load_p_range"] = df["site_load_p_max"] - df["site_load_p_min"]

    df["pv_p_slope_10"] = (df["pv_p"] - df["pv_p"].shift(10)) / 10.0
    df["pv_p_slope_30"] = (df["pv_p"] - df["pv_p"].shift(30)) / 30.0
    df["solar_radiation_slope_10"] = (df["solar_radiation"] - df["solar_radiation"].shift(10)) / 10.0
    df["excess_now_slope_10"] = (df["excess_now_w"] - df["excess_now_w"].shift(10)) / 10.0

    df["time_of_day_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24.0)
    df["time_of_day_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24.0)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    df[DELTA_TARGET] = df[RAW_TARGET] - df["excess_now_w"]

    before = len(df)
    df = df.dropna(subset=FEATURES + [RAW_TARGET, DELTA_TARGET])
    df = df[df["pv_p"] >= DAYTIME_PV_W]
    print(f"  {path}: {len(df)} / {before} rows after reindex + lag-feature dropna + "
          f"daytime filter (pv_p >= {DAYTIME_PV_W}W).")
    return df


print("Loading + engineering features (daytime-only, with volatility features)...")
train_df = load_and_engineer("train_data.csv")
val_df = load_and_engineer("val_data.csv")

X_train, y_train = train_df[FEATURES], train_df[DELTA_TARGET]
X_val = val_df[FEATURES]

print(f"Training HistGradientBoostingRegressor on {len(X_train)} rows, {len(FEATURES)} features, "
      f"delta target...")
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    validation_fraction=0.1,
    early_stopping=True,
    n_iter_no_change=15,
)
model.fit(X_train, y_train)
print(f"Stopped after {model.n_iter_} iterations.")

pred_delta = model.predict(X_val)
pred_future = val_df["excess_now_w"].to_numpy() + pred_delta
model_mae = mean_absolute_error(val_df[RAW_TARGET], pred_future)
persistence_mae = mean_absolute_error(val_df[RAW_TARGET], val_df["excess_now_w"])
print(f"\nDaytime val set (n={len(val_df)}):")
print(f"  Model MAE:       {model_mae:.1f} W")
print(f"  Persistence MAE: {persistence_mae:.1f} W")

backtest_df = val_df[val_df["excess_solar_watts"].notna()]
backtest_pred_delta = model.predict(backtest_df[FEATURES])
backtest_pred_future = backtest_df["excess_now_w"].to_numpy() + backtest_pred_delta
model_backtest_mae = mean_absolute_error(backtest_df[RAW_TARGET], backtest_pred_future)
heuristic_mae = mean_absolute_error(backtest_df[RAW_TARGET], backtest_df["excess_solar_watts"])
persistence_backtest_mae = mean_absolute_error(backtest_df[RAW_TARGET], backtest_df["excess_now_w"])
print(f"\nDaytime backtest subset with logged heuristic predictions (n={len(backtest_df)}):")
print(f"  Model MAE:       {model_backtest_mae:.1f} W")
print(f"  Heuristic MAE:   {heuristic_mae:.1f} W")
print(f"  Persistence MAE: {persistence_backtest_mae:.1f} W")

joblib.dump({"model": model, "features": FEATURES, "target": DELTA_TARGET}, "model_run3.joblib")
print("\nSaved model_run3.joblib")
