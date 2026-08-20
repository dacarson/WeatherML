import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

STATE_FEATURES = ["pv_p", "net_p", "site_load_p", "baseline_house_load_w", "excess_now_w"]
WEATHER_FEATURES = [
    "solar_radiation", "illuminance", "uv", "wind_avg", "wind_gust", "wind_lull",
    "wind_direction", "relative_humidity", "station_pressure", "temperature", "rain_accumulated",
]
SLOPE_FEATURES = ["pv_p_slope_10", "pv_p_slope_30", "solar_radiation_slope_10", "excess_now_slope_10"]
TIME_FEATURES = ["time_of_day_sin", "time_of_day_cos", "day_of_year_sin", "day_of_year_cos"]
FEATURES = STATE_FEATURES + WEATHER_FEATURES + SLOPE_FEATURES + TIME_FEATURES
TARGET = "excess_future_w"


def load_and_engineer(path):
    """Load an export_and_join.py CSV, reindex onto a regular 1-min grid (so shift-based lag
    features are time-correct across the export's small gaps), and add slope/cyclic features."""
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full_idx)

    df["pv_p_slope_10"] = (df["pv_p"] - df["pv_p"].shift(10)) / 10.0
    df["pv_p_slope_30"] = (df["pv_p"] - df["pv_p"].shift(30)) / 30.0
    df["solar_radiation_slope_10"] = (df["solar_radiation"] - df["solar_radiation"].shift(10)) / 10.0
    df["excess_now_slope_10"] = (df["excess_now_w"] - df["excess_now_w"].shift(10)) / 10.0

    df["time_of_day_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24.0)
    df["time_of_day_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24.0)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"  {path}: {len(df)} / {before} rows after reindex + lag-feature dropna "
          f"({(before - len(df)) / before:.1%} dropped).")
    return df


print("Loading + engineering features...")
train_df = load_and_engineer("train_data.csv")
val_df = load_and_engineer("val_data.csv")

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val, y_val = val_df[FEATURES], val_df[TARGET]

print(f"Training HistGradientBoostingRegressor on {len(X_train)} rows, {len(FEATURES)} features...")
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

pred = model.predict(X_val)
model_mae = mean_absolute_error(y_val, pred)
persistence_mae = mean_absolute_error(y_val, val_df["excess_now_w"])
print(f"\nFull val set (n={len(val_df)}):")
print(f"  Model MAE:       {model_mae:.1f} W")
print(f"  Persistence MAE: {persistence_mae:.1f} W")

backtest_df = val_df[val_df["excess_solar_watts"].notna()]
backtest_pred = model.predict(backtest_df[FEATURES])
model_backtest_mae = mean_absolute_error(backtest_df[TARGET], backtest_pred)
heuristic_mae = mean_absolute_error(backtest_df[TARGET], backtest_df["excess_solar_watts"])
persistence_backtest_mae = mean_absolute_error(backtest_df[TARGET], backtest_df["excess_now_w"])
print(f"\nBacktest subset with logged heuristic predictions (n={len(backtest_df)}):")
print(f"  Model MAE:       {model_backtest_mae:.1f} W")
print(f"  Heuristic MAE:   {heuristic_mae:.1f} W")
print(f"  Persistence MAE: {persistence_backtest_mae:.1f} W")

importances = pd.Series(
    model.feature_importances_ if hasattr(model, "feature_importances_") else None,
    index=FEATURES,
) if hasattr(model, "feature_importances_") else None
if importances is not None:
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False))

joblib.dump({"model": model, "features": FEATURES}, "model_run1.joblib")
print("\nSaved model_run1.joblib")
