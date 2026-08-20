import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

TARGET = "excess_future_w"


def load_and_engineer(path):
    """Duplicated from train_run1.py (module has top-level training code, so it can't be
    imported without re-running it) — keep in sync if train_run1.py's feature engineering changes."""
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
    return df


bundle = joblib.load("model_run1.joblib")
model = bundle["model"]
FEATURES = bundle["features"]

val_df = load_and_engineer("val_data.csv").dropna(subset=FEATURES + [TARGET])

for label, df in [
    ("All val rows", val_df),
    ("Daytime only (pv_p >= 500W, matches controller's predicted_excess branch)",
     val_df[val_df["pv_p"] >= 500]),
]:
    backtest_df = df[df["excess_solar_watts"].notna()]
    if len(backtest_df) == 0:
        print(f"\n{label}: no rows with logged heuristic prediction, skipping.")
        continue
    pred = model.predict(backtest_df[FEATURES])
    model_mae = mean_absolute_error(backtest_df[TARGET], pred)
    heuristic_mae = mean_absolute_error(backtest_df[TARGET], backtest_df["excess_solar_watts"])
    persistence_mae = mean_absolute_error(backtest_df[TARGET], backtest_df["excess_now_w"])
    print(f"\n{label} (n={len(backtest_df)}):")
    print(f"  Model MAE:       {model_mae:.1f} W")
    print(f"  Heuristic MAE:   {heuristic_mae:.1f} W")
    print(f"  Persistence MAE: {persistence_mae:.1f} W")
