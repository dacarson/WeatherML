"""Run 5a: retrain Run 4's exact architecture/features/hyperparameters on data extended
through today (export_and_join.py's TRAIN_END/VAL_START moved 2026-06-30/07-01 ->
2026-07-16/07-17, see SOLARCHARGE_EXPERIMENT_LOG.md Run 5a). Isolates "did more data help"
from any loss-function change, which Run 5b tests separately from this same dataset."""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

from feature_engineering import FEATURES, RAW_TARGET, DELTA_TARGET, load_and_engineer

print("Loading + engineering features (30s grid, daytime-only, extended through today)...")
train_df = load_and_engineer("train_data.csv")
val_df = load_and_engineer("val_data.csv")

X_train, y_train = train_df[FEATURES], train_df[DELTA_TARGET]
X_val = val_df[FEATURES]

print(f"Training HistGradientBoostingRegressor on {len(X_train)} rows, {len(FEATURES)} features, "
      f"delta target (squared_error, same hyperparams as Run 4)...")
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

joblib.dump({"model": model, "features": FEATURES, "target": DELTA_TARGET}, "model_run5a.joblib")
print("\nSaved model_run5a.joblib")
