"""Run 5c: bracket the quantile sweep below 0.3, since Run 5b found q=0.3 best of {0.3,0.4,0.5}
by a wide margin with a mechanism (correcting determine_target_amperage's round-up-always
convention) that doesn't obviously saturate at 0.3. See SOLARCHARGE_EXPERIMENT_LOG.md Run 5c."""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

from feature_engineering import FEATURES, RAW_TARGET, DELTA_TARGET, load_and_engineer

print("Loading + engineering features (30s grid, daytime-only, extended through today)...")
train_df = load_and_engineer("train_data.csv")
val_df = load_and_engineer("val_data.csv")

X_train, y_train = train_df[FEATURES], train_df[DELTA_TARGET]
X_val = val_df[FEATURES]
backtest_df = val_df[val_df["excess_solar_watts"].notna()]

persistence_mae = mean_absolute_error(val_df[RAW_TARGET], val_df["excess_now_w"])
heuristic_mae = mean_absolute_error(backtest_df[RAW_TARGET], backtest_df["excess_solar_watts"])

for quantile in (0.10, 0.15, 0.20, 0.25):
    print(f"\n=== quantile={quantile} ===")
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
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

    backtest_pred_delta = model.predict(backtest_df[FEATURES])
    backtest_pred_future = backtest_df["excess_now_w"].to_numpy() + backtest_pred_delta
    model_backtest_mae = mean_absolute_error(backtest_df[RAW_TARGET], backtest_pred_future)
    bias = (backtest_pred_future - backtest_df[RAW_TARGET]).mean()

    print(f"  Full val MAE:      {model_mae:.1f} W  (persistence: {persistence_mae:.1f} W)")
    print(f"  Backtest MAE:      {model_backtest_mae:.1f} W  (heuristic: {heuristic_mae:.1f} W)")
    print(f"  Mean signed error: {bias:+.1f} W  (negative = under-predicts excess, on average)")

    q_tag = f"q{int(quantile * 100)}"
    path = f"model_run5c_{q_tag}.joblib"
    joblib.dump({"model": model, "features": FEATURES, "target": DELTA_TARGET}, path)
    print(f"  Saved {path}")
