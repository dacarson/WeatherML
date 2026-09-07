"""
Evaluate every SolarChargeML candidate on the metrics that actually matter: amp-decision
accuracy against a perfect-hindsight ideal amp, and simulated PG&E $ cost — not just watts MAE.
See SOLARCHARGE_EXPERIMENT_LOG.md's "Live shadow validation results" section for why watts MAE
alone was found to be an insufficient/misleading optimization target.

All candidates are evaluated on the SAME val_data.csv (extended through today for Run 5), so
this also gives a fair "did retraining/loss-change help" comparison for model_run4 unchanged
vs. model_run5a/5b, isolating the effect from the val-window shift.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from feature_engineering import FEATURES, RAW_TARGET, load_and_engineer
import decision_policy as dp

DECISION_STRIDE = 10  # 30s bin * 10 = 5 min, matching HORIZON_MIN / the real control cadence


def load_model_predictions(path, df):
    bundle = joblib.load(path)
    model, features = bundle["model"], bundle["features"]
    pred_delta = model.predict(df[features])
    return df["excess_now_w"].to_numpy() + pred_delta


def oscillation_rate(amp, index, tou_period=None, period_filter=None):
    """0-to-nonzero amp transitions per day — a stop/start-cycling proxy. amp/index must
    already be at decision cadence (one row per real control decision), not the raw 30s grid."""
    amp = np.asarray(amp)
    is_on = (amp > 0).astype(int)
    flips = np.abs(np.diff(is_on))
    if period_filter is not None:
        mask = (np.asarray(tou_period) == period_filter)
        flips = flips[mask[1:] & mask[:-1]]
    n_days = (index.max() - index.min()).total_seconds() / 86400.0
    return flips.sum() / n_days


def evaluate_candidate(name, predicted_excess, df):
    """df must already have tou_period/season columns (see decision_policy.add_tou_and_season)
    and an 'ideal_amp' column. predicted_excess is this candidate's excess_future_w estimate
    (or, for 'ideal', just pass df[RAW_TARGET] itself)."""
    actual = df[RAW_TARGET].to_numpy()
    mae = mean_absolute_error(actual, predicted_excess)

    amp = dp.amp_decisions(predicted_excess, df["tou_period"].to_numpy())
    ideal_amp = df["ideal_amp"].to_numpy()
    exact_match = (amp == ideal_amp)
    amp_error = np.abs(amp - ideal_amp)

    rows = []
    for period in ["peak", "part_peak", "off_peak"]:
        mask = (df["tou_period"] == period).to_numpy()
        if mask.sum() == 0:
            continue
        rows.append({
            "candidate": name, "tou_period": period, "n": int(mask.sum()),
            "exact_match_pct": exact_match[mask].mean() * 100,
            "mean_amp_error": amp_error[mask].mean(),
        })
    rows.append({
        "candidate": name, "tou_period": "ALL", "n": len(df),
        "exact_match_pct": exact_match.mean() * 100,
        "mean_amp_error": amp_error.mean(),
    })
    amp_df = pd.DataFrame(rows)

    # $ simulation on decision-cadence-matched rows only (every DECISION_STRIDE-th row ~= one
    # real 5-min control decision) — see SOLARCHARGE_EXPERIMENT_LOG.md Run 5a for why summing
    # every 30s row would 10x-overcount overlapping hypothetical decisions.
    decision_df = df.iloc[::DECISION_STRIDE].copy()
    decision_amp = dp.amp_decisions(
        predicted_excess[::DECISION_STRIDE], decision_df["tou_period"].to_numpy()
    )
    dt_hours = 5 / 60.0  # HORIZON_MIN=5 minutes per decision
    cost, grid_kwh, solar_kwh = dp.simulate_cost(
        decision_amp, decision_df[RAW_TARGET].to_numpy(),
        decision_df["tou_period"].to_numpy(), decision_df["season"].to_numpy(), dt_hours,
    )
    decision_df["cost"] = cost
    cost_by_period = decision_df.groupby("tou_period")["cost"].sum()
    total_cost = cost.sum()
    n_days = (df.index.max() - df.index.min()).total_seconds() / 86400.0
    monthly_cost = total_cost / n_days * 30.4375

    off_peak_flips = oscillation_rate(
        decision_amp, decision_df.index, decision_df["tou_period"].to_numpy(), "off_peak"
    )
    all_flips = oscillation_rate(decision_amp, decision_df.index)

    return mae, amp_df, total_cost, monthly_cost, cost_by_period, off_peak_flips, all_flips


def main():
    print("Loading + engineering val_data.csv...")
    val_df = load_and_engineer("val_data.csv")
    backtest_df = val_df[val_df["excess_solar_watts"].notna()].copy()
    print(f"Backtest set: {len(backtest_df)} rows ({backtest_df.index.min()} -> "
          f"{backtest_df.index.max()}), {len(backtest_df) / len(val_df):.1%} of daytime val.")

    backtest_df = dp.add_tou_and_season(backtest_df)
    backtest_df["ideal_amp"] = dp.amp_decisions(
        backtest_df[RAW_TARGET].to_numpy(), backtest_df["tou_period"].to_numpy()
    )

    candidates = {
        "heuristic": backtest_df["excess_solar_watts"].to_numpy(),
        "ideal": backtest_df[RAW_TARGET].to_numpy(),
    }
    for name, path in [
        ("model_run4", "model_run4.joblib"),
        ("model_run5a", "model_run5a.joblib"),
        ("model_run5b_q30", "model_run5b_q30.joblib"),
        ("model_run5b_q40", "model_run5b_q40.joblib"),
        ("model_run5b_q50", "model_run5b_q50.joblib"),
        ("model_run5c_q10", "model_run5c_q10.joblib"),
        ("model_run5c_q15", "model_run5c_q15.joblib"),
        ("model_run5c_q20", "model_run5c_q20.joblib"),
        ("model_run5c_q25", "model_run5c_q25.joblib"),
    ]:
        candidates[name] = load_model_predictions(path, backtest_df)

    print(f"\n{'='*110}")
    print(f"{'Candidate':<18}{'Watts MAE':>12}{'ALL exact%':>13}{'ALL |err|':>12}"
          f"{'Monthly $':>13}{'off_pk flips/d':>16}{'ALL flips/d':>13}")
    print("=" * 110)

    all_amp_rows = []
    monthly_costs = {}
    for name, predicted_excess in candidates.items():
        mae, amp_df, total_cost, monthly_cost, cost_by_period, off_peak_flips, all_flips = (
            evaluate_candidate(name, predicted_excess, backtest_df)
        )
        all_amp_rows.append(amp_df)
        monthly_costs[name] = monthly_cost
        overall = amp_df[amp_df["tou_period"] == "ALL"].iloc[0]
        print(f"{name:<18}{mae:>10.1f} W{overall['exact_match_pct']:>12.1f}%"
              f"{overall['mean_amp_error']:>12.2f}{monthly_cost:>13.2f}"
              f"{off_peak_flips:>16.2f}{all_flips:>13.2f}")

    heuristic_monthly = monthly_costs["heuristic"]
    print(f"\n$ savings vs. heuristic (positive = candidate saves more per month):")
    for name, cost in monthly_costs.items():
        if name in ("heuristic",):
            continue
        print(f"  {name:<18} {heuristic_monthly - cost:+7.2f} $/month")

    print(f"\n{'='*100}\nAmp-decision accuracy by TOU period\n{'='*100}")
    amp_all = pd.concat(all_amp_rows, ignore_index=True)
    for period in ["peak", "part_peak", "off_peak", "ALL"]:
        sub = amp_all[amp_all["tou_period"] == period]
        print(f"\n-- {period} --")
        print(sub[["candidate", "n", "exact_match_pct", "mean_amp_error"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    amp_all.to_csv("run5_amp_accuracy.csv", index=False)
    print("\nWrote run5_amp_accuracy.csv")


if __name__ == "__main__":
    main()
