import pandas as pd

df = pd.read_csv("train_data.csv", parse_dates=["time"])
cols = ["pv_p", "solar_radiation", "excess_now_w", "excess_future_w",
        "baseline_house_load_w", "charging_power_watts"]
print(df[cols].describe())
print()
print("corr(pv_p, solar_radiation):", df["pv_p"].corr(df["solar_radiation"]))
print("corr(excess_now_w, excess_future_w):", df["excess_now_w"].corr(df["excess_future_w"]))
nonzero_charge = (df["charging_power_watts"] > 0).mean()
print(f"fraction of rows with EV charging > 0: {nonzero_charge:.1%}")
