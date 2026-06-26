import pandas as pd

# Load source datasets
sf_train = pd.read_csv("train_data_sf.csv")
sf_val   = pd.read_csv("val_data_sf.csv")
ps_train = pd.read_csv("train_data_ps_2026.csv")
ps_val   = pd.read_csv("val_data_ps_2026.csv")

# Convert timestamps for date alignment
for df in [sf_train, sf_val, ps_train, ps_val]:
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s")

# PS starts 2020-07-25, SF starts 2022-06-22. PS has earlier data with no SF counterpart;
# that's fine — both stations contribute independently to the combined training set.
ps_train_start = ps_train["timestamp_dt"].min()
sf_train_start = sf_train["timestamp_dt"].min()
print(f"SF train start: {sf_train_start.date()}  ({len(sf_train):,} rows)")
print(f"PS train start: {ps_train_start.date()}  ({len(ps_train):,} rows)")

# Validation sets align (both start 2025-06-23)
print(f"SF val start: {sf_val['timestamp_dt'].min()}  ({len(sf_val):,} rows)")
print(f"PS val start: {ps_val['timestamp_dt'].min()}  ({len(ps_val):,} rows)")

# Drop helper column before saving
for df in [sf_train, sf_val, ps_train, ps_val]:
    df.drop(columns=["timestamp_dt"], inplace=True)

# Combine and sort
train_combined = pd.concat([sf_train, ps_train], ignore_index=True)
train_combined = train_combined.sort_values("timestamp").reset_index(drop=True)

val_combined = pd.concat([sf_val, ps_val], ignore_index=True)
val_combined = val_combined.sort_values("timestamp").reset_index(drop=True)

# Save
train_combined.to_csv("train_data_combined.csv", index=False)
val_combined.to_csv("val_data_combined.csv", index=False)

print(f"\n✅ train_data_combined.csv: {len(train_combined):,} rows")
print(f"✅ val_data_combined.csv:   {len(val_combined):,} rows")
