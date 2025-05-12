import pandas as pd

# Load CSVs without date parsing
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

# Rename raw integer timestamp column to keep it
train_df["timestamp_raw"] = train_df["timestamp"]
val_df["timestamp_raw"] = val_df["timestamp"]

# Convert to datetime
train_df["timestamp"] = pd.to_datetime(train_df["timestamp_raw"], unit="s")
val_df["timestamp"] = pd.to_datetime(val_df["timestamp_raw"], unit="s")

# Combine and sort
combined_df = pd.concat([train_df, val_df], ignore_index=True)
combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

# Save
combined_df.to_csv("combined_data.csv", index=False)

print("✅ Combined dataset saved as 'combined_data.csv'")
