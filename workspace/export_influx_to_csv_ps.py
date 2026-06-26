from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime, timezone, timedelta

# Connect to InfluxDB
client = InfluxDBClient(
    host="10.0.1.188",
    port=8086,
    username="admin",
    password="24planet",
    database="ps_smartweather"
)

print("Querying data from InfluxDB...")

# Fetch all data in 60-day chunks to stay under max-select-point limit
chunk_days = 60
fetch_start = datetime(2020, 7, 25, tzinfo=timezone.utc)
fetch_end   = datetime(2026, 6, 24, tzinfo=timezone.utc)

chunks = []
chunk_start = fetch_start
while chunk_start < fetch_end:
    chunk_end = min(chunk_start + timedelta(days=chunk_days), fetch_end)
    start_str = chunk_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str   = chunk_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    query = (
        f'SELECT * FROM "wf/obs_st" '
        f"WHERE time >= '{start_str}' AND time < '{end_str}'"
    )
    print(f"  Fetching {start_str} → {end_str}...")
    result = client.query(query)
    points = list(result.get_points())
    if points:
        chunks.append(pd.DataFrame(points))
    chunk_start = chunk_end

df = pd.concat(chunks, ignore_index=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df.sort_index(inplace=True)

# Overwrite the WF device 'timestamp' field with the InfluxDB index (always valid,
# unlike the device field which is absent/zero in older backfilled records).
df['timestamp'] = df.index.astype('int64') // 1_000_000_000

# Drop columns not used in training
df.drop(columns=['humidity', 'firmware_revision', 'report_interval', 'wind_sample_interval'],
        errors='ignore', inplace=True)

# Add derived features
df['day_of_year'] = df.index.dayofyear
df['time_of_day'] = df.index.hour + df.index.minute / 60.0

# Create prediction targets
df['temp_t+1hr'] = df['temperature'].shift(-60)
df['temp_t+2hr'] = df['temperature'].shift(-120)
df['temp_t+3hr'] = df['temperature'].shift(-180)

# Drop rows with missing inputs or targets
required_fields = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_direction',
    'wind_gust', 'wind_lull', 'rain_accumulated', 'precipitation_type',
    'day_of_year', 'time_of_day',
    'temp_t+1hr', 'temp_t+2hr', 'temp_t+3hr'
]
df_clean = df.dropna(subset=required_fields)

# Training: all data up to 2025-04-06; Validation: last year (2025-04-07 to 2026-04-07)
train_start = pd.Timestamp('2020-07-25T00:00:00Z')
train_end   = pd.Timestamp('2025-06-22T23:59:59Z')
val_start   = pd.Timestamp('2025-06-23T00:00:00Z')
val_end     = pd.Timestamp('2026-06-23T23:59:59Z')

train_df = df_clean[(df_clean.index >= train_start) & (df_clean.index <= train_end)]
val_df   = df_clean[(df_clean.index >= val_start)   & (df_clean.index <= val_end)]

# Save to CSV
train_df.to_csv("train_data_ps_2026.csv", index=False)
val_df.to_csv("val_data_ps_2026.csv", index=False)

print(f"✅ Wrote {len(train_df)} training rows to train_data_ps_2026.csv")
print(f"✅ Wrote {len(val_df)} validation rows to val_data_ps_2026.csv")
