import subprocess
import argparse
from influxdb import InfluxDBClient

SCRIPT      = "Inference_InfluxDB_Writer_track_a.py"
DEFAULT_RUN = "run6"


def _locations_for(run_name: str):
    return {
        "sf": {"db": "weather",         "measurement": f"model_5c_tft_{run_name}",    "label": "San Francisco"},
        "ps": {"db": "ps_smartweather", "measurement": f"model_5c_tft_{run_name}_ps", "label": "Palm Springs"},
    }


def _check_last_timestamp(db, measurement):
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet", database=db)
        res = client.query(f'SELECT LAST("pred_1hr_temperature") FROM "{measurement}"')
        pts = list(res.get_points())
        if pts:
            return pts[0]["time"]
    except Exception as e:
        print(f"Could not query InfluxDB: {e}")
    return None


parser = argparse.ArgumentParser(
    description="Run Model 5c Track A (TFT) FP32 inference with auto-restart. "
                "No TPU rotation — Track A never runs on Edge TPU (permanent TFLite export blocker).")
parser.add_argument("--fresh", action="store_true",
    help="Drop existing predictions and start from the beginning of the dataset.")
parser.add_argument("--run", default=DEFAULT_RUN,
    help=f"Trained run to serve, e.g. run6 (selects ./results_5c_<run>). Default: {DEFAULT_RUN}.")
parser.add_argument("--location", choices=["sf", "ps"], default="sf",
    help="Weather data source: sf (San Francisco, default) or ps (Palm Springs).")
args = parser.parse_args()

LOCATIONS   = _locations_for(args.run)
loc         = LOCATIONS[args.location]
DB          = loc["db"]
MEASUREMENT = loc["measurement"]
print(f"Location: {loc['label']} | Run: {args.run} | DB: {DB} | Measurement: {MEASUREMENT}")

if args.fresh:
    print(f"--fresh: dropping '{MEASUREMENT}' from '{DB}'...")
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet", database=DB)
        client.query(f'DROP MEASUREMENT "{MEASUREMENT}"')
        print(f"  Dropped '{MEASUREMENT}'")
    except Exception as e:
        print(f"  Could not drop: {e}")
else:
    last_ts = _check_last_timestamp(DB, MEASUREMENT)
    if last_ts:
        print(f"Resuming; last prediction at {last_ts}")
    else:
        print("No existing predictions - starting from the beginning.")

while True:
    cmd = ["python3", SCRIPT, "--location", args.location, "--run", args.run]
    print(f"Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 88:
        print("Restart requested (memory-bound backfill checkpoint or next batch).")
        continue
    else:
        print(f"Script exited with code {result.returncode}")
        break
