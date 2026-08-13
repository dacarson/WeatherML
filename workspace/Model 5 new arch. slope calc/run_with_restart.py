import subprocess
import time
import argparse
from influxdb import InfluxDBClient

SCRIPT = "Inference_InfluxDB_Writer.py"
MEASUREMENT = "model_5a"

parser = argparse.ArgumentParser(
    description="Run Model 5 inference with auto-restart.")
parser.add_argument("--fresh", action="store_true",
    help="Drop existing predictions and start from the beginning of the dataset.")
args = parser.parse_args()

if args.fresh:
    print(f"🧹 --fresh: dropping '{MEASUREMENT}' for fresh start...")
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet",
                                database="weather")
        client.query(f'DROP MEASUREMENT "{MEASUREMENT}"')
        print(f"  ✔ Dropped '{MEASUREMENT}'")
    except Exception as e:
        print(f"  ⚠️ Could not drop '{MEASUREMENT}': {e}")
else:
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet",
                                database="weather")
        res = client.query(f'SELECT LAST("pred_1hr_temperature") FROM "{MEASUREMENT}"')
        pts = list(res.get_points())
        if pts:
            print(f"📁 Resuming; last prediction at {pts[0]['time']}")
        else:
            print("📄 No existing predictions — starting from the beginning.")
    except Exception as e:
        print(f"⚠️  Could not query InfluxDB: {e}")

while True:
    print("🚀 Launching inference script...")
    result = subprocess.run(["python3", SCRIPT])

    if result.returncode == 88:
        print("🔁 Restart requested by script. Waiting 2s to avoid TPU conflict...")
        time.sleep(2)
        continue
    else:
        print("🛑 Script exited with code", result.returncode)
        break
