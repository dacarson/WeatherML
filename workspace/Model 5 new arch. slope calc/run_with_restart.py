import subprocess
import time
from influxdb import InfluxDBClient

SCRIPT = "Inference_InfluxDB_Writer.py"
MEASUREMENT = "model_5a"

print(f"🧹 Dropping '{MEASUREMENT}' for fresh start...")
try:
    client = InfluxDBClient(host="localhost", port=8086,
                            username="admin", password="24planet",
                            database="weather")
    client.query(f'DROP MEASUREMENT "{MEASUREMENT}"')
    print(f"  ✔ Dropped '{MEASUREMENT}'")
except Exception as e:
    print(f"  ⚠️ Could not drop '{MEASUREMENT}': {e}")

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
