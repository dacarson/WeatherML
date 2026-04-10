import subprocess
import time
import os

# Remove progress.json if it exists
if os.path.exists("progress_diff.json"):
    os.remove("progress_diff.json")

SCRIPT = "Inference_InfluxDB_Writer_1_diff.py"

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
