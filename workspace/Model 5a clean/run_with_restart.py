import subprocess
import argparse
from influxdb import InfluxDBClient

SCRIPT = "Inference_InfluxDB_Writer.py"
MEASUREMENT = "model_5a_clean"
DEFAULT_TPUS = "auto"
MAX_TPU_PROBE = 8


def _parse_tpus(tpus_arg: str):
    tpus = [t.strip() for t in tpus_arg.split(",") if t.strip()]
    if not tpus:
        raise ValueError("At least one TPU id must be provided.")
    return tpus


def _discover_tpus_auto():
    # Probe delegate by index (works without PyCoral).
    try:
        import tflite_runtime.interpreter as tflite

        detected = []
        for i in range(MAX_TPU_PROBE):
            device = f":{i}"
            delegate = None
            try:
                delegate = tflite.load_delegate('libedgetpu.so.1', {"device": device})
                detected.append(str(i))
            except Exception:
                continue
            finally:
                # Release delegate handle before running the actual script.
                if delegate is not None:
                    del delegate

        if detected:
            print(f"🔎 Auto-detected {len(detected)} TPU(s) by probe: {detected}")
            return detected
    except Exception as e:
        print(f"⚠️ Delegate probing unavailable: {e}")

    print("⚠️ Could not auto-detect TPUs; defaulting to TPU 0.")
    return ["0"]


def _check_last_timestamp():
    """Query InfluxDB for the last prediction timestamp."""
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet",
                                database="weather")
        result = client.query(f'SELECT LAST("pred_1hr_temperature") FROM "{MEASUREMENT}"')
        points = list(result.get_points())
        if points:
            return points[0]['time']
    except Exception as e:
        print(f"⚠️ Could not query InfluxDB for last timestamp: {e}")
    return None


parser = argparse.ArgumentParser(description="Run inference script and rotate TPUs on restart.")
parser.add_argument(
    "--tpus",
    default=DEFAULT_TPUS,
    help='Comma-separated TPU device IDs, or "auto" to discover available TPUs (default: auto).',
)
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Drop existing predictions and start from the beginning of the dataset.",
)
args = parser.parse_args()
tpus = _discover_tpus_auto() if args.tpus.strip().lower() == "auto" else _parse_tpus(args.tpus)

if args.fresh:
    print(f"🧹 --fresh flag set: dropping '{MEASUREMENT}' for a clean run...")
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet",
                                database="weather")
        client.query(f'DROP MEASUREMENT "{MEASUREMENT}"')
        print(f"  ✔ Dropped '{MEASUREMENT}'")
    except Exception as e:
        print(f"  ⚠️ Could not drop '{MEASUREMENT}': {e}")
else:
    last_ts = _check_last_timestamp()
    if last_ts:
        print(f"📁 Resuming from InfluxDB; last prediction timestamp = {last_ts}")
    else:
        print("📄 No existing predictions found in InfluxDB; starting from the beginning.")

idx = 0
while True:
    tpu_id = tpus[idx]
    print(f"🚀 Launching inference script on TPU {tpu_id}...")
    result = subprocess.run(["python3", SCRIPT, "--tpu", tpu_id])

    if result.returncode == 88:
        idx = (idx + 1) % len(tpus)
        print(f"🔁 Restart requested by script. Switching to next TPU ({tpus[idx]}).")
        continue
    else:
        print("🛑 Script exited with code", result.returncode)
        break
