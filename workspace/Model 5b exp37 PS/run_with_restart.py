import subprocess
import argparse
from influxdb import InfluxDBClient

MEASUREMENT = "model_5b_exp37"
SCRIPT = "Inference_InfluxDB_Writer.py"
DEFAULT_TPUS = "auto"
MAX_TPU_PROBE = 8


def _parse_tpus(tpus_arg: str):
    tpus = [t.strip() for t in tpus_arg.split(",") if t.strip()]
    if not tpus:
        raise ValueError("At least one TPU id must be provided.")
    return tpus


def _discover_tpus_auto():
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
                if delegate is not None:
                    del delegate

        if detected:
            print(f"🔎 Auto-detected {len(detected)} TPU(s) by probe: {detected}")
            return detected
    except Exception as e:
        print(f"⚠️ Delegate probing unavailable: {e}")

    print("⚠️ Could not auto-detect TPUs; defaulting to TPU 0.")
    return ["0"]


parser = argparse.ArgumentParser(description="Run Exp 37 inference script and rotate TPUs on restart.")
parser.add_argument(
    "--tpus",
    default=DEFAULT_TPUS,
    help='Comma-separated TPU device IDs, or "auto" to discover available TPUs (default: auto).',
)
parser.add_argument(
    "--fresh",
    action="store_true",
    help=f"Drop the '{MEASUREMENT}' measurement before starting for a clean backfill.",
)
args = parser.parse_args()

if args.fresh:
    print(f"🧹 Dropping '{MEASUREMENT}' for fresh start...")
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet",
                                database="weather")
        client.query(f'DROP MEASUREMENT "{MEASUREMENT}"')
        print(f"  ✔ Dropped '{MEASUREMENT}'")
    except Exception as e:
        print(f"  ⚠️ Could not drop '{MEASUREMENT}': {e}")

tpus = _discover_tpus_auto() if args.tpus.strip().lower() == "auto" else _parse_tpus(args.tpus)
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
