import subprocess
import argparse
from influxdb import InfluxDBClient

SCRIPT       = "Inference_InfluxDB_Writer.py"
DEFAULT_TPUS = "auto"
MAX_TPU_PROBE = 8

LOCATIONS = {
    "sf": {"db": "weather",        "measurement": "model_5c_trackb",    "label": "San Francisco"},
    "ps": {"db": "ps_smartweather","measurement": "model_5c_trackb_ps", "label": "Palm Springs"},
}


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
            delegate = None
            try:
                delegate = tflite.load_delegate("libedgetpu.so.1", {"device": f":{i}"})
                detected.append(str(i))
            except Exception:
                continue
            finally:
                if delegate is not None:
                    del delegate
        if detected:
            print(f"🔎 Auto-detected {len(detected)} TPU(s): {detected}")
            return detected
    except Exception as e:
        print(f"⚠️  Delegate probing unavailable: {e}")
    print("⚠️  No TPUs found; defaulting to TPU 0.")
    return ["0"]


def _check_last_timestamp(db, measurement):
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet",
                                database=db)
        res = client.query(f'SELECT LAST("pred_1hr_temperature") FROM "{measurement}"')
        pts = list(res.get_points())
        if pts:
            return pts[0]["time"]
    except Exception as e:
        print(f"⚠️  Could not query InfluxDB: {e}")
    return None


parser = argparse.ArgumentParser(
    description="Run Model 5c Track B inference with TPU rotation and auto-restart.")
parser.add_argument("--tpus", default=DEFAULT_TPUS,
    help='Comma-separated TPU IDs, or "auto" to discover (default: auto).')
parser.add_argument("--fresh", action="store_true",
    help="Drop existing predictions and start from the beginning of the dataset.")
parser.add_argument("--no-tpu", action="store_true",
    help="Disable EdgeTPU; use FP32 model on CPU (Mac development / no Coral TPU).")
parser.add_argument("--location", choices=list(LOCATIONS.keys()), default="sf",
    help="Weather data source: sf (San Francisco, default) or ps (Palm Springs).")
args = parser.parse_args()

loc         = LOCATIONS[args.location]
DB          = loc["db"]
MEASUREMENT = loc["measurement"]
print(f"📍 Location: {loc['label']} | DB: {DB} | Measurement: {MEASUREMENT}")

if args.fresh:
    print(f"🧹 --fresh: dropping '{MEASUREMENT}' from '{DB}'...")
    try:
        client = InfluxDBClient(host="localhost", port=8086,
                                username="admin", password="24planet",
                                database=DB)
        client.query(f'DROP MEASUREMENT "{MEASUREMENT}"')
        print(f"  ✔ Dropped '{MEASUREMENT}'")
    except Exception as e:
        print(f"  ⚠️  Could not drop: {e}")
else:
    last_ts = _check_last_timestamp(DB, MEASUREMENT)
    if last_ts:
        print(f"📁 Resuming; last prediction at {last_ts}")
    else:
        print("📄 No existing predictions — starting from the beginning.")

if args.no_tpu:
    tpus = [""]  # single pass, no TPU rotation needed
else:
    tpus = _discover_tpus_auto() if args.tpus.strip().lower() == "auto" else _parse_tpus(args.tpus)

idx = 0
while True:
    tpu_id = tpus[idx]
    cmd = ["python3", SCRIPT, "--location", args.location]
    if tpu_id:
        cmd += ["--tpu", tpu_id]
    if args.no_tpu:
        cmd.append("--no-tpu")

    print(f"🚀 Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 88:
        idx = (idx + 1) % len(tpus)
        print(f"🔁 Restart requested. Next TPU: {tpus[idx]}")
        continue
    else:
        print(f"🛑 Script exited with code {result.returncode}")
        break
