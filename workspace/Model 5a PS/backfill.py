#!/usr/bin/env python3
"""
Weatherflow → InfluxDB backfill for ps_smartweather database
Station: Palm Springs (24833), Device: ST-00014305 (device_id 80826)

Modes:
  (default)          — discover and fill data gaps in InfluxDB
  --migrate-humidity — find rows with integer 'humidity' field (old backfill
                       artifact), re-fetch from WeatherFlow API, and store as
                       float 'relative_humidity'. Target the NAS with --host.

Examples:
  python3 backfill.py                               # gap-fill to localhost
  python3 backfill.py --host http://10.0.1.188:8086 --migrate-humidity
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import datetime
import argparse
import time

# ── CLI args (parsed first so INFLUX_HOST can be set from --host) ─────────────

parser = argparse.ArgumentParser(description="WeatherFlow → InfluxDB backfill / migration")
parser.add_argument(
    "--host",
    default="http://localhost:8086",
    help="InfluxDB base URL (default: http://localhost:8086)",
)
parser.add_argument(
    "--migrate-humidity",
    action="store_true",
    help="Re-fetch windows that have integer 'humidity' but no 'relative_humidity' "
         "and store as float 'relative_humidity'. Use --host to target the NAS.",
)
args = parser.parse_args()

# ── Configuration ────────────────────────────────────────────────────────────

API_TOKEN   = "88e930eb-ee19-423e-ac78-f33117885fd3"
DEVICE_ID   = 80826
INFLUX_HOST = args.host
INFLUX_DB   = "ps_smartweather"
MEASUREMENT = "wf/obs_st"

SCAN_START         = 1686000000  # fallback only: 2023-06-06 UTC
today    = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
SCAN_END = int((today - datetime.timedelta(days=1)).timestamp())

CHUNK_SECONDS      = 3600   # gap-scan granularity: 1 hour
FILL_CHUNK_SECONDS = 86400  # fetch granularity: 1 day
MIN_EXPECTED_RATIO = 0.80   # flag chunk if <80% of expected points present

MIGRATE_SCAN_CHUNK = 86400  # migration scan granularity: 1 day (~3650 queries for 10 yrs)

# Gaps with only partial data that Weatherflow cannot fill further.
SKIP_GAPS = [
    (1595649600, 1595653200),  # 2020-07-25 04:00 → 2020-07-25 05:00 UTC
    (1602806400, 1602810000),  # 2020-10-16 00:00 → 2020-10-16 01:00 UTC
    (1612648800, 1612652400),  # 2021-02-06 22:00 → 2021-02-06 23:00 UTC
    (1612882800, 1612886400),  # 2021-02-09 15:00 → 2021-02-09 16:00 UTC
    (1612936800, 1612954800),  # 2021-02-10 06:00 → 2021-02-10 11:00 UTC
    (1612958400, 1612962000),  # 2021-02-10 12:00 → 2021-02-10 13:00 UTC
    (1624413600, 1624417200),  # 2021-06-23 02:00 → 2021-06-23 03:00 UTC
    (1624438800, 1624442400),  # 2021-06-23 09:00 → 2021-06-23 10:00 UTC
    (1653584400, 1653591600),  # 2022-05-26 17:00 → 2022-05-26 19:00 UTC
    (1677661200, 1677668400),  # 2023-03-01 09:00 → 2023-03-01 11:00 UTC
    (1624356000, 1624377600),  # 2021-06-22 10:00 → 2021-06-22 16:00 UTC
    (1646942400, 1646956800),  # 2022-03-10 20:00 → 2022-03-11 00:00 UTC
]

# ── Field mapping (obs_st indices 0–17, matching UDP protocol) ────────────────

# (field_name, is_integer)
FIELDS = [
    ("time_epoch",                     True),
    ("wind_lull",                      False),
    ("wind_avg",                       False),
    ("wind_gust",                      False),
    ("wind_direction",                 True),
    ("wind_sample_interval",           True),
    ("station_pressure",               False),
    ("temperature",                    False),
    ("relative_humidity",              False),   # float, not integer
    ("illuminance",                    True),
    ("uv",                             False),
    ("solar_radiation",                True),
    ("rain_accumulated",               False),
    ("precipitation_type",             True),
    ("lightning_strike_avg_distance",  True),
    ("lightning_strike_count",         True),
    ("battery",                        False),
    ("report_interval",                True),
]

def obs_to_lp(obs):
    """Convert a single obs_st array to InfluxDB line protocol (all fields)."""
    if not obs or len(obs) < 18:
        return None
    ts = obs[0]
    if ts is None:
        return None
    field_parts = []
    for i, (name, is_int) in enumerate(FIELDS[1:], start=1):
        if i < len(obs) and obs[i] is not None:
            val = obs[i]
            if is_int:
                field_parts.append(f"{name}={int(val)}i")
            else:
                field_parts.append(f"{name}={float(val)}")
    if not field_parts:
        return None
    return f'{MEASUREMENT} {",".join(field_parts)} {ts}'

def obs_to_lp_rh_only(obs):
    """Write only relative_humidity (float) for a single obs_st observation.

    Used by --migrate-humidity to add the corrected field without touching
    any other fields already stored in InfluxDB.
    """
    if not obs or len(obs) < 9:
        return None
    ts = obs[0]
    rh = obs[8]
    if ts is None or rh is None:
        return None
    return f'{MEASUREMENT} relative_humidity={float(rh)} {ts}'

# ── InfluxDB helpers ──────────────────────────────────────────────────────────

def influx_query(query):
    """Run a SELECT query against InfluxDB, return parsed JSON result."""
    url = f"{INFLUX_HOST}/query?db={INFLUX_DB}&q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": "wf-backfill/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"    InfluxDB query error: {e}")
        return None

def count_field(field, time_start, time_end):
    """Return count of non-null values for a specific field in [time_start, time_end)."""
    query = (f'SELECT count({field}) FROM "{MEASUREMENT}" '
             f'WHERE time >= {time_start}s AND time < {time_end}s')
    result = influx_query(query)
    if result is None:
        return 0
    series = result.get("results", [{}])[0].get("series")
    if not series:
        return 0
    return series[0]["values"][0][1] or 0

def count_points(time_start, time_end):
    """Return count of points in [time_start, time_end) or 0 on error."""
    return count_field("station_pressure", time_start, time_end)

def get_influx_first_timestamp():
    """Return the earliest timestamp in InfluxDB for this measurement, or None."""
    query = f'SELECT first(station_pressure) FROM "{MEASUREMENT}"'
    result = influx_query(query)
    if result is None:
        return None
    series = result.get("results", [{}])[0].get("series")
    if not series:
        return None
    time_str = series[0]["values"][0][0]
    dt = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())

def write_to_influx(points_lp):
    if not points_lp:
        return 0
    body = "\n".join(points_lp).encode()
    url  = f"{INFLUX_HOST}/write?db={INFLUX_DB}&precision=s"
    req  = urllib.request.Request(url, data=body, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return len(points_lp)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"    InfluxDB write error {e.code}: {body}")
        return 0
    except Exception as e:
        print(f"    InfluxDB write error: {e}")
        return 0

# ── Weatherflow API ───────────────────────────────────────────────────────────

def get_weatherflow_first_timestamp():
    """
    Find the first observation timestamp by probing backward from InfluxDB start
    in weekly chunks until we hit a window with no data.
    """
    influx_first = get_influx_first_timestamp()
    if influx_first is None:
        return None

    print("  Probing Weatherflow API for data before InfluxDB start...", flush=True)

    probe_end      = influx_first
    probe_start    = probe_end - (7 * 86400)
    found_earliest = influx_first
    limit          = int(datetime.datetime(2016, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    consecutive_empty = 0

    while probe_start >= limit:
        data = fetch_obs(probe_start, probe_end)
        if data and data.get("obs"):
            obs_list = [o for o in data["obs"] if o and o[0] is not None]
            if obs_list:
                found_earliest    = obs_list[0][0]
                probe_end         = probe_start
                probe_start       = probe_end - (7 * 86400)
                consecutive_empty = 0
                dt = datetime.datetime.utcfromtimestamp(found_earliest).strftime('%Y-%m-%d')
                print(f"    Found data back to {dt}, probing further...")
                continue

        consecutive_empty += 1
        if consecutive_empty >= 2:
            break
        probe_end   = probe_start
        probe_start = probe_end - (7 * 86400)

    return found_earliest

def fetch_obs(time_start, time_end, _max_retries=5):
    """Fetch observations from Weatherflow REST API with retry on 429 rate-limit."""
    url = (f"https://swd.weatherflow.com/swd/rest/observations/device/{DEVICE_ID}"
           f"?time_start={time_start}&time_end={time_end}&token={API_TOKEN}")
    req = urllib.request.Request(url, headers={"User-Agent": "wf-backfill/1.0"})
    delay = 10
    for attempt in range(_max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    Rate limited (429) — waiting {delay}s before retry {attempt + 1}/{_max_retries}...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)  # exponential backoff, cap at 2 min
            else:
                print(f"    API error: {e}")
                return None
        except Exception as e:
            print(f"    API error: {e}")
            return None
    print(f"    API error: gave up after {_max_retries} retries (persistent 429)")
    return None

# ── Gap detection & filling ───────────────────────────────────────────────────

def find_gaps(scan_start, scan_end):
    """
    Walk [scan_start, scan_end) in CHUNK_SECONDS increments.
    Returns list of (start, end, has_empty) tuples for gaps below the
    expected point threshold, with adjacent chunks merged into contiguous ranges.
    """
    print(f"Scanning {datetime.datetime.utcfromtimestamp(scan_start).strftime('%Y-%m-%d')} "
          f"→ {datetime.datetime.utcfromtimestamp(scan_end).strftime('%Y-%m-%d')} "
          f"in {CHUNK_SECONDS // 60}-minute chunks...\n")

    raw_gaps     = []
    t            = scan_start
    total_chunks = (scan_end - scan_start) // CHUNK_SECONDS
    chunk_num    = 0

    while t < scan_end:
        t_end    = min(t + CHUNK_SECONDS, scan_end)
        expected = (t_end - t) / 60
        count    = count_points(t, t_end)

        if count < expected * MIN_EXPECTED_RATIO:
            raw_gaps.append((t, t_end))

        chunk_num += 1
        if chunk_num % 500 == 0:
            pct = 100 * chunk_num / total_chunks
            dt  = datetime.datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')
            print(f"  ... {pct:.0f}% scanned (at {dt})")

        t = t_end

    if not raw_gaps:
        return []

    merged = [list(raw_gaps[0])]
    for start, end in raw_gaps[1:]:
        if start == merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    merged = [(s, e) for s, e in merged]

    classified = []
    for gs, ge in merged:
        total_count    = count_points(gs, ge)
        total_expected = (ge - gs) / 60
        is_sparse      = total_count < total_expected * MIN_EXPECTED_RATIO
        classified.append((gs, ge, not is_sparse))

    result = [
        (gs, ge, has_empty) for gs, ge, has_empty in classified
        if (gs, ge) not in SKIP_GAPS
    ]
    return result

def fill_gap(gap_start, gap_end):
    """
    Fetch and write all observations for [gap_start, gap_end).
    Returns (points_written, unfilled_chunks).
    """
    total_written = 0
    unfilled      = []
    chunk_start   = gap_start

    while chunk_start < gap_end:
        chunk_end = min(chunk_start + FILL_CHUNK_SECONDS, gap_end)
        cs_dt = datetime.datetime.utcfromtimestamp(chunk_start).strftime('%Y-%m-%d %H:%M')
        ce_dt = datetime.datetime.utcfromtimestamp(chunk_end).strftime('%Y-%m-%d %H:%M')
        print(f"  Fetching {cs_dt} → {ce_dt} UTC ...", end=" ", flush=True)

        data = fetch_obs(chunk_start, chunk_end)
        if data is None or data.get("obs") is None:
            print("⚠ no data (null response)")
            unfilled.append((chunk_start, chunk_end, "null response"))
            chunk_start = chunk_end
            continue

        obs_list = data.get("obs", [])
        if not obs_list:
            print("⚠ 0 observations returned by API")
            unfilled.append((chunk_start, chunk_end, "empty obs array"))
            chunk_start = chunk_end
            continue

        points  = [lp for obs in obs_list if (lp := obs_to_lp(obs))]
        written = write_to_influx(points)
        total_written += written
        print(f"{written} points written")

        chunk_start = chunk_end

    return total_written, unfilled

# ── Humidity migration ────────────────────────────────────────────────────────

def find_humidity_migration_ranges(scan_start, scan_end):
    """
    Walk [scan_start, scan_end) in MIGRATE_SCAN_CHUNK (daily) increments.
    Returns list of (start, end) ranges where integer 'humidity' exists
    but float 'relative_humidity' does not — these need migration.

    Uses daily chunks to keep query count manageable (~3650 for 10 years).
    """
    print(f"Scanning {datetime.datetime.utcfromtimestamp(scan_start).strftime('%Y-%m-%d')} "
          f"→ {datetime.datetime.utcfromtimestamp(scan_end).strftime('%Y-%m-%d')} "
          f"for days with 'humidity' but no 'relative_humidity'...\n")

    raw_ranges    = []
    t             = scan_start
    total_chunks  = max(1, (scan_end - scan_start) // MIGRATE_SCAN_CHUNK)
    chunk_num     = 0

    while t < scan_end:
        t_end = min(t + MIGRATE_SCAN_CHUNK, scan_end)

        hum_count = count_field("humidity", t, t_end)
        rh_count  = count_field("relative_humidity", t, t_end)

        if hum_count > 0 and rh_count == 0:
            raw_ranges.append((t, t_end))

        chunk_num += 1
        if chunk_num % 100 == 0:
            pct = 100 * chunk_num / total_chunks
            dt  = datetime.datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')
            print(f"  ... {pct:.0f}% scanned (at {dt})")

        t = t_end

    if not raw_ranges:
        return []

    # Merge contiguous day-ranges into larger spans
    merged = [list(raw_ranges[0])]
    for start, end in raw_ranges[1:]:
        if start == merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def migrate_humidity_range(range_start, range_end):
    """
    For [range_start, range_end): fetch from WeatherFlow API and write only
    the relative_humidity field (float) for each observation.
    Returns (points_written, unfilled_chunks).
    """
    total_written = 0
    unfilled      = []
    chunk_start   = range_start

    while chunk_start < range_end:
        chunk_end = min(chunk_start + FILL_CHUNK_SECONDS, range_end)
        cs_dt = datetime.datetime.utcfromtimestamp(chunk_start).strftime('%Y-%m-%d %H:%M')
        ce_dt = datetime.datetime.utcfromtimestamp(chunk_end).strftime('%Y-%m-%d %H:%M')
        print(f"  Fetching {cs_dt} → {ce_dt} UTC ...", end=" ", flush=True)

        data = fetch_obs(chunk_start, chunk_end)
        if data is None or data.get("obs") is None:
            print("⚠ no data (null response)")
            unfilled.append((chunk_start, chunk_end, "null response"))
            chunk_start = chunk_end
            continue

        obs_list = data.get("obs", [])
        if not obs_list:
            print("⚠ 0 observations returned by API")
            unfilled.append((chunk_start, chunk_end, "empty obs array"))
            chunk_start = chunk_end
            continue

        points  = [lp for obs in obs_list if (lp := obs_to_lp_rh_only(obs))]
        written = write_to_influx(points)
        total_written += written
        print(f"{written} points written")

        chunk_start = chunk_end

    return total_written, unfilled

# ── Main ──────────────────────────────────────────────────────────────────────

def main_migrate_humidity():
    print(f"=== Humidity migration mode (target: {INFLUX_HOST}) ===\n")

    influx_first = get_influx_first_timestamp()
    if influx_first is None:
        print("❌ No data found in InfluxDB — nothing to migrate.")
        return

    effective_start = (influx_first // MIGRATE_SCAN_CHUNK) * MIGRATE_SCAN_CHUNK

    ranges = find_humidity_migration_ranges(effective_start, SCAN_END)

    if not ranges:
        print("✓ No migration needed — all data already has 'relative_humidity'.")
        return

    total_days = sum((e - s) // 86400 for s, e in ranges)
    print(f"\nFound {len(ranges)} range(s) to migrate ({total_days} day(s) total):\n")
    for i, (rs, re) in enumerate(ranges, 1):
        rs_dt    = datetime.datetime.utcfromtimestamp(rs).strftime('%Y-%m-%d')
        re_dt    = datetime.datetime.utcfromtimestamp(re).strftime('%Y-%m-%d')
        duration = (re - rs) / 86400
        print(f"  {i:2}. {rs_dt} → {re_dt}  ({duration:.0f} days)")

    print()
    total_written = 0
    all_unfilled  = []

    for i, (rs, re) in enumerate(ranges, 1):
        rs_dt    = datetime.datetime.utcfromtimestamp(rs).strftime('%Y-%m-%d')
        re_dt    = datetime.datetime.utcfromtimestamp(re).strftime('%Y-%m-%d')
        duration = (re - rs) / 86400
        print(f"{'='*60}")
        print(f"Range {i}/{len(ranges)}: {rs_dt} → {re_dt} ({duration:.0f} days)")

        written, unfilled = migrate_humidity_range(rs, re)
        print(f"  → Range total: {written} points written")
        if unfilled:
            print(f"  ⚠ {len(unfilled)} chunk(s) returned no data from API")

        total_written += written
        all_unfilled.extend(unfilled)

    print(f"\n{'='*60}")
    print(f"DONE. Total relative_humidity points written: {total_written}")

    if all_unfilled:
        print(f"\n⚠ {len(all_unfilled)} chunk(s) could not be filled:")
        for cs, ce, reason in all_unfilled:
            cs_dt = datetime.datetime.utcfromtimestamp(cs).strftime('%Y-%m-%d %H:%M')
            ce_dt = datetime.datetime.utcfromtimestamp(ce).strftime('%Y-%m-%d %H:%M')
            print(f"  {cs_dt} → {ce_dt} UTC  ({reason})")
    else:
        print("✓ All ranges successfully migrated.")


def main_gap_fill():
    print("Checking data boundaries...\n")

    influx_first = get_influx_first_timestamp()
    wf_first     = get_weatherflow_first_timestamp()

    if wf_first and influx_first and wf_first < influx_first:
        delta_days = (influx_first - wf_first) / 86400
        print(f"\n  ⚠ Weatherflow has data {delta_days:.1f} days before InfluxDB starts.")
        print(f"  Will scan from Weatherflow device creation date.")
        effective_start = wf_first
    elif influx_first:
        print(f"\n  ✓ No earlier data found on Weatherflow — starting from InfluxDB first record.")
        effective_start = influx_first
    else:
        print(f"\n  Falling back to hardcoded SCAN_START.")
        effective_start = SCAN_START

    effective_start = (effective_start // 3600) * 3600

    print()
    gaps = find_gaps(effective_start, SCAN_END)

    if not gaps:
        print("✓ No gaps found — database looks complete.")
        return

    print(f"\nFound {len(gaps)} gap(s):\n")
    for i, (gs, ge, has_empty) in enumerate(gaps, 1):
        gs_dt    = datetime.datetime.utcfromtimestamp(gs).strftime('%Y-%m-%d %H:%M')
        ge_dt    = datetime.datetime.utcfromtimestamp(ge).strftime('%Y-%m-%d %H:%M')
        duration = (ge - gs) / 3600
        flag     = "" if has_empty else "  [sparse]"
        print(f"  {i:2}. {gs_dt} → {ge_dt} UTC  ({duration:.1f}h){flag}")

    print()
    all_unfilled  = []
    total_written = 0
    still_sparse  = []

    for i, (gs, ge, has_empty) in enumerate(gaps, 1):
        gs_dt    = datetime.datetime.utcfromtimestamp(gs).strftime('%Y-%m-%d %H:%M')
        ge_dt    = datetime.datetime.utcfromtimestamp(ge).strftime('%Y-%m-%d %H:%M')
        duration = (ge - gs) / 3600
        print(f"{'='*60}")
        print(f"Gap {i}/{len(gaps)}: {gs_dt} → {ge_dt} UTC ({duration:.1f}h)")

        written, unfilled = fill_gap(gs, ge)
        print(f"  → Gap total: {written} points")
        if unfilled:
            print(f"  ⚠ {len(unfilled)} chunk(s) returned no data from API")

        total_written += written
        all_unfilled.extend(unfilled)

        if not has_empty:
            t           = gs
            still_below = True
            while t < ge:
                t_end    = min(t + CHUNK_SECONDS, ge)
                expected = (t_end - t) / 60
                c        = count_points(t, t_end)
                if c >= expected * MIN_EXPECTED_RATIO:
                    still_below = False
                    break
                t = t_end
            if still_below:
                still_sparse.append((gs, ge))

    print(f"\n{'='*60}")
    print(f"DONE. Total points written: {total_written}")

    if still_sparse:
        print(f"\n⚠ {len(still_sparse)} gap(s) are still sparse after filling.")
        print("  Weatherflow likely has no additional data for these periods.")
        print("  Add this block to the top of the script to suppress them:\n")
        print("SKIP_GAPS = [")
        for gs, ge in still_sparse:
            gs_dt = datetime.datetime.utcfromtimestamp(gs).strftime('%Y-%m-%d %H:%M')
            ge_dt = datetime.datetime.utcfromtimestamp(ge).strftime('%Y-%m-%d %H:%M')
            print(f"    ({gs}, {ge}),  # {gs_dt} → {ge_dt} UTC")
        print("]")

    if all_unfilled:
        print(f"\n⚠ WARNING: {len(all_unfilled)} chunk(s) could not be filled (API returned no data):")
        for cs, ce, reason in all_unfilled:
            cs_dt = datetime.datetime.utcfromtimestamp(cs).strftime('%Y-%m-%d %H:%M')
            ce_dt = datetime.datetime.utcfromtimestamp(ce).strftime('%Y-%m-%d %H:%M')
            print(f"  {cs_dt} → {ce_dt} UTC  ({reason})")
        print("\nThese may be genuine data gaps on Weatherflow's servers,")
        print("or transient API failures — consider retrying or checking")
        print("the Weatherflow app for those time periods.")
    else:
        print("✓ All gaps successfully filled.")


if __name__ == "__main__":
    if args.migrate_humidity:
        main_migrate_humidity()
    else:
        main_gap_fill()
