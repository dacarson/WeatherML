import json
import numpy as np

with open("results_5c_run4/attention_maps_tft_run1.json") as f:
    d = json.load(f)

# 1. VSN feature importance — sorted
print("=== VSN Feature Importance (sorted) ===")
vsn = sorted(d["vsn_feature_importance"].items(), key=lambda x: x[1], reverse=True)
for feat, w in vsn:
    bar = "█" * int(w * 200)
    print(f"  {feat:22s} {w:.4f}  {bar}")

# 2. Full attention curve over the 180-step window
print("\n=== Attention over time (mean across heads, query=last timestep) ===")
attn = np.array(d["attention_mean_over_heads"])  # shape (180,) — position 0 = oldest
lags = np.arange(len(attn) - 1, -1, -1)         # convert position → lag in minutes
# Print in 10-minute buckets
print("  lag_min  weight")
for i in range(0, 180, 10):
    pos = 179 - i   # position in the array
    w = attn[pos]
    bar = "█" * int(w * 1000)
    print(f"  t-{i:3d}min  {w:.5f}  {bar}")

# 3. Top attended timesteps
print("\n=== Top 10 attended timesteps ===")
for t in d["top_attended_timesteps"]:
    print(f"  t-{t['lag_minutes']:3d}min  (pos {t['position']:3d})  weight={t['attention_weight']:.5f}")

# 4. Per-head attention — see if heads specialize
print("\n=== Per-head attention at last-query position 0 (t-179min) ===")
for head, weights in d["attention_at_last_timestep_per_head"].items():
    w0 = weights[0]   # oldest position
    w_last = weights[-1]  # most recent position (t-0)
    print(f"  {head}: oldest={w0:.4f}  newest={w_last:.4f}")

