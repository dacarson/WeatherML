# Model 5f: Shallow, INT8-Robust Architecture with Curated Features

## Project Goal

Combine Track B Run 2's demonstrated INT8-robustness (shallow, no raw-sequence architecture) with
Track A/B's curated, TFT-validated feature set — neither existing checkpoint has both.

**Predecessor context**: Model 5e concluded 2026-08-12 (QAT doesn't rescue Track B's deep
`SEQ_LEN=180` architecture). A post-conclusion investigation found Track B **Run 2**
(`SEQ_LEN=1`, pre-breakthrough, 23 raw/uncurated features) beats every deep architecture at INT8
— confirmed with a controlled n=500 eval: 0.211/0.333/0.432°C (1/2/3hr), vs Track B Run 11's
0.522/0.588/0.898°C. A same-day follow-up tested Model 5d's Run 3 (also shallow, but with Track
A/B's curated 20-feature set) the same way: 0.643/0.497/0.634°C — better than Run 11, but *worse*
than Run 2, despite the better features. Full detail in `../Model 5c TFT/
MODEL_5C_TRACK_B_EXPERIMENT_LOG.md` and `../Model 5d/MODEL_5D_EXPERIMENT_LOG.md`'s
post-conclusion addenda.

---

## Why Run 2 Beats Run 3 Is Not Yet Understood — Open Design Variables

Run 2 and Run 3 differ on four axes simultaneously; no experiment has isolated which one(s)
actually drive the INT8 gap:

| | Track B Run 2 (INT8 0.432°C @ 3hr) | Model 5d Run 3 (INT8 0.634°C @ 3hr) |
|---|---|---|
| Features | 23, uncurated (includes some Track A/B flagged harmful for the deep architecture: raw `relative_humidity`, `station_pressure`, no `temp_slope_*`/`temp_diff_vs_5hr/6hr`) | 20, TFT-curated (drops the above, adds `temp_slope_15/30/60`, `temp_diff_vs_5hr/6hr`) |
| Normalization | BatchNorm after every Dense layer | None |
| Activation | plain `relu` (unbounded) | `relu6` (bounded [0,6]) |
| Width | 512→256→128→64 | 128→64→32 |
| FP32 3hr MAE | 0.783°C | 0.790°C (nearly identical) |

Notably, BatchNorm was something Track B's *deep* `SEQ_LEN=180` architecture explicitly
**removed** as an INT8 fix (Run 11) — "BN γ unconstrained by L2, produces wide pre-clip
activation range." Here, in a shallow architecture, the direction may be reversed. This is a real
open question, not an assumption to build on.

## Approach

Rather than changing all four variables at once (which would leave us unable to attribute
whatever result comes out), isolate them one at a time starting from Run 2's proven-good
architecture:

- **Run 1**: Run 2's exact architecture (BatchNorm, `relu`, 512-256-128-64) + Track A/B's curated
  feature set (Run 3's 20 features, or an updated set incorporating anything learned since). Tests
  the features variable alone, holding architecture fixed at what's known to quantize well.
- **Run 2+ (depending on Run 1's result)**: if Run 1 beats Run 2's 0.432°C, curated features help
  and the project can move toward finalizing a deployment candidate. If Run 1 doesn't clearly
  beat it, follow up by testing BatchNorm-vs-none and `relu`-vs-`relu6` individually on top of
  the curated feature set, rather than assuming which one matters.

## Targets

Beat Track B Run 2's INT8 3hr MAE (0.432°C) — the current best-known deployable checkpoint in the
Model 5-series. Stretch target: get FP32 accuracy closer to Track B Run 11's (0.121°C at 3hr)
while keeping INT8 robustness, since neither Run 2 nor Run 3's FP32 ceiling (~0.78-0.79°C) is
actually that close to what the deep architecture achieves in FP32 — there may be headroom left
by combining curated features with more capacity, as long as it doesn't reintroduce the deep
architecture's INT8 fragility.

## Success Criteria

- INT8 3hr MAE at or below 0.432°C (Run 2's confirmed number) — the bar to beat, using the same
  n=500 methodology used throughout this investigation for a fair comparison.
- If achieved: this becomes the new best-known deployable checkpoint recommendation, ahead of
  both Track B Run 2 and Run 11.
