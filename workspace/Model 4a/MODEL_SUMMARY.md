# Model 4a — Hybrid Precision Encoding (INT32 for Wide-Range Features)

## Overview

Refinement of Model 4's MSB/LSB encoding that applies per-feature precision analysis. Wide-range features (like `illuminance`, which needs ~17 bits to represent its full dynamic range) are encoded as 4-byte INT32 (4 inputs per feature), while all other features continue using 2-byte INT16 encoding. This makes the input width 26 features (11 × 2 + 1 × 4), compared to Model 4's uniform 24.

No results files exist — the model was written and configured for training but was never run to completion.

## Features (12 base → 26 encoded)

Base features (12) same as Model 4. Precision allocation:

| Encoding | Features |
|----------|----------|
| INT32 → 4 bytes | `illuminance` (needs 17 bits) |
| INT16 → 2 bytes | All 11 remaining features |

Total encoded input: 11 × 2 + 1 × 4 = **26 features**

### Encoding Process

For each base feature, scaled to [0, 1]:
- **INT16**: multiply by 65535, cast to `uint16`, split LSB (bits 0-7) and MSB (bits 8-15), both normalized to [0, 1]
- **INT32**: multiply by 2^32-1, cast to `uint32`, split into 4 bytes B0/B1/B2/B3, each normalized to [0, 1]

## Architecture

```
Input (26)
      │
LayerNormalization
      │
Wide path:  Dense(16) ────────────────────────────────────────────┐
Deep path:  Dense(64, relu, L2=1e-4) → Dropout(0.2)              │
            Dense(32, relu, L2=1e-4) + shortcut(Dense(32))        │
                                 Add → Dense(16, relu, L2=1e-4)   │
                                              Concatenate(wide+deep)
                                                                   │
                                              Dense(1) × 3 (t1hr, t2hr, t3hr)
```

Differences from Model 4:
- Slightly wider initial Dense (same 64), but shallower residual (32→16 vs 64→32)
- 26 inputs instead of 24
- Single run configured (`range(1, 2)`)

- Optimizer: Adam lr=1e-5
- Loss: MSE
- Batch size: 32
- Early stopping: patience=5
- Quantization: INT8 PTQ (same as Model 4)

## Results

**No results** — no JSON result files exist in the directory. The model was not successfully trained.

## Key Notes

- Incomplete experiment — never produced results
- The precision-analysis-driven approach (analyzing each feature's actual bit requirements before choosing encoding width) is methodologically sound but the execution didn't run
- Both Model 4 and Model 4a (the MSB/LSB family) were abandoned after Model 4 showed val_loss of 0.017 — far worse than flat floating-point models
- Model 5 and later work moved back to standard float inputs with hand-crafted feature engineering
