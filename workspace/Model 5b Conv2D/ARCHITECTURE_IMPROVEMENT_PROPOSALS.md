# Architectural Improvements to Beat Model 5a

## Current Situation
- **Model 5a**: val_loss=0.00068, val_mae=0.00445 (excellent)
- **Model 5b (Exp 5)**: val_loss=0.0062, val_mae=0.0143 (best so far, but 9x worse than 5a)

## Key Differences Between Models

### Features
- **Model 5a**: 20 features (no `temperature`, no `temp_delta_1`)
- **Model 5b**: 29 features (includes `temperature` and `temp_delta_1`)

### Target Scaling
- **Model 5a**: Global scaling (all targets use same min/max: `min().min()` and `max().max()`)
- **Model 5b**: Per-horizon scaling (each target has its own min/max)

### Architecture
- **Model 5a**: Simple (wide=16, deep=128→64→32, interaction=16→32, no lag extraction, no final merged layer)
- **Model 5b**: Complex (wide=24, deep=192→96→64→48, interaction=24→48, lag extraction branch, final merged layer=256)

### Training
- **Model 5a**: LR=1e-5, MSE, patience=5
- **Model 5b**: LR=2e-5, MSE, patience=10, LR warmup + scheduling

---

## Proposed Architectural Changes

### Option 1: Model 5a Architecture + Model 5b Features + Global Scaling ⭐ **MOST PROMISING**

**Strategy**: Use Model 5a's proven simple architecture with Model 5b's additional features, plus Model 5a's global target scaling.

**Changes**:
1. **Architecture**: Use Model 5a's exact architecture (wide=16, deep=128→64→32, interaction=16→32)
2. **Remove**: Lag extraction branch (redundant - lag features already in input)
3. **Remove**: Final merged layer (Model 5a goes directly to output)
4. **Keep**: Model 5b's additional features (`temperature`, `temp_delta_1`)
5. **Target Scaling**: Switch to Model 5a's global scaling (single min/max for all targets)
6. **Training**: Use Model 5a's config (LR=1e-5, MSE, patience=5)

**Rationale**: 
- Model 5a's architecture is proven to work excellently
- Additional features (`temperature`, `temp_delta_1`) should provide more information
- Global scaling might be more stable than per-horizon
- Simple architecture may generalize better

**Expected Outcome**: Should match or exceed Model 5a because we have more input information with the same proven architecture.

---

### Option 2: Remove Redundant Features + Model 5a Architecture

**Strategy**: Match Model 5a's feature set exactly, use Model 5a's architecture, but keep Model 5b's per-horizon scaling.

**Changes**:
1. **Remove**: `temperature` and `temp_delta_1` features (match Model 5a exactly)
2. **Architecture**: Use Model 5a's exact architecture
3. **Target Scaling**: Keep per-horizon (test if it helps)
4. **Training**: Use Model 5a's config

**Rationale**: 
- Test if `temperature` and `temp_delta_1` are actually hurting performance
- If this matches Model 5a, then the additional features are the problem

---

### Option 3: Hybrid Architecture - Moderate Size, No Lag Extraction

**Strategy**: Remove lag extraction (redundant) but use moderate architecture size between Model 5a and Model 5b.

**Changes**:
1. **Architecture**: wide=20, deep=160→80→48, interaction=20→40
2. **Remove**: Lag extraction branch
3. **Remove**: Final merged layer (go directly to output like Model 5a)
4. **Keep**: All Model 5b features
5. **Target Scaling**: Try global scaling (Model 5a's approach)
6. **Training**: LR=1e-5 (Model 5a's rate)

**Rationale**: 
- Moderate capacity for 29 features (more than Model 5a's 16, less than Model 5b's 24)
- Remove redundant lag extraction
- Simpler architecture like Model 5a

---

### Option 4: Feature Gating/Attention Mechanism

**Strategy**: Add a learnable attention/gating mechanism to focus on important features.

**Changes**:
1. **Add**: Feature attention layer that learns to weight features
2. **Architecture**: Keep Model 5a's size but add attention before dense layers
3. **Keep**: All features and lag extraction

**Rationale**: 
- Help model focus on most important features
- Could improve performance with 29 features

**Complexity**: Higher - requires new layer implementation

---

### Option 5: Remove Lag Extraction, Keep Full Architecture

**Strategy**: Remove redundant lag extraction but keep Experiment 5's larger architecture.

**Changes**:
1. **Architecture**: Keep full Model 5b size (wide=24, deep=192→96→64→48)
2. **Remove**: Lag extraction branch (redundant)
3. **Keep**: Final merged layer (256 units)
4. **Target Scaling**: Try global scaling
5. **Training**: LR=2e-5 (Experiment 5's rate)

**Rationale**: 
- Full capacity for 29 features
- Remove redundant lag extraction
- Test if global scaling helps

---

## Recommended Approach

**Start with Option 1** (Model 5a architecture + Model 5b features + global scaling):

1. **Why it's most promising**:
   - Uses Model 5a's proven architecture (val_loss=0.00068)
   - Adds Model 5b's additional features (more information)
   - Uses Model 5a's proven global scaling
   - Simple and clean - no redundant components

2. **If Option 1 doesn't beat Model 5a**, try:
   - **Option 2**: Remove `temperature` and `temp_delta_1` to match Model 5a exactly
   - This will tell us if the additional features are hurting

3. **If Option 1 works but not enough**, try:
   - **Option 3**: Hybrid architecture with moderate capacity
   - **Option 5**: Full architecture without lag extraction

---

## Implementation Priority

1. ✅ **Option 1**: Model 5a architecture + Model 5b features + global scaling
2. ⏸️ **Option 2**: Remove redundant features (if Option 1 doesn't work)
3. ⏸️ **Option 3**: Hybrid architecture (if Option 1 needs more capacity)
4. ⏸️ **Option 5**: Full architecture without lag extraction (if simpler doesn't work)
5. ⏸️ **Option 4**: Feature attention (most complex, try last)

---

## Key Insights

1. **Simplicity wins**: Model 5a's simple architecture achieved excellent results
2. **More features ≠ better**: Additional features might be hurting if not handled properly
3. **Scaling matters**: Global scaling might be more stable than per-horizon
4. **Redundancy hurts**: Lag extraction is redundant since lag features are already in input
5. **Architecture size must match complexity**: But Model 5a shows simple can work for complex problems

---

## Questions to Answer

1. Are `temperature` and `temp_delta_1` helping or hurting?
2. Is per-horizon scaling better or worse than global scaling?
3. Is the lag extraction branch actually helping despite redundancy?
4. Does Model 5a's architecture work with 29 features, or does it need more capacity?

---

*Generated: [Current Date]*
*Status: Proposal - Ready for implementation*
