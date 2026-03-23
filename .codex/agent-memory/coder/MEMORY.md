# Coder Agent Memory

## Recent Changes

### 2026-03-21: OFI Calculation Fix (Task from lit_expert)

**File Modified**: `python_lab/src/dataset.py`

**Function Updated**: `compute_ofi_from_lob()` (lines 95-137)

**Problem**: Previous implementation computed OFI as simple `sum(bid_v[:, :3]) - sum(ask_v[:, :3])`, ignoring price changes. This did not match Cont-Kukanov-Stoikov (CKS) formula.

**Solution Implemented**:
- Added `depth` parameter (default=3) for configurable LOB depth
- Proper CKS logic using price and volume differences:
  - `buyer_mask = (bid_price_diff > 0) | ((bid_price_diff == 0) & (bid_vol_diff > 0))`
  - `seller_mask = (ask_price_diff < 0) | ((ask_price_diff == 0) & (ask_vol_diff < 0))`
  - `buy_contrib = np.where(buyer_mask, bid_vol_diff, 0).sum(axis=1)`
  - `sell_contrib = np.where(seller_mask, ask_vol_diff, 0).sum(axis=1)`
  - `ofi_deltas = buy_contrib - sell_contrib`
- Uses `np.diff(..., prepend=...)` for vectorized diff calculation
- Applies `is_update` mask to only count actual LOB updates
- Returns `np.float32` array of shape `(N,)` (non-cumulative per-tick OFI)

**Compatibility**: `compute_ofi_from_lob_cache()` continues to work correctly (converts log1p volumes via `exp(x)-1` before calling).

**Key Insight**: OFI should capture order flow initiation based on both price movements AND volume changes when prices are unchanged. Pure volume sum is insufficient.

### 2026-03-21: OOM Memory Optimizations (CUDA OOM Fix)

**Files Modified**:
- `python_lab/src/train.py`
- `python_lab/src/lit_model.py`

**Changes Implemented**:

1. **Mixed Precision** (`train.py:2490`):
   - Changed `precision="32"` → `precision="16-mixed" if torch.cuda.is_available() else 32`
   - Reduces memory usage by ~50%

2. **Gradient Accumulation** (`train.py:1462`, `train.py:2495`):
   - Added `--accumulate_grad_batches` arg (default=1)
   - Passed to `pl.Trainer(accumulate_grad_batches=args.accumulate_grad_batches)`
   - Enables larger effective batch size without memory increase

3. **Gradient Checkpointing**:
   - Added `use_gradient_checkpointing: bool = False` to `LiTConfig` (`lit_model.py:14-34`)
   - `CustomTransformerEncoderLayer` (`lit_model.py:147-`) now accepts and stores `use_gradient_checkpointing`
   - Modified `forward` to use `torch.checkpoint.checkpoint(..., use_reentrant=False)` for `_sa_block` and `_ff_block` when flag enabled and training
   - Added `use_gradient_checkpointing` param to `LiTModel.__init__` and passed to all encoder layers
   - Added CLI arg `--use_gradient_checkpointing` to `train.py` and passed through all `LiTModule` instantiations (trial, student, teacher, CV fold)

**Important**: All changes maintain backward compatibility (defaults False). `nan_to_num` protections and residual connections preserved. Used `use_reentrant=False` for dropout stability.
