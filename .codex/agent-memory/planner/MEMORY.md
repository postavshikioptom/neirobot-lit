# Planner Memory

## Project Structure
- Task docs go in `docs/` as `NNN-english-slug.md`
- Task list is `docs/000-tasks_list.md` — append new entries at the end
- Python ML code is in `python_lab/src/`
- Key files: `dataset.py` (features/normalization), `lit_model.py` (model architecture), `train.py` (training loop), `labels.py` (labeling logic)
- Latest completed task number is tracked in `000-tasks_list.md` — always check before creating new plan

## Conventions
- One task per plan file, with detailed sub-tasks
- Each sub-task references specific file paths and line numbers
- Include verification steps for each sub-task
- Final sub-task is always "run training and compare metrics"
- Plans are written in Russian with English code snippets

## Channel Architecture (current state as of task 318)
- Dataset produces channels as (seq_len, n_channels, 50) tensors
- Channel order (13 channels): MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread, CumOFI, ImbAccel
- `_calculate_6_channels_raw()` is the single source of truth for channel computation
- Normalization is done per-channel in `normalize_channel()` using normalizer params
- `in_channels` must match between `LiTConfig` (lit_model.py), hardcoded value (train.py), and actual output from dataset
- Task 319 plans to reduce to 11 channels (remove CumOFI, ImbAccel) and rework LOBPatching
