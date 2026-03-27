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

## Current Training State
- Active Python training pipeline is `python_lab/src/`, not `python_ab/`
- Current dataset channel layout is 11 channels: MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread
- Main training path in `train_data.py` still uses chronological `70/15/15` split
- `PurgedKFold` exists in `utils.py` and is used by `train_cv.py`, but not by the main train split
