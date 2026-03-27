## Stable Repo Facts

- 2026-03-26: Current Python/Rust LiT pipeline is effectively 11-channel, not 13-channel. Active channels in dataset/model/export are `MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread`. See `python_lab/src/train_data.py`, `python_lab/src/lit_model.py`, `python_lab/src/export_onnx.py`.
- 2026-03-26: Current labeler uses ternary labels from simple future mid-price return with fixed row/event horizon and symmetric deadband: `label = sign((mid_{t+h}-mid_t)/mid_t, threshold)`. Default CLI is `horizon=100`, `threshold=0.0005`. See `python_lab/src/labels.py` and `python_lab/src/train_cli.py`.
- 2026-03-26: Default train path uses chronological `70/15/15` split without purge/embargo. Purged K-Fold exists, but only in `mode=cv`; default `train` mode does not use it. See `python_lab/src/train_data.py` and `python_lab/src/train_cv.py`.
- 2026-03-26: Normalizer fit is train-only in the main pipeline and dynamic channels use `symlog -> robust -> clamp`, with guardrails for saturation/zero collapse. See `python_lab/src/train_data.py` and `python_lab/src/normalization.py`.
- 2026-03-26: Validation and holdout decisions are mostly `argmax(logits)`; there is calibration/ECE logging, but no default class-specific decision thresholds or abstention rule in the main training path. See `python_lab/src/train_module.py`.
- 2026-03-26: Current `LiTModel` is transformer-only with `LOBPatching`; it does not include the LSTM stage from the original LiT paper. See `python_lab/src/lit_model.py`.
