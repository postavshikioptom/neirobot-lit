"""
train_cli.py - CLI argument parsing for train.py.
Extracted from train.py during task 322.2.
"""
import argparse


def build_train_parser() -> argparse.ArgumentParser:
    """Create and return ArgumentParser with all training flags."""
    parser = argparse.ArgumentParser(description="Train LiT model on LOB data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to train on")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--horizon", type=int, default=100, help="Prediction horizon for labels (single horizon, deprecated)")
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated list of horizons for multi-horizon prediction (e.g., '10,50,100')")
    parser.add_argument("--horizon_weights", type=str, default=None, help="Comma-separated list of weights for each horizon (e.g., '0.4,0.3,0.3')")
    parser.add_argument("--use_horizon_embedding", action="store_true", help="Use Horizon Embedding instead of separate heads")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Static return threshold (0.0005 = 0.05%)")
    parser.add_argument("--horizon_sweep", type=str, default=None,
                        help="Comma-separated sweep horizons (e.g. '10,20,50,100')")
    parser.add_argument("--threshold_sweep", type=str, default=None,
                        help="Comma-separated sweep thresholds (e.g. '0.0001,0.0005,0.0015')")
    parser.add_argument("--sweep_baseline_path", type=str, default=None,
                        help="Base path for sweep artifacts without extension")
    parser.add_argument("--sweep_use_event_rows", action=argparse.BooleanOptionalAction, default=False,
                        help="Use deduplicated event rows (feat_update_id changes only) for sweep baseline and mini-train")
    parser.add_argument("--sweep_train_topk", type=int, default=3,
                        help="Run mini-train only for top-k shortlisted sweep candidates")
    parser.add_argument("--sweep_epochs", type=int, default=1,
                        help="Epochs for mini-train inside sweep mode")
    parser.add_argument("--sweep_limit_train_batches", type=int, default=20,
                        help="Limit number of train batches per mini-train epoch in sweep mode (0 = no limit)")
    parser.add_argument("--sweep_limit_val_batches", type=int, default=20,
                        help="Limit number of val batches per mini-train epoch in sweep mode (0 = no limit)")
    parser.add_argument("--class_weight_smooth", type=float, default=1.0, help="Smoothing for class weights calculation")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for CrossEntropyLoss")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal"], help="Loss function type")
    parser.add_argument("--focal_gamma", type=float, default=3.0, help="Gamma parameter for Focal Loss")
    parser.add_argument("--metric_contract", type=str, default="standard", choices=["standard", "hft", "strict"],
                        help="Validation metric contract preset stored in checkpoint hparams")
    parser.add_argument("--metric_log_prefix", type=str, default="val",
                        help="Prefix metadata for validation metric contract reproduction")
    parser.add_argument("--metric_directional_base", type=str, default="predicted", choices=["predicted", "truth", "union"],
                        help="Directional-base metadata stored with validation contract")
    parser.add_argument("--report_fee_bps", type=float, default=0.0,
                        help="Fee in bps used for cost-aware validation edge reporting")
    parser.add_argument("--report_slippage_bps", type=float, default=0.0,
                        help="Slippage in bps used for cost-aware validation edge reporting")
    parser.add_argument("--report_half_spread_bps", type=float, default=0.0,
                        help="Half-spread in bps used for cost-aware validation edge reporting")
    parser.add_argument("--past_returns_lags", type=str, default="10,50,100", help="Comma-separated list of lags for past returns")
    parser.add_argument("--activation", type=str, default="gelu_exact", choices=["relu", "gelu_exact", "gelu_tanh", "silu"], help="Activation function type")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=96, help="Model embedding dimension (d_model)")
    parser.add_argument("--nhead", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory")

    # Data loading
    parser.add_argument("--data_mode", type=str, default="memory", choices=["memory"], help="Data loading mode: only 'memory' is supported")

    # LR scheduler
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["onecycle", "plateau", "cosine", "step", "none"], help="Learning rate scheduler type")
    parser.add_argument("--div_factor", type=float, default=25.0, help="Initial LR divisor for OneCycle/Cosine warmup")
    parser.add_argument("--final_div_factor", type=float, default=10000.0, help="Final LR divisor for OneCycle")
    parser.add_argument("--pct_start", type=float, default=0.3, help="Percentage of cycle spent increasing LR in OneCycle")
    parser.add_argument("--plateau_factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau")
    parser.add_argument("--plateau_patience", type=int, default=2, help="Patience for ReduceLROnPlateau")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for StepLR scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer")

    # Gradient clipping
    parser.add_argument("--clip_mode", type=str, default="norm", choices=["none", "norm", "agc"], help="Gradient clipping mode")
    parser.add_argument("--clip_val", type=float, default=0.5, help="Clipping threshold")

    # Time weighting
    parser.add_argument("--use_time_weighting", action="store_true", help="Enable time-decay weighting")
    parser.add_argument("--half_life_hours", type=float, default=24.0, help="Weight decay half-life in hours")
    parser.add_argument("--min_sample_weight", type=float, default=0.1, help="Minimum weight for old samples")

    # Augmentation
    parser.add_argument("--augment_prob", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--use_symmetric_flip", action="store_true", help="Enable Bid/Ask flipping with label reversal")
    parser.add_argument("--volume_jitter_range", type=float, default=0.1, help="Max relative volume change")
    parser.add_argument("--aug_seed", type=int, default=42, help="Seed for reproducible augmentation")

    parser.add_argument("--balance_method", type=str, default="none", choices=["none", "smote", "bgmm", "adasyn"], help="Dataset balancing method (deprecated)")
    parser.add_argument("--balance_ratio", type=float, default=0.5, help="Target ratio for minority classes")

    # Knowledge distillation
    parser.add_argument("--mode", type=str, default="train", choices=["train", "distill", "cv"], help="Training mode: train, distill, or cv")
    parser.add_argument("--teacher_path", type=str, default=None, help="Path to teacher model checkpoint (required for distill mode)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Weight for soft loss in distillation")
    parser.add_argument("--temperature", type=float, default=3.0, help="Temperature for softening logits in distillation")
    parser.add_argument("--student_d_model", type=int, default=64, help="Student model d_model")
    parser.add_argument("--student_nhead", type=int, default=4, help="Student model number of attention heads")
    parser.add_argument("--student_num_layers", type=int, default=2, help="Student model number of transformer layers")

    # Purged K-Fold CV
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--purge_buffer_events", type=int, default=100, help="Number of events to purge before validation fold")
    parser.add_argument("--embargo_buffer_events", type=int, default=50, help="Number of events to embargo after validation fold")

    # Optuna pruning
    parser.add_argument("--pruner_type", type=str, default="median", choices=["median", "hyperband", "patience"],
                        help="Pruner type for Optuna")
    parser.add_argument("--min_resource", type=int, default=1, help="Minimum resource (epochs) for HyperbandPruner")
    parser.add_argument("--max_resource", type=int, default=20, help="Maximum resource (epochs) for HyperbandPruner")
    parser.add_argument("--n_startup_trials", type=int, default=20, help="Number of startup trials before pruning starts")
    parser.add_argument("--n_warmup_steps", type=int, default=25, help="Number of warmup steps before first pruning check")
    parser.add_argument("--patience", type=int, default=3, help="Patience for PatientPruner")

    # TensorBoard
    parser.add_argument("--tb_dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--tb_hist_freq", type=int, default=10, help="Frequency of writing full histograms")
    parser.add_argument("--tb_embedding_samples", type=int, default=1000, help="Max samples for TensorBoard Projector")

    # Model pruning
    parser.add_argument("--prune_mode", type=str, default="none", choices=["none", "unstructured", "structured_2_4"],
                        help="Pruning mode")
    parser.add_argument("--prune_amount", type=float, default=0.5, help="Target sparsity level (0.0-0.6)")
    parser.add_argument("--prune_iterations", type=int, default=3, help="Number of prune-and-finetune iterations")
    parser.add_argument("--prune_finetune_epochs", type=int, default=2, help="Epochs of fine-tuning after each pruning iteration")

    # Micro-trades imbalance
    parser.add_argument("--trade_imb_windows", type=str, nargs="+", default=["1s", "5s", "15s", "60s"], help="Windows for trade imbalance aggregation")
    parser.add_argument("--trade_imb_agg", type=str, default="vol", choices=["vol", "count"], help="Aggregation type for imbalance")
    parser.add_argument("--trade_noise_filter_pct", type=float, default=0.05, help="Noise filter percentage")

    # Precision mode
    parser.add_argument("--precision_mode", type=str, default="auto", choices=["auto", "32", "16-mixed"],
                        help="Precision mode for A/B debug: auto | 32 | 16-mixed")

    # Curvature regularization
    parser.add_argument("--use_curvature_reg", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable curvature regularization penalty")
    parser.add_argument("--curvature_lambda", type=float, default=1e-4, help="Curvature penalty coefficient")
    parser.add_argument("--input_noise_std", type=float, default=0.005, help="Standard deviation for input noise injection")

    # Robust scaling
    parser.add_argument("--scaler_type", type=str, default="robust", choices=["zscore", "robust", "winsor_robust"], help="Scaler type")
    parser.add_argument("--winsor_limits", type=str, default="0.01,0.99", help="Winsorization limits as comma-separated floats")
    parser.add_argument("--scale_multiplier", type=float, default=1.5, help="Multiplier for Normalizer scale")

    # Optuna seq_len search
    parser.add_argument("--optuna_seq_len_search", action="store_true", help="Enable Optuna hyperparameter search for seq_len")
    parser.add_argument("--optuna_n_trials", type=int, default=10, help="Number of Optuna trials for seq_len search")
    parser.add_argument("--optuna_pruner", type=str, default="median", choices=["median", "hyperband", "patient"], help="Optuna pruner type")

    # PyTorch Profiler
    parser.add_argument("--enable_profiler", action="store_true", help="Enable PyTorch Profiler")
    parser.add_argument("--profiler_wait_steps", type=int, default=1, help="Steps to wait before profiling starts")
    parser.add_argument("--profiler_warmup_steps", type=int, default=1, help="Warmup steps for profiler")
    parser.add_argument("--profiler_active_steps", type=int, default=3, help="Active profiling steps")

    # DataLoader and Trainer
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True, help="Pin memory for DataLoader")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True, help="Persistent workers for DataLoader")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for DataLoader")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0, help="Number of sanity validation steps")
    parser.add_argument("--enable_progress_bar", action=argparse.BooleanOptionalAction, default=False, help="Enable progress bar")
    parser.add_argument("--enable_tb_embeddings", action=argparse.BooleanOptionalAction, default=False, help="Enable TensorBoard embeddings logging")
    parser.add_argument("--enable_epoch_end_plots", action=argparse.BooleanOptionalAction, default=False, help="Enable heavy epoch-end plots")
    parser.add_argument("--skip_epoch0_artifacts", action=argparse.BooleanOptionalAction, default=True, help="Skip heavy artifacts on epoch 0")
    parser.add_argument("--val_batch_log_interval", type=int, default=100, help="Log validation progress every N batches")
    parser.add_argument("--train_batch_log_interval", type=int, default=0, help="Log training batch statistics every N batches")
    parser.add_argument("--enable_vol_debug", action=argparse.BooleanOptionalAction, default=False, help="Enable volume debug prints")
    parser.add_argument("--grad_finite_check_interval", type=int, default=0, help="Interval (steps) for gradient finiteness check")
    parser.add_argument("--limit_train_batches", type=int, default=0, help="Limit number of training batches per epoch (0 = no limit)")
    parser.add_argument("--limit_val_batches", type=int, default=0, help="Limit number of validation batches per epoch (0 = no limit)")

    # Task 324.5: guard for dynamic scale
    parser.add_argument("--allow-bad-dynamic-scale", action="store_true", dest="allow_bad_dynamic_scale",
                        help="Allow training even when dynamic scale diagnostics are bad (saturation > 10%). "
                             "Default is False and training stops with RuntimeError.")

    return parser


def parse_train_args(argv=None):
    """Parse command-line arguments and return namespace."""
    parser = build_train_parser()
    return parser.parse_args(argv)


def parse_winsor_limits(raw_value: str) -> tuple:
    """Parse winsor_limits string into tuple (low, high)."""
    try:
        parts = [float(x.strip()) for x in raw_value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"winsor_limits must have exactly 2 values, got {len(parts)}")
        return tuple(parts)
    except ValueError as exc:
        raise ValueError(f"Invalid --winsor_limits format: {exc}. Expected comma-separated floats like '0.01,0.99'")


def resolve_horizon_config(args):
    """
    Resolve horizon configuration from args.
    Return (horizons_obj, num_horizons, horizon_weights_list_or_None).
    """
    if args.horizons is not None:
        horizons = [int(x.strip()) for x in args.horizons.split(",")]
        num_horizons = len(horizons)
        if args.horizon_weights is not None:
            horizon_weights = [float(x.strip()) for x in args.horizon_weights.split(",")]
            if len(horizon_weights) != num_horizons:
                raise ValueError(f"horizon_weights length ({len(horizon_weights)}) must match horizons length ({num_horizons})")
        else:
            horizon_weights = None
    else:
        horizons = args.horizon
        num_horizons = 1
        horizon_weights = None
    return horizons, num_horizons, horizon_weights
