"""
train_cli.py - CLI argument parsing for train.py.
Extracted from train.py during task 322.2.
"""
import argparse

BASELINE_PROFILE = "lit_scalping_baseline"
TASK332_PROFILE = "task332_execution_recalibration"
PROFILE_NONE = "none"
APPROVED_LABEL_CONTRACT_VERSION = "label_contract_v2"
APPROVED_METRICS_CONTRACT_VERSION = "metrics_contract_v2"

PROFILE_OVERRIDES = {
    BASELINE_PROFILE: {
        "horizons": "100",
        "freeze_experimental_features": True,
        "use_horizon_embedding": False,
        "label_mode": "execution_mid_return",
        "split_strategy": "purged_holdout",
        "loss_type": "focal",
        "decision_rule": "argmax",
        "enable_channel_attribution": False,
    },
    TASK332_PROFILE: {
        "horizons": "100",
        "freeze_experimental_features": True,
        "use_horizon_embedding": False,
        "label_mode": "execution_mid_return",
        "split_strategy": "purged_holdout",
        "loss_type": "focal",
        "decision_rule": "flat_bias",
        "decision_hold_threshold": 0.62,
        "margin_threshold": 0.02,
        "cost_floor_bps": 1.0,
        "fee_bps": 0.5,
        "slippage_bps": 0.5,
        "use_spread_floor": True,
        "report_fee_bps": 0.5,
        "report_slippage_bps": 0.5,
        "report_half_spread_bps": 0.5,
        "narrow_threshold_sweep": True,
        "threshold_sweep_span": 0.0002,
        "threshold_sweep_step": 0.0001,
        "sweep_train_topk": 2,
        "decision_rule_ablation": True,
        "quality_gate_enabled": True,
        "quality_gate_min_coverage_directional": 0.18,
        "quality_gate_fail_run": False,
        "enable_channel_attribution": False,
    }
}


def _apply_profile_overrides(args):
    overrides = PROFILE_OVERRIDES.get(args.profile)
    if overrides is None:
        return
    for key, value in overrides.items():
        setattr(args, key, value)


def _validate_frozen_experimental_paths(args):
    if not args.freeze_experimental_features:
        return
    if args.horizons is not None:
        horizons_list = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
        if len(horizons_list) > 1:
            raise ValueError(
                "Multi-horizon path is frozen: --horizons запрещён при --freeze_experimental_features, "
                "если указано более одного горизонта. Используйте single horizon."
            )
    if args.use_horizon_embedding:
        raise ValueError("Experimental horizon embedding frozen: отключите --use_horizon_embedding.")
    if args.mode == "distill":
        raise ValueError("Distillation path is frozen: --mode distill запрещён при --freeze_experimental_features.")
    if args.dynamic_threshold:
        raise ValueError("Legacy dynamic threshold frozen: --dynamic_threshold запрещён при --freeze_experimental_features.")
    if args.balance_method != "none":
        raise ValueError("Legacy balancing branch frozen: используйте --balance_method none.")
    if args.prune_mode != "none":
        raise ValueError("Pruning path is frozen: --prune_mode запрещён при --freeze_experimental_features.")
    if args.optuna_seq_len_search:
        raise ValueError("Optuna seq_len search is frozen: --optuna_seq_len_search запрещён.")
    if args.precision_mode != "auto":
        raise ValueError("Precision mode tuning is frozen: --precision_mode запрещён.")


def build_train_parser() -> argparse.ArgumentParser:
    """Create and return ArgumentParser with all training flags."""
    parser = argparse.ArgumentParser(description="Train LiT model on LOB data")
    stable_group = parser.add_argument_group("stable")
    experimental_group = parser.add_argument_group("experimental")
    deprecated_group = parser.add_argument_group("deprecated")

    stable_group.add_argument("--profile", type=str, default=PROFILE_NONE, choices=[PROFILE_NONE, BASELINE_PROFILE, TASK332_PROFILE],
                              help="Training profile. lit_scalping_baseline and task332_execution_recalibration force stable contract presets.")
    stable_group.add_argument("--freeze_experimental_features", action=argparse.BooleanOptionalAction, default=True,
                              help="Freeze multi-horizon/distillation/legacy branches.")

    stable_group.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to train on")
    stable_group.add_argument("--seq_len", type=int, default=100, help="Sequence length for the model")
    stable_group.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    stable_group.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    stable_group.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")

    deprecated_group.add_argument("--horizon", type=int, default=100,
                                  help="Prediction horizon for labels (single horizon, deprecated)")
    experimental_group.add_argument("--horizons", type=str, default=None,
                                    help="Comma-separated list of horizons for multi-horizon prediction (e.g., '10,50,100')")
    experimental_group.add_argument("--horizon_weights", type=str, default=None,
                                    help="Comma-separated list of weights for each horizon (e.g., '0.4,0.3,0.3')")
    experimental_group.add_argument("--use_horizon_embedding", action="store_true",
                                    help="Use Horizon Embedding instead of separate heads")

    stable_group.add_argument("--threshold", type=float, default=0.0005,
                              help="Static return threshold (0.0005 = 0.05%)")
    deprecated_group.add_argument("--dynamic_threshold", action=argparse.BooleanOptionalAction, default=False,
                                  help="Use legacy rolling_std*K threshold. Forbidden with --label_mode execution_mid_return.")
    stable_group.add_argument("--label_mode", type=str, default="legacy_mid_return",
                              choices=["legacy_mid_return", "execution_mid_return"],
                              help="Label contract: legacy mid-return or execution-aware mid-return")
    stable_group.add_argument("--time_mode", type=str, default="row", choices=["row", "event", "ms"],
                              help="How horizon is interpreted: rows, update events, or milliseconds")
    stable_group.add_argument("--label_contract_version", type=str, default=APPROVED_LABEL_CONTRACT_VERSION,
                              help="Label contract version marker for startup invariants.")
    stable_group.add_argument("--metrics_contract_version", type=str, default=APPROVED_METRICS_CONTRACT_VERSION,
                              help="Metrics contract version marker for startup invariants.")

    stable_group.add_argument("--event_time_column", type=str, default="feat_update_id",
                              help="Column used for event-time indexing; ms mode still uses timestamp_ms")
    stable_group.add_argument("--cost_floor_bps", type=float, default=0.0,
                              help="Minimum execution-aware cost floor in bps")
    stable_group.add_argument("--fee_bps", type=float, default=0.0,
                              help="Per-side fee in bps for execution-aware labels")
    stable_group.add_argument("--slippage_bps", type=float, default=0.0,
                              help="Slippage floor in bps for execution-aware labels")
    stable_group.add_argument("--use_spread_floor", action=argparse.BooleanOptionalAction, default=False,
                              help="Include current spread floor into execution-aware effective threshold")

    experimental_group.add_argument("--horizon_sweep", type=str, default=None,
                                    help="Comma-separated sweep horizons (e.g. '10,20,50,100')")
    experimental_group.add_argument("--threshold_sweep", type=str, default=None,
                                    help="Comma-separated sweep thresholds (e.g. '0.0001,0.0005,0.0015')")
    experimental_group.add_argument("--narrow_threshold_sweep", action=argparse.BooleanOptionalAction, default=False,
                                    help="Restrict threshold sweep to a narrow local band around --threshold")
    experimental_group.add_argument("--threshold_sweep_span", type=float, default=0.0002,
                                    help="Total span for narrow threshold sweep around --threshold")
    experimental_group.add_argument("--threshold_sweep_step", type=float, default=0.0001,
                                    help="Step for narrow threshold sweep candidates")
    experimental_group.add_argument("--sweep_baseline_path", type=str, default=None,
                                    help="Base path for sweep artifacts without extension")
    experimental_group.add_argument("--sweep_use_event_rows", action=argparse.BooleanOptionalAction, default=False,
                                    help="Use deduplicated event rows (feat_update_id changes only) for sweep baseline and mini-train")
    experimental_group.add_argument("--sweep_train_topk", type=int, default=3,
                                    help="Run mini-train only for top-k shortlisted sweep candidates")
    experimental_group.add_argument("--sweep_epochs", type=int, default=1,
                                    help="Epochs for mini-train inside sweep mode")
    experimental_group.add_argument("--sweep_limit_train_batches", type=int, default=20,
                                    help="Limit number of train batches per mini-train epoch in sweep mode (0 = no limit)")
    experimental_group.add_argument("--sweep_limit_val_batches", type=int, default=20,
                                    help="Limit number of val batches per mini-train epoch in sweep mode (0 = no limit)")

    stable_group.add_argument("--class_weight_smooth", type=float, default=1.0, help="Smoothing for class weights calculation")
    stable_group.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for CrossEntropyLoss")
    stable_group.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal"], help="Loss function type")
    stable_group.add_argument("--focal_gamma", type=float, default=3.0, help="Gamma parameter for Focal Loss")
    stable_group.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True,
                              help="Enable/disable class weights in classification loss")
    stable_group.add_argument("--multi_task", action=argparse.BooleanOptionalAction, default=True,
                              help="Enable/disable multi-task loss (classification + volatility)")
    stable_group.add_argument("--cls_loss_weight", type=float, default=1.0, help="Weight for classification loss")
    stable_group.add_argument("--vol_loss_weight", type=float, default=1.0,
                              help="Weight for volatility loss (ignored if --no-multi_task)")
    stable_group.add_argument("--metric_contract", type=str, default="standard", choices=["standard", "hft", "strict"],
                              help="Validation metric contract preset stored in checkpoint hparams")
    stable_group.add_argument("--metric_log_prefix", type=str, default="val",
                              help="Prefix metadata for validation metric contract reproduction")
    stable_group.add_argument("--metric_directional_base", type=str, default="predicted",
                              choices=["predicted", "truth", "union"],
                              help="Directional-base metadata stored with validation contract")
    stable_group.add_argument("--decision_rule", type=str, default="argmax",
                              choices=["argmax", "confidence_gap", "class_specific_thresholds", "flat_bias"],
                              help="Decision rule applied over calibrated probabilities")
    stable_group.add_argument("--decision_rule_ablation", action=argparse.BooleanOptionalAction, default=False,
                              help="Evaluate all decision rules on validation without changing model architecture")
    stable_group.add_argument("--decision_rule_ablation_rules", type=str,
                              default="argmax,confidence_gap,class_specific_thresholds,flat_bias",
                              help="Comma-separated rules for decision-rule ablation")
    stable_group.add_argument("--decision_confidence", type=float, default=0.5,
                              help="Minimum max-probability to accept directional trade")
    stable_group.add_argument("--decision_hold_threshold", type=float, default=0.6,
                              help="Directional prob threshold for flat_bias rule")
    stable_group.add_argument("--flat_prob_threshold", type=float, default=0.34,
                              help="Flat class probability threshold for class_specific_thresholds")
    stable_group.add_argument("--up_prob_threshold", type=float, default=0.34,
                              help="Up class probability threshold for class_specific_thresholds")
    stable_group.add_argument("--down_prob_threshold", type=float, default=0.34,
                              help="Down class probability threshold for class_specific_thresholds")
    stable_group.add_argument("--margin_threshold", type=float, default=0.0,
                              help="Minimum top1-top2 probability gap for confidence_gap/flat_bias rules")
    stable_group.add_argument("--report_fee_bps", type=float, default=0.0,
                              help="Fee in bps used for cost-aware validation edge reporting")
    stable_group.add_argument("--report_slippage_bps", type=float, default=0.0,
                              help="Slippage in bps used for cost-aware validation edge reporting")
    stable_group.add_argument("--report_half_spread_bps", type=float, default=0.0,
                              help="Half-spread in bps used for cost-aware validation edge reporting")
    stable_group.add_argument("--quality_gate_enabled", action=argparse.BooleanOptionalAction, default=False,
                              help="Enable run/epoch quality-gate checks for directional metrics")
    stable_group.add_argument("--quality_gate_min_coverage_directional", type=float, default=0.0,
                              help="Minimum acceptable coverage_directional for quality-gate")
    stable_group.add_argument("--quality_gate_require_non_negative_net_edge", action=argparse.BooleanOptionalAction, default=True,
                              help="Require net_edge_total >= 0 for quality-gate")
    stable_group.add_argument("--quality_gate_require_mcc_growth", action=argparse.BooleanOptionalAction, default=True,
                              help="Require val_mcc_primary growth versus previous epoch for quality-gate")
    stable_group.add_argument("--quality_gate_fail_run", action=argparse.BooleanOptionalAction, default=False,
                              help="Raise RuntimeError when run-level quality-gate fails")

    stable_group.add_argument("--past_returns_lags", type=str, default="10,50,100", help="Comma-separated list of lags for past returns")
    stable_group.add_argument("--activation", type=str, default="gelu_exact", choices=["relu", "gelu_exact", "gelu_tanh", "silu"], help="Activation function type")

    stable_group.add_argument("--d_model", type=int, default=96, help="Model embedding dimension (d_model)")
    stable_group.add_argument("--nhead", type=int, default=6, help="Number of attention heads")
    stable_group.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    stable_group.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    stable_group.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory")

    stable_group.add_argument("--data_mode", type=str, default="memory", choices=["memory"], help="Data loading mode: only 'memory' is supported")

    stable_group.add_argument("--scheduler", type=str, default="plateau", choices=["onecycle", "plateau", "cosine", "step", "none"], help="Learning rate scheduler type")
    stable_group.add_argument("--div_factor", type=float, default=25.0, help="Initial LR divisor for OneCycle/Cosine warmup")
    stable_group.add_argument("--final_div_factor", type=float, default=10000.0, help="Final LR divisor for OneCycle")
    stable_group.add_argument("--pct_start", type=float, default=0.3, help="Percentage of cycle spent increasing LR in OneCycle")
    stable_group.add_argument("--plateau_factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau")
    stable_group.add_argument("--plateau_patience", type=int, default=2, help="Patience for ReduceLROnPlateau")
    stable_group.add_argument("--step_size", type=int, default=10, help="Step size for StepLR scheduler")
    stable_group.add_argument("--gamma", type=float, default=0.5, help="Gamma for StepLR scheduler")
    stable_group.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer")

    stable_group.add_argument("--clip_mode", type=str, default="norm", choices=["none", "norm", "agc"], help="Gradient clipping mode")
    stable_group.add_argument("--clip_val", type=float, default=0.5, help="Clipping threshold")

    stable_group.add_argument("--use_time_weighting", action="store_true", help="Enable time-decay weighting")
    stable_group.add_argument("--half_life_hours", type=float, default=24.0, help="Weight decay half-life in hours")
    stable_group.add_argument("--min_sample_weight", type=float, default=0.1, help="Minimum weight for old samples")

    stable_group.add_argument("--augment_prob", type=float, default=0.5, help="Probability of applying augmentation")
    stable_group.add_argument("--use_symmetric_flip", action="store_true", help="Enable Bid/Ask flipping with label reversal")
    stable_group.add_argument("--volume_jitter_range", type=float, default=0.1, help="Max relative volume change")
    stable_group.add_argument("--aug_seed", type=int, default=42, help="Seed for reproducible augmentation")

    deprecated_group.add_argument("--balance_method", type=str, default="none", choices=["none", "smote", "bgmm", "adasyn"], help="Dataset balancing method (deprecated)")
    stable_group.add_argument("--balance_ratio", type=float, default=0.5, help="Target ratio for minority classes")

    experimental_group.add_argument("--mode", type=str, default="train", choices=["train", "distill", "cv"], help="Training mode: train, distill, or cv")
    experimental_group.add_argument("--teacher_path", type=str, default=None, help="Path to teacher model checkpoint (required for distill mode)")
    experimental_group.add_argument("--alpha", type=float, default=0.9, help="Weight for soft loss in distillation")
    experimental_group.add_argument("--temperature", type=float, default=3.0, help="Temperature for softening logits in distillation")
    experimental_group.add_argument("--student_d_model", type=int, default=64, help="Student model d_model")
    experimental_group.add_argument("--student_nhead", type=int, default=4, help="Student model number of attention heads")
    experimental_group.add_argument("--student_num_layers", type=int, default=2, help="Student model number of transformer layers")

    stable_group.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation")
    stable_group.add_argument("--split_strategy", type=str, default="purged_holdout",
                              choices=["chronological", "purged_holdout", "walk_forward"],
                              help="Data split strategy for single-run training/evaluation")
    stable_group.add_argument("--embargo_seconds", type=int, default=0,
                              help="Embargo duration in seconds near validation/test boundaries")
    stable_group.add_argument("--purge_buffer_events", type=int, default=100, help="Number of events to purge before validation fold")
    stable_group.add_argument("--embargo_buffer_events", type=int, default=50, help="Number of events to embargo after validation fold")
    stable_group.add_argument("--holdout_days", type=float, default=1.0,
                              help="Validation/test holdout window size in days for purged_holdout/walk_forward; fractional values are allowed")
    stable_group.add_argument("--training_window_days", type=int, default=7,
                              help="Training window size in days for walk_forward")

    stable_group.add_argument("--pruner_type", type=str, default="median", choices=["median", "hyperband", "patience"],
                              help="Pruner type for Optuna")
    stable_group.add_argument("--min_resource", type=int, default=1, help="Minimum resource (epochs) for HyperbandPruner")
    stable_group.add_argument("--max_resource", type=int, default=20, help="Maximum resource (epochs) for HyperbandPruner")
    stable_group.add_argument("--n_startup_trials", type=int, default=20, help="Number of startup trials before pruning starts")
    stable_group.add_argument("--n_warmup_steps", type=int, default=25, help="Number of warmup steps before first pruning check")
    stable_group.add_argument("--patience", type=int, default=3, help="Patience for PatientPruner")

    stable_group.add_argument("--tb_dir", type=str, default=None, help="TensorBoard log directory")
    stable_group.add_argument("--tb_hist_freq", type=int, default=10, help="Frequency of writing full histograms")
    stable_group.add_argument("--tb_embedding_samples", type=int, default=1000, help="Max samples for TensorBoard Projector")

    experimental_group.add_argument("--prune_mode", type=str, default="none", choices=["none", "unstructured", "structured_2_4"], help="Pruning mode")
    experimental_group.add_argument("--prune_amount", type=float, default=0.5, help="Target sparsity level (0.0-0.6)")
    experimental_group.add_argument("--prune_iterations", type=int, default=3, help="Number of prune-and-finetune iterations")
    experimental_group.add_argument("--prune_finetune_epochs", type=int, default=2, help="Epochs of fine-tuning after each pruning iteration")

    experimental_group.add_argument("--trade_imb_windows", type=str, nargs="+", default=["1s", "5s", "15s", "60s"], help="Windows for trade imbalance aggregation")
    experimental_group.add_argument("--trade_imb_agg", type=str, default="vol", choices=["vol", "count"], help="Aggregation type for imbalance")
    experimental_group.add_argument("--trade_noise_filter_pct", type=float, default=0.05, help="Noise filter percentage")

    experimental_group.add_argument("--precision_mode", type=str, default="auto", choices=["auto", "32", "16-mixed"],
                                    help="Precision mode for A/B debug: auto | 32 | 16-mixed")

    experimental_group.add_argument("--use_curvature_reg", action=argparse.BooleanOptionalAction, default=False,
                                    help="Enable/disable curvature regularization penalty")
    experimental_group.add_argument("--curvature_lambda", type=float, default=1e-4, help="Curvature penalty coefficient")
    experimental_group.add_argument("--input_noise_std", type=float, default=0.005, help="Standard deviation for input noise injection")

    experimental_group.add_argument("--scaler_type", type=str, default="robust", choices=["zscore", "robust", "winsor_robust"], help="Scaler type")
    experimental_group.add_argument("--winsor_limits", type=str, default="0.01,0.99", help="Winsorization limits as comma-separated floats")
    experimental_group.add_argument("--scale_multiplier", type=float, default=1.5, help="Multiplier for Normalizer scale")

    experimental_group.add_argument("--optuna_seq_len_search", action="store_true", help="Enable Optuna hyperparameter search for seq_len")
    experimental_group.add_argument("--optuna_n_trials", type=int, default=10, help="Number of Optuna trials for seq_len search")
    experimental_group.add_argument("--optuna_pruner", type=str, default="median", choices=["median", "hyperband", "patient"], help="Optuna pruner type")

    experimental_group.add_argument("--enable_profiler", action="store_true", help="Enable PyTorch Profiler")
    experimental_group.add_argument("--profiler_wait_steps", type=int, default=1, help="Steps to wait before profiling starts")
    experimental_group.add_argument("--profiler_warmup_steps", type=int, default=1, help="Warmup steps for profiler")
    experimental_group.add_argument("--profiler_active_steps", type=int, default=3, help="Active profiling steps")

    stable_group.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    stable_group.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True, help="Pin memory for DataLoader")
    stable_group.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True, help="Persistent workers for DataLoader")
    stable_group.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for DataLoader")
    stable_group.add_argument("--num_sanity_val_steps", type=int, default=0, help="Number of sanity validation steps")
    stable_group.add_argument("--enable_progress_bar", action=argparse.BooleanOptionalAction, default=False, help="Enable progress bar")
    stable_group.add_argument("--enable_tb_embeddings", action=argparse.BooleanOptionalAction, default=False, help="Enable TensorBoard embeddings logging")
    stable_group.add_argument("--enable_epoch_end_plots", action=argparse.BooleanOptionalAction, default=False, help="Enable heavy epoch-end plots")
    stable_group.add_argument("--skip_epoch0_artifacts", action=argparse.BooleanOptionalAction, default=True, help="Skip heavy artifacts on epoch 0")
    stable_group.add_argument("--enable_channel_attribution", action=argparse.BooleanOptionalAction, default=False,
                              help="Enable post-hoc channel attribution logging on validation epoch end")
    stable_group.add_argument("--channel_attribution_samples", type=int, default=128,
                              help="Maximum number of validation samples used for channel attribution per epoch")
    stable_group.add_argument("--channel_attribution_method", type=str, default="grad_x_input", choices=["grad_x_input", "occlusion"],
                              help="Channel attribution method: gradient x input or channel occlusion")
    stable_group.add_argument("--val_batch_log_interval", type=int, default=100, help="Log validation progress every N batches")
    stable_group.add_argument("--train_batch_log_interval", type=int, default=0, help="Log training batch statistics every N batches")
    stable_group.add_argument("--enable_vol_debug", action=argparse.BooleanOptionalAction, default=False, help="Enable volume debug prints")
    stable_group.add_argument("--grad_finite_check_interval", type=int, default=0, help="Interval (steps) for gradient finiteness check")
    stable_group.add_argument("--limit_train_batches", type=int, default=0, help="Limit number of training batches per epoch (0 = no limit)")
    stable_group.add_argument("--limit_val_batches", type=int, default=0, help="Limit number of validation batches per epoch (0 = no limit)")

    stable_group.add_argument("--allow-bad-dynamic-scale", action="store_true", dest="allow_bad_dynamic_scale",
                              help="Allow training even when dynamic scale diagnostics are bad (saturation > 10%). "
                                   "Default is False and training stops with RuntimeError.")

    return parser


def parse_train_args(argv=None):
    """Parse command-line arguments and return namespace."""
    parser = build_train_parser()
    args = parser.parse_args(argv)
    _apply_profile_overrides(args)
    _validate_frozen_experimental_paths(args)
    if args.label_mode == "execution_mid_return" and args.dynamic_threshold:
        raise ValueError(
            "--label_mode execution_mid_return несовместим с --dynamic_threshold. "
            "Legacy dynamic threshold разрешён только для legacy/debug режима."
        )
    if args.cls_loss_weight < 0.0:
        raise ValueError("--cls_loss_weight must be >= 0.0")
    if args.vol_loss_weight < 0.0:
        raise ValueError("--vol_loss_weight must be >= 0.0")

    def _validate_prob(name: str, value: float):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name} must be in [0.0, 1.0], got {value}")

    _validate_prob("decision_confidence", args.decision_confidence)
    _validate_prob("decision_hold_threshold", args.decision_hold_threshold)
    _validate_prob("flat_prob_threshold", args.flat_prob_threshold)
    _validate_prob("up_prob_threshold", args.up_prob_threshold)
    _validate_prob("down_prob_threshold", args.down_prob_threshold)
    _validate_prob("margin_threshold", args.margin_threshold)
    if args.quality_gate_min_coverage_directional < 0.0 or args.quality_gate_min_coverage_directional > 1.0:
        raise ValueError(
            "--quality_gate_min_coverage_directional must be in [0.0, 1.0], "
            f"got {args.quality_gate_min_coverage_directional}"
        )
    if args.threshold_sweep_span < 0.0:
        raise ValueError("--threshold_sweep_span must be >= 0.0")
    if args.threshold_sweep_step <= 0.0:
        raise ValueError("--threshold_sweep_step must be > 0.0")
    ablation_rules = [item.strip() for item in args.decision_rule_ablation_rules.split(",") if item.strip()]
    allowed_rules = {"argmax", "confidence_gap", "class_specific_thresholds", "flat_bias"}
    invalid_rules = sorted(set(ablation_rules) - allowed_rules)
    if invalid_rules:
        raise ValueError(f"--decision_rule_ablation_rules contains unsupported rules: {invalid_rules}")
    args.decision_rule_ablation_rules = ",".join(ablation_rules)
    if args.embargo_seconds < 0:
        raise ValueError("--embargo_seconds must be >= 0")
    if args.purge_buffer_events < 0:
        raise ValueError("--purge_buffer_events must be >= 0")
    if args.embargo_buffer_events < 0:
        raise ValueError("--embargo_buffer_events must be >= 0")
    if args.holdout_days <= 0:
        raise ValueError("--holdout_days must be > 0")
    if args.training_window_days <= 0:
        raise ValueError("--training_window_days must be > 0")
    if args.split_strategy == "walk_forward" and args.mode == "cv":
        raise ValueError("--split_strategy walk_forward несовместим с --mode cv")
    if args.channel_attribution_samples <= 0:
        raise ValueError("--channel_attribution_samples must be > 0")
    return args


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
        horizons_list = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
        if not horizons_list:
            raise ValueError("--horizons provided but no valid horizon values parsed.")
        if len(horizons_list) == 1:
            horizons = horizons_list[0]
            num_horizons = 1
            if args.horizon_weights is not None:
                print("[WARN] --horizon_weights ignored for single-horizon run.")
            horizon_weights = None
        else:
            horizons = horizons_list
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
