"""
train.py - Thin entrypoint for LiT model training.
Refactoring was done during task 322.
"""
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .train_cli import parse_train_args, parse_winsor_limits, resolve_horizon_config
from .train_cv import run_cross_validation
from .train_data import (
    clone_args_with_overrides,
    collect_sweep_baseline,
    export_sweep_baseline,
    is_sweep_mode,
    load_feature_frame,
    prepare_training_data,
    resolve_sweep_grid,
    select_event_rows,
    shortlist_sweep_candidates,
)
from .train_metadata import update_model_metadata
from .train_model_factory import build_training_module
from .train_module import ProfilerCallback
from .train_optuna import run_optuna_seq_len_search
from .train_postprocess import (
    compare_teacher_student,
    copy_best_checkpoint_to_target,
    run_holdout_evaluation,
    run_mc_dropout_uncertainty,
    run_model_pruning,
)
from .train_runtime import (
    build_train_paths,
    resolve_trainer_precision,
    seed_training,
    warn_if_dataset_may_exceed_ram,
)


def _build_logger(tb_dir: str):
    from .utils import cleanup_old_tensorboard_logs, setup_custom_scalars_layout

    cleanup_old_tensorboard_logs(tb_dir, max_runs=50)
    logger = TensorBoardLogger(tb_dir, name="lit_training")
    setup_custom_scalars_layout(logger.experiment)
    return logger


def _log_hparams(args, logger):
    from .utils import log_hparams

    hparams_dict = {
        "lr": 1e-4,
        "d_model": args.d_model if args.mode != "distill" else args.student_d_model,
        "nhead": args.nhead if args.mode != "distill" else args.student_nhead,
        "num_layers": args.num_layers if args.mode != "distill" else args.student_num_layers,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "activation": args.activation,
        "scheduler": args.scheduler,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
    }
    log_hparams(logger.experiment, hparams_dict, {})
    return hparams_dict


def _build_callbacks(args, checkpoint_dir: Path, *, patience: int = 15, save_top_k: int = 3):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="lit-{epoch:02d}-{val_mcc_primary:.4f}",
        save_top_k=save_top_k,
        monitor="val_mcc_primary",
        mode="max",
    )
    callbacks = [
        EarlyStopping(monitor="val_mcc_primary", patience=patience, mode="max"),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.enable_profiler:
        callbacks.append(
            ProfilerCallback(
                wait_steps=args.profiler_wait_steps,
                warmup_steps=args.profiler_warmup_steps,
                active_steps=args.profiler_active_steps,
                profiler_dir=f"profiler_logs/{args.symbol}",
            )
        )
        print(f"\n[PROFILER] PyTorch Profiler enabled. Results will be saved to profiler_logs/{args.symbol}")

    return callbacks, checkpoint_callback


def _build_trainer(args, callbacks, logger, *, max_epochs: int, limit_train_batches: int, limit_val_batches: int):
    trainer_precision = resolve_trainer_precision(args)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision=trainer_precision,
        log_every_n_steps=100,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=False,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=limit_train_batches if limit_train_batches > 0 else 1.0,
        limit_val_batches=limit_val_batches if limit_val_batches > 0 else 1.0,
    )
    return trainer


def _prepare_model_for_fit(args, prepared, winsor_limits):
    model_class_weights = None if args.use_time_weighting else prepared.class_weights
    if prepared.class_weight_metadata.get("label_mode") != args.label_mode:
        raise ValueError(
            "Class weight label_mode mismatch: "
            f"weights={prepared.class_weight_metadata.get('label_mode')} vs args={args.label_mode}"
        )
    built = build_training_module(
        args,
        in_channels=prepared.in_channels,
        past_returns_lags=prepared.past_returns_lags,
        num_horizons=prepared.num_horizons,
        horizon_weights=prepared.horizon_weights,
        model_class_weights=model_class_weights,
        regime_detector=prepared.regime_detector,
        regime_weights=prepared.regime_weights,
        num_regimes=prepared.num_regimes,
        winsor_limits=winsor_limits,
        label_columns=prepared.label_columns,
        class_weight_metadata=prepared.class_weight_metadata,
    )
    return built.module, built.teacher_model


def _sanity_check_prepared(prepared, args):
    print("Performing sanity check on data...")
    try:
        batch = next(iter(prepared.train_loader))
        x_check = batch[0]
        if not torch.isfinite(x_check).all():
            raise ValueError("NaN or Inf detected in input features before training!")
        print(f"Sanity check passed. Input shape: {x_check.shape}, range: [{x_check.min():.4f}, {x_check.max():.4f}]")
    except Exception as exc:
        print(f"Sanity check failed: {exc}")
        if not args.optuna_seq_len_search:
            raise


def _fit_model(
    args,
    prepared,
    winsor_limits,
    *,
    symbol_tag: str,
    checkpoint_dir: Path,
    tb_dir: str,
    max_epochs: int,
    limit_train_batches: int,
    limit_val_batches: int,
    patience: int,
    save_top_k: int,
):
    model, teacher_model = _prepare_model_for_fit(args, prepared, winsor_limits)
    callbacks, checkpoint_callback = _build_callbacks(args, checkpoint_dir, patience=patience, save_top_k=save_top_k)
    logger = _build_logger(tb_dir)
    hparams_dict = _log_hparams(args, logger)
    trainer = _build_trainer(
        args,
        callbacks,
        logger,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer.symbol = symbol_tag

    effective_label_smoothing = 0.0 if args.loss_type == "focal" else args.label_smoothing
    print(f"Label smoothing: requested={args.label_smoothing}, effective={effective_label_smoothing}")

    model.hparams.val_batch_log_interval = args.val_batch_log_interval
    model.hparams.enable_epoch_end_plots = args.enable_epoch_end_plots
    model.hparams.skip_epoch0_artifacts = args.skip_epoch0_artifacts
    model.hparams.enable_tb_embeddings = args.enable_tb_embeddings

    _sanity_check_prepared(prepared, args)
    print("Starting training...")
    trainer.fit(model, prepared.train_loader, prepared.val_loader)

    from .utils import log_hparams

    best_val_mcc_primary = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0.0
    log_hparams(logger.experiment, hparams_dict, {"hparam/best_val_mcc_primary": best_val_mcc_primary})

    return trainer, model, teacher_model, checkpoint_callback


def _extract_mini_train_metrics(trainer) -> dict:
    def _metric(name: str) -> float | None:
        value = trainer.callback_metrics.get(name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    return {
        "mini_train_mcc": _metric("val_mcc_primary"),
        "mini_train_coverage_directional": _metric("coverage_directional"),
        "mini_train_net_edge_total": _metric("net_edge_total"),
    }


def _append_train_log(paths, csv_path: Path, json_path: Path, baselines_path: Path, topk: int):
    train_logs_path = paths.base_path / "docs" / "train_logs.md"
    entry = (
        "\n## Р—Р°РґР°С‡Р° 326 | Baseline sweep\n\n"
        f"- CSV: `{csv_path.relative_to(paths.base_path).as_posix()}`\n"
        f"- JSON: `{json_path.relative_to(paths.base_path).as_posix()}`\n"
        f"- РћС‚С‡С‘С‚: `{baselines_path.relative_to(paths.base_path).as_posix()}`\n"
        f"- Shortlist top-k: `{topk}`\n"
    )
    existing = train_logs_path.read_text(encoding="utf-8") if train_logs_path.exists() else ""
    if entry not in existing:
        train_logs_path.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")


def _run_sweep_mode(args, paths, winsor_limits):
    horizons, thresholds = resolve_sweep_grid(args)
    feature_df = load_feature_frame(args, paths, use_event_rows=False)

    artifacts = collect_sweep_baseline(
        feature_df,
        horizons=horizons,
        thresholds=thresholds,
        use_event_rows=args.sweep_use_event_rows,
    )
    artifacts.symbol = args.symbol
    artifacts.horizons = list(horizons)
    artifacts.thresholds = list(thresholds)

    shortlisted = shortlist_sweep_candidates(artifacts.grid, args.sweep_train_topk)
    candidate_feature_df = select_event_rows(feature_df) if args.sweep_use_event_rows else feature_df

    for row in shortlisted:
        print(f"[SWEEP] Mini-train for h={row.horizon}, thr={row.threshold:.4f}")
        candidate_args = clone_args_with_overrides(
            args,
            threshold=row.threshold,
            horizon=row.horizon,
            horizons=None,
            epochs=args.sweep_epochs,
            limit_train_batches=args.sweep_limit_train_batches,
            limit_val_batches=args.sweep_limit_val_batches,
        )

        prepared = prepare_training_data(
            candidate_args,
            paths,
            winsor_limits,
            row.horizon,
            1,
            None,
            feature_df=candidate_feature_df,
        )
        run_tag = f"{args.symbol}_sweep_h{row.horizon}_thr{str(row.threshold).replace('.', '_')}"
        checkpoint_dir = paths.base_path / "artifacts" / run_tag / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = str(paths.base_path / "runs" / run_tag)
        trainer, _, _, _ = _fit_model(
            candidate_args,
            prepared,
            winsor_limits,
            symbol_tag=run_tag,
            checkpoint_dir=checkpoint_dir,
            tb_dir=tb_dir,
            max_epochs=args.sweep_epochs,
            limit_train_batches=args.sweep_limit_train_batches,
            limit_val_batches=args.sweep_limit_val_batches,
            patience=max(1, args.sweep_epochs),
            save_top_k=1,
        )
        metrics = _extract_mini_train_metrics(trainer)
        row.mini_train_mcc = metrics["mini_train_mcc"]
        row.mini_train_coverage_directional = metrics["mini_train_coverage_directional"]
        row.mini_train_net_edge_total = metrics["mini_train_net_edge_total"]

    artifacts.shortlist = [
        {
            "candidate_id": f"h{row.horizon}_thr{row.threshold:.4f}".replace('.', 'p'),
            "candidate_rank": row.shortlist_rank,
            "horizon": row.horizon,
            "threshold": row.threshold,
            "trade_share": row.trade_share,
            "share_flat": row.share_flat,
            "share_up": row.share_up,
            "share_down": row.share_down,
            "threshold_to_spread_ratio": row.threshold_to_spread_ratio,
            "mini_train_mcc": row.mini_train_mcc,
            "mini_train_coverage_directional": row.mini_train_coverage_directional,
            "mini_train_net_edge_total": row.mini_train_net_edge_total,
        }
        for row in shortlisted
    ]

    csv_path, json_path = export_sweep_baseline(paths, args, artifacts)

    print(f"[SWEEP] Baseline CSV saved to {csv_path}")
    print(f"[SWEEP] Baseline JSON saved to {json_path}")
    print(f"[SWEEP] Baselines report saved to {paths.base_path / 'docs' / 'baselines.md'}")


def train():
    args = parse_train_args()
    if args.label_mode == "execution_mid_return" and args.dynamic_threshold:
        raise ValueError(
            "dynamic_threshold разрешен только для legacy/debug режима. "
            "Сочетание execution_mid_return + dynamic_threshold запрещено."
        )
    winsor_limits = parse_winsor_limits(args.winsor_limits)
    horizons, num_horizons, horizon_weights = resolve_horizon_config(args)
    print(f"Scaler configuration: type={args.scaler_type}, winsor_limits={winsor_limits}")

    if args.prune_mode != "none":
        if args.prune_amount < 0.0 or args.prune_amount > 0.6:
            raise ValueError(f"--prune_amount must be in [0.0, 0.6], got: {args.prune_amount}")
        if args.prune_iterations < 1:
            raise ValueError(f"--prune_iterations must be >= 1, got: {args.prune_iterations}")
        if args.prune_finetune_epochs < 1:
            raise ValueError(f"--prune_finetune_epochs must be >= 1, got: {args.prune_finetune_epochs}")
        if args.mode == "cv":
            raise ValueError("Pruning is not supported in cv mode.")

    seed_training()
    paths = build_train_paths(__file__, args.symbol)
    warn_if_dataset_may_exceed_ram(paths, args.symbol, args.seq_len)

    if is_sweep_mode(args):
        _run_sweep_mode(args, paths, winsor_limits)
        return

    prepared = prepare_training_data(args, paths, winsor_limits, horizons, num_horizons, horizon_weights)
    update_model_metadata(paths.base_path, args.symbol, args, winsor_limits, paths.norm_params_path)

    if args.optuna_seq_len_search:
        prepared = run_optuna_seq_len_search(args, paths, prepared, winsor_limits)

    if args.mode == "cv":
        run_cross_validation(args, paths, prepared, winsor_limits)
        return

    checkpoint_dir = paths.checkpoint_dir
    trainer, model, teacher_model, checkpoint_callback = _fit_model(
        args,
        prepared,
        winsor_limits,
        symbol_tag=args.symbol,
        checkpoint_dir=checkpoint_dir,
        tb_dir=args.tb_dir if args.tb_dir else f"runs/{args.symbol}",
        max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        patience=15,
        save_top_k=3,
    )

    run_mc_dropout_uncertainty(
        model,
        prepared.val_loader,
        checkpoint_dir,
        in_channels=prepared.in_channels,
        seq_len=args.seq_len,
    )

    model = run_model_pruning(
        model,
        args,
        prepared.train_loader,
        prepared.val_loader,
        checkpoint_callback,
        trainer.logger,
        paths.base_path,
        args.symbol,
    )

    best_model_path = checkpoint_callback.best_model_path
    y_true, y_pred, best_model = run_holdout_evaluation(
        best_model_path,
        prepared.test_loader,
        paths.base_path,
        args.symbol,
    )

    if y_true is not None and args.mode == "distill":
        compare_teacher_student(
            args,
            y_true,
            y_pred,
            best_model,
            prepared.test_loader,
            teacher_model,
            paths.base_path,
            args.symbol,
        )

    copy_best_checkpoint_to_target(args, best_model_path, paths.base_path, args.symbol)
    print("\nEvaluation completed. Run 'python evaluate.py --checkpoint PATH' for uncertainty and interpretability analysis.")


if __name__ == "__main__":
    train()


