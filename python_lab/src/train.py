"""
train.py — Thin entrypoint для обучения LiT модели.
Рефакторинг выполнен в рамках задачи 322.
Вся логика вынесена в отдельные модули:
  train_cli, train_runtime, train_data, train_model_factory,
  train_optuna, train_cv, train_postprocess, train_module, train_metadata.
"""
import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .train_cli import parse_train_args, parse_winsor_limits, resolve_horizon_config
from .train_runtime import (
    build_train_paths,
    seed_training,
    warn_if_dataset_may_exceed_ram,
    resolve_trainer_precision,
)
from .train_data import prepare_training_data
from .train_model_factory import build_training_module
from .train_optuna import run_optuna_seq_len_search
from .train_cv import run_cross_validation
from .train_postprocess import (
    run_mc_dropout_uncertainty,
    run_model_pruning,
    run_holdout_evaluation,
    compare_teacher_student,
    copy_best_checkpoint_to_target,
)
from .train_module import ProfilerCallback
from .train_metadata import update_model_metadata


def train():
    # 1. CLI + winsor limits
    args = parse_train_args()
    winsor_limits = parse_winsor_limits(args.winsor_limits)
    horizons, num_horizons, horizon_weights = resolve_horizon_config(args)
    print(f'Scaler configuration: type={args.scaler_type}, winsor_limits={winsor_limits}')

    # 2. Валидация аргументов прунинга
    if args.prune_mode != 'none':
        if args.prune_amount < 0.0 or args.prune_amount > 0.6:
            raise ValueError(f'--prune_amount должен быть в диапазоне [0.0, 0.6], получено: {args.prune_amount}')
        if args.prune_iterations < 1:
            raise ValueError(f'--prune_iterations должен быть >= 1, получено: {args.prune_iterations}')
        if args.prune_finetune_epochs < 1:
            raise ValueError(f'--prune_finetune_epochs должен быть >= 1, получено: {args.prune_finetune_epochs}')
        if args.mode == 'cv':
            raise ValueError('Pruning не поддерживается в режиме cv.')

    # 3. Seed + paths
    seed_training()
    paths = build_train_paths(__file__, args.symbol)
    warn_if_dataset_may_exceed_ram(paths, args.symbol, args.seq_len)

    # 4. Data pipeline
    prepared = prepare_training_data(args, paths, winsor_limits, horizons, num_horizons, horizon_weights)
    update_model_metadata(paths.base_path, args.symbol, args, winsor_limits, paths.norm_params_path)

    # 5. Optuna seq_len search (опционально)
    if args.optuna_seq_len_search:
        prepared = run_optuna_seq_len_search(args, paths, prepared, winsor_limits)

    # 6. CV режим — отдельная ветка
    if args.mode == "cv":
        run_cross_validation(args, paths, prepared, winsor_limits)
        return

    # 7. Сборка модели (train / distill)
    model_class_weights = None if args.use_time_weighting else prepared.class_weights
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
    )
    model = built.module
    teacher_model = built.teacher_model

    # 8. Callbacks + Logger
    checkpoint_dir = paths.checkpoint_dir
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="lit-{epoch:02d}-{val_mcc:.4f}",
        save_top_k=3,
        monitor="val_mcc",
        mode="max",
    )
    callbacks = [
        EarlyStopping(monitor="val_mcc", patience=15, mode="max"),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.enable_profiler:
        callbacks.append(ProfilerCallback(
            wait_steps=args.profiler_wait_steps,
            warmup_steps=args.profiler_warmup_steps,
            active_steps=args.profiler_active_steps,
            profiler_dir=f"profiler_logs/{args.symbol}",
        ))
        print(f"\n[PROFILER] PyTorch Profiler enabled. Results will be saved to profiler_logs/{args.symbol}")

    tb_dir = args.tb_dir if args.tb_dir else f"runs/{args.symbol}"
    from .utils import cleanup_old_tensorboard_logs, setup_custom_scalars_layout, log_hparams
    cleanup_old_tensorboard_logs(tb_dir, max_runs=50)
    logger = TensorBoardLogger(tb_dir, name="lit_training")
    setup_custom_scalars_layout(logger.experiment)

    hparams_dict = {
        'lr': 1e-4,
        'd_model': args.d_model if args.mode != 'distill' else args.student_d_model,
        'nhead': args.nhead if args.mode != 'distill' else args.student_nhead,
        'num_layers': args.num_layers if args.mode != 'distill' else args.student_num_layers,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'activation': args.activation,
        'scheduler': args.scheduler,
        'loss_type': args.loss_type,
        'label_smoothing': args.label_smoothing,
    }
    log_hparams(logger.experiment, hparams_dict, {})

    # 9. Trainer
    trainer_precision = resolve_trainer_precision(args)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision=trainer_precision,
        log_every_n_steps=100,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=False,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
    )
    trainer.symbol = args.symbol

    # 10. Sanity check
    print("Performing sanity check on data...")
    try:
        batch = next(iter(prepared.train_loader))
        x_check = batch[0]
        if not torch.isfinite(x_check).all():
            raise ValueError("NaN or Inf detected in input features before training!")
        print(f"Sanity check passed. Input shape: {x_check.shape}, range: [{x_check.min():.4f}, {x_check.max():.4f}]")
    except Exception as e:
        print(f"Sanity check failed: {e}")
        if not args.optuna_seq_len_search:
            raise e

    effective_label_smoothing = 0.0 if args.loss_type == "focal" else args.label_smoothing
    print(f"Label smoothing: requested={args.label_smoothing}, effective={effective_label_smoothing}")

    # 11. Передача hparams в модель для хуков
    model.hparams.val_batch_log_interval = args.val_batch_log_interval
    model.hparams.enable_epoch_end_plots = args.enable_epoch_end_plots
    model.hparams.skip_epoch0_artifacts = args.skip_epoch0_artifacts
    model.hparams.enable_tb_embeddings = args.enable_tb_embeddings

    # 12. Обучение
    print("Starting training...")
    trainer.fit(model, prepared.train_loader, prepared.val_loader)

    # Логируем итоговые метрики
    best_val_mcc = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0.0
    log_hparams(logger.experiment, hparams_dict, {'hparam/best_val_mcc': best_val_mcc})

    # 13. MC Dropout
    run_mc_dropout_uncertainty(
        model, prepared.val_loader, checkpoint_dir,
        in_channels=prepared.in_channels, seq_len=args.seq_len,
    )

    # 14. Pruning
    model = run_model_pruning(
        model, args,
        prepared.train_loader, prepared.val_loader,
        checkpoint_callback, logger,
        paths.base_path, args.symbol,
    )

    # 15. Holdout evaluation
    best_model_path = checkpoint_callback.best_model_path
    y_true, y_pred, best_model = run_holdout_evaluation(
        best_model_path, prepared.test_loader, paths.base_path, args.symbol,
    )

    # 16. Teacher vs Student (только distill)
    if y_true is not None and args.mode == "distill":
        compare_teacher_student(
            args, y_true, y_pred, best_model,
            prepared.test_loader, teacher_model,
            paths.base_path, args.symbol,
        )

    # 17. Копируем лучший checkpoint
    copy_best_checkpoint_to_target(args, best_model_path, paths.base_path, args.symbol)

    print("\nEvaluation completed. Run 'python evaluate.py --checkpoint PATH' for uncertainty and interpretability analysis.")


if __name__ == "__main__":
    train()
