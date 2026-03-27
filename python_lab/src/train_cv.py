"""
train_cv.py — Purged K-Fold Cross-Validation для train.py.
Вынесено из train.py в рамках задачи 322.8.
"""
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import matthews_corrcoef
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .train_module import LiTModule, TrainSubset
from .train_runtime import build_dataloader_kwargs, resolve_trainer_precision


def run_cross_validation(args, paths, prepared, winsor_limits):
    """
    Purged K-Fold Cross-Validation (Задача 153).
    Fold-модели создаются fresh, без reuse основного model.
    Пишет cv_results.json и оценивает holdout лучшей fold-моделью.
    """
    from .utils import PurgedKFold
    from .lit_model import LiTConfig

    base_path = paths.base_path
    checkpoint_dir = paths.checkpoint_dir

    full_dataset = prepared.full_dataset
    test_ds = prepared.test_ds
    test_loader = prepared.test_loader
    in_channels = prepared.in_channels
    past_returns_lags = prepared.past_returns_lags
    num_horizons = prepared.num_horizons
    horizon_weights = prepared.horizon_weights
    regime_detector = prepared.regime_detector
    regime_weights = prepared.regime_weights
    num_regimes = prepared.num_regimes
    weights = prepared.class_weights

    model_class_weights = None if args.use_time_weighting else weights

    print("\n" + "=" * 70)
    print("PURGED K-FOLD CROSS-VALIDATION MODE")
    print("=" * 70)

    cv = PurgedKFold(
        n_splits=args.n_splits,
        purge_buffer_events=args.purge_buffer_events,
        embargo_buffer_events=args.embargo_buffer_events,
    )

    timestamps = full_dataset.get_timestamps()
    total_len = len(full_dataset)
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    cv_size = train_size + val_size

    cv_indices = np.arange(cv_size)
    cv_timestamps = timestamps[:cv_size]
    cv_labels = full_dataset.labels[:cv_size]

    fold_mccs = []
    fold_results = []

    print(f"\nCV Configuration:")
    print(f"  - Number of folds: {args.n_splits}")
    print(f"  - Purge buffer: {args.purge_buffer_events} events")
    print(f"  - Embargo buffer: {args.embargo_buffer_events} events")
    print(f"  - Total CV samples: {cv_size}")
    print(f"  - Holdout test samples: {len(test_ds)}")
    print()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(cv_indices, cv_labels, cv_timestamps)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{args.n_splits}")
        print(f"{'='*70}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        fold_train_ds = TrainSubset(full_dataset, list(train_idx))
        fold_val_ds = Subset(full_dataset, list(val_idx))

        fold_train_loader = DataLoader(fold_train_ds, **build_dataloader_kwargs(args, shuffle=True))
        fold_val_loader = DataLoader(fold_val_ds, **build_dataloader_kwargs(args, shuffle=False))

        fold_config = LiTConfig(
            seq_len=args.seq_len,
            in_channels=in_channels,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            multi_task=True,
            num_horizons=num_horizons,
            use_horizon_embedding=args.use_horizon_embedding,
        )

        fold_model = LiTModule(
            seq_len=fold_config.seq_len,
            lr=1e-4,
            class_weights=model_class_weights,
            label_smoothing=args.label_smoothing,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            activation=fold_config.activation,
            use_time_weighting=args.use_time_weighting,
            in_channels=fold_config.in_channels,
            past_returns_lags=past_returns_lags,
            d_model=fold_config.d_model,
            nhead=fold_config.nhead,
            num_layers=fold_config.num_layers,
            num_horizons=num_horizons,
            horizon_weights=horizon_weights,
            use_horizon_embedding=args.use_horizon_embedding,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            dropout=fold_config.dropout,
            multi_task=fold_config.multi_task,
            scheduler=args.scheduler,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            pct_start=args.pct_start,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            step_size=args.step_size,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            clip_mode=args.clip_mode,
            clip_val=args.clip_val,
            use_curvature_reg=args.use_curvature_reg,
            curvature_lambda=args.curvature_lambda,
            input_noise_std=args.input_noise_std,
            scaler_type=args.scaler_type,
            winsor_limits=list(winsor_limits) if winsor_limits else None,
            metric_contract=args.metric_contract,
            metric_log_prefix=args.metric_log_prefix,
            metric_directional_base=args.metric_directional_base,
            report_fee_bps=args.report_fee_bps,
            report_slippage_bps=args.report_slippage_bps,
            report_half_spread_bps=args.report_half_spread_bps,
        )

        fold_checkpoint_dir = checkpoint_dir / f"fold_{fold_idx + 1}"
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        fold_checkpoint_callback = ModelCheckpoint(
            dirpath=fold_checkpoint_dir,
            filename="lit-{epoch:02d}-{val_mcc_primary:.4f}",
            save_top_k=1,
            monitor="val_mcc_primary",
            mode="max",
        )

        fold_callbacks = [
            EarlyStopping(monitor="val_mcc_primary", patience=15, mode="max"),
            fold_checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ]

        fold_logger = TensorBoardLogger("tb_logs", name=f"lit_{args.symbol}_fold{fold_idx + 1}")
        trainer_precision = resolve_trainer_precision(args)

        fold_trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=fold_callbacks,
            logger=fold_logger,
            accelerator="auto",
            devices=1,
            precision=trainer_precision,
            enable_progress_bar=False,
            log_every_n_steps=100,
            accumulate_grad_batches=args.accumulate_grad_batches,
            num_sanity_val_steps=args.num_sanity_val_steps,
            limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else 1.0,
            limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
        )
        fold_trainer.symbol = args.symbol

        print(f"\nTraining fold {fold_idx + 1}...")
        fold_model.hparams.val_batch_log_interval = args.val_batch_log_interval
        fold_model.hparams.enable_epoch_end_plots = args.enable_epoch_end_plots
        fold_model.hparams.skip_epoch0_artifacts = args.skip_epoch0_artifacts
        fold_model.hparams.enable_tb_embeddings = args.enable_tb_embeddings
        fold_trainer.fit(fold_model, fold_train_loader, fold_val_loader)

        best_fold_model_path = fold_checkpoint_callback.best_model_path
        if best_fold_model_path:
            print(f"\nEvaluating fold {fold_idx + 1}...")
            best_fold_model = LiTModule.load_from_checkpoint(best_fold_model_path)
            best_fold_model.eval()
            best_fold_model.freeze()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            best_fold_model.to(device)

            fold_y_true, fold_y_pred = [], []
            with torch.no_grad():
                for batch in fold_val_loader:
                    x, y, ts, mid, label, extra_data = batch
                    regime_id = extra_data["regime_id"]
                    x = x.to(device)
                    r_id = regime_id.to(device) if regime_id is not None else None
                    logits, _ = best_fold_model(x, regime_id=r_id)
                    preds = torch.argmax(logits, dim=1)
                    fold_y_true.extend(y.cpu().numpy())
                    fold_y_pred.extend(preds.cpu().numpy())

            fold_mcc = matthews_corrcoef(fold_y_true, fold_y_pred)
            fold_mccs.append(fold_mcc)
            fold_results.append({
                'fold': fold_idx + 1,
                'mcc': fold_mcc,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'best_model_path': best_fold_model_path,
            })
            print(f"\nFold {fold_idx + 1} Results: MCC={fold_mcc:.4f}")

    # Агрегируем результаты
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")

    mean_mcc = np.mean(fold_mccs)
    std_mcc = np.std(fold_mccs)

    for result in fold_results:
        print(f"  Fold {result['fold']}: {result['mcc']:.4f}")

    print(f"\nAggregated: Mean MCC={mean_mcc:.4f} ± {std_mcc:.4f}, "
          f"Min={np.min(fold_mccs):.4f}, Max={np.max(fold_mccs):.4f}")

    cv_results = {
        'n_splits': args.n_splits,
        'purge_buffer_events': args.purge_buffer_events,
        'embargo_buffer_events': args.embargo_buffer_events,
        'mean_mcc': float(mean_mcc),
        'std_mcc': float(std_mcc),
        'min_mcc': float(np.min(fold_mccs)),
        'max_mcc': float(np.max(fold_mccs)),
        'folds': [
            {
                'fold': r['fold'],
                'mcc': float(r['mcc']),
                'train_size': int(r['train_size']),
                'val_size': int(r['val_size']),
            }
            for r in fold_results
        ],
    }

    cv_results_path = base_path / "bots" / args.symbol / "models" / "cv_results.json"
    cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nCV results saved to: {cv_results_path}")

    # Финальная оценка на holdout (лучший фолд)
    best_fold_idx = np.argmax(fold_mccs)
    best_fold_result = fold_results[best_fold_idx]

    print(f"\n{'='*70}")
    print(f"HOLDOUT TEST EVALUATION (using best fold: {best_fold_result['fold']})")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = LiTModule.load_from_checkpoint(best_fold_result['best_model_path'])
    best_model.eval()
    best_model.freeze()
    best_model.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y, ts, mid, label, extra_data = batch
            regime_id = extra_data["regime_id"]
            x = x.to(device)
            r_id = regime_id.to(device) if regime_id is not None else None
            logits, _ = best_model(x, regime_id=r_id)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    holdout_mcc = matthews_corrcoef(y_true, y_pred)
    mcc_diff = abs(mean_mcc - holdout_mcc)
    mcc_diff_pct = (mcc_diff / mean_mcc) * 100 if mean_mcc != 0 else 0

    print(f"\nHoldout Test Results: MCC={holdout_mcc:.4f}")
    print(f"CV vs Holdout: CV Mean={mean_mcc:.4f}, Holdout={holdout_mcc:.4f}, "
          f"Diff={mcc_diff:.4f} ({mcc_diff_pct:.2f}%)")

    if mcc_diff_pct > 15:
        print(f"\n⚠️  WARNING: MCC difference > 15% detected! Possible data drift.")

    print(f"\n{'='*70}")
    print("Cross-validation completed successfully!")
    print(f"{'='*70}\n")
