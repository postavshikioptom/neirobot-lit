"""
train_optuna.py — Optuna seq_len search для train.py.
Вынесено из train.py в рамках задачи 322.7.
"""
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from optuna.pruners import MedianPruner, HyperbandPruner, PatientPruner
from optuna.exceptions import TrialPruned
from sklearn.metrics import matthews_corrcoef
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .dataset import LOBDataset
from .train_module import LiTModule, TrainSubset
from .train_runtime import build_dataloader_kwargs, resolve_trainer_precision
from .utils import CalibrationMetrics


def objective_seq_len_search(
    trial, args, base_path, df,
    in_channels, past_returns_lags, num_horizons, horizon_weights,
    weights, normalizer, regime_detector, regime_weights, num_regimes,
):
    """
    Optuna objective для поиска оптимальной seq_len (Задача 055).
    Использует те же factory-и что и основной pipeline.
    """
    seq_len = trial.suggest_int("seq_len", 10, 100, step=10)
    print(f"\n[Optuna Trial] Testing seq_len={seq_len}")

    scheduler = trial.suggest_categorical("scheduler", ["onecycle", "plateau", "cosine", "step", "none"])
    div_factor = trial.suggest_float("div_factor", 10.0, 40.0, log=True)
    final_div_factor = trial.suggest_float("final_div_factor", 1000.0, 10000.0, log=True)
    pct_start = trial.suggest_float("pct_start", 0.1, 0.5)
    plateau_factor = trial.suggest_float("plateau_factor", 0.1, 0.9)
    plateau_patience = trial.suggest_int("plateau_patience", 2, 10)
    step_size = trial.suggest_int("step_size", 5, 30)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)

    print(f"[Optuna Trial] Scheduler: {scheduler}, div_factor={div_factor:.2f}, "
          f"final_div_factor={final_div_factor:.0f}, pct_start={pct_start:.2f}")

    try:
        if args.use_time_weighting:
            time_weighting_params = {
                'half_life_hours': args.half_life_hours,
                'min_weight': args.min_sample_weight,
                'class_weights': weights,
            }
        else:
            time_weighting_params = {
                'half_life_hours': 24.0,
                'min_weight': 1.0,
                'class_weights': None,
            }

        trial_dataset = LOBDataset(
            df,
            seq_len=seq_len,
            n_past_returns=len(past_returns_lags),
            data_mode="memory",
            is_train=False,
            augment_prob=args.augment_prob,
            use_symmetric_flip=args.use_symmetric_flip,
            volume_jitter_range=args.volume_jitter_range,
            aug_seed=args.aug_seed,
            regime_detector=regime_detector,
            regime_window=1000,
            scaler_type=args.scaler_type,
            winsor_limits=tuple([float(x.strip()) for x in args.winsor_limits.split(",")]),
            normalizer=normalizer,
            **time_weighting_params,
        )

        total_len = len(trial_dataset)
        train_size = int(0.8 * total_len)

        trial_train_ds = TrainSubset(trial_dataset, list(range(0, train_size)))
        trial_val_ds = Subset(trial_dataset, list(range(train_size, total_len)))

        trial_train_loader = DataLoader(trial_train_ds, **build_dataloader_kwargs(args, shuffle=True))
        trial_val_loader = DataLoader(trial_val_ds, **build_dataloader_kwargs(args, shuffle=False))

    except Exception as e:
        print(f"[Optuna Trial] Error creating dataset with seq_len={seq_len}: {e}")
        raise TrialPruned()

    from .lit_model import LiTConfig
    trial_config = LiTConfig(
        seq_len=seq_len,
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

    trial_model = LiTModule(
        seq_len=trial_config.seq_len,
        lr=1e-4,
        class_weights=weights,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        activation=trial_config.activation,
        use_time_weighting=args.use_time_weighting,
        use_regime_weighting=(regime_detector is not None),
        regime_weights=regime_weights,
        in_channels=trial_config.in_channels,
        past_returns_lags=past_returns_lags,
        d_model=trial_config.d_model,
        nhead=trial_config.nhead,
        num_layers=trial_config.num_layers,
        dropout=trial_config.dropout,
        multi_task=trial_config.multi_task,
        num_regimes=num_regimes,
        regime_embedding_dim=16,
        num_horizons=num_horizons,
        horizon_weights=horizon_weights,
        use_horizon_embedding=args.use_horizon_embedding,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        scheduler=scheduler,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        pct_start=pct_start,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        weight_decay=args.weight_decay,
        clip_mode=args.clip_mode,
        clip_val=args.clip_val,
        tb_hist_freq=args.tb_hist_freq,
        tb_embedding_samples=args.tb_embedding_samples,
        use_curvature_reg=args.use_curvature_reg,
        curvature_lambda=args.curvature_lambda,
        input_noise_std=args.input_noise_std,
        scaler_type=args.scaler_type,
        winsor_limits=list(tuple([float(x.strip()) for x in args.winsor_limits.split(",")])) if args.winsor_limits else None,
        enable_channel_attribution=args.enable_channel_attribution,
        channel_attribution_samples=args.channel_attribution_samples,
        channel_attribution_method=args.channel_attribution_method,
    )

    trial_checkpoint_callback = ModelCheckpoint(
        dirpath=base_path / "bots" / args.symbol / "models" / "optuna_checkpoints" / f"seq_len_{seq_len}",
        filename="lit-{epoch:02d}-{val_mcc_primary:.4f}",
        save_top_k=1,
        monitor="val_mcc_primary",
        mode="max",
    )

    trial_callbacks = [
        EarlyStopping(monitor="val_mcc_primary", patience=5, mode="max"),
        trial_checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trial_logger = TensorBoardLogger(f"runs/{args.symbol}/optuna", name=f"seq_len_{seq_len}")
    trainer_precision = resolve_trainer_precision(args)

    trial_trainer = pl.Trainer(
        max_epochs=min(20, args.epochs),
        callbacks=trial_callbacks,
        logger=trial_logger,
        accelerator="auto",
        devices=1,
        precision=trainer_precision,
        enable_progress_bar=False,
        log_every_n_steps=100,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
    )

    try:
        trial_model.hparams.val_batch_log_interval = args.val_batch_log_interval
        trial_model.hparams.enable_epoch_end_plots = args.enable_epoch_end_plots
        trial_model.hparams.skip_epoch0_artifacts = args.skip_epoch0_artifacts
        trial_model.hparams.enable_tb_embeddings = args.enable_tb_embeddings
        trial_model.hparams.enable_channel_attribution = args.enable_channel_attribution
        trial_model.hparams.channel_attribution_samples = args.channel_attribution_samples
        trial_model.hparams.channel_attribution_method = args.channel_attribution_method
        trial_trainer.fit(trial_model, trial_train_loader, trial_val_loader)
    except Exception as e:
        print(f"[Optuna Trial] Training failed for seq_len={seq_len}: {e}")
        raise TrialPruned()

    # Вычисляем MCC
    trial_model.eval()
    all_preds, all_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_model.to(device)

    with torch.no_grad():
        for batch in trial_val_loader:
            x, y, ts, mid, label, extra_data = batch
            device = next(trial_model.parameters()).device
            x = x.to(device)
            y = y.to(device)
            regime_id = extra_data["regime_id"].to(device) if extra_data["regime_id"] is not None else None
            output = trial_model(x, regime_id=regime_id)
            logits = output[0] if isinstance(output, tuple) else output
            if logits.dim() == 3:
                logits = logits[:, 0, :]
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(y.cpu())

    val_preds = torch.cat(all_preds).numpy()
    val_labels_tensor = torch.cat(all_labels)
    val_labels = val_labels_tensor[:, 0].numpy() if val_labels_tensor.dim() == 2 else val_labels_tensor.numpy()
    val_mcc = matthews_corrcoef(val_labels, val_preds)

    # Вычисляем ECE
    all_logits = []
    with torch.no_grad():
        for batch in trial_val_loader:
            x, y, ts, mid, label, extra_data = batch
            device = next(trial_model.parameters()).device
            x = x.to(device)
            regime_id = extra_data["regime_id"].to(device) if extra_data["regime_id"] is not None else None
            output = trial_model(x, regime_id=regime_id)
            logits = output[0] if isinstance(output, tuple) else output
            if logits.dim() == 3:
                logits = logits[:, 0, :]
            all_logits.append(logits.cpu())

    val_logits_tensor = torch.cat(all_logits)
    calibration_metrics = CalibrationMetrics(n_bins=15)
    ece, _, _ = calibration_metrics.calculate(val_logits_tensor, torch.from_numpy(val_labels).long())

    score = val_mcc - (ece * 0.5)
    print(f"[Optuna Trial] seq_len={seq_len}, val_mcc={val_mcc:.4f}, ece={ece:.4f}, score={score:.4f}")
    return score


def run_optuna_seq_len_search(args, paths, prepared, winsor_limits):
    """
    Запускает Optuna поиск seq_len и пересоздаёт датасеты с лучшим seq_len.
    Возвращает обновлённый PreparedTrainingData.
    """
    import optuna
    from .train_data import PreparedTrainingData, build_full_dataset, split_dataset_chronologically
    from .train_runtime import build_dataloader_kwargs

    base_path = paths.base_path

    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER SEARCH FOR seq_len")
    print("=" * 70)
    print(f"Number of trials: {args.optuna_n_trials}")
    print(f"Pruner type: {args.optuna_pruner}")
    print("=" * 70 + "\n")

    if args.optuna_pruner == "hyperband":
        pruner = HyperbandPruner(min_resource=1, max_resource=min(20, args.epochs), reduction_factor=3)
    elif args.optuna_pruner == "patient":
        pruner = PatientPruner(patience=3)
    else:
        pruner = MedianPruner()

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective_seq_len_search(
            trial, args, base_path, prepared.df,
            prepared.in_channels, prepared.past_returns_lags,
            prepared.num_horizons, prepared.horizon_weights,
            prepared.class_weights, prepared.normalizer,
            prepared.regime_detector, prepared.regime_weights, prepared.num_regimes,
        ),
        n_trials=args.optuna_n_trials,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_seq_len = best_trial.params["seq_len"]
    best_mcc = best_trial.value

    print(f"\n{'='*70}")
    print(f"OPTUNA SEARCH COMPLETED")
    print(f"Best seq_len: {best_seq_len}, Best MCC: {best_mcc:.4f}")
    print(f"{'='*70}\n")

    # Сохраняем конфиг
    config_path = base_path / "bots" / args.symbol / "models" / "optuna_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    optuna_config = {
        "best_seq_len": int(best_seq_len),
        "best_mcc": float(best_mcc),
        "n_trials": args.optuna_n_trials,
        "pruner_type": args.optuna_pruner,
        "all_trials": [
            {
                "seq_len": int(t.params["seq_len"]),
                "mcc": float(t.value) if t.value is not None else None,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(config_path, 'w') as f:
        json.dump(optuna_config, f, indent=2)
    print(f"Optuna config saved to: {config_path}")

    # Обновляем seq_len и пересоздаём датасеты
    print(f"\nUpdating seq_len from {args.seq_len} to {best_seq_len} for training")
    args.seq_len = best_seq_len
    print("Recreating datasets with best seq_len...")

    new_full_dataset = build_full_dataset(
        prepared.df, args, prepared.past_returns_lags, winsor_limits,
        prepared.normalizer, prepared.regime_detector, prepared.time_weighting_params,
    )

    new_train_ds, new_val_ds, new_test_ds, _, _, _ = split_dataset_chronologically(new_full_dataset)

    new_train_loader = DataLoader(new_train_ds, **build_dataloader_kwargs(args, shuffle=True))
    new_val_loader = DataLoader(new_val_ds, **build_dataloader_kwargs(args, shuffle=False))
    new_test_loader = DataLoader(new_test_ds, **build_dataloader_kwargs(args, shuffle=False))

    print("Datasets recreated successfully\n")

    # Возвращаем обновлённый PreparedTrainingData
    from dataclasses import replace
    return replace(
        prepared,
        full_dataset=new_full_dataset,
        train_ds=new_train_ds,
        val_ds=new_val_ds,
        test_ds=new_test_ds,
        train_loader=new_train_loader,
        val_loader=new_val_loader,
        test_loader=new_test_loader,
    )
