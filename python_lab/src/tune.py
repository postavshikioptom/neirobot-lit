import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse

from .train import LiTModule
from .dataset import LOBPyTorchDataset, LOBDataset, LOBDataLoader
from .features import FeatureEngineer
from .labels import Labeler
from .normalization import Normalizer
from .utils import CalibrationMetrics, print_pruning_stats

from pytorch_lightning.callbacks import EarlyStopping

"""
Optuna Hyperparameter Tuning with Pruning (Задача 156)

Этот модуль реализует автоматический подбор гиперпараметров с использованием Optuna
и поддержкой различных стратегий pruning для эффективного отсечения бесперспективных
конфигураций на ранних этапах обучения.

Основные возможности:
1. Три типа pruner'ов:
   - median: базовый вариант (MedianPruner)
   - hyperband: бюджетирование ресурсов (HyperbandPruner)
   - patience: устойчивость к шуму (PatientPruner) - рекомендуется для LOB данных

2. Оптимизация по MCC (Matthews Correlation Coefficient) как основной метрике качества

3. Поддержка режимов:
   - train: обычное обучение
   - distill: knowledge distillation
   - cv: purged k-fold cross-validation

4. Автоматическая статистика pruning:
   - Количество pruned trials
   - Средняя эпоха отсечения
   - Оценка эффективности pruning

Примеры использования:

# Базовый вариант с MedianPruner
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \\
    --pruner_type median --n_startup_trials 20 --n_warmup_steps 25

# HyperbandPruner для бюджетирования ресурсов
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \\
    --pruner_type hyperband --min_resource 1 --max_resource 20

# PatientPruner для шумных метрик (рекомендуется для LOB)
python -m python_lab.src.tune --symbol BTCUSDT --trials 50 \\
    --pruner_type patience --n_startup_trials 20 --n_warmup_steps 25 --patience 3

# Distillation режим с pruning
python -m python_lab.src.tune --symbol BTCUSDT --mode distill \\
    --teacher_path bots/BTCUSDT/models/teacher_lit.pt \\
    --pruner_type patience --trials 30

Параметры pruning:
- --pruner_type: тип pruner (median, hyperband, patience)
- --n_startup_trials: количество полных триалов перед началом pruning (мин 20)
- --n_warmup_steps: количество эпох до первой проверки на pruning (мин 25)
- --patience: терпение для PatientPruner (шаги без улучшения)
- --min_resource, --max_resource: ресурсы для HyperbandPruner

Рекомендации для LOB данных:
- Используйте --pruner_type patience для обработки шумных метрик
- Установите --n_startup_trials >= 20 для стабилизации
- Установите --n_warmup_steps >= 25 для трансформеров
- Используйте --patience 3-5 для баланса между скоростью и качеством
"""

# Глобальная переменная для данных
raw_df = None

def prepare_data(args):
    """Загрузка и базовая подготовка данных один раз."""
    global raw_df
    
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "bots" / args.symbol / "data" / "raw"

    print(f"Pre-loading raw data for tuning {args.symbol}...")
    loader = LOBDataLoader(str(data_path), args.symbol)
    df = loader.load_data()

    # Сохраняем сырые данные (без feature engineering)
    # Feature engineering будет выполняться в каждом trial с разными лагами
    raw_df = df

def objective(trial):
    global raw_df
    
    # Получаем аргументы из глобальной переменной (будет установлена в run_tuning)
    args = trial.study.user_attrs.get('args')
    
    # Режим distillation или обычное обучение
    is_distillation = args.mode == "distill" if hasattr(args, 'mode') else False
    
    # Гиперпараметры для подбора
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    seq_len = trial.suggest_int("seq_len", 10, 100, step=10)
    
    # Валидация архитектуры (d_model должен делиться на nhead)
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()
        
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    
    # Выбор функции активации
    activation = trial.suggest_categorical("activation", ["relu", "gelu_exact", "gelu_tanh", "silu"])
    
    # Выбор LR Scheduler
    scheduler = trial.suggest_categorical("scheduler", ["onecycle", "plateau", "cosine", "step", "none"])
    
    # Параметры для OneCycle и Cosine
    div_factor = trial.suggest_float("div_factor", 10.0, 40.0) if scheduler in ["onecycle", "cosine"] else 25.0
    
    # Параметры для OneCycle
    pct_start = trial.suggest_float("pct_start", 0.2, 0.4) if scheduler == "onecycle" else 0.3
    final_div_factor = trial.suggest_float("final_div_factor", 1000.0, 20000.0, log=True) if scheduler == "onecycle" else 10000.0
    
    # Параметры для Plateau
    plateau_factor = trial.suggest_float("plateau_factor", 0.3, 0.7) if scheduler == "plateau" else 0.5
    plateau_patience = trial.suggest_int("plateau_patience", 3, 7) if scheduler == "plateau" else 5
    
    # Параметры для StepLR
    step_size = trial.suggest_int("step_size", 5, 15) if scheduler == "step" else 10
    gamma = trial.suggest_float("gamma", 0.3, 0.7) if scheduler == "step" else 0.5
    
    # Weight decay для AdamW
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Параметры Adaptive Gradient Clipping (Задача 154)
    clip_mode = trial.suggest_categorical("clip_mode", ["none", "norm", "agc"])
    
    # clip_val зависит от режима: для norm обычно 1.0, для agc 0.01-0.1
    if clip_mode == "agc":
        clip_val = trial.suggest_float("clip_val", 0.01, 0.1)
    elif clip_mode == "norm":
        clip_val = trial.suggest_float("clip_val", 0.5, 2.0)
    else:  # none
        clip_val = 0.01  # Значение по умолчанию (не используется)
    
    # Выбор набора лагов для Past Returns
    lags_set = trial.suggest_categorical("past_returns_lags", [
        "[10, 50, 100]",
        "[5, 20, 50]",
        "[20, 100, 200]",
        "[10, 30, 60, 120]"
    ])
    past_returns_lags = eval(lags_set)  # Преобразуем строку в список
    
    # 0. Выбор функции потерь
    loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])
    focal_gamma = 2.0
    if loss_type == "focal":
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
    
    # Параметры аугментации (Задача 124)
    augment_prob = trial.suggest_float("augment_prob", 0.0, 0.8)
    use_symmetric_flip = trial.suggest_categorical("use_symmetric_flip", [True, False])
    volume_jitter_range = trial.suggest_float("volume_jitter_range", 0.0, 0.3)
    aug_seed = 42  # Фиксированный seed для воспроизводимости
    
    # Параметры Knowledge Distillation (Задача 151)
    teacher_model = None
    alpha = 0.9
    temperature = 3.0
    
    if is_distillation:
        # Подбираем оптимальные параметры distillation
        alpha = trial.suggest_float("alpha", 0.7, 0.95)
        temperature = trial.suggest_float("temperature", 2.0, 5.0)
        
        # Загружаем teacher модель один раз
        if not hasattr(trial.study, '_teacher_model'):
            print(f"Loading teacher model from: {args.teacher_path}")
            teacher_module = LiTModule.load_from_checkpoint(args.teacher_path)
            trial.study._teacher_model = teacher_module.model
            trial.study._teacher_model.eval()
            trial.study._teacher_model.requires_grad_(False)
        
        teacher_model = trial.study._teacher_model

    # 1. Feature Engineering
    fe = FeatureEngineer(n_levels=50)
    df_feat = fe.transform(raw_df)
    
    # 2. Добавление меток (поддержка multi-horizon)
    # Определяем horizons из args или используем фиксированное значение
    if hasattr(args, 'horizons') and args.horizons:
        horizons = [int(h.strip()) for h in args.horizons.split(',')]
        num_horizons = len(horizons)
        
        # Подбираем оптимальные веса для горизонтов (Задача 160)
        horizon_weights = []
        for h_idx in range(num_horizons):
            weight = trial.suggest_float(f"horizon_weight_{h_idx}", 0.1, 1.0)
            horizon_weights.append(weight)
        
        # Нормализуем веса (сумма = 1.0)
        total_weight = sum(horizon_weights)
        horizon_weights = [w / total_weight for w in horizon_weights]
        
        print(f"Trial {trial.number}: Testing horizon weights: {horizon_weights}")
    else:
        horizons = 100  # Single horizon (обратная совместимость)
        num_horizons = 1
        horizon_weights = None
    
    labeler = Labeler(horizon=horizons, threshold=0.0005)
    df_feat = labeler.add_labels(df_feat)
    
    # 3. Нормализация (fit на данных)
    # В тюнинге делаем упрощенно. Normalizer требует путь, но мы его не будем сохранять
    normalizer = Normalizer("/tmp/tune_norm.json")
    
    # Задача 311: Обучаем на каналах
    temp_ds = LOBDataset(df_feat, seq_len=1, data_mode="memory", is_train=False)
    channels_df = temp_ds._compute_channels_for_normalization(list(range(len(df_feat))))
    normalizer.fit(channels_df)

    # 4. Создание Dataset с новым seq_len и количеством лагов
    n_past_returns = len(past_returns_lags)
    full_dataset = LOBPyTorchDataset(
        df_feat,  # ПЕРЕДАЕМ RAW ДАННЫЕ (Задача 311)
        seq_len=seq_len, 
        n_past_returns=n_past_returns,
        past_returns_lags=past_returns_lags, # ПЕРЕДАЕМ ЛАГИ (Задача 311)
        is_train=False,  # Будет переопределено для train_ds
        augment_prob=augment_prob,
        use_symmetric_flip=use_symmetric_flip,
        volume_jitter_range=volume_jitter_range,
        aug_seed=aug_seed,
        normalizer=normalizer  # ПЕРЕДАЕМ NORMALIZER (Задача 311)
    )
    
    # Проверяем режим CV
    is_cv_mode = args.mode == "cv" if hasattr(args, 'mode') else False
    
    if is_cv_mode:
        # ============================================================================
        # Purged K-Fold Cross-Validation Mode для Optuna (Задача 153)
        # ============================================================================
        from .utils import PurgedKFold
        from torch.utils.data import Subset
        
        # Инициализируем PurgedKFold
        cv = PurgedKFold(
            n_splits=args.n_splits,
            purge_buffer_events=args.purge_buffer_events,
            embargo_buffer_events=args.embargo_buffer_events
        )
        
        # Получаем timestamps для проверки сортировки
        timestamps = full_dataset.get_timestamps()
        
        # Используем весь датасет для CV (без holdout в tuning)
        cv_indices = np.arange(len(full_dataset))
        cv_timestamps = timestamps
        cv_labels = full_dataset.labels
        
        # Список для сбора MCC по фолдам
        fold_mccs = []
        
        # Цикл по фолдам
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(cv_indices, cv_labels, cv_timestamps)):
            # Создаем Subset для train и val
            fold_train_ds = Subset(full_dataset, train_idx)
            fold_val_ds = Subset(full_dataset, val_idx)
            
            # Расчет весов для текущего фолда
            fold_train_labels = full_dataset.labels[train_idx]
            classes, counts = np.unique(fold_train_labels, return_counts=True)
            full_counts = np.zeros(3, dtype=np.int64)
            for cls, count in zip(classes, counts):
                if 0 <= cls < 3: full_counts[int(cls)] = count
            
            total_samples = np.sum(full_counts)
            fold_weights = total_samples / (3 * (full_counts + 1.0))
            fold_weights = fold_weights / np.mean(fold_weights)
            
            # Создаем DataLoaders для фолда
            fold_train_loader = DataLoader(
                fold_train_ds,
                batch_size=128,
                shuffle=True,
                num_workers=0
            )
            
            fold_val_loader = DataLoader(
                fold_val_ds,
                batch_size=128,
                shuffle=False,
                num_workers=0
            )
            
            # Инициализация модели с fresh weights для каждого фолда
            fold_model = LiTModule(
                seq_len=seq_len,
                lr=lr,
                class_weights=None if is_distillation else fold_weights,
                label_smoothing=0.0 if is_distillation else label_smoothing,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                activation=activation,
                teacher_model=teacher_model,
                alpha=alpha,
                temperature=temperature,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                past_returns_lags=past_returns_lags,
                scheduler=scheduler,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                pct_start=pct_start,
                plateau_factor=plateau_factor,
                plateau_patience=plateau_patience,
                step_size=step_size,
                gamma=gamma,
                weight_decay=weight_decay,
                # Параметры gradient clipping (Задача 154)
                clip_mode=clip_mode,
                clip_val=clip_val,
                # Параметры multi-horizon (Задача 160)
                num_horizons=num_horizons,
                horizon_weights=horizon_weights,
                use_horizon_embedding=args.use_horizon_embedding if hasattr(args, 'use_horizon_embedding') else False
            )
            
            # Настройка Trainer для фолда с pruning
            fold_callbacks = [
                PyTorchLightningPruningCallback(trial, monitor="val_mcc"),
                EarlyStopping(monitor="val_loss", patience=3, mode="min")  # Меньше patience для ускорения
            ]
            
            fold_trainer = pl.Trainer(
                max_epochs=15,  # Меньше эпох для ускорения tuning
                min_epochs=2,
                accelerator="auto",
                devices=1,
                enable_checkpointing=False,
                logger=False,
                callbacks=fold_callbacks,
                enable_progress_bar=False
            )
            
            # Обучение фолда
            fold_trainer.fit(fold_model, fold_train_loader, fold_val_loader)
            
            # Получаем MCC для фолда
            fold_mcc = fold_trainer.callback_metrics.get("val_mcc", torch.tensor(0.0)).item()
            fold_mccs.append(fold_mcc)
            
            # Промежуточный отчет для pruning
            # Если средний MCC по первым фолдам значительно хуже, Optuna может прервать триал
            trial.report(np.mean(fold_mccs), fold_idx)
            
            # Проверка на pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Возвращаем средний MCC по всем фолдам
        mean_mcc = np.mean(fold_mccs)
        std_mcc = np.std(fold_mccs)
        
        # Логируем метрики
        trial.set_user_attr("mean_mcc", mean_mcc)
        trial.set_user_attr("std_mcc", std_mcc)
        trial.set_user_attr("min_mcc", np.min(fold_mccs))
        trial.set_user_attr("max_mcc", np.max(fold_mccs))
        
        return mean_mcc
    
    else:
        # Обычный режим (train или distill) - без CV
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_ds, val_ds = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Включаем аугментацию для тренировочного набора
        # Создаем обертку для train_ds
        class TrainWrapper:
            def __init__(self, subset, dataset):
                self.subset = subset
                self.dataset = dataset
                self.dataset.is_train = True
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                return self.subset[idx]
        
        train_ds = TrainWrapper(train_ds, full_dataset)

        # Расчет весов для текущего разбиения
        train_indices = train_ds.subset.indices
        train_labels = full_dataset.labels[train_indices]
        classes, counts = np.unique(train_labels, return_counts=True)
        full_counts = np.zeros(3, dtype=np.int64)
        for cls, count in zip(classes, counts):
            if 0 <= cls < 3: full_counts[int(cls)] = count
        
        total_samples = np.sum(full_counts)
        weights = total_samples / (3 * (full_counts + 1.0))
        weights = weights / np.mean(weights)

        train_loader = DataLoader(
            train_ds, 
            batch_size=128, 
            shuffle=True, 
            num_workers=0 
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=128, 
            shuffle=False, 
            num_workers=0
        )

        # 5. Инициализация модели
        model = LiTModule(
            seq_len=seq_len,
            lr=lr,
            class_weights=None if is_distillation else weights,  # Не используем веса для distillation
            label_smoothing=0.0 if is_distillation else label_smoothing,  # Не используем smoothing для distillation
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            activation=activation,  # Передаем тип активации
            teacher_model=teacher_model,  # Передаем teacher для distillation
            alpha=alpha,
            temperature=temperature,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            past_returns_lags=past_returns_lags,
            # Параметры scheduler
            scheduler=scheduler,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            pct_start=pct_start,
            plateau_factor=plateau_factor,
            plateau_patience=plateau_patience,
            step_size=step_size,
            gamma=gamma,
            weight_decay=weight_decay,
            # Параметры gradient clipping (Задача 154)
            clip_mode=clip_mode,
            clip_val=clip_val,
            # Параметры multi-horizon (Задача 160)
            num_horizons=num_horizons,
            horizon_weights=horizon_weights,
            use_horizon_embedding=args.use_horizon_embedding if hasattr(args, 'use_horizon_embedding') else False
        )

        # 6. Настройка Trainer с прунингом и EarlyStopping
        callbacks = [
            PyTorchLightningPruningCallback(trial, monitor="val_mcc"),
            EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ]
        
        trainer = pl.Trainer(
            max_epochs=20,
            min_epochs=3,
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            callbacks=callbacks
        )
        
        # Обучение
        trainer.fit(model, train_loader, val_loader)
        
        # Вычисляем ECE на валидационном наборе для дополнительных метрик
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        val_logits = []
        val_labels = []
        
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                logits, _ = model(x.to(device))
                val_logits.append(logits.cpu())
                val_labels.append(y)
        
        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        # Вычисляем ECE
        calibration_metrics = CalibrationMetrics(n_bins=15)
        ece, mce, _ = calibration_metrics.calculate(val_logits, val_labels)
        
        # Получаем метрики из trainer
        val_loss = trainer.callback_metrics["val_loss"].item()
        val_mcc = trainer.callback_metrics["val_mcc"].item()
        
        # Целевая функция: максимизируем MCC (Задача 156, пункт 2)
        # Optuna настроен на максимизацию, поэтому возвращаем MCC напрямую
        
        # Логируем дополнительные метрики для анализа
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("ece", ece)
        trial.set_user_attr("mce", mce)
        trial.set_user_attr("val_mcc", val_mcc)
        
        return val_mcc

def run_tuning():
    parser = argparse.ArgumentParser(description="Tune LiT model with Optuna")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256) # Больше батч для ускорения тюнинга
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.0005)
    parser.add_argument("--trials", type=int, default=50)
    
    # Параметры Knowledge Distillation (Задача 151)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "distill", "cv"], help="Tuning mode: train, distill, or cv (purged k-fold)")
    parser.add_argument("--teacher_path", type=str, default=None, help="Path to teacher model (required for distill mode)")
    
    # Параметры Purged K-Fold Cross-Validation (Задача 153)
    parser.add_argument("--n_splits", type=int, default=3, help="Number of folds for CV tuning (reduced from 5 for speed)")
    parser.add_argument("--purge_buffer_events", type=int, default=100, help="Number of events to purge before validation fold")
    parser.add_argument("--embargo_buffer_events", type=int, default=50, help="Number of events to embargo after validation fold")
    
    # Параметры Pruning (Задача 156)
    parser.add_argument("--pruner_type", type=str, default="median", choices=["median", "hyperband", "patience"], 
                        help="Pruner type: median (baseline), hyperband (resource budgeting), patience (noise tolerance)")
    parser.add_argument("--min_resource", type=int, default=1, help="Minimum resource (epochs) for HyperbandPruner")
    parser.add_argument("--max_resource", type=int, default=20, help="Maximum resource (epochs) for HyperbandPruner")
    parser.add_argument("--n_startup_trials", type=int, default=20, help="Number of startup trials before pruning starts (min 20 for LOB data)")
    parser.add_argument("--n_warmup_steps", type=int, default=25, help="Number of warmup steps (epochs) before first pruning check (min 25 for transformers)")
    parser.add_argument("--patience", type=int, default=3, help="Patience for PatientPruner (steps without improvement)")
    
    args = parser.parse_args()
    
    # Валидация параметров distillation
    if args.mode == "distill" and args.teacher_path is None:
        raise ValueError("--teacher_path is required for distillation mode")
    
    # Валидация параметров pruning (Задача 156)
    if args.n_startup_trials < 20:
        print(f"⚠️  WARNING: n_startup_trials={args.n_startup_trials} is below recommended minimum of 20 for LOB data")
    if args.n_warmup_steps < 25:
        print(f"⚠️  WARNING: n_warmup_steps={args.n_warmup_steps} is below recommended minimum of 25 for transformers")

    # Загружаем данные один раз
    prepare_data(args)

    # Настройка Optuna с TPESampler (по умолчанию для всех режимов)
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # Создание pruner на основе выбранного типа (Задача 156)
    print(f"\n{'='*60}")
    print(f"Pruning Configuration:")
    print(f"  - Pruner type: {args.pruner_type}")
    print(f"  - Startup trials: {args.n_startup_trials}")
    print(f"  - Warmup steps: {args.n_warmup_steps}")
    
    if args.pruner_type == "median":
        # MedianPruner - базовый вариант
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=1
        )
        print(f"  - Strategy: Median stopping rule")
        
    elif args.pruner_type == "hyperband":
        # HyperbandPruner - бюджетирование ресурсов
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=args.min_resource,
            max_resource=args.max_resource,
            reduction_factor=3
        )
        print(f"  - Strategy: Hyperband (resource budgeting)")
        print(f"  - Min resource: {args.min_resource} epochs")
        print(f"  - Max resource: {args.max_resource} epochs")
        
    elif args.pruner_type == "patience":
        # PatientPruner - устойчивость к шуму (критично для LOB данных)
        base_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=1
        )
        pruner = optuna.pruners.PatientPruner(
            wrapped_pruner=base_pruner,
            patience=args.patience
        )
        print(f"  - Strategy: Patient (noise tolerance)")
        print(f"  - Patience: {args.patience} steps without improvement")
        print(f"  - Wrapped: MedianPruner")
    
    print(f"{'='*60}\n")
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"lit_hpo_{args.symbol}_{args.mode}",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )
    
    # Сохраняем args в study для доступа из objective
    study.set_user_attr('args', args)
    
    mode_str = "CV" if args.mode == "cv" else ("Distillation" if args.mode == "distill" else "Normal")
    print(f"Starting {mode_str} HPO for {args.symbol} (Trials: {args.trials})...")
    if args.mode == "distill":
        print(f"Teacher model: {args.teacher_path}")
    elif args.mode == "cv":
        print(f"CV folds: {args.n_splits}, Purge: {args.purge_buffer_events}, Embargo: {args.embargo_buffer_events}")
    
    study.optimize(objective, n_trials=args.trials, timeout=3600*4)
    
    # Статистика pruning (Задача 156, пункт 5)
    print_pruning_stats(study, n_warmup_steps=args.n_warmup_steps)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    if args.mode == "cv":
        # Вывод для CV режима
        print(f"Best Mean MCC: {study.best_value:.4f}")
        print(f"Best trial metrics:")
        print(f"  - mean_mcc: {study.best_trial.user_attrs.get('mean_mcc', 'N/A'):.4f}")
        print(f"  - std_mcc: {study.best_trial.user_attrs.get('std_mcc', 'N/A'):.4f}")
        print(f"  - min_mcc: {study.best_trial.user_attrs.get('min_mcc', 'N/A'):.4f}")
        print(f"  - max_mcc: {study.best_trial.user_attrs.get('max_mcc', 'N/A'):.4f}")
    else:
        # Вывод для обычного режима (Задача 156: MCC как основная метрика)
        print(f"Best MCC: {study.best_value:.4f}")
        print(f"Best trial metrics:")
        print(f"  - val_mcc: {study.best_trial.user_attrs.get('val_mcc', 'N/A'):.4f}")
        print(f"  - val_loss: {study.best_trial.user_attrs.get('val_loss', 'N/A'):.4f}")
        print(f"  - ECE: {study.best_trial.user_attrs.get('ece', 'N/A'):.4f}")
        print(f"  - MCE: {study.best_trial.user_attrs.get('mce', 'N/A'):.4f}")
    
    print(f"Best params: {study.best_params}")
    
    if args.mode == "distill":
        print(f"\nDistillation params:")
        print(f"  - alpha: {study.best_params.get('alpha', 'N/A'):.3f}")
        print(f"  - temperature: {study.best_params.get('temperature', 'N/A'):.2f}")
    
    print("="*60)

if __name__ == "__main__":
    run_tuning()
