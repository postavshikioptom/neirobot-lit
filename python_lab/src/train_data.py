"""
train_data.py — Data orchestration для train.py.
Вынесено из train.py в рамках задачи 322.5.

Порядок инициализации (нельзя менять):
  LOBDataLoader -> FeatureEngineer -> Labeler -> LOBDataset
  -> chronological split -> normalizer.fit(train only) -> DataLoaders -> class weights
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from .dataset import LOBDataset, LOBDataLoader
from .features import FeatureEngineer
from .labels import Labeler
from .normalization import Normalizer, symlog_transform
from .train_module import TrainSubset
from .train_runtime import build_dataloader_kwargs


@dataclass
class PreparedTrainingData:
    df: object
    normalizer: object
    full_dataset: object
    train_ds: object
    val_ds: object
    test_ds: object
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights: np.ndarray
    past_returns_lags: list
    in_channels: int
    num_horizons: int
    horizon_weights: object  # list[float] | None
    regime_detector: object
    regime_weights: object
    num_regimes: int
    # Вспомогательные поля для пересоздания датасетов (Optuna)
    n_past_returns: int = 0
    time_weighting_params: dict = field(default_factory=dict)


def _parse_past_returns_lags(raw: str) -> list:
    """Парсит строку лагов past returns в список int."""
    return [int(x.strip()) for x in raw.split(',')]


def build_full_dataset(df, args, past_returns_lags, winsor_limits, normalizer,
                        regime_detector, time_weighting_params):
    """Создаёт LOBDataset в memory режиме."""
    n_past_returns = len(past_returns_lags)
    return LOBDataset(
        df,
        seq_len=args.seq_len,
        n_past_returns=n_past_returns,
        past_returns_lags=past_returns_lags,
        data_mode="memory",
        is_train=False,  # val/test работают в eval mode; train через TrainSubset
        augment_prob=args.augment_prob,
        use_symmetric_flip=args.use_symmetric_flip,
        volume_jitter_range=args.volume_jitter_range,
        aug_seed=args.aug_seed,
        regime_detector=regime_detector,
        regime_window=1000,
        scaler_type=args.scaler_type,
        winsor_limits=winsor_limits,
        scale_multiplier=args.scale_multiplier,
        normalizer=normalizer,
        **time_weighting_params
    )


def split_dataset_chronologically(full_dataset):
    """Хронологическое разделение 70/15/15."""
    total_len = len(full_dataset)
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_len))

    train_ds = TrainSubset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)
    return train_ds, val_ds, test_ds, train_indices, val_indices, test_indices


def _fit_normalizer_on_train(full_dataset, train_ds, normalizer, args, winsor_limits):
    """Обучает нормализатор только на train-части (channel-space)."""
    print("\nFitting normalizer on original training set (channels-based)...")
    train_indices_for_fit = train_ds.indices
    train_channels_df = full_dataset._compute_channels_for_normalization(train_indices_for_fit)
    print(f"Static features dimension check: {train_channels_df.shape[1]} features (8 static channels × 50 levels, dynamic channels fitted separately)")

    # Задача 324.2: Используем полные кэши train-части, а не три суррогатных столбца
    # Берём все значения ofi_cache, delta_imb_cache, delta_spread_cache для train-индексов
    train_idx_arr = np.array(train_indices_for_fit)

    ofi_raw = full_dataset.ofi_cache[train_idx_arr]
    delta_imb_raw = full_dataset.delta_imb_cache[train_idx_arr]
    delta_spread_raw = full_dataset.delta_spread_cache[train_idx_arr]

    # Задача 324.3: Применяем symlog через ту же функцию что и в runtime (_apply_dynamic_transform).
    # Путь ИДЕНТИЧЕН: symlog_transform → transform_dynamic (median/iqr) → clamp[-4,4].
    # symlog_transform импортирован из normalization.py — единый источник.
    ofi_sym = symlog_transform(ofi_raw)
    delta_imb_sym = symlog_transform(delta_imb_raw)
    delta_spread_sym = symlog_transform(delta_spread_raw)

    dynamic_data = {
        "ofi": ofi_sym,
        "delta_imb": delta_imb_sym,
        "delta_spread": delta_spread_sym
    }
    normalizer.fit(train_channels_df, winsor_limits=winsor_limits, dynamic_data=dynamic_data)
    normalizer.save(scaler_type=args.scaler_type, winsor_limits=winsor_limits)
    print(f"✓ Normalizer fitted on {len(train_channels_df)} samples")

    # Задача 324.4: Агрегированная диагностика по всему train split после fit
    # Применяем полный pipeline (symlog → robust → clamp) и считаем статистику
    diag_metrics = _log_dynamic_train_diagnostics(
        {"ofi": ofi_sym, "delta_imb": delta_imb_sym, "delta_spread": delta_spread_sym},
        normalizer,
        clip_limit=4.0
    )

    # Задача 324.5: Hard guard — останавливаем обучение при плохом scale
    allow_bad = getattr(args, 'allow_bad_dynamic_scale', False)
    _check_dynamic_scale_guard(diag_metrics, allow_bad_scale=allow_bad)


def _log_dynamic_train_diagnostics(dynamic_sym: dict, normalizer, clip_limit: float = 4.0):
    """
    Задача 324.4: Агрегированная диагностика dynamic-каналов по всему train split.
    Печатается один раз после fit normalizer, до старта первой эпохи.
    Возвращает dict с метриками для последующего guard-check.
    """
    print("\n" + "=" * 70)
    print("DYNAMIC CHANNEL TRAIN DIAGNOSTICS (после fit normalizer)")
    print("=" * 70)
    metrics = {}
    for name, sym_arr in dynamic_sym.items():
        p = normalizer.dynamic_params.get(name, {})
        median = p.get("median", 0.0)
        iqr = p.get("iqr", 1.0)
        eps = normalizer.eps

        normed = (sym_arr - median) / (iqr + eps)

        n = len(normed)
        below = np.sum(normed < -clip_limit)
        above = np.sum(normed > clip_limit)
        sat_pct = (below + above) / n * 100 if n > 0 else 0.0
        zero_pct = np.sum(normed == 0.0) / n * 100 if n > 0 else 0.0

        p01 = float(np.percentile(normed, 1))
        p50 = float(np.percentile(normed, 50))
        p99 = float(np.percentile(normed, 99))
        dyn_range = p99 - p01

        print(f"\n  [{name}]")
        print(f"    fit params: median={median:.6f}, iqr={iqr:.6f}")
        print(f"    min={normed.min():.4f}, max={normed.max():.4f}, mean={normed.mean():.4f}, std={normed.std():.4f}")
        print(f"    p01={p01:.4f}, p50={p50:.4f}, p99={p99:.4f}, range(p99-p01)={dyn_range:.4f}")
        print(f"    saturation: below={below/n*100:.2f}%, above={above/n*100:.2f}%, total={sat_pct:.2f}%")
        print(f"    zero%={zero_pct:.2f}%")

        metrics[name] = {
            "sat_pct": sat_pct,
            "zero_pct": zero_pct,
            "dyn_range": dyn_range,
        }

    print("=" * 70 + "\n")
    return metrics


def _check_dynamic_scale_guard(metrics: dict, allow_bad_scale: bool = False):
    """
    Задача 324.5: Hard guard — останавливает обучение при заведомо плохом dynamic scale.
    Проверяет каждый канал на три условия:
      1. saturation > 10%
      2. zero% > 95% (канал схлопнулся в ноль)
      3. range(p99-p01) < 0.01 (канал фактически константный)
    """
    SAT_THRESHOLD = 10.0
    ZERO_THRESHOLD = 95.0
    RANGE_THRESHOLD = 0.01

    violations = []
    for name, m in metrics.items():
        if m["sat_pct"] > SAT_THRESHOLD:
            violations.append(
                f"  [{name}] saturation={m['sat_pct']:.2f}% > {SAT_THRESHOLD}% — слишком много значений за пределами clamp"
            )
        if m["zero_pct"] > ZERO_THRESHOLD:
            violations.append(
                f"  [{name}] zero%={m['zero_pct']:.2f}% > {ZERO_THRESHOLD}% — канал схлопнулся в ноль"
            )
        if m["dyn_range"] < RANGE_THRESHOLD:
            violations.append(
                f"  [{name}] range(p99-p01)={m['dyn_range']:.6f} < {RANGE_THRESHOLD} — канал фактически константный"
            )

    if violations:
        msg = (
            "\n" + "!" * 70 + "\n"
            "DYNAMIC SCALE GUARD: Обнаружены проблемы с нормализацией dynamic-каналов!\n"
            "Обучение остановлено. Исправьте pipeline или используйте --allow-bad-dynamic-scale.\n\n"
            "Нарушения:\n" + "\n".join(violations) + "\n"
            "!" * 70
        )
        if allow_bad_scale:
            print(msg)
            print("[WARN] --allow-bad-dynamic-scale активен, продолжаем несмотря на нарушения.\n")
        else:
            raise RuntimeError(msg)


def _run_normalized_nan_checks(train_ds, full_dataset):
    """Проверка данных на NaN после нормализации (sample-based)."""
    print("\nПроверка данных на NaN после нормализации (sampling)...")
    nan_check_samples = min(100, len(train_ds))
    nan_found = False

    for i in range(0, nan_check_samples, 10):
        try:
            ds = train_ds if i < len(train_ds) else full_dataset
            sample = ds[i]
            x, y, vol_target, weight = sample[:4]

            if i == 0:
                print(f"Sample Normalized Tensor (first 5 features of channel 0): {x[0, :5]}")

            if torch.isnan(x).any():
                print(f"⚠️  WARNING: NaN обнаружен в признаках (x) на индексе {i}")
                nan_found = True
            if torch.isnan(torch.tensor(y)).any():
                print(f"⚠️  WARNING: NaN обнаружен в метках (y) на индексе {i}")
                nan_found = True
            if torch.isnan(torch.tensor(vol_target)).any():
                print(f"⚠️  WARNING: NaN обнаружен в целевой волатильности (vol_target) на индексе {i}")
                nan_found = True
        except Exception as e:
            print(f"⚠️  WARNING: Ошибка при проверке индекса {i}: {e}")
            nan_found = True

    if nan_found:
        print("\n" + "!" * 80)
        print("⚠️  CRITICAL WARNING: Обнаружены NaN значения в данных!")
        print("   Это может привести к нестабильности обучения и NaN в метриках.")
        print("   Рекомендации:")
        print("   1. Проверьте качество исходных Parquet файлов")
        print("   2. Проверьте параметры нормализации (scaler_type, winsor_limits)")
        print("   3. Проверьте параметры feature engineering")
        print("!" * 80 + "\n")
    else:
        print(f"✓ Проверка завершена: NaN не обнаружены в {nan_check_samples} проверенных примерах")


def _compute_class_weights(full_dataset, train_ds, args):
    """Вычисляет веса классов на основе тренировочного набора."""
    print("Calculating class weights from training set...")
    train_labels = full_dataset.labels[train_ds.indices]
    classes, counts_list = np.unique(train_labels, return_counts=True)

    counts = np.zeros(3, dtype=np.int64)
    for cls, count in zip(classes, counts_list):
        if 0 <= cls < 3:
            counts[int(cls)] = count

    total_samples = np.sum(counts)
    smoothing = args.class_weight_smooth
    n_classes = 3

    weights = total_samples / (n_classes * (counts + smoothing))
    weights = weights / np.mean(weights)

    flat_ratio = counts[0] / total_samples if total_samples > 0 else 1.0
    if flat_ratio > 0.85:
        amplification = 5.0
        print(f"[ADJUST] Flat class dominating: {flat_ratio:.1%}. Amplifying Up/Down weights by {amplification}x.")
        weights[1] *= amplification
        weights[2] *= amplification

    print(f"Effective class weights: [Flat: {weights[0]:.2f}, Up: {weights[1]:.2f}, Down: {weights[2]:.2f}]")
    return weights


def prepare_training_data(args, paths, winsor_limits, horizons, num_horizons, horizon_weights) -> PreparedTrainingData:
    """
    Полный data pipeline: load -> feature engineer -> labeler -> dataset
    -> chronological split -> normalizer.fit(train only) -> dataloaders -> class weights.

    Порядок нельзя менять.
    """
    base_path = paths.base_path
    data_path = paths.data_path
    norm_params_path = paths.norm_params_path

    # 1. Парсим лаги past returns
    past_returns_lags = _parse_past_returns_lags(args.past_returns_lags)
    n_past_returns = len(past_returns_lags)
    in_channels = 11  # Задача 319: 11 каналов

    print(f'Using past returns lags: {past_returns_lags}')
    print(f'Total input channels: {in_channels} (MicropriceDev, Vol, Imb, OFI, VIB, Ret_10, Ret_50, Ret_100, Spread, DeltaImb, DeltaSpread)')
    print(f'Data loading mode: {args.data_mode}')

    # 2. Загрузка данных
    print(f"Loading data for {args.symbol} from {data_path}...")
    loader = LOBDataLoader(str(data_path), args.symbol)
    df = loader.load_data(lazy=False)

    # 3. Feature Engineering
    print("Engineering features...")
    fe = FeatureEngineer(n_levels=50)
    df = fe.transform(df)

    # 4. Разметка (Labeler)
    print("Adding labels...")
    labeler = Labeler(
        horizon=horizons,
        threshold=args.threshold,
        dynamic_threshold=False
    )
    df = labeler.add_labels(df)

    # 5. Инициализация Normalizer (fit будет позже на train set)
    print("Initializing normalizer...")
    normalizer = Normalizer(norm_params_path, scale_multiplier=args.scale_multiplier)

    # 5.5. RegimeDetector — временно отключён (Задача 155 приостановлена)
    regime_detector = None
    regime_weights = None
    num_regimes = 0

    # 6. Параметры временного взвешивания
    if args.use_time_weighting:
        # Placeholder: class_weights будут вычислены позже, передаём None пока
        time_weighting_params = {
            'half_life_hours': args.half_life_hours,
            'min_weight': args.min_sample_weight,
            'class_weights': None
        }
        print(f"Time weighting enabled: half_life={args.half_life_hours}h, min_weight={args.min_sample_weight}")
    else:
        time_weighting_params = {
            'half_life_hours': 24.0,
            'min_weight': 1.0,
            'class_weights': None
        }

    # 7. Создание полного датасета
    print(f"Creating dataset in 'memory' mode (raw features)...")
    full_dataset = build_full_dataset(
        df, args, past_returns_lags, winsor_limits, normalizer,
        regime_detector, time_weighting_params
    )

    # 7.1. Проверка NaN в сырых данных
    if np.isnan(full_dataset.x_raw).any():
        raise ValueError("КРИТИЧНО: Входящие features содержат NaN строки для запуска обучения!")

    # 8. Хронологическое разделение 70/15/15
    train_ds, val_ds, test_ds, train_indices, val_indices, test_indices = \
        split_dataset_chronologically(full_dataset)

    total_len = len(full_dataset)
    print(f"\nChronological split verification:")
    print(f"  Train: indices {train_indices[0]}-{train_indices[-1]} ({len(train_ds)} samples, {len(train_ds)/total_len*100:.1f}%)")
    print(f"  Val:   indices {val_indices[0]}-{val_indices[-1]} ({len(val_ds)} samples, {len(val_ds)/total_len*100:.1f}%)")
    print(f"  Test:  indices {test_indices[0]}-{test_indices[-1]} ({len(test_ds)} samples, {len(test_ds)/total_len*100:.1f}%)")

    # 9. Fit нормализатора только на train-части
    _fit_normalizer_on_train(
        full_dataset, train_ds, normalizer, args, winsor_limits
    )

    # 10. NaN диагностика после нормализации
    _run_normalized_nan_checks(train_ds, full_dataset)

    print(f"Dataset split (Chronological): Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    if args.use_symmetric_flip or args.volume_jitter_range > 0:
        print(f"Augmentation enabled for training: flip={args.use_symmetric_flip}, jitter={args.volume_jitter_range}, prob={args.augment_prob}")

    # 11. DataLoaders
    train_loader = DataLoader(train_ds, **build_dataloader_kwargs(args, shuffle=True))
    val_loader = DataLoader(val_ds, **build_dataloader_kwargs(args, shuffle=False))
    test_loader = DataLoader(test_ds, **build_dataloader_kwargs(args, shuffle=False))

    # 12. Веса классов
    class_weights = _compute_class_weights(full_dataset, train_ds, args)

    return PreparedTrainingData(
        df=df,
        normalizer=normalizer,
        full_dataset=full_dataset,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        past_returns_lags=past_returns_lags,
        in_channels=in_channels,
        num_horizons=num_horizons,
        horizon_weights=horizon_weights,
        regime_detector=regime_detector,
        regime_weights=regime_weights,
        num_regimes=num_regimes,
        n_past_returns=n_past_returns,
        time_weighting_params=time_weighting_params,
    )
