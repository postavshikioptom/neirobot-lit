#!/usr/bin/env python3
"""
Скрипт калибровки вероятностей модели с использованием Temperature Scaling.

Temperature Scaling - это простой и эффективный метод пост-обработки для калибровки
вероятностей нейронных сетей. Метод находит оптимальную температуру T, которая
минимизирует Negative Log Likelihood (NLL) на валидационной выборке.

Калиброванные вероятности: p_calibrated = softmax(logits / T)

Использование:
    python calibrate.py --symbol BTCUSDT --checkpoint path/to/model.ckpt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset

# Импорты из проекта
from src.train import LiTModule
from src.dataset import LOBDataset, LOBDataLoader
from src.features import FeatureEngineer
from src.labels import Labeler
from src.normalization import Normalizer
from src.utils import calculate_ece, plot_reliability_diagram


class TemperatureScaler(nn.Module):
    """
    Модуль для калибровки вероятностей с использованием Temperature Scaling.
    
    Temperature Scaling применяет единственный скалярный параметр T ко всем логитам:
    calibrated_logits = logits / T
    
    Оптимальное значение T находится путем минимизации NLL на валидационной выборке.
    """
    def __init__(self, initial_temperature=1.5):
        super().__init__()
        # Инициализируем температуру как обучаемый параметр
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits):
        """
        Применяет temperature scaling к логитам.
        
        Args:
            logits: тензор логитов (N, C)
        
        Returns:
            calibrated_logits: логиты, разделенные на температуру (N, C)
        """
        return logits / self.temperature

    def get_temperature(self):
        """Возвращает текущее значение температуры."""
        return self.temperature.item()


def find_temperature(logits, labels, initial_temperature=1.5, max_iter=50, verbose=True):
    """
    Находит оптимальную температуру для калибровки модели.
    
    Использует LBFGS оптимизатор для минимизации Negative Log Likelihood (NLL)
    на валидационной выборке. LBFGS - это квази-ньютоновский метод, который
    эффективен для оптимизации одного параметра.
    
    Args:
        logits: тензор логитов модели (N, C)
        labels: истинные метки (N,)
        initial_temperature: начальное значение температуры (по умолчанию 1.5)
        max_iter: максимальное количество итераций LBFGS (по умолчанию 50)
        verbose: выводить ли информацию о процессе оптимизации
    
    Returns:
        float: оптимальное значение температуры
    """
    # Создаем scaler
    scaler = TemperatureScaler(initial_temperature=initial_temperature)
    
    # LBFGS - стандарт для поиска температуры
    # Используем небольшой learning rate для стабильности
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)
    
    # Функция потерь - Cross Entropy (эквивалентна NLL для классификации)
    criterion = nn.CrossEntropyLoss()
    
    # Переводим данные на устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler.to(device)
    logits = logits.to(device)
    labels = labels.to(device)
    
    # Вычисляем начальные метрики
    with torch.no_grad():
        initial_loss = criterion(logits, labels).item()
        initial_probs = torch.softmax(logits, dim=1)
        initial_ece = calculate_ece(initial_probs, labels)
    
    if verbose:
        print(f"\nНачальные метрики (T=1.0):")
        print(f"  NLL: {initial_loss:.4f}")
        print(f"  ECE: {initial_ece:.4f}")
        print(f"\nНачало оптимизации температуры (initial_T={initial_temperature})...")
    
    # Функция для вычисления loss (требуется для LBFGS)
    def eval_loss():
        optimizer.zero_grad()
        calibrated_logits = scaler(logits)
        loss = criterion(calibrated_logits, labels)
        loss.backward()
        return loss
    
    # Запускаем оптимизацию
    optimizer.step(eval_loss)
    
    # Получаем оптимальную температуру
    optimal_temperature = scaler.get_temperature()
    
    # Вычисляем финальные метрики
    with torch.no_grad():
        calibrated_logits = scaler(logits)
        final_loss = criterion(calibrated_logits, labels).item()
        final_probs = torch.softmax(calibrated_logits, dim=1)
        final_ece = calculate_ece(final_probs, labels)
    
    if verbose:
        print(f"\nОптимизация завершена!")
        print(f"  Оптимальная температура: {optimal_temperature:.4f}")
        print(f"\nФинальные метрики:")
        print(f"  NLL: {final_loss:.4f} (изменение: {final_loss - initial_loss:+.4f})")
        print(f"  ECE: {final_ece:.4f} (изменение: {final_ece - initial_ece:+.4f})")
    
    return optimal_temperature


def calibrate_model(args):
    """
    Основная функция калибровки модели.
    
    1. Загружает обученную модель
    2. Получает логиты на валидационной выборке
    3. Находит оптимальную температуру
    4. Сохраняет температуру в metadata.json
    5. Генерирует reliability diagrams до и после калибровки
    """
    print(f"=== Калибровка модели для {args.symbol} ===\n")
    
    # 1. Настройка путей
    base_path = Path(__file__).parent.parent
    data_path = base_path / "bots" / args.symbol / "data" / "raw"
    norm_params_path = base_path / "bots" / args.symbol / "model" / "norm_params.json"
    model_dir = base_path / "bots" / args.symbol / "model"
    metadata_path = model_dir / "metadata.json"
    reports_dir = model_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Загрузка модели
    print(f"Загрузка модели из {args.checkpoint}...")
    model_module = LiTModule.load_from_checkpoint(args.checkpoint)
    model = model_module.model
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Модель загружена на устройство: {device}")
    
    # Получаем параметры из модели
    seq_len = model_module.hparams.get("seq_len", 100)
    past_returns_lags = model_module.hparams.get("past_returns_lags", [10, 50, 100])
    n_past_returns = len(past_returns_lags)
    horizon = args.horizon
    threshold = args.threshold
    
    print(f"Параметры модели: seq_len={seq_len}, past_returns_lags={past_returns_lags}")
    
    # 3. Загрузка и подготовка данных
    print(f"\nЗагрузка данных для {args.symbol}...")
    loader = LOBDataLoader(str(data_path), args.symbol)
    df = loader.load_data(lazy=False)
    
    # Генерация признаков
    print("Генерация признаков...")
    fe = FeatureEngineer(n_levels=50)
    df = fe.transform(df)
    
    # Разметка
    print("Добавление меток...")
    labeler = Labeler(horizon=horizon, threshold=threshold)
    df = labeler.add_labels(df)
    
    # Нормализация
    print("Нормализация данных...")
    normalizer = Normalizer(norm_params_path)
    if not normalizer.params:
        print("Параметры нормализации не найдены, выполняется fit...")
        normalizer.fit(df)
        normalizer.save()
    df = normalizer.transform(df)
    
    # 4. Создание датасета и разделение
    print("Создание датасета...")
    full_dataset = LOBDataset(df, seq_len=seq_len, n_past_returns=n_past_returns, data_mode="memory")
    
    total_len = len(full_dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    
    # Валидационная выборка: 70%-85% данных
    val_ds = Subset(full_dataset, range(train_size, train_size + val_size))
    
    print(f"Размер валидационной выборки: {len(val_ds)}")
    
    # 5. Получение логитов на валидационной выборке
    print("\nПолучение логитов на валидационной выборке...")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Inference"):
            logits = model(x.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(y)
    
    # Объединяем все батчи
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print(f"Получено логитов: {logits_tensor.shape}")
    
    # 6. Вычисляем метрики до калибровки
    print("\n=== Метрики ДО калибровки ===")
    with torch.no_grad():
        probs_before = torch.softmax(logits_tensor, dim=1)
        ece_before = calculate_ece(probs_before, labels_tensor)
        nll_before = nn.CrossEntropyLoss()(logits_tensor, labels_tensor).item()
    
    print(f"NLL: {nll_before:.4f}")
    print(f"ECE: {ece_before:.4f}")
    
    # Сохраняем reliability diagram до калибровки
    plot_reliability_diagram(
        probs_before.numpy(),
        labels_tensor.numpy(),
        reports_dir / "reliability_before_calibration.png",
        bins=args.bins
    )
    
    # 7. Поиск оптимальной температуры
    print("\n=== Поиск оптимальной температуры ===")
    optimal_temperature = find_temperature(
        logits_tensor,
        labels_tensor,
        initial_temperature=args.initial_temperature,
        max_iter=args.max_iter,
        verbose=True
    )
    
    # 8. Вычисляем метрики после калибровки
    print("\n=== Метрики ПОСЛЕ калибровки ===")
    with torch.no_grad():
        calibrated_logits = logits_tensor / optimal_temperature
        probs_after = torch.softmax(calibrated_logits, dim=1)
        ece_after = calculate_ece(probs_after, labels_tensor)
        nll_after = nn.CrossEntropyLoss()(calibrated_logits, labels_tensor).item()
    
    print(f"NLL: {nll_after:.4f}")
    print(f"ECE: {ece_after:.4f}")
    
    # Сохраняем reliability diagram после калибровки
    plot_reliability_diagram(
        probs_after.numpy(),
        labels_tensor.numpy(),
        reports_dir / "reliability_after_calibration.png",
        bins=args.bins
    )
    
    # 9. Сохранение температуры в metadata.json
    print(f"\nСохранение температуры в {metadata_path}...")
    
    # Загружаем существующий metadata или создаем новый
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Добавляем температуру и метрики калибровки
    metadata['temperature'] = float(optimal_temperature)
    metadata['calibration'] = {
        'ece_before': float(ece_before),
        'ece_after': float(ece_after),
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'bins': args.bins
    }
    
    # Сохраняем обновленный metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Температура сохранена: T = {optimal_temperature:.4f}")
    
    # 10. Итоговая сводка
    print("\n" + "="*60)
    print("ИТОГОВАЯ СВОДКА КАЛИБРОВКИ")
    print("="*60)
    print(f"Оптимальная температура: {optimal_temperature:.4f}")
    print(f"\nУлучшение метрик:")
    print(f"  ECE: {ece_before:.4f} → {ece_after:.4f} ({(ece_after - ece_before)/ece_before * 100:+.2f}%)")
    print(f"  NLL: {nll_before:.4f} → {nll_after:.4f} ({(nll_after - nll_before)/nll_before * 100:+.2f}%)")
    print(f"\nГрафики сохранены в: {reports_dir}")
    print(f"  - reliability_before_calibration.png")
    print(f"  - reliability_after_calibration.png")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Калибровка модели с использованием Temperature Scaling"
    )
    
    # Обязательные параметры
    parser.add_argument("--symbol", type=str, required=True, 
                       help="Символ для калибровки (например, BTCUSDT)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Путь к checkpoint модели (.ckpt)")
    
    # Параметры данных
    parser.add_argument("--horizon", type=int, default=100,
                       help="Горизонт предсказания для меток (по умолчанию 100)")
    parser.add_argument("--threshold", type=float, default=0.0005,
                       help="Порог для меток (по умолчанию 0.0005)")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Размер батча для inference (по умолчанию 256)")
    
    # Параметры калибровки
    parser.add_argument("--initial_temperature", type=float, default=1.5,
                       help="Начальное значение температуры (по умолчанию 1.5)")
    parser.add_argument("--max_iter", type=int, default=50,
                       help="Максимальное количество итераций LBFGS (по умолчанию 50)")
    parser.add_argument("--bins", type=int, default=15,
                       help="Количество корзин для ECE и reliability diagram (по умолчанию 15)")
    
    args = parser.parse_args()
    
    # Запускаем калибровку
    calibrate_model(args)


if __name__ == "__main__":
    main()
