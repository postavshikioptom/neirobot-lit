"""
Скрипт для оценки важности признаков методом перестановки (Permutation Importance).
Задача №152: Анализ важности признаков для оптимизации входного тензора.

Использует глобальную перестановку по всему валидационному набору для корректной оценки
важности признаков в контексте временных рядов.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
import sys

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import LOBPyTorchDataset, get_val_loader
from lit_model import LiTModel
from normalization import Normalizer
from utils import plot_feature_importance_bar, plot_lob_importance_heatmap


def permute_feature_global(X_val, feature_idx, n_repeats=5, seed=42):
    """
    Глобальная перестановка признака по всему валидационному набору.
    
    Args:
        X_val: тензор (N, seq_len, channels, levels)
        feature_idx: индекс признака для перестановки (channel_idx, level_idx)
        n_repeats: количество повторений для расчета mean и std
        seed: seed для воспроизводимости
    
    Returns:
        list: список перемешанных тензоров
    """
    N = X_val.shape[0]
    channel_idx, level_idx = feature_idx
    
    permuted_tensors = []
    
    for repeat in range(n_repeats):
        # Создаем новый seed для каждого повторения
        torch.manual_seed(seed + repeat)
        
        # Глобальная перестановка по всем сэмплам
        perm = torch.randperm(N)
        
        # Клонируем тензор
        X_perm = X_val.clone()
        
        # Перемешиваем значения признака между всеми сэмплами
        # Сохраняем временную структуру внутри seq_len
        X_perm[:, :, channel_idx, level_idx] = X_val[perm, :, channel_idx, level_idx]
        
        permuted_tensors.append(X_perm)
    
    return permuted_tensors


def permute_group_global(X_val, group_indices, n_repeats=5, seed=42):
    """
    Глобальная перестановка группы признаков.
    
    Args:
        X_val: тензор (N, seq_len, channels, levels)
        group_indices: список индексов признаков [(channel_idx, level_idx), ...]
        n_repeats: количество повторений
        seed: seed для воспроизводимости
    
    Returns:
        list: список перемешанных тензоров
    """
    N = X_val.shape[0]
    permuted_tensors = []
    
    for repeat in range(n_repeats):
        torch.manual_seed(seed + repeat)
        perm = torch.randperm(N)
        
        X_perm = X_val.clone()
        
        # Перемешиваем все признаки группы одновременно
        for channel_idx, level_idx in group_indices:
            X_perm[:, :, channel_idx, level_idx] = X_val[perm, :, channel_idx, level_idx]
        
        permuted_tensors.append(X_perm)
    
    return permuted_tensors


def compute_mcc(model, X, y, device, batch_size=256):
    """
    Вычисляет MCC на заданных данных.
    
    Args:
        model: обученная модель
        X: тензор признаков (N, seq_len, channels, levels)
        y: тензор меток (N,)
        device: устройство для вычислений
        batch_size: размер батча
    
    Returns:
        float: значение MCC
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            
            # Получаем логиты
            outputs = model(batch_X)
            
            # Если модель возвращает tuple (multi-task), берем только логиты
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Получаем предсказания
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    
    return matthews_corrcoef(y_np, all_preds)


def compute_feature_importance(model, X_val, y_val, device, n_repeats=5, seed=42):
    """
    Вычисляет важность каждого признака методом перестановки.
    
    Args:
        model: обученная модель
        X_val: валидационный тензор (N, seq_len, channels, levels)
        y_val: валидационные метки (N,)
        device: устройство для вычислений
        n_repeats: количество повторений для расчета статистики
        seed: seed для воспроизводимости
    
    Returns:
        dict: словарь с результатами {feature_name: {mean_importance, std_dev}}
    """
    # Вычисляем baseline MCC
    print("\nВычисление baseline MCC...")
    baseline_mcc = compute_mcc(model, X_val, y_val, device)
    print(f"Baseline MCC: {baseline_mcc:.4f}")
    
    N, seq_len, channels, levels = X_val.shape
    
    # Словарь для хранения результатов
    importance_results = {}
    
    # Названия каналов
    channel_names = ['ask_p', 'ask_v', 'bid_p', 'bid_v']
    if channels > 4:
        # Добавляем past returns каналы
        for i in range(channels - 4):
            channel_names.append(f'past_return_{i+1}')
    
    print(f"\nАнализ важности {channels * levels} признаков...")
    print(f"Каналы: {channel_names}")
    print(f"Уровни: {levels}")
    
    # Итерируемся по всем признакам
    total_features = channels * levels
    
    with tqdm(total=total_features, desc="Feature Importance") as pbar:
        for channel_idx in range(channels):
            for level_idx in range(levels):
                feature_name = f"{channel_names[channel_idx]}_{level_idx}"
                
                # Перемешиваем признак n_repeats раз
                permuted_tensors = permute_feature_global(
                    X_val, 
                    (channel_idx, level_idx), 
                    n_repeats=n_repeats,
                    seed=seed
                )
                
                # Вычисляем MCC для каждой перестановки
                mcc_scores = []
                for X_perm in permuted_tensors:
                    mcc = compute_mcc(model, X_perm, y_val, device)
                    mcc_scores.append(mcc)
                
                # Вычисляем падение метрики (importance)
                # Положительное значение = признак важен (MCC упал)
                importance_scores = [baseline_mcc - mcc for mcc in mcc_scores]
                
                mean_importance = np.mean(importance_scores)
                std_importance = np.std(importance_scores)
                
                importance_results[feature_name] = {
                    'mean_importance': float(mean_importance),
                    'std_dev': float(std_importance),
                    'channel': channel_names[channel_idx],
                    'level': int(level_idx),
                    'channel_idx': int(channel_idx),
                    'level_idx': int(level_idx)
                }
                
                pbar.update(1)
    
    return importance_results, baseline_mcc


def compute_group_importance(model, X_val, y_val, device, n_repeats=5, seed=42):
    """
    Вычисляет важность групп признаков.
    
    Args:
        model: обученная модель
        X_val: валидационный тензор (N, seq_len, channels, levels)
        y_val: валидационные метки (N,)
        device: устройство для вычислений
        n_repeats: количество повторений
        seed: seed для воспроизводимости
    
    Returns:
        dict: словарь с результатами по группам
    """
    print("\nАнализ важности групп признаков...")
    
    # Вычисляем baseline MCC
    baseline_mcc = compute_mcc(model, X_val, y_val, device)
    
    N, seq_len, channels, levels = X_val.shape
    
    # Определяем группы признаков
    groups = {}
    
    # Группа 1: Price Levels (ask_p и bid_p для всех уровней)
    price_indices = []
    for level_idx in range(levels):
        price_indices.append((0, level_idx))  # ask_p
        price_indices.append((2, level_idx))  # bid_p
    groups['Price Levels'] = price_indices
    
    # Группа 2: Volume Levels (ask_v и bid_v для всех уровней)
    volume_indices = []
    for level_idx in range(levels):
        volume_indices.append((1, level_idx))  # ask_v
        volume_indices.append((3, level_idx))  # bid_v
    groups['Volume Levels'] = volume_indices
    
    # Группа 3: Spread/Imbalance (первые уровни + дополнительные признаки)
    # Spread = ask_p_0 - bid_p_0, Imbalance = (bid_v_0 - ask_v_0) / (bid_v_0 + ask_v_0)
    spread_imbalance_indices = [
        (0, 0),  # ask_p_0
        (1, 0),  # ask_v_0
        (2, 0),  # bid_p_0
        (3, 0),  # bid_v_0
    ]
    
    # Если есть дополнительные каналы (past returns или другие признаки),
    # проверяем наличие spread/imbalance признаков
    # Примечание: В текущей архитектуре spread/imbalance вычисляются из Level 0,
    # но если в будущем они будут отдельными каналами, они будут автоматически включены
    groups['Spread/Imbalance'] = spread_imbalance_indices
    
    # Если есть past returns, добавляем их как отдельную группу
    if channels > 4:
        past_returns_indices = []
        for channel_idx in range(4, channels):
            for level_idx in range(levels):
                past_returns_indices.append((channel_idx, level_idx))
        groups['Past Returns'] = past_returns_indices
    
    # Анализируем каждую группу
    group_results = {}
    
    for group_name, group_indices in groups.items():
        print(f"\nАнализ группы: {group_name} ({len(group_indices)} признаков)")
        
        # Перемешиваем группу n_repeats раз
        permuted_tensors = permute_group_global(
            X_val,
            group_indices,
            n_repeats=n_repeats,
            seed=seed
        )
        
        # Вычисляем MCC для каждой перестановки
        mcc_scores = []
        for X_perm in permuted_tensors:
            mcc = compute_mcc(model, X_perm, y_val, device)
            mcc_scores.append(mcc)
        
        # Вычисляем падение метрики
        importance_scores = [baseline_mcc - mcc for mcc in mcc_scores]
        
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)
        
        group_results[group_name] = {
            'mean_importance': float(mean_importance),
            'std_dev': float(std_importance),
            'n_features': len(group_indices)
        }
        
        print(f"  Mean Importance: {mean_importance:.4f} ± {std_importance:.4f}")
    
    return group_results


def save_importance_results(results, group_results, baseline_mcc, output_path, metadata):
    """
    Сохраняет результаты анализа важности в JSON.
    
    Args:
        results: словарь с результатами по признакам
        group_results: словарь с результатами по группам
        baseline_mcc: baseline MCC
        output_path: путь для сохранения
        metadata: дополнительные метаданные
    """
    # Сортируем результаты по убыванию важности
    sorted_results = dict(sorted(
        results.items(),
        key=lambda x: x[1]['mean_importance'],
        reverse=True
    ))
    
    # Формируем итоговый JSON
    output_data = {
        'metadata': {
            'baseline_mcc': float(baseline_mcc),
            'n_features': len(results),
            'n_repeats': metadata.get('n_repeats', 5),
            'symbol': metadata.get('symbol', 'UNKNOWN'),
            'model_path': metadata.get('model_path', ''),
            'timestamp': metadata.get('timestamp', '')
        },
        'individual_features': sorted_results,
        'group_importance': group_results
    }
    
    # Создаем директорию если не существует
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Результаты сохранены: {output_path}")
    
    # Выводим топ-10 признаков
    print("\nТоп-10 наиболее важных признаков:")
    for i, (feature_name, data) in enumerate(list(sorted_results.items())[:10], 1):
        print(f"  {i}. {feature_name}: {data['mean_importance']:.4f} ± {data['std_dev']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Feature Importance Analysis')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., CAKEUSDT)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='bots/{symbol}/data/raw', help='Path to parquet data')
    parser.add_argument('--output_dir', type=str, default='bots/{symbol}/model', help='Output directory')
    parser.add_argument('--n_repeats', type=int, default=5, help='Number of permutation repeats')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--skip_individual', action='store_true', help='Skip individual feature analysis')
    parser.add_argument('--skip_groups', action='store_true', help='Skip group analysis')
    
    args = parser.parse_args()
    
    # Подставляем symbol в пути
    args.data_path = args.data_path.format(symbol=args.symbol)
    args.output_dir = args.output_dir.format(symbol=args.symbol)
    
    # Устанавливаем seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Проверяем устройство
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем модель
    print(f"\nЗагрузка модели из {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Извлекаем гиперпараметры из checkpoint
    hparams = checkpoint.get('hyper_parameters', {})
    
    model = LiTModel(
        seq_len=hparams.get('seq_len', 100),
        in_channels=hparams.get('in_channels', 3),
        d_model=hparams.get('d_model', 64),
        nhead=hparams.get('nhead', 4),
        num_layers=hparams.get('num_layers', 2),
        dropout=hparams.get('dropout', 0.1),
        activation=hparams.get('activation', 'gelu_exact'),
        multi_task=hparams.get('multi_task', True)
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Модель загружена")
    print(f"  Параметры: seq_len={hparams.get('seq_len', 100)}, "
          f"in_channels={hparams.get('in_channels', 3)}, "
          f"d_model={hparams.get('d_model', 64)}")
    
    # Загружаем валидационные данные
    print(f"\nЗагрузка валидационных данных из {args.data_path}...")
    
    # Примечание: Можно использовать get_val_loader для упрощения загрузки,
    # но для глобальной перестановки нам нужен весь тензор в памяти,
    # поэтому мы загружаем данные напрямую
    
    # Используем LOBDataset напрямую (без DataLoader для глобальной перестановки)
    # Важно: shuffle=False для фиксированного порядка
    import polars as pl
    
    # Загружаем данные
    data_path = Path(args.data_path)
    pattern = f"{args.symbol}_*.parquet"
    files = list(data_path.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No parquet files found for {args.symbol} in {data_path}")
    
    df = pl.read_parquet(data_path / pattern)
    print(f"✓ Загружено {len(df)} строк")
    
    # Применяем нормализацию Z-score (задача 024)
    # Данные должны быть нормализованы так же, как при обучении модели
    norm_path = Path(args.model_path).parent / 'normalization_params.json'
    if norm_path.exists():
        print(f"\nПрименение нормализации из {norm_path}...")
        normalizer = Normalizer(norm_path)
        normalizer.load()
        df = normalizer.transform(df)
        print("✓ Нормализация применена")
    else:
        print(f"\n⚠️  WARNING: Файл нормализации не найден: {norm_path}")
        print("   Анализ будет проведен на ненормализованных данных!")
        print("   Это может привести к некорректным результатам.")
    
    # Создаем валидационный датасет
    # Используем последние 20% данных для валидации
    val_split = int(len(df) * 0.8)
    val_df = df.slice(val_split)
    
    print(f"Валидационная выборка: {len(val_df)} строк")
    
    val_ds = LOBPyTorchDataset(
        val_df,
        seq_len=hparams.get('seq_len', 100),
        n_past_returns=hparams.get('n_past_returns', 0),
        data_mode='memory',
        is_train=False  # Отключаем аугментацию
    )
    
    print(f"✓ Валидационный датасет создан: {len(val_ds)} сэмплов")
    
    # Собираем весь валидационный тензор в память
    print("\nСбор валидационного тензора в память...")
    X_val_list = []
    y_val_list = []
    
    for i in tqdm(range(len(val_ds)), desc="Loading data"):
        x, y, _, _ = val_ds[i]
        X_val_list.append(x)
        y_val_list.append(y)
    
    X_val = torch.stack(X_val_list)
    y_val = torch.stack(y_val_list)
    
    print(f"✓ Тензор собран: X_val.shape = {X_val.shape}, y_val.shape = {y_val.shape}")
    
    # Метаданные для сохранения
    from datetime import datetime
    metadata = {
        'symbol': args.symbol,
        'model_path': args.model_path,
        'n_repeats': args.n_repeats,
        'timestamp': datetime.now().isoformat()
    }
    
    # Анализ важности индивидуальных признаков
    individual_results = {}
    baseline_mcc = 0.0
    
    if not args.skip_individual:
        individual_results, baseline_mcc = compute_feature_importance(
            model, X_val, y_val, device,
            n_repeats=args.n_repeats,
            seed=args.seed
        )
    
    # Анализ важности групп
    group_results = {}
    
    if not args.skip_groups:
        group_results = compute_group_importance(
            model, X_val, y_val, device,
            n_repeats=args.n_repeats,
            seed=args.seed
        )
    
    # Сохраняем результаты
    output_path = Path(args.output_dir) / 'feature_importance.json'
    save_importance_results(
        individual_results,
        group_results,
        baseline_mcc,
        output_path,
        metadata
    )
    
    # Создаем визуализации
    if individual_results:
        print("\nСоздание визуализаций...")
        
        # Bar Chart
        bar_path = Path(args.output_dir) / 'feature_importance_bar.png'
        plot_feature_importance_bar(individual_results, top_k=20, save_path=str(bar_path))
        
        # LOB Heatmap
        heatmap_path = Path(args.output_dir) / 'lob_importance_heatmap.png'
        plot_lob_importance_heatmap(individual_results, n_levels=50, save_path=str(heatmap_path))
    
    print("\n✓ Анализ завершен!")


if __name__ == '__main__':
    main()
