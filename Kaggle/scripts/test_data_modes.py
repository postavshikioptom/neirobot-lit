"""
Тестовый скрипт для проверки всех режимов загрузки данных (memory, streaming, memmap).
Проверяет:
1. Корректность загрузки в каждом режиме
2. Идентичность результатов между режимами (parity check)
3. Производительность каждого режима
"""

import sys
import time
import numpy as np
import polars as pl
import psutil
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import LOBPyTorchDataset, LOBDataset
from features import FeatureEngineer
from labels import Labeler
from normalization import Normalizer

def get_memory_usage():
    """Возвращает текущее использование памяти в GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def test_data_mode(mode: str, df, seq_len: int, n_past_returns: int, cache_dir=None):
    """Тестирует один режим загрузки данных"""
    print(f"\n{'='*60}")
    print(f"Testing {mode.upper()} mode")
    print(f"{'='*60}")
    
    mem_before = get_memory_usage()
    start_time = time.time()
    
    # Создаем датасет
    if mode == "streaming":
        dataset = LOBPyTorchDataset(
            df.lazy() if isinstance(df, pl.DataFrame) else df,
            seq_len=seq_len,
            n_past_returns=n_past_returns,
            data_mode=mode
        )
    else:
        dataset = LOBPyTorchDataset(
            df,
            seq_len=seq_len,
            n_past_returns=n_past_returns,
            data_mode=mode,
            cache_dir=cache_dir
        )
    
    init_time = time.time() - start_time
    mem_after_init = get_memory_usage()
    
    print(f"Dataset initialized in {init_time:.2f}s")
    print(f"Memory usage: {mem_after_init:.2f} GB (delta: {mem_after_init - mem_before:.2f} GB)")
    print(f"Dataset length: {len(dataset)}")
    
    # Тестируем доступ к элементам
    n_samples = min(100, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    
    samples = []
    labels = []
    
    start_time = time.time()
    for idx in indices:
        x, y = dataset[idx]
        samples.append(x.numpy())
        labels.append(y.item())
    
    access_time = time.time() - start_time
    mem_after_access = get_memory_usage()
    
    print(f"Accessed {n_samples} samples in {access_time:.2f}s ({access_time/n_samples*1000:.2f}ms per sample)")
    print(f"Memory usage after access: {mem_after_access:.2f} GB (delta: {mem_after_access - mem_after_init:.2f} GB)")
    
    # Проверяем форму данных
    sample_shape = samples[0].shape
    print(f"Sample shape: {sample_shape}")
    
    # Проверяем распределение классов
    class_dist = dataset.get_class_distribution()
    print(f"Class distribution: Up={class_dist[0]}, Down={class_dist[1]}, Flat={class_dist[2]}")
    
    return {
        'mode': mode,
        'init_time': init_time,
        'access_time': access_time,
        'mem_delta_init': mem_after_init - mem_before,
        'mem_delta_access': mem_after_access - mem_after_init,
        'samples': np.array(samples),
        'labels': np.array(labels),
        'sample_shape': sample_shape,
        'class_dist': class_dist
    }

def compare_results(results):
    """Сравнивает результаты между режимами (parity check)"""
    print(f"\n{'='*60}")
    print("PARITY CHECK: Comparing results between modes")
    print(f"{'='*60}")
    
    modes = list(results.keys())
    reference_mode = modes[0]
    reference = results[reference_mode]
    
    all_match = True
    
    for mode in modes[1:]:
        current = results[mode]
        
        print(f"\nComparing {reference_mode} vs {mode}:")
        
        # Сравниваем формы
        if reference['sample_shape'] != current['sample_shape']:
            print(f"  ❌ Shape mismatch: {reference['sample_shape']} vs {current['sample_shape']}")
            all_match = False
        else:
            print(f"  ✓ Shape match: {reference['sample_shape']}")
        
        # Сравниваем распределение классов
        if not np.array_equal(reference['class_dist'], current['class_dist']):
            print(f"  ❌ Class distribution mismatch:")
            print(f"     {reference_mode}: {reference['class_dist']}")
            print(f"     {mode}: {current['class_dist']}")
            all_match = False
        else:
            print(f"  ✓ Class distribution match")
        
        # Сравниваем метки
        if not np.array_equal(reference['labels'], current['labels']):
            print(f"  ❌ Labels mismatch")
            all_match = False
        else:
            print(f"  ✓ Labels match")
        
        # Сравниваем сэмплы (с учетом float precision)
        max_diff = np.max(np.abs(reference['samples'] - current['samples']))
        if max_diff > 1e-5:
            print(f"  ❌ Samples mismatch (max diff: {max_diff})")
            all_match = False
        else:
            print(f"  ✓ Samples match (max diff: {max_diff:.2e})")
    
    if all_match:
        print(f"\n✅ PARITY CHECK PASSED: All modes produce identical results")
    else:
        print(f"\n❌ PARITY CHECK FAILED: Some modes produce different results")
    
    return all_match

def print_performance_summary(results):
    """Выводит сводку по производительности"""
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Mode':<12} {'Init Time':<12} {'Access Time':<15} {'Mem Init':<12} {'Mem Access':<12}")
    print("-" * 60)
    
    for mode, data in results.items():
        print(f"{mode:<12} {data['init_time']:>10.2f}s {data['access_time']:>13.2f}s "
              f"{data['mem_delta_init']:>10.2f}GB {data['mem_delta_access']:>10.2f}GB")

def main():
    # Настройка
    symbol = "BTCUSDT"
    seq_len = 100
    n_past_returns = 3
    
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "bots" / symbol / "data" / "raw"
    cache_dir = base_path / "bots" / symbol / "models" / "cache_test"
    
    print(f"Testing data loading modes for {symbol}")
    print(f"Data path: {data_path}")
    print(f"Cache dir: {cache_dir}")
    
    # Проверяем доступную память
    mem = psutil.virtual_memory()
    print(f"\nSystem memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
    
    # Загружаем и подготавливаем данные
    print("\nLoading and preparing data...")
    loader = LOBDataset(str(data_path), symbol)
    df = loader.load_data(lazy=False)
    
    # Генерация признаков
    fe = FeatureEngineer(n_levels=50)
    df = fe.transform(df)
    
    # Разметка
    labeler = Labeler(horizon=100, threshold=0.0005)
    df = labeler.add_labels(df)
    
    # Нормализация
    norm_params_path = base_path / "bots" / symbol / "models" / "norm_params_test.json"
    normalizer = Normalizer(norm_params_path)
    normalizer.fit(df)
    df = normalizer.transform(df)
    
    print(f"Data prepared: {len(df)} rows")
    
    # Тестируем все режимы
    results = {}
    
    # 1. Memory mode
    try:
        results['memory'] = test_data_mode('memory', df, seq_len, n_past_returns)
    except Exception as e:
        print(f"❌ Memory mode failed: {e}")
    
    # 2. Streaming mode
    try:
        results['streaming'] = test_data_mode('streaming', df, seq_len, n_past_returns)
    except Exception as e:
        print(f"❌ Streaming mode failed: {e}")
    
    # 3. Memmap mode
    try:
        results['memmap'] = test_data_mode('memmap', df, seq_len, n_past_returns, cache_dir)
    except Exception as e:
        print(f"❌ Memmap mode failed: {e}")
    
    # Сравниваем результаты
    if len(results) > 1:
        parity_passed = compare_results(results)
    else:
        print("\n⚠️  Not enough modes succeeded to perform parity check")
        parity_passed = False
    
    # Выводим сводку по производительности
    if results:
        print_performance_summary(results)
    
    # Итоговый результат
    print(f"\n{'='*60}")
    if len(results) == 3 and parity_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
