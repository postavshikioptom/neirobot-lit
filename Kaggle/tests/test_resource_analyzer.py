#!/usr/bin/env python3
"""
Тесты для задачи 225: resource_analyzer.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import sys

# Добавляем путь к скриптам
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from resource_analyzer import (
    parse_log_file,
    calculate_correlations,
    generate_report,
)


@pytest.fixture
def sample_log_file():
    """Создает временный лог-файл с тестовыми данными"""
    content = """
2024-02-16T10:00:00.000Z INFO System metrics: CPU=45.2%, MEM=512000KB, DISK_R=1024B, DISK_W=2048B, LEAK=false
2024-02-16T10:00:00.100Z INFO Order executed: internal_latency_us=1500
2024-02-16T10:00:05.000Z INFO System metrics: CPU=50.5%, MEM=513000KB, DISK_R=1124B, DISK_W=2148B, LEAK=false
2024-02-16T10:00:05.100Z INFO Order executed: internal_latency_us=1600
2024-02-16T10:00:10.000Z INFO System metrics: CPU=55.8%, MEM=514000KB, DISK_R=1224B, DISK_W=2248B, LEAK=false
2024-02-16T10:00:10.100Z INFO Order executed: internal_latency_us=1700
2024-02-16T10:00:15.000Z INFO System metrics: CPU=60.1%, MEM=515000KB, DISK_R=1324B, DISK_W=2348B, LEAK=false
2024-02-16T10:00:15.100Z INFO Order executed: internal_latency_us=1800
2024-02-16T10:00:20.000Z INFO System metrics: CPU=65.4%, MEM=516000KB, DISK_R=1424B, DISK_W=2448B, LEAK=false
2024-02-16T10:00:20.100Z INFO Order executed: internal_latency_us=1900
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()


def test_parse_log_file(sample_log_file):
    """Тест парсинга лог-файла"""
    df = parse_log_file(sample_log_file)
    
    assert len(df) == 5, "Should parse 5 records"
    assert 'timestamp' in df.columns
    assert 'cpu_usage_pct' in df.columns
    assert 'memory_rss_kb' in df.columns
    assert 'internal_latency_us' in df.columns
    
    # Проверяем значения
    assert df['cpu_usage_pct'].iloc[0] == 45.2
    assert df['memory_rss_kb'].iloc[0] == 512000
    assert df['internal_latency_us'].iloc[0] == 1500


def test_parse_log_file_empty():
    """Тест парсинга пустого файла"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("No valid data here\n")
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="No data found"):
            parse_log_file(temp_path)
    finally:
        temp_path.unlink()


def test_calculate_correlations():
    """Тест вычисления корреляций"""
    # Создаем тестовые данные с известной корреляцией
    df = pd.DataFrame({
        'cpu_usage_pct': [10, 20, 30, 40, 50],
        'memory_rss_kb': [1000, 2000, 3000, 4000, 5000],
        'internal_latency_us': [100, 200, 300, 400, 500],
    })
    
    correlations = calculate_correlations(df)
    
    assert 'cpu_latency_pearson' in correlations
    assert 'cpu_latency_spearman' in correlations
    assert 'memory_latency_pearson' in correlations
    assert 'memory_latency_spearman' in correlations
    
    # Проверяем, что корреляция близка к 1.0 (идеальная линейная зависимость)
    assert correlations['cpu_latency_pearson'] > 0.99
    assert correlations['memory_latency_pearson'] > 0.99


def test_calculate_correlations_no_correlation():
    """Тест вычисления корреляций при отсутствии зависимости"""
    # Создаем данные без корреляции
    np.random.seed(42)
    df = pd.DataFrame({
        'cpu_usage_pct': np.random.rand(100) * 100,
        'memory_rss_kb': np.random.rand(100) * 10000,
        'internal_latency_us': np.random.rand(100) * 1000,
    })
    
    correlations = calculate_correlations(df)
    
    # Корреляция должна быть близка к 0
    assert abs(correlations['cpu_latency_pearson']) < 0.5
    assert abs(correlations['memory_latency_pearson']) < 0.5


def test_generate_report():
    """Тест генерации HTML отчета"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
        'cpu_usage_pct': [50.0] * 10,
        'memory_rss_kb': [500000] * 10,
        'internal_latency_us': [1000] * 10,
    })
    
    correlations = {
        'cpu_latency_pearson': 0.5,
        'cpu_latency_spearman': 0.6,
        'memory_latency_pearson': 0.3,
        'memory_latency_spearman': 0.4,
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        generate_report(df, correlations, output_path)
        
        # Проверяем, что файл создан
        assert output_path.exists()
        
        # Проверяем содержимое
        content = output_path.read_text(encoding='utf-8')
        assert 'System Resource Analysis Report' in content
        assert 'CPU Usage' in content
        assert 'Memory RSS' in content
        assert 'Internal Latency' in content
        assert '0.5000' in content  # Корреляция
    finally:
        output_path.unlink()


def test_correlation_interpretation():
    """Тест интерпретации корреляций"""
    # Сильная положительная корреляция
    df_strong = pd.DataFrame({
        'cpu_usage_pct': [10, 20, 30, 40, 50],
        'memory_rss_kb': [1000, 2000, 3000, 4000, 5000],
        'internal_latency_us': [100, 200, 300, 400, 500],
    })
    
    corr_strong = calculate_correlations(df_strong)
    assert corr_strong['cpu_latency_pearson'] > 0.9
    
    # Слабая корреляция
    np.random.seed(42)
    df_weak = pd.DataFrame({
        'cpu_usage_pct': np.random.rand(50) * 100,
        'memory_rss_kb': np.random.rand(50) * 10000,
        'internal_latency_us': np.random.rand(50) * 1000,
    })
    
    corr_weak = calculate_correlations(df_weak)
    assert abs(corr_weak['cpu_latency_pearson']) < 0.5


def test_data_sorting():
    """Тест сортировки данных по времени"""
    # Создаем данные в неправильном порядке
    df = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2024-01-01 10:02:00'),
            pd.Timestamp('2024-01-01 10:00:00'),
            pd.Timestamp('2024-01-01 10:01:00'),
        ],
        'cpu_usage_pct': [30, 10, 20],
        'memory_rss_kb': [3000, 1000, 2000],
        'internal_latency_us': [300, 100, 200],
    })
    
    # Сортируем
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Проверяем порядок
    assert df_sorted['cpu_usage_pct'].tolist() == [10, 20, 30]
    assert df_sorted['memory_rss_kb'].tolist() == [1000, 2000, 3000]


def test_missing_data_handling():
    """Тест обработки отсутствующих данных"""
    # Создаем данные с пропусками
    df = pd.DataFrame({
        'cpu_usage_pct': [10, 20, np.nan, 40, 50],
        'memory_rss_kb': [1000, 2000, 3000, np.nan, 5000],
        'internal_latency_us': [100, 200, 300, 400, 500],
    })
    
    # Удаляем строки с NaN
    df_clean = df.dropna()
    
    assert len(df_clean) == 3
    assert not df_clean.isnull().any().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
