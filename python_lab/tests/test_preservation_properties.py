"""
Property-Based Tests for Preservation Requirements

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

Эти тесты захватывают наблюдаемое поведение на неисправленном коде для корректных входных данных.
Они должны ПРОХОДИТЬ на неисправленном коде и продолжать проходить после исправлений.

Preservation Requirements:
1. Все импорты работают корректно для корректных путей
2. Все функции работают так же для корректных входных данных
3. Все существующие тесты проходят
4. Результаты обработки данных идентичны для корректных входных данных
"""

import sys
from pathlib import Path

# Добавляем src в sys.path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import polars as pl
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import List, Union

# Импорты модулей для тестирования
from dataset import compute_intensity
from features import FeatureEngineer
from labels import Labeler
from normalization import Normalizer, symlog_transform
from utils import compute_metrics, count_parameters


class TestPreservationImports:
    """
    Property 1: Все импорты работают корректно для корректных путей
    
    Проверяем, что все модули импортируются без ошибок.
    """
    
    def test_import_dataset_module(self):
        """Проверяем импорт dataset модуля"""
        from dataset import compute_intensity, DataLoader
        assert callable(compute_intensity)
        assert DataLoader is not None
    
    def test_import_features_module(self):
        """Проверяем импорт features модуля"""
        from features import FeatureEngineer
        assert FeatureEngineer is not None
    
    def test_import_labels_module(self):
        """Проверяем импорт labels модуля"""
        from labels import Labeler
        assert Labeler is not None
    
    def test_import_normalization_module(self):
        """Проверяем импорт normalization модуля"""
        from normalization import Normalizer, symlog_transform
        assert Normalizer is not None
        assert callable(symlog_transform)
    
    def test_import_utils_module(self):
        """Проверяем импорт utils модуля"""
        from utils import compute_metrics, count_parameters
        assert callable(compute_metrics)
        assert callable(count_parameters)
    
    def test_import_train_module(self):
        """Проверяем импорт train модуля"""
        from train import LiTModule, TrainSubset
        assert LiTModule is not None
        assert TrainSubset is not None


class TestPreservationComputeIntensity:
    """
    Property 2: compute_intensity работает корректно для корректных входных данных
    
    Проверяем, что функция compute_intensity производит ожидаемые результаты.
    """
    
    @given(
        n=st.integers(min_value=1, max_value=1000),
        window=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_compute_intensity_output_shape(self, n, window):
        """
        Property: compute_intensity возвращает массив правильной формы
        
        Для любого n и window, результат должен быть массивом длины n.
        """
        timestamps = np.arange(n, dtype=np.float64)
        intensity = compute_intensity(timestamps, window=window)
        
        assert isinstance(intensity, np.ndarray)
        assert len(intensity) == n
        assert intensity.dtype in [np.float64, np.float32, np.int64, np.int32]
    
    @given(
        n=st.integers(min_value=1, max_value=1000),
        window=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_compute_intensity_values_positive(self, n, window):
        """
        Property: compute_intensity возвращает положительные значения
        
        Интенсивность должна быть положительной для всех точек.
        """
        timestamps = np.arange(n, dtype=np.float64)
        intensity = compute_intensity(timestamps, window=window)
        
        assert np.all(intensity > 0)
    
    @given(
        n=st.integers(min_value=1, max_value=1000),
        window=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_compute_intensity_monotonic_growth(self, n, window):
        """
        Property: compute_intensity растет монотонно до размера окна
        
        Интенсивность должна расти до window, затем оставаться постоянной.
        """
        timestamps = np.arange(n, dtype=np.float64)
        intensity = compute_intensity(timestamps, window=window)
        
        # Проверяем, что интенсивность растет до window
        for i in range(1, min(window, n)):
            assert intensity[i] >= intensity[i-1]
        
        # Проверяем, что после window интенсивность остается постоянной
        if n > window:
            for i in range(window, n):
                assert intensity[i] == intensity[i-1]


class TestPreservationFeatureEngineer:
    """
    Property 3: FeatureEngineer работает корректно для корректных входных данных
    
    Проверяем, что трансформация признаков работает правильно.
    """
    
    @staticmethod
    def create_valid_lob_dataframe(n_rows: int, n_levels: int = 50) -> pl.DataFrame:
        """Создает валидный DataFrame со схемой LOB"""
        data = {"timestamp_ms": list(range(n_rows))}
        
        # Добавляем ask_p и ask_v
        for i in range(n_levels):
            data[f"ask_p_{i}"] = [101.0 + i*0.1 + j*0.01 for j in range(n_rows)]
            data[f"ask_v_{i}"] = [10.0 + i + j*0.1 for j in range(n_rows)]
        
        # Добавляем bid_p и bid_v
        for i in range(n_levels):
            data[f"bid_p_{i}"] = [99.0 - i*0.1 + j*0.01 for j in range(n_rows)]
            data[f"bid_v_{i}"] = [15.0 + i + j*0.1 for j in range(n_rows)]
        
        return pl.DataFrame(data)
    
    @given(n_rows=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_engineer_output_shape(self, n_rows):
        """
        Property: FeatureEngineer возвращает DataFrame с правильной формой
        
        Результат должен иметь n_rows строк и 200+ признаков.
        """
        df = self.create_valid_lob_dataframe(n_rows, n_levels=50)
        fe = FeatureEngineer(n_levels=50)
        result = fe.transform(df)
        
        assert isinstance(result, pl.DataFrame)
        assert result.height == n_rows
        # 200 признаков LOB + метаданные
        assert result.width >= 200
    
    @given(n_rows=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_engineer_has_required_columns(self, n_rows):
        """
        Property: FeatureEngineer создает все требуемые колонки
        
        Результат должен содержать feat_ask_p, feat_bid_p, feat_ask_v, feat_bid_v.
        """
        df = self.create_valid_lob_dataframe(n_rows, n_levels=50)
        fe = FeatureEngineer(n_levels=50)
        result = fe.transform(df)
        
        # Проверяем наличие требуемых колонок
        assert any(c.startswith("feat_ask_p_") for c in result.columns)
        assert any(c.startswith("feat_bid_p_") for c in result.columns)
        assert any(c.startswith("feat_ask_v_") for c in result.columns)
        assert any(c.startswith("feat_bid_v_") for c in result.columns)
        assert "mid_price" in result.columns
    
    @given(n_rows=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_engineer_no_nan_values(self, n_rows):
        """
        Property: FeatureEngineer не создает NaN значения для корректных входных данных
        
        Все признаки должны быть конечными числами.
        """
        df = self.create_valid_lob_dataframe(n_rows, n_levels=50)
        fe = FeatureEngineer(n_levels=50)
        result = fe.transform(df)
        
        # Проверяем, что нет NaN значений
        for col in result.columns:
            if result[col].dtype in [pl.Float32, pl.Float64]:
                assert not result[col].is_nan().any()


class TestPreservationLabeler:
    """
    Property 4: Labeler работает корректно для корректных входных данных
    
    Проверяем, что генерация меток работает правильно.
    """
    
    @staticmethod
    def create_valid_price_dataframe(n_rows: int) -> pl.DataFrame:
        """Создает валидный DataFrame с ценами"""
        # Создаем цены с небольшими изменениями
        prices = [100.0 + i*0.01 + np.sin(i*0.1)*0.5 for i in range(n_rows)]
        return pl.DataFrame({"mid_price": prices})
    
    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        horizon=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_labeler_single_horizon_output_shape(self, n_rows, horizon):
        """
        Property: Labeler возвращает DataFrame с правильной формой
        
        Результат должен иметь n_rows - horizon строк и колонку 'label'.
        """
        df = self.create_valid_price_dataframe(n_rows)
        labeler = Labeler(horizon=horizon, threshold=0.0005)
        result = labeler.add_labels(df)
        
        assert isinstance(result, pl.DataFrame)
        assert "label" in result.columns
        # Последние horizon строк удаляются
        assert result.height == n_rows - horizon
    
    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        horizon=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_labeler_label_values_valid(self, n_rows, horizon):
        """
        Property: Labeler создает валидные значения меток
        
        Метки должны быть в диапазоне [-100, 0, 1, 2].
        """
        df = self.create_valid_price_dataframe(n_rows)
        labeler = Labeler(horizon=horizon, threshold=0.0005)
        result = labeler.add_labels(df)
        
        labels = result["label"].to_list()
        valid_labels = {-100, 0, 1, 2}
        assert all(label in valid_labels for label in labels)
    
    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        horizon=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_labeler_multi_horizon_output_shape(self, n_rows, horizon):
        """
        Property: Labeler с multi-horizon возвращает DataFrame с правильной формой
        
        Результат должен иметь колонки для каждого горизонта.
        """
        df = self.create_valid_price_dataframe(n_rows)
        horizons = [horizon, horizon*2, horizon*3]
        labeler = Labeler(horizon=horizons, threshold=0.0005)
        result = labeler.add_labels(df)
        
        assert isinstance(result, pl.DataFrame)
        for h in horizons:
            assert f"label_h{h}" in result.columns


class TestPreservationNormalizer:
    """
    Property 5: Normalizer работает корректно для корректных входных данных
    
    Проверяем, что нормализация работает правильно.
    """
    
    @staticmethod
    def create_valid_feature_dataframe(n_rows: int, n_features: int = 10) -> pl.DataFrame:
        """Создает валидный DataFrame с признаками"""
        data = {}
        for i in range(n_features):
            # Создаем признаки с разными масштабами
            data[f"feat_{i}"] = [np.random.randn() * (i+1) + i*10 for _ in range(n_rows)]
        return pl.DataFrame(data)
    
    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_normalizer_fit_returns_params(self, n_rows, n_features):
        """
        Property: Normalizer.fit возвращает параметры нормализации
        
        Результат должен быть словарем с параметрами для каждого признака.
        """
        df = self.create_valid_feature_dataframe(n_rows, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        
        normalizer = Normalizer()
        params = normalizer.fit(df, feature_names=feature_names)
        
        assert isinstance(params, dict)
        assert len(params) == n_features
    
    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_normalizer_transform_output_shape(self, n_rows, n_features):
        """
        Property: Normalizer.transform возвращает DataFrame с правильной формой
        
        Результат должен иметь ту же форму, что и входной DataFrame.
        """
        df = self.create_valid_feature_dataframe(n_rows, n_features)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        
        normalizer = Normalizer()
        normalizer.fit(df, feature_names=feature_names)
        result = normalizer.transform(df)
        
        assert isinstance(result, pl.DataFrame)
        assert result.height == n_rows
        assert result.width == n_features
    
    def test_symlog_transform_preserves_sign(self):
        """
        Property: symlog_transform сохраняет знак значения
        
        Положительные значения остаются положительными, отрицательные - отрицательными.
        """
        positive_values = np.array([0.1, 1.0, 10.0, 100.0])
        negative_values = np.array([-0.1, -1.0, -10.0, -100.0])
        
        result_pos = symlog_transform(positive_values)
        result_neg = symlog_transform(negative_values)
        
        assert np.all(result_pos > 0)
        assert np.all(result_neg < 0)


class TestPreservationUtils:
    """
    Property 6: Функции в utils работают корректно для корректных входных данных
    
    Проверяем, что утилиты работают правильно.
    """
    
    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_classes=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_compute_metrics_output_type(self, n_samples, n_classes):
        """
        Property: compute_metrics возвращает словарь метрик
        
        Результат должен быть словарем с метриками.
        """
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    def test_count_parameters_returns_positive_int(self):
        """
        Property: count_parameters возвращает положительное целое число
        
        Результат должен быть положительным целым числом.
        """
        import torch
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        param_count = count_parameters(model)
        
        assert isinstance(param_count, int)
        assert param_count > 0


class TestPreservationIntegration:
    """
    Property 7: Все существующие тесты проходят
    
    Проверяем, что базовая функциональность работает в интеграции.
    """
    
    def test_full_pipeline_imports(self):
        """
        Property: Полный пайплайн импортов работает
        
        Все модули должны импортироваться без ошибок.
        """
        # Импортируем все модули
        from dataset import compute_intensity, DataLoader
        from features import FeatureEngineer
        from labels import Labeler
        from normalization import Normalizer
        from utils import compute_metrics
        from train import LiTModule
        
        # Проверяем, что все импорты успешны
        assert all([
            compute_intensity,
            DataLoader,
            FeatureEngineer,
            Labeler,
            Normalizer,
            compute_metrics,
            LiTModule
        ])
    
    def test_feature_engineering_pipeline(self):
        """
        Property: Полный пайплайн обработки признаков работает
        
        От LOB данных до нормализованных признаков.
        """
        # Создаем LOB данные
        n_rows = 10
        n_levels = 50
        data = {"timestamp_ms": list(range(n_rows))}
        
        for i in range(n_levels):
            data[f"ask_p_{i}"] = [101.0 + i*0.1 for _ in range(n_rows)]
            data[f"ask_v_{i}"] = [10.0 + i for _ in range(n_rows)]
            data[f"bid_p_{i}"] = [99.0 - i*0.1 for _ in range(n_rows)]
            data[f"bid_v_{i}"] = [15.0 + i for _ in range(n_rows)]
        
        df = pl.DataFrame(data)
        
        # Применяем трансформацию признаков
        fe = FeatureEngineer(n_levels=50)
        feat_df = fe.transform(df)
        
        # Проверяем результат
        assert feat_df.height == n_rows
        assert feat_df.width >= 200
        assert "mid_price" in feat_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
