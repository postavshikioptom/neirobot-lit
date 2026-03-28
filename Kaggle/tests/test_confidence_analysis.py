"""
Задача 224: Тесты для анализа распределения уверенности инференса

Тесты для model_confidence_analysis.py
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Добавляем путь к scripts для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from model_confidence_analysis import (
    calculate_entropy,
    calculate_ece,
    ks_test_comparison
)


class TestEntropyCalculation:
    """Тесты для расчета энтропии"""
    
    def test_uniform_distribution(self):
        """Равномерное распределение имеет максимальную энтропию"""
        probs = np.array([[1/3, 1/3, 1/3]])
        entropy = calculate_entropy(probs)
        
        # Максимальная энтропия для 3 классов = ln(3) ≈ 1.0986
        expected = np.log(3)
        np.testing.assert_almost_equal(entropy[0], expected, decimal=5)
    
    def test_certain_prediction(self):
        """Уверенное предсказание имеет минимальную энтропию"""
        probs = np.array([[1.0, 0.0, 0.0]])
        entropy = calculate_entropy(probs)
        
        # Минимальная энтропия = 0
        np.testing.assert_almost_equal(entropy[0], 0.0, decimal=5)
    
    def test_partial_certainty(self):
        """Частичная уверенность"""
        probs = np.array([[0.7, 0.2, 0.1]])
        entropy = calculate_entropy(probs)
        
        # Вычисляем вручную
        expected = -0.7 * np.log(0.7) - 0.2 * np.log(0.2) - 0.1 * np.log(0.1)
        np.testing.assert_almost_equal(entropy[0], expected, decimal=5)
    
    def test_batch_entropy(self):
        """Расчет энтропии для батча"""
        probs = np.array([
            [1.0, 0.0, 0.0],
            [1/3, 1/3, 1/3],
            [0.7, 0.2, 0.1]
        ])
        entropy = calculate_entropy(probs)
        
        assert len(entropy) == 3
        assert entropy[0] < entropy[2] < entropy[1]  # Упорядочение по энтропии
    
    def test_entropy_symmetry(self):
        """Энтропия не зависит от порядка вероятностей"""
        probs1 = np.array([[0.5, 0.3, 0.2]])
        probs2 = np.array([[0.2, 0.5, 0.3]])
        probs3 = np.array([[0.3, 0.2, 0.5]])
        
        entropy1 = calculate_entropy(probs1)
        entropy2 = calculate_entropy(probs2)
        entropy3 = calculate_entropy(probs3)
        
        np.testing.assert_almost_equal(entropy1, entropy2, decimal=5)
        np.testing.assert_almost_equal(entropy2, entropy3, decimal=5)


class TestECECalculation:
    """Тесты для Expected Calibration Error"""
    
    def test_perfect_calibration(self):
        """Идеально калиброванная модель имеет ECE = 0"""
        # Модель предсказывает 0.8 уверенности и правильна в 80% случаев
        confidences = np.array([0.8] * 10)
        accuracies = np.array([True] * 8 + [False] * 2)
        
        ece = calculate_ece(confidences, accuracies, n_bins=5)
        
        # ECE должна быть близка к 0
        assert ece < 0.01
    
    def test_overconfident_model(self):
        """Переуверенная модель имеет высокую ECE"""
        # Модель предсказывает 0.9 уверенности, но правильна только в 50% случаев
        confidences = np.array([0.9] * 10)
        accuracies = np.array([True] * 5 + [False] * 5)
        
        ece = calculate_ece(confidences, accuracies, n_bins=5)
        
        # ECE должна быть высокой (около 0.4)
        assert ece > 0.3
    
    def test_underconfident_model(self):
        """Недоуверенная модель имеет высокую ECE"""
        # Модель предсказывает 0.6 уверенности, но правильна в 90% случаев
        confidences = np.array([0.6] * 10)
        accuracies = np.array([True] * 9 + [False] * 1)
        
        ece = calculate_ece(confidences, accuracies, n_bins=5)
        
        # ECE должна быть заметной (около 0.3)
        assert ece > 0.2
    
    def test_binary_example(self):
        """Тест на примере из документации"""
        # Пример из TowardsDataScience статьи
        samples = np.array([
            [0.78, 0.22],
            [0.36, 0.64],
            [0.08, 0.92],
            [0.58, 0.42],
            [0.49, 0.51],
            [0.85, 0.15],
            [0.30, 0.70],
            [0.63, 0.37],
            [0.17, 0.83]
        ])
        true_labels = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1])
        
        confidences = np.max(samples, axis=1)
        predicted_labels = np.argmax(samples, axis=1)
        accuracies = (predicted_labels == true_labels)
        
        ece = calculate_ece(confidences, accuracies, n_bins=5)
        
        # Ожидаемое значение около 0.104
        np.testing.assert_almost_equal(ece, 0.104, decimal=2)
    
    def test_ece_range(self):
        """ECE должна быть в диапазоне [0, 1]"""
        np.random.seed(42)
        confidences = np.random.uniform(0.5, 1.0, 100)
        accuracies = np.random.choice([True, False], 100)
        
        ece = calculate_ece(confidences, accuracies, n_bins=10)
        
        assert 0.0 <= ece <= 1.0


class TestKSTest:
    """Тесты для Kolmogorov-Smirnov теста"""
    
    def test_identical_distributions(self):
        """Идентичные распределения не должны показывать дрейф"""
        np.random.seed(42)
        entropy1 = np.random.normal(0.5, 0.1, 1000)
        entropy2 = np.random.normal(0.5, 0.1, 1000)
        
        statistic, p_value = ks_test_comparison(entropy1, entropy2)
        
        # p-value должно быть высоким (нет значимого различия)
        assert p_value > 0.05
    
    def test_different_distributions(self):
        """Различные распределения должны показывать дрейф"""
        np.random.seed(42)
        entropy_baseline = np.random.normal(0.5, 0.1, 1000)
        entropy_drifted = np.random.normal(0.8, 0.1, 1000)  # Сдвиг среднего
        
        statistic, p_value = ks_test_comparison(entropy_drifted, entropy_baseline)
        
        # p-value должно быть низким (значимое различие)
        assert p_value < 0.05
        # Статистика должна быть заметной
        assert statistic > 0.3
    
    def test_variance_change(self):
        """Изменение дисперсии также детектируется"""
        np.random.seed(42)
        entropy_baseline = np.random.normal(0.5, 0.1, 1000)
        entropy_volatile = np.random.normal(0.5, 0.3, 1000)  # Увеличенная дисперсия
        
        statistic, p_value = ks_test_comparison(entropy_volatile, entropy_baseline)
        
        # Должно детектировать изменение распределения
        assert p_value < 0.05


class TestIntegration:
    """Интеграционные тесты"""
    
    def test_entropy_ece_relationship(self):
        """Высокая энтропия коррелирует с низкой уверенностью"""
        np.random.seed(42)
        
        # Генерируем предсказания с разной энтропией
        # Низкая энтропия (уверенные предсказания)
        probs_confident = np.random.dirichlet([10, 1, 1], 100)
        # Высокая энтропия (неуверенные предсказания)
        probs_uncertain = np.random.dirichlet([3, 3, 3], 100)
        
        entropy_confident = calculate_entropy(probs_confident)
        entropy_uncertain = calculate_entropy(probs_uncertain)
        
        # Средняя энтропия неуверенных предсказаний должна быть выше
        assert entropy_uncertain.mean() > entropy_confident.mean()
        
        # Уверенность (max prob) должна быть обратно пропорциональна энтропии
        conf_confident = np.max(probs_confident, axis=1)
        conf_uncertain = np.max(probs_uncertain, axis=1)
        
        assert conf_confident.mean() > conf_uncertain.mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
