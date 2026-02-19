"""
Модуль для определения режимов рынка через Hidden Markov Models (HMM).

Согласно задаче 155, используем HMM вместо K-Means для учета временной
последовательности состояний рынка (тренд/флэт/волатильность).
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:
    """
    Детектор режимов рынка на основе Hidden Markov Model.
    
    Использует признаки:
    - Intensity: количество событий в окне
    - Volatility: log std mid_price
    - Spread Z-Score: нормализованный спред
    - OFI: Order Flow Imbalance
    
    Attributes:
        n_components: количество скрытых состояний (режимов)
        model: обученная модель GaussianHMM
        feature_means: средние значения признаков для нормализации
        feature_stds: стандартные отклонения признаков для нормализации
    """
    
    def __init__(self, n_components: int = 3, covariance_type: str = "diag", 
                 n_iter: int = 1000, random_state: int = 42):
        """
        Инициализация детектора режимов.
        
        Args:
            n_components: количество скрытых состояний (режимов)
            covariance_type: тип ковариационной матрицы ("diag", "full", "tied", "spherical")
            n_iter: максимальное количество итераций EM-алгоритма
            random_state: seed для воспроизводимости
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        
        self.feature_means = None
        self.feature_stds = None
        self.is_fitted = False
    
    def _normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Нормализует признаки (Z-score normalization).
        
        Args:
            features: массив признаков формы (n_samples, n_features)
            fit: если True, вычисляет и сохраняет параметры нормализации
        
        Returns:
            нормализованные признаки
        """
        if fit:
            self.feature_means = np.mean(features, axis=0)
            self.feature_stds = np.std(features, axis=0) + 1e-8
        
        if self.feature_means is None or self.feature_stds is None:
            raise ValueError("Feature normalization parameters not fitted")
        
        return (features - self.feature_means) / self.feature_stds
    
    def fit(self, features: np.ndarray, lengths: Optional[np.ndarray] = None) -> 'RegimeDetector':
        """
        Обучает HMM на признаках режима.
        
        Args:
            features: массив признаков формы (n_samples, n_features)
                     где n_features = 4 (intensity, volatility, spread_zscore, ofi)
            lengths: массив длин последовательностей (для нескольких независимых серий)
        
        Returns:
            self
        """
        # Нормализуем признаки
        features_normalized = self._normalize_features(features, fit=True)
        
        # Обучаем HMM
        if lengths is not None:
            self.model.fit(features_normalized, lengths=lengths)
        else:
            self.model.fit(features_normalized)
        
        self.is_fitted = True
        return self
    
    def predict_state(self, features: np.ndarray) -> int:
        """
        Предсказывает текущий режим рынка (online inference).
        
        Args:
            features: вектор признаков формы (n_features,) или (1, n_features)
        
        Returns:
            индекс режима (0, 1, ..., n_components-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Приводим к формату (1, n_features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Нормализуем
        features_normalized = self._normalize_features(features, fit=False)
        
        # Предсказываем состояние
        state = self.model.predict(features_normalized)[0]
        return int(state)
    
    def predict_states(self, features: np.ndarray) -> np.ndarray:
        """
        Предсказывает последовательность режимов для всех семплов.
        
        Args:
            features: массив признаков формы (n_samples, n_features)
        
        Returns:
            массив индексов режимов формы (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features_normalized = self._normalize_features(features, fit=False)
        return self.model.predict(features_normalized)
    
    def get_regime_distribution(self, features: np.ndarray) -> np.ndarray:
        """
        Возвращает распределение режимов в данных.
        
        Args:
            features: массив признаков формы (n_samples, n_features)
        
        Returns:
            массив с количеством семплов в каждом режиме
        """
        states = self.predict_states(features)
        distribution = np.bincount(states, minlength=self.n_components)
        return distribution
    
    def compute_silhouette_score(self, features: np.ndarray) -> float:
        """
        Вычисляет Silhouette Score для оценки качества кластеризации.
        
        Args:
            features: массив признаков формы (n_samples, n_features)
        
        Returns:
            Silhouette Score (от -1 до 1, чем выше - тем лучше)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features_normalized = self._normalize_features(features, fit=False)
        states = self.model.predict(features_normalized)
        
        # Silhouette Score требует минимум 2 кластера
        if len(np.unique(states)) < 2:
            return -1.0
        
        return silhouette_score(features_normalized, states)
    
    def save(self, path: str) -> None:
        """
        Сохраняет параметры HMM в JSON файл.
        
        Args:
            path: путь к файлу для сохранения
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        config = {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "startprob": self.model.startprob_.tolist(),
            "transmat": self.model.transmat_.tolist(),
            "means": self.model.means_.tolist(),
            "covars": self.model.covars_.tolist(),
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist()
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str) -> 'RegimeDetector':
        """
        Загружает параметры HMM из JSON файла.
        
        Args:
            path: путь к файлу с параметрами
        
        Returns:
            self
        """
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.n_components = config["n_components"]
        self.covariance_type = config["covariance_type"]
        self.n_iter = config["n_iter"]
        self.random_state = config["random_state"]
        
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.startprob_ = np.array(config["startprob"])
        self.model.transmat_ = np.array(config["transmat"])
        self.model.means_ = np.array(config["means"])
        self.model.covars_ = np.array(config["covars"])
        
        self.feature_means = np.array(config["feature_means"])
        self.feature_stds = np.array(config["feature_stds"])
        
        self.is_fitted = True
        return self


def optimize_n_components_optuna(features: np.ndarray, 
                                  min_components: int = 2,
                                  max_components: int = 10,
                                  n_trials: int = 20,
                                  criterion: str = "silhouette") -> Tuple[int, float]:
    """
    Оптимизирует количество состояний HMM через Optuna.
    
    Args:
        features: массив признаков формы (n_samples, n_features)
        min_components: минимальное количество состояний
        max_components: максимальное количество состояний
        n_trials: количество попыток оптимизации
        criterion: критерий оптимизации ("silhouette", "aic", "bic")
    
    Returns:
        tuple: (оптимальное количество состояний, лучший score)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Optuna not installed. Install with: pip install optuna")
    
    def objective(trial):
        n_components = trial.suggest_int("n_components", min_components, max_components)
        
        detector = RegimeDetector(n_components=n_components)
        detector.fit(features)
        
        if criterion == "silhouette":
            # Maximize Silhouette Score (от -1 до 1, чем выше - тем лучше)
            score = detector.compute_silhouette_score(features)
            return score
        elif criterion == "aic":
            # Minimize AIC (чем меньше - тем лучше)
            # Optuna максимизирует, поэтому возвращаем -AIC
            aic = detector.model.aic(detector._normalize_features(features, fit=False))
            return -aic
        elif criterion == "bic":
            # Minimize BIC (чем меньше - тем лучше)
            # Optuna максимизирует, поэтому возвращаем -BIC
            bic = detector.model.bic(detector._normalize_features(features, fit=False))
            return -bic
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'silhouette', 'aic', or 'bic'")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_n_components = study.best_params["n_components"]
    best_score = study.best_value
    
    return best_n_components, best_score
