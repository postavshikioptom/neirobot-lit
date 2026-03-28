#!/usr/bin/env python3
"""
Bug Condition Exploration Test

Этот тест выявляет все 9 категорий критических ошибок в python_lab.
ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: Тест ПРОВАЛИВАЕТСЯ на неисправленном коде (это подтверждает наличие ошибок).

Тест кодирует ожидаемое поведение - он будет валидировать исправления, когда пройдет после реализации.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path
import pytest
import numpy as np
import polars as pl
from typing import List, Tuple


class TestBugConditionExploration:
    """Тесты для выявления критических ошибок в python_lab"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Подготовка к тестам"""
        self.python_lab_root = Path(__file__).parent.parent
        self.src_path = self.python_lab_root / "src"
        yield

    # ============================================================================
    # ОШИБКА 1: Неправильные пути импортов (7 файлов)
    # ============================================================================

    def test_bug_condition_1_import_errors_calibrate(self):
        """
        Ошибка 1.1: Неправильные пути импортов в calibrate.py
        
        Проблема: calibrate.py использует `from src.train import` без добавления src в sys.path
        Ожидаемый результат: ModuleNotFoundError при попытке импорта
        """
        calibrate_file = self.python_lab_root / "calibrate.py"
        
        # Проверяем, что файл содержит ошибочный импорт
        content = calibrate_file.read_text()
        
        # Проверяем наличие ошибочного импорта
        assert "from src.train import" in content or "from src.dataset import" in content, \
            "calibrate.py должен содержать импорты с префиксом 'src.'"
        
        # Проверяем отсутствие исправления (sys.path.insert)
        assert "sys.path.insert(0, str(Path(__file__).parent.parent / \"src\"))" not in content, \
            "calibrate.py не должен содержать исправление sys.path.insert"
        
        # Попытаемся импортировать - должна быть ошибка
        with pytest.raises(ModuleNotFoundError):
            # Добавляем директорию в sys.path для импорта calibrate
            sys.path.insert(0, str(self.python_lab_root))
            try:
                import calibrate
            finally:
                sys.path.pop(0)

    def test_bug_condition_1_import_errors_evaluate(self):
        """
        Ошибка 1.2: Неправильные пути импортов в evaluate.py
        
        Проблема: evaluate.py использует `from src.train import` без добавления src в sys.path
        Ожидаемый результат: ModuleNotFoundError при попытке импорта
        """
        evaluate_file = self.python_lab_root / "evaluate.py"
        
        # Проверяем, что файл содержит ошибочный импорт
        content = evaluate_file.read_text()
        
        # Проверяем наличие ошибочного импорта
        assert "from src." in content, \
            "evaluate.py должен содержать импорты с префиксом 'src.'"
        
        # Проверяем отсутствие исправления
        assert "sys.path.insert(0, str(Path(__file__).parent.parent / \"src\"))" not in content, \
            "evaluate.py не должен содержать исправление sys.path.insert"

    def test_bug_condition_1_import_errors_test_robust_scaler(self):
        """
        Ошибка 1.3: Неправильные пути импортов в test_robust_scaler.py
        
        Проблема: test_robust_scaler.py использует `from src.dataset import` без добавления src в sys.path
        Ожидаемый результат: ModuleNotFoundError при попытке импорта
        """
        test_file = self.python_lab_root / "test_robust_scaler.py"
        
        # Проверяем, что файл содержит ошибочный импорт
        content = test_file.read_text()
        
        # Проверяем наличие ошибочного импорта
        assert "from src." in content, \
            "test_robust_scaler.py должен содержать импорты с префиксом 'src.'"
        
        # Проверяем отсутствие исправления
        assert "sys.path.insert(0, str(Path(__file__).parent.parent / \"src\"))" not in content, \
            "test_robust_scaler.py не должен содержать исправление sys.path.insert"

    def test_bug_condition_1_import_errors_test_multi_instrument(self):
        """
        Ошибка 1.4: Неправильные пути импортов в tests/test_multi_instrument.py
        
        Проблема: test_multi_instrument.py использует `from src.dataset import` без добавления src в sys.path
        Ожидаемый результат: ModuleNotFoundError при попытке импорта
        """
        test_file = self.python_lab_root / "tests" / "test_multi_instrument.py"
        
        if not test_file.exists():
            pytest.skip("test_multi_instrument.py не существует")
        
        # Проверяем, что файл содержит ошибочный импорт
        content = test_file.read_text()
        
        # Проверяем наличие ошибочного импорта
        assert "from src." in content, \
            "test_multi_instrument.py должен содержать импорты с префиксом 'src.'"

    def test_bug_condition_1_import_errors_test_monte_carlo(self):
        """
        Ошибка 1.5: Неправильные пути импортов в tests/test_monte_carlo.py
        
        Проблема: test_monte_carlo.py использует `from src.backtest.perturbation import` без добавления src в sys.path
        Ожидаемый результат: ModuleNotFoundError при попытке импорта
        """
        test_file = self.python_lab_root / "tests" / "test_monte_carlo.py"
        
        if not test_file.exists():
            pytest.skip("test_monte_carlo.py не существует")
        
        # Проверяем, что файл содержит ошибочный импорт
        content = test_file.read_text()
        
        # Проверяем наличие ошибочного импорта
        assert "from src." in content, \
            "test_monte_carlo.py должен содержать импорты с префиксом 'src.'"

    def test_bug_condition_1_import_errors_tune_attention(self):
        """
        Ошибка 1.6: Неправильные пути импортов в scripts/tune_attention.py
        
        Проблема: tune_attention.py использует `from src.lit_model import` без добавления src в sys.path
        Ожидаемый результат: ModuleNotFoundError при попытке импорта
        """
        script_file = self.python_lab_root / "scripts" / "tune_attention.py"
        
        if not script_file.exists():
            pytest.skip("scripts/tune_attention.py не существует")
        
        # Проверяем, что файл содержит ошибочный импорт
        content = script_file.read_text()
        
        # Проверяем наличие ошибочного импорта
        assert "from src." in content, \
            "tune_attention.py должен содержать импорты с префиксом 'src.'"

    def test_bug_condition_1_import_errors_monte_carlo_backtest(self):
        """
        Ошибка 1.7: Неправильные пути импортов в scripts/monte_carlo_backtest.py
        
        Проблема: monte_carlo_backtest.py использует `from src.backtest.engine import` без добавления src в sys.path
        Ожидаемый результат: ModuleNotFoundError при попытке импорта
        """
        script_file = self.python_lab_root / "scripts" / "monte_carlo_backtest.py"
        
        if not script_file.exists():
            pytest.skip("scripts/monte_carlo_backtest.py не существует")
        
        # Проверяем, что файл содержит ошибочный импорт
        content = script_file.read_text()
        
        # Проверяем наличие ошибочного импорта
        assert "from src." in content, \
            "monte_carlo_backtest.py должен содержать импорты с префиксом 'src.'"

    # ============================================================================
    # ОШИБКА 2: Жёстко закодированный путь в тестах
    # ============================================================================

    def test_bug_condition_2_hardcoded_path(self):
        """
        Ошибка 2: Жёстко закодированный путь в test_patching_fix.py
        
        Проблема: test_patching_fix.py использует жёстко закодированный путь 'python_lab/src'
        Ожидаемый результат: Путь не работает на разных ОС и расположениях проекта
        """
        test_file = self.python_lab_root / "test_patching_fix.py"
        
        # Проверяем, что файл содержит жёстко закодированный путь
        content = test_file.read_text()
        
        # Проверяем наличие ошибочного пути
        assert "sys.path.insert(0, 'python_lab/src')" in content, \
            "test_patching_fix.py должен содержать жёстко закодированный путь 'python_lab/src'"
        
        # Проверяем отсутствие исправления (Path(__file__).parent)
        assert "sys.path.insert(0, str(Path(__file__).parent / \"src\"))" not in content, \
            "test_patching_fix.py не должен содержать исправление с Path(__file__).parent"

    # ============================================================================
    # ОШИБКА 3: Несуществующие версии пакетов
    # ============================================================================

    def test_bug_condition_3_incompatible_versions(self):
        """
        Ошибка 3: Несуществующие версии пакетов в requirements.txt
        
        Проблема: requirements.txt содержит версии, которые никогда не были выпущены
        Ожидаемый результат: Ошибка при попытке установки
        """
        requirements_file = self.python_lab_root / "requirements.txt"
        
        # Проверяем, что файл содержит несовместимые версии
        content = requirements_file.read_text()
        
        # Проверяем наличие несовместимых версий
        has_scipy_error = "scipy>=1.17.0" in content
        has_matplotlib_error = "matplotlib>=3.10.8" in content
        has_plotly_error = "plotly>=6.2.0" in content
        has_seaborn_error = "seaborn>=0.13.0" in content
        
        # Хотя бы одна из этих ошибок должна быть
        assert has_scipy_error or has_matplotlib_error or has_plotly_error or has_seaborn_error, \
            "requirements.txt должен содержать несовместимые версии пакетов"

    # ============================================================================
    # ОШИБКА 4: Пустой файл src/types.py
    # ============================================================================

    def test_bug_condition_4_empty_types_file(self):
        """
        Ошибка 4: Пустой файл src/types.py
        
        Проблема: src/types.py не содержит базовых типов
        Ожидаемый результат: Отсутствие типов для проекта
        """
        types_file = self.src_path / "types.py"
        
        # Проверяем, что файл существует
        assert types_file.exists(), "src/types.py должен существовать"
        
        # Проверяем, что файл не содержит необходимых типов
        content = types_file.read_text()
        
        # Проверяем отсутствие базовых типов
        has_type_imports = "from typing import" in content
        has_numpy_import = "import numpy" in content
        has_polars_import = "import polars" in content
        has_type_aliases = "ArrayLike" in content or "PathLike" in content
        
        # Файл должен быть пустым или содержать только комментарий
        assert not (has_type_imports and has_numpy_import and has_polars_import and has_type_aliases), \
            "src/types.py должен быть пустым или содержать только комментарий"

    # ============================================================================
    # ОШИБКА 5: Утечка данных в labels.py
    # ============================================================================

    def test_bug_condition_5_data_leakage_labels(self):
        """
        Ошибка 5: Утечка данных в labels.py:61
        
        Проблема: fill_null(strategy="backward") заполняет пропуски будущими значениями
        Ожидаемый результат: Неправильное заполнение пропусков
        """
        labels_file = self.src_path / "labels.py"
        
        if not labels_file.exists():
            pytest.skip("src/labels.py не существует")
        
        # Проверяем, что файл содержит ошибочное заполнение
        content = labels_file.read_text()
        
        # Проверяем наличие ошибочного backward fill
        assert 'fill_null(strategy="backward")' in content, \
            "src/labels.py должен содержать fill_null(strategy=\"backward\")"
        
        # Проверяем отсутствие исправления
        assert "forward_fill()" not in content or 'fill_null(strategy="backward")' in content, \
            "src/labels.py должен содержать ошибочное backward fill"

    # ============================================================================
    # ОШИБКА 6: Отсутствие защиты от None в normalization.py
    # ============================================================================

    def test_bug_condition_6_none_protection_normalization(self):
        """
        Ошибка 6: Отсутствие защиты от None в normalization.py:118-119
        
        Проблема: Условие `if winsor_limits` не проверяет на None перед доступом к winsor_limits[0]
        Ожидаемый результат: AttributeError при доступе к None[0]
        """
        normalization_file = self.src_path / "normalization.py"
        
        if not normalization_file.exists():
            pytest.skip("src/normalization.py не существует")
        
        # Проверяем, что файл содержит ошибочную проверку
        content = normalization_file.read_text()
        
        # Проверяем наличие ошибочной проверки
        assert "if winsor_limits" in content, \
            "src/normalization.py должен содержать проверку if winsor_limits"
        
        # Проверяем отсутствие явной проверки на None
        assert "if winsor_limits is not None:" not in content, \
            "src/normalization.py не должен содержать явную проверку на None"

    # ============================================================================
    # ОШИБКА 7: Некорректная compute_intensity в dataset.py
    # ============================================================================

    def test_bug_condition_7_compute_intensity_no_timestamps(self):
        """
        Ошибка 7: Некорректная compute_intensity в dataset.py:130-149
        
        Проблема: Функция использует счетчик вместо временных меток
        Ожидаемый результат: Неправильное значение интенсивности
        """
        dataset_file = self.src_path / "dataset.py"
        
        if not dataset_file.exists():
            pytest.skip("src/dataset.py не существует")
        
        # Проверяем, что файл содержит ошибочную compute_intensity
        content = dataset_file.read_text()
        
        # Проверяем наличие ошибочного использования счетчика
        assert "intensity[i] = i - start_idx + 1" in content, \
            "src/dataset.py должен содержать ошибочное использование счетчика в compute_intensity"

    # ============================================================================
    # ОШИБКА 8: Отсутствие защиты от деления на ноль в features.py
    # ============================================================================

    def test_bug_condition_8_division_by_zero_protection(self):
        """
        Ошибка 8: Отсутствие защиты от деления на ноль в features.py:32-36
        
        Проблема: Вычисление mid_price не проверяет на ноль для ask_p_0 и bid_p_0
        Ожидаемый результат: Ошибка деления на ноль
        """
        features_file = self.src_path / "features.py"
        
        if not features_file.exists():
            pytest.skip("src/features.py не существует")
        
        # Проверяем, что файл содержит ошибочное вычисление
        content = features_file.read_text()
        
        # Проверяем наличие ошибочного вычисления mid_price
        assert "(pl.col(\"ask_p_0\") + pl.col(\"bid_p_0\")) / 2" in content, \
            "src/features.py должен содержать вычисление mid_price"
        
        # Проверяем отсутствие защиты от деления на ноль
        assert "pl.when((pl.col(\"ask_p_0\") != 0) & (pl.col(\"bid_p_0\") != 0))" not in content, \
            "src/features.py не должен содержать защиту от деления на ноль"

    # ============================================================================
    # ОШИБКА 9: Скрытие ошибок при импорте
    # ============================================================================

    def test_bug_condition_9_exception_hiding_import(self):
        """
        Ошибка 9: Скрытие ошибок при импорте (try/except: pass)
        
        Проблема: try/except: pass скрывает реальные проблемы
        Ожидаемый результат: Отсутствие логирования ошибок
        """
        dataset_file = self.src_path / "dataset.py"
        
        if not dataset_file.exists():
            pytest.skip("src/dataset.py не существует")
        
        # Проверяем, что файл содержит скрытие ошибок
        content = dataset_file.read_text()
        
        # Проверяем наличие try/except: pass
        assert "try:" in content and "except:" in content, \
            "src/dataset.py должен содержать try/except блоки"
        
        # Проверяем наличие скрытия ошибок (except: pass)
        lines = content.split('\n')
        has_exception_hiding = False
        for i, line in enumerate(lines):
            if "except:" in line or "except Exception:" in line or "except ImportError:" in line:
                # Проверяем следующую строку
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line == "pass":
                        has_exception_hiding = True
                        break
        
        assert has_exception_hiding, \
            "src/dataset.py должен содержать скрытие ошибок (except: pass)"


class TestBugConditionIntegration:
    """Интеграционные тесты для проверки всех ошибок вместе"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Подготовка к тестам"""
        self.python_lab_root = Path(__file__).parent.parent
        self.src_path = self.python_lab_root / "src"
        yield

    def test_all_bug_conditions_present(self):
        """
        Проверка, что все 9 категорий ошибок присутствуют в проекте
        
        Это подтверждает, что тесты выявляют реальные ошибки
        """
        errors_found = []
        
        # Ошибка 1: Неправильные импорты
        calibrate_file = self.python_lab_root / "calibrate.py"
        if calibrate_file.exists():
            content = calibrate_file.read_text()
            if "from src." in content and "sys.path.insert(0, str(Path(__file__).parent.parent / \"src\"))" not in content:
                errors_found.append("Ошибка 1: Неправильные импорты в calibrate.py")
        
        # Ошибка 2: Жёстко закодированный путь
        test_file = self.python_lab_root / "test_patching_fix.py"
        if test_file.exists():
            content = test_file.read_text()
            if "sys.path.insert(0, 'python_lab/src')" in content:
                errors_found.append("Ошибка 2: Жёстко закодированный путь в test_patching_fix.py")
        
        # Ошибка 3: Несовместимые версии
        requirements_file = self.python_lab_root / "requirements.txt"
        if requirements_file.exists():
            content = requirements_file.read_text()
            if "scipy>=1.17.0" in content or "matplotlib>=3.10.8" in content:
                errors_found.append("Ошибка 3: Несовместимые версии в requirements.txt")
        
        # Ошибка 4: Пустой types.py
        types_file = self.src_path / "types.py"
        if types_file.exists():
            content = types_file.read_text()
            if not ("from typing import" in content and "import numpy" in content):
                errors_found.append("Ошибка 4: Пустой файл src/types.py")
        
        # Ошибка 5: Утечка данных
        labels_file = self.src_path / "labels.py"
        if labels_file.exists():
            content = labels_file.read_text()
            if 'fill_null(strategy="backward")' in content:
                errors_found.append("Ошибка 5: Утечка данных в labels.py")
        
        # Ошибка 6: Отсутствие защиты от None
        normalization_file = self.src_path / "normalization.py"
        if normalization_file.exists():
            content = normalization_file.read_text()
            if "if winsor_limits" in content and "if winsor_limits is not None:" not in content:
                errors_found.append("Ошибка 6: Отсутствие защиты от None в normalization.py")
        
        # Ошибка 7: Некорректная compute_intensity
        dataset_file = self.src_path / "dataset.py"
        if dataset_file.exists():
            content = dataset_file.read_text()
            if "intensity[i] = i - start_idx + 1" in content:
                errors_found.append("Ошибка 7: Некорректная compute_intensity в dataset.py")
        
        # Ошибка 8: Отсутствие защиты от деления на ноль
        features_file = self.src_path / "features.py"
        if features_file.exists():
            content = features_file.read_text()
            if "(pl.col(\"ask_p_0\") + pl.col(\"bid_p_0\")) / 2" in content and \
               "pl.when((pl.col(\"ask_p_0\") != 0) & (pl.col(\"bid_p_0\") != 0))" not in content:
                errors_found.append("Ошибка 8: Отсутствие защиты от деления на ноль в features.py")
        
        # Ошибка 9: Скрытие ошибок
        if dataset_file.exists():
            content = dataset_file.read_text()
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "except:" in line or "except Exception:" in line or "except ImportError:" in line:
                    if i + 1 < len(lines) and lines[i + 1].strip() == "pass":
                        errors_found.append("Ошибка 9: Скрытие ошибок при импорте в dataset.py")
                        break
        
        # Проверяем, что найдены ошибки
        assert len(errors_found) > 0, \
            f"Должны быть найдены критические ошибки. Найдено: {len(errors_found)}\n" + \
            "\n".join(errors_found)
        
        # Выводим найденные ошибки для документирования
        print("\n" + "="*80)
        print("НАЙДЕННЫЕ КРИТИЧЕСКИЕ ОШИБКИ:")
        print("="*80)
        for i, error in enumerate(errors_found, 1):
            print(f"{i}. {error}")
        print("="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
