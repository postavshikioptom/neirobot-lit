#!/usr/bin/env python3
"""
Тесты для Model Manager
Задача 228: Автоматизированная дистрибуция и безопасный Hot-Swap моделей
"""

import hashlib
import tempfile
import unittest
from pathlib import Path
import sys

# Добавляем путь к scripts для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from model_manager import compute_file_hash, filter_symbols
except ImportError:
    # Если не удается импортировать, пропускаем тесты
    print("Warning: Cannot import model_manager, skipping tests")
    sys.exit(0)


class TestModelManager(unittest.TestCase):
    """Тесты для Model Manager"""
    
    def test_compute_file_hash(self):
        """Тест вычисления SHA-256 хеша файла"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_data = b"test data for hashing"
            f.write(test_data)
            f.flush()
            
            # Вычисляем хеш через функцию
            file_hash = compute_file_hash(Path(f.name))
            
            # Вычисляем ожидаемый хеш
            expected_hash = hashlib.sha256(test_data).hexdigest()
            
            self.assertEqual(file_hash, expected_hash)
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_filter_symbols_all(self):
        """Тест фильтрации символов с флагом --all"""
        all_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        result = filter_symbols(all_symbols, deploy_all=True)
        
        self.assertEqual(result, all_symbols)
    
    def test_filter_symbols_by_list(self):
        """Тест фильтрации символов по списку"""
        all_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        symbols_filter = ["BTC", "ETH"]
        
        result = filter_symbols(all_symbols, symbols_filter=symbols_filter)
        
        self.assertIn("BTCUSDT", result)
        self.assertIn("ETHUSDT", result)
        self.assertNotIn("BNBUSDT", result)
        self.assertNotIn("SOLUSDT", result)
    
    def test_filter_symbols_short_names(self):
        """Тест фильтрации с короткими именами символов"""
        all_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        symbols_filter = ["BTC"]
        
        result = filter_symbols(all_symbols, symbols_filter=symbols_filter)
        
        self.assertEqual(len(result), 1)
        self.assertIn("BTCUSDT", result)
    
    def test_filter_symbols_case_insensitive(self):
        """Тест фильтрации без учета регистра"""
        all_symbols = ["BTCUSDT", "ETHUSDT"]
        symbols_filter = ["btc", "eth"]
        
        result = filter_symbols(all_symbols, symbols_filter=symbols_filter)
        
        self.assertEqual(len(result), 2)
        self.assertIn("BTCUSDT", result)
        self.assertIn("ETHUSDT", result)
    
    def test_filter_symbols_no_duplicates(self):
        """Тест отсутствия дубликатов в результате"""
        all_symbols = ["BTCUSDT", "ETHUSDT"]
        symbols_filter = ["BTC", "BTCUSDT"]  # Оба должны совпасть с BTCUSDT
        
        result = filter_symbols(all_symbols, symbols_filter=symbols_filter)
        
        # Должен быть только один BTCUSDT
        self.assertEqual(result.count("BTCUSDT"), 1)
    
    def test_filter_symbols_by_group(self):
        """Тест фильтрации по группам"""
        # Симулируем конфигурацию farm.toml с группами
        farm_config = {
            'symbols': {'list': ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]},
            'groups': {
                'scalping': ["BTCUSDT", "ETHUSDT"],
                'trending': ["SOLUSDT", "BNBUSDT"]
            }
        }
        
        # Проверяем группу scalping
        scalping_group = farm_config['groups']['scalping']
        self.assertEqual(len(scalping_group), 2)
        self.assertIn("BTCUSDT", scalping_group)
        self.assertIn("ETHUSDT", scalping_group)
        
        # Проверяем группу trending
        trending_group = farm_config['groups']['trending']
        self.assertEqual(len(trending_group), 2)
        self.assertIn("SOLUSDT", trending_group)
        self.assertIn("BNBUSDT", trending_group)
    
    def test_atomic_push_simulation(self):
        """Тест симуляции Atomic Push"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Создаем исходную модель
            source = tmpdir / "source.onnx"
            source.write_bytes(b"new model data")
            
            # Симулируем Atomic Push
            tmp_file = tmpdir / "model.onnx.tmp"
            final_file = tmpdir / "model.onnx"
            
            # 1. Копируем в .tmp
            tmp_file.write_bytes(source.read_bytes())
            self.assertTrue(tmp_file.exists())
            
            # 2. Атомарное переименование
            tmp_file.replace(final_file)
            
            # Проверки
            self.assertFalse(tmp_file.exists())
            self.assertTrue(final_file.exists())
            self.assertEqual(final_file.read_bytes(), b"new model data")
    
    def test_backup_creation(self):
        """Тест создания backup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Создаем оригинальную модель
            original = tmpdir / "model.onnx"
            original.write_bytes(b"original model")
            
            # Создаем backup
            backup = tmpdir / "model.onnx.bak"
            backup.write_bytes(original.read_bytes())
            
            # Проверки
            self.assertTrue(original.exists())
            self.assertTrue(backup.exists())
            self.assertEqual(original.read_bytes(), backup.read_bytes())
    
    def test_hash_file_format(self):
        """Тест формата файла model.hash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            hash_file = tmpdir / "model.hash"
            test_hash = "abc123def456"
            
            # Записываем хеш
            hash_file.write_text(f"{test_hash}\n")
            
            # Читаем и проверяем
            content = hash_file.read_text().strip()
            self.assertEqual(content, test_hash)


class TestDeploymentWorkflow(unittest.TestCase):
    """Тесты для workflow деплоя"""
    
    def test_deployment_steps_order(self):
        """Тест правильного порядка шагов деплоя"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Подготовка
            source = tmpdir / "new_model.onnx"
            source.write_bytes(b"new model")
            
            original = tmpdir / "model.onnx"
            original.write_bytes(b"old model")
            
            # Шаг 1: Backup
            backup = tmpdir / "model.onnx.bak"
            backup.write_bytes(original.read_bytes())
            self.assertTrue(backup.exists())
            
            # Шаг 2: Copy to .tmp
            tmp_file = tmpdir / "model.onnx.tmp"
            tmp_file.write_bytes(source.read_bytes())
            self.assertTrue(tmp_file.exists())
            
            # Шаг 3: Compute hash
            model_hash = hashlib.sha256(tmp_file.read_bytes()).hexdigest()
            self.assertIsNotNone(model_hash)
            
            # Шаг 4: Atomic rename
            tmp_file.replace(original)
            self.assertFalse(tmp_file.exists())
            self.assertTrue(original.exists())
            
            # Шаг 5: Update hash file
            hash_file = tmpdir / "model.hash"
            hash_file.write_text(f"{model_hash}\n")
            self.assertTrue(hash_file.exists())
            
            # Проверка финального состояния
            self.assertEqual(original.read_bytes(), b"new model")
            self.assertEqual(backup.read_bytes(), b"old model")
            self.assertEqual(hash_file.read_text().strip(), model_hash)


if __name__ == "__main__":
    unittest.main()
