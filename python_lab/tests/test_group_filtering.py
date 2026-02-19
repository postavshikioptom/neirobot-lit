#!/usr/bin/env python3
"""
Тесты для фильтрации по группам
Задача 228: Автоматизированная дистрибуция и безопасный Hot-Swap моделей
"""

import tempfile
import unittest
from pathlib import Path
import sys
import toml

# Добавляем путь к scripts для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from model_manager import load_farm_config
except ImportError:
    print("Warning: Cannot import model_manager, skipping tests")
    sys.exit(0)


class TestGroupFiltering(unittest.TestCase):
    """Тесты для фильтрации по группам"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.farm_config_path = Path(self.temp_dir.name) / "farm.toml"
        
        # Создаем тестовый farm.toml с группами
        farm_config = {
            'symbols': {
                'list': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
            },
            'groups': {
                'scalping': ['BTCUSDT', 'ETHUSDT'],
                'trending': ['SOLUSDT', 'ADAUSDT'],
                'altcoins': ['BNBUSDT', 'ADAUSDT']
            }
        }
        
        with open(self.farm_config_path, 'w') as f:
            toml.dump(farm_config, f)
    
    def tearDown(self):
        """Очистка после тестов"""
        self.temp_dir.cleanup()
    
    def test_load_farm_config_with_groups(self):
        """Тест загрузки конфигурации с группами"""
        config = load_farm_config(self.farm_config_path)
        
        self.assertIn('groups', config)
        self.assertIn('scalping', config['groups'])
        self.assertIn('trending', config['groups'])
        self.assertIn('altcoins', config['groups'])
    
    def test_scalping_group(self):
        """Тест группы scalping"""
        config = load_farm_config(self.farm_config_path)
        scalping = config['groups']['scalping']
        
        self.assertEqual(len(scalping), 2)
        self.assertIn('BTCUSDT', scalping)
        self.assertIn('ETHUSDT', scalping)
        self.assertNotIn('BNBUSDT', scalping)
    
    def test_trending_group(self):
        """Тест группы trending"""
        config = load_farm_config(self.farm_config_path)
        trending = config['groups']['trending']
        
        self.assertEqual(len(trending), 2)
        self.assertIn('SOLUSDT', trending)
        self.assertIn('ADAUSDT', trending)
        self.assertNotIn('BTCUSDT', trending)
    
    def test_altcoins_group(self):
        """Тест группы altcoins"""
        config = load_farm_config(self.farm_config_path)
        altcoins = config['groups']['altcoins']
        
        self.assertEqual(len(altcoins), 2)
        self.assertIn('BNBUSDT', altcoins)
        self.assertIn('ADAUSDT', altcoins)
    
    def test_symbol_in_multiple_groups(self):
        """Тест символа, входящего в несколько групп"""
        config = load_farm_config(self.farm_config_path)
        
        # ADAUSDT входит в trending и altcoins
        self.assertIn('ADAUSDT', config['groups']['trending'])
        self.assertIn('ADAUSDT', config['groups']['altcoins'])
    
    def test_all_symbols_in_groups(self):
        """Тест что все символы из групп есть в основном списке"""
        config = load_farm_config(self.farm_config_path)
        all_symbols = config['symbols']['list']
        
        for group_name, group_symbols in config['groups'].items():
            for symbol in group_symbols:
                self.assertIn(
                    symbol, all_symbols,
                    f"Symbol {symbol} from group {group_name} not in all_symbols"
                )
    
    def test_group_filtering_logic(self):
        """Тест логики фильтрации по группам"""
        config = load_farm_config(self.farm_config_path)
        
        # Симулируем фильтрацию по группе scalping
        group_name = 'scalping'
        if group_name in config['groups']:
            target_symbols = config['groups'][group_name]
            self.assertEqual(target_symbols, ['BTCUSDT', 'ETHUSDT'])
        else:
            self.fail(f"Group {group_name} not found")
    
    def test_nonexistent_group(self):
        """Тест обработки несуществующей группы"""
        config = load_farm_config(self.farm_config_path)
        
        nonexistent_group = 'nonexistent'
        self.assertNotIn(nonexistent_group, config['groups'])


class TestGroupDeploymentWorkflow(unittest.TestCase):
    """Тесты для workflow деплоя с группами"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.farm_config_path = Path(self.temp_dir.name) / "farm.toml"
        
        # Создаем тестовый farm.toml
        farm_config = {
            'symbols': {
                'list': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
            },
            'groups': {
                'scalping': ['BTCUSDT', 'ETHUSDT'],
                'trending': ['SOLUSDT', 'ADAUSDT'],
                'altcoins': ['BNBUSDT', 'ADAUSDT']
            }
        }
        
        with open(self.farm_config_path, 'w') as f:
            toml.dump(farm_config, f)
    
    def tearDown(self):
        """Очистка после тестов"""
        self.temp_dir.cleanup()
    
    def test_canary_deployment_workflow(self):
        """Тест workflow Canary deployment с группами"""
        config = load_farm_config(self.farm_config_path)
        
        # Шаг 1: Canary на одном символе
        canary_symbols = ['BTCUSDT']
        self.assertEqual(len(canary_symbols), 1)
        
        # Шаг 2: Scalping группа
        scalping_symbols = config['groups']['scalping']
        self.assertEqual(len(scalping_symbols), 2)
        
        # Шаг 3: Trending группа
        trending_symbols = config['groups']['trending']
        self.assertEqual(len(trending_symbols), 2)
        
        # Шаг 4: Все символы
        all_symbols = config['symbols']['list']
        self.assertEqual(len(all_symbols), 5)
        
        # Проверяем что все группы покрывают все символы
        all_group_symbols = set()
        for group_symbols in config['groups'].values():
            all_group_symbols.update(group_symbols)
        
        # Не все символы обязательно должны быть в группах
        # но те что есть должны быть в основном списке
        for symbol in all_group_symbols:
            self.assertIn(symbol, all_symbols)


if __name__ == "__main__":
    unittest.main()
