# Python Lab for Neirobot LiT

1. Создать виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Запустить среду для экспериментов:
   ```bash
   jupyter lab
   ```


## Анализ распределения меток

Для анализа распределения классов (Up/Down/Flat) и матрицы переходов используйте скрипт `analyze_labels.py`:

```bash
python scripts/analyze_labels.py --data_path bots/CAKEUSDT/data/raw
```

Скрипт выполняет:
- Анализ распределения классов с вычислением Imbalance Ratio
- Построение матрицы переходов между состояниями
- Проверку консистентности между train и val выборками
- Сохранение метаданных в JSON для использования в train.py
- Генерацию визуализаций (гистограммы и тепловые карты)

Опциональные параметры:
- `--output_dir` - путь для сохранения результатов (по умолчанию: data_path)
- `--consistency_threshold` - порог различия в % для предупреждения (по умолчанию: 5.0)
