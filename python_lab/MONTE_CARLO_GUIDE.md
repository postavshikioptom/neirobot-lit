# Руководство по Монте-Карло симуляции задержек

## Задача 214: Монте-Карло симуляция вариативности задержек (Latency Perturbation)

Система стресс-тестирования стратегии путем многократного прогона бэктеста с внесением случайного «джиттера» (jitter) в задержки исполнения. Позволяет оценить устойчивость прибыли к нестабильности сети и лагам биржи.

## Быстрый старт

### Базовый запуск с реальными данными (один символ)

```bash
python python_lab/scripts/monte_carlo_backtest.py \
    --symbols BTCUSDT \
    --iterations 100
```

### Запуск с несколькими символами (Задача 213 Integration)

```bash
python python_lab/scripts/monte_carlo_backtest.py \
    --symbols BTCUSDT,ETHUSDT,BNBUSDT \
    --iterations 100
```

### Запуск с реальными параметрами задержек

Если у вас есть файл `execution_quality.csv` из задачи 202:

```bash
python python_lab/scripts/monte_carlo_backtest.py \
    --symbols BTCUSDT \
    --iterations 100 \
    --execution-quality-csv ./bots/BTCUSDT/reports/execution_quality.csv
```

### Запуск с кастомными параметрами

```bash
python python_lab/scripts/monte_carlo_backtest.py \
    --symbols ETHUSDT,BNBUSDT \
    --iterations 200 \
    --initial-balance 5000 \
    --order-size 500 \
    --mean-latency 30 \
    --std-latency 20 \
    --workers 8 \
    --output-dir ./bots/ETHUSDT/reports
```

## Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--symbols` | Список символов через запятую (e.g., BTCUSDT,ETHUSDT) | BTCUSDT |
| `--iterations` | Количество итераций Монте-Карло | 100 |
| `--initial-balance` | Начальный баланс в USD | 1000.0 |
| `--order-size` | Размер ордера в USD | 100.0 |
| `--mean-latency` | Среднее значение задержки (ms) | 20.0 |
| `--std-latency` | Стандартное отклонение задержки (ms) | 15.0 |
| `--execution-quality-csv` | Путь к CSV с реальными задержками | None |
| `--output-dir` | Директория для отчетов | ./bots/SYMBOL/reports/ |
| `--seed` | Базовый random seed | 42 |
| `--workers` | Количество worker-процессов | CPU count |

## Выходные файлы

После запуска создаются следующие файлы в директории `./bots/SYMBOL/reports/`:

1. **monte_carlo_results.csv** - Таблица с результатами всех итераций
   - Колонки: iteration, seed, pnl, final_balance, max_drawdown_pct, total_trades, maker_rate, avg_slippage_bps, p95_latency_ms, mean_latency_ms, std_latency_ms

2. **monte_carlo_pnl.png** - Графики с 4 панелями:
   - Гистограмма распределения PnL
   - Кривая устойчивости (PnL vs 95-й перцентиль задержки)
   - Распределение максимальной просадки
   - Boxplot ключевых метрик

## Интерпретация результатов

### Метрики риска

**PnL at Risk (5th percentile)** - Худший ожидаемый результат при "плохой" сети
- Если < -100 USD: Стратегия хрупкая, требует оптимизации
- Если > 0 USD: Стратегия устойчива даже в худших сценариях

**Reliability Score** - Процент итераций, оставшихся в прибыли
- Если < 50%: Стратегия нестабильна
- Если > 70%: Стратегия надежна

### Кривая устойчивости

График PnL vs 95-й перцентиль задержки показывает:
- **Отрицательный наклон**: Стратегия чувствительна к задержкам (плохо)
- **Горизонтальная линия**: Стратегия устойчива к задержкам (хорошо)
- **Положительный наклон**: Стратегия выигрывает от задержек (редко, но возможно)

### Распределение PnL

"Облако" возможных исходов показывает:
- **Узкое распределение**: Предсказуемая стратегия
- **Широкое распределение**: Высокая вариативность результатов
- **Смещение влево (в минус)**: Стратегия хрупкая
- **Смещение вправо (в плюс)**: Стратегия устойчива

## Архитектура

### Ключевые исправления (Задача 214 v2)

1. **Джиттер применяется ТОЛЬКО к OrderEvent**
   - MarketData остаются неизменными (историческая константа)
   - SignalData остаются неизменными
   - Только OrderEvent получают джиттер в timestamp
   - Это предотвращает нарушение причинно-следственной связи

2. **Загрузка реальных данных из Parquet**
   - Используется `load_backtest_events()` для загрузки из Parquet
   - Fallback на синтетические данные, если Parquet недоступны
   - Данные загружаются из `./bots/SYMBOL/data/raw/`

3. **Абсолютные пути через base_path**
   - Все пути вычисляются относительно базовой директории проекта
   - Отчеты сохраняются в `./bots/SYMBOL/reports/monte_carlo_pnl.png`

### Модуль perturbation.py

Генератор задержек с логнормальным распределением:

```python
from python_lab.src.backtest.perturbation import LatencyGenerator

# Создание генератора
gen = LatencyGenerator(
    mean_ms=20.0,
    std_ms=15.0,
    seed=42,
    execution_quality_csv='./execution_quality.csv'  # опционально
)

# Генерация задержек
latencies = gen.generate(size=100)  # массив из 100 задержек
single_latency = gen.generate_single()  # одна задержка

# Статистика
stats = gen.get_stats(n_samples=10000)
print(f"P95: {stats['p95_ms']:.2f}ms")
```

### Скрипт monte_carlo_backtest.py

Параллельный запуск бэктестов:

1. **Загрузка данных**: Загружаются события из Parquet или синтетические данные
2. **Параллелизация**: Используется `multiprocessing.Pool` для распределения итераций
3. **Worker-функция**: Каждая итерация запускает бэктест с уникальным набором задержек
   - Джиттер применяется только к OrderEvent
   - MarketData и SignalData остаются неизменными
4. **Сбор результатов**: Метрики собираются в DataFrame
5. **Визуализация**: Построение графиков и вывод отчета

## Логнормальное распределение

Почему используется логнормальное распределение для задержек?

1. **Положительность**: Задержки всегда > 0
2. **Длинный правый хвост**: Редкие большие лаги (spike latency)
3. **Реалистичность**: Соответствует реальным сетевым задержкам
4. **Стандарт индустрии**: Используется в HFT для моделирования latency

### Преобразование параметров

Для задания mean и std в миллисекундах используются формулы:

```
mu = ln(mean^2 / sqrt(mean^2 + std^2))
sigma = sqrt(ln(1 + (std/mean)^2))
```

Затем в scipy: `lognorm(s=sigma, scale=exp(mu))`

## Тестирование

Запуск тестов:

```bash
pytest python_lab/tests/test_monte_carlo.py -v
```

Тесты проверяют:
- Воспроизводимость результатов при одинаковом seed
- Положительность всех задержек
- Корректность fallback на дефолтные параметры
- Загрузку параметров из CSV
- Свойства логнормального распределения

## Интеграция с реальными данными

Скрипт автоматически загружает данные из Parquet файлов:

```bash
# Данные должны быть в директории:
./bots/BTCUSDT/data/raw/*.parquet

# Скрипт автоматически найдет и загрузит их
python python_lab/scripts/monte_carlo_backtest.py --symbol BTCUSDT
```

Если Parquet файлы недоступны, скрипт использует синтетические данные для демонстрации.

Структура Parquet файлов должна содержать:
- `timestamp_ms` - временная метка в миллисекундах
- `bid_price_0`, `bid_volume_0` - лучшая цена/объем покупки
- `ask_price_0`, `ask_volume_0` - лучшая цена/объем продажи
- `bid_price_1..50`, `ask_price_1..50` - остальные уровни стакана

## Рекомендации

1. **Количество итераций**: Минимум 100 для статистической значимости, 500+ для production
2. **Параметры задержек**: Используйте реальные данные из execution_quality.csv
3. **Seed**: Фиксируйте seed для воспроизводимости результатов
4. **Workers**: Используйте все доступные CPU cores для ускорения
5. **Интерпретация**: Обращайте внимание на PnL at Risk и Reliability Score

## Примеры использования

### Сравнение двух стратегий

```bash
# Стратегия A (агрессивная)
python python_lab/scripts/monte_carlo_backtest.py \
    --symbol BTCUSDT \
    --iterations 200 \
    --order-size 1000 \
    --output-dir ./reports/strategy_a

# Стратегия B (консервативная)
python python_lab/scripts/monte_carlo_backtest.py \
    --symbol BTCUSDT \
    --iterations 200 \
    --order-size 100 \
    --output-dir ./reports/strategy_b

# Сравните PnL at Risk и Reliability Score
```

### Тестирование при разных условиях сети

```bash
# Хорошая сеть (низкие задержки)
python python_lab/scripts/monte_carlo_backtest.py \
    --mean-latency 10 --std-latency 5 \
    --output-dir ./reports/good_network

# Плохая сеть (высокие задержки)
python python_lab/scripts/monte_carlo_backtest.py \
    --mean-latency 50 --std-latency 30 \
    --output-dir ./reports/bad_network
```

## Troubleshooting

**Проблема**: Слишком долгий запуск
- **Решение**: Уменьшите `--iterations` или `--n-samples`, увеличьте `--workers`

**Проблема**: Out of memory
- **Решение**: Уменьшите `--workers` или `--n-samples`

**Проблема**: Результаты не воспроизводятся
- **Решение**: Убедитесь, что используете одинаковый `--seed`

**Проблема**: Все итерации дают одинаковый результат
- **Решение**: Проверьте, что задержки действительно применяются к событиям

## Связанные задачи

- **Задача 202**: Execution Quality Analysis (источник параметров задержек)
- **Задача 212**: Limit Order Queue Simulation (используется в бэктестере)
- **Задача 213**: Multi-Instrument Support (поддержка нескольких символов)

## Ссылки

- [SciPy lognorm documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)
- [Multiprocessing best practices](https://docs.python.org/3/library/multiprocessing.html)
- [Monte Carlo methods in finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance)
