# Задача 205: Ежедневные PnL-отчеты

## Описание

Система автоматизированной ежедневной отчетности, которая объединяет данные об исполнении (Rust-логи) с реальным состоянием счета на бирже (Bybit REST). Отчет содержит ключевые риск-метрики для оценки качества работы бота.

## Компоненты

### 1. Rust-ядро (Data Source Sync)

**Файлы:**
- `src/trading/rest_client.rs` - Структуры WalletBalance с полями totalEquity, totalAvailableBalance, totalMarginBalance
- `src/utils/logger.rs` - Структура `EquityLog` и функции логирования
- `src/bin/run-bot.rs` - Асинхронная задача для синхронизации баланса

**Функционал:**
- Периодический запрос баланса к эндпоинту Bybit `/v5/account/wallet-balance` (каждые 3 минуты)
- Логирование данных в `bots/SYMBOL/logs/equity.csv`
- Логирование данных о slippage в `bots/SYMBOL/logs/slippage.csv`
- Асинхронная задача не блокирует основной цикл WebSocket

**Колонки equity.csv:**
- `timestamp` - Временная метка в миллисекундах
- `rest_equity` - Общий equity с биржи (totalEquity)
- `available_balance` - Доступный баланс (totalAvailableBalance)
- `rest_margin` - Используемая маржа (totalMarginBalance)
- `local_unrealized_pnl` - Нереализованный PnL на основе стакана
- `total_pnl_delta` - Изменение PnL

**Колонки slippage.csv:**
- `timestamp_utc` - Временная метка в миллисекундах
- `signal_price` - Mid price в момент генерации сигнала
- `fill_price` - Средневзвешенная цена исполнения
- `slippage_bps` - Проскальзывание в базисных пунктах
- `latency_ms` - Задержка от сигнала до исполнения
- `spread_bps` - Спред в момент исполнения

### 2. Python Lab (Advanced Analytics)

**Файл:** `python_lab/scripts/daily_report.py`

**Использование:**

```bash
# Генерировать отчет на сегодня
python python_lab/scripts/daily_report.py --symbol BTCUSDT

# Генерировать отчет на конкретную дату
python python_lab/scripts/daily_report.py --symbol BTCUSDT --date 20250215

# Указать корневую директорию ботов
python python_lab/scripts/daily_report.py --symbol BTCUSDT --bot-root /path/to/bots
```

**Выходные файлы:**
- `bots/SYMBOL/reports/daily_YYYYMMDD.json` - JSON отчет для UI/Dashboards
- `bots/SYMBOL/reports/daily_YYYYMMDD.md` - Markdown отчет для человека

## Метрики

### Net PnL
Чистая прибыль за период, рассчитанная как разница между конечным и начальным equity.

```
Net PnL = Final Equity - Initial Equity
```

### Sharpe Ratio
Коэффициент Шарпа - риск-скорректированная доходность. Показывает, сколько единиц доходности приходится на единицу риска.

```
Sharpe = (Mean Return - Risk Free Rate) / Std Dev * sqrt(252)
```

Где 252 - количество торговых дней в году.

### Max Drawdown
Максимальная просадка - наибольшее падение equity от пика до дна.

```
Max DD = Min((Equity - Running Max) / Running Max) * 100%
```

### Calmar Ratio
Отношение годовой доходности к максимальной просадке. Показывает, сколько доходности приходится на единицу просадки.

```
Calmar = Annual Return / |Max Drawdown|
```

### Win Rate
Процент прибыльных сделок. Рассчитывается на основе realized_pnl из trades.csv (только закрытые позиции).

```
Win Rate = (Trades with realized_pnl > 0 / Total Trades with realized_pnl) * 100%
```

### Slippage Leakage
Общие потери на проскальзывании в базисных пунктах. Суммирует все значения `slippage_bps` из `slippage.csv`.

```
Slippage Leakage = Sum(slippage_bps)
```

## Примеры

### Пример JSON отчета

```json
{
  "symbol": "BTCUSDT",
  "date": "20250215",
  "timestamp": "2025-02-15T12:00:00.000000",
  "metrics": {
    "net_pnl": 1250.50,
    "sharpe_ratio": 1.2345,
    "max_drawdown": -5.67,
    "calmar_ratio": 2.1876,
    "win_rate": 65.43,
    "slippage_leakage": -125.50
  }
}
```

### Пример Markdown отчета

```markdown
# Daily PnL Report - BTCUSDT

**Date:** 2025-02-15  
**Generated:** 2025-02-15T12:00:00.000000

## Performance Metrics

| Metric | Value |
|--------|-------|
| Net PnL | $1250.50 |
| Sharpe Ratio | 1.2345 |
| Max Drawdown | -5.67% |
| Calmar Ratio | 2.1876 |
| Win Rate | 65.43% |
| Slippage Leakage | -125.50 bps |

...
```

## Интеграция с Cron

Для автоматического запуска отчетов каждый день в 00:00 UTC:

```bash
# Добавить в crontab
0 0 * * * cd /path/to/neirobot-lit && python python_lab/scripts/daily_report.py --symbol BTCUSDT
```

## Требования

- Python 3.8+
- polars >= 1.9.0
- numpy >= 2.0.0

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Neirobot LiT Bot                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Rust-ядро (run-bot.rs)                               │  │
│  │                                                      │  │
│  │ ┌────────────────────────────────────────────────┐  │  │
│  │ │ Асинхронная задача (bg_handle)                │  │  │
│  │ │ - Каждые 3 минуты запрашивает баланс          │  │  │
│  │ │ - Отправляет в канал equity_tx                │  │  │
│  │ └────────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │ ┌────────────────────────────────────────────────┐  │  │
│  │ │ spawn_equity_logger()                          │  │  │
│  │ │ - Получает EquityLog из канала                │  │  │
│  │ │ - Записывает в equity.csv                     │  │  │
│  │ └────────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │ ┌────────────────────────────────────────────────┐  │  │
│  │ │ log_trade_execution()                          │  │  │
│  │ │ - Логирует slippage данные                    │  │  │
│  │ │ - Записывает в slippage.csv                   │  │  │
│  │ └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  bots/SYMBOL/logs/equity.csv                               │
│  bots/SYMBOL/logs/trades.csv (из CsvTradeLogger)           │
│  bots/SYMBOL/logs/slippage.csv                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Python Lab (daily_report.py)                   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ load_equity_data() + load_trades_data() +            │  │
│  │ load_slippage_data()                                 │  │
│  │ - Читает CSV файлы                                  │  │
│  │ - Фильтрует по дате                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Расчет метрик                                        │  │
│  │ - calculate_net_pnl()                               │  │
│  │ - calculate_sharpe_ratio()                          │  │
│  │ - calculate_max_drawdown()                          │  │
│  │ - calculate_calmar_ratio()                          │  │
│  │ - calculate_win_rate() (из trades.csv)              │  │
│  │ - calculate_slippage_leakage() (из slippage.csv)    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Генерация отчетов                                    │  │
│  │ - generate_json_report()                            │  │
│  │ - generate_markdown_report()                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  bots/SYMBOL/reports/daily_YYYYMMDD.json                   │
│  bots/SYMBOL/reports/daily_YYYYMMDD.md                     │
└─────────────────────────────────────────────────────────────┘
```

## Примечания

1. **Асинхронность**: Синхронизация баланса выполняется в отдельной асинхронной задаче в фоновом рантайме, что не блокирует основной цикл обработки WebSocket-событий.

2. **Точность**: Используется REST API баланс как "якорь истины", а не оценка по `mid_price`, что избегает накопления ошибок округления и комиссий.

3. **Независимость**: Python скрипт полностью независим и может запускаться вручную, по расписанию (cron) или через вызов из Rust при `on_exit`.

4. **Масштабируемость**: Система поддерживает несколько ботов одновременно - каждый бот имеет свою директорию `bots/SYMBOL` с собственными логами и отчетами.

5. **Разделение данных**: 
   - `trades.csv` - Полная история сделок из CsvTradeLogger (с realized_pnl)
   - `slippage.csv` - Данные о проскальзывании для анализа качества исполнения
   - `equity.csv` - Данные о балансе для расчета метрик производительности
