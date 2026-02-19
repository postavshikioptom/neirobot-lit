# Задача №170: Фильтр по ставкам финансирования (Funding Rate Arbitrage Filter)

**Цель**: Защита от потерь на "дорогом" финансировании (Funding) и фильтрация сигналов в периоды аномальных ставок. Бот должен избегать входа в "токсичные" с точки зрения стоимости удержания сделки, особенно перед моментом клиринга.

---

## План реализации для Gemini AI Coder:

### 1. Изменения в [./src/config/types.rs](./src/config/types.rs)
Добавить структуру `FundingFilterConfig` в `StrategyConfig`:
- **`max_funding_rate_bps`**: `u32` (макс. допустимая ставка, например 30 bps = 0.03%).
- **`avoid_settlement_window_ms`**: `u64` (окно перед клирингом, когда вход запрещен при плохой ставке).
- **`min_confidence_to_ignore_funding`**: `f64` (порог Alpha, при котором мы игнорируем фандинг).

Обновить структуру `Ticker` в `[./src/data/types.rs](./src/data/types.rs)`:
- **`funding_rate`**: `f64`.
- **`next_funding_time`**: `u64`.

### 2. Сбор данных (WebSocket & REST)
- **WebSocket ([./src/data/websocket.rs](./src/data/websocket.rs))**:
    - Подписаться на топик `tickers.{SYMBOL}` (category linear).
    - В методе `on_message` парсить JSON: `fundingRate` и `nextFundingTime`. Обновлять состояние `current_funding` и `next_funding` в `BotState` (per-token).
- **REST ([./src/trading/rest-client.rs](./src/trading/rest-client.rs))**:
    - В задаче [061](./docs/000-tasks_list.md) добавить метод `get_tickers(symbol)` для первичной синхронизации при запуске бота (fallback, если WS еще не прислал апдейт).

### 3. Логика в [./src/trading/execution.rs](./src/trading/execution.rs)
Внедрить проверку в `can_execute`:
- **Определение Adverse (вредного) направления**:
  ```rust
  let is_adverse = (funding_rate > 0.0 && side == Side::Buy) || (funding_rate < 0.0 && side == Side::Sell);
  let rate_bps = (funding_rate.abs() * 10000.0) as u32;
  ```
- **Условие блокировки**:
  Если `is_adverse && rate_bps > config.max_funding_rate_bps`:
    1. Проверить время: Если `now + config.avoid_settlement_window_ms > next_funding_time`.
    2. Проверить сигнал: Если `signal.confidence < config.min_confidence_to_ignore_funding`.
    3. Итог: `return false` (блокировка входа).

### 4. Учет в PnL ([./src/trading/position_manager.rs](./src/trading/position_manager.rs))
- Интегрировать списание/начисление фандинга в `realized_pnl` при достижении `next_funding_time` (если позиция открыта), чтобы статистика соответствовала реальному балансу биржи.

### 5. Тестирование в `tests/execution_flow.rs`
- **Mock**: Установить `funding_rate = 0.0005` (50 bps), `next_funding` через 5 минут.
- **Test**: Подать сигнал `Long` (Buy) с `confidence = 0.7` (ниже порога).
- **Expectation**: `can_execute` должен вернуть `false`.

---

## Аргументация (ответы на замечания Grok):
1. **Архитектура данных**: Отказались от нового `bybit_client.rs`. Используем существующий `websocket.rs` для подписки на `tickers` и `rest-client.rs` для `reqwest` вызовов. Это сохраняет чистоту слоев.
2. **Типы данных**: Используем `abs(rate)` для оценки величины и логику `side` для определения вредности. Теперь расчет `rate_bps` корректен.
3. **Bybit Specifics**: Подписка на `tickers` (category linear) — самый быстрый способ получать актуальный фандинг в реальном времени без лишних REST-запросов в горячем цикле.
4. **Alpha vs Cost**: Введен параметр `min_confidence_to_ignore_funding`. Если наша модель уверена в мощном движении (например, на 2%), нет смысла скипать вход из-за фандинга в 0.01%.

**Gemini, твоя задача — сделать так, чтобы бот не "платил за вход" в рынок больше, чем он может там заработать.**