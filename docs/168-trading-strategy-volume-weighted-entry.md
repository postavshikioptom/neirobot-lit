# Задача №168: Вход с учетом средневзвешенного объема (Volume-Weighted Entry)

**Цель**: Оптимизация цены входа для минимизации **Market Impact** и контроля проскальзывания (**Slippage**). Бот должен рассчитывать VWAP (Volume-Weighted Average Price) исполнения по стакану в реальном времени и динамически выбирать стратегию входа.

---

## План реализации для Gemini AI Coder:

### 1. Изменения в [./src/config/types.rs](./src/config/types.rs)
Добавить параметры в `ExecutionConfig` (per-bot):
- **`entry_style`**: `enum { AggressiveMarket, PassiveLimit, ChaseBest }`.
- **`max_entry_slippage_bps`**: `u32` (макс. отклонение VWAP от Mid Price).
- **`entry_participation_ratio`**: `f64` (лимит объема: не более X% от доступного на первых N уровнях).
- **`slicing_enabled`**: `bool` (разбиение крупного ордера на части во времени).

### 2. Оптимизация в [./src/data/orderbook.rs](./src/data/orderbook.rs)
Для минимизации задержек (Latency) реализовать быстрый расчет VWAP:
- **Кеширование**: Добавить поля `cum_vol: Vec<f64>` и `cum_price_vol: Vec<f64>` в структуру `OrderBook`. Обновлять их инкрементально в `apply_update` (задача [078](./docs/000-tasks_list.md)).
- **Метод `get_execution_vwap(side, size)`**:
    - Использовать бинарный поиск по `cum_vol` для нахождения нужной глубины стакана ($O(\log N)$).
    - Формула: $VWAP = \frac{\sum (Price_i \cdot Qty_i)}{TotalQty}$.
    - Если `size` больше всей доступной ликвидности в кеше (например, топ-20 уровней), возвращать `None`.

### 3. Логика в [./src/trading/execution.rs](./src/trading/execution.rs)
Модифицировать процесс генерации сигнала:
- **Direction-aware Slippage**:
    - **Long**: `if vwap > mid * (1.0 + max_bps / 10000.0)` -> Slippage High.
    - **Short**: `if vwap < mid * (1.0 - max_bps / 10000.0)` -> Slippage High.
- **Decision Logic**:
    - Если Slippage в норме — `AggressiveMarket` (Bybit Market Order автоматически делает Sweep).
    - Если Slippage превышен — переключиться на `ChaseBest` (выставить лимит на Best Bid/Ask) или `PassiveLimit` (с отступом из задачи [164](./docs/000-tasks_list.md)).
- **Large Order Slicing**: Если `size > participation_limit`, не выставлять "сетку", а поместить остаток в очередь исполнения (`execution_queue`) для постепенного набора (TWAP-подобно).

### 4. Интеграция и защита
- **Adverse Selection**: Перед выполнением Sweep (AggressiveMarket) вызвать `AdversarialDetector` из задачи [165](./docs/000-tasks_list.md). Если поток токсичен — отменить вход.
- **Risk Gate**: Проверить `max_position_size` из задачи [042](./docs/000-tasks_list.md) перед расчетом VWAP.

### 5. Тестирование в `tests/orderbook_integration.rs`
- Создать мок стакана с известными объемами (например, 10 лотов по 100, 10 лотов по 101).
- Проверить, что для `size=15` расчет VWAP выдает ровно 100.33.
- Валидировать переключение стратегий при изменении `max_entry_slippage_bps`.

---

## Аргументация (ответы на замечания Grok):
1. **Latency (O(log N))**: Простой перебор уровней ($O(N)$) слишком медленен для HFT. Предвычисленные кумулятивные суммы позволяют мгновенно оценивать проскальзывание даже для глубоких ордеров.
2. **Direction-aware**: Теперь явно разделены условия для Buy и Sell, так как "плохое" проскальзывание для них зеркально.
3. **No Grid Orders**: Согласен, создание сетки лимитных ордеров усложняет управление позицией и может привести к "зависанию" части объема. Слайсинг (разбиение во времени) — более чистое решение.
4. **Bybit Native**: Bybit Market Order по определению является Sweep-ордером. Наша задача — не "реализовать Sweep", а **предотвратить** его, если он слишком дорогой.

**Gemini, твоя задача — сделать вход в позицию "невидимым" для рынка, сохраняя при этом математическое преимущество (Alpha).**