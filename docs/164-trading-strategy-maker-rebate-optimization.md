# Задача №164: Оптимизация Maker-рибейтов (Maker Rebate Optimization)

**Цель**: Трансформация бота из "потребителя ликвидности" в "поставщика", минимизация издержек за счет возврата комиссий (рибейтов). Бот должен агрессивно бороться за очередь в стакане, используя флаг `Post-Only` и динамический ре-пеггинг, сохраняя при этом гибкость для входа по рынку при сверхвысокой уверенности модели.

---

## План реализации для Gemini AI Coder:

### 1. Изменения в [./src/config/types.rs](./src/config/types.rs)
Добавить параметры в `ExecutionConfig` и `OrderConfig`:
- **`force_taker_confidence`**: `f64` (порог уверенности, например 0.95, для немедленного Market-ордера).
- **`maker_offset_step_ticks`**: `u32` (базовый отступ в тиках от Best Bid/Ask).
- **`max_post_only_rejects`**: `u32` (лимит отклонений Post-Only до переключения в Taker-режим или экспоненциального увеличения отступа).
- **`repeg_threshold_ticks`**: `u32` (дистанция от цены ордера до Best Price, триггер для переустановки).

### 2. Логика принятия решения в [./src/trading/execution.rs](./src/trading/execution.rs)
Модифицировать логику открытия позиции:
- **Safety Valve (Force-Taker)**: Если `signal.confidence > config.force_taker_confidence` и сигнал свежий (< 50мс), отправлять `MarketOrder` для гарантированного входа.
- **Liquidity Check**: Перед отправкой Maker-ордера вызвать `get_liquidity_at_depth` (task [162](./docs/000-tasks_list.md)). Если объем на топ-уровнях < `order_size * 3`, увеличить отступ или пропустить вход (защита от Adverse Selection).
- **Initial Placement**: Отправлять `LimitOrder` с флагом `PostOnly` (task [069](./docs/000-tasks_list.md)) и рассчитанным `offset`.

### 3. Управление жизненным циклом в [./src/trading/order_manager.rs](./src/trading/order_manager.rs)
Перенести логику сопровождения Maker-ордера сюда:
- **Обработка Post-Only Rejects**: При ошибке `PostOnlyOrderWouldExecute` инкрементировать счетчик режектов. Если `count < max_post_only_rejects`, повторить с `offset * 1.5`, иначе — вход по рынку (Graceful Degradation).
- **Dynamic Re-pegging**: В цикле мониторинга (task [108](./docs/000-tasks_list.md)) проверять актуальность цены. Если `abs(order_price - best_price) > repeg_threshold_ticks`, выполнять `amend_order` на новый уровень.
- **Rebate Timeout**: Если ордер не исполнен за `rebate_wait_timeout_ms`, отменить его или "ударить" по рынку.

### 4. Расчет PnL в [./src/trading/position_manager.rs](./src/trading/position_manager.rs)
- **Fee Integration**: Обновить метод обработки филлов (task [064](./docs/000-tasks_list.md)). Использовать поле `fee` из WebSocket-сообщения биржи.
- **Net PnL**: `realized_pnl += (qty * price * sign) + fee`. Поскольку для Maker-сделок `fee` будет отрицательным (рибейт), это автоматически увеличит профит.

### 5. Метрики в [./src/risk/risk_manager.rs](./src/risk/risk_manager.rs)
- Вместо жесткого стопа внедрить **Soft Limit**: если `current_taker_ratio > 0.2`, принудительно увеличивать `maker_offset_step_ticks` для более глубокого захода в стакан, вместо остановки торгов.

---

## Аргументация и проверка:
1. **Почему `order_manager.rs`?**: Ре-пеггинг и работа с таймаутами — это типичный жизненный цикл ордера. `execution.rs` должен оставаться "чистым" местом принятия стратегических решений.
2. **Защита от Adverse Selection**: Вход лимитным ордером в пустой стакан (низкая ликвидность перед нами) — это риск немедленного "пробития" и убытка. Проверка объема из задачи [162](./docs/000-tasks_list.md) обязательна.
3. **Graceful Degradation**: Постоянные режекты Post-Only на узком спреде могут оставить бота без позиции в сильном движении. Фоллбек в Taker после N попыток — необходимый компромисс.

**Gemini, твоя задача — обеспечить "умный" пассивный вход, который не боится пропустить сделку, если она невыгодна по комиссии, но умеет быстро реагировать на сверхприбыльные сигналы.**