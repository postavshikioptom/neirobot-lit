# **Задача №177: Расширенный фильтр отклонения цены (Extreme Price Deviation & Fat Finger Protection)**

**Цель**: Эволюция задачи [075](./docs/000-tasks_list.md). Защита от "Fat Finger" ошибок и исполнения в условиях аномальных рыночных зазоров. Бот должен блокировать любой ордер (Limit или Market), цена которого отклоняется от рыночного эталона (Mid/Last/Best) более чем на заданный критический порог.

---

## **План реализации для Gemini AI Coder:**

### **1. Изменения в [./src/config/types.rs](./src/config/types.rs)**
Добавить/расширить параметры в `RiskConfig` (изоляция **per-bot**):
- **`max_price_deviation_bps`**: `u32` (макс. отклонение от эталона, например 500 bps = 5%).
- **`price_reference_source`**: `enum { MidPrice, LastPrice, Both }` (эталон для сравнения).
- **`halt_on_extreme_deviation`**: `bool` (полная остановка бота при срабатывании).

### **2. Логика в [./src/risk/risk_manager.rs](./src/risk/risk_manager.rs)**
Реализовать метод `check_price_sanity`, объединяющий защиту лимитных и маркет-ордеров:
- **Выбор эталона**:
    - **Primary**: `ob.get_mid_price()` (задача [015](./docs/000-tasks_list.md)).
    - **Optional**: `ob.last_trade_price` (из WebSocket-ленты сделок).
    - Если `Both`, отклонение проверяется по обоим значениям — ошибка, если хотя бы одно превышено.
- **Валидация Limit ордеров**:
    - Сравнивать `order_price` с противоположной стороной стакана (`best_ask` для Buy, `best_bid` для Sell).
    - **Long**: `if price > best_ask * (1.0 + limit_bps/10000.0) -> Err`.
- **Валидация Market ордеров**:
    - Использовать `vwap` из задачи [168](./docs/100-trading-strategy-volume-weighted-entry.md) или `mid_price` как ожидаемую цену исполнения.
    - Проверять отклонение ожидаемой цены от эталона (`ref_price`).

### **3. Интеграция в [./src/trading/execution.rs](./src/trading/execution.rs)**
Внедрить вызов в основной пайплайн:
```rust
let sanity_result = risk_manager.check_price_sanity(side, order_price, &order_book);
if let Err(e) = sanity_result {
    log::error!("[RISK] Price Deviation Block: {:?}", e);
    if config.halt_on_extreme_deviation {
        return ExecutionAction::EmergencyStop;
    }
    return ExecutionAction::Skip;
}
```

### **4. Обработка "пустых" эталонов**
- Если `last_trade_price` отсутствует (первый запуск), использовать `MidPrice` как fallback.
- Если стакан не проинициализирован (нет уровней), блокировать любые операции до получения `Snapshot` (задача [020](./docs/000-tasks_list.md)).

### **5. Тестирование в [./tests/risk_gates.rs](./tests/risk_gates.rs)**
- **Limit Test**: `Mid=100`, `Limit Buy=110` (при лимите 5%). Убедиться в блокировке.
- **Market Test**: Симулировать "тонкий" стакан, где покупка 100 лотов дает `VWAP=110` при `Mid=100`. Убедиться в блокировке.
- **Both Ref Test**: Проверить ситуацию, когда `Mid` в норме, но `LastTrade` на 10% выше (прострел) — фильтр должен сработать.

---

## **Аргументация (согласовано с Grok):**
1. **Слияние с 075**: Эта задача полностью заменяет и расширяет базовый чек 075, добавляя учет направления сделки и поддержку маркет-ордеров.
2. **Best Bid/Ask**: Для лимитных ордеров сравнение с `best_bid/ask` точнее, чем с `mid`, так как оно учитывает реальную возможность исполнения "внутри" или "снаружи" спреда.
3. **Market Sanity**: Даже если ордер маркетный (без цены), мы обязаны оценить его `Expected Price` (через VWAP из 168), иначе Flash Crash съест баланс мгновенно.
4. **Reference Both**: Режим `Both` защищает от манипуляций в стакане, когда `Mid` искусственно сдвинут фиктивными ордерами, но реальные сделки (`Last`) проходят по другой цене.

**Gemini, твоя задача — внедрить "предохранитель", который сработает быстрее, чем рынок успеет забрать твои деньги из-за ошибки в расчетах или внезапной паники.**