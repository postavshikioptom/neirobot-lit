# **Задача №175: Гейт лимита отклонений ордеров (Order Rejection Limit Gate)**

**Цель**: Предотвращение зацикливания бота при критических ошибках (неверные параметры, нехватка маржи, блокировки). Бот должен останавливаться, если количество отклоненных ордеров превышает лимит подряд или внутри скользящего окна.

---

## **План реализации для Gemini AI Coder:**

### **1. Изменения в [./src/config/types.rs](./src/config/types.rs)**
Добавить параметры в `RiskConfig` (изоляция **per-bot**):
- **`max_consecutive_rejections`**: `u32` (лимит идущих подряд ошибок, например 5).
- **`max_total_rejections_in_window`**: `u32` (лимит внутри окна, например 10).
- **`rejection_window_ms`**: `u64` (размер окна, например 60000).
- **`ignored_rejection_codes`**: `Vec<i32>` (коды Bybit, которые не считаются ошибкой, например `34026` — PostOnly reject).

### **2. Состояние в [./src/risk/risk_manager.rs](./src/risk/risk_manager.rs)**
Реализовать эффективный учет отклонений с ленивой очисткой:
- **Поля**: `consecutive_rejections: u32`, `rejection_history: VecDeque<u64>` (таймстампы).
- **Метод `report_rejection()`**: 
    - Добавить `now` в историю.
    - **Lazy Cleanup**: Использовать `while` для удаления устаревших меток из начала `VecDeque` (амортизированная сложность $O(1)$).
    - Увеличить `consecutive_rejections`.
- **Метод `report_success()`**: Сбросить `consecutive_rejections = 0`.

### **3. Интеграция в [./src/trading/order_manager.rs](./src/trading/order_manager.rs)**
Внедрить вызовы в обработку ответов API:
```rust
if resp.ret_code != 0 {
    if !config.ignored_rejection_codes.contains(&resp.ret_code) {
        risk_manager.report_rejection();
        // Включая ошибки Rate Limit (10004/429) из задачи 067
        log::error!("[Order] Rejected: {} - {}", resp.ret_code, resp.ret_msg);
    }
} else {
    risk_manager.report_success();
}
```

### **4. Логика гейта в `check_risk_gates`**
Добавить проверку условий блокировки:
- `consecutive_rejections >= max_consecutive_rejections` **ИЛИ** `rejection_history.len() >= max_total_rejections_in_window`.
- **Action**: Возвращать `Err(RiskError::TooManyRejections)`, инициировать `emergency_market_close` и остановить бота.

### **5. Тестирование в [./tests/risk_gates.rs](./tests/risk_gates.rs)**
- **Consecutive Test**: Симулировать 5 ошибок подряд -> Проверить `bot_active == false`.
- **Window Cleanup Test**: 
    1. Добавить 10 ошибок.
    2. "Перемотать" время (mock) на `window + 1ms`.
    3. Вызвать `report_rejection()`.
    4. Убедиться, что старые ошибки удалены и `len()` истории корректен.
- **Ignore List Test**: Симулировать ошибку `34026` и убедиться, что счетчики не выросли.

---

## **Аргументация (согласовано с Grok):**
1. **Производительность**: Очистка истории перенесена из `check_risk_gates` (горячий цикл перед каждым ордером) в `report_rejection` (только при ошибке), что минимизирует Latency.
2. **Гибкость**: Список `ignored_rejection_codes` позволяет адаптироваться к специфике стратегий (например, частые PostOnly режекты при агрессивном маркет-мейкинге).
3. **Rate Limit Integration**: Ошибки 10004 (Bybit Rate Limit) теперь считаются отклонениями, что предотвращает бан IP-адреса при "зацикливании" неверных запросов.
4. **Sample Variance**: Комбинация `consecutive` (защита от мгновенных сбоев) и `window` (защита от скрытой деградации) обеспечивает полную безопасность исполнения.

**Gemini, твоя задача — научить бота "сдаваться" при серии технических неудач, сохраняя контроль над аккаунтом.**