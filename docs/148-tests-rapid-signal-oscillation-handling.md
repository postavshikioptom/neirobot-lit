# Задача 148: Testing Rapid Signal Oscillation and Race Conditions

## 1. Цель
Протестировать устойчивость торгового движка к сверхбыстрой смене сигналов (Signal Flips) и убедиться в отсутствии дублирования ордеров, "зависания" позиций или гонок состояний при получении противоречивых команд от ML-модели.

## 2. План реализации
1.  **Mock ML**: Создать тестовый мок для [./src/trading/onnx.rs](./src/trading/onnx.rs), позволяющий детерминировано отправлять последовательность сигналов (например, Up, через 10мс Down, еще через 10мс Up).
2.  **Execution Loop Test**: В интеграционном тесте [./tests/integration_trading.rs](./tests/integration_trading.rs) запустить цикл бота и подать "дрожащий" сигнал.
3.  **State Verification**:
    - Убедиться, что при получении сигнала Down, когда Long еще в `PendingNew`, система вызывает `Cancel` (если это предусмотрено стратегией) или дожидается финализации перед переворотом.
    - Проверить, что в `OrderManager` не создается более одного активного ордера на вход для одного и того же сигнала.
4.  **Throttling Logic**: Проверить (и при необходимости внедрить в `execution.rs`) минимальный интервал между переворотами позиции (`min_flip_interval_ms`).

## 3. Технические детали
- **Test Scenario**:
```rust
#[tokio::test]
async fn test_rapid_signal_flip() {
    let mut bot = setup_test_bot().await;
    // 1. Send UP
    bot.inject_signal(Signal::Up).await; 
    assert!(bot.order_manager.has_active_orders());
    
    // 2. Immediate DOWN (while order is PendingNew)
    bot.inject_signal(Signal::Down).await;
    
    // 3. Verify: No double orders, correct transition
    let orders = bot.order_manager.get_all_orders();
    assert!(orders.len() <= 2); // Original + Close/Flip, no spam
}
```
- **Metrics**: Инкрементировать `bot_signal_oscillations_handled_total` в `ExecutionManager`, если новый сигнал пришел до завершения обработки предыдущего.

## 4. Критерии приемки
- [ ] Система не спамит ордерами при получении >5 сигналов в секунду.
- [ ] Все ордера корректно отслеживаются в машине состояний, "брошенных" (orphan) ордеров не остается.
- [ ] Тест подтверждает, что при смене сигнала `Up -> Down` старый ордер `Up` либо отменяется, либо учитывается при расчете объема для `Down`.
- [ ] Логгер фиксирует попытки осцилляции с уровнем `WARN` и дампом текущих `link_id`.

-