Задача 074: Реализация риск-гейта Max Open Orders Limit
Цель: Предотвратить избыточное количество активных лимитных ордеров («спам»), ограничив число одновременно висящих в стакане заявок. Это защищает от программных петель (loops) и превышения лимитов API.

1. Изменения в ./src/config/types.rs
RiskConfig: Использовать Option для гибкого включения/выключения лимита.
pub struct RiskConfig {
    // ...
    pub max_open_orders: Option<u32>, // None = лимит отключен
}
2. Изменения в ./src/risk/risk_manager.rs
Метод check_orders_limit_gate(active_orders_count: usize) -> bool:
Логика:
Если max_open_orders равен None -> return true (лимит не активен).
limit = max_open_orders.unwrap().
Мониторинг:
Если active_orders_count >= limit -> warn!("MAX ORDERS REACHED: {}/{}", active_orders_count, limit); return false.
Если active_orders_count >= (limit * 80 / 100) -> info!("Approaching orders limit: {}/{}", active_orders_count, limit).
Иначе -> debug!("Active orders: {}/{}", active_orders_count, limit); return true.
3. Изменения в ./src/trading/execution.rs
Интеграция в рабочий цикл:
Перед вызовом order_manager.place_limit_order():
Получить количество ожидающих (pending) ордеров из OrderManager (те, что в статусах New, PartiallyFilled, Untracked согласно задаче 068).
Вызвать risk_manager.check_orders_limit_gate(count).
Если гейт закрыт — пропустить сигнал на вход, не прерывая работу бота.
Важно: Гейт не должен блокировать запросы на отмену ордеров (cancel_order) или закрытие позиций.
4. Почему этот план лучше (Аргументы против упрощений):
Option Support: Позволяет легко отключить проверку в конфигурации exchange.toml без необходимости ставить «магические» большие числа (типа 9999).
Pending Focus: Мы считаем только те ордера, которые занимают место в OrderManager и на бирже (лимитки). Рыночные ордера исполняются мгновенно и не должны попадать под этот лимит.
Proactive Info: Лог на уровне 80% помогает заметить проблему (например, ордера не отменяются вовремя) до того, как торговля будет полностью заблокирована.
Bybit Context: Хотя Bybit позволяет до 500 ордеров на символ, для стратегии LiT (на основе LOB Transformer) типичное число ордеров — 1-5. Лимит в 10-20 является безопасным дефолтом.
5. Тестирование
Unit test: Проверить Option::None (всегда true).
Unit test: Проверить пороговые значения (80% — info, 100% — warn/false).
Integration test: Убедиться, что при достижении лимита новые ордера не создаются, но старые успешно отменяются.