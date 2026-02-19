Задача 068: Отслеживание жизненного цикла ордеров (Order Lifecycle)
Цель: Реализовать надежную машину состояний для ордеров, обеспечивающую точность учета позиций при частичных исполнениях и корректную зачистку завершенных заявок.

Инструкции для реализации:
1. Изменения в ./src/trading/types.rs
Расширенный OrderStatus:
pub enum OrderStatus {
    Created,          // Локальный статус: подготовка к отправке
    New,              // Принят биржей, активен в стакане
    PartiallyFilled,  // Частично исполнен (не финальный)
    Filled,           // Полностью исполнен (финальный)
    Cancelled,        // Отменен (финальный)
    Rejected,         // Отклонен при создании (финальный)
    Expired,          // Истек (например, PostOnly-отклонение или IOC) (финальный)
}
Поля OrderInfo: Добавить updated_at: u64 (timestamp) для будущего анализа "зависших" ордеров и cum_exec_qty: Decimal для учета прогресса исполнения.
2. Изменения в ./src/trading/order_manager.rs
Метод update_order_state(&mut self, order_link_id: &str, event: OrderUpdate):
Partial Fill Handling:
Если новый статус PartiallyFilled, вычислить дельту: delta_qty = event.cum_exec_qty - self.orders[id].cum_exec_qty.
ВАЖНО: Немедленно отправить FillEvent в PositionManager для обновления позиции на эту дельту.
Final States:
Если статус Filled, Cancelled, Rejected или Expired, пометить ордер как завершенный.
Удалять ордер из active_orders только после того, как все дельты исполнения переданы в PositionManager.
Метод reconcile_with_exchange(&mut self, remote_orders: Vec<RemoteOrder>):
Сравнить локальный список с данными из /v5/order/realtime.
Если локальный ордер отсутствует в списке "активных" на бирже — инициировать запрос в /v5/order/history, чтобы понять, был он Filled или Cancelled, и корректно закрыть его локально.
3. Логирование
info!("Order {} status: {:?} -> {:?}", id, old, new).
warn!("Order {} Expired/Rejected by exchange. Reason: {}", id, reason).
Аргументация изменений:
Expired Status: В скальпинге мы часто будем использовать PostOnly. Если цена изменилась в момент отправки, Bybit не "отклонит" ордер, а пометит его как Expired. Без этого статуса бот будет бесконечно ждать исполнения несуществующей заявки.
Incremental Position Update: Ожидание полного исполнения (Filled) для обновления позиции опасно. Если ордер на 1 BTC исполнился на 0.9 BTC и "завис", бот должен знать, что 0.9 BTC уже в рынке, чтобы RiskManager видел реальную загрузку.
Reconciliation Strategy: Разделение на realtime (быстрый чек) и history (уточнение причин исчезновения) — это стандартный паттерн для высоконадежных торговых систем.
Критическое требование: При переходе в статус Filled необходимо убедиться, что итоговая сумма всех delta_qty (переданных в PositionManager) точно равна полному объему ордера, чтобы избежать накопления ошибки округления.