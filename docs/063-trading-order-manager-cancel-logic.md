Задача 063: Реализация отмены ордеров в OrderManager
Цель: Разработать надежный механизм отмены ордеров, включая эффективную очистку всех открытых заявок одним запросом.

Инструкции для реализации:
1. Изменения в ./src/trading/types.rs
Добавить CancelOrderRequest:
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelOrderRequest {
    pub category: String,
    pub symbol: String,
    pub order_link_id: Option<String>, // Отмена по нашему ID
    pub order_id: Option<String>,      // ИЛИ по ID биржи
}
Добавить CancelAllOrdersRequest:
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelAllOrdersRequest {
    pub category: String,
    pub symbol: String,
}
2. Изменения в ./src/trading/order_manager.rs
Метод cancel_order(&mut self, order_link_id: &str, force: bool):
Если !force и ордер не найден в active_orders — вернуть ошибку.
Если force — отправить запрос в любом случае (для "зачистки").
Обработка ответа: При коде 110001 (Order not exists) или успехе — удалить ордер из локальной мапы.
Метод cancel_all_orders(&mut self):
Оптимизация: Использовать ОДИН запрос к /v5/order/cancel-all с указанием symbol и category.
После успеха: Вызвать self.active_orders.clear() и залогировать массовую отмену.
Rate Limiting: Перед каждым вызовом REST API вызывать rate_limiter.wait() (подготовка к задаче 067).
3. Обработка исключений
Логировать warn! при попытке отмены несуществующего на бирже ордера, но считать это успехом для локального состояния (синхронизация).
Аргументация изменений:
Cancel-All: В критических ситуациях (например, резкий пролив против позиции) бот должен мгновенно очистить стакан. Цикл по 10 ордерам может занять секунду и упереться в Rate Limit. Один запрос cancel-all отрабатывает мгновенно на стороне биржи.
Force Cancel: Позволяет выполнять "гигиеническую" очистку при старте бота или после реконнекта, даже если мы потеряли локальное состояние.
Flexibility: Поддержка обоих типов ID (order_id и order_link_id) делает клиент более универсальным.
Критическое требование: Поле category в запросах на отмену должно строго соответствовать категории из конфига (например, "linear"). Ошибка в категории приведет к тому, что биржа "не увидит" ваши ордера.