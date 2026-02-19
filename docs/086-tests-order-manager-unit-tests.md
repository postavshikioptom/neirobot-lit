Задача 086: Unit-тесты для OrderManager
Цель: Реализовать комплексное покрытие логики OrderManager тестами, используя мокирование внешних зависимостей (Bybit REST API) и симуляцию различных рыночных сценариев.

1. Изменения в Cargo.toml
Добавить в [dev-dependencies]:
mockall = "0.13" (для асинхронных моков).
tokio = { version = "1.0", features = ["macros", "rt", "rt-multi-thread"] }.
2. Архитектура тестов
Файл: Создать отдельный файл ./tests/order_manager_tests.rs (интеграционный стиль тестов для сложной логики).
Trait-based Mocking: Чтобы мокировать BybitRestClient, необходимо выделить его методы в трейт (если это не сделано в задаче 061).
#[automock]
#[async_trait]
pub trait BybitRestClientTrait {
    async fn place_limit_order(&self, params: PlaceOrderParams) -> anyhow::Result<String>;
    async fn cancel_order(&self, symbol: &str, order_link_id: &str) -> anyhow::Result<()>;
    async fn get_active_orders(&self, symbol: &str) -> anyhow::Result<Vec<OrderInfo>>;
}
3. Тестовые сценарии (Scenarios)
Размещение и ID (Placement):
Проверить уникальность order_link_id (генерация префикса + таймштамп).
Убедиться, что при ошибке API ордер не сохраняется в локальном стейте active_orders.
Жизненный цикл и частичное исполнение (Fills):
Multiple Partial Fills: Имитировать 3 сообщения об исполнении (например, по 0.1 при общем объеме 0.5) и проверить корректность cum_exec_qty.
Duplicate Messages: Проверить идемпотентность (обработка двух одинаковых сообщений об исполнении не должна удваивать объем).
Синхронизация (Reconciliation):
Ghost Orders: Имитировать ситуацию, когда в локальном списке 3 ордера, а API вернул 2. Проверить, что "лишний" ордер удален (маркирован как Cancelled/Untracked).
Обработка статусов:
Проверить переходы: New -> PartiallyFilled -> Filled.
Проверить обработку Rejected и Expired.
4. Почему этот план лучше (Аргументы Grok):
Separate Test File: Вынос тестов в /tests позволяет избежать раздувания основного кода order_manager.rs и четче разделяет unit и integration логику.
Mockall 0.13: Поддерживает асинхронные методы через async_trait, что необходимо для reqwest клиента.
Decimal Precision: Использование rust_decimal в тестах гарантирует, что 0.1 + 0.2 будет строго равно 0.3, что критично для проверки полных исполнений (Fills).
Reconciliation Focus: Самая сложная часть менеджера — синхронизация при реконнекте. Тест на "ghost orders" предотвращает зависание несуществующих ордеров в памяти бота.
5. Критические требования
Async tests: Использовать #[tokio::test].
Isolation: Никакого сетевого взаимодействия. Использовать Box::pin(async { ... }) в возвратах мока.
Thread Safety: Если OrderManager используется через Arc<Mutex>, добавить тест с одновременным вызовом place и update из разных задач.
6. Запуск
cargo test --test order_manager_tests