# Order Manager Unit Tests

Комплексное покрытие логики OrderManager тестами с мокированием REST API.

## Запуск тестов

```bash
# Запустить все тесты
cargo test --test order_manager_tests

# Запустить с выводом
cargo test --test order_manager_tests -- --nocapture

# Запустить конкретный тест
cargo test --test order_manager_tests test_add_order_success

# Запустить только REST API тесты
cargo test --test order_manager_tests rest_api_tests
```

## Покрытие тестов

### Базовые операции (13 тестов)
- ✅ Добавление ордера
- ✅ Проверка на дубликаты client_oid
- ✅ Обновление статуса ордера
- ✅ Перемещение в историю при терминальном статусе

### Жизненный цикл и исполнение (9 тестов)
- ✅ Множественные частичные исполнения (3 fill по 0.1)
- ✅ Идемпотентность дубликатов сообщений
- ✅ Переходы статусов: Created → New → PartiallyFilled → Filled
- ✅ Обработка Rejected и Expired статусов
- ✅ Распознавание PostOnlyRejected
- ✅ Автокоррекция cum_exec_qty при Filled статусе

### REST API операции (7 тестов)
- ✅ place_limit_order - успешное размещение
- ✅ place_limit_order - ошибка API (ордер не сохраняется)
- ✅ cancel_order - успешная отмена
- ✅ cancel_order - ордер не найден на бирже (110001)
- ✅ cancel_all_orders - массовая отмена
- ✅ reconcile_with_exchange - все ордера активны
- ✅ reconcile_with_exchange - ghost order cleanup

### Thread Safety (2 теста)
- ✅ Одновременное обновление нескольких ордеров
- ✅ Конкурентные операции place и update

### Синхронизация (3 теста)
- ✅ Ghost orders (ордера не найденные на бирже)
- ✅ Уникальность order_link_id
- ✅ Поиск по exchange_id

### Edge Cases (4 теста)
- ✅ Ордера с нулевым объемом
- ✅ Отрицательные цены
- ✅ Очень большие объемы
- ✅ Точность Decimal (0.1 + 0.2 = 0.3)

## Структура тестов

```
tests/order_manager_tests.rs
├── Базовые тесты (test_*)
├── async_tests (асинхронные и thread safety)
├── reconciliation_tests (сверка с биржей)
├── edge_cases (граничные случаи)
└── rest_api_tests (мокирование REST API)
```

## Архитектура мокирования

### Трейт-based подход
Используется трейт `BybitRestClientTrait` из `src/trading/rest_client.rs`:

```rust
#[async_trait]
pub trait BybitRestClientTrait: Send + Sync {
    async fn post<T, R>(&self, endpoint: &str, body: &T) -> Result<R>;
    async fn get_signed<R>(&self, endpoint: &str, params: &str) -> Result<R>;
}
```

### Мокирование с mockall
```rust
mock! {
    pub RestClient {}
    
    #[async_trait::async_trait]
    impl BybitRestClientTrait for RestClient {
        // методы трейта
    }
}
```

### Пример использования
```rust
let mut mock_client = MockRestClient::new();

mock_client
    .expect_post::<serde_json::Value, BybitOrderResult>()
    .withf(|endpoint, _| endpoint == "/v5/order/create")
    .times(1)
    .returning(|_, _| Ok(BybitOrderResult { ... }));

manager.place_limit_order(&mock_client, ...).await;
```

## Зависимости

- `mockall = "0.13"` - для мокирования REST клиента
- `async-trait = "0.1"` - для асинхронных трейтов
- `tokio` - для асинхронных тестов
- `rust_decimal` - для точных вычислений

## Критические проверки

1. **Изоляция**: Все тесты изолированы, нет сетевого взаимодействия
2. **Идемпотентность**: Дубликаты сообщений не удваивают объемы
3. **Точность**: rust_decimal гарантирует 0.1 + 0.2 = 0.3
4. **Thread Safety**: Arc<Mutex<>> для многопоточного доступа
5. **Ghost Orders**: Ордера не найденные на бирже корректно удаляются
6. **API Errors**: При ошибке API ордер не сохраняется локально

## Всего тестов: 38

- Базовые операции: 13
- Жизненный цикл: 9
- REST API: 7
- Thread Safety: 2
- Reconciliation: 3
- Edge Cases: 4
