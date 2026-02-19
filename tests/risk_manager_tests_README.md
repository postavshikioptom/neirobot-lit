# Тесты RiskManager - Документация

## Обзор

Этот файл содержит комплексные тесты для проверки всех риск-гейтов (защитных механизмов) в `RiskManager`. Тесты покрывают математические расчеты, логику блокировки и edge cases.

## Структура тестов

### 1. Spread BPS Filter (Задача 073)
- **test_spread_bps_calculation_and_blocking**: Проверка расчета BPS и блокировки при превышении лимита
- **test_spread_gate_with_zero_mid_price**: Edge case с нулевой средней ценой (защита от деления на ноль)
- **test_spread_gate_config_none**: Проверка, что при отключенном лимите гейт всегда пропускает
- **test_spread_bps_exact_calculation**: Точность расчета BPS
- **test_spread_bps_at_80_percent_threshold**: Логирование при приближении к лимиту (80%)
- **test_inverted_spread_blocking**: Блокировка инвертированного спреда (ask < bid)
- **test_negative_prices_blocking**: Блокировка отрицательных цен

### 2. Drawdown & Peak Logic (Задача 072)
- **test_drawdown_peak_logic**: Проверка логики пиков и просадок
- **test_drawdown_reset_logic**: Проверка сброса блокировки после полуночи
- **test_cumulative_drawdown_blocking**: Блокировка при глобальной просадке
- **test_daily_drawdown_percentage_blocking**: Блокировка при дневной просадке в процентах

### 3. Max Position & Orders
- **test_max_orders_limit_blocking**: Блокировка при достижении лимита ордеров
- **test_max_position_size_blocking**: Блокировка при превышении максимального размера позиции
- **test_max_notional_blocking**: Блокировка при превышении максимального номинала
- **test_max_margin_blocking**: Блокировка при превышении максимальной маржи

### 4. Price Deviation & Tick (Задача 075)
- **test_price_deviation_blocking**: Блокировка при превышении отклонения цены
- **test_price_tick_precision_blocking**: Блокировка при некратности цены шагу тика
- **test_price_deviation_config_none**: Проверка отключенного лимита отклонения
- **test_price_validation_with_zero_tick**: Проверка с нулевым tick size

### 5. Дополнительные тесты
- **test_fail_fast_gate_chain**: Проверка цепочки гейтов (fail-fast логика)
- **test_reduce_only_always_allowed**: Проверка, что закрытие позиций всегда разрешено
- **test_min_notional_blocking**: Блокировка ордеров ниже минимального номинала
- **test_multiple_limits_first_violation_wins**: Проверка приоритета ошибок
- **test_short_position_reduce_only**: Проверка reduce-only для short позиций

## Использование tracing-test

Тесты используют библиотеку `tracing-test` для захвата и проверки логов:

```rust
#[test]
#[tracing_test::traced_test]
fn test_example() {
    // ... код теста ...
    
    // Проверка логов
    assert!(logs_contain("Expected log message"));
}
```

Функция `logs_contain` автоматически инжектируется макросом `#[traced_test]`.

## Запуск тестов

```bash
# Запуск всех тестов RiskManager
cargo test --test risk_manager_tests

# Запуск конкретного теста
cargo test --test risk_manager_tests test_spread_bps_calculation_and_blocking

# Запуск с выводом логов
cargo test --test risk_manager_tests -- --nocapture
```

## Критические требования

1. **Log Assertions**: Все тесты с блокировкой проверяют наличие соответствующих сообщений в логах
2. **Fail-Fast**: Проверяется, что при блокировке первого гейта остальные не вызываются
3. **No Floating Point**: Все расчеты используют `Decimal` для точности
4. **Zero-Division Guard**: Проверяется защита от деления на ноль

## Покрытие

Тесты покрывают:
- ✅ Все ветки `if` и `match` в методах валидации
- ✅ Edge cases (нулевые значения, отрицательные числа, инвертированные спреды)
- ✅ Математические расчеты (BPS, отклонения, маржа)
- ✅ Логику блокировки и сброса
- ✅ Конфигурационную гибкость (Option::None)
- ✅ Reduce-only логику
