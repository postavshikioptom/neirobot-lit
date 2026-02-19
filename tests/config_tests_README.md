# Тесты ConfigLoader - Документация

## Обзор

Этот файл содержит комплексные тесты для проверки корректности работы кастомного загрузчика конфигураций (`ConfigLoader`). Тесты проверяют логику ручного слияния (merge) уровней настроек и работу валидаторов.

Всего реализовано **16 тестов**, покрывающих все аспекты работы с конфигурацией.

## Структура тестов

### 1. Ручное слияние (Manual Merge)

**test_manual_merge_hierarchy**
- Проверяет иерархию переопределения: Bot > Exchange > Global
- Тестирует, что `max_spread_bps` из bot config (10) переопределяет значение из global (50)
- Проверяет наследование `ws_ping_interval_secs` от exchange config

**test_three_level_override**
- Комплексная проверка переопределения на всех трех уровнях
- Проверяет приоритеты: bot > exchange > global
- Тестирует несколько полей одновременно

### 2. Обработка Option-полей

**test_option_fields_inheritance**
- Проверяет, что поля типа `Option<T>`, отсутствующие в bot config, корректно наследуются из global
- Тестирует `max_position_size` и `max_notional_usd`

**test_validation_negative_max_position_size**
- Проверяет отклонение отрицательных значений для `max_position_size`
- Ожидаемая ошибка: "max_position_size must be positive"

**test_validation_nonexistent_model_path**
- Проверяет валидацию существования пути к файлу модели
- Ожидаемая ошибка: "model file not found"

**test_validation_invalid_thresholds**
- Проверяет бизнес-правило: `threshold_buy` должен быть больше `threshold_sell`
- Ожидаемая ошибка: "threshold_buy must be greater than threshold_sell"

**test_validation_threshold_flat_out_of_range**
- Проверяет диапазон значений для `threshold_flat` (должен быть между 0.0 и 1.0)
- Ожидаемая ошибка: "threshold_flat must be between 0.0 and 1.0"

**test_validation_empty_symbol**
- Проверяет, что символ не может быть пустым
- Ожидаемая ошибка: "symbol is empty"

**test_validation_symbol_format_with_dash**
- Проверяет формат символа (не должен содержать дефисы)
- Ожидаемая ошибка: "invalid format"

**test_validation_symbol_format_lowercase**
- Проверяет, что символ должен быть в верхнем регистре
- Ожидаемая ошибка: "must be in uppercase"

**test_validation_zero_max_open_orders**
- Проверяет, что `max_open_orders` не может быть равен 0
- Ожидаемая ошибка: "max_open_orders must be > 0"

**test_validation_zero_max_spread**
- Проверяет, что `max_spread_bps` не может быть равен 0
- Ожидаемая ошибка: "max_spread_bps must be > 0"

### 4. Синтаксические ошибки

**test_invalid_toml_syntax**
- Проверяет обработку невалидного TOML (пропущенные кавычки)
- Ожидаемая ошибка: "Failed to parse TOML"

### 5. Значения по умолчанию

**test_default_values**
- Проверяет, что поля с дефолтными значениями корректно инициализируются
- Тестирует: `seq_len`, `features_dim`, `leverage`, `post_only`

### 6. Environment Overrides (Secrets)

**test_environment_overrides_secrets**
- Проверяет, что метод `load_secrets()` корректно читает переменные окружения
- Тестирует приоритет: переменные окружения > .env файл

**test_secrets_missing_env_vars**
- Проверяет обработку отсутствующих переменных окружения
- Ожидаемая ошибка: "BYBIT_API_KEY not found"

### 7. Default Traits (отсутствие конфигов)

**test_missing_global_config_error**
- Проверяет поведение при отсутствии `global.toml`
- Ожидаемая ошибка: "Failed to read config file"

**test_missing_exchange_config_error**
- Проверяет поведение при отсутствии `exchange.toml`
- Ожидаемая ошибка: "Failed to read config file"

## Использование tempfile

Тесты используют библиотеку `tempfile` для создания временных конфигурационных файлов:

```rust
fn setup_test_configs(
    global_content: &str,
    exchange_content: &str,
    bot_content: &str,
) -> Result<(TempDir, std::path::PathBuf)>
```

Эта функция:
1. Создает временную директорию
2. Создает файлы `global.toml`, `exchange.toml` и `bot config.toml`
3. Создает фиктивный файл модели для прохождения валидации
4. Автоматически удаляет все файлы после завершения теста

## Запуск тестов

```bash
# Запуск всех тестов конфигурации
cargo test --test config_tests

# Запуск конкретного теста
cargo test --test config_tests test_manual_merge_hierarchy

# Запуск с выводом логов
cargo test --test config_tests -- --nocapture
```

## Критические требования

1. **No Panics**: Загрузчик всегда возвращает `anyhow::Result`, никогда не паникует
2. **Exact Match**: Используется `assert_eq!` для точной проверки всех полей
3. **Custom Validation**: Проверяются бизнес-правила, специфичные для торговой логики
4. **Tempfile Cleanup**: Автоматическое удаление тестовых конфигов после прохождения тестов

## Покрытие

Тесты покрывают:
- ✅ Иерархию слияния конфигураций (Bot > Exchange > Global)
- ✅ Обработку Option-полей и наследование значений
- ✅ Валидацию положительных значений (max_position_size, max_open_orders, max_spread_bps)
- ✅ Валидацию существования путей к файлам
- ✅ Валидацию бизнес-правил (пороги, диапазоны)
- ✅ Валидацию формата символа (верхний регистр, без дефисов)
- ✅ Обработку синтаксических ошибок TOML
- ✅ Проверку значений по умолчанию
- ✅ Environment Overrides для секретов (API ключи)
- ✅ Обработку отсутствующих конфигурационных файлов
- ✅ Изоляцию конфигураций ботов (Single Bot Isolation)

## Архитектурные принципы

### No Figment
Тесты проверяют именно тот код, который написан вручную (используя `toml::from_str` и merge), что исключает скрытое поведение внешних библиотек.

### Custom Validation
В отличие от простого парсинга, проверяются "бизнес-правила", которые критичны для торговой логики.

### Single Bot Isolation
Тесты подтверждают архитектуру, где каждый бот имеет свой изолированный конфиг, и настройки одного не могут случайно переопределить настройки другого.

### Tempfile Cleanup
Автоматическое удаление тестовых конфигов после прохождения тестов предотвращает мусор в репозитории.
