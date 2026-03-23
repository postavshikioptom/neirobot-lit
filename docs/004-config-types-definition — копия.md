# Задача 004: Определение структур конфигурации (Config Types)
Цель: Создать файл src/config/types.rs, содержащий строго типизированные Rust-структуры для всех уровней конфигурации. Это обеспечит безопасную десериализацию TOML и предотвратит ошибки в логике бота из-за неверных типов данных.

Файл: src/config/types.rs

Инструкции для Gemini:
Реализовать структуры с использованием serde и функций для значений по умолчанию.

1. Подключить зависимости:

use serde::{Deserialize, Serialize};
2. Реализовать вспомогательные функции для #[serde(default)]: Создать функции типа fn default_level() -> String { "info".to_string() } для всех полей, чтобы обеспечить корректную загрузку при частичном заполнении файлов.

3. Структуры для GlobalConfig (согласно 002):

GeneralConfig: env (String), timezone (String).
LoggingConfig: level (String), format (String), file_name (String), rotate (bool), max_size_mb (u64).
TradingDefaultsConfig: max_latency_ms (u64), retry_interval_ms (u64), max_retries (u32).
RiskDefaultsConfig: drawdown_stop_pct (f64), max_orders_per_minute (u32).
4. Структуры для ExchangeConfig (согласно 003):

BybitConfig: category (String), api_key_path (String).
WebsocketConfig: public_url (String), private_url (String), ping_interval_sec (u64), pong_timeout_sec (u64), max_subscriptions_per_connection (u32).
RestConfig: base_url (String), request_timeout_sec (u64).
RateLimitsConfig: rest_requests_per_second (u32), private_endpoint_per_minute (u32), backoff_base_ms (u64).
5. Структура BotConfig (per-symbol):

symbol: String (обязательно).
model_path: String (путь к .onnx).
threshold_buy: f32 (дефолт 0.5).
threshold_sell: f32 (дефолт 0.5).
threshold_flat: f32 (дефолт 0.3).
max_position_size: Option (переопределение глобального риска).
6. Структуры верхнего уровня:
```
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct GlobalConfig {
    pub general: GeneralConfig,
    pub logging: LoggingConfig,
    pub trading_defaults: TradingDefaultsConfig,
    pub risk_defaults: RiskDefaultsConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ExchangeConfig {
    pub bybit: BybitConfig,
    pub websocket: WebsocketConfig,
    pub rest: RestConfig,
    pub rate_limits: RateLimitsConfig,
}
```
Технические требования:
Публичность: Все структуры и их поля должны быть помечены как pub.
Derive: Добавить #[derive(Debug, Deserialize, Serialize, Clone, Default)] для всех структур.
Дефолты: Для полей thresholds, file_name, env и т.д. использовать аннотации #[serde(default = "path_to_fn")].
Изоляция: BotConfig должен содержать только те поля, которые специфичны для одного токена.
Почему это важно:
Это "контракт" между TOML-файлами и кодом Rust. Если мы добавим новое поле в exchange.toml, но не опишем его здесь, бот его проигнорирует. Использование Option<T> и default функций позволяет гибко переопределять настройки.