Задача 075: Реализация риск-гейта Price Deviation & Precision Check
Цель: Предотвратить отправку ордеров с некорректной ценой (fat-finger, ошибки алгоритма) и обеспечить соответствие цены шагу тика (tickSize) биржи.

1. Изменения в ./src/config/types.rs
BotConfig: Гейт должен настраиваться индивидуально для каждого бота (пары).
pub struct RiskConfig {
    // ... другие лимиты
    pub max_price_deviation_pct: Option<Decimal>, // Например, 0.02 (2%)
}

pub struct BotConfig {
    // ...
    pub risk: RiskConfig,
}
Default: В config/loader.rs установить значение по умолчанию 0.02 (2%), если поле не задано в bot.toml.
2. Изменения в ./src/risk/risk_manager.rs
Метод validate_order_price:
pub fn validate_order_price(
    &self,
    order_price: Decimal,
    mid_price: Decimal,
    tick_size: Decimal,
) -> anyhow::Result<bool> {
    // 1. Проверка на кратность tick_size (Bybit rejection 10001)
    if !(order_price % tick_size).is_zero() {
        anyhow::bail!("Price {} is not a multiple of tick_size {}", order_price, tick_size);
    }

    // 2. Проверка отклонения (Price Deviation)
    let limit = match self.config.max_price_deviation_pct {
        Some(l) => l,
        None => return Ok(true), // Лимит отключен
    };

    if mid_price <= Decimal::ZERO {
        anyhow::bail!("Invalid mid_price: {}", mid_price);
    }

    let deviation = (order_price - mid_price).abs() / mid_price;

    if deviation > limit {
        warn!("Price deviation too high: {:.2}% (Order: {}, Mid: {})", 
               deviation * Decimal::from(100), order_price, mid_price);
        return Ok(false);
    }

    Ok(true)
}
3. Изменения в ./src/trading/execution.rs
Интеграция:
Использовать crate::data::orderbook::get_mid_price(&self.orderbook) для получения актуальной цены.
Получить tick_size из MarketInfo (загруженного в задаче 065).
Вызвать risk_manager.validate_order_price(...).
При Ok(false) или Err — блокировать отправку ордера.
4. Почему этот план лучше (Аргументы против упрощений):
Tick Size Validation: Отправка цены, не кратной шагу тика, гарантированно приведет к ошибке API (retCode 10001). Проверка "на берегу" экономит лимиты и время.
Anyhow Result: Возврат Result вместо макросов error! позволяет вызывающему коду (Execution) корректно обрабатывать сбои (например, пропустить один сигнал вместо падения всего потока).
Per-Bot Risk: Позволяет ставить узкий лимит (0.5%) на стабильных парах и широкий (5%) на волатильных щиткоинах.
Precision: Использование rust_decimal версии 1.34+ (с поддержкой % оператора для Decimal) критично для точности.
5. Тестирование
Unit test:
price=10.0001, tick=0.01 -> Ошибка (кратность).
price=10.2, mid=10.0, limit=0.01 -> Ok(false) (превышение 2%).
Integration test: Использовать mock OrderBook (BTreeMap) для имитации цен и проверки реакции Execution.