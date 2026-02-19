Задача 062-trading-order-manager-place-limit.md
Цель: Реализовать логику отправки лимитных ордеров через OrderManager, обеспечив надежную идентификацию сделок и корректное взаимодействие с API Bybit V5.

Инструкции для реализации:
1. Создание ./src/trading/types.rs
Создать модуль для DTO (Data Transfer Objects), используемых в торговой логике:

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateOrderRequest {
    pub category: String,       // "linear" для USDT Perpetuals
    pub symbol: String,         // Например, "BTCUSDT" (в верхнем регистре)
    pub side: String,           // "Buy" или "Sell"
    pub order_type: String,     // "Limit"
    pub qty: String,            // Decimal as string
    pub price: String,          // Decimal as string
    pub time_in_force: String,  // "GTC" или "PostOnly"
    pub order_link_id: String,  // Наш уникальный ID
    pub position_idx: i32,      // 0 для One-Way mode, 1 для Long (Hedge), 2 для Short (Hedge)
}
2. Изменения в ./src/trading/order_manager.rs
Генерация order_link_id:
Использовать формат: {prefix}_{nanos}_{random_u16}.
Пример: LIT_BTC_1706965400123456789_42.
Проверка: Перед отправкой убедиться, что ID отсутствует в локальной мапе active_orders.
Метод place_limit_order:
Принимать side, price (Decimal), qty (Decimal).
Брать symbol и position_idx из BotConfig.
Формировать CreateOrderRequest и отправлять через rest_client.post("/v5/order/create", body).
Локальный реестр:
При retCode == 0 сохранять структуру OrderInfo (ID, цена, объем, статус "Placed") в BTreeMap<String, OrderInfo>.
3. Обработка ошибок (Error Mapping)
Расширить обработку BybitError, добавив распознавание кодов:
110001: Invalid symbol (ошибка в конфиге).
110007: Insufficient balance.
30003: Price out of range (слишком далеко от mid_price).
20006: Duplicate order_link_id.
4. Конфигурация
По умолчанию использовать time_in_force: "GTC".
Убедиться, что symbol в запросе всегда в UPPERCASE.
Аргументация изменений:
Наносекунды + Random: В HFT-системах обычные миллисекунды могут дублироваться при высокой частоте сигналов. Наносекунды сводят риск коллизий к минимуму.
Архитектура Types: Разделение структур для данных (data/types.rs), инференса (ml/types.rs) и торговли (trading/types.rs) упрощает поддержку кода и предотвращает циклические зависимости.
Position Index: Для Bybit V5 поле positionIdx является обязательным при торговле фьючерсами. Ошибка в этом поле — самая частая причина отклонения ордеров в Hedge-режиме.
Критическое требование: Gemini должен использовать Decimal::to_string() для всех числовых полей в JSON. Использование f64 или serde_json::Number приведет к ошибкам округления на стороне биржи и отказу в приеме ордера.