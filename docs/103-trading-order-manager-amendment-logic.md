# Задача 103: Trading Order Manager Amendment Logic

## 1. Цель
Реализовать логику изменения (amendment) активных ордеров в [src/trading/order_manager.rs](./src/trading/order_manager.rs). Это позволит боту динамически обновлять параметры ордера (цену, объем, триггерную цену) через метод `POST /v5/order/amend`, что эффективнее, чем связка `cancel` + `place`.

## 2. Изменения

### Файл: [src/trading/rest_client.rs](./src/trading/rest_client.rs)
- Добавить структуру параметров и ответа:
```rust
pub struct AmendParams {
    pub category: String,        // "linear", "inverse", "spot"
    pub symbol: String,
    pub order_link_id: String,
    pub price: Option<Decimal>,
    pub qty: Option<Decimal>,
    pub trigger_price: Option<Decimal>, // Для условных ордеров
}

pub struct AmendedResponse {
    pub order_id: String,
    pub order_link_id: String,
}
```
- Реализовать метод `amend_order`:
```rust
pub async fn amend_order(&self, params: AmendParams) -> Result<AmendedResponse, anyhow::Error> {
    let mut body = json!({
        "category": params.category,
        "symbol": params.symbol,
        "orderLinkId": params.order_link_id,
    });
    
    // Добавляем опциональные поля только если они Some
    if let Some(p) = params.price { body["price"] = json!(p.to_string()); }
    if let Some(q) = params.qty { body["qty"] = json!(q.to_string()); }
    if let Some(tp) = params.trigger_price { body["triggerPrice"] = json!(tp.to_string()); }

    self.rate_limiter.wait().await;
    let resp: BybitResponse<AmendedResponse> = self.signed_post("/v5/order/amend", &body).await?;
    
    if resp.ret_code == 0 {
        Ok(resp.result)
    } else {
        Err(anyhow::anyhow!("Bybit Error {}: {}", resp.ret_code, resp.ret_msg))
    }
}
```

### Файл: [src/trading/order_manager.rs](./src/trading/order_manager.rs)
- **Метод amend_active_order**:
    - Проверять локальный `OrderInfo`: если новые `price`/`qty` совпадают с текущими, **не отправлять** запрос (экономия лимитов).
    - **Обработка ошибок**:
        - `110004` (Order not modified): игнорировать (считать успехом).
        - `110001` (Order not found): удалить ордер из локального списка (считаем исполненным/отмененным).
        - `170139` (Order qty out of range): логировать ошибку `error!` и не ретраить.
    - **Синхронизация**: При успехе REST-запроса обновить локальные данные, но окончательную истину принимать через приватный WebSocket канал (задача 080).

## 3. Критические требования
- **Category**: Поле `category` обязательно. Брать из `BotConfig` (по умолчанию "linear" для USDT-бессрочных).
- **Идемпотентность**: Использовать только `orderLinkId` для идентификации ордера.
- **Decimal**: Все числовые значения передавать в Bybit как строки через `to_string()`.

## 4. Зависимости
- `src/trading/rest_client.rs` (метод API).
- `src/utils/rate_limiter.rs` (контроль частоты запросов).
