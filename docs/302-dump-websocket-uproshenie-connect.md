# Задача 302: Упрощение WebSocket (Fix TLS Handshake)

## Описание
Текущий способ подключения к WebSocket Bybit через ручную настройку сокетов (`socket2`) и `TokioTlsConnector` вызывает ошибки `TLS handshake failed` и `STATUS_STACK_BUFFER_OVERRUN` в Windows. Это происходит из-за конфликта низкоуровневых настроек TCP с системным TLS (SChannel).

По примеру из `dum-primer` (старый работающий код), необходимо перейти на стандартный высокоуровневый `connect_async`.

## Цель
Удалить неоправданную сложность и обеспечить стабильное подключение к бирже без ошибок SSL/TLS.

## План изменений

### 1. Файл `src/data/websocket.rs`
- **[Удалить]**: функцию `create_optimized_socket` (она больше не нужна).
- **[Изменить]**: `BybitWsClient::connect_and_subscribe` (Public WS).
    - Заменить всю логику открытия сокета на:
      ```rust
      let (ws_stream, _) = tokio_tungstenite::connect_async(url).await?;
      ```
- **[Изменить]**: `BybitPrivateWsClient::connect_and_auth` (Private WS).
    - Также перейти на `connect_async(url)`.
- **[Очистить]**: Удалить неиспользуемые импорты: `socket2`, `native_tls`, `tokio_native_tls`.

### 2. Файл `src/data/parser.rs`
- **[Выполнено]**: Поля в структуре `BybitMarkPriceData` уже исправлены (заменены `s`/`p` на `symbol`/`markPrice` для соответствия V5 API).

## Верификация
1.  **Компиляция**: `cargo check --bin dump` (проверка отсутствия ошибок типов).
2.  **Запуск**: `cargo run --bin dump -- --hours 0.05`.
3.  **Логи**: В логах должно появиться `✅ WebSocket connected successfully!` без попыток реконнекта (Attempt 1, 2...).

## Ожидаемый результат
Дампер запускается, подключается к `wss://stream.bybit.com/v5/public/linear` через стандартный TLS и стабильно выгружает данные в течение заданного времени.

=========
# Задача 302-2 Debugging and Fixing Empty Parquet Data
The user reported that the Parquet dump for FARTCOINUSDT contains 3592 rows with 0.0 values. Another AI model suggested that the logger might be blocking in Windows and that last_update_id is updated prematurely in OrderBook::apply_update.

Proposed Changes
Infrastructure Layer
[MODIFY] logger.rs
Wrap std::io::stdout in NonBlocking to prevent blocking the main thread when Windows console is in QuickEdit mode.
Data Layer
[MODIFY] orderbook.rs
Defer last_update_id update: Move self.last_update_id and self.timestamp_ms updates to the END of 
reset_with_snapshot
 and 
apply_update
 to ensure the book is only marked as "ready" if data is successfully processed.
Add info! logging: Log the number of bids and asks received in snapshots.
Add warn! logging: Alert if Decimal::from_f64 returns ZERO for a non-zero price value.
[MODIFY] parser.rs
Add debug! logging in 
parse_orderbook_msg
 to log the raw 
ts
 and u values.
[MODIFY] dump.rs
Add a sanity check in 
push_snapshot
: if last_update_id > 0 but both best prices are 0.0, log a warn!.