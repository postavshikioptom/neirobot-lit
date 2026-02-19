# Задача 231: Обработка ошибок при нехватке маржи (Insufficient Margin Recovery)

Реализация механизма восстановления после специфических ошибок Bybit V5 (`110007` — Insufficient margin, `110004` — Insufficient wallet balance). Бот должен автоматически высвобождать ликвидность и временно снижать нагрузку на баланс.

## 1. Цель задачи
Предотвратить цикличные попытки входа при дефиците средств, автоматически отменять конкурирующие ордера и реализовать адаптивное снижение объема (Sizing) для продолжения торговли.

## 2. Инструкции по реализации для Gemini

### А. Rust: Детекция и логика восстановления ([./src/trading/order_manager.rs](./src/trading/order_manager.rs))
1.  **Обработка API-ответа**:
    *   В функции отправки ордера реализовать `match` по коду возврата:
        ```rust
        match response.ret_code {
            110004 | 110007 => self.handle_margin_error().await?,
            0 => Ok(response),
            _ => Err(OrderError::ApiError(response.ret_msg)),
        }
        ```
2.  **Метод `handle_margin_error`**:
    *   **Action 1: Cancel All**: Выполнить немедленную отмену всех активных ордеров по текущему символу через `POST /v5/order/cancel-all` (params: `{"category": "linear", "symbol": symbol}`).
    *   **Action 2: Backoff & Sizing**: Установить флаг `is_margin_limited` в состояние `true` и запустить таймер на `margin_error_backoff_minutes`.
    *   **Action 3: Logging**: Использовать `tracing::error!` для детального логирования инцидента.

### Б. Интеграция с Risk Manager ([./src/risk/risk_manager.rs](./src/risk/risk_manager.rs))
1.  **Поля состояния**: Добавить `margin_multiplier: f64` (по умолчанию `1.0`) и `last_margin_error_ts: Option<Instant>`.
2.  **Метод `apply_margin_penalty(&mut self)`**:
    *   Уменьшить `margin_multiplier` до значения `BotConfig.margin_penalty_multiplier` (например, `0.5`).
    *   В методе расчета объема (`calculate_size`) учитывать этот множитель.
3.  **Сброс состояния**: В основном цикле проверять время: если прошло более `backoff_minutes`, вернуть `margin_multiplier` к `1.0`.

## 3. Изменения в конфигурации ([./src/config/types.rs](./src/config/types.rs))
Добавить в `BotConfig` новые поля:
-   **margin_error_backoff_minutes**: `u64` (время «штрафного» периода).
-   **margin_penalty_multiplier**: `f64` (на сколько снижать размер позиции после ошибки).

## 4. Ожидаемый результат
1.  При получении ошибки маржи бот мгновенно очищает «зависшие» ордера, высвобождая `blocked_balance`.
2.  Следующие попытки входа в течение заданного времени выполняются уменьшенным объемом (защита от повторной ошибки).
3.  Информация об инциденте фиксируется в `logs/bot.log` с тегом `CRITICAL`.

## 5. Необходимые зависимости
-   **Rust**: `reqwest`, `tracing`, `tokio`.