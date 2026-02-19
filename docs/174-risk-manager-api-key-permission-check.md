# **Задача №174: Проверка разрешений API-ключей (API Key Permission Check)**

**Цель**: Предотвращение сбоев исполнения и защита от блокировок по причине некорректных настроек API на стороне биржи. Бот должен при запуске и каждые 24 часа проверять наличие необходимых прав (Trade/Contract), статус IP-вайтлиста и срок действия ключа.

---

## **План реализации для Gemini AI Coder:**

### **1. Изменения в [./src/config/types.rs](./src/config/types.rs)**
Добавить параметры в `ExchangeConfig` (из задачи [003](./docs/000-tasks_list.md)):
- **`required_permissions`**: `Vec<String>` (например, `["ContractTrade", "Order", "Position"]`).
- **`check_api_expiry`**: `bool` (по умолчанию `true`).
- **`min_api_days_left`**: `u32` (порог для предупреждения, например, 7 дней).

### **2. Реализация в [./src/trading/rest-client.rs](./src/trading/rest-client.rs)**
Добавить метод для запроса данных ключа (Bybit V5 API):
- **Endpoint**: `GET /v5/user/query-api-key`.
- **Логика**: Использовать `signed_get` с ретраями и `exponential_backoff` (задача [085](./docs/000-tasks_list.md)).
- **Парсинг полей (Bybit 2026 format)**:
  - `permissions`: `Vec<String>`.
  - `ipRestrict`: `bool` (true = ограничен по IP).
  - `expiredAt`: `u64` (время истечения в мс, `0` = бессрочно).

### **3. Интеграция в [./src/risk/health_monitor.rs](./src/risk/health_monitor.rs)**
Добавить метод `validate_api_permissions()`:
- **Проверка прав**: Сравнить `permissions` из ответа с `required_permissions` из конфига. Если критическое право отсутствует — `self.bot_active = false`.
- **IP Security**: Если `ipRestrict == false`, выводить `warn!("[Security] No IP restriction on API key!")`.
- **Expiry Logic**:
  ```rust
  if config.check_api_expiry && info.expired_at != 0 {
      let days_left = (info.expired_at as i64 - current_unix_ms() as i64) / 86_400_000;
      if days_left < config.min_api_days_left as i64 {
          warn!("[Health] API Key expires in {} days", days_left);
      }
  }
  ```

### **4. Запуск и периодичность в [./src/run-bot.rs](./src/run-bot.rs)**
- **Startup**: Выполнить первичную проверку перед инициализацией WebSocket.
- **Periodic**: Запустить `tokio::spawn` с интервалом 24 часа для фонового мониторинга статуса ключа (на случай отзыва прав во время работы).

### **5. Тестирование в [./tests/risk_gates.rs](./tests/risk_gates.rs)**
- **Mock Failure**: Подменить ответ API Bybit: `{"result": {"apiKeyInfo": [{"permissions": ["ReadOnly"]}]}}`.
- **Assertion**: Убедиться, что бот переходит в состояние `bot_active = false` и логирует `Insufficient permissions`.
- **Mock Expiry**: Симулировать `expiredAt` через 2 дня и проверить наличие `warning`.

---

## **Аргументация (согласовано с Grok):**
1. **Bybit V5 API**: Используется актуальный эндпоинт и корректные типы полей (`ipRestrict` как `bool`, а не `1/0`).
2. **Handle Infinite Keys**: Добавлена проверка `expiredAt != 0`, так как многие ключи не имеют срока действия.
3. **Periodic Check**: Проверка раз в сутки в `health_monitor` защищает от ручного отзыва ключа администратором в консоли биржи без остановки бота.
4. **Security Awareness**: Автоматический аудит `ipRestrict` — обязательное требование для систем, работающих с реальным капиталом, для минимизации рисков при утечке `API_SECRET`.
5. **Robustness**: Интеграция с `exponential_backoff` предотвращает падение бота при временных сетевых ошибках или превышении лимитов (Rate Limit) на публичных эндпоинтах управления аккаунтом.

**Gemini, твоя задача — гарантировать, что у бота всегда есть "лицензия на торговлю" и его права доступа в полном порядке.**