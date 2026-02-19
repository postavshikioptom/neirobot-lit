# **Задача №172: Защита от десинхронизации времени (Clock Drift Protection)**

**Цель**: Обеспечение точности временных меток (timestamps), что критично для HFT-стратегий и корректного расчета возраста сигнала (задача [169](./docs/000-tasks_list.md)). Бот должен автоматически проверять расхождение между локальным временем сервера и временем биржи Bybit, блокируя торговлю при превышении допустимого дрейфа (Clock Drift).

---

## **План реализации для Gemini AI Coder:**

### **1. Изменения в [./src/config/types.rs](./src/config/types.rs)**
Добавить параметры в структуру `HealthCheckConfig` (из задачи [171](./docs/000-tasks_list.md)) для унификации:
- **`max_clock_drift_ms`**: `i64` (макс. расхождение, например 100мс).
- **`clock_sync_interval_s`**: `u64` (интервал проверки, например 300с).
- **`drift_action`**: `enum { StopBot, LogWarning }` (переиспользовать или расширить существующий `emergency_action`).

### **2. Реализация в [./src/utils/helpers.rs](./src/utils/helpers.rs)**
Создать единую функцию `calculate_clock_drift` (расширяя идею из [169](./docs/000-tasks_list.md)):
- **Логика**:
    1. Замер `t1` (`Instant::now()`).
    2. Вызов `GET /v5/market/time` через `RestClient` (задача [061](./docs/000-tasks_list.md)).
    3. Парсинг ответа: `result.timeNano` (Bybit 2026 format).
    4. Замер `t2` (`Instant::now()`).
- **Расчет**:
    - `rtt_ms = t2.duration_since(t1).as_millis() as u64`.
    - `server_ms = response.result.time_nano.parse::<u64>()? / 1_000_000`.
    - `adjusted_server_ms = server_ms + (rtt_ms / 2)`.
    - `drift = current_unix_ms() as i64 - adjusted_server_ms as i64`.
- **Отказоустойчивость**: При сетевой ошибке использовать `exponential_backoff` (задача [085](./docs/000-tasks_list.md)) и не блокировать бота сразу (только `warn`).

### **3. Интеграция в [./src/risk/health_monitor.rs](./src/risk/health_monitor.rs)**
Выполнять проверку в рамках жизненного цикла `HealthMonitor` (задача [171](./docs/000-tasks_list.md)):
- Добавить поле `last_clock_drift: i64`.
- В методе `run_periodic_checks` добавить вызов `calculate_clock_drift` по таймеру.
- Если `abs(drift) > max_clock_drift_ms`, устанавливать внутренний флаг `is_clock_stale = true`.

### **4. Интеграция в [./src/risk/risk_manager.rs](./src/risk/risk_manager.rs)**
- **Метод `check_risk_gates`**: Добавить проверку `if health_monitor.is_clock_stale() { return Err(RiskError::ClockDriftExceeded); }`.
- **Действие**: Если `drift_action == StopBot`, вызвать `emergency_market_close` (задача [109](./docs/000-tasks_list.md)).

### **5. Тестирование в [./tests/risk_gates.rs](./tests/risk_gates.rs)**
- **Mock Time**: Подменить JSON ответ Bybit:
  ```json
  { "retCode": 0, "result": { "timeNano": "1720000000123456789" } }
  ```
- **Validation**: Проверить корректность парсинга `timeNano` в миллисекунды и срабатывание `RiskError` при искусственном дрифте в 500мс.

---

## **Аргументы (согласовано с Grok):**
1. **Единая функция**: `calculate_clock_drift` в `helpers.rs` исключает дублирование кода между задачами 169 и 172.
2. **Локация вызова**: Размещение логики в `HealthMonitor` логично, так как это часть общего "здоровья" системы, и позволяет избежать лишних потоков (race conditions).
3. **Bybit API**: Явное указание на парсинг `result.timeNano` предотвратит ошибки Gemini при работе с вложенными структурами Bybit V5.
4. **RTT коррекция**: Использование `rtt / 2` критично для HFT, так как сетевая задержка в 20-40мс может быть ошибочно принята за дрифт часов.
5. **Robustness**: Добавление ретраев и мягкой обработки ошибок (LogWarning) предотвращает остановку бота из-за кратковременных сбоев сети при запросе времени.

**Gemini, твоя задача — синхронизировать время бота с пульсом биржи, обеспечив математическую точность всех временных проверок.**