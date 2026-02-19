# Задача №169: Проверка свежести сигнала (Signal Staleness Check)

**Цель**: Предотвращение исполнения сделок по устаревшим сигналам. Бот должен гарантировать, что время от момента получения данных с биржи до принятия решения в `execution.rs` не превышает допустимый порог (Latency Budget).

---

## План реализации для Gemini AI Coder:

### 1. Изменения в структурах данных
- **[./src/ml/types.rs](./src/ml/types.rs)**: Добавить поле `source_timestamp_ms: u64` в перечисление/структуру `Signal` (согласно задаче [033](./docs/000-tasks_list.md)). Это время получения (`receive_ts`) исходного снепшота стакана.
- **[./src/ml/onnx.rs](./src/ml/onnx.rs)**: Обновить результат инференса:
  ```rust
  pub struct InferenceResult {
      pub signal: Signal,
      pub duration_us: u64, // Время выполнения модели в микросекундах
  }
  ```

### 2. Изменения в [./src/config/types.rs](./src/config/types.rs)
Добавить параметры в `ExecutionConfig` (изоляция per-bot через `BotConfig`):
- **`max_signal_age_ms`**: `u64` (лимит "протухания", например, 100мс).
- **`max_clock_skew_ms`**: `i64` (максимально допустимое расхождение времени с биржей).
- **`staleness_action`**: `enum { Skip, LogOnly }`.

### 3. Синхронизация времени в [./src/utils/helpers.rs](./src/utils/helpers.rs)
Реализовать проверку `check_clock_skew` через REST-клиент (задача [061](./docs/000-tasks_list.md)):
- **Метод**: Выполнить `GET /v5/market/time`.
- **Логика**: `delta = local_ms - server_ms`. Если `abs(delta) > max_clock_skew_ms`, выдать `critical log`, так как расчеты возраста сигнала станут недостоверными.

### 4. Логика в [./src/trading/execution.rs](./src/trading/execution.rs)
Внедрить финальный гейт в метод `can_execute`:
- **Расчет возраста**: `let age = helpers::current_unix_ms() - signal.source_timestamp_ms`.
- **Валидация**:
  ```rust
  if age > self.config.max_signal_age_ms {
      logger::warn!("[Stale] Age {}ms > {}ms limit", age, self.config.max_signal_age_ms);
      if self.config.staleness_action == StalenessAction::Skip {
          return false; 
      }
  }
  ```

### 5. Мониторинг и Риски ([./src/risk/risk_manager.rs](./src/risk/risk_manager.rs))
- **Rolling Ratio**: Использовать `VecDeque` для хранения статуса последних сигналов.
- **Circuit Breaker**: Если за последние 5 минут более 50% сигналов оказались `Stale` — инициировать остановку бота (признак деградации сетевого канала или перегрузки CPU).

### 6. Тестирование в `tests/risk_gates.rs`
- **Mock**: Создать сигнал с `source_timestamp_ms`, который на 200мс меньше текущего времени.
- **Expectation**: При `max_signal_age_ms = 100`, `can_execute` должен вернуть `false`.

---

## Аргументация и проверка (ответы на замечания Grok):
1. **Локация Signal**: Согласен с Grok, `Signal` относится к ML-слою ([./src/ml/types.rs](./src/ml/types.rs)), а не к `data/types.rs`.
2. **Источник времени**: Используем `receive_ts` (локальное время получения сообщения из WebSocket), чтобы измерять именно задержку обработки внутри нашего контура, исключая плавающий сетевой лаг биржи.
3. **Inference Duration**: Важно замерять длительность инференса через `Instant::elapsed()`, так как это основной источник задержки в ML-боте.
4. **Clock Skew**: Вместо сложного NTP-протокола используем простой запрос к API Bybit `/v5/market/time`, что достаточно для оперативного контроля десинхронизации.

**Gemini, твоя задача — превратить бота в "детектор лжи" для сигналов. Если данные опоздали хоть на мгновение — они не должны попасть в стакан.**