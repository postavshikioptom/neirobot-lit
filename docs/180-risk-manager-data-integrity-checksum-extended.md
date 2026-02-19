# **Задача №180: Расширенная проверка целостности данных (Data Integrity Checksum Extended)**

**Цель**: Эволюция задачи [049](./docs/000-tasks_list.md). Реализация строгого механизма валидации локального стакана (LOB) через контрольные суммы (**CRC32 IEEE**) согласно спецификации Bybit V5. Это гарантирует, что ML-модель не получит искаженные данные из-за пропущенных WebSocket-апдейтов.

---

## **План реализации для Gemini AI Coder:**

### **1. Изменения в [Cargo.toml](./Cargo.toml)**
Добавить высокопроизводительный крейт для расчета контрольных сумм:
```toml
crc32fast = "1.4"
```

### **2. Изменения в [./src/config/types.rs](./src/config/types.rs)**
Добавить параметры в `HealthCheckConfig` (задача [171](./docs/000-tasks_list.md)):
- **`checksum_validation_enabled`**: `bool` (по умолчанию `true`).
- **`max_checksum_mismatches`**: `u32` (лимит ошибок до полного сброса, например `3`).

### **3. Логика в [./src/data/orderbook.rs](./src/data/orderbook.rs)**
Реализовать метод `calculate_checksum` с учетом требований Bybit (Top 25 Bids/Asks):
- **Форматирование строки**:
    - Использовать предварительно аллоцированный буфер `String` (задача [078](./docs/000-tasks_list.md)).
    - Формат: `{price}:{qty}` без лишних пробелов и научной нотации. Разделитель — `|`.
    - **Порядок**: Bids в порядке убывания цены (`rev()`), Asks в порядке возрастания.
    - **Важно**: Удалить последний символ `|` перед хешированием.
- **Пример реализации**:
  ```rust
  use crc32fast::Hasher;
  use std::fmt::Write;

  pub fn calculate_checksum(&self) -> u32 {
      let mut hasher = Hasher::new();
      let mut buffer = String::with_capacity(1024);
      // Bids desc
      for (&price, &qty) in self.bids.iter().rev().take(25) {
          write!(&mut buffer, "{}:{}|", price, qty).unwrap();
      }
      // Asks asc
      for (&price, &qty) in self.asks.iter().take(25) {
          write!(&mut buffer, "{}:{}|", price, qty).unwrap();
      }
      if buffer.ends_with('|') { buffer.pop(); } // Remove trailing |
      hasher.update(buffer.as_bytes());
      hasher.finalize()
  }
  ```

### **4. Интеграция в [./src/risk/health_monitor.rs](./src/risk/health_monitor.rs)**
Добавить обработку несоответствий:
- **Метод `checksum_mismatch()`**:
    - Инкрементировать счетчик ошибок.
    - Если `errors >= max_checksum_mismatches`:
        1. Установить `status = Corrupted` и заблокировать торговлю.
        2. Вызвать `ws_reconnect()` (задача [048](./docs/000-tasks_list.md)).
        3. Выполнить `orderbook.clear()`.
        4. Запросить `REST Snapshot` (задача [061](./docs/000-tasks_list.md)).

### **5. Обработка в [./src/data/websocket.rs](./src/data/websocket.rs)**
В парсере сообщений (задача [019](./docs/000-tasks_list.md)):
- Если поле `cs` (checksum) присутствует в JSON:
  ```rust
  if let Some(expected_cs) = json.cs {
      if !orderbook.verify_checksum(expected_cs) {
          health_monitor.checksum_mismatch();
      }
  }
  ```

---

## **Аргументация и проверка (согласовано с Grok):**
1. **CRC32Fast**: Выбор `crc32fast` обусловлен HFT-спецификой: он использует SIMD-инструкции, что делает расчет контрольной суммы практически "бесплатным" для CPU.
2. **Точность форматирования**: Строка `price:qty` должна быть идентична той, что генерирует Bybit. Отсутствие завершающего `|` — критическое требование документации.
3. **Full Recovery Cycle**: При расхождении данных недостаточно просто обновить стакан по REST. Мы должны переподключить WebSocket, так как `checksum mismatch` — верный признак потери пакетов или десинхронизации потока.
4. **Take 25**: Bybit V5 проверяет только топ-25 уровней с каждой стороны. Использование `take(25)` с итераторами `BTreeMap` обеспечивает $O(1)$ по памяти и высокую скорость.

**Gemini, твоя задача — гарантировать 100% идентичность данных в боте и на бирже. Любое расхождение — это немедленный сигнал к остановке и восстановлению.**