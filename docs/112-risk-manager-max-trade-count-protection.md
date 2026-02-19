
# Задача 112: Лимит количества сделок за период (v2.0)

## 1. Изменения в конфигурации [./src/config/types.rs](./src/config/types.rs)
Добавь параметры в `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub max_trades_limit: usize,         // Макс. кол-во сделок (0 = отключено)
    pub max_trades_window_sec: u64,     // Скользящее окно в секундах
}
```

## 2. Расширение стейта в [./src/trading/types.rs](./src/trading/types.rs)
```rust
pub struct BotState {
    // ...
    pub recent_trade_timestamps: Vec<i64>, // Для сохранения на диск
}
```

## 3. Реализация в [./src/risk/risk_manager.rs](./src/risk/risk_manager.rs)

### Инициализация и структура:
```rust
use std::collections::VecDeque;
use chrono::Utc;

pub struct RiskManager {
    // ...
    // Внутреннее хранилище для быстрой очистки
    trade_history: VecDeque<i64>, 
}

impl RiskManager {
    pub fn new(config: BotConfig, state: &BotState) -> Self {
        Self {
            config,
            trade_history: VecDeque::from(state.recent_trade_timestamps.clone()),
            // ...
        }
    }

    // Вызывается при исполнении ордера (Fill)
    pub fn register_fill(&mut self, timestamp_ms: i64) {
        self.trade_history.push_back(timestamp_ms);
    }

    pub fn check_overtrading_limit(&mut self, state: &mut BotState) -> bool {
        // Если лимит 0 — проверка отключена
        if self.config.max_trades_limit == 0 {
            return true;
        }

        let now_ms = Utc::now().timestamp_millis();
        let cutoff = now_ms - (self.config.max_trades_window_sec as i64 * 1000);

        // 1. Очистка устаревших записей
        let initial_len = self.trade_history.len();
        self.trade_history.retain(|&ts| ts > cutoff);

        // 2. Синхронизация со стейтом только если были изменения
        if self.trade_history.len() != initial_len {
            state.recent_trade_timestamps = self.trade_history.iter().copied().collect();
            // Опционально: вызвать save_state() здесь, если окно очистилось
        }

        // 3. Проверка лимита
        if self.trade_history.len() >= self.config.max_trades_limit {
            tracing::error!(
                "OVERTRADING PROTECT: {} trades in {}s window (Limit: {})", 
                self.trade_history.len(), 
                self.config.max_trades_window_sec,
                self.config.max_trades_limit
            );
            return false;
        }
        
        true
    }
}
```

## 4. Интеграция в [./src/trading/execution.rs](./src/trading/execution.rs)
1.  **Обработка Fill**: При получении сообщения об исполнении ордера (из WebSocket или REST), вызывай `risk_manager.register_fill(timestamp)`.
2.  **Перед выставлением ордера**:
    - Вызывай `risk_manager.check_overtrading_limit(&mut self.state)`.
    - Если вернулось `false`, **не отправляй** новый ордер.
    - **Важно**: Если торговля заблокирована по этому лимиту, логгируй это с уровнем `WARN` не чаще одного раза в минуту (чтобы не спамить лог).

## 5. Особенности реализации
- **Startup**: При запуске `RiskManager` загружает историю из `BotState`. Сразу после загрузки выполни `check_overtrading_limit`, чтобы очистить записи, которые устарели, пока бот был выключен.
- **Performance**: Мы используем `retain` на `VecDeque`. Для небольших лимитов (до 1000 сделок) это практически мгновенно.
- **I/O Optimization**: `save_state` вызывается только в основном цикле сохранения стейта или при критических событиях, а не на каждой регистрации сделки.

---

## Аргументация для Планировщика:
1.  **Zero-Disable**: Удобно для тестирования — установил 0 и лимит не мешает.
2.  **Consistency**: Использование `recent_trade_timestamps` из стейта гарантирует, что "память" о сделках сохраняется между перезапусками.
3.  **Encapsulation**: Логика очистки окна скрыта внутри `check_overtrading_limit`, что делает интерфейс `RiskManager` чистым.

**Gemini, реализуй эту защиту, уделяя внимание корректной конвертации `VecDeque` обратно в `Vec` для стейта при изменениях.**