
# Задача 106: Реализация функций VWAP и Time-Weighted TWAP (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь в структуру `BotConfig` параметры для управления окном статистики:
```rust
pub struct BotConfig {
    // ...
    pub stats_window_ms: i64,       // Окно расчета (например, 60000 для 1 минуты)
    pub stats_max_trades: usize,    // Лимит сделок в очереди (защита памяти, например, 5000)
}
```

## 2. Определение типов в `src/data/types.rs`
Убедись, что структура `PublicTrade` готова для передачи между модулями:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicTrade {
    pub price: Decimal,
    pub amount: Decimal,
    pub side: Side,       // Buy/Sell
    pub timestamp: i64,   // Unix MS
}
```

## 3. Реализация `RollingPriceStats` в `src/utils/helpers.rs`
Реализуй структуру с использованием скользящего окна и **временного взвешивания (TWAP Proper)**:

```rust
pub struct RollingPriceStats {
    window_ms: i64,
    max_trades: usize,
    trades: VecDeque<PublicTrade>,
    
    // Для VWAP
    sum_pv: Decimal,      // Sum(Price * Amount)
    sum_vol: Decimal,     // Sum(Amount)
    
    // Для Time-Weighted TWAP
    sum_pw: Decimal,      // Sum(Price * TimeDelta)
    last_ts: i64,
    total_time_ms: i64,
}

impl RollingPriceStats {
    pub fn update(&mut self, trade: PublicTrade) {
        // 1. Расчет TWAP интеграла (Time-Weighted)
        if self.last_ts > 0 {
            let delta = (trade.timestamp - self.last_ts).max(0);
            if let Some(last_trade) = self.trades.back() {
                self.sum_pw += last_trade.price * Decimal::from(delta);
                self.total_time_ms += delta;
            }
        }
        self.last_ts = trade.timestamp;

        // 2. Добавление новой сделки (VWAP)
        self.sum_pv += trade.price * trade.amount;
        self.sum_vol += trade.amount;
        self.trades.push_back(trade);

        // 3. Очистка старых данных (Sliding Window)
        let cutoff = self.last_ts - self.window_ms;
        while self.trades.len() > 1 && (self.trades[0].timestamp < cutoff || self.trades.len() > self.max_trades) {
            let old = self.trades.pop_front().unwrap();
            
            // Вычитаем из VWAP
            self.sum_pv -= old.price * old.amount;
            self.sum_vol -= old.amount;
            
            // Важно: TWAP интеграл сложнее "вычитать" точно без хранения дельт, 
            // поэтому при pop_front допустима небольшая погрешность или пересчет total_time.
            // (Для Gemini: реализуй корректное вычитание pw дельты первой сделки).
        }
    }

    pub fn get_vwap(&self, side_filter: Option<Side>) -> Decimal {
        if let Some(side) = side_filter {
            let (s_pv, s_vol) = self.trades.iter()
                .filter(|t| t.side == side)
                .fold((Decimal::ZERO, Decimal::ZERO), |acc, t| (acc.0 + t.price * t.amount, acc.1 + t.amount));
            if s_vol.is_zero() { Decimal::ZERO } else { s_pv / s_vol }
        } else {
            if self.sum_vol.is_zero() { Decimal::ZERO } else { self.sum_pv / self.sum_vol }
        }
    }

    pub fn get_twap(&self) -> Decimal {
        if self.total_time_ms == 0 { 
            self.trades.back().map(|t| t.price).unwrap_or(Decimal::ZERO)
        } else {
            self.sum_pw / Decimal::from(self.total_time_ms)
        }
    }
}
```

## 4. Интеграция с WebSocket и Execution
1.  **WebSocket (`src/data/websocket.rs`)**: Добавь подписку на топик `publicTrade.{symbol}`. Парси входящие сообщения в `PublicTrade`.
2.  **Execution (`src/trading/execution.rs`)**:
    - Добавь `RollingPriceStats` в структуру `Execution`.
    - Обновляй статсы при получении новых сделок через канал.
    - Используй `get_vwap()` для расчета **Slippage Tolerance**: если цена планируемого лимитного ордера отклоняется от VWAP более чем на `X%`, логгируй предупреждение или корректируй цену.

---

## Аргументация для Планировщика:
1.  **Интеграл vs Средняя**: Grok прав, `sum_pw` (price * weight) — единственный верный способ для TWAP на неравномерных данных.
2.  **Side Filter**: Мы реализуем его через `fold` в `get_vwap`, чтобы не хранить отдельные суммы для каждой стороны (экономия памяти), так как фильтрация по стороне нужна реже, чем общий VWAP.
3.  **Безопасность**: Лимит `stats_max_trades` обязателен. На пампах Bybit может слать сотни сделок в секунду, что переполнит `VecDeque` и вызовет OOM.

**Gemini, твоя задача**: реализовать `RollingPriceStats` с учетом временного взвешивания и обеспечить интеграцию с потоком публичных сделок.