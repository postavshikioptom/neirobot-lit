
# Задача 117: Гейт дисбаланса стакана (Order Book Imbalance Gate) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры в структуру `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub obi_threshold: f64,             // Порог блокировки (например, 0.7 для 85/15 дисбаланса)
    pub obi_depth: usize,               // Глубина стакана (например, 10 уровней)
}
```

## 2. Реализация расчета в `src/data/orderbook.rs`
Добавь метод расчета OBI с защитой от пустых данных и использованием `fold`:

```rust
impl OrderBook {
    pub fn calculate_imbalance(&self, depth: usize) -> f64 {
        // Защита от выхода за пределы стакана
        let d = depth.min(self.bids.len().min(self.asks.len()));
        if d == 0 { return 0.0; }

        let bid_vol = self.bids.values().take(d)
            .fold(Decimal::ZERO, |acc, l| acc + l.volume);
        let ask_vol = self.asks.values().take(d)
            .fold(Decimal::ZERO, |acc, l| acc + l.volume);
        
        let total_vol = bid_vol + ask_vol;
        if total_vol.is_zero() {
            return 0.0;
        }

        // Конвертируем в f64 для вычисления коэффициента
        let b = bid_vol.to_f64().unwrap_or(0.0);
        let a = ask_vol.to_f64().unwrap_or(0.0);
        
        (b - a) / (b + a)
    }
}
```

## 3. Реализация в `src/risk/risk_manager.rs`
Метод проверки гейта:
```rust
impl RiskManager {
    pub fn check_imbalance_gate(&self, side: Side, current_obi: f64) -> bool {
        if self.config.obi_threshold <= 0.0 { return true; }

        match side {
            Side::Buy => {
                // Блокируем покупку, если в стакане доминируют продавцы
                if current_obi < -self.config.obi_threshold {
                    tracing::warn!("BUY BLOCKED: OBI {:.2} < -{:.2}", current_obi, self.config.obi_threshold);
                    return false;
                }
            }
            Side::Sell => {
                // Блокируем продажу, если в стакане доминируют покупатели
                if current_obi > self.config.obi_threshold {
                    tracing::warn!("SELL BLOCKED: OBI {:.2} > {:.2}", current_obi, self.config.obi_threshold);
                    return false;
                }
            }
        }
        true
    }
}
```

## 4. Интеграция в `src/trading/execution.rs`
В методе `on_signal` (задача 043/044):

```rust
// 1. Считаем OBI на нужной глубине
let current_obi = self.orderbook.calculate_imbalance(self.config.obi_depth);

// 2. Логируем для отладки (debug уровень)
tracing::debug!("Current OBI: {:.4}", current_obi);

// 3. Проверяем через RiskManager
if !self.risk_manager.check_imbalance_gate(signal_side, current_obi) {
    return Ok(()); // Сделка отклонена гейтом
}
```

---

## Аргументация для Планировщика:
1.  **Safety**: Использование `min()` гарантирует, что мы не упадем, если в стакане Bybit внезапно станет меньше уровней, чем указано в `obi_depth`.
2.  **Performance**: `fold` эффективнее, чем создание промежуточного вектора для суммы.
3.  **Threshold Logic**: Положительный OBI (>0) означает перевес покупателей (поддержка для Buy), отрицательный (<0) — перевес продавцов. Гейт блокирует вход «против стены» лимитных заявок.

**Gemini, реализуй этот гейт, обеспечив корректные импорты `rust_decimal::prelude::ToPrimitive` для работы метода `to_f64()`.**