
# Задача 111: Лимит максимального дневного убытка (Daily Hard Stop) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь параметры в структуру `BotConfig`:
```rust
pub struct BotConfig {
    // ...
    pub max_daily_loss_pct: Decimal,    // Лимит в % (например, 2.0)
    pub daily_reset_hour_utc: u32,      // Час сброса по UTC (0-23)
}
```

## 2. Обновление стейта в `src/trading/types.rs`
```rust
pub struct BotState {
    // ...
    pub day_start_pnl: Decimal,         // Накопленный PnL на начало текущих суток
    pub last_pnl_reset_ts: i64,         // Таймстемп последнего сброса (ms)
}
```

## 3. Реализация в `src/risk/risk_manager.rs`

### Логика сброса статистики:
```rust
fn should_reset_daily_stats(&self, now: DateTime<Utc>) -> bool {
    let last_reset = DateTime::from_timestamp(self.state.last_pnl_reset_ts / 1000, 0)
        .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
    
    // Сброс если: сменилась дата ИЛИ наступил час сброса (если в прошлые сутки не сбрасывали)
    now.date_naive() != last_reset.date_naive() && now.hour() >= self.config.daily_reset_hour_utc
}
```

### Основная проверка лимита:
```rust
pub async fn check_daily_limit(&mut self) -> anyhow::Result<bool> {
    let now = Utc::now();
    let mid_price = self.orderbook.get_mid_price();
    
    // 1. Получаем актуальный PnL (реализованный + нереализованный)
    let current_pnl = self.position_manager.get_total_pnl(mid_price);
    
    // 2. Получаем актуальный Equity с биржи (с ретраями из задачи 085)
    let equity = self.rest_client.get_equity_with_retry(3).await?;

    // 3. Проверка и выполнение сброса начала дня
    if self.should_reset_daily_stats(now) {
        self.state.day_start_pnl = current_pnl;
        self.state.last_pnl_reset_ts = now.timestamp_millis();
        self.save_state()?; // Немедленное сохранение
        tracing::info!("Daily PnL limit reset. New day_start_pnl: {}", current_pnl);
    }

    // 4. Расчет просадки за сегодня
    let daily_pnl = current_pnl - self.state.day_start_pnl;
    
    if daily_pnl < Decimal::ZERO {
        let loss_pct = (daily_pnl.abs() / equity);
        let limit_pct = self.config.max_daily_loss_pct / Decimal::from(100);
        
        if loss_pct > limit_pct {
            tracing::error!(
                "DAILY LOSS LIMIT BREACHED: {:.2}% (Limit: {}%)", 
                (loss_pct * Decimal::from(100)), 
                self.config.max_daily_loss_pct
            );
            return Ok(false); 
        }
    }
    
    Ok(true)
}
```

## 4. Интеграция в `src/trading/execution.rs`
Перед каждым расчетом объема ордера (`calculate_order_size`):
1.  Вызывай `risk_manager.check_daily_limit().await`.
2.  Если результат `false` или `Err` (критическая ошибка API после ретраев):
    - Установи `self.emergency_mode = true`.
    - Выполни `self.emergency_market_close().await` (задача 109).
    - Выведи в лог: `FATAL: Bot stopped due to daily risk limit or API failure`.

## 5. Обновление `PositionManager` (`src/trading/position_manager.rs`)
Убедись, что метод `get_total_pnl` корректно считает:
- `Realized PnL` (уже закрытые сделки).
- `Unrealized PnL` = `(mid_price - avg_price) * position_size * side_multiplier`.

---

## Аргументация для Планировщика:
1.  **Fresh Data**: Ретраи при получении `equity` гарантируют, что мы не остановим бота из-за кратковременного сетевого лага, но и не пропустим реальную просадку.
2.  **Date Naive**: Сравнение через `date_naive()` — самый надежный способ в Rust обработать смену дня по UTC.
3.  **Decimal Precision**: Деление `daily_pnl / equity` в формате `Decimal` исключает "плавающую запятую", что критично для финансовых расчетов.

**Gemini, реализуй эту логику, обеспечив атомарное сохранение стейта при сбросе дня.**