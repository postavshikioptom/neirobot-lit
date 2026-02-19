
# Задача 105: Динамический скейлинг объема на основе волатильности (v2.0)

## 1. Изменения в конфигурации [./src/config/types.rs](./src/config/types.rs)
Добавь в структуру `BotConfig` следующие поля:
```rust
pub struct BotConfig {
    // ... существующие поля
    pub volatility_target_bps: f64,    // Целевая волатильность (например, 5.0 - 10.0 для HFT)
    pub volatility_window: usize,      // Размер окна (например, 100 снапшотов)
    pub volatility_default: f64,    // Значение по умолчанию до заполнения окна
    pub size_min_multiplier: f64,   // Минимум (например, 0.5)
    pub size_max_multiplier: f64,   // Максимум (например, 2.0)
}
```

## 2. Реализация буфера и расчета в [./src/trading/execution.rs](./src/trading/execution.rs)
Переносим логику расчета из `OrderBook` в `Execution`. 

### Обновление структуры `Execution`:
Добавь поля для эффективного расчета волатильности без пересчета всего вектора:
```rust
pub struct Execution {
    mid_history: VecDeque<f64>,
    sum_returns: f64,
    sum_returns_sq: f64,
    // ...
}
```

### Метод обновления данных:
Внедрить в цикл обработки обновлений (после получения нового `mid_price` из `OrderBook`):
```rust
fn update_volatility(&mut self, new_mid: f64, window: usize) {
    if let Some(&last_mid) = self.mid_history.back() {
        let ret = (new_mid - last_mid) / last_mid;
        self.mid_history.push_back(new_mid);
        
        // Обновляем суммы для O(1) расчета дисперсии
        self.sum_returns += ret;
        self.sum_returns_sq += ret * ret;

        if self.mid_history.len() > window + 1 {
            // Реализация удаления старого значения (нужно хранить сами returns или пересчитывать)
            // Для упрощения и точности на малых окнах допустим Vec::windows(2).map(...)
            self.mid_history.pop_front();
        }
    } else {
        self.mid_history.push_back(new_mid);
    }
}
```

## 3. Расчет объема в `calculate_order_size` [./src/trading/execution.rs](./src/trading/execution.rs)
Используй `rust_decimal` для финального объема:

```rust
fn calculate_scaled_size(&self, base_size: Decimal) -> Decimal {
    let current_vol = self.get_current_vol(); // Расчет StdDev в bps
    
    // Защита от деления на ноль и холодного старта
    let effective_vol = if self.mid_history.len() < self.config.volatility_window {
        self.config.volatility_default
    } else if current_vol < 1e-9 {
        1e-9
    } else {
        current_vol
    };

    let multiplier = (self.config.volatility_target_bps / effective_vol)
        .clamp(self.config.size_min_multiplier, self.config.size_max_multiplier);

    let multiplier_dec = Decimal::from_f64(multiplier).unwrap_or(Decimal::ONE);
    
    // Финальный объем с учетом шага лота биржи
    let scaled_size = base_size * multiplier_dec;
    round_to_lot_step(scaled_size, self.exchange_config.min_lot_step)
}
```

## 4. Связь с [./src/risk/risk_manager.rs](./src/risk/risk_manager.rs)
**Критически важно**: `Execution` должен передавать `scaled_size` в `RiskManager::can_open_position`. 
Если `RiskManager` возвращает `false` (превышение `max_pos_size`), `Execution` должен либо:
1.  Урезать объем до максимально допустимого остатка.
2.  Отменить сделку (в зависимости от настроек стратегии).

---

## Аргументация для спора с Grok (Planning Notes):
1.  **Типы**: Я полностью согласен с Grok по `Decimal`. Использование `f64` для цен в торговом ядре недопустимо.
2.  **Welford**: Я против классического Welford для **Sliding Window**, так как он требует хранения всех значений окна для корректного удаления "хвоста". Проще использовать `VecDeque` для хранения `returns` и обновлять суммы. Это даст те же $O(1)$ и будет чище.
3.  **Default Vol**: Обязательно добавляем `volatility_default` в [./config.toml](./config.toml), иначе первые 100 тиков бот будет выдавать непредсказуемый объем.
