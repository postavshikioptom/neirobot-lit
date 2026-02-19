# Задача 107: Персистентность состояния торгового движка (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Добавь в `BotConfig` параметр допустимого расхождения:
```rust
pub struct BotConfig {
    // ...
    pub desync_tolerance_pct: f64, // Например, 0.01 (1%)
}
```

## 2. Определение типов в `src/trading/types.rs`
Добавь структуру состояния:
```rust
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct BotState {
    pub symbol: String,
    pub position_size: Decimal,
    pub avg_price: Decimal,
    pub cumulative_pnl: Decimal,
    pub active_orders: HashMap<String, OrderInfo>, // link_id -> info
    pub last_update_ts: i64,
}
```

## 3. Модуль управления состоянием `src/trading/state.rs`
Реализуй функции сохранения и загрузки с обработкой ошибок и атомарной записью:

```rust
pub fn save_state(state: &BotState, path: &Path) -> anyhow::Result<()> {
    let tmp_path = path.with_extension("tmp");
    let file = File::create(&tmp_path).map_err(|e| anyhow!("Failed to create tmp file: {}", e))?;
    
    // Пишем красиво для дебага
    serde_json::to_writer_pretty(file, state)?;
    
    // Атомарный перенос
    fs::rename(tmp_path, path).map_err(|e| anyhow!("Failed to rename state file: {}", e))?;
    Ok(())
}

pub fn load_state(path: &Path) -> anyhow::Result<BotState> {
    if !path.exists() {
        return Ok(BotState::default());
    }
    let file = File::open(path)?;
    let state = serde_json::from_reader(file)?;
    Ok(state)
}
```

## 4. Логика синхронизации (Reconciliation) в `src/trading/execution.rs`
При старте (метод `init` или `sync`):
1.  Загрузи `local_state` через `load_state`.
2.  Получи `exchange_state` через REST API Bybit.
3.  **Сверь данные**:
```rust
let diff = (local.position_size - ex.position_size).abs();
let tolerance = Decimal::from_f64(config.desync_tolerance_pct).unwrap_or(Decimal::ZERO);

if ex.position_size != Decimal::ZERO && (diff / ex.position_size) > tolerance {
    tracing::error!("CRITICAL DESYNC: Local {} vs Exchange {}. STOPPING.", local.position_size, ex.position_size);
    self.emergency_mode = true; // Блокировка торговли
} else {
    // Мягкая синхронизация: обновляем локальный стейт данными биржи
    self.state.position_size = ex.position_size;
    self.state.avg_price = ex.avg_price;
    save_state(&self.state, &path)?;
}
```

## 5. Обработка событий
- **Fill/Cancel**: После каждого изменения `position_size` или `active_orders` вызывай `save_state`.
- **Emergency Mode**: Если `emergency_mode == true`, метод `on_signal` должен немедленно возвращать `Error` или игнорировать сигналы до ручного сброса (удаления `state.json`).

---

## Аргументация для Планировщика:
1.  **Типы**: `types.rs` — правильное место, это упрощает импорты.
2.  **Безопасность**: `emergency_mode` предотвращает "войну" бота с биржей при багах API.
3.  **Атомарность**: Использование `.tmp` + `rename` исключает повреждение JSON при внезапном отключении питания.

**Gemini, реализуй эту логику, обеспечив логирование всех расхождений в `bots/{SYMBOL}/logs/bot.log`.**