
# Задача 132: Периодический дамп снимков стакана (v2.0)

## 1. Структура снимка в `src/data/orderbook.rs`
Добавь структуру `OrderBookSnapshot` и поддержку «грязного» флага. Используем `f64` для цен и объемов для максимальной производительности при записи.

```rust
// В [./src/data/orderbook.rs](./src/data/orderbook.rs)
pub struct OrderBookSnapshot {
    pub timestamp: i64,
    pub symbol: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

impl OrderBook {
    // ... существующие поля
    // dirty: bool, // флаг изменений

    pub fn take_snapshot(&self) -> OrderBookSnapshot {
        OrderBookSnapshot {
            // Используем системный хелпер из 009
            timestamp: crate::utils::helpers::unix_ms(),
            symbol: self.symbol.clone(),
            // Копируем только TOP-50 уровней
            bids: self.bids.iter().take(50).map(|(p, l)| (*p as f64, l.volume as f64)).collect(),
            asks: self.asks.iter().take(50).map(|(p, l)| (*p as f64, l.volume as f64)).collect(),
        }
    }
}
```

## 2. Фоновый воркер (Background Writer)
Реализуй асинхронную задачу, которая принимает снимки через канал и сохраняет их. JSON перезаписывается (всегда актуальный стейт), Parquet — дополняется (append batches).

```rust
// В новом модуле [./src/data/dump.rs](./src/data/dump.rs) или [./src/data/parquet.rs](./src/data/parquet.rs)
pub async fn start_snapshot_writer(mut rx: mpsc::Receiver<OrderBookSnapshot>, bot_path: PathBuf) {
    let mut buffer: Vec<OrderBookSnapshot> = Vec::with_capacity(100);
    
    while let Some(snap) = rx.recv().await {
        // 1. Быстрое сохранение JSON для восстановления (0107)
        let json_path = bot_path.join("data/last_snapshot.json");
        if let Ok(f) = File::create(json_path) {
            serde_json::to_writer(f, &snap).ok();
        }

        // 2. Накопление батча для Parquet
        buffer.push(snap);
        if buffer.len() >= 100 {
            flush_to_parquet(&mut buffer, &bot_path).await;
        }
    }
}
```

## 3. Интеграция в основной цикл `run-bot.rs`
Логика проверки интервала и отправки данных должна находиться в главном цикле обработки сообщений, чтобы не блокировать WebSocket-поток.

```rust
// В [./src/bin/run-bot.rs](./src/bin/run-bot.rs)
let mut last_dump = 0;
let dump_interval = config.snapshot_interval_ms;

loop {
    // ... обработка сообщений
    let now = helpers::unix_ms();
    if now - last_dump > dump_interval && orderbook.is_dirty() {
        let snap = orderbook.take_snapshot();
        // Отправка в фоновый поток без блокировки
        let _ = snapshot_tx.try_send(snap); 
        orderbook.reset_dirty();
        last_dump = now;
    }
}
```

## 4. Спорные моменты и корректировки (Grok + Zencoder)

- **Decimal vs f64**: Согласен с Grok. Хотя внутри трейдинг-логики мы используем `Decimal` (задача 065), для дампов истории `f64` предпочтительнее: Polars в Python нативно работает с `float64`, и это экономит 20-30% времени при записи и чтении.
- **Timestamp**: Используем `utils::helpers::unix_ms()` (задача 009). Никакого `chrono` в асинхронном горячем цикле для минимизации оверхеда.
- **JSON Recovery**: Файл `last_snapshot.json` всегда содержит ОДИН последний снимок. Это позволяет боту при старте мгновенно «увидеть» спред, даже если WebSocket еще не прислал первый `depth` апдейт.
- **Parquet Append**: Используем `polars` для формирования `DataFrame` из накопленного буфера и записи через `zstd` компрессию (задача 017). Это обеспечит компактное хранение гигабайтов истории.

## 5. Инструкции для Gemini (Coder AI):
1. **[./src/data/orderbook.rs](./src/data/orderbook.rs)**: Добавить `OrderBookSnapshot`, флаг `dirty` и методы `take_snapshot`, `is_dirty`, `reset_dirty`.
2. **[./src/data/dump.rs](./src/data/dump.rs)**: Реализовать асинхронный воркер `start_snapshot_writer` с дублированием в JSON и Parquet.
3. **[./src/bin/run-bot.rs](./src/bin/run-bot.rs)**: Интегрировать проверку таймера и отправку в канал в основном цикле.
4. **[./src/config/types.rs](./src/config/types.rs)**: Добавить поле `snapshot_interval_ms` (u64) в `BotConfig`.

**Результат**: Надежная система сохранения данных, которая обеспечивает и выживаемость бота при сбоях, и накопление идеального датасета для обучения ML-моделей.
