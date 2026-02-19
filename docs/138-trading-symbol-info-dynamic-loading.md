# Задача 138: Динамическая загрузка параметров символа (v2.0)

## 1. Загрузка данных в [./src/trading/rest.rs](./src/trading/rest.rs)
Реализуй получение спецификаций контракта с поддержкой повторных попыток (retries) при ошибках сети или Rate Limit.

```rust
// В [./src/trading/rest.rs](./src/trading/rest.rs)
impl RestClient {
    pub async fn fetch_symbol_info(&self, symbol: &str) -> Result<SymbolInfo> {
        let endpoint = "/v5/market/instruments-info";
        let params = [("category", "linear"), ("symbol", symbol)];

        // Используем хелпер ретраев из задачи 085
        let resp = utils::retry_with_backoff(3, || async {
            self.get::<BybitResponse<InstrumentsInfo>>(endpoint, Some(&params)).await
        }).await?;

        let data = resp.result.list.first().ok_or("Symbol not found")?;
        
        // Парсим строки в f64, так как Bybit API возвращает числа в кавычках
        let tick_size = f64::from_str(&data.price_filter.tick_size)?;
        let qty_step = f64::from_str(&data.lot_size_filter.qty_step)?;
        
        // Рассчитываем количество знаков для форматирования цен
        let price_precision = if tick_size > 0.0 {
            (-tick_size.log10().floor()) as usize
        } else {
            8
        };

        Ok(SymbolInfo {
            lot_filter: LotFilter {
                min_qty: f64::from_str(&data.lot_size_filter.min_order_qty)?,
                max_qty: f64::from_str(&data.lot_size_filter.max_order_qty)?,
                qty_step,
            },
            price_filter: PriceFilter {
                tick_size,
                price_precision,
                min_price: f64::from_str(&data.price_filter.min_price)?,
                max_price: f64::from_str(&data.price_filter.max_price)?,
            },
        })
    }
}
```

## 2. Кэширование в `bots/SYMBOL/cache/symbol_info.json`
Чтобы бот мог стартовать при временных сбоях API, сохраняем последний успешный ответ на диск.

```rust
// В [./src/bin/run-bot.rs](./src/bin/run-bot.rs)
let cache_path = bot_path.join("cache").join("symbol_info.json");
std::fs::create_dir_all(cache_path.parent().unwrap())?;

let symbol_info = match rest_client.fetch_symbol_info(&config.symbol).await {
    Ok(info) => {
        // Сохраняем в кэш
        let f = File::create(&cache_path)?;
        serde_json::to_writer_pretty(f, &info)?;
        info
    }
    Err(e) => {
        tracing::warn!("Failed to fetch symbol info: {}. Trying cache...", e);
        let f = File::open(&cache_path).map_err(|_| "No cache available")?;
        serde_json::from_reader(f)?
    }
};
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

*   **Placement**: Согласен с Grok. Объединение рыночных данных и торговых запросов в `src/trading/rest.rs` логично, так как `SymbolInfo` напрямую влияет на параметры выставляемых ордеров.
*   **Precision Handling**: Bybit возвращает `tick_size` как `"0.5"`. Мы парсим это в `f64` и рассчитываем `price_precision` (например, `0.1` -> `1`, `0.001` -> `3`). Это позволит в `execution.rs` использовать `format!("{:.1}", price)` без ошибок.
*   **Clamping**: В логике `calculate_order_size` (задача 065) теперь ОБЯЗАТЕЛЬНО использовать `clamp_qty` из задачи 137, используя загруженные `min_qty` и `max_qty`.
*   **Retries**: Использование экспоненциального отката критично при запуске множества ботов одновременно, чтобы не поймать бан по IP от Bybit.

## 4. Инструкции для Gemini (Coder AI):
1.  **[./src/trading/rest.rs](./src/trading/rest.rs)**: Добавить метод `fetch_symbol_info` с парсингом `String -> f64`.
2.  **[./src/trading/types.rs](./src/trading/types.rs)**: Обновить структуры `LotFilter` и `PriceFilter`, добавив `price_precision`.
3.  **[./src/bin/run-bot.rs](./src/bin/run-bot.rs)**: Реализовать логику старта: Попытка API -> Fallback на JSON кэш -> Ошибка.
4.  **Integration**: Добавить логирование всех параметров контракта при старте (`min_qty`, `tick_size`, `max_leverage`).

**Результат**: Бот становится автономным и устойчивым к изменениям параметров контрактов на бирже, а также защищен от сбоев API при старте за счет локального кэша.
