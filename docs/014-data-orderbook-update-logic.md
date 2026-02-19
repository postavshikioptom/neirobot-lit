# 014 - Data Orderbook Update Logic

Цель задачи: Реализовать и протестировать логику обновления стакана (LOB) в src/data/orderbook.rs. Основной фокус на обеспечении целостности данных: игнорирование дельт до первого снапшота, обработка последовательности (last_update_id) и обязательная обрезка (pruning) стакана до 50 уровней для предотвращения утечек памяти и накопления устаревших цен.

Файлы для изменения:

src/data/orderbook.rs
Инструкции для Gemini:

Реализовать обновленную логику в apply_update: Использовать ссылку &OrderBookUpdate для исключения лишних аллокаций.

impl OrderBook {
    pub fn apply_update(&mut self, update: &OrderBookUpdate) {
        // 1. Снапшот сбрасывает состояние
        if update.is_snapshot {
            self.bids.clear();
            self.asks.clear();
        } else {
            // 2. Проверка: нельзя применять дельту, если еще не было снапшота
            if self.last_update_id == 0 {
                warn!("[{}] Received delta before first snapshot. Skipping.", self.symbol);
                return;
            }
            // 3. Проверка последовательности (Bybit: 'u' должен расти)
            if update.last_update_id <= self.last_update_id {
                warn!("[{}] Out-of-order update for {}: {} <= {}. Skipping.", 
                      self.symbol, self.symbol, update.last_update_id, self.last_update_id);
                return;
            }
        }

        self.last_update_id = update.last_update_id;
        self.timestamp_ms = update.timestamp_ms;

        // 4. Применение Bids
        for level in &update.bids {
            let price = OrderedFloat(level.price);
            if level.size == 0.0 {
                self.bids.remove(&price);
            } else {
                self.bids.insert(price, level.size);
            }
        }

        // 5. Применение Asks
        for level in &update.asks {
            let price = OrderedFloat(level.price);
            if level.size == 0.0 {
                self.asks.remove(&price);
            } else {
                self.asks.insert(price, level.size);
            }
        }

        // 6. Обрезка (Pruning): Оставляем только топ-50
        // BTreeMap сортирует по возрастанию. 
        // В Bids (покупка) нам нужны САМЫЕ ВЫСОКИЕ цены. Удаляем самые низкие (начало карты).
        while self.bids.len() > LOB_DEPTH {
            let first_key = *self.bids.keys().next().unwrap();
            self.bids.remove(&first_key);
        }
        // В Asks (продажа) нам нужны САМЫЕ НИЗКИЕ цены. Удаляем самые высокие (конец карты).
        while self.asks.len() > LOB_DEPTH {
            let last_key = *self.asks.keys().next_back().unwrap();
            self.asks.remove(&last_key);
        }
    }
}
Добавить Unit-тесты: Протестировать граничные случаи:

test_orderbook_snapshot: Загрузка снапшота и проверка get_flat_snapshot (наличие padding нулями, если уровней < 50).
test_orderbook_delta_update: Изменение объема на существующем уровне.
test_orderbook_delta_remove: Удаление уровня через size: 0.0.
test_orderbook_pruning: Добавление 60-го уровня и проверка, что в BTreeMap осталось ровно 50.
test_orderbook_out_of_order: Проверка, что дельта с меньшим id игнорируется.
Технические требования:

Использовать LOB_DEPTH = 50.
Обязательно warn! при пропуске данных, чтобы видеть проблемы с сетью в логах.
В тестах проверять порядок: bids должны идти от max цены к min, asks от min к max.
Почему это важно: Обрезка стакана (pruning) критична: без неё BTreeMap будет расти бесконечно, по мере того как цена движется по рынку, что приведет к деградации производительности и неверным данным в get_flat_snapshot (мы будем брать "хвосты" из истории вместо актуального края стакана).

Grok, 014 готова.

Добавлена обрезка while len > 50.
Добавлен скип дельт до снапшота.
Добавлен warn на out-of-order.