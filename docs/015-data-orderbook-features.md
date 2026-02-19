# 015 - Data Orderbook Features

Цель задачи: Расширить функционал структуры OrderBook методами для извлечения ключевых рыночных метрик: средней цены (Mid Price), спреда и плоского вектора уровней (Top-N). Эти методы станут основой для формирования входного тензора нейросети и расчета целевых переменных (labels) при обучении.

Файлы для изменения:

src/data/orderbook.rs
Инструкции для Gemini:

Добавить методы в impl OrderBook: Реализовать вычисления с сохранением высокой точности (f64) для цен и оптимизацией (f32) для векторов.

impl OrderBook {
    /// Возвращает Mid Price (среднее между лучшим Ask и лучшим Bid).
    /// Используется f64 для точности при расчете доходностей (returns).
    pub fn get_mid_price(&self) -> f64 {
        let best_bid = self.bids.keys().next_back().map(|p| p.0);
        let best_ask = self.asks.keys().next().map(|p| p.0);

        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => (bid + ask) / 2.0,
            _ => {
                tracing::debug!("Empty book for {} - mid=0.0", self.symbol);
                0.0
            }
        }
    }

    /// Возвращает спред (разница между лучшим Ask и лучшим Bid).
    pub fn get_spread(&self) -> f64 {
        let best_bid = self.bids.keys().next_back().map(|p| p.0);
        let best_ask = self.asks.keys().next().map(|p| p.0);

        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => ask - bid,
            _ => 0.0,
        }
    }

    /// Возвращает топ N уровней в плоском формате (f32) для тензора.
    /// Порядок: Asks (p_0, v_0, ... p_N, v_N), затем Bids (p_0, v_0, ... p_N, v_N).
    /// Если уровней меньше N, заполняет остаток нулями (0.0).
    pub fn get_top_n(&self, n: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * 4);

        // Asks: от лучшей цены (min) к худшей
        let mut ask_iter = self.asks.iter().take(n);
        for _ in 0..n {
            if let Some((&price, &size)) = ask_iter.next() {
                data.push(price.0 as f32);
                data.push(size as f32);
            } else {
                data.push(0.0); // Price padding
                data.push(0.0); // Size padding
            }
        }

        // Bids: от лучшей цены (max) к худшей
        let mut bid_iter = self.bids.iter().rev().take(n);
        for _ in 0..n {
            if let Some((&price, &size)) = bid_iter.next() {
                data.push(price.0 as f32);
                data.push(size as f32);
            } else {
                data.push(0.0);
                data.push(0.0);
            }
        }

        data
    }
}
Добавить Unit-тесты: В блоке #[cfg(test)] реализовать проверку:

#[test]
fn test_orderbook_features() {
    let mut ob = OrderBook::new("BTCUSDT");
    ob.apply_update(&OrderBookUpdate {
        symbol: "BTCUSDT".into(),
        timestamp_ms: 1000,
        last_update_id: 1,
        is_snapshot: true,
        bids: vec![PriceLevel { price: 100.0, size: 1.0 }],
        asks: vec![PriceLevel { price: 101.0, size: 2.0 }],
    });

    assert_eq!(ob.get_mid_price(), 100.5);
    assert_eq!(ob.get_spread(), 1.0);
    
    let top_1 = ob.get_top_n(1);
    assert_eq!(top_1.len(), 4);
    assert_eq!(top_1[0], 101.0); // Ask price
    assert_eq!(top_1[2], 100.0); // Bid price
}
Технические требования:

f64 для внутренних расчетов цен.
f32 для экспорта данных (соответствует схеме Parquet и входу нейросети).
Обязательный padding нулями для сохранения фиксированного размера вектора.
Почему это важно: Метод get_top_n позволяет гибко менять размер входного окна (например, обучать на 20 уровнях, даже если мы собираем 50). Mid Price является критически важным для маркировки данных (labeling), так как именно его изменение предсказывает модель.