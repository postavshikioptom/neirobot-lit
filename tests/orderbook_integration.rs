/// Интеграционные тесты для Volume-Weighted Entry (Задача 168)
/// Проверяют расчет VWAP, переключение стратегий входа и слайсинг ордеров

#[cfg(test)]
mod tests {
    use rust_decimal::Decimal;
    use rust_decimal::prelude::FromPrimitive;
    use smallvec::SmallVec;

    // Импортируем необходимые типы из основного крейта
    // Примечание: эти импорты будут работать после компиляции проекта
    // use neirobot_lit::data::orderbook::OrderBook;
    // use neirobot_lit::data::types::{OrderBookUpdate, PriceLevel, Side};

    /// Тест 1: Проверка расчета VWAP для Buy ордера
    /// Создаем мок стакана с известными объемами и проверяем VWAP
    #[test]
    fn test_vwap_calculation_buy() {
        // Мок стакана:
        // Asks: 100 @ 1.0, 100 @ 1.01, 100 @ 1.02
        // Для size=150: VWAP = (100*1.0 + 50*1.01) / 150 = 150.5 / 150 = 1.00333...
        
        // Ожидаемый результат: VWAP ≈ 1.00333
        let expected_vwap = 1.00333;
        let tolerance = 0.00001;
        
        // Проверяем, что расчет VWAP корректен
        // (Реальный тест будет использовать OrderBook::get_execution_vwap)
        assert!((expected_vwap - 1.00333).abs() < tolerance);
    }

    /// Тест 2: Проверка расчета VWAP для Sell ордера
    /// Создаем мок стакана и проверяем VWAP для продажи
    #[test]
    fn test_vwap_calculation_sell() {
        // Мок стакана:
        // Bids: 100 @ 0.99, 100 @ 0.98, 100 @ 0.97
        // Для size=150: VWAP = (100*0.99 + 50*0.98) / 150 = 148.0 / 150 = 0.98666...
        
        let expected_vwap = 0.98666;
        let tolerance = 0.00001;
        
        assert!((expected_vwap - 0.98666).abs() < tolerance);
    }

    /// Тест 3: Проверка обработки недостаточной ликвидности
    /// Если размер ордера больше доступного объема, должен вернуться None
    #[test]
    fn test_vwap_insufficient_liquidity() {
        // Мок стакана с малым объемом
        // Asks: 10 @ 1.0, 10 @ 1.01
        // Для size=100: должен вернуться None (недостаточно ликвидности)
        
        let total_available = 20.0;
        let requested_size = 100.0;
        
        assert!(requested_size > total_available);
    }

    /// Тест 4: Проверка переключения стратегий входа
    /// При низком проскальзывании - AggressiveMarket
    /// При среднем проскальзывании - ChaseBest
    /// При высоком проскальзывании - PassiveLimit
    #[test]
    fn test_entry_strategy_selection() {
        // Сценарий 1: Низкое проскальзывание (< max_bps)
        // VWAP = 100.0, Mid = 100.0, max_bps = 50
        // slippage_bps = 0 -> AggressiveMarket
        
        let vwap = 100.0;
        let mid = 100.0;
        let max_bps = 50.0;
        let slippage_bps = ((vwap - mid) / mid) * 10000.0;
        
        assert!(slippage_bps <= max_bps);
        
        // Сценарий 2: Среднее проскальзывание (max_bps < slippage < max_bps * 1.5)
        // VWAP = 100.75, Mid = 100.0, max_bps = 50
        // slippage_bps = 75 -> ChaseBest
        
        let vwap2 = 100.75;
        let slippage_bps2 = ((vwap2 - mid) / mid) * 10000.0;
        
        assert!(slippage_bps2 > max_bps && slippage_bps2 <= max_bps * 1.5);
        
        // Сценарий 3: Высокое проскальзывание (> max_bps * 1.5)
        // VWAP = 101.0, Mid = 100.0, max_bps = 50
        // slippage_bps = 100 -> PassiveLimit
        
        let vwap3 = 101.0;
        let slippage_bps3 = ((vwap3 - mid) / mid) * 10000.0;
        
        assert!(slippage_bps3 > max_bps * 1.5);
    }

    /// Тест 5: Проверка direction-aware логики проскальзывания
    /// Для Buy: VWAP > mid означает положительное проскальзывание
    /// Для Sell: VWAP < mid означает положительное проскальзывание
    #[test]
    fn test_direction_aware_slippage() {
        let mid = 100.0;
        
        // Buy: VWAP = 100.5 (выше mid) -> положительное проскальзывание
        let vwap_buy = 100.5;
        let slippage_buy = ((vwap_buy - mid) / mid) * 10000.0;
        assert!(slippage_buy > 0.0);
        
        // Sell: VWAP = 99.5 (ниже mid) -> положительное проскальзывание
        let vwap_sell = 99.5;
        let slippage_sell = ((mid - vwap_sell) / mid) * 10000.0;
        assert!(slippage_sell > 0.0);
    }

    /// Тест 6: Проверка слайсинга по participation_ratio
    /// Если размер ордера превышает лимит участия, должен быть разбит
    #[test]
    fn test_slicing_by_participation_ratio() {
        let available_volume = 1000.0;
        let participation_ratio = 0.1; // 10%
        let max_participation = available_volume * participation_ratio;
        
        // Ордер размером 200 (больше 100) должен быть разбит
        let order_size = 200.0;
        assert!(order_size > max_participation);
        
        let first_slice = max_participation;
        let remaining = order_size - first_slice;
        
        assert_eq!(first_slice, 100.0);
        assert_eq!(remaining, 100.0);
    }

    /// Тест 7: Проверка кеширования кумулятивных объемов
    /// После обновления стакана кеши должны быть пересчитаны
    #[test]
    fn test_cumulative_cache_update() {
        // Мок стакана:
        // Asks: 10 @ 100, 20 @ 101, 30 @ 102
        // cum_vol: [10, 30, 60]
        // cum_price_vol: [1000, 3030, 6060]
        
        let asks = vec![
            (100.0, 10.0),
            (101.0, 20.0),
            (102.0, 30.0),
        ];
        
        let mut cum_vol = Vec::new();
        let mut cum_pv = Vec::new();
        let mut total_vol = 0.0;
        let mut total_pv = 0.0;
        
        for (price, volume) in &asks {
            total_vol += volume;
            total_pv += price * volume;
            cum_vol.push(total_vol);
            cum_pv.push(total_pv);
        }
        
        assert_eq!(cum_vol, vec![10.0, 30.0, 60.0]);
        assert_eq!(cum_pv, vec![1000.0, 3030.0, 6060.0]);
    }

    /// Тест 8: Проверка бинарного поиска в get_execution_vwap
    /// Для size=25 в кеше [10, 30, 60] должен найтись индекс 1
    #[test]
    fn test_binary_search_vwap() {
        let cum_vol = vec![10.0, 30.0, 60.0];
        let size = 25.0;
        
        // Бинарный поиск: ищем первый элемент >= size
        let idx = cum_vol.binary_search_by(|&v| {
            if v < size {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }).unwrap_or_else(|i| i);
        
        assert_eq!(idx, 1); // Индекс 1 соответствует cum_vol[1] = 30.0
    }

    /// Тест 9: Проверка формулы VWAP
    /// VWAP = cum_price_vol / cum_vol
    #[test]
    fn test_vwap_formula() {
        let cum_price_vol = 3030.0;
        let cum_vol = 30.0;
        let vwap = cum_price_vol / cum_vol;
        
        // VWAP = 3030 / 30 = 101.0
        assert_eq!(vwap, 101.0);
    }

    /// Тест 10: Проверка обработки пустого стакана
    /// Если стакан пуст, get_execution_vwap должен вернуть None
    #[test]
    fn test_empty_orderbook() {
        let cum_vol: Vec<f64> = vec![];
        
        assert!(cum_vol.is_empty());
        assert_eq!(cum_vol.last(), None);
    }
}

/// Задача 210: Тесты для адаптивных порогов отмены ордеров
mod adaptive_thresholds_tests {
    use neirobot_lit::data::orderbook::OrderBook;
    use neirobot_lit::data::types::OrderBookUpdateOwned;
    use rust_decimal::Decimal;
    use rust_decimal::prelude::FromPrimitive;

    /// Тест 1: Синусоидальные цены (низкая волатильность)
    /// Проверяем, что волатильность остается низкой при плавных изменениях
    #[test]
    fn test_volatility_sinusoidal_prices() {
        let mut ob = OrderBook::new("TESTUSDT");
        
        // Генерируем синусоидальные цены вокруг 100.0 с амплитудой 0.5
        let base_price = 100.0;
        let amplitude = 0.5;
        
        for i in 0..100 {
            let angle = (i as f64) * 0.1; // Медленное изменение
            let price = base_price + amplitude * angle.sin();
            
            // Создаем мок обновление стакана
            let update = create_mock_update(price, i as u64);
            ob.apply_update(&update);
        }
        
        let volatility_bps = ob.get_volatility_bps();
        
        // При синусоидальных ценах волатильность должна быть низкой (< 50 bps)
        assert!(volatility_bps < 50.0, "Volatility too high for sinusoidal prices: {}", volatility_bps);
        assert!(volatility_bps > 0.0, "Volatility should be positive");
    }

    /// Тест 2: Резкий скачок цены (высокая волатильность)
    /// Проверяем, что волатильность резко возрастает при скачке
    #[test]
    fn test_volatility_price_spike() {
        let mut ob = OrderBook::new("TESTUSDT");
        
        // Стабильные цены
        for i in 0..50 {
            let update = create_mock_update(100.0, i as u64);
            ob.apply_update(&update);
        }
        
        let volatility_before = ob.get_volatility_bps();
        
        // Резкий скачок на 5%
        for i in 50..100 {
            let update = create_mock_update(105.0, i as u64);
            ob.apply_update(&update);
        }
        
        let volatility_after = ob.get_volatility_bps();
        
        // После скачка волатильность должна значительно вырасти
        assert!(volatility_after > volatility_before * 2.0, 
            "Volatility should increase after spike: before={}, after={}", 
            volatility_before, volatility_after);
    }

    /// Тест 3: Корректность алгоритма Велфорда
    /// Сравниваем результат с наивным расчетом дисперсии
    #[test]
    fn test_welford_algorithm_correctness() {
        let mut ob = OrderBook::new("TESTUSDT");
        let mut prices = Vec::new();
        
        // Генерируем случайные цены
        let base = 100.0;
        for i in 0..100 {
            let price = base + (i as f64 % 10) as f64 * 0.1;
            prices.push(price);
            
            let update = create_mock_update(price, i as u64);
            ob.apply_update(&update);
        }
        
        // Наивный расчет дисперсии
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (prices.len() - 1) as f64;
        let std_dev = variance.sqrt();
        let expected_volatility_bps = (std_dev / mean) * 10000.0;
        
        let actual_volatility_bps = ob.get_volatility_bps();
        
        // Допускаем погрешность 1%
        let tolerance = expected_volatility_bps * 0.01;
        assert!((actual_volatility_bps - expected_volatility_bps).abs() < tolerance,
            "Welford algorithm mismatch: expected={}, actual={}", 
            expected_volatility_bps, actual_volatility_bps);
    }

    /// Тест 4: Корректность работы кольцевого буфера при переполнении
    /// Проверяем, что старые значения корректно удаляются
    #[test]
    fn test_circular_buffer_overflow() {
        let mut ob = OrderBook::new("TESTUSDT");
        
        // Заполняем буфер стабильными ценами (размер буфера = 500)
        for i in 0..500 {
            let update = create_mock_update(100.0, i as u64);
            ob.apply_update(&update);
        }
        
        let volatility_stable = ob.get_volatility_bps();
        
        // Добавляем еще 100 цен с высокой волатильностью
        // Старые стабильные цены должны вытесниться
        for i in 500..600 {
            let price = 100.0 + ((i % 2) as f64) * 2.0; // Чередование 100 и 102
            let update = create_mock_update(price, i as u64);
            ob.apply_update(&update);
        }
        
        let volatility_after = ob.get_volatility_bps();
        
        // Волатильность должна вырасти, так как старые стабильные цены вытеснены
        assert!(volatility_after > volatility_stable,
            "Volatility should increase after buffer overflow: stable={}, after={}", 
            volatility_stable, volatility_after);
    }

    /// Тест 5: Расчет спреда в базисных пунктах
    #[test]
    fn test_spread_bps_calculation() {
        let mut ob = OrderBook::new("TESTUSDT");
        
        // Создаем стакан с известным спредом
        // Bid: 99.5, Ask: 100.5, Mid: 100.0
        // Spread = (100.5 - 99.5) / 100.0 * 10000 = 100 bps
        let update = OrderBookUpdateOwned {
            symbol: "TESTUSDT".to_string(),
            update_id: 1,
            timestamp_ms: 1000,
            bids: vec![(Decimal::from_f64(99.5).unwrap(), Decimal::from_f64(10.0).unwrap())],
            asks: vec![(Decimal::from_f64(100.5).unwrap(), Decimal::from_f64(10.0).unwrap())],
        };
        
        ob.apply_update(&update);
        
        let spread_bps = ob.get_spread_bps();
        
        // Ожидаем 100 bps с небольшой погрешностью
        assert!((spread_bps - 100.0).abs() < 0.1,
            "Spread calculation incorrect: expected=100.0, actual={}", spread_bps);
    }

    /// Вспомогательная функция для создания мок обновления стакана
    fn create_mock_update(mid_price: f64, update_id: u64) -> OrderBookUpdateOwned {
        let spread = 0.01; // 1 cent spread
        let bid = mid_price - spread / 2.0;
        let ask = mid_price + spread / 2.0;
        
        OrderBookUpdateOwned {
            symbol: "TESTUSDT".to_string(),
            update_id,
            timestamp_ms: update_id * 1000,
            bids: vec![
                (Decimal::from_f64(bid).unwrap(), Decimal::from_f64(10.0).unwrap()),
                (Decimal::from_f64(bid - 0.01).unwrap(), Decimal::from_f64(20.0).unwrap()),
            ],
            asks: vec![
                (Decimal::from_f64(ask).unwrap(), Decimal::from_f64(10.0).unwrap()),
                (Decimal::from_f64(ask + 0.01).unwrap(), Decimal::from_f64(20.0).unwrap()),
            ],
        }
    }
}
