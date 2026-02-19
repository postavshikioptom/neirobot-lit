use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use neirobot_lit::trading::position_manager::PositionManager;
use neirobot_lit::trading::types::{OrderSide, FillEvent};

// Вспомогательная функция для создания FillEvent
fn create_fill(side: OrderSide, qty: Decimal, price: Decimal) -> FillEvent {
    FillEvent {
        symbol: "BTCUSDT".to_string(),
        side,
        exec_qty: qty,
        exec_price: price,
        exec_fee: dec!(0),
        is_maker: true,
        exec_id: "test_exec_id".to_string(),
        order_id: "test_order_id".to_string(),
        order_link_id: Some("test_link_id".to_string()),
        timestamp: 1000000,
    }
}

// Вспомогательная функция для создания FillEvent с комиссией
fn create_fill_with_fee(side: OrderSide, qty: Decimal, price: Decimal, fee: Decimal) -> FillEvent {
    FillEvent {
        symbol: "BTCUSDT".to_string(),
        side,
        exec_qty: qty,
        exec_price: price,
        exec_fee: fee,
        is_maker: false,
        exec_id: "test_exec_id".to_string(),
        order_id: "test_order_id".to_string(),
        order_link_id: Some("test_link_id".to_string()),
        timestamp: 1000000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_averaging_down_long() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Первая покупка: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(1.0));
        assert_eq!(pos.avg_price, dec!(100.0));
        
        // Вторая покупка: 1.0 @ 90.0 (усреднение вниз)
        let fill2 = create_fill(OrderSide::Buy, dec!(1.0), dec!(90.0));
        manager.update_from_fill(fill2);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(2.0));
        assert_eq!(pos.avg_price, dec!(95.0)); // (100 + 90) / 2 = 95
    }

    #[tokio::test]
    async fn test_averaging_up_short() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Первая продажа: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Sell, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(-1.0));
        assert_eq!(pos.avg_price, dec!(100.0));
        
        // Вторая продажа: 1.0 @ 110.0 (усреднение вверх в шорт)
        let fill2 = create_fill(OrderSide::Sell, dec!(1.0), dec!(110.0));
        manager.update_from_fill(fill2);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(-2.0));
        assert_eq!(pos.avg_price, dec!(105.0)); // (100 + 110) / 2 = 105
    }

    #[tokio::test]
    async fn test_position_flip_long_to_short() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Long: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(1.0));
        assert_eq!(pos.avg_price, dec!(100.0));
        
        // Переворот: Sell 2.5 @ 90.0
        // Закрываем 1.0 Long с убытком: (90 - 100) * 1.0 = -10.0
        // Открываем 1.5 Short @ 90.0
        let fill2 = create_fill(OrderSide::Sell, dec!(2.5), dec!(90.0));
        let realized_pnl = manager.update_from_fill(fill2);
        
        assert!(realized_pnl.is_some());
        assert_eq!(realized_pnl.unwrap(), dec!(-10.0)); // Убыток по закрытому лонгу
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(-1.5)); // Short 1.5
        assert_eq!(pos.avg_price, dec!(90.0)); // Цена входа для нового шорта
        assert_eq!(pos.realized_pnl, dec!(-10.0));
    }

    #[tokio::test]
    async fn test_position_flip_short_to_long() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Short: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Sell, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(-1.0));
        assert_eq!(pos.avg_price, dec!(100.0));
        
        // Переворот: Buy 2.5 @ 110.0
        // Закрываем 1.0 Short с убытком: (110 - 100) * 1.0 * -1 = -10.0
        // Открываем 1.5 Long @ 110.0
        let fill2 = create_fill(OrderSide::Buy, dec!(2.5), dec!(110.0));
        let realized_pnl = manager.update_from_fill(fill2);
        
        assert!(realized_pnl.is_some());
        assert_eq!(realized_pnl.unwrap(), dec!(-10.0)); // Убыток по закрытому шорту
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(1.5)); // Long 1.5
        assert_eq!(pos.avg_price, dec!(110.0)); // Цена входа для нового лонга
        assert_eq!(pos.realized_pnl, dec!(-10.0));
    }

    #[tokio::test]
    async fn test_partial_close_long() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Long: 2.0 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(2.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        // Частичное закрытие: Sell 1.0 @ 110.0
        // PnL: (110 - 100) * 1.0 = +10.0
        let fill2 = create_fill(OrderSide::Sell, dec!(1.0), dec!(110.0));
        let realized_pnl = manager.update_from_fill(fill2);
        
        assert!(realized_pnl.is_some());
        assert_eq!(realized_pnl.unwrap(), dec!(10.0)); // Прибыль
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(1.0)); // Остался 1.0 Long
        assert_eq!(pos.avg_price, dec!(100.0)); // Средняя цена не меняется при закрытии
        assert_eq!(pos.realized_pnl, dec!(10.0));
    }

    #[tokio::test]
    async fn test_full_close_long() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Long: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        // Полное закрытие: Sell 1.0 @ 105.0
        // PnL: (105 - 100) * 1.0 = +5.0
        let fill2 = create_fill(OrderSide::Sell, dec!(1.0), dec!(105.0));
        let realized_pnl = manager.update_from_fill(fill2);
        
        assert!(realized_pnl.is_some());
        assert_eq!(realized_pnl.unwrap(), dec!(5.0));
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(0)); // Flat
        assert_eq!(pos.avg_price, dec!(0)); // Сброшена при полном закрытии
        assert_eq!(pos.realized_pnl, dec!(5.0));
    }

    #[tokio::test]
    async fn test_leveraged_unrealized_pnl() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(10), dec!(0.001)); // 10x leverage
        
        // Открываем Long: 1.0 @ 100.0
        let fill = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill);
        
        // Обновляем unrealized PnL при mid_price = 105.0
        manager.update_unrealized_pnl(dec!(105.0), dec!(105.0));
        
        let pos = manager.get_position();
        
        // Nominal PnL: (105 - 100) * 1.0 = +5.0
        assert_eq!(pos.unrealized_pnl, dec!(5.0));
        
        // Leveraged ROI: 
        // entry_value = 100 * 1.0 = 100
        // entry_margin = 100 / 10 = 10
        // ROI = (5 / 10) * 100 = 50%
        assert_eq!(pos.unrealized_pnl_pct, dec!(50.0));
    }

    #[tokio::test]
    async fn test_leveraged_unrealized_pnl_short() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(10), dec!(0.001)); // 10x leverage
        
        // Открываем Short: 1.0 @ 100.0
        let fill = create_fill(OrderSide::Sell, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill);
        
        // Обновляем unrealized PnL при mid_price = 95.0
        manager.update_unrealized_pnl(dec!(95.0), dec!(95.0));
        
        let pos = manager.get_position();
        
        // Nominal PnL: (95 - 100) * -1.0 = +5.0
        assert_eq!(pos.unrealized_pnl, dec!(5.0));
        
        // Leveraged ROI: 
        // entry_value = 100 * 1.0 = 100
        // entry_margin = 100 / 10 = 10
        // ROI = (5 / 10) * 100 = 50%
        assert_eq!(pos.unrealized_pnl_pct, dec!(50.0));
    }

    #[tokio::test]
    async fn test_unrealized_pnl_flat_position() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(10), dec!(0.001));
        
        // Без позиции
        manager.update_unrealized_pnl(dec!(100.0), dec!(100.0));
        
        let pos = manager.get_position();
        assert_eq!(pos.unrealized_pnl, dec!(0));
        assert_eq!(pos.unrealized_pnl_pct, dec!(0));
        assert_eq!(pos.mark_pnl, dec!(0));
    }

    #[tokio::test]
    async fn test_multiple_fills_accumulation() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Fill 1: Buy 0.5 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(0.5), dec!(100.0));
        manager.update_from_fill(fill1);
        
        // Fill 2: Buy 0.3 @ 105.0
        let fill2 = create_fill(OrderSide::Buy, dec!(0.3), dec!(105.0));
        manager.update_from_fill(fill2);
        
        // Fill 3: Buy 0.2 @ 110.0
        let fill3 = create_fill(OrderSide::Buy, dec!(0.2), dec!(110.0));
        manager.update_from_fill(fill3);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(1.0)); // 0.5 + 0.3 + 0.2
        
        // Средневзвешенная цена: (0.5*100 + 0.3*105 + 0.2*110) / 1.0
        // = (50 + 31.5 + 22) / 1.0 = 103.5
        assert_eq!(pos.avg_price, dec!(103.5));
    }

    #[tokio::test]
    async fn test_realized_pnl_accumulation() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Long: 2.0 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(2.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        // Закрываем частично: Sell 1.0 @ 110.0 (PnL = +10)
        let fill2 = create_fill(OrderSide::Sell, dec!(1.0), dec!(110.0));
        let pnl1 = manager.update_from_fill(fill2);
        assert_eq!(pnl1.unwrap(), dec!(10.0));
        
        // Закрываем остаток: Sell 1.0 @ 105.0 (PnL = +5)
        let fill3 = create_fill(OrderSide::Sell, dec!(1.0), dec!(105.0));
        let pnl2 = manager.update_from_fill(fill3);
        assert_eq!(pnl2.unwrap(), dec!(5.0));
        
        let pos = manager.get_position();
        assert_eq!(pos.realized_pnl, dec!(15.0)); // 10 + 5
        assert_eq!(pos.qty, dec!(0)); // Flat
    }

    #[tokio::test]
    async fn test_loss_scenario() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Long: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        // Закрываем с убытком: Sell 1.0 @ 90.0
        // PnL: (90 - 100) * 1.0 = -10.0
        let fill2 = create_fill(OrderSide::Sell, dec!(1.0), dec!(90.0));
        let realized_pnl = manager.update_from_fill(fill2);
        
        assert!(realized_pnl.is_some());
        assert_eq!(realized_pnl.unwrap(), dec!(-10.0));
        
        let pos = manager.get_position();
        assert_eq!(pos.realized_pnl, dec!(-10.0));
    }

    #[tokio::test]
    async fn test_short_profit_scenario() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Short: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Sell, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        // Закрываем с прибылью: Buy 1.0 @ 90.0
        // PnL: (90 - 100) * 1.0 * -1 = +10.0
        let fill2 = create_fill(OrderSide::Buy, dec!(1.0), dec!(90.0));
        let realized_pnl = manager.update_from_fill(fill2);
        
        assert!(realized_pnl.is_some());
        assert_eq!(realized_pnl.unwrap(), dec!(10.0));
        
        let pos = manager.get_position();
        assert_eq!(pos.realized_pnl, dec!(10.0));
    }

    #[tokio::test]
    async fn test_leverage_change() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(5), dec!(0.001));
        
        // Открываем позицию с 5x
        let fill = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill);
        
        manager.update_unrealized_pnl(dec!(110.0), dec!(110.0));
        let pos = manager.get_position();
        
        // ROI с 5x: entry_margin = 100/5 = 20, PnL = 10, ROI = 50%
        assert_eq!(pos.unrealized_pnl_pct, dec!(50.0));
        
        // Меняем плечо на 10x
        manager.set_leverage(dec!(10));
        manager.update_unrealized_pnl(dec!(110.0), dec!(110.0));
        let pos = manager.get_position();
        
        // ROI с 10x: entry_margin = 100/10 = 10, PnL = 10, ROI = 100%
        assert_eq!(pos.unrealized_pnl_pct, dec!(100.0));
    }

    #[tokio::test]
    async fn test_zero_leverage_edge_case() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(0), dec!(0.001)); // Без плеча
        
        let fill = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill);
        
        manager.update_unrealized_pnl(dec!(110.0), dec!(110.0));
        let pos = manager.get_position();
        
        // При leverage = 0, entry_margin = entry_value
        // ROI = (10 / 100) * 100 = 10%
        assert_eq!(pos.unrealized_pnl, dec!(10.0));
        assert_eq!(pos.unrealized_pnl_pct, dec!(10.0));
    }

    #[tokio::test]
    async fn test_dust_cleanup() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001)); // min_qty_step = 0.001
        
        // Открываем Long: 1.0 @ 100.0
        let fill1 = create_fill(OrderSide::Buy, dec!(1.0), dec!(100.0));
        manager.update_from_fill(fill1);
        
        let pos = manager.get_position();
        assert_eq!(pos.qty, dec!(1.0));
        
        // Закрываем почти всё: Sell 0.9999 @ 100.0
        // Остаток: 1.0 - 0.9999 = 0.0001 < 0.001 (min_qty_step)
        let fill2 = create_fill(OrderSide::Sell, dec!(0.9999), dec!(100.0));
        manager.update_from_fill(fill2);
        
        let pos = manager.get_position();
        // Dust cleanup должен сработать
        assert_eq!(pos.qty, dec!(0)); // Должно стать Flat
        assert_eq!(pos.avg_price, dec!(0)); // Сброшена
    }

    #[tokio::test]
    async fn test_fee_accounting() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Long с комиссией: 1.0 @ 100.0, fee = 0.055 (Taker 0.055%)
        let fill = create_fill_with_fee(OrderSide::Buy, dec!(1.0), dec!(100.0), dec!(0.055));
        manager.update_from_fill(fill);
        
        let pos = manager.get_position();
        // Комиссия сразу уменьшает realized_pnl
        assert_eq!(pos.realized_pnl, dec!(-0.055));
        assert_eq!(pos.qty, dec!(1.0));
        assert_eq!(pos.avg_price, dec!(100.0));
    }

    #[tokio::test]
    async fn test_fee_accumulation() {
        let mut manager = PositionManager::new("BTCUSDT".to_string(), dec!(1), dec!(0.001));
        
        // Открываем Long с комиссией
        let fill1 = create_fill_with_fee(OrderSide::Buy, dec!(1.0), dec!(100.0), dec!(0.055));
        manager.update_from_fill(fill1);
        
        // Закрываем с комиссией и прибылью
        // PnL от сделки: (110 - 100) * 1.0 = +10.0
        // Комиссия: -0.06
        // Итого: -0.055 (от входа) + 10.0 (от закрытия) - 0.06 (от закрытия) = 9.885
        let fill2 = create_fill_with_fee(OrderSide::Sell, dec!(1.0), dec!(110.0), dec!(0.06));
        let pnl = manager.update_from_fill(fill2);
        
        assert_eq!(pnl.unwrap(), dec!(10.0)); // PnL от сделки
        
        let pos = manager.get_position();
        // Общий realized_pnl = -0.055 - 0.06 + 10.0 = 9.885
        assert_eq!(pos.realized_pnl, dec!(9.885));
    }
}
