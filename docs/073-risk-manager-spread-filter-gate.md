Задача 073: Реализация риск-гейта Spread Filter (Wide Spread Protection)
Цель: Предотвратить вход в сделки при низкой ликвидности, инвертированном спреде или аномальном расширении спреда (Wide Spread), чтобы минимизировать издержки (slippage).

1. Изменения в ./src/config/types.rs
RiskConfig: Добавить параметр максимально допустимого спреда.
pub struct RiskConfig {
    // ...
    pub max_spread_bps: Option<u32>, // Макс. спред в базисных пунктах (1 bps = 0.01%)
}
2. Изменения в ./src/risk/risk_manager.rs
Метод check_spread_gate(best_bid: Decimal, bid_vol: Decimal, best_ask: Decimal, ask_vol: Decimal) -> bool:
Валидация данных (Data Integrity):
Если best_bid <= 0 ИЛИ best_ask <= 0 -> return false (некорректные цены).
Если best_ask <= best_bid -> error!("Inverted spread detected: ask {} <= bid {}", best_ask, best_bid); return false.
Ликвидность: Если bid_vol.is_zero() ИЛИ ask_vol.is_zero() -> warn!("Zero volume at top levels"); return false.
Расчет:
spread_abs = best_ask - best_bid.
mid_price = (best_ask + best_bid) / Decimal::from(2).
spread_bps = (spread_abs / mid_price) * Decimal::from(10_000).
Логика проверки:
Если max_spread_bps не задан -> return true.
Если spread_bps > Decimal::from(max_spread_bps) -> warn!("Spread too wide: {} bps", spread_bps); return false.
Если spread_bps близок к лимиту (например, > 80% от лимита) -> info!("Spread nearing limit: {} bps", spread_bps).
В штатном режиме -> debug!("Spread OK: {} bps", spread_bps); return true.
3. Изменения в ./src/trading/execution.rs
В логике генерации сигнала (перед вызовом OrderManager):
Получить best_bid, best_ask и их объемы из OrderBook.
Вызвать risk_manager.check_spread_gate(...).
Если гейт закрыт — отклонить сигнал (Signal::Flat).
4. Почему этот план лучше (Аргументы против упрощений):
Inverted Spread Check: На волатильном рынке WebSocket может прислать пакеты не по порядку или с задержкой, что приведет к ask <= bid. Попытка торговать в такой момент приведет к немедленному убытку.
Volume Check: Наличие цены в стакане без объема означает отсутствие ликвидности для исполнения даже минимального ордера.
Decimal Precision: Использование Decimal::from(10_000) исключает потерю точности при сравнении bps.
Future Proof: Поле max_spread_bps закладывает основу для динамического фильтра (задача 104), который будет рассчитываться как avg_spread * multiplier.
5. Тестирование
Unit test: Подать bid=10.00, ask=9.99 — должен вернуть false (инверсия).
Unit test: Подать bid_vol=0 — должен вернуть false.
Unit test: Подать spread=50 bps при лимите 10 bps — должен вернуть false.