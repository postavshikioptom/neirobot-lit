Задача 065: Расчет объема ордера в Execution
Цель: Реализовать математически точный расчет объема сделки, гарантирующий прохождение риск-гейтов, наличие средств на комиссии и соответствие правилам лотности Bybit.

Инструкции для реализации:
1. Подготовка данных (MarketInfo)
В ./src/trading/types.rs добавить структуру:
pub struct MarketInfo {
    pub qty_step: Decimal,       // Шаг объема (например, 0.01)
    pub min_order_qty: Decimal,  // Минимальный лот
    pub max_order_qty: Decimal,  // Максимальный лот
    pub tick_size: Decimal,      // Шаг цены (для будущего)
}
Эти данные должны подгружаться один раз при старте бота через REST-метод get_instruments_info.
2. Изменения в ./src/trading/execution.rs
Функция calculate_order_size:
Вход: available_balance, current_price, bot_config, market_info.
Логика:
Учет комиссий и буфера:
fee_rate = bot_config.taker_fee_bps / 10000 (берем худший вариант).
buffer = bot_config.buffer_pct (например, 0.01 для 1%).
effective_balance = available_balance / (1.0 + fee_rate + buffer).
Целевой объем: target_qty = min(effective_balance * leverage, bot_config.max_pos_size_usd) / current_price.
Округление до шага (Floor to Step):
final_qty = (target_qty / market_info.qty_step).floor() * market_info.qty_step.
Валидация по лимитам:
Если final_qty < market_info.min_order_qty -> возвращаем 0.
Если final_qty > market_info.max_order_qty -> final_qty = market_info.max_order_qty.
Риск-проверка: Вызвать risk_manager.can_open_position(final_qty). Если false -> 0.
3. Изменения в ./src/config/types.rs
Добавить buffer_pct: Decimal (default: 0.01) в BotConfig или RiskConfig.
4. Логирование
info!("Order size calculated: {} (original target: {}, balance used: {})", final_qty, target_qty, effective_balance).
Аргументация изменений:
Floor to Step: Простое round_dp может дать 0.100001, что Bybit отклонит. Математическое floor(qty/step)*step гарантирует, что объем будет кратен минимальному лоту.
Effective Balance: Учет (1 + fee + buffer) в знаменателе — единственный способ на 100% избежать ошибки Insufficient Balance при маркет-ордерах (Taker) в условиях волатильности.
Static MarketInfo: Подгрузка спецификаций контракта при инициализации — это "лучшая практика", предотвращающая ошибки при попытке торговать символами с разной лотностью (например, BTC vs SHIB).
Критическое требование: Округление должно выполняться только один раз в самом конце расчета, чтобы избежать накопления погрешности на промежуточных этапах. Используйте методы rust_decimal для всех операций.