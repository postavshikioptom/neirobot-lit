Задача 072: Реализация риск-гейта Drawdown Stop (HardStop)
Цель: Создать механизм защиты капитала на основе High-Water Mark (пика эквити).


Structuring the Risk Manager


Блокировка торговли при достижении лимита просадки от внутридневного пика.

1. Изменения в ./src/config/types.rs
RiskConfig: Добавить поддержку динамических лимитов и автосброса.
pub struct RiskConfig {
    pub max_daily_drawdown_usd: Option<Decimal>,
    pub max_daily_drawdown_pct: Option<Decimal>, // % от пика дня
    pub auto_reset_at_midnight: bool,           // Сбрасывать блок в 00:00 UTC
}
2. Изменения в ./src/risk/risk_manager.rs
Состояние (State):
peak_daily_equity: Decimal — "высшая точка" баланса + PnL за сегодня.
last_reset_date: Option<NaiveDate> — для отслеживания смены суток по UTC.
is_blocked: bool — флаг HardStop.
Логика High-Water Mark (update_metrics):
Вызывается при каждом тике (изменение unrealized_pnl в задаче 070) и синхронизации баланса (задача 066).
current_equity = balance + unrealized_pnl.
Если current_equity > peak_daily_equity -> peak_daily_equity = current_equity.
Расчет Просадки (check_drawdown):
drawdown_usd = peak_daily_equity - current_equity.
drawdown_pct = (drawdown_usd / peak_daily_equity) * 100.
Trigger: Если drawdown_usd >= limit_usd ИЛИ drawdown_pct >= limit_pct -> is_blocked = true.
Автосброс (Midnight Reset):
В начале каждой проверки сравнивать Utc::now().date() с last_reset_date.
Если дата изменилась: is_blocked = false, peak_daily_equity = current_equity, last_reset_date = today.
3. Изменения в ./src/trading/execution.rs и position_manager.rs
Метод emergency_market_close():
В PositionManager добавить метод, который формирует рыночный ордер с флагом reduce_only: true и объемом, равным текущей позиции.
Реакция на HardStop:
Если risk_manager.check_drawdown() возвращает false:
critical!("HARD STOP TRIGGERED: DD from peak reached. Peak: {}, Current: {}", peak, current).
order_manager.cancel_all_orders() (очистка лимиток).
position_manager.emergency_market_close() (выход по рынку).
Перевод системы в режим Blocked (только чтение данных до сброса лимита).
4. Почему этот план лучше (Аргументы против упрощений):
Peak vs Start-of-Day: Использование пика (HWM) защищает накопленную за день прибыль. Если бот заработал +5% и потерял их, initial_equity этого не заметит, а peak_equity зафиксирует просадку и остановит торговлю.
Midnight UTC: Необходим для корректной работы на Bybit, где торговые сутки и отчеты привязаны к UTC.
Reduce Only: Экстренное закрытие обязано быть reduce_only, чтобы при сетевых лагах не открыть позицию в обратную сторону.
5. Тестирование
Тест на пик: Убедиться, что peak_daily_equity растет вместе с эквити, но не падает при убытках.
Тест на полночь: Имитировать смену даты и проверить автоматическую разблокировку.