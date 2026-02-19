Задача 070: Расчет нереализованного PnL в реальном времени
Цель: Реализовать в PositionManager высокоточный расчет плавающего PnL и Leveraged ROI (доходности на маржу) для принятия торговых решений и контроля рисков.

Инструкции для реализации:
1. Изменения в ./src/trading/position_manager.rs
Дополнительные поля:
leverage: Decimal — текущее плечо (из конфига бота).
unrealized_pnl: Decimal — PnL по Mid Price (для стратегии).
unrealized_pnl_pct: Decimal — Leveraged ROI в процентах.
mark_pnl: Decimal — PnL по Mark Price (для риск-гейтов).
Метод update_unrealized_pnl(&mut self, mid_price: Decimal, mark_price: Decimal):
PnL (Absolute):
self.unrealized_pnl = (mid_price - self.avg_price) * self.qty.
self.mark_pnl = (mark_price - self.avg_price) * self.qty.
Leveraged ROI (%):
entry_value = self.avg_price * self.qty.abs().
entry_margin = entry_value / self.leverage.
self.unrealized_pnl_pct = if entry_margin.is_zero() { 0 } else { (self.unrealized_pnl / entry_margin) * 100 }.
Сброс: Если self.qty == 0, все PnL-поля сбрасываются в 0.
2. Источники цен
mid_price берется из OrderBook (каждый тик).
mark_price берется из WebSocket-потока tickers или tickers.SYMBOL (задача 080). До реализации WS можно использовать mid_price как временную заглушку.
3. Синхронизация и дрейф
В методе sync_from_remote (из задачи 066) добавить сравнение:
remote_pnl (с биржи) должен сопоставляться с нашим self.mark_pnl, так как биржа считает по Mark Price.
Логировать разницу между unrealized_pnl (Mid) и mark_pnl (Mark), чтобы видеть влияние спреда на текущую оценку позиции.
4. Логирование
debug!("PnL: {:+} USDT (ROI: {:+}%), Mark PnL: {:+}", self.unrealized_pnl, self.unrealized_pnl_pct, self.mark_pnl).
Аргументация изменений:
Leveraged ROI: Если вы зашли с плечом 10x и цена выросла на 1%, ваша прибыль на вложенный капитал составляет 10%. Именно это значение критично для риск-менеджмента и психологии трейдинга.
Mid vs Mark: Mid Price отражает реальность исполнения (Taker exit). Mark Price — это "справедливая" цена биржи, защищенная от манипуляций. Использование обеих цен дает полный контроль над ситуацией.
Entry Margin: Корректный расчет маржи — фундамент для будущих задач по расчету ликвидаций и маржин-коллов.
Критическое требование: Убедитесь, что leverage берется из актуального конфига бота. Если плечо было изменено в процессе работы через REST, PositionManager должен обновить это поле, чтобы расчет ROI оставался корректным.