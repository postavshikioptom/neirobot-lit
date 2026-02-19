Задача 064: Обновление позиции на основе данных об исполнении (Fills)
Цель: Реализовать в PositionManager математически точный учет позиции с использованием знаковых объемов и расчетом реализованного PnL.

Инструкции для реализации:
1. Изменения в ./src/trading/types.rs
Enum Side: Создать перечисление Side { Buy, Sell } с реализацией FromStr для парсинга ответов Bybit.
Обновить FillEvent: Использовать Side вместо String.
2. Изменения в ./src/trading/position_manager.rs
Поля PositionManager:
qty: Decimal — Знаковое число. Положительное = Long, Отрицательное = Short.
avg_price: Decimal — Всегда положительное (средняя цена входа).
realized_pnl: Decimal — Накопленная прибыль/убыток (для задачи 083).
Метод update_from_fill(&mut self, fill: FillEvent):
Определить знак изменения: qty_change = if fill.side == Side::Buy { fill.exec_qty } else { -fill.exec_qty }.
Логика изменения:
Увеличение позиции (знаки qty и qty_change одинаковы):
new_avg = (self.qty * self.avg_price + qty_change * fill.exec_price) / (self.qty + qty_change).
Уменьшение / Закрытие (знаки разные, qty_change.abs() <= qty.abs()):
avg_price не меняется.
Расчет PnL: realized_pnl += (fill.exec_price - self.avg_price) * qty_change.abs() * (if Long { 1 } else { -1 }).
Переворот (Flip) (знаки разные, qty_change.abs() > qty.abs()):
Закрыть текущую позицию (рассчитать PnL для self.qty).
Открыть новую на остаток (excess_qty = qty_change + self.qty).
self.avg_price = fill.exec_price.
self.qty = excess_qty.
Сброс: Если self.qty == 0, то self.avg_price = 0.
3. Логирование
Логировать каждое изменение: info!("Fill received: {} @ {}. New Pos: {}, Avg: {}", fill.side, fill.exec_price, self.qty, self.avg_price).
При фиксации прибыли: info!("Realized PnL updated: {}", self.realized_pnl).
Аргументация изменений:
Signed Qty: Использование +1.0 для Long и -1.0 для Short упрощает все формулы. Проверка self.qty.is_sign_positive() заменяет сложные проверки строковых полей "Buy/Sell".
Flip Accuracy: При перевороте (например, из Long 1.0 в Short 0.5 одним ордером на 1.5) средняя цена Short-позиции должна быть ровно ценой последней сделки. Моя прошлая логика могла это усреднить, что неверно.
Realized PnL: Накопление реализованного PnL в PositionManager — самый надежный способ передачи данных в logger.csv (задача 083), так как менеджер позиций является "источником истины".
Критическое требование: Убедиться, что avg_price всегда рассчитывается как абсолютное значение, а знак направления хранится только в qty. Любое деление в формулах должно быть защищено от NaN.