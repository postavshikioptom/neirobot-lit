Задача 087: Тестирование консистентности PnL и логики PositionManager
Цель: Реализовать набор асинхронных unit-тестов для PositionManager, гарантирующих точность расчетов Avg Price, Realized PnL и корректный Position Flip (переворот) в соответствии с логикой Bybit V5.

1. Архитектура тестов
Файл: ./tests/position_manager_tests.rs.
Инструментарий: rust_decimal для точности, tokio::test для асинхронности, dec! макрос для лаконичности.
2. Тестовые сценарии (Scenarios)
Усреднение (Averaging Down/Up):

Long: 1.0 @ 100.0 + 1.0 @ 90.0 -> qty = 2.0, avg_price = 95.0.
Short: -1.0 @ 100.0 + -1.0 @ 110.0 -> qty = -2.0, avg_price = 105.0 (усреднение в шорт при росте цены).
Переворот позиции (Position Flip):

Критический сценарий: Переход из Long в Short одним исполнением.
Текущая позиция: 1.0 @ 100.0.
Входящий Fill: Sell 2.5 @ 90.0.
Assert (PnL): realized_pnl должен быть строго -10.0 (убыток по закрытому лонгу: (90 - 100) * 1.0).
Assert (State): qty = -1.5 (Short), avg_price = 90.0 (цена входа для оставшегося объема шорта).
Рычажный ROI (Leveraged Unrealized PnL):

Позиция: 1.0 @ 100.0, Leverage: 10x.
Mid Price: 105.0.
Assert (Nominal): unrealized_pnl = +5.0.
Assert (ROI %): unrealized_pnl_pct = +50.0% (формула: pnl / (entry_value / leverage) * 100).
Очистка "пыли" (Dust Cleanup):

Позиция: 1.0 @ 100.0.
Fill: Sell 0.9999 @ 100.0.
Assert: Если остаток 0.0001 < min_qty_step, позиция должна стать Flat (qty = 0, avg_price = 0).
Учет комиссий (Fees):

Вход 1.0 @ 100.0 (Taker 0.055%).
Assert: realized_pnl = -0.055 (убыток сразу после входа за счет комиссий).
3. Почему этот план лучше (Аргументы Grok):
FIFO-like Flip: Мы четко разделяем закрытие старой позиции (по старой avg_price) и открытие новой (по цене fill). Это предотвращает математические искажения средней цены.
Leveraged ROI: В крипто-торговле ROI 50% при движении цены на 5% (с 10х плечом) — стандарт. Тесты подтверждают, что бот видит реальную доходность на вложенную маржу.
Dust Management: Предотвращает ошибки "insufficient balance" и замусоривание логов микро-позициями, которые невозможно закрыть из-за ограничений биржи (qty_step).
4. Критические требования
Decimal: Использование rust_decimal версии 1.34+.
Async: Тесты должны быть async, так как PositionManager интегрирован с асинхронным TradeLogger (задача 083).
Signed Qty: Положительный qty — Long, отрицательный — Short.
5. Тестирование
Запуск: cargo test --test position_manager_tests.
Покрытие: Минимум 8 сценариев, включая реверсивный флип (из Short в Long).