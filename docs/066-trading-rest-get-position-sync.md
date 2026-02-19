Задача 066: Синхронизация позиции через REST API
Цель: Реализовать механизм сверки локального состояния PositionManager с «источником истины» на бирже (Bybit V5) для устранения расхождений в объеме и цене.

Инструкции для реализации:
1. Изменения в ./src/trading/rest_client.rs
Метод get_position(symbol: &str):
Запрос: GET /v5/position/list с параметрами category=linear и symbol.
Парсинг знакового объема (Signed Qty):
Bybit возвращает size (всегда положительный) и side ("Buy" или "Sell").
Логика: let qty = if side == "Buy" { size } else { -size }; (используя Decimal).
Дополнительные поля: Извлекать avgPrice и unrealisedPnl (для мониторинга из задачи 070).
Фильтрация: Искать запись, где positionIdx соответствует значению из BotConfig (по умолчанию 0 для One-Way Mode).
2. Изменения в ./src/trading/position_manager.rs
Метод sync_from_remote(&mut self, remote_qty: Decimal, remote_avg_price: Decimal, market_info: &MarketInfo):
Порог дрейфа: Считать расхождение существенным только если (self.qty - remote_qty).abs() >= market_info.qty_step.
Действие: Если дрейф обнаружен — принудительно перезаписать self.qty и self.avg_price.
Логирование: При расхождении писать WARN, при совпадении — DEBUG.
3. Изменения в ./src/config/types.rs
Добавить в BotConfig или ExchangeConfig:
position_sync_interval_secs: u64 (по умолчанию 60).
position_idx: i32 (по умолчанию 0).
4. Интеграция в ./src/bin/run-bot.rs
Initial Sync: Выполнить один запрос перед запуском основного цикла. Если позиция на бирже уже открыта (например, бот упал и перезапустился), PositionManager должен подхватить её сразу.
Periodic Task: Использовать tokio::time::interval(Duration::from_secs(config.sync_interval)).
Аргументация изменений:
Signed Qty Conversion: Bybit не присылает -1.0 для шорта. Мы обязаны реализовать это преобразование в REST-клиенте, чтобы сохранить консистентность с логикой PositionManager (задача 064).
Qty Step Filter: Использование qty_step из MarketInfo предотвращает "панику" бота из-за микро-расхождений (например, 0.000000001), вызванных разницей в точности представления чисел.
Unrealized PnL: Получение этого значения из REST API — самый дешевый способ проверить, совпадает ли наше понимание прибыли с расчетами биржи, не дожидаясь инференса.
Критическое требование: Если запрос /v5/position/list возвращает пустой список (list: []), это означает, что позиция на бирже закрыта (qty = 0). Код должен корректно обрабатывать этот случай, а не выдавать ошибку "Position not found".