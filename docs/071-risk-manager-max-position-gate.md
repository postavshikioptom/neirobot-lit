Задача 071: Реализация риск-гейта MaxPositionSize
Цель: Создать защитный барьер, ограничивающий максимальный объем позиции (по номиналу и марже), гарантируя соблюдение риск-параметров бота.

Инструкции для реализации:
1. Изменения в ./src/config/types.rs
В BotConfig или RiskConfig определить поля:
max_notional_usd: Decimal — жесткий лимит номинальной стоимости (например, 1000 USDT).
max_margin_usd: Decimal — лимит используемой маржи (например, 100 USDT).
2. Изменения в ./src/risk/risk_manager.rs
Метод check_order_size:
Вход: side, qty, mid_price, current_position_qty, leverage.
Логика:
Проекция: projected_qty = current_position_qty + (if side == Buy { qty } else { -qty }).
Снижение риска (Reduce-Only):
Если projected_qty.abs() < current_position_qty.abs() — ВСЕГДА OK. Любая сделка, уменьшающая текущую экспозицию, разрешена.
Валидация по лимитам (для открытия/увеличения):
projected_notional = projected_qty.abs() * mid_price.
projected_margin = projected_notional / leverage.
Условие блокировки:
Если projected_notional > max_notional_usd -> Err(MaxNotionalExceeded).
Если projected_margin > max_margin_usd -> Err(MaxMarginExceeded).
Выход: Result<(), RiskError>.
3. Интеграция
Вызывать проверку в execution.rs перед вызовом OrderManager::place_limit_order.
Если проверка не пройдена, логировать причину и отменять выставление ордера.
4. Логирование
warn!("Risk Gate: Order blocked. Projected Notional: {} (Limit: {}), Margin: {} (Limit: {})", projected_notional, max_notional_usd, projected_margin, max_margin_usd).
Аргументация изменений:
Margin vs Notional: Если у вас max_notional = 1000 USDT, при плече 1x вы тратите 1000 маржи, а при 100x — всего 10. max_margin_usd защищает от чрезмерного залога, а max_notional_usd — от чрезмерной экспозиции на движение цены.
Mid Price Priority: Мы используем цену стакана (Mid), так как это наиболее реалистичная оценка того, сколько капитала будет "задействовано" в случае немедленного исполнения.
Absolute Safety: Безусловное разрешение на уменьшение позиции (Reduce-Only) — критически важный механизм безопасности. Бот всегда должен иметь возможность выйти из рынка, даже если его лимиты риска были изменены внешне.
Критическое требование: Убедиться, что при расчете projected_margin используется leverage из актуальных настроек бота. Если плечо равно 1, лимиты маржи и номинала будут фактически дублировать друг друга, что безопасно.