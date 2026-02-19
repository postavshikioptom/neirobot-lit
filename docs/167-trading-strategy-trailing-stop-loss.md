# Задача №167: Динамический скользящий стоп-лосс (Trailing Stop Loss)

**Цель**: Защита накопленной прибыли путем автоматического подтягивания уровня Stop Loss вслед за ценой. Реализация должна поддерживать как логику на стороне бота (Bot-side), так и нативный функционал биржи (Exchange-side).

---

## План реализации для Gemini AI Coder:

### 1. Изменения в [./src/config/types.rs](./src/config/types.rs)
Добавить структуру `TrailingStopConfig` в `StrategyConfig`:
- **`tsl_mode`**: `enum { Bot, Exchange }` (место исполнения логики).
- **`tsl_activation_bps`**: `u32` (профит в базисных пунктах для активации трейлинга).
- **`tsl_distance_bps`**: `u32` (дистанция отступа от экстремума).
- **`tsl_step_bps`**: `u32` (минимальный шаг обновления уровня).

### 2. Состояние в [./src/trading/position_manager.rs](./src/trading/position_manager.rs)
Обновить структуру `Position` (инициализируется per-bot в [./src/run-bot.rs](./src/run-bot.rs)):
- **`extreme_water_mark`**: `f64` (максимальная цена для Long, минимальная для Short).
- **`current_stop_loss`**: `f64` (актуальный уровень стопа).
- **`tsl_active`**: `bool` (флаг состояния активации).

### 3. Логика в [./src/trading/execution.rs](./src/trading/execution.rs) (Bot-side)
Реализовать расчет в методе `update_tsl`:
- **Активация**:
  ```rust
  let profit_bps = if position.side == Side::Buy {
      (mid_price - entry) / entry * 10000.0
  } else {
      (entry - mid_price) / entry * 10000.0
  };
  if profit_bps >= config.tsl_activation_bps { position.tsl_active = true; }
  ```
- **Обновление Extreme и Stop**:
  - **Long**: `if mid_price > extreme_water_mark { extreme_water_mark = mid_price; }`
    `new_sl = extreme_water_mark * (1.0 - distance / 10000.0)`
  - **Short**: `if mid_price < extreme_water_mark { extreme_water_mark = mid_price; }`
    `new_sl = extreme_water_mark * (1.0 + distance / 10000.0)`
- **Фильтр шага**: Обновлять `current_stop_loss`, только если `abs(new_sl - current_sl) / current_sl * 10000.0 >= tsl_step_bps`.

### 4. Исполнение в [./src/trading/order_manager.rs](./src/trading/order_manager.rs)
- **Exchange-side**: Если `tsl_mode == Exchange`, при открытии позиции (задача [061](./docs/000-tasks_list.md)) отправлять параметры `trailing_stop` и `active_price` напрямую в Bybit API.
- **Bot-side Trigger**: Если `tsl_mode == Bot` и цена пересекла `current_stop_loss`, инициировать немедленное закрытие по рынку.

### 5. Тестирование в `tests/execution_flow.rs`
Создать тест-кейс для проверки цепочки событий:
1. `Entry` на 100.0.
2. `Mid` растет до 102.0 (активация при 200 bps).
3. `Mid` растет до 110.0 (TSL подтягивается к ~108.9 при дистанции 100 bps).
4. `Mid` падает до 108.5 -> Триггер на закрытие.

---

## Аргументация (ответы на замечания Grok):
1. **Extreme Water Mark**: Название исправлено для устранения путаницы (High vs Low). Теперь это универсальный показатель лучшей достигнутой цены.
2. **Точные формулы**: Исключена неопределенность `sign`. Использованы явные условия для Long (`- distance`) и Short (`+ distance`).
3. **Bybit Native**: Добавлена опция `tsl_mode`. Использование нативного трейлинга биржи снижает риски при потере связи с ботом, но лишает гибкости (например, нельзя менять дистанцию динамически на основе волатильности).
4. **Profit-only Activation**: Трейлинг активируется только при достижении профита, что предотвращает "зажатие" позиции стопом сразу после входа в условиях рыночного шума.

**Gemini, твоя задача — реализовать надежный механизм сопровождения прибыли, который не позволит рыночному откату "съесть" заработанное.**