# Position Manager PnL Consistency Tests

Комплексное тестирование консистентности расчетов PnL и логики управления позициями.

## Запуск тестов

```bash
# Запустить все тесты
cargo test --test position_manager_tests

# Запустить с выводом
cargo test --test position_manager_tests -- --nocapture

# Запустить конкретный тест
cargo test --test position_manager_tests test_position_flip_long_to_short
```

## Покрытие тестов (18 сценариев)

### Усреднение позиций (2 теста)
- ✅ **Averaging Down Long**: 1.0 @ 100 + 1.0 @ 90 → qty=2.0, avg=95.0
- ✅ **Averaging Up Short**: -1.0 @ 100 + -1.0 @ 110 → qty=-2.0, avg=105.0

### Переворот позиции (2 теста)
- ✅ **Long → Short Flip**: 1.0 @ 100 → Sell 2.5 @ 90
  - Realized PnL: -10.0 (убыток по закрытому лонгу)
  - New Position: -1.5 @ 90.0
- ✅ **Short → Long Flip**: -1.0 @ 100 → Buy 2.5 @ 110
  - Realized PnL: -10.0 (убыток по закрытому шорту)
  - New Position: 1.5 @ 110.0

### Закрытие позиций (2 теста)
- ✅ **Partial Close**: 2.0 @ 100 → Sell 1.0 @ 110 (PnL: +10, остаток: 1.0)
- ✅ **Full Close**: 1.0 @ 100 → Sell 1.0 @ 105 (PnL: +5, flat)

### Leveraged ROI (4 теста)
- ✅ **Long 10x**: 1.0 @ 100, price → 105
  - Nominal PnL: +5.0
  - ROI: +50% (5 / (100/10) * 100)
- ✅ **Short 10x**: -1.0 @ 100, price → 95
  - Nominal PnL: +5.0
  - ROI: +50%
- ✅ **Flat Position**: PnL = 0, ROI = 0
- ✅ **Zero Leverage**: ROI = 10% (без плеча)

### Накопление (2 теста)
- ✅ **Multiple Fills**: 0.5@100 + 0.3@105 + 0.2@110 → avg=103.5
- ✅ **Realized PnL Accumulation**: Два частичных закрытия → total PnL = 15.0

### Прибыль/Убыток (2 теста)
- ✅ **Loss Scenario**: Long 1.0@100 → Sell@90 (PnL: -10)
- ✅ **Short Profit**: Short 1.0@100 → Buy@90 (PnL: +10)

### Изменение плеча (2 теста)
- ✅ **Leverage Change**: 5x → 10x (ROI: 50% → 100%)
- ✅ **Zero Leverage Edge Case**: Без плеча ROI = 10%

### Dust Cleanup (1 тест)
- ✅ **Dust Cleanup**: 1.0 @ 100 → Sell 0.9999 @ 100
  - Остаток 0.0001 < min_qty_step (0.001)
  - Позиция автоматически закрывается (qty=0, avg_price=0)

### Учет комиссий (2 теста)
- ✅ **Fee Accounting**: Вход 1.0@100 с комиссией 0.055
  - realized_pnl = -0.055 (убыток сразу после входа)
- ✅ **Fee Accumulation**: Вход с fee=0.055, выход с fee=0.06, PnL=+10
  - Total realized_pnl = 9.885 (-0.055 - 0.06 + 10.0)

## Ключевые проверки

### 1. FIFO-like Position Flip
При перевороте позиции:
1. Сначала закрывается старая позиция по старой avg_price
2. Рассчитывается realized PnL
3. Открывается новая позиция по цене fill

```rust
// Long 1.0 @ 100 → Sell 2.5 @ 90
// Закрытие: (90 - 100) * 1.0 = -10.0
// Открытие: -1.5 @ 90.0
```

### 2. Leveraged ROI Formula
```
entry_value = avg_price * |qty|
entry_margin = entry_value / leverage
ROI% = (unrealized_pnl / entry_margin) * 100
```

Пример с 10x:
- Entry: 1.0 @ 100 (value=100, margin=10)
- Price → 105 (PnL=+5)
- ROI = (5/10)*100 = 50%

### 3. Средневзвешенная цена
При пирамидинге (увеличении позиции):
```
new_avg = (old_qty * old_avg + qty_change * exec_price) / new_qty
```

### 4. Realized PnL при закрытии
```
side_sign = if Long { 1 } else { -1 }
pnl = (exec_price - avg_price) * closed_qty * side_sign
```

### 5. Учет комиссий
Комиссия вычитается из realized_pnl сразу при каждом fill:
```rust
self.position.realized_pnl -= fill.exec_fee;
```

### 6. Dust Cleanup
Если остаток позиции меньше min_qty_step, позиция автоматически закрывается:
```rust
if self.position.qty.abs() < self.min_qty_step {
    self.position.qty = Decimal::zero();
    self.position.avg_price = Decimal::zero();
}
```

## Использование rust_decimal

Все тесты используют `rust_decimal` для точности:

```rust
use rust_decimal_macros::dec;

// Точные вычисления
assert_eq!(dec!(0.1) + dec!(0.2), dec!(0.3)); // ✅ Работает
```

## Асинхронность

Все тесты асинхронные (используют `#[tokio::test]`), что соответствует требованиям задачи 087:

```rust
#[tokio::test]
async fn test_position_flip_long_to_short() {
    // ...
}
```

## Критические требования

1. **Decimal Precision**: rust_decimal гарантирует точность
2. **Signed Qty**: Положительный = Long, отрицательный = Short
3. **PnL Consistency**: Realized PnL накапливается корректно
4. **Leverage Impact**: ROI учитывает плечо правильно
5. **Fee Accounting**: Комиссии вычитаются из realized_pnl
6. **Dust Management**: Микро-позиции автоматически закрываются

## Структура тестов

```
tests/position_manager_tests.rs
├── Усреднение (averaging_down/up)
├── Переворот (position_flip)
├── Закрытие (partial/full_close)
├── Leveraged ROI (unrealized_pnl)
├── Накопление (accumulation)
├── Прибыль/Убыток (profit/loss)
├── Изменение плеча (leverage_change)
├── Dust Cleanup
└── Учет комиссий (fee_accounting)
```

## Зависимости

- `rust_decimal = "1.40"` - точные вычисления
- `rust_decimal_macros` - макрос `dec!` для лаконичности
- `tokio` - для асинхронных тестов

## Примечания

- Все тесты асинхронные (`#[tokio::test]`)
- Используется макрос `dec!` для удобства записи Decimal
- Тесты покрывают все edge cases из задачи 087
- Проверяется консистентность PnL при различных сценариях
- Добавлены тесты на dust cleanup и fee accounting

