
# Задача 137: Логика округления и фильтрация «пыли» (v2.0)

## 1. Математические хелперы в `src/utils/helpers.rs`
Реализуй функции для прецизионной работы с лотами биржи Bybit.

```rust
// В [./src/utils/helpers.rs](./src/utils/helpers.rs)
pub fn round_down_to_step(qty: f64, step: f64) -> f64 {
    if step <= 0.0 { return qty; }
    // Стандартное округление вниз до шага лота
    (qty / step).floor() * step
}

pub fn is_dust(qty: f64, min_qty: f64) -> bool {
    // Учитываем микро-погрешность float (epsilon)
    qty < (min_qty - 1e-10)
}

pub fn clamp_qty(qty: f64, min_qty: f64, max_qty: f64, step: f64) -> f64 {
    let clamped = qty.max(min_qty).min(max_qty);
    round_down_to_step(clamped, step)
}
```

## 2. Структура фильтров в `src/trading/types.rs`
Объедини параметры лота в единую структуру для удобной передачи в методы ордера.

```rust
// В [./src/trading/types.rs](./src/trading/types.rs)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LotFilter {
    pub min_qty: f64,
    pub max_qty: f64,
    pub qty_step: f64,
}
```

## 3. Обновление логики в `src/trading/order.rs`
Метод `transition` (из задачи 136) должен учитывать фильтры лота при обработке частичных исполнений.

```rust
// В [./src/trading/order.rs](./src/trading/order.rs)
impl Order {
    pub fn apply_trade(&mut self, exec_qty: f64, filter: &LotFilter) {
        self.executed_qty += exec_qty;
        
        // Расчитываем реальный остаток, который можно выставить на биржу
        let remaining = self.qty - self.executed_qty;
        let rounded_remaining = crate::utils::helpers::round_down_to_step(remaining, filter.qty_step);

        if crate::utils::helpers::is_dust(rounded_remaining, filter.min_qty) {
            // Если остаток — "пыль", помечаем ордер как полностью исполненный
            self.state = OrderState::Filled;
            if rounded_remaining > 0.0 {
                tracing::debug!("Dust detected ({}), marking order {} as Filled", rounded_remaining, self.link_id);
            }
        } else {
            self.state = OrderState::PartiallyFilled;
        }
    }
}
```

## 4. Спорные моменты и корректировки (Grok + Zencoder)

- **f64 vs Decimal**: Согласен с Grok. Для округления объемов в крипто-трейдинге `f64` достаточно, если использовать метод `(qty / step).floor()`. Это на порядок быстрее `Decimal` и не вызывает ошибок `Invalid Qty` при правильном `step` (который Bybit всегда отдает как `f64`).
- **Placement**: Переносим всё в [./src/utils/helpers.rs](./src/utils/helpers.rs). Лишние файлы только усложняют навигацию.
- **Max Qty**: Обязательно добавляем `max_qty`. Bybit жестко ограничивает максимальный объем одного ордера (особенно на щиткоинах), и `clamp_qty` защитит нас от отклонения (Reject) всей транзакции.
- **Dust Handling**: Пыль фильтруется именно в `Order::transition`. Если после сделки остаток меньше `min_qty`, мы не переходим в `PartiallyFilled` (что могло бы спровоцировать попытку `chase` несуществующего объема), а сразу идем в `Filled`.

## 5. Инструкции для Gemini (Coder AI):
1. **[./src/utils/helpers.rs](./src/utils/helpers.rs)**: Реализовать `round_down_to_step`, `is_dust` и `clamp_qty`.
2. **[./src/trading/types.rs](./src/trading/types.rs)**: Добавить структуру `LotFilter`.
3. **[./src/trading/order.rs](./src/trading/order.rs)**: Обновить логику обработки события `Trade`, внедрив проверку пыли через `LotFilter`.
4. **Tests**: Написать расширенный тест в `src/utils/helpers.rs` (через `#[cfg(test)]`), покрывающий кейсы с разными `qty_step` (напр. 0.001, 1.0, 10.0) и проверку пыли.

**Результат**: Безопасная работа с частичными исполнениями, отсутствие «битых» ордеров из-за неверного округления и корректный учет позиции без микро-остатков.
