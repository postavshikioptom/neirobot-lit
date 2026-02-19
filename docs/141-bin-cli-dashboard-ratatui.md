# Задача 141: Интерактивный CLI Дашборд на Ratatui (v2.0)

## 1. Контекст и цели
Создание изолированного инструмента мониторинга `cli-dashboard`, который не влияет на производительность основного бота.
*   **Zero-Overhead**: Отдельный процесс, опрашивающий `/health` эндпоинт.
*   **Visuals**: Использование `Sparkline` для цен, `Gauge` для сигналов и `Table` для ордеров.
*   **Robustness**: Автоматическое отображение статуса "Disconnected" при сбоях сети.

## 2. Реализация в [./src/bin/dashboard.rs](./src/bin/dashboard.rs)
Реализуй синхронный цикл обработки событий и отрисовки.

```rust
use ratatui::{backend::CrosstermBackend, widgets::*, Terminal};
use crossterm::{event::{self, Event, KeyCode}, execute, terminal::*};
use std::{io, time::{Duration, Instant}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Инициализация терминала
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, crossterm::event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let tick_rate = Duration::from_millis(200);
    let mut last_tick = Instant::now();

    loop {
        // 2. Отрисовка
        terminal.draw(|f| ui(f, &fetch_data()))?;

        // 3. Обработка ввода (q для выхода)
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Char('q') = key.code {
                    break;
                }
            }
        }
        
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // 4. Восстановление терминала
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, crossterm::event::DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

- **Sync Loop**: Согласен с Grok. Использование `tokio` для TUI часто приводит к конфликтам с `crossterm` при обработке сигналов. Синхронный цикл `poll/draw` — более стабильное решение для мониторинга.
- **Deprecated API**: Переходим на ручное управление `EnterAlternateScreen`. Это дает полный контроль над терминалом и гарантирует корректное восстановление при панике.
- **Sparklines**: Добавляем `Sparkline` для отображения `mid_price` за последние 60 секунд. Это позволит визуально определять микро-тренды, не глядя в TradingView.
- **Error Handling**: Если запрос к `localhost:8080/health` (задача 135) падает, UI должен окрашиваться в **красный** цвет с крупным текстом "BOT DISCONNECTED".
- **Performance**: Ограничиваем частоту обновления до **5 Гц** (200мс). Этого достаточно для человеческого глаза и почти не потребляет ресурсы CPU.

## 4. Виджеты дашборда
1.  **Price Chart (Top)**: `Sparkline` с историей цен.
2.  **OrderBook (Left)**: `Table` с 5 уровнями Bids/Asks (зеленый/красный).
3.  **Signal Strength (Right)**: `Gauge` с прогрессом уверенности модели `Up` (Green) или `Down` (Red).
4.  **Bot Info (Bottom)**: `Paragraph` со статусом, аптаймом и версией.

## 5. Инструкции для Gemini (Coder AI):
1.  **Cargo.toml**: Добавить `ratatui = "0.26"`, `crossterm = "0.27"`, `reqwest = { version = "0.11", features = ["blocking", "json"] }`.
2.  **[./src/bin/dashboard.rs](./src/bin/dashboard.rs)**: Реализовать TUI с разделением экрана на блоки (`Layout`).
3.  **Data Fetch**: Использовать `reqwest::blocking` для упрощения кода внутри синхронного цикла.
4.  **UI**: Применить `Constraint::Percentage` для адаптивной верстки под разный размер терминала.

**Результат**: Удобный и быстрый консольный дашборд для оперативного контроля работы бота, который запускается простой командой `cargo run --bin dashboard`.

