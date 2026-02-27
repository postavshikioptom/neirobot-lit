use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Sparkline, Table},
    Frame, Terminal,
};
use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant},
};

/// Структура для десериализации ответа от /health эндпоинта
#[derive(Debug, Clone, serde::Deserialize)]
struct HealthStatus {
    status: String,
    uptime_sec: u64,
    last_update_ms_ago: u64,
    ws_connected: bool,
    dropped_logs: u64,
}

/// Данные для дашборда
struct DashboardData {
    health: Option<HealthStatus>,
    price_history: VecDeque<u64>,
    current_price: f64,
    signal_strength: f64,
    signal_direction: String,
}

impl Default for DashboardData {
    fn default() -> Self {
        Self {
            health: None,
            price_history: VecDeque::with_capacity(60),
            current_price: 0.0,
            signal_strength: 0.0,
            signal_direction: "Neutral".to_string(),
        }
    }
}

/// Получение данных из /health эндпоинта
fn fetch_data(data: &mut DashboardData) {
    match reqwest::blocking::get("http://localhost:8080/health") {
        Ok(response) => {
            if let Ok(health) = response.json::<HealthStatus>() {
                data.health = Some(health);
                
                // Mock данные для демонстрации (в реальности нужен отдельный эндпоинт)
                // Генерируем случайную цену для sparkline
                let time = Instant::now().elapsed().as_secs_f64();
                data.current_price = 100.0 + (time * 0.5).sin() * 10.0;
                
                // Добавляем в историю (масштабируем для sparkline)
                let scaled_price = (data.current_price * 10.0) as u64;
                data.price_history.push_back(scaled_price);
                if data.price_history.len() > 60 {
                    data.price_history.pop_front();
                }
                
                // Mock сигнал модели
                data.signal_strength = ((time * 0.3).sin() + 1.0) / 2.0; // 0.0 - 1.0
                data.signal_direction = if (time * 0.3).sin() > 0.0 {
                    "Up".to_string()
                } else {
                    "Down".to_string()
                };
            } else {
                data.health = None;
            }
        }
        Err(_) => {
            data.health = None;
        }
    }
}

/// Отрисовка UI
fn ui(f: &mut Frame, data: &DashboardData) {
    // Если бот отключен - показываем красный экран
    if data.health.is_none() {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Red))
            .style(Style::default().bg(Color::Red));
        
        let text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "BOT DISCONNECTED",
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Cannot connect to localhost:8080/health",
                Style::default().fg(Color::White).bg(Color::Red),
            )),
        ];
        
        let paragraph = Paragraph::new(text)
            .block(block)
            .alignment(ratatui::layout::Alignment::Center);
        
        f.render_widget(paragraph, f.area());
        return;
    }
    
    let health = data.health.as_ref().unwrap();
    
    // Основной layout: Top (Price) | Middle (OrderBook + Signal) | Bottom (Info)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(25),  // Price chart
            Constraint::Percentage(50),  // OrderBook + Signal
            Constraint::Percentage(25),  // Bot info
        ])
        .split(f.area());
    
    // 1. Price Chart (Sparkline)
    render_price_chart(f, chunks[0], data);
    
    // 2. Middle section: OrderBook (Left) + Signal (Right)
    let middle_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);
    
    render_orderbook(f, middle_chunks[0]);
    render_signal_gauge(f, middle_chunks[1], data);
    
    // 3. Bot Info (Bottom)
    render_bot_info(f, chunks[2], health);
}

/// Отрисовка графика цен (Sparkline)
fn render_price_chart(f: &mut Frame, area: Rect, data: &DashboardData) {
    let price_data: Vec<u64> = data.price_history.iter().copied().collect();
    
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Price: ${:.2} ", data.current_price))
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .data(&price_data)
        .style(Style::default().fg(Color::Green));
    
    f.render_widget(sparkline, area);
}

/// Отрисовка orderbook (Table) - mock данные
fn render_orderbook(f: &mut Frame, area: Rect) {
    let header = Row::new(vec!["Bid Price", "Bid Size", "Ask Price", "Ask Size"])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
    
    // Mock данные для 5 уровней (Bids - зеленый, Asks - красный)
    let rows = vec![
        Row::new(vec![
            Span::styled("100.50", Style::default().fg(Color::Green)),
            Span::styled("1.25", Style::default().fg(Color::Green)),
            Span::styled("100.55", Style::default().fg(Color::Red)),
            Span::styled("0.85", Style::default().fg(Color::Red)),
        ]),
        Row::new(vec![
            Span::styled("100.45", Style::default().fg(Color::Green)),
            Span::styled("2.10", Style::default().fg(Color::Green)),
            Span::styled("100.60", Style::default().fg(Color::Red)),
            Span::styled("1.50", Style::default().fg(Color::Red)),
        ]),
        Row::new(vec![
            Span::styled("100.40", Style::default().fg(Color::Green)),
            Span::styled("0.95", Style::default().fg(Color::Green)),
            Span::styled("100.65", Style::default().fg(Color::Red)),
            Span::styled("2.20", Style::default().fg(Color::Red)),
        ]),
        Row::new(vec![
            Span::styled("100.35", Style::default().fg(Color::Green)),
            Span::styled("1.80", Style::default().fg(Color::Green)),
            Span::styled("100.70", Style::default().fg(Color::Red)),
            Span::styled("1.10", Style::default().fg(Color::Red)),
        ]),
        Row::new(vec![
            Span::styled("100.30", Style::default().fg(Color::Green)),
            Span::styled("3.00", Style::default().fg(Color::Green)),
            Span::styled("100.75", Style::default().fg(Color::Red)),
            Span::styled("0.75", Style::default().fg(Color::Red)),
        ]),
    ];
    
    let table = Table::new(
        rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" OrderBook (Top 5) ")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    
    f.render_widget(table, area);
}

/// Отрисовка силы сигнала (Gauge)
fn render_signal_gauge(f: &mut Frame, area: Rect, data: &DashboardData) {
    let signal_percent = (data.signal_strength * 100.0) as u16;
    
    let color = if data.signal_direction == "Up" {
        Color::Green
    } else {
        Color::Red
    };
    
    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Signal: {} ", data.signal_direction))
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .gauge_style(Style::default().fg(color).add_modifier(Modifier::BOLD))
        .percent(signal_percent)
        .label(format!("{}%", signal_percent));
    
    f.render_widget(gauge, area);
}

/// Отрисовка информации о боте (Paragraph)
fn render_bot_info(f: &mut Frame, area: Rect, health: &HealthStatus) {
    let status_color = if health.status == "up" {
        Color::Green
    } else {
        Color::Red
    };
    
    let uptime_hours = health.uptime_sec / 3600;
    let uptime_minutes = (health.uptime_sec % 3600) / 60;
    let uptime_seconds = health.uptime_sec % 60;
    
    let text = vec![
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(
                &health.status,
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(format!(
            "Uptime: {:02}:{:02}:{:02}",
            uptime_hours, uptime_minutes, uptime_seconds
        )),
        Line::from(format!(
            "WebSocket: {}",
            if health.ws_connected { "Connected" } else { "Disconnected" }
        )),
        Line::from(format!(
            "Last Update: {}ms ago",
            health.last_update_ms_ago
        )),
        Line::from(format!("Dropped Logs: {}", health.dropped_logs)),
        Line::from(format!("Version: {}", env!("CARGO_PKG_VERSION"))),
        Line::from(""),
        Line::from(Span::styled(
            "Press 'q' to quit",
            Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
        )),
    ];
    
    let paragraph = Paragraph::new(text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Bot Info ")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    
    f.render_widget(paragraph, area);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Инициализация терминала
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    // Настройки цикла
    let tick_rate = Duration::from_millis(200); // 5 Hz
    let mut last_tick = Instant::now();
    let mut data = DashboardData::default();
    
    // Основной цикл
    loop {
        // Получаем данные
        fetch_data(&mut data);
        
        // Отрисовка
        terminal.draw(|f| ui(f, &data))?;
        
        // Обработка ввода (q для выхода)
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
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    
    Ok(())
}
