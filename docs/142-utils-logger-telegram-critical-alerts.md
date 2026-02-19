# Задача 142: Telegram-оповещения для критических событий (v2.0)

## 1. Модуль воркера в [./src/utils/telegram.rs](./src/utils/telegram.rs)
Реализуй фоновый процесс, который читает сообщения из очереди, соблюдает **Rate Limit** (1 сообщение в сек) и экранирует символы для Telegram MarkdownV2.

```rust
// В [./src/utils/telegram.rs](./src/utils/telegram.rs)
use reqwest::Client;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};

pub fn escape_markdown_v2(s: &str) -> String {
    // Список спецсимволов Telegram MarkdownV2, требующих экранирования \
    let chars = r"\_*[]()~`>#+-=|{}.!";
    let mut escaped = String::with_capacity(s.len());
    for c in s.chars() {
        if chars.contains(c) { escaped.push('\\'); }
        escaped.push(c);
    }
    escaped
}

pub struct TelegramWorker {
    client: Client,
    token: String,
    chat_id: String,
}

impl TelegramWorker {
    pub async fn run(self, mut rx: mpsc::Receiver<String>, rate_ms: u64) {
        let mut interval = interval(Duration::from_millis(rate_ms));
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);

        while let Some(text) = rx.recv().await {
            interval.tick().await; // Соблюдаем лимит
            let escaped_text = escape_markdown_v2(&text);
            
            let params = [
                ("chat_id", &self.chat_id),
                ("text", &escaped_text),
                ("parse_mode", &"MarkdownV2".to_string()),
            ];

            let _ = self.client.post(&url).form(&params).send().await;
        }
    }
}
```

## 2. Кастомный Layer в [./src/utils/logger.rs](./src/utils/logger.rs)
Добавь слой для `tracing`, который перехватывает события нужного уровня и отправляет их в канал.

```rust
// В [./src/utils/logger.rs](./src/utils/logger.rs)
pub struct TelegramLayer {
    tx: mpsc::Sender<String>,
    level: tracing::Level,
}

impl<S> tracing_subscriber::Layer<S> for TelegramLayer 
where S: tracing::Subscriber 
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if event.metadata().level() <= &self.level {
            // Формируем текст (можно расширить полями события)
            let msg = format!("🚨 *{}*: {}", event.metadata().level(), "Critical error detected"); 
            // Неблокирующая отправка: если канал полон, алерт отбрасывается
            let _ = self.tx.try_send(msg);
        }
    }
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

- **Zero-Stall**: Категорически запрещено использовать `await` внутри `on_event`. Только `try_send` в ограниченный (bounded) канал. Если Telegram-воркер не успевает отправлять сообщения, мы жертвуем алертом, чтобы не заблокировать торговый поток.
- **MarkdownV2**: Без функции `escape_markdown_v2` Telegram API будет возвращать `400 Bad Request` на любое сообщение, содержащее точку или дефис.
- **Secrets**: Токен и Chat ID **не хранятся** в `BotConfig` (TOML). Они загружаются в [./src/config/loader.rs](./src/config/loader.rs) через `dotenvy` из переменных окружения `TELEGRAM_TOKEN` и `TELEGRAM_CHAT_ID`.
- **Rate Limiter**: Использование `interval.tick()` гарантирует, что мы не превысим лимиты Telegram (30 сообщений в секунду на бота, но мы ставим 1/сек для безопасности и чистоты чата).

## 4. Инструкции для Gemini (Coder AI):
1.  **Cargo.toml**: Добавить `dotenvy = "0.15"`.
2.  **[./src/config/loader.rs](./src/config/loader.rs)**: Добавить загрузку `TELEGRAM_TOKEN` и `CHAT_ID` через `std::env::var`.
3.  **[./src/utils/telegram.rs](./src/utils/telegram.rs)**: Реализовать воркер и функцию экранирования.
4.  **[./src/utils/logger.rs](./src/utils/logger.rs)**: Интегрировать `TelegramLayer` в `init_logger`. Запускать воркер через `tokio::spawn`.

**Результат**: Надежная система оповещения в Telegram, которая мгновенно уведомит об ошибках в продакшене, не создавая рисков для производительности бота.
