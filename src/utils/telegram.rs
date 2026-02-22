use reqwest::Client;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};

/// Экранирование специальных символов для Telegram MarkdownV2
/// 
/// Согласно документации Telegram Bot API, следующие символы должны быть экранированы:
/// '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'
pub fn escape_markdown_v2(s: &str) -> String {
    // Список спецсимволов Telegram MarkdownV2, требующих экранирования \
    let chars = r"\_*[]()~`>#+-=|{}.!";
    let mut escaped = String::with_capacity(s.len() * 2); // Резервируем больше места для экранирования
    
    for c in s.chars() {
        if chars.contains(c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    
    escaped
}

/// Фоновый воркер для отправки сообщений в Telegram
/// 
/// Соблюдает Rate Limit (по умолчанию 1 сообщение в секунду) и автоматически
/// экранирует специальные символы для MarkdownV2
pub struct TelegramWorker {
    client: Client,
    token: String,
    chat_id: String,
}

impl TelegramWorker {
    /// Создает новый экземпляр TelegramWorker
    pub fn new(token: String, chat_id: String) -> Self {
        Self {
            client: Client::new(),
            token,
            chat_id,
        }
    }

    /// Запускает воркер, который читает сообщения из канала и отправляет их в Telegram
    /// 
    /// # Параметры
    /// - `rx`: Receiver для получения сообщений
    /// - `rate_ms`: Интервал между отправками в миллисекундах (по умолчанию 1000)
    pub async fn run(self, mut rx: mpsc::Receiver<String>, rate_ms: u64) {
        let mut interval = interval(Duration::from_millis(rate_ms));
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);

        while let Some(text) = rx.recv().await {
            interval.tick().await; // Соблюдаем лимит
            
            let escaped_text = escape_markdown_v2(&text);
            
            let params = serde_json::json!({
                "chat_id": &self.chat_id,
                "text": &escaped_text,
                "parse_mode": "MarkdownV2",
            });

            // Отправляем запрос, игнорируя ошибки (не блокируем воркер)
            if let Err(e) = self.client.post(&url).json(&params).send().await {
                tracing::warn!("Failed to send Telegram message: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_markdown_v2() {
        // Тест базовых символов
        assert_eq!(escape_markdown_v2("Hello World"), "Hello World");
        
        // Тест специальных символов
        assert_eq!(escape_markdown_v2("Test."), r"Test\.");
        assert_eq!(escape_markdown_v2("Error!"), r"Error\!");
        assert_eq!(escape_markdown_v2("Price: $100"), r"Price: $100");
        
        // Тест множественных специальных символов
        assert_eq!(
            escape_markdown_v2("Error: [critical] (code-500)"),
            r"Error: \[critical\] \(code\-500\)"
        );
        
        // Тест всех специальных символов
        let all_special = r"_*[]()~`>#+-=|{}.!";
        let escaped = escape_markdown_v2(all_special);
        assert!(escaped.contains(r"\_"));
        assert!(escaped.contains(r"\*"));
        assert!(escaped.contains(r"\!"));
    }
}
