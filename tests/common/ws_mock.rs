use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use serde_json::{json, Value};
use tokio_tungstenite::{accept_async, tungstenite::Message};

/// Mock WebSocket сервер для тестирования
pub struct WsMockServer {
    pub port: u16,
    pub tx: broadcast::Sender<Message>,
    handle: tokio::task::JoinHandle<()>,
}

impl WsMockServer {
    /// Создает новый mock WebSocket сервер на динамическом порту
    pub async fn new() -> Self {
        Self::with_options(true, true).await
    }

    /// Создает mock сервер с настройками
    /// 
    /// # Параметры
    /// * `auto_pong` - автоматически отвечать на ping сообщения
    /// * `auto_auth` - автоматически подтверждать авторизацию
    pub async fn with_options(auto_pong: bool, auto_auth: bool) -> Self {
        // Привязываемся к динамическому порту
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind to address");
        
        let port = listener.local_addr().unwrap().port();
        
        // Канал для отправки сообщений из тестов в сервер (broadcast для множественных подключений)
        let (tx, _rx) = broadcast::channel::<Message>(100);
        
        let tx_clone = tx.clone();
        
        // Запускаем сервер в отдельной задаче
        let handle = tokio::spawn(async move {
            Self::run_server(listener, tx_clone, auto_pong, auto_auth).await;
        });

        WsMockServer { port, tx, handle }
    }

    /// Основной цикл сервера
    async fn run_server(
        listener: TcpListener,
        tx: broadcast::Sender<Message>,
        auto_pong: bool,
        auto_auth: bool,
    ) {
        while let Ok((stream, _)) = listener.accept().await {
            let mut rx = tx.subscribe();
            let handle_connection = Self::handle_connection(
                stream,
                &mut rx,
                auto_pong,
                auto_auth,
            );
            
            tokio::spawn(handle_connection);
        }
    }

    /// Обработка одного WebSocket соединения
    async fn handle_connection(
        stream: TcpStream,
        rx: &mut broadcast::Receiver<Message>,
        auto_pong: bool,
        auto_auth: bool,
    ) {
        let ws_stream = match accept_async(stream).await {
            Ok(ws) => ws,
            Err(e) => {
                eprintln!("WebSocket handshake error: {}", e);
                return;
            }
        };

        let (mut write, mut read) = ws_stream.split();

        loop {
            tokio::select! {
                // Получаем сообщения от клиента
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                                // Обработка auth
                                if auto_auth && json.get("op").and_then(|v| v.as_str()) == Some("auth") {
                                    // Проверка формата args (задача 080): должен быть массив из 3 элементов
                                    let is_valid_format = json.get("args")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| arr.len() == 3)
                                        .unwrap_or(false);
                                    
                                    let response = if is_valid_format {
                                        json!({
                                            "success": true,
                                            "ret_msg": "",
                                            "conn_id": "test-connection",
                                            "op": "auth"
                                        })
                                    } else {
                                        json!({
                                            "success": false,
                                            "ret_msg": "Invalid auth format: expected [api_key, expires, signature]",
                                            "conn_id": "test-connection",
                                            "op": "auth"
                                        })
                                    };
                                    
                                    if write.send(Message::Text(response.to_string())).await.is_err() {
                                        break;
                                    }
                                }
                                
                                // Обработка subscribe
                                if json.get("op").and_then(|v| v.as_str()) == Some("subscribe") {
                                    let response = json!({
                                        "success": true,
                                        "ret_msg": "",
                                        "conn_id": "test-connection",
                                        "op": "subscribe"
                                    });
                                    
                                    if write.send(Message::Text(response.to_string())).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            // Автоматический ответ на ping
                            if auto_pong {
                                if write.send(Message::Pong(data)).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            break;
                        }
                        Some(Err(e)) => {
                            eprintln!("WebSocket error: {}", e);
                            break;
                        }
                        _ => {}
                    }
                }
                
                // Получаем сообщения из теста для отправки клиенту
                Ok(msg) = rx.recv() => {
                    if write.send(msg).await.is_err() {
                        break;
                    }
                }
            }
        }
    }

    /// Отправляет сообщение клиенту
    pub async fn send_message(&self, msg: Message) -> Result<(), broadcast::error::SendError<Message>> {
        self.tx.send(msg).map(|_| ())
    }

    /// Отправляет текстовое сообщение клиенту
    pub async fn send_text(&self, text: String) -> Result<(), broadcast::error::SendError<Message>> {
        self.send_message(Message::Text(text)).await
    }

    /// Отправляет JSON сообщение клиенту
    pub async fn send_json(&self, json: Value) -> Result<(), broadcast::error::SendError<Message>> {
        self.send_text(json.to_string()).await
    }

    /// Возвращает URL для подключения к серверу
    pub fn url(&self) -> String {
        format!("ws://127.0.0.1:{}", self.port)
    }

    /// Останавливает сервер
    pub fn shutdown(self) {
        self.handle.abort();
    }
}

impl Drop for WsMockServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
