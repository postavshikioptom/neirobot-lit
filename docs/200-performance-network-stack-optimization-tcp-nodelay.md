# Задача №200: Оптимизация сетевого стека: TCP_NODELAY и буферы сокетов (Windows)

[./docs/200-performance-network-stack-optimization-tcp-nodelay.md](./docs/200-performance-network-stack-optimization-tcp-nodelay.md)

## 1. План реализации

### 1.1. Обновление зависимостей
- Добавить в `Cargo.toml` крейт `socket2 = "0.5.7"` для низкоуровневой настройки WinSock.

### 1.2. Изменения в [./src/config/types.rs](./src/config/types.rs)
- Добавить в структуру `NetworkConfig` следующие параметры:
    - `tcp_nodelay: bool` (по умолчанию `true`).
    - `socket_recv_buffer_size: usize` (рекомендуется от `1048576` (1 МБ) до `8388608` (8 МБ)).
    - `socket_send_buffer_size: usize` (рекомендуется `1048576`).

### 1.3. Изменения в [./src/data/websocket.rs](./src/data/websocket.rs)
- Переработать процесс установления соединения, заменив прямой вызов `connect_async` на ручное создание сокета через `socket2`:
    1.  Создать объект `socket2::Socket` (IPv4, Stream, TCP).
    2.  Вызвать `.set_nodelay(config.tcp_nodelay)?`.
    3.  Установить размеры буферов: `.set_recv_buffer_size(config.socket_recv_buffer_size)?` и `.set_send_buffer_size(config.socket_send_buffer_size)?`.
    4.  Выполнить подключение к адресу биржи через `.connect(&addr.into())?`.
    5.  Преобразовать в `std::net::TcpStream`, установить неблокирующий режим `.set_nonblocking(true)?`.
    6.  Преобразовать в `tokio::net::TcpStream::from_std(std_tcp)` для дальнейшего использования в `tokio_tungstenite::client_async`.

### 1.4. Логирование и проверка
- Сразу после настройки сокета считать фактические значения буферов через `.recv_buffer_size()?` и `.send_buffer_size()?`.
- Вывести в лог сравнение: `[Network] Socket buffer: requested {} KB, actual {} KB`. Это критично для Windows, так как система может ограничивать (cap) размер буфера.

## 2. Пример реализации (фрагмент)

```rust
use socket2::{Socket, Domain, Type, Protocol};
use tokio::net::TcpStream;

// Ручная настройка сокета для Windows WinSock
let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;
socket.set_nodelay(true)?;
socket.set_recv_buffer_size(config.recv_buf_size)?;

let actual_buf = socket.recv_buffer_size()?;
info!("[Network] WinSock buffer set: requested {}, actual {}", config.recv_buf_size, actual_buf);

socket.connect(&addr.into())?;
let std_tcp: std::net::TcpStream = socket.into();
std_tcp.set_nonblocking(true)?;
let tokio_tcp = TcpStream::from_std(std_tcp)?;
```

## 3. Ожидаемый результат
- **Минимизация задержек**: Отключение алгоритма Нагла устраняет задержки в 10–50 мс при отправке мелких пакетов (ордеров).
- **Стабильность в пиках**: Увеличенные буферы WinSock предотвращают потерю рыночных данных при всплесках активности (bursts) в стакане.
- **Прозрачность**: Логи показывают реальные системные лимиты Windows для сетевых буферов.