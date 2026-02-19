Задача 061-trading-rest-client-init.md
Цель: Создать защищенный и высокоточный асинхронный REST-клиент для Bybit V5, обеспечивающий надежное выполнение торговых операций и синхронизацию баланса.

Инструкции для реализации:
1. Добавление зависимостей (Cargo.toml)
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }
hmac = "0.12", sha2 = "0.10", hex = "0.4"
rust_decimal = { version = "1.34", features = ["serde-float"] } — для точности
secrecy = { version = "0.8", features = ["serde"] } — для защиты ключей
chrono = "0.4" — для работы с миллисекундами Bybit.
2. Создание ./src/trading/rest_client.rs
Структура BybitRestClient:

api_key: String
api_secret: Secret<String> — обертка для предотвращения логирования секрета.
base_url: String (из exchange.toml).
recv_window: u64 (по умолчанию 5000).
Алгоритм подписи (Bybit V5):

Для GET: Параметры запроса должны быть отсортированы по алфавиту перед конкатенацией.
Для POST: Использовать JSON-строку тела запроса.
Pre-hash string: timestamp + api_key + recv_window + (query_string или body_string).
HMAC-SHA256: Вычисляется от секрета и pre-hash строки, результат в lowercase hex.
3. Реализация методов (MVP)
get_wallet_balance():
Эндпоинт: /v5/account/wallet-balance?accountType=UNIFIED.
Важно: Все числовые значения (equity, availableBalance) парсить строго в rust_decimal::Decimal.
get_server_time(): Для синхронизации X-BAPI-TIMESTAMP (Bybit допускает отклонение не более recv_window).
4. Обработка ошибок BybitError
Парсить поля: retCode (i64), retMsg (String) и retExtInfo (Value) для детальной диагностики.
Если retCode != 0, возвращать Err(BybitError).
5. Безопасность и конфигурация
Инициализировать клиент через BybitRestClient::new(config: &ExchangeConfig).
Считывать BYBIT_API_SECRET из .env напрямую в Secret<String>.
Запрещено: Выводить BybitRestClient или api_secret в логи через Debug.
Аргументация изменений:
Decimal vs Float: В крипто-торговле 0.000000010.000000010.00000001 имеет значение. f64 накапливает ошибки округления при расчете PnL и объемов, что недопустимо.
Secrecy: Использование Secret<String> гарантирует, что даже при panic! или случайном println!("{:?}", client) ваш API-секрет не попадет в логи.
Сортировка параметров: Bybit V5 требует алфавитного порядка ключей в QueryString. Без этого подпись будет валидна для одних запросов и невалидна для других.
Критическое требование: Убедиться, что base_url подставляется корректно (Mainnet/Testnet) из файла ./exchange.toml, чтобы избежать случайной торговли на реальные деньги во время тестов.