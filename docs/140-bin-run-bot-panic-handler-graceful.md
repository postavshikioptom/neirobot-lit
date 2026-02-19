# Задача 140: Обработка паник и экстренная отмена ордеров (v2.0)

## 1. Синхронный модуль отмены в [./src/trading/emergency.rs](./src/trading/emergency.rs)
Создай изолированный модуль, который выполняет прямой POST-запрос к Bybit без использования асинхронного рантайма Tokio.

```rust
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub fn cancel_all_sync(key: &str, secret: &str, symbol: &str) {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis().to_string();
    let recv_window = "5000";
    let body = format!(r#"{{"category":"linear","symbol":"{}"}}"#, symbol);
    
    // Bybit V5 Signature: timestamp + api_key + recv_window + payload
    let sign_data = format!("{}{}{}{}", ts, key, recv_window, body);
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
    mac.update(sign_data.as_bytes());
    let signature = hex::encode(mac.finalize().into_bytes());

    let res = ureq::post("https://api.bybit.com/v5/order/cancel-all")
        .set("X-BAPI-API-KEY", key)
        .set("X-BAPI-TIMESTAMP", &ts)
        .set("X-BAPI-SIGN", &signature)
        .set("X-BAPI-RECV-WINDOW", recv_window)
        .set("Content-Type", "application/json")
        .send_string(&body);

    if let Err(e) = res {
        eprintln!("!!! EMERGENCY CANCEL FAILED: {} !!!", e);
    } else {
        eprintln!("✓ Emergency cancel-all request sent successfully.");
    }
}
```

## 2. Глобальный хук паники в [./src/bin/run-bot.rs](./src/bin/run-bot.rs)
Настрой перехватчик, который сработает в любом потоке и выполнит очистку перед жестким выходом.

```rust
use std::sync::Arc;
use std::panic;
use std::process;

pub fn setup_panic_handler(key: Arc<String>, secret: Arc<String>, symbol: String) {
    let default_hook = panic::take_hook();

    panic::set_hook(Box::new(move |panic_info| {
        eprintln!("\nFATAL ERROR: {}", panic_info);
        
        // Выполняем экстренную отмену (синхронно, блокируя поток до завершения I/O)
        crate::trading::emergency::cancel_all_sync(&key, &secret, &symbol);

        default_hook(panic_info);
        
        // Гарантируем выход с ошибкой для рестарта супервизором
        process::exit(1);
    }));
}
```

## 3. Спорные моменты и корректировки (Grok + Zencoder)

- **Sync Isolation**: Согласен с Grok. Весь «грязный» синхронный код (ureq, hmac) должен жить в [./src/trading/emergency.rs](./src/trading/emergency.rs). Это защищает основной асинхронный пайплайн от случайного использования блокирующих вызовов.
- **Bybit V5 Signature**: Grok предложил параметры в URL, но Bybit V5 для POST-запросов требует подпись через заголовки (`X-BAPI-SIGN`). Мы реализовали стандарт V5: `timestamp + api_key + recv_window + body`.
- **Arc Ownership**: Используем `Arc<String>` для ключей. Это позволяет хуку владеть данными (ownership) даже если `main` уже начал разрушаться.
- **No Async in Hook**: Категорически запрещено использовать `tokio::spawn` или `await` внутри хука. Рантайм может быть в состоянии `deadlock` или `panic`. Только синхронный `ureq`.
- **Best Effort**: Мы не обрабатываем ошибки API внутри хука сложным образом. Если запрос не прошел — печатаем в `stderr` и выходим. Главное — попытаться спасти депозит.

## 4. Инструкции для Gemini (Coder AI):
1. **Cargo.toml**: Добавить `ureq`, `hmac`, `sha2`, `hex`.
2. **[./src/trading/emergency.rs](./src/trading/emergency.rs)**: Реализовать логику формирования подписи V5 и синхронный POST-запрос.
3. **[./src/bin/run-bot.rs](./src/bin/run-bot.rs)**: Инициализировать `setup_panic_handler` в `main` сразу после загрузки конфига.
4. **Safety**: Убедиться, что в `Cargo.toml` **НЕТ** настройки `panic = "abort"`, иначе хуки не будут вызваны.

**Результат**: Гарантированная попытка очистки активных ордеров при любом критическом сбое программы, что минимизирует финансовые риски в непредвиденных ситуациях.
