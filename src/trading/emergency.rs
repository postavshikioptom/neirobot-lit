use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Синхронная функция экстренной отмены всех ордеров
/// Используется в panic handler для гарантированной попытки очистки
/// без зависимости от асинхронного рантайма Tokio
pub fn cancel_all_sync(key: &str, secret: &str, symbol: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string();
    let recv_window = "5000";
    let body = format!(r#"{{"category":"linear","symbol":"{}"}}"#, symbol);
    
    // Bybit V5 Signature: timestamp + api_key + recv_window + payload
    let sign_data = format!("{}{}{}{}", ts, key, recv_window, body);
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(sign_data.as_bytes());
    let signature = hex::encode(mac.finalize().into_bytes());

    let base_url = std::env::var("BYBIT_API_URL")
        .unwrap_or_else(|_| "https://api.bybit.com".to_string());
    let url = format!("{}/v5/order/cancel-all", base_url);

    let res = ureq::post(&url)
        .header("X-BAPI-API-KEY", key)
        .header("X-BAPI-TIMESTAMP", &ts)
        .header("X-BAPI-SIGN", &signature)
        .header("X-BAPI-RECV-WINDOW", recv_window)
        .header("Content-Type", "application/json")
        .send(body);

    if let Err(e) = res {
        eprintln!("!!! EMERGENCY CANCEL FAILED: {} !!!", e);
    } else {
        eprintln!("✓ Emergency cancel-all request sent successfully.");
    }
}
