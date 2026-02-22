//! Криптографический модуль для шифрования API-ключей при хранении
//! 
//! Использует AES-256-GCM для аутентифицированного шифрования и Argon2id для деривации ключа.
//! Формат зашифрованных данных: ENC:<base64(salt(16) + nonce(12) + ciphertext_with_tag)>

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Nonce, Key
};
use argon2::{Argon2, Algorithm, Version, Params};
use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Размер соли для Argon2 (16 байт)
const SALT_SIZE: usize = 16;

/// Размер nonce для AES-GCM (12 байт)
const NONCE_SIZE: usize = 12;

/// Размер ключа AES-256 (32 байта)
const KEY_SIZE: usize = 32;

/// Префикс для зашифрованных значений
pub const ENCRYPTED_PREFIX: &str = "ENC:";

/// Структура для безопасного хранения мастер-ключа в памяти
#[derive(Zeroize, ZeroizeOnDrop)]
struct MasterKey {
    key: [u8; KEY_SIZE],
}

impl MasterKey {
    /// Создает новый мастер-ключ из пароля с использованием Argon2id
    fn from_password(password: &str, salt: &[u8]) -> Result<Self> {
        if salt.len() != SALT_SIZE {
            bail!("Salt must be exactly {} bytes", SALT_SIZE);
        }

        let mut key = [0u8; KEY_SIZE];
        
        // Параметры Argon2id: m=19456 KiB, t=2, p=1
        let params = Params::new(
            19456, // memory cost in KiB
            2,     // iterations
            1,     // parallelism
            Some(KEY_SIZE)
        ).map_err(|e| anyhow::anyhow!("Failed to create Argon2 params: {}", e))?;

        let argon2 = Argon2::new(
            Algorithm::Argon2id,
            Version::V0x13,
            params,
        );

        argon2
            .hash_password_into(password.as_bytes(), salt, &mut key)
            .map_err(|e| anyhow::anyhow!("Argon2 key derivation failed: {}", e))?;

        Ok(Self { key })
    }

    /// Возвращает ссылку на ключ
    fn as_bytes(&self) -> &[u8; KEY_SIZE] {
        &self.key
    }
}

/// Шифрует plaintext с использованием мастер-пароля
/// 
/// Возвращает строку в формате: ENC:<base64(salt + nonce + ciphertext_with_tag)>
pub fn encrypt(plaintext: &str, master_password: &str) -> Result<String> {
    // Генерируем случайную соль
    let mut salt = [0u8; SALT_SIZE];
    use aes_gcm::aead::rand_core::RngCore;
    OsRng.fill_bytes(&mut salt);

    // Деривируем ключ из пароля
    let master_key = MasterKey::from_password(master_password, &salt)?;

    // Создаем cipher
    let key = Key::<Aes256Gcm>::from(*master_key.as_bytes());
    let cipher = Aes256Gcm::new(&key);

    // Генерируем случайный nonce
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    // Шифруем
    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_bytes())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    // Разделяем ciphertext и tag (tag — последние 16 байт)
    if ciphertext.len() < 16 {
        bail!("Ciphertext too short to contain tag");
    }
    let (ciphertext_only, tag) = ciphertext.split_at(ciphertext.len() - 16);

    // Упаковываем: salt + nonce + tag + ciphertext (как указано в требованиях)
    let mut packed = Vec::with_capacity(SALT_SIZE + NONCE_SIZE + 16 + ciphertext_only.len());
    packed.extend_from_slice(&salt);
    packed.extend_from_slice(&nonce[..]);
    packed.extend_from_slice(tag);
    packed.extend_from_slice(ciphertext_only);

    // Кодируем в base64 и добавляем префикс
    let encoded = BASE64.encode(&packed);
    Ok(format!("{}{}", ENCRYPTED_PREFIX, encoded))
}

/// Расшифровывает зашифрованную строку с использованием мастер-пароля
/// 
/// Ожидает строку в формате: ENC:<base64(salt + nonce + ciphertext_with_tag)>
pub fn decrypt(encrypted: &str, master_password: &str) -> Result<String> {
    // Проверяем префикс
    if !encrypted.starts_with(ENCRYPTED_PREFIX) {
        bail!("Invalid encrypted string: missing '{}' prefix", ENCRYPTED_PREFIX);
    }

    // Убираем префикс и декодируем base64
    let encoded = &encrypted[ENCRYPTED_PREFIX.len()..];
    let packed = BASE64
        .decode(encoded)
        .context("Failed to decode base64")?;

    // Проверяем минимальный размер (salt + nonce + минимум 16 байт для tag)
    if packed.len() < SALT_SIZE + NONCE_SIZE + 16 {
        bail!("Invalid encrypted data: too short");
    }

    // Распаковываем: salt + nonce + tag + ciphertext
    let salt = &packed[0..SALT_SIZE];
    let nonce_bytes = &packed[SALT_SIZE..SALT_SIZE + NONCE_SIZE];
    let tag = &packed[SALT_SIZE + NONCE_SIZE..SALT_SIZE + NONCE_SIZE + 16];
    let ciphertext_only = &packed[SALT_SIZE + NONCE_SIZE + 16..];

    // Собираем обратно ciphertext + tag для расшифровки
    let mut ciphertext_with_tag = Vec::with_capacity(ciphertext_only.len() + 16);
    ciphertext_with_tag.extend_from_slice(ciphertext_only);
    ciphertext_with_tag.extend_from_slice(tag);

    // Деривируем ключ из пароля
    let master_key = MasterKey::from_password(master_password, salt)?;

    // Создаем cipher
    let key = Key::<Aes256Gcm>::from(*master_key.as_bytes());
    let cipher = Aes256Gcm::new(&key);

    // Создаем nonce
    let nonce_array: &[u8; 12] = nonce_bytes.try_into().context("Invalid nonce size")?;
    let nonce = Nonce::from(*nonce_array);

    // Расшифровываем
    let plaintext_bytes = cipher
        .decrypt(&nonce, ciphertext_with_tag.as_ref())
        .map_err(|e| anyhow::anyhow!("Decryption failed (wrong password or tampered data): {}", e))?;

    // Конвертируем в строку
    let plaintext = String::from_utf8(plaintext_bytes)
        .context("Decrypted data is not valid UTF-8")?;

    Ok(plaintext)
}

/// Проверяет, является ли строка зашифрованной (имеет префикс ENC:)
pub fn is_encrypted(value: &str) -> bool {
    value.starts_with(ENCRYPTED_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let plaintext = "my_secret_api_key";
        let password = "test_master_password";

        let encrypted = encrypt(plaintext, password).unwrap();
        assert!(encrypted.starts_with(ENCRYPTED_PREFIX));

        let decrypted = decrypt(&encrypted, password).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_wrong_password() {
        let plaintext = "my_secret_api_key";
        let password = "correct_password";
        let wrong_password = "wrong_password";

        let encrypted = encrypt(plaintext, password).unwrap();
        let result = decrypt(&encrypted, wrong_password);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Decryption failed"));
    }

    #[test]
    fn test_tampered_data() {
        let plaintext = "my_secret_api_key";
        let password = "test_password";

        let mut encrypted = encrypt(plaintext, password).unwrap();
        
        // Изменяем один символ в зашифрованных данных
        let bytes = encrypted.as_bytes().to_vec();
        let mut modified = bytes.clone();
        if let Some(byte) = modified.last_mut() {
            *byte = byte.wrapping_add(1);
        }
        let tampered = String::from_utf8(modified).unwrap();

        let result = decrypt(&tampered, password);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_encrypted() {
        assert!(is_encrypted("ENC:somebase64data"));
        assert!(!is_encrypted("plaintext"));
        assert!(!is_encrypted(""));
    }
}
