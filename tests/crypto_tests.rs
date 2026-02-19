//! Интеграционные тесты для модуля шифрования API-ключей
//! 
//! Задача 216: Шифрование API-ключей при хранении

use neirobot_lit::utils::crypto;

#[test]
fn test_roundtrip_encryption_decryption() {
    // Тест: зашифровал -> расшифровал -> получил исходник
    let plaintext = "my_secret_api_key_12345";
    let password = "strong_master_password";

    let encrypted = crypto::encrypt(plaintext, password)
        .expect("Encryption should succeed");
    
    // Проверяем формат
    assert!(encrypted.starts_with("ENC:"), "Encrypted value should start with ENC:");
    assert!(encrypted.len() > 4, "Encrypted value should have content after prefix");

    let decrypted = crypto::decrypt(&encrypted, password)
        .expect("Decryption should succeed");
    
    assert_eq!(decrypted, plaintext, "Decrypted value should match original");
}

#[test]
fn test_tamper_detection() {
    // Тест: изменение одного бита в Base64 строке должно приводить к ошибке аутентификации
    let plaintext = "sensitive_data";
    let password = "test_password";

    let encrypted = crypto::encrypt(plaintext, password)
        .expect("Encryption should succeed");
    
    // Изменяем последний символ в зашифрованной строке
    let mut chars: Vec<char> = encrypted.chars().collect();
    let last_idx = chars.len() - 1;
    chars[last_idx] = if chars[last_idx] == 'A' { 'B' } else { 'A' };
    let tampered: String = chars.into_iter().collect();

    // Попытка расшифровки должна провалиться
    let result = crypto::decrypt(&tampered, password);
    assert!(result.is_err(), "Decryption of tampered data should fail");
    
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("Decryption failed") || error_msg.contains("decode"),
        "Error should indicate decryption failure or decode error, got: {}",
        error_msg
    );
}

#[test]
fn test_invalid_master_key() {
    // Тест: попытка расшифровки с неверным мастер-паролем должна возвращать ошибку
    let plaintext = "api_secret_value";
    let correct_password = "correct_password_123";
    let wrong_password = "wrong_password_456";

    let encrypted = crypto::encrypt(plaintext, correct_password)
        .expect("Encryption should succeed");
    
    // Попытка расшифровки с неверным паролем
    let result = crypto::decrypt(&encrypted, wrong_password);
    assert!(result.is_err(), "Decryption with wrong password should fail");
    
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("Decryption failed"),
        "Error should indicate decryption failure, got: {}",
        error_msg
    );
}

#[test]
fn test_multiple_encryptions_produce_different_ciphertexts() {
    // Тест: одинаковый plaintext с одинаковым паролем должен давать разные ciphertext
    // (из-за случайной соли и nonce)
    let plaintext = "same_secret";
    let password = "same_password";

    let encrypted1 = crypto::encrypt(plaintext, password)
        .expect("First encryption should succeed");
    let encrypted2 = crypto::encrypt(plaintext, password)
        .expect("Second encryption should succeed");
    
    assert_ne!(encrypted1, encrypted2, "Two encryptions should produce different ciphertexts");
    
    // Но оба должны расшифровываться в исходный текст
    let decrypted1 = crypto::decrypt(&encrypted1, password)
        .expect("First decryption should succeed");
    let decrypted2 = crypto::decrypt(&encrypted2, password)
        .expect("Second decryption should succeed");
    
    assert_eq!(decrypted1, plaintext);
    assert_eq!(decrypted2, plaintext);
}

#[test]
fn test_is_encrypted_detection() {
    // Тест: функция is_encrypted правильно определяет зашифрованные значения
    assert!(crypto::is_encrypted("ENC:somebase64data"));
    assert!(crypto::is_encrypted("ENC:"));
    assert!(!crypto::is_encrypted("plaintext_value"));
    assert!(!crypto::is_encrypted(""));
    assert!(!crypto::is_encrypted("EN:missing_C"));
}

#[test]
fn test_decrypt_without_prefix_fails() {
    // Тест: попытка расшифровать строку без префикса ENC: должна провалиться
    let password = "test_password";
    let result = crypto::decrypt("not_encrypted_value", password);
    
    assert!(result.is_err(), "Decryption without ENC: prefix should fail");
    assert!(result.unwrap_err().to_string().contains("missing 'ENC:' prefix"));
}

#[test]
fn test_empty_plaintext() {
    // Тест: шифрование пустой строки
    let plaintext = "";
    let password = "test_password";

    let encrypted = crypto::encrypt(plaintext, password)
        .expect("Encryption of empty string should succeed");
    
    let decrypted = crypto::decrypt(&encrypted, password)
        .expect("Decryption should succeed");
    
    assert_eq!(decrypted, plaintext);
}

#[test]
fn test_unicode_plaintext() {
    // Тест: шифрование Unicode текста
    let plaintext = "Секретный ключ 🔐 with émojis";
    let password = "пароль";

    let encrypted = crypto::encrypt(plaintext, password)
        .expect("Encryption of Unicode should succeed");
    
    let decrypted = crypto::decrypt(&encrypted, password)
        .expect("Decryption should succeed");
    
    assert_eq!(decrypted, plaintext);
}

#[test]
fn test_long_plaintext() {
    // Тест: шифрование длинного текста
    let plaintext = "a".repeat(10000);
    let password = "test_password";

    let encrypted = crypto::encrypt(&plaintext, password)
        .expect("Encryption of long text should succeed");
    
    let decrypted = crypto::decrypt(&encrypted, password)
        .expect("Decryption should succeed");
    
    assert_eq!(decrypted, plaintext);
}
