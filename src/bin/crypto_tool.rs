//! CLI утилита для шифрования/расшифровки API-ключей
//! 
//! Использование:
//!   cargo run --bin crypto_tool -- --encrypt "my_secret_key"
//!   cargo run --bin crypto_tool -- --decrypt "ENC:base64data..."
//! 
//! Мастер-пароль берется из переменной окружения NEIRO_MASTER_KEY
//! или запрашивается интерактивно.

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "crypto_tool")]
#[command(about = "Утилита для шифрования/расшифровки API-ключей", long_about = None)]
struct Cli {
    /// Зашифровать plaintext
    #[arg(long)]
    encrypt: Option<String>,
    
    /// Расшифровать ciphertext
    #[arg(long)]
    decrypt: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Проверяем, что указана ровно одна операция
    let operations_count = [cli.encrypt.is_some(), cli.decrypt.is_some()]
        .iter()
        .filter(|&&x| x)
        .count();
    
    if operations_count != 1 {
        anyhow::bail!("Please specify exactly one operation: --encrypt or --decrypt");
    }

    // Получаем мастер-пароль
    let master_password = get_master_password()?;

    if let Some(plaintext) = cli.encrypt {
        let encrypted = neirobot_lit::utils::crypto::encrypt(&plaintext, &master_password)
            .context("Encryption failed")?;
        
        println!("Encrypted value:");
        println!("{}", encrypted);
        println!();
        println!("You can now use this value in your .env file:");
        println!("BYBIT_API_KEY={}", encrypted);
    } else if let Some(ciphertext) = cli.decrypt {
        let decrypted = neirobot_lit::utils::crypto::decrypt(&ciphertext, &master_password)
            .context("Decryption failed")?;
        
        println!("Decrypted value:");
        println!("{}", decrypted);
    }

    Ok(())
}

/// Получает мастер-пароль из переменной окружения или запрашивает интерактивно
fn get_master_password() -> Result<String> {
    // Сначала пробуем получить из переменной окружения
    if let Ok(password) = std::env::var("NEIRO_MASTER_KEY") {
        if !password.is_empty() {
            return Ok(password);
        }
    }

    // Если не найден, запрашиваем интерактивно
    print!("Enter master password: ");
    io::stdout().flush()?;
    
    let mut password = String::new();
    io::stdin().read_line(&mut password)?;
    
    let password = password.trim().to_string();
    
    if password.is_empty() {
        anyhow::bail!("Master password cannot be empty");
    }

    Ok(password)
}
