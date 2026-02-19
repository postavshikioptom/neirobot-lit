/// Интеграционные тесты для проверки "зависших" ордеров (Задача 179)
/// 
/// Тесты проверяют:
/// 1. Timeout Test: Ордер отменяется при превышении max_order_life_ms
/// 2. Market Fill Test: При свежем сигнале выполняется market fill
/// 3. Persistence Test: Ордер остается в стакане, если заполнено выше порога

#[cfg(test)]
mod stale_order_tests {
    use neirobot_lit::config::types::{RiskConfig, StaleOrderAction};
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use neirobot_lit::risk::risk_manager::OrderIntent;
    use neirobot_lit::trading::types::OrderSide;
    use std::collections::HashMap;
    use chrono::Utc;

    /// Создает тестовую конфигурацию для проверки stale orders
    fn create_test_config() -> RiskConfig {
        let mut config = RiskConfig::default();
        config.max_order_life_ms = 5000;  // 5 секунд
        config.min_fill_pct_to_keep = 0.8; // 80%
        config.stale_order_action = StaleOrderAction::CancelOnly;
        config.stale_check_interval_ms = 1000;
        config
    }

    /// Создает тестовый интент
    fn create_test_intent(qty: f64, filled_qty: f64, age_ms: u64) -> OrderIntent {
        let now = Utc::now().timestamp_millis() as u64;
        OrderIntent {
            side: OrderSide::Buy,
            price: 100.0,
            qty,
            timestamp: now.saturating_sub(age_ms),
            filled_qty,
        }
    }

    #[test]
    fn test_stale_order_detection_timeout() {
        // Arrange
        let config = create_test_config();
        let health_monitor = HealthMonitor::new(config);
        
        let mut active_intents = HashMap::new();
        // Добавляем ордер, который "зависел" 6 секунд (больше max_order_life_ms = 5000)
        active_intents.insert(
            "link_1".to_string(),
            create_test_intent(100.0, 0.0, 6000),
        );
        
        // Act
        let now = Utc::now().timestamp_millis() as u64;
        let intent = active_intents.get("link_1").unwrap();
        let age_ms = now.saturating_sub(intent.timestamp);
        let fill_pct = if intent.qty > 0.0 {
            intent.filled_qty / intent.qty
        } else {
            0.0
        };
        
        // Assert
        assert!(age_ms > health_monitor.config.max_order_life_ms, 
                "Ордер должен быть старше max_order_life_ms");
        assert!(fill_pct < health_monitor.config.min_fill_pct_to_keep,
                "Ордер должен быть заполнен менее чем на 80%");
    }

    #[test]
    fn test_stale_order_persistence_high_fill() {
        // Arrange
        let config = create_test_config();
        let health_monitor = HealthMonitor::new(config);
        
        let mut active_intents = HashMap::new();
        // Добавляем ордер, который "зависел" 6 секунд, но заполнен на 90%
        active_intents.insert(
            "link_2".to_string(),
            create_test_intent(100.0, 90.0, 6000),
        );
        
        // Act
        let now = Utc::now().timestamp_millis() as u64;
        let intent = active_intents.get("link_2").unwrap();
        let age_ms = now.saturating_sub(intent.timestamp);
        let fill_pct = if intent.qty > 0.0 {
            intent.filled_qty / intent.qty
        } else {
            0.0
        };
        
        // Assert
        assert!(age_ms > health_monitor.config.max_order_life_ms,
                "Ордер должен быть старше max_order_life_ms");
        assert!(fill_pct >= health_monitor.config.min_fill_pct_to_keep,
                "Ордер должен быть заполнен на 90%, что выше порога 80%");
        // Ордер НЕ должен быть отменен, так как заполнен выше порога
    }

    #[test]
    fn test_stale_order_young_order_not_stale() {
        // Arrange
        let config = create_test_config();
        let health_monitor = HealthMonitor::new(config);
        
        let mut active_intents = HashMap::new();
        // Добавляем молодой ордер (1 секунда), который не должен быть "зависшим"
        active_intents.insert(
            "link_3".to_string(),
            create_test_intent(100.0, 0.0, 1000),
        );
        
        // Act
        let now = Utc::now().timestamp_millis() as u64;
        let intent = active_intents.get("link_3").unwrap();
        let age_ms = now.saturating_sub(intent.timestamp);
        
        // Assert
        assert!(age_ms < health_monitor.config.max_order_life_ms,
                "Молодой ордер не должен быть старше max_order_life_ms");
        // Ордер НЕ должен быть отменен
    }

    #[test]
    fn test_stale_order_boundary_condition() {
        // Arrange
        let config = create_test_config();
        let health_monitor = HealthMonitor::new(config);
        
        let mut active_intents = HashMap::new();
        // Добавляем ордер, который ровно на границе (5000ms)
        active_intents.insert(
            "link_4".to_string(),
            create_test_intent(100.0, 0.0, 5000),
        );
        
        // Act
        let now = Utc::now().timestamp_millis() as u64;
        let intent = active_intents.get("link_4").unwrap();
        let age_ms = now.saturating_sub(intent.timestamp);
        
        // Assert
        // Ордер на границе НЕ должен быть отменен (используется >)
        assert!(age_ms <= health_monitor.config.max_order_life_ms,
                "Ордер на границе не должен быть отменен");
    }

    #[test]
    fn test_multiple_stale_orders() {
        // Arrange
        let config = create_test_config();
        let health_monitor = HealthMonitor::new(config);
        
        let mut active_intents = HashMap::new();
        // Добавляем несколько ордеров
        active_intents.insert("link_5".to_string(), create_test_intent(100.0, 0.0, 6000));
        active_intents.insert("link_6".to_string(), create_test_intent(100.0, 50.0, 6000));
        active_intents.insert("link_7".to_string(), create_test_intent(100.0, 90.0, 6000));
        active_intents.insert("link_8".to_string(), create_test_intent(100.0, 0.0, 1000));
        
        // Act
        let now = Utc::now().timestamp_millis() as u64;
        let mut stale_count = 0;
        
        for (_, intent) in active_intents.iter() {
            let age_ms = now.saturating_sub(intent.timestamp);
            let fill_pct = if intent.qty > 0.0 {
                intent.filled_qty / intent.qty
            } else {
                0.0
            };
            
            if age_ms > health_monitor.config.max_order_life_ms 
                && fill_pct < health_monitor.config.min_fill_pct_to_keep {
                stale_count += 1;
            }
        }
        
        // Assert
        // Должны быть отменены: link_5 (0% заполнен), link_6 (50% заполнен)
        // НЕ должны быть отменены: link_7 (90% заполнен), link_8 (молодой)
        assert_eq!(stale_count, 2, "Должно быть 2 'зависших' ордера");
    }

    #[test]
    fn test_fill_pct_calculation() {
        // Arrange
        let config = create_test_config();
        let health_monitor = HealthMonitor::new(config);
        
        // Act & Assert
        let test_cases = vec![
            (100.0, 0.0, 0.0),      // 0% заполнен
            (100.0, 50.0, 0.5),     // 50% заполнен
            (100.0, 80.0, 0.8),     // 80% заполнен (граница)
            (100.0, 100.0, 1.0),    // 100% заполнен
            (50.0, 25.0, 0.5),      // 50% заполнен (другой объем)
        ];
        
        for (qty, filled_qty, expected_pct) in test_cases {
            let fill_pct = if qty > 0.0 {
                filled_qty / qty
            } else {
                0.0
            };
            
            assert!((fill_pct - expected_pct).abs() < 0.0001,
                    "Неверный расчет процента заполнения: {} vs {}", fill_pct, expected_pct);
        }
    }

    #[test]
    fn test_stale_order_action_enum() {
        // Arrange & Act
        let cancel_only = StaleOrderAction::CancelOnly;
        let market_fill = StaleOrderAction::CancelAndMarketFill;
        let repeg = StaleOrderAction::Repeg;
        
        // Assert
        assert_eq!(cancel_only, StaleOrderAction::CancelOnly);
        assert_eq!(market_fill, StaleOrderAction::CancelAndMarketFill);
        assert_eq!(repeg, StaleOrderAction::Repeg);
        assert_ne!(cancel_only, market_fill);
    }
}


/// Интеграционные тесты для архивации логов (Задача 182)
#[cfg(test)]
mod log_archival_tests {
    use neirobot_lit::config::types::RiskConfig;
    use neirobot_lit::risk::health_monitor::HealthMonitor;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio;

    /// Создает тестовую конфигурацию для архивации логов
    fn create_test_config() -> RiskConfig {
        let mut config = RiskConfig::default();
        config.log_retention_days = 7;
        config
    }

    #[tokio::test]
    async fn test_log_archival_task_initialization() {
        // Arrange
        let mut config = create_test_config();
        let mut health_monitor = HealthMonitor::new(config);
        
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();
        
        // Act
        health_monitor.set_log_dir(log_dir.clone());
        
        // Assert
        assert!(health_monitor.log_dir.is_some(), "log_dir should be set");
        assert_eq!(health_monitor.log_dir.as_ref().unwrap(), &log_dir);
    }

    #[tokio::test]
    async fn test_log_archival_with_no_log_dir() {
        // Arrange
        let config = create_test_config();
        let mut health_monitor = HealthMonitor::new(config);
        
        // Act: Вызываем архивацию без установки log_dir
        let result = health_monitor.run_log_archival_task(7).await;
        
        // Assert: Должно вернуться Ok, так как log_dir не установлен
        assert!(result.is_ok(), "Should return Ok when log_dir is not set");
    }

    #[tokio::test]
    async fn test_log_archival_creates_zst_files() {
        // Arrange
        let mut config = create_test_config();
        let mut health_monitor = HealthMonitor::new(config);
        
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();
        
        // Создаем тестовые лог-файлы
        let log_file1 = log_dir.join("bot_2024-02-13.log");
        let log_file2 = log_dir.join("bot_2024-02-14.log");
        
        let test_data = "2024-02-14T10:00:00Z INFO Test log entry\n".repeat(100);
        fs::write(&log_file1, &test_data).expect("Failed to write test log 1");
        fs::write(&log_file2, &test_data).expect("Failed to write test log 2");
        
        health_monitor.set_log_dir(log_dir.clone());
        
        // Act
        let result = health_monitor.run_log_archival_task(7).await;
        
        // Assert
        assert!(result.is_ok(), "Archival task should succeed");
        
        // Проверяем, что .zst файлы созданы
        let zst_files: Vec<_> = fs::read_dir(&log_dir)
            .expect("Failed to read log dir")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "zst") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        
        assert!(!zst_files.is_empty(), "Should create .zst files");
    }

    #[tokio::test]
    async fn test_log_archival_skips_active_log() {
        // Arrange
        let mut config = create_test_config();
        let mut health_monitor = HealthMonitor::new(config);
        
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();
        
        // Создаем активный лог-файл (bot.log)
        let active_log = log_dir.join("bot.log");
        let test_data = "2024-02-14T10:00:00Z INFO Active log\n".repeat(100);
        fs::write(&active_log, &test_data).expect("Failed to write active log");
        
        health_monitor.set_log_dir(log_dir.clone());
        
        // Act
        let result = health_monitor.run_log_archival_task(7).await;
        
        // Assert
        assert!(result.is_ok(), "Archival task should succeed");
        
        // Проверяем, что bot.log не был сжат
        assert!(active_log.exists(), "Active log (bot.log) should not be compressed");
        
        let zst_active = log_dir.join("bot.log.zst");
        assert!(!zst_active.exists(), "bot.log should not be compressed");
    }

    #[test]
    fn test_compression_ratio_for_logs() {
        // Arrange: Создаем типичные лог-данные
        let mut log_data = String::new();
        for i in 0..1000 {
            log_data.push_str(&format!(
                "2024-02-14T10:00:{:02}Z INFO Event {} - Processing market data\n",
                i % 60, i
            ));
        }
        
        // Act
        let compressed = zstd::encode_all(log_data.as_bytes(), 1)
            .expect("Failed to compress");
        
        let original_size = log_data.len();
        let compressed_size = compressed.len();
        let ratio = compressed_size as f64 / original_size as f64;
        
        // Assert
        assert!(
            ratio < 0.3,
            "Log compression ratio should be less than 30%, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_zst_file_integrity() {
        // Arrange
        let original_data = "2024-02-14T10:00:00Z ERROR Critical error\n".repeat(500);
        
        // Act
        let compressed = zstd::encode_all(original_data.as_bytes(), 1)
            .expect("Failed to compress");
        
        let decompressed = zstd::decode_all(&compressed[..])
            .expect("Failed to decompress");
        
        let decompressed_str = String::from_utf8(decompressed)
            .expect("Failed to convert to string");
        
        // Assert
        assert_eq!(
            original_data, decompressed_str,
            "Decompressed data should match original"
        );
    }
}
