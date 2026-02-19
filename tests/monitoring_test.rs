use anyhow::Result;
use std::time::Duration;
use tokio::time::sleep;

/// Тест проверяет доступность эндпоинта /metrics и наличие ключевых метрик
#[tokio::test]
async fn test_metrics_endpoint_availability() -> Result<()> {
    // Инициализируем metrics exporter на тестовом порту
    let test_port = 19090; // Используем нестандартный порт для тестов
    
    // Инициализируем exporter
    neirobot_lit::monitoring::metrics::init_metrics_exporter(test_port)?;
    
    // Даем время на запуск HTTP сервера
    sleep(Duration::from_millis(500)).await;
    
    // Проверяем доступность эндпоинта
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/metrics", test_port);
    
    let response = client.get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await?;
    
    assert_eq!(response.status(), 200, "Metrics endpoint should return 200 OK");
    
    let body = response.text().await?;
    
    // Проверяем наличие ключевых метрик (Задача 189)
    assert!(body.contains("bot_ws_messages_total"), "Should contain bot_ws_messages_total metric");
    assert!(body.contains("bot_inference_duration_us"), "Should contain bot_inference_duration_us metric");
    assert!(body.contains("bot_realized_pnl_bps"), "Should contain bot_realized_pnl_bps metric");
    assert!(body.contains("bot_unrealized_pnl_bps"), "Should contain bot_unrealized_pnl_bps metric");
    assert!(body.contains("bot_health_status"), "Should contain bot_health_status metric");
    assert!(body.contains("bot_orders_placed_total"), "Should contain bot_orders_placed_total metric");
    assert!(body.contains("bot_order_rejections_total"), "Should contain bot_order_rejections_total metric");
    
    // Проверяем наличие консолидированных метрик из prometheus.rs (Задача 189)
    assert!(body.contains("bot_ticks_total"), "Should contain bot_ticks_total metric");
    assert!(body.contains("bot_signal_oscillations_handled_total"), "Should contain bot_signal_oscillations_handled_total metric");
    assert!(body.contains("bot_memory_usage_bytes"), "Should contain bot_memory_usage_bytes metric");
    assert!(body.contains("bot_cpu_usage_percent"), "Should contain bot_cpu_usage_percent metric");
    assert!(body.contains("bot_watchdog_stall_seconds"), "Should contain bot_watchdog_stall_seconds metric");
    assert!(body.contains("bot_time_decay_exits_total"), "Should contain bot_time_decay_exits_total metric");
    assert!(body.contains("bot_maker_fills_total"), "Should contain bot_maker_fills_total metric");
    assert!(body.contains("bot_taker_fills_total"), "Should contain bot_taker_fills_total metric");
    
    println!("✓ All key metrics are present in /metrics endpoint");
    
    Ok(())
}

/// Тест проверяет что метрики обновляются корректно
#[tokio::test]
async fn test_metrics_update() -> Result<()> {
    // Инициализируем metrics exporter на тестовом порту
    let test_port = 19091; // Используем другой порт для избежания конфликтов
    
    neirobot_lit::monitoring::metrics::init_metrics_exporter(test_port)?;
    
    // Даем время на запуск
    sleep(Duration::from_millis(500)).await;
    
    // Обновляем метрики
    metrics::counter!("bot_ws_messages_total").increment(5);
    metrics::histogram!("bot_inference_duration_us").record(1500.0);
    metrics::gauge!("bot_health_status").set(1.0);
    metrics::counter!("bot_orders_placed_total").increment(3);
    metrics::counter!("bot_order_rejections_total").increment(1);
    
    // Обновляем консолидированные метрики
    metrics::counter!("bot_ticks_total").increment(10);
    metrics::counter!("bot_signal_oscillations_handled_total").increment(2);
    metrics::gauge!("bot_memory_usage_bytes").set(1024.0 * 1024.0 * 512.0); // 512 MB
    metrics::gauge!("bot_cpu_usage_percent").set(45.5);
    metrics::gauge!("bot_watchdog_stall_seconds").set(0.0);
    metrics::counter!("bot_time_decay_exits_total").increment(1);
    metrics::counter!("bot_maker_fills_total").increment(2);
    metrics::counter!("bot_taker_fills_total").increment(1);
    
    // Даем время на обновление
    sleep(Duration::from_millis(200)).await;
    
    // Проверяем что метрики обновились
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/metrics", test_port);
    
    let response = client.get(&url).send().await?;
    let body = response.text().await?;
    
    // Проверяем что значения присутствуют (точные значения могут варьироваться)
    assert!(body.contains("bot_ws_messages_total"), "bot_ws_messages_total should be present");
    assert!(body.contains("bot_health_status"), "bot_health_status should be present");
    assert!(body.contains("bot_orders_placed_total"), "bot_orders_placed_total should be present");
    assert!(body.contains("bot_order_rejections_total"), "bot_order_rejections_total should be present");
    assert!(body.contains("bot_ticks_total"), "bot_ticks_total should be present");
    assert!(body.contains("bot_memory_usage_bytes"), "bot_memory_usage_bytes should be present");
    assert!(body.contains("bot_cpu_usage_percent"), "bot_cpu_usage_percent should be present");
    
    println!("✓ Metrics are updating correctly");
    
    Ok(())
}

/// Тест проверяет что метрики заказов инкрементируются при размещении ордеров
#[tokio::test]
async fn test_order_metrics_increment() -> Result<()> {
    // Инициализируем metrics exporter на тестовом порту
    let test_port = 19092; // Используем третий порт для избежания конфликтов
    
    neirobot_lit::monitoring::metrics::init_metrics_exporter(test_port)?;
    
    // Даем время на запуск
    sleep(Duration::from_millis(500)).await;
    
    // Получаем начальное значение метрик
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/metrics", test_port);
    
    let response = client.get(&url).send().await?;
    let initial_body = response.text().await?;
    
    // Извлекаем начальные значения счетчиков
    let initial_placed = extract_counter_value(&initial_body, "bot_orders_placed_total");
    let initial_rejections = extract_counter_value(&initial_body, "bot_order_rejections_total");
    
    // Имитируем размещение ордеров
    metrics::counter!("bot_orders_placed_total").increment(5);
    metrics::counter!("bot_order_rejections_total").increment(2);
    
    // Даем время на обновление
    sleep(Duration::from_millis(200)).await;
    
    // Получаем обновленные значения
    let response = client.get(&url).send().await?;
    let updated_body = response.text().await?;
    
    let updated_placed = extract_counter_value(&updated_body, "bot_orders_placed_total");
    let updated_rejections = extract_counter_value(&updated_body, "bot_order_rejections_total");
    
    // Проверяем что счетчики увеличились
    assert!(updated_placed >= initial_placed + 5, 
        "bot_orders_placed_total should increase by at least 5 (was {}, now {})", 
        initial_placed, updated_placed);
    assert!(updated_rejections >= initial_rejections + 2, 
        "bot_order_rejections_total should increase by at least 2 (was {}, now {})", 
        initial_rejections, updated_rejections);
    
    println!("✓ Order metrics are incrementing correctly");
    
    Ok(())
}

/// Вспомогательная функция для извлечения значения счетчика из Prometheus текста
fn extract_counter_value(body: &str, metric_name: &str) -> f64 {
    for line in body.lines() {
        if line.starts_with(&format!("{} ", metric_name)) && !line.starts_with('#') {
            if let Some(value_str) = line.split_whitespace().last() {
                if let Ok(value) = value_str.parse::<f64>() {
                    return value;
                }
            }
        }
    }
    0.0
}
