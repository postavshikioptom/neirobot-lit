// Задача 150: Stress Testing High Frequency Pipeline with Criterion
// Комплексное стресс-тестирование торгового конвейера под нагрузкой 20,000+ msg/sec

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neirobot_lit::data::orderbook::OrderBook;
use neirobot_lit::data::types::{OrderBookUpdate, PriceLevel};
use neirobot_lit::ml::types::{Signal, InferenceOutput};
use neirobot_lit::ml::onnx::OnnxEngine;
use neirobot_lit::config::types::OnnxConfig;
use neirobot_lit::monitoring::prometheus::{WATCHDOG_CHECK_GAUGE, init_metrics};
use smallvec::SmallVec;
use tokio::sync::mpsc;
use std::time::{Duration, Instant};
use std::path::Path;
use rand::Rng;
use ndarray::Array2;
use chrono::Utc;
use anyhow::Result;

/// Режим работы ML движка
enum MlEngineMode {
    /// Быстрый мок для CI/CD тестов
    Mock(MockOnnxEngine),
    /// Реальная ONNX модель для production validation
    Real(OnnxEngine),
}

impl MlEngineMode {
    fn predict(&self, input_data: &[f32]) -> Result<InferenceOutput> {
        match self {
            MlEngineMode::Mock(engine) => Ok(engine.predict(input_data)),
            MlEngineMode::Real(engine) => engine.predict(input_data),
        }
    }
}

/// Mock ONNX Engine для быстрого инференса без реальной модели
struct MockOnnxEngine {
    seq_len: usize,
    input_features: usize,
}

impl MockOnnxEngine {
    fn new(seq_len: usize, input_features: usize) -> Self {
        Self { seq_len, input_features }
    }

    /// Симулирует инференс с минимальной задержкой (~10-50 микросекунд)
    fn predict(&self, _input_data: &[f32]) -> InferenceOutput {
        // Симулируем небольшую вычислительную нагрузку
        let mut sum = 0.0f32;
        for i in 0..100 {
            sum += (i as f32).sin();
        }
        
        // Генерируем случайные вероятности
        let mut rng = rand::thread_rng();
        let logits = [
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
        ];
        
        // Softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        let probs = [
            exps[0] / sum_exps,
            exps[1] / sum_exps,
            exps[2] / sum_exps,
        ];
        
        // Определяем сигнал
        let signal_idx = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        InferenceOutput {
            signal: Signal::from(signal_idx),
            probabilities: probs.to_vec(),
            probs: Array2::from_shape_vec((1, 3), probs.to_vec()).unwrap(),
            source_timestamp_ms: 0,
        }
    }
}

/// Генератор реалистичных LOB Delta обновлений
struct LobUpdateGenerator {
    symbol: String,
    base_price: f64,
    update_id: u64,
}

impl LobUpdateGenerator {
    fn new(symbol: &str, base_price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            base_price,
            update_id: 1,
        }
    }

    /// Генерирует случайное LOB обновление с реалистичными ценами и объемами
    fn generate(&mut self) -> OrderBookUpdate {
        let mut rng = rand::thread_rng();
        
        // Генерируем небольшое отклонение от базовой цены
        let price_deviation = rng.gen_range(-0.001..0.001);
        let mid_price = self.base_price * (1.0 + price_deviation);
        
        // Генерируем спред (0.01% - 0.05%)
        let spread_pct = rng.gen_range(0.0001..0.0005);
        let half_spread = mid_price * spread_pct;
        
        let best_bid = mid_price - half_spread;
        let best_ask = mid_price + half_spread;
        
        // Генерируем 1-3 уровня обновлений
        let num_levels = rng.gen_range(1..=3);
        
        let mut bids = SmallVec::new();
        let mut asks = SmallVec::new();
        
        for i in 0..num_levels {
            let bid_price = best_bid - (i as f64 * 0.01);
            let ask_price = best_ask + (i as f64 * 0.01);
            
            let bid_size = rng.gen_range(0.1..10.0);
            let ask_size = rng.gen_range(0.1..10.0);
            
            bids.push(PriceLevel {
                price: bid_price,
                size: bid_size,
            });
            
            asks.push(PriceLevel {
                price: ask_price,
                size: ask_size,
            });
        }
        
        self.update_id += 1;
        
        OrderBookUpdate {
            symbol: self.symbol.clone(),
            timestamp_ms: Utc::now().timestamp_millis() as u64,
            last_update_id: self.update_id,
            is_snapshot: false,
            bids,
            asks,
            checksum: None,
        }
    }
}

/// Полный торговый конвейер для бенчмаркинга
struct TradingPipeline {
    orderbook: OrderBook,
    ml_engine: MlEngineMode,
    feature_buffer: Vec<f32>,
    processed_count: u64,
    symbol: String,
}

impl TradingPipeline {
    fn new_mock(symbol: &str, seq_len: usize, input_features: usize) -> Self {
        Self {
            orderbook: OrderBook::new(symbol),
            ml_engine: MlEngineMode::Mock(MockOnnxEngine::new(seq_len, input_features)),
            feature_buffer: vec![0.0f32; seq_len * input_features],
            processed_count: 0,
            symbol: symbol.to_string(),
        }
    }

    fn new_real(symbol: &str, model_path: &Path, seq_len: usize, input_features: usize) -> Result<Self> {
        // Конфигурация для CPU (для бенчмарков)
        let onnx_config = OnnxConfig {
            execution_provider: "cpu".to_string(),
            device_id: 0,
            intra_threads: Some(4),
            inter_threads: Some(1),
        };

        let engine = OnnxEngine::load(model_path, seq_len, input_features, &onnx_config, symbol, None)?;

        Ok(Self {
            orderbook: OrderBook::new(symbol),
            ml_engine: MlEngineMode::Real(engine),
            feature_buffer: vec![0.0f32; seq_len * input_features],
            processed_count: 0,
            symbol: symbol.to_string(),
        })
    }

    /// Обрабатывает одно LOB обновление через весь конвейер
    async fn process(&mut self, update: OrderBookUpdate) -> Result<InferenceOutput> {
        // 1. Обновляем OrderBook
        self.orderbook.apply_update(&update);
        
        // 2. Извлекаем фичи из стакана
        self.orderbook.fill_flat_buffer(10, &mut self.feature_buffer[..40]);
        
        // 3. Запускаем ML инференс (реальный или мок)
        let output = self.ml_engine.predict(&self.feature_buffer)?;
        
        // 4. Симулируем обработку в ExecutionEngine
        // (в реальности здесь был бы вызов on_inference_output)
        self.processed_count += 1;
        
        // 5. Обновляем watchdog метрику
        if let Some(gauge) = WATCHDOG_CHECK_GAUGE.get() {
            gauge.with_label_values(&[&self.symbol]).set(Utc::now().timestamp() as f64);
        }
        
        Ok(output)
    }
}

/// Setup функция для инициализации полного конвейера с моком
fn setup_full_pipeline() -> TradingPipeline {
    TradingPipeline::new_mock("BTCUSDT", 100, 40)
}

/// Setup функция для инициализации полного конвейера с реальной моделью
fn setup_real_pipeline(model_path: &Path) -> Result<TradingPipeline> {
    TradingPipeline::new_real("BTCUSDT", model_path, 100, 40)
}

/// Основной бенчмарк: End-to-End латентность при 20k TPS
fn bench_full_hot_path(c: &mut Criterion) {
    // Инициализируем метрики для watchdog
    init_metrics("BTCUSDT");
    
    let mut group = c.benchmark_group("hot_path");
    
    // Настройка для точных измерений
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);
    
    group.bench_function("pipeline_tick_20k_tps", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .unwrap();
        
        b.to_async(&rt).iter(|| async {
            let mut pipeline = setup_full_pipeline();
            let mut generator = LobUpdateGenerator::new("BTCUSDT", 50000.0);
            
            let msg = generator.generate();
            black_box(pipeline.process(msg).await).unwrap();
            
            // Проверяем, что watchdog обновляется
            check_watchdog_health("BTCUSDT");
        });
    });
    
    group.finish();
}

/// Проверка здоровья watchdog метрики
fn check_watchdog_health(symbol: &str) {
    if let Some(gauge) = WATCHDOG_CHECK_GAUGE.get() {
        let last_check = gauge.with_label_values(&[symbol]).get();
        let now = Utc::now().timestamp() as f64;
        let stall_duration = now - last_check;
        
        // Если watchdog не обновлялся более 5 секунд - это проблема
        if stall_duration > 5.0 {
            panic!("Watchdog stall detected! Last check was {} seconds ago", stall_duration);
        }
    }
}


/// Бенчмарк: Burst нагрузка 50k msg/sec в течение 10 секунд
fn bench_burst_load(c: &mut Criterion) {
    init_metrics("BTCUSDT");
    
    let mut group = c.benchmark_group("burst_load");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(100);
    
    group.bench_function("burst_50k_tps_10sec", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(8)
            .build()
            .unwrap();
        
        b.to_async(&rt).iter(|| async {
            let mut pipeline = setup_full_pipeline();
            let mut generator = LobUpdateGenerator::new("BTCUSDT", 50000.0);
            
            let start = Instant::now();
            let target_duration = Duration::from_secs(10);
            let target_rate = 50_000; // msg/sec
            let interval_us = 1_000_000 / target_rate;
            
            let mut processed = 0u64;
            let mut next_tick = start;
            
            while start.elapsed() < target_duration {
                let msg = generator.generate();
                if let Ok(_) = pipeline.process(msg).await {
                    processed += 1;
                }
                
                // Проверяем watchdog каждые 1000 сообщений
                if processed % 1000 == 0 {
                    check_watchdog_health("BTCUSDT");
                }
                
                // Rate limiting
                next_tick += Duration::from_micros(interval_us);
                let now = Instant::now();
                if next_tick > now {
                    tokio::time::sleep(next_tick - now).await;
                }
            }
            
            black_box(processed)
        });
    });
    
    group.finish();
}

/// Бенчмарк: Sustained нагрузка 20k msg/sec с мониторингом памяти
fn bench_sustained_load_with_memory(c: &mut Criterion) {
    init_metrics("BTCUSDT");
    
    let mut group = c.benchmark_group("sustained_load");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);
    
    group.bench_function("sustained_20k_tps_60sec", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .unwrap();
        
        b.to_async(&rt).iter(|| async {
            let mut pipeline = setup_full_pipeline();
            let mut generator = LobUpdateGenerator::new("BTCUSDT", 50000.0);
            
            // Получаем начальное использование памяти
            let initial_memory = get_memory_usage_mb();
            
            let start = Instant::now();
            let target_duration = Duration::from_secs(60);
            let target_rate = 20_000;
            let interval_us = 1_000_000 / target_rate;
            
            let mut processed = 0u64;
            let mut next_tick = start;
            let mut max_memory = initial_memory;
            
            while start.elapsed() < target_duration {
                let msg = generator.generate();
                if let Ok(_) = pipeline.process(msg).await {
                    processed += 1;
                }
                
                // Проверяем память и watchdog каждые 10 секунд
                if processed % 200_000 == 0 {
                    let current_memory = get_memory_usage_mb();
                    max_memory = max_memory.max(current_memory);
                    check_watchdog_health("BTCUSDT");
                }
                
                // Rate limiting
                next_tick += Duration::from_micros(interval_us);
                let now = Instant::now();
                if next_tick > now {
                    tokio::time::sleep(next_tick - now).await;
                }
            }
            
            let final_memory = get_memory_usage_mb();
            let memory_growth = final_memory - initial_memory;
            
            // Проверяем критерий: рост памяти должен быть минимальным
            assert!(
                memory_growth < 100.0,
                "Memory growth too high: {} MB (initial: {}, final: {})",
                memory_growth, initial_memory, final_memory
            );
            
            black_box((processed, memory_growth))
        });
    });
    
    group.finish();
}

/// Бенчмарк: Backlog detection - проверка заполненности каналов
fn bench_backlog_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("backlog_detection");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);
    
    group.bench_function("channel_capacity_check", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .unwrap();
        
        b.to_async(&rt).iter(|| async {
            let mut pipeline = setup_full_pipeline();
            let mut generator = LobUpdateGenerator::new("BTCUSDT", 50000.0);
            
            // Создаем канал для симуляции межкомпонентной связи
            let (tx, mut rx) = mpsc::channel::<InferenceOutput>(1000);
            
            let start = Instant::now();
            let target_duration = Duration::from_secs(5);
            let target_rate = 20_000;
            let interval_us = 1_000_000 / target_rate;
            
            let mut processed = 0u64;
            let mut next_tick = start;
            let mut min_capacity = 1000usize;
            
            // Запускаем consumer
            let consumer_handle = tokio::spawn(async move {
                while let Some(_output) = rx.recv().await {
                    // Симулируем обработку
                    tokio::time::sleep(Duration::from_micros(10)).await;
                }
            });
            
            while start.elapsed() < target_duration {
                let msg = generator.generate();
                if let Ok(output) = pipeline.process(msg).await {
                    // Проверяем capacity канала
                    let current_capacity = tx.capacity();
                    min_capacity = min_capacity.min(current_capacity);
                    
                    // Отправляем в канал (non-blocking)
                    let _ = tx.try_send(output);
                    
                    processed += 1;
                }
                
                // Rate limiting
                next_tick += Duration::from_micros(interval_us);
                let now = Instant::now();
                if next_tick > now {
                    tokio::time::sleep(next_tick - now).await;
                }
            }
            
            drop(tx);
            let _ = consumer_handle.await;
            
            // Проверяем критерий: канал не должен быть переполнен
            assert!(
                min_capacity > 0,
                "Channel backlog detected! Min capacity: {}",
                min_capacity
            );
            
            black_box((processed, min_capacity))
        });
    });
    
    group.finish();
}

/// Бенчмарк: Latency distribution - проверка P99.9
fn bench_latency_distribution(c: &mut Criterion) {
    init_metrics("BTCUSDT");
    
    let mut group = c.benchmark_group("latency_distribution");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    
    group.bench_function("p999_latency_check", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .unwrap();
        
        b.to_async(&rt).iter(|| async {
            let mut pipeline = setup_full_pipeline();
            let mut generator = LobUpdateGenerator::new("BTCUSDT", 50000.0);
            
            let mut latencies = Vec::with_capacity(100_000);
            
            let start = Instant::now();
            let target_duration = Duration::from_secs(30);
            let target_rate = 20_000;
            let interval_us = 1_000_000 / target_rate;
            
            let mut next_tick = start;
            
            while start.elapsed() < target_duration {
                let msg = generator.generate();
                
                let tick_start = Instant::now();
                let _ = pipeline.process(msg).await;
                let tick_latency = tick_start.elapsed();
                
                latencies.push(tick_latency.as_micros() as u64);
                
                // Rate limiting
                next_tick += Duration::from_micros(interval_us);
                let now = Instant::now();
                if next_tick > now {
                    tokio::time::sleep(next_tick - now).await;
                }
            }
            
            // Сортируем для расчета перцентилей
            latencies.sort_unstable();
            
            let p50_idx = latencies.len() / 2;
            let p99_idx = (latencies.len() as f64 * 0.99) as usize;
            let p999_idx = (latencies.len() as f64 * 0.999) as usize;
            
            let p50 = latencies.get(p50_idx).copied().unwrap_or(0);
            let p99 = latencies.get(p99_idx).copied().unwrap_or(0);
            let p999 = latencies.get(p999_idx).copied().unwrap_or(0);
            let mean = latencies.iter().sum::<u64>() / latencies.len() as u64;
            
            // Проверяем критерии приемки
            assert!(
                mean < 500,
                "Mean latency too high: {} us (target: < 500 us)",
                mean
            );
            
            assert!(
                p999 < 2000,
                "P99.9 latency too high: {} us (target: < 2000 us)",
                p999
            );
            
            println!("\nLatency Statistics:");
            println!("  Mean: {} us", mean);
            println!("  P50:  {} us", p50);
            println!("  P99:  {} us", p99);
            println!("  P99.9: {} us", p999);
            
            // Финальная проверка watchdog
            check_watchdog_health("BTCUSDT");
            
            black_box((mean, p50, p99, p999))
        });
    });
    
    group.finish();
}

/// Бенчмарк: Реальная ONNX модель - влияние высокой частоты на session.run()
/// Этот бенчмарк требует наличия реальной модели в bots/BTCUSDT/model/model.onnx
fn bench_real_model_inference(c: &mut Criterion) {
    init_metrics("BTCUSDT");
    
    // Проверяем наличие модели
    let model_path = Path::new("bots/BTCUSDT/model/model.onnx");
    if !model_path.exists() {
        eprintln!("Skipping real model benchmark: model not found at {:?}", model_path);
        return;
    }
    
    let mut group = c.benchmark_group("real_model");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(50);
    
    group.bench_function("real_onnx_20k_tps", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .unwrap();
        
        b.to_async(&rt).iter(|| async {
            // Создаем pipeline с реальной моделью
            let mut pipeline = match setup_real_pipeline(model_path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Failed to load real model: {}", e);
                    return;
                }
            };
            
            let mut generator = LobUpdateGenerator::new("BTCUSDT", 50000.0);
            
            let start = Instant::now();
            let target_duration = Duration::from_secs(30);
            let target_rate = 20_000;
            let interval_us = 1_000_000 / target_rate;
            
            let mut processed = 0u64;
            let mut next_tick = start;
            let mut inference_times = Vec::with_capacity(10_000);
            
            while start.elapsed() < target_duration {
                let msg = generator.generate();
                
                let inference_start = Instant::now();
                if let Ok(_) = pipeline.process(msg).await {
                    let inference_time = inference_start.elapsed();
                    inference_times.push(inference_time.as_micros() as u64);
                    processed += 1;
                }
                
                // Проверяем watchdog каждые 1000 сообщений
                if processed % 1000 == 0 {
                    check_watchdog_health("BTCUSDT");
                }
                
                // Rate limiting
                next_tick += Duration::from_micros(interval_us);
                let now = Instant::now();
                if next_tick > now {
                    tokio::time::sleep(next_tick - now).await;
                }
            }
            
            // Анализ влияния высокой частоты на inference
            if !inference_times.is_empty() {
                inference_times.sort_unstable();
                let mean = inference_times.iter().sum::<u64>() / inference_times.len() as u64;
                let p99_idx = (inference_times.len() as f64 * 0.99) as usize;
                let p99 = inference_times.get(p99_idx).copied().unwrap_or(0);
                
                println!("\nReal Model Inference Statistics:");
                println!("  Processed: {} messages", processed);
                println!("  Mean inference: {} us", mean);
                println!("  P99 inference: {} us", p99);
            }
            
            black_box(processed)
        });
    });
    
    group.finish();
}

/// Вспомогательная функция для получения использования памяти (RSS) в MB
fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        // Fallback для других ОС
        use sysinfo::{System, SystemExt, ProcessExt, PidExt};
        let mut sys = System::new_all();
        sys.refresh_all();
        
        if let Some(process) = sys.process(sysinfo::get_current_pid().unwrap()) {
            return process.memory() as f64 / 1024.0 / 1024.0; // Convert bytes to MB
        }
    }
    
    0.0
}

// Регистрация всех бенчмарков
criterion_group!(
    benches,
    bench_full_hot_path,
    bench_burst_load,
    bench_sustained_load_with_memory,
    bench_backlog_detection,
    bench_latency_distribution,
    bench_real_model_inference
);

criterion_main!(benches);
