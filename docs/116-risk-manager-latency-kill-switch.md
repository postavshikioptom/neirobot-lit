# Задача 116: Предохранитель по задержке (Latency Kill Switch) (v2.0)

## 1. Изменения в конфигурации `src/config/types.rs`
Обнови параметры на микросекунды и добавь лимит повторных отказов:
```rust
pub struct BotConfig {
    // ...
    pub max_network_latency_micros: u64,   // Лимит сети (например, 200000 для 200мс)
    pub max_inference_latency_micros: u64, // Лимит инференса (например, 50000 для 50мс)
    pub max_total_latency_micros: u64,     // Общий лимит (например, 250000)
    pub max_latency_rejects_limit: usize,  // Сколько подряд отказов до блокировки (например, 3)
}
```

## 2. Прогрев модели в `src/ml/onnx.rs`
При инициализации сессии добавь цикл «холостых» прогонов:
```rust
impl OnnxModel {
    pub async fn new(config: &BotConfig) -> Self {
        let session = // ... инициализация ...
        
        // Warm-up: 3 холостых прогона для аллокации памяти и JIT
        let dummy_input = Array4::<f32>::zeros((1, 1, seq_len, features));
        for _ in 0..3 {
            let _ = session.run(dummy_input.clone()).await;
        }
        
        Self { session, ... }
    }
}
```

## 3. Реализация в `src/risk/risk_manager.rs`
Метод проверки с учетом счетчика отказов:
```rust
pub struct RiskManager {
    // ...
    pub consecutive_latency_rejects: usize,
}

impl RiskManager {
    pub fn check_latency(&mut self, network_micros: u64, inference_micros: u64) -> bool {
        let total = network_micros + inference_micros;
        let limit = self.config.max_total_latency_micros;

        if total > limit && limit > 0 {
            self.consecutive_latency_rejects += 1;
            tracing::warn!("LATENCY REJECT: total {}µs (Net: {}µs, Inf: {}µs). Count: {}", 
                total, network_micros, inference_micros, self.consecutive_latency_rejects);
            return false;
        }

        self.consecutive_latency_rejects = 0; // Сброс при успешном проходе
        true
    }
}
```

## 4. Интеграция и замеры в `src/trading/execution.rs`
Реализуй замеры в микросекундах и логику экстренной остановки:

```rust
// В методе обработки сообщения (on_update):

// 1. Сетевая задержка (из WS сообщения)
let now_ms = Utc::now().timestamp_millis();
let network_micros = if now_ms >= msg.exchange_ts {
    (now_ms - msg.exchange_ts) as u64 * 1000
} else {
    tracing::warn!("Clock skew detected: local < exchange_ts");
    0
};

// 2. Замер инференса
let start_inf = Instant::now();
let inference = self.ml.predict(tensor).await?;
let inference_micros = start_inf.elapsed().as_micros() as u64;

// 3. Проверка через RiskManager
if !self.risk_manager.check_latency(network_micros, inference_micros) {
    // Если слишком много отказов подряд — уходим в Waiting Mode (задача 113)
    if self.risk_manager.consecutive_latency_rejects >= self.config.max_latency_rejects_limit {
        self.handle_inactivity_trigger().await; // Используем логику из 113
        tracing::error!("FATAL LATENCY: Too many rejects. Trading suspended.");
    }
    return Ok(());
}
```

---

## Аргументация для Планировщика:
1.  **Microseconds**: В HFT задержка в 5-10мс может быть критичной. Микросекунды дают необходимую детализацию для профилирования.
2.  **Warm-up**: Без прогрева первый сигнал после запуска бота почти гарантированно будет отброшен из-за долгой инициализации тензоров внутри ONNX Runtime.
3.  **Emergency Mode**: Если задержка высокая на 3-х сигналах подряд, значит проблема системная (сеть или перегрузка CPU), и продолжать торговлю опасно.

**Gemini, реализуй эту защиту, обеспечив точные замеры `as_micros()` на каждом этапе конвейера данных.**