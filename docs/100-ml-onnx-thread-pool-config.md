# Задача 100: ML ONNX Thread Pool Config

## 1. Цель
Настроить пулы потоков ONNX Runtime (`ort`) в [src/ml/onnx.rs](./src/ml/onnx.rs). Это необходимо для предотвращения **CPU Contention** (соперничества за ресурсы процессора) при запуске нескольких экземпляров ботов на одном сервере. Без этой настройки каждый бот попытается занять все доступные ядра процессора, что резко увеличит задержки (latency).

## 2. Изменения

### Файл: [src/config/types.rs](./src/config/types.rs)
- Добавить структуру `OnnxConfig` (если не была добавлена в задаче 096):
```rust
#[derive(Debug, Deserialize, Clone)]
pub struct OnnxConfig {
    pub intra_threads: Option<usize>, // Потоки внутри операторов
    pub inter_threads: Option<usize>, // Потоки между операторами
    pub execution_provider: String,   // "cpu", "cuda", "tensorrt"
}
```
- Интегрировать её в `BotConfig`.

### Файл: [src/ml/onnx.rs](./src/ml/onnx.rs)
- **Использование num_cpus**: Добавить зависимость `num_cpus = "1.16"` в `Cargo.toml`.
- **Логика расчета потоков**:
    - Если `intra_threads` не задан в конфиге, рассчитывать его как `num_cpus::get_physical() / количество_активных_ботов` (минимум 1).
    - `inter_threads` для небольших моделей (LOB) рекомендуется устанавливать в **1**, чтобы избежать лишних переключений контекста.
- **Настройка сессии**:
```rust
let mut builder = Session::builder()?;

// Настройка intra-op (внутри операторов)
let intra = config.onnx.intra_threads.unwrap_or_else(|| {
    let cp_total = num_cpus::get_physical();
    (cp_total / 2).max(1) // Консервативный дефолт
});
builder.with_intra_op_num_threads(intra as i32)?;

// Настройка inter-op (между операторами)
let inter = config.onnx.inter_threads.unwrap_or(1);
builder.with_inter_op_num_threads(inter as i32)?;
```

## 3. Критические требования
- **Изоляция ресурсов**: Каждый экземпляр бота должен иметь жестко ограниченный пул потоков, чтобы не мешать другим процессам и ОС.
- **Логирование**: При инициализации сессии выводить `tracing::info!` с фактическими значениями `intra` и `inter` потоков.
- **Совместимость с EP**: Учитывать, что при использовании **CUDA/TensorRT** нагрузка на CPU снижается, но пулы потоков все равно должны быть настроены для пре-процессинга и управления очередью.

## 4. Зависимости
- `num_cpus` (для динамического определения ядер).
- `ort` (актуальная версия с поддержкой `SessionBuilder`).

