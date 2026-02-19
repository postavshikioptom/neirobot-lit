Задача 082: Мониторинг задержек инференса (Inference & Tensor Bench)
Цель: Реализовать детальный замер времени выполнения ML-пайплайна (сборка тензора + инференс) с поддержкой прогрева модели и мониторингом перцентилей.

1. Создание ./src/monitoring/latency.rs
Структура InferenceStats: Использовать атомарные типы (std::sync::atomic::AtomicU64) для потокобезопасного сбора статистики без блокировок.
pub struct InferenceStats {
    pub tensor_build_sum_us: AtomicU64,
    pub model_run_sum_us: AtomicU64,
    pub max_run_us: AtomicU64,
    pub count: AtomicU64,
}
// Реализовать методы update(tensor_us, model_us) и reset()
2. Изменения в ./src/ml/onnx.rs
Метод warmup: Добавить метод для "прогрева" ONNX-сессии (lazy-init оптимизация).
Выполнить 10-50 прогонов session.run с нулевыми тензорами при старте бота. Это исключит "пики" задержки на первых итерациях.
Замеры в методе predict:
let start_build = Instant::now();
Сборка тензора (задача 035). Использовать ort::Value::from_array для zero-copy передачи данных из ndarray.
let build_us = start_build.elapsed().as_micros() as u64;
let start_run = Instant::now();
self.session.run(...) (задача 036).
let run_us = start_run.elapsed().as_micros() as u64;
Обновить InferenceStats.
3. Мониторинг и логирование
Пороги (Thresholds):
Если model_run_us > 15_000 (15мс) -> warn!("Slow inference detected: {}us", run_us).
Периодический отчет: В цикле run_loop (./src/trading/execution.rs) добавить tokio::time::interval(60s):
Вычислять avg_build = sum_build / count.
Вычислять avg_run = sum_run / count.
info!("[ML Stats] Build: {}us, Model: {}us, Max: {}us, Samples: {}", ...).
4. Почему этот план лучше (Аргументы Grok):
Correct Paths: Модуль monitoring/ выделен специально для латентности (081-082), а ml/onnx.rs отвечает за инференс.
Model Warmup: ONNX (особенно с CUDA/TensorRT, если будут добавлены) требует прогрева для компиляции графа. Без этого первые сигналы будут приходить с задержкой 100мс+.
15ms Limit: Для микро-трендов на LOB задержка выше 15-20мс делает сигнал неактуальным. 50мс было слишком много.
Zero-Alloc: Использование ort::Value из ndarray (задача 035) минимизирует аллокации в горячем цикле.
5. Критические требования
Microseconds: Только as_micros().
Atomic Performance: Использовать Ordering::Relaxed для обновления счетчиков (максимальная скорость).
Warmup Call: Вызвать warmup() в main.rs сразу после инициализации OnnxEngine.
6. Тестирование
Unit test: Подать фиксированные значения задержек и проверить правильность расчета avg.
Integration test: Убедиться, что после вызова warmup первый реальный инференс проходит быстро (< 5мс).