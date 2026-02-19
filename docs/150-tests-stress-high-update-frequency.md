# Задача 150: Stress Testing High Frequency Pipeline with Criterion

## 1. Цель
Провести комплексное стресс-тестирование всего торгового конвейера (от получения WS-сообщения до формирования торгового приказа) под нагрузкой 20,000+ msg/sec, используя `criterion` для выявления микро-задержек.

## 2. План реализации
1.  **Micro-benchmarking**: Создать бенчмарк в `benches/hot_path.rs` с использованием `criterion`.
2.  **Full Loop Simulation**:
    - **Data**: Генерация потока валидных LOB Delta обновлений.
    - **ML**: Вызов мока или реальной модели в [./src/trading/onnx.rs](./src/trading/onnx.rs).
    - **Execution**: Обработка результата в [./src/trading/execution.rs](./src/trading/execution.rs).
3.  **Backlog Detection**: Мониторить заполненность каналов связи между компонентами.
4.  **Inference Latency**: Замерить влияние высокой частоты входящих данных на время выполнения `session.run()` (нагрев GPU/CPU, конкуренция за шину).

## 3. Технические детали
- **Criterion Configuration**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_full_hot_path(c: &mut Criterion) {
    let mut rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();
    let mut pipeline = setup_full_pipeline(); // Data + ML + Exec

    c.bench_function("pipeline_tick_20k_tps", |b| {
        b.to_async(&rt).iter(|| async {
            let msg = generate_mock_update();
            black_box(pipeline.process(msg).await).unwrap();
        });
    });
}
```
- **Monitoring Integration**: Во время теста проверять `METRICS.watchdog_last_check` (задача 146). Если Watchdog детектирует Stall при высокой нагрузке — тест провален.
- **Queue Check**: `assert!(tx.capacity() > 0)` — проверка, что каналы не переполнены.

## 4. Критерии приемки
- [ ] Среднее время прохождения всего цикла (End-to-End) < 500 микросекунд при 20k TPS.
- [ ] P99.9 задержка не превышает 2мс (отсутствие тяжелых хвостов распределения).
- [ ] Отсутствие роста памяти (RSS) в течение 10-минутного теста (задача 145).
- [ ] `criterion` генерирует отчет (HTML) с графиками распределения задержек.
- [ ] Система выдерживает "всплеск" до 50,000 msg/sec в течение 10 секунд без паники или потери дескрипторов.
