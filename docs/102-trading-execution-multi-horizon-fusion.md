# Задача 102: Trading Execution Multi-Horizon Fusion

## 1. Цель
Реализовать механизм **слияния (fusion)** предсказаний для разных временных горизонтов в [src/trading/execution.rs](./src/trading/execution.rs). Модель выводит вероятности для нескольких горизонтов планирования (например, 10, 50, 100 тиков). Задача — объединить их в единый вектор вероятностей `[Flat, Up, Down]` перед применением асимметричных порогов из задачи 101.

## 2. Изменения

### Файл: [src/ml/types.rs](./src/ml/types.rs)
- Обновить структуру `InferenceOutput` для поддержки матрицы вероятностей:
```rust
use ndarray::Array2;

pub struct InferenceOutput {
    /// Матрица формы [Horizons, 3], где 3 — это классы [Flat, Up, Down]
    pub probs: Array2<f32>,
}
```

### Файл: [src/config/types.rs](./src/config/types.rs)
- Добавить конфигурацию слияния в `BotConfig`:
```rust
#[derive(Debug, Deserialize, Clone)]
pub enum FusionMethod {
    WeightedAverage, // Взвешенное среднее
    Consensus,       // Согласие большинства
    Principal,       // Приоритет одного горизонта
}

pub struct FusionConfig {
    pub method: FusionMethod,
    pub weights: Vec<Decimal>,    // Веса горизонтов
    pub min_horizons: usize,      // Для Consensus
    pub principal_idx: usize,     // Для Principal
}
```

### Файл: [src/trading/execution.rs](./src/trading/execution.rs)
- **Реализация слияния**:
```rust
fn fuse_probs(&self, output: &InferenceOutput) -> [f32; 3] {
    match self.config.fusion.method {
        FusionMethod::WeightedAverage => {
            let mut fused = [0.0f32; 3];
            for (i, weight) in self.config.fusion.weights.iter().enumerate() {
                let w = weight.to_f32().unwrap_or(0.0);
                for cls in 0..3 {
                    fused[cls] += output.probs[[i, cls]] * w;
                }
            }
            fused
        },
        FusionMethod::Consensus => {
            let mut votes_up = 0;
            let mut votes_down = 0;
            let long_th = self.config.long_threshold.to_f32().unwrap_or(0.6);
            let short_th = self.config.short_threshold.to_f32().unwrap_or(0.6);

            for i in 0..output.probs.shape()[0] {
                if output.probs[[i, 1]] > long_th { votes_up += 1; }
                if output.probs[[i, 2]] > short_th { votes_down += 1; }
            }

            if votes_up >= self.config.fusion.min_horizons { [0.0, 1.0, 0.0] }
            else if votes_down >= self.config.fusion.min_horizons { [0.0, 0.0, 1.0] }
            else { [1.0, 0.0, 0.0] }
        },
        FusionMethod::Principal => {
            let idx = self.config.fusion.principal_idx;
            [output.probs[[idx, 0]], output.probs[[idx, 1]], output.probs[[idx, 2]]]
        }
    }
}
```

## 3. Критические требования
- **Порядок классов**: Строго соблюдать индексы: `0: Flat`, `1: Up`, `2: Down` (согласно экспорту модели из задачи 056).
- **Валидация**: В `config/loader.rs` проверять, что `weights.len()` или `principal_idx` соответствуют количеству строк в матрице `probs`.
- **Интеграция с 101**: Метод `fuse_probs` вызывается первым, затем к результату применяются `long_threshold` и `short_threshold`.

## 4. Зависимости
- `ndarray = "0.15"` (обработка матриц).
- `rust_decimal` (конвертация весов).
