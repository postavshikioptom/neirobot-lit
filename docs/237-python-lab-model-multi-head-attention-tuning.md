# Задача 237: Оптимизация Multi-Head Attention (Heads/Embedding Tuning)

Тонкая настройка архитектуры внимания в модели LiT. Необходимо найти оптимальное сочетание количества голов (`num_heads`) и размерности эмбеддингов (`embed_dim`) для максимизации предсказательной способности при строгом ограничении задержки инференса (latency) на CPU.

## 1. Цель задачи
Использовать байесовскую оптимизацию (**Optuna**) для подбора параметров MHA (Multi-Head Attention), минимизируя вычислительную сложность модели без потери качества классификации (MCC).

## 2. Инструкции по реализации для Gemini

### А. Модель ([./python_lab/src/lit_model.py](./python_lab/src/lit_model.py))
1.  **Параметризация архитектуры**:
    *   Обновить конструктор `LitModel`:
        ```python
        class LitModel(nn.Module):
            def __init__(self, embed_dim: int = 64, num_heads: int = 8, dropout: float = 0.1, use_gqa: bool = False):
                super().__init__()
                assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                
                # Основной механизм: Scaled Dot Product Attention (Flash-ready)
                # Опционально: Реализация Grouped Query Attention (GQA) для сравнения
                self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        ```
2.  **Инференс**: В методе `forward` использовать `torch.nn.functional.scaled_dot_product_attention` для автоматической активации Flash Attention на поддерживаемом железе.

### Б. Оптимизация ([./python_lab/scripts/tune_attention.py](./python_lab/scripts/tune_attention.py))
1.  **Optuna Study**:
    *   **Пространство поиска**:
        *   `num_heads`: [2, 4, 8, 16]
        *   `embed_dim`: [32, 64, 128, 256] (с условием кратности)
    *   **Целевая функция (Objective)**:
        *   `score = validation_mcc - lambda * inference_latency_ms`
        *   Замерять `inference_latency_ms` через `onnxruntime` на **CPU** (имитация реального бота).
2.  **Latency Constraint**: Установить жесткий порог `latency < 2.0ms` на снапшот. Испытания проводить на `onnxruntime.InferenceSession` с `CPUExecutionProvider`.

## 3. Спорные моменты и аргументация

-   **MHA vs GQA (По Grok)**: Хотя GQA эффективнее на сверхдлинных последовательностях (LLM), для фиксированных окон LOB (задача 026) стандартный **MHA** часто показывает лучшую точность при сопоставимой скорости. Оставляем GQA как вторичный эксперимент.
-   **Почему Optuna?**: В отличие от Grid Search, Optuna позволяет использовать **Pruning** (остановка заведомо плохих попыток), что критично при переборе тяжелых архитектур внимания.
-   **Inference Benchmark**: Замер латентности должен происходить **вне** PyTorch (в ONNX), так как именно этот результат критичен для Rust-ядра.

## 4. Ожидаемый результат
1.  Файл конфигурации `best_mha_config.json` с оптимальными `num_heads` и `embed_dim`.
2.  MCC модели не снижается более чем на 1% при сокращении времени инференса на 30%+.
3.  График фронта Парето (Accuracy vs Latency) в папке `reports/`.

## 5. Необходимые зависимости
-   **Python**: `pytorch >= 2.0`, `optuna`, `onnxruntime`, `flash-attn` (опционально для ускорения обучения).