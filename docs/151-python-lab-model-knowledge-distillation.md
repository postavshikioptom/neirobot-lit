# Задача №151: Внедрение Knowledge Distillation (Дистилляция знаний) на PyTorch

Согласно [./docs/000-architecture.md](./docs/000-architecture.md), мы стремимся к минимальной латентности. Дистилляция позволит сжать тяжелый Трансформер (Teacher) в компактную модель (Student) для Rust-инференса без критической потери точности (MCC).

## План реализации для Gemini AI Coder:

### 1. Подготовка: Обучение "Тяжелого" Учителя (Step 0)
В текущем списке задач нет этапа обучения Teacher. Задача 151 должна начинаться с этого:
- **Действие**: Запустить `train.py` с "тяжелым" конфигом (например, `n_layers=8`, `n_heads=8`, `d_model=256`).
- **Результат**: Сохранить веса в `bots/SYMBOL/model/teacher_lit.pt`. Это "эталон", который мы будем дистиллировать.

### 2. Интеграция в [./python_lab/scripts/train.py](./python_lab/scripts/train.py)
Не создавать новый файл, добавить режим `--mode distill`.
- **Параметры**: `--teacher_path`, `--alpha` (default 0.9), `--temperature` (default 3.0).
- **Загрузка**:
    - Загрузить Teacher, перевести в `teacher.eval()`.
    - Использовать `requires_grad_(False)` для всех параметров учителя.
- **Цикл обучения**:
    ```python
    with torch.no_grad():
        teacher_logits = teacher(inputs) # Экономия памяти
    student_logits = student(inputs)
    
    # Расчет потерь (KD Loss)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    ```
    *Примечание: Выбор **alpha=0.9** приоритетнее для LOB-данных, чтобы Student максимально перенимал вероятностное распределение Teacher на несбалансированном классе Flat.*

### 3. Изменения в [./python_lab/src/lit_model.py](./python_lab/src/lit_model.py)
- Добавить поддержку `student_config` (например, `n_layers=2`, `n_heads=4`, `d_model=64`).
- Реализовать метод для инициализации модели по переданному объекту конфигурации.

### 4. Метрики и Бенчмаркинг в [./python_lab/src/utils.py](./python_lab/src/utils.py)
Добавить функцию замера латентности:
- **Инструмент**: Использовать `torch.cuda.Event(enable_timing=True)` для точного замера времени инференса на GPU.
- **Сравнение**: После обучения выводить в лог:
    - **Teacher**: MCC, Mean Latency (ms), Params Count.
    - **Student**: MCC, Mean Latency (ms), Params Count.
    - **Speedup**: Коэффициент ускорения (например, 4.5x).

### 5. Интеграция с Optuna (см. задачу 030)
Добавить поиск оптимальной пары `(T, alpha)`. Для зашумленных LOB-данных "размягчение" через `T` критично, чтобы Student не переобучался на шуме Hard Labels.

### 6. Экспорт (см. задачу 031)
Сохранять итогового Student-а в `bots/SYMBOL/model/lit.pt`, который затем будет сконвертирован в `lit.onnx` для Rust.

## Аргументация (Спор с Grok):
- **Согласен**: Использование `torch.no_grad()` для Teacher обязательно для экономии VRAM.
- **Согласен**: `alpha=0.9` (высокий вес на soft loss) лучше для LOB, так как "мягкие" метки учителя содержат информацию о близости классов, которую теряет обычная кросс-энтропия.
- **Согласен**: Замер латентности прямо в Python Lab необходим для ранней валидации ускорения перед деплоем в Rust.

**Gemini, приступай к реализации. Только PyTorch. Только высокая производительность.**