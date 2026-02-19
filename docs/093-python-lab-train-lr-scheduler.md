# Задача 093: Внедрение планировщика скорости обучения (LR Scheduler)

**Цель**: Реализовать гибкую систему управления Learning Rate (LR) и импульсом (Momentum) для ускорения обучения и стабилизации градиентов в глубоких трансформерах.

## 1. Изменения в [./python/lit_model.py](./python/lit_model.py)
*   **Метод `configure_optimizers`**:
    *   Реализовать поддержку нескольких стратегий через `SequentialLR` или встроенные функции:
        1.  **Linear Warmup + OneCycleLR**: Комбинированный подход (разогрев в первые 10% шагов для стабилизации эмбеддингов, затем цикл).
        2.  **ReduceLROnPlateau**: Адаптивное снижение LR при стагнации `val_loss` (идеально для зашумленных LOB-данных).
        3.  **CosineAnnealingWithWarmup**: Плавное затухание по косинусу.
    *   **AdamW Momentum**: При использовании `OneCycleLR` активировать циклическое изменение импульса (`cycle_momentum=True`), что улучшает обобщающую способность.
*   **Логирование**: Добавить `self.log("lr", ..., on_step=True)` и `self.log("momentum", ...)` для мониторинга в TensorBoard.

## 2. Интеграция с [./python/train.py](./python/train.py) (Optuna)
*   **Поиск гиперпараметров**:
    ```python
    scheduler = trial.suggest_categorical("scheduler", ["onecycle", "plateau", "cosine", "step", "none"])
    # Параметры для OneCycle
    div_factor = trial.suggest_float("div_factor", 10.0, 40.0) 
    ```
*   **Динамический расчет**: Использовать `self.trainer.estimated_stepping_batches` для автоматического вычисления `total_steps` в зависимости от `max_epochs` и `batch_size`.

## 3. Почему этот план лучше (Аргументы Grok):
*   **Explicit Warmup**: Трансформеры крайне чувствительны к большим градиентам на старте. Линейный разогрев предотвращает "развал" весов до начала основного цикла.
*   **ReduceLROnPlateau**: Позволяет модели "доучиваться" на плато, что часто дает прирост в 0.5-1% **MCC** на финальных стадиях.
*   **SequentialLR**: Использование стандартного `SequentialLR` из PyTorch гарантирует корректный переход от Warmup к основной стратегии без скачков LR.

## 4. Тестирование
*   **Stability Test**: Проверка отсутствия `NaN` при высоком `initial_lr` с использованием Warmup.
*   **Ablation**: Сравнение `Constant LR` vs `OneCycle`. Ожидается более высокая точность при меньшем количестве эпох.

---
**Напиши следующую задачу: 094-python-lab-cross-validation-logic.md**