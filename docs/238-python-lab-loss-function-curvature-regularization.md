# Задача 238: Регуляризация кривизны и устойчивость к шуму (Curvature Regularization)

Реализация механизмов регуляризации поверхности функции потерь для повышения обобщающей способности модели LiT. Это необходимо для предотвращения переобучения на микро-шуме стакана и обеспечения стабильности предсказаний.

## 1. Цель задачи
Снизить чувствительность модели к незначительным флуктуациям входов (LOB-снапшотов) через добавление штрафа за кривизну (Curvature Penalty) и аугментацию шумом в процессе обучения.

## 2. Инструкции по реализации для Gemini

### А. Модуль регуляризации ([./python_lab/src/lit_model.py](./python_lab/src/lit_model.py))
1.  **Реализация Curvature Penalty**:
    Добавить функцию для аппроксимации кривизны через конечные разности (Finite Differences), что вычислительно эффективнее прямого Гессиана:
    ```python
    def compute_curvature_penalty(model, inputs, outputs, lambda_=1e-4, epsilon=1e-3):
        # Генерируем случайное направление шума
        v = torch.randn_like(inputs)
        v = v / (torch.norm(v, p=2) + 1e-6) # Нормализация вектора
        
        # Инференс с возмущенными входами
        perturbed_inputs = inputs + epsilon * v
        perturbed_outputs = model(perturbed_inputs)
        
        # Штраф за разницу предсказаний (L2)
        diff = perturbed_outputs - outputs
        return lambda_ * (diff ** 2).mean()
    ```

### Б. Тренировочный цикл ([./python_lab/scripts/train.py](./python_lab/scripts/train.py))
1.  **Интеграция в Loss**:
    В основном цикле обучения добавить штраф к основной функции потерь (Cross-Entropy/Focal Loss):
    ```python
    logits = model(x)
    task_loss = criterion(logits, y)
    
    if config.use_curvature_reg:
        reg_loss = compute_curvature_penalty(model, x, logits, config.curvature_lambda)
        total_loss = task_loss + reg_loss
    ```
2.  **Noise Injection**:
    Добавить опциональный слой или функцию `apply_input_noise(x, std=0.01)`, которая накладывает гауссов шум на нормализованные признаки перед подачей в модель.

## 3. Конфигурация
Добавить в настройки обучения:
-   **use_curvature_reg**: `bool` (по умолчанию `true`).
-   **curvature_lambda**: `f64` (рекомендуется `1e-4` – `1e-3`).
-   **input_noise_std**: `f64` (например, `0.005`).

## 4. Ожидаемый результат
1.  Уменьшение «дрожания» вероятностей (Confidence Jitter) при стабильных рыночных условиях.
2.  Повышение устойчивости модели к спуфингу и фиктивным заявкам в стакане.
3.  Улучшение метрики **MCC** на валидационном сете с искусственно добавленным шумом.

## 5. Необходимые зависимости
-   **Python**: `pytorch >= 2.0`, `numpy`.