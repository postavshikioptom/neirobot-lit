
# Задача 126: Интерпретируемость и оптимизация признаков через SHAP (v2.0)

## 1. Модуль анализа в `python_lab/src/interpretability.py`
Реализуй расчет вкладов признаков, используя современный `GradientExplainer`. Это позволит нам не только понимать модель, но и физически удалять «шумные» уровни стакана для ускорения Rust-инференса.

```python
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

def run_shap_analysis(model, train_data, test_data, depth, save_path):
    """
    model: PyTorch модель в eval mode.
    train_data: тензор для калибровки (background).
    test_data: тензор для анализа.
    depth: глубина стакана (для генерации имен фич).
    """
    # 1. Генерация имен фич (соответствие логике из 022)
    feature_names = []
    for i in range(depth):
        feature_names.extend([f'ask_p_{i}', f'ask_v_{i}', f'bid_p_{i}', f'bid_v_{i}'])
    
    # 2. Подготовка фоновых данных (kmeans эффективнее случайного сэмпла)
    # Используем 100 репрезентативных центроидов
    background = shap.kmeans(train_data.numpy(), 100).data
    background = torch.from_numpy(background).to(train_data.device)
    
    # 3. Инициализация GradientExplainer (рекомендовано для PyTorch)
    explainer = shap.GradientExplainer(model, background)
    
    # Считаем SHAP values для батча (возвращает список для каждого класса: [Up, Down, Flat])
    shap_values = explainer.shap_values(test_data)
    
    return shap_values, feature_names

def prune_features(shap_values, feature_names, threshold=0.01):
    """
    Находит фичи, чей средний абсолютный вклад < threshold (1% от общего).
    """
    # Усредняем по всем классам и сэмплам
    # shap_values shape: [classes][samples, features]
    abs_shap = np.mean([np.abs(v).mean(0) for v in shap_values], axis=0)
    total_impact = np.sum(abs_shap)
    
    importance_pct = abs_shap / total_impact
    to_keep = [name for name, imp in zip(feature_names, importance_pct) if imp >= threshold]
    to_drop = [name for name, imp in zip(feature_names, importance_pct) if imp < threshold]
    
    return to_keep, to_drop, importance_pct
```

## 2. Визуализация и локальный дебаг
Добавь генерацию графиков для каждого торгового сигнала (Long/Short).

```python
def plot_shap_results(shap_values, test_data, feature_names, save_path):
    classes = ['Up', 'Down', 'Flat']
    for i, class_name in enumerate(classes):
        plt.figure()
        # Summary plot для конкретного класса
        shap.summary_plot(
            shap_values[i], 
            test_data.numpy(), 
            feature_names=feature_names, 
            plot_type="bar",
            show=False
        )
        plt.title(f"Feature Importance for {class_name}")
        plt.savefig(f"{save_path}/shap_bar_{class_name.lower()}.png")
        plt.close()

    # Локальное объяснение для самого уверенного прогноза (Force Plot)
    # Выбираем сэмпл с макс. вероятностью и сохраняем его логику
    # (Требуется JS/HTML для полного force_plot, в лабе сохраняем как статичное описание)
```

## 3. Спорные моменты и корректировки (Grok/Zencoder)

*   **GradientExplainer vs DeepExplainer**: Согласен с Grok. `GradientExplainer` лучше работает с графами PyTorch и быстрее сходится на данных временных рядов (LOB).
*   **Multi-class Handling**: Обязательно разделять графики. Нельзя смешивать важность фич для `Up` и `Down`. Для шорта важны аски, для лонга — биды.
*   **Feature Pruning**: Это главная цель. Если `ask_v_19` имеет вклад 0.0001, мы должны вырезать его из `BotConfig` и `src/data/orderbook.rs`, чтобы модель не тратила такты процессора на умножение на ноль.
*   **Background Data**: Только `shap.kmeans`. Случайный сэмпл может пропустить редкие состояния стакана (например, огромный дисбаланс), которые критичны для обучения.

## 4. Инструкции для Gemini (Coder AI):
1.  **python_lab/src/interpretability.py**: Реализовать `run_shap_analysis` и `prune_features`.
2.  **python_lab/evaluate.py**: Добавить вызов анализа. В конце выводить в консоль список рекомендованных к удалению признаков.
3.  **Visuals**: Сохранять результаты в `python_lab/results/interpret/`.
4.  **Dependencies**: Добавить `shap` в `requirements.txt` (или проверить наличие).

**Результат**: Мы получаем список «важных» фич, который позволит нам сократить размер входного вектора модели в Rust, уменьшив задержку (latency) на 10-20%.

---
**Статус**: Задача полностью скорректирована. Жду следующую или команду на выполнение.