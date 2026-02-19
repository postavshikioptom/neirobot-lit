Задача 057-python-lab-implement-focal-loss.md
Цель: Реализовать эффективную и численно стабильную версию Focal Loss для фокусировки модели на сложных рыночных событиях (Up/Down).

Инструкции для реализации:
1. Изменения в ./python_lab/src/utils.py
Класс FocalLoss(nn.Module):
Инициализация: Принимает alpha (тензор весов), gamma (focusing parameter) и label_smoothing.
Метод forward(inputs, targets):
Вычислить базовый ce_loss через F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.label_smoothing).
Получить вероятности ptp_tpt​: pt = torch.exp(-ce_loss).
Вычислить финальный лосс: focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean().
Преимущество: Такой подход автоматически учитывает и веса классов (alpha), и сглаживание меток (label_smoothing), оставаясь при этом максимально быстрым.
2. Изменения в ./python_lab/scripts/train.py
Логика выбора в Optuna (Задача 030):
Использовать условные параметры для оптимизации пространства поиска:
loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])
if loss_type == "focal":
    gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
    criterion = FocalLoss(alpha=class_weights, gamma=gamma, label_smoothing=args.label_smoothing)
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
CLI Аргументы: Добавить --focal_gamma (default: 2.0).
3. Обработка весов (alpha)
Убедиться, что class_weights передается как torch.Tensor на правильное устройство (GPU/CPU).
Внутри FocalLoss веса должны применяться к каждому образцу через встроенный механизм F.cross_entropy.
Аргументация изменений:
Эффективность: Использование F.cross_entropy с reduction='none' — это "золотой стандарт" PyTorch. Мы делегируем сложные вычисления оптимизированному C++ ядру, а не пишем циклы на Python.
Гибкость эксперимента: Мы не запрещаем комбинацию Focal + Smoothing, а позволяем Optuna самой найти оптимальное сочетание. Это может быть полезно на крайне зашумленных данных Bybit.
Стабильность: Расчет через exp(-ce_loss) предотвращает проблемы с логарифмом нуля, которые часто возникают при наивной реализации Focal Loss.
Критическое требование: Убедиться, что reduction='mean' применяется в самом конце, чтобы loss оставался сопоставимым по масштабу с обычной кросс-энтропией для корректной работы EarlyStopping.