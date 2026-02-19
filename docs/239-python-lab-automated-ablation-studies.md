Задача 239: Автоматизированные исследования абляции (Automated Ablation Studies)
Реализация системы автоматического тестирования влияния отдельных компонентов (признаков, слоев, голов внимания) на итоговое качество модели. Система должна выявлять «мертвый вес» в данных и избыточность в архитектуре для оптимизации инференса.

1. Цель задачи
Создать инструмент для систематического удаления групп признаков или модулей нейросети с последующим переобучением и сравнением метрик (MCC, Latency) относительно базовой модели (Baseline).

2. Инструкции по реализации для Gemini
А. Конфигура

Analyzing Task Revisions


ция экспериментов (ablation_config.yaml) Создать YAML-файл со структурой групп для тестирования:

feature_groups:
  lob_depth: [bid_price_1, ask_price_1, ...]
  trade_imb: [trade_imb_1s, trade_imb_5s]
  technical: [rsi, macd, vwap_dist]
arch_variants:
  heads: [2, 4, 8]
  layers: [1, 2]
Б. Скрипт абляции (./python_lab/scripts/ablation_study.py)
Baseline & SHAP:
Запустить обучение базовой модели.
Выполнить быстрый расчет важности через SHAP GradientExplainer для предварительной оценки.
Цикл переобучения:
Для каждой группы из ablation_config.yaml:
Сформировать временный конфиг (через OmegaConf или Hydra).
Запустить сокращенное обучение (например, 10 эпох с EarlyStopping).
Использовать Optuna Pruning (задача 030) для быстрой остановки бесперспективных веток.
Сбор метрик:
def run_ablation():
    base_results = run_train(base_config)
    results = []
    for group_name, features in config.feature_groups.items():
        # Override: exclude features
        metrics = run_train(override_config(exclude=features), epochs=10)
        delta_mcc = metrics['mcc'] - base_results['mcc']
        results.append({"group": group_name, "delta_mcc": delta_mcc, "latency": metrics['latency']})
    
    pd.DataFrame(results).to_markdown("reports/ablation_report.md")
В. Модификация датасета (./python_lab/src/dataset.py)
Добавить в конструктор Dataset параметр exclude_features: list[str] = None.
Реализовать фильтрацию колонок df.drop(exclude_features) перед конвертацией в тензоры.
3. Аргументация и уточнения (по Grok)
SHAP vs Ablation: SHAP дает быструю оценку на уже обученной модели, но только абляция (переобучение без фичи) показывает истинную ценность признака, так как модель может адаптироваться к отсутствию данных, используя коррелирующие переменные.
Эффективность: Использование EarlyStopping на плато валидации и сокращение количества эпох до 10-15 достаточно для определения Δ\DeltaΔ (дельты) качества.
Hydra/OmegaConf: Рекомендуется использовать эти библиотеки для удобного переопределения параметров конфига из командной строки без создания временных файлов.
4. Ожидаемый результат
Отчет ablation_report.md со списком групп признаков, ранжированных по их вкладу в MCC.
Список кандидатов на удаление (фичи с ΔMCC≈0\Delta MCC \approx 0ΔMCC≈0).
Подтверждение оптимальности количества голов/слоев Attention для целевой задержки инференса.
5. Необходимые зависимости
Python: hydra-core, shap, pandas, optuna.