# 030 - Python Lab Optuna Integration
Цель задачи: Реализовать автоматизированный поиск гиперпараметров (HPO) с использованием Optuna. Скрипт должен подбирать архитектуру модели и параметры обучения, используя Pruning (отсечение плохих попыток) для экономии времени. Главная метрика для оптимизации — val_mcc.

Файлы: python_lab/src/tune.py (создать)

Инструкции для Gemini:

Загрузка данных: Загрузить данные один раз перед запуском study, чтобы не тратить время на I/O в каждом trial.
Функция objective(trial):
Предложить параметры: d_model (32, 64, 128), nhead (4, 8 — с проверкой d_model % nhead == 0), num_layers (1, 2, 3), lr (1e-5..1e-3 log), dropout (0.1, 0.2).
Создать LiTModule, LOBDataset и DataLoader.
Настроить pl.Trainer: max_epochs=20, min_epochs=3, accelerator="auto", и PyTorchLightningPruningCallback.
Вернуть val_mcc.
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from .train import LiTModule
from .dataset import LOBDataset
# ... импорты данных ...

def objective(trial):
    # Параметры
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned() # Валидация архитектуры
        
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Модель и обучение
    model = LiTModule(d_model=d_model, nhead=nhead, num_layers=num_layers, lr=lr)
    trainer = pl.Trainer(
        max_epochs=20,
        min_epochs=3, # Даем модели шанс перед прунингом
        accelerator="auto",
        enable_checkpointing=False, # Не забиваем диск весами триалов
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_mcc")]
    )
    
    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_mcc"].item()

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize", 
        study_name="lit_hpo",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        sampler=sampler
    )
    study.optimize(objective, n_trials=50, timeout=3600*4) # 4 часа максимум
    
    print(f"Best MCC: {study.best_value}")
    print(f"Best params: {study.best_params}")
Технические требования:

TPESampler: Использовать явно с seed=42 для воспроизводимости.
Pruning: PyTorchLightningPruningCallback обязателен.
SQLite: Сохранять прогресс в optuna.db.
Ограничения: Добавить timeout (например, 4 часа) и n_trials.
Архитектура: Проверять совместимость d_model и nhead перед созданием модели.
Почему это важно: Оптимальные гиперпараметры критичны для трансформеров. Optuna позволяет не гадать, а научно найти конфигурацию, которая выжимает максимум из данных стакана. Прунинг позволяет отсекать «мусорные» варианты уже на 3-й эпохе, ускоряя поиск в разы.

Ожидаемый результат: Скрипт запускает тюнинг, сохраняет результаты в базу и выводит лучшие параметры в конце.