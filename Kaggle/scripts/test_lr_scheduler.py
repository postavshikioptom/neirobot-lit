"""
Тестовый скрипт для проверки работы LR Scheduler.
Визуализирует изменение learning rate и momentum для разных стратегий.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def test_scheduler(scheduler_type, max_epochs=20, steps_per_epoch=100):
    """
    Тестирует scheduler и возвращает историю LR и momentum.
    
    Args:
        scheduler_type: Тип scheduler ("onecycle", "cosine", "plateau", "step", "none")
        max_epochs: Количество эпох
        steps_per_epoch: Количество шагов в эпохе
    """
    # Создаем простую модель
    model = nn.Linear(10, 3)
    lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    total_steps = max_epochs * steps_per_epoch
    
    # Создаем scheduler
    if scheduler_type == "none":
        scheduler = None
        interval = "epoch"
    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )
        interval = "step"
    elif scheduler_type == "cosine":
        warmup_steps = int(0.1 * total_steps)
        cosine_steps = total_steps - warmup_steps
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0/25.0,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=lr / 10000.0
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        interval = "step"
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.5
        )
        interval = "epoch"
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )
        interval = "epoch"
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Симулируем обучение
    lr_history = []
    momentum_history = []
    step_count = 0
    
    for epoch in range(max_epochs):
        for step in range(steps_per_epoch):
            # Записываем текущий LR и momentum
            current_lr = optimizer.param_groups[0]['lr']
            current_momentum = optimizer.param_groups[0]['betas'][0]
            
            lr_history.append(current_lr)
            momentum_history.append(current_momentum)
            
            # Имитируем backward и optimizer.step()
            optimizer.zero_grad()
            loss = torch.randn(1)  # Фейковый loss
            
            # Шаг scheduler (если interval="step")
            if scheduler is not None and interval == "step":
                scheduler.step()
            
            step_count += 1
        
        # Шаг scheduler (если interval="epoch")
        if scheduler is not None and interval == "epoch":
            if scheduler_type == "plateau":
                # Для plateau нужна метрика (симулируем улучшение/ухудшение)
                fake_metric = 0.5 + 0.01 * epoch + np.random.randn() * 0.05
                scheduler.step(fake_metric)
            else:
                scheduler.step()
    
    return lr_history, momentum_history

def plot_schedulers():
    """Визуализирует все типы schedulers."""
    scheduler_types = ["onecycle", "cosine", "plateau", "step", "none"]
    
    fig, axes = plt.subplots(2, len(scheduler_types), figsize=(20, 8))
    fig.suptitle("LR Scheduler Comparison (20 epochs, 100 steps/epoch)", fontsize=16)
    
    for idx, sched_type in enumerate(scheduler_types):
        print(f"Testing {sched_type} scheduler...")
        lr_hist, momentum_hist = test_scheduler(sched_type, max_epochs=20, steps_per_epoch=100)
        
        # График LR
        axes[0, idx].plot(lr_hist, linewidth=2)
        axes[0, idx].set_title(f"{sched_type.upper()}")
        axes[0, idx].set_xlabel("Step")
        axes[0, idx].set_ylabel("Learning Rate")
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_yscale('log')
        
        # График Momentum
        axes[1, idx].plot(momentum_hist, linewidth=2, color='orange')
        axes[1, idx].set_xlabel("Step")
        axes[1, idx].set_ylabel("Momentum (beta1)")
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_ylim([0.8, 1.0])
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = Path(__file__).parent.parent / "lr_scheduler_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранен: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование LR Schedulers")
    print("=" * 60)
    
    plot_schedulers()
    
    print("\n✅ Все schedulers протестированы успешно!")
    print("\nОсновные выводы:")
    print("1. OneCycle: Быстрый рост LR, затем плавное снижение + циклический momentum")
    print("2. Cosine: Warmup 10%, затем косинусное затухание")
    print("3. Plateau: Адаптивное снижение при стагнации метрики")
    print("4. Step: Ступенчатое снижение каждые 5 эпох")
    print("5. None: Константный LR")
