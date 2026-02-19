# Исправление критической ошибки в LOBPatching (Задача 026)

## Проблема
В конструкторе и методе forward класса `LOBPatching` в `python_lab/src/layers.py` были жестко прописаны размерности:
- Строка 22: `level_pos_emb` размер 100 (жестко)
- Строка 26: `time_pos_emb` размер 512 (жестко)
- Строка 46: `view(b * s, 1, 200)` - жестко 200 фич
- Строка 51: `view(b, s, 100, ...)` - жестко 100 патчей

Это вызывало ошибку размерности при использовании параметров по умолчанию:
- `in_channels=3, n_levels=50` → `3 * 50 = 150` фич
- Но код ожидал 200 фич → ошибка при `view(b * s, 1, 200)`

## Решение
Реализован динамический расчет размерностей на основе параметров конструктора:

### Изменения в `python_lab/src/layers.py`

#### 1. Конструктор (строки 8-35)
```python
def __init__(self, seq_len=100, n_levels=50, in_channels=3, d_model=64):
    # Сохраняем параметры
    self.in_channels = in_channels
    self.n_levels = n_levels
    self.seq_len = seq_len
    
    # Динамический расчет размерностей
    self.num_features = in_channels * n_levels  # 3 * 50 = 150
    self.num_patches = self.num_features // 2   # 150 // 2 = 75
    
    # Level Positional Embedding (динамический размер)
    self.level_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
    
    # Temporal Positional Embedding (строго seq_len, не 512)
    self.time_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
```

#### 2. Forward метод (строки 37-65)
```python
def forward(self, x):
    # Используем динамические размеры вместо жестких значений
    x_flat = x_flat_seq.view(b * s, 1, self.num_features)  # Вместо 200
    x_patched = x_patched.view(b, s, self.num_patches, self.d_model)  # Вместо 100
```

### Изменения в `python_lab/src/lit_model.py`

#### Forward метод (строка 283)
Обновлен комментарий для ясности:
```python
def forward(self, x, mask=None, regime_id=None):
    """
    x: (Batch, Seq, in_channels, n_levels) - входные данные из dataset
       По умолчанию: (Batch, Seq, 3, 50) - 3 канала × 50 уровней
    """
```

## Результаты

### Параметры по умолчанию (in_channels=3, n_levels=50)
- `num_features = 3 * 50 = 150`
- `num_patches = 150 // 2 = 75`
- `level_pos_emb` размер: `(1, 75, d_model)`
- `time_pos_emb` размер: `(1, seq_len, d_model)`

### Гибкость
Теперь слой работает с любыми значениями `in_channels` и `n_levels`:
- `in_channels=6, n_levels=100` → `num_features=600, num_patches=300`
- `in_channels=4, n_levels=50` → `num_features=200, num_patches=100`

## Соответствие плану 026
✓ Динамический расчет `num_features = in_channels * n_levels`
✓ Динамический расчет `num_patches = num_features // 2`
✓ Level Positional Embedding размер `(1, num_patches, d_model)`
✓ Temporal Positional Embedding размер `(1, seq_len, d_model)`
✓ Conv1d с kernel=2, stride=2
✓ LayerNorm после всех сложений с pos_emb
✓ CLS Token в LiTModel
✓ Агрегация уровней в один Snapshot Token

## Тестирование
Создан тестовый скрипт `python_lab/test_patching_fix.py` для проверки:
1. LOBPatching с параметрами по умолчанию
2. LOBPatching с пользовательскими параметрами
3. LiTModel с параметрами по умолчанию
4. LiTModel с multi-horizon

Все тесты проходят успешно без ошибок размерности.
