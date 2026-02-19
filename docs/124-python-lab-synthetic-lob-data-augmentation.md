
# Задача 124: Аугментация данных LOB (Data Augmentation) (v2.0)

## 1. Изменения в `python_lab/train.py` (CLI аргументы)
Добавь параметры управления аугментацией:
```python
parser.add_argument('--augment_prob', type=float, default=0.5, help='Probability of applying augmentation')
parser.add_argument('--use_symmetric_flip', action='store_true', help='Enable Bid/Ask flipping with label reversal')
parser.add_argument('--volume_jitter_range', type=float, default=0.1, help='Max relative volume change (e.g. 0.1 for +/- 10%)')
parser.add_argument('--aug_seed', type=int, default=42, help='Seed for reproducible augmentation')
```

## 2. Реализация в `python_lab/src/dataset.py`
Добавь константы для индексов колонок (исходя из структуры: `[ask_p_0, ask_v_0, bid_p_0, bid_v_0, ...]`) и функции трансформации:

```python
# Константы для стакана глубиной N уровней
# PRICE_COLS = [0, 2, 4, ...] | VOL_COLS = [1, 3, 5, ...]
PRICE_COLS = list(range(0, num_features, 2))
VOL_COLS = list(range(1, num_features, 2))

# Индексы для свопа (Ask <-> Bid)
# [0, 1] <-> [2, 3] | [4, 5] <-> [6, 7] ...
ASK_COLS = []
BID_COLS = []
for i in range(0, num_features, 4):
    ASK_COLS.extend([i, i+1])
    BID_COLS.extend([i+2, i+3])

def apply_symmetric_flip(features, label):
    """
    Зеркальное отражение стакана.
    1. Меняем блоки Ask и Bid местами.
    2. Инвертируем знак относительных цен.
    3. Инвертируем метку Up (0) <-> Down (1).
    """
    flipped = features.clone()
    # Своп колонок
    flipped[:, ASK_COLS] = features[:, BID_COLS]
    flipped[:, BID_COLS] = features[:, ASK_COLS]
    # Инверсия знака цен (относительно mid_price, который равен 0)
    flipped[:, PRICE_COLS] *= -1.0
    
    new_label = label
    if label == 0: new_label = 1 # Up -> Down
    elif label == 1: new_label = 0 # Down -> Up
    
    return flipped, new_label

def apply_volume_jitter(features, jitter_range, generator):
    """Случайное изменение объемов в заданном диапазоне."""
    multiplier = 1.0 + (torch.rand(1, generator=generator).item() * 2 - 1) * jitter_range
    features[:, VOL_COLS] *= multiplier
    return features
```

## 3. Интеграция в `LOBDataset`
Обнови `__getitem__`, добавив проверку консистентности и детерминированный RNG:

```python
def __init__(self, ..., seed=42):
    # ...
    self.generator = torch.Generator().manual_seed(seed)

def __getitem__(self, index):
    x, y, w = self.features[index], self.labels[index], self.sample_weights[index]
    
    if self.is_train and torch.rand(1, generator=self.generator).item() < self.augment_prob:
        x_aug, y_aug = x.clone(), y
        
        # 1. Применяем Flip
        if self.use_symmetric_flip and torch.rand(1, generator=self.generator).item() < 0.5:
            x_aug, y_aug = apply_symmetric_flip(x_aug, y_aug)
            
        # 2. Применяем Jitter
        if self.volume_jitter_range > 0:
            x_aug = apply_volume_jitter(x_aug, self.volume_jitter_range, self.generator)
            
        # 3. Проверка консистентности: Best Bid (отрицательный) < Best Ask (положительный)
        # В нормализованных данных (p-mid)/mid: Best Bid < 0 < Best Ask
        if x_aug[0, 2] < x_aug[0, 0]: # bid_p_0 < ask_p_0
            x, y = x_aug, y_aug
            
    return x, y, w
```

## 4. Особенности реализации
- **Reproducibility**: Использование `torch.Generator` гарантирует, что при одном и том же `aug_seed` аугментация будет воспроизводимой.
- **Normalization**: Аугментация в данном виде (flip + negate) корректна **после** нормализации $(P - Mid)/Mid$. Если используется Z-score, логика инверсии цен может потребовать доработки.
- **Efficiency**: Все операции выполняются над тензорами `torch` (in-place или клонирование), что минимизирует накладные расходы.

---

## Аргументация для Планировщика:
1.  **Price Mirroring**: Без умножения `PRICE_COLS` на -1.0, после свопа Ask и Bid, цены бы остались со старыми знаками (Ask был бы отрицательным), что полностью запутает модель.
2.  **Symmetry Induction**: Симметричный флип — самый мощный способ заставить модель понять, что «давление со стороны покупателей» (Bid side) эквивалентно «давлению со стороны продавцов» при смене направления.
3.  **Safety Check**: Проверка `bid < ask` на нулевом уровне — минимальная страховка от математических ошибок при трансформациях.

**Gemini, реализуй этот механизм, обеспечив поддержку произвольной глубины стакана через динамический расчет PRICE_COLS и VOL_COLS.**