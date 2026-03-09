# Задача 308: Переделываем расчет VIP и Past Returns с нормализацией Robusta.

Я изучил глубокий анализ от Grok и полностью принимаю его архитектурные замечания, особенно в части **глобального вычисления признаков** в `__init__`. Это в разы эффективнее, чем считать суммы по уровням для каждого сэмпла во время обучения.

Вот детальный план для задачи **308: Реализация "On-the-fly" признаков VIB и PastRet из Orderbook**.

---

### ЗАДАЧА 308 — ФИНАЛЬНЫЙ ПЛАН (Claude + Grok) ✅ ЗАВЕРШЕНО

#### 1. Вспомогательные функции (Вне класса) ✅ ЗАВЕРШЕНО
**Файл**: [./python_lab/src/dataset.py](./python_lab/src/dataset.py)
- **Функция `compute_depth_imbalance_globally(bid_v, ask_v)`**:
    - Векторизованный расчет по всей матрице (N, 50). 
    - Формула: `(sum(bid_v) - sum(ask_v)) / (sum(bid_v) + sum(ask_v) + 1e-8)`.
- **Функция `compute_past_returns_globally(mid_prices, lags=[10, 50, 100])`**:
    - Использование `np.log(np.maximum(mid_prices, 1e-10))`.
    - Сдвиг через `returns[lag:] = log_p[lag:] - log_p[:-lag]`.

#### 2. Кэширование и поддержка режимов (Memmap/Memory) ✅ ЗАВЕРШЕНО
**Файл**: [./python_lab/src/dataset.py](./python_lab/src/dataset.py)
- В `__init__`, если признаки отсутствуют:
    - **Режим `memory`**: Сохранять в `self.vib_cache` (N,) и `self.past_ret_cache` (N, 3).
    - **Режим `memmap`**: Проверять наличие файлов `vib_cache.npy` и `past_ret_cache.npy` рядом с исходными данными. Если их нет — вычислить 1 раз и сохранить через `np.lib.format.open_memmap`.
    - **Режим `streaming`**: Вычислять признаки внутри `_load_next_batch` для каждого нового загруженного куска данных.

#### 3. Консистентная нормализация (Robust/Winsor) ✅ ЗАВЕРШЕНО
**Файл**: [./python_lab/src/dataset.py](./python_lab/src/dataset.py)
- Не использовать простой Z-score. Вместо этого:
    - Создать временный `pl.DataFrame` из вычисленных кэшей.
    - Инициализировать наш `Normalizer` с тем же `scaler_type` и `winsor_limits`, что и для основных данных.
    - Применить `fit_transform()` к кэшам, чтобы их масштаб (Scale) был идентичен остальным 4-м каналам.

#### 4. Обновление сигнатур и __getitem__ ✅ ЗАВЕРШЕНО
**Файл**: [./python_lab/src/dataset.py](./python_lab/src/dataset.py)
- **Изменить `__getitem__(self, idx)`**:
    - Извлекать `vib_val = self.vib_cache[idx]` и `ret_val = self.past_ret_cache[idx, -1]`.
    - Передавать их в `_process_sample`.
- **Изменить `_process_sample(self, x_raw, y, v, w, regime_id, vib_val=None, past_ret_val=None)`**:
    - Если `vib_val` передан — использовать его для заполнения `ch[4]`.
    - Если `past_ret_val` передан — использовать его для `ch[5]`.

#### 5. Integrity Log 2.0 ✅ ЗАВЕРШЕНО
**Файл**: [./python_lab/src/dataset.py](./python_lab/src/dataset.py)
- Обновить блок логов, добавленный в задаче 307.
- Теперь он должен выводить: `[LOBDataset] On-the-fly features generated and normalized via RobustScaler`.
- Проверить, что `mean` для каналов 4 и 5 в логе больше не равен `0.0000`.

---

**Итог**: Это решение делает датасет полностью автономным. Мы больше не зависим от того, какие колонки "забыл" положить в Parquet сборщик данных или Rust-дампер. 

**Grok, теперь план идеален. Все нюансы учтены.**




================= ОБЩАЯ ИНФОРМАЦИЯ ===========

# ПРОБЛЕМА
Переделываем расчет VIP и Past Returns с нормализацией Robusta. Раньше эти два индикатора ошибочно загружались из сырых данных parquet. Код ожидал, что они там есть и не считал их на лету в Python. Теперь мы разобрались, что инчего в parquet кроме сырых ask, bid и времени нет. Там нет и не будет никаких индикаторов ,все считается в pythonyf ktne/

Логи обучения, Канал 4,5 одни нули при загрузке данных в обучение
Channel 0 (Price): min=-4.2309, max=3.6262, mean=-0.0728
Channel 1 (Vol): min=-1.4621, max=1.6363, mean=-0.0866
Channel 2 (Imb): min=-1.7050, max=2.1016, mean=0.0760
Channel 3 (OFI): min=-10.4854, max=6.7569, mean=-0.4892
Channel 4 (VIB): min=0.0000, max=0.0000, mean=0.0000
Channel 5 (PastRet): min=0.0000, max=0.0000, mean=0.0000
[LOBDataset] Loaded 691980 samples. Data mode: memory

---------
# ДИАЛОГ АНАЛИЗА ПРОБЛЕМЫ:
Проблема в том, что текущая реализация LOBDataset в dataset.py ожидает, что VIB и PastRet уже лежат в Parquet файле с префиксами feat_, а если их там нет — она просто заполняет каналы нулями.

Поскольку в твоих сырых данных есть только ask, bid и время, эти два канала (4 и 5) остаются пустыми (0.0000), и модель на 1/3 "слепа".

В чем конкретно ошибка в коде:
VIB (Channel 4): Код ищет колонку feat_trade_side или feat_vib. Если ты не загрузил файл сделок (trades_*.parquet), расчет VIB (Volume Imbalance) не запускается.
PastRet (Channel 5): Код ищет заранее рассчитанные лаги (например, feat_past_ret_100). Если их нет в Parquet, он ставит torch.zeros.
Как это исправить (План "Оживления" каналов):
Нам нужно заставить LOBDataset рассчитывать эти признаки внутри себя, если их нет в исходном файле.

1. Исправление VIB (Channel 4) — "Агрессия сделок"
Если нет файла сделок, мы можем эмулировать VIB через изменение объема в стакане (Internal VIB).

Файл: dataset.py
Что изменить: В методе _process_sample, если vib_idx < 0, вместо нулей брать разность total_volume между текущим и предыдущим шагом.
2. Исправление PastRet (Channel 5) — "Доходность"
У нас уже есть функция compute_past_returns в начале файла dataset.py, но она не вызывается автоматически для формирования тензора.

Файл: dataset.py
Что изменить: В __init__ добавить вызов compute_past_returns для колонки mid_price и сохранить результат в self.past_returns_cache. Затем в _process_sample брать данные оттуда.
Детальный код для исправления dataset.py:
Шаг А: В __init__ (после загрузки mid_price):
# Найти место, где определяется self.mid_price (примерно 700-750 строки)
# Добавить расчет Past Returns на лету:
if not self.past_ret_indices:
    print(f"[{self.__class__.__name__}] Warning: PastRet columns not found. Computing on the fly...")
    lags = [10, 50, 100]
    self.past_returns_cache = compute_past_returns(self.mid_price, lags)
    self.has_on_the_fly_ret = True
else:
    self.has_on_the_fly_ret = False
Шаг Б: В _process_sample (исправление 5-го канала):
# Найти блок ch[5] (примерно 1090-1100 строки)
<<<<
        if self.past_ret_indices:
            pr_idx = self.past_ret_indices[-1]
            pr = torch.from_numpy(x_raw[:, pr_idx].copy()).float()
        else:
            pr = torch.zeros(ask_p.shape[0], device=ask_p.device)
====
        if self.has_on_the_fly_ret:
            # Берем из кэша, который мы посчитали в __init__
            # idx - это индекс текущего сэмпла в большом массиве
            pr_val = self.past_returns_cache[idx, -1] # берем лаг 100
            pr = torch.full((ask_p.shape[0],), pr_val, dtype=torch.float32)
        elif self.past_ret_indices:
            pr_idx = self.past_ret_indices[-1]
            pr = torch.from_numpy(x_raw[:, pr_idx].copy()).float()
        else:
            pr = torch.zeros(ask_p.shape[0], device=ask_p.device)
>>>>
Grok, подтверди: если мы не будем считать PastRet на лету, модель потеряет важнейший сигнал о тренде. Нам нужно внедрить этот расчет в dataset.py немедленно.

Это вернет жизнь в каналы 4 и 5 и поднимет MCC выше нуля.

MarinaDash, 4m ago
коротко овтеть, получается VIИ рачситывается только на trades, не на Orderbook? Больше не вспоминай про trades вообще, мы работаем только с Orderbook. Изучи тогда в нашем списке задачи 000-tasks_list.md зачем мы его вобще добавляли и на основание чего он работает

**СПОР-ПЛАН**

Reviewing the Documentation

а, ты прав. В текущем коде dataset.py (строки 1076-1085) VIB (Volume Imbalance) реализован как накопительная агрессия сделок (cumsum по объемам сделок). Именно поэтому в твоих логах там 0.0000 — раз файлов сделок нет, считать нечего.

Согласно docs/000-tasks_list.md и логике работы только со стаканом (Orderbook only), вот краткий анализ:

Зачем добавляли (Задача 236): Изначально планировалось отслеживать "микро-сделки", чтобы видеть, куда бьют рыночные ордера. Но раз мы перешли на чистый стакан, этот канал должен давать информацию о давлении глубины.
На чем должен работать: Раз мы исключаем сделки, VIB должен базироваться на задаче 053 (Bid-Ask Imbalance), но расширенной на всю глубину. Это должен быть Cumulative Depth Imbalance.
Вместо сделок он должен считать разницу суммарных объемов Bid и Ask по всем 50 уровням.
Это покажет модели не просто "кто сильнее на первой линии", а у кого больше "стен" (ликвидности) во всем стакане.
Проблема PastRet (Channel 5):
Он тоже 0.0000, потому что код ищет готовую колонку feat_past_ret_... в Parquet. Раз её нет, он просто ставит нули.

Твой вердикт для реализации:
Чтобы каналы ожили без сделок:

VIB (ch 4): Должен считаться как (sum(bid_v) - sum(ask_v)) / (sum(bid_v) + sum(ask_v)) по всем 50 уровням стакана.
PastRet (ch 5): Должен считаться в Python на лету как log(mid_price_t) - log(mid_price_{t-100}).
