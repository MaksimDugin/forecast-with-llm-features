# NOTEBOOK_REVIEW_4AAPL.md

Дата разбора: 2026-06-24  
Файл: `4aapl.ipynb`  
Режим: статический разбор, без перезапуска notebook.

---

## 1. Краткий вывод

`4aapl.ipynb` нельзя брать как готовый воспроизводимый эксперимент.

Его можно использовать как черновик для:
- загрузки `TheFinAI/flare-sm-bigdata`;
- парсинга табличного контекста и текстов;
- получения CLS-embedding из FinBERT;
- базовой LSTM-архитектуры.

Но перед любыми результатами нужно исправить P0-проблемы:

1. В коде фильтруется `ticker = "aapl"`, но цены скачиваются для `CSCO`.
2. Target сейчас считается как движение `Close_t > Close_{t-1}`, а не как заранее зафиксированный next-day target.
3. Не доказано, что признаки не содержат данные из дня прогноза или будущего.
4. Данные не сортируются по дате перед `shuffle=False` split.
5. Нет validation split.
6. Нет scaler / normalization protocol.
7. Нет baseline-моделей.
8. Нет seed для воспроизводимости.
9. Ячейка отладки вызывает функцию до её определения.
10. Результаты не сохраняются в таблицы/файлы.

---

## 2. Карта notebook

| Cell | Что делает | Можно использовать | Риск / проблема | Минимальная правка |
|---:|---|---|---|---|
| 0 | Импорты, загрузка `ProsusAI/finbert` | Частично | нет seed, интернет-зависимость, нет фиксации версий | добавить `set_seed`, requirements, device log |
| 1 | Загружает `TheFinAI/flare-sm-bigdata`, фильтрует `aapl`, извлекает `pred_date` из `query` | Частично | не описан источник, нет проверки формата query, нет сортировки по дате | добавить dataset summary, sort by `pred_date` |
| 2 | Загружает цены через `yfinance` | Нет в текущем виде | критическая ошибка: скачивается `CSCO`, а не `AAPL` | заменить на `ticker.upper()` / `AAPL` |
| 3 | Отладка дат и парсинга | Нет в текущем виде | вызывает `parse_text_and_table` до определения функции | перенести после cell 4 или удалить |
| 4 | Парсит raw text/table, строит X/y | Частично | target не соответствует MVP; не фильтрует даты признаков; нет названий 11 числовых признаков | явно задать feature window и target |
| 5 | Train/test split и DataLoader | Частично | нет validation; split по текущему порядку строк, не гарантированно по времени | сначала sort by date, потом train/val/test |
| 6 | LSTM-модель | Частично | `Sigmoid + BCELoss`, `squeeze()` может ломать batch size 1; `pos_weight` используется вручную | лучше `BCEWithLogitsLoss`, `squeeze(-1)` |
| 7 | Обучение | Частично | scheduler по train loss; нет validation loss; нет early stopping | добавить val loop, save best model |
| 8 | Test metrics | Частично | нет baseline; нет сохранения predictions; только threshold 0.5 | добавить majority/price-only baseline и `predictions.csv` |

---

## 3. Что сейчас делает pipeline

Фактическая логика:

```text
TheFinAI/flare-sm-bigdata
    ↓
filter query contains "aapl"
    ↓
extract pred_date from query by regex "at YYYY-MM-DD"
    ↓
download Yahoo prices for CSCO
    ↓
target = 1{Close_pred_date > Close_previous_trading_day}
    ↓
parse raw text:
    table rows -> 11 numeric features
    date:text rows -> FinBERT CLS embedding
    ↓
X shape: samples × 10 days × 779 features
    779 = 768 FinBERT embedding + 11 numeric features
    ↓
train/test split 80/20, shuffle=False
    ↓
bidirectional LSTM
    ↓
classification_report + confusion_matrix
```

Главная проблема: по названию notebook и фильтру датасета это AAPL, но target берётся по CSCO. В таком виде результат методологически невалиден.

---

## 4. Target

### Текущий target в notebook

В cell 2:

```python
stock_data['prev_close'] = stock_data['Close'].shift(1)
stock_data['direction'] = (stock_data['Close'] > stock_data['prev_close']).astype(int)
```

В cell 4:

```python
pred_date = row['pred_date']
price_row = stock_data[stock_data['Date'] == pred_date]
direction = price_row['direction'].values[0]
```

То есть фактически:

```text
y_d = 1{Close_d > Close_{d-1}}
```

где `d = pred_date`.

### Для MVP лучше зафиксировать

```text
y_t = 1{Close_{t+1} > Close_t}
```

где:
- `t` — дата принятия решения;
- признаки доступны только на дату `t` и раньше;
- `t+1` — следующий торговый день.

Тогда в коде нужно создать target так:

```python
prices["target"] = (prices["Close"].shift(-1) > prices["Close"]).astype(int)
```

И после этого удалить последнюю строку, где `target` неизвестен.

---

## 5. Leakage-риски

| Риск | Где | Почему опасно | Что сделать |
|---|---|---|---|
| AAPL/CSCO mismatch | cell 1 vs cell 2 | модель учится на текстах AAPL, а target от CSCO | заменить `CSCO` на `ticker.upper()` |
| Same-day target | cell 2 + 4 | если признаки содержат новости/цены дня `pred_date`, можно предсказывать уже известное движение | перейти на `Close_{t+1} > Close_t` |
| Нет фильтра дат внутри `parse_text_to_features` | cell 4 | берутся последние строки таблицы без проверки относительно `pred_date` | передавать `decision_date` и фильтровать `date <= decision_date` |
| Неизвестен состав 11 numeric features | cell 4 | часть признаков может быть рассчитана с будущими ценами | распарсить header и назвать признаки |
| Split не гарантированно временной | cell 5 | `shuffle=False` не спасает, если `df_ticker` не отсортирован | `df_model.sort_values("pred_date")` перед split |
| Нет scaler protocol | все ячейки | числовые признаки не нормализованы; нельзя описать fit на train | добавить scaler fit только на train |
| FinBERT считается на всём датасете до split | cell 4 | для frozen embeddings это не leakage по label, но плохо для контроля pipeline | лучше precompute без y или после split с cache |
| Нет validation | cell 5–7 | нельзя выбирать эпоху и scheduler по train loss | добавить train/val/test |
| Нет baseline | cell 8 | accuracy LSTM нельзя интерпретировать | majority + price-only logistic/RF |

---

## 6. Минимальный план исправления notebook

### Шаг 1. Починить запуск

- Убрать/перенести cell 3 после определения `parse_text_and_table`.
- Заменить `CSCO` на `ticker.upper()`.
- Добавить сортировку по `pred_date`.

### Шаг 2. Зафиксировать методологию

- Явно выбрать:
  - `decision_date = t`;
  - target: `Close_{t+1} > Close_t`;
  - feature window: `[t-9, ..., t]` или `[t-10, ..., t-1]`.
- Для минимальной защиты лучше выбрать `[t-10, ..., t-1]`, если есть сомнение во времени публикации новостей внутри дня.

### Шаг 3. Сделать baseline до LSTM

Минимально:

```text
majority baseline
price-only logistic regression
price-only random forest
FinBERT-only logistic regression
combined logistic regression
```

LSTM оставить как P1, если есть время.

### Шаг 4. Сделать воспроизводимость

- `random_state`;
- `np.random.seed`;
- `torch.manual_seed`;
- фиксированный split по датам;
- `requirements.txt`;
- сохранение `dataset_summary.csv`, `metrics.csv`, `predictions.csv`.

---

## 7. Рекомендация по архитектуре

Не чинить notebook как основной код.

Правильный минимальный вариант:

```text
notebooks/
  4aapl_original.ipynb
  01_mvp_aapl_check.ipynb

src/
  data/
    load_flare.py
    load_prices.py
    make_dataset.py
  features/
    finbert_embeddings.py
    price_features.py
  models/
    baselines.py
    lstm.py
  evaluation/
    metrics.py
```

Но в минимальные сроки можно сначала сделать один чистый notebook:

```text
notebooks/01_mvp_aapl_check.ipynb
```

А потом вынести стабильные функции в `src/`.

---

## 8. Решение по задачам

Закрыты статическим разбором:

- `CODE-001` — структура notebook понятна.
- `CODE-002` — target найден.
- `CODE-003` — price/numeric features найдены, но названия требуют уточнения.
- `CODE-004` — FinBERT features найдены.
- `CODE-005` — leakage-риски найдены.
- `CODE-006` — split найден, scaler отсутствует.

Следующая задача:

```text
CODE-007 — решить: чинить notebook или выносить в .py-модули
```

Рекомендация: для скорости — сделать чистый notebook `01_mvp_aapl_check.ipynb`, не переписывать сразу весь проект в модули.
