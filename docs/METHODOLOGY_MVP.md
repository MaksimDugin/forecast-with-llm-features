# METHODOLOGY_MVP.md

Дата: 2026-06-25  
Статус: draft для MVP.

---

## 1. Цель

Проверить, улучшают ли признаки из финансовых текстов, полученные через LLM / FinBERT, прогноз next-day direction по сравнению с price-only baseline.

Минимальный актив: `AAPL`.  
Частота: daily.  
Горизонт: 1 следующий торговый день.

---

## 2. Target

Пусть `P_t` — цена закрытия в день `t`. Target:

```text
y_t = 1{P_{t+1} > P_t}
```

То есть:

- `y_t = 1`, если `Close_{t+1} > Close_t`;
- `y_t = 0`, если `Close_{t+1} <= Close_t`.

Последний день в выборке удаляется, потому что для него нет `Close_{t+1}`.

---

## 3. Временная шкала

Решение принимается после закрытия торгового дня `t`. Разрешены только признаки, доступные к этому моменту:

- price features up to `t`;
- text features dated up to `t`;
- LLM features extracted from texts dated up to `t`.

Если у текстов нет точного intraday timestamp, используется более осторожное правило:

```text
texts <= t-1, prices <= t, target = Close_{t+1} > Close_t
```

Выбранное правило должно быть записано в `dataset_summary`.

---

## 4. Feature groups

### Prices-only

Минимально:

- close;
- daily return;
- lagged returns;
- rolling volatility;
- volume / volume change.

Запрещены признаки, рассчитанные с использованием `Close_{t+1}` или более поздних цен.

### FinBERT sentiment

- positive score;
- neutral score;
- negative score;
- sentiment label.

Если за день несколько текстов, нужна агрегация: mean / max / count / share positive / share negative.

### FinBERT embeddings

Для каждого текста извлекается embedding. Дневной embedding агрегируется средним по текстам дня.

### Theoretical text features

Минимально допустимые:

- attention: число текстов за день;
- uncertainty: score или частотность слов неопределённости;
- negative attention: attention × negative sentiment.

---

## 5. No-leakage checklist

- [ ] target считается через `shift(-1)`;
- [ ] строки отсортированы по `decision_date`;
- [ ] признаки строки `t` не используют `Close_{t+1}`;
- [ ] rolling features считаются только по прошлому и текущему окну;
- [ ] scaler обучается только на train;
- [ ] validation/test не используются при fit scaler и feature selection;
- [ ] временной ряд не перемешивается перед split;
- [ ] даты train/validation/test не пересекаются.

---

## 6. Train / validation / test split

Split строго хронологический:

- train: первые 70% наблюдений;
- validation: следующие 15%;
- test: последние 15%.

Validation используется для выбора эпохи, early stopping и гиперпараметров. Test используется только для финальной оценки.

---

## 7. Normalization protocol

Scaler fit только на train:

```text
fit scaler on train
transform train
transform validation
transform test
```

Нельзя делать fit scaler на всей выборке.

---

## 8. Baselines

Обязательные baseline:

1. Majority baseline: всегда предсказывать самый частый класс train set.
2. Price-only baseline: logistic regression или random forest на price features.
3. LLM-only baseline: модель на FinBERT sentiment или embeddings.
4. Combined model: price features + LLM features.

Главное сравнение:

```text
combined model vs price-only baseline
```

---

## 9. Metrics

Обязательные метрики:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC, если есть probability score;
- confusion matrix;
- TP, TN, FP, FN.

Accuracy обязательно сравнивается с majority baseline.

---

## 10. Минимальные артефакты

- `reports/tables/dataset_summary.csv`;
- `reports/tables/metrics.csv`;
- `reports/tables/confusion_matrix.csv`;
- `reports/tables/predictions.csv`.

В `dataset_summary` должны быть: источник датасета, тикер, период, число текстов, число daily observations, правила пропусков, split sizes и commit hash.
