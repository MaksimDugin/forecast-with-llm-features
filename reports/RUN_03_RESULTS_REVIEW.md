# RUN_03_RESULTS_REVIEW.md

Дата: 2026-06-25  
Run: row-balanced date-safe split  
Входные артефакты: `dataset_summary.csv`, `metrics.csv`, `confusion_matrix.csv`, `predictions.csv`, `lstm_loss_history.csv`, `model_index.csv`.

---

## 1. Проверка split

RUN 03 достиг нужного технического состояния split:

| Поле | Значение |
|---|---:|
| split_type | chronological_row_balanced_by_decision_date |
| observations | 1578 |
| unique dates | 117 |
| train rows | 1104 |
| validation rows | 252 |
| test rows | 222 |
| train unique dates | 52 |
| validation unique dates | 10 |
| test unique dates | 55 |
| train period | 2020-04-01 — 2020-06-15 |
| validation period | 2020-06-16 — 2020-06-29 |
| test period | 2020-06-30 — 2020-09-30 |

Date leakage по split исправлен: train, validation и test не пересекаются по `decision_date`.

---

## 2. Проверка признаков

Имена числовых признаков распознаны корректно:

```text
open, high, low, close, adj-close, inc-5, inc-10, inc-15, inc-20, inc-25, inc-30
```

Это закрывает проблему `numeric_0 ... numeric_10`.

---

## 3. Метрики RUN 03

| Model | Feature group | Accuracy | Balanced accuracy | F1 | ROC-AUC | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | none | 0.532 | 0.500 | 0.694 | 0.500 | 118 | 0 | 104 | 0 |
| LogisticRegression | prices_only | 0.410 | 0.400 | 0.502 | 0.379 | 66 | 25 | 79 | 52 |
| LogisticRegression | finbert_only | 0.550 | 0.529 | 0.667 | 0.512 | 100 | 22 | 82 | 18 |
| LogisticRegression | prices_plus_finbert | 0.486 | 0.474 | 0.584 | 0.427 | 80 | 28 | 76 | 38 |
| LSTM | prices_plus_finbert | 0.432 | 0.445 | 0.315 | 0.449 | 29 | 67 | 37 | 89 |

---

## 4. Основной вывод

На текущем MVP-протоколе LLM/FinBERT-признаки не дают убедительного улучшения относительно baseline.

Самый сильный результат среди LLM-вариантов:

```text
LogisticRegression + FinBERT-only
accuracy = 0.550
balanced_accuracy = 0.529
F1 = 0.667
ROC-AUC = 0.512
```

Но это улучшение относительно majority baseline слабое:

```text
Majority accuracy = 0.532
FinBERT-only accuracy = 0.550
```

ROC-AUC близок к случайному уровню:

```text
FinBERT-only ROC-AUC = 0.512
```

Combined model (`prices + FinBERT`) работает хуже, чем FinBERT-only и хуже majority по accuracy.

LSTM также не даёт улучшения и показывает худший F1 среди моделей с LLM features.

---

## 5. LSTM training

Лучшее значение validation loss:

```text
epoch = 4
train_loss = 0.4219
val_loss = 0.4615
```

После этого validation loss растёт, а train loss снижается. Это указывает на переобучение.

---

## 6. Как формулировать в работе

Безопасная формулировка:

```text
В рамках MVP-эксперимента на AAPL с хронологическим date-safe split признаки FinBERT не дали устойчивого улучшения относительно baseline-моделей. FinBERT-only logistic regression немного превышает majority baseline по accuracy и balanced accuracy, однако ROC-AUC остаётся близким к 0.5, а combined price + FinBERT model не улучшает результат. Следовательно, на данном этапе признаки LLM не демонстрируют устойчивой добавочной предсказательной силы.
```

---

## 7. Можно ли переходить дальше

Да, можно переходить к следующему этапу: интерпретация результатов и написание разделов методологии/результатов.

Не нужно продолжать бесконечно улучшать модель в рамках MVP. Текущий результат отрицательный/слабый, но методологически пригодный для честного описания.

Следующие задачи:

1. Зафиксировать RUN 03 как основной MVP-run.
2. Написать `RESULTS_MVP.md`.
3. Обновить task tracker.
4. Перейти к тексту курсовой: постановка задачи, данные, методология, результаты.
