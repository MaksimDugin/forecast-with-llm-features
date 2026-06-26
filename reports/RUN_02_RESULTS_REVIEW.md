# RUN_02_RESULTS_REVIEW.md

Дата: 2026-06-25  
Notebook: `notebooks/02_mvp_aapl_date_split.ipynb`  
Входные артефакты: `dataset_summary.csv`, `metrics.csv`, `confusion_matrix.csv`, `predictions.csv`, `lstm_loss_history.csv`, `model_index.csv`.

---

## 1. Что исправлено в RUN 02

RUN 02 исправил две проблемы RUN 01:

1. Имена числовых признаков теперь распознаны корректно:
   - `open`
   - `high`
   - `low`
   - `close`
   - `adj-close`
   - `inc-5`
   - `inc-10`
   - `inc-15`
   - `inc-20`
   - `inc-25`
   - `inc-30`

2. Split теперь сделан по уникальным `decision_date`, поэтому одна дата не попадает одновременно в train, validation и test.

---

## 2. Dataset summary

| Поле | Значение |
|---|---:|
| ticker | AAPL |
| target | `y_t = 1{Close_{t+1} > Close_t}` |
| frequency | daily |
| horizon | 1 trading day |
| split type | chronological_by_unique_decision_date |
| text lag | 1 day |
| observations | 1578 |
| unique dates | 117 |
| train rows | 1542 |
| validation rows | 18 |
| test rows | 18 |
| train unique dates | 81 |
| validation unique dates | 18 |
| test unique dates | 18 |
| train period | 2020-04-01 — 2020-07-30 |
| validation period | 2020-07-31 — 2020-09-01 |
| test period | 2020-09-02 — 2020-09-30 |

---

## 3. Main issue

RUN 02 is methodologically safer than RUN 01, but it creates a new practical problem:

```text
test_size = 18
validation_size = 18
```

This is too small for a stable conclusion. Metrics on 18 test observations are highly unstable.

The reason is the distribution of rows by date: early dates contain many rows, but late dates mostly contain one row per date. Splitting by unique dates gives good no-leakage protection but weak statistical power.

---

## 4. Metrics

| Model | Feature group | Accuracy | Balanced accuracy | F1 | ROC-AUC | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | none | 0.611 | 0.500 | 0.759 | 0.500 | 11 | 0 | 7 | 0 |
| LogisticRegression | prices_only | 0.556 | 0.584 | 0.556 | 0.714 | 5 | 5 | 2 | 6 |
| LogisticRegression | finbert_only | 0.389 | 0.500 | 0.000 | 0.500 | 0 | 7 | 0 | 11 |
| LogisticRegression | prices_plus_finbert | 0.278 | 0.305 | 0.235 | 0.377 | 2 | 3 | 4 | 9 |
| LSTM | prices_plus_finbert | 0.389 | 0.396 | 0.421 | 0.416 | 4 | 3 | 4 | 7 |

---

## 5. Interpretation

Do not use RUN 02 metrics as final evidence.

Safe interpretation:

```text
RUN 02 confirms that the no-leakage date split and feature-name parsing work, but the resulting validation and test sets are too small for reliable model comparison.
```

Current metric pattern:

- Majority baseline has highest accuracy because the test set is small and class 1 dominates.
- Price-only logistic regression has the best ROC-AUC and balanced accuracy among non-majority models.
- LSTM does not outperform majority baseline or price-only baseline in RUN 02.
- Combined price + FinBERT logistic regression performs worse than price-only logistic regression.

This does not prove that LLM features are useless; it only means this split is not adequate for a final conclusion.

---

## 6. LSTM training

Best validation loss occurs at epoch 5:

```text
epoch = 5
train_loss = 0.3941
val_loss = 0.4131
```

After epoch 5, validation loss increases while train loss continues to decrease. This is overfitting behavior.

---

## 7. Next fix

Create RUN 03 with a date-safe but row-balanced split.

Recommended split logic:

```text
1. group rows by decision_date
2. sort dates chronologically
3. choose split boundaries by cumulative row count, not by number of unique dates
4. never split the same date across train/validation/test
```

For the current data, a good approximate boundary is:

```text
train:      2020-04-01 — 2020-06-15    about 1104 rows
validation: 2020-06-16 — 2020-06-26    about 229 rows
test:       2020-06-29 — 2020-09-30    about 245 rows
```

This keeps no date overlap while making validation/test large enough for a more stable first comparison.

---

## 8. Decision

RUN 02 is useful as an audit run, but not as final results. The next notebook should be:

```text
notebooks/03_mvp_aapl_row_balanced_date_split.ipynb
```
