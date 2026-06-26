# PRIMARY_EXPERIMENT.md

Дата фиксации: 2026-06-25  
Основной эксперимент: **RUN 03**

---

## Primary run

В качестве основного MVP-эксперимента фиксируется RUN 03.

Причина выбора:

1. Target определён формально: `y_t = 1{Close_{t+1} > Close_t}`.
2. Тикер согласован: тексты и рыночные данные относятся к `AAPL`.
3. Split хронологический и не разрезает одну дату между train, validation и test.
4. Validation и test имеют достаточный размер для первичного сравнения моделей.
5. Scaler обучается только на train subset.
6. Есть baseline-сравнения и confusion matrix.

---

## Main artifacts

| Artifact | Description |
|---|---|
| `notebooks/02_mvp_aapl_date_split.ipynb` | notebook после правки Cell 7 на row-balanced date-safe split |
| `reports/RUN_03_RESULTS_REVIEW.md` | технический разбор RUN 03 |
| `reports/RESULTS_MVP.md` | текстовое резюме результатов для курсовой |
| `reports/tables/dataset_summary.csv` | summary датасета и split |
| `reports/tables/metrics.csv` | метрики моделей |
| `reports/tables/confusion_matrix.csv` | TP/TN/FP/FN |
| `reports/tables/predictions.csv` | предсказания по test subset |
| `reports/tables/lstm_loss_history.csv` | train/validation loss для LSTM |
| `reports/tables/model_index.csv` | row-level split audit |

---

## Main conclusion

RUN 03 не показывает устойчивого преимущества LLM/FinBERT-признаков относительно baseline.

Короткая формулировка:

```text
FinBERT-only logistic regression немного превосходит majority baseline по accuracy, однако ROC-AUC остаётся близким к 0.5, а combined model не улучшает качество. Поэтому в рамках MVP нельзя утверждать, что LLM-generated features дают устойчивую добавочную предсказательную силу.
```

---

## Next stage

Следующий этап — не дальнейший тюнинг модели, а перенос результатов в текст работы:

1. методология;
2. данные;
3. признаки;
4. экспериментальный протокол;
5. результаты;
6. ограничения;
7. bibliography.
