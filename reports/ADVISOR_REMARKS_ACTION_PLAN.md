# ADVISOR_REMARKS_ACTION_PLAN.md

Дата: 2026-06-25  
Источник: `Федоров_Станислав_замечания_2026_06_23.pdf`  
Цель: превратить замечания в контролируемые задачи.

---

## Краткий вывод

Замечания не про косметику. Основная проблема — эксперимент пока не является воспроизводимым и методологически защищённым. Поэтому порядок исправления такой:

1. Сначала формальная постановка: target, горизонт, временная шкала, доступность признаков.
2. Потом данные: тикер, период, число наблюдений, правила агрегации и удаления пропусков.
3. Потом протокол обучения: split, validation, scaler, seed, hyperparameters.
4. Потом baseline и интерпретация.
5. Только после этого — LSTM/FinBERT/BERT-теория, библиография и редактура.

---

## Таблица: замечание → действие → задача

| № | Суть замечания | Что нужно сделать | Task ID | Приоритет | Артефакт |
|---:|---|---|---|---|---|
| 1 | Не определена метка `y_t`: Rise/Fall, горизонт, цена, связь даты текстов и даты прогноза | Зафиксировать target, горизонт, цену и момент принятия решения | METH-001–005, INBOX-010 | P0 | `docs/METHODOLOGY_MVP.md` |
| 2 | Неясно, за какой день берутся `open/high/low/close/adj-close/inc-*`; возможна утечка | Нарисовать временную шкалу и проверить все признаки на доступность до прогноза | METH-004–005, FEAT-005, INBOX-009 | P0 | no-leakage checklist |
| 3 | Ошибка в формулах LSTM и не определены размерности | Исправить обозначения `W(o)`, `U(o)`, определить все переменные и размерности | TEXT-004 | P1 | раздел про LSTM |
| 4 | В FinBERT/BERT не определены softmax, LayerNorm, Concat, erf, ось softmax; нет первоисточников | Добавить определения и ссылки на Transformer, BERT, FinBERT, LSTM | TEXT-004, TEXT-007, INBOX-011 | P1/P0 | раздел архитектур + библиография |
| 5 | Не описан `flare-sm-bigdata`: URL, период, число дней/сообщений/примеров, пропуски, тикер, версия кода | Сформировать dataset summary и ссылку на конкретный commit | DATA-002–007, INBOX-010 | P0 | `reports/tables/dataset_summary.csv`, commit hash |
| 6 | Неполный train/validation/test protocol; неясны scaler, seed, hyperparameters | Описать split, validation, scaler fit only train, seed, число запусков | MODEL-006, INBOX-006–008, INBOX-010 | P0 | methodology + code |
| 7 | Accuracy 0.57 почти равна majority baseline 179/316 ≈ 0.566; нужны baseline и confusion matrix | Добавить majority, price-only, FinBERT-only, combined; вывести TP/TN/FP/FN | MODEL-001–004, EVAL-001–005 | P0 | `reports/tables/metrics.csv` |
| 8 | Train loss без validation/test loss не доказывает качество, может быть переобучение | Добавить validation loss, early stopping, выбор лучшей эпохи | EVAL-006, INBOX-007 | P0 | loss table/plot |
| 9 | Библиография требует переработки, дубль [9]/[13], нет DOI/URL, первоисточников и датасета | Составить список литературы на проверку и исправить `.bib`/раздел | TEXT-007, INBOX-011 | P0 | `reports/BIBLIOGRAPHY_FIX_LIST.md` |
| 10 | Редакционные дефекты по тексту | Сделать финальную редактуру после методологии и результатов | TEXT-001–007 | P1 | clean draft |

---

## Что исправлять первым

### P0. Первый пакет исправлений

1. `AAPL/CSCO mismatch` в notebook.
2. Target: `y_t = 1{Close_{t+1} > Close_t}`.
3. Временная шкала признаков.
4. Time-based train/validation/test split.
5. Scaler fit only on train.
6. Majority baseline и price-only baseline.
7. Confusion matrix и метрики.

### P1. Второй пакет исправлений

1. LSTM formulas.
2. BERT/FinBERT definitions.
3. Библиография.
4. Редактура текста.

---

## Минимальный документ методологии

Следующий артефакт: `docs/METHODOLOGY_MVP.md`.

Он должен содержать:

```text
1. Постановка задачи
2. Target definition
3. Data availability timeline
4. Feature groups
5. Train/validation/test split
6. Normalization protocol
7. Baselines
8. Metrics
9. Reproducibility settings
```

---

## Решение

`INBOX-001` закрыт. Следующая задача по критическому пути — `INBOX-010`: подготовить `docs/METHODOLOGY_MVP.md`.
