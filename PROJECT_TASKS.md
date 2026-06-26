# PROJECT_TASKS.md

Проект: **LLM-generated features для прогнозирования финансовых временных рядов**  
Режим: **минимальные сроки, контроль по маленьким задачам**  
Дата старта трекера: **2026-06-24**  
Последнее обновление: **2026-06-25**

---

## 1. Минимальная цель проекта

Сделать воспроизводимый MVP-эксперимент, который проверяет, дают ли LLM/FinBERT-признаки дополнительную предсказательную информацию относительно классических рыночных признаков.

Основной вопрос:

```text
Улучшают ли LLM-generated features качество прогноза next-day direction по сравнению с baseline-моделями?
```

---

## 2. Текущий статус

Основной MVP-run зафиксирован:

```text
RUN 03 = primary experiment
```

Главный артефакт результатов:

```text
reports/RESULTS_MVP.md
```

Основной вывод:

```text
В рамках текущего MVP FinBERT-признаки не показали устойчивого преимущества относительно baseline-моделей.
```

---

## 3. Primary experiment

| Поле | Значение |
|---|---|
| Primary run | RUN 03 |
| Notebook | `notebooks/02_mvp_aapl_date_split.ipynb` с заменённой Cell 7 |
| Split | `chronological_row_balanced_by_decision_date` |
| Asset | AAPL |
| Target | `y_t = 1{Close_{t+1} > Close_t}` |
| Frequency | daily |
| Horizon | 1 trading day |
| Train | 1104 rows, 2020-04-01 -- 2020-06-15 |
| Validation | 252 rows, 2020-06-16 -- 2020-06-29 |
| Test | 222 rows, 2020-06-30 -- 2020-09-30 |

Primary artifacts:

```text
reports/PRIMARY_EXPERIMENT.md
reports/RUN_03_RESULTS_REVIEW.md
reports/RESULTS_MVP.md
reports/tables/dataset_summary.csv
reports/tables/metrics.csv
reports/tables/confusion_matrix.csv
reports/tables/predictions.csv
reports/tables/lstm_loss_history.csv
reports/tables/model_index.csv
```

---

## 4. P0 status

| ID | Task | Status | Artifact |
|---|---|---|---|
| INBOX-001 | Разобрать замечания научного руководителя | DONE | `reports/ADVISOR_REMARKS_ACTION_PLAN.md` |
| INBOX-002 | Разобрать `4aapl.ipynb` | DONE | `reports/NOTEBOOK_REVIEW_4AAPL.md` |
| INBOX-004 | Исправить mismatch AAPL/CSCO | DONE | `notebooks/01_mvp_aapl_check.ipynb` |
| INBOX-005 | Переписать target на next-day direction | DONE | `docs/METHODOLOGY_MVP.md` |
| INBOX-006 | Гарантировать временной split | DONE | RUN 03 |
| INBOX-007 | Добавить validation и baseline | DONE | `reports/tables/metrics.csv` |
| INBOX-008 | Добавить normalization protocol | DONE | `docs/METHODOLOGY_MVP.md` |
| INBOX-009 | Проверить `raw_text` и 11 numeric features | DONE | RUN 02/RUN 03 parser fix |
| INBOX-010 | Подготовить MVP methodology | DONE | `docs/METHODOLOGY_MVP.md` |
| RUN-001 | Зафиксировать RUN 03 как основной эксперимент | DONE | `reports/PRIMARY_EXPERIMENT.md` |
| RUN-002 | Написать summary результатов MVP | DONE | `reports/RESULTS_MVP.md` |

---

## 5. Main metrics: RUN 03

| Model | Feature group | Accuracy | Balanced accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Majority | none | 0.532 | 0.500 | 0.694 | 0.500 |
| Logistic Regression | prices only | 0.410 | 0.400 | 0.502 | 0.379 |
| Logistic Regression | FinBERT only | 0.550 | 0.529 | 0.667 | 0.512 |
| Logistic Regression | prices + FinBERT | 0.486 | 0.474 | 0.584 | 0.427 |
| LSTM | prices + FinBERT | 0.432 | 0.445 | 0.315 | 0.449 |

---

## 6. Next tasks

Теперь не нужно продолжать тюнинг MVP-модели. Следующий блок — текст работы.

| ID | Task | Priority | Status | Owner | Output |
|---|---|---|---|---|---|
| TEXT-001 | Написать постановку задачи | P0 | NEXT | Максим | раздел курсовой |
| TEXT-002 | Написать описание данных | P0 | TODO | Максим + Стас | раздел курсовой |
| TEXT-003 | Написать описание признаков | P0 | TODO | Максим | раздел курсовой |
| TEXT-004 | Написать описание моделей | P0 | TODO | Стас | раздел курсовой |
| TEXT-005 | Написать протокол эксперимента | P0 | TODO | Максим + Стас | раздел курсовой |
| TEXT-006 | Написать результаты | P0 | TODO | Максим + Стас | раздел курсовой |
| TEXT-007 | Исправить bibliography | P0 | TODO | Максим + Стас | bibliography |
| REPO-001 | Синхронизировать локальный git без тяжёлых файлов | P0 | TODO | Максим | clean local repo |

---

## 7. Решения

| Дата | Решение | Причина |
|---|---|---|
| 2026-06-24 | MVP делаем на одном активе | нужно быстро получить воспроизводимый результат |
| 2026-06-24 | LLM используется как feature generator | это соответствует теме проекта |
| 2026-06-25 | RUN 03 фиксируется как основной MVP-run | split date-safe, validation/test достаточно крупные |
| 2026-06-25 | Переходим от тюнинга модели к тексту | результат уже интерпретируемый, дальнейший тюнинг не нужен для MVP |

---

## 8. Формулировка вывода

Рабочий вывод для курсовой:

```text
В рамках MVP-эксперимента на AAPL признаки, извлечённые из финансовых текстов с помощью FinBERT, не продемонстрировали устойчивой добавочной предсказательной силы. FinBERT-only logistic regression немного превысила majority baseline по accuracy, однако ROC-AUC остался близким к 0.5, а объединение рыночных и текстовых признаков не улучшило качество прогноза.
```
