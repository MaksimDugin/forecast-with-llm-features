# COURSEWORK_ASSEMBLY_PIPELINE.md

Дата: 2026-06-26  
Назначение: показать, из каких файлов собирать текст курсовой и в каком порядке.

---

## 1. Главный LaTeX-файл

Основной файл текущей сборки:

```text
coursework_latex/main_current_draft.tex
```

Он подключает секции через `\input{...}`.

---

## 2. Порядок сборки текста

| Порядок | Раздел курсовой | LaTeX-файл | Статус |
|---:|---|---|---|
| 1 | Введение | `coursework_latex/sections/00_introduction.tex` | LaTeX готов |
| 2 | 1. Исследование литературы | `coursework_latex/sections/01_literature_review.tex` | LaTeX готов, библиографию ещё проверить |
| 3 | 2.1 Постановка задачи + 2.2 Данные | `coursework_latex/sections/02_experiment_setup.tex` | LaTeX готов |
| 4 | 2.3 Признаки | `coursework_latex/sections/02_3_features.tex` | LaTeX готов |
| 5 | 2.4 Методология сравнения групп признаков | `coursework_latex/sections/02_4_methodology.tex` | LaTeX готов |
| 6 | 2.5 Модели как инструмент проверки признаков | `coursework_latex/sections/02_5_models.tex` | LaTeX готов |
| 7 | 2.6 Результаты: влияние признаков на прогноз | `coursework_latex/sections/02_6_results.tex` | LaTeX готов |
| 8 | 2.7 Интерпретация и ограничения | пока нет отдельного LaTeX-файла | нужно написать |
| 9 | Заключение | пока нет отдельного LaTeX-файла | нужно написать |
| 10 | Список литературы | `coursework_latex/references_draft.bib` | черновик, нужно проверить |

---

## 3. Markdown-источники, из которых сделана LaTeX-версия

| LaTeX-файл | Источник / база |
|---|---|
| `00_introduction.tex` | `reports/INTRODUCTION_FEATURE_CENTRIC_DRAFT.md` |
| `01_literature_review.tex` | `reports/LITERATURE_REVIEW_DRAFT.md` |
| `02_experiment_setup.tex` | черновики постановки задачи, данных и RUN 03 summaries |
| `02_3_features.tex` | `reports/FEATURES_SECTION_DRAFT.md` |
| `02_4_methodology.tex` | `docs/METHODOLOGY_MVP.md` + feature-centric правка |
| `02_5_models.tex` | `reports/MODELS_FEATURE_CENTRIC_DRAFT.md` |
| `02_6_results.tex` | `reports/RESULTS_FEATURE_LEVEL_DRAFT.md` + `reports/RUN_03_RESULTS_REVIEW.md` |
| `references_draft.bib` | `reports/LITERATURE_REVIEW_DRAFT.md` + Deep Research + `statii.pdf` |

---

## 4. Как вставлять в шаблон курсовой

Если шаблон уже содержит `main.tex`, можно не использовать `main_current_draft.tex`, а просто перенести секции в нужные места шаблона.

Минимальный порядок вставки:

```latex
\input{sections/00_introduction}
\input{sections/01_literature_review}
\input{sections/02_experiment_setup}
\input{sections/02_3_features}
\input{sections/02_4_methodology}
\input{sections/02_5_models}
\input{sections/02_6_results}
```

Для библиографии:

```latex
\bibliographystyle{gost2008}
\bibliography{references_draft}
```

Если в шаблоне используется `biblatex`, тогда вместо этого нужно подключать `references_draft.bib` через `\addbibresource{references_draft.bib}` и печатать список через `\printbibliography`.

---

## 5. Что ещё не готово

### Нужно написать

```text
coursework_latex/sections/02_7_interpretation_limitations.tex
coursework_latex/sections/03_conclusion.tex
```

### Нужно проверить

```text
coursework_latex/references_draft.bib
```

В `references_draft.bib` часть свежих источников оформлена как черновик. Перед финальной сдачей нужно сверить авторов, год, журнал/конференцию, DOI или arXiv ID.

---

## 6. Текущая логика работы

Курсовая собирается не вокруг максимизации качества модели, а вокруг влияния признаков:

```text
рыночные признаки
→ FinBERT-generated features
→ объединённые признаки
→ сравнение качества
→ вывод о добавочной предсказательной силе
```

Главный результат RUN 03:

```text
FinBERT-only даёт слабый самостоятельный сигнал,
но устойчивая добавочная сила FinBERT-признаков поверх рыночных признаков не подтверждается.
```

---

## 7. Следующий шаг

Следующий файл для написания:

```text
coursework_latex/sections/02_7_interpretation_limitations.tex
```

После него:

```text
coursework_latex/sections/03_conclusion.tex
```
