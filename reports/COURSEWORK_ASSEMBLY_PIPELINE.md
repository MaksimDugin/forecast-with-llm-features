# COURSEWORK_ASSEMBLY_PIPELINE.md

Дата: 2026-06-26  
Назначение: показать, из каких файлов собирать текст курсовой и в каком порядке.

---

## 1. Главный LaTeX-файл

Для твоего текущего шаблона с `inputenc`, `fontenc` и `babel` нужно использовать:

```text
coursework_latex/main_msu_template.tex
```

Это pdfLaTeX-compatible версия с титульником и без `fontspec`.

Файл ниже оставлен как технический черновик под XeLaTeX/LuaLaTeX и сейчас не является основным:

```text
coursework_latex/main_current_draft.tex
```

Если компилируешь через pdfLaTeX, `main_current_draft.tex` даст ошибку `fontspec requires either XeTeX or LuaTeX`.

---

## 2. Почему была ошибка `fontspec`

Ошибка:

```text
Package fontspec Error: The fontspec package requires either XeTeX or LuaTeX.
```

возникает, потому что `main_current_draft.tex` использует:

```latex
\usepackage{fontspec}
\usepackage{polyglossia}
```

Эти пакеты нельзя компилировать через pdfLaTeX. У тебя шаблон на классической связке:

```latex
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[english,russian]{babel}
```

Поэтому нужно либо:

1. компилировать `main_current_draft.tex` через XeLaTeX;  
2. либо использовать `main_msu_template.tex` через pdfLaTeX.

Для твоего случая лучше второй вариант.

---

## 3. Порядок сборки текста

| Порядок | Раздел курсовой | LaTeX-файл | Статус |
|---:|---|---|---|
| 1 | Введение | `coursework_latex/sections/00_introduction.tex` | LaTeX готов, ссылки усилены |
| 2 | 1. Исследование литературы | `coursework_latex/sections/01_literature_review.tex` | LaTeX готов, нужно ещё усилить источниками |
| 3 | 2.1 Постановка задачи + 2.2 Данные | `coursework_latex/sections/02_experiment_setup.tex` | LaTeX готов |
| 4 | 2.3 Признаки | `coursework_latex/sections/02_3_features.tex` | LaTeX готов |
| 5 | 2.4 Методология сравнения групп признаков | `coursework_latex/sections/02_4_methodology.tex` | LaTeX готов |
| 6 | 2.5 Модели как инструмент проверки признаков | `coursework_latex/sections/02_5_models.tex` | LaTeX готов |
| 7 | 2.6 Результаты: влияние признаков на прогноз | `coursework_latex/sections/02_6_results.tex` | LaTeX готов |
| 8 | 2.7 Интерпретация и ограничения | пока нет отдельного LaTeX-файла | нужно написать |
| 9 | Заключение | пока нет отдельного LaTeX-файла | нужно написать |
| 10 | Список литературы | `coursework_latex/references_draft.bib` | расширен, но нужно проверить |

---

## 4. Markdown-источники, из которых сделана LaTeX-версия

| LaTeX-файл | Источник / база |
|---|---|
| `00_introduction.tex` | `reports/INTRODUCTION_FEATURE_CENTRIC_DRAFT.md` + `reports/SOURCE_EXPANSION_PLAN.md` |
| `01_literature_review.tex` | `reports/LITERATURE_REVIEW_DRAFT.md` |
| `02_experiment_setup.tex` | черновики постановки задачи, данных и RUN 03 summaries |
| `02_3_features.tex` | `reports/FEATURES_SECTION_DRAFT.md` |
| `02_4_methodology.tex` | `docs/METHODOLOGY_MVP.md` + feature-centric правка |
| `02_5_models.tex` | `reports/MODELS_FEATURE_CENTRIC_DRAFT.md` |
| `02_6_results.tex` | `reports/RESULTS_FEATURE_LEVEL_DRAFT.md` + `reports/RUN_03_RESULTS_REVIEW.md` |
| `references_draft.bib` | `reports/LITERATURE_REVIEW_DRAFT.md` + Deep Research + `statii.pdf` + `reports/SOURCE_EXPANSION_PLAN.md` |

---

## 5. Как вставлять секции в твой шаблон

После титульного листа и оглавления вставляй секции так:

```latex
\tableofcontents
\clearpage

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

Если `gost2008` не установлен, временно можно заменить на:

```latex
\bibliographystyle{plain}
\bibliography{references_draft}
```

---

## 6. Как компилировать

Из папки `coursework_latex`:

```bash
pdflatex main_msu_template.tex
bibtex main_msu_template
pdflatex main_msu_template.tex
pdflatex main_msu_template.tex
```

Важно: файл картинки должен лежать здесь:

```text
coursework_latex/photo/emb.jpg
```

Если картинки нет, будет ошибка по `\includegraphics{photo/emb.jpg}`. Тогда временно закомментируй блок с `\includegraphics`.

---

## 7. Что ещё не готово

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

## 8. Текущая логика работы

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

## 9. Следующий шаг

Следующий файл для написания:

```text
coursework_latex/sections/02_7_interpretation_limitations.tex
```

После него:

```text
coursework_latex/sections/03_conclusion.tex
```
