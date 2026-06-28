# SOURCE_EXPANSION_PLAN.md

Дата: 2026-06-26  
Назначение: усилить курсовую ссылками на математику, финансовую теорию, textual finance и NLP/LLM.

---

## 1. Что сейчас не хватает

Текущая версия текста слишком быстро переходит к эксперименту. Нужно добавить источники в три слоя:

1. **Математика / ML-основание признаков**: почему вообще сравниваются группы признаков и что такое feature contribution.
2. **Финансовая теория и textual finance**: почему текст может быть связан с ценами и доходностями.
3. **NLP / FinBERT / LLM**: почему BERT/FinBERT embeddings можно использовать как признаки.

---

## 2. Источники по признакам и ML-основанию

| Источник | Куда поставить | Зачем нужен |
|---|---|---|
| Guyon, Elisseeff (2003), *An Introduction to Variable and Feature Selection* | Введение; 2.3 Признаки | Обосновать, что качество модели зависит от информативности входных признаков |
| Hochreiter, Schmidhuber (1997), *Long Short-Term Memory* | 1.1; 2.5 Модели | Обосновать LSTM как модель для последовательностей |
| Vaswani et al. (2017), *Attention Is All You Need* | 1.3 | Обосновать Transformer-family |
| Devlin et al. (2019), *BERT* | 1.3 | Обосновать контекстные embedding-представления |

---

## 3. Источники по финансовой теории и текстам

| Источник | Куда поставить | Зачем нужен |
|---|---|---|
| Fama (1970), *Efficient Capital Markets* | Введение; 1.1 | Обосновать сложность прогнозирования и роль информации |
| Tetlock (2007), *Giving Content to Investor Sentiment* | Введение; 1.2 | Связь медиа-содержания и фондового рынка |
| Loughran, McDonald (2011), *When Is a Liability Not a Liability?* | 1.2; 1.3 | Почему общие словари плохо работают в финансовом языке |
| Kearney, Liu (2014), *Textual Sentiment in Finance* | 1.2 | Survey по textual sentiment in finance |
| Nassirtoussi et al. (2014), *Text Mining for Market Prediction* | 1.2 | Systematic review по text mining для market prediction |
| Bollen, Mao, Zeng (2011), *Twitter Mood Predicts the Stock Market* | 1.2 | Ранний пример превращения текста/настроения в прогнозный сигнал |
| Mao, Counts, Bollen (2011), *Predicting Financial Markets* | 1.2 | Сравнение survey, news, Twitter и search data |

---

## 4. Источники по FinBERT и LLM

| Источник | Куда поставить | Зачем нужен |
|---|---|---|
| Araci (2019), *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* | Введение; 1.3; 2.3 | Главный источник по FinBERT для финансового sentiment analysis |
| Yang, Uy, Huang (2020), *FinBERT: A Pretrained Language Model for Financial Communications* | 1.3 | Дополнительное обоснование domain-specific FinBERT |
| Luo, Gong (2024), *Pre-trained Large Language Models for Financial Sentiment Analysis* | 1.3; ограничения | Актуальный LLM-контекст 2024 |

---

## 5. Куда добавить ссылки в тексте

### Введение

Добавить:

```latex
Прогнозирование финансовых временных рядов осложняется информационной эффективностью рынка и быстрым включением новой информации в цены \cite{fama1970efficient}.
```

После тезиса про признаки:

```latex
С точки зрения машинного обучения качество прогноза зависит не только от выбранного алгоритма, но и от информативности входных признаков \cite{guyon2003feature}.
```

После тезиса про тексты:

```latex
Ранние исследования textual finance показывают, что тональность новостного потока и финансовых сообщений может быть связана с последующей рыночной динамикой \cite{tetlock2007media,loughran2011liability,kearney2014textual}.
```

### 1.2 Текстовые данные

Добавить ссылки:

```latex
\cite{tetlock2007media,loughran2011liability,kearney2014textual,nassirtoussi2014text,bollen2011twitter,mao2011sentiment}
```

### 1.3 Языковые модели

Добавить ссылки:

```latex
\cite{vaswani2017attention,devlin2019bert,araci2019finbert,yang2020finbert,luo2024financialllm}
```

### 2.3 Признаки

Добавить:

```latex
Сравнение групп признаков в данной работе следует логике feature ablation: если при добавлении новой группы признаков качество не улучшается, её добавочная предсказательная сила в данном протоколе не подтверждается \cite{guyon2003feature}.
```

---

## 6. Следующий практический шаг

Нужно пройти по LaTeX-файлам и добавить ссылки:

```text
coursework_latex/sections/00_introduction.tex
coursework_latex/sections/01_literature_review.tex
coursework_latex/sections/02_3_features.tex
```

После этого нужно проверить `references_draft.bib` и заменить черновые записи на точные библиографические данные.
