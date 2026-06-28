# RUN 04. Feature ablation and dimensionality reduction

Дата: 2026-06-29  
Статус: код эксперимента добавлен, результаты появятся после локального запуска на полном датасете.

---

## 1. Цель RUN 04

RUN 04 нужен не для механического увеличения accuracy, а для углубления feature-centric эксперимента.

Главный вопрос:

```text
Какие группы признаков действительно дают вклад в прогноз направления цены?
```

В RUN 03 сравнение было укрупнённым:

```text
prices-only
FinBERT-only
prices + FinBERT
```

RUN 04 делает более детальное сравнение:

```text
market features
volume / activity features
FinBERT full embeddings
FinBERT PCA embeddings
combined features
text metadata
```

---

## 2. Основной риск RUN 03

В RUN 03 объединённый вектор признаков имел размерность:

```latex
x_t = [p_t; e_t] \in \mathbb{R}^{779},
```

где:

```text
p_t: 11 рыночных признаков
e_t: 768 FinBERT embedding-компонент
```

При train size около 1104 наблюдений отношение:

```latex
\frac{n_{train}}{d} = \frac{1104}{779} \approx 1.42.
```

Это высокая размерность относительно числа наблюдений. Поэтому плохой результат `prices + FinBERT` может быть связан не только с отсутствием текстового сигнала, но и с шумом в embedding-пространстве.

---

## 3. Что делает RUN 04

Скрипт:

```text
experiments/run_04_feature_ablation.py
```

Он выполняет:

1. автоматическое определение date column;
2. построение target, если он не передан явно;
3. генерацию market features;
4. генерацию rolling / residual features;
5. генерацию volume/activity features, если есть volume;
6. определение FinBERT embedding columns;
7. PCA по FinBERT embeddings только на train subset;
8. Logistic Regression по группам признаков;
9. подбор `C` и threshold только на validation;
10. финальную оценку только на test.

---

## 4. Feature groups

RUN 04 проверяет следующие группы:

| Feature group | Состав | Что проверяет |
|---|---|---|
| `majority` | нет признаков | минимальный baseline |
| `previous_direction` | $\hat y_t = y_{t-1}$ | временной baseline |
| `prices_raw` | исходные price/inc признаки | качество старого price baseline |
| `returns_rolling` | returns, rolling mean, rolling vol, MA residuals | улучшенные рыночные признаки |
| `volume_rolling` | log-volume, relative volume, volume z-score | торговая активность |
| `market_full` | prices + returns + volume | расширенный market baseline |
| `finbert_full` | 768 FinBERT embedding-компонент | полный текстовый сигнал |
| `finbert_pca_16` | PCA(16) от FinBERT | сжатый текстовый сигнал |
| `finbert_pca_32` | PCA(32) от FinBERT | сжатый текстовый сигнал |
| `finbert_pca_64` | PCA(64) от FinBERT | сжатый текстовый сигнал |
| `market_full_finbert_pca_k` | market + PCA(k) | добавочная сила текста |
| `market_full_finbert_pca_k_text_meta` | market + PCA(k) + text metadata | текст + информационная активность |

---

## 5. Математическая логика

FinBERT embedding:

```latex
 e_t = FinBERT(text_t) \in \mathbb{R}^{768}.
```

PCA-сжатие:

```latex
 e_t^{(k)} = PCA_k(e_t), \qquad e_t^{(k)} \in \mathbb{R}^{k}.
```

Объединённый вектор:

```latex
 x_t^{(k)} = [p_t; e_t^{(k)}].
```

Добавочная сила текстовых признаков:

```latex
\Delta_{text}^{M}(k) = M(X_{market}, E_k) - M(X_{market}),
```

где $M$ — выбранная метрика качества.

Если:

```latex
\Delta_{text}^{M}(k) > 0,
```

то PCA-сжатые FinBERT-признаки улучшают расширенный market baseline по метрике $M$.

---

## 6. Как запустить

Пример запуска:

```bash
python experiments/run_04_feature_ablation.py \
  --input data/primary_dataset.csv \
  --output-dir artifacts/run_04 \
  --date-col decision_date \
  --target-col y
```

Если target ещё не сохранён в датасете, скрипт может построить его из `close`:

```latex
y_t = \mathbf{1}\{Close_{t+1} > Close_t\}.
```

Тогда можно запустить так:

```bash
python experiments/run_04_feature_ablation.py \
  --input data/primary_dataset.csv \
  --output-dir artifacts/run_04 \
  --date-col decision_date
```

---

## 7. Ожидаемые outputs

Скрипт сохраняет:

```text
artifacts/run_04/metrics.csv
artifacts/run_04/confusion_matrix.csv
artifacts/run_04/predictions.csv
artifacts/run_04/validation_selection.csv
artifacts/run_04/feature_groups.json
artifacts/run_04/dataset_summary.json
```

Главный файл для текста курсовой:

```text
artifacts/run_04/metrics.csv
```

---

## 8. Как интерпретировать результаты

Возможны три основных сценария.

### Сценарий A: PCA-FinBERT улучшает market baseline

Тогда можно писать:

```text
После снижения размерности FinBERT embeddings текстовые признаки дают добавочную предсказательную силу относительно расширенного рыночного baseline.
```

Это лучший сценарий для улучшения результата.

### Сценарий B: FinBERT-only лучше, но combined не лучше market

Тогда вывод остаётся близким к RUN 03:

```text
Текстовые признаки содержат слабый самостоятельный сигнал, но их комплементарность с рыночными признаками не подтверждается.
```

### Сценарий C: market_full сильно лучше остальных

Тогда работа усиливается за счёт вывода:

```text
После добавления volume/rolling features основной вклад дают рыночные признаки, а добавочная сила LLM-generated features не подтверждается.
```

Это тоже хороший научный вывод, потому что RUN 04 делает отрицательный результат более убедительным.

---

## 9. Как вставить в курсовую после запуска

После локального запуска нужно будет добавить раздел:

```text
2.8. Дополнительный эксперимент: ablation признаков и снижение размерности
```

В него войдут:

1. таблица feature groups;
2. таблица `metrics.csv`;
3. сравнение `market_full` и `market_full_finbert_pca_k`;
4. обновлённый вывод о добавочной силе FinBERT-признаков.

---

## 10. Почему это методологически лучше, чем просто дольше тренировать LSTM

Долгая тренировка LSTM проверяет в первую очередь архитектуру. RUN 04 проверяет именно признаки:

```text
рынок → объём → rolling dynamics → текст → PCA-текст → combined
```

Это соответствует фокусу текущей курсовой: не «какая модель лучше», а «как LLM/FinBERT-generated features влияют на прогноз».
