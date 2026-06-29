# RUN 05. Candidate Kaggle datasets

Цель: быстро проверить несколько датасетов и выбрать те, где текстовые признаки с наибольшей вероятностью могут дать прирост относительно market baseline.

## Основные кандидаты

```text
oliviervha/crypto-news
miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
equinxx/stock-tweets-for-sentiment-analysis-and-prediction
aaron7sun/stocknews
yash612/stockmarket-sentiment-dataset
jeet2016/us-financial-news-articles
ankurzing/sentiment-analysis-for-financial-news
sbhatti/financial-sentiment-analysis
sudalairajkumar/cryptocurrencypricehistory
mczielinski/bitcoin-historical-data
```

## Как использовать

Вставить список в `DATASET_SLUGS` в notebook:

```text
notebooks/05_multi_kaggle_dataset_research.ipynb
```

или создать рядом файл:

```text
dataset_slugs.txt
```

и вставить туда slugs по одному на строку.

## Приоритеты

### 1. Crypto/news datasets

Лучше всего подходят для RUN 05, если есть:

- timestamp/date;
- title/text/body;
- упоминания BTC/ETH/SOL/etc.;
- достаточное пересечение с price data.

Главный кандидат:

```text
oliviervha/crypto-news
```

### 2. Stock news datasets with ticker/company alignment

Лучше всего подходят, если есть:

- дата публикации;
- ticker/company;
- headline/body;
- много компаний и длинный период.

Главный кандидат:

```text
miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
```

### 3. Stock tweets datasets

Могут дать более живой sentiment-сигнал, но обычно шумнее.

Кандидат:

```text
equinxx/stock-tweets-for-sentiment-analysis-and-prediction
```

### 4. Sentiment-only datasets

Полезны для проверки модели текста, но обычно хуже подходят для price prediction, если нет даты и актива.

Кандидаты:

```text
ankurzing/sentiment-analysis-for-financial-news
sbhatti/financial-sentiment-analysis
yash612/stockmarket-sentiment-dataset
```

### 5. Price-only datasets

Нужны не для текстового сигнала, а как источник price baseline, если `yfinance` не подходит.

Кандидаты:

```text
sudalairajkumar/cryptocurrencypricehistory
mczielinski/bitcoin-historical-data
```

## Что считать хорошим датасетом

Минимальные требования:

```text
>= 1000 текстов
>= 200 уникальных дат
есть date/timestamp
есть title/text/body
есть asset/ticker или текстовые упоминания актива
цены можно подтянуть через yfinance или есть внутри датасета
```

Для полного RUN 05 лучше:

```text
>= 10000 текстов
>= 365 уникальных дат
несколько активов
не слишком перекошенный target: positive_share примерно 0.4--0.6
```

## Что смотреть после запуска notebook

Главные файлы:

```text
artifacts/run_05_multi_dataset_research/all_table_profiles.csv
artifacts/run_05_multi_dataset_research/all_asset_text_coverage.csv
artifacts/run_05_multi_dataset_research/all_price_overlap_candidates.csv
artifacts/run_05_multi_dataset_research/dataset_leaderboard_top50.csv
artifacts/run_05_multi_dataset_research/shortlist_for_run05.csv
```

Финальный выбор делаем по `shortlist_for_run05.csv` и `dataset_leaderboard_top50.csv`.
