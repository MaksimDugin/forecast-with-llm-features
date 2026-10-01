# Воспроизведение FinTexTS и реконструкция FinMultiTime

Код добавлен отдельно от локального FNSPID. Все длительные обучения требуют
`--execute`; без флага runner показывает план. Точные табличные результаты статей
ещё не воспроизведены. Проверены реальные короткие CPU-запуски, см. VALIDATION.json.

## Реализовано

- `prepare.py`: данные на фиксированных HF revisions, размеры и SHA-256;
  повреждённые существующие файлы не перезаписываются. Получение чистого author
  checkout FinTexTS на коммите из inputs.lock.json.
- `embed_fintexts.py`: чтение опубликованных parquet, отбор акций, фиксированный
  384-dimensional SBERT, среднее по присутствующим категориям каждой из пяти
  групп. Пустая группа = NaN; author loader усредняет доступные группы и заменяет
  полностью отсутствующий текст нулями. Модель all-MiniLM-L6-v2 и её revision
  записываются в embedding_manifest.json. Это наш реконструированный энкодер:
  точное имя SBERT авторов не опубликовано. Длинный текст обрезается согласно
  max_seq_length энкодера. Новое LLM-сопоставление новостей здесь не выполняется.
- `train_fintexts.py`: 12 неизменённых авторских моделей, author TextModel и
  training step; orchestration сохраняет лучший по validation checkpoint,
  историю, scaled MSE/MAE, persistence MSE и прогнозы. Test оценивается после
  выбора лучшей эпохи. Чекпойнт включает scaler и веса обеих моделей.
- `finmultitime.py`: восстановленные RNN/LSTM/GRU (2 слоя, hidden=64), CNN
  (2 слоя) и TimeNetReconstructed (гипотеза 4 convolution layers). Прогноз OHLC,
  история 96 торговых строк, горизонты 24/48/96, Adam lr=0.001, ранняя остановка
  после 5 эпох без улучшения. Архитектуры, границы сплитов, список акций,
  seed и отсутствие авторского удаления выброса явно отличаются от нераскрытого
  протокола статьи. TimeNetReconstructed не выдаётся за авторский TimeNet.
- `news_features.py`: HS300 новости, стабильные article_id, дневное количество,
  экспорт текстов для sentiment scoring. Опционально импортирует scores CSV
  article_id,sentiment в [-1,1] с обязательным model/revision provenance.
  Самостоятельный авторский sentiment scorer пока не восстановлен. Counts —
  отдельный восстановленный признак, а не paper sentiment.
- `run_matrix.py`: последовательная серия ticker × model × seed × horizon,
  отдельные логи, остановка при сбое, summary с полнотой матрицы. Метрики сначала
  усредняются по seeds внутри акции, затем с равным весом по акциям. Smoke runs
  помечаются и не считаются результатами статьи. Автоматического resume нет.
- `run_fintexts.py`: прежний запуск полностью неизменённого author CLI.
  Этот путь не сохраняет веса; для checkpoint используется train_fintexts.py.

## Окружение и данные

Из корня репозитория, в отдельном Python окружении:

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r reproductions/paper_tools/requirements-workflows.txt
python -m pip check
python reproductions/paper_tools/prepare.py source fintexts --destination .reproduction-cache/FinTexTS
python reproductions/paper_tools/prepare.py download fintexts --destination .reproduction-cache/fintexts-published
python reproductions/paper_tools/prepare.py download finmultitime --destination .reproduction-cache/finmultitime-published
```

FinTexTS: 305 MB. Выбранные HS300 файлы FinMultiTime: около 217 MB;
это не весь опубликованный датасет. `verify` вместо `download` проверяет офлайн.
Энкодер один раз скачивается отдельно с HF, затем поддерживается `--offline`.
Глобальное Python окружение изменять не требуется.

```bash
python reproductions/paper_tools/embed_fintexts.py --published .reproduction-cache/fintexts-published --output .reproduction-cache/fintexts-prepared --tickers AAPL MSFT
```

Для всех 100 компаний передать тикеры из fintexts.100-tickers.reconstructed.json.
Скрипт сохраняет отдельный parquet и manifest; существующий output не заменяется.
Подготовка всех пяти групп может быть длительной даже до начала обучения.

## FinTexTS: единичный запуск и матрица

```bash
python reproductions/paper_tools/train_fintexts.py --author .reproduction-cache/FinTexTS --data .reproduction-cache/fintexts-prepared/AAPL.parquet --output .reproduction-cache/fintexts-dlinear --model DLinear --execute
python reproductions/paper_tools/run_matrix.py --config reproductions/paper_tools/fintexts.matrix.example.json --output .reproduction-cache/fintexts-matrix
```

Последняя команда только печатает план. Добавить `--execute` для обучения.
Для короткой проверки: `--epochs 1 --smoke-batches 1 --execute` у train_fintexts.py;
у matrix указать epochs=1, smoke_batches=1 в собственной копии конфигурации.

Протокол author loader: OHLC, 64 history, 16 decoder history, horizon 3;
2019-01-01 <= date < 2023-12-19; train до 2022, validation 2022, test 2023.
`--scaler author-full` сохраняет авторскую нормализацию по всем данным, включая
validation/test. `--scaler train-only` — отдельный изменённый протокол без этой
утечки scaler. Публикуемые тексты требуют отдельного аудита временной доступности;
одна нормализация train-only не доказывает отсутствие других утечек.

Autoformer в pinned source принудительно вызывает .cuda() при inference.
`--portable-autoformer` включает узкую замену .cuda() на .to(values.device)
в копии двух методов в памяти. Исходные файлы не меняются. Флаг записан в manifest.
CPU example matrix включает эту поправку; полностью неизменённый Autoformer
на CPU не проходит inference. RED/green тесты и реальный исправленный запуск проверены.

fintexts.100-tickers.reconstructed.json содержит 100 фактических тикеров
опубликованного датасета × 12 моделей × seeds [7,17,27] = 3600 запусков.
Это реконструкция: значения seed выбраны нами, cap 10 epochs взят из author CLI,
не из восстановленного конечного конфига статьи. Эмбеддинги должны быть готовы.

## FinMultiTime: HS300 и дополнительные модальности

```bash
python reproductions/paper_tools/finmultitime.py --published .reproduction-cache/finmultitime-published --ticker 000001.SZ --train-end 2022-01-01 --val-end 2023-01-01 --model GRU --horizon 24 --output .reproduction-cache/fmt-gru
python reproductions/paper_tools/news_features.py --published .reproduction-cache/finmultitime-published --ticker 000001.SZ --output .reproduction-cache/fmt-news
```

Первая команда печатает план. Даты обязательны и заданы нами, это не авторские
границы. Ни один target window не пересекает границы сплитов. Scaler fit train only.
Для обучения добавить --execute. Пример матрицы содержит 45 price-only запусков
на одной акции: 5 моделей × 3 seed × 3 горизонта.

Для news/table/image режима передать `--modality`, `--features features.csv`
и `--feature-columns ...`. CSV требует available_at с достоверным временем
доступности и числовые признаки. Для HS300 используется backward as-of join
на 15:00 Asia/Shanghai каждого торгового дня; будущие признаки и backfill запрещены.
News counts агрегируются до следующей местной полуночи, пропуски дней = 0.
До первой доступной записи признаки = 0. Остальные признаки переносятся вперёд
до следующей записи; выбор признаков и timestamp provenance остаются явными.
У matrix эти параметры задаются modality, feature_files (ticker→path), feature_columns.

В финансовых таблицах опубликованы end_date без надёжной даты объявления.
Поэтому end_date не используется как available_at автоматически. Полугодовой
K-line chart также нельзя автоматически считать известным в начале полугодия.
Извлечение этих модальностей без проверенной доступности не заявляется выполненным.
SP500 adapter, CALF, ChatTime, FTS-Text-MoE, точные 35+35 компаний и авторское
удаление выброса ещё не реализованы; подтверждённый training repo FinMultiTime не найден.

## Podman / Ubuntu 22.04

```bash
podman build -f reproductions/paper_tools/Containerfile.fintexts-cpu -t localhost/paper-workflows:cpu reproductions/paper_tools
podman run --rm localhost/paper-workflows:cpu
```

Образ содержит инструменты обеих статей и pinned source FinTexTS. Источник,
данные и результаты контейнера доступны как обычные volume mounts. CMD печатает
help и не запускает обучение. Прямые зависимости зафиксированы; transitive freeze
пишется в /opt/environment.freeze.txt. Base digest и OS packages ещё не locked.
Образ здесь не собран: проверено отдельное Linux Python 3.12 CPU окружение,
а совместимость Ubuntu 22.04/Python 3.10 остаётся проверкой сборки.
Для Windows указывать явное --connection fnspid; пути volume должны быть доступны
этой machine. Локальная проблема cgroups не исправляется этим Containerfile.
Деплой и полное серверное обучение не запускались.

## Проверки

```bash
python -m unittest discover -s reproductions/paper_tools -p 'test_*.py' -v
python -m compileall -q reproductions/paper_tools
```

19 тестов: SHA/неполные загрузки, формат embeddings, split boundaries,
as-of availability, pooling, gradients всех восстановленных архитектур,
безопасная матрица, averaging и отдельная device correction. Для pytest/ruff
при необходимости установить pytest==8.3.5 ruff==0.11.13 в dev окружение.
Реальные smoke запуски используют одну эпоху и один batch каждого сплита;
их метрики показывают работоспособность и не оценивают качество обучения.

## Three server images

See [the three-image Podman workflow](../containers/README.md) for separate FNSPID,
FinTexTS and FinMultiTime CPU images. FinMultiTime coverage remains five reconstructed
models; three multimodal paper models are explicitly blocked. Server image/model
acceptance is pending, independent of earlier local smoke/pilot evidence.
