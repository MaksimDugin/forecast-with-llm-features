# Три контейнера для воспроизведения статей

Ubuntu 22.04, Python 3.10, linux/amd64, CPU. Определения образов подготовлены;
сборка и приёмка моделей на сервере ещё не выполнены. Это не GPU-образы.

| Образ | Модели | Реализация |
|---|---|---|
| fnspid | CNN, RNN, LSTM, GRU, Transformer, TimesNet | Зафиксированный авторский код; Keras и PyTorch в отдельных venv |
| fintexts | DLinear, PatchTST, Informer, Autoformer, iTransformer, Reformer, Crossformer, Transformer, FiLM, Nonstationary_Transformer, TSMixer, TiDE | Авторские модели и наш runner |
| finmultitime | RNN, LSTM, GRU, CNN, TimeNetReconstructed | Наши восстановленные реализации |

FinMultiTime пока **не покрывает все модели статьи**: CALF, ChatTime и
FTS-Text-MoE не имеют подключённых и проверенных адаптеров её эксперимента.
Наличие upstream CALF/ChatTime само по себе не воспроизводит протокол статьи.
Эти три модели перечислены в `blocked_models`; запуск неизвестной модели
завершается ошибкой. `TimeNetReconstructed` не объявляется авторским TimeNet.
Авторский FNSPID TimesNet — именно реализация из закреплённого репозитория.

## Сборка на сервере через Podman

Использовать отдельный чистый checkout ветки `reproduce/fintexts-finmultitime-tools`,
содержащий эту папку. Внутри репозитория:

```bash
python3 reproductions/containers/manage.py build --output "$HOME/paper-images-build-01"
python3 reproductions/containers/manage.py check --output "$HOME/paper-images-check-01"
```

Каждый каталог output должен быть новым и находиться **вне checkout**.
Сборщик копирует только committed-файлы `containers` и `paper_tools`, проверяет
их чистоту и создаёт минимальный build context. Загрузки датасетов и обучение
во время сборки не запускаются. Для сборки нужен доступ к Ubuntu registry,
apt, PyPI, PyTorch CPU index и GitHub. Пользователь сервера устанавливает Podman.

Теги: `localhost/fnspid:cpu-<12 символов HEAD>`, аналогично `fintexts`,
`finmultitime`. Все три создаются по умолчанию; для одного можно задать
`--profiles fintexts`. Логи, image IDs, исходный commit, хеши context и статусы
сохраняются в `manifest.json` и отдельных файлах. По умолчанию контейнер только
выводит inventory. Команда `check` проверяет зависимости и доступность исходников/
модулей, **не делает forward/backward и не подтверждает метрики статьи**.
FNSPID TimesNet при этой проверке разбирается через AST, потому что импорт
авторского run.py может запустить обучение.

На Windows возможны `--podman "C:/.../podman.exe" --connection fnspid`.
Только для отдельной локальной проверки допускается `check --local-cgroups-disabled`.
На сервере этот обход cgroups не включён. Общие ресурсы/лимиты назначаются
параметрами Podman при запуске эксперимента; OMP/MKL default — 4 потока.

## Перенос уже собранных образов

```bash
python3 reproductions/containers/manage.py export --output "$HOME/paper-images-export-01"
```

Получаются три OCI-архива и SHA-256 в manifest. Перед переносом сверить хеши;
на сервере загрузить каждый через `podman load -i fnspid.oci.tar` и аналогично
остальные. Имена тегов сохранены в manifest. Архивы образов не содержат датасеты,
подготовленные эмбеддинги, опубликованные checkpoints и результаты локальных запусков.
Эти файлы переносятся отдельно, с исходными manifest и SHA-256.

## Команды внутри образов

У всех образов: `inventory`, `check`, `run --help`. `run` без `--execute`
создаёт/печатает план согласно выбранному runner; обучение требует `--execute`.
Ниже примеры после сборки из того же checkout:

```bash
REV=$(git rev-parse --short=12 HEAD)
podman run --rm "localhost/fnspid:cpu-$REV" inventory
podman run --rm "localhost/fintexts:cpu-$REV" run --help
podman run --rm "localhost/finmultitime:cpu-$REV" run --help
```

Входы монтируются read-only в `/data`, результаты — в `/runs`, кэш — в `/cache`.
На SELinux-хостах использовать подходящую метку тома (`:Z` для выделенного
каталога проекта); доступ к общим данным не менять без необходимости.
Пример **плана**, ещё без обучения, с существующими абсолютными каталогами:

```bash
podman run --rm --network=none \
  -v /srv/papers/fintexts:/data:ro -v /srv/papers/runs:/runs \
  "localhost/fintexts:cpu-$REV" run \
  --data /data/AAPL.parquet --output /runs/aapl-dlinear-seed7 \
  --model DLinear --prefixes all --seed 7 --epochs 10 --device cpu
```

FinTexTS ожидает подготовленный parquet с эмбеддингами; исходные цены недостаточны.
В образе также есть `prepare.py`, `embed_fintexts.py`, `run_matrix.py` и
`inputs.lock.json` в `/opt/paper_tools`. Можно вызвать их через
`--entrypoint /opt/venv/bin/python IMAGE /opt/paper_tools/<script>.py --help`.
Скачивание данных/весов энкодера — отдельный этап с доступом к сети. Для Autoformer
на CPU требуется **`--portable-autoformer`**: документированная поправка устройства
в памяти, без изменения author checkout. `author-full` scaler повторяет авторское
масштабирование с validation/test; `train-only` — отдельный контрольный протокол.

FNSPID `run --model CNN --data-dir /data --output /runs/cnn` проверяет CSV и
показывает план. Runner собирает объединение авторских списков 5/25/50, сопоставляет
регистр файлов, проверяет колонки, длину и записывает хеши. CSV нужны внешние,
включая для RNN/TimesNet, где у авторов отсутствовала отдельная папка данных.
`--execute` копирует код/входы в новый output и запускает неизменённый run.py.
Он исполняет только варианты/эпохи, включённые в самом авторском скрипте:
**это не runner полной матрицы 72 случаев**. Stock lists и sentiment flags
авторов неодинаковы; упаковка не исправляет эти расхождения. Финальные метрики
требуют отдельного аудита. Сохранённые локально checkpoints автоматически
не подгружаются. Копии исходников и данных могут занимать дополнительное место.

FinMultiTime `run --help` описывает `--published`, ticker, train/val даты,
history/horizon, modality и опциональные признаки с временем доступности.
Модель по умолчанию GRU; прогнозы/метрики/checkpoint сохраняются в новый output.
Этот runner не заменяет ещё отсутствующий протокол мультимодальных моделей.

## Приёмка на сервере

Последовательность: build всех трёх → check окружений → на закреплённых входах
forward/backward для каждой доступной модели → короткое обучение и сохранение/
загрузка checkpoint → полные матрицы с одинаковыми splits/seeds и persistence.
Полная приёмка FinMultiTime требует отдельно закрыть три blocked-модели.
`manifest.models_acceptance=PENDING_SERVER_TESTS` остаётся после build/check/export;
успешный импорт не считается успешным экспериментом.

Закреплены Ubuntu manifest digest, авторские commit/blob hashes, прямые Python
зависимости. Разрешённые pip транзитивные версии записаны в image
`/opt/venv/environment.freeze.txt` или `/opt/{torch,keras}/environment.freeze.txt`;
OS версии — `/opt/os-packages.txt`. apt/transitive pip ещё не являются полным
восстановимым lockfile: для идентичного переноса использовать OCI-архив и image ID.
