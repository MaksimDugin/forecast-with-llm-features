# Подготовка FinTexTS и FinMultiTime

Это отдельное дополнение к репозиторию. Оно не меняет локальные
`reproductions/fnspid`, `fintexts/sources.json`, `finmultitime/sources.json`.
Точные результаты статей ещё не воспроизведены.

## Что реализовано

- `prepare.py`: загрузка конкретных ревизий опубликованных данных, проверка
  размера и SHA-256, отказ перезаписывать повреждённый существующий файл.
  FinMultiTime: только перечисленные HS300-архивы и две таблицы наличия данных;
  это не весь датасет и не полный набор экспериментов статьи.
- Получение чистого авторского FinTexTS на фиксированном коммите.
- `run_fintexts.py`: проверка prepared parquet и непустых окон train/val/test,
  запуск неизменённого `forecasting_task.run`, уникальная папка результатов,
  хеш входного файла, команда запуска, лог и статус. Без `--execute` только план.
- Отдельное CPU-окружение Ubuntu 22.04 для Podman. Образ в этой среде ещё
  НЕ собран. Прямые пакеты частично зафиксированы, остальные разрешаются при
  сборке; freeze сохраняется в образе. Это не восстановленный авторский lockfile.

## Подготовка (из корня проекта)

```bash
python reproductions/paper_tools/prepare.py list fintexts
python reproductions/paper_tools/prepare.py source fintexts --destination .reproduction-cache/FinTexTS
python reproductions/paper_tools/prepare.py download fintexts --destination .reproduction-cache/fintexts-published
python reproductions/paper_tools/prepare.py download finmultitime --destination .reproduction-cache/finmultitime-published
```

Загрузка FinTexTS около 305 MB, выбранных файлов FinMultiTime около 217 MB.
Повторная проверка без сети: та же команда с `verify` вместо `download`.
При несовпадении хеша существующий файл сохраняется, команда завершается ошибкой.
Незавершённый git checkout также не удаляется и не перезаписывается автоматически.

## Контракт FinTexTS

Авторский загрузчик требует отдельный parquet для каждой акции:
`date` (строка YYYY-MM-DD), `ticker`, `open`, `high`, `low`, `close`,
`{prefix}_emb0` ... `{prefix}_emb383` для выбранных категорий
`macro`, `sector`, `targetCompany`, `relatedCompany`, `filing`.
Опубликованный сырой текст не заменяет эти векторы.
Энкодер и способ получения именно опубликованных 384-мерных признаков не
восстановлены; новый энкодер означал бы отдельную реконструкцию эксперимента.
Эта версия инструмента не генерирует эмбеддинги и не выдумывает их из текста.

Авторский протокол: 64 строки истории, 16 decoder history, горизонт 3 строки,
2019-01-01 <= date < 2023-12-19; train до 2022, validation 2022, test 2023.
Defaults скрипта — 10 эпох, seed 7, batch 32, text_weight 0.1.
Это defaults КОДА, а не утверждение об окончательной матрице статьи v3.
Scaler авторов обучается на всех train/validation/test значениях.
Мы сохраняем это поведение в режиме повторения авторского кода и явно
фиксируем ограничение. Вывод об отсутствии утечки делать нельзя.
Авторский runner сохраняет `result.txt` и `test_io.pt`, но не веса моделей.

```bash
python reproductions/paper_tools/run_fintexts.py --author .reproduction-cache/FinTexTS --data /absolute/path/NVDA.parquet --output /absolute/path/new-run --model PatchTST
```

Для обучения добавить `--execute`. Нужны зависимости author requirements и
`pyarrow`; рекомендуется отдельный контейнер, а не глобальный Python.
Не передавайте raw dataset вместо prepared parquet. Если text_weight=0,
авторский загрузчик всё равно требует embedding-колонки: это не обходится.

## Podman (Linux или настроенный Windows remote client)

```bash
podman build -f reproductions/paper_tools/Containerfile.fintexts-cpu -t localhost/fintexts-cpu:prepare reproductions/paper_tools
podman run --rm localhost/fintexts-cpu:prepare
```

Пример Linux для проверки prepared файла без обучения:

```bash
podman run --rm --network=none -v /absolute/prepared:/data:ro -v /absolute/results:/results localhost/fintexts-cpu:prepare python /opt/paper_tools/run_fintexts.py --author /opt/FinTexTS --data /data/NVDA.parquet --output /results/new-run
```

Для Windows используются пути, доступные конкретной Podman machine.
Проблема cgroups локальной машины не исправляется этим Containerfile.
Отключение cgroups не включено в серверные команды. Деплой выполняет пользователь.

## FinMultiTime: что остаётся открытым

В v2 перечислены RNN/LSTM/GRU/CNN/TimeNet и CALF/ChatTime/FTS-Text-MoE.
Таблицы 7–8 описывают модальности и горизонты; таблица 9 — масштабы 5/15/35 акций
двух рынков. Раздел 4.3 описывает 96 дней истории, следующие 24 дня и 100 эпох;
рядом встречаются горизонты в часах. Не переносим эти параметры на все таблицы
без подтверждения. Авторская архитектура TimeNet не заменяется молча TimesNet.
Не найден подтверждённый авторский training repository, отсутствуют точные
конфиги/сплиты/идентичность выброса/полная схема модальных признаков.
Поэтому команда обучения FinMultiTime пока не предоставлена как «повторение».

## Проверки

```bash
python -m unittest discover -s reproductions/paper_tools -p 'test_*.py' -v
python -m compileall -q reproductions/paper_tools
```

Тесты используют небольшие синтетические записи, проверяют защиту входов,
формат эмбеддингов и вычисление непустых окон. Они не подтверждают обучение,
валидность опубликованного датасета или метрики статьи.
