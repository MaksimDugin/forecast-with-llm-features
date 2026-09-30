# Ошибки подготовки

## Повторное использование неполных файлов из временного рабочего каталога

- Mistake: начато чтение ранее скачанного ZIP без повторной проверки размера и хеша.
- Why it was wrong: сохранённый manifest не гарантирует сохранность текущего файла.
- Evidence: HS300_time_series.zip имеет 43005440 байт вместо 101014468 и выдаёт
  BadZipFile; первый FinTexTS parquet имеет 32810496 вместо 72261370 байт.
- Prevention rule: перед чтением и запуском проверять фактические bytes и SHA-256
  по pinned manifest; не объявлять старые результаты аудита текущим runtime PASS.

## Workflow runtime check (2026-09-30)

The first FinTexTS smoke invocation stopped at import: the isolated environment lacked matplotlib, which AutoCorrelation imports. No training occurred. Added matplotlib==3.10.7 to workflow dependencies and reran in a new output directory; preserved the failed-run manifest.

## Изолированная установка дополнительной зависимости

Separate pip install matplotlib без одновременного numpy pin обновил numpy до
несовместимой версии. Исправлено только тестовое venv: numpy==1.26.4,
contourpy==1.3.2. pip check проходит. В Containerfile зависимости устанавливаются
одной командой с pins из requirements-workflows.txt.

## Autoformer CPU inference

Реальный smoke запуск выявил .cuda() в авторской AutoCorrelation. Ошибка сохранена
в runtime validation, после RED теста добавлена отдельно включаемая поправка
устройства в памяти. Author checkout остаётся clean. Непоправленный CPU путь
не объявляется работающим.
