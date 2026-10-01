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

## CSV discovery in FNSPID packaging (2026-10-01)

The first static preflight collected every `.csv` AST constant, including output
filename suffixes, and incorrectly demanded 52 inputs instead of 50. A failing
regression test reproduced this without training. Discovery now reads only literal
`names_5`, `names_25`, `names_50` lists; all six pinned scripts resolve correctly.
Prevention: derive inputs from input declarations, not all filename-like strings.

## Unbounded source-tree output (2026-10-01)

Printed an entire large FNSPID recursive tree during discovery, causing output
truncation. Filter source entries before displaying metadata; store raw responses
without printing them. The packaging manifest includes only selected source files.

## Attachment path resolution during SSH deployment preparation (2026-10-01)

- Mistake: attempted to read `upload/deploy_project(1).py` relative to the repository checkout.
- Why it was wrong: the attachment directory belongs to the scratch root, not the Git checkout.
- Evidence: read-only `sed` returned file-not-found; no files were changed by that command.
- Prevention rule: use the exact absolute workspace path supplied for attachments, independently of command cwd.
