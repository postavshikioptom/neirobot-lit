# Persistent Agent Memory for giter

## Статистика

- Коммит `8e4af6c` по задаче 328 затронул `python_lab/src/train.py`, `python_lab/src/train_cli.py`, `python_lab/src/train_model_factory.py`, `python_lab/src/train_module.py`, `python_lab/src/utils.py`.
- Часто меняемые файлы вокруг train-контура: `python_lab/src/train.py`, `python_lab/src/train_cli.py`, `python_lab/src/train_module.py`, `python_lab/src/utils.py`, `python_lab/src/train_data.py`.

## Формат коммитов

- Пользователь ожидает осмысленный commit на русском с описанием именно `.py`/`.rs` изменений.
- Сообщение должно быть подробным: 7-10 предложений, с привязкой к конкретным файлам и смыслу изменений.
- Документацию можно коммитить вместе с кодом, но в сообщении акцент делать на Python/Rust части.

## Ошибки git

- Если запускать `git add`, `git commit` и `git push` параллельно, можно получить зависший `.git/index.lock`; для git-операций выполнять шаги последовательно.
- После неудачного `git commit` staged-индекс может сохраниться, поэтому сначала проверять `git status`, затем повторять только недостающий шаг.
- В PowerShell для удаления `index.lock` безопаснее использовать `cmd /c "if exist .git\\index.lock del /f /q .git\\index.lock"`.

## Особенности проекта

- Основная рабочая ветка: `main`, push идёт в `origin/main`.
- В репозитории часто коммитятся одновременно код в `python_lab/src/*` и документы задач в `docs/*.md`.
