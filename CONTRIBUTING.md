# Contributing

1. Create a Python 3.12 environment with `scripts/bootstrap.ps1 -SkipModels -Dev`
   or `./scripts/bootstrap.sh --skip-models --dev`.
   With uv installed, both commands use the committed `uv.lock` automatically.
2. Keep source images immutable. Render only to a distinct output folder.
3. Run `uv run --frozen --no-sync pytest` before opening a pull request.
4. When changing the UI, update `docs/PARITY_LEDGER.md` and verify desktop and
   390 px mobile layouts.
5. Bump the version in the same commit as the change, and size the bump to the
   change:
   - **minor** (`0.6.0` → `0.7.0`) for anything a user can notice as new
     capability: a new feature, a new setting, a new API surface, or a behavioural
     change to an existing flow;
   - **patch** (`0.6.0` → `0.6.1`) for fixes that restore intended behaviour
     without adding capability;
   - **major** only for a change that breaks existing saved data, job records, or
     the request/response contract.

   A version lives in three places and `tests/test_version.py` fails if they
   disagree: `src/manga_localizer/__init__.py` (`__version__`, which the API, the
   CLI, and the UI footer report), `pyproject.toml` (what gets built), and
   `uv.lock` (the editable package `uv sync --frozen` resolves). Add the release
   heading to `CHANGELOG.md` in the same commit.

By contributing, you agree that your contribution is licensed under MIT.
