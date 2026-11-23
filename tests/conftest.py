"""Run-wide setup that has to happen before any test imports paddle.

paddlex reads ``PADDLE_PDX_CACHE_HOME`` once, at import
(``paddlex/utils/cache.py:29``: ``CACHE_DIR = os.environ.get(...)``), and the
first test that builds the app freezes it for the rest of the run --
``tests/test_api.py`` constructs ``ModelManager``, which calls
``configure_model_caches`` with that test's temporary home.  A later
``monkeypatch.setenv`` cannot undo that, so the OCR test's own redirect only
works when it runs alone.

Pinning the variable here, at collection time, makes a full run use the models
this checkout already has instead of downloading 97 MB of them into pytest's
temp directory.  On CI the directory does not exist and nothing is set, which
leaves the previous behaviour intact.
"""
import os
from pathlib import Path

# tests/conftest.py -> tests -> app -> project root
_MODELS = Path(__file__).resolve().parents[2] / "data" / "models" / "paddlex"
if _MODELS.is_dir():
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_MODELS))
