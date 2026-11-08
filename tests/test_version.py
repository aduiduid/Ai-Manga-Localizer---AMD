import re
from pathlib import Path

from manga_localizer import __version__


ROOT = Path(__file__).parents[1]


def _declared(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"{pattern!r} not found"
    return match.group(1)


def test_the_version_number_has_one_truth_in_every_file_that_declares_it():
    # `__version__` is what /api/system, the CLI and the UI footer report;
    # pyproject.toml is what gets built and installed; uv.lock pins the editable
    # package that `uv sync --frozen` resolves.  A bump that touches fewer than
    # all three either ships a stale lock or reports a version nobody installed.
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert _declared(r'^version = "([^"]+)"', project) == __version__
    assert (
        _declared(
            r'name = "manga-localizer-studio"\nversion = "([^"]+)"', lock
        )
        == __version__
    )
    # Every release note has to open with the version that is actually shipping,
    # otherwise the newest heading describes an unreleased build.
    assert f"## {__version__} - " in changelog


def test_the_version_number_is_a_plain_release_triple():
    # Two numbers (or a stray suffix) would confuse the release tags, which are
    # `vX.Y.Z`.  Keep the shape checkable so a bump cannot invent a new one.
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__
