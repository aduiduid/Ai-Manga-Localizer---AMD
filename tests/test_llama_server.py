"""Tests for the llama.cpp process manager.

The real server takes half a minute and 7 GB of VRAM to start, so everything
here runs against fakes: what matters is the logic around the process -- which
blob a tag resolves to, what happens when the configured port is taken, and
whether a finished job really gives the VRAM back.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from manga_localizer import llama_server
from manga_localizer.config import AppPaths, UserSettings
from manga_localizer.llama_server import (
    DEFAULT_REVIEW_PORT,
    DEFAULT_VISION_PORT,
    LlamaServerError,
    LlamaServerManager,
    LlamaSlot,
    OllamaModelFiles,
    discover_ollama_models,
    hold_llama_servers,
    manager_for,
    open_slots,
    open_vision_slot,
    release_llama_servers,
    running_llama_slots,
)


def _manifest(folder: Path, tag: str, layers: list[dict]) -> Path:
    """Ollama stores the manifest as a file named after the tag, no extension."""
    path = folder / tag
    path.write_text(json.dumps({"layers": layers}), encoding="utf-8")
    return path


def _layered_blobs(root: Path) -> tuple[Path, Path]:
    """A manifest tree and blob store for two tags, one with a projector."""
    manifests = root / "manifests-v2" / "ollama.com" / "library"
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "sha256-aaaa").write_bytes(b"GGUF" + b"0" * 12)
    (blobs / "sha256-bbbb").write_bytes(b"GGUF" + b"1" * 12)
    (blobs / "sha256-cccc").write_bytes(b"GGUF" + b"2" * 12)
    # Ollama lays a manifest out as <name>/<tag>, so the version file's name is
    # the tag and its parent directory is the model name.
    deep = manifests / "DeepSeek-9B"
    deep.mkdir(parents=True)
    _manifest(
        deep,
        "latest",
        [
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:aaaa"},
            {
                "mediaType": "application/vnd.ollama.image.projector",
                "digest": "sha256:bbbb",
            },
        ],
    )
    fast = manifests / "hy-mt2-1.8b"
    fast.mkdir(parents=True)
    _manifest(
        fast,
        "q8",
        [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:cccc"}],
    )
    return manifests, blobs


def test_a_manifest_maps_each_tag_to_its_gguf_blob(tmp_path):
    manifests, blobs = _layered_blobs(tmp_path)
    found = discover_ollama_models(manifests, blobs)
    # The tag is "<parent>:<file>", which is exactly the Ollama spelling the
    # settings already carry, so no translation layer is needed.
    assert set(found) == {"DeepSeek-9B:latest", "hy-mt2-1.8b:q8"}
    assert found["DeepSeek-9B:latest"].weights.name == "sha256-aaaa"
    assert found["DeepSeek-9B:latest"].projector is not None
    assert found["hy-mt2-1.8b:q8"].projector is None


def test_a_missing_blob_drops_the_tag_instead_of_pointing_at_nothing(tmp_path):
    manifests, blobs = _layered_blobs(tmp_path)
    (blobs / "sha256-cccc").unlink()
    found = discover_ollama_models(manifests, blobs)
    assert "hy-mt2-1.8b:q8" not in found


def test_resolve_tolerates_a_tag_written_without_the_latest_suffix(tmp_path):
    manifests, blobs = _layered_blobs(tmp_path)
    manager = LlamaServerManager(tmp_path / "runtime-models")
    manager._discovered = discover_ollama_models(manifests, blobs)
    assert manager.resolve("DeepSeek-9B") is not None
    assert manager.resolve("DeepSeek-9B:latest") is not None
    assert manager.resolve("nope") is None


class _FakeProcess:
    """Stands in for ``subprocess.Popen`` so no real server is launched."""

    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated += 1
        self.exit_code = 0

    def kill(self):
        self.killed += 1
        self.exit_code = -9

    def wait(self, timeout=None):
        return self.exit_code


def _fake_start(monkeypatch, manager, *, busy=()):
    """Wire ``start`` up against fakes and return the processes it created.

    ``_server_alias`` answers per port, like the real server: the readiness
    probe blocks until that port serves the requested alias, so a fake that
    always answered ``None`` would spin for the whole 300 s timeout.
    """
    created: list[_FakeProcess] = []
    state = {"serving": {}, "last_port": ""}

    monkeypatch.setattr(manager, "find_executable", lambda: Path("llama-server.exe"))
    monkeypatch.setattr(manager, "_stage_directory", lambda _exe: tmp_stage(manager))
    monkeypatch.setattr(manager, "_model_path", lambda files: (files.weights, None))
    monkeypatch.setattr(
        llama_server, "_port_is_free", lambda port, host="127.0.0.1": port not in busy
    )
    monkeypatch.setattr(
        llama_server,
        "_server_alias",
        lambda url, timeout=2.0: state["serving"].get(int(url.rsplit(":", 1)[-1])),
    )
    monkeypatch.setattr(llama_server, "_healthy", lambda url, timeout=2.0: True)

    def fake_popen(args, **kwargs):
        process = _FakeProcess()
        created.append(process)
        port = int(args[args.index("--port") + 1])
        state["last_port"] = str(port)
        state["serving"][port] = args[args.index("--alias") + 1]
        return process

    monkeypatch.setattr(llama_server.subprocess, "Popen", fake_popen)
    return created, state


def tmp_stage(manager: LlamaServerManager) -> Path:
    stage = manager.models_root / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    return stage


def _files(tmp_path: Path) -> OllamaModelFiles:
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"GGUF")
    return OllamaModelFiles(tag="DeepSeek-9B:latest", weights=weights)


def test_a_free_port_is_used_as_configured(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    created, notes = _fake_start(monkeypatch, manager)
    slot = manager.start(
        "editor", _files(tmp_path), port=8080, context=4096
    )
    assert slot.port == 8080
    assert notes["last_port"] == "8080"
    assert len(created) == 1
    assert manager.port_notes() == {}


def test_a_busy_port_moves_to_the_next_free_one_and_reports_why(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    created, notes = _fake_start(monkeypatch, manager, busy={8080, 8081})
    slot = manager.start("editor", _files(tmp_path), port=8080, context=4096)
    # The user asked for 8080 and is not getting it, so the move is recorded
    # rather than silently happening.
    assert slot.port == 8082
    assert "8080" in manager.port_notes()["editor"]
    assert "8082" in manager.port_notes()["editor"]
    assert notes["last_port"] == "8082"


def test_a_fully_busy_range_fails_with_a_readable_message(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_server, "_port_is_free", lambda port, host="": False)
    with pytest.raises(LlamaServerError) as excinfo:
        llama_server._find_free_port(8080, span=3)
    assert "8080-8082" in str(excinfo.value)


def test_a_server_already_serving_the_alias_is_reused(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    created, _ = _fake_start(monkeypatch, manager)
    first = manager.start("editor", _files(tmp_path), port=8080, context=4096)
    monkeypatch.setattr(
        llama_server, "_server_alias", lambda url, timeout=2.0: first.alias
    )
    second = manager.start("editor", _files(tmp_path), port=8080, context=4096)
    # Two processes for one model would double the VRAM for no gain.
    assert len(created) == 1
    assert second is first
    assert manager.running() == [first]


def test_a_process_that_dies_immediately_reports_its_log(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    _fake_start(monkeypatch, manager)

    def dying_popen(args, **kwargs):
        return _FakeProcess(exit_code=1)

    monkeypatch.setattr(llama_server.subprocess, "Popen", dying_popen)
    monkeypatch.setattr(llama_server, "_healthy", lambda url, timeout=2.0: False)
    with pytest.raises(LlamaServerError) as excinfo:
        manager.start("editor", _files(tmp_path), port=8080, context=4096)
    assert "立即退出" in str(excinfo.value)
    # The dead slot must not be left behind, or the next start would reuse it.
    assert manager.running() == []


def test_stop_all_terminates_every_slot(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    created, _ = _fake_start(monkeypatch, manager)
    manager.start("editor", _files(tmp_path), port=8080, context=4096)
    manager.start("review", _files(tmp_path), port=8081, context=8192)
    assert len(manager.running()) == 2
    manager.stop_all()
    assert manager.running() == []
    assert all(process.terminated == 1 for process in created)


def test_release_keeps_the_servers_while_a_batch_is_holding_them(tmp_path, monkeypatch):
    manager = manager_for(tmp_path / "models")
    created, _ = _fake_start(monkeypatch, manager)
    manager.start("editor", _files(tmp_path), port=8080, context=4096)
    with hold_llama_servers():
        release_llama_servers()
        # A per-folder release inside a batch must not stop the model that the
        # next folder is about to use.
        assert len(running_llama_slots()) == 1
    # Leaving the outermost hold is what hands the VRAM back.
    assert running_llama_slots() == []
    assert created[0].terminated == 1


def test_release_stops_the_servers_when_no_scope_is_holding_them(tmp_path, monkeypatch):
    manager = manager_for(tmp_path / "models")
    _fake_start(monkeypatch, manager)
    manager.start("editor", _files(tmp_path), port=8080, context=4096)
    release_llama_servers()
    assert running_llama_slots() == []


def test_a_finished_batch_reports_the_llama_slots_it_freed(tmp_path, monkeypatch):
    from manga_localizer.batches import release_runtime

    live = [
        LlamaSlot(
            "editor",
            "DeepSeek-9B",
            "http://127.0.0.1:8080",
            8080,
            65536,
            Path("m.gguf"),
        )
    ]
    monkeypatch.setattr("manga_localizer.batches.running_llama_slots", lambda: live)
    stopped: list[bool] = []
    monkeypatch.setattr(
        "manga_localizer.batches.stop_llama_servers", lambda: stopped.append(True)
    )
    # The OCR side stays on Ollama, so this also asks it to evict its model;
    # stub that call out so the test only describes the llama half.
    monkeypatch.setattr(
        "manga_localizer.batches.release_ollama_model", lambda *args: False
    )
    settings = UserSettings(inference_backend="llama")
    released = release_runtime(settings)
    assert stopped == [True]
    assert released == ["DeepSeek-9B@8080"]


def test_a_live_slot_is_found_by_tag_whatever_role_started_it(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    _fake_start(monkeypatch, manager)
    editor = manager.start("editor", _files(tmp_path), port=8080, context=4096)
    # The vision stage asks for the model, not for a role: the same 9B may
    # already be loaded for translation and a second copy would not fit.
    assert manager.slot_for_tag("DeepSeek-9B:latest") is editor
    assert manager.slot_for_tag("other:latest") is None


def _vision_files(tmp_path: Path, tag: str = "Vision-9B:latest") -> OllamaModelFiles:
    weights = tmp_path / "vision.gguf"
    weights.write_bytes(b"GGUF")
    projector = tmp_path / "vision-mmproj.gguf"
    projector.write_bytes(b"GGUF")
    return OllamaModelFiles(tag=tag, weights=weights, projector=projector)


def test_vision_ocr_shares_the_translation_slot_instead_of_a_second_copy(
    tmp_path, monkeypatch
):
    manager = LlamaServerManager(tmp_path / "models")
    created, _ = _fake_start(monkeypatch, manager)
    files = _vision_files(tmp_path)
    monkeypatch.setattr(manager, "resolve", lambda tag: files)
    editor = manager.start("editor", files, port=8080, context=4096)

    slot = open_vision_slot(manager, UserSettings(ollama_ocr_model=files.tag))

    assert slot is editor
    assert len(created) == 1


def test_vision_ocr_starts_the_vision_port_for_a_fresh_model(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    created, notes = _fake_start(monkeypatch, manager)
    files = _vision_files(tmp_path)
    monkeypatch.setattr(manager, "resolve", lambda tag: files)
    messages: list[str] = []

    slot = open_vision_slot(
        manager, UserSettings(ollama_ocr_model=files.tag), emit=messages.append
    )

    # Reading a page and translating a page cannot share one slot: the server
    # holds one request at a time, and OCR runs before translation begins.
    assert slot.port == DEFAULT_VISION_PORT
    assert notes["last_port"] == str(DEFAULT_VISION_PORT)
    assert messages == ["启动 llama.cpp 视觉 OCR 模型 Vision-9B:latest"]
    assert len(created) == 1


def test_vision_ocr_needs_the_mmproj_projector(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    _fake_start(monkeypatch, manager)
    text_only = OllamaModelFiles(tag="hy-mt2-1.8b:q8", weights=_files(tmp_path).weights)
    monkeypatch.setattr(manager, "resolve", lambda tag: text_only)

    with pytest.raises(LlamaServerError) as excinfo:
        open_vision_slot(manager, UserSettings(ollama_ocr_model=text_only.tag))
    # Without a projector the server cannot open a page at all, so this fails
    # up front instead of on the first request.
    assert "mmproj" in str(excinfo.value)


def test_vision_ocr_reports_a_model_that_is_not_downloaded(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    _fake_start(monkeypatch, manager)
    monkeypatch.setattr(manager, "resolve", lambda tag: None)

    with pytest.raises(LlamaServerError) as excinfo:
        open_vision_slot(manager, UserSettings(ollama_ocr_model="nope:latest"))
    assert "nope:latest" in str(excinfo.value)


def test_ocr_on_llama_cpp_alone_still_frees_the_servers(tmp_path, monkeypatch):
    from manga_localizer.batches import release_runtime

    live = [
        LlamaSlot(
            "vision",
            "DeepSeek-9B",
            "http://127.0.0.1:8082",
            8082,
            65536,
            Path("m.gguf"),
        )
    ]
    monkeypatch.setattr("manga_localizer.batches.running_llama_slots", lambda: live)
    stopped: list[bool] = []
    monkeypatch.setattr(
        "manga_localizer.batches.stop_llama_servers", lambda: stopped.append(True)
    )
    monkeypatch.setattr(
        "manga_localizer.batches.release_ollama_model", lambda *args: False
    )
    # Translation stays on Ollama here; OCR was the stage that started the
    # server, so the release cannot key off the inference backend alone.
    settings = UserSettings(inference_backend="ollama", ocr_backend="llama")
    released = release_runtime(settings)
    assert stopped == [True]
    assert released == ["DeepSeek-9B@8082"]
def test_restart_replaces_the_server_serving_that_url(tmp_path, monkeypatch):
    manifests, blobs = _layered_blobs(tmp_path)
    manager = LlamaServerManager(tmp_path / "models")
    manager._discovered = discover_ollama_models(manifests, blobs)
    created, state = _fake_start(monkeypatch, manager)
    monkeypatch.setattr(llama_server, "_MANAGERS", {"test": manager})
    manager.start("editor", manager.resolve("DeepSeek-9B:latest"), port=8080, context=4096)
    assert len(created) == 1
    # A real terminate() frees the port; the fake has to be told that it died,
    # otherwise start() finds the alias still being served and hands back a stub.
    state["serving"].clear()
    restarted = llama_server.restart_llama_server("http://127.0.0.1:8080")
    assert restarted == "http://127.0.0.1:8080"
    # A new process, not the collapsed one still holding the port.
    assert len(created) == 2
    assert len(manager.running()) == 1


def test_restart_reports_a_server_this_process_does_not_own(tmp_path, monkeypatch):
    manifests, blobs = _layered_blobs(tmp_path)
    manager = LlamaServerManager(tmp_path / "models")
    manager._discovered = discover_ollama_models(manifests, blobs)
    created, state = _fake_start(monkeypatch, manager)
    monkeypatch.setattr(llama_server, "_MANAGERS", {"test": manager})
    # A server left over from an earlier session answers before we start one, so
    # the slot is a stub: killing it would fight whoever owns it.
    state["serving"][8080] = "DeepSeek-9B-latest"
    manager.start("editor", manager.resolve("DeepSeek-9B:latest"), port=8080, context=4096)
    assert created == []
    assert llama_server.restart_llama_server("http://127.0.0.1:8080") is None


def test_restart_ignores_a_url_no_slot_serves(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_server, "_MANAGERS", {})
    assert llama_server.restart_llama_server("http://127.0.0.1:8080") is None


def test_a_server_past_its_uptime_is_replaced_instead_of_reused(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    created, _ = _fake_start(monkeypatch, manager)
    slot = manager.start("editor", _files(tmp_path), port=8080, context=4096)
    assert slot.started_at > 0.0
    # This build leaks and one process died of heap corruption after 93 minutes,
    # so a server that has been up past the limit is replaced even though it
    # still answers: its age is the only warning available.
    stale = LlamaSlot(
        "editor",
        slot.alias,
        "http://127.0.0.1:8080",
        8080,
        4096,
        slot.weights,
        started_at=llama_server.time.monotonic() - llama_server.SERVER_MAX_UPTIME_S - 60,
    )
    manager._slots["editor"] = (stale, created[0])
    assert manager.reusable_slot("DeepSeek-9B:latest") is None
    assert "分钟" in manager.port_notes()["editor"]


def test_a_young_server_is_still_reused(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    _fake_start(monkeypatch, manager)
    slot = manager.start("editor", _files(tmp_path), port=8080, context=4096)
    assert manager.reusable_slot("DeepSeek-9B:latest") is slot


def _bare_files(tmp_path: Path, tag: str) -> OllamaModelFiles:
    """A GGUF dropped into the models directory under its own name."""
    weights = tmp_path / f"{tag}.gguf"
    weights.write_bytes(b"GGUF" + b"0" * 12)
    return OllamaModelFiles(tag=tag, weights=weights)


def test_a_gguf_in_the_models_directory_is_found_by_its_own_name(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "Local-7B-Q6_K.gguf").write_bytes(b"GGUF" + b"0" * 12)
    (models / "Local-7B-Q6_K-mmproj.gguf").write_bytes(b"GGUF" + b"1" * 12)
    manager = LlamaServerManager(models)
    manager._discovered = {}

    files = manager.resolve("Local-7B-Q6_K")

    # ``_model_path`` derives exactly this path from the tag, so a file the user
    # dropped in needs no Ollama manifest and no second copy of 6 GB.
    assert files is not None
    assert files.weights == models / "Local-7B-Q6_K.gguf"
    assert files.projector == models / "Local-7B-Q6_K-mmproj.gguf"
    assert manager.resolve("Local-7B-Q6_K:q8") is None


def test_a_pair_of_weight_files_past_the_limit_is_not_resident(tmp_path):
    small = _files(tmp_path)
    big = OllamaModelFiles(tag="Local-7B", weights=tmp_path / "big.gguf")
    big.weights.write_bytes(b"0" * 64)

    assert llama_server.pair_fits_resident(small, small) is True

    monkey = llama_server.PAIR_RESIDENT_LIMIT_BYTES
    try:
        llama_server.PAIR_RESIDENT_LIMIT_BYTES = 16
        assert llama_server.pair_fits_resident(small, big) is False
    finally:
        llama_server.PAIR_RESIDENT_LIMIT_BYTES = monkey


def _two_tag_manager(tmp_path: Path, monkeypatch) -> LlamaServerManager:
    """A manager whose editor and candidate resolve, with the pair rule forced."""
    manager = LlamaServerManager(tmp_path / "models")
    editor = _vision_files(tmp_path, "DeepSeek-9B:latest")
    candidate = OllamaModelFiles(
        tag="Local-7B:latest", weights=editor.weights, projector=None
    )
    monkeypatch.setattr(
        manager,
        "resolve",
        lambda tag: editor if tag == editor.tag else candidate,
    )
    return manager


def test_a_candidate_that_cannot_share_the_card_waits_for_its_pass(
    tmp_path, monkeypatch
):
    manager = _two_tag_manager(tmp_path, monkeypatch)
    created, _ = _fake_start(monkeypatch, manager)
    monkeypatch.setattr(llama_server, "pair_fits_resident", lambda a, b: False)
    messages: list[str] = []

    slots = open_slots(
        manager,
        UserSettings(
            llama_model="DeepSeek-9B:latest",
            llama_review_model="Local-7B:latest",
            quality_profile="quality",
        ),
        emit=messages.append,
    )

    # One process only: the candidate is registered and left for the pass that
    # needs it, so 8,083 MB and 6,503 MB never sit on the card together.
    assert slots["review"] is None
    assert manager.can_relaunch("review")
    assert len(created) == 1
    assert "无法同时驻留" in messages[-1]

    slot = manager.relaunch("review")

    assert slot.port == DEFAULT_REVIEW_PORT
    assert len(created) == 2


def test_a_candidate_that_fits_is_started_beside_the_editor(tmp_path, monkeypatch):
    manager = _two_tag_manager(tmp_path, monkeypatch)
    created, _ = _fake_start(monkeypatch, manager)

    slots = open_slots(
        manager,
        UserSettings(
            llama_model="DeepSeek-9B:latest",
            llama_review_model="Local-7B:latest",
            quality_profile="quality",
        ),
    )

    # The 1.8B candidate fits, and starting it up front is what keeps the
    # candidate phase from paying a model load in the middle of a book.
    assert slots["review"] is not None
    assert len(created) == 2


def test_adopting_a_slot_moves_it_to_the_role_that_will_stop_it(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    _fake_start(monkeypatch, manager)
    vision = manager.start("vision", _files(tmp_path), port=8082, context=4096)

    adopted = manager.adopt("editor", vision)

    assert adopted.role == "editor"
    assert [slot.role for slot in manager.running()] == ["editor"]
    manager.stop("editor")
    assert manager.running() == []


def test_relaunch_starts_the_role_again_from_its_recipe(tmp_path, monkeypatch):
    manager = LlamaServerManager(tmp_path / "models")
    created, _ = _fake_start(monkeypatch, manager)
    manager.register("review", _files(tmp_path), 8081, 4096)

    slot = manager.relaunch("review")

    assert (slot.role, slot.port, slot.context) == ("review", 8081, 4096)
    assert len(created) == 1
    assert manager.relaunch("review") is slot


def test_relaunch_without_a_recipe_says_so(tmp_path):
    manager = LlamaServerManager(tmp_path / "models")

    with pytest.raises(LlamaServerError) as excinfo:
        manager.relaunch("review")

    assert "可重启" in str(excinfo.value)
