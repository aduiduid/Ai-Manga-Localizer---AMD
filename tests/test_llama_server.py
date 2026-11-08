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
    LlamaServerError,
    LlamaServerManager,
    LlamaSlot,
    OllamaModelFiles,
    discover_ollama_models,
    hold_llama_servers,
    manager_for,
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
