import json
import time

from fastapi.testclient import TestClient
from PIL import Image

from manga_localizer.api import create_app
from manga_localizer.config import AppPaths
from manga_localizer.jobs import JobStore
from manga_localizer.pipeline import PipelineRequest


class FakePipeline:
    def __init__(self, paths):
        self.paths = paths

    def run(self, request, callback):
        files = sorted(request.source.glob("*.png"))
        request.output.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files, 1):
            callback("ocr", index, len(files), source.name)
            Image.open(source).save(request.output / source.name)
        callback("complete", len(files), len(files), "done")
        return {"pages": len(files), "output": str(request.output), "images": []}


class SlowPipeline(FakePipeline):
    """Stalls between pages so a test can interrupt it mid-run."""

    def run(self, request, callback):
        files = sorted(request.source.glob("*.png"))
        request.output.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files, 1):
            time.sleep(0.05)
            callback("ocr", index, len(files), source.name)
        callback("complete", len(files), len(files), "done")
        return {"pages": len(files), "output": str(request.output), "images": []}


def make_paths(tmp_path):
    root = tmp_path / "home"
    return AppPaths(
        root, root / "models", root / "cache", root / "jobs", root / "settings.json"
    )


def test_health_models_and_job_flow(monkeypatch, tmp_path):
    monkeypatch.setattr("manga_localizer.api.ollama_tags", lambda *args, **kwargs: None)
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    assert client.get("/api/health").json()["status"] == "ok"
    local = [
        item
        for item in client.get("/api/models").json()["models"]
        if item["kind"] == "local"
    ]
    assert len(local) == 4
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (32, 32), "white").save(source / "1.png")
    output = tmp_path / "output"
    response = client.post(
        "/api/jobs", json={"source": str(source), "output": str(output)}
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"complete", "failed"}:
            break
        time.sleep(0.01)
    assert job["status"] == "complete"
    assert job["total"] == 1
    assert output.joinpath("1.png").exists()


def test_cancelled_job_stops_and_is_not_marked_complete(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=SlowPipeline))
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        Image.new("RGB", (16, 16), "white").save(source / f"{index}.png")
    output = tmp_path / "output"
    job_id = client.post(
        "/api/jobs", json={"source": str(source), "output": str(output)}
    ).json()["id"]
    time.sleep(0.06)
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.01)
    # The worker must not overwrite the cancelled state with "complete".
    time.sleep(0.15)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "cancelled"
    assert job["current"] < job["total"]


def test_cancel_unknown_job_returns_404(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_paused_job_holds_its_page_until_resumed(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=SlowPipeline))
    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        Image.new("RGB", (16, 16), "white").save(source / f"{index}.png")
    output = tmp_path / "output"
    job_id = client.post(
        "/api/jobs", json={"source": str(source), "output": str(output)}
    ).json()["id"]
    time.sleep(0.08)
    paused = client.post(f"/api/jobs/{job_id}/pause").json()
    assert paused["status"] == "paused"
    # Let the worker reach the next tick and block there.
    time.sleep(0.2)
    held = client.get(f"/api/jobs/{job_id}").json()
    assert held["status"] == "paused"
    assert 0 < held["current"] < held["total"]
    # Nothing may advance while it is parked.
    time.sleep(0.2)
    assert client.get(f"/api/jobs/{job_id}").json()["current"] == held["current"]
    assert client.post(f"/api/jobs/{job_id}/resume").json()["status"] == "running"
    for _ in range(400):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "complete":
            break
        time.sleep(0.01)
    assert job["status"] == "complete"
    assert job["current"] == job["total"]


def test_job_records_its_clock_and_ignores_the_pause_gap(tmp_path):
    # The sidebar timer needs start/end stamps and must not count paused time.
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=SlowPipeline))
    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        Image.new("RGB", (16, 16), "white").save(source / f"{index}.png")
    job_id = client.post(
        "/api/jobs", json={"source": str(source), "output": str(tmp_path / "out")}
    ).json()["id"]
    time.sleep(0.08)
    assert client.post(f"/api/jobs/{job_id}/pause").json()["status"] == "paused"
    time.sleep(0.25)
    held = client.get(f"/api/jobs/{job_id}").json()
    assert held["started_at"]
    assert held["paused_at"]
    assert client.post(f"/api/jobs/{job_id}/resume").json()["status"] == "running"
    for _ in range(400):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "complete":
            break
        time.sleep(0.01)
    assert job["status"] == "complete"
    assert job["finished_at"]
    assert job["paused_at"] is None
    assert job["paused_seconds"] >= 0.2


def test_paused_job_can_still_be_cancelled(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=SlowPipeline))
    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        Image.new("RGB", (16, 16), "white").save(source / f"{index}.png")
    job_id = client.post(
        "/api/jobs", json={"source": str(source), "output": str(tmp_path / "out")}
    ).json()["id"]
    time.sleep(0.08)
    assert client.post(f"/api/jobs/{job_id}/pause").json()["status"] == "paused"
    time.sleep(0.15)
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    time.sleep(0.2)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"


def test_pause_and_resume_unknown_job_return_404(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    assert client.post("/api/jobs/nope/pause").status_code == 404
    assert client.post("/api/jobs/nope/resume").status_code == 404


def test_leftover_paused_job_is_cancelled_on_startup(tmp_path):
    # A paused job from a previous process has no gate left to drive it, so it
    # must not linger as a resumable entry nobody can start again.
    paths = make_paths(tmp_path)
    store = JobStore(paths.jobs)
    job = store.create(PipelineRequest(source=tmp_path / "s", output=tmp_path / "o"))
    job.update(status="paused", phase="ocr", current=3, total=9)
    store.save(job)
    client = TestClient(create_app(paths, pipeline_factory=FakePipeline))
    restored = client.get(f"/api/jobs/{job['id']}").json()
    assert restored["status"] == "cancelled"
    assert restored["current"] == 3


def test_shutdown_requires_a_server_handle(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    assert client.post("/api/shutdown").status_code == 409


def test_shutdown_stops_the_server_and_cancels_running_jobs(tmp_path):
    app = create_app(make_paths(tmp_path), pipeline_factory=SlowPipeline)

    class FakeServer:
        should_exit = False

    server = FakeServer()
    app.state.server = server
    client = TestClient(app)
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        Image.new("RGB", (16, 16), "white").save(source / f"{index}.png")
    output = tmp_path / "out"
    job = client.post(
        "/api/jobs", json={"source": str(source), "output": str(output)}
    ).json()
    time.sleep(0.06)
    assert client.post("/api/shutdown").status_code == 200
    for _ in range(60):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert state["status"] == "cancelled"
    time.sleep(0.6)
    assert server.should_exit is True


def test_rejects_source_as_output(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (16, 16), "white").save(source / "1.png")
    response = client.post(
        "/api/jobs", json={"source": str(source), "output": str(source)}
    )
    assert response.status_code == 422


def test_derives_non_destructive_output_path(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    source = tmp_path / "chapter-01"
    response = client.post("/api/paths/derive-output", json={"initial": str(source)})
    assert response.status_code == 200
    assert response.json()["path"] == str(tmp_path / "chapter-01--已汉化")


def test_derives_output_under_the_picked_output_root(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    source = tmp_path / "chapter-01"
    root = tmp_path / "output"
    response = client.post(
        "/api/paths/derive-output", json={"initial": str(source), "output": str(root)}
    )
    assert response.status_code == 200
    assert response.json()["path"] == str(root / "chapter-01--已汉化")


def test_online_secret_is_memory_only_and_builtin_model_becomes_optional(tmp_path):
    app_paths = make_paths(tmp_path)
    client = TestClient(create_app(app_paths, pipeline_factory=FakePipeline))
    response = client.put(
        "/api/settings",
        json={
            "inference_backend": "online",
            "online_base_url": "https://example.test/v1",
            "online_model": "translator-a",
            "online_api_key": "do-not-persist",
        },
    )
    assert response.status_code == 200
    assert response.json()["online_api_key_configured"] is True
    assert "online_api_key" not in response.json()
    assert "do-not-persist" not in app_paths.settings.read_text(encoding="utf-8")
    models = {item["id"]: item for item in client.get("/api/models").json()["models"]}
    assert models["hy-mt2"]["required"] is False


def test_notification_switches_round_trip_through_settings(tmp_path):
    app_paths = make_paths(tmp_path)
    client = TestClient(create_app(app_paths, pipeline_factory=FakePipeline))
    # Both default to on: the app is useless if a long book finishes silently
    # while the user is in another window.
    defaults = client.get("/api/settings").json()
    assert defaults["notify_on_complete"] is True
    assert defaults["notify_on_failure"] is True
    response = client.put(
        "/api/settings",
        json={"notify_on_complete": False, "notify_on_failure": True},
    )
    assert response.status_code == 200
    assert response.json()["notify_on_complete"] is False
    assert response.json()["notify_on_failure"] is True
    reloaded = json.loads(app_paths.settings.read_text(encoding="utf-8"))
    assert reloaded["notify_on_complete"] is False
    assert reloaded["notify_on_failure"] is True


def test_notification_volume_round_trips_and_rejects_out_of_range(tmp_path):
    app_paths = make_paths(tmp_path)
    client = TestClient(create_app(app_paths, pipeline_factory=FakePipeline))
    assert client.get("/api/settings").json()["notify_volume"] == 70
    response = client.put("/api/settings", json={"notify_volume": 25})
    assert response.status_code == 200
    assert response.json()["notify_volume"] == 25
    assert json.loads(app_paths.settings.read_text(encoding="utf-8"))["notify_volume"] == 25
    # A percentage outside 0-100 would drive the gain past full scale.
    for invalid in (-10, 150):
        rejected = client.put("/api/settings", json={"notify_volume": invalid})
        assert rejected.status_code == 422
    assert client.get("/api/settings").json()["notify_volume"] == 25


def test_models_page_lists_the_ollama_tags_that_actually_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "manga_localizer.api.ollama_tags",
        lambda *args, **kwargs: {
            "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K:latest": 6_557_000_000,
            "hy-mt2-1.8b:q8": 1_900_000_000,
        },
    )
    monkeypatch.setenv("MLS_HY_MT2_OLLAMA", "hy-mt2-1.8b:q8")
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    rows = {item["id"]: item for item in client.get("/api/models").json()["models"]}
    assert rows["ollama:DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K:latest"]["size"] == "~6.1 GB"
    assert rows["ollama:hy-mt2-1.8b:q8"]["size"] == "~1.8 GB"
    assert rows["ollama:hy-mt2-1.8b:q8"]["status_code"] == "pulled"
    assert rows["ollama:hy-mt2-1.8b:q8"]["kind"] == "ollama"
    # Served from Ollama, so the torch weights on disk are no longer required.
    assert rows["hy-mt2"]["required"] is False


def test_models_page_says_so_when_ollama_is_not_answering(monkeypatch, tmp_path):
    monkeypatch.setattr("manga_localizer.api.ollama_tags", lambda *args, **kwargs: None)
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    rows = {item["id"]: item for item in client.get("/api/models").json()["models"]}
    row = rows["ollama:DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K:latest"]
    assert row["ready"] is False
    assert row["status_code"] == "offline"
    assert row["size"] == ""


def test_inference_check_lists_remote_models(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "manga_localizer.api.available_remote_models",
        lambda backend, base_url, api_key: ["qwen-test"],
    )
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    response = client.post(
        "/api/inference/check",
        json={
            "inference_backend": "ollama",
            "ocr_backend": "ollama",
            "quality_profile": "fast",
            "ollama_base_url": "http://127.0.0.1:11434",
            "ollama_model": "qwen-test",
            "ollama_ocr_model": "qwen-test",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "backend": "ollama",
        "message": "已连接，发现 1 个模型",
        "models": ["qwen-test"],
    }


def test_quality_inference_check_reuses_the_selected_9b_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "manga_localizer.api.available_remote_models",
        lambda backend, base_url, api_key: ["draft-model"],
    )
    monkeypatch.setattr("manga_localizer.api.ModelManager.is_ready", lambda *_: True)
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    response = client.post(
        "/api/inference/check",
        json={
            "inference_backend": "ollama",
            "ocr_backend": "ollama",
            "quality_profile": "quality",
            "ollama_model": "draft-model",
            "ollama_ocr_model": "draft-model",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "backend": "ollama",
        "message": "已连接，发现 1 个模型",
        "models": ["draft-model"],
    }


def _vision_manager(monkeypatch, projector: bool):
    """A manager that resolves the vision tag, with or without a projector."""
    from pathlib import Path

    from manga_localizer.llama_server import OllamaModelFiles

    files = OllamaModelFiles(
        "vision:q8",
        Path("vision.gguf"),
        Path("vision-mmproj.gguf") if projector else None,
    )

    class FakeManager:
        def resolve(self, tag):
            return files if tag == "vision:q8" else None

        def find_executable(self):
            return Path("llama-server.exe")

    monkeypatch.setattr(
        "manga_localizer.api.manager_from_settings", lambda *args, **kwargs: FakeManager()
    )


def test_inference_check_reports_the_llama_vision_model_when_only_ocr_moved(
    monkeypatch, tmp_path
):
    # The translator still answers through the daemon; only the page reader runs
    # on llama.cpp, so the answer has to describe the vision side rather than
    # fall through and report the daemon.
    _vision_manager(monkeypatch, projector=True)
    monkeypatch.setattr(
        "manga_localizer.api.available_remote_models",
        lambda backend, base_url, api_key: ["draft-model"],
    )
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    response = client.post(
        "/api/inference/check",
        json={
            "inference_backend": "ollama",
            "ocr_backend": "llama",
            "quality_profile": "quality",
            "ollama_model": "draft-model",
            "ollama_ocr_model": "vision:q8",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "backend": "ollama",
        "message": "llama.cpp 视觉 OCR 就绪：llama-server.exe、vision:q8",
        "models": ["vision:q8"],
    }


def test_inference_check_needs_the_vision_projector(monkeypatch, tmp_path):
    _vision_manager(monkeypatch, projector=False)
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    response = client.post(
        "/api/inference/check",
        json={
            "inference_backend": "ollama",
            "ocr_backend": "llama",
            "quality_profile": "quality",
            "ollama_model": "draft-model",
            "ollama_ocr_model": "vision:q8",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "backend": "ollama",
        "message": "视觉 OCR 模型 vision:q8 缺少 mmproj 投影器",
        "models": [],
    }


def test_preview_of_a_deleted_source_folder_is_a_404(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (16, 16), "white").save(source / "1.png")
    job_id = client.post(
        "/api/jobs", json={"source": str(source), "output": str(tmp_path / "output")}
    ).json()["id"]
    assert client.get(f"/api/jobs/{job_id}/preview/source/1").status_code == 200
    for image in source.iterdir():
        image.unlink()
    source.rmdir()
    response = client.get(f"/api/jobs/{job_id}/preview/source/1")
    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_inference_check_rejects_local_models_above_9b(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    response = client.post(
        "/api/inference/check",
        json={
            "inference_backend": "ollama",
            "ocr_backend": "ollama",
            "quality_profile": "quality",
            "ollama_model": "qwen3.5:14b",
            "ollama_ocr_model": "qwen3.5:9b",
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert "最多支持 9B" in response.json()["message"]
