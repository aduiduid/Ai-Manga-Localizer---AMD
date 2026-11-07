"""Batch (multi-folder queue) scheduling: ordering, failures, cancel, release."""

import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from manga_localizer.api import create_app
from manga_localizer.batches import BatchStore, make_task
from manga_localizer.config import AppPaths


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


class FlakyPipeline(FakePipeline):
    """Fails on exactly one folder, so the batch has to carry on without it."""

    failing = "chapter-02"

    def run(self, request, callback):
        if request.source.name == self.failing:
            callback("ocr", 1, 1, request.source.name)
            raise RuntimeError("OCR exploded")
        return super().run(request, callback)


class SlowPipeline(FakePipeline):
    """Stalls between pages so a test can interrupt a folder mid-run."""

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


def make_folder(root, name, pages=1):
    folder = root / name
    folder.mkdir(parents=True)
    for page in range(1, pages + 1):
        Image.new("RGB", (16, 16), "white").save(folder / f"{page}.png")
    return folder


def start_batch(client, sources, output=""):
    return client.post(
        "/api/batches",
        json={"source": [str(path) for path in sources], "output": str(output)},
    )


def wait_for_batch(client, batch_id, timeout=5.0):
    deadline = time.time() + timeout
    batch = None
    while time.time() < deadline:
        batch = client.get(f"/api/jobs/{batch_id}").json()
        if batch["status"] in {"completed", "failed", "cancelled"}:
            return batch
        time.sleep(0.01)
    raise AssertionError(f"batch never finished: {batch}")


def wait_for_running_task(client, batch_id, index=0, timeout=5.0):
    deadline = time.time() + timeout
    batch = None
    while time.time() < deadline:
        batch = client.get(f"/api/jobs/{batch_id}").json()
        if batch["tasks"][index]["status"] == "running":
            return batch
        time.sleep(0.01)
    raise AssertionError(f"folder never started: {batch}")


def test_a_failing_folder_is_recorded_and_the_rest_still_run(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FlakyPipeline))
    folders = [
        make_folder(tmp_path, name)
        for name in ("chapter-01", "chapter-02", "chapter-03")
    ]
    output = tmp_path / "out"

    response = start_batch(client, folders, output)
    assert response.status_code == 200
    batch = wait_for_batch(client, response.json()["id"])

    assert batch["kind"] == "batch"
    assert batch["status"] == "completed"
    assert [task["status"] for task in batch["tasks"]] == [
        "complete",
        "failed",
        "complete",
    ]
    assert (batch["completed"], batch["failed"], batch["cancelled"]) == (2, 1, 0)
    assert batch["message"] == "成功 2 个 / 失败 1 个"
    assert batch["progress"] == 100
    # The failing folder keeps its reason for the queue list...
    assert "OCR exploded" in batch["tasks"][1]["error"]
    # ...and the folder after it still produced its pages.
    assert (output / "chapter-01--已汉化" / "1.png").exists()
    assert (output / "chapter-03--已汉化" / "1.png").exists()
    assert not (output / "chapter-02--已汉化").exists()
    # Every queued folder owns a job record, so pause/preview/history work.
    assert all(task["job_id"] for task in batch["tasks"])
    assert len(client.get("/api/jobs").json()["jobs"]) == 3


def test_every_bad_path_is_reported_before_anything_runs(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    good = make_folder(tmp_path, "chapter-01")
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "missing"

    response = start_batch(client, [good, missing, empty])

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert str(missing) in detail and str(empty) in detail
    # Validation happens at enqueue time, so nothing was queued at all.
    assert client.get("/api/batches").json()["batches"] == []
    assert client.get("/api/jobs").json()["jobs"] == []


def test_duplicate_folders_are_dropped_and_the_order_is_kept(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    first = make_folder(tmp_path, "chapter-01")
    second = make_folder(tmp_path, "chapter-02")

    response = start_batch(client, [first, second, f"{first}\\", str(first).upper()])

    batch = wait_for_batch(client, response.json()["id"])
    assert [task["name"] for task in batch["tasks"]] == ["chapter-01", "chapter-02"]
    assert batch["duplicates_removed"] == 2
    assert batch["total"] == 2


def test_cancel_stops_the_running_folder_and_drops_the_queue(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=SlowPipeline))
    folders = [
        make_folder(tmp_path, name, pages=6)
        for name in ("chapter-01", "chapter-02", "chapter-03")
    ]
    output = tmp_path / "out"
    batch_id = start_batch(client, folders, output).json()["id"]
    wait_for_running_task(client, batch_id)

    cancelled = client.post(f"/api/jobs/{batch_id}/cancel").json()
    assert cancelled["status"] == "cancelled"
    assert [task["status"] for task in cancelled["tasks"]] == [
        "cancelled",
        "cancelled",
        "cancelled",
    ]

    deadline = time.time() + 5
    while time.time() < deadline:
        batch = client.get(f"/api/jobs/{batch_id}").json()
        if not any(task["status"] == "running" for task in batch["tasks"]):
            break
        time.sleep(0.01)
    # The queued folders never started, so they have no job and no output.
    assert batch["tasks"][1]["job_id"] is None
    assert batch["tasks"][2]["job_id"] is None
    assert not (output / "chapter-02--已汉化").exists()
    assert not (output / "chapter-03--已汉化").exists()


def test_pause_and_resume_reach_the_running_folder(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=SlowPipeline))
    folders = [
        make_folder(tmp_path, "chapter-01", pages=4),
        make_folder(tmp_path, "chapter-02", pages=4),
    ]
    batch_id = start_batch(client, folders).json()["id"]
    wait_for_running_task(client, batch_id)

    paused = client.post(f"/api/jobs/{batch_id}/pause").json()
    assert paused["message"] == "已暂停"
    assert paused["tasks"][0]["status"] == "paused"

    resumed = client.post(f"/api/jobs/{batch_id}/resume").json()
    assert resumed["message"] == "继续运行"

    batch = wait_for_batch(client, batch_id)
    assert batch["status"] == "completed"
    assert batch["completed"] == 2


def test_the_output_root_is_optional_and_defaults_beside_the_source(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    folder = make_folder(tmp_path, "chapter-01")

    batch = start_batch(client, [folder], "").json()
    produced = Path(batch["tasks"][0]["output"])
    assert produced.name == "chapter-01--已汉化"
    # An empty root is remembered as empty, so a reloaded page can put the same
    # folders and the same (absent) root back into the form.
    assert batch["request"]["source"] == [str(folder.resolve())]
    assert batch["request"]["output"] == ""

    finished = wait_for_batch(client, batch["id"])
    assert (Path(finished["tasks"][0]["output"]) / "1.png").exists()


def test_a_batch_remembers_the_output_root_for_a_reloaded_page(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    first = make_folder(tmp_path, "chapter-01")
    second = make_folder(tmp_path, "chapter-02")
    root = tmp_path / "out"

    batch = start_batch(client, [first, second], root).json()
    assert batch["request"]["source"] == [str(first.resolve()), str(second.resolve())]
    assert batch["request"]["output"] == str(root)
    assert [task["output"] for task in batch["tasks"]] == [
        str(root / "chapter-01--已汉化"),
        str(root / "chapter-02--已汉化"),
    ]

    # The listing a refreshed page reads carries the same request.
    reloaded = client.get("/api/batches").json()["batches"]
    assert reloaded[0]["id"] == batch["id"]
    assert reloaded[0]["request"]["output"] == str(root)

    # Drain the batch: a consumer thread that outlives its test would release
    # models through whatever monkeypatch the next test installs.
    finished = wait_for_batch(client, batch["id"])
    assert finished["status"] == "completed"


def test_a_finished_batch_releases_the_ollama_models(tmp_path, monkeypatch):
    released = []
    monkeypatch.setattr(
        "manga_localizer.batches.release_ollama_model",
        lambda base_url, model: released.append(model) or True,
    )
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    folder = make_folder(tmp_path, "chapter-01")

    response = client.post(
        "/api/batches",
        json={
            "source": [str(folder)],
            "output": "",
            "inference_backend": "ollama",
            "ocr_backend": "hybrid",
            "ollama_model": "model-a",
            "ollama_ocr_model": "model-b",
        },
    )

    batch = wait_for_batch(client, response.json()["id"])
    assert released == ["model-a", "model-b"]
    assert batch["released"] == ["model-a", "model-b"]


def test_a_single_string_source_still_creates_one_job(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    folder = make_folder(tmp_path, "chapter-01")

    response = client.post(
        "/api/jobs", json={"source": str(folder), "output": str(tmp_path / "out")}
    )

    assert response.status_code == 200
    job = response.json()
    assert "tasks" not in job
    assert job["request"]["source"] == str(folder.resolve())


def test_subfolders_are_listed_with_their_image_counts(tmp_path):
    client = TestClient(create_app(make_paths(tmp_path), pipeline_factory=FakePipeline))
    parent = tmp_path / "volumes"
    (parent / "vol-01").mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(parent / "vol-01" / "1.png")
    (parent / "vol-02").mkdir()

    response = client.get("/api/paths/children", params={"path": str(parent)})
    folders = response.json()["folders"]
    assert [item["name"] for item in folders] == ["vol-01", "vol-02"]
    assert [item["images"] for item in folders] == [1, 0]
    assert client.get(
        "/api/paths/children", params={"path": str(tmp_path / "missing")}
    ).status_code == 422


def test_a_batch_left_live_by_a_previous_process_is_marked_interrupted(tmp_path):
    paths = make_paths(tmp_path)
    paths.ensure()
    store = BatchStore(paths.batches)
    batch = store.create(
        [make_task(0, tmp_path / "chapter-01", tmp_path / "chapter-01--已汉化", 3)]
    )
    batch["status"] = "running"
    batch["tasks"][0]["status"] = "running"
    store.save(batch)

    client = TestClient(create_app(paths, pipeline_factory=FakePipeline))
    recovered = client.get(f"/api/batches/{batch['id']}").json()

    assert recovered["status"] == "cancelled"
    assert recovered["tasks"][0]["status"] == "cancelled"
    assert "服务重启" in recovered["message"]
