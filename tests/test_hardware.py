from manga_localizer.hardware import (
    CpuInfo,
    apply_cpu_threads,
    detect_cpu,
    normalize_cpu_threads,
)


def test_detect_cpu_reports_a_usable_machine():
    cpu = detect_cpu()
    assert cpu.name
    assert cpu.logical_cores >= 1
    # Windows always reports at least as many logical processors as cores, and
    # a failed probe must not invent more cores than threads.
    assert 1 <= cpu.physical_cores <= cpu.logical_cores


def test_thread_choices_list_every_count_down_to_one():
    cpu = CpuInfo(name="Test CPU", physical_cores=4, logical_cores=8)
    assert cpu.choices == [8, 7, 6, 5, 4, 3, 2, 1]


def test_thread_choices_keep_every_step_on_large_machines():
    # A coarse ladder would deny "use 10 of 16" and "pin to 3 of 12", so every
    # value in the range is offered even though the dropdown gets long.
    cpu = CpuInfo(name="Test CPU", physical_cores=16, logical_cores=32)
    assert cpu.choices == list(range(32, 0, -1))
    assert 31 in cpu.choices and 29 in cpu.choices and 3 in cpu.choices


def test_zero_means_library_defaults():
    cpu = CpuInfo(name="Test CPU", physical_cores=4, logical_cores=8)
    assert normalize_cpu_threads(0, cpu) == 0
    assert normalize_cpu_threads(None, cpu) == 0
    assert normalize_cpu_threads("", cpu) == 0


def test_requested_threads_are_clamped_to_the_machine():
    cpu = CpuInfo(name="Test CPU", physical_cores=4, logical_cores=8)
    assert normalize_cpu_threads(3, cpu) == 3
    assert normalize_cpu_threads(8, cpu) == 8
    # A settings file copied from a bigger machine must not oversubscribe this
    # one, and a negative number is treated as "no limit" rather than an error.
    assert normalize_cpu_threads(32, cpu) == 8
    assert normalize_cpu_threads(-4, cpu) == 0
    assert normalize_cpu_threads("not a number", cpu) == 0


def test_apply_cpu_threads_returns_the_effective_count():
    cpu = CpuInfo(name="Test CPU", physical_cores=4, logical_cores=8)
    assert apply_cpu_threads(3, cpu) == 3
    assert apply_cpu_threads(0, cpu) == 0


def test_apply_cpu_threads_survives_a_missing_runtime(monkeypatch):
    # The CLI imports this module before torch or paddle exist; probing for them
    # must never raise just because this machine has no ML runtime installed.
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith(("torch", "paddle")):
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    cpu = CpuInfo(name="Test CPU", physical_cores=2, logical_cores=4)
    assert apply_cpu_threads(2, cpu) == 2


def test_pipeline_request_carries_the_thread_budget():
    from pathlib import Path

    from manga_localizer.config import UserSettings
    from manga_localizer.pipeline import request_from_settings

    settings = UserSettings(cpu_threads=3)
    request = request_from_settings(Path("src"), Path("out"), settings)
    assert request.cpu_threads == 3
