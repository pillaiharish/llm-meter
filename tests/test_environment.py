from __future__ import annotations

import subprocess
from importlib import metadata

from llm_meter.environment import (
    ENVIRONMENT_PROVENANCE_VERSION,
    NVIDIA_SMI_TIMEOUT_SECONDS,
    collect_environment,
    collect_nvidia_provenance,
    collect_package_versions,
    collect_python_provenance,
    collect_system_provenance,
)


def test_system_provenance_uses_non_identifying_platform_fields(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.platform.system", lambda: "TestOS")
    monkeypatch.setattr("llm_meter.environment.platform.release", lambda: "1.2")
    monkeypatch.setattr("llm_meter.environment.platform.machine", lambda: "test64")
    monkeypatch.setattr("llm_meter.environment.os.cpu_count", lambda: None)

    system = collect_system_provenance()

    assert system.source == "python_platform"
    assert system.os_name == "TestOS"
    assert system.os_release == "1.2"
    assert system.architecture == "test64"
    assert system.logical_cpu_count is None
    assert set(vars(system)) == {
        "source",
        "os_name",
        "os_release",
        "architecture",
        "logical_cpu_count",
    }


def test_python_runtime_provenance(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.platform.python_version", lambda: "3.12.7")
    monkeypatch.setattr("llm_meter.environment.platform.python_implementation", lambda: "CPython")

    runtime = collect_python_provenance()

    assert runtime.source == "python_runtime"
    assert runtime.version == "3.12.7"
    assert runtime.implementation == "CPython"


def test_package_versions_are_limited_and_missing_is_none(monkeypatch) -> None:
    def version(name: str) -> str:
        if name == "httpx":
            return "0.28.1"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr("llm_meter.environment.metadata.version", version)

    packages = collect_package_versions()

    assert packages.httpx == "0.28.1"
    assert packages.tokenizers is None
    assert set(vars(packages)) == {"httpx", "tokenizers"}


def test_nvidia_smi_absent_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.shutil.which", lambda _: None)

    result = collect_nvidia_provenance()

    assert result.status == "unavailable"
    assert result.source == "nvidia_smi"
    assert result.reason == "nvidia_smi_not_found"
    assert result.devices == []


def test_nvidia_smi_timeout_is_error_without_stderr(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.shutil.which", lambda _: "/bin/nvidia-smi")

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], stderr="private path")

    monkeypatch.setattr("llm_meter.environment.subprocess.run", run)

    result = collect_nvidia_provenance()

    assert result.status == "error"
    assert result.reason == "nvidia_smi_timeout"
    assert "private path" not in repr(result)


def test_nvidia_smi_nonzero_is_error(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.shutil.which", lambda _: "/bin/nvidia-smi")
    monkeypatch.setattr(
        "llm_meter.environment.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "private"),
    )

    result = collect_nvidia_provenance()

    assert result.status == "error"
    assert result.reason == "nvidia_smi_failed"
    assert "private" not in repr(result)


def test_nvidia_smi_malformed_rows_are_errors(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.shutil.which", lambda _: "/bin/nvidia-smi")

    for output in ("", "0, GPU", "bad, GPU, uuid, 80, 550", "0, GPU, uuid, bad, 550"):
        monkeypatch.setattr(
            "llm_meter.environment.subprocess.run",
            lambda *args, output=output, **kwargs: subprocess.CompletedProcess(
                args[0], 0, output, ""
            ),
        )
        result = collect_nvidia_provenance()
        assert result.status == "error"
        assert result.reason == "nvidia_smi_parse_error"
        assert result.devices == []


def test_nvidia_smi_parses_quoted_multi_gpu_csv_in_index_order(monkeypatch) -> None:
    calls = []
    output = (
        '3,"NVIDIA GPU, Special",GPU-b,81559,550.54\n'
        "1,NVIDIA H100, GPU-a, 40960 ,550.54\n"
    )
    monkeypatch.setattr("llm_meter.environment.shutil.which", lambda _: "/bin/nvidia-smi")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, output, "ignored")

    monkeypatch.setattr("llm_meter.environment.subprocess.run", run)

    result = collect_nvidia_provenance()

    assert result.status == "available"
    assert result.reason is None
    assert [device.index for device in result.devices] == [1, 3]
    assert result.devices[0].memory_total_mib == 40960
    assert result.devices[0].uuid == "GPU-a"
    assert result.devices[0].driver_version == "550.54"
    assert result.devices[1].name == "NVIDIA GPU, Special"
    command, options = calls[0]
    assert command == [
        "/bin/nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    assert options["shell"] is False
    assert options["timeout"] == NVIDIA_SMI_TIMEOUT_SECONDS
    assert "CUDA" not in " ".join(command)


def test_nvidia_smi_parses_one_gpu_with_optional_fields(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.shutil.which", lambda _: "/bin/nvidia-smi")
    monkeypatch.setattr(
        "llm_meter.environment.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "7,NVIDIA Test GPU,N/A,N/A,N/A\n", ""
        ),
    )

    result = collect_nvidia_provenance()

    assert result.status == "available"
    assert len(result.devices) == 1
    assert result.devices[0].index == 7
    assert result.devices[0].uuid is None
    assert result.devices[0].memory_total_mib is None
    assert result.devices[0].driver_version is None


def test_collect_environment_is_deterministic_with_mocked_sources(monkeypatch) -> None:
    monkeypatch.setattr("llm_meter.environment.datetime", _FixedDatetime)
    monkeypatch.setattr("llm_meter.environment.platform.system", lambda: "TestOS")
    monkeypatch.setattr("llm_meter.environment.platform.release", lambda: "1")
    monkeypatch.setattr("llm_meter.environment.platform.machine", lambda: "test64")
    monkeypatch.setattr("llm_meter.environment.os.cpu_count", lambda: 8)
    monkeypatch.setattr("llm_meter.environment.platform.python_version", lambda: "3.12")
    monkeypatch.setattr("llm_meter.environment.platform.python_implementation", lambda: "CPython")
    monkeypatch.setattr("llm_meter.environment.metadata.version", lambda name: f"{name}-version")
    monkeypatch.setattr("llm_meter.environment.shutil.which", lambda _: None)

    first = collect_environment()
    second = collect_environment()

    assert first == second
    assert first.version == ENVIRONMENT_PROVENANCE_VERSION
    assert first.captured_at_utc == "2025-01-01T00:00:00+00:00"


class _FixedDatetime:
    @classmethod
    def now(cls, timezone):
        from datetime import datetime

        return datetime.fromisoformat("2025-01-01T00:00:00+00:00")
