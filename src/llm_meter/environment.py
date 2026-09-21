from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib import metadata

from llm_meter.models import (
    CollectionStatus,
    EnvironmentProvenance,
    NvidiaDeviceProvenance,
    NvidiaProvenance,
    PackageVersions,
    PythonRuntimeProvenance,
    SystemProvenance,
)

ENVIRONMENT_PROVENANCE_VERSION = "1"
NVIDIA_SMI_TIMEOUT_SECONDS = 3
NVIDIA_SMI_SOURCE = "nvidia_smi"


def collect_system_provenance() -> SystemProvenance:
    return SystemProvenance(
        source="python_platform",
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine(),
        logical_cpu_count=os.cpu_count(),
    )


def collect_python_provenance() -> PythonRuntimeProvenance:
    return PythonRuntimeProvenance(
        source="python_runtime",
        version=platform.python_version(),
        implementation=platform.python_implementation(),
    )


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_package_versions() -> PackageVersions:
    return PackageVersions(
        httpx=_package_version("httpx"),
        tokenizers=_package_version("tokenizers"),
    )


def _nvidia_result(status: CollectionStatus, reason: str | None) -> NvidiaProvenance:
    return NvidiaProvenance(
        status=status.value,
        source=NVIDIA_SMI_SOURCE,
        reason=reason,
        devices=[],
    )


def _optional_text(value: str) -> str | None:
    value = value.strip()
    return None if not value or value.casefold() == "n/a" else value


def _optional_int(value: str) -> int | None:
    text = _optional_text(value)
    return None if text is None else int(text)


def _parse_nvidia_devices(output: str) -> list[NvidiaDeviceProvenance]:
    devices: list[NvidiaDeviceProvenance] = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 5:
            raise ValueError("unexpected nvidia-smi field count")
        index_text, name, uuid, memory_total_mib, driver_version = row
        if not name.strip():
            raise ValueError("missing NVIDIA device name")
        devices.append(
            NvidiaDeviceProvenance(
                index=int(index_text.strip()),
                name=name.strip(),
                uuid=_optional_text(uuid),
                memory_total_mib=_optional_int(memory_total_mib),
                driver_version=_optional_text(driver_version),
            )
        )
    if not devices:
        raise ValueError("nvidia-smi returned no devices")
    return sorted(devices, key=lambda device: device.index)


def collect_nvidia_provenance() -> NvidiaProvenance:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return _nvidia_result(
            CollectionStatus.UNAVAILABLE,
            "nvidia_smi_not_found",
        )

    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            shell=False,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _nvidia_result(CollectionStatus.ERROR, "nvidia_smi_timeout")
    except OSError:
        return _nvidia_result(CollectionStatus.ERROR, "nvidia_smi_failed")

    if completed.returncode != 0:
        return _nvidia_result(CollectionStatus.ERROR, "nvidia_smi_failed")

    try:
        devices = _parse_nvidia_devices(completed.stdout)
    except (ValueError, csv.Error):
        return _nvidia_result(CollectionStatus.ERROR, "nvidia_smi_parse_error")

    return NvidiaProvenance(
        status=CollectionStatus.AVAILABLE.value,
        source=NVIDIA_SMI_SOURCE,
        reason=None,
        devices=devices,
    )


def collect_environment() -> EnvironmentProvenance:
    return EnvironmentProvenance(
        version=ENVIRONMENT_PROVENANCE_VERSION,
        captured_at_utc=datetime.now(UTC).isoformat(),
        system=collect_system_provenance(),
        python=collect_python_provenance(),
        packages=collect_package_versions(),
        nvidia=collect_nvidia_provenance(),
    )
