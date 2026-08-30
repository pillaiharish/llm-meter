from __future__ import annotations

import subprocess
import sys

import llm_meter
from llm_meter.cli import main as cli_main


def test_version_attribute() -> None:
    assert isinstance(llm_meter.__version__, str)
    assert llm_meter.__version__.startswith("0.")


def test_cli_version(capsys: object) -> None:
    try:
        cli_main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "llm-meter" in captured.out


def test_cli_help(capsys: object) -> None:
    try:
        cli_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "llm-meter" in captured.out


def test_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "llm_meter.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "llm-meter" in result.stdout
