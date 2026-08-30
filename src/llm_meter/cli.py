from __future__ import annotations

import argparse
import sys

from llm_meter import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-meter",
        description=(
            "Transparent, reproducible metrology for LLM inference. "
            "Under active development; V1 is being designed."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"llm-meter {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())