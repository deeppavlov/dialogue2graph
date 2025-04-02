import pathlib
from typing import List

import black
from flake8.main.cli import main as flake_main


_STANDARD_PATHS = ["dialogue2graph", "scripts", "tests", ".github"]
_STANDARD_PATHS_LEN = 150


def _get_paths(paths: List[str]) -> List[pathlib.Path]:
    return [path for dir in paths for path in pathlib.Path(dir).glob("**/*.py")]


def _run_flake():
    lint_result = 0
    flake8_configs = [
        "--select=E,W,F",
        # black formats binary operators after line breaks
        "--ignore=E501,E203,W503,W293,F824",
        "--per-file-ignores="
        # allow imports in init files without use
        "**/__init__.py:F401 ",
    ]
    lint_result += flake_main([f"--max-line-length={_STANDARD_PATHS_LEN}"] + flake8_configs + _STANDARD_PATHS)

    exit(lint_result)


def _run_black(modify: bool):
    report = black.Report(check=not modify, quiet=False)
    write = black.WriteBack.YES if modify else black.WriteBack.CHECK
    for path in _get_paths(_STANDARD_PATHS):
        mode = black.Mode(line_length=_STANDARD_PATHS_LEN)
        black.reformat_one(path, False, write, mode, report)
    exit(report.return_code)
