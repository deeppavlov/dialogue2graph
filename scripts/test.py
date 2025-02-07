import pytest


def _test() -> int:
    """
    Run framework tests, located in `tests/` dir.
    """
    args = ["--tb=long", "-vv", "--cache-clear", "tests/"]
    return pytest.main(args)
