import pytest


def _test() -> int:
    """
    Run framework tests, located in `tests/` dir.
    """
    args = ["--tb=long", "-vv", "--cache-clear", "tests/"]
    return pytest.main(args)


def quick_test():
    exit(_test())


def coverage():
    """
    Run pytest with coverage options and generate coverage reports.

    This function constructs a list of arguments for pytest to measure code
    coverage, enforce a minimum coverage threshold, and generate both HTML
    and terminal coverage reports. It then executes pytest with the specified
    arguments.

    Returns:
        int: The exit code returned by pytest. A non-zero value indicates
             that tests failed or the coverage threshold was not met.
    """
    test_coverage_threshold = 80

    args = [
        "tests/",
        "--cov=dialogue2graph",
        f"--cov-fail-under={test_coverage_threshold}",
        "--cov-report=html",
        "--cov-report=term",
        "--tb=long",
        "-vv",
    ]
    return pytest.main(args)
