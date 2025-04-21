import subprocess


def _check():
    """Run ruff to check code style."""
    subprocess.run(["ruff", "check", "--fix", "."], check=True)


def _format():
    """Run ruff to format code."""
    subprocess.run(["ruff", "format", "."], check=True)
