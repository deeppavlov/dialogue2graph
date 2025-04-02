import subprocess

def _check():
    """Run ruff to check code style."""
    subprocess.run(["ruff", "check", "./dialogue2graph"], check=True)

def _format():
    """Run ruff to format code."""
    subprocess.run(["ruff", "format", "./dialogue2graph"], check=True)