# test/test_style.py
import subprocess


def test_flake8_compliance():
    """Run flake8 to check for coding standards and style violations."""
    result = subprocess.run(
        ["flake8", "src"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert (
        result.returncode == 0
    ), f"flake8 found violations:\n{result.stdout.decode()}"
