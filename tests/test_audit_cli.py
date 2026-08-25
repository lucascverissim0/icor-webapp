import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_audit_cli_is_read_only_and_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    before = (ROOT / "data" / "passenger_car_data.xlsx").read_bytes()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_baseline.py"),
        "--root",
        str(ROOT),
        "--output",
        str(output),
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    first_bytes = output.read_bytes() if output.exists() else b""
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == second.returncode == 0
    assert first_bytes == output.read_bytes()
    assert json.loads(first_bytes)["file_count"] == 22
    assert (ROOT / "data" / "passenger_car_data.xlsx").read_bytes() == before
