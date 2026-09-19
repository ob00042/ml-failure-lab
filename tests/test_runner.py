import json
import subprocess
import sys

import pytest

from failure_lab import runner

EXPECTED_CASES = {"unstable-loss", "exploding-gradients", "broadcasting-bug", "precision-failure"}


def test_cli_lists_every_case():
    result = subprocess.run(
        [sys.executable, "-m", "failure_lab.runner", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(result.stdout.splitlines()) == EXPECTED_CASES


def test_cli_runs_all_and_saves_strict_json(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "failure_lab.runner", "run", "all", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert {p.stem for p in tmp_path.glob("*.json")} == EXPECTED_CASES
    for path in tmp_path.glob("*.json"):
        report = json.loads(path.read_text(), parse_constant=reject_nonstandard_number)
        assert report["environment"]["device"] == "cpu"
        assert all(report["verification"].values())
        assert all(
            report[key]
            for key in (
                "symptom",
                "hypothesis",
                "evidence",
                "root_cause",
                "repair",
                "broken",
                "repaired",
            )
        )


def reject_nonstandard_number(value):
    raise AssertionError(f"Nonstandard JSON number: {value}")


def test_failed_verification_is_not_silent(tmp_path, monkeypatch):
    class FailingCase:
        @staticmethod
        def run():
            return {
                "broken": {},
                "repaired": {},
                "root_cause": "test",
                "repair": "test",
                "verification": {"finite_repair": False},
            }

    monkeypatch.setattr(runner, "import_module", lambda _: FailingCase)
    with pytest.raises(RuntimeError, match="verification failed"):
        runner.run_case("unstable-loss", tmp_path)
    assert (tmp_path / "unstable-loss.json").exists()


def test_unknown_case_exits_nonzero():
    result = subprocess.run(
        [sys.executable, "-m", "failure_lab.runner", "run", "unknown"], capture_output=True
    )
    assert result.returncode != 0
