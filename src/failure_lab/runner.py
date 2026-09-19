"""Run reproducible CPU experiments and write strict JSON reports."""

import argparse
import importlib
import json
from pathlib import Path

from failure_lab.diagnostics import environment, seed_everything

CASES = {
    "unstable-loss": "unstable_loss",
    "exploding-gradients": "exploding_gradients",
    "broadcasting-bug": "broadcasting_bug",
    "precision-failure": "precision_failure",
}


def run_case(name: str, output: Path) -> dict:
    seed_everything()
    result = importlib.import_module(f"failure_lab.cases.{CASES[name]}").run()
    report = {"schema_version": 1, "case": name, "environment": environment(), **result}
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\ncase: {name}\nbroken: {result['broken']}\n")
    print(f"diagnosis: {result['root_cause']}\nrepair: {result['repair']}")
    print(f"repaired: {result['repaired']}\nverification: {result['verification']}\nsaved: {path}")
    if not all(result["verification"].values()):
        raise RuntimeError(f"{name}: verification failed; inspect {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("case", choices=[*CASES, "all"])
    run_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(CASES))
        return
    for name in CASES if args.case == "all" else [args.case]:
        run_case(name, args.output_dir)


if __name__ == "__main__":
    main()
