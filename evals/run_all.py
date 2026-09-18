#!/usr/bin/env python3
"""Run safety + games evals and collect metrics.json from each folder.

Does not change how either suite scores examples. Just launches the existing
harnesses and writes a combined snapshot at evals/summary.json.

Usage (from repo root, vLLM already up):
    python evals/run_all.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bootstrap import ensure_eval_python  # noqa: E402

ensure_eval_python()


def run_suite(name: str) -> dict:
    script = ROOT / name / "run_eval.py"
    print(f"\n=== {name} ===\n", flush=True)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT.parent))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    metrics_path = ROOT / name / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {
        "suite": name,
        "metrics_path": str(metrics_path.relative_to(ROOT.parent)),
        "results_path": str((ROOT / name / "results.json").relative_to(ROOT.parent)),
        "report_path": str((ROOT / name / "report.md").relative_to(ROOT.parent)),
        "metrics": metrics,
    }


def main() -> None:
    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "suites": [run_suite("safety"), run_suite("games")],
    }
    out = ROOT / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
