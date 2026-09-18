"""Use the sentence_coach interpreter when this process has no openai package.

Running `python evals/...` from conda base hits LanguageModel with
ImportError, which the harness used to report as 'could not reach vLLM'.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_eval_python() -> None:
    try:
        import openai  # noqa: F401
        return
    except ImportError:
        pass
    current = Path(sys.executable).resolve()
    roots = [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path(os.environ.get("CONDA_EXE", "")).resolve().parent.parent if os.environ.get("CONDA_EXE") else None,
        Path("/opt/conda"),
    ]
    for root in roots:
        if not root:
            continue
        py = root / "envs" / "sentence_coach" / "bin" / "python"
        if py.exists() and py.resolve() != current:
            os.execv(str(py), [str(py), *sys.argv])
    sys.exit(
        "This Python is missing the openai package, so it cannot talk to vLLM.\n"
        "You are likely in conda base. Activate the app env and retry:\n"
        "  conda activate sentence_coach\n"
        "  python evals/games/run_eval.py"
    )
