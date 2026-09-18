#!/usr/bin/env python3
"""Real evaluation harness for the game-answer judge.

Loads evals/games/dataset.json (100 gold-labeled examples) and calls the
ACTUAL deployed judge model (agent.py's LanguageModel, pointed at whatever
vLLM/OpenAI-compatible endpoint config.json specifies) with a 4-way judge
prompt built for this eval. This produces real model_prediction values,
not simulated/fabricated ones — the whole point is to measure whether the
system you actually run is accurate, not to demo a metrics formula.

Usage:
    python evals/games/run_eval.py

Writes:
    evals/games/results.json  (dataset + model_prediction/prediction_reason)
    evals/games/report.md     (accuracy, macro F1, confusion matrix, per-game
                                accuracy, error analysis)
"""
from __future__ import annotations

import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

EVALS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALS))
from bootstrap import ensure_eval_python  # noqa: E402

ensure_eval_python()

ROOT = EVALS.parent
sys.path.insert(0, str(ROOT))

from agent import LanguageModel, parse_llm_json  # noqa: E402

LABELS = ["correct", "partially_correct", "incorrect", "no_response"]

SYSTEM_EVAL_JUDGE = textwrap.dedent("""\
    You are an expert evaluator for a children's (ages 5-8) audio English-learning app.
    You will judge a child's spoken response against a learning prompt and a target answer.
    Do not require an exact string match — judge whether the child demonstrates the
    intended learning objective. Longer answers, full sentences, valid alternative
    answers, and minor grammar mistakes that don't change the meaning are all correct.

    Use exactly one of these four labels:
    - correct: demonstrates the intended learning objective.
    - partially_correct: shows some understanding but is incomplete, vague, or
      has the right idea without the target word/concept.
    - incorrect: unrelated, contradicts the prompt/story, or shows no understanding.
    - no_response: the response is empty, "silence", "inaudible", or "[no response]".

    JSON only: {"model_prediction": "correct|partially_correct|incorrect|no_response", "prediction_reason": "..."}
""")


def build_user_payload(example: dict) -> str:
    if example["game"] == "complete_the_sentence":
        payload = {
            "prompt": example["prompt"],
            "target_answer": example["target_answer"],
            "child_response": example["child_response"],
        }
    else:
        payload = {
            "story": example["story"],
            "question": example["question"],
            "target_answer": example["target_answer"],
            "child_response": example["child_response"],
        }
    return json.dumps(payload)


def judge_one(llm: LanguageModel, example: dict) -> dict:
    if not str(example.get("child_response") or "").strip() or example["child_response"].lower() in {
        "silence", "inaudible", "[no response]",
    }:
        # No need to call the LLM for the clear-cut no-response marker cases —
        # mirrors how the real app's is_garbled/empty-transcript checks work
        # before anything reaches an LLM judge call.
        return {"model_prediction": "no_response", "prediction_reason": "Empty or explicit no-response marker."}
    try:
        raw = llm.complete(SYSTEM_EVAL_JUDGE, build_user_payload(example), max_tokens=120, temperature=0.1)
        parsed = parse_llm_json(raw)
        label = str(parsed.get("model_prediction") or "").lower().strip()
        if label not in LABELS:
            label = "incorrect"
        reason = str(parsed.get("prediction_reason") or "").strip() or "(no reason given)"
        return {"model_prediction": label, "prediction_reason": reason}
    except Exception as exc:  # noqa: BLE001
        return {"model_prediction": "incorrect", "prediction_reason": f"Judge call failed: {exc}"}


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct_preds = sum(1 for r in results if r["model_prediction"] == r["expected_label"])
    accuracy = correct_preds / total if total else 0.0

    per_label = {}
    for label in LABELS:
        tp = sum(1 for r in results if r["model_prediction"] == label and r["expected_label"] == label)
        fp = sum(1 for r in results if r["model_prediction"] == label and r["expected_label"] != label)
        fn = sum(1 for r in results if r["model_prediction"] != label and r["expected_label"] == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(LABELS)

    confusion = {a: {p: 0 for p in LABELS} for a in LABELS}
    for r in results:
        confusion[r["expected_label"]][r["model_prediction"]] += 1

    by_game = {}
    for game in ("complete_the_sentence", "story_challenge"):
        rows = [r for r in results if r["game"] == game]
        by_game[game] = sum(1 for r in rows if r["model_prediction"] == r["expected_label"]) / len(rows) if rows else 0.0

    return {
        "total": total,
        "correct_predictions": correct_preds,
        "accuracy": accuracy,
        "per_label": per_label,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "accuracy_by_game": by_game,
    }


def render_report(metrics: dict, results: list[dict]) -> str:
    lines = ["# Game Judge Evaluation Report", ""]
    lines.append("## Overall accuracy")
    lines.append(f"- Correct predictions: {metrics['correct_predictions']} / {metrics['total']}")
    lines.append(f"- Accuracy: {metrics['accuracy'] * 100:.1f}%")
    lines.append("")
    lines.append("## Macro F1 score")
    lines.append("| Label | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in LABELS:
        m = metrics["per_label"][label]
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines.append(f"\n**Macro F1: {metrics['macro_f1']:.3f}**")
    lines.append("")
    lines.append("## Confusion matrix (rows = expected, columns = predicted)")
    header = "| Expected \\ Predicted | " + " | ".join(LABELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(LABELS) + 1))
    for actual in LABELS:
        row = metrics["confusion_matrix"][actual]
        lines.append(f"| {actual} | " + " | ".join(str(row[p]) for p in LABELS) + " |")
    lines.append("")
    lines.append("## Accuracy by game")
    for game, acc in metrics["accuracy_by_game"].items():
        lines.append(f"- {game}: {acc * 100:.1f}%")
    lines.append("")
    lines.append("## Error analysis")
    errors = [r for r in results if r["model_prediction"] != r["expected_label"]]
    lines.append(f"{len(errors)} of {metrics['total']} examples were misclassified.")
    error_pairs = Counter((r["expected_label"], r["model_prediction"]) for r in errors)
    for (expected, predicted), count in error_pairs.most_common(5):
        lines.append(f"- {count}x expected `{expected}`, predicted `{predicted}`")
    lines.append("")
    lines.append("Sample misclassifications:")
    for r in errors[:8]:
        resp = r["child_response"] or "(empty)"
        lines.append(
            f"- [{r['id']}] \"{resp}\" — expected `{r['expected_label']}`, "
            f"got `{r['model_prediction']}` ({r['prediction_reason']})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    dataset_path = Path(__file__).parent / "dataset.json"
    config_path = ROOT / "config.json"
    dataset = json.loads(dataset_path.read_text())
    config = json.loads(config_path.read_text())
    llm_cfg = config.get("llm", {})
    llm = LanguageModel(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        model=llm_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
        api_key=llm_cfg.get("api_key", "local"),
        timeout=float(llm_cfg.get("timeout", 10.0)),
    )

    try:
        probe = llm.complete("Reply with JSON only.", '{"ping": true}', max_tokens=10, temperature=0.0)
        if not probe:
            raise RuntimeError("empty response")
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: could not reach the judge model at {llm.base_url} ({exc}).\n"
            "Refusing to run — a report generated while the model is unreachable would just show every\n"
            "call failing as 'incorrect', which is not a real measurement. Start vLLM (scripts/1_vllm.sh)\n"
            "and try again."
        )
        sys.exit(1)

    results = []
    for i, example in enumerate(dataset, 1):
        prediction = judge_one(llm, example)
        result = {**example, **prediction}
        results.append(result)
        print(f"[{i}/{len(dataset)}] {example['id']}: expected={example['expected_label']} predicted={prediction['model_prediction']}")

    metrics = compute_metrics(results)
    out_dir = Path(__file__).parent
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = render_report(metrics, results)
    (out_dir / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
