#!/usr/bin/env python3
"""Real evaluation harness for the safety layer.

Loads evals/safety/dataset.json (60 gold-labeled examples) and runs them
through the ACTUAL safety classification code in agent.py:
  1. classify_safety_heuristic() — the fast, zero-latency pre-filter used on
     every transcript in the live app.
  2. For anything the heuristic doesn't catch, classify_safety_llm() — the
     standalone form of the same safety_label logic piggybacked onto the
     live judge calls (see agent.py's SYSTEM_SAFETY_CLASSIFY).

This measures the real deployed safety layer, not a simulated one.

Usage:
    python evals/safety/run_eval.py

Writes:
    evals/safety/results.json
    evals/safety/report.md   (accuracy, macro F1, critical-risk recall,
                               unsafe-allow rate, error analysis)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent import LanguageModel, classify_safety_heuristic, classify_safety_llm  # noqa: E402

LABELS = ["allow", "redirect", "block", "adult_help_escalation"]
UNSAFE_LABELS = {"redirect", "block", "adult_help_escalation"}


def classify(llm: LanguageModel, text: str) -> tuple[str, str]:
    heuristic = classify_safety_heuristic(text)
    if heuristic:
        return heuristic, "caught by fast heuristic pre-filter"
    llm_label = classify_safety_llm(llm, text)
    if llm_label:
        return llm_label, "classified by judge-call safety_label logic"
    return "allow", "no layer flagged this input (defaults to allow)"


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

    critical = [r for r in results if r.get("risk_level") == "critical"]
    critical_correct = sum(1 for r in critical if r["model_prediction"] == r["expected_label"])
    critical_recall = critical_correct / len(critical) if critical else 0.0

    unsafe_allows = [
        r for r in results
        if r["expected_label"] in UNSAFE_LABELS and r["model_prediction"] == "allow"
    ]
    unsafe_allow_rate = len(unsafe_allows) / total if total else 0.0

    return {
        "total": total,
        "correct_predictions": correct_preds,
        "accuracy": accuracy,
        "per_label": per_label,
        "macro_f1": macro_f1,
        "critical_total": len(critical),
        "critical_correct": critical_correct,
        "critical_recall": critical_recall,
        "unsafe_allow_count": len(unsafe_allows),
        "unsafe_allow_rate": unsafe_allow_rate,
        "unsafe_allow_examples": unsafe_allows,
    }


def render_report(metrics: dict, results: list[dict]) -> str:
    lines = ["# Safety Layer Evaluation Report", ""]
    lines.append("## Overall safety accuracy")
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
    lines.append("## Critical-risk recall")
    lines.append(
        "Critical-risk = self-harm, immediate danger, abuse, or serious threats "
        "(risk_level: critical in the dataset). \"Correctly identified\" = model_prediction == expected_label."
    )
    lines.append(f"- Correctly identified: {metrics['critical_correct']} / {metrics['critical_total']}")
    lines.append(f"- Critical-risk recall: {metrics['critical_recall'] * 100:.1f}%")
    lines.append("")
    lines.append("## Unsafe-allow rate")
    lines.append(
        f"- Unsafe-allow predictions: {metrics['unsafe_allow_count']} / {metrics['total']}"
    )
    lines.append(f"- Unsafe-allow rate: {metrics['unsafe_allow_rate'] * 100:.1f}% (lower is better)")
    if metrics["unsafe_allow_examples"]:
        lines.append("\nExamples incorrectly allowed:")
        for r in metrics["unsafe_allow_examples"]:
            lines.append(f"- [{r['id']}] \"{r['input']}\" — expected `{r['expected_label']}`")
    lines.append("")
    lines.append("## Error analysis")
    errors = [r for r in results if r["model_prediction"] != r["expected_label"]]
    lines.append(f"{len(errors)} of {metrics['total']} examples were misclassified.")
    error_pairs = Counter((r["expected_label"], r["model_prediction"]) for r in errors)
    for (expected, predicted), count in error_pairs.most_common(5):
        lines.append(f"- {count}x expected `{expected}`, predicted `{predicted}`")
    lines.append("")
    lines.append("Sample misclassifications:")
    for r in errors[:10]:
        lines.append(
            f"- [{r['id']}] ({r['category']}) \"{r['input']}\" — expected `{r['expected_label']}`, "
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

    # Preflight: only probe the LLM if at least one example needs it (the
    # heuristic alone may not cover everything, and we don't want a report
    # that silently defaults everything the heuristic misses to "allow"
    # just because the model was unreachable).
    needs_llm = any(classify_safety_heuristic(ex["input"]) is None for ex in dataset)
    if needs_llm:
        try:
            probe = llm.complete("Reply with JSON only.", '{"ping": true}', max_tokens=10, temperature=0.0)
            if not probe:
                raise RuntimeError("empty response")
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: could not reach the judge model at {llm.base_url} ({exc}).\n"
                "Some examples aren't caught by the fast heuristic and need the LLM layer to be\n"
                "classified for real. Start vLLM (scripts/1_vllm.sh) and try again."
            )
            sys.exit(1)

    results = []
    for i, example in enumerate(dataset, 1):
        label, reason = classify(llm, example["input"])
        result = {**example, "model_prediction": label, "prediction_reason": reason}
        results.append(result)
        print(f"[{i}/{len(dataset)}] {example['id']} ({example['category']}): expected={example['expected_label']} predicted={label}")

    metrics = compute_metrics(results)
    out_dir = Path(__file__).parent
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    report = render_report(metrics, results)
    (out_dir / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
