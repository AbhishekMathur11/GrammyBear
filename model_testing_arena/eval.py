#!/usr/bin/env python3
"""Grammar-coaching model evaluation arena (config-driven)."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STOP = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "at",
    "is", "are", "was", "were", "be", "been", "am", "it", "this", "that", "with",
}

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
WORD_RE = re.compile(r"[a-z0-9']+")
MAJOR_FAIL_SEVERITIES = {"substantive", "major_failure"}
MAJOR_CATEGORIES = {
    "subject_verb_agreement",
    "verb_tense",
    "articles",
    "prepositions",
    "spelling",
    "punctuation",
    "capitalization",
    "pronouns",
    "word_order",
    "sentence_structure",
    "fragments",
    "run_on",
    "awkward_phrasing",
    "natural_english",
    "word_choice",
    "redundant_wording",
    "concise_rewriting",
    "already_correct",
    "multiple_errors",
    "multiple_valid",
    "meaning_preservation",
    "subtle_distinctions",
    "context_dependent",
    "ambiguity",
    "formal_informal",
    "judge_completion",
    "judge_mistake",
    "adversarial",
}


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = data["examples"] if isinstance(data, dict) else data
    if not examples:
        raise SystemExit("empty dataset")
    ids = [e["id"] for e in examples]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate example ids")
    return examples


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(normalize(text))


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOP and len(t) > 1}


def extract_first_json(text: str) -> Any | None:
    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = JSON_RE.search(raw)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return None


def seq_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    from collections import Counter

    cp, cr = Counter(p), Counter(r)
    overlap = sum((cp & cr).values())
    prec = overlap / max(1, sum(cp.values()))
    rec = overlap / max(1, sum(cr.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def best_ref_match(pred: str, refs: list[str]) -> tuple[float, str]:
    best, best_ref = 0.0, refs[0] if refs else ""
    for ref in refs:
        n_pred, n_ref = normalize(pred).rstrip(" .!?"), normalize(ref).rstrip(" .!?")
        if n_pred == n_ref:
            return 1.0, ref
        score = max(seq_ratio(pred, ref), token_f1(pred, ref))
        if n_pred in n_ref or n_ref in n_pred:
            score = max(score, 0.82)
        if score > best:
            best, best_ref = score, ref
    return best, best_ref


def strip_model_output(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.I | re.S)
    s = re.sub(r"</?think>", "", s, flags=re.I)
    s = s.strip()
    s = re.sub(r'^corrected sentence:\s*', "", s, flags=re.I)
    s = re.sub(r'^(answer|output|correction)\s*:\s*', "", s, flags=re.I)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s.strip()


def first_sentence(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    m = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)
    return m[0].strip()


def verbosity_penalty(pred: str, refs: list[str]) -> float:
    pw = max(1, len(tokens(pred)))
    rw = max(1, max(len(tokens(r)) for r in refs) if refs else 8)
    if pw <= max(12, int(2.2 * rw)):
        return 1.0
    extra = pw / max(12, 2.2 * rw)
    return max(0.15, 1.0 / extra)


def lecture_penalty(pred: str) -> float:
    low = pred.lower()
    hits = 0
    for p in (
        "as an ai",
        "i cannot",
        "here is the corrected",
        "the correct sentence is",
        "let me explain",
        "sure,",
        "of course",
        "i'd be happy",
    ):
        if p in low:
            hits += 1
    return max(0.0, 1.0 - 0.25 * hits)


def fluency_score(pred: str) -> float:
    s = pred.strip()
    if not s:
        return 0.0
    score = 0.55
    if s[0].isupper() or s[0] in "\"'({[":
        score += 0.12
    if s[-1] in ".!?\"'":
        score += 0.10
    words = tokens(s)
    if words:
        repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
        score -= min(0.25, 0.08 * repeats)
    if re.search(r"\b(\w+)(\s+\1){2,}\b", s, re.I):
        score -= 0.2
    if len(words) >= 2:
        score += 0.08
    letters = re.sub(r"[^a-z]", "", s.lower())
    if letters and len(set(letters)) <= 2 and len(letters) >= 6:
        score -= 0.4
    return float(min(1.0, max(0.0, score)))


def kid_unsafe(text: str) -> bool:
    low = normalize(text)
    banned = ("kill", "suicide", "porn", "damn", "hell ", "shit", "fuck", "sex")
    return any(b in low for b in banned)


def score_invent(example: dict, raw: str) -> dict:
    pred = strip_model_output(raw)
    obj = extract_first_json(pred)
    instruction = 0.15
    correctness = 0.0
    semantic = 0.4
    agreement = 0.0
    fluency = fluency_score(pred)
    details = {"parsed": bool(obj), "issues": []}
    if obj is None:
        details["issues"].append("not_json")
        instruction = 0.05
    elif example["task"] == "invent_complete":
        instruction = 0.4
        stem = str(obj.get("stem", "")).strip()
        expected = str(obj.get("expected", "")).strip()
        full = str(obj.get("full", "")).strip()
        sw = stem.split()
        if 6 <= len(sw) <= 12:
            correctness += 0.35
        else:
            details["issues"].append("stem_length")
        if expected and expected.lower() not in {"the", "a", "an"}:
            correctness += 0.2
        if full and stem and stem.lower() in full.lower():
            correctness += 0.25
            semantic = 0.8
        if any(b.lower() in stem.lower() for b in example.get("banned_stems") or []):
            correctness *= 0.2
            details["issues"].append("copied_banned_stem")
        if kid_unsafe(stem + " " + full):
            correctness *= 0.2
            details["issues"].append("unsafe")
        instruction = 0.9 if set(k.lower() for k in obj) >= {"stem", "expected", "full"} else 0.45
        agreement = correctness
        fluency = fluency_score(full or stem)
    else:
        spoken = str(obj.get("spoken", "")).strip()
        correct = str(obj.get("correct", "")).strip()
        kind = str(obj.get("kind", "")).strip().lower()
        hint = str(obj.get("hint", "")).strip()
        nw = len(spoken.split())
        if 6 <= nw <= 10:
            correctness += 0.25
        else:
            details["issues"].append("spoken_length")
        if spoken and correct and normalize(spoken) != normalize(correct):
            correctness += 0.25
        else:
            details["issues"].append("no_single_error_diff")
        if kind in {"grammar", "pronunciation"}:
            correctness += 0.2
        if hint:
            correctness += 0.1
        allowed = ("don't", "doesn", "was", "were", "goed", "went", "have", "has", "see", "sea", "free", "three", "sink", "think")
        blob = normalize(spoken + " " + correct)
        if any(a in blob for a in allowed):
            correctness += 0.2
        instruction = 0.9 if set(k.lower() for k in obj) >= {"spoken", "correct", "kind", "hint"} else 0.4
        agreement = correctness
        semantic = 0.75 if spoken and correct else 0.2
        fluency = fluency_score(correct or spoken)
        if kid_unsafe(spoken + " " + correct):
            correctness *= 0.2
    return finalize_scores(example, pred, correctness, semantic, agreement, instruction, fluency, details)


def score_judge(example: dict, raw: str) -> dict:
    pred = strip_model_output(raw)
    obj = extract_first_json(pred)
    expected = bool(example["ground_truth"]["correct"])
    instruction = 0.2
    details = {"parsed": False, "predicted_correct": None, "expected_correct": expected}
    predicted = None
    if isinstance(obj, dict):
        details["parsed"] = True
        predicted = boolish(obj.get("correct"))
        if predicted is None:
            for k, v in obj.items():
                predicted = boolish(v)
                if predicted is not None:
                    break
        details["predicted_correct"] = predicted
        extra_keys = [k for k in obj.keys() if k.lower() not in {"correct", "reason", "explanation"}]
        instruction = 0.95 if predicted is not None else 0.4
        if extra_keys and len(obj) > 4:
            instruction *= 0.7
    else:
        low = pred.lower()
        if "true" in low and "false" not in low:
            predicted = True
            instruction = 0.45
        elif "false" in low and "true" not in low:
            predicted = False
            instruction = 0.45
        details["predicted_correct"] = predicted
    if predicted is None:
        correctness = 0.0
        semantic = 0.0
        agreement = 0.0
    else:
        correctness = 1.0 if predicted == expected else 0.0
        semantic = 1.0 if predicted == expected else 0.0
        agreement = correctness
    fluency = 1.0 if details["parsed"] else 0.45
    # verbosity: JSON-only preferred
    instruction *= verbosity_penalty(pred, ['{"correct": true}'])
    return finalize_scores(example, pred, correctness, semantic, agreement, instruction, fluency, details)


def score_correct(example: dict, raw: str) -> dict:
    pred = strip_model_output(raw)
    details: dict[str, Any] = {}
    if not pred:
        return finalize_scores(example, pred, 0.0, 0.0, 0.0, 0.0, 0.0, {"empty": True})
    refs = [str(r) for r in example.get("acceptable_answers") or [example["ground_truth"]]]
    refs = [r for r in refs if isinstance(r, str)]
    core = first_sentence(pred) if len(tokens(pred)) > max(18, 3 * len(tokens(refs[0]))) else pred
    agreement, matched = best_ref_match(core, refs)
    details["matched_ref"] = matched
    details["agreement_raw"] = agreement

    must_not = [normalize(x) for x in (example.get("must_not_contain") or []) if x]
    ncore = normalize(core)
    violations = []
    for b in must_not:
        if not b:
            continue
        if re.search(r"(?<![a-z])" + re.escape(b) + r"(?![a-z])", ncore):
            violations.append(b)
    details["must_not_violations"] = violations

    meaning = [normalize(t) for t in (example.get("meaning_tokens") or []) if t]
    present = [t for t in meaning if t in normalize(core)]
    semantic = (len(present) / len(meaning)) if meaning else max(agreement, 0.5)
    src_c = content_tokens(example.get("source_sentence") or "")
    pred_c = content_tokens(core)
    ref_c = content_tokens(" ".join(refs))
    keep = src_c & pred_c
    if src_c:
        semantic = 0.55 * semantic + 0.45 * (len(keep) / max(1, len(src_c)))
    invented = pred_c - src_c - ref_c
    if invented and len(invented) >= 3:
        semantic *= 0.75
        details["invented_content"] = sorted(invented)[:8]
    semantic = float(min(1.0, max(0.0, semantic)))

    key_reqs = example.get("key_requirements") or []
    req_hits = 0
    for req in key_reqs:
        r = normalize(req)
        pieces = [p.strip() for p in re.split(r"\bor\b", r) if p.strip()]
        if any(p in normalize(core) or p in normalize(matched) for p in pieces if len(p) > 3):
            req_hits += 1
        elif agreement >= 0.9:
            req_hits += 1
    req_frac = (req_hits / len(key_reqs)) if key_reqs else agreement

    correctness = 0.62 * agreement + 0.25 * req_frac + 0.13 * (0.0 if violations else 1.0)
    if example["category"] == "already_correct":
        src_n = normalize(example["source_sentence"]).rstrip(" .!?")
        pr_n = normalize(core).rstrip(" .!?")
        if src_n == pr_n:
            correctness = max(correctness, 0.98)
            semantic = max(semantic, 0.98)
        elif agreement < 0.9:
            correctness *= 0.7
            details["over_edit"] = True
    if violations:
        correctness = min(correctness, 0.38)
        agreement = min(agreement, 0.55)
        details["unfixed_error_tokens"] = violations

    instruction = 0.55 * lecture_penalty(pred) + 0.45 * verbosity_penalty(pred, refs)
    if extract_first_json(pred) and example["task"] == "correct_sentence":
        instruction *= 0.55
        details["unexpected_json"] = True
    fluency = fluency_score(core)
    return finalize_scores(example, pred, correctness, semantic, agreement, instruction, fluency, details)


def severity_of(correctness: float, semantic: float, instruction: float) -> str:
    if correctness >= 0.92 and semantic >= 0.90 and instruction >= 0.80:
        return "completely_correct"
    if correctness >= 0.70 and semantic >= 0.75:
        return "minor_issue"
    if correctness >= 0.40:
        return "substantive"
    return "major_failure"


def finalize_scores(example, pred, correctness, semantic, agreement, instruction, fluency, details) -> dict:
    correctness = float(min(1.0, max(0.0, correctness)))
    semantic = float(min(1.0, max(0.0, semantic)))
    agreement = float(min(1.0, max(0.0, agreement)))
    instruction = float(min(1.0, max(0.0, instruction)))
    fluency = float(min(1.0, max(0.0, fluency)))
    example_score = (
        0.32 * correctness
        + 0.22 * semantic
        + 0.16 * agreement
        + 0.16 * instruction
        + 0.14 * fluency
    )
    sev = severity_of(correctness, semantic, instruction)
    passed = (
        sev in {"completely_correct", "minor_issue"}
        and correctness >= 0.75
        and semantic >= 0.80
        and instruction >= 0.55
    )
    return {
        "example_id": example["id"],
        "task": example["task"],
        "category": example["category"],
        "difficulty": example["difficulty"],
        "prompt": example["prompt"],
        "ground_truth": example["ground_truth"],
        "acceptable_answers": example.get("acceptable_answers"),
        "model_output": pred,
        "scores": {
            "correctness": round(correctness, 4),
            "semantic_preservation": round(semantic, 4),
            "ground_truth_agreement": round(agreement, 4),
            "instruction_following": round(instruction, 4),
            "fluency": round(fluency, 4),
            "example_score": round(example_score, 4),
        },
        "error_severity": sev,
        "passed": passed,
        "catastrophic": sev == "major_failure" and correctness < 0.20,
        "scoring_details": details,
    }


def score_example(example: dict, raw: str) -> dict:
    task = example.get("task")
    try:
        if task in {"judge_complete", "judge_mistake"}:
            return score_judge(example, raw)
        if task in {"invent_complete", "invent_mistake"}:
            return score_invent(example, raw)
        return score_correct(example, raw)
    except Exception as exc:
        rec = finalize_scores(example, strip_model_output(raw), 0.0, 0.0, 0.0, 0.0, 0.0, {"scorer_error": str(exc)})
        rec["error_severity"] = "major_failure"
        rec["passed"] = False
        rec["catastrophic"] = True
        return rec


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    m = mean(xs)
    if len(xs) < 2:
        return m, m, m
    se = pstdev(xs) / math.sqrt(len(xs))
    z = 1.96
    return m, max(0.0, m - z * se), min(1.0, m + z * se)


def wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(model_name: str, records: list[dict], load_error: str | None = None) -> dict:
    n = len(records)
    scores = [r["scores"]["example_score"] for r in records]
    corr = [r["scores"]["correctness"] for r in records]
    sem = [r["scores"]["semantic_preservation"] for r in records]
    agr = [r["scores"]["ground_truth_agreement"] for r in records]
    ins = [r["scores"]["instruction_following"] for r in records]
    flu = [r["scores"]["fluency"] for r in records]
    passed = sum(1 for r in records if r["passed"])
    failed = n - passed
    major = sum(1 for r in records if r["error_severity"] in MAJOR_FAIL_SEVERITIES)
    minor = sum(1 for r in records if r["error_severity"] == "minor_issue")
    complete = sum(1 for r in records if r["error_severity"] == "completely_correct")
    catas = sum(1 for r in records if r.get("catastrophic"))
    by_cat: dict[str, list[dict]] = defaultdict(list)
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
        by_diff[r["difficulty"]].append(r)

    def pack(rs: list[dict]) -> dict:
        sc = [x["scores"]["example_score"] for x in rs]
        cc = [x["scores"]["correctness"] for x in rs]
        return {
            "n": len(rs),
            "example_score": round(mean(sc), 4),
            "correctness": round(mean(cc), 4),
            "pass_rate": round(sum(1 for x in rs if x["passed"]) / max(1, len(rs)), 4),
            "major_error_rate": round(sum(1 for x in rs if x["error_severity"] in MAJOR_FAIL_SEVERITIES) / max(1, len(rs)), 4),
        }

    cat_stats = {k: pack(v) for k, v in sorted(by_cat.items())}
    grammar_cats = [c for c in cat_stats if c in MAJOR_CATEGORIES]
    worst = None
    if grammar_cats:
        worst = min(grammar_cats, key=lambda c: cat_stats[c]["correctness"])
    hard = [r for r in records if r["difficulty"] in {"hard", "adversarial"}]
    m_score, lo, hi = mean_ci(scores)
    p_rate, plo, phi = wilson_ci(passed, n) if n else (0, 0, 0)
    return {
        "model": model_name,
        "load_error": load_error,
        "status": "failed_to_load" if load_error else "evaluated",
        "n": n,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "pass_rate_ci95": [round(plo, 4), round(phi, 4)],
        "overall": {
            "example_score": round(m_score, 4),
            "example_score_ci95": [round(lo, 4), round(hi, 4)],
            "correctness": round(mean(corr), 4),
            "semantic_preservation": round(mean(sem), 4),
            "ground_truth_agreement": round(mean(agr), 4),
            "instruction_following": round(mean(ins), 4),
            "fluency": round(mean(flu), 4),
            "score_std": round(pstdev(scores), 4),
            "score_variance": round(pstdev(scores) ** 2, 4),
        },
        "error_counts": {
            "completely_correct": complete,
            "minor_issue": minor,
            "substantive": sum(1 for r in records if r["error_severity"] == "substantive"),
            "major_failure": sum(1 for r in records if r["error_severity"] == "major_failure"),
            "catastrophic": catas,
        },
        "major_error_rate": round(major / n, 4) if n else 1.0,
        "minor_error_rate": round(minor / n, 4) if n else 0.0,
        "catastrophic_rate": round(catas / n, 4) if n else 1.0,
        "difficult_example_score": round(mean([r["scores"]["example_score"] for r in hard]), 4) if hard else 0.0,
        "difficult_n": len(hard),
        "per_category": cat_stats,
        "per_difficulty": {k: pack(v) for k, v in sorted(by_diff.items())},
        "worst_category": worst,
        "worst_category_correctness": cat_stats[worst]["correctness"] if worst else 0.0,
    }


def evaluate_threshold(summary: dict, thresholds: dict) -> dict:
    reasons = []
    if summary.get("load_error"):
        return {"passed": False, "reasons": [f"load_error: {summary['load_error']}"]}
    o = summary["overall"]
    checks = [
        (o["example_score"] >= thresholds["min_example_score"], f"example_score {o['example_score']} < {thresholds['min_example_score']}"),
        (o["correctness"] >= thresholds["min_correctness"], f"correctness {o['correctness']} < {thresholds['min_correctness']}"),
        (o["semantic_preservation"] >= thresholds["min_semantic_preservation"], f"semantic {o['semantic_preservation']} < {thresholds['min_semantic_preservation']}"),
        (o["instruction_following"] >= thresholds["min_instruction_following"], f"instruction {o['instruction_following']} < {thresholds['min_instruction_following']}"),
        (o["fluency"] >= thresholds["min_fluency"], f"fluency {o['fluency']} < {thresholds['min_fluency']}"),
        (summary["major_error_rate"] <= thresholds["max_major_error_rate"], f"major_error_rate {summary['major_error_rate']} > {thresholds['max_major_error_rate']}"),
        (summary["catastrophic_rate"] <= thresholds["max_catastrophic_rate"], f"catastrophic_rate {summary['catastrophic_rate']} > {thresholds['max_catastrophic_rate']}"),
        (summary["pass_rate"] >= thresholds["min_pass_rate"], f"pass_rate {summary['pass_rate']} < {thresholds['min_pass_rate']}"),
        (summary["difficult_example_score"] >= thresholds["min_difficult_score"], f"difficult_score {summary['difficult_example_score']} < {thresholds['min_difficult_score']}"),
        (o["score_std"] <= thresholds["max_score_std"], f"score_std {o['score_std']} > {thresholds['max_score_std']}"),
        (summary["worst_category_correctness"] >= thresholds["min_category_correctness"], f"worst_category {summary.get('worst_category')} correctness {summary['worst_category_correctness']} < {thresholds['min_category_correctness']}"),
    ]
    for cat, st in summary.get("per_category", {}).items():
        if cat in MAJOR_CATEGORIES and st["n"] >= 5:
            if st["correctness"] < thresholds["min_category_correctness"]:
                checks.append((False, f"category {cat} correctness {st['correctness']} < {thresholds['min_category_correctness']}"))
    for ok, reason in checks:
        if not ok:
            reasons.append(reason)
    return {"passed": not reasons, "reasons": reasons}


def svg_bars(path: Path, title: str, labels: list[str], values: list[float], ymax: float = 1.0) -> None:
    w, h = 920, 420
    left, right, top, bottom = 160, 24, 48, 56
    inner_w = w - left - right
    inner_h = h - top - bottom
    n = max(1, len(labels))
    bw = inner_w / n * 0.7
    gap = inner_w / n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0f1419"/>',
        f'<text x="{w/2}" y="28" fill="#e8eef4" text-anchor="middle" font-size="16" font-family="sans-serif">{escape_xml(title)}</text>',
    ]
    for i in range(5):
        y = top + inner_h * i / 4
        val = ymax * (1 - i / 4)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#2a3340" />')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" fill="#9aa7b5" text-anchor="end" font-size="11" font-family="sans-serif">{val:.2f}</text>')
    colors = ["#5b8def", "#3ccf91", "#f0c14b", "#e07a5f", "#c77dff", "#80cbc4", "#ffb4a2", "#90caf9"]
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = left + i * gap + (gap - bw) / 2
        bh = inner_h * (max(0.0, min(ymax, val)) / ymax)
        y = top + inner_h - bh
        c = colors[i % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{c}" rx="4"/>')
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{h-18}" fill="#c5d0da" text-anchor="middle" font-size="10" font-family="sans-serif">{escape_xml(lab[:18])}</text>'
        )
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" fill="#e8eef4" text-anchor="middle" font-size="11" font-family="sans-serif">{val:.2f}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_grouped_svg(path: Path, title: str, series: dict[str, dict[str, float]], cats: list[str]) -> None:
    models = list(series.keys())
    w = max(980, 80 * len(cats) + 180)
    h = 480
    left, right, top, bottom = 48, 24, 64, 90
    inner_w = w - left - right
    inner_h = h - top - bottom
    group = inner_w / max(1, len(cats))
    bw = group / max(1, len(models)) * 0.8
    colors = ["#5b8def", "#3ccf91", "#f0c14b", "#e07a5f", "#c77dff", "#80cbc4"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0f1419"/>',
        f'<text x="{w/2}" y="28" fill="#e8eef4" text-anchor="middle" font-size="16" font-family="sans-serif">{escape_xml(title)}</text>',
    ]
    for i, cat in enumerate(cats):
        gx = left + i * group
        for j, model in enumerate(models):
            val = series[model].get(cat, 0.0)
            x = gx + j * (group / max(1, len(models))) + 4
            bh = inner_h * max(0.0, min(1.0, val))
            y = top + inner_h - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{colors[j % len(colors)]}" rx="2"/>')
        parts.append(
            f'<text x="{gx+group/2:.1f}" y="{h-36}" fill="#c5d0da" text-anchor="middle" font-size="9" font-family="sans-serif">{escape_xml(cat[:16])}</text>'
        )
    for j, model in enumerate(models):
        parts.append(
            f'<rect x="{left+j*140}" y="{h-22}" width="10" height="10" fill="{colors[j % len(colors)]}"/>'
            f'<text x="{left+j*140+16}" y="{h-13}" fill="#c5d0da" font-size="11" font-family="sans-serif">{escape_xml(model[:22])}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_analytics(results_dir: Path, config: dict, summaries: list[dict], run_meta: dict) -> dict:
    plots = results_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    evaluated = [s for s in summaries if s["status"] == "evaluated" and s["n"] > 0]
    labels = [s["model"] for s in evaluated]
    svg_bars(plots / "overall_scores.svg", "Overall example score", labels, [s["overall"]["example_score"] for s in evaluated])
    svg_bars(plots / "correctness.svg", "Correctness", labels, [s["overall"]["correctness"] for s in evaluated])
    svg_bars(plots / "pass_rate.svg", "Pass rate", labels, [s["pass_rate"] for s in evaluated])
    svg_bars(plots / "consistency_std.svg", "Score std (lower is more consistent)", labels, [s["overall"]["score_std"] for s in evaluated])
    svg_bars(plots / "major_error_rate.svg", "Major error rate", labels, [s["major_error_rate"] for s in evaluated])
    svg_bars(plots / "semantic_preservation.svg", "Semantic preservation", labels, [s["overall"]["semantic_preservation"] for s in evaluated])

    all_cats = sorted({c for s in evaluated for c in s["per_category"]})
    series = {s["model"]: {c: s["per_category"].get(c, {}).get("correctness", 0.0) for c in all_cats} for s in evaluated}
    write_grouped_svg(plots / "per_category.svg", "Per-category correctness", series, all_cats)
    diffs = sorted({d for s in evaluated for d in s["per_difficulty"]})
    dseries = {s["model"]: {d: s["per_difficulty"].get(d, {}).get("example_score", 0.0) for d in diffs} for s in evaluated}
    write_grouped_svg(plots / "per_difficulty.svg", "Performance by difficulty", dseries, diffs)
    eseries = {
        s["model"]: {
            "complete": s["error_counts"]["completely_correct"] / max(1, s["n"]),
            "minor": s["error_counts"]["minor_issue"] / max(1, s["n"]),
            "substantive": s["error_counts"]["substantive"] / max(1, s["n"]),
            "major": s["error_counts"]["major_failure"] / max(1, s["n"]),
        }
        for s in evaluated
    }
    write_grouped_svg(
        plots / "error_distribution.svg",
        "Error / success mix (share of examples)",
        eseries,
        ["complete", "minor", "substantive", "major"],
    )

    thresholds = config["thresholds"]
    verdicts = []
    for s in summaries:
        v = evaluate_threshold(s, thresholds) if s["status"] == "evaluated" else {"passed": False, "reasons": [s.get("load_error") or "not evaluated"]}
        verdicts.append({"model": s["model"], **v, "example_score": None if not s.get("overall") else s["overall"]["example_score"]})

    qualified = [v for v in verdicts if v["passed"]]
    scored = [s for s in evaluated]
    highest = max(scored, key=lambda s: s["overall"]["example_score"]) if scored else None
    winner = None
    winner_note = "NO MODEL PASSED THE REQUIRED THRESHOLD"
    if qualified:
        qmap = {v["model"]: True for v in qualified}
        winner = max([s for s in scored if s["model"] in qmap], key=lambda s: s["overall"]["example_score"])
        winner_note = winner["model"]

    # differentiation: category variance across models
    diffs_cat = []
    for cat in all_cats:
        vals = [s["per_category"].get(cat, {}).get("correctness", 0.0) for s in evaluated]
        if len(vals) >= 2:
            diffs_cat.append((cat, max(vals) - min(vals), vals))
    diffs_cat.sort(key=lambda x: x[1], reverse=True)

    most_consistent = min(evaluated, key=lambda s: s["overall"]["score_std"]) if evaluated else None
    analytics = {
        "generated_from": "evaluation records",
        "run": run_meta,
        "thresholds": thresholds,
        "winner_rule": (
            "A model qualifies only if ALL threshold checks pass, including per-major-category "
            "correctness floors. Among qualified models, highest overall example_score wins. "
            "If none qualify, winner is null and message is NO MODEL PASSED THE REQUIRED THRESHOLD."
        ),
        "highest_scoring_model": None if not highest else {
            "name": highest["model"],
            "example_score": highest["overall"]["example_score"],
            "qualified": any(v["model"] == highest["model"] and v["passed"] for v in verdicts),
        },
        "threshold_qualified_winner": None if not winner else {
            "name": winner["model"],
            "example_score": winner["overall"]["example_score"],
        },
        "winner_message": winner_note if winner else "NO MODEL PASSED THE REQUIRED THRESHOLD",
        "qualification": verdicts,
        "failed_qualification": [v for v in verdicts if not v["passed"]],
        "most_consistent_model": None if not most_consistent else {
            "name": most_consistent["model"],
            "score_std": most_consistent["overall"]["score_std"],
        },
        "categories_that_differentiate_most": [
            {"category": c, "spread": round(sp, 4)} for c, sp, _ in diffs_cat[:12]
        ],
        "summaries": summaries,
        "plots": [str(p.relative_to(results_dir)) for p in sorted(plots.glob("*.svg"))],
    }
    (results_dir / "analytics.json").write_text(json.dumps(analytics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (results_dir / "summary.json").write_text(
        json.dumps(
            {
                "winner_message": analytics["winner_message"],
                "highest_scoring_model": analytics["highest_scoring_model"],
                "threshold_qualified_winner": analytics["threshold_qualified_winner"],
                "models": [
                    {
                        "model": s["model"],
                        "status": s["status"],
                        "load_error": s.get("load_error"),
                        "n": s["n"],
                        "passed": s.get("passed"),
                        "failed": s.get("failed"),
                        "overall": s.get("overall"),
                        "major_error_rate": s.get("major_error_rate"),
                        "catastrophic_rate": s.get("catastrophic_rate"),
                    }
                    for s in summaries
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return analytics


def apply_chat_template(tokenizer, messages: list[dict], contender: dict) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not contender.get("enable_thinking", False):
        kwargs["enable_thinking"] = False
        kwargs["thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        kwargs.pop("thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            # fallback ChatML
            chunks = []
            for m in messages:
                chunks.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            chunks.append("<|im_start|>assistant\n")
            return "\n".join(chunks)


def build_messages(example: dict, contender: dict) -> list[dict]:
    sys_msg = (
        "You are a precise English grammar coach for children ages 5-8. "
        "Follow the user instructions exactly. Do not think out loud. "
        "Do not add extra commentary."
    )
    user = example["prompt"]
    if contender.get("no_think_suffix"):
        user = user + "\n\n/no_think"
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def classify_failure(text: str) -> str:
    t = (text or "").lower()
    if any(s in t for s in ("out of memory", "cuda out of memory", "engine core initialization failed")):
        return "oom"
    if any(
        s in t
        for s in (
            "localentrynotfound",
            "couldn't find them in the cached",
            "outgoing traffic has been disabled",
            "not found on huggingface",
            "404 client error",
            "repository not found",
        )
    ):
        return "missing_weights"
    if any(s in t for s in ("401", "403", "gated", "access to model")):
        return "gated_repo"
    return "other"


def is_oom(exc: BaseException | str) -> bool:
    return classify_failure(str(exc)) == "oom"


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def wait_for_gpu(min_free_frac: float = 0.45, timeout: float = 45.0) -> dict:
    info = {"free_gib": None, "total_gib": None, "waited_s": 0.0}
    t0 = time.time()
    while True:
        release_cuda()
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                info["free_gib"] = round(free / 1024**3, 2)
                info["total_gib"] = round(total / 1024**3, 2)
                if total and free / total >= min_free_frac:
                    info["waited_s"] = round(time.time() - t0, 2)
                    return info
        except Exception:
            break
        if time.time() - t0 >= timeout:
            info["waited_s"] = round(time.time() - t0, 2)
            return info
        time.sleep(2)


def memory_profiles(config: dict, contender: dict) -> list[dict]:
    profiles = list(config.get("memory_profiles") or [])
    if not profiles:
        profiles = [
            {
                "gpu_memory_utilization": 0.78,
                "max_model_len": 768,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 512,
            }
        ]
    first = dict(profiles[0])
    for key in ("gpu_memory_utilization", "max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if contender.get(key) is not None:
            first[key] = contender[key]
    out = [first]
    for p in profiles[1:]:
        if p != first:
            out.append(p)
    return out


def generate_chunked(llm, prompts: list[str], sp, chunk_sizes: list[int]) -> list[str]:
    last_err: Exception | None = None
    for chunk in chunk_sizes:
        texts: list[str] = []
        try:
            for i in range(0, len(prompts), chunk):
                batch = prompts[i : i + chunk]
                outs = llm.generate(batch, sp, use_tqdm=True)
                for o in outs:
                    texts.append(o.outputs[0].text if o.outputs else "")
            return texts
        except Exception as exc:
            last_err = exc
            if not is_oom(exc):
                raise
            print(f"[arena] generate OOM at chunk={chunk}; retrying smaller batches", flush=True)
            release_cuda()
            time.sleep(2)
    raise last_err or RuntimeError("generate failed")


def run_vllm(contender: dict, examples: list[dict], gen: dict, config: dict) -> tuple[list[str], dict]:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    local_only = bool(config.get("hf_local_files_only", False))
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    gpu = wait_for_gpu()
    print(f"[arena] gpu before load: {gpu}", flush=True)

    hf_id = contender["hf_id"]
    tok = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=bool(contender.get("trust_remote_code", True)),
        local_files_only=local_only,
    )
    prompts = [apply_chat_template(tok, build_messages(ex, contender), contender) for ex in examples]
    sp_kwargs = dict(
        temperature=float(gen.get("temperature", 0.0)),
        top_p=float(gen.get("top_p", 1.0)),
        max_tokens=int(gen.get("max_tokens", 96)),
        repetition_penalty=float(gen.get("repetition_penalty", 1.0)),
    )
    try:
        sp = SamplingParams(seed=int(gen.get("seed", 42)), **sp_kwargs)
    except TypeError:
        sp = SamplingParams(**sp_kwargs)

    vllm_defaults = config.get("vllm") or {}
    chunks = list(config.get("generate_chunk_sizes") or [32, 8, 1])
    attempts: list[dict] = []
    last_exc: Exception | None = None
    t0 = time.time()

    for i, profile in enumerate(memory_profiles(config, contender), 1):
        llm = None
        kwargs = {
            "model": hf_id,
            "trust_remote_code": bool(contender.get("trust_remote_code", True)),
            "enforce_eager": bool(vllm_defaults.get("enforce_eager", True)),
            "dtype": contender.get("dtype") or vllm_defaults.get("dtype") or "auto",
            "seed": 42,
            "enable_prefix_caching": bool(vllm_defaults.get("enable_prefix_caching", False)),
            "gpu_memory_utilization": float(profile["gpu_memory_utilization"]),
            "max_model_len": int(profile["max_model_len"]),
            "max_num_seqs": int(profile["max_num_seqs"]),
            "max_num_batched_tokens": int(profile["max_num_batched_tokens"]),
        }
        q = contender.get("quantization")
        if q:
            kwargs["quantization"] = q
        print(f"[arena] load attempt {i}/{len(memory_profiles(config, contender))} {kwargs}", flush=True)
        try:
            try:
                llm = LLM(**kwargs)
            except TypeError:
                kwargs.pop("enable_prefix_caching", None)
                kwargs.pop("max_num_batched_tokens", None)
                llm = LLM(**kwargs)
            texts = generate_chunked(llm, prompts, sp, chunks)
            meta = {
                "backend": "vllm",
                "seconds": round(time.time() - t0, 2),
                "n_prompts": len(prompts),
                "profile": profile,
                "attempts": attempts + [{"profile": profile, "ok": True}],
                "gpu_before": gpu,
            }
            return texts, meta
        except Exception as exc:
            last_exc = exc
            kind = classify_failure(exc)
            attempts.append({"profile": profile, "ok": False, "class": kind, "error": str(exc).splitlines()[-1][:300]})
            print(f"[arena] load/generate failed ({kind}): {attempts[-1]['error']}", flush=True)
            if kind != "oom":
                break
            release_cuda()
            time.sleep(3)
            wait_for_gpu()
        finally:
            if llm is not None:
                try:
                    del llm
                except Exception:
                    pass
            release_cuda()

    del tok
    release_cuda()
    raise last_exc or RuntimeError("vLLM failed")


def evaluate_contender(config: dict, contender: dict, examples: list[dict], out_dir: Path) -> dict:
    name = contender["name"]
    gen = dict(config.get("generation") or {})
    gen["seed"] = config.get("seed", 42)
    random.seed(gen["seed"])
    raw_path = out_dir / "raw" / f"{name}.jsonl"
    rec_path = out_dir / "raw" / f"{name}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    load_error = None
    fail_class = None
    outputs: list[str] = []
    inf_meta: dict = {}
    try:
        if contender.get("provider") != "vllm":
            raise RuntimeError(f"unsupported provider {contender.get('provider')}")
        outputs, inf_meta = run_vllm(contender, examples, gen, config)
    except Exception:
        load_error = traceback.format_exc()
        fail_class = classify_failure(load_error)
        outputs = []
    records = []
    if load_error:
        short = load_error.strip().splitlines()[-1][:500]
        (out_dir / "raw" / f"{name}.load_error.txt").write_text(load_error, encoding="utf-8")
        fail_rec = {
            "model": name,
            "example_id": None,
            "status": "failed_to_load",
            "failure_class": fail_class,
            "load_error": short,
            "attempted_examples": len(examples),
        }
        raw_path.write_text(json.dumps(fail_rec, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = summarize(name, [], load_error=short)
        summary["failure_class"] = fail_class
        summary["attempted_examples"] = len(examples)
        summary["example_ids_planned"] = [e["id"] for e in examples]
    else:
        err_txt = out_dir / "raw" / f"{name}.load_error.txt"
        if err_txt.exists():
            err_txt.unlink()
        for ex, text in zip(examples, outputs):
            rec = score_example(ex, text)
            rec["model"] = name
            rec["raw_output"] = text
            rec["inference_error"] = False
            records.append(rec)
        with raw_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary = summarize(name, records, load_error=None)
        summary["failure_class"] = None
    summary["inference"] = inf_meta
    summary["contender"] = {k: contender[k] for k in contender if k != "notes"}
    rec_path.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "raw" / f"{name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def successful_eval(path: Path, n_examples: int) -> bool:
    if not path.exists():
        return False
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return s.get("status") == "evaluated" and int(s.get("n") or 0) == n_examples and not s.get("load_error")


def self_test() -> int:
    ok = True
    ex = {
        "id": "t1",
        "task": "correct_sentence",
        "category": "subject_verb_agreement",
        "difficulty": "easy",
        "prompt": "x",
        "source_sentence": "She don't like apples.",
        "ground_truth": "She doesn't like apples.",
        "acceptable_answers": ["She doesn't like apples.", "She does not like apples."],
        "key_requirements": ["doesn't or does not"],
        "must_not_contain": ["don't"],
        "meaning_tokens": ["she", "like", "apples"],
    }
    good = score_example(ex, "She doesn't like apples.")
    bad = score_example(ex, "She don't like apples.")
    verbose = score_example(ex, "Sure, as an AI I would say the correct sentence is: She doesn't like apples. Let me explain why at length. " * 8)
    if good["scores"]["correctness"] < 0.9 or not good["passed"]:
        print("FAIL good correction", good)
        ok = False
    if bad["scores"]["correctness"] > 0.5 or bad["passed"]:
        print("FAIL unfixed", bad)
        ok = False
    if verbose["scores"]["instruction_following"] >= good["scores"]["instruction_following"]:
        print("FAIL verbosity", verbose["scores"], good["scores"])
        ok = False
    jex = {
        "id": "j1",
        "task": "judge_complete",
        "category": "judge_completion",
        "difficulty": "easy",
        "prompt": "p",
        "ground_truth": {"correct": True},
        "acceptable_answers": [{"correct": True}],
        "key_requirements": ["json"],
        "must_not_contain": [],
        "meaning_tokens": [],
    }
    jg = score_example(jex, '{"correct": true}')
    jb = score_example(jex, '{"correct": false}')
    if jg["scores"]["correctness"] != 1.0 or jb["scores"]["correctness"] != 0.0:
        print("FAIL judge", jg, jb)
        ok = False
    fake = summarize("m", [good] * 10)
    fake["per_category"] = {
        "subject_verb_agreement": {"n": 10, "correctness": 0.99, "example_score": 0.99, "pass_rate": 1, "major_error_rate": 0}
    }
    th = {
        "min_example_score": 0.88,
        "min_correctness": 0.90,
        "min_semantic_preservation": 0.92,
        "min_instruction_following": 0.85,
        "min_fluency": 0.80,
        "max_major_error_rate": 0.04,
        "max_catastrophic_rate": 0.02,
        "min_category_correctness": 0.80,
        "min_difficult_score": 0.78,
        "min_pass_rate": 0.85,
        "max_score_std": 0.22,
    }
    # all-good copies should pass most numeric floors
    print("self-test summary", json.dumps({k: fake[k] for k in ("overall", "pass_rate", "major_error_rate")}))
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def validate(config: dict, dataset_path: Path) -> None:
    if not config.get("contenders"):
        raise SystemExit("no contenders configured")
    names = [c["name"] for c in config["contenders"]]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate contender names")
    for c in config["contenders"]:
        for key in ("name", "hf_id", "provider"):
            if key not in c:
                raise SystemExit(f"contender missing {key}")
    if not dataset_path.exists() or dataset_path.stat().st_size < 10:
        raise SystemExit(f"dataset missing: {dataset_path}")
    examples = load_dataset(dataset_path)
    print(f"config ok: {len(config['contenders'])} contenders, {len(examples)} examples")


def cmd_run_one() -> int:
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "contenders.config"))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--ensure-dataset", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--contender", default="")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-evaluate models that already succeeded")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    cfg_path = Path(args.config)
    config = load_config(cfg_path)
    dataset_path = Path(args.dataset) if args.dataset else HERE / config.get("dataset_path", "prompts.json")
    results_dir = Path(args.results_dir) if args.results_dir else HERE / config.get("results_dir", "results")

    if args.ensure_dataset:
        from dataset_builder import main as build_main

        build_main()
        return 0

    if args.validate:
        if not dataset_path.exists():
            from dataset_builder import main as build_main

            build_main()
        validate(config, dataset_path)
        return 0

    if not dataset_path.exists() or dataset_path.stat().st_size < 10:
        from dataset_builder import main as build_main

        build_main()

    examples = load_dataset(dataset_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(exist_ok=True)

    if args.contender:
        contenders = [c for c in config["contenders"] if c["name"] == args.contender]
        if not contenders:
            print(f"unknown contender {args.contender}", file=sys.stderr)
            return 2
        summary = evaluate_contender(config, contenders[0], examples, results_dir)
        print(json.dumps({"model": summary["model"], "status": summary["status"], "load_error": bool(summary.get("load_error")), "n": summary["n"], "score": summary.get("overall")}, indent=2))
        return 0 if summary["status"] == "evaluated" else 3

    if args.aggregate_only or args.run:
        run_meta = {
            "seed": config.get("seed", 42),
            "generation": config.get("generation"),
            "vllm": config.get("vllm"),
            "hf_local_files_only": config.get("hf_local_files_only", True),
            "n_examples": len(examples),
            "example_ids": [e["id"] for e in examples],
            "started_unix": int(time.time()),
        }
        summaries = []
        if args.run:
            py = args.python

            def _fail_summary(model: str, err: str) -> dict:
                return {
                    "model": model,
                    "load_error": err,
                    "status": "failed_to_load",
                    "n": 0,
                    "passed": 0,
                    "failed": 0,
                    "overall": {
                        "example_score": 0,
                        "example_score_ci95": [0, 0],
                        "correctness": 0,
                        "semantic_preservation": 0,
                        "ground_truth_agreement": 0,
                        "instruction_following": 0,
                        "fluency": 0,
                        "score_std": 0,
                        "score_variance": 0,
                    },
                    "per_category": {},
                    "per_difficulty": {},
                    "major_error_rate": 1.0,
                    "catastrophic_rate": 1.0,
                    "pass_rate": 0.0,
                    "error_counts": {
                        "completely_correct": 0,
                        "minor_issue": 0,
                        "substantive": 0,
                        "major_failure": 0,
                        "catastrophic": 0,
                    },
                    "difficult_example_score": 0.0,
                    "worst_category": None,
                    "worst_category_correctness": 0.0,
                }

            for skipped in config.get("ineligible") or []:
                print(f"[arena] skipping ineligible {skipped.get('name')}: {skipped.get('reason')}", flush=True)

            for c in config["contenders"]:
                sum_path = results_dir / "raw" / f"{c['name']}.summary.json"
                resume = bool(config.get("resume_successful", True)) and not args.force
                if resume and successful_eval(sum_path, len(examples)):
                    print(f"\n=== resume {c['name']} (already evaluated n={len(examples)}) ===", flush=True)
                    summaries.append(json.loads(sum_path.read_text(encoding="utf-8")))
                    continue
                print(f"\n=== evaluating {c['name']} ({c['hf_id']}) ===", flush=True)
                try:
                    proc = subprocess.run(
                        [
                            py,
                            str(HERE / "eval.py"),
                            "--config",
                            str(cfg_path),
                            "--contender",
                            c["name"],
                            "--results-dir",
                            str(results_dir),
                            "--dataset",
                            str(dataset_path),
                        ],
                        cwd=str(HERE),
                        timeout=int(os.environ.get("ARENA_MODEL_TIMEOUT", "5400")),
                    )
                except subprocess.TimeoutExpired:
                    summaries.append(_fail_summary(c["name"], "timeout"))
                    continue
                if sum_path.exists():
                    summaries.append(json.loads(sum_path.read_text(encoding="utf-8")))
                else:
                    summaries.append(_fail_summary(c["name"], f"subprocess_exit_{proc.returncode}_no_summary"))
        else:
            for c in config["contenders"]:
                p = results_dir / "raw" / f"{c['name']}.summary.json"
                if p.exists():
                    summaries.append(json.loads(p.read_text(encoding="utf-8")))
        run_meta["finished_unix"] = int(time.time())
        (results_dir / "run_config.json").write_text(
            json.dumps({"config": config, "run": run_meta}, indent=2) + "\n", encoding="utf-8"
        )
        analytics = write_analytics(results_dir, config, summaries, run_meta)
        print("\n===== ARENA RESULT =====")
        print(analytics["winner_message"])
        print("highest:", analytics["highest_scoring_model"])
        print("qualified winner:", analytics["threshold_qualified_winner"])
        # harness failure if zero models produced any summary file after run
        if args.run and not any(s.get("n", 0) > 0 or s.get("load_error") for s in summaries):
            return 1
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
