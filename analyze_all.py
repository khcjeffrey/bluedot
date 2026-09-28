"""
Full analysis of CoT faithfulness experiments.

Produces summary tables for:
  1. Baseline accuracy (with CoT)
  2. Level 0 accuracy (no CoT)
  3. Truncation intervention (per-question, sentence-level, first/second half)
  4. Mistakes intervention (per-question, sentence-level, first/second half)
  5. CoT length comparison
  6. Restart detection (mistakes only)
  7. Answer format compliance (baseline + interventions)

Usage:
    python analyze_all.py
    python analyze_all.py --output results/summary.json
"""

import json
import math
import re
from pathlib import Path

RESULTS_DIR = Path("results")

BASELINES = {
    ("FL", "freeform"): "cot_gemini-3_5-flash-lite_freeform_20260913_131538.json",
    ("Flash", "freeform"): "cot_gemini-3_5-flash_freeform_20260913_205023.json",
    ("FL", "structured"): "cot_gemini-3_5-flash-lite_structured_20260914_201713.json",
}

TRUNCATION = {
    ("FL", "freeform"): "truncation_gemini-3_5-flash-lite_20260913_222237.json",
    ("Flash", "freeform"): "truncation_gemini-3_5-flash_20260913_222302.json",
    ("FL", "structured"): "truncation_gemini-3_5-flash-lite_structured_20260914_204752.json",
}

MISTAKES = {
    ("FL", "freeform"): "mistakes_gemini-3_5-flash-lite_20260914_170321.json",
    ("Flash", "freeform"): "mistakes_gemini-3_5-flash_20260914_175334.json",
    ("FL", "structured"): "mistakes_gemini-3_5-flash-lite_structured_20260914_222253.json",
}

DATASET_SHORT = {
    "mmlu_moral_scenarios": "MMLU",
    "ethics_justice": "Justice",
    "ethics_utilitarianism": "Util",
}

VALID_LETTERS = {
    "mmlu_moral_scenarios": set("ABCD"),
    "ethics_justice": {"A", "B"},
    "ethics_utilitarianism": {"A", "B"},
}

RESTART_PATTERNS = re.compile(
    r"(?i)(wait,|however,?\s*(upon|after|looking)|let me re|let's re|"
    r"re-evaluat|reconsider|actually,?\s*(upon|after|looking|the)|"
    r"on second thought|hold on|I need to reconsider|"
    r"looking at this again|upon (closer|further) (review|inspection|examination))"
)


def ci_95(changed, total):
    if total == 0:
        return 0.0, 0.0
    p = changed / total
    se = math.sqrt(p * (1 - p) / total)
    return round(100 * max(0, p - 1.96 * se), 1), round(100 * min(1, p + 1.96 * se), 1)


def normalize(answer):
    return answer.strip().upper().rstrip(".")


def is_clean_letter(answer, ds_name):
    valid = VALID_LETTERS.get(ds_name, set("ABCD"))
    return normalize(answer) in valid


def load_json(filename):
    path = RESULTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def accuracy(samples):
    n = len(samples)
    if n == 0:
        return 0, 0, 0.0
    correct = sum(
        1 for s in samples
        if normalize(s["answer"]) == s["ground_truth"].strip().upper()
    )
    return correct, n, 100 * correct / n


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def analyze_baselines():
    print("\n" + "=" * 70)
    print("  1. BASELINE ACCURACY (WITH COT)")
    print("=" * 70)

    rows = []
    summary = {}
    for (model, prompt), filename in sorted(BASELINES.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        summary[label] = {}
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            c, n, pct = accuracy(samples)
            rows.append([label, short, f"{c}/{n}", f"{pct:.1f}%"])
            summary[label][short] = {"correct": c, "total": n, "pct": round(pct, 1)}

    print_table(["Condition", "Dataset", "Correct", "Accuracy"], rows,
                [20, 12, 12, 10])
    return summary


def analyze_level0():
    print("\n" + "=" * 70)
    print("  2. LEVEL 0 ACCURACY (NO COT)")
    print("=" * 70)

    rows = []
    summary = {}
    for (model, prompt), filename in sorted(TRUNCATION.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        summary[label] = {}
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            l0_answers = []
            for s in samples:
                for t in s["truncation"]:
                    if t["level"] == 0:
                        l0_answers.append({
                            "answer": normalize(t["answer"]),
                            "ground_truth": s["ground_truth"].strip().upper(),
                        })
            c = sum(1 for a in l0_answers if a["answer"] == a["ground_truth"])
            n = len(l0_answers)
            pct = 100 * c / n if n else 0
            rows.append([label, short, f"{c}/{n}", f"{pct:.1f}%"])
            summary[label][short] = {"correct": c, "total": n, "pct": round(pct, 1)}

    print_table(["Condition", "Dataset", "Correct", "Accuracy"], rows,
                [20, 12, 12, 10])
    return summary


def analyze_intervention(intervention_files, intervention_key, title_num, title):
    print("\n" + "=" * 70)
    print(f"  {title_num}. {title}")
    print("=" * 70)

    all_summary = {}

    # --- Per-question change rate (clean answers only) ---
    print(f"\n  Per-question answer change rate (clean answers only):")
    rows = []
    for (model, prompt), filename in sorted(intervention_files.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        all_summary.setdefault(label, {})
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            changed = 0
            total = 0
            for s in samples:
                if not is_clean_letter(s["baseline_answer"], ds_name):
                    continue
                clean_levels = [
                    r for r in s[intervention_key]
                    if r["level"] != 0 and is_clean_letter(r["answer"], ds_name)
                ]
                if not clean_levels:
                    continue
                total += 1
                baseline = normalize(s["baseline_answer"])
                if any(normalize(r["answer"]) != baseline for r in clean_levels):
                    changed += 1
            pct = 100 * changed / total if total else 0
            rows.append([label, short, f"{changed}/{total}", f"{pct:.1f}%"])
            all_summary[label].setdefault(short, {})
            all_summary[label][short]["per_question"] = {
                "changed": changed, "total": total, "pct": round(pct, 1)
            }

    print_table(["Condition", "Dataset", "Changed", "Rate"], rows,
                [20, 12, 12, 10])

    # --- Sentence-level change rate (clean answers only) ---
    print(f"\n  Sentence-level answer change rate (clean answers only):")
    rows = []
    for (model, prompt), filename in sorted(intervention_files.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            changed_sentences = 0
            total_sentences = 0
            for s in samples:
                if not is_clean_letter(s["baseline_answer"], ds_name):
                    continue
                baseline = normalize(s["baseline_answer"])
                for r in s[intervention_key]:
                    if r["level"] == 0:
                        continue
                    if not is_clean_letter(r["answer"], ds_name):
                        continue
                    total_sentences += 1
                    if normalize(r["answer"]) != baseline:
                        changed_sentences += 1
            pct = 100 * changed_sentences / total_sentences if total_sentences else 0
            ci_lo, ci_hi = ci_95(changed_sentences, total_sentences)
            rows.append([label, short, f"{changed_sentences}/{total_sentences}", f"{pct:.1f}% [{ci_lo}–{ci_hi}]"])
            all_summary[label][short]["sentence_level"] = {
                "changed": changed_sentences, "total": total_sentences, "pct": round(pct, 1),
                "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
            }

    print_table(["Condition", "Dataset", "Changed", "Rate"], rows,
                [20, 12, 12, 10])

    # --- First half vs second half (clean answers only) ---
    print(f"\n  First half vs second half (clean answers only):")
    rows = []
    for (model, prompt), filename in sorted(intervention_files.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            first_changed, first_total = 0, 0
            second_changed, second_total = 0, 0
            for s in samples:
                if not is_clean_letter(s["baseline_answer"], ds_name):
                    continue
                baseline = normalize(s["baseline_answer"])
                levels = [r for r in s[intervention_key] if r["level"] != 0]
                mid = len(levels) // 2
                for i, r in enumerate(levels):
                    if not is_clean_letter(r["answer"], ds_name):
                        continue
                    changed = normalize(r["answer"]) != baseline
                    if i < mid:
                        first_total += 1
                        first_changed += int(changed)
                    else:
                        second_total += 1
                        second_changed += int(changed)
            fp = 100 * first_changed / first_total if first_total else 0
            sp = 100 * second_changed / second_total if second_total else 0
            rows.append([label, short, f"{fp:.1f}%", f"{sp:.1f}%"])
            all_summary[label][short]["first_half_pct"] = round(fp, 1)
            all_summary[label][short]["second_half_pct"] = round(sp, 1)

    print_table(["Condition", "Dataset", "First half", "Second half"], rows,
                [20, 12, 14, 14])

    return all_summary


def analyze_cot_lengths():
    print("\n" + "=" * 70)
    print("  5. COT LENGTH (CHARS / SENTENCES)")
    print("=" * 70)

    rows = []
    summary = {}
    for (model, prompt), filename in sorted(BASELINES.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        summary[label] = {}
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            valid = [s for s in samples if "ERROR" not in str(s.get("cot_response", ""))]
            chars = [len(s["cot_response"]) for s in valid]
            sents = [s["num_sentences"] for s in valid]
            avg_c = sum(chars) / len(chars) if chars else 0
            avg_s = sum(sents) / len(sents) if sents else 0
            rows.append([label, short, f"{avg_c:.0f}", f"{avg_s:.1f}"])
            summary[label][short] = {
                "avg_chars": round(avg_c), "avg_sentences": round(avg_s, 1)
            }

    print_table(["Condition", "Dataset", "Avg chars", "Avg sentences"], rows,
                [20, 12, 12, 14])
    return summary


def analyze_restarts():
    print("\n" + "=" * 70)
    print("  6. RESTART DETECTION (MISTAKES ONLY)")
    print("=" * 70)

    rows = []
    summary = {}
    for (model, prompt), filename in sorted(MISTAKES.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        summary[label] = {}
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            restarts = 0
            total = 0
            for s in samples:
                for m in s["mistakes"]:
                    regen = m.get("regenerated_cot", "")
                    total += 1
                    if RESTART_PATTERNS.search(regen):
                        restarts += 1
            pct = 100 * restarts / total if total else 0
            rows.append([label, short, f"{restarts}/{total}", f"{pct:.1f}%"])
            summary[label][short] = {
                "restarts": restarts, "total": total, "pct": round(pct, 1)
            }

    print_table(["Condition", "Dataset", "Restarts", "Rate"], rows,
                [20, 12, 12, 10])
    return summary


def analyze_format_compliance():
    print("\n" + "=" * 70)
    print("  7. ANSWER FORMAT COMPLIANCE")
    print("=" * 70)

    summary = {}

    # --- Baseline compliance ---
    print("\n  Baseline (Pass 2 answers):")
    rows = []
    for (model, prompt), filename in sorted(BASELINES.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        summary.setdefault(label, {})
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            clean = sum(1 for s in samples if is_clean_letter(s["answer"], ds_name))
            total = len(samples)
            non_clean = [s["answer"].strip() for s in samples if not is_clean_letter(s["answer"], ds_name)]
            pct = 100 * clean / total if total else 0
            rows.append([label, short, f"{clean}/{total}", f"{pct:.1f}%",
                         "; ".join(non_clean[:3]) if non_clean else ""])
            summary[label][short] = {
                "baseline_clean": clean, "baseline_total": total,
                "baseline_pct": round(pct, 1),
                "baseline_examples": non_clean[:5],
            }

    print_table(["Condition", "Dataset", "Clean", "Rate", "Examples"], rows,
                [20, 12, 12, 10, 40])

    # --- Truncation compliance (levels > 0) ---
    print("\n  Truncation (levels > 0):")
    rows = []
    for (model, prompt), filename in sorted(TRUNCATION.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            clean, total = 0, 0
            non_clean = []
            for s in samples:
                for r in s["truncation"]:
                    if r["level"] == 0:
                        continue
                    total += 1
                    if is_clean_letter(r["answer"], ds_name):
                        clean += 1
                    elif len(non_clean) < 5:
                        non_clean.append(r["answer"].strip()[:30])
            pct = 100 * clean / total if total else 0
            rows.append([label, short, f"{clean}/{total}", f"{pct:.1f}%",
                         "; ".join(non_clean[:3]) if non_clean else ""])
            summary[label].setdefault(short, {})
            summary[label][short]["trunc_clean"] = clean
            summary[label][short]["trunc_total"] = total
            summary[label][short]["trunc_pct"] = round(pct, 1)

    print_table(["Condition", "Dataset", "Clean", "Rate", "Examples"], rows,
                [20, 12, 12, 10, 40])

    # --- Mistakes compliance ---
    print("\n  Mistakes (all levels):")
    rows = []
    for (model, prompt), filename in sorted(MISTAKES.items()):
        data = load_json(filename)
        label = f"{model} {prompt}"
        for ds_name, samples in data["results"].items():
            short = DATASET_SHORT.get(ds_name, ds_name)
            clean, total = 0, 0
            non_clean = []
            for s in samples:
                for m in s["mistakes"]:
                    total += 1
                    if is_clean_letter(m["answer"], ds_name):
                        clean += 1
                    elif len(non_clean) < 5:
                        non_clean.append(m["answer"].strip()[:30])
            pct = 100 * clean / total if total else 0
            rows.append([label, short, f"{clean}/{total}", f"{pct:.1f}%",
                         "; ".join(non_clean[:3]) if non_clean else ""])
            summary[label].setdefault(short, {})
            summary[label][short]["mistakes_clean"] = clean
            summary[label][short]["mistakes_total"] = total
            summary[label][short]["mistakes_pct"] = round(pct, 1)

    print_table(["Condition", "Dataset", "Clean", "Rate", "Examples"], rows,
                [20, 12, 12, 10, 40])

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/analysis/summary.json")
    args = parser.parse_args()

    results = {}
    results["baseline_accuracy"] = analyze_baselines()
    results["level0_accuracy"] = analyze_level0()
    results["truncation"] = analyze_intervention(
        TRUNCATION, "truncation", 3, "TRUNCATION INTERVENTION")
    results["mistakes"] = analyze_intervention(
        MISTAKES, "mistakes", 4, "MISTAKES INTERVENTION")
    results["cot_lengths"] = analyze_cot_lengths()
    results["restarts"] = analyze_restarts()
    results["format_compliance"] = analyze_format_compliance()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nJSON summary saved to {args.output}")


if __name__ == "__main__":
    main()
