"""
Analyze truncation results: extract answer patterns and flag changes.

Usage:
    python analyze_truncation.py
    python analyze_truncation.py results/truncation_test_20260904_201633.json
"""

import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path("results")


def extract_letter(answer):
    clean = answer.strip().upper()
    if len(clean) == 1 and clean in "ABCD":
        return clean
    match = re.findall(r'\b([ABCD])\b', clean)
    return match[-1] if match else "?"


def load_latest():
    files = sorted(RESULTS_DIR.glob("truncation_test_*.json"))
    if not files:
        print("No truncation results found")
        sys.exit(1)
    return files[-1]


def analyze(path):
    with open(path) as f:
        data = json.load(f)

    print(f"File: {path}\n")

    analysis = {"source_file": str(path), "datasets": {}}
    total_stable, total_changed = 0, 0

    for dataset_name, samples in data["results"].items():
        print(f"{'='*60}")
        print(f"  {dataset_name}")
        print(f"{'='*60}")

        sample_results = []
        for sample in samples:
            baseline = extract_letter(sample["baseline_answer"])
            gt = sample.get("ground_truth", "").strip().upper()
            letters = [extract_letter(r["answer"]) for r in sample["truncation"]]
            n = len(letters)
            stable = all(l == baseline for l in letters)

            if stable:
                total_stable += 1
            else:
                total_changed += 1

            tag = "STABLE" if stable else "CHANGED"
            correct = "correct" if baseline == gt else "WRONG"
            print(f"  [{tag:7s}] id={sample['id']:<5d} baseline={baseline}[{correct}]  "
                  f"{','.join(letters)}")

            sample_results.append({
                "id": sample["id"],
                "baseline": baseline,
                "ground_truth": gt,
                "pattern": letters,
                "stable": stable,
            })

        analysis["datasets"][dataset_name] = sample_results

    total = total_stable + total_changed
    print(f"\nOverall: {total_stable}/{total} stable, {total_changed}/{total} changed")

    out_path = RESULTS_DIR / f"analysis_{path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else load_latest()
    analyze(path)
