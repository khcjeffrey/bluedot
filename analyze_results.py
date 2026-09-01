"""
Analyze CoT generation results: sentence length distributions, accuracy, and answer format compliance.

Usage:
    python analyze_results.py
    python analyze_results.py results/cot_gemini_20260831_220105.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


RESULTS_DIR = Path("results")


def load_latest():
    files = sorted(RESULTS_DIR.glob("cot_*.json"))
    if not files:
        print("No results found in results/")
        sys.exit(1)
    return files[-1]


def analyze(path):
    with open(path) as f:
        data = json.load(f)

    analysis = {
        "source_file": str(path),
        "metadata": data["metadata"],
        "datasets": {},
    }

    print(f"File: {path}")
    print(f"Provider: {data['metadata']['provider']}")
    print(f"Samples per dataset: {data['metadata']['samples_per_dataset']}")
    print(f"Timestamp: {data['metadata']['timestamp']}")

    for dataset_name, samples in data["results"].items():
        print(f"\n{'='*60}")
        print(f"  {dataset_name}")
        print(f"{'='*60}")

        valid = [s for s in samples if "ERROR" not in str(s.get("cot_response", ""))]
        errors = len(samples) - len(valid)

        # Sentence counts
        sentence_counts = [s["num_sentences"] for s in valid]
        avg_sentences = sum(sentence_counts) / len(sentence_counts)
        print(f"\n  Samples: {len(valid)} valid, {errors} errors")
        print(f"  Sentences: min={min(sentence_counts)}, max={max(sentence_counts)}, avg={avg_sentences:.1f}")

        # Accuracy
        correct = sum(
            1 for s in valid
            if s["answer"].strip().upper().startswith(s["ground_truth"].upper())
        )
        print(f"  Accuracy: {correct}/{len(valid)} ({100*correct/len(valid):.1f}%)")

        # Answer format compliance
        expected_formats = {
            "mmlu_moral_scenarios": set("ABCD"),
            "ethics_justice": {"JUST", "UNJUST"},
            "ethics_utilitarianism": {"A", "B"},
        }
        compliant_count = len(valid)
        non_compliant = []
        if dataset_name in expected_formats:
            valid_answers = expected_formats[dataset_name]
            compliant_count = sum(
                1 for s in valid
                if s["answer"].strip().upper() in valid_answers
                or s["answer"].strip().upper()[:1] in valid_answers
            )
            non_compliant = [
                s["answer"].strip() for s in valid
                if s["answer"].strip().upper() not in valid_answers
                and s["answer"].strip().upper()[:1] not in valid_answers
            ]
            print(f"  Format compliance: {compliant_count}/{len(valid)} ({100*compliant_count/len(valid):.1f}%)")
            if non_compliant:
                print(f"  Non-compliant answers: {non_compliant[:5]}")

        # Answer distribution
        answers = Counter(s["answer"].strip() for s in valid)
        print(f"  Answer distribution: {dict(answers.most_common())}")

        # CoT length in chars
        cot_lengths = [len(s["cot_response"]) for s in valid]
        avg_cot_len = sum(cot_lengths) / len(cot_lengths)
        print(f"  CoT length (chars): min={min(cot_lengths)}, max={max(cot_lengths)}, avg={avg_cot_len:.0f}")

        analysis["datasets"][dataset_name] = {
            "valid_samples": len(valid),
            "errors": errors,
            "sentences": {
                "min": min(sentence_counts),
                "max": max(sentence_counts),
                "avg": round(avg_sentences, 1),
            },
            "accuracy": {
                "correct": correct,
                "total": len(valid),
                "pct": round(100 * correct / len(valid), 1),
            },
            "format_compliance": {
                "compliant": compliant_count,
                "total": len(valid),
                "pct": round(100 * compliant_count / len(valid), 1),
                "non_compliant_examples": non_compliant[:5],
            },
            "answer_distribution": dict(answers.most_common()),
            "cot_length_chars": {
                "min": min(cot_lengths),
                "max": max(cot_lengths),
                "avg": round(avg_cot_len),
            },
        }

    # Save analysis
    out_path = RESULTS_DIR / f"analysis_{path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n\nAnalysis saved to {out_path}")

    return analysis


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = load_latest()
    analyze(path)
