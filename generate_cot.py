"""
Step 1: Generate Chain-of-Thought responses on ethical reasoning datasets.

Two-pass approach (matching Lanham et al. 2023):
  Pass 1: Generate free-form CoT reasoning
  Pass 2: Feed CoT back to model, prompt for final answer

Outputs include NLTK sentence-segmented CoT for downstream interventions.

Usage:
    python generate_cot.py
    python generate_cot.py --provider groq
    python generate_cot.py --samples 3
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from data import load_data
from utils import PROVIDERS, two_pass, segment_cot

RESULTS_DIR = Path("results")
SAMPLES_PER_DATASET = 2


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="gemini", choices=PROVIDERS.keys())
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_DATASET)
    args = parser.parse_args()

    call_api = PROVIDERS[args.provider]
    datasets = load_data(n=args.samples)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "provider": args.provider,
            "samples_per_dataset": args.samples,
            "seed": SEED,
            "timestamp": timestamp,
        },
        "results": {},
    }

    for dataset_name, samples in datasets.items():
        print(f"\n{'='*60}")
        print(f"Generating CoT for: {dataset_name}")
        print(f"{'='*60}")

        dataset_results = []
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} (id={sample['id']}) ---")
            print(f"Ground truth: {sample['ground_truth']}")

            try:
                cot, answer = two_pass(
                    call_api, sample["question"], sample["answer_format"]
                )
                sentences = segment_cot(cot)
                print(f"CoT: {len(cot)} chars, {len(sentences)} sentences")
                print(f"Answer: {answer.strip()}")
            except Exception as e:
                cot, answer, sentences = f"ERROR: {e}", f"ERROR: {e}", []
                print(f"Error: {e}")

            dataset_results.append({
                "id": sample["id"],
                "question": sample["question"],
                "answer_format": sample["answer_format"],
                "ground_truth": sample["ground_truth"],
                "cot_response": cot,
                "cot_sentences": sentences,
                "num_sentences": len(sentences),
                "answer": answer.strip() if isinstance(answer, str) else answer,
            })

            time.sleep(15)

        results["results"][dataset_name] = dataset_results

    out_path = RESULTS_DIR / f"cot_{args.provider}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
