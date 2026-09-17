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
from functools import partial
from pathlib import Path

from data import load_data, SEED
from prompts import COT_INSTRUCTION, COT_INSTRUCTION_STRUCTURED
from utils import PROVIDERS, two_pass, segment_cot

RESULTS_DIR = Path("results")
SAMPLES_PER_DATASET = 40


def save_results(results, out_path):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="gemini", choices=PROVIDERS.keys())
    parser.add_argument("--model", default="gemini-3.5-flash-lite",
                        choices=["gemini-3.5-flash-lite", "gemini-3.5-flash"])
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_DATASET)
    parser.add_argument("--prompt-style", default="freeform",
                        choices=["freeform", "structured"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing output file to resume from")
    args = parser.parse_args()

    cot_instruction = COT_INSTRUCTION if args.prompt_style == "freeform" else COT_INSTRUCTION_STRUCTURED
    call_api = partial(PROVIDERS[args.provider], model=args.model)
    datasets = load_data(n=args.samples)

    RESULTS_DIR.mkdir(exist_ok=True)
    model_short = args.model.replace(".", "_")

    if args.resume:
        out_path = Path(args.resume)
        with open(out_path) as f:
            results = json.load(f)
        done_ids = {}
        for ds_name, ds_results in results["results"].items():
            done_ids[ds_name] = {r["id"] for r in ds_results}
        print(f"Resuming from {out_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"cot_{model_short}_{args.prompt_style}_{timestamp}.json"
        results = {
            "metadata": {
                "provider": args.provider,
                "model": args.model,
                "samples_per_dataset": args.samples,
                "seed": SEED,
                "timestamp": timestamp,
            },
            "results": {},
        }
        done_ids = {}

    for dataset_name, samples in datasets.items():
        print(f"\n{'='*60}")
        print(f"Generating CoT for: {dataset_name}")
        print(f"{'='*60}")

        dataset_results = results["results"].setdefault(dataset_name, [])
        skip_ids = done_ids.get(dataset_name, set())

        for i, sample in enumerate(samples):
            if sample["id"] in skip_ids:
                print(f"\n--- Sample {i+1}/{len(samples)} (id={sample['id']}) --- SKIPPED")
                continue

            print(f"\n--- Sample {i+1}/{len(samples)} (id={sample['id']}) ---")
            print(f"Ground truth: {sample['ground_truth']}")

            try:
                cot, answer = two_pass(
                    call_api, sample["question"], sample["answer_format"],
                    cot_instruction=cot_instruction,
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

            save_results(results, out_path)
            time.sleep(2)

    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
