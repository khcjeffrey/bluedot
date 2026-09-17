"""Run mistakes intervention on baseline samples and save results."""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from datetime import datetime
from pathlib import Path

from functools import partial
from utils import call_gemini, segment_cot
from prompts import COT_INSTRUCTION, COT_INSTRUCTION_STRUCTURED
from mistakes import run_mistakes

RESULTS_DIR = Path("results")


def find_baseline(model, prompt_style="freeform"):
    model_short = model.replace(".", "_")
    files = sorted(RESULTS_DIR.glob(f"cot_{model_short}_{prompt_style}_*.json"))
    if not files:
        files = sorted(RESULTS_DIR.glob(f"cot_{model_short}_*.json"))
    return files[-1] if files else None


def save_results(results, out_path):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--corrupt-model", default="gemini-3.5-flash-lite",
                        help="Model used for corruption (default: gemini-3.5-flash-lite)")
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--prompt-style", default="freeform",
                        choices=["freeform", "structured"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing output file to resume from")
    args = parser.parse_args()

    cot_instruction = COT_INSTRUCTION if args.prompt_style == "freeform" else COT_INSTRUCTION_STRUCTURED

    baseline_path = Path(args.baseline) if args.baseline else find_baseline(args.model, args.prompt_style)
    with open(baseline_path) as f:
        baseline = json.load(f)

    call_api = partial(call_gemini, model=args.model)
    corrupt_api = partial(call_gemini, model=args.corrupt_model)
    model_short = args.model.replace(".", "_")

    print(f"Baseline: {baseline_path}")
    print(f"Model: {args.model}")

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
        out_path = RESULTS_DIR / f"mistakes_{model_short}_{args.prompt_style}_{timestamp}.json"
        results = {
            "metadata": {
                "baseline_file": str(baseline_path),
                "intervention": "mistakes",
                "model": args.model,
                "timestamp": timestamp,
            },
            "results": {},
        }
        done_ids = {}

    for dataset_name, samples in baseline["results"].items():
        print(f"\n{'='*60}")
        print(f"  {dataset_name}")
        print(f"{'='*60}")

        max_samples = args.samples if args.samples else len(samples)
        dataset_results = results["results"].setdefault(dataset_name, [])
        skip_ids = done_ids.get(dataset_name, set())

        for i, sample in enumerate(samples[:max_samples]):
            if sample["id"] in skip_ids:
                print(f"\n--- Sample {i+1} (id={sample['id']}) --- SKIPPED")
                continue

            sample["cot_sentences"] = segment_cot(sample["cot_response"])
            sample["num_sentences"] = len(sample["cot_sentences"])
            n = sample["num_sentences"]
            baseline_answer = sample["answer"].strip()
            print(f"\n--- Sample {i+1} (id={sample['id']}, "
                  f"{n} sentences, baseline={baseline_answer}) ---")

            mistakes = run_mistakes(call_api, sample, corrupt_api=corrupt_api, cot_instruction=cot_instruction)

            for r in mistakes:
                match = "=" if r["answer"].upper().startswith(baseline_answer[0].upper()) else "!"
                print(f"  Level {r['level']:2d}/{n}: {r['answer'][:20]:20s} {match}= baseline  "
                      f"| corrupt: {r['corrupted_sentence'][:50]}")

            dataset_results.append({
                "id": sample["id"],
                "question": sample["question"],
                "ground_truth": sample.get("ground_truth", ""),
                "baseline_answer": baseline_answer,
                "num_sentences": n,
                "mistakes": mistakes,
            })

            save_results(results, out_path)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
