"""Explore dataset structures and print sample examples from each."""

import json
from datasets import load_dataset

ETHICS_BASE = "hf://datasets/hendrycks/ethics/data"


def preview_mmlu(n=2):
    print(f"\n{'='*60}")
    print("cais/mmlu (moral_scenarios)")
    print(f"{'='*60}")

    ds = load_dataset("cais/mmlu", "moral_scenarios", split="test")
    print(f"Size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    for i in range(n):
        print(f"\n--- Example {i} ---")
        print(json.dumps(ds[i], indent=2))


def preview_ethics(subset, n=2):
    print(f"\n{'='*60}")
    print(f"hendrycks/ethics ({subset})")
    print(f"{'='*60}")

    ds = load_dataset("csv", data_files=f"{ETHICS_BASE}/{subset}/test.csv", split="train")
    print(f"Size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    for i in range(n):
        print(f"\n--- Example {i} ---")
        print(json.dumps(ds[i], indent=2))


if __name__ == "__main__":
    preview_mmlu()
    preview_ethics("utilitarianism")
    preview_ethics("justice")
