import random

from datasets import load_dataset

from prompts import (
    format_mmlu_prompt, parse_mmlu_answer,
    format_justice_prompt, parse_justice_answer,
    format_utilitarianism_prompt,
)

ETHICS_BASE = "hf://datasets/hendrycks/ethics/data"
SEED = 42


def load_data(n=2):
    random.seed(SEED)
    datasets = {}

    # MMLU moral_scenarios
    ds = load_dataset("cais/mmlu", "moral_scenarios", split="test")
    indices = random.sample(range(len(ds)), n)
    datasets["mmlu_moral_scenarios"] = []
    for idx in indices:
        prompts = format_mmlu_prompt(ds[idx])
        datasets["mmlu_moral_scenarios"].append({
            "id": idx,
            "ground_truth": parse_mmlu_answer(ds[idx]),
            "raw": ds[idx],
            **prompts,
        })

    # Ethics justice
    ds = load_dataset("csv", data_files=f"{ETHICS_BASE}/justice/test.csv", split="train")
    indices = random.sample(range(len(ds)), n)
    datasets["ethics_justice"] = []
    for idx in indices:
        prompts = format_justice_prompt(ds[idx])
        datasets["ethics_justice"].append({
            "id": idx,
            "ground_truth": parse_justice_answer(ds[idx]),
            "raw": ds[idx],
            **prompts,
        })

    # Ethics utilitarianism
    ds = load_dataset("csv", data_files=f"{ETHICS_BASE}/utilitarianism/test.csv", split="train")
    indices = random.sample(range(len(ds)), n)
    datasets["ethics_utilitarianism"] = []
    for idx in indices:
        prompts = format_utilitarianism_prompt(ds[idx])
        correct = "A" if prompts["presented_order"][0] == ds[idx]["baseline"] else "B"
        datasets["ethics_utilitarianism"].append({
            "id": idx,
            "ground_truth": correct,
            "raw": ds[idx],
            **prompts,
        })

    return datasets
