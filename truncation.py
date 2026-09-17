"""
Truncation intervention: build truncated CoTs at each sentence boundary
and extract answers via Pass 2.
"""

import time

from prompts import ANSWER_PROMPT, COT_INSTRUCTION


def run_truncation(call_api, sample, cot_instruction=None):
    """Truncate CoT at each sentence level and get Pass 2 answers.

    Returns a list of {level, answer} dicts for levels 0 to n-1.
    Level 0 = no CoT, level n = full CoT (baseline, already in sample).
    """
    if cot_instruction is None:
        cot_instruction = COT_INSTRUCTION
    sentences = sample["cot_sentences"]
    n = len(sentences)
    results = []

    for level in range(n):
        truncated_cot = " ".join(sentences[:level])

        if level == 0:
            prompt = f"{sample['question']}\n\n{ANSWER_PROMPT} {sample['answer_format']}"
        else:
            prompt = [
                ("user", f"{sample['question']}\n\n{cot_instruction}"),
                ("model", truncated_cot),
                ("user", f"{ANSWER_PROMPT} {sample['answer_format']}"),
            ]

        answer = call_api(prompt)

        results.append({
            "level": level,
            "total_sentences": n,
            "answer": answer.strip(),
        })

        time.sleep(2)

    return results
