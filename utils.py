import os
import re
import time

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

from prompts import COT_INSTRUCTION, ANSWER_PROMPT


def segment_cot(cot_text):
    raw = sent_tokenize(cot_text)
    # Split off trailing numbered markers (e.g., "intro\n\n1." -> "intro", "1.")
    split = []
    for s in raw:
        match = re.search(r'\n(\d+\.)$', s)
        if match and len(s[:match.start()].strip()) >= 10:
            split.append(s[:match.start()])
            split.append(match.group(1))
        else:
            split.append(s)
    # Merge short segments (<10 chars) into next sentence
    merged = []
    for s in split:
        if merged and len(merged[-1]) < 10:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    if len(merged) > 1 and len(merged[-1]) < 10:
        merged[-2] = merged[-2] + " " + merged.pop()
    return merged


def call_gemini(prompt, model="gemini-3.5-flash-lite", max_retries=3):
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    if isinstance(prompt, list):
        contents = [
            {"role": role, "parts": [{"text": text}]}
            for role, text in prompt
        ]
    else:
        contents = prompt

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={"temperature": 0},
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1 and ("503" in str(e) or "UNAVAILABLE" in str(e) or "500" in str(e)):
                wait = 2 ** (attempt + 2)
                print(f"  Retry {attempt+1}/{max_retries-1} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def call_groq(prompt, model="llama-3.3-70b-versatile"):
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


PROVIDERS = {
    "gemini": call_gemini,
    "groq": call_groq,
}


def two_pass(call_api, question, answer_format, cot_instruction=None):
    """Pass 1: generate CoT. Pass 2: feed CoT back (chat format) and extract answer."""
    if cot_instruction is None:
        cot_instruction = COT_INSTRUCTION
    cot_prompt = f"{question}\n\n{cot_instruction}"
    cot = call_api(cot_prompt)
    time.sleep(2)

    answer = call_api([
        ("user", cot_prompt),
        ("model", cot),
        ("user", f"{ANSWER_PROMPT} {answer_format}"),
    ])

    return cot, answer
