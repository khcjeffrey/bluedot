import os
import time

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

from prompts import COT_INSTRUCTION, ANSWER_PROMPT


def segment_cot(cot_text):
    return sent_tokenize(cot_text)


def call_gemini(prompt, model="gemini-3.5-flash-lite"):
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client = genai.GenerativeModel(model)
    response = client.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0),
    )
    return response.text


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


def two_pass(call_api, question, answer_format):
    """Pass 1: generate CoT. Pass 2: feed CoT back and extract answer."""
    cot_prompt = f"{question}\n\n{COT_INSTRUCTION}"
    cot = call_api(cot_prompt)
    time.sleep(15)

    answer_prompt = f"{question}\n\n{cot}\n\n{ANSWER_PROMPT} {answer_format}"
    answer = call_api(answer_prompt)

    return cot, answer
