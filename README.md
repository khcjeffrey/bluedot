# Measuring Chain-of-Thought Necessity and the Effect of Reasoning Constraints

Extending [Lanham et al. (2023)](https://arxiv.org/abs/2307.13702) to measure CoT necessity across ethical reasoning domains, and testing whether constraining CoT generation (e.g. with format templates) affects necessity.

Write-up: *[link to LessWrong post]*

## Key Findings

1. **Model size is the dominant factor.** Larger models depend far less on CoT, consistent with Lanham et al.'s findings on factual tasks.
2. **Constraining CoT generation may increase necessity.** Structured prompts reduce the model's control over its reasoning, and the model tends to be more dependent on its CoT content as a result.

## Datasets

| Dataset | Subset | Task | Samples |
|---|---|---|---|
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | moral_scenarios | Multiple choice: which scenario is morally wrong | 150 |
| [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) | justice | Binary: is the action just | 150 |
| [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) | utilitarianism | Binary: which scenario is more pleasant | 150 |

## Scripts

| Script | Purpose |
|---|---|
| `data.py` | Dataset loading |
| `prompts.py` | Prompt definitions (freeform and structured) |
| `utils.py` | API calls, two-pass setup, answer extraction |
| `generate_cot.py` | Pass 1: generate CoT reasoning |
| `truncation.py` | Truncation intervention logic |
| `mistakes.py` | Mistakes intervention logic |
| `test_truncation.py` | Run truncation experiments |
| `test_mistakes.py` | Run mistakes experiments |
| `analyze_all.py` | Comprehensive analysis with 95% CIs |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install google-generativeai python-dotenv datasets nltk
cp .env.example .env  # Add your GEMINI_API_KEY
```

## References

- Lanham et al. (2023). [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)
- Jia et al. (2025). [Faithfulness as Information Flow](https://arxiv.org/abs/2605.24286)
- Hendrycks et al. (2020). [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- Hendrycks et al. (2021). [Aligning AI With Shared Human Values](https://arxiv.org/abs/2008.02275)
