# CoT Faithfulness in Ethical Reasoning

Extending [Lanham et al. 2023 ("Measuring Faithfulness in Chain-of-Thought Reasoning")](https://arxiv.org/abs/2307.13702) to investigate whether LLMs' chain-of-thought reasoning is more or less faithful on ethically charged questions compared to neutral domains.

## Research Questions

1. **Ethical faithfulness gap:** Do models show different CoT faithfulness on ethical/social reasoning tasks compared to the neutral tasks studied in the original paper?
2. **Evaluation awareness:** Does model awareness of being evaluated (via system prompt framing) affect CoT faithfulness, and does this interact with ethical content?

## Methodology

Following the paper's two-pass approach:
- **Pass 1:** Generate free-form CoT reasoning (with "Let's think step by step")
- **Pass 2:** Feed the CoT back to the model and prompt for a final answer only

CoT responses are segmented into sentences using NLTK's `sent_tokenize`, matching the paper's granularity.

### Interventions (planned)
- **Truncation (early answering):** Progressively truncate CoT at each sentence boundary and measure answer stability
- **Adding mistakes:** Use an LLM to corrupt individual CoT sentences, resample the CoT tail, and measure answer changes

### Evaluation awareness conditions (planned)
- **Evaluated:** System prompt signals the model's reasoning is being assessed
- **Unmonitored:** Naturalistic task framing with no mention of evaluation

## Datasets

| Dataset | Subset | Size | Task |
|---|---|---|---|
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | moral_scenarios | 895 | Multiple choice (A/B/C/D): which scenario is morally wrong |
| [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) | justice | 2,704 | Binary (JUST/UNJUST): is the described action just |
| [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) | utilitarianism | 4,807 | Binary (A/B): which scenario is more pleasant |

## Progress

### Step 1: Validate CoT generation (complete)
- Confirmed all three datasets produce rich, multi-step CoT reasoning
- Two-pass approach implemented and working with Gemini 3.5 Flash Lite
- 40 samples per dataset (120 total) generated successfully
- Average CoT lengths: ~11-16 sentences per response
- Baseline accuracy and format compliance measured via `analyze_results.py`

### Step 2: Experiment design (in progress)
- Intervention approach scoped: truncation first, adding mistakes second
- Cost model established: `m x (2 + 4n)` API calls per question (m=samples, n=avg sentences)
- Evaluation awareness dimension: system prompt manipulation (evaluated vs. naturalistic framing)
- Target: m=20 CoT chains per question at non-zero temperature

### Step 3: Cost estimation and scaling (upcoming)
- Estimate total API calls and token costs based on Step 1 sentence length data
- Determine question count per dataset within ~$200 budget

## Scripts

| Script | Purpose |
|---|---|
| `explore_datasets.py` | Preview dataset structures and sample examples |
| `generate_cot.py` | Two-pass CoT generation with NLTK segmentation |
| `analyze_results.py` | Analyze results: accuracy, sentence lengths, format compliance |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install google-generativeai python-dotenv datasets nltk
cp .env.example .env  # Add your GEMINI_API_KEY
```

## Running

```bash
# Generate CoT (default: 2 samples per dataset)
python generate_cot.py
python generate_cot.py --samples 40

# Analyze latest results
python analyze_results.py
python analyze_results.py results/cot_gemini_20260831_220105.json
```
