# Self-Awareness Module v2 — Code

## Overview

SA module on top of ReasoningBank. Tested on 4 knowledge boundary benchmarks.

## Results

| Dataset | Baseline F1 | RB F1 | RB+SA F1 | SA uplift |
|---------|-------------|-------|----------|-----------|
| SelfAware | 0.425 | 0.483 | 0.506 | +0.024 |
| KUQ | 0.265 | 0.336 | 0.474 | +0.138 |
| FalseQA | 0.680 | 0.674 | 0.736 | +0.062 |
| UnknownBench | 0.481 | 0.551 | 0.821 | +0.271 |

## File Structure

```
src/
├── llm_client.py           # LLM client (talks to flask server on port 8080)
├── memory.py               # Core data structures: MemoryItem, SAMemoryItem,
│                           # SAMemoryBank (two-level awareness memory), SelfModel
├── extraction/
│   └── extractor.py        # MemoryExtractor (RB) + SAExtractor (SA)
│                           # Key: failure-type-specific prompts
├── retrieval/
│   └── retriever.py        # MemoryRetriever + SARetriever
│                           # Key: CORRECT episodes filtered from SA retrieval
├── judge/
│   └── evaluator.py        # Math judge (for math experiments)
├── run_selfaware.py        # SelfAware experiment (F1=0.506)
├── run_falseqa.py          # FalseQA experiment (F1=0.736)
└── run_unknownbench.py     # UnknownBench experiment (F1=0.821)
```

## Key Design Decisions

1. **Deterministic failure typing** — computed from evaluation outcomes, not asked from LLM
2. **Failure-type-specific prompts** — different prompt for each failure type (mirrors RB's success/failure split)
3. **Two-level awareness memory** — low-level episodic + high-level policies (promoted after 3+ recurrences)
4. **CORRECT episode filtering** — only failure episodes retrieved during testing
5. **Standing guidelines** — high-level policies always injected into system prompt

## Setup

```bash
# Install dependencies
pip install transformers flask torch sentence-transformers datasets tqdm openai

# Start flask server (Qwen2.5-7B-Instruct)
python simple_server.py  # separate terminal

# Run experiment
python src/run_selfaware.py
python src/run_falseqa.py
python src/run_unknownbench.py
```

## Model

- **LLM:** Qwen2.5-7B-Instruct
- **Embedding:** Qwen/Qwen3-Embedding-0.6B
