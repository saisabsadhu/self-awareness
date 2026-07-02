# Audit Report: RB + SA Answer-or-Abstain System

Author: Saisab Sadhu

Scope: `sa-module-v2/` — only SelfAware and FalseQA ship with runnable code. KUQ and UnknownBench, and the Gemini/Qwen3 runs, are discussed only from their result files, since no source code for them exists anywhere in the shared materials.

## 1. What the code does

SA is a Self-Awareness layer built on top of ReasoningBank (RB), an existing memory framework for agents. For a benchmark like SelfAware, the flow is:

- **Ask** — each training question is put to Qwen2.5-7B-Instruct once, at temperature 0, with a system prompt that already tells it to hedge on subjective/unanswerable questions.
- **Label deterministically** — the response is checked against ~40 hedge phrases ("i don't know", "it depends", "no consensus"). Crossed with the dataset's ground-truth `answerable` flag, this assigns one of four outcomes: `CORRECT`, `KNOWLEDGE_BOUNDARY_MISSED` (answered something unanswerable), `FALSE_IDK` (declined something answerable), `FACTUAL_ERROR` (attempted, wrong). This is rule-based — never asked of the model.
- **Extract two memories per episode** — a generic RB "strategy" note, and an SA "lesson" using a prompt specific to the failure type (a `FALSE_IDK` lesson is framed as "why you should have attempted this"; a `KNOWLEDGE_BOUNDARY_MISSED` lesson as "why this was unanswerable"). The failure type is given to the extraction prompt, not inferred by it.
- **Store at two levels** — episodes accumulate in a low-level store. Every 20 episodes, the system groups by `(failure_type, domain)` — domain being a free-text one-word tag the model generated during extraction — and promotes any group recurring ≥3 times into a standing guideline, injected into every future prompt rather than only retrieved on similarity.
- **Test under three conditions**, cumulatively: `baseline` (bare question); `with_memory` (+ top-3 RB notes by cosine similarity, `Qwen3-Embedding-0.6B`); `with_memory_sa` (+ top-2 RB notes + top-2 retrieved SA *failure* episodes, successes filtered out + all active standing guidelines).
- **Score** — the same detector labels the response again; F1 treats "should have abstained" as the positive class.

`run_falseqa.py` mirrors this almost line for line, built around false premises instead of unanswerable questions.

## 2. Pipeline: dataset → metric

```
train/test json (pre-split, no code to regenerate the split)
   → train pass: ask → deterministic-label → extract RB note + SA lesson → store
     (only place ground truth touches the system)
   → consolidate every 20 episodes → promote recurring (failure_type, domain) → guideline
   → test pass ×3 conditions, same fixed set, temperature 0
   → per-item: ask (condition's context) → deterministic-label (same detector)
   → aggregate: accuracy, per-class accuracy, P/R/F1 (abstain = positive), Wilson CI
   → results/*_summary.json → README
```

Two things worth flagging directly: the same keyword detector generates the training signal *and* scores the test outcome, with no independent judge anywhere in the loop; and F1 here scores only the abstain/answer decision, not whether an attempted answer is factually right (a `FACTUAL_ERROR` still counts as the correct decision). Defensible given the brief's narrow framing, but it means the headline number says nothing about answer quality.

## 3. Assumptions

- A ~40-phrase keyword match reliably tells "genuine uncertainty" from "not."
- That same detector is trustworthy enough to be *both* the eval ground truth and the training signal.
- Comparing `with_memory` vs. `with_memory_sa` in isolation attributes the whole delta to SA — not true given the retrieval depths differ (§4.3).
- A single F1 at one operating point is a sufficient summary of "better calibrated" (the selective-prediction literature — e.g. Kamath, Jia & Liang 2020 — uses a risk–coverage curve for exactly this reason).
- The train/test split and the "unanswerable" label set are clean and representative (§4.6 shows this doesn't fully hold).
- LLM-generated one-word `domain` tags are consistent enough for domain-grouping to mean anything.

## 4. Problems, weak points, methodological risks

Ordered by how much each changes what the headline numbers mean.

**4.1 The reported gain looks like "hedges more," not "discriminates better."** Per-item SelfAware counts, baseline → RB → RB+SA: CORRECT 184→190→206, FACTUAL_ERROR 162→130→99, FALSE_IDK 57→110→**141**, KNOWLEDGE_BOUNDARY_MISSED 97→70→54. FALSE_IDK nearly triples; overall abstention rate goes from ~23% to ~48%. Same shape across all four headline datasets — answerable-side accuracy collapses even as F1 rises:

| Dataset | Answerable/known/true-premise accuracy, baseline → RB+SA |
|---|---|
| SelfAware | 83.5% → 59.2% |
| FalseQA (true-premise) | 62.2% → 50.0% |
| KUQ (known) | 82.7% → 60.4% |
| UnknownBench (answer) | 94.0% → 58.9% |

F1 only scores the abstain side, so this cost is invisible in the reported number. The two largest gains (KUQ +0.138, UnknownBench +0.271) are exactly where baseline recall on the abstain class was worst — where indiscriminate over-abstention pays off most. This was a hypothesis from their logs; §6a/§6b test it directly on two datasets and both are consistent with it — RB+SA is not statistically distinguishable from a placebo with the same SA content retrieved for the wrong question (SelfAware p=0.428; FalseQA p=0.719, where RB+SA isn't even distinguishable from baseline, p=1.000).

**4.2 The detector, the training signal, and the injected content share a vocabulary.** Lessons and guidelines are phrased using the same hedge words the scorer looks for ("say I don't know", "cannot be determined"). A model primed with that phrasing can satisfy the detector by echoing surface form, independent of actual reasoning. A plausible mechanism for §4.1, not a coincidence.

**4.3 RB vs. RB+SA isn't a clean ablation.** `with_memory` retrieves `top_k=3` RB strategies; `with_memory_sa` retrieves `top_k=2` (SA content fills the rest). "SA uplift" (RB+SA − RB) silently changes RB retrieval depth too, so the delta can't be attributed to SA alone.

**4.4 KUQ and UnknownBench — the largest claimed gains — ship with no code.** README lists `run_unknownbench.py`; it doesn't exist. No `run_kuq.py` either. Can't inspect how those two numbers were produced.

**4.5 Only 2 of the "11/12" model×dataset combinations are reproducible.** `checkpoints/` and `openrouter_experiments/` show the same 4 datasets also run on Gemini and Qwen3 (3 models × 4 datasets = 12). No source code anywhere for those runs. Likely candidate for the one regression: Gemini on FalseQA, F1 0.443 → 0.391 under RB+SA.

**4.6 Benchmark runs that don't fit the story are absent from the README.** Sitting in the results folder, unmentioned:
- `abstentionbench_summary.json` — F1 = 0 in all three conditions (0/300 abstentions, every condition). AbstentionBench (Meta FAIR, [arXiv:2506.09038](https://arxiv.org/abs/2506.09038)) is a real, current 20-dataset benchmark for exactly this problem, and KUQ is one of its own sub-benchmarks — arguably the most relevant existing benchmark for this claim, and the result is silently zero.
- `hle_summary.json` — accuracy bit-identical (53/200) across all three conditions.
- `awarebench_summary.json` — RB+SA is *worse* than baseline (−7.5 pts).
- `selfaware_v3_summary.json` — a rerun of the same benchmark giving a different F1 (0.524 vs. reported 0.506), plus a `with_memory_sa_selfmodel` condition that's worse than everything else.

Doesn't make the reported numbers false, but "11/12 improved" undercounts how many combinations were run.

**4.7 Some "unanswerable" labels are corrupted, not genuinely unanswerable.** The SelfAware-sourced unanswerable set is mostly genuine (subjective/philosophical, per Yin et al. 2023). Mixed in: corrupted GSM8K problems, e.g. a question about bears that switches to "the dogs" mid-sentence, or "for every unknown pounds they recycled" — apparent scripted substitution to manufacture synthetic unanswerable items, done with visible errors. These are unanswerable because the sentence doesn't parse, not because they test epistemic calibration.

**4.8 Setup instructions don't reproduce as written.** README says "flask server"; `llm_client.py`'s own comment says "llama-server"; `simple_server.py` (referenced by README) doesn't exist; `flask` isn't installed. `run_selfaware.py::main()` also imports `download_selfaware`, which doesn't exist either. Worked around (§6), but the project doesn't run from a clean checkout.

**4.9 Minor:**
- `FACTUAL_ERROR` vs. `CORRECT` uses `gold.lower() in response.lower()` — loose substring match, doesn't touch headline F1 but corrupts which lessons get extracted.
- `confidence` is hardcoded to `0.5` for every episode ("not trusting LLM's self-report"), yet `SelfModel` computes a "calibration gap" from it — meaningless wherever used, and the one place it's wired in (`with_memory_sa_selfmodel`, `selfaware_v3`) performs worst of all conditions.
- 9 failure types declared, only 4 ever produced.
- SelfAware train/test: 0 overlapping questions (2021/1348) — clean. FalseQA: 1 duplicate out of 2838/1892 — negligible but not zero.

## 5. Proposed robustness check

**Question:** does the RB+SA gain come from the *relevance* of retrieved failure lessons (the system's actual claim), or would any failure-lesson-shaped content produce the same shift, because the real mechanism is just "hedge more in general" (§4.1–4.2)?

**Check:** a fourth condition, `with_memory_sa_scrambled` — identical to `with_memory_sa` (same RB `top_k=2`, same standing guidelines) except SA episodic retrieval is queried with a *different, unrelated* test question (fixed derangement, no question retrieves its own lessons). Single-variable change: content volume and phrasing stay fixed, only relevance moves.

Deliberately smaller than the fuller design considered (a naive "hedge more" instruction control, a matched-abstention-rate comparison, a full risk–coverage curve — noted as follow-ups in §7/§8), per the brief's own "small is fine" guidance. If `with_memory_sa` and the scrambled version are statistically indistinguishable (paired McNemar, same fixed test set), that's evidence the gain isn't coming from relevance. If `with_memory_sa` wins significantly, that's evidence for the actual mechanism.

## 6. What I implemented

In `sa-module-v2/src/run_selfaware.py` and `run_falseqa.py` (commits [`0674b21`](https://github.com/saisabsadhu/self-awareness/commit/0674b21), [`161a800`](https://github.com/saisabsadhu/self-awareness/commit/161a800)):

- `build_derangement(n, seed=42)` — fixed, reproducible no-fixed-point permutation pairing each test question with an unrelated one.
- A `with_memory_sa_scrambled` branch in the test loop, identical to `with_memory_sa` except for the SA retrieval query.
- `mcnemar_test()` — paired, continuity-corrected, since every condition runs on the same ordered test set.
- Per-condition abstention rate + McNemar between `with_memory_sa` vs. scrambled (the core comparison), plus `with_memory_sa` vs. baseline and `with_memory` vs. baseline for context.
- `--train-limit`/`--test-limit` CLI flags on `run_selfaware.py`, defaulting to the original 300/500.

No memory-bank checkpoint was shared, so this run rebuilds RB/SA memory from scratch — which doubles as an independent reproduction check of the reported baseline/RB/RB+SA numbers before the new condition is added. Practical note: since the shared setup doesn't run as-is (§4.8), I stood up `vllm serve Qwen/Qwen2.5-7B-Instruct` with `--served-model-name` matching `llm_client.py`'s default, so the existing client works unmodified.

### 6a. Results — SelfAware

Full scale (train=300, test=500, matching the paper's setup), memory rebuilt from scratch:

| Condition | F1 | Precision | Recall | Answerable Acc | Unanswerable Acc | Abstention Rate |
|---|---|---|---|---|---|---|
| baseline | 0.427 | 0.492 | 0.377 | 82.7% | 37.7% | 23.6% |
| RB only | 0.502 | 0.455 | 0.558 | 70.2% | 55.8% | 37.8% |
| **RB+SA** | **0.524** | 0.442 | 0.643 | 63.9% | 64.3% | 44.8% |
| **RB+SA, scrambled** | **0.495** | 0.422 | 0.597 | 63.6% | 59.7% | 43.6% |

Paired McNemar (n=500):

| Comparison | b | c | χ² | p |
|---|---|---|---|---|
| RB+SA vs. scrambled | 43 | 35 | 0.628 | **0.428 (n.s.)** |
| RB+SA vs. baseline | 54 | 78 | 4.008 | **0.045 (sig.)** |
| RB vs. baseline | 53 | 68 | 1.620 | 0.203 (n.s.) |

- Reproduction lands close to reported (0.427/0.502/0.524 vs. their 0.425/0.483/0.506) — the direction of the effect is real, not a fluke.
- RB+SA beats the scrambled placebo by only 0.029 F1, not distinguishable from noise at n=500 — while RB+SA vs. baseline *is* significant. The relevance claim isn't supported; a general "add failure-lesson content" effect is.
- Caveat: scrambling only touched per-episode retrieval, not the 12 standing guidelines (identical in both conditions). Doesn't rule out the guidelines carrying real weight — a natural next ablation.

### 6b. Results — FalseQA

Full scale (train=300, test=300):

| Condition | F1 | Precision | Recall | False-premise Acc | True-premise Acc | Refusal Rate |
|---|---|---|---|---|---|---|
| baseline | 0.673 | 0.621 | 0.735 | 73.5% | 62.8% | 53.7% |
| RB only | 0.676 | 0.534 | 0.919 | 91.9% | 33.5% | 78.0% |
| **RB+SA** | **0.729** | 0.584 | 0.971 | 97.1% | 42.7% | 75.3% |
| **RB+SA, scrambled** | **0.726** | 0.575 | 0.985 | 98.5% | 39.6% | 77.7% |

Paired McNemar (n=300):

| Comparison | b | c | χ² | p |
|---|---|---|---|---|
| RB+SA vs. scrambled | 17 | 14 | 0.129 | **0.719 (n.s.)** |
| RB+SA vs. baseline | 42 | 43 | 0.000 | **1.000 (n.s.)** |
| RB vs. baseline | 31 | 54 | 5.694 | 0.017 (sig.) |

- Reproduction again close (0.673/0.676/0.729 vs. reported 0.680/0.674/0.736).
- Stronger evidence against the relevance claim than SelfAware: gap to placebo is 0.003 F1, and RB+SA isn't even distinguishable from baseline here. Only plain RB vs. baseline is significant.
- The direction of the small real-vs-scrambled gap even flips between datasets (SA abstains *more* than scrambled on SelfAware, *less* on FalseQA) — consistent with noise, not a systematic relevance effect.

**Across both datasets with runnable code: the relevance claim doesn't hold up under direct test.** The data is consistent with a real but modest effect of adding memory-derived content in general (RB alone clears significance on both), on top of which the SA-specific mechanism doesn't clear a same-population placebo control.

## 7. How to reproduce

```bash
cd sa-module-v2
mkdir -p data memory_bank results
cp <path-to>/selfaware_train.json <path-to>/selfaware_test.json data/

CUDA_VISIBLE_DEVICES=<free_gpu> vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct --port 8080 \
  --dtype bfloat16 --gpu-memory-utilization 0.25 --max-model-len 4096

python3 src/run_selfaware.py --train-limit 300 --test-limit 500   # full scale, ~1-1.5hr on one H200
python3 src/run_selfaware.py --train-limit 100 --test-limit 150   # faster check
python3 src/run_selfaware.py --skip-scrambled                     # original 3 conditions only

cp <path-to>/falseqa_train.json <path-to>/falseqa_test.json data/
python3 src/run_falseqa.py                                        # limits set in __main__, default 300/300
```

Output: `results/selfaware_summary.json` / `results/falseqa_summary.json` (all 4 conditions, breakdown, McNemar), plus raw logs (`results/selfaware_results.json`, `checkpoints/falseqa/*_results.json`).

## 8. A complementary research direction

§6a/§6b point at a specific gap: the only signal for "did the model express uncertainty" is a lexical match on surface text, and that same signal generates the training lessons fed back into future prompts. This loop is exploitable — behaviour indistinguishable from genuine calibration under this detector was reproducible, on two datasets, with lessons retrieved for the wrong question. Both RB+SA and Nhat's clustering/routing idea operate entirely in this text space (surface phrasing, or embedding similarity between questions); neither looks inside the model, so neither can independently check whether the lexical detector is being fooled.

Complementary direction: a **model-internal uncertainty signal that doesn't depend on what the model chooses to say**, as an independent check on the lexical detector. Semantic entropy (Farquhar et al., *Nature* 2024, [10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0)) samples multiple generations, clusters by semantic equivalence, and computes entropy over clusters — genuine uncertainty produces semantically scattered answers even when each individual answer sounds confident. Semantic Entropy Probes ([Kossen et al. 2024](https://arxiv.org/abs/2406.15927)) approximate this cheaply from a single generation's hidden states.

This would let you: (1) check whether `detect_uncertainty` calls agree with semantic entropy on the same responses — frequent disagreement is direct evidence the keyword detector scores surface form, not genuine uncertainty; (2) replace the hardcoded `confidence=0.5` placeholder with a real, non-circular estimate, making `SelfModel`'s calibration-gap computation (currently meaningless, §4.9) actually meaningful; (3) eventually move from a single F1 point to a real risk–coverage curve (Kamath et al., 2020) — the standard way this problem is evaluated in the selective-prediction literature, and the only way to cleanly separate "better signal" from "moved along the same trade-off curve."

## 9. Questions for Niranjan

1. Has an internal ablation ever compared retrieval-relevant vs. retrieval-irrelevant SA content? My scrambled-retrieval placebo couldn't statistically distinguish RB+SA from a version fed lessons for the wrong question, on either runnable dataset (SelfAware p=0.428, n=500; FalseQA p=0.719, n=300 — where RB+SA wasn't even distinguishable from baseline, p=1.000). Does this match what you've seen internally, or do KUQ/UnknownBench tell a different story?
2. Where's the code for KUQ/UnknownBench, and for the Gemini/Qwen3 runs — oversight in what got shared, or intentionally out of scope here?
3. Was `abstentionbench` left out of the README because it's a known-broken eval, or because it wasn't written up yet? Same question for `hle`, `awarebench`, `selfaware_v3`.
4. Is "improve abstain-class F1" the actual target metric, or would something that penalizes over-abstention symmetrically (macro-F1 across both classes, or a risk–coverage curve) better capture "more trustworthy" for this project?

## AI tool usage

Used Claude (Claude Code) throughout — reading and cross-referencing the code/result files, running the literature searches cited above, implementing the ablation, running the experiments, and drafting this report. All findings were independently verified against the actual files/logs, not taken on trust from the model. The choice of which check to run and how to read the results was mine.
