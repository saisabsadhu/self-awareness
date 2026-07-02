# Audit Report: RB + SA Answer-or-Abstain System

Author: Saisab Sadhu
Scope: `sa-module-v2/` (SelfAware and FalseQA experiments — the two with runnable code), cross-checked against all result files shared in the Drive folder.

---

## 1. What the code does, step by step

The system builds a Self-Awareness (SA) layer on top of ReasoningBank (RB), an existing memory-for-agents method. For a given benchmark (e.g. SelfAware):

1. **Ask.** For each training question, the model (Qwen2.5-7B-Instruct) is prompted once, temperature 0, with a system prompt that already tells it to hedge on subjective/unanswerable questions (`run_selfaware.py::_ask_question`).
2. **Label the outcome deterministically.** `evaluate_selfaware()` checks the response for ~40 hedge phrases ("i don't know", "it depends", "no consensus", ...) via substring match (`detect_uncertainty`), crosses that against the dataset's ground-truth `answerable` flag, and assigns exactly one of four failure types: `CORRECT`, `KNOWLEDGE_BOUNDARY_MISSED` (answered something unanswerable), `FALSE_IDK` (abstained on something answerable), `FACTUAL_ERROR` (attempted, wrong answer). This labeling is rule-based, not judged by the model.
3. **Extract two kinds of memory from the same episode:**
   - An RB "strategy" note (generic, mirrors the original ReasoningBank success/failure prompt split).
   - An SA "lesson" (`SAExtractor.extract_sa_lesson`), using a *different prompt per failure type* — a `FALSE_IDK` lesson is phrased "here's why you should have attempted this," a `KNOWLEDGE_BOUNDARY_MISSED` lesson is phrased "here's why this was unanswerable." The failure type itself is given to the prompt, not asked of the model.
4. **Store at two levels.** Individual episodes go into `SAMemoryBank.episodes` (low level). Every 20 episodes, `consolidate()` groups episodes by `(failure_type, domain)` — domain being a free-text, one-word tag the model itself generated during extraction — and promotes any group with ≥3 occurrences into a standing guideline (high level), composed by a further LLM call.
5. **Test.** The same held-out questions run under three conditions, built cumulatively:
   - `baseline` — bare question, no memory.
   - `with_memory` — question + top-3 retrieved RB strategy notes (cosine similarity via `Qwen/Qwen3-Embedding-0.6B`).
   - `with_memory_sa` — question + top-2 RB notes + top-2 retrieved SA *failure* episodes (successes are filtered out of SA retrieval) + all active standing guidelines, all injected into the system prompt.
6. **Score.** `evaluate_selfaware()` runs again on each response; F1 is computed treating "should have abstained" as the positive class (precision penalizes over-abstention, recall penalizes under-abstention).

`run_falseqa.py` follows the identical structure, adapted to false-premise questions instead of unanswerable ones.

## 2. The pipeline end to end: dataset → metric

```
dataset json (train/test, pre-split, no code to regenerate the split)
   │
   ▼
train pass: ask → deterministic-label → extract RB note + SA lesson → store
   │              (this is the ONLY place ground truth touches the system)
   ▼
consolidate every 20 episodes → promote recurring (failure_type, domain) → standing guideline
   │
   ▼
test pass ×3 conditions, same fixed held-out set, temperature 0
   │
   ▼
per-item: ask (with condition's injected context) → deterministic-label (same detector)
   │
   ▼
aggregate: accuracy, per-class accuracy, precision/recall/F1 (abstain = positive class), Wilson CI
   │
   ▼
results/*_results.json (raw) + results/*_summary.json (aggregated) → reported in README
```

Two things worth being explicit about: (a) the *same* keyword-based labeling function is used both to generate the training signal and to score the test-time outcome — there's no independent judge; (b) F1 here measures only the abstain/answer *decision*, not whether an attempted answer is factually correct (a `FACTUAL_ERROR` still counts as "correct decision" in the F1 computation, since attempting was the right call). That's a defensible scope choice given the brief's narrow framing of the problem, but it means the headline number says nothing about answer quality.

## 3. Assumptions the system makes

- A ~40-phrase keyword/substring match reliably distinguishes "the model expressed genuine uncertainty" from "the model didn't."
- That same detector is trustworthy enough to serve as *both* the evaluation ground truth *and* the training signal that produces the lessons injected back into the prompt.
- Comparing the `with_memory` and `with_memory_sa` conditions in isolation attributes the entire difference to the SA module specifically (see §4.3 — this isn't actually true given how the code is written).
- A single F1 score, computed at one fixed operating point, is a sufficient summary of "the model is better calibrated." (The selective-prediction literature — e.g. Kamath, Jia & Liang, *Selective Question Answering Under Domain Shift*, 2020 — evaluates this class of problem with a risk–coverage curve precisely because a single point can't distinguish "better signal" from "moved along the same trade-off curve.")
- The train/test split and the "unanswerable" label set are themselves clean and representative of what the benchmark's authors intended (see §4.6 — this doesn't fully hold).
- LLM-generated one-word `domain` tags are consistent enough across episodes to make grouping-by-domain for the promotion mechanism meaningful.

## 4. Problems, weak points, and methodological risks

Ranked by how much they change what the headline numbers mean, most important first.

### 4.1 The reported gain is consistent with "the model just hedges more," not "the model got better at telling the two cases apart"

I pulled the raw per-item log (`Results/qwen2.5-7b-results/results/selfaware_results.json`). Going baseline → RB → RB+SA:

| | CORRECT | FACTUAL_ERROR | FALSE_IDK | KNOWLEDGE_BOUNDARY_MISSED |
|---|---|---|---|---|
| baseline | 184 | 162 | 57 | 97 |
| with_memory | 190 | 130 | 110 | 70 |
| with_memory_sa | 206 | 99 | **141** | 54 |

`FALSE_IDK` (wrongly abstaining on an answerable question) nearly triples. Overall abstention rate goes from ~23% to ~48% of all 500 test questions. The same shape shows up in every one of the four headline datasets — the "should-answer" side accuracy collapses under RB+SA even as the reported F1 goes up:

| Dataset | Answerable/known/true-premise accuracy: baseline → RB+SA |
|---|---|
| SelfAware | 83.5% → 59.2% |
| FalseQA (true-premise) | 62.2% → 50.0% |
| KUQ (known) | 82.7% → 60.4% |
| UnknownBench (answer) | 94.0% → 58.9% |

Because the F1 formula only scores the abstain side, this real cost is invisible in the single reported number. The two largest claimed gains (KUQ +0.138, UnknownBench +0.271) are exactly the datasets where baseline recall on the abstain class was worst — i.e., where "abstain more, indiscriminately" pays off most under this metric. This is the central competing explanation for the whole result. **This was a hypothesis when I first read their logs; §6a below tests it directly and the result is consistent with it** — RB+SA is not statistically distinguishable from a placebo with the same amount of failure-lesson content retrieved for the wrong question.

### 4.2 The detector, the training signal, and the injected content share a vocabulary

The lessons and standing guidelines that get injected into the prompt are themselves phrased using exactly the hedge words the scoring function looks for ("say 'I don't know'", "cannot be determined"). A model primed with that phrasing could satisfy the detector by echoing surface form, independent of whether it reasoned correctly about the specific question. This is a plausible mechanism for §4.1, not just a coincidence.

### 4.3 The RB vs. RB+SA comparison isn't a clean ablation

In both `run_selfaware.py` and `run_falseqa.py`, `with_memory` retrieves `top_k=3` RB strategies, while `with_memory_sa` retrieves `top_k=2` RB strategies (plus SA content). So "SA uplift" (RB+SA − RB) also silently changes the RB retrieval depth. The delta can't be attributed to the SA module alone as currently measured.

### 4.4 KUQ and UnknownBench — the two largest claimed gains — ship with no code

The README lists `run_unknownbench.py`; it does not exist anywhere in the shared folder. There is no `run_kuq.py` either. Only `run_selfaware.py` and `run_falseqa.py` are present and runnable. I cannot inspect how the other two headline numbers were produced.

### 4.5 Of the "11/12 model×dataset combinations," only 2 are reproducible from what was shared

`Results/checkpoints/` and `Results/openrouter_experiments/` show the same 4 datasets were also run on Gemini and Qwen3 (3 models × 4 datasets = 12, matching the email's "11 out of 12" claim). There is no source code anywhere for the Gemini/Qwen3 runs — no OpenRouter client, no run scripts, nothing beyond the raw checkpoint JSON. I also found the likely "1 of 12 that regressed": Gemini on FalseQA is *worse* under RB+SA than baseline (F1 0.443 → 0.391).

### 4.6 Benchmark runs that don't fit the reported story are absent from the README, not discussed

Sitting in the results folder, never mentioned in the README or brief:
- `abstentionbench_summary.json` — **F1 = 0 in all three conditions.** The model abstained 0/300 times, in every condition. AbstentionBench (Meta FAIR, [arXiv:2506.09038](https://arxiv.org/abs/2506.09038)) is a real, current, 20-dataset benchmark specifically built to stress-test abstention — and KUQ (one of this project's own headline datasets) is literally one of its constituent sub-benchmarks. This is arguably the single most relevant existing benchmark for this exact claim, and the result on it is silently zero.
- `hle_summary.json` — accuracy bit-for-bit identical (53/200 correct) across baseline, RB, and RB+SA. Zero uplift.
- `awarebench_summary.json` — RB+SA is *worse* than baseline (−7.5 points), only partially recovering the drop caused by RB itself.
- `selfaware_v3_summary.json` — a rerun of the *same* nominal benchmark giving a different F1 (0.524) than the one reported (0.506), plus an extra `with_memory_sa_selfmodel` condition that performs worse than everything else.

None of this makes the reported numbers false, but "11/12 combos improved" undercounts how many combos were actually run, and the ones that don't fit the story aren't discussed anywhere.

### 4.7 Data quality: some "unanswerable" training labels are corrupted, not genuinely unanswerable

The `SelfAware`-sourced unanswerable set is mostly genuine (Yin et al., ACL 2023 — subjective/philosophical/no-consensus questions like "Do truly good people exist?"). But mixed in are corrupted GSM8K word problems, e.g.: *"They put the **dogs** onto shelves..."* in a question that starts with **bears**, and *"For every **unknown** pounds they recycled..."* — apparently a scripted attempt to manufacture synthetic unanswerable math questions by substituting a quantity, done with visible substitution errors. These are trivially "unanswerable" because the sentence doesn't parse, not because they test epistemic calibration, which muddies what the unanswerable-accuracy numbers actually mean.

### 4.8 Setup instructions don't reproduce as written

README says to start "a flask server"; `llm_client.py`'s own comment says it's a "llama-server" client; `simple_server.py` (referenced by the README) doesn't exist anywhere in the shared folder; `flask` isn't even installed in the environment. `main()` in `run_selfaware.py` also references `download_selfaware.py`, which doesn't exist either. None of this is fatal (I worked around it — see §6), but as shared, the project doesn't start from a clean checkout.

### 4.9 Minor findings
- Answer-correctness checking for `FACTUAL_ERROR` vs `CORRECT` uses `any(gold.lower() in response.lower() for gold in answers)` — a loose substring containment that can misfire on short/common gold answers. This doesn't touch the headline F1 (which only scores the abstain decision) but does corrupt which lessons get extracted and promoted.
- `confidence` is a hardcoded `0.5` placeholder for every episode (`extractor.py`, with the comment "not trusting LLM's self-report"), yet `SelfModel.build()` computes a "calibration gap" from this constant — numerically meaningless wherever it's actually used (and `selfaware_v3`'s `with_memory_sa_selfmodel` condition, the one place this machinery is wired in, performs worse than everything else).
- `memory.py` declares 9 failure types; only 4 are ever produced by the evaluation functions actually used. The other 5 are unexercised code paths.
- Train/test overlap for SelfAware checked out clean — 0 shared questions between the 2021 train and 1348 test items in `openrouter_experiments/data/`. Worth stating since not everything I looked at was a problem.

---

## 5. Proposed robustness check

**Question:** is the RB+SA gain driven by the specific *relevance* of retrieved failure lessons to the question at hand (the system's actual mechanistic claim), or would any failure-lesson-shaped content produce a similar shift, because what's really happening is the model being primed to hedge more in general (§4.1–4.2)?

**Check:** add a fourth condition, `with_memory_sa_scrambled`, identical to `with_memory_sa` in every respect — same RB retrieval (`top_k=2`), same standing guidelines — except the SA episodic-lesson retrieval is queried with a *different, unrelated* test question (a fixed derangement of the test set, so no question ever retrieves its own lessons) instead of the real one. This is a minimal, single-variable change: it keeps the total amount of injected SA content, its surface phrasing, and everything else about the pipeline fixed, and isolates exactly the one claim that matters — retrieval relevance.

This is deliberately small in scope (per the brief's own guidance — "a focused check beats a broad one") rather than the fuller design I considered and cut: a naive "just hedge more" instruction-only control, a matched-abstention-rate comparison, and a full risk–coverage curve. Those are real follow-ups (noted in §7 and §8) but this one condition, plus proper paired statistics, directly adjudicates the sharpest open question with the least new surface area.

**Reading the result:** if `with_memory_sa` and `with_memory_sa_scrambled` are statistically indistinguishable (paired McNemar test on the same 500 items, since all conditions share the same fixed test set), that's strong evidence the gain is not coming from the lessons' relevance — it's coming from their mere presence. If `with_memory_sa` is significantly better, that's real evidence for the system's actual mechanism.

## 6. What I implemented

Changes are in `sa-module-v2/src/run_selfaware.py` and `sa-module-v2/src/run_falseqa.py` ([diff](https://github.com/saisabsadhu/self-awareness/commit/0674b21)):
- `build_derangement(n, seed=42)` — fixed, reproducible no-fixed-point permutation used to pair each test question with an unrelated one.
- A new `with_memory_sa_scrambled` branch in `run_test`, structurally identical to `with_memory_sa` except for the SA retrieval query.
- `mcnemar_test(correct_a, correct_b)` — paired, continuity-corrected McNemar's test, since every condition runs on the same ordered test set (a paired design, which a plain CI-overlap eyeball test ignores).
- `save_results()` now reports per-condition abstention rate and runs McNemar between `with_memory_sa` vs. `with_memory_sa_scrambled` (the core question), plus `with_memory_sa` vs. `baseline` and `with_memory` vs. `baseline` for context.
- `--train-limit` / `--test-limit` CLI args, defaulting to the paper's original 300/500, so the check can be run at full scale or a faster reduced scale.

I could not reuse the original memory-bank checkpoint used to produce the reported numbers — none was included in the shared materials (only final summaries and one raw per-item log survive). So this run rebuilds RB/SA memory from the training set from scratch, which also serves as an independent reproduction check of the reported baseline/RB/RB+SA SelfAware numbers, before adding the new condition on top.

**Practical note:** the shared setup instructions don't run as-is (§4.8). I stood up an actual OpenAI-compatible server via `vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct` on a free GPU, which lets the existing `llm_client.py` work completely unmodified (the served name matches its default).

### 6a. Results — SelfAware

Full-scale reproduction (train=300, test=500, matching the paper's own setup exactly), retrained from scratch since no memory-bank checkpoint was shared:

| Condition | F1 | Precision | Recall | Answerable Acc | Unanswerable Acc | Abstention Rate |
|---|---|---|---|---|---|---|
| baseline | 0.427 | 0.492 | 0.377 | 82.7% | 37.7% | 23.6% |
| RB only | 0.502 | 0.455 | 0.558 | 70.2% | 55.8% | 37.8% |
| **RB+SA** | **0.524** | 0.442 | 0.643 | 63.9% | 64.3% | 44.8% |
| **RB+SA, scrambled SA retrieval** | **0.495** | 0.422 | 0.597 | 63.6% | 59.7% | 43.6% |

Paired McNemar (all conditions share the same 500 items in the same order):

| Comparison | b (A right, B wrong) | c (A wrong, B right) | χ² | p-value |
|---|---|---|---|---|
| RB+SA vs. RB+SA-scrambled | 43 | 35 | 0.628 | **0.428 (not significant)** |
| RB+SA vs. baseline | 54 | 78 | 4.008 | **0.045 (significant)** |
| RB only vs. baseline | 53 | 68 | 1.620 | 0.203 (not significant) |

**Reading this:**
- The overall pattern (baseline < RB < RB+SA) replicates independently from scratch (0.427/0.502/0.524 vs. their reported 0.425/0.483/0.506) — the reported direction of the effect is real, not a one-off or fabricated run.
- RB+SA beats the scrambled-retrieval placebo by only 0.029 F1, and that gap is **not statistically distinguishable from noise** at n=500 (p=0.428) — while RB+SA vs. doing nothing *is* significant (p=0.045). The system's actual claim — that retrieving *relevant* past failures helps — is not what this data supports. What the data supports is that adding failure-lesson-shaped content to the prompt shifts behavior, largely independent of whether that content is about the question being asked.
- Caveat I want to be explicit about: the scrambling only touched per-episode retrieval, not the 12 standing guidelines promoted during training (identical in both `with_memory_sa` and the scrambled condition). This experiment doesn't tell us whether the guidelines themselves are doing real work — a natural next ablation would scramble or drop those too.

### 6b. Results — FalseQA

*(same scrambled-retrieval ablation, applied to `run_falseqa.py`; run in progress, numbers to follow)*

## 7. How to reproduce

```bash
cd sa-module-v2
mkdir -p data memory_bank results
cp <path-to>/selfaware_train.json <path-to>/selfaware_test.json data/

# Stand up a model server matching what llm_client.py expects:
CUDA_VISIBLE_DEVICES=<free_gpu> vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct --port 8080 \
  --dtype bfloat16 --gpu-memory-utilization 0.25 --max-model-len 4096

# Full scale (matches the paper's reported setup, ~1-1.5hr on one H200):
python3 src/run_selfaware.py --train-limit 300 --test-limit 500

# Faster check:
python3 src/run_selfaware.py --train-limit 100 --test-limit 150

# Skip the new condition (reproduce only the original 3):
python3 src/run_selfaware.py --skip-scrambled

# Same ablation on FalseQA (train/test limits are set in __main__, default 300/300):
cp <path-to>/falseqa_train.json <path-to>/falseqa_test.json data/
python3 src/run_falseqa.py
```
Output: `results/selfaware_summary.json` / `results/falseqa_summary.json` (all 4 conditions, full breakdown, McNemar tests), plus raw per-item logs (`results/selfaware_results.json`, `checkpoints/falseqa/*_results.json`).

## 8. A complementary research direction

The result in §6a points at a specific gap: the system's only signal for "did the model express uncertainty" is a ~40-phrase keyword match on the *surface text* of the response, and that same signal is used to generate the training lessons that get fed back into future prompts. §6a shows this loop is exploitable — behavior that looks identical to genuine calibration under this detector was reproducible with lessons retrieved for the wrong question. Both the current RB+SA mechanism and Nhat's clustering/routing idea operate entirely in this same text space (surface phrasing, or lexical/embedding similarity between questions); neither one looks at what's happening inside the model when it generates an answer, so neither can independently check whether the lexical detector is being fooled.

A complementary direction: use a **model-internal uncertainty signal that doesn't depend on what the model chooses to say**, as an independent channel to validate (or replace) the lexical detector. Concretely, semantic entropy (Farquhar et al., *Nature* 2024, [10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0)) samples multiple generations, clusters them by semantic equivalence, and computes entropy over the clusters — a question the model is genuinely uncertain about produces semantically scattered answers even if every individual answer sounds confident. Semantic Entropy Probes ([Kossen et al. 2024](https://arxiv.org/abs/2406.15927)) show this can be approximated cheaply from a single generation's hidden states, avoiding the cost of sampling many completions.

Concretely this would let you: (1) check whether the lexical `detect_uncertainty` calls agree with semantic entropy on the same responses — if they diverge often, that's direct evidence the keyword detector is scoring surface form rather than genuine uncertainty, which is exactly the mechanism §6a's null result is consistent with; (2) replace the hardcoded `confidence=0.5` placeholder with a real, non-circular estimate, making `SelfModel`'s calibration-gap computation (currently meaningless — see §4.9) actually meaningful; (3) eventually swap the single F1 operating point for a real risk–coverage curve (Kamath et al., 2020), which is the standard way this problem is evaluated in the selective-prediction literature and the only way to cleanly separate "better discriminative signal" from "moved along the same trade-off curve" — which is the exact ambiguity this whole audit has been circling.

## 9. Questions for Niranjan

1. Has an internal ablation ever compared retrieval-relevant vs. retrieval-irrelevant SA content? My scrambled-retrieval placebo (§6a) couldn't statistically distinguish RB+SA from a version fed lessons retrieved for the wrong question (p=0.428, n=500), while RB+SA vs. no-memory-at-all was significant (p=0.045). If that holds up on more datasets, it suggests most of the measured effect may come from the standing guidelines and/or a general increase in hedging rather than from retrieval relevance specifically — which would be a fairly different story from the one in the README.
2. Where's the code for KUQ/UnknownBench, and for the Gemini/Qwen3 runs? Was that an oversight in what got shared, or intentionally out of scope for this exercise?
3. Was `abstentionbench` excluded from the README because it's a known-broken eval, or because it hadn't been written up yet? Same question for `hle`, `awarebench`, and `selfaware_v3`.
4. Is "improve abstain-class F1" the metric you actually want optimized, or would a metric that penalizes over-abstention symmetrically (e.g. macro-F1 across both classes, or a risk–coverage curve) better reflect what "more trustworthy" means for this project?

## AI tool usage

Used Claude (Claude Code) throughout: reading and cross-referencing the code/result files, running the literature searches cited above, implementing the ablation script, running the experiment, and drafting this report. All findings were independently verified against the actual files/logs (not taken on trust from the model), and the choice of which check to prioritize and how to scope it was mine.
