# Audit Report: RB + SA Answer-or-Abstain System

Author: Saisab Sadhu

This covers `sa-module-v2/`, the only part of the shared materials with runnable code. KUQ, UnknownBench, and the Gemini/Qwen3 runs are discussed only from their result files, since no source code for them was shared.

## 1. What the code does

SA is a Self-Awareness layer on top of ReasoningBank (RB), an existing agent-memory framework. Each training question is put to Qwen2.5-7B-Instruct once at temperature 0, with a system prompt that already tells it to hedge on subjective or unanswerable questions. The response is labelled by a deterministic function that checks it against roughly forty hedge phrases and crosses that against the dataset's own `answerable` flag, giving one of four outcomes: `CORRECT`, `KNOWLEDGE_BOUNDARY_MISSED` (attempted something unanswerable), `FALSE_IDK` (declined something answerable), or `FACTUAL_ERROR` (attempted, wrong). This is rule-based, never asked of the model.

Each episode produces two memories: a generic RB "strategy" note, and an SA "lesson" written with a prompt specific to the failure type, the failure type itself being handed to the prompt rather than inferred. Episodes accumulate in a low-level store, and every twenty of them the system groups by failure type and a free-text domain tag the model generated, promoting any pattern that recurs three or more times into a standing guideline injected into every future prompt. Testing runs the same held-out questions under three cumulative conditions — a bare baseline, `with_memory` (top-3 RB notes by embedding similarity), and `with_memory_sa` (top-2 RB notes, top-2 retrieved SA failure episodes, and all active guidelines) — and scores each response with the same detector, computing F1 with "should have abstained" as the positive class. `run_falseqa.py` mirrors this almost exactly, built around false premises instead.

## 2. Pipeline, dataset to metric

Data arrives pre-split into train and test JSON with no code to regenerate that split. Training — ask, label, extract, store — is the only point where ground truth touches the system; testing then repeats across the three conditions on the same fixed set and aggregates into accuracy, precision, recall, F1, and a Wilson interval. Worth flagging directly: the same keyword detector both generates the training signal and scores the test outcome, with no independent judge anywhere in the loop, and F1 here only scores the abstain/answer decision, not whether an attempted answer is actually correct, so the headline number says nothing about answer quality on its own.

## 3. Assumptions

The system assumes a forty-phrase keyword list reliably tells genuine uncertainty from confident hedging, and trusts that same detector to be both the evaluation ground truth and the training signal. It assumes comparing `with_memory` against `with_memory_sa` attributes the whole difference to SA, which isn't true given their retrieval depths differ (§4.3). It assumes a single F1 at one operating point is enough to call a model better calibrated, where the selective-prediction literature (Kamath, Jia & Liang, 2020) generally prefers a risk–coverage curve for exactly this reason. And it assumes the train/test split and the "unanswerable" label set are themselves clean, which §4.6 shows doesn't fully hold.

## 4. Problems, weak points, methodological risks

The most consequential finding is that the reported gain looks like the model simply hedging more, not discriminating better. Per-item SelfAware counts, baseline → RB → RB+SA, run 184/162/57/97 → 190/130/110/70 → 206/99/141/54 for CORRECT/FACTUAL_ERROR/FALSE_IDK/KNOWLEDGE_BOUNDARY_MISSED — FALSE_IDK nearly triples, and overall abstention rises from ~23% to ~48%. The same shape recurs across all four headline datasets, where answerable-side accuracy collapses even as F1 rises (83.5%→59.2% on SelfAware, 62.2%→50.0% on FalseQA, 82.7%→60.4% on KUQ, 94.0%→58.9% on UnknownBench), and the two largest claimed gains sit exactly where baseline recall on the abstain class was weakest — where indiscriminate over-abstention pays off most. A likely mechanism: the lessons and guidelines injected into the prompt are phrased using the same hedge vocabulary the detector looks for, so a primed model can satisfy the scorer by echoing surface form rather than reasoning correctly. This was a hypothesis from their logs; §6a/§6b test it directly and both datasets are consistent with it — RB+SA is not statistically distinguishable from a placebo fed the same content retrieved for the wrong question.

Separately, the RB-versus-RB+SA comparison isn't a clean ablation, since `with_memory` retrieves three RB strategies while `with_memory_sa` retrieves only two, so the reported "SA uplift" also silently changes RB's retrieval depth. And the two largest claimed gains, on KUQ and UnknownBench, ship with no code at all — the README lists `run_unknownbench.py` but it doesn't exist, nor does `run_kuq.py`, so there's no way to inspect how those numbers were produced. The same is true, more broadly, for eight of the "eleven out of twelve" reported model-dataset combinations: the checkpoints folders show the four datasets were also run on Gemini and Qwen3, but no source code for those runs exists anywhere, and going through the raw checkpoints turned up the likely regression — Gemini on FalseQA, F1 falling from 0.443 to 0.391 under RB+SA.

Several result files never made it into the README at all. `abstentionbench_summary.json` reports F1 of zero in every condition, the model having abstained precisely 0 out of 300 times regardless — notable because AbstentionBench (Meta FAIR, [arXiv:2506.09038](https://arxiv.org/abs/2506.09038)) is a real, current benchmark built specifically for this problem, with KUQ as one of its own sub-benchmarks. `hle_summary.json` shows accuracy bit-identical across all three conditions; `awarebench_summary.json` shows RB+SA doing worse than baseline; and `selfaware_v3_summary.json` reruns the same benchmark to a different F1 (0.524 against the reported 0.506), with an extra condition that underperforms everything else. None of this makes the reported numbers false, but "11/12 improved" undercounts how many combinations were actually run.

A few smaller things: the SelfAware unanswerable set is mostly genuine subjective or philosophical questions, but includes corrupted GSM8K word problems (one switches from "bears" to "the dogs" mid-sentence) that are unanswerable only because the sentence doesn't parse. The setup instructions don't reproduce as written — flask versus llama-server naming inconsistencies, a `simple_server.py` and a `download_selfaware` module that both don't exist. The factual-correctness check is a loose substring match that doesn't touch the headline F1 but does corrupt which lessons get extracted. Every episode's confidence is hardcoded to 0.5, yet `SelfModel` computes a "calibration gap" from it regardless, meaninglessly. And on the reassuring side, SelfAware's train and test sets have zero overlapping questions; FalseQA has one duplicate out of 2838/1892, negligible but not literally zero.

## 5. Proposed robustness check

The open question is whether the RB+SA gain comes from the relevance of retrieved failure lessons, the system's actual claim, or whether any failure-lesson-shaped content produces the same shift because the real effect is general hedge-priming. The check adds a fourth condition, `with_memory_sa_scrambled`, identical to `with_memory_sa` — same RB depth, same guidelines — except SA episodic retrieval is queried with a different, unrelated test question, paired through a fixed derangement so no question retrieves its own lessons. It's a single-variable change, kept deliberately smaller than a fuller design I considered (a naive "hedge more" control, a matched-abstention-rate comparison, a full risk–coverage curve), per the brief's own preference for a focused check over a broad one. If the two conditions are statistically indistinguishable under a paired test, that points at presence rather than relevance; if `with_memory_sa` wins clearly, that supports the claimed mechanism.

## 6. What I implemented

Both run scripts now have a `build_derangement` function and a `with_memory_sa_scrambled` branch identical to `with_memory_sa` except for the SA retrieval query, plus a paired, continuity-corrected `mcnemar_test` since every condition runs on the same ordered test set. Results now report abstention rate per condition and McNemar between the real and scrambled SA conditions, and against baseline for context; `run_selfaware.py` also gained `--train-limit`/`--test-limit` flags defaulting to the paper's own 300/500 (commits [`0674b21`](https://github.com/saisabsadhu/self-awareness/commit/0674b21), [`161a800`](https://github.com/saisabsadhu/self-awareness/commit/161a800)). No memory-bank checkpoint was shared, so this rebuilds RB/SA memory from scratch, which doubles as an independent reproduction check before the new condition is added. Since the shared setup doesn't run as-is, I stood up `vllm serve Qwen/Qwen2.5-7B-Instruct` directly with `--served-model-name` matching what `llm_client.py` expects, so the existing client needed no changes.

### 6a. Results — SelfAware

Full scale, train=300, test=500, memory rebuilt from scratch:

| Condition | F1 | Precision | Recall | Answerable Acc | Unanswerable Acc |
|---|---|---|---|---|---|
| baseline | 0.427 | 0.492 | 0.377 | 82.7% | 37.7% |
| RB only | 0.502 | 0.455 | 0.558 | 70.2% | 55.8% |
| RB+SA | 0.524 | 0.442 | 0.643 | 63.9% | 64.3% |
| RB+SA, scrambled | 0.495 | 0.422 | 0.597 | 63.6% | 59.7% |

Paired McNemar: RB+SA vs. scrambled, p=0.428 (not significant); RB+SA vs. baseline, p=0.045 (significant); RB vs. baseline, p=0.203 (not significant).

Our from-scratch numbers sit close to their reported 0.425/0.483/0.506, so the effect direction is real. But RB+SA beats the scrambled placebo by only 0.029 F1, not distinguishable from noise at n=500, while RB+SA against baseline is significant — the relevance claim isn't what this data supports, a general effect of adding failure-lesson content is. One caveat: scrambling only touched per-episode retrieval, not the twelve standing guidelines, identical in both conditions, so this doesn't rule out the guidelines carrying real weight.

### 6b. Results — FalseQA

Full scale, train=300, test=300:

| Condition | F1 | Precision | Recall | False-premise Acc | True-premise Acc |
|---|---|---|---|---|---|
| baseline | 0.673 | 0.621 | 0.735 | 73.5% | 62.8% |
| RB only | 0.676 | 0.534 | 0.919 | 91.9% | 33.5% |
| RB+SA | 0.729 | 0.584 | 0.971 | 97.1% | 42.7% |
| RB+SA, scrambled | 0.726 | 0.575 | 0.985 | 98.5% | 39.6% |

Paired McNemar: RB+SA vs. scrambled, p=0.719 (not significant); RB+SA vs. baseline, p=1.000 — not distinguishable from doing nothing at all; RB vs. baseline, p=0.017, the only comparison that clears significance.

The reproduction again lands close to reported figures, and the result is if anything stronger evidence against the relevance claim than SelfAware's. Across both datasets with runnable code, the claim that relevant retrieval drives the gain doesn't survive direct testing, though a real, modest effect of adding memory-derived content in general does appear present.

## 7. How to reproduce

```bash
cd sa-module-v2
mkdir -p data memory_bank results
cp <path-to>/selfaware_train.json <path-to>/selfaware_test.json data/

CUDA_VISIBLE_DEVICES=<free_gpu> vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct --port 8080 \
  --dtype bfloat16 --gpu-memory-utilization 0.25 --max-model-len 4096

python3 src/run_selfaware.py --train-limit 300 --test-limit 500   # full scale
python3 src/run_selfaware.py --skip-scrambled                     # original 3 conditions only

cp <path-to>/falseqa_train.json <path-to>/falseqa_test.json data/
python3 src/run_falseqa.py                                        # limits set in __main__, default 300/300
```

## 8. A complementary research direction

The only signal for "did the model express uncertainty" is a lexical match on surface text, and that same signal generates the lessons fed back into future prompts — a loop shown above to be exploitable. Both RB+SA and Nhat's clustering/routing idea work entirely in this text space and can't independently check whether the detector is being fooled. A complementary direction would use a model-internal uncertainty signal that doesn't depend on what the model says out loud. Semantic entropy (Farquhar et al., *Nature* 2024, [10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0)) samples several generations and computes entropy over semantically clustered answers rather than surface wording, and Semantic Entropy Probes ([Kossen et al. 2024](https://arxiv.org/abs/2406.15927)) approximate this cheaply from a single generation's hidden states. This would let one check whether the lexical detector agrees with semantic entropy on the same responses, replace the hardcoded confidence placeholder with a real estimate, and eventually move from a single F1 point to a proper risk–coverage curve.

## 9. Questions I'd want to clarify

Whether an internal ablation has ever compared retrieval-relevant against retrieval-irrelevant SA content, since my own placebo couldn't statistically distinguish the two on either runnable dataset, and whether KUQ/UnknownBench tell a different story. Where the code for KUQ, UnknownBench, and the Gemini/Qwen3 runs actually is. Whether `abstentionbench`, `hle`, `awarebench`, and `selfaware_v3` were left out of the README because they're known-broken or simply not yet written up. And whether "improve abstain-class F1" is really the target metric, or whether something that penalises over-abstention symmetrically would better capture what "more trustworthy" is meant to mean here.

## AI tool usage

Used Claude (Claude Code) throughout — reading the code and result files, running the literature searches cited above, implementing the ablation, running the experiments, and drafting this report. Every finding was independently verified against the actual files and logs, and the choice of what to check and how to read the results was mine.
