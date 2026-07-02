# Audit Report: RB + SA Answer-or-Abstain System

Author: Saisab Sadhu

This covers `sa-module-v2/`, the only part of the shared materials with runnable code. KUQ, UnknownBench, and the Gemini/Qwen3 runs are discussed only from their result files below, since no source code for them exists anywhere in what was shared.

## 1. What the code does

SA is a Self-Awareness layer built on top of ReasoningBank (RB), an existing memory framework for agents. For a benchmark like SelfAware, each training question is put to Qwen2.5-7B-Instruct once at temperature 0, with a system prompt that already tells it to hedge on subjective or unanswerable questions. The response is then labelled by a deterministic function that checks it against roughly forty hedge phrases ("i don't know", "it depends", "no consensus") and crosses that against the dataset's own `answerable` flag, giving one of four outcomes: `CORRECT`, `KNOWLEDGE_BOUNDARY_MISSED` (attempted something unanswerable), `FALSE_IDK` (declined something answerable), or `FACTUAL_ERROR` (attempted, wrong). This labelling is rule-based, never asked of the model itself.

From each episode two memories get extracted: a generic RB "strategy" note, and an SA "lesson" written with a prompt specific to the failure type — a `FALSE_IDK` lesson is framed as "why you should have attempted this," a `KNOWLEDGE_BOUNDARY_MISSED` one as "why this was unanswerable," with the failure type handed to the prompt rather than inferred by it. Episodes accumulate in a low-level store, and every 20 of them the system groups by failure type and a free-text domain tag the model generated during extraction, promoting any group that has recurred three or more times into a standing guideline that gets injected into every future prompt rather than only retrieved on similarity.

Testing runs the same held-out questions under three conditions built cumulatively: a bare `baseline`; `with_memory`, which adds the top three RB strategy notes by cosine similarity (`Qwen3-Embedding-0.6B`); and `with_memory_sa`, which adds the top two RB notes, the top two retrieved SA failure episodes (successes are filtered out of SA retrieval), and every active standing guideline. The same detector scores the response again, and F1 is computed treating "should have abstained" as the positive class. `run_falseqa.py` follows this almost exactly, built around false premises instead of unanswerable questions.

## 2. Pipeline, dataset to metric

The data arrives pre-split into train and test JSON, with no code anywhere to regenerate that split. The training pass — ask, label, extract, store — is the only point where ground truth actually touches the system; everything downstream of it runs on the model's own retrieved memory. Testing then repeats across the three conditions on the same fixed set, always at temperature 0, and the results get aggregated into accuracy, per-class accuracy, precision/recall/F1, and a Wilson interval, which is what eventually lands in the README.

Two things worth being direct about. First, the same keyword detector generates the training signal and scores the test outcome — there's no independent judge anywhere in this loop. Second, F1 here scores only the decision to answer or abstain, not whether an attempted answer is actually correct, since a `FACTUAL_ERROR` still counts as the right decision. That's a defensible scope given how narrowly the brief frames the problem, but it means the headline number says nothing about answer quality on its own.

## 3. Assumptions

The system leans on a fairly long chain of assumptions. That a forty-phrase keyword list reliably tells genuine uncertainty from confident hedging. That the same detector can be trusted as both the evaluation ground truth and the training signal that produces the lessons fed back into the prompt. That comparing `with_memory` against `with_memory_sa` in isolation attributes the whole difference to SA, which isn't actually true given the retrieval depths differ between the two conditions (§4.3). That a single F1 at one operating point is enough to say a model is "better calibrated," when the selective-prediction literature — Kamath, Jia & Liang's 2020 paper on selective QA under domain shift, for instance — generally uses a risk–coverage curve for exactly this reason, since one point can't tell a genuinely better signal from a system that simply moved along the same trade-off curve. And that the train/test split and the "unanswerable" label set are themselves clean, which §4.6 shows doesn't fully hold.

## 4. Problems, weak points, methodological risks

Ordered by how much each one changes what the headline numbers actually mean.

The most consequential finding is that the reported gain looks a lot like the model simply hedging more, rather than getting better at telling the two cases apart. The raw per-item SelfAware counts, going baseline → RB → RB+SA, are 184/162/57/97 for CORRECT/FACTUAL_ERROR/FALSE_IDK/KNOWLEDGE_BOUNDARY_MISSED under baseline, 190/130/110/70 under RB, and 206/99/141/54 under RB+SA — FALSE_IDK, a wrongly declined but perfectly answerable question, nearly triples, and overall abstention climbs from about 23% to about 48%. The same shape shows up in every headline dataset: answerable-side accuracy collapses even as F1 rises, from 83.5% to 59.2% on SelfAware, 62.2% to 50.0% on FalseQA's true-premise side, 82.7% to 60.4% on KUQ's known questions, 94.0% to 58.9% on UnknownBench. Since F1 only scores the abstain side, this cost is invisible in the reported number, and it's telling that the two largest claimed gains — KUQ at +0.138 and UnknownBench at +0.271 — are exactly the datasets where baseline recall on the abstain class was weakest, which is where abstaining more indiscriminately pays off most under this metric. This started as a hypothesis from reading their logs; §6a and §6b below test it directly on two datasets and both are consistent with it, showing RB+SA is not statistically distinguishable from a placebo fed the same amount of failure-lesson content retrieved for the wrong question (SelfAware p=0.428, FalseQA p=0.719, where RB+SA isn't even distinguishable from baseline, p=1.000).

A plausible reason for this sits right underneath it: the lessons and standing guidelines injected into the prompt are phrased using the same hedge vocabulary the scoring function looks for, things like "say I don't know" or "cannot be determined." A model primed with that phrasing can satisfy the detector simply by echoing its surface form, regardless of whether it reasoned correctly about the question in front of it.

Separately, the comparison between RB and RB+SA isn't actually a clean ablation. `with_memory` retrieves the top three RB strategies, while `with_memory_sa` retrieves only the top two, making up the difference with SA content — so the "SA uplift" quietly changes RB's retrieval depth at the same time, and the observed delta can't be attributed to SA on its own as currently measured.

The two largest claimed gains, on KUQ and UnknownBench, also ship with no code at all. The README lists `run_unknownbench.py`, which doesn't exist anywhere in the folder, and there's no `run_kuq.py` either — only `run_selfaware.py` and `run_falseqa.py` are actually runnable, so there's no way to inspect how those two numbers were produced. Widening the lens further, the checkpoints and openrouter_experiments folders show the same four datasets were also run against Gemini and Qwen3, which together with Qwen2.5-7B accounts for the "eleven out of twelve" figure in the email. There's no source code anywhere for those Gemini or Qwen3 runs, and going through the raw checkpoints turned up what's likely the one combination that didn't improve: Gemini on FalseQA, where F1 actually drops from 0.443 to 0.391 under RB+SA.

There are also several result files sitting in the folder that never made it into the README. One, `abstentionbench_summary.json`, reports F1 of zero in all three conditions because the model abstained precisely zero times out of three hundred, every time — and AbstentionBench (Meta FAIR, [arXiv:2506.09038](https://arxiv.org/abs/2506.09038)) is a real, current, twenty-dataset benchmark built specifically to stress-test abstention, with KUQ as one of its own sub-benchmarks, which arguably makes it the single most relevant existing benchmark for this claim. Another, `hle_summary.json`, shows accuracy bit-identical at 53/200 across all three conditions, zero uplift of any kind. A third, `awarebench_summary.json`, shows RB+SA doing worse than baseline by 7.5 points. And a fourth, `selfaware_v3_summary.json`, is a rerun of the same nominal benchmark giving a different F1, 0.524 against the reported 0.506, plus an extra `with_memory_sa_selfmodel` condition that underperforms everything else in the file. None of this makes the reported numbers false, but "11/12 improved" undercounts how many combinations were actually run.

The unanswerable label set has its own quality problem. Most of the SelfAware-sourced unanswerable questions are genuine subjective or philosophical ones, but mixed in are corrupted GSM8K word problems — one starts with bears and switches to "the dogs" mid-sentence, another asks how many points were earned "for every unknown pounds" recycled — apparently a scripted attempt to manufacture synthetic unanswerable questions by substituting a number, done with visible errors. These are unanswerable because the sentence doesn't parse, not because they test any real epistemic boundary.

The setup instructions don't reproduce as written either: the README says to start "a flask server," `llm_client.py`'s own comment calls it a "llama-server" client, `simple_server.py` doesn't exist anywhere, flask isn't installed, and `run_selfaware.py::main()` imports a `download_selfaware` module that also doesn't exist. None of this is fatal — worked around it, see §6 — but the project doesn't run from a clean checkout as shared.

A handful of smaller things round this out. The check for whether an attempted answer is factually correct is a loose substring match, `gold.lower() in response.lower()`, which doesn't touch the headline F1 but does corrupt which lessons get extracted. Every episode's `confidence` field is hardcoded to 0.5, with a comment saying the model's self-reported confidence isn't trusted, and yet `SelfModel` computes a "calibration gap" from that same constant — meaningless wherever it's used, and the one place it actually is used, `with_memory_sa_selfmodel` in `selfaware_v3`, performs worse than every other condition in that file. Nine failure types are declared in the code but only four are ever produced. And on the reassuring side, SelfAware's train and test sets have zero overlapping questions out of 2021 and 1348 respectively; FalseQA has one duplicate out of 2838 and 1892, which is negligible but not literally zero.

## 5. Proposed robustness check

The open question is whether the RB+SA gain comes from the relevance of retrieved failure lessons to the actual question, which is the system's own claimed mechanism, or whether any failure-lesson-shaped content would produce much the same shift because the real effect is just general hedge-priming. The check adds a fourth condition, `with_memory_sa_scrambled`, identical to `with_memory_sa` in every respect — same RB retrieval depth, same standing guidelines — except the SA episodic retrieval is queried using a different, unrelated test question, the pairing fixed through a derangement so no question ever retrieves its own lessons. It's a single-variable change: content volume and phrasing stay fixed, only relevance moves.

This was kept deliberately smaller than a fuller design I considered, which included a naive "just hedge more" instruction control, a version matched for overall abstention rate, and a full risk–coverage curve — real follow-ups, noted again below, but this one condition plus a proper paired significance test answers the sharpest open question with the least new surface area. If `with_memory_sa` and the scrambled version turn out indistinguishable under a paired test, that points at presence rather than relevance. If `with_memory_sa` wins clearly, that's real evidence for the claimed mechanism.

## 6. What I implemented

Both `run_selfaware.py` and `run_falseqa.py` now have a `build_derangement` function producing a fixed, reproducible, no-fixed-point permutation, and a `with_memory_sa_scrambled` branch in the test loop that's identical to `with_memory_sa` except for the SA retrieval query. A paired, continuity-corrected `mcnemar_test` was added since every condition runs on the same ordered test set, which makes it a paired design that a plain comparison of confidence intervals would misrepresent. Results now report abstention rate per condition and McNemar between `with_memory_sa` and the scrambled version, plus `with_memory_sa` and `with_memory` against baseline for context. `run_selfaware.py` also gained `--train-limit`/`--test-limit` flags, defaulting to the paper's own 300/500. These changes are commits [`0674b21`](https://github.com/saisabsadhu/self-awareness/commit/0674b21) and [`161a800`](https://github.com/saisabsadhu/self-awareness/commit/161a800) in the repo.

No memory-bank checkpoint was shared, so this rebuilds RB/SA memory from the training set from scratch, which usefully doubles as an independent reproduction check of the reported numbers before the new condition gets added. Since the shared setup doesn't run as-is (§4.8), I stood up `vllm serve Qwen/Qwen2.5-7B-Instruct` directly, with `--served-model-name` set to match what `llm_client.py` expects by default, so the existing client code needed no changes at all.

### 6a. Results — SelfAware

Full scale, train=300 and test=500, matching the paper's own setup, memory rebuilt from scratch:

| Condition | F1 | Precision | Recall | Answerable Acc | Unanswerable Acc | Abstention Rate |
|---|---|---|---|---|---|---|
| baseline | 0.427 | 0.492 | 0.377 | 82.7% | 37.7% | 23.6% |
| RB only | 0.502 | 0.455 | 0.558 | 70.2% | 55.8% | 37.8% |
| RB+SA | 0.524 | 0.442 | 0.643 | 63.9% | 64.3% | 44.8% |
| RB+SA, scrambled | 0.495 | 0.422 | 0.597 | 63.6% | 59.7% | 43.6% |

Paired McNemar on the same 500 items: RB+SA against the scrambled version gives b=43, c=35, χ²=0.628, p=0.428, not significant; RB+SA against baseline gives b=54, c=78, χ²=4.008, p=0.045, significant; RB alone against baseline gives p=0.203, not significant.

The overall pattern replicates independently — our from-scratch numbers, 0.427/0.502/0.524, sit close to their reported 0.425/0.483/0.506 — so the direction of the effect is real rather than a one-off. But RB+SA beats the scrambled placebo by only 0.029 F1, and that gap isn't distinguishable from noise at this sample size, while RB+SA against doing nothing at all is significant. The relevance claim isn't what this data supports; a general effect of adding failure-lesson-shaped content is. One caveat worth stating plainly: scrambling only touched per-episode retrieval, not the twelve standing guidelines promoted during training, which are identical in both conditions, so this doesn't tell us whether the guidelines themselves carry real weight.

### 6b. Results — FalseQA

Full scale, train=300 and test=300:

| Condition | F1 | Precision | Recall | False-premise Acc | True-premise Acc | Refusal Rate |
|---|---|---|---|---|---|---|
| baseline | 0.673 | 0.621 | 0.735 | 73.5% | 62.8% | 53.7% |
| RB only | 0.676 | 0.534 | 0.919 | 91.9% | 33.5% | 78.0% |
| RB+SA | 0.729 | 0.584 | 0.971 | 97.1% | 42.7% | 75.3% |
| RB+SA, scrambled | 0.726 | 0.575 | 0.985 | 98.5% | 39.6% | 77.7% |

RB+SA against the scrambled version: b=17, c=14, χ²=0.129, p=0.719, not significant. RB+SA against baseline: b=42, c=43, χ²≈0, p=1.000 — not distinguishable from doing nothing at all. RB alone against baseline: p=0.017, the only comparison here that clears significance.

The reproduction again lands close to reported (0.673/0.676/0.729 against 0.680/0.674/0.736), so the effect direction replicates a second time, but the ablation result is if anything stronger evidence against the relevance claim than SelfAware's — the gap to the placebo is a mere 0.003 F1, and RB+SA can't even be told apart from baseline here. It's worth noting the small real-versus-scrambled gap even flips direction between the two datasets, real SA abstaining slightly more than scrambled on SelfAware but slightly less on FalseQA, which is itself consistent with noise rather than a systematic relevance effect. Across both datasets with runnable code, the same conclusion holds: the claim that relevant retrieval drives the gain doesn't survive direct testing, though a real, modest, dataset-dependent effect of adding memory-derived content in general does appear to be there.

## 7. How to reproduce

```bash
cd sa-module-v2
mkdir -p data memory_bank results
cp <path-to>/selfaware_train.json <path-to>/selfaware_test.json data/

CUDA_VISIBLE_DEVICES=<free_gpu> vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct --port 8080 \
  --dtype bfloat16 --gpu-memory-utilization 0.25 --max-model-len 4096

python3 src/run_selfaware.py --train-limit 300 --test-limit 500   # full scale, ~1-1.5hr on one H200
python3 src/run_selfaware.py --skip-scrambled                     # original 3 conditions only

cp <path-to>/falseqa_train.json <path-to>/falseqa_test.json data/
python3 src/run_falseqa.py                                        # limits set in __main__, default 300/300
```

This writes `results/selfaware_summary.json` and `results/falseqa_summary.json`, each with all four conditions and the McNemar tests, alongside the raw per-item logs.

## 8. A complementary research direction

The results above point at a specific gap: the only signal for "did the model express uncertainty" is a lexical match on surface text, and that same signal generates the training lessons fed back into future prompts. That loop is exploitable — behaviour indistinguishable from genuine calibration under this detector was reproducible, on two datasets, using lessons retrieved for the wrong question entirely. Both RB+SA and Nhat's clustering/routing idea work in this same text space, whether that's surface phrasing or embedding similarity between questions, and neither looks inside the model, so neither can check whether the lexical detector is being fooled.

A complementary angle would use a model-internal uncertainty signal that doesn't depend on what the model chooses to say, as an independent check on the lexical detector. Semantic entropy (Farquhar et al., *Nature* 2024, [10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0)) samples several generations, clusters them by semantic equivalence, and computes entropy over the clusters — genuine uncertainty shows up as semantically scattered answers even when each one, read alone, sounds confident. Semantic Entropy Probes ([Kossen et al. 2024](https://arxiv.org/abs/2406.15927)) approximate this cheaply from a single generation's hidden states. This would let one check whether `detect_uncertainty` agrees with semantic entropy on the same responses, which is a direct test of whether the keyword detector is scoring surface form rather than genuine uncertainty; it would also give a real, non-circular replacement for the hardcoded confidence placeholder, and eventually support a proper risk–coverage curve in place of a single F1 point — the standard way this class of problem is evaluated in the selective-prediction literature.

## 9. Questions I'd want to clarify

I'd first want to ask whether an internal ablation has ever compared retrieval-relevant against retrieval-irrelevant SA content, since my own scrambled-retrieval placebo couldn't statistically distinguish RB+SA from a version fed lessons for the wrong question on either runnable dataset, and on FalseQA RB+SA wasn't even distinguishable from baseline. I'd want to know whether that matches what's been seen internally, or whether KUQ and UnknownBench, where the largest gains are reported, tell a different story. I'd also want to know where the code for KUQ, UnknownBench, and the Gemini/Qwen3 runs actually is — whether that was an oversight in what got shared or intentionally out of scope here — and whether `abstentionbench`, `hle`, `awarebench`, and `selfaware_v3` were left out of the README because they're known-broken or simply not yet written up. And finally, whether "improve abstain-class F1" is really the target you want optimised, or whether a metric that penalises over-abstention symmetrically would better capture what "more trustworthy" is meant to mean for this project.

## AI tool usage

Used Claude (Claude Code) throughout — reading and cross-referencing the code and result files, running the literature searches cited above, implementing the ablation, running the experiments, and drafting this report. Every finding was independently verified against the actual files and logs rather than taken on trust from the model's output, and the choice of which check to run and how to read the results was mine.
