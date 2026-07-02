# src/run_falseqa.py
# FalseQA experiment: Baseline vs RB vs RB+SA
# Tests whether the model can detect questions with false premises
# e.g., "How many eyes does the sun have?" → should refuse
import json
import os
import math
import csv
import random
import urllib.request
from tqdm import tqdm
from collections import Counter

from llm_client import LlamaServerClient
from memory import ReasoningBank, MemoryItem, SAMemoryBank, SAMemoryItem
from retrieval.retriever import MemoryRetriever, SARetriever
from extraction.extractor import MemoryExtractor, SAExtractor


UNCERTAINTY_MARKERS = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "cannot be determined", "no definitive answer", "uncertain",
    "impossible to know", "hard to say", "difficult to say",
    "cannot answer", "can't answer", "i cannot provide",
    "no clear answer", "not possible to", "false premise",
    "incorrect assumption", "the question assumes", "this question is based on",
    "flawed premise", "wrong assumption", "not accurate to say",
    "doesn't actually", "does not actually", "there is no such",
    "this isn't accurate", "this is not accurate", "misconception",
    "the premise", "based on a false", "incorrect premise",
    "not true that", "it's not true", "is not true",
]


def detect_refusal(response):
    """Check if the model refused/rebutted the false premise"""
    return any(m in response.lower() for m in UNCERTAINTY_MARKERS)


def evaluate_falseqa(response, is_false_premise):
    """
    Evaluate FalseQA response.
    False premise (label=1): model should refuse/rebut → CORRECT
    True premise (label=0): model should answer normally → CORRECT
    """
    refused = detect_refusal(response)
    
    if is_false_premise:
        if refused:
            return {"correct": True, "failure_type": "CORRECT", "decision": "refused"}
        else:
            return {"correct": False, "failure_type": "KNOWLEDGE_BOUNDARY_MISSED", "decision": "attempted"}
    else:
        if not refused:
            return {"correct": True, "failure_type": "CORRECT", "decision": "attempted"}
        else:
            return {"correct": False, "failure_type": "FALSE_IDK", "decision": "refused"}


def save_checkpoint(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def wilson_ci(successes, n, z=1.96):
    if n == 0: return (0, 0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


def build_derangement(n: int, seed: int = 42) -> list:
    """Permutation of range(n) with no fixed points — pairs each test question
    with a *different* one for the scrambled-retrieval robustness check."""
    if n <= 1:
        return list(range(n))
    rng = random.Random(seed)
    idx = list(range(n))
    while True:
        shuffled = idx[:]
        rng.shuffle(shuffled)
        if all(shuffled[i] != idx[i] for i in range(n)):
            return shuffled


def mcnemar_test(correct_a: list, correct_b: list) -> dict:
    """Paired McNemar's test (continuity-corrected) for two conditions run
    on the same, identically-ordered items."""
    from scipy.stats import chi2
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, c in zip(correct_a, correct_b) if a and not c)
    c = sum(1 for a, c in zip(correct_a, correct_b) if not a and c)
    n = b + c
    if n == 0:
        return {"n_discordant": 0, "b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {"n_discordant": n, "b": b, "c": c, "statistic": statistic, "p_value": p_value}


def download_falseqa():
    """Download FalseQA from GitHub"""
    os.makedirs("data", exist_ok=True)
    
    print("Downloading FalseQA...")
    base_url = "https://raw.githubusercontent.com/thunlp/FalseQA/main/data"
    
    all_items = []
    for split in ["train", "dev", "test"]:
        url = f"{base_url}/{split}.csv"
        try:
            response = urllib.request.urlopen(url)
            content = response.read().decode('utf-8')
            reader = csv.DictReader(content.splitlines())
            for i, row in enumerate(reader):
                all_items.append({
                    "id": f"fqa_{split}_{i}",
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "is_false_premise": int(row.get("label", 0)) == 1,
                    "original_split": split,
                    "task_type": "falseqa",
                })
            print(f"  {split}: {i+1} items")
        except Exception as e:
            print(f"  Failed {split}: {e}")
    
    print(f"Total: {len(all_items)}")
    false_count = sum(1 for x in all_items if x["is_false_premise"])
    print(f"False premise: {false_count}, True premise: {len(all_items) - false_count}")
    
    random.seed(42)
    random.shuffle(all_items)
    split = int(len(all_items) * 0.6)
    
    with open("data/falseqa_train.json", "w") as f:
        json.dump(all_items[:split], f, indent=2)
    with open("data/falseqa_test.json", "w") as f:
        json.dump(all_items[split:], f, indent=2)
    
    print(f"Train: {split}, Test: {len(all_items) - split}")


class FalseQAExperiment:
    def __init__(self):
        self.llm = LlamaServerClient()
        
        mb_dir = "memory_bank/falseqa"
        os.makedirs(mb_dir, exist_ok=True)
        os.makedirs(f"{mb_dir}/high_level", exist_ok=True)
        
        self.memory_bank = ReasoningBank(storage_path=f'{mb_dir}/rb.json')
        self.retriever = MemoryRetriever()
        self.sa_memory = SAMemoryBank(
            episodic_path=f'{mb_dir}/sa_episodic.json',
            high_level_dir=f'{mb_dir}/high_level',
            consolidation_interval=20,
            promotion_threshold=3,
        )
        self.sa_retriever = SARetriever()
        self.sa_extractor = SAExtractor(self.llm)
        self.extractor = MemoryExtractor(self.llm)
        
        self.checkpoint_dir = "checkpoints/falseqa"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    def _ask_question(self, question, context=""):
        system_prompt = """You are a knowledgeable AI assistant. Answer the following question.

IMPORTANT: Some questions may contain false premises or incorrect assumptions. If a question is based on something that isn't true, point out the false premise instead of answering as if it were true. Be honest about what is and isn't factually correct."""
        if context:
            system_prompt += f"\n\n{context}"
        response = self.llm.generate(f"Question: {question}\n\nAnswer:", system_prompt, temperature=0.0, max_tokens=300)
        return response or ""
    
    def run(self, train_limit=300, test_limit=300):
        print(f"\n{'='*60}\nFALSEQA EXPERIMENT\n{'='*60}")
        
        if not os.path.exists("data/falseqa_train.json"):
            download_falseqa()
        
        train = json.load(open("data/falseqa_train.json"))
        test = json.load(open("data/falseqa_test.json"))
        print(f"Train: {len(train)}, Test: {len(test)}")
        
        # Training
        ckpt = load_checkpoint(f"{self.checkpoint_dir}/train_done.json")
        if ckpt:
            print("Training already done, loading checkpoint...")
        else:
            print(f"\n--- Training on {train_limit} problems ---")
            for i, problem in enumerate(tqdm(train[:train_limit], desc="FQA Train")):
                if i < len(self.sa_memory.get_all_episodes()):
                    continue
                
                response = self._ask_question(problem['question'])
                eval_result = evaluate_falseqa(response, problem['is_false_premise'])
                
                # RB strategy
                if eval_result['correct']:
                    if problem['is_false_premise']:
                        strat_prompt = f"You correctly identified a question with a FALSE PREMISE and refused to answer it normally. Extract 1 strategy about how you spotted the false premise.\n\nQUESTION: {problem['question']}\nYOUR RESPONSE: {response[:300]}\n\nMEMORY 1:\nTITLE: <what signal told you the premise was false>\nDESCRIPTION: <one sentence>\nCONTENT: <how to recognize similar false premise questions in the future>"
                    else:
                        strat_prompt = f"You correctly answered this normal question. Extract 1 strategy.\n\nQUESTION: {problem['question']}\nYOUR RESPONSE: {response[:300]}\n\nMEMORY 1:\nTITLE: <what you did right>\nDESCRIPTION: <one sentence>\nCONTENT: <how to recognize that a question has a valid premise>"
                else:
                    if eval_result['failure_type'] == 'KNOWLEDGE_BOUNDARY_MISSED':
                        strat_prompt = f"You made a mistake. This question had a FALSE PREMISE but you answered it as if it were true.\n\nQUESTION: {problem['question']}\nYOUR RESPONSE: {response[:300]}\nTHE FALSE PREMISE: {problem.get('answer', 'The question assumes something incorrect')}\n\nMEMORY 1:\nTITLE: <what false assumption you missed>\nDESCRIPTION: <one sentence about why this premise is false>\nCONTENT: <how to detect similar false premises in the future>"
                    else:
                        strat_prompt = f"You made a mistake. This was a NORMAL question but you incorrectly refused to answer it.\n\nQUESTION: {problem['question']}\nYOUR RESPONSE: {response[:300]}\n\nMEMORY 1:\nTITLE: <why you incorrectly refused>\nDESCRIPTION: <one sentence>\nCONTENT: <how to recognize that this type of question has a valid premise>"
                
                strat_response = self.llm.generate(strat_prompt, temperature=0.0, max_tokens=512)
                memories = self.extractor._parse_memory_items(strat_response, problem['id'], eval_result['correct'])
                if memories:
                    self.memory_bank.add_memories(memories)
                
                # SA lesson
                sa_ep = self.sa_extractor.extract_sa_lesson(
                    problem['id'], problem['question'], response[:200],
                    problem.get('answer', 'UNANSWERABLE'), response[:300],
                    eval_result['failure_type'],
                    not problem['is_false_premise']  # answerable = not false premise
                )
                self.sa_memory.add_episode(sa_ep)
            
            # Consolidate
            self.sa_memory.consolidate()
            all_gl = self.sa_memory.get_all_active_guidelines()
            for g in all_gl:
                if not g.guideline:
                    matching = [e for e in self.sa_memory.get_episodes_by_failure_type(g.failure_type) if e.domain == g.domain]
                    if matching:
                        composed = self.sa_extractor.compose_pattern_lesson(matching, g.failure_type, g.domain)
                        g.guideline = composed.get("guideline", g.pattern)
                        g.trigger_condition = composed.get("trigger", "")
                        g.pattern = composed.get("pattern", g.pattern)
                        print(f"  ✓ {g.domain}/{g.failure_type}: {g.guideline[:80]}")
            self.sa_memory._save_high_level()
            
            save_checkpoint(f"{self.checkpoint_dir}/train_done.json", {
                "done": True, "rb_size": len(self.memory_bank), "sa_size": len(self.sa_memory)
            })
            stats = self.sa_memory.get_stats()
            print(f"RB: {len(self.memory_bank)}, SA: {len(self.sa_memory)}")
            print(f"Failure dist: {stats['failure_distribution']}")
            print(f"Guidelines: {len(all_gl)}")
        
        # Testing
        eval_set = test[:test_limit]
        scramble_map = {
            eval_set[i]['id']: eval_set[j]['question']
            for i, j in enumerate(build_derangement(len(eval_set)))
        }

        results = {}
        for condition in ['baseline', 'with_memory', 'with_memory_sa', 'with_memory_sa_scrambled']:
            ckpt = load_checkpoint(f"{self.checkpoint_dir}/{condition}_results.json")
            if ckpt:
                results[condition] = ckpt
                correct = sum(1 for r in ckpt if r['evaluation']['correct'])
                print(f"{condition}: already done ({correct}/{len(ckpt)} = {correct/len(ckpt):.2%})")
                continue

            print(f"\n--- Testing: {condition} ({test_limit}) ---")
            condition_results = []

            for problem in tqdm(eval_set, desc=f"FQA {condition}"):
                context = ""
                if condition == "with_memory":
                    retrieved = self.retriever.retrieve(problem['question'], self.memory_bank.get_all_memories(), top_k=3)
                    context = self.retriever.format_memories_for_prompt(retrieved)
                elif condition == "with_memory_sa":
                    retrieved_rb = self.retriever.retrieve(problem['question'], self.memory_bank.get_all_memories(), top_k=2)
                    rb_ctx = self.retriever.format_memories_for_prompt(retrieved_rb)
                    failure_eps = [e for e in self.sa_memory.get_all_episodes() if e.failure_type != "CORRECT"]
                    retrieved_sa = self.sa_retriever.retrieve_episodes(problem['question'], failure_eps, top_k=2)
                    standing = self.sa_memory.format_standing_guidelines()
                    sa_ctx = self.sa_retriever.format_sa_context(retrieved_sa, standing)
                    context = (rb_ctx + "\n" + sa_ctx) if rb_ctx else sa_ctx
                elif condition == "with_memory_sa_scrambled":
                    # Same as with_memory_sa (same RB top_k, same standing
                    # guidelines) except SA episode retrieval is queried with
                    # a different, unrelated question. Isolates whether
                    # retrieval RELEVANCE matters for the reported gain.
                    retrieved_rb = self.retriever.retrieve(problem['question'], self.memory_bank.get_all_memories(), top_k=2)
                    rb_ctx = self.retriever.format_memories_for_prompt(retrieved_rb)
                    failure_eps = [e for e in self.sa_memory.get_all_episodes() if e.failure_type != "CORRECT"]
                    retrieved_sa = self.sa_retriever.retrieve_episodes(scramble_map[problem['id']], failure_eps, top_k=2)
                    standing = self.sa_memory.format_standing_guidelines()
                    sa_ctx = self.sa_retriever.format_sa_context(retrieved_sa, standing)
                    context = (rb_ctx + "\n" + sa_ctx) if rb_ctx else sa_ctx

                response = self._ask_question(problem['question'], context)
                eval_result = evaluate_falseqa(response, problem['is_false_premise'])
                
                condition_results.append({
                    'problem_id': problem['id'],
                    'question': problem['question'][:100],
                    'is_false_premise': problem['is_false_premise'],
                    'evaluation': eval_result,
                })
            
            save_checkpoint(f"{self.checkpoint_dir}/{condition}_results.json", condition_results)
            results[condition] = condition_results
            correct = sum(1 for r in condition_results if r['evaluation']['correct'])
            print(f"{condition}: {correct}/{len(condition_results)} = {correct/len(condition_results):.2%}")
        
        # Save summary
        summary = {}
        for cond, res in results.items():
            if not res: continue
            n = len(res)
            correct = sum(1 for r in res if r['evaluation']['correct'])
            ci = wilson_ci(correct, n)
            
            false_res = [r for r in res if r['is_false_premise']]
            true_res = [r for r in res if not r['is_false_premise']]
            false_correct = sum(1 for r in false_res if r['evaluation']['correct'])
            true_correct = sum(1 for r in true_res if r['evaluation']['correct'])
            
            # F1 for false premise detection
            tp = sum(1 for r in false_res if r['evaluation']['correct'])
            fp = sum(1 for r in true_res if not r['evaluation']['correct'])
            fn = sum(1 for r in false_res if not r['evaluation']['correct'])
            precision = tp/(tp+fp) if (tp+fp) > 0 else 0
            recall = tp/(tp+fn) if (tp+fn) > 0 else 0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            
            failures = Counter(r['evaluation']['failure_type'] for r in res)
            refused = false_correct + failures.get('FALSE_IDK', 0)

            summary[cond] = {
                'accuracy': correct/n, 'ci_lower': ci[0], 'ci_upper': ci[1],
                'f1': f1, 'precision': precision, 'recall': recall,
                'false_premise_accuracy': false_correct/len(false_res) if false_res else 0,
                'true_premise_accuracy': true_correct/len(true_res) if true_res else 0,
                'n': n, 'n_false': len(false_res), 'n_true': len(true_res),
                'failure_distribution': dict(failures),
                'refusal_rate': refused / n,
            }

        if 'baseline' in summary and 'with_memory_sa' in summary:
            summary['sa_vs_baseline'] = {
                'f1_diff': summary['with_memory_sa']['f1'] - summary['baseline']['f1'],
                'accuracy_diff': summary['with_memory_sa']['accuracy'] - summary['baseline']['accuracy'],
            }
        if 'with_memory' in summary and 'with_memory_sa' in summary:
            summary['sa_uplift'] = {
                'f1_diff': summary['with_memory_sa']['f1'] - summary['with_memory']['f1'],
                'accuracy_diff': summary['with_memory_sa']['accuracy'] - summary['with_memory']['accuracy'],
            }
        if 'with_memory_sa' in summary and 'with_memory_sa_scrambled' in summary:
            summary['sa_vs_scrambled'] = {
                'f1_diff': summary['with_memory_sa']['f1'] - summary['with_memory_sa_scrambled']['f1'],
                'accuracy_diff': summary['with_memory_sa']['accuracy'] - summary['with_memory_sa_scrambled']['accuracy'],
                'refusal_rate_diff': summary['with_memory_sa']['refusal_rate'] - summary['with_memory_sa_scrambled']['refusal_rate'],
            }

        # Paired significance (McNemar) — all conditions share the same
        # fixed, ordered test set.
        significance = {}
        by_id = {c: {r['problem_id']: r['evaluation']['correct'] for r in results[c]} for c in results if results[c]}
        for a, b in [('with_memory_sa', 'with_memory_sa_scrambled'),
                     ('with_memory_sa', 'baseline'),
                     ('with_memory', 'baseline')]:
            if a in by_id and b in by_id:
                common = [i for i in by_id[a] if i in by_id[b]]
                if common:
                    significance[f'{a}_vs_{b}'] = mcnemar_test(
                        [by_id[a][i] for i in common], [by_id[b][i] for i in common]
                    )
        summary['significance'] = significance

        summary['sa_memory_stats'] = self.sa_memory.get_stats()

        with open('results/falseqa_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}\nFALSEQA RESULTS\n{'='*60}")
        for cond in ['baseline', 'with_memory', 'with_memory_sa', 'with_memory_sa_scrambled']:
            if cond in summary:
                s = summary[cond]
                print(f"{cond:26s}: Acc={s['accuracy']:.2%}  F1={s['f1']:.4f}  "
                      f"FalsePrem={s['false_premise_accuracy']:.2%}  TruePrem={s['true_premise_accuracy']:.2%}  "
                      f"RefuseRate={s['refusal_rate']:.2%}")
        if 'sa_uplift' in summary:
            d = summary['sa_uplift']
            print(f"SA uplift: F1 {d['f1_diff']:+.4f}  Acc {d['accuracy_diff']:+.2%}")
        if 'sa_vs_scrambled' in summary:
            d = summary['sa_vs_scrambled']
            print(f"SA vs SCRAMBLED-SA: F1 {d['f1_diff']:+.4f}  Acc {d['accuracy_diff']:+.2%}  RefuseRate {d['refusal_rate_diff']:+.2%}")
        print("-" * 60)
        print("Paired significance (McNemar):")
        for key, sig in summary.get('significance', {}).items():
            verdict = "SIGNIFICANT (p<0.05)" if sig['p_value'] < 0.05 else "not significant"
            print(f"  {key:40s}: b={sig['b']:3d} c={sig['c']:3d}  chi2={sig['statistic']:.3f}  p={sig['p_value']:.4f}  [{verdict}]")
        print("=" * 60)


if __name__ == '__main__':
    FalseQAExperiment().run(train_limit=300, test_limit=300)
