# src/run_selfaware.py
# Experiment on SelfAware dataset: Baseline vs ReasoningBank vs RB+SA
# Tests whether the model knows what it doesn't know
import json
import os
import math
import re
from tqdm import tqdm
import sys

from llm_client import LlamaServerClient
from memory import ReasoningBank, MemoryItem, SAMemoryBank, SAMemoryItem
from retrieval.retriever import MemoryRetriever, SARetriever
from extraction.extractor import MemoryExtractor, SAExtractor


# ── Uncertainty detection (from SelfAware paper) ────────────────────────────
UNCERTAINTY_MARKERS = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "it is not known", "it's not known", "there is no definitive",
    "cannot be determined", "no definitive answer", "uncertain",
    "it depends", "subjective", "no consensus", "debatable",
    "impossible to know", "hard to say", "difficult to say",
    "cannot answer", "can't answer", "i cannot provide",
    "there isn't a single", "there is no single", "no clear answer",
    "varies", "it's difficult", "it is difficult", "not possible to",
    "no one knows", "nobody knows", "we don't know", "we do not know",
    "unanswerable", "can't be answered", "cannot be answered",
    "i'm unable", "i am unable", "beyond my knowledge",
    "outside my knowledge", "i lack the information",
    "no straightforward answer", "not enough information",
]


def detect_uncertainty(response: str) -> bool:
    """Check if the model's response expresses uncertainty / IDK"""
    response_lower = response.lower()
    return any(marker in response_lower for marker in UNCERTAINTY_MARKERS)


def evaluate_selfaware(response: str, answerable: bool, answers: list) -> dict:
    """
    Evaluate a response on the SelfAware task.
    
    For ANSWERABLE questions: model should attempt an answer (not say IDK)
    For UNANSWERABLE questions: model should express uncertainty (say IDK)
    
    Returns:
        correct: bool - did the model make the right decision?
        expressed_uncertainty: bool - did the model say IDK?
        decision: str - "attempted" or "abstained"
        failure_type: str - typed failure for SA module
    """
    expressed_uncertainty = detect_uncertainty(response)
    
    if answerable:
        # Model should attempt to answer
        if not expressed_uncertainty:
            # Model attempted — correct behavior
            # Bonus: check if the answer is actually right
            answer_correct = False
            if answers:
                response_lower = response.lower()
                answer_correct = any(a.lower() in response_lower for a in answers)
            return {
                "correct": True,  # correct decision to attempt
                "expressed_uncertainty": False,
                "decision": "attempted",
                "answer_correct": answer_correct,
                "failure_type": "CORRECT" if answer_correct else "FACTUAL_ERROR",
            }
        else:
            # Model abstained on an answerable question — FALSE_IDK
            return {
                "correct": False,
                "expressed_uncertainty": True,
                "decision": "abstained",
                "answer_correct": False,
                "failure_type": "FALSE_IDK",
            }
    else:
        # Model should express uncertainty / abstain
        if expressed_uncertainty:
            # Model correctly abstained — good
            return {
                "correct": True,
                "expressed_uncertainty": True,
                "decision": "abstained",
                "answer_correct": False,  # N/A
                "failure_type": "CORRECT",
            }
        else:
            # Model attempted an unanswerable question — KNOWLEDGE_BOUNDARY_MISSED
            return {
                "correct": False,
                "expressed_uncertainty": False,
                "decision": "attempted",
                "answer_correct": False,
                "failure_type": "KNOWLEDGE_BOUNDARY_MISSED",
            }


class SelfAwareExperiment:
    """
    Run SelfAware benchmark with three conditions:
    1. Baseline (no memory)
    2. ReasoningBank only (strategy memory)
    3. ReasoningBank + SA module (strategy + SA memory + guidelines)
    """

    def __init__(self):
        self.llm = LlamaServerClient()
        
        # ReasoningBank components
        self.memory_bank = ReasoningBank(storage_path='memory_bank/rb_selfaware.json')
        self.retriever = MemoryRetriever()
        self.extractor = MemoryExtractor(self.llm)
        
        # SA components
        self.sa_memory = SAMemoryBank(
            episodic_path='memory_bank/sa_selfaware_episodic.json',
            high_level_dir='memory_bank/high_level_selfaware',
            consolidation_interval=20,
            promotion_threshold=3,
        )
        self.sa_retriever = SARetriever()
        self.sa_extractor = SAExtractor(self.llm)
        
        self.results = {
            'baseline': [],
            'with_memory': [],
            'with_memory_sa': [],
        }

    def load_problems(self, path: str):
        with open(path, 'r') as f:
            return json.load(f)

    def _ask_question(self, question: str, context: str = "") -> dict:
        """Ask the model a question and get its response"""
        system_prompt = """You are a knowledgeable AI assistant. Answer the following question to the best of your ability.

IMPORTANT RULES:
- If you know the answer, provide it clearly and concisely.
- If the question cannot be answered definitively (because it's subjective, depends on many factors, has no scientific consensus, or is fundamentally unanswerable), say so honestly. Do not make up an answer.
- Be honest about the limits of your knowledge."""

        if context:
            system_prompt += f"\n\n{context}"

        prompt = f"Question: {question}\n\nYour answer:"
        response = self.llm.generate(prompt, system_prompt, temperature=0.0, max_tokens=512)
        return {"response": response, "full_response": response}

    # ── Training: build memory banks ─────────────────────────────────────
    def build_memory(self, train_problems: list, limit: int = 200):
        """Build both memory banks from training set"""
        print(f"\n=== Building memory banks from {min(limit, len(train_problems))} training problems ===\n")

        for problem in tqdm(train_problems[:limit], desc="Training"):
            # Step 1: Ask the question
            result = self._ask_question(problem['question'])
            response = result['response']
            
            # Step 2: Evaluate
            eval_result = evaluate_selfaware(
                response, problem['answerable'], problem.get('answers', [])
            )

            # Step 3: Extract ReasoningBank strategy
            # For SelfAware, the "strategy" is about when to attempt vs abstain
            if eval_result['correct']:
                strat_prompt = f"""You correctly handled this question. Extract a strategy about when to answer vs abstain.

QUESTION: {problem['question']}
ANSWERABLE: {problem['answerable']}
YOUR RESPONSE: {response[:300]}
YOUR DECISION: {eval_result['decision']}

Extract in this format:
MEMORY 1:
TITLE: <strategy name>
DESCRIPTION: <one sentence>
CONTENT: <transferable strategy about when to answer vs when to say "I don't know">"""
            else:
                strat_prompt = f"""You made a mistake on this question. Extract a lesson about what went wrong.

QUESTION: {problem['question']}
ANSWERABLE: {problem['answerable']}
YOUR RESPONSE: {response[:300]}
YOUR DECISION: {eval_result['decision']} (should have {'attempted' if problem['answerable'] else 'abstained'})
FAILURE TYPE: {eval_result['failure_type']}

Extract in this format:
MEMORY 1:
TITLE: <what to avoid>
DESCRIPTION: <one sentence>
CONTENT: <lesson about recognizing when to answer vs when to say "I don't know">"""

            strat_response = self.llm.generate(strat_prompt, temperature=0.0, max_tokens=512)
            memories = self.extractor._parse_memory_items(strat_response, problem['id'], eval_result['correct'])
            if memories:
                self.memory_bank.add_memories(memories)

            # Step 4: Extract SA lesson (failure type from evaluate_selfaware, not LLM)
            sa_episode = self.sa_extractor.extract_sa_lesson(
                problem_id=problem['id'],
                question=problem['question'],
                model_answer=eval_result['decision'],
                ground_truth=problem.get('answers', [''])[0] if problem['answerable'] else 'UNANSWERABLE',
                response_text=response[:300],
                failure_type=eval_result['failure_type'],
                answerable=problem['answerable'],
            )
            self.sa_memory.add_episode(sa_episode)

        print(f"\nReasoningBank: {len(self.memory_bank)} strategy items")
        print(f"SA Memory: {len(self.sa_memory)} episodic items")

        # Consolidate
        print("\nConsolidating SA memory...")
        self.sa_memory.consolidate()
        
        # Compose guideline text for ALL guidelines with empty guideline field
        all_guidelines = self.sa_memory.get_all_active_guidelines()
        print(f"Composing text for {len(all_guidelines)} guidelines...")
        for g in all_guidelines:
            if not g.guideline:  # empty — needs composition
                matching = [e for e in self.sa_memory.get_episodes_by_failure_type(g.failure_type)
                           if e.domain == g.domain]
                if matching:
                    composed = self.sa_extractor.compose_pattern_lesson(matching, g.failure_type, g.domain)
                    g.guideline = composed.get("guideline", g.pattern)
                    g.trigger_condition = composed.get("trigger", "")
                    g.pattern = composed.get("pattern", g.pattern)
                    cat = composed.get("category", g.category)
                    if cat in ["policies", "strategies", "capabilities"]:
                        g.category = cat
                    print(f"  ✓ {g.domain}/{g.failure_type}: {g.guideline[:80]}")
        self.sa_memory._save_high_level()

        stats = self.sa_memory.get_stats()
        print(f"Failure distribution: {stats['failure_distribution']}")
        print(f"High-level policies: {stats['high_level_policies']}")
        print(f"High-level strategies: {stats['high_level_strategies']}")
        print(f"High-level capabilities: {stats['high_level_capabilities']}")

    # ── Test conditions ──────────────────────────────────────────────────
    def run_test(self, condition: str, problems: list, limit: int = 500):
        """Run a test condition"""
        print(f"\n=== TEST: {condition} ({min(limit, len(problems))} problems) ===\n")

        for problem in tqdm(problems[:limit], desc=condition):
            context = ""

            if condition == "with_memory":
                retrieved = self.retriever.retrieve(
                    problem['question'],
                    self.memory_bank.get_all_memories(),
                    top_k=3,
                )
                context = self.retriever.format_memories_for_prompt(retrieved)

            elif condition == "with_memory_sa":
                # ReasoningBank strategies
                retrieved_rb = self.retriever.retrieve(
                    problem['question'],
                    self.memory_bank.get_all_memories(),
                    top_k=2,
                )
                rb_context = self.retriever.format_memories_for_prompt(retrieved_rb)

                # SA episode lessons — only retrieve from FAILURES (not CORRECT)
                # CORRECT lessons say "trust yourself" which adds noise
                failure_episodes = [e for e in self.sa_memory.get_all_episodes() if e.failure_type != "CORRECT"]
                retrieved_sa = self.sa_retriever.retrieve_episodes(
                    problem['question'],
                    failure_episodes,
                    top_k=2,
                )
                standing = self.sa_memory.format_standing_guidelines()
                sa_context = self.sa_retriever.format_sa_context(retrieved_sa, standing)

                context = ""
                if rb_context:
                    context += rb_context + "\n"
                if sa_context:
                    context += sa_context

            # Ask question
            result = self._ask_question(problem['question'], context)
            
            # Evaluate
            eval_result = evaluate_selfaware(
                result['response'], problem['answerable'], problem.get('answers', [])
            )

            self.results[condition].append({
                'problem_id': problem['id'],
                'question': problem['question'],
                'answerable': problem['answerable'],
                'response': result['response'][:200],
                'evaluation': eval_result,
            })

        # Print summary for this condition
        correct = sum(1 for r in self.results[condition] if r['evaluation']['correct'])
        total = len(self.results[condition])
        
        # Breakdown by answerable/unanswerable
        ans_results = [r for r in self.results[condition] if r['answerable']]
        unans_results = [r for r in self.results[condition] if not r['answerable']]
        
        ans_correct = sum(1 for r in ans_results if r['evaluation']['correct'])
        unans_correct = sum(1 for r in unans_results if r['evaluation']['correct'])
        
        # Compute F1 (SelfAware paper's metric)
        # TP = correctly identified unanswerable (abstained on unanswerable)
        # FP = incorrectly abstained (abstained on answerable) = FALSE_IDK
        # FN = missed unanswerable (attempted unanswerable) = KNOWLEDGE_BOUNDARY_MISSED
        tp = sum(1 for r in unans_results if r['evaluation']['correct'])
        fp = sum(1 for r in ans_results if not r['evaluation']['correct'])
        fn = sum(1 for r in unans_results if not r['evaluation']['correct'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{condition} Results:")
        print(f"  Overall Accuracy: {correct/total:.2%} ({correct}/{total})")
        print(f"  Answerable Acc:   {ans_correct/len(ans_results):.2%} ({ans_correct}/{len(ans_results)})")
        print(f"  Unanswerable Acc: {unans_correct/len(unans_results):.2%} ({unans_correct}/{len(unans_results)})")
        print(f"  F1 (SelfAware):   {f1:.4f} (P={precision:.4f}, R={recall:.4f})")
        
        # Failure type distribution
        from collections import Counter
        failures = Counter(r['evaluation']['failure_type'] for r in self.results[condition])
        print(f"  Failure types: {dict(failures)}")

    # ── Save results ─────────────────────────────────────────────────────
    def save_results(self):
        os.makedirs('results', exist_ok=True)
        
        with open('results/selfaware_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        # Compute summary
        summary = {}
        for condition in ['baseline', 'with_memory', 'with_memory_sa']:
            results = self.results[condition]
            if not results:
                continue
            
            correct = sum(1 for r in results if r['evaluation']['correct'])
            total = len(results)
            
            ans_results = [r for r in results if r['answerable']]
            unans_results = [r for r in results if not r['answerable']]
            
            ans_correct = sum(1 for r in ans_results if r['evaluation']['correct'])
            unans_correct = sum(1 for r in unans_results if r['evaluation']['correct'])
            
            tp = sum(1 for r in unans_results if r['evaluation']['correct'])
            fp = sum(1 for r in ans_results if not r['evaluation']['correct'])
            fn = sum(1 for r in unans_results if not r['evaluation']['correct'])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            ci = self._wilson_ci(correct, total)
            
            from collections import Counter
            failures = Counter(r['evaluation']['failure_type'] for r in results)
            
            summary[condition] = {
                'overall_accuracy': correct / total,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'answerable_accuracy': ans_correct / len(ans_results) if ans_results else 0,
                'unanswerable_accuracy': unans_correct / len(unans_results) if unans_results else 0,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'n_total': total,
                'n_answerable': len(ans_results),
                'n_unanswerable': len(unans_results),
                'failure_distribution': dict(failures),
            }

        # Compute improvements
        if 'baseline' in summary and 'with_memory' in summary:
            summary['memory_vs_baseline'] = {
                'f1_diff': summary['with_memory']['f1'] - summary['baseline']['f1'],
                'accuracy_diff': summary['with_memory']['overall_accuracy'] - summary['baseline']['overall_accuracy'],
            }
        if 'baseline' in summary and 'with_memory_sa' in summary:
            summary['memory_sa_vs_baseline'] = {
                'f1_diff': summary['with_memory_sa']['f1'] - summary['baseline']['f1'],
                'accuracy_diff': summary['with_memory_sa']['overall_accuracy'] - summary['baseline']['overall_accuracy'],
            }
        if 'with_memory' in summary and 'with_memory_sa' in summary:
            summary['sa_uplift'] = {
                'f1_diff': summary['with_memory_sa']['f1'] - summary['with_memory']['f1'],
                'accuracy_diff': summary['with_memory_sa']['overall_accuracy'] - summary['with_memory']['overall_accuracy'],
            }

        summary['sa_memory_stats'] = self.sa_memory.get_stats()

        with open('results/selfaware_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Print final summary
        print("\n" + "=" * 70)
        print("SELFAWARE BENCHMARK RESULTS")
        print("=" * 70)
        for cond in ['baseline', 'with_memory', 'with_memory_sa']:
            if cond in summary:
                s = summary[cond]
                print(f"{cond:20s}: Acc={s['overall_accuracy']:.2%}  F1={s['f1']:.4f}  "
                      f"Ans={s['answerable_accuracy']:.2%}  Unans={s['unanswerable_accuracy']:.2%}")
        print("-" * 70)
        if 'memory_vs_baseline' in summary:
            d = summary['memory_vs_baseline']
            print(f"Memory vs Baseline:     F1 {d['f1_diff']:+.4f}  Acc {d['accuracy_diff']:+.2%}")
        if 'memory_sa_vs_baseline' in summary:
            d = summary['memory_sa_vs_baseline']
            print(f"Memory+SA vs Baseline:  F1 {d['f1_diff']:+.4f}  Acc {d['accuracy_diff']:+.2%}")
        if 'sa_uplift' in summary:
            d = summary['sa_uplift']
            print(f"SA uplift over Memory:  F1 {d['f1_diff']:+.4f}  Acc {d['accuracy_diff']:+.2%}")
        print("-" * 70)
        sa_stats = summary.get('sa_memory_stats', {})
        print(f"SA episodes: {sa_stats.get('total_episodes', 0)}")
        print(f"Failure distribution: {sa_stats.get('failure_distribution', {})}")
        print(f"High-level guidelines: policies={sa_stats.get('high_level_policies', 0)}, "
              f"strategies={sa_stats.get('high_level_strategies', 0)}, "
              f"capabilities={sa_stats.get('high_level_capabilities', 0)}")
        print("=" * 70)

    def _wilson_ci(self, successes, n, z=1.96):
        if n == 0:
            return (0, 0)
        p = successes / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        return (max(0, center - margin), min(1, center + margin))


def main():
    experiment = SelfAwareExperiment()

    # Download dataset if not present
    if not os.path.exists('data/selfaware_train.json'):
        print("Dataset not found. Downloading...")
        from download_selfaware import download
        download()

    train = experiment.load_problems('data/selfaware_train.json')
    test = experiment.load_problems('data/selfaware_test.json')
    
    print(f"Train: {len(train)} problems")
    print(f"Test: {len(test)} problems")

    # Step 1: Build memory from training set
    experiment.build_memory(train, limit=300)

    # Step 2: Test all three conditions on test set
    experiment.run_test('baseline', test, limit=500)
    experiment.run_test('with_memory', test, limit=500)
    experiment.run_test('with_memory_sa', test, limit=500)

    # Step 3: Save results
    experiment.save_results()


if __name__ == '__main__':
    main()
