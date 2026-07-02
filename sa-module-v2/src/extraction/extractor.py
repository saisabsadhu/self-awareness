# src/extraction/extractor.py
# Extractor: ReasoningBank strategies + SA awareness lessons
# SA extraction mirrors ReasoningBank's prompt structure exactly:
#   - Different prompts for different outcomes (like RB's success/failure)
#   - Simple TITLE/DESCRIPTION/CONTENT format (same as RB)
#   - Focused on ONE thing per prompt (not 8 fields at once)
#   - Failure type computed deterministically, not asked from LLM
import sys
import re
sys.path.append('..')
from llm_client import LlamaServerClient
from memory import MemoryItem, SAMemoryItem, SCHEMA_CATEGORIES, FAILURE_TYPES
from datetime import datetime
from typing import List, Dict, Optional


class MemoryExtractor:
    """Original ReasoningBank extractor — unchanged"""

    def __init__(self, llm_client: LlamaServerClient):
        self.llm = llm_client

    def extract_from_trajectory(self, problem_id: str, question: str,
                                 solution: Dict, success: bool) -> List[MemoryItem]:
        if success:
            prompt = self._create_success_prompt(question, solution['reasoning'])
        else:
            prompt = self._create_failure_prompt(question, solution['reasoning'],
                                                  solution.get('expected', ''))

        response = self.llm.generate(prompt, temperature=0.0, max_tokens=2048)
        memories = self._parse_memory_items(response, problem_id, success)
        return memories

    def _create_success_prompt(self, question: str, reasoning: str) -> str:
        return f"""You successfully solved this math problem. Extract 1-3 generalizable strategies that led to success.

PROBLEM: {question}

YOUR SOLUTION: {reasoning}

Extract strategies in this format:

MEMORY 1:
TITLE: <concise strategy name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed transferable strategy>

MEMORY 2:
...

Focus on WHY the approach worked and how it could apply to similar problems."""

    def _create_failure_prompt(self, question: str, reasoning: str, expected: str) -> str:
        return f"""You attempted this math problem but got it wrong. Extract 1-3 lessons about what went wrong.

PROBLEM: {question}

YOUR ATTEMPT: {reasoning}

EXPECTED: {expected}

Extract lessons in this format:

MEMORY 1:
TITLE: <what to avoid or check>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed lesson or preventive strategy>

MEMORY 2:
...

Focus on the mistake and how to prevent it in future similar problems."""

    def _parse_memory_items(self, response: str, problem_id: str, success: bool) -> List[MemoryItem]:
        memories = []
        parts = response.split('MEMORY ')
        for part in parts[1:]:
            try:
                title = self._extract_field(part, 'TITLE:')
                description = self._extract_field(part, 'DESCRIPTION:')
                content = self._extract_field(part, 'CONTENT:')
                if title and description and content:
                    memory = MemoryItem(
                        title=title, description=description, content=content,
                        source_problem_id=problem_id, success=success,
                        created_at=datetime.now().isoformat()
                    )
                    memories.append(memory)
            except:
                continue
        return memories

    def _extract_field(self, text: str, field_name: str) -> str:
        if field_name not in text:
            return ""
        start = text.index(field_name) + len(field_name)
        next_field_markers = ['TITLE:', 'DESCRIPTION:', 'CONTENT:', 'MEMORY ']
        end = len(text)
        for marker in next_field_markers:
            if marker in text[start:]:
                candidate_end = start + text[start:].index(marker)
                if candidate_end < end:
                    end = candidate_end
        return text[start:end].strip()


# ─────────────────────────────────────────────────────────────────────────────
# SA Extractor — mirrors ReasoningBank's structure
# Key differences from old version:
#   1. Failure type is GIVEN (computed deterministically), not asked from LLM
#   2. One focused prompt per failure type (like RB's success vs failure)
#   3. Same TITLE/DESCRIPTION/CONTENT output format as RB
#   4. LLM only does what it's good at: explaining WHY and composing lessons
# ─────────────────────────────────────────────────────────────────────────────

class SAExtractor:
    """Self-Awareness extractor — same structure as ReasoningBank"""

    def __init__(self, llm_client: LlamaServerClient):
        self.llm = llm_client

    def extract_sa_lesson(self, problem_id: str, question: str,
                          model_answer: str, ground_truth: str,
                          response_text: str, failure_type: str,
                          answerable: bool) -> SAMemoryItem:
        """
        Extract a self-awareness lesson from one episode.
        
        Key difference from old version:
        - failure_type is GIVEN (computed by evaluate_selfaware), not asked
        - prompt is specific to the failure type
        - output is simple TITLE/DESCRIPTION/CONTENT like ReasoningBank
        """

        # Pick the right prompt based on failure type
        if failure_type == "CORRECT":
            prompt = self._prompt_correct(question, response_text, answerable)
        elif failure_type == "KNOWLEDGE_BOUNDARY_MISSED":
            prompt = self._prompt_boundary_missed(question, response_text)
        elif failure_type == "FALSE_IDK":
            prompt = self._prompt_false_idk(question, response_text, ground_truth)
        elif failure_type == "FACTUAL_ERROR":
            prompt = self._prompt_factual_error(question, response_text, ground_truth)
        else:
            prompt = self._prompt_generic_failure(question, response_text, ground_truth, failure_type)

        llm_response = self.llm.generate(prompt, temperature=0.0, max_tokens=512)
        parsed = self._parse_sa_response(llm_response)

        return SAMemoryItem(
            question=question,
            model_answer=model_answer,
            ground_truth=ground_truth,
            domain=parsed.get("domain", "general"),
            source_problem_id=problem_id,
            created_at=datetime.now().isoformat(),
            outcome="correct" if failure_type == "CORRECT" else "incorrect",
            failure_type=failure_type,  # deterministic, not from LLM
            confidence=0.5,  # placeholder — not trusting LLM's self-report
            knowledge_gap=parsed.get("knowledge_gap", ""),
            schema_categories=["strategy"],
            episode_lesson=parsed.get("lesson", ""),
            strategy_title=parsed.get("title", ""),
            strategy_content=parsed.get("content", ""),
            success=(failure_type == "CORRECT"),
        )

    # ── Prompts per failure type (mirrors RB's success/failure split) ────

    def _prompt_correct(self, question: str, response: str, answerable: bool) -> str:
        if answerable:
            return f"""You correctly answered this question. Extract 1 lesson about how you recognized this was answerable and responded well.

QUESTION: {question}

YOUR RESPONSE: {response[:300]}

CORRECT DECISION: You attempted to answer, which was right.

Extract in this format:
TITLE: <what you did right>
DESCRIPTION: <one sentence>
CONTENT: <how to recognize similar answerable questions in the future>
DOMAIN: <one word topic domain>"""
        else:
            return f"""You correctly recognized this question as unanswerable and expressed uncertainty. Extract 1 lesson about how you recognized it.

QUESTION: {question}

YOUR RESPONSE: {response[:300]}

CORRECT DECISION: You said "I don't know" or expressed uncertainty, which was right because this question has no definitive answer.

Extract in this format:
TITLE: <what signal told you this was unanswerable>
DESCRIPTION: <one sentence>
CONTENT: <how to recognize similar unanswerable questions in the future — what made this question impossible to answer definitively?>
DOMAIN: <one word topic domain>"""

    def _prompt_boundary_missed(self, question: str, response: str) -> str:
        return f"""You made a mistake. This question was UNANSWERABLE — it has no definitive answer. But you attempted to answer it anyway instead of saying "I don't know."

QUESTION: {question}

YOUR RESPONSE: {response[:300]}

THE PROBLEM: This question cannot be answered definitively. You should have expressed uncertainty or said "I don't know."

Extract 1 lesson about what went wrong:
TITLE: <what type of unanswerable question this was>
DESCRIPTION: <one sentence about why you should have abstained>
CONTENT: <detailed explanation of why this question is unanswerable, and what signals you should look for to recognize similar unanswerable questions in the future. Be specific.>
KNOWLEDGE_GAP: <what you failed to recognize about this question>
DOMAIN: <one word topic domain>"""

    def _prompt_false_idk(self, question: str, response: str, ground_truth: str) -> str:
        return f"""You made a mistake. This question WAS answerable, but you said "I don't know" instead of attempting an answer.

QUESTION: {question}

YOUR RESPONSE: {response[:300]}

CORRECT ANSWER: {ground_truth}

THE PROBLEM: You should have attempted to answer this question. The answer exists and is knowable.

Extract 1 lesson about what went wrong:
TITLE: <why you incorrectly abstained>
DESCRIPTION: <one sentence about why you should have answered>
CONTENT: <detailed explanation of why this question IS answerable, and what signals you should look for to avoid incorrectly saying "I don't know" on similar questions in the future. Be specific.>
KNOWLEDGE_GAP: <what knowledge or confidence you were missing>
DOMAIN: <one word topic domain>"""

    def _prompt_factual_error(self, question: str, response: str, ground_truth: str) -> str:
        return f"""You attempted this question but got it wrong. Extract 1 lesson about what went wrong.

QUESTION: {question}

YOUR ANSWER: {response[:300]}

CORRECT ANSWER: {ground_truth}

Extract in this format:
TITLE: <what kind of mistake this was>
DESCRIPTION: <one sentence>
CONTENT: <detailed lesson about the mistake and how to prevent it on similar questions>
KNOWLEDGE_GAP: <what specific knowledge was missing or wrong>
DOMAIN: <one word topic domain>"""

    def _prompt_generic_failure(self, question: str, response: str, ground_truth: str, failure_type: str) -> str:
        return f"""You made a mistake on this question. The failure type was: {failure_type}

QUESTION: {question}

YOUR RESPONSE: {response[:300]}

EXPECTED: {ground_truth}

Extract 1 lesson about what went wrong:
TITLE: <what went wrong>
DESCRIPTION: <one sentence>
CONTENT: <detailed lesson about how to avoid this type of failure>
KNOWLEDGE_GAP: <what was missing>
DOMAIN: <one word topic domain>"""

    # ── Parsing (simple, like RB) ────────────────────────────────────────

    def _parse_sa_response(self, response: str) -> Dict:
        """Parse TITLE/DESCRIPTION/CONTENT/DOMAIN/KNOWLEDGE_GAP from response"""
        data = {}
        data["title"] = self._extract_line_value(response, "TITLE:")
        data["description"] = self._extract_line_value(response, "DESCRIPTION:")
        data["content"] = self._extract_line_value(response, "CONTENT:")
        data["domain"] = self._extract_line_value(response, "DOMAIN:") or "general"
        data["knowledge_gap"] = self._extract_line_value(response, "KNOWLEDGE_GAP:") or ""

        # Compose the lesson from title + content
        if data["title"] and data["content"]:
            data["lesson"] = f"{data['title']}: {data['content']}"
        elif data["content"]:
            data["lesson"] = data["content"]
        else:
            data["lesson"] = data.get("description", "")

        return data

    def _extract_line_value(self, text: str, prefix: str) -> str:
        """Extract value after a prefix, handling multi-line content"""
        if prefix not in text:
            return ""
        
        start = text.index(prefix) + len(prefix)
        # Find the next field marker
        next_markers = ['TITLE:', 'DESCRIPTION:', 'CONTENT:', 'DOMAIN:', 
                       'KNOWLEDGE_GAP:', 'MEMORY ']
        end = len(text)
        for marker in next_markers:
            if marker != prefix and marker in text[start:]:
                candidate = start + text[start:].index(marker)
                if candidate < end:
                    end = candidate
        
        return text[start:end].strip()

    # ── High-level guideline composition ─────────────────────────────────

    def compose_pattern_lesson(self, episodes: List[SAMemoryItem],
                                failure_type: str, domain: str) -> Dict:
        """Compose a high-level guideline from recurring failures.
        Prompt mirrors RB's failure prompt — direct, specific, focused."""

        examples = []
        for ep in episodes[:5]:
            examples.append(f"- Q: {ep.question[:80]}... | {ep.failure_type} | Gap: {ep.knowledge_gap[:60]}")

        # Different prompts depending on failure type
        if failure_type == "KNOWLEDGE_BOUNDARY_MISSED":
            prompt = f"""The model repeatedly attempted to answer UNANSWERABLE questions in {domain} instead of saying "I don't know." This happened {len(episodes)} times.

EXAMPLES OF QUESTIONS IT SHOULD HAVE REFUSED:
{chr(10).join(examples)}

Write a specific guideline to prevent this. The guideline should help the model recognize unanswerable {domain} questions.

PATTERN: <one sentence: what kind of {domain} questions the model incorrectly attempts>
TRIGGER: <specific signal that a {domain} question is unanswerable>
GUIDELINE: <concrete rule: "When [trigger], say 'I don't know' because [reason]">"""

        elif failure_type == "FALSE_IDK":
            prompt = f"""The model repeatedly said "I don't know" on ANSWERABLE questions in {domain} when it should have attempted an answer. This happened {len(episodes)} times.

EXAMPLES OF QUESTIONS IT SHOULD HAVE ANSWERED:
{chr(10).join(examples)}

Write a specific guideline to prevent this. The guideline should help the model recognize when it CAN answer {domain} questions.

PATTERN: <one sentence: what kind of {domain} questions the model incorrectly refuses>
TRIGGER: <specific signal that a {domain} question IS answerable>
GUIDELINE: <concrete rule: "When [trigger], attempt an answer because [reason]">"""

        elif failure_type == "FACTUAL_ERROR":
            prompt = f"""The model repeatedly gave wrong factual answers in {domain}. This happened {len(episodes)} times.

EXAMPLES OF QUESTIONS IT GOT WRONG:
{chr(10).join(examples)}

Write a specific guideline to prevent this. The guideline should help the model avoid factual errors in {domain}.

PATTERN: <one sentence: what kind of {domain} questions the model gets wrong>
TRIGGER: <specific signal that a {domain} question needs extra care>
GUIDELINE: <concrete rule: "When [trigger], [specific action to avoid errors]">"""

        else:
            prompt = f"""The model made {failure_type} errors in {domain} repeatedly ({len(episodes)} times).

EXAMPLES:
{chr(10).join(examples)}

PATTERN: <one sentence describing the recurring pattern>
TRIGGER: <specific condition>
GUIDELINE: <concrete action to take>"""

        response = self.llm.generate(prompt, temperature=0.0, max_tokens=512)
        return self._parse_pattern_response(response)

    def _parse_pattern_response(self, response: str) -> Dict:
        return {
            "pattern": self._extract_line_value(response, "PATTERN:") or "",
            "trigger": self._extract_line_value(response, "TRIGGER:") or "",
            "guideline": self._extract_line_value(response, "GUIDELINE:") or "",
        }
