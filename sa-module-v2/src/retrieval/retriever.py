# src/retrieval/retriever.py
# Extended retriever: retrieves from both strategy memory and SA memory
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import re
import sys
sys.path.append('..')
from memory import MemoryItem, SAMemoryItem, HighLevelMemory, TIER_COLD


class MemoryRetriever:
    """Original retriever — unchanged for backward compat"""

    def __init__(self, embedding_model_path='Qwen/Qwen3-Embedding-0.6B'):
        self.model = SentenceTransformer(embedding_model_path)
        print(f"Loaded embedding model: {embedding_model_path}")

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_memories(self, memories: List[MemoryItem]):
        texts = [f"{m.title}. {m.description}" for m in memories]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        for memory, embedding in zip(memories, embeddings):
            memory.embedding = embedding.tolist()

    def _has_answer_leak(self, memory: MemoryItem, expected_value: str) -> bool:
        if not expected_value:
            return False
        memory_text = f"{memory.title} {memory.description} {memory.content}".lower()
        expected_numbers = set(re.findall(r'\b\d+\.?\d*\b', str(expected_value)))
        if not expected_numbers:
            return False
        result_keywords = ['answer', 'result', 'total', 'equals', 'final',
                          '=', 'is', 'solution', 'outcome']
        for num in expected_numbers:
            num_positions = [m.start() for m in re.finditer(r'\b' + re.escape(num) + r'\b', memory_text)]
            for pos in num_positions:
                context_start = max(0, pos - 50)
                context_end = min(len(memory_text), pos + 50)
                context = memory_text[context_start:context_end]
                if any(keyword in context for keyword in result_keywords):
                    return True
        return False

    def retrieve(self, query: str, memories: List[MemoryItem], top_k: int = 3,
                 expected_value: Optional[str] = None) -> List[Tuple[MemoryItem, float]]:
        if not memories:
            return []
        if any(m.embedding is None for m in memories):
            self.embed_memories(memories)
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        memory_embeddings = np.array([m.embedding for m in memories])
        similarities = np.dot(memory_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-(top_k * 3):][::-1]
        candidates = [(memories[idx], float(similarities[idx])) for idx in top_indices]
        if expected_value:
            filtered = [(m, s) for m, s in candidates if not self._has_answer_leak(m, expected_value)]
            results = filtered[:top_k]
        else:
            results = candidates[:top_k]
        return results

    def format_memories_for_prompt(self, retrieved: List[Tuple[MemoryItem, float]]) -> str:
        if not retrieved:
            return ""
        formatted = "## Past Strategy Hints:\n\n"
        formatted += "**Important**: These are STRATEGY hints only. Do NOT copy any numbers from them.\n\n"
        for idx, (memory, score) in enumerate(retrieved, 1):
            status = "✓ Success Strategy" if memory.success else "✗ Lesson from Failure"
            formatted += f"### Strategy {idx} ({status}):\n"
            formatted += f"**{memory.title}**\n"
            formatted += f"{memory.content}\n\n"
        return formatted


class SARetriever:
    """Self-Awareness retriever — retrieves from SA episodic memory"""

    def __init__(self, embedding_model_path='Qwen/Qwen3-Embedding-0.6B'):
        self.model = SentenceTransformer(embedding_model_path)

    def embed_episodes(self, episodes: List[SAMemoryItem]):
        """Compute embeddings for SA episodes"""
        texts = [f"{ep.question}. {ep.episode_lesson}" for ep in episodes]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        for ep, emb in zip(episodes, embeddings):
            ep.embedding = emb.tolist()

    def retrieve_episodes(self, query: str, episodes: List[SAMemoryItem],
                          top_k: int = 3, exclude_cold: bool = True) -> List[Tuple[SAMemoryItem, float]]:
        """Retrieve top-k most relevant SA episodes"""
        if not episodes:
            return []

        # Filter out COLD tier if requested
        active_episodes = [e for e in episodes if not (exclude_cold and e.tier == TIER_COLD)]
        if not active_episodes:
            return []

        # Ensure embeddings exist
        needs_embedding = [e for e in active_episodes if e.embedding is None]
        if needs_embedding:
            self.embed_episodes(needs_embedding)

        query_embedding = self.model.encode(query, convert_to_numpy=True)
        ep_embeddings = np.array([e.embedding for e in active_episodes])
        similarities = np.dot(ep_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            ep = active_episodes[idx]
            ep.retrieval_count += 1  # track usage for aging
            results.append((ep, float(similarities[idx])))

        return results

    def retrieve_by_failure_type(self, failure_type: str,
                                  episodes: List[SAMemoryItem],
                                  top_k: int = 3) -> List[SAMemoryItem]:
        """Retrieve past episodes with same failure type"""
        matching = [e for e in episodes if e.failure_type == failure_type]
        return matching[-top_k:]  # most recent

    def format_sa_context(self, retrieved_episodes: List[Tuple[SAMemoryItem, float]],
                          standing_guidelines: str = "") -> str:
        """Format SA lessons + standing guidelines for prompt injection"""
        parts = []

        # Standing guidelines (always present)
        if standing_guidelines:
            parts.append(standing_guidelines)

        # Retrieved episode lessons (per-task)
        if retrieved_episodes:
            parts.append("## Relevant Past Lessons:\n")
            for idx, (ep, score) in enumerate(retrieved_episodes, 1):
                if ep.episode_lesson:
                    parts.append(f"{idx}. {ep.episode_lesson}")
            parts.append("")

        return "\n".join(parts)
