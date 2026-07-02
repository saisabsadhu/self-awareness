# src/memory.py
# Evolving Self-Awareness system with two-level, multi-schema support
# Terminology: low-level AWARENESS (episodic) + high-level AWARENESS (policies)
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter


# ── Schema categories for low-level awareness ───────────────────────────────
SCHEMA_CATEGORIES = [
    "values",           # did response align with principles?
    "safety",           # was the action safe?
    "capability",       # did model know what it can/can't do?
    "user_understanding",  # did model respect user preferences?
    "env_understanding",   # did model use tools/constraints correctly?
    "strategy",         # reasoning approach (original ReasoningBank)
]

# ── Failure types ────────────────────────────────────────────────────────────
FAILURE_TYPES = [
    "CORRECT",
    "KNOWLEDGE_BOUNDARY_MISSED",  # attempted unanswerable
    "FALSE_IDK",                  # abstained on answerable
    "OVERCONFIDENT_WRONG",        # high confidence, wrong
    "UNDERCONFIDENT_RIGHT",       # low confidence, correct
    "CAPABILITY_OVERCLAIM",       # claimed capability it lacks
    "CAPABILITY_UNDERCLAIM",      # denied capability it has
    "FACTUAL_ERROR",              # wrong fact, mid confidence
    "REASONING_ERROR",            # correct knowledge, wrong logic
]

# ── Tiering for memory aging ────────────────────────────────────────────────
TIER_HOT = "hot"      # recent, frequently retrieved
TIER_WARM = "warm"    # older, occasionally useful
TIER_COLD = "cold"    # old, rarely used — candidate for archival


# ── Low-level: Episodic Memory Item ─────────────────────────────────────────
@dataclass
class MemoryItem:
    """Original ReasoningBank memory item — kept for backward compatibility"""
    title: str
    description: str
    content: str
    source_problem_id: str
    success: bool
    created_at: str
    embedding: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SAMemoryItem:
    """Self-Awareness enriched memory item (low-level, episodic)"""
    # ── context ──
    question: str
    model_answer: str
    ground_truth: str
    domain: str                              # extracted topic/domain
    source_problem_id: str
    created_at: str

    # ── outcome ──
    outcome: str                             # "correct" | "incorrect"
    
    # ── SA signals ──
    failure_type: str                        # one of FAILURE_TYPES
    confidence: float                        # verbal confidence 0-1
    knowledge_gap: str                       # what specifically was missing
    schema_categories: List[str]             # which schemas this touches

    # ── lesson ──
    episode_lesson: str                      # "When [X], model did [Y]. Next time: [Z]"

    # ── original strategy (backward compat with ReasoningBank) ──
    strategy_title: str = ""
    strategy_content: str = ""
    success: bool = False

    # ── retrieval ──
    embedding: Optional[List[float]] = None
    tier: str = TIER_HOT                     # hot / warm / cold
    retrieval_count: int = 0                 # how often retrieved

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── High-level: Policy / Guideline ──────────────────────────────────────────
# Three categories inspired by OpenClaw:
#   policies   ≈ SOUL.md   (long-term values, rules)
#   strategies ≈ AGENTS.md (reasoning guidelines)
#   capabilities ≈ TOOLS.md (what model can/can't do)

HIGH_LEVEL_CATEGORIES = ["policies", "strategies", "capabilities"]


@dataclass
class HighLevelMemory:
    """Distilled high-level guideline / policy"""
    id: str
    category: str                            # policies | strategies | capabilities
    pattern: str                             # observed pattern
    guideline: str                           # "When [trigger], do [action]"
    trigger_condition: str                   # explicit trigger
    source_episodes: List[str]               # which low-level episodes contributed
    n_episodes: int                          # how many episodes support this
    domain: str                              # domain this applies to
    failure_type: str                        # associated failure type
    created_at: str
    updated_at: str
    active: bool = True                      # can be deactivated if contradicted

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── ReasoningBank (original, kept for backward compat) ──────────────────────
class ReasoningBank:
    """Original memory storage — unchanged"""

    def __init__(self, storage_path='memory_bank/reasoning_bank.json'):
        self.storage_path = storage_path
        self.memories: List[MemoryItem] = []
        self.load()

    def add_memory(self, memory: MemoryItem):
        self.memories.append(memory)
        self.save()

    def add_memories(self, memories: List[MemoryItem]):
        self.memories.extend(memories)
        self.save()

    def get_all_memories(self) -> List[MemoryItem]:
        return self.memories

    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump([m.to_dict() for m in self.memories], f, indent=2)

    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.memories = [MemoryItem.from_dict(m) for m in data]
        except FileNotFoundError:
            self.memories = []

    def clear(self):
        self.memories = []
        self.save()

    def __len__(self):
        return len(self.memories)


# ── SA Memory Bank (two-level) ──────────────────────────────────────────────
class SAMemoryBank:
    """Two-level memory bank with multi-schema support"""

    def __init__(self,
                 episodic_path='memory_bank/sa_episodic.json',
                 high_level_dir='memory_bank/high_level',
                 consolidation_interval=20,
                 promotion_threshold=3):
        self.episodic_path = episodic_path
        self.high_level_dir = high_level_dir
        self.consolidation_interval = consolidation_interval
        self.promotion_threshold = promotion_threshold

        self.episodes: List[SAMemoryItem] = []
        self.high_level: Dict[str, List[HighLevelMemory]] = {
            cat: [] for cat in HIGH_LEVEL_CATEGORIES
        }

        self._load_episodes()
        self._load_high_level()
        self._hl_counter = 0  # for generating unique IDs

    # ── Add episodes ─────────────────────────────────────────────────────
    def add_episode(self, episode: SAMemoryItem):
        """Add a single episodic memory and check if consolidation is needed"""
        self.episodes.append(episode)
        self._save_episodes()

        # Trigger consolidation every N episodes
        if len(self.episodes) % self.consolidation_interval == 0:
            self.consolidate()

    def add_episodes(self, episodes: List[SAMemoryItem]):
        self.episodes.extend(episodes)
        self._save_episodes()

    # ── Get episodes ─────────────────────────────────────────────────────
    def get_all_episodes(self) -> List[SAMemoryItem]:
        return self.episodes

    def get_episodes_by_schema(self, schema: str) -> List[SAMemoryItem]:
        return [e for e in self.episodes if schema in e.schema_categories]

    def get_episodes_by_failure_type(self, ftype: str) -> List[SAMemoryItem]:
        return [e for e in self.episodes if e.failure_type == ftype]

    def get_recent_episodes(self, n: int = 20) -> List[SAMemoryItem]:
        return self.episodes[-n:]

    # ── Get high-level ───────────────────────────────────────────────────
    def get_policies(self) -> List[HighLevelMemory]:
        return [h for h in self.high_level["policies"] if h.active]

    def get_strategies(self) -> List[HighLevelMemory]:
        return [h for h in self.high_level["strategies"] if h.active]

    def get_capabilities(self) -> List[HighLevelMemory]:
        return [h for h in self.high_level["capabilities"] if h.active]

    def get_all_active_guidelines(self) -> List[HighLevelMemory]:
        """Get all active high-level guidelines across categories"""
        guidelines = []
        for cat in HIGH_LEVEL_CATEGORIES:
            guidelines.extend([h for h in self.high_level[cat] if h.active])
        return guidelines

    # ── Consolidation: promote low-level → high-level ────────────────────
    def consolidate(self) -> List[HighLevelMemory]:
        """
        Group recent episodes by failure_type + domain.
        If a pattern recurs >= promotion_threshold times, promote to high-level.
        Returns list of newly promoted guidelines.
        """
        new_guidelines = []

        # Group by (failure_type, domain)
        groups: Dict[tuple, List[SAMemoryItem]] = {}
        for ep in self.episodes:
            key = (ep.failure_type, ep.domain)
            if key not in groups:
                groups[key] = []
            groups[key].append(ep)

        for (ftype, domain), group in groups.items():
            if ftype == "CORRECT":
                continue  # don't promote "everything is fine" patterns

            if len(group) >= self.promotion_threshold:
                # Check if a guideline for this pattern already exists
                existing = self._find_existing_guideline(ftype, domain)
                if existing:
                    # Update existing guideline with new evidence
                    existing.n_episodes = len(group)
                    existing.source_episodes = [ep.source_problem_id for ep in group[-10:]]
                    existing.updated_at = datetime.now().isoformat()
                    continue

                # Create new high-level guideline
                self._hl_counter += 1
                guideline = HighLevelMemory(
                    id=f"hl_{self._hl_counter:04d}",
                    category=self._classify_category(ftype),
                    pattern=f"Model shows {ftype} pattern in {domain} ({len(group)} episodes)",
                    guideline="",       # will be filled by LLM during composition
                    trigger_condition="",  # will be filled by LLM
                    source_episodes=[ep.source_problem_id for ep in group[-10:]],
                    n_episodes=len(group),
                    domain=domain,
                    failure_type=ftype,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                )
                new_guidelines.append(guideline)

        # Add new guidelines to the appropriate category
        for g in new_guidelines:
            self.high_level[g.category].append(g)

        self._save_high_level()
        return new_guidelines

    def _find_existing_guideline(self, ftype: str, domain: str) -> Optional[HighLevelMemory]:
        for cat in HIGH_LEVEL_CATEGORIES:
            for g in self.high_level[cat]:
                if g.failure_type == ftype and g.domain == domain and g.active:
                    return g
        return None

    def _classify_category(self, failure_type: str) -> str:
        """Map failure type to high-level category"""
        if failure_type in ["CAPABILITY_OVERCLAIM", "CAPABILITY_UNDERCLAIM"]:
            return "capabilities"
        elif failure_type in ["KNOWLEDGE_BOUNDARY_MISSED", "FALSE_IDK",
                              "OVERCONFIDENT_WRONG", "UNDERCONFIDENT_RIGHT",
                              "FACTUAL_ERROR", "REASONING_ERROR"]:
            return "strategies"
        else:
            return "policies"

    # ── Memory aging ─────────────────────────────────────────────────────
    def age_memories(self):
        """Move memories from HOT → WARM → COLD based on age and retrieval count"""
        now = datetime.now()
        for ep in self.episodes:
            try:
                created = datetime.fromisoformat(ep.created_at)
                age_days = (now - created).days
            except:
                continue

            if ep.tier == TIER_HOT and age_days > 30 and ep.retrieval_count < 3:
                ep.tier = TIER_WARM
            elif ep.tier == TIER_WARM and age_days > 90 and ep.retrieval_count < 5:
                ep.tier = TIER_COLD

        self._save_episodes()

    # ── Format for prompt injection ──────────────────────────────────────
    def format_standing_guidelines(self) -> str:
        """Format high-level policies for injection into every prompt"""
        guidelines = self.get_all_active_guidelines()
        if not guidelines:
            return ""

        formatted = "## Standing Guidelines (from past experience):\n\n"
        for cat in HIGH_LEVEL_CATEGORIES:
            cat_guidelines = [g for g in guidelines if g.category == cat]
            if not cat_guidelines:
                continue
            formatted += f"### {cat.title()}:\n"
            for g in cat_guidelines:
                formatted += f"- {g.guideline}\n"
            formatted += "\n"

        return formatted

    # ── Stats ────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        """Get summary statistics of the memory bank"""
        failure_counts = Counter(ep.failure_type for ep in self.episodes)
        schema_counts = Counter(
            schema for ep in self.episodes for schema in ep.schema_categories
        )
        return {
            "total_episodes": len(self.episodes),
            "failure_distribution": dict(failure_counts),
            "schema_distribution": dict(schema_counts),
            "high_level_policies": len(self.get_policies()),
            "high_level_strategies": len(self.get_strategies()),
            "high_level_capabilities": len(self.get_capabilities()),
            "tier_distribution": dict(Counter(ep.tier for ep in self.episodes)),
        }

    # ── Persistence ──────────────────────────────────────────────────────
    def _save_episodes(self):
        with open(self.episodic_path, 'w') as f:
            json.dump([e.to_dict() for e in self.episodes], f, indent=2)

    def _load_episodes(self):
        try:
            with open(self.episodic_path, 'r') as f:
                data = json.load(f)
                self.episodes = [SAMemoryItem.from_dict(d) for d in data]
        except FileNotFoundError:
            self.episodes = []

    def _save_high_level(self):
        import os
        os.makedirs(self.high_level_dir, exist_ok=True)
        for cat in HIGH_LEVEL_CATEGORIES:
            path = f"{self.high_level_dir}/{cat}.json"
            with open(path, 'w') as f:
                json.dump([h.to_dict() for h in self.high_level[cat]], f, indent=2)

    def _load_high_level(self):
        import os
        for cat in HIGH_LEVEL_CATEGORIES:
            path = f"{self.high_level_dir}/{cat}.json"
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.high_level[cat] = [HighLevelMemory.from_dict(d) for d in data]
            except FileNotFoundError:
                self.high_level[cat] = []

    def clear(self):
        self.episodes = []
        self.high_level = {cat: [] for cat in HIGH_LEVEL_CATEGORIES}
        self._save_episodes()
        self._save_high_level()

    def __len__(self):
        return len(self.episodes)


# ── Self-Model: aggregated self-awareness from episodes ─────────────────────
class SelfModel:
    """
    The Evolving Self-Model.
    
    This is NOT a separate data structure — it is computed from the SA memory bank.
    It answers the question: "What does the model know about itself?"
    
    Four dimensions:
    1. Domain competence — accuracy and failure rates per domain
    2. Calibration quality — is confidence aligned with actual accuracy?
    3. Failure profile — what kinds of errors does the model make?
    4. Capability boundary — what does the model over/under-claim?
    
    The self-model is recomputed after training and used during testing
    to enable self-aware behavior (pre-task self-query).
    """

    def __init__(self, sa_memory: SAMemoryBank):
        self.sa_memory = sa_memory
        self.domain_stats: Dict[str, Dict] = {}
        self.overall_stats: Dict = {}
        self.calibration: Dict[str, float] = {}
        self.failure_profile: Dict[str, int] = {}
        self.weak_domains: List[str] = []
        self.strong_domains: List[str] = []
        self.domain_thresholds: Dict[str, float] = {}

    def build(self):
        """Build/rebuild the self-model from all episodes in memory"""
        episodes = self.sa_memory.get_all_episodes()
        if not episodes:
            return

        # ── 1. Domain competence ─────────────────────────────────────────
        domain_groups: Dict[str, List[SAMemoryItem]] = {}
        for ep in episodes:
            domain = ep.domain
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(ep)

        for domain, eps in domain_groups.items():
            total = len(eps)
            correct = sum(1 for e in eps if e.outcome == "correct")
            accuracy = correct / total if total > 0 else 0
            avg_confidence = sum(e.confidence for e in eps) / total if total > 0 else 0.5

            # Calibration gap: |avg_confidence - accuracy|
            calibration_gap = abs(avg_confidence - accuracy)

            # Failure type distribution for this domain
            domain_failures = Counter(e.failure_type for e in eps if e.outcome != "correct")

            self.domain_stats[domain] = {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "calibration_gap": calibration_gap,
                "failure_types": dict(domain_failures),
                "most_common_failure": domain_failures.most_common(1)[0][0] if domain_failures else "CORRECT",
            }

        # ── 2. Identify weak and strong domains ──────────────────────────
        domain_accuracies = [(d, s["accuracy"]) for d, s in self.domain_stats.items() if s["total"] >= 3]
        if domain_accuracies:
            avg_acc = sum(a for _, a in domain_accuracies) / len(domain_accuracies)
            self.weak_domains = [d for d, a in domain_accuracies if a < avg_acc - 0.1]
            self.strong_domains = [d for d, a in domain_accuracies if a > avg_acc + 0.1]

        # ── 3. Domain-specific confidence thresholds ─────────────────────
        # For each domain, compute the optimal IDK threshold
        # If accuracy is low, threshold should be lower (more willing to say IDK)
        for domain, stats in self.domain_stats.items():
            if stats["total"] < 3:
                self.domain_thresholds[domain] = 0.5  # default
            else:
                # Lower accuracy → lower threshold → more likely to abstain
                self.domain_thresholds[domain] = max(0.2, min(0.8, stats["accuracy"]))

        # ── 4. Overall failure profile ───────────────────────────────────
        self.failure_profile = dict(Counter(ep.failure_type for ep in episodes))

        # ── 5. Overall calibration ───────────────────────────────────────
        total = len(episodes)
        overall_correct = sum(1 for e in episodes if e.outcome == "correct")
        overall_accuracy = overall_correct / total if total > 0 else 0
        overall_avg_conf = sum(e.confidence for e in episodes) / total if total > 0 else 0.5
        self.overall_stats = {
            "total_episodes": total,
            "overall_accuracy": overall_accuracy,
            "overall_avg_confidence": overall_avg_conf,
            "overall_calibration_gap": abs(overall_avg_conf - overall_accuracy),
            "n_domains": len(self.domain_stats),
            "n_weak_domains": len(self.weak_domains),
            "n_strong_domains": len(self.strong_domains),
        }

    def query(self, domain: str) -> Dict:
        """
        Pre-task self-query: given a domain, what does the model know about
        its own performance in that domain?
        
        This is the key self-awareness function — called BEFORE answering each question.
        """
        if domain in self.domain_stats:
            stats = self.domain_stats[domain]
            return {
                "known_domain": True,
                "accuracy": stats["accuracy"],
                "avg_confidence": stats["avg_confidence"],
                "calibration_gap": stats["calibration_gap"],
                "most_common_failure": stats["most_common_failure"],
                "is_weak": domain in self.weak_domains,
                "is_strong": domain in self.strong_domains,
                "confidence_threshold": self.domain_thresholds.get(domain, 0.5),
                "n_episodes": stats["total"],
            }
        else:
            # Unknown domain — no self-knowledge, flag uncertainty
            return {
                "known_domain": False,
                "accuracy": None,
                "avg_confidence": None,
                "calibration_gap": None,
                "most_common_failure": None,
                "is_weak": None,
                "is_strong": None,
                "confidence_threshold": 0.5,
                "n_episodes": 0,
            }

    def format_self_awareness_context(self, domain: str) -> str:
        """
        Format self-model query results for prompt injection.
        This is what makes the module SELF-AWARE rather than just memory.
        
        Injected before each task so the model knows its own strengths/weaknesses.
        """
        q = self.query(domain)
        if not q["known_domain"]:
            return (
                "## Self-Awareness Note:\n"
                f"This question is in the domain '{domain}', which I have not encountered before. "
                "I should be cautious and express uncertainty if I'm not confident.\n"
            )

        parts = ["## Self-Awareness Note:"]
        parts.append(f"Domain: {domain} (seen {q['n_episodes']} similar questions before)")
        parts.append(f"My accuracy in this domain: {q['accuracy']:.0%}")

        if q["is_weak"]:
            parts.append(f"⚠ This is a WEAK domain for me. My most common error here: {q['most_common_failure']}.")
            parts.append("I should be extra cautious and willing to say 'I don't know' if uncertain.")
        elif q["is_strong"]:
            parts.append("This is a strong domain for me. I can answer with reasonable confidence.")

        if q["calibration_gap"] and q["calibration_gap"] > 0.15:
            parts.append(f"⚠ My confidence is poorly calibrated in this domain (gap: {q['calibration_gap']:.2f}). "
                        "I should not trust my gut feeling about how confident I am.")

        if q["most_common_failure"] == "KNOWLEDGE_BOUNDARY_MISSED":
            parts.append("I tend to answer questions in this domain even when I shouldn't. "
                        "If this question seems unanswerable, I should say so.")
        elif q["most_common_failure"] == "FALSE_IDK":
            parts.append("I tend to say 'I don't know' too often in this domain. "
                        "I should attempt an answer if I have any relevant knowledge.")

        return "\n".join(parts) + "\n"

    def format_summary(self) -> str:
        """Human-readable summary of the self-model"""
        lines = ["=" * 60, "EVOLVING SELF-MODEL SUMMARY", "=" * 60]

        if self.overall_stats:
            o = self.overall_stats
            lines.append(f"Total episodes: {o['total_episodes']}")
            lines.append(f"Overall accuracy: {o['overall_accuracy']:.2%}")
            lines.append(f"Overall calibration gap: {o['overall_calibration_gap']:.3f}")
            lines.append(f"Domains seen: {o['n_domains']}")

        if self.weak_domains:
            lines.append(f"\nWeak domains: {', '.join(self.weak_domains)}")
        if self.strong_domains:
            lines.append(f"Strong domains: {', '.join(self.strong_domains)}")

        lines.append(f"\nFailure profile: {self.failure_profile}")

        if self.domain_stats:
            lines.append("\nPer-domain breakdown:")
            for domain, stats in sorted(self.domain_stats.items(), key=lambda x: x[1]["accuracy"]):
                weak_marker = " ⚠ WEAK" if domain in self.weak_domains else ""
                strong_marker = " ✓ STRONG" if domain in self.strong_domains else ""
                lines.append(
                    f"  {domain:25s}: acc={stats['accuracy']:.0%}, "
                    f"conf={stats['avg_confidence']:.2f}, "
                    f"cal_gap={stats['calibration_gap']:.2f}, "
                    f"n={stats['total']}{weak_marker}{strong_marker}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "domain_stats": self.domain_stats,
            "overall_stats": self.overall_stats,
            "failure_profile": self.failure_profile,
            "weak_domains": self.weak_domains,
            "strong_domains": self.strong_domains,
            "domain_thresholds": self.domain_thresholds,
        }

    def save(self, path: str = "memory_bank/self_model.json"):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
