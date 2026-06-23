from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.ai import taxonomy


class ConversationSummary(BaseModel):
    id: str
    name: str
    cover_image_url: str | None = None
    assigned_by: str | None = None  # teacher_id when teacher-assigned
    status: str = "reviewing"  # reviewing | completed


class AssignQuestionResponse(BaseModel):
    conversation_id: str
    problem: str


class GenerateQuestionRequest(BaseModel):
    prompt: str


class GenerateQuestionResponse(BaseModel):
    problem: str


class StudentSummaryResponse(BaseModel):
    summary: str


class StudentOverviewStats(BaseModel):
    total_conversations: int
    assigned_count: int
    practice_count: int
    done_count: int
    total_turns: int


class StudentOverviewResponse(BaseModel):
    student_id: str
    username: str | None = None
    conversations: list[ConversationSummary]
    stats: StudentOverviewStats


class AssignRequest(BaseModel):
    problem: str
    name: str = "שאלה ממורה"


class BulkAssignRequest(BaseModel):
    problem: str
    student_ids: list[str]
    name: str = "שאלה ממורה"


class BulkAssignResult(BaseModel):
    student_id: str
    conversation_id: str


class BulkAssignResponse(BaseModel):
    problem: str
    results: list[BulkAssignResult]


class CreateConversationRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Display name for the conversation")


class HomeworkImage(BaseModel):
    filename: str
    key: str
    url: str | None = None  # presigned GET URL so the frontend can render it


class TurnHistoryItem(BaseModel):
    turn: int
    homework_files: list[HomeworkImage]
    ai_feedback: dict[str, Any] | None


class ConversationHistory(BaseModel):
    conversation_id: str
    conversation_name: str
    history: list[TurnHistoryItem]
    status: str = "reviewing"  # reviewing | completed


class SubmitConversationResponse(BaseModel):
    review: str
    status: str = "completed"


class NextStepResponse(BaseModel):
    """Proactive recommendation of what the student should practice next."""

    concept: str | None = None  # canonical id, or null when nothing is due
    he_name: str | None = None  # Hebrew display name
    difficulty: str | None = None  # easier | same | harder
    practice_problem: str | None = None  # optionally LLM-generated, on demand


class PracticeStartResponse(BaseModel):
    """A freshly-created practice conversation, seeded with a generated problem."""

    conversation_id: str
    concept: str
    he_name: str
    difficulty: str
    problem: str


class PostTurnResult(BaseModel):
    status: str
    message: str
    turn: int
    image_keys: list[str]
    response_key: str
    ai_feedback: dict[str, Any]  # the tutor reply the backend just generated


# --- The "brain": persistent memory + per-turn reasoning artifacts -----------
#
# These three models back the smarter tutor harness. ``LearnerProfile`` is the
# student-level adaptation store, ``ConversationContext`` is the per-conversation
# managed-history store, and ``TutoringPlan`` is the structured "think before you
# speak" step. None of them change the stored turn JSON or the frontend contract;
# they live in their own sidecar S3 files (profile.json / context.json).


class ConceptMastery(BaseModel):
    """Running tally for a single math concept (e.g. "fractions").

    Mastery is tracked as a *recency-weighted* estimate (``ewma``), not a
    lifetime average, so a rough start doesn't drag the picture down forever and
    recent improvement shows up fast. ``attempts``/``correct`` are kept verbatim
    for teacher-facing stats. ``last_attempt_turn`` (the value of
    ``LearnerProfile.total_turns`` when this concept was last graded) drives
    spaced-repetition "due for review" decisions.
    """

    # How much each new attempt moves the recency-weighted mastery estimate.
    EWMA_ALPHA: ClassVar[float] = 0.5
    # Need at least this many attempts before a concept can read as "mastered".
    MASTERY_MIN_ATTEMPTS: ClassVar[int] = 3

    attempts: int = 0
    correct: int = 0
    recent_errors: list[str] = Field(default_factory=list)  # last few error_types
    # None on legacy profiles -> derived from lifetime accuracy on first use, so
    # an upgraded profile continues smoothly instead of resetting to zero.
    ewma: float | None = None
    last_attempt_turn: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts else 1.0

    def ewma_value(self) -> float:
        """Recency-weighted correctness in [0, 1] (back-compat: falls back to accuracy)."""
        return self.ewma if self.ewma is not None else self.accuracy

    @property
    def mastery_level(self) -> str:
        """Coarse bucket: unseen | struggling | developing | proficient | mastered."""
        if self.attempts == 0:
            return "unseen"
        e = self.ewma_value()
        if e < 0.4:
            return "struggling"
        if e < 0.7:
            return "developing"
        if e < 0.9:
            return "proficient"
        # Need sustained evidence before certifying full mastery.
        return "mastered" if self.attempts >= self.MASTERY_MIN_ATTEMPTS else "proficient"


class LearnerProfile(BaseModel):
    """Student-level adaptation store, accumulated across all conversations.

    Defaults match the old fixed constants, so a cold-start profile reproduces
    today's behaviour exactly and only diverges as evidence accumulates.
    """

    concepts: dict[str, ConceptMastery] = Field(default_factory=dict)
    style: str = "step_by_step"
    pace: str = "normal"
    grade: int = 8  # default middle-school grade; made per-student/dynamic later
    confidence: str = "med"
    total_turns: int = 0

    # Difficulty to practice a concept at, by how well the student knows it.
    _DIFFICULTY_BY_LEVEL: ClassVar[dict[str, str]] = {
        "unseen": "easier",
        "struggling": "easier",
        "developing": "same",
        "proficient": "harder",
        "mastered": "harder",
    }

    # How many graded attempts may elapse before an un-mastered concept is
    # considered "due" for spaced-repetition review.
    SPACED_REVIEW_GAP: ClassVar[int] = 8

    def record(self, concept: str, is_correct: bool, error_type: str | None) -> None:
        """Deterministically fold one graded attempt into mastery (no LLM math).

        ``total_turns`` is the *graded-attempt clock*: it only advances here, on
        a graded turn, so ``last_attempt_turn`` gaps measure graded attempts (the
        unit the spaced-repetition logic reasons in). Text-only turns must not
        call this.
        """
        self.total_turns += 1
        if not concept:
            return
        mastery = self.concepts.setdefault(concept, ConceptMastery())
        hit = 1.0 if is_correct else 0.0
        if mastery.attempts == 0 and mastery.ewma is None:
            # First evidence ever for a brand-new concept: take it at face value.
            mastery.ewma = hit
        else:
            # Otherwise blend; for a legacy concept (ewma=None, attempts>0) this
            # seeds from lifetime accuracy so the upgrade is continuous.
            prev = mastery.ewma_value()
            mastery.ewma = ConceptMastery.EWMA_ALPHA * hit + (1 - ConceptMastery.EWMA_ALPHA) * prev
        mastery.attempts += 1
        mastery.last_attempt_turn = self.total_turns
        if is_correct:
            mastery.correct += 1
        elif error_type and error_type != "none":
            mastery.recent_errors.append(error_type)
            mastery.recent_errors = mastery.recent_errors[-5:]

    def migrate_concepts(self) -> bool:
        """Fold legacy free-form concept keys onto the canonical taxonomy.

        Old profiles may hold non-canonical keys ("fraction", "שברים"); merge
        each into its canonical id so mastery aggregates and the recommender
        doesn't see duplicates. Returns True if anything changed (so the caller
        can persist once). Pure — no I/O.
        """
        merged: dict[str, ConceptMastery] = {}
        changed = False
        for raw_key, mastery in self.concepts.items():
            canonical = taxonomy.normalize_concept(raw_key)
            if canonical != raw_key:
                changed = True
            existing = merged.get(canonical)
            if existing is None:
                merged[canonical] = mastery
                continue
            # Merge two masteries that collapsed onto the same canonical id.
            existing.attempts += mastery.attempts
            existing.correct += mastery.correct
            existing.recent_errors = (existing.recent_errors + mastery.recent_errors)[-5:]
            existing.last_attempt_turn = max(
                existing.last_attempt_turn, mastery.last_attempt_turn
            )
            # Keep the more-attempted side's recency-weighted estimate.
            if mastery.attempts > existing.attempts - mastery.attempts:
                existing.ewma = mastery.ewma_value()
        if changed:
            self.concepts = merged
        return changed

    def is_due(self, concept: str, *, gap: int | None = None) -> bool:
        """True when an un-mastered concept hasn't been practiced in a while."""
        m = self.concepts.get(concept)
        if m is None or m.attempts == 0 or m.mastery_level == "mastered":
            return False
        threshold = self.SPACED_REVIEW_GAP if gap is None else gap
        return (self.total_turns - m.last_attempt_turn) >= threshold

    def struggle_summary(self) -> str:
        """Short text for the tutor prompt: the 1-2 weakest concepts + error type.

        Returns "" when there is nothing notable yet, so the prompt stays clean.
        Kept terse (Hebrew concept name + last error) to respect the prompt budget.
        """
        weak = [
            (name, m)
            for name, m in self.concepts.items()
            if m.mastery_level in ("struggling", "developing")
        ]
        if not weak:
            return ""
        # Weakest first (lowest recency-weighted mastery); break ties by most-attempted.
        weak.sort(key=lambda kv: (kv[1].ewma_value(), -kv[1].attempts))
        parts: list[str] = []
        for name, m in weak[:2]:
            label = taxonomy.display_name(name)
            err = m.recent_errors[-1] if m.recent_errors else ""
            parts.append(f"{label} ({err})" if err else label)
        return "struggles with " + ", ".join(parts)

    def adapt(self, *, affect: str = "neutral") -> None:
        """Re-derive the prompt dials (confidence, pace) from accumulated evidence.

        Pure and deterministic. Uses coarse buckets with wide deadbands so a
        single attempt rarely flips a dial (no turn-to-turn flapping), and treats
        ``affect`` (read from the student's message) as a one-band nudge, never a
        setter. ``style`` stays preference-driven; ``grade`` stays fixed
        (difficulty adapts via :meth:`recommend_next`). No-op until there is
        evidence, so a cold-start profile reproduces today's fixed defaults.
        """
        active = [m for m in self.concepts.values() if m.attempts > 0]
        if not active:
            return

        mean_ewma = sum(m.ewma_value() for m in active) / len(active)
        struggle_density = sum(
            1 for m in active if m.mastery_level in ("struggling", "developing")
        ) / len(active)

        # confidence: coarse 3-way target with wide deadbands...
        target = "high" if mean_ewma >= 0.75 else "low" if mean_ewma < 0.45 else "med"
        affect = affect if affect in ("frustrated", "neutral", "confident") else "neutral"
        if affect == "frustrated":
            target = {"high": "med", "med": "low", "low": "low"}[target]
        elif affect == "confident":
            target = {"low": "med", "med": "high", "high": "high"}[target]
        # ...then step at most ONE band toward it (hysteresis): confidence can
        # never lurch high<->low in a single turn, which kills oscillation.
        bands = ["low", "med", "high"]
        cur = bands.index(self.confidence) if self.confidence in bands else 1
        tgt = bands.index(target)
        self.confidence = bands[cur + (1 if tgt > cur else -1 if tgt < cur else 0)]

        # pace: slow when broadly struggling, fast when broadly solid.
        if struggle_density >= 0.6:
            self.pace = "slow"
        elif struggle_density <= 0.2 and mean_ewma >= 0.8:
            self.pace = "fast"
        else:
            self.pace = "normal"

    def recommend_next(self) -> dict[str, str] | None:
        """Pick the most valuable concept to practice next, with a difficulty.

        Prefers spaced-repetition-due concepts, then the weakest non-mastered
        one; redirects to a missing prerequisite when one is unseen/struggling so
        the student shores up the foundation first. Returns ``None`` when nothing
        needs attention (everything mastered or unseen).
        """
        # "other" is a catch-all bucket, not a teachable concept — never recommend it.
        due = [cid for cid in self.concepts if cid != "other" and self.is_due(cid)]
        candidates = due or [
            cid
            for cid, m in self.concepts.items()
            if cid != "other" and m.mastery_level in ("struggling", "developing")
        ]
        if not candidates:
            return None

        target = min(candidates, key=lambda c: self.concepts[c].ewma_value())
        # Shore up a weak/unseen prerequisite before the target itself.
        for pre in taxonomy.prerequisites_of(target):
            pm = self.concepts.get(pre)
            if pm is None or pm.mastery_level in ("unseen", "struggling"):
                target = pre
                break

        return {
            "concept": target,
            "he_name": taxonomy.display_name(target),
            "difficulty": self.difficulty_for(target),
        }

    def difficulty_for(self, concept: str) -> str:
        """Difficulty to practice a concept at, from the student's mastery of it."""
        level = self.concepts[concept].mastery_level if concept in self.concepts else "unseen"
        return self._DIFFICULTY_BY_LEVEL[level]


class Exchange(BaseModel):
    """One verbatim student/tutor round, kept for short-term recall."""

    student: str = ""
    tutor: str = ""


class ConversationContext(BaseModel):
    """Per-conversation managed history — the dynamic "context file".

    Instead of re-rendering raw turn files each turn, the tutor reads this
    compact, continuously-updated memory: a rolling summary, the current problem,
    and the last couple of verbatim exchanges.
    """

    rolling_summary: str = ""
    current_problem: str | None = None
    recent_exchanges: list[Exchange] = Field(default_factory=list)
    turn_count: int = 0
    # Within-problem hint ladder: a normalized signature of the problem the
    # student is currently working, and how many times they've attempted *this*
    # problem without solving it. scaffold_level is derived from the count.
    problem_signature: str = ""
    attempts_on_problem: int = 0

    @property
    def scaffold_level(self) -> int:
        """0 = gentle nudge, 1 = more specific, 2 = near-worked-step (capped)."""
        return min(max(self.attempts_on_problem - 1, 0), 2)

    def compute_problem_state(
        self, problem: str | None, is_correct: bool | None
    ) -> tuple[str, int]:
        """Return the (signature, attempts_on_problem) this turn yields, no mutation.

        - text-only turn (no problem) -> unchanged (student still on same problem)
        - unclear/empty problem        -> unchanged (don't reset on a bad photo)
        - correct                      -> attempts reset to 0 (problem is solved)
        - same problem, still wrong    -> attempts + 1 (escalate the hint ladder)
        - a new problem                -> attempts reset to 1
        """
        old_sig, old_attempts = self.problem_signature, self.attempts_on_problem
        if problem is None:
            return old_sig, old_attempts
        new_sig = taxonomy.problem_signature(problem)
        if not new_sig:
            return old_sig, old_attempts
        if is_correct:
            return new_sig, 0
        if taxonomy.same_problem(new_sig, old_sig):
            return new_sig, old_attempts + 1
        return new_sig, 1

    def render(self) -> str:
        """Flatten the memory into a prompt-ready block ("" when empty)."""
        parts: list[str] = []
        if self.current_problem:
            parts.append(f"Current problem: {self.current_problem}")
        if self.rolling_summary:
            parts.append(f"Session so far: {self.rolling_summary}")
        if self.recent_exchanges:
            parts.append("Recent exchanges:")
            for ex in self.recent_exchanges:
                if ex.student:
                    parts.append(f"  Student: {ex.student}")
                if ex.tutor:
                    parts.append(f"  Tutor: {ex.tutor}")
        return "\n".join(parts)


class TutoringPlan(BaseModel):
    """Structured private plan the model produces before it writes the reply."""

    misconception: str  # the precise underlying misunderstanding
    next_move: str  # the single pedagogical step to take now
    do_not_reveal: str  # what must NOT be said (never the final answer)
    guiding_question: str  # one question that nudges the student forward
    # Affect read from the student's message, used to nudge the confidence dial.
    # Rides this existing call (no extra LLM round-trip); validated on read.
    student_affect: str = "neutral"  # frustrated | neutral | confident

    def render(self) -> str:
        return (
            f"Misconception: {self.misconception}\n"
            f"Next move: {self.next_move}\n"
            f"Do not reveal: {self.do_not_reveal}\n"
            f"Guiding question: {self.guiding_question}"
        )
