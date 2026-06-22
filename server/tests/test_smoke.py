"""Fast, dependency-light smoke tests for the AI-enabled backend.

No real Gemini key, no S3/MinIO required. The Gemini calls are monkeypatched
and the S3-backed conversation service is replaced with an in-memory fake, so
this runs offline as a CI gate. Run with: python -m pytest server/tests -q
(from repo root) or `pytest` from inside server/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.ai.llm as llm  # noqa: E402
from app.ai.prompts import (  # noqa: E402
    STYLES,
    SYSTEM_PROMPT_BUDGET,
    AnalysisResult,
    build_tutor_system,
)
from app.config import settings as app_settings  # noqa: E402
from app.exceptions import FileKeyNotFoundError  # noqa: E402
from app.schemas.tutor import (  # noqa: E402
    ConceptMastery,
    ConversationContext,
    LearnerProfile,
    TutoringPlan,
)


class FakeS3:
    """In-memory stand-in for AsyncS3Service's JSON get/put used by the brain."""

    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    async def get_object_as_json(self, bucket: str, key: str):
        if key not in self.store:
            raise FileKeyNotFoundError(bucket, key)
        return self.store[key]

    async def put_object_json(self, bucket: str, key: str, data) -> None:
        self.store[key] = data


def _fake_plan(*_a, **_k) -> TutoringPlan:
    return TutoringPlan(
        misconception="forgot common denominator",
        next_move="ask for the common denominator",
        do_not_reveal="the final sum",
        guiding_question="What is the common denominator?",
    )


# --- Prompt budget / style rules (ported from the AI-engine branch) ----------

def test_prompt_stays_under_budget():
    pathological = "weak: " + ", ".join(["fractions"] * 50)
    system = build_tutor_system("socratic", "fast", 8, "low", pathological)
    assert len(system) <= SYSTEM_PROMPT_BUDGET


def test_prompt_changes_by_style():
    rendered = {build_tutor_system(s, "normal", 6, "med") for s in STYLES}
    assert len(rendered) == len(STYLES)  # each style yields a distinct prompt


def test_unknown_enum_falls_back():
    system = build_tutor_system("bogus", "bogus", 6, "bogus")
    assert "middle-school math tutor" in system


# --- End-to-end turn flow (mocked Gemini + in-memory conversation store) -----

def _fake_analysis():
    return AnalysisResult(
        problem="2/3 + 1/4",
        is_correct=False,
        error_type="arithmetic",
        concept="fractions",
        confidence=0.9,
    )


def _make_client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(llm, "analyze", lambda *a, **k: _fake_analysis())
    monkeypatch.setattr(llm, "tutor", lambda system, context: "What is the common denominator?")
    monkeypatch.setattr(llm, "plan", _fake_plan)
    monkeypatch.setattr(llm, "summarize", lambda prev, latest: "session summary")

    from app.main import app
    from app.dependencies import get_conversation_service, get_tutor_ai_service
    from app.schemas.tutor import PostTurnResult
    from app.services.context_service import ContextService
    from app.services.profile_service import ProfileService
    from app.services.tutor_ai_service import TutorAIService

    class FakeConversationService:
        def __init__(self):
            self.posted = []

        async def get_history(self, student_id, conversation_id):
            from app.exceptions import ConversationNotFoundError

            raise ConversationNotFoundError(student_id, conversation_id)

        async def post_turn(self, **kwargs):
            self.posted.append(kwargs)
            return PostTurnResult(
                status="success",
                message="ok",
                turn=kwargs["turn_number"],
                image_keys=["k.jpg"] if kwargs["images"] else [],
                response_key="resp.json",
                ai_feedback=kwargs["feedback_data"],
            )

    fake = FakeConversationService()
    fake_s3 = FakeS3()
    profiles = ProfileService(fake_s3, app_settings)
    contexts = ContextService(fake_s3, app_settings, fake)
    ai = TutorAIService(profiles, contexts, app_settings)

    app.dependency_overrides[get_conversation_service] = lambda: fake
    app.dependency_overrides[get_tutor_ai_service] = lambda: ai
    return TestClient(app), fake


def test_turn_with_image_generates_feedback(monkeypatch):
    client, fake = _make_client(monkeypatch)
    try:
        resp = client.post(
            "/students/demo/conversations/c1/turn",
            data={"conversation_name": "Fractions", "turn_number": "0"},
            files={"images": ("hw.jpg", b"\xff\xd8\xff fake-jpeg-bytes", "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ai_feedback"]["reply"] == "What is the common denominator?"
        assert body["ai_feedback"]["concept"] == "fractions"
        assert body["ai_feedback"]["is_correct"] is False
        assert fake.posted[0]["images"]  # image was forwarded to storage
    finally:
        client.app.dependency_overrides.clear()


def test_text_only_turn_generates_reply(monkeypatch):
    client, fake = _make_client(monkeypatch)
    try:
        resp = client.post(
            "/students/demo/conversations/c1/turn",
            data={
                "conversation_name": "Fractions",
                "turn_number": "1",
                "student_text": "I still don't get it",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ai_feedback"]["reply"] == "What is the common denominator?"
        # text-only turn: no analysis verdict, but the message is persisted
        assert body["ai_feedback"]["is_correct"] is None
        assert body["ai_feedback"]["student_text"] == "I still don't get it"
        assert not fake.posted[0]["images"]
    finally:
        client.app.dependency_overrides.clear()


def test_path_traversal_ids_rejected(monkeypatch):
    client, _ = _make_client(monkeypatch)
    try:
        # An id containing a dot (the "../" building block) is rejected by the
        # pattern before any S3 key is built — proves the regex guard fires.
        resp = client.get("/students/demo/conversations/a.b")
        assert resp.status_code == 422, resp.text
        # Encoded slashes / ".." never route to the handler at all.
        for bad in ["..%2f..%2fsecret", ".."]:
            resp = client.get(f"/students/demo/conversations/{bad}")
            assert resp.status_code in (404, 422), (bad, resp.status_code)
        resp = client.get("/students/a%2fb/conversations/c1")
        assert resp.status_code in (404, 422)
    finally:
        client.app.dependency_overrides.clear()


def test_disallowed_extension_falls_back():
    from app.utils.files import get_file_extension

    assert get_file_extension("evil.php") == "jpg"
    assert get_file_extension("../../x.svg") == "jpg"
    assert get_file_extension("photo.PNG") == "png"


def test_empty_turn_rejected(monkeypatch):
    client, _ = _make_client(monkeypatch)
    try:
        resp = client.post(
            "/students/demo/conversations/c1/turn",
            data={"conversation_name": "Fractions", "turn_number": "2"},
        )
        assert resp.status_code == 400
    finally:
        client.app.dependency_overrides.clear()


# --- Adaptation: deterministic mastery math (no S3, no LLM) -------------------

def test_profile_records_mastery_and_ranks_struggle():
    from app.ai.taxonomy import display_name

    p = LearnerProfile()
    p.record("fractions", is_correct=False, error_type="arithmetic")
    p.record("fractions", is_correct=False, error_type="sign")
    p.record("decimals", is_correct=True, error_type="none")

    assert p.total_turns == 3
    assert p.concepts["fractions"].attempts == 2
    assert p.concepts["fractions"].correct == 0
    assert p.concepts["decimals"].accuracy == 1.0

    summary = p.struggle_summary()
    assert display_name("fractions") in summary  # weakest concept surfaces (Hebrew name)
    assert "sign" in summary  # carries the latest error type
    assert display_name("decimals") not in summary  # all-correct concept is not a struggle


def test_profile_struggle_empty_when_no_evidence():
    assert LearnerProfile().struggle_summary() == ""
    p = LearnerProfile()
    # A single correct answer is "proficient", not a struggle.
    p.record("fractions", is_correct=True, error_type="none")
    assert p.struggle_summary() == ""


# --- Phase 1: canonical taxonomy ---------------------------------------------

def test_normalize_concept_maps_to_canonical():
    from app.ai.taxonomy import CONCEPTS, normalize_concept

    assert normalize_concept("fraction") == "fractions"
    assert normalize_concept("שברים") == "fractions"
    assert normalize_concept("solving equations") == "linear_equations"
    assert normalize_concept("fractions") == "fractions"  # canonical passes through
    assert normalize_concept("totally unknown topic") == "other"
    assert normalize_concept("") == "other"
    assert "other" in CONCEPTS  # the fallback is a real entry


def test_prerequisites_and_candidates():
    from app.ai.taxonomy import candidate_ids, prerequisites_of

    assert "whole_number_arithmetic" in prerequisites_of("fractions")
    assert prerequisites_of("other") == ()
    assert "other" not in candidate_ids()  # not offered to the grader


def test_profile_migrates_legacy_concept_keys():
    # A legacy profile with free-form keys folds onto the taxonomy on migrate.
    p = LearnerProfile.model_validate(
        {"concepts": {"fraction": {"attempts": 2, "correct": 1},
                       "שברים": {"attempts": 1, "correct": 0}}}
    )
    assert p.migrate_concepts() is True
    assert set(p.concepts) == {"fractions"}  # both collapsed onto one canonical id
    assert p.concepts["fractions"].attempts == 3
    assert p.migrate_concepts() is False  # idempotent


# --- Phase 2: EWMA mastery model + spaced repetition -------------------------

def test_mastery_levels_by_evidence():
    p = LearnerProfile()
    assert ConceptMastery().mastery_level == "unseen"

    # Three correct from cold -> mastered (sustained evidence).
    for _ in range(3):
        p.record("decimals", is_correct=True, error_type="none")
    assert p.concepts["decimals"].mastery_level == "mastered"

    # One correct is strong but not yet certified as mastered.
    p.record("percentages", is_correct=True, error_type="none")
    assert p.concepts["percentages"].mastery_level == "proficient"

    # Three wrong from cold -> struggling.
    for _ in range(3):
        p.record("integers", is_correct=False, error_type="sign")
    assert p.concepts["integers"].mastery_level == "struggling"


def test_mastery_backward_compat_from_accuracy():
    # Legacy mastery with no ewma derives from lifetime accuracy, then continues.
    m = ConceptMastery(attempts=5, correct=4)  # ewma is None
    assert m.ewma_value() == 0.8
    p = LearnerProfile(concepts={"fractions": m}, total_turns=5)
    p.record("fractions", is_correct=True, error_type="none")
    # Continued from 0.8 (0.5*1 + 0.5*0.8 = 0.9), not reset to 0.
    assert abs(p.concepts["fractions"].ewma - 0.9) < 1e-9


def test_spaced_repetition_due():
    p = LearnerProfile()
    p.record("fractions", is_correct=False, error_type="arithmetic")  # turn 1
    assert p.is_due("fractions") is False  # just practiced
    p.total_turns += 8  # simulate 8 graded attempts on other concepts
    assert p.is_due("fractions") is True
    # A mastered concept is never "due".
    for _ in range(3):
        p.record("decimals", is_correct=True, error_type="none")
    p.total_turns += 50
    assert p.is_due("decimals") is False


# --- Phase 3: closing the adaptation loop ------------------------------------

def test_adapt_noop_on_cold_start():
    p = LearnerProfile()
    p.adapt(affect="frustrated")
    assert (p.style, p.pace, p.grade, p.confidence) == ("step_by_step", "normal", 8, "med")


def test_adapt_sets_pace_and_confidence_from_evidence():
    strong = LearnerProfile()
    for _ in range(3):
        strong.record("decimals", is_correct=True, error_type="none")
    strong.adapt()
    assert strong.confidence == "high"
    assert strong.pace == "fast"

    weak = LearnerProfile()
    for _ in range(3):
        weak.record("integers", is_correct=False, error_type="sign")
    weak.adapt()
    assert weak.confidence == "low"
    assert weak.pace == "slow"


def test_adapt_affect_is_a_one_band_nudge():
    p = LearnerProfile()
    for _ in range(3):
        p.record("fractions", is_correct=True, error_type="none")  # mean ~ "high"
    p.adapt(affect="frustrated")
    assert p.confidence == "med"  # nudged down exactly one band, not to "low"


def test_adapt_confidence_never_skips_a_band():
    # Alternating correct/wrong is a worst case for oscillation; hysteresis must
    # keep confidence from ever lurching low<->high in a single turn.
    bands = ["low", "med", "high"]
    p = LearnerProfile()
    seq = []
    for i in range(6):
        p.record("fractions", is_correct=(i % 2 == 0), error_type="arithmetic")
        p.adapt()
        seq.append(p.confidence)
    for prev, cur in zip(seq, seq[1:]):
        assert abs(bands.index(prev) - bands.index(cur)) <= 1  # no skipped band
    assert len(set(seq)) > 1  # still responsive, not frozen


# --- Phase 4: within-problem hint ladder -------------------------------------

def test_problem_signature_absorbs_transcription_noise():
    from app.ai.taxonomy import problem_signature, same_problem

    a = problem_signature("2/3 + 1/4")
    b = problem_signature("2/3+1/4")
    c = problem_signature("חשבי 2/3 + 1/4")
    assert same_problem(a, b)
    assert same_problem(a, c)
    assert not same_problem(a, problem_signature("5*7 - 2"))
    assert problem_signature("(unclear)") == ""
    assert not same_problem("", "")  # empty never matches


def test_hint_ladder_escalates_and_resets():
    ctx = ConversationContext()
    # First wrong attempt on a problem -> attempts 1 (scaffold 0).
    sig, n = ctx.compute_problem_state("2/3 + 1/4", is_correct=False)
    ctx.problem_signature, ctx.attempts_on_problem = sig, n
    assert ctx.attempts_on_problem == 1 and ctx.scaffold_level == 0
    # Same problem, still wrong -> escalate.
    sig, n = ctx.compute_problem_state("2/3+1/4", is_correct=False)
    ctx.problem_signature, ctx.attempts_on_problem = sig, n
    assert ctx.attempts_on_problem == 2 and ctx.scaffold_level == 1
    # Text-only follow-up preserves the count.
    sig, n = ctx.compute_problem_state(None, is_correct=None)
    assert (sig, n) == (ctx.problem_signature, 2)
    # Getting it right resets the ladder.
    sig, n = ctx.compute_problem_state("2/3 + 1/4", is_correct=True)
    assert n == 0
    # A new problem resets to 1.
    sig, n = ctx.compute_problem_state("5*7 - 2", is_correct=False)
    assert n == 1


# --- Phase 5: proactive next-step --------------------------------------------

def test_recommend_next_picks_weakest_and_redirects_to_prereq():
    p = LearnerProfile()
    # Struggling on decimals (whose prereq is fractions); fractions unseen.
    for _ in range(3):
        p.record("decimals", is_correct=False, error_type="arithmetic")
    p.record("fractions", is_correct=False, error_type="arithmetic")  # struggling prereq
    rec = p.recommend_next()
    assert rec is not None
    # Redirects to the weak prerequisite rather than the surface concept.
    assert rec["concept"] == "fractions"
    assert rec["difficulty"] == "easier"


def test_recommend_next_none_when_nothing_weak():
    p = LearnerProfile()
    for _ in range(3):
        p.record("decimals", is_correct=True, error_type="none")  # mastered
    assert p.recommend_next() is None


def test_recommend_next_skips_other_bucket():
    # The "other" catch-all is not a teachable concept and must never be recommended.
    p = LearnerProfile()
    for _ in range(3):
        p.record("other", is_correct=False, error_type="conceptual")
    assert p.recommend_next() is None
    # ...but a real struggling concept alongside "other" is still recommended.
    # "angles" has no prerequisites, so it isn't redirected to a foundation.
    for _ in range(3):
        p.record("angles", is_correct=False, error_type="conceptual")
    assert p.recommend_next()["concept"] == "angles"


def test_practice_endpoint_creates_seeded_conversation(monkeypatch):
    from fastapi.testclient import TestClient

    from app.dependencies import get_conversation_service, get_profile_service
    from app.main import app
    from app.schemas.tutor import ConversationSummary, PostTurnResult
    from app.services.profile_service import ProfileService

    monkeypatch.setattr(llm, "generate_practice", lambda he, diff, grade=8: "תרגיל: 2/7 + 3/7 = ?")

    fake_s3 = FakeS3()
    profiles = ProfileService(fake_s3, app_settings)
    seeded = LearnerProfile()
    for _ in range(3):
        seeded.record("angles", is_correct=False, error_type="conceptual")  # no prereqs
    fake_s3.store["students/demo/profile.json"] = seeded.model_dump()

    class FakeConv:
        def __init__(self):
            self.turns = []

        async def create_conversation(self, sid, name):
            return ConversationSummary(id="practice-c1", name=name)

        async def post_turn(self, **kw):
            self.turns.append(kw)
            return PostTurnResult(
                status="success", message="ok", turn=kw["turn_number"],
                image_keys=[], response_key="r.json", ai_feedback=kw["feedback_data"],
            )

    fake_conv = FakeConv()
    app.dependency_overrides[get_profile_service] = lambda: profiles
    app.dependency_overrides[get_conversation_service] = lambda: fake_conv
    try:
        resp = TestClient(app).post("/students/demo/practice")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["conversation_id"] == "practice-c1"
        assert body["concept"] == "angles"
        assert "2/7" in body["problem"]
        # The opening turn is a tutor-only message (no images), seeding the problem.
        seeded_turn = fake_conv.turns[0]
        assert seeded_turn["turn_number"] == 0
        assert seeded_turn["images"] == []
        assert seeded_turn["feedback_data"]["reply"] == body["problem"]
        assert seeded_turn["feedback_data"]["is_correct"] is None
    finally:
        app.dependency_overrides.clear()


def test_practice_endpoint_honors_chosen_subject_and_grade(monkeypatch):
    from fastapi.testclient import TestClient

    from app.dependencies import get_conversation_service, get_profile_service
    from app.main import app
    from app.schemas.tutor import ConversationSummary, PostTurnResult
    from app.services.profile_service import ProfileService

    seen = {}

    def fake_gen(he, diff, grade=8):
        seen["he"], seen["diff"], seen["grade"] = he, diff, grade
        return "בעיה במשולשים"

    monkeypatch.setattr(llm, "generate_practice", fake_gen)

    fake_s3 = FakeS3()
    profiles = ProfileService(fake_s3, app_settings)
    fake_s3.store["students/demo/profile.json"] = LearnerProfile(grade=8).model_dump()

    class FakeConv:
        async def create_conversation(self, sid, name):
            return ConversationSummary(id="c-pick", name=name)

        async def post_turn(self, **kw):
            return PostTurnResult(status="ok", message="", turn=0, image_keys=[],
                                  response_key="r", ai_feedback=kw["feedback_data"])

    app.dependency_overrides[get_profile_service] = lambda: profiles
    app.dependency_overrides[get_conversation_service] = lambda: FakeConv()
    try:
        client = TestClient(app)
        # /concepts lists pickable subjects (never the "other" bucket).
        cats = client.get("/students/demo/concepts").json()
        ids = {c["concept"] for c in cats}
        assert "angles" in ids and "other" not in ids
        # Picking a subject overrides the recommendation; grade is passed through.
        resp = client.post("/students/demo/practice", params={"concept": "angles"})
        assert resp.status_code == 201, resp.text
        assert resp.json()["concept"] == "angles"
        assert seen["grade"] == 8  # grade-aligned generation
        # An unknown subject is rejected.
        assert client.post("/students/demo/practice", params={"concept": "bogus"}).status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_struggle_summary_stays_under_budget_with_many_concepts():
    # A pathological profile (every concept struggling) must not blow the prompt
    # budget once its Hebrew struggle text is rendered into the system prompt.
    from app.ai.taxonomy import CONCEPTS

    p = LearnerProfile()
    for cid in CONCEPTS:
        for _ in range(3):
            p.record(cid, is_correct=False, error_type="conceptual")
    system = build_tutor_system("socratic", "slow", 9, "low", p.struggle_summary())
    assert len(system) <= SYSTEM_PROMPT_BUDGET


def test_next_endpoint_returns_recommendation(monkeypatch):
    from fastapi.testclient import TestClient

    from app.dependencies import get_profile_service
    from app.main import app
    from app.services.profile_service import ProfileService

    fake_s3 = FakeS3()
    profiles = ProfileService(fake_s3, app_settings)
    seeded = LearnerProfile()
    for _ in range(3):
        seeded.record("whole_number_arithmetic", is_correct=True, error_type="none")  # prereq solid
    for _ in range(3):
        seeded.record("fractions", is_correct=False, error_type="arithmetic")  # struggling
    fake_s3.store["students/demo/profile.json"] = seeded.model_dump()

    app.dependency_overrides[get_profile_service] = lambda: profiles
    try:
        resp = TestClient(app).get("/students/demo/next")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["concept"] == "fractions"
        assert body["difficulty"] == "easier"
        assert body["practice_problem"] is None  # not requested
    finally:
        app.dependency_overrides.clear()


def test_profile_defaults():
    # A cold-start profile: fixed style/pace/confidence; grade defaults to 8
    # (middle-school) until per-student grade is wired up.
    p = LearnerProfile()
    assert (p.style, p.pace, p.grade, p.confidence) == ("step_by_step", "normal", 8, "med")


# --- Managed history: context trimming + rolling summary ----------------------

def test_context_update_trims_to_two_and_summarizes(monkeypatch):
    import asyncio

    from app.services.context_service import ContextService

    monkeypatch.setattr(llm, "summarize", lambda prev, latest: "running summary")

    class _NoHistoryConv:
        async def get_history(self, *a, **k):
            from app.exceptions import ConversationNotFoundError

            raise ConversationNotFoundError("s", "c")

    contexts = ContextService(FakeS3(), app_settings, _NoHistoryConv())
    ctx = ConversationContext()

    async def run():
        for i in range(3):
            await contexts.update(
                "s", "c", ctx, student_text=f"msg{i}", analysis=None, reply=f"reply{i}"
            )

    asyncio.run(run())

    assert ctx.turn_count == 3
    assert len(ctx.recent_exchanges) == 2  # only the last two kept verbatim
    assert ctx.recent_exchanges[-1].student == "msg2"
    assert ctx.rolling_summary == "running summary"
    # render() exposes the memory as a prompt-ready block
    rendered = ctx.render()
    assert "running summary" in rendered
    assert "msg2" in rendered


# --- Verdict branching: the graded result drives the tutor directive ----------

def _situation_text(analysis):
    from app.services.tutor_ai_service import TutorAIService

    return "\n".join(TutorAIService._situation(analysis, student_text=""))


def test_correct_answer_directive_affirms_and_asks_whats_next():
    analysis = AnalysisResult(
        problem="2/3 + 1/4",
        is_correct=True,
        error_type="none",
        concept="fractions",
        confidence=0.95,
        student_answer="11/12",
        observation="found a common denominator of 12 correctly",
    )
    text = _situation_text(analysis)
    assert "CORRECT" in text
    assert "what they'd like to do next" in text
    assert "common denominator of 12" in text  # grounded in the observation


def test_wrong_answer_directive_guides_to_fix():
    analysis = AnalysisResult(
        problem="2/3 + 1/4",
        is_correct=False,
        error_type="arithmetic",
        concept="fractions",
        confidence=0.9,
        student_answer="3/7",
        observation="added numerators and denominators directly",
    )
    text = _situation_text(analysis)
    assert "INCORRECT" in text
    assert "arithmetic" in text and "fractions" in text
    assert "without revealing the answer" in text


def test_low_confidence_directive_asks_to_confirm():
    analysis = AnalysisResult(
        problem="(unclear)",
        is_correct=False,
        error_type="none",
        concept="unknown",
        confidence=0.2,  # below LOW_CONFIDENCE
        student_answer="",
        observation="",
    )
    text = _situation_text(analysis)
    assert "confirm or re-share" in text
    assert "CORRECT" not in text  # must not pretend to have graded it
