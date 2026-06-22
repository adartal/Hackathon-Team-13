"""Student-level endpoints that live OUTSIDE the conversations prefix.

The tutor router is mounted at ``/students/{student_id}/conversations``; the
proactive "what should I practice next" recommendation is a property of the
student, not of any one conversation, so it gets its own router here.
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.concurrency import run_in_threadpool

from app.ai import llm, taxonomy
from app.dependencies import get_conversation_service, get_profile_service
from app.schemas.tutor import NextStepResponse, PracticeStartResponse
from app.services.conversation_service import ConversationService
from app.services.profile_service import ProfileService

# Same guard the tutor router uses — ids are interpolated into S3 keys.
StudentId = Path(..., pattern=r"^[A-Za-z0-9_-]{1,128}$", description="Student identifier")

router = APIRouter(prefix="/students/{student_id}", tags=["Student"])


@router.get(
    "/next",
    summary="Recommend Next Practice",
    response_model=NextStepResponse,
)
async def recommend_next(
    student_id: str = StudentId,
    generate: bool = Query(
        False, description="Also generate a fresh practice problem for the concept"
    ),
    profiles: ProfileService = Depends(get_profile_service),
) -> NextStepResponse:
    """Pick the most valuable concept for this student to practice next.

    Deterministic selection from the learner profile (spaced-repetition-due or
    weakest non-mastered, honoring prerequisites). Returns an empty body when
    nothing needs attention. ``?generate=true`` adds one LLM-authored problem.
    """
    profile = await profiles.load(student_id)
    rec = profile.recommend_next()
    if rec is None:
        return NextStepResponse()

    practice = None
    if generate:
        try:
            practice = await run_in_threadpool(
                llm.generate_practice, rec["he_name"], rec["difficulty"], profile.grade
            )
        except Exception:  # pragma: no cover - recommendation still useful without it
            practice = None

    return NextStepResponse(
        concept=rec["concept"],
        he_name=rec["he_name"],
        difficulty=rec["difficulty"],
        practice_problem=practice,
    )


@router.get("/concepts", summary="List Pickable Practice Subjects")
async def list_concepts(
    student_id: str = StudentId,
    profiles: ProfileService = Depends(get_profile_service),
) -> list[dict[str, object]]:
    """Subjects the student can choose to practice (grade-appropriate listed first)."""
    profile = await profiles.load(student_id)
    return taxonomy.catalog(profile.grade)


@router.post(
    "/practice",
    summary="Start a Practice Conversation",
    response_model=PracticeStartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_practice(
    student_id: str = StudentId,
    concept: str | None = Query(
        None, description="Subject to practice; omit to use the recommended one"
    ),
    profiles: ProfileService = Depends(get_profile_service),
    conversations: ConversationService = Depends(get_conversation_service),
) -> PracticeStartResponse:
    """Open a practice conversation on a concept (chosen or recommended).

    The student may pick any subject; when none is given we fall back to the
    recommendation. Either way we generate a grade-aligned problem and seed it as
    the opening tutor message (turn 0, no homework), so the frontend just navigates
    into the chat.
    """
    profile = await profiles.load(student_id)

    if concept:
        if concept not in taxonomy.CONCEPTS or concept == "other":
            raise HTTPException(status_code=400, detail=f"Unknown subject '{concept}'.")
        cid, he_name = concept, taxonomy.display_name(concept)
        difficulty = profile.difficulty_for(concept)
    else:
        rec = profile.recommend_next()
        if rec is None:
            raise HTTPException(status_code=404, detail="Nothing to practice right now.")
        cid, he_name, difficulty = rec["concept"], rec["he_name"], rec["difficulty"]

    problem = await run_in_threadpool(
        llm.generate_practice, he_name, difficulty, profile.grade
    )
    convo = await conversations.create_conversation(student_id, f"תרגול: {he_name}")
    # Seed turn 0 as a tutor-only message (no images) presenting the problem.
    opening = {
        "reply": problem,
        "is_correct": None,
        "concept": cid,
        "error_type": None,
    }
    await conversations.post_turn(
        student_id=student_id,
        conversation_id=convo.id,
        conversation_name=convo.name,
        turn_number=0,
        feedback_data=opening,
        images=[],
    )
    return PracticeStartResponse(
        conversation_id=convo.id,
        concept=cid,
        he_name=he_name,
        difficulty=difficulty,
        problem=problem,
    )
