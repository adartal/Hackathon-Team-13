import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from fastapi.concurrency import run_in_threadpool

from app.ai import llm, taxonomy
from app.dependencies import get_conversation_service, get_profile_service, get_teacher_service
from app.schemas.auth import StudentEntry  # noqa: F401 — re-exported via response_model
from app.schemas.tutor import (
    AssignQuestionResponse,
    AssignRequest,
    BulkAssignRequest,
    BulkAssignResponse,
    BulkAssignResult,
    GenerateQuestionRequest,
    GenerateQuestionResponse,
    StudentOverviewResponse,
    StudentOverviewStats,
    StudentSummaryResponse,
)
from app.services.conversation_service import ConversationService
from app.services.profile_service import ProfileService
from app.services.teacher_service import TeacherService

_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
TeacherId = Path(..., pattern=_ID_PATTERN, description="Teacher identifier")
StudentId = Path(..., pattern=_ID_PATTERN, description="Student user_id")

router = APIRouter(prefix="/teachers", tags=["Teachers"])


@router.get("/{teacher_id}/students", response_model=list[StudentEntry])
async def get_students(
    teacher_id: str = TeacherId,
    service: TeacherService = Depends(get_teacher_service),
) -> list[StudentEntry]:
    return await service.get_students(teacher_id)


@router.post("/{teacher_id}/students", response_model=list[StudentEntry])
async def add_student(
    teacher_id: str = TeacherId,
    username: str = Body(..., embed=True),
    service: TeacherService = Depends(get_teacher_service),
) -> list[StudentEntry]:
    try:
        return await service.add_student(teacher_id, username)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{teacher_id}/students/{student_id}", response_model=list[StudentEntry])
async def remove_student(
    teacher_id: str = TeacherId,
    student_id: str = StudentId,
    service: TeacherService = Depends(get_teacher_service),
) -> list[StudentEntry]:
    return await service.remove_student(teacher_id, student_id)


@router.get(
    "/{teacher_id}/students/{student_id}/overview",
    response_model=StudentOverviewResponse,
)
async def get_student_overview(
    teacher_id: str = TeacherId,
    student_id: str = StudentId,
    teacher_service: TeacherService = Depends(get_teacher_service),
    conversations: ConversationService = Depends(get_conversation_service),
    profiles: ProfileService = Depends(get_profile_service),
) -> StudentOverviewResponse:
    """Return a teacher-facing overview of one student: full conversation list + engagement stats."""
    roster = await teacher_service.get_students(teacher_id)
    entry = next((s for s in roster if s.user_id == student_id), None)

    convos = await conversations.list_conversations(student_id)
    profile = await profiles.load(student_id)

    assigned = [c for c in convos if c.assigned_by]
    practice = [c for c in convos if not c.assigned_by]
    done = [c for c in convos if c.status == "completed"]

    return StudentOverviewResponse(
        student_id=student_id,
        username=entry.username if entry else None,
        conversations=convos,
        stats=StudentOverviewStats(
            total_conversations=len(convos),
            assigned_count=len(assigned),
            practice_count=len(practice),
            done_count=len(done),
            total_turns=profile.total_turns,
        ),
    )


@router.get(
    "/{teacher_id}/students/{student_id}/ai-summary",
    response_model=StudentSummaryResponse,
)
async def get_student_ai_summary(
    teacher_id: str = TeacherId,
    student_id: str = StudentId,
    conversations: ConversationService = Depends(get_conversation_service),
    profiles: ProfileService = Depends(get_profile_service),
) -> StudentSummaryResponse:
    """Generate an AI summary focused on completed session reviews."""
    convos = await conversations.list_conversations(student_id)
    profile = await profiles.load(student_id)

    completed = [c for c in convos if c.status == "completed"]

    async def _get_review(c) -> tuple[str, str]:
        try:
            history = await conversations.get_history(student_id, c.id)
            last_reply = next(
                (
                    turn.ai_feedback.get("reply", "")
                    for turn in reversed(history.history)
                    if turn.ai_feedback and turn.ai_feedback.get("reply")
                ),
                "",
            )
            return (c.name, last_reply)
        except Exception:
            return (c.name, "")

    session_reviews = list(await asyncio.gather(*[_get_review(c) for c in completed]))

    concept_mastery = {
        concept: {
            "mastery_level": mastery.mastery_level,
            "attempts": mastery.attempts,
            "correct": mastery.correct,
            "he_name": taxonomy.display_name(concept),
        }
        for concept, mastery in profile.concepts.items()
        if mastery.attempts > 0
    }

    summary = await run_in_threadpool(
        llm.generate_student_summary,
        session_reviews,
        concept_mastery,
        profile.total_turns,
    )
    return StudentSummaryResponse(summary=summary)


@router.post(
    "/{teacher_id}/generate-question",
    response_model=GenerateQuestionResponse,
)
async def generate_question(
    teacher_id: str = TeacherId,
    body: GenerateQuestionRequest = Body(...),
) -> GenerateQuestionResponse:
    """Generate a Hebrew math question from a free-form prompt without persisting anything."""
    problem = await run_in_threadpool(llm.generate_from_prompt, body.prompt)
    return GenerateQuestionResponse(problem=problem)


@router.post(
    "/{teacher_id}/students/{student_id}/assign",
    response_model=AssignQuestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def assign_question(
    teacher_id: str = TeacherId,
    student_id: str = StudentId,
    body: AssignRequest = Body(...),
    conversations: ConversationService = Depends(get_conversation_service),
) -> AssignQuestionResponse:
    """Seed a pre-generated question as a new conversation for the student."""
    convo = await conversations.create_conversation(
        student_id, body.name, assigned_by=teacher_id
    )
    opening = {"reply": body.problem, "is_correct": None, "concept": None, "error_type": None}
    await conversations.post_turn(
        student_id=student_id,
        conversation_id=convo.id,
        conversation_name=convo.name,
        turn_number=0,
        feedback_data=opening,
        images=[],
    )
    return AssignQuestionResponse(conversation_id=convo.id, problem=body.problem)


@router.post(
    "/{teacher_id}/assign-bulk",
    response_model=BulkAssignResponse,
    status_code=status.HTTP_201_CREATED,
)
async def assign_bulk(
    teacher_id: str = TeacherId,
    body: BulkAssignRequest = Body(...),
    conversations: ConversationService = Depends(get_conversation_service),
) -> BulkAssignResponse:
    """Assign a pre-generated question to multiple students at once."""
    if not body.student_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No students selected")

    async def _seed(student_id: str) -> BulkAssignResult:
        convo = await conversations.create_conversation(
            student_id, body.name, assigned_by=teacher_id
        )
        opening = {"reply": body.problem, "is_correct": None, "concept": None, "error_type": None}
        await conversations.post_turn(
            student_id=student_id,
            conversation_id=convo.id,
            conversation_name=convo.name,
            turn_number=0,
            feedback_data=opening,
            images=[],
        )
        return BulkAssignResult(student_id=student_id, conversation_id=convo.id)

    results = await asyncio.gather(*[_seed(sid) for sid in body.student_ids])
    return BulkAssignResponse(problem=body.problem, results=list(results))
