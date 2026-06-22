import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from fastapi.concurrency import run_in_threadpool

from app.ai import llm
from app.dependencies import get_conversation_service, get_teacher_service
from app.schemas.auth import StudentEntry  # noqa: F401 — re-exported via response_model
from app.schemas.tutor import (
    AssignQuestionResponse,
    AssignRequest,
    BulkAssignRequest,
    BulkAssignResponse,
    BulkAssignResult,
    GenerateQuestionRequest,
    GenerateQuestionResponse,
)
from app.services.conversation_service import ConversationService
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
