from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Path, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.ai import llm
from app.dependencies import get_conversation_service, get_tutor_ai_service
from app.schemas.images import ImageUpload
from app.schemas.tutor import (
    ConversationHistory,
    ConversationSummary,
    CreateConversationRequest,
    PostTurnResult,
    SubmitConversationResponse,
)
from app.services.conversation_service import ConversationService
from app.services.tutor_ai_service import TutorAIService

# Path identifiers are interpolated into S3 keys, so reject anything that could
# escape the intended prefix (slashes, dots/"..", encoded variants all fail this).
_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
StudentId = Path(..., pattern=_ID_PATTERN, description="Student identifier")
ConversationId = Path(..., pattern=_ID_PATTERN, description="Conversation identifier")

router = APIRouter(
    prefix="/students/{student_id}/conversations",
    tags=["Math Tutor Conversations"],
)


@router.get("", summary="List Student Conversations", response_model=list[ConversationSummary])
async def list_conversations(
    student_id: str = StudentId,
    service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationSummary]:
    """
    Scans S3 prefixes under the student path to discover conversation folders.
    Reads each folder's meta.json to extract the conversation name.
    """
    return await service.list_conversations(student_id)


@router.post(
    "",
    summary="Create Conversation",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    student_id: str = StudentId,
    body: CreateConversationRequest = Body(...),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationSummary:
    """
    Creates a new conversation with a generated ID and writes meta.json to S3
    containing the conversation name.
    """
    return await service.create_conversation(student_id, body.name)


@router.get(
    "/{conversation_id}",
    summary="Get Conversation History",
    response_model=ConversationHistory,
)
async def get_conversation_history(
    student_id: str = StudentId,
    conversation_id: str = ConversationId,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationHistory:
    """
    Collects and aggregates all files inside the conversation prefix.
    Reads meta.json for the name and pairs files matching the same turn prefix chronologically.
    """
    return await service.get_history(student_id, conversation_id)


@router.post(
    "/{conversation_id}/turn",
    summary="Post Conversation Turn",
    response_model=PostTurnResult,
)
async def post_conversation_turn(
    student_id: str = StudentId,
    conversation_id: str = ConversationId,
    conversation_name: str = Form(
        ..., description="The name of the conversation to update or create"
    ),
    turn_number: int = Form(
        ..., description="Zero-based or one-based turn index"
    ),
    student_text: str = Form(
        "", description="Optional message the student typed for this turn"
    ),
    images: list[UploadFile] = File(
        default=[], description="Optional student homework images for this turn"
    ),
    service: ConversationService = Depends(get_conversation_service),
    ai: TutorAIService = Depends(get_tutor_ai_service),
) -> PostTurnResult:
    """
    Submits a student turn (homework photos and/or a text message). The backend
    runs the AI tutor to generate feedback, then writes the images and the
    generated feedback to S3 and returns the feedback for the chat to render.
    """
    if turn_number < 0:
        raise HTTPException(
            status_code=400,
            detail="Turn number must be greater than or equal to 0.",
        )

    uploads: list[ImageUpload] = []
    for image in images:
        data = await image.read()
        if not data:
            continue
        uploads.append(
            ImageUpload(
                data=data,
                filename=image.filename,
                content_type=image.content_type or "image/octet-stream",
            )
        )

    if not uploads and not student_text.strip():
        raise HTTPException(
            status_code=400,
            detail="A turn needs at least one image or a text message.",
        )

    # The "brain" loads its own memory (profile + context) and folds this turn
    # back into it; the student's typed message is recorded inside run_turn.
    first_image = uploads[0] if uploads else None
    feedback_data = await ai.run_turn(
        student_id=student_id,
        conversation_id=conversation_id,
        image=first_image.data if first_image else None,
        image_mime=first_image.content_type if first_image else "image/jpeg",
        student_text=student_text,
    )

    return await service.post_turn(
        student_id=student_id,
        conversation_id=conversation_id,
        conversation_name=conversation_name,
        turn_number=turn_number,
        feedback_data=feedback_data,
        images=uploads,
    )


def _build_dialogue(history: ConversationHistory) -> str:
    lines: list[str] = []
    for turn in history.history:
        if not turn.ai_feedback:
            continue
        if turn.turn == 0:
            reply = turn.ai_feedback.get("reply", "")
            if reply:
                lines.append(f"שאלה: {reply}")
        else:
            student_text = turn.ai_feedback.get("student_text", "")
            if student_text:
                lines.append(f"תלמיד: {student_text}")
            reply = turn.ai_feedback.get("reply", "")
            if reply:
                lines.append(f"מורה AI: {reply}")
    return "\n".join(lines) if lines else "(שיחה ריקה)"


@router.post(
    "/{conversation_id}/submit",
    summary="Submit Conversation",
    response_model=SubmitConversationResponse,
)
async def submit_conversation(
    student_id: str = StudentId,
    conversation_id: str = ConversationId,
    service: ConversationService = Depends(get_conversation_service),
) -> SubmitConversationResponse:
    """Mark a conversation as completed and post an AI-generated Hebrew review."""
    history = await service.get_history(student_id, conversation_id)

    if history.status == "completed":
        # Already submitted — return the last tutor message as the review.
        last_reply = next(
            (
                turn.ai_feedback.get("reply", "")
                for turn in reversed(history.history)
                if turn.ai_feedback and turn.ai_feedback.get("reply")
            ),
            "",
        )
        return SubmitConversationResponse(review=last_reply, status="completed")

    dialogue = _build_dialogue(history)
    review = await run_in_threadpool(
        llm.generate_conversation_review, history.conversation_name, dialogue
    )

    next_turn = len(history.history)
    await service.post_turn(
        student_id=student_id,
        conversation_id=conversation_id,
        conversation_name=history.conversation_name,
        turn_number=next_turn,
        feedback_data={
            "reply": review,
            "is_correct": None,
            "concept": None,
            "error_type": None,
        },
        images=[],
    )

    await service.mark_completed(student_id, conversation_id)

    return SubmitConversationResponse(review=review, status="completed")
