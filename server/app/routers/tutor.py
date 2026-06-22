import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import get_conversation_service
from app.schemas.images import ImageUpload
from app.schemas.tutor import (
    ConversationHistory,
    ConversationSummary,
    CreateConversationRequest,
    PostTurnResult,
)
from app.services.conversation_service import ConversationService

router = APIRouter(
    prefix="/students/{student_id}/conversations",
    tags=["Math Tutor Conversations"],
)


@router.get("", summary="List Student Conversations", response_model=list[ConversationSummary])
async def list_conversations(
    student_id: str,
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
    student_id: str,
    body: CreateConversationRequest,
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
    student_id: str,
    conversation_id: str,
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
    student_id: str,
    conversation_id: str,
    conversation_name: str = Form(
        ..., description="The name of the conversation to update or create"
    ),
    turn_number: int = Form(
        ..., description="Zero-based or one-based turn index"
    ),
    ai_feedback_json: str = Form(
        ..., description="Stringified JSON response payload from AI"
    ),
    images: list[UploadFile] = File(
        ..., description="One or more student homework images for this turn"
    ),
    service: ConversationService = Depends(get_conversation_service),
) -> PostTurnResult:
    """
    Submits student work and AI response for a specific conversation turn.
    Idempotently updates meta.json, and writes all images and AI feedback payloads to S3.
    """
    try:
        feedback_data = json.loads(ai_feedback_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="ai_feedback_json form parameter must be a valid JSON string.",
        )

    if turn_number < 0:
        raise HTTPException(
            status_code=400,
            detail="Turn number must be greater than or equal to 0.",
        )

    if not images:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required.",
        )

    uploads: list[ImageUpload] = []
    for image in images:
        uploads.append(
            ImageUpload(
                data=await image.read(),
                filename=image.filename,
                content_type=image.content_type or "image/octet-stream",
            )
        )

    return await service.post_turn(
        student_id=student_id,
        conversation_id=conversation_id,
        conversation_name=conversation_name,
        turn_number=turn_number,
        feedback_data=feedback_data,
        images=uploads,
    )
