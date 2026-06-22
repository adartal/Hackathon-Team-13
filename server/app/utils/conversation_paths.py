def student_conversations_prefix(student_id: str) -> str:
    return f"students/{student_id}/conversations/"


def conversation_prefix(student_id: str, conversation_id: str) -> str:
    return f"{student_conversations_prefix(student_id)}{conversation_id}/"


def meta_key(student_id: str, conversation_id: str) -> str:
    return f"{conversation_prefix(student_id, conversation_id)}meta.json"


def turn_homework_key(
    student_id: str, conversation_id: str, turn: int, index: int, ext: str
) -> str:
    return (
        f"{conversation_prefix(student_id, conversation_id)}"
        f"turn_{turn:02d}_homework_{index:02d}.{ext}"
    )


def turn_response_key(student_id: str, conversation_id: str, turn: int) -> str:
    return f"{conversation_prefix(student_id, conversation_id)}turn_{turn:02d}_response.json"
