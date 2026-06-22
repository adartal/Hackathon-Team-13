import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class S3Exception(Exception):
    """Base exception for all S3/MinIO operations."""
    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class S3ConnectionError(S3Exception):
    """Raised when the S3 service connection fails."""
    def __init__(self, message: str = "Could not establish connection to S3/MinIO service") -> None:
        super().__init__(message, status_code=503)

class BucketNotFoundError(S3Exception):
    """Raised when a requested bucket does not exist."""
    def __init__(self, bucket_name: str) -> None:
        super().__init__(f"Bucket '{bucket_name}' not found", status_code=404)

class BucketAlreadyExistsError(S3Exception):
    """Raised when attempting to create a bucket that already exists."""
    def __init__(self, bucket_name: str) -> None:
        super().__init__(f"Bucket '{bucket_name}' already exists", status_code=400)

class FileKeyNotFoundError(S3Exception):
    """Raised when a specific file/key is missing in a bucket."""
    def __init__(self, bucket_name: str, file_key: str) -> None:
        super().__init__(f"File '{file_key}' not found in bucket '{bucket_name}'", status_code=404)

class FileOperationError(S3Exception):
    """Raised when a file upload or download operation fails."""
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)

class ConversationNotFoundError(S3Exception):
    """Raised when a conversation folder does not exist in S3."""
    def __init__(self, student_id: str, conversation_id: str) -> None:
        super().__init__(
            f"Conversation '{conversation_id}' for student '{student_id}' not found.",
            status_code=404,
        )

async def s3_exception_handler(request: Request, exc: S3Exception) -> JSONResponse:
    """Captured S3Exceptions are formatted and returned as JSON responses."""
    logger.error(f"S3 Error occurred during request {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "error_type": exc.__class__.__name__
        }
    )

def register_exception_handlers(app: FastAPI) -> None:
    """Registers exception handlers to capture S3Exceptions and format them as JSON responses."""
    app.add_exception_handler(S3Exception, s3_exception_handler)
