from typing import NamedTuple


class ImageUpload(NamedTuple):
    data: bytes
    filename: str | None
    content_type: str
