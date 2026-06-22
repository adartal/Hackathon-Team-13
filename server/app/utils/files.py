def get_file_extension(filename: str | None, default: str = "jpg") -> str:
    if filename and "." in filename:
        return filename.rsplit(".", 1)[-1]
    return default
