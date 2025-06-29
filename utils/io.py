import tempfile
import os
from fastapi import UploadFile

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    tmp_path = f"/tmp/{file.filename}"
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)
    return tmp_path

def cleanup_temp_file(file_path: str):
    """Remove temporary file"""
    if os.path.exists(file_path):
        os.remove(file_path)
