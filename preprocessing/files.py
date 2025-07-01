# ==========================================
# 💡 Support Assistant Project .gitignore
# For FastAPI + PostgreSQL + React + Docker
# Author: Shivam Srivastav
# ==========================================
# File: backend/api/files.py    
# ==========================================
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os

from file_parser import parse_file, clean_json
# from utils.email_notify import send_upload_notification

router = APIRouter()

UPLOAD_DIR = Path("uploads") 
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".json", ".xlsx" , ".txt", ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp"}
MAX_FILE_SIZE_MB = 10

@router.post("/upload", summary="Upload training/WhatsApp file")
async def upload_file(file: UploadFile = File(...)):
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"❌ Unsupported file type: {ext}")

    # Validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"❌ File too large (max {MAX_FILE_SIZE_MB}MB)")

    # Save file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(contents)

    # Optional: parse and preview first rows
    try:
        preview = parse_file(file_path, limit=5)
        preview = clean_json(preview)  # Clean JSON data if needed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to parse file: {str(e)}")

    # Optional: send email/slack alert to admin
    # send_upload_notification("vastavshivam@gmai.com", "uploaded successfully.", body="File uploaded ...")

    return JSONResponse(content={
        "message": f"✅ File {file.filename} uploaded successfully.",
        "filename": file.filename,
        "preview": preview  # Optional: parsed preview from CSV/JSON/XLSX
    })

# async def upload_file(file: UploadFile = File(...)):
#     # Validate extension
#     ext = Path(file.filename).suffix.lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(status_code=400, detail=f"❌ Unsupported file type: {ext}")

#     # Validate size
#     contents = await file.read()
#     size_mb = len(contents) / (1024 * 1024)
#     if size_mb > MAX_FILE_SIZE_MB:
#         raise HTTPException(status_code=400, detail=f"❌ File too large (max {MAX_FILE_SIZE_MB}MB)")

#     # Save file
#     file_path = UPLOAD_DIR / file.filename
#     with open(file_path, "wb") as f:
#         f.write(contents)

#     # Optional: parse and preview first rows
#     try:
#         preview = parse_file(file_path, limit=5)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Failed to parse file: {str(e)}")

#     # Optional: send email/slack alert to admin
#     send_upload_notification('vastavshivam@gmai.com', "uploaded successfully.", body="File uploaded ...")

#     return JSONResponse(content={
#     "message": f"✅ File {file.filename} uploaded successfully.",
#     "filename": file.filename,
#     "preview": preview  # Optional: parsed preview from CSV/JSON/XLSX
# })
