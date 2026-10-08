import requests
import json
from pathlib import Path

file_path = Path(r"C:\Users\DARSHAN\OneDrive\Desktop\PROGO-OCR\PROGO-OCR\greencare-ai\temp_uploads\4065a2b2-02d7-4a5f-bc51-830d0201cf78_WhatsApp Image 2026-05-30 at 11.31.29 AM.jpeg")
if not file_path.exists():
    print("File not found.")
    exit(1)

with open(file_path, "rb") as f:
    print("Sending POST request to /api/v1/ingest...")
    try:
        resp = requests.post("http://127.0.0.1:8000/api/v1/ingest", files={"file": ("test.jpeg", f.read())}, timeout=10)
        print("Status code:", resp.status_code)
        print("Response:", resp.text)
    except Exception as e:
        print("Error:", e)
