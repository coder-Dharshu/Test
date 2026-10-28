import requests
import json
from pathlib import Path

def main():
    sample_files = list(Path("./temp_uploads").glob("*.*"))
    if not sample_files:
        print("No test file found in ./temp_uploads.")
        return
    file_path = sample_files[0]

    with open(file_path, "rb") as f:
        print("Sending POST request to /api/v1/ingest...")
        try:
            resp = requests.post("http://127.0.0.1:8000/api/v1/ingest", files={"file": ("test.jpeg", f.read())}, timeout=10)
            print("Status code:", resp.status_code)
            print("Response:", resp.text)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()

