"""Google Drive helpers — list and download recipe files."""

import io
import json
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _get_service():
    """Build a Drive v3 service from env-var credentials."""
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )
    else:
        path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
        creds = service_account.Credentials.from_service_account_file(
            path, scopes=SCOPES
        )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_image_files(folder_id: str) -> list[dict]:
    """Return all image files in *folder_id* as [{id, name, mimeType}]."""
    svc = _get_service()
    q = (
        f"'{folder_id}' in parents "
        f"and trashed = false "
        f"and mimeType contains 'image/'"
    )
    res = (
        svc.files()
        .list(q=q, fields="files(id, name, mimeType)", pageSize=200)
        .execute()
    )
    return res.get("files", [])


def download_bytes(file_id: str) -> bytes:
    """Download a Drive file and return its raw bytes."""
    svc = _get_service()
    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()
