"""Shared helpers for HTTP and Google Drive downloads."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path


def download_url(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"


def download_gdrive(file_id: str, dest: Path, *, label: str = "archive") -> None:
    """Download a large Google Drive file (handles virus-scan confirm page)."""
    import requests

    session = requests.Session()
    landing = session.get(
        "https://drive.google.com/uc?export=download",
        params={"id": file_id},
    )
    landing.raise_for_status()

    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', landing.text)
    if uuid_match:
        params = {
            "id": file_id,
            "export": "download",
            "confirm": "t",
            "uuid": uuid_match.group(1),
        }
        response = session.get(
            "https://drive.usercontent.google.com/download",
            params=params,
            stream=True,
        )
    else:
        response = landing
        for key, value in landing.cookies.items():
            if key.startswith("download_warning"):
                response = session.get(
                    "https://drive.google.com/uc?export=download",
                    params={"id": file_id, "confirm": value},
                    stream=True,
                )
                break

    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise RuntimeError(
            f"Google Drive returned an HTML page instead of the {label}. "
            f"Try again later or download manually: "
            f"https://drive.google.com/file/d/{file_id}/view"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    if not is_valid_zip(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a valid zip ({label}). "
            f"Remove {dest} and retry, or download manually: "
            f"https://drive.google.com/file/d/{file_id}/view"
        )
