"""Image path helpers: local COCO train2014 + optional per-file download."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from . import config
from src.paths import coco_train2014_dir

COCO_TRAIN_URL = "http://images.cocodataset.org/train2014/{filename}"


def image_filename(image_id: str | int) -> str:
    return f"COCO_train2014_{int(image_id):012d}.jpg"


def image_path(image_id: str | int, root: Path | None = None) -> Path:
    root = root or coco_train2014_dir()
    return root / image_filename(image_id)


def ensure_image(
    image_id: str | int,
    *,
    root: Path | None = None,
    allow_download: bool | None = None,
) -> Path:
    """Return local path, downloading a single file if missing and allowed."""
    root = root or coco_train2014_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = image_path(image_id, root)
    if path.is_file():
        return path
    if allow_download is None:
        allow_download = config.ALLOW_IMAGE_DOWNLOAD
    if not allow_download:
        raise FileNotFoundError(
            f"Missing image {path}. Set ALLOW_IMAGE_DOWNLOAD or fetch train2014."
        )
    url = COCO_TRAIN_URL.format(filename=path.name)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path
