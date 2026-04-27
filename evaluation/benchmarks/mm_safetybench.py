"""
MM-SafetyBench loader (HuggingFace `PKU-Alignment/MM-SafetyBench`).

Expected on-disk layout after running evaluation/scripts/download_mm_safetybench.py:

    data/mm-safetybench/
    ├── README.md
    └── data/
        ├── HateSpeech/
        │   ├── SD.parquet         (image-encoded queries, harmful concept rendered as SD)
        │   ├── SD_TYPO.parquet    (SD + typography overlay)
        │   ├── TYPO.parquet       (text-only typography images; "OCR" in ShiftDC notation)
        │   └── Text_only.parquet  (no image; text question contains the harmful concept)
        ├── Illegal_Activitiy/   (note: HF mirror keeps the canonical typo)
        ├── Malware_Generation/
        ├── Physical_Harm/
        ├── EconomicHarm/
        ├── Fraud/
        ├── Sex/                 (== "Pornography" in ShiftDC notation)
        ├── Political_Lobbying/
        ├── Privacy_Violence/
        ├── Legal_Opinion/
        ├── Financial_Advice/
        ├── Health_Consultation/
        └── Gov_Decision/        (== "Government_Decision" in ShiftDC notation)

Each parquet has columns: `id` (int), `question` (str), `image` (bytes-or-None).
For SD / SD_TYPO / TYPO the question is the rephrased "the image shows ..." form;
for Text_only the question is the original harmful instruction with no image.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

from . import EvalSample


SCENARIO_NAMES: dict[int, str] = {
    1: "Illegal_Activity",
    2: "HateSpeech",
    3: "Malware_Generation",
    4: "Physical_Harm",
    5: "Economic_Harm",
    6: "Fraud",
    7: "Pornography",
    8: "Political_Lobbying",
    9: "Privacy_Violence",
    10: "Legal_Opinion",
    11: "Financial_Advice",
    12: "Health_Consultation",
    13: "Government_Decision",
}

# HF mirror's directory names (sometimes with a typo or short form). We accept
# multiple aliases so newer/older mirrors both work.
_SCENARIO_DIR_ALIASES: dict[int, list[str]] = {
    1:  ["Illegal_Activitiy", "Illegal_Activity"],
    2:  ["HateSpeech"],
    3:  ["Malware_Generation"],
    4:  ["Physical_Harm"],
    5:  ["EconomicHarm", "Economic_Harm"],
    6:  ["Fraud"],
    7:  ["Sex", "Pornography"],
    8:  ["Political_Lobbying"],
    9:  ["Privacy_Violence"],
    10: ["Legal_Opinion"],
    11: ["Financial_Advice"],
    12: ["Health_Consultation"],
    13: ["Gov_Decision", "Government_Decision"],
}

# image_type -> parquet filename stem
_IMAGE_TYPE_PARQUETS = {
    "SD":      "SD.parquet",
    "OCR":     "TYPO.parquet",       # ShiftDC's "OCR" == HF's "TYPO"
    "SD_TYPO": "SD_TYPO.parquet",
}


def _missing(data_dir: Path) -> str:
    return (
        f"MM-SafetyBench not found at {data_dir}/data/.\n"
        "Download with:\n"
        "    python evaluation/scripts/download_mm_safetybench.py\n"
        "(this fetches the HuggingFace mirror PKU-Alignment/MM-SafetyBench, "
        "which embeds images directly in the parquet files — no separate "
        "image folder needed)."
    )


def _resolve_scenario_dir(data_root: Path, scenario_id: int) -> Optional[Path]:
    for alias in _SCENARIO_DIR_ALIASES[scenario_id]:
        p = data_root / alias
        if p.is_dir():
            return p
    return None


def _decode_image(blob) -> Optional[Image.Image]:
    """Decode a parquet `image` cell (raw bytes or HF-style {bytes, path} dict)."""
    if blob is None:
        return None
    if isinstance(blob, dict):
        blob = blob.get("bytes")
    if not blob:
        return None
    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return None


def load_mm_safetybench(
    data_dir: str | Path = "data/mm-safetybench",
    scenarios: Optional[list[int]] = None,
    image_types: Optional[list[str]] = None,
    limit_per_scenario: Optional[int] = None,
) -> list[EvalSample]:
    data_dir = Path(data_dir)
    data_root = data_dir / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(_missing(data_dir))

    scenarios = scenarios or sorted(SCENARIO_NAMES.keys())
    image_types = image_types or ["SD", "OCR", "SD_TYPO"]
    for it in image_types:
        if it not in _IMAGE_TYPE_PARQUETS:
            raise ValueError(
                f"Unknown image type '{it}'. Allowed: {sorted(_IMAGE_TYPE_PARQUETS)}"
            )

    samples: list[EvalSample] = []
    for sid in scenarios:
        scen_dir = _resolve_scenario_dir(data_root, sid)
        if scen_dir is None:
            print(f"  [mm-safetybench] No directory for scenario {sid:02d} "
                  f"({SCENARIO_NAMES[sid]}); skipping.")
            continue
        for image_type in image_types:
            parquet_path = scen_dir / _IMAGE_TYPE_PARQUETS[image_type]
            if not parquet_path.exists():
                continue
            df = pd.read_parquet(parquet_path)
            # Required columns: id, question, image. Be defensive about names.
            cols_lower = {c.lower(): c for c in df.columns}
            id_col = cols_lower.get("id") or "id"
            q_col = cols_lower.get("question") or "question"
            img_col = cols_lower.get("image") or "image"

            n = 0
            for _, row in df.iterrows():
                question = row.get(q_col)
                if not isinstance(question, str) or not question.strip():
                    continue
                image = _decode_image(row.get(img_col))
                if image is None:
                    continue
                idx = int(row.get(id_col))
                samples.append(EvalSample(
                    id=f"mmsb_{sid:02d}_{image_type}_{idx:04d}",
                    question=question,
                    image=image,
                    benchmark="mm_safetybench",
                    scenario_id=sid,
                    scenario_name=SCENARIO_NAMES[sid],
                    image_type=image_type,
                ))
                n += 1
                if limit_per_scenario is not None and n >= limit_per_scenario:
                    break
    return samples
