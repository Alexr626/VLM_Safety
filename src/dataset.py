"""
HoliSafe-Bench dataset loading and subset filtering.

HoliSafe-Bench has 5 combination categories:
  UiUt  - unsafe image + unsafe text  -> unsafe output
  UiSt  - unsafe image + safe text    -> (various)
  SiUt  - safe image  + unsafe text   -> (various)
  SiSt→S - safe image + safe text     -> safe output   [our SSS subset]
  SiSt→U - safe image + safe text     -> unsafe output [our SSU subset]

We filter for SSS and SSU and expose them with a uniform sample dict:
  {
    "id":             int,
    "image_path":     str or None  (local path if downloaded),
    "image_pil":      PIL.Image or None,
    "text":           str,
    "label":          "SSS" or "SSU",
    "label_idx":      0 (SSS) or 1 (SSU),
    "subset_type":    str  (raw value from dataset),
    "category":       str,
  }
"""

import json
import os
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple

from PIL import Image

# ── Canonical subset names in HoliSafe-Bench ────────────────────────────────

# All known spellings of the two subsets we care about
_SSS_LABELS = {
    "SiSt->S", "SiSt→S", "SiSt-S", "SiSt_S",
    "safe_safe_safe", "sist_s", "SSS",
    "SiSt->Safe", "SiSt→Safe",
}
_SSU_LABELS = {
    "SiSt->U", "SiSt→U", "SiSt-U", "SiSt_U",
    "safe_safe_unsafe", "sist_u", "SSU",
    "SiSt->Unsafe", "SiSt→Unsafe",
}

# Broader: any SiSt* that ends in S / U (case-insensitive fallback)
def _classify_entry(raw_type: str) -> Optional[str]:
    """Return 'SSS', 'SSU', or None for other subsets."""
    t = raw_type.strip()
    if t in _SSS_LABELS:
        return "SSS"
    if t in _SSU_LABELS:
        return "SSU"
    # fuzzy: must contain SiSt (or variant) + terminal S/U
    tl = t.lower().replace("→", "->").replace("_", "->")
    if "sist" in tl or "si_st" in tl:
        if tl.endswith("->s") or tl.endswith("safe"):
            return "SSS"
        if tl.endswith("->u") or tl.endswith("unsafe"):
            return "SSU"
    return None


# ── Schema detection helpers ─────────────────────────────────────────────────

_TYPE_FIELD_CANDIDATES = [
    "type", "category_type", "subset", "combination_type",
    "harm_type", "combination", "label_combo",
]
_TEXT_FIELD_CANDIDATES = [
    "query", "question", "instruction", "text", "prompt", "caption",
]
_IMAGE_FIELD_CANDIDATES = [
    "image", "image_path", "img", "img_path", "image_file",
]
_CATEGORY_FIELD_CANDIDATES = [
    "category", "harm_category", "safety_category", "topic",
]


def _detect_field(entry: dict, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in entry:
            return c
    return None


def inspect_schema(data) -> dict:
    """
    Print and return schema info for the dataset.
    Call this with --inspect to understand the dataset before running.
    """
    entries = data if isinstance(data, list) else list(data.values())
    print(f"\n{'='*60}")
    print(f"  HoliSafe-Bench Schema Inspection")
    print(f"{'='*60}")
    print(f"  Total entries : {len(entries)}")

    if not entries:
        print("  No entries found!")
        return {}

    sample = entries[0]
    print(f"  Keys          : {sorted(sample.keys())}")
    print(f"\n  Sample entry:")
    for k, v in sample.items():
        vstr = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
        print(f"    {k}: {vstr}")

    # Find and count subset types
    type_field = _detect_field(sample, _TYPE_FIELD_CANDIDATES)
    if type_field:
        type_values = [e.get(type_field) for e in entries]
        counts = Counter(type_values)
        print(f"\n  Type field    : '{type_field}'")
        print(f"  Unique values : {dict(counts)}")
    else:
        print(f"\n  WARNING: Could not detect a type/subset field.")
        print(f"  Will try label-based filtering (image_label, text_label, combined_label).")

    print(f"{'='*60}\n")

    return {
        "type_field": type_field,
        "text_field": _detect_field(sample, _TEXT_FIELD_CANDIDATES),
        "image_field": _detect_field(sample, _IMAGE_FIELD_CANDIDATES),
        "category_field": _detect_field(sample, _CATEGORY_FIELD_CANDIDATES),
    }


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_holisafe(
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[list, str]:
    """
    Load HoliSafe-Bench, saving it into this repository under ``data/holisafe-bench/``
    so the dataset is easy to inspect locally.

    Download only happens once.  On subsequent calls the local copy is reused.

    Args:
        local_dir: Where to store the dataset.  Defaults to
                   ``<repo_root>/data/holisafe-bench/``.
        cache_dir: Optional HuggingFace download cache directory.

    Returns:
        (entries, images_base_dir)
        entries:        list of raw dicts (one per sample).
        images_base_dir: local directory that image paths are relative to.
    """
    REPO = "etri-vilab/holisafe-bench"

    # Default to <repo_root>/data/holisafe-bench/
    if local_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        local_dir = str(repo_root / "data" / "holisafe-bench")

    json_path = os.path.join(local_dir, "holisafe_bench.json")

    # ── If already downloaded locally, just load it ───────────────────────
    if os.path.exists(json_path):
        print(f"Loading HoliSafe-Bench from local copy: {local_dir}")
        with open(json_path, "r") as f:
            raw = json.load(f)
        entries = raw if isinstance(raw, list) else list(raw.values())
        print(f"  Loaded {len(entries)} entries.")
        return entries, local_dir

    # ── Download the full dataset into local_dir ──────────────────────────
    from huggingface_hub import snapshot_download
    print(f"Downloading '{REPO}' into {local_dir}/ ...")
    print(f"  (this may take a few minutes on the first run)")
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=local_dir,
        cache_dir=cache_dir,
    )

    # Verify the JSON landed where expected
    if not os.path.exists(json_path):
        # Some repos nest files; search for it
        for root, _, files in os.walk(local_dir):
            if "holisafe_bench.json" in files:
                json_path = os.path.join(root, "holisafe_bench.json")
                break

    with open(json_path, "r") as f:
        raw = json.load(f)

    entries = raw if isinstance(raw, list) else list(raw.values())
    print(f"  Loaded {len(entries)} entries.  Dataset saved to: {local_dir}")
    return entries, local_dir


def _build_sample(
    idx: int,
    entry: dict,
    label: str,
    images_base: str,
    type_field: Optional[str],
    text_field: Optional[str],
    image_field: Optional[str],
    category_field: Optional[str],
) -> dict:
    """Build a normalised sample dict from a raw entry."""

    # ── Text ─────────────────────────────────────────────────────────────
    if text_field:
        text = str(entry.get(text_field, ""))
    else:
        # search common names
        text = ""
        for k in _TEXT_FIELD_CANDIDATES:
            if k in entry and isinstance(entry[k], str):
                text = entry[k]
                break

    # ── Image ────────────────────────────────────────────────────────────
    image_pil = None
    image_path = None

    def _resolve_img_path(raw_img_str: str) -> str:
        """Resolve a relative image path against images_base.
        HoliSafe-Bench stores paths relative to an images/ subdirectory,
        so we try  images_base/images/<path>  first, then  images_base/<path>."""
        if os.path.isabs(raw_img_str):
            return raw_img_str
        if images_base:
            # Try with images/ subdirectory first (HoliSafe-Bench layout)
            candidate = os.path.join(images_base, "images", raw_img_str)
            if os.path.exists(candidate):
                return candidate
            # Fall back to direct join
            return os.path.join(images_base, raw_img_str)
        return raw_img_str

    if image_field and image_field in entry:
        raw_img = entry[image_field]
        if isinstance(raw_img, Image.Image):
            image_pil = raw_img.convert("RGB")
        elif isinstance(raw_img, str):
            image_path = _resolve_img_path(raw_img)
        elif isinstance(raw_img, bytes):
            from io import BytesIO
            image_pil = Image.open(BytesIO(raw_img)).convert("RGB")
    else:
        # Try all candidate image fields
        for k in _IMAGE_FIELD_CANDIDATES:
            if k in entry:
                raw_img = entry[k]
                if isinstance(raw_img, Image.Image):
                    image_pil = raw_img.convert("RGB")
                    break
                elif isinstance(raw_img, str):
                    image_path = _resolve_img_path(raw_img)
                    break

    # ── Category ─────────────────────────────────────────────────────────
    category = "unknown"
    if category_field:
        category = str(entry.get(category_field, "unknown"))
    else:
        for k in _CATEGORY_FIELD_CANDIDATES:
            if k in entry:
                category = str(entry[k])
                break

    return {
        "id": idx,
        "image_path": image_path,
        "image_pil": image_pil,   # pre-loaded PIL if available
        "text": text,
        "label": label,
        "label_idx": 0 if label == "SSS" else 1,
        "subset_type": str(entry.get(type_field, "")) if type_field else "",
        "category": category,
        "raw": entry,
    }


def filter_subsets(
    entries: list,
    images_base: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Filter dataset into SSS and SSU subsets.

    Returns:
        (sss_samples, ssu_samples)
    """
    if not entries:
        raise ValueError("Dataset is empty.")

    sample0 = entries[0]
    type_field = _detect_field(sample0, _TYPE_FIELD_CANDIDATES)
    text_field = _detect_field(sample0, _TEXT_FIELD_CANDIDATES)
    image_field = _detect_field(sample0, _IMAGE_FIELD_CANDIDATES)
    category_field = _detect_field(sample0, _CATEGORY_FIELD_CANDIDATES)

    sss, ssu = [], []

    if type_field is not None:
        for idx, entry in enumerate(entries):
            raw_type = str(entry.get(type_field, "")).strip()
            lbl = _classify_entry(raw_type)
            if lbl is None:
                continue
            s = _build_sample(idx, entry, lbl, images_base,
                               type_field, text_field, image_field, category_field)
            (sss if lbl == "SSS" else ssu).append(s)
    else:
        # Fallback: use per-modality safety labels
        print("  Using per-label fields to filter (image_label, text_label, combined_label).")
        for idx, entry in enumerate(entries):
            il = str(entry.get("image_label", entry.get("image_safety", ""))).lower()
            tl = str(entry.get("text_label",  entry.get("text_safety",  ""))).lower()
            cl = str(entry.get("combined_label", entry.get("output_label", entry.get("safety", "")))).lower()

            if "safe" in il and "safe" in tl:
                if "unsafe" in cl or "harmful" in cl:
                    lbl = "SSU"
                elif "safe" in cl:
                    lbl = "SSS"
                else:
                    continue
                s = _build_sample(idx, entry, lbl, images_base,
                                   None, text_field, image_field, category_field)
                (sss if lbl == "SSS" else ssu).append(s)

    print(f"  SSS (safe→safe)  : {len(sss)} samples")
    print(f"  SSU (safe→unsafe): {len(ssu)} samples")
    return sss, ssu


def filter_reference_subsets(entries: list, images_base: str) -> dict:
    """
    Return samples from ALL five subsets keyed by subset name.
    Used by Method 2 to get clearly-unsafe text examples (SiUt, UiUt).
    """
    sample0 = entries[0]
    type_field = _detect_field(sample0, _TYPE_FIELD_CANDIDATES)
    text_field = _detect_field(sample0, _TEXT_FIELD_CANDIDATES)
    image_field = _detect_field(sample0, _IMAGE_FIELD_CANDIDATES)
    category_field = _detect_field(sample0, _CATEGORY_FIELD_CANDIDATES)

    buckets: dict = {}

    if type_field is None:
        return buckets

    for idx, entry in enumerate(entries):
        raw_type = str(entry.get(type_field, "")).strip()
        s = _build_sample(idx, entry, "OTHER", images_base,
                           type_field, text_field, image_field, category_field)
        s["subset_type"] = raw_type
        buckets.setdefault(raw_type, []).append(s)

    return buckets


# ── Image loading helper ──────────────────────────────────────────────────────

def load_image_for_sample(sample: dict, hf_repo: str = "etri-vilab/holisafe-bench") -> Optional[Image.Image]:
    """
    Return a PIL RGB image for a sample, downloading from HF if needed.
    Returns None on failure.
    """
    # Already loaded (from datasets library)
    if sample.get("image_pil") is not None:
        return sample["image_pil"]

    path = sample.get("image_path")
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  Warning: could not open image {path}: {e}")
            return None

    # Try HF hub download as fallback
    if path:
        # Build the repo-relative filename (must include images/ prefix)
        if "images/" in path:
            filename = "images/" + path.split("images/")[-1]
        else:
            filename = "images/" + os.path.basename(path)
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                repo_type="dataset",
            )
            return Image.open(local).convert("RGB")
        except Exception as e:
            print(f"  Warning: could not download image '{filename}': {e}")

    return None

# ── Reference dataset loaders (for ShiftDC Method 2) ─────────────────────────

_MMSB_SCENARIOS_01_07_09 = [
    "Illegal_Activitiy",   # note: typo in HF config name (double 'i')
    "HateSpeech",
    "Malware_Generation",
    "Physical_Harm",
    "EconomicHarm",
    "Fraud",
    "Sex",
    "Privacy_Violence",
]


def load_mmsafetybench_reference(
    n_samples: int = 160,
    scenarios: Optional[List[str]] = None,
    split: str = "SD",
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Load and sample from MM-SafetyBench for **unsafe** reference data.

    Following ShiftDC Appendix A.3: scenarios 01-07 & 09, SD split, ~160 samples.
    The text queries are NOT malicious on their own — harmful content is in the
    images.  Captioning the images is required to create meaningful text-only
    counterparts.

    Returns list of sample dicts compatible with ``generate_captions()`` and
    ``load_image_for_sample()``.
    """
    import random

    if scenarios is None:
        scenarios = list(_MMSB_SCENARIOS_01_07_09)

    repo_root = Path(__file__).resolve().parent.parent
    local_cache = Path(cache_dir) if cache_dir else repo_root / "data" / "mm-safetybench-ref"
    meta_path = local_cache / f"samples_n{n_samples}_seed{seed}.json"
    images_dir = local_cache / "images"

    # ── Cached? ───────────────────────────────────────────────────────────
    if meta_path.exists():
        print(f"  Loading cached MM-SafetyBench reference ({meta_path})")
        with open(meta_path) as f:
            samples_meta = json.load(f)
        samples = []
        for m in samples_meta:
            img_path = str(images_dir / m["image_filename"])
            samples.append({
                "id": m["id"],
                "text": m["text"],
                "image_pil": None,
                "image_path": img_path,
                "source": "mm-safetybench",
                "scenario": m["scenario"],
            })
        print(f"  {len(samples)} cached samples loaded.")
        return samples

    # ── Download from HuggingFace ─────────────────────────────────────────
    from datasets import load_dataset as _load_dataset

    all_rows = []
    for scenario in scenarios:
        print(f"    Loading MM-SafetyBench: {scenario} [{split}]...")
        try:
            ds = _load_dataset(
                "PKU-Alignment/MM-SafetyBench",
                name=scenario,
                split=split,
            )
            for row in ds:
                all_rows.append({
                    "id_raw": row.get("id", ""),
                    "question": row.get("question", ""),
                    "image": row.get("image"),          # PIL.Image
                    "scenario": scenario,
                })
        except Exception as e:
            print(f"    Warning: could not load scenario '{scenario}': {e}")

    print(f"    Total rows across {len(scenarios)} scenarios: {len(all_rows)}")

    rng = random.Random(seed)
    if len(all_rows) > n_samples:
        all_rows = rng.sample(all_rows, n_samples)

    # ── Save images + metadata to cache ───────────────────────────────────
    images_dir.mkdir(parents=True, exist_ok=True)
    samples: List[dict] = []
    samples_meta: list = []

    for row in all_rows:
        sid = f"mmsb_{row['scenario']}_{row['id_raw']}"
        img_filename = f"{sid}.png"
        img_path = images_dir / img_filename

        pil_img = row["image"]
        if pil_img is not None:
            pil_img = pil_img.convert("RGB")
            pil_img.save(str(img_path))

        samples.append({
            "id": sid,
            "text": row["question"],
            "image_pil": pil_img,
            "image_path": str(img_path),
            "source": "mm-safetybench",
            "scenario": row["scenario"],
        })
        samples_meta.append({
            "id": sid,
            "text": row["question"],
            "image_filename": img_filename,
            "scenario": row["scenario"],
        })

    local_cache.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(samples_meta, f, indent=2)
    print(f"    Cached {len(samples)} samples → {local_cache}")
    return samples


def load_llava_instruct_reference(
    n_samples: int = 160,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Load and sample from LLaVA-Instruct-80k for **safe** reference data.

    Downloads only the ~``n_samples`` COCO images actually needed.

    Following ShiftDC Appendix A.3: 160 samples from LLaVA-Instruct-80k,
    each with a unique image paired with a single instruction.

    Returns list of sample dicts compatible with ``generate_captions()`` and
    ``load_image_for_sample()``.
    """
    import random
    import urllib.request
    from tqdm import tqdm as _tqdm

    repo_root = Path(__file__).resolve().parent.parent
    local_cache = Path(cache_dir) if cache_dir else repo_root / "data" / "llava-instruct-ref"
    meta_path = local_cache / f"samples_n{n_samples}_seed{seed}.json"
    images_dir = local_cache / "images"

    COCO_BASE = "http://images.cocodataset.org/train2014/COCO_train2014_{filename}"

    # ── Cached? ───────────────────────────────────────────────────────────
    if meta_path.exists():
        print(f"  Loading cached LLaVA-Instruct reference ({meta_path})")
        with open(meta_path) as f:
            samples_meta = json.load(f)
        samples = []
        for m in samples_meta:
            img_path = str(images_dir / m["image_filename"])
            samples.append({
                "id": m["id"],
                "text": m["text"],
                "image_pil": None,
                "image_path": img_path,
                "source": "llava-instruct-80k",
            })
        print(f"  {len(samples)} cached samples loaded.")
        return samples

    # ── Download JSON from HuggingFace ────────────────────────────────────
    from huggingface_hub import hf_hub_download

    print("    Downloading llava_instruct_80k.json ...")
    json_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="llava_instruct_80k.json",
        repo_type="dataset",
    )
    with open(json_path) as f:
        all_entries = json.load(f)
    print(f"    Loaded {len(all_entries)} entries from LLaVA-Instruct-80k")

    rng = random.Random(seed)
    sampled = rng.sample(all_entries, min(n_samples, len(all_entries)))

    # ── Download COCO images ──────────────────────────────────────────────
    images_dir.mkdir(parents=True, exist_ok=True)
    samples: List[dict] = []
    samples_meta: list = []

    for entry in _tqdm(sampled, desc="    Downloading COCO images"):
        coco_filename = entry["image"]              # e.g. "000000215677.jpg"

        # Extract the human turn, stripping the <image> tag
        text = ""
        for turn in entry.get("conversations", []):
            if turn.get("from") == "human":
                text = turn.get("value", "")
                for prefix in ("<image>\n", "<image>"):
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                text = text.strip()
                break

        sid = f"llava_{entry['id']}"
        img_local = images_dir / coco_filename

        if not img_local.exists():
            url = COCO_BASE.format(filename=coco_filename)
            try:
                urllib.request.urlretrieve(url, str(img_local))
            except Exception as e:
                print(f"    Warning: failed to download {url}: {e}")
                continue

        samples.append({
            "id": sid,
            "text": text,
            "image_pil": None,
            "image_path": str(img_local),
            "source": "llava-instruct-80k",
        })
        samples_meta.append({
            "id": sid,
            "text": text,
            "image_filename": coco_filename,
        })

    local_cache.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(samples_meta, f, indent=2)
    print(f"    Cached {len(samples)} samples → {local_cache}")
    return samples


# ── Reference dataset registry ────────────────────────────────────────────────
#
# Maps dataset name → {role, loader}.
# To add a new reference dataset:
#   1. Write a loader function that returns List[dict] with keys:
#      id, text, image_pil, image_path, source
#   2. Add an entry here with role "safe" or "unsafe".
#
REFERENCE_REGISTRY: Dict[str, dict] = {
    "mm-safetybench": {
        "role":      "unsafe",
        "loader":    load_mmsafetybench_reference,
        "text_only": False,   # has images → captions required
    },
    "llava-instruct": {
        "role":      "safe",
        "loader":    load_llava_instruct_reference,
        "text_only": False,   # has images → captions required
    },
    # To add a text-only dataset (e.g. prompt/response pairs with no images):
    #   1. Write a loader function returning List[dict] with keys: id, text, image_pil=None, image_path=None
    #   2. Add an entry here with text_only=True
    #   3. No need to run generate_captions.py for it
}


if __name__ == "__main__":
    load_holisafe()