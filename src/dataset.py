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

    # Prefer the dataset's own id field over the enumeration index
    sample_id = entry.get("id", idx)

    return {
        "id": sample_id,
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


# ── CatQA contrastive pair loaders ────────────────────────────────────────────

def _load_catqa_pairs(cache_dir: Optional[str] = None) -> List[dict]:
    """Load the CatQA contrastive pairs JSON generated by generate_catqa_harmless_pairs.py."""
    repo_root = Path(__file__).resolve().parent.parent
    if cache_dir:
        pairs_path = Path(cache_dir) / "catqa_contrastive_pairs.json"
    else:
        pairs_path = repo_root / "data" / "catqa-contrastive" / "catqa_contrastive_pairs.json"

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"CatQA contrastive pairs not found: {pairs_path}\n"
            "Run first: python generate_catqa_harmless_pairs.py"
        )
    with open(pairs_path) as f:
        return json.load(f)


def load_catqa_harmful_reference(
    n_samples: int = 160,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Load harmful questions from CatQA contrastive pairs as **unsafe** reference data.

    These are the original harmful questions from CategoricalHarmfulQA, paired
    with minimal-edit harmless counterparts for contrastive safety direction extraction.
    """
    import random

    pairs = _load_catqa_pairs(cache_dir)
    # Filter out any pairs where generation failed (empty harmless)
    pairs = [p for p in pairs if p.get("question_harmless")]

    rng = random.Random(seed)
    if len(pairs) > n_samples:
        pairs = rng.sample(pairs, n_samples)

    samples = []
    for i, p in enumerate(pairs):
        samples.append({
            "id": f"catqa_harmful_{i}",
            "text": p["question_harmful"],
            "image_pil": None,
            "image_path": None,
            "source": "catqa-harmful",
            "category": p.get("category", "unknown"),
            "subcategory": p.get("subcategory", "unknown"),
        })
    print(f"  Loaded {len(samples)} CatQA harmful samples.")
    return samples


def load_catqa_harmless_reference(
    n_samples: int = 160,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Load harmless counterparts from CatQA contrastive pairs as **safe** reference data.

    These are minimal-edit harmless rewrites of CategoricalHarmfulQA questions,
    designed to differ minimally from the harmful versions so that the contrastive
    direction isolates the concept of safety rather than topic differences.
    """
    import random

    pairs = _load_catqa_pairs(cache_dir)
    pairs = [p for p in pairs if p.get("question_harmless")]

    rng = random.Random(seed)
    if len(pairs) > n_samples:
        pairs = rng.sample(pairs, n_samples)

    samples = []
    for i, p in enumerate(pairs):
        samples.append({
            "id": f"catqa_harmless_{i}",
            "text": p["question_harmless"],
            "image_pil": None,
            "image_path": None,
            "source": "catqa-harmless",
            "category": p.get("category", "unknown"),
            "subcategory": p.get("subcategory", "unknown"),
        })
    print(f"  Loaded {len(samples)} CatQA harmless samples.")
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
    "catqa-harmful": {
        "role":      "unsafe",
        "loader":    load_catqa_harmful_reference,
        "text_only": True,    # text-only → no captions needed
    },
    "catqa-harmless": {
        "role":      "safe",
        "loader":    load_catqa_harmless_reference,
        "text_only": True,    # text-only → no captions needed
    },
}


def split_holisafe_train_eval(
    sss_samples: List[dict],
    ssu_samples: List[dict],
    n_eval: int = 175,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Partition SSS and SSU samples into train/eval sets, stratified by harm category.

    On subsequent calls with matching seed and n_eval, loads the saved split
    rather than recomputing.

    Args:
        sss_samples: list of SSS sample dicts
        ssu_samples: list of SSU sample dicts
        n_eval: target number of evaluation samples per group (SSS and SSU each).
                Allocated proportionally per category. The rest of each group goes
                to training. Default 175 ≈ 25% eval for the typical HoliSafe pool
                (yielding the desired 75/25 train/eval split for the probes).
        seed: random seed for reproducibility
        save_dir: directory to save/load the split JSON.
                  Defaults to <repo_root>/data/holisafe-bench/

    Returns:
        (sss_train, sss_eval, ssu_train, ssu_eval)
    """
    import random

    repo_root = Path(__file__).resolve().parent.parent
    if save_dir is None:
        save_dir = str(repo_root / "data" / "holisafe-bench")
    split_path = Path(save_dir) / "train_eval_split.json"

    # Build lookup by id for both groups
    sss_by_id = {s["id"]: s for s in sss_samples}
    ssu_by_id = {s["id"]: s for s in ssu_samples}

    # Try loading existing split (only honor it if it was saved with the new
    # n_eval-keyed schema and matches; older n_train-keyed splits are discarded).
    if split_path.exists():
        with open(split_path) as f:
            saved = json.load(f)
        if saved.get("seed") == seed and saved.get("n_eval") == n_eval:
            print(f"  Loading saved train/eval split from {split_path}")
            sss_train = [sss_by_id[i] for i in saved["sss_train_ids"] if i in sss_by_id]
            sss_eval = [sss_by_id[i] for i in saved["sss_eval_ids"] if i in sss_by_id]
            ssu_train = [ssu_by_id[i] for i in saved["ssu_train_ids"] if i in ssu_by_id]
            ssu_eval = [ssu_by_id[i] for i in saved["ssu_eval_ids"] if i in ssu_by_id]
            print(f"  SSS: {len(sss_train)} train, {len(sss_eval)} eval")
            print(f"  SSU: {len(ssu_train)} train, {len(ssu_eval)} eval")
            return sss_train, sss_eval, ssu_train, ssu_eval

    def _stratified_split(samples, n_eval_group, rng):
        """Split samples stratified by category, allocating n_eval_group to eval."""
        by_cat = {}
        for s in samples:
            by_cat.setdefault(s["category"], []).append(s)

        total = len(samples)
        train_ids, eval_ids = [], []

        for cat, cat_samples in sorted(by_cat.items()):
            rng.shuffle(cat_samples)
            # Proportional allocation of eval; rest goes to train.
            n_cat_eval = max(1, round(len(cat_samples) * n_eval_group / total))
            n_cat_eval = min(n_cat_eval, len(cat_samples) - 1)  # keep at least 1 for train
            eval_ids.extend(cat_samples[:n_cat_eval])
            train_ids.extend(cat_samples[n_cat_eval:])

        return train_ids, eval_ids

    rng = random.Random(seed)
    sss_train, sss_eval = _stratified_split(sss_samples, n_eval, rng)
    ssu_train, ssu_eval = _stratified_split(ssu_samples, n_eval, rng)

    # Persist
    split_data = {
        "seed": seed,
        "n_eval": n_eval,
        "sss_train_ids": [s["id"] for s in sss_train],
        "sss_eval_ids": [s["id"] for s in sss_eval],
        "ssu_train_ids": [s["id"] for s in ssu_train],
        "ssu_eval_ids": [s["id"] for s in ssu_eval],
    }
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"  Saved train/eval split → {split_path}")
    print(f"  SSS: {len(sss_train)} train, {len(sss_eval)} eval")
    print(f"  SSU: {len(ssu_train)} train, {len(ssu_eval)} eval")

    return sss_train, sss_eval, ssu_train, ssu_eval


def extend_holisafe_eval_compositional(
    reference_subsets: dict,
    n_eval: int = 175,
    seed: int = 42,
    save_dir: Optional[str] = None,
    subsets: Optional[List[str]] = None,
) -> dict:
    """
    Append eval-only id lists for the compositional-unsafety subsets
    (USU, SUU, UUU) to the existing train_eval_split.json.

    Each list is stratified by harm category to mirror the proportions in the
    full pool of that subset, and is sized at ~n_eval. Idempotent: keys that
    are already present (and non-empty) are not regenerated. SSS/SSU keys
    are never touched.

    Args:
        reference_subsets: output of `filter_reference_subsets(entries, ...)`.
                           A dict keyed by raw HoliSafe `type` string
                           (SSS / SSU / SUU / USU / UUU) with sample lists.
        n_eval: target eval-set size per subset.
        seed: master seed; per-subset deterministic streams are derived from it.
        save_dir: directory holding train_eval_split.json. Defaults to
                  <repo_root>/data/holisafe-bench/.
        subsets: subsets to add (defaults to ["SUU", "USU", "UUU"]).

    Returns:
        The full updated split dict (after read-modify-write).

    Raises:
        FileNotFoundError if train_eval_split.json does not exist (call
        `split_holisafe_train_eval` first).
        ValueError if the existing split's seed/n_eval don't match.
    """
    import random

    if subsets is None:
        subsets = ["SUU", "USU", "UUU"]

    repo_root = Path(__file__).resolve().parent.parent
    if save_dir is None:
        save_dir = str(repo_root / "data" / "holisafe-bench")
    split_path = Path(save_dir) / "train_eval_split.json"

    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found. Run split_holisafe_train_eval(...) first "
            "to create the SSS/SSU split."
        )

    with open(split_path) as f:
        split = json.load(f)

    if split.get("seed") != seed or split.get("n_eval") != n_eval:
        raise ValueError(
            f"Existing split has seed={split.get('seed')}, "
            f"n_eval={split.get('n_eval')}; refusing to extend with "
            f"seed={seed}, n_eval={n_eval}. Either rerun "
            "split_holisafe_train_eval with these values, or pass matching "
            "seed/n_eval here."
        )

    def _stratified_eval_only(samples: List[dict], n_eval_target: int,
                              rng: "random.Random") -> List[dict]:
        """Pick a stratified-by-category subset of ~n_eval_target samples."""
        by_cat: Dict[str, List[dict]] = {}
        for s in samples:
            by_cat.setdefault(s.get("category", "unknown"), []).append(s)
        total = len(samples)
        out: List[dict] = []
        for cat, cat_samples in sorted(by_cat.items()):
            cat_samples = list(cat_samples)
            rng.shuffle(cat_samples)
            n_cat = max(1, round(len(cat_samples) * n_eval_target / total))
            n_cat = min(n_cat, len(cat_samples))
            out.extend(cat_samples[:n_cat])
        return out

    changed = False
    for subset in subsets:
        key = f"{subset.lower()}_eval_ids"
        if split.get(key):
            print(f"  [{subset}] {key} already present "
                  f"({len(split[key])} ids); skipping.")
            continue
        samples = reference_subsets.get(subset)
        if not samples:
            print(f"  [{subset}] no samples available in reference_subsets; "
                  "skipping.")
            continue
        rng_seed = seed + sum(ord(c) for c in subset)
        rng = random.Random(rng_seed)
        picked = _stratified_eval_only(samples, n_eval, rng)
        split[key] = sorted(s["id"] for s in picked)
        print(f"  [{subset}] picked {len(picked)} eval samples (target {n_eval}, "
              f"pool {len(samples)})")
        changed = True

    if changed:
        # Atomic-ish write: write to .tmp then rename.
        tmp_path = split_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(split, f, indent=2)
        tmp_path.replace(split_path)
        print(f"  Saved extended split → {split_path}")
    else:
        print("  No new keys added; train_eval_split.json unchanged.")

    return split


def split_catqa_train_eval(
    n_samples: int = 550,
    n_eval: int = 132,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    """
    Stratified train/eval split of the CatQA pool used during reference
    activation extraction, by category.

    Returns indices into the rows of `activation_matrices.npz`. To stay aligned
    with the rows produced by `extract_ref_activations.py`, the same loader
    (and same `n_samples`/`seed`) is used here to recover the row order:
    activation row `i` corresponds to the i-th sample returned by
    `load_catqa_harmful_reference(n_samples=n_samples, seed=seed)` (whose `id`
    is `catqa_harmful_{i}`).

    Args:
        n_samples: must match the `n_samples` used at extraction time (read
                   from each model's `metadata.json`).
        n_eval: target total number of eval examples across categories,
                allocated proportionally. The final eval size is approximate
                because per-category counts are integer-rounded.
        seed: random seed; must match extraction-time seed for row alignment.
        cache_dir: directory containing catqa_contrastive_pairs.json.
        save_dir: directory to persist the split JSON. Defaults to
                  `<repo_root>/data/catqa-contrastive/splits/`. The split file
                  is keyed by (n_samples, seed, n_eval) so different
                  extractions don't collide.

    Returns:
        (train_indices, eval_indices) — sorted lists of integer row indices.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if cache_dir is None:
        cache_dir = str(repo_root / "data" / "catqa-contrastive")
    if save_dir is None:
        save_dir = str(repo_root / "data" / "catqa-contrastive" / "splits")
    split_path = (Path(save_dir) /
                  f"train_eval_split_n{n_samples}_seed{seed}_neval{n_eval}.json")

    # Use the same loader that extraction used, so id catqa_harmful_i
    # corresponds to row i of the activation matrix.
    samples = load_catqa_harmful_reference(
        n_samples=n_samples, cache_dir=cache_dir, seed=seed
    )
    n_total = len(samples)
    if n_eval >= n_total:
        raise ValueError(
            f"n_eval={n_eval} >= n_total={n_total}: not enough samples to split. "
            "Re-extract reference activations with a larger REF_SAMPLES."
        )
    if n_eval > n_total * 0.5:
        print(f"  WARNING: n_eval={n_eval} is more than half of n_total={n_total}; "
              "consider re-extracting with a larger REF_SAMPLES.")

    if split_path.exists():
        with open(split_path) as f:
            saved = json.load(f)
        if (saved.get("seed") == seed
                and saved.get("n_eval") == n_eval
                and saved.get("n_samples") == n_samples
                and saved.get("n_total") == n_total):
            print(f"  Loading saved CatQA split from {split_path}")
            train_idx = sorted(saved["train_indices"])
            eval_idx = sorted(saved["eval_indices"])
            print(f"  CatQA: {len(train_idx)} train, {len(eval_idx)} eval")
            return train_idx, eval_idx

    import random
    by_cat: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_cat.setdefault(s.get("category", "unknown"), []).append(i)

    rng = random.Random(seed + 1)  # offset to decouple from loader's RNG
    train_idx, eval_idx = [], []
    for cat, idxs in sorted(by_cat.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_cat_eval = max(1, round(len(idxs) * n_eval / n_total))
        n_cat_eval = min(n_cat_eval, len(idxs) - 1)
        eval_idx.extend(idxs[:n_cat_eval])
        train_idx.extend(idxs[n_cat_eval:])

    train_idx.sort()
    eval_idx.sort()

    eval_cat_counts = {c: 0 for c in by_cat}
    train_cat_counts = {c: 0 for c in by_cat}
    for i in eval_idx:
        eval_cat_counts[samples[i].get("category", "unknown")] += 1
    for i in train_idx:
        train_cat_counts[samples[i].get("category", "unknown")] += 1

    split_data = {
        "seed": seed,
        "n_samples": n_samples,
        "n_eval": n_eval,
        "n_total": n_total,
        "train_indices": train_idx,
        "eval_indices": eval_idx,
        "train_category_counts": train_cat_counts,
        "eval_category_counts": eval_cat_counts,
    }
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"  Saved CatQA train/eval split → {split_path}")
    print(f"  CatQA: {len(train_idx)} train, {len(eval_idx)} eval")

    return train_idx, eval_idx


# ── MSSBench (diagnostic-side loader; eval-side loader lives at evaluation/benchmarks/mssbench.py) ──
#
# MSSBench has paired SSS/SSU samples on the same query — the IMAGE alone
# determines whether a refusal is the safe response. Each chat record produces
# 2 × len(queries) diagnostic-side samples, so per-pair PCA is well-defined:
# for matching (rec_idx, q_idx), the SSS sample uses safe_image_path and the
# SSU sample uses unsafe_image_path with the same question.
#
# The id format below MUST match `EvalSample.id` produced by
# `evaluation/benchmarks/mssbench.py:153` so that activations cached on the
# diagnostic side can be addressed by the eval-side EvalSample.id (and vice
# versa). When changing one, change both.
def load_mssbench(
    data_dir: Optional[str] = None,
    splits: Tuple[str, ...] = ("chat",),
) -> List[dict]:
    """Load MSSBench as diagnostic-side sample dicts (parallel to HoliSafe).

    Per-sample dict schema:
      {
        "id":           "mssbench_{rec_idx:04d}_{SSS|SSU}_{stem}_q{q_idx}",
        "image_path":   str (absolute),
        "image_pil":    PIL.Image,
        "text":         str (the query),
        "label":        "SSS" or "SSU",
        "label_idx":    0 or 1,
        "subset_type":  "SSS" or "SSU",
        "category":     str (record's "Type" field, e.g. "harmful"|"property"|...),
        "rec_idx":      int,
        "q_idx":        int,
        "raw":          dict (full record),
      }
    """
    repo_root = Path(__file__).resolve().parent.parent
    if data_dir is None:
        data_dir = str(repo_root / "data" / "mssbench")
    data_dir = Path(data_dir)
    records_path = data_dir / "combined.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"MSSBench combined.json not found at {records_path}.\n"
            "Run: python evaluation/scripts/download_mssbench.py"
        )
    with open(records_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        records: List[dict] = []
        for split in splits:
            if split in raw and isinstance(raw[split], list):
                records.extend(raw[split])
        if not records:
            raise ValueError(
                f"No records in combined.json for splits {splits}. "
                f"Available: {list(raw.keys())}"
            )
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unexpected combined.json type: {type(raw).__name__}")

    def _resolve(rel: str) -> Optional[Path]:
        if not rel:
            return None
        for split in splits:
            cand = data_dir / split / rel
            if cand.exists():
                return cand
        cand = data_dir / rel
        return cand if cand.exists() else None

    samples: List[dict] = []
    for rec_idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        queries = record.get("queries") or []
        if not isinstance(queries, list):
            continue
        rec_type = record.get("Type") or record.get("type") or "unknown"
        for variant_label, path_field in (
            ("SSS", "safe_image_path"),
            ("SSU", "unsafe_image_path"),
        ):
            rel = record.get(path_field)
            if not isinstance(rel, str) or not rel.strip():
                continue
            img_path = _resolve(rel)
            if img_path is None:
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [mssbench] failed to open {img_path}: {e}")
                continue
            stem = img_path.stem
            for q_idx, question in enumerate(queries):
                if not isinstance(question, str) or not question.strip():
                    continue
                sid = f"mssbench_{rec_idx:04d}_{variant_label}_{stem}_q{q_idx}"
                samples.append({
                    "id": sid,
                    "image_path": str(img_path),
                    "image_pil": image,
                    "text": question,
                    "label": variant_label,
                    "label_idx": 0 if variant_label == "SSS" else 1,
                    "subset_type": variant_label,
                    "category": rec_type,
                    "rec_idx": rec_idx,
                    "q_idx": q_idx,
                    "raw": record,
                })
    return samples


def split_mssbench_train_eval(
    samples: Optional[List[dict]] = None,
    train_frac: float = 0.75,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> dict:
    """Stratified-by-Type, **record-level** train/eval split of MSSBench.

    Both image variants (SSS + SSU) of every (rec_idx, q_idx) pair stay in
    the same split — the unit being shuffled is the record, not the sample.

    Persists a `train_eval_split.json` under `data/mssbench/`. Idempotent:
    a saved file matching seed and train_frac is reused.

    Returns the persisted split dict.
    """
    import random

    repo_root = Path(__file__).resolve().parent.parent
    if save_dir is None:
        save_dir = str(repo_root / "data" / "mssbench")
    split_path = Path(save_dir) / "train_eval_split.json"

    if split_path.exists():
        with open(split_path) as f:
            saved = json.load(f)
        if (saved.get("seed") == seed
                and abs(float(saved.get("train_frac", -1)) - train_frac) < 1e-9):
            print(f"  Loading saved MSSBench split from {split_path}")
            return saved

    if samples is None:
        samples = load_mssbench()

    # Group sample ids by record; record's Type drives stratification.
    by_record: Dict[int, dict] = {}
    for s in samples:
        rec = by_record.setdefault(s["rec_idx"], {
            "rec_idx": s["rec_idx"],
            "type": s["category"],
            "sample_ids": [],
            "pair_keys": set(),
        })
        rec["sample_ids"].append(s["id"])
        rec["pair_keys"].add((s["rec_idx"], s["q_idx"]))

    # Stratify rec_idx by type.
    by_type: Dict[str, List[int]] = {}
    for rec_idx, rec in by_record.items():
        by_type.setdefault(rec["type"], []).append(rec_idx)

    rng = random.Random(seed)
    train_record_ids: List[int] = []
    eval_record_ids: List[int] = []
    category_counts: Dict[str, dict] = {}
    for cat, rec_list in sorted(by_type.items()):
        rec_list = list(rec_list)
        rng.shuffle(rec_list)
        n_train = max(1, round(len(rec_list) * train_frac))
        n_train = min(n_train, len(rec_list) - 1)  # keep at least 1 for eval
        train_record_ids.extend(rec_list[:n_train])
        eval_record_ids.extend(rec_list[n_train:])
        category_counts[cat] = {
            "train_records": n_train,
            "eval_records": len(rec_list) - n_train,
        }

    train_set = set(train_record_ids)
    train_sample_ids: List[str] = []
    eval_sample_ids: List[str] = []
    train_pair_keys: List[List[int]] = []
    eval_pair_keys: List[List[int]] = []
    for rec_idx, rec in by_record.items():
        bucket_samples = train_sample_ids if rec_idx in train_set else eval_sample_ids
        bucket_pairs = train_pair_keys if rec_idx in train_set else eval_pair_keys
        bucket_samples.extend(sorted(rec["sample_ids"]))
        bucket_pairs.extend(sorted([list(pk) for pk in rec["pair_keys"]]))

    split_data = {
        "seed": seed,
        "train_frac": train_frac,
        "train_record_ids": sorted(train_record_ids),
        "eval_record_ids": sorted(eval_record_ids),
        "train_sample_ids": sorted(train_sample_ids),
        "eval_sample_ids": sorted(eval_sample_ids),
        "train_pair_keys": sorted(train_pair_keys),
        "eval_pair_keys": sorted(eval_pair_keys),
        "stratified_by": "Type",
        "category_counts": category_counts,
    }
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"  Saved MSSBench train/eval split → {split_path}")
    print(f"  Records: {len(train_record_ids)} train / {len(eval_record_ids)} eval")
    print(f"  Samples: {len(train_sample_ids)} train / {len(eval_sample_ids)} eval")
    return split_data


# ── Source-of-truth dataset path mapping ─────────────────────────────────────
# Used by extract_vl/tt.py, compositional_safety_direction.py, and safety_probes.py
# so per-dataset paths flow from a single registry rather than scattered hardcodes.
DATASET_DATA_DIRS: Dict[str, str] = {
    "holisafe": "holisafe-bench",
    "mssbench": "mssbench",
    "mm_safetybench": "mm-safetybench",
    "figstep": "figstep",
}


def model_data_root(benchmark: str, model_short: str,
                    project_root: Optional[Path] = None) -> Path:
    """Return data/{benchmark_dir}/{model_short}/.

    Convention: every per-model artifact under a benchmark lives at
        data/{benchmark_dir}/{model_short}/{activations|responses}/...
    """
    root = project_root or Path(__file__).resolve().parent.parent
    return root / "data" / DATASET_DATA_DIRS.get(benchmark, benchmark) / model_short


def model_activations_dir(benchmark: str, model_short: str,
                          project_root: Optional[Path] = None) -> Path:
    return model_data_root(benchmark, model_short, project_root) / "activations"


def model_responses_dir(benchmark: str, model_short: str,
                        intervention: str = "vanilla",
                        project_root: Optional[Path] = None) -> Path:
    """Return data/{benchmark_dir}/{model_short}/responses/{intervention}/."""
    return model_data_root(benchmark, model_short, project_root) / "responses" / intervention


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Persist train/eval splits for downstream diagnostics. "
                    "Without --mssbench_split: extends HoliSafe "
                    "train_eval_split.json with eval-only id lists for the "
                    "compositional-unsafety subsets (USU/SUU/UUU). "
                    "With --mssbench_split: produces "
                    "data/mssbench/train_eval_split.json (record-level, "
                    "stratified by Type)."
    )
    p.add_argument("--mssbench_split", action="store_true",
                   help="Generate the MSSBench 75/25 record-level split JSON.")
    p.add_argument("--mssbench_train_frac", type=float, default=0.75,
                   help="Train fraction for --mssbench_split (default 0.75).")
    p.add_argument("--n_eval", type=int, default=175,
                   help="HoliSafe target eval-set size per subset (default 175).")
    p.add_argument("--seed", type=int, default=42,
                   help="Master seed for stratified sampling.")
    p.add_argument("--cache_dir", default=None,
                   help="HuggingFace cache dir for HoliSafe download.")
    p.add_argument("--subsets", nargs="+", default=["SUU", "USU", "UUU"],
                   choices=["SSS", "SSU", "SUU", "USU", "UUU"],
                   help="HoliSafe subsets to add (default: SUU USU UUU).")
    args = p.parse_args()

    if args.mssbench_split:
        split_mssbench_train_eval(
            train_frac=args.mssbench_train_frac, seed=args.seed,
        )
    else:
        entries, images_base = load_holisafe(cache_dir=args.cache_dir)
        refs = filter_reference_subsets(entries, images_base)
        print("Pool sizes per subset:",
              {k: len(v) for k, v in sorted(refs.items())})
        extend_holisafe_eval_compositional(
            refs, n_eval=args.n_eval, seed=args.seed, subsets=args.subsets,
        )