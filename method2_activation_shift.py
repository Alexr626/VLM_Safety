#!/usr/bin/env python3
"""
Method 2: Safety-Relevant Activation Shift Analysis (ShiftDC-style)
====================================================================
From: "Understanding and Rectifying Safety Perception Distortion in VLMs"
       (Zou et al., 2025)

Diagnostic question
-------------------
When an image is introduced (vs. its caption), does the VLM's activation
shift toward the "safe" side of the safety boundary?  Does this modality-
induced shift differ between SSS and SSU examples — and does it explain
why models miss emergent harm?

Steps
-----
  1. Generate captions for every SSS / SSU image using the VLM itself.
  2. Compute safety direction s^l at each layer:
       s^l = mean_activations^l(safe_text) − mean_activations^l(unsafe_text)
     Safe text:   benign instructions from reference_data/safe_instructions.txt
     Unsafe text: text from SiUt / UiUt examples in HoliSafe-Bench
                  (text labeled unsafe even if image is safe/ignored)
  3. For each SSS / SSU sample:
       VL  activation x^l(vl)  = image + text → last-token hidden state at l
       TT  activation x^l(tt)  = caption + text (text-only) → last-token hidden state at l
       modality shift  m^l = x^l(vl) − x^l(tt)
       cosine_sim^l    = cosine(m^l, s^l)
       proj_magnitude^l = (m^l · s^l) / ||s^l||²
  4. Aggregate and compare SSS vs SSU per layer; run t-test.

Outputs (under {output_dir}/{model_name}/method2_activation_shift/)
-------
  captions.json               — generated captions per sample
  safety_direction_vectors.npz — s^l per layer
  per_sample_shifts.json       — cosine_sim, projection, label per sample per layer
  aggregate_stats.json         — mean cosine/projection + t-test p-values per layer
  sample_metadata.json

Usage
-----
  python method2_activation_shift.py
  python method2_activation_shift.py --model llava-hf/llava-1.5-7b-hf \\
      --output_dir outputs --limit 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import (
    load_holisafe, inspect_schema, filter_subsets,
    filter_reference_subsets, load_image_for_sample,
)
from src.model import VLMWrapper
from src.extraction import (
    ActivationCache, get_last_token_activations,
    cleanup_gpu, save_json, save_pickle, save_npz,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Method 2: Activation Shift Analysis")
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit samples per class (for quick testing)")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--skip_captions", action="store_true",
                   help="Skip caption generation if captions.json already exists")
    p.add_argument("--skip_extraction", action="store_true",
                   help="Skip forward passes if activations are already cached")
    p.add_argument("--safe_ref_file",
                   default="reference_data/safe_instructions.txt",
                   help="File of safe reference instructions (one per line)")
    p.add_argument("--unsafe_ref_file", default=None,
                   help="Optional file of unsafe reference instructions. "
                        "If not provided, uses SiUt/UiUt text from HoliSafe-Bench.")
    p.add_argument("--max_ref_samples", type=int, default=100,
                   help="Max reference samples per class for safety direction")
    return p.parse_args()


# ── Reference data loading ────────────────────────────────────────────────────

def load_safe_instructions(path: str, max_n: int) -> list:
    """Load safe text instructions from a flat text file (# = comment)."""
    instructions = []
    if not os.path.exists(path):
        print(f"  Warning: safe reference file '{path}' not found. Using fallback.")
        instructions = [
            "What is the capital of France?",
            "How do I bake a cake?",
            "Explain the water cycle.",
            "What are tips for better sleep?",
            "How does photosynthesis work?",
        ]
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    instructions.append(line)
    return instructions[:max_n]


def load_unsafe_instructions(path: str, max_n: int) -> list:
    """Load unsafe text instructions from a flat text file."""
    if not os.path.exists(path):
        return []
    instructions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                instructions.append(line)
    return instructions[:max_n]


def extract_unsafe_from_holisafe(ref_buckets: dict, max_n: int) -> list:
    """
    Extract text instructions from SiUt (safe image, unsafe text) or UiUt
    (unsafe image, unsafe text) HoliSafe subsets.  These texts are labeled as
    unsafe regardless of image, making them good "unsafe text-only" references.
    """
    unsafe_texts = []
    for k, samples in ref_buckets.items():
        k_norm = k.lower().replace("→", "->").replace("_", "->")
        # SiUt: safe image + unsafe text, UiUt: unsafe image + unsafe text
        if ("siut" in k_norm or "si_ut" in k_norm or
                "uiut" in k_norm or "ui_ut" in k_norm or
                ("ut" in k_norm and "unsafe" in k_norm)):
            for s in samples:
                if s["text"]:
                    unsafe_texts.append(s["text"])
                if len(unsafe_texts) >= max_n:
                    return unsafe_texts

    # Fallback: try any subset whose name suggests unsafe text
    if not unsafe_texts:
        for k, samples in ref_buckets.items():
            k_norm = k.lower()
            if "unsafe" in k_norm:
                for s in samples:
                    if s["text"]:
                        unsafe_texts.append(s["text"])
                    if len(unsafe_texts) >= max_n:
                        return unsafe_texts

    return unsafe_texts[:max_n]


# ── Caption generation ────────────────────────────────────────────────────────

def generate_captions(
    samples: list,
    wrapper: VLMWrapper,
    captions_path: str,
    skip_if_exists: bool = False,
) -> dict:
    """
    Generate captions for each sample image. Returns {sample_id: caption_str}.
    Saves / loads from captions_path.
    """
    if skip_if_exists and os.path.exists(captions_path):
        print(f"  Loading existing captions from {captions_path}")
        with open(captions_path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    captions = {}
    for sample in tqdm(samples, desc="Generating captions"):
        image = load_image_for_sample(sample)
        if image is None:
            captions[sample["id"]] = ""
            continue
        try:
            cap = wrapper.generate_caption(image)
            captions[sample["id"]] = cap
        except Exception as e:
            print(f"  Warning: caption failed for sample {sample['id']}: {e}")
            captions[sample["id"]] = ""
        cleanup_gpu()

    # Save
    Path(captions_path).parent.mkdir(parents=True, exist_ok=True)
    with open(captions_path, "w") as f:
        json.dump({str(k): v for k, v in captions.items()}, f, indent=2)
    print(f"  Saved captions → {captions_path}")
    return captions


# ── Safety direction computation ──────────────────────────────────────────────

def compute_safety_direction(
    safe_texts: list,
    unsafe_texts: list,
    wrapper: VLMWrapper,
    layer_indices: range,
) -> dict:
    """
    Compute s^l = mean_safe^l − mean_unsafe^l for each layer.

    Returns dict {layer_idx: np.ndarray of shape (hidden_dim,)}.
    """
    def _extract_means(text_list: list, desc: str) -> dict:
        layer_sums = {l: np.zeros(wrapper.hidden_dim, dtype=np.float64)
                      for l in layer_indices}
        count = 0
        for text in tqdm(text_list, desc=desc):
            try:
                hidden, _, _ = wrapper.forward_text(text)
                acts = get_last_token_activations(hidden)
                for l in layer_indices:
                    if l in acts:
                        layer_sums[l] += acts[l].astype(np.float64)
                count += 1
            except Exception as e:
                print(f"  Warning: text forward pass failed: {e}")
            cleanup_gpu()
        if count == 0:
            return {l: layer_sums[l] for l in layer_indices}
        return {l: layer_sums[l] / count for l in layer_indices}

    print(f"  Computing mean activations for {len(safe_texts)} safe instructions...")
    mean_safe   = _extract_means(safe_texts,   desc="Safe ref")

    print(f"  Computing mean activations for {len(unsafe_texts)} unsafe instructions...")
    mean_unsafe = _extract_means(unsafe_texts, desc="Unsafe ref")

    safety_dir = {l: (mean_safe[l] - mean_unsafe[l]).astype(np.float32)
                  for l in layer_indices}
    return safety_dir


# ── TT (text-only counterpart) activations ───────────────────────────────────

def build_tt_prompt(text: str, caption: str) -> str:
    """Construct the text-only counterpart by prepending the image caption."""
    if caption:
        return f"Image description: {caption}\n\n{text}"
    return text


def extract_tt_activations(
    samples: list,
    captions: dict,
    wrapper: VLMWrapper,
    cache: ActivationCache,
    skip_if_cached: bool = True,
):
    """Extract text-only (caption + text) activations and cache as suffix='tt'."""
    todo = [s for s in samples
            if not (skip_if_cached and cache.exists(s["id"], "tt"))]
    if not todo:
        print(f"  All TT activations already cached.")
        return

    print(f"  Extracting TT activations for {len(todo)} samples...")
    for sample in tqdm(todo, desc="TT forward passes"):
        cap = captions.get(sample["id"], "")
        tt_text = build_tt_prompt(sample["text"], cap)
        try:
            hidden, _, _ = wrapper.forward_text(tt_text)
            acts = get_last_token_activations(hidden)
            cache.save(sample["id"], acts, suffix="tt")
            del hidden, acts
        except Exception as e:
            print(f"  Warning: TT pass failed for sample {sample['id']}: {e}")
        cleanup_gpu()


# ── Shift computation ─────────────────────────────────────────────────────────

def compute_shifts(
    samples: list,
    safety_dir: dict,
    cache: ActivationCache,
    layer_indices,
) -> tuple:
    """
    For each sample, compute the modality-induced activation shift and its
    projection onto the safety direction.

    Returns:
        per_sample_shifts: list of dicts per sample (averaged over layers for JSON)
        layer_data: dict {layer_idx: {"sss_cos": [], "ssu_cos": [],
                                      "sss_proj": [], "ssu_proj": []}}
    """
    per_sample_shifts = []  # full record per sample
    layer_data = {l: {"sss_cos": [], "sss_proj": [], "ssu_cos": [], "ssu_proj": []}
                  for l in layer_indices}

    def _cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

    def _projection(m, s):
        s_norm_sq = float(np.dot(s, s))
        if s_norm_sq < 1e-12:
            return float("nan")
        return float(np.dot(m, s) / s_norm_sq)

    print(f"  Computing activation shifts for {len(samples)} samples...")
    for sample in tqdm(samples, desc="Computing shifts"):
        sid = sample["id"]
        lbl = sample["label"]

        vl_acts = cache.load_or_none(sid, "vl")
        tt_acts = cache.load_or_none(sid, "tt")
        if vl_acts is None or tt_acts is None:
            continue

        sample_record = {
            "sample_id": sid,
            "label": lbl,
            "category": sample["category"],
            "per_layer": {},
        }

        for l in layer_indices:
            if l not in vl_acts or l not in tt_acts or l not in safety_dir:
                continue

            vl_h = vl_acts[l].astype(np.float64)
            tt_h = tt_acts[l].astype(np.float64)
            s    = safety_dir[l].astype(np.float64)

            m = vl_h - tt_h          # modality-induced shift
            cos_sim  = _cosine(m, s)
            proj_mag = _projection(m, s)

            sample_record["per_layer"][str(l)] = {
                "cosine_sim_with_safety": cos_sim,
                "projection_magnitude": proj_mag,
            }

            key = f"{lbl.lower()}_cos"
            if not np.isnan(cos_sim):
                layer_data[l][key].append(cos_sim)
            key = f"{lbl.lower()}_proj"
            if not np.isnan(proj_mag):
                layer_data[l][key].append(proj_mag)

        per_sample_shifts.append(sample_record)

    return per_sample_shifts, layer_data


def compute_aggregate_stats(layer_data: dict) -> list:
    """
    For each layer, compute mean cosine/projection for SSS and SSU, plus t-test.
    Returns list of dicts (one per layer).
    """
    results = []
    for l in sorted(layer_data.keys()):
        d = layer_data[l]
        sss_cos  = np.array(d["sss_cos"])
        ssu_cos  = np.array(d["ssu_cos"])
        sss_proj = np.array(d["sss_proj"])
        ssu_proj = np.array(d["ssu_proj"])

        def _safe_ttest(a, b):
            if len(a) < 2 or len(b) < 2:
                return float("nan")
            return float(stats.ttest_ind(a, b).pvalue)

        results.append({
            "layer": l,
            "SSS_mean_cosine":        float(sss_cos.mean())  if len(sss_cos)  else None,
            "SSU_mean_cosine":        float(ssu_cos.mean())  if len(ssu_cos)  else None,
            "SSS_mean_projection":    float(sss_proj.mean()) if len(sss_proj) else None,
            "SSU_mean_projection":    float(ssu_proj.mean()) if len(ssu_proj) else None,
            "t_test_p_value_cosine":  _safe_ttest(sss_cos, ssu_cos),
            "t_test_p_value_projection": _safe_ttest(sss_proj, ssu_proj),
            "n_sss": len(sss_cos),
            "n_ssu": len(ssu_cos),
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model_name = args.model.split("/")[-1]
    out_base   = Path(args.output_dir) / model_name
    out_m2     = out_base / "method2_activation_shift"
    act_dir    = out_base / "activations"
    out_m2.mkdir(parents=True, exist_ok=True)

    cache = ActivationCache(str(act_dir))

    # ── Load dataset ──────────────────────────────────────────────────────
    print("\n[1/6] Loading HoliSafe-Bench...")
    entries, images_base = load_holisafe(cache_dir=args.cache_dir)

    if args.inspect:
        inspect_schema(entries)
        sys.exit(0)

    sss_samples, ssu_samples = filter_subsets(entries, images_base)
    if args.limit:
        sss_samples = sss_samples[:args.limit]
        ssu_samples = ssu_samples[:args.limit]

    all_samples = sss_samples + ssu_samples
    ref_buckets = filter_reference_subsets(entries, images_base)

    # ── Reference text data for safety direction ──────────────────────────
    safe_texts = load_safe_instructions(
        args.safe_ref_file, args.max_ref_samples
    )

    if args.unsafe_ref_file:
        unsafe_texts = load_unsafe_instructions(
            args.unsafe_ref_file, args.max_ref_samples
        )
    else:
        unsafe_texts = extract_unsafe_from_holisafe(
            ref_buckets, args.max_ref_samples
        )

    print(f"  Reference data: {len(safe_texts)} safe, {len(unsafe_texts)} unsafe instructions")
    if not unsafe_texts:
        print("  Warning: No unsafe reference text found. Safety direction will be uninformative.")
        print("  Provide --unsafe_ref_file or ensure HoliSafe-Bench has SiUt/UiUt subsets.")

    # Save sample metadata
    metadata = [
        {"id": s["id"], "label": s["label"], "category": s["category"],
         "text_snippet": s["text"][:120]}
        for s in all_samples
    ]
    save_json(metadata, str(out_m2 / "sample_metadata.json"))

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[2/6] Loading model...")
    wrapper = VLMWrapper(args.model).load()
    num_layers = wrapper.num_layers
    layer_indices = range(num_layers + 1)  # 0 = embedding layer

    # ── Generate captions ─────────────────────────────────────────────────
    print("\n[3/6] Generating image captions...")
    captions_path = str(out_m2 / "captions.json")
    captions = generate_captions(
        all_samples, wrapper, captions_path,
        skip_if_exists=args.skip_captions,
    )

    # ── Extract VL activations ────────────────────────────────────────────
    print("\n[4/6] Extracting VL activations (image + text)...")
    # Reuse cache from Method 1 if already populated
    vl_todo = [s for s in all_samples
               if not (args.skip_extraction and cache.exists(s["id"], "vl"))]
    if vl_todo:
        for sample in tqdm(vl_todo, desc="VL forward passes"):
            image = load_image_for_sample(sample)
            if image is None:
                continue
            try:
                hidden, _, _ = wrapper.forward_vl(image, sample["text"])
                acts = get_last_token_activations(hidden)
                cache.save(sample["id"], acts, suffix="vl")
                del hidden, acts
            except Exception as e:
                print(f"  Warning: sample {sample['id']} VL pass failed: {e}")
            cleanup_gpu()
    else:
        print("  VL activations already cached.")

    # ── Extract TT activations ────────────────────────────────────────────
    print("\n[4b/6] Extracting TT activations (caption + text, no image)...")
    extract_tt_activations(
        all_samples, captions, wrapper, cache,
        skip_if_cached=args.skip_extraction,
    )

    # ── Compute safety direction ──────────────────────────────────────────
    print("\n[5/6] Computing safety direction vectors...")
    safety_dir = compute_safety_direction(
        safe_texts, unsafe_texts, wrapper, layer_indices,
    )

    # Save safety direction vectors
    save_npz(
        {f"layer_{l}": safety_dir[l] for l in layer_indices},
        str(out_m2 / "safety_direction_vectors.npz"),
    )

    # Done with model — free GPU memory
    cleanup_gpu()
    del wrapper

    # ── Compute shifts ────────────────────────────────────────────────────
    print("\n[6/6] Computing modality-induced activation shifts...")
    per_sample_shifts, layer_data = compute_shifts(
        all_samples, safety_dir, cache, layer_indices,
    )

    aggregate_stats = compute_aggregate_stats(layer_data)

    # ── Save results ──────────────────────────────────────────────────────
    save_json(per_sample_shifts, str(out_m2 / "per_sample_shifts.json"))
    save_json(aggregate_stats,   str(out_m2 / "aggregate_stats.json"))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Method 2 complete.")
    if aggregate_stats:
        # Show layer with biggest SSS-SSU cosine difference
        def _diff(r):
            a, b = r.get("SSS_mean_cosine"), r.get("SSU_mean_cosine")
            return abs(a - b) if (a is not None and b is not None) else 0.0
        best = max(aggregate_stats, key=_diff)
        print(f"  Largest SSS/SSU cosine split at layer {best['layer']}:")
        print(f"    SSS mean cosine = {best['SSS_mean_cosine']:.4f}")
        print(f"    SSU mean cosine = {best['SSU_mean_cosine']:.4f}")
        print(f"    p-value         = {best['t_test_p_value_cosine']:.4e}")
    print(f"  Results → {out_m2}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
