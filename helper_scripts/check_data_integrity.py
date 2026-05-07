#!/usr/bin/env python3
"""
Data integrity check for VLM_Safety experiments.
Run this before method2_activation_shift.py to verify all required data is present.

How it works:
  - Reads holisafe_bench.json and checks whether each sample's "image" field
    (e.g. "self_harm/suicide/suicide_8.jpg") exists under data/holisafe-bench/images/.
  - Checks MM-SafetyBench and LLaVA-Instruct cached reference sets similarly.
  - Reports counts of existing activation cache files.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
HOLISAFE_DIR = DATA_DIR / "holisafe-bench"
MMSAFETY_DIR = DATA_DIR / "mm-safetybench-ref"
LLAVA_DIR    = DATA_DIR / "llava-instruct-ref"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

issues = []

def check(ok, pass_msg, fail_msg):
    if ok:
        print(f"  {PASS} {pass_msg}")
    else:
        print(f"  {FAIL} {fail_msg}")
        issues.append(fail_msg)

def warn(msg):
    print(f"  {WARN} {msg}")


# ── 1. HoliSafe-Bench ────────────────────────────────────────────────────────
print("\n[1] HoliSafe-Bench")

json_path = HOLISAFE_DIR / "holisafe_bench.json"
check(json_path.exists(),
      f"holisafe_bench.json found",
      f"holisafe_bench.json missing: {json_path}")

if json_path.exists():
    with open(json_path) as f:
        data = json.load(f)
    print(f"  {PASS} {len(data)} total entries")

    sss_total = ssu_total = 0
    sss_ok    = ssu_ok    = 0
    missing_by_cat: dict[str, int] = {}

    for entry in data:
        t = entry.get("type", "")
        if t not in ("SSS", "SSU"):
            continue
        img    = entry.get("image", "")
        path1  = HOLISAFE_DIR / "images" / img
        path2  = HOLISAFE_DIR / img
        exists = path1.exists() or path2.exists()
        cat    = entry.get("category", "unknown")

        if t == "SSS":
            sss_total += 1
            if exists: sss_ok += 1
        else:
            ssu_total += 1
            if exists: ssu_ok += 1

        if not exists:
            missing_by_cat[cat] = missing_by_cat.get(cat, 0) + 1

    sss_pct = 100 * sss_ok / sss_total if sss_total else 0
    ssu_pct = 100 * ssu_ok / ssu_total if ssu_total else 0

    check(sss_ok == sss_total,
          f"SSS images complete: {sss_ok}/{sss_total}",
          f"SSS images incomplete: {sss_ok}/{sss_total} present ({sss_pct:.1f}%)")
    check(ssu_ok == ssu_total,
          f"SSU images complete: {ssu_ok}/{ssu_total}",
          f"SSU images incomplete: {ssu_ok}/{ssu_total} present ({ssu_pct:.1f}%)")

    if missing_by_cat:
        print(f"  Missing images by category:")
        for cat, cnt in sorted(missing_by_cat.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {cnt}")
        total_missing = sum(missing_by_cat.values())
        warn(f"{total_missing} images missing — affected samples will be skipped during VL pass")


# ── 2. MM-SafetyBench reference ───────────────────────────────────────────────
print("\n[2] MM-SafetyBench reference dataset")

mm_json = MMSAFETY_DIR / "samples_n160_seed42.json"
if mm_json.exists():
    with open(mm_json) as f:
        mm_samples = json.load(f)
    print(f"  {PASS} metadata JSON found — {len(mm_samples)} samples")
    mm_img_dir = MMSAFETY_DIR / "images"
    mm_img_ok  = sum(1 for s in mm_samples
                     if (mm_img_dir / s.get("image_filename", "")).exists())
    check(mm_img_ok == len(mm_samples),
          f"Images complete: {mm_img_ok}/{len(mm_samples)}",
          f"Images incomplete: {mm_img_ok}/{len(mm_samples)} present")
else:
    warn("MM-SafetyBench cache not found — will be downloaded automatically on first run")


# ── 3. LLaVA-Instruct reference ───────────────────────────────────────────────
print("\n[3] LLaVA-Instruct-80k reference dataset")

llava_json = LLAVA_DIR / "samples_n160_seed42.json"
if llava_json.exists():
    with open(llava_json) as f:
        llava_samples = json.load(f)
    print(f"  {PASS} metadata JSON found — {len(llava_samples)} samples")
    llava_img_dir = LLAVA_DIR / "images"
    llava_img_ok  = sum(1 for s in llava_samples
                        if (llava_img_dir / s.get("image_filename", "")).exists())
    check(llava_img_ok == len(llava_samples),
          f"Images complete: {llava_img_ok}/{len(llava_samples)}",
          f"Images incomplete: {llava_img_ok}/{len(llava_samples)} present")
else:
    warn("LLaVA-Instruct cache not found — will be downloaded automatically on first run")


# ── 4. Activation cache ───────────────────────────────────────────────────────
print("\n[4] Activation cache")

act_dir = (PROJECT_ROOT / "data" / "holisafe-bench" / "llava-1.5-7b-hf"
           / "activations")
if act_dir.exists():
    vl_files = list(act_dir.glob("*_vl.npz"))
    tt_files = list(act_dir.glob("*_tt.npz"))
    print(f"  Cached VL activations : {len(vl_files)} files")
    print(f"  Cached TT activations : {len(tt_files)} files")
    if len(vl_files) != len(tt_files):
        warn(f"VL ({len(vl_files)}) and TT ({len(tt_files)}) counts differ — "
             f"some samples may have incomplete activation pairs")
else:
    print(f"  Activation cache directory does not exist (method2 not yet run)")


# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if issues:
    print(f"  {len(issues)} issue(s) found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n  Recommendation: download the complete dataset before running experiments.")
    sys.exit(1)
else:
    print("  All checks passed. Ready to run experiments.")
    sys.exit(0)
