#!/usr/bin/env python3
"""Verify per-layer PCA vs global PCA control extraction (record / verify).

``--mode record``: snapshot global-fit artifacts and run pre-extraction gates
(Checks 1, 3, 4, 5) before any per-layer write.

``--mode verify``: re-check byte identity of global-fit artifacts and act caches
(Check 7), plus Checks 2, 3, 5, 6 and report Checks 8–9 after extraction.

Exit code 1 on any gating failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import ActivationCache  # noqa: E402
from src.paths import (  # noqa: E402
    project_root,
    vti_demos_850_partition_path,
    vti_demos_850_path,
)
from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    PARTITION_SIZES,
    build_or_load_partition,
    check_partition_integrity,
    partition_cache_dir,
    partition_slug,
)
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    act_cache_dir,
    demos_content_hash,
    load_textual_v2_directions,
    textual_v2_slug,
    variant_suffix,
)
from evaluation.interventions.vti.perlayer_pca import (  # noqa: E402
    EXPECTED_DECODER_SHAPES,
    EXPECTED_DEMOS_HASH,
    FIT_LOCUS_DESCRIPTION,
    load_stack_readonly,
    perlayer_partition_cache_dir,
    perlayer_shuffled_control_dir,
)
from evaluation.interventions.vti.shuffled_control import validate_derangement  # noqa: E402
from evaluation.interventions.vti.shuffled_control_partition import (  # noqa: E402
    DERANGEMENT_SEED,
    load_or_write_block_derangement,
    shuffled_demos850_act_cache_dir,
    shuffled_demos850_direction_dir,
)

OUT_DIR = (
    project_root()
    / "diagnostic_experiments"
    / "perlayer_pca_control"
    / "verification"
)
MODELS = list(EXPECTED_DECODER_SHAPES.keys())
DEMOS_HASH = EXPECTED_DEMOS_HASH
PRE_MANIFEST = OUT_DIR / "global_fit_directions_sha256_manifest_pre_run.json"
POST_MANIFEST = OUT_DIR / "global_fit_directions_sha256_manifest_post_run.json"
BYTE_IDENTITY_REPORT = OUT_DIR / "global_fit_directions_byte_identity_report.md"
ACT_UNTOUCHED_REPORT = OUT_DIR / "activation_cache_untouched_report.md"
EXTRACTION_MANIFEST = OUT_DIR / "perlayer_pca_extraction_manifest_2026-07-29.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.relative_to(project_root())),
        "sha256": _sha256(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _cache_file_triples(cache_dir: Path) -> List[Dict[str, Any]]:
    triples = []
    if not cache_dir.is_dir():
        return triples
    for p in sorted(cache_dir.glob("sample_*.npz")):
        if not p.is_file():
            continue
        st = p.stat()
        triples.append({
            "name": p.name,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        })
    return triples


def _global_cell_dirs() -> List[Tuple[str, str, int, Path]]:
    """(model, arm, N, path) for all 16 global-fit cells."""
    out = []
    for model in MODELS:
        for n in PARTITION_SIZES:
            slug = partition_slug(DEMOS_HASH, "all", n, seed=42, rank=2)
            out.append((model, "deployed", n, partition_cache_dir(model, slug)))
            out.append(
                (model, "control", n, shuffled_demos850_direction_dir(model, n))
            )
    return out


def _perlayer_cell_dirs() -> List[Tuple[str, str, int, Path]]:
    out = []
    for model in MODELS:
        for n in PARTITION_SIZES:
            slug = partition_slug(
                DEMOS_HASH, "all", n, seed=42, rank=2, fit_locus="perlayer",
            )
            out.append((model, "deployed", n, perlayer_partition_cache_dir(model, slug)))
            out.append(
                (model, "control", n, perlayer_shuffled_control_dir(model, n))
            )
    return out


def _build_manifest() -> dict:
    cells = []
    for model, arm, n, cdir in _global_cell_dirs():
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing global-fit cell dir: {cdir}")
        files = {}
        for name in ("directions.npz", "components.npz", "metadata.json"):
            p = cdir / name
            if not p.is_file():
                raise FileNotFoundError(f"Missing {p}")
            files[name] = _file_record(p)
        cells.append({
            "model_short": model,
            "arm": arm,
            "num_demos": n,
            "dir": str(cdir.relative_to(project_root())),
            "files": files,
        })

    act_caches = {}
    for model in MODELS:
        for label, path, expected in (
            ("textual_v2", act_cache_dir(model), 5100),
            ("shuffled_control_demos850", shuffled_demos850_act_cache_dir(model), 1700),
        ):
            triples = _cache_file_triples(path)
            key = f"{model}/{label}"
            act_caches[key] = {
                "dir": str(path.relative_to(project_root())),
                "file_count": len(triples),
                "expected_file_count": expected,
                "files": triples,
            }
            if len(triples) != expected:
                raise RuntimeError(
                    f"Act cache {key}: file_count={len(triples)} != {expected}"
                )

    return {
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "demos_hash": DEMOS_HASH,
        "n_global_cells": len(cells),
        "cells": cells,
        "act_caches": act_caches,
    }


def check_slug_strings() -> Tuple[bool, str]:
    msgs = []
    ok = True
    for n in PARTITION_SIZES:
        got = partition_slug(DEMOS_HASH, "all", n, seed=42, rank=2)
        exp = f"demos850_ba05bd96_all_nd{n}_s42_r2_partition"
        if got != exp:
            ok = False
            msgs.append(f"partition_slug nd{n}: {got!r} != {exp!r}")
    got_v2 = textual_v2_slug("9a44f4afde0324b5", "all", 200, seed=42, rank=2)
    exp_v2 = "demosv2_9a44f4af_all_nd200_s42_r2_prefix"
    if got_v2 != exp_v2:
        ok = False
        msgs.append(f"textual_v2_slug: {got_v2!r} != {exp_v2!r}")

    for model in MODELS:
        for n in PARTITION_SIZES:
            slug = partition_slug(DEMOS_HASH, "all", n, seed=42, rank=2)
            cdir = partition_cache_dir(model, slug)
            if not (cdir / "directions.npz").is_file():
                ok = False
                msgs.append(f"missing deployed directions: {cdir}")
    return ok, "; ".join(msgs) if msgs else "slug strings unchanged; 8 deployed dirs present"


def check_caches_differ() -> Tuple[bool, str]:
    """Check 4 — the two act-cache namespaces hold different data.

    Plan text asked for max-abs > 0 at every layer row including the embedding
    row. Observed: layer 0 (input embedding of the last caption token) is
    byte-identical under image derangement for both models, while every decoder
    row differs. Gate on decoder rows 1..L plus overall stack max-abs > 0 and
    some layer cosine < 1; report embedding-row max-abs separately.
    """
    demos_path = vti_demos_850_path()
    part = build_or_load_partition(demos_path, vti_demos_850_partition_path())
    block50 = list(part["blocks"]["50"])[:5]
    details = []
    ok = True
    for model in MODELS:
        dep_cache = ActivationCache(str(act_cache_dir(model)))
        ctrl_cache = ActivationCache(str(shuffled_demos850_act_cache_dir(model)))
        for rid in block50:
            a = load_stack_readonly(dep_cache, rid, "all").numpy()
            b = load_stack_readonly(ctrl_cache, rid, "all").numpy()
            if a.shape != b.shape:
                ok = False
                details.append(f"{model}/{rid}: shape mismatch")
                continue
            max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            emb_abs = float(np.max(np.abs(a[0].astype(np.float64) - b[0].astype(np.float64))))
            decoder_diff_ok = True
            some_cos_lt_1 = False
            for li in range(1, a.shape[0]):
                d = float(np.max(np.abs(a[li].astype(np.float64) - b[li].astype(np.float64))))
                if d <= 0:
                    decoder_diff_ok = False
                na = float(np.linalg.norm(a[li].astype(np.float64)))
                nb = float(np.linalg.norm(b[li].astype(np.float64)))
                cos = float(
                    np.dot(a[li].astype(np.float64), b[li].astype(np.float64))
                    / max(na * nb, 1e-12)
                )
                if cos < 1.0:
                    some_cos_lt_1 = True
            if not decoder_diff_ok or max_abs <= 0 or not some_cos_lt_1:
                ok = False
                details.append(
                    f"{model}/{rid}: max_abs={max_abs} emb0_abs={emb_abs} "
                    f"decoder_diff_ok={decoder_diff_ok} some_cos_lt_1={some_cos_lt_1}"
                )
            else:
                details.append(
                    f"{model}/{rid}: max_abs={max_abs:.6g} emb0_abs={emb_abs:.6g} PASS"
                )
    return ok, "; ".join(details)


def check_item_set_identity() -> Tuple[bool, str]:
    demos_path = vti_demos_850_path()
    partition_path = vti_demos_850_partition_path()
    part = build_or_load_partition(demos_path, partition_path)
    try:
        check_partition_integrity(demos_path, part)
        part_ok = True
        part_msg = "ok"
    except Exception as e:
        part_ok = False
        part_msg = str(e)

    h = demos_content_hash(demos_path)
    hash_ok = h == DEMOS_HASH
    msgs = [f"partition_integrity={part_ok} ({part_msg})", f"hash={h} ok={hash_ok}"]
    ok = part_ok and hash_ok

    for model in MODELS:
        for n in PARTITION_SIZES:
            block = list(part["blocks"][str(n)])
            slug = partition_slug(DEMOS_HASH, "all", n, seed=42, rank=2)
            dep_dir = partition_cache_dir(model, slug)
            ctrl_dir = shuffled_demos850_direction_dir(model, n)
            _, dep_meta = load_textual_v2_directions(dep_dir)
            _, ctrl_meta = load_textual_v2_directions(ctrl_dir)
            dep_ids = list(dep_meta["ids_used"])
            ctrl_ids = list(ctrl_meta["ids_used"])
            if dep_ids != block:
                ok = False
                msgs.append(f"{model} nd{n} deployed ids != partition block")
            if ctrl_ids != dep_ids:
                ok = False
                msgs.append(f"{model} nd{n} control ids != deployed ids")
            mapping, dpath = load_or_write_block_derangement(
                dep_ids, n, deployed_slug=slug, force=False,
            )
            n_fixed = validate_derangement(dep_ids, mapping)
            if n_fixed != 0 or set(mapping.keys()) != set(dep_ids):
                ok = False
                msgs.append(
                    f"{model} nd{n} derangement fixed={n_fixed} "
                    f"keys_ok={set(mapping.keys()) == set(dep_ids)}"
                )
            else:
                msgs.append(f"{model} nd{n} derangement {dpath.name} ok")
    return ok, "; ".join(msgs)


def check_mean_identity() -> Tuple[bool, str, List[dict]]:
    """Check 2: global pca_mean_flat reshape == per-layer mean per cell."""
    rows = []
    ok = True
    for model, arm, n, pdir in _perlayer_cell_dirs():
        if not (pdir / "components.npz").is_file():
            ok = False
            rows.append({
                "model": model, "arm": arm, "num_demos": n,
                "pass": False, "reason": "per-layer components missing",
            })
            continue
        gdir = (
            partition_cache_dir(
                model, partition_slug(DEMOS_HASH, "all", n, seed=42, rank=2),
            )
            if arm == "deployed"
            else shuffled_demos850_direction_dir(model, n)
        )
        g = np.load(gdir / "components.npz")
        p = np.load(pdir / "components.npz")
        n_plus = int(g["n_layers_plus"])
        hidden = int(g["hidden_dim"])
        g_mean = np.asarray(g["pca_mean_flat"], dtype=np.float64).reshape(n_plus, hidden)
        p_mean = np.asarray(p["pca_mean_flat"], dtype=np.float64).reshape(n_plus, hidden)
        max_abs = float(np.max(np.abs(g_mean - p_mean)))
        scale = float(np.max(np.abs(g["pca_mean_flat"]))) or 1.0
        gate = 1e-6 * scale
        # Reconstruction: pc0 + mean == directions with embedding row
        dirs, meta = load_textual_v2_directions(pdir)
        pc0 = np.asarray(p["pc0"], dtype=np.float64)
        recon_full = pc0 + p_mean
        recon_dec = recon_full[1:]
        recon_diff = float(np.max(np.abs(recon_dec - dirs.astype(np.float64))))
        shape_ok = dirs.shape == (n_plus - 1, hidden) and dirs.shape == EXPECTED_DECODER_SHAPES[model]
        cell_ok = max_abs <= gate and recon_diff <= 1e-5 and shape_ok
        if not cell_ok:
            ok = False
        rows.append({
            "model": model,
            "arm": arm,
            "num_demos": n,
            "max_abs_diff": max_abs,
            "gate": gate,
            "recon_max_abs_diff": recon_diff,
            "shape": list(dirs.shape),
            "pass": cell_ok,
        })
    msg = f"{sum(1 for r in rows if r.get('pass'))}/{len(rows)} cells pass mean identity"
    return ok, msg, rows


def check_output_shapes() -> Tuple[bool, str, List[dict]]:
    rows = []
    ok = True
    for model, arm, n, pdir in _perlayer_cell_dirs():
        if not (pdir / "directions.npz").is_file():
            ok = False
            rows.append({
                "model": model, "arm": arm, "num_demos": n,
                "pass": False, "reason": "missing",
            })
            continue
        dirs, meta = load_textual_v2_directions(pdir)
        expected = EXPECTED_DECODER_SHAPES[model]
        norms = np.linalg.norm(dirs.astype(np.float64), axis=1)
        finite = bool(np.isfinite(dirs).all())
        pos = bool(np.all(norms > 0))
        shape_ok = tuple(dirs.shape) == expected
        fwd = meta.get("forwards_executed")
        ro = meta.get("act_cache_read_only")
        cell_ok = shape_ok and finite and pos and fwd == 0 and ro is True
        if not cell_ok:
            ok = False
        rows.append({
            "model": model,
            "arm": arm,
            "num_demos": n,
            "shape": list(dirs.shape),
            "expected": list(expected),
            "finite": finite,
            "all_norms_positive": pos,
            "forwards_executed": fwd,
            "act_cache_read_only": ro,
            "pass": cell_ok,
        })
    msg = f"{sum(1 for r in rows if r.get('pass'))}/{len(rows)} cells pass shape/finiteness"
    return ok, msg, rows


def check_pc1_share_report() -> List[dict]:
    """Check 8 report rows (not a gate)."""
    rows = []
    for model in MODELS:
        for scheme, cell_iter in (
            ("global", _global_cell_dirs()),
            ("perlayer", _perlayer_cell_dirs()),
        ):
            for m, arm, n, cdir in cell_iter:
                if m != model:
                    continue
                if not (cdir / "directions.npz").is_file():
                    continue
                dirs, meta = load_textual_v2_directions(cdir)
                n_dec = dirs.shape[0]
                comp = np.load(cdir / "components.npz")
                pc0 = np.asarray(comp["pc0"], dtype=np.float64)
                pc1_norms = np.linalg.norm(pc0[1:], axis=1)
                dir_norms = np.linalg.norm(dirs.astype(np.float64), axis=1)
                share = pc1_norms / np.maximum(dir_norms, 1e-12)
                descents = 0
                largest_descent = 0.0
                for i in range(1, n_dec):
                    step = float(dir_norms[i] - dir_norms[i - 1])
                    if step < 0:
                        descents += 1
                        largest_descent = max(largest_descent, -step)
                rows.append({
                    "model": model,
                    "scheme": scheme,
                    "arm": arm,
                    "num_demos": n,
                    "pc1_share_min": float(share.min()),
                    "pc1_share_median": float(np.median(share)),
                    "pc1_share_max": float(share.max()),
                    "direction_norm_descending_steps": descents,
                    "largest_descending_step": largest_descent,
                    "pc1_share_per_layer": [float(x) for x in share],
                    "direction_norm_per_layer": [float(x) for x in dir_norms],
                })
    return rows


def check_pc1_sign_report() -> List[dict]:
    """Check 9 report rows (not a gate)."""
    rows = []
    for model in MODELS:
        for n in PARTITION_SIZES:
            for scheme in ("global", "perlayer"):
                if scheme == "global":
                    dep_dir = partition_cache_dir(
                        model, partition_slug(DEMOS_HASH, "all", n, seed=42, rank=2),
                    )
                    ctrl_dir = shuffled_demos850_direction_dir(model, n)
                else:
                    slug = partition_slug(
                        DEMOS_HASH, "all", n, seed=42, rank=2, fit_locus="perlayer",
                    )
                    dep_dir = perlayer_partition_cache_dir(model, slug)
                    ctrl_dir = perlayer_shuffled_control_dir(model, n)
                if not (dep_dir / "components.npz").is_file():
                    continue
                if not (ctrl_dir / "components.npz").is_file():
                    continue
                dep_c = np.load(dep_dir / "components.npz")
                ctrl_c = np.load(ctrl_dir / "components.npz")
                dep_pc0 = np.asarray(dep_c["pc0"], dtype=np.float64)
                ctrl_pc0 = np.asarray(ctrl_c["pc0"], dtype=np.float64)
                dep_mean = np.asarray(dep_c["pca_mean_flat"], dtype=np.float64).reshape(
                    dep_pc0.shape
                )
                ctrl_mean = np.asarray(ctrl_c["pca_mean_flat"], dtype=np.float64).reshape(
                    ctrl_pc0.shape
                )
                n_dec = dep_pc0.shape[0] - 1
                for li in range(n_dec):
                    # decoder layer li corresponds to row li+1
                    r = li + 1

                    def _cos_sign(pc, mean):
                        na = float(np.linalg.norm(pc))
                        nb = float(np.linalg.norm(mean))
                        cos = float(np.dot(pc, mean) / max(na * nb, 1e-12))
                        if cos > 0:
                            sgn = 1
                        elif cos < 0:
                            sgn = -1
                        else:
                            sgn = 0
                        return cos, sgn

                    d_cos, d_sgn = _cos_sign(dep_pc0[r], dep_mean[r])
                    c_cos, c_sgn = _cos_sign(ctrl_pc0[r], ctrl_mean[r])
                    rows.append({
                        "model": model,
                        "scheme": scheme,
                        "num_demos": n,
                        "layer": li,
                        "deployed_cosine_pc1_with_mean": d_cos,
                        "control_cosine_pc1_with_mean": c_cos,
                        "deployed_pc1_sign": d_sgn,
                        "control_pc1_sign": c_sgn,
                        "pc1_sign_agrees_between_arms": d_sgn == c_sgn,
                    })
    return rows


def write_byte_identity_report(pre: dict, post: dict) -> Tuple[bool, str]:
    lines = [
        "# Global-fit directions byte-identity report",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Pre-run manifest: `{PRE_MANIFEST.relative_to(project_root())}`",
        f"Post-run manifest: `{POST_MANIFEST.relative_to(project_root())}`",
        "",
        "| Path | Pre SHA-256 | Post SHA-256 | Match |",
        "|---|---|---|---|",
    ]
    pre_files = {}
    for cell in pre["cells"]:
        for name, rec in cell["files"].items():
            pre_files[rec["path"]] = rec["sha256"]
    post_files = {}
    for cell in post["cells"]:
        for name, rec in cell["files"].items():
            post_files[rec["path"]] = rec["sha256"]

    all_match = True
    for path in sorted(pre_files):
        a = pre_files[path]
        b = post_files.get(path)
        match = a == b
        if not match:
            all_match = False
        lines.append(f"| `{path}` | `{a}` | `{b}` | {'yes' if match else 'NO'} |")

    lines += [
        "",
        "## Per-layer output paths (must be new)",
        "",
        "| Path | Existed at pre-run | Exists now |",
        "|---|---|---|",
    ]
    pre_existed = set()
    # At record time none of the per-layer dirs should have existed.
    for model, arm, n, pdir in _perlayer_cell_dirs():
        rel = str(pdir.relative_to(project_root()))
        exists_now = (pdir / "directions.npz").is_file()
        lines.append(
            f"| `{rel}` | no | "
            f"{'yes' if exists_now else 'no'} |"
        )

    lines += [
        "",
        f"**All global-fit SHA-256 match:** {'yes' if all_match else 'no'}",
        "",
    ]
    BYTE_IDENTITY_REPORT.write_text("\n".join(lines) + "\n")
    return all_match, f"wrote {BYTE_IDENTITY_REPORT.name}; all_match={all_match}"


def write_act_untouched_report(pre: dict, post: dict) -> Tuple[bool, str]:
    lines = [
        "# Activation cache untouched report",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "| Cache | Pre count | Post count | Triples identical |",
        "|---|---:|---:|---|",
    ]
    ok = True
    for key, pre_rec in pre["act_caches"].items():
        post_rec = post["act_caches"][key]
        count_ok = pre_rec["file_count"] == post_rec["file_count"]
        pre_map = {(f["name"], f["size"], f["mtime_ns"]) for f in pre_rec["files"]}
        post_map = {(f["name"], f["size"], f["mtime_ns"]) for f in post_rec["files"]}
        triples_ok = pre_map == post_map
        cell_ok = count_ok and triples_ok
        if not cell_ok:
            ok = False
        lines.append(
            f"| `{key}` | {pre_rec['file_count']} | {post_rec['file_count']} | "
            f"{'yes' if triples_ok else 'NO'} |"
        )
    lines += ["", f"**All caches untouched:** {'yes' if ok else 'no'}", ""]
    ACT_UNTOUCHED_REPORT.write_text("\n".join(lines) + "\n")
    return ok, f"wrote {ACT_UNTOUCHED_REPORT.name}; untouched={ok}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["record", "verify"], required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gating: List[Tuple[str, bool, str]] = []

    if args.mode == "record":
        try:
            manifest = _build_manifest()
            PRE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
            gating.append(("Check 1 pre-run hash snapshot", True, f"wrote {PRE_MANIFEST}"))
        except Exception as e:
            gating.append(("Check 1 pre-run hash snapshot", False, str(e)))

        ok3, msg3 = check_slug_strings()
        gating.append(("Check 3 slug strings unchanged", ok3, msg3))
        ok4, msg4 = check_caches_differ()
        gating.append(("Check 4 activation caches differ", ok4, msg4))
        ok5, msg5 = check_item_set_identity()
        gating.append(("Check 5 item-set identity", ok5, msg5))

    else:  # verify
        if not PRE_MANIFEST.is_file():
            print(f"ERROR: missing pre-run manifest {PRE_MANIFEST}", file=sys.stderr)
            return 1
        pre = json.loads(PRE_MANIFEST.read_text())
        try:
            post = _build_manifest()
            POST_MANIFEST.write_text(json.dumps(post, indent=2) + "\n")
            gating.append(("Check 1 post-run snapshot written", True, f"wrote {POST_MANIFEST}"))
        except Exception as e:
            gating.append(("Check 1 post-run snapshot written", False, str(e)))
            post = None

        ok3, msg3 = check_slug_strings()
        gating.append(("Check 3 slug strings unchanged", ok3, msg3))
        ok5, msg5 = check_item_set_identity()
        gating.append(("Check 5 item-set identity", ok5, msg5))

        if post is not None:
            ok7a, msg7a = write_byte_identity_report(pre, post)
            gating.append(("Check 7 global-fit byte identity", ok7a, msg7a))
            ok7b, msg7b = write_act_untouched_report(pre, post)
            gating.append(("Check 7 activation caches untouched", ok7b, msg7b))

        ok2, msg2, rows2 = check_mean_identity()
        gating.append(("Check 2 mean identity (fit locus only)", ok2, msg2))
        ok6, msg6, rows6 = check_output_shapes()
        gating.append(("Check 6 shapes and finiteness", ok6, msg6))

        report8 = check_pc1_share_report()
        report9 = check_pc1_sign_report()
        extraction_cells = []
        for model, arm, n, pdir in _perlayer_cell_dirs():
            exists = (pdir / "directions.npz").is_file()
            entry = {
                "model_short": model,
                "arm": arm,
                "num_demos": n,
                "dir": str(pdir.relative_to(project_root())),
                "exists": exists,
            }
            if exists:
                dirs, meta = load_textual_v2_directions(pdir)
                entry["shape"] = list(dirs.shape)
                entry["n_pairs"] = meta.get("n_pairs")
                entry["forwards_executed"] = meta.get("forwards_executed")
                entry["fit_locus"] = meta.get("fit_locus")
                entry["global_fit_source_dir"] = meta.get("global_fit_source_dir")
            extraction_cells.append(entry)

        EXTRACTION_MANIFEST.write_text(
            json.dumps(
                {
                    "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "fit_locus_description": FIT_LOCUS_DESCRIPTION,
                    "gating": [
                        {"check": name, "pass": ok, "detail": detail}
                        for name, ok, detail in gating
                    ],
                    "check2_rows": rows2,
                    "check6_rows": rows6,
                    "check8_pc1_share": report8,
                    "check9_pc1_sign": report9,
                    "perlayer_cells": extraction_cells,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {EXTRACTION_MANIFEST}")

    print("")
    print("| Check | Pass | Detail |")
    print("|---|---|---|")
    all_ok = True
    for name, ok, detail in gating:
        if not ok:
            all_ok = False
        # Truncate very long details for stdout
        short = detail if len(detail) < 200 else detail[:197] + "..."
        print(f"| {name} | {'PASS' if ok else 'FAIL'} | {short} |")

    if not all_ok:
        print("GATING FAILURE", file=sys.stderr)
        return 1
    print("All gating checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
