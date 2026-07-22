#!/usr/bin/env python3
"""Steered capture dump: last-prefill acts/norms + first-token yes/no scores + response.

Dumps land under ``data/{amber|pope}/dumps/{model_short}/{run_tag}/``.

Example::

    CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/perception_diag/run_dump.py \\
      --model llava-hf/llava-1.5-7b-hf \\
      --augmented_jsonl data/amber/augmented_amber25.jsonl \\
      --cells all \\
      --run_tag amber25_all_steering_settings

Windowed steering grid (layer bands × configs × betas)::

    CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/perception_diag/run_dump.py \\
      --model llava-hf/llava-1.5-7b-hf \\
      --augmented_jsonl data/pope/augmented_pope30.jsonl \\
      --windowed_grid --conditions gold_conditional \\
      --run_tag pope30_windowed_steering
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture.dump_writer import DumpWriter  # noqa: E402
from capture.steered_capture import (  # noqa: E402
    run_capture_generation,
    verify_steered_capture_equality,
)
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    compute_or_load_textual_directions_v2,
    demos_content_hash,
    textual_v2_slug,
)
from src.model import _normalize_model_name, create_wrapper  # noqa: E402
from src.paths import (  # noqa: E402
    infer_benchmark_from_run_tag,
    perception_dump_dir,
    project_root,
    vti_demos_v2_path,
)
from src.prompt_spans import layer_windows  # noqa: E402

CELL_SPECS = {
    "baseline": {"method": "none", "site": None, "strength": 0.0, "readout": "full"},
    "rotation_mlp_0.2": {
        "method": "uniform_rotation", "site": "mlp", "strength": 0.2, "readout": "full",
    },
    "rotation_mlp_0.5": {
        "method": "uniform_rotation", "site": "mlp", "strength": 0.5, "readout": "full",
    },
    "rotation_mlp_0.9": {
        "method": "uniform_rotation", "site": "mlp", "strength": 0.9, "readout": "full",
    },
    "rotation_layer_0.2": {
        "method": "uniform_rotation", "site": "layer", "strength": 0.2, "readout": "full",
    },
    "rotation_layer_0.5": {
        "method": "uniform_rotation", "site": "layer", "strength": 0.5, "readout": "full",
    },
    "rotation_layer_0.9": {
        "method": "uniform_rotation", "site": "layer", "strength": 0.9, "readout": "full",
    },
    "additive_mlp_0.2": {
        "method": "additive", "site": "mlp", "strength": 0.2, "readout": "capture_only",
    },
    "additive_mlp_0.5": {
        "method": "additive", "site": "mlp", "strength": 0.5, "readout": "capture_only",
    },
    "additive_mlp_0.9": {
        "method": "additive", "site": "mlp", "strength": 0.9, "readout": "capture_only",
    },
    "additive_layer_0.2": {
        "method": "additive", "site": "layer", "strength": 0.2, "readout": "capture_only",
    },
    "additive_layer_0.5": {
        "method": "additive", "site": "layer", "strength": 0.5, "readout": "capture_only",
    },
    "additive_layer_0.9": {
        "method": "additive", "site": "layer", "strength": 0.9, "readout": "capture_only",
    },
}

CELL_ORDER = [
    "baseline",
    "rotation_mlp_0.2", "rotation_mlp_0.5", "rotation_mlp_0.9",
    "rotation_layer_0.2", "rotation_layer_0.5", "rotation_layer_0.9",
    "additive_mlp_0.2", "additive_mlp_0.5", "additive_mlp_0.9",
    "additive_layer_0.2", "additive_layer_0.5", "additive_layer_0.9",
]

# Qwen2.5 AMBER-25 dump subset (no additive_layer 0.2 / 0.9).
DEFAULT_SUBSET_CELLS = [
    "baseline",
    "rotation_mlp_0.2", "rotation_mlp_0.5", "rotation_mlp_0.9",
    "rotation_layer_0.2", "rotation_layer_0.5", "rotation_layer_0.9",
    "additive_mlp_0.2", "additive_mlp_0.5", "additive_mlp_0.9",
    "additive_layer_0.5",
]

# Windowed-grid configs: (cell_name_prefix, method, site)
WINDOWED_CONFIGS: List[Tuple[str, str, str]] = [
    ("additive_mlp", "additive", "mlp"),
    ("rotation_mlp", "uniform_rotation", "mlp"),
    ("rotation_layer", "uniform_rotation", "layer"),
]
WINDOWED_STRENGTHS = (0.2, 0.5, 0.9)


def _parse_cells(s: str) -> list[str]:
    key = s.strip().lower()
    if key == "all":
        return list(CELL_ORDER)
    if key in ("default_subset", "subset", "smoke"):
        return list(DEFAULT_SUBSET_CELLS)
    wanted = [c.strip() for c in s.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in CELL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown cells: {unknown}")
    return sorted(wanted, key=lambda c: CELL_ORDER.index(c))


def _load_augmented(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _gold_opposing_condition(gold: str) -> str:
    g = (gold or "").strip().lower()
    if g == "no":
        return "assertive_toward_yes"
    if g == "yes":
        return "assertive_toward_no"
    raise ValueError(f"Unexpected gold label for gold_conditional filter: {gold!r}")


def apply_gold_conditional(rows: list[dict]) -> list[dict]:
    """Keep neutral + gold-opposing assertive variant per item."""
    out: list[dict] = []
    n_yes = 0
    n_no = 0
    for item in rows:
        gold = item.get("gold")
        if gold == "yes":
            n_yes += 1
        elif gold == "no":
            n_no += 1
        opposing = _gold_opposing_condition(str(gold))
        keep = {"neutral", opposing}
        variants = [v for v in item["variants"] if v["condition_id"] in keep]
        found = {v["condition_id"] for v in variants}
        missing = keep - found
        if missing:
            raise SystemExit(
                f"Item {item.get('item_id')}: missing variants {sorted(missing)} "
                f"(have {[v['condition_id'] for v in item['variants']]})"
            )
        # Preserve neutral-first order
        ordered = []
        by_id = {v["condition_id"]: v for v in variants}
        for cid in ("neutral", opposing):
            ordered.append(by_id[cid])
        out.append({**item, "variants": ordered})
    # Cheap inline gold-split check (AMBER-100 expect 60 no / 40 yes; POPE-30: 0/30)
    print(
        f"[gold_conditional] filtered items={len(out)} gold_yes={n_yes} gold_no={n_no}",
        flush=True,
    )
    return out


def build_windowed_cell_specs(
    num_layers: int,
) -> Tuple[Dict[str, dict], List[str], dict]:
    """Baseline + 3 configs × 3 strengths × (layer_windows + all-layers).

    Returns (specs_dict, ordered_cell_ids, grid_metadata).
    ``baseline`` (method=none) is first so no-intervention is always in-grid.
    """
    windows = layer_windows(num_layers, width=10, stride=5)
    window_entries: List[Tuple[str, Optional[List[int]]]] = [
        (f"layers_{s}_{e}", list(range(s, e + 1))) for s, e in windows
    ]
    window_entries.append(("layers_all", None))  # None → all layers in vti_hook_ctx

    specs: Dict[str, dict] = {
        "baseline": {
            "method": "none",
            "site": None,
            "strength": 0.0,
            "readout": "full",
        },
    }
    order: List[str] = ["baseline"]
    for prefix, method, site in WINDOWED_CONFIGS:
        for strength in WINDOWED_STRENGTHS:
            for win_label, layer_idxs in window_entries:
                cell_id = f"{prefix}_{strength}_{win_label}"
                spec: Dict[str, Any] = {
                    "method": method,
                    "site": site,
                    "strength": float(strength),
                    "readout": "full",
                }
                if layer_idxs is not None:
                    spec["layer_indices"] = layer_idxs
                specs[cell_id] = spec
                order.append(cell_id)

    grid_meta = {
        "num_layers": num_layers,
        "windows_inclusive": [{"start": s, "end": e} for s, e in windows],
        "windows_plus_all": [w[0] for w in window_entries],
        "configs": [
            {"prefix": p, "method": m, "site": s} for p, m, s in WINDOWED_CONFIGS
        ],
        "strengths": list(WINDOWED_STRENGTHS),
        "includes_baseline": True,
        "n_cells": len(order),
        "cell_ids": order,
    }
    return specs, order, grid_meta


def _oom_or_error_record(wrapper, status: str) -> dict:
    n_layers = getattr(wrapper, "num_layers", 32)
    hdim = getattr(wrapper, "hidden_dim", 4096)
    return {
        "response": "",
        "parsed_outcome": "unparseable",
        "first_vs_parsed_agree": False,
        "degeneracy_flag": False,
        "truncated": False,
        "status": status,
        "scores": {
            "p_yes_raw": float("nan"),
            "p_no_raw": float("nan"),
            "answer_mass": float("nan"),
            "p_yes_norm": float("nan"),
            "yes_logit_sum": float("nan"),
            "no_logit_sum": float("nan"),
            "logit_margin_yes_minus_no": float("nan"),
            "first_token_pred": "unparseable",
            "prefill_seq_len": 0,
        },
        "last_token_acts_fp16": np.zeros((n_layers + 1, hdim), dtype=np.float16),
        "prefill_pos_norms_fp16": np.zeros((n_layers + 1, 1), dtype=np.float16),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--augmented_jsonl", required=True)
    p.add_argument("--cells", default="default_subset",
                   help="Comma list of cell ids, 'all', or 'default_subset'. "
                        "Ignored when --windowed_grid is set.")
    p.add_argument("--windowed_grid", action="store_true",
                   help="Build cell specs: baseline + 3 configs × 3 strengths × "
                        "(layer_windows + all-layers) after model load.")
    p.add_argument("--conditions", default="all",
                   choices=("all", "gold_conditional"),
                   help="'all' keeps every variant; 'gold_conditional' keeps "
                        "neutral + assertive opposing gold.")
    p.add_argument("--run_tag", default="dump",
                   help="Dump folder name; amber*/pope* prefix selects data/{amber|pope}/.")
    p.add_argument("--benchmark", default=None, choices=("amber", "pope"),
                   help="Override benchmark data dir (default: inferred from --run_tag).")
    p.add_argument("--num_demos", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Generation budget for captured responses.")
    p.add_argument("--max_pixels", type=int, default=1003520)
    p.add_argument("--eps_coeff", type=float, default=0.1)
    p.add_argument("--limit_items", type=int, default=None)
    p.add_argument("--verify_g1", action="store_true")
    p.add_argument("--device_map", default=None,
                   help="Pass 'cuda:0' to force single-GPU unsharded load.")
    args = p.parse_args()

    model_short = _normalize_model_name(args.model)
    rows = _load_augmented(Path(args.augmented_jsonl))
    if args.limit_items is not None:
        rows = rows[: args.limit_items]

    condition_filter = args.conditions
    if condition_filter == "gold_conditional":
        rows = apply_gold_conditional(rows)
        # Soft gold-split assert for AMBER-100 (60 no / 40 yes)
        n_yes = sum(1 for r in rows if r.get("gold") == "yes")
        n_no = sum(1 for r in rows if r.get("gold") == "no")
        if len(rows) == 100 and (n_no, n_yes) != (60, 40):
            warnings.warn(
                f"AMBER-100 gold split expected 60 no / 40 yes, got {n_no} no / {n_yes} yes",
                stacklevel=1,
            )

    aug_meta_path = (
        project_root() / "diagnostic_experiments" / "perception_diag"
        / "augment" / "outputs" / "augmentation_meta.json"
    )
    aug_meta = json.loads(aug_meta_path.read_text()) if aug_meta_path.exists() else {}

    demos_path = vti_demos_v2_path()
    demos_hash = demos_content_hash(demos_path)
    slug = textual_v2_slug(demos_hash, "all", args.num_demos)

    benchmark = args.benchmark or infer_benchmark_from_run_tag(args.run_tag)
    out_dir = perception_dump_dir(benchmark, model_short, args.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper_kwargs = {}
    if args.device_map:
        wrapper_kwargs["device_map"] = args.device_map
    if "qwen2" in args.model.lower():
        wrapper_kwargs["max_pixels"] = args.max_pixels
    wrapper = create_wrapper(args.model, **wrapper_kwargs).load()

    window_grid_meta = None
    if args.windowed_grid:
        cell_specs, cells, window_grid_meta = build_windowed_cell_specs(
            wrapper.num_layers
        )
        print(
            f"[windowed_grid] num_layers={wrapper.num_layers} n_cells={len(cells)}",
            flush=True,
        )
    else:
        cell_specs = CELL_SPECS
        cells = _parse_cells(args.cells)

    need_dirs = any(cell_specs[c]["method"] != "none" for c in cells)
    directions = None
    if need_dirs:
        directions = compute_or_load_textual_directions_v2(
            wrapper, model_short, dimension="all", num_demos=args.num_demos,
        )

    base_meta = {
        "model": args.model,
        "model_short": model_short,
        "run_tag": args.run_tag,
        "demos_path": str(demos_path),
        "demos_hash": demos_hash,
        "direction_slug": slug,
        "num_demos": args.num_demos,
        "augmentation_meta": aug_meta,
        "augmented_jsonl": str(args.augmented_jsonl),
        "max_new_tokens": args.max_new_tokens,
        "eps_coeff": args.eps_coeff,
        "n_items": len(rows),
        "condition_filter": condition_filter,
        "windowed_grid": bool(args.windowed_grid),
    }
    if window_grid_meta is not None:
        base_meta["window_grid"] = window_grid_meta
    (out_dir / "run_metadata.json").write_text(json.dumps(base_meta, indent=2) + "\n")

    if args.verify_g1 and directions is not None:
        probe_cell_id = next(c for c in cells if cell_specs[c]["method"] != "none")
        probe = dict(cell_specs[probe_cell_id])
        probe["eps_coeff"] = args.eps_coeff
        item0 = rows[0]
        prompt0 = next(v["prompt"] for v in item0["variants"] if v["condition_id"] == "neutral")
        img0 = Image.open(item0["image_path"]).convert("RGB")
        g1 = verify_steered_capture_equality(
            wrapper, img0, prompt0, directions, probe,
        )
        (out_dir / "g1_steered_capture_check.json").write_text(json.dumps(g1, indent=2) + "\n")
        print(f"G1 steered-capture check: {g1}")
        if not g1.get("pass"):
            print("WARNING: G1 failed — continuing dump but flag for review")

    for cell_id in cells:
        spec = dict(cell_specs[cell_id])
        spec["eps_coeff"] = args.eps_coeff
        cell_meta = {**base_meta, "cell_id": cell_id, "cell": spec}
        writer = DumpWriter(out_dir, cell_id, cell_meta)
        n_skip = 0
        n_run = 0
        for item in tqdm(rows, desc=f"{cell_id}"):
            img = Image.open(item["image_path"]).convert("RGB")
            for var in item["variants"]:
                if writer.is_done(item["item_id"], var["condition_id"]):
                    n_skip += 1
                    continue
                cell_arg = None if spec["method"] == "none" else spec
                try:
                    rec = run_capture_generation(
                        wrapper, img, var["prompt"],
                        directions=directions,
                        cell=cell_arg,
                        max_new_tokens=args.max_new_tokens,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rec = _oom_or_error_record(wrapper, "oom")
                    print(
                        f"OOM {cell_id} {item['item_id']} {var['condition_id']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 — per-item status column
                    rec = _oom_or_error_record(wrapper, "error")
                    print(
                        f"ERROR {cell_id} {item['item_id']} "
                        f"{var['condition_id']}: {exc}",
                        flush=True,
                    )
                writer.write(
                    item_id=item["item_id"],
                    condition_id=var["condition_id"],
                    template_id=var["template_id"],
                    gold=item.get("gold"),
                    record=rec,
                    extra={"qtype": item.get("qtype"), "cell_id": cell_id},
                )
                n_run += 1
        print(f"[{cell_id}] ran={n_run} skipped_resume={n_skip}")

    wrapper.cleanup()
    print(f"Done. Dumps at {out_dir}")


if __name__ == "__main__":
    main()
