#!/usr/bin/env python3
"""Steered capture dump: last-prefill acts/norms + first-token yes/no scores + response.

Example::

    CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/perception_diag/run_dump.py \\
      --model llava-hf/llava-1.5-7b-hf \\
      --augmented_jsonl diagnostic_experiments/perception_diag/augment/outputs/augmented_amber25.jsonl \\
      --cells all \\
      --run_tag amber25_all_steering_settings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from src.paths import project_root, vti_demos_v2_path  # noqa: E402

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--augmented_jsonl", required=True)
    p.add_argument("--cells", default="default_subset",
                   help="Comma list of cell ids, 'all', or 'default_subset'.")
    p.add_argument("--run_tag", default="dump")
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
    cells = _parse_cells(args.cells)
    rows = _load_augmented(Path(args.augmented_jsonl))
    if args.limit_items is not None:
        rows = rows[: args.limit_items]

    aug_meta_path = (
        project_root() / "diagnostic_experiments" / "perception_diag"
        / "augment" / "outputs" / "augmentation_meta.json"
    )
    aug_meta = json.loads(aug_meta_path.read_text()) if aug_meta_path.exists() else {}

    demos_path = vti_demos_v2_path()
    demos_hash = demos_content_hash(demos_path)
    slug = textual_v2_slug(demos_hash, "all", args.num_demos)

    out_dir = (
        project_root() / "diagnostic_experiments" / "perception_diag"
        / model_short / "dumps" / args.run_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper_kwargs = {}
    if args.device_map:
        wrapper_kwargs["device_map"] = args.device_map
    if "qwen2" in args.model.lower():
        wrapper_kwargs["max_pixels"] = args.max_pixels
    wrapper = create_wrapper(args.model, **wrapper_kwargs).load()

    need_dirs = any(CELL_SPECS[c]["method"] != "none" for c in cells)
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
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(base_meta, indent=2) + "\n")

    if args.verify_g1 and directions is not None:
        # Use first item/neutral prompt with first non-baseline cell that has a method
        probe_cell_id = next(c for c in cells if CELL_SPECS[c]["method"] != "none")
        probe = dict(CELL_SPECS[probe_cell_id])
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
        spec = dict(CELL_SPECS[cell_id])
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
                    # Minimal failed record so resume-by-id skips retries unless deleted.
                    n_layers = getattr(wrapper, "num_layers", 32)
                    hdim = getattr(wrapper, "hidden_dim", 4096)
                    rec = {
                        "response": "",
                        "parsed_outcome": "unparseable",
                        "first_vs_parsed_agree": False,
                        "degeneracy_flag": False,
                        "truncated": False,
                        "status": "oom",
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
                        "last_token_acts_fp16": np.zeros(
                            (n_layers + 1, hdim), dtype=np.float16
                        ),
                        "prefill_pos_norms_fp16": np.zeros(
                            (n_layers + 1, 1), dtype=np.float16
                        ),
                    }
                    print(
                        f"OOM {cell_id} {item['item_id']} {var['condition_id']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 — per-item status column
                    n_layers = getattr(wrapper, "num_layers", 32)
                    hdim = getattr(wrapper, "hidden_dim", 4096)
                    rec = {
                        "response": "",
                        "parsed_outcome": "unparseable",
                        "first_vs_parsed_agree": False,
                        "degeneracy_flag": False,
                        "truncated": False,
                        "status": "error",
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
                        "last_token_acts_fp16": np.zeros(
                            (n_layers + 1, hdim), dtype=np.float16
                        ),
                        "prefill_pos_norms_fp16": np.zeros(
                            (n_layers + 1, 1), dtype=np.float16
                        ),
                    }
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
