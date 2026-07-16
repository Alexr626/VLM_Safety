#!/usr/bin/env python3
"""Split demos_v2 qualitative HTML galleries by intervention / dimension / nd.

Writes one gallery per (model, benchmark, intervention, dimension, num_demos)::

    evaluation/results/{run_date}/_samples/{model_short}/{benchmark}/
      {additive_layer|additive_mlp|uniform_rotation_layer|uniform_rotation_mlp}/
        {all|existence|attribute|counting|relation}/
            {50|100|200|500}/
              {model}_{benchmark}_{intervention}_{dimension}_nd{N}.html

Each page shows baseline ``no_intervention`` plus that intervention at
β ∈ {0.5, 0.2, 0.9}. CHAIR pages annotate every response with per-caption
CHAIR_s / CHAIR_i (computed from the caption vs COCO GT). The filename
encodes the same variables as the directory path so downloads do not collide.

Fixes the prior demosv2 gallery regex that silently dropped
``vti_textual_uniform_rotation_{layer,mlp}`` cells.

Usage::

    python helper_scripts/render_demosv2_qual_split_review.py --run_date 2026-07-13
    python helper_scripts/render_demosv2_qual_split_review.py \\
      --run_date 2026-07-13 --model llava-hf/llava-1.5-7b-hf --benchmark amber
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "helper_scripts"))

from review_lib import (  # noqa: E402
    GalleryPanel,
    ResponseSet,
    infer_benchmark,
    sample_bundle_dir,
    write_gallery_html,
)
from sample_responses import _build_amber_index, _meta_for_panel  # noqa: E402
from src.model import _normalize_model_name  # noqa: E402

DEFAULT_MODELS = (
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)
DEFAULT_SUBSET = _PROJECT_ROOT / "data" / "vti" / "qual_subset_chair5_amber25.json"

INTERVENTIONS = (
    "vti_textual_additive_layer",
    "vti_textual_additive_mlp",
    "vti_textual_uniform_rotation_layer",
    "vti_textual_uniform_rotation_mlp",
)
DIMENSIONS = ("all", "existence", "attribute", "counting", "relation")
NUM_DEMOS = (50, 100, 200, 500)
BETAS = (0.2, 0.5, 0.9)  # ascending in galleries for easier per-example comparison

# Matches additive_* and uniform_rotation_* (multi-token variant names).
_CELL_RE = re.compile(
    r"^(?P<iv>vti_textual_(?:additive|uniform_rotation|gated_rotation)_(?:mlp|layer))"
    r"__b(?P<beta>[0-9.]+)"
    r"__d(?P<dim>[a-z]+)"
    r"__nd(?P<nd>\d+)$"
)


def iv_short(full_name: str) -> str:
    return full_name.removeprefix("vti_textual_")


def cell_dirname(iv: str, beta: float, dim: str, nd: int) -> str:
    return f"{iv}__b{beta}__d{dim}__nd{nd}"


def _load_responses(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        return {}
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        return {}
    return {str(r["id"]): r for r in records if "id" in r}


def _chair_note_for_record(rec: dict) -> str:
    """Per-caption CHAIR_s / CHAIR_i for gallery annotation."""
    from evaluation.classifiers.chair_objects import (
        gt_objects_for_image,
        parse_caption_objects,
    )

    caption = rec.get("response") or ""
    meta = rec.get("metadata") or {}
    raw = meta.get("raw") or {}
    image_id = raw.get("coco_id")
    if image_id is None:
        return "CHAIR unavailable (no coco_id)"
    if not caption.strip():
        return "CHAIR undefined (empty caption)"
    mentioned = parse_caption_objects(caption)
    gt = gt_objects_for_image(image_id)
    hallucinated = mentioned - gt
    chair_s = 1 if hallucinated else 0
    chair_i = (len(hallucinated) / len(mentioned)) if mentioned else 0.0
    hall_str = ",".join(sorted(hallucinated)) if hallucinated else "—"
    return (
        f"CHAIR_s={chair_s} · CHAIR_i={chair_i:.2f} · "
        f"halluc=[{hall_str}] · n_mentioned={len(mentioned)}"
    )


def _response_set_for_cell(
    sample_id: str,
    cell_dir: Path,
    label: str,
    *,
    with_chair_scores: bool,
) -> Optional[ResponseSet]:
    by_id = _load_responses(cell_dir / "responses.json")
    rec = by_id.get(sample_id)
    if rec is None:
        return None
    rs = ResponseSet(label, cell_dir / "responses.json")
    rs.question = rec.get("question")
    rs.ground_truth = rec.get("ground_truth")
    note = _chair_note_for_record(rec) if with_chair_scores else None
    rs.add("response", rec.get("response", ""), note)
    return rs


def _out_dir(
    run_date: str,
    model_short: str,
    benchmark: str,
    intervention: str,
    dimension: str,
    num_demos: int,
    output_dir: str,
) -> Path:
    return (
        sample_bundle_dir(run_date, model_short, benchmark, output_dir)
        / iv_short(intervention)
        / dimension
        / str(num_demos)
    )


def _out_html_name(
    model_short: str,
    benchmark: str,
    intervention: str,
    dimension: str,
    num_demos: int,
) -> str:
    """Download-safe filename encoding every split axis (no collisions)."""
    return (
        f"{model_short}_{benchmark}_{iv_short(intervention)}_"
        f"{dimension}_nd{num_demos}.html"
    )


def render_one(
    *,
    results_root: Path,
    model_short: str,
    benchmark: str,
    intervention: str,
    dimension: str,
    num_demos: int,
    sample_ids: List[str],
    amber_index: Optional[dict],
    run_date: str,
    output_dir: str,
) -> Optional[Path]:
    bench_dir = results_root / model_short / benchmark
    baseline_dir = bench_dir / "no_intervention"
    cell_dirs: List[Tuple[str, Path]] = []
    if (baseline_dir / "responses.json").is_file():
        cell_dirs.append(("baseline · no_intervention", baseline_dir))
    for beta in BETAS:
        name = cell_dirname(intervention, beta, dimension, num_demos)
        d = bench_dir / name
        if (d / "responses.json").is_file():
            cell_dirs.append((f"{iv_short(intervention)} · β={beta}", d))

    # Need at least one intervention cell (not only baseline).
    if len(cell_dirs) <= (1 if cell_dirs and cell_dirs[0][1] == baseline_dir else 0):
        return None
    if not any(d.name.startswith(intervention) for _, d in cell_dirs):
        return None

    with_chair = benchmark == "chair"
    panels: List[GalleryPanel] = []
    for sid in sample_ids:
        b = infer_benchmark(sid, benchmark)
        meta = _meta_for_panel(sid, b, amber_index)
        resp_sets: List[ResponseSet] = []
        for label, cell_dir in cell_dirs:
            rs = _response_set_for_cell(
                sid, cell_dir, label, with_chair_scores=with_chair,
            )
            if rs is not None:
                resp_sets.append(rs)
        panels.append(
            GalleryPanel(
                sid,
                b,
                meta,
                resp_sets,
                section_heading=(
                    f"{benchmark.upper()} · {iv_short(intervention)} · "
                    f"{dimension} · nd={num_demos}"
                ),
            )
        )

    out_html = _out_dir(
        run_date, model_short, benchmark, intervention, dimension, num_demos,
        output_dir,
    ) / _out_html_name(
        model_short, benchmark, intervention, dimension, num_demos,
    )
    subtitle = (
        f"demos_v2 split · {model_short} · {benchmark.upper()} · "
        f"{iv_short(intervention)} · dim={dimension} · nd={num_demos} · "
        f"{len(panels)} samples · {len(cell_dirs)} columns "
        f"(baseline + βs)"
    )
    if with_chair:
        subtitle += " · per-response CHAIR_s/i in yellow notes"
    write_gallery_html(
        (
            f"demos_v2 · {benchmark.upper()} · {iv_short(intervention)} · "
            f"{dimension} · nd={num_demos} ({run_date})"
        ),
        subtitle,
        panels,
        out_html,
    )
    return out_html


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run_date", required=True)
    p.add_argument("--model", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--subset_ids_file", type=Path, default=DEFAULT_SUBSET)
    p.add_argument("--benchmark", choices=("amber", "chair", "both"), default="both")
    p.add_argument("--interventions", nargs="+", default=list(INTERVENTIONS))
    p.add_argument("--dimensions", nargs="+", default=list(DIMENSIONS))
    p.add_argument("--num_demos", nargs="+", type=int, default=list(NUM_DEMOS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.output_dir) / args.run_date
    if not results_root.is_dir():
        raise SystemExit(f"No results at {results_root}")
    subset = json.loads(args.subset_ids_file.read_text())
    benchmarks = ("amber", "chair") if args.benchmark == "both" else (args.benchmark,)

    written: List[str] = []
    skipped = 0
    for model in args.model:
        model_short = _normalize_model_name(model)
        for bench in benchmarks:
            ids = list(subset.get(bench) or [])
            if not ids:
                print(f"[warning] no ids for {bench}", file=sys.stderr)
                continue
            amber_index = (
                _build_amber_index(list(subset.get("amber") or []))
                if bench == "amber"
                else None
            )
            for iv in args.interventions:
                for dim in args.dimensions:
                    for nd in args.num_demos:
                        path = render_one(
                            results_root=results_root,
                            model_short=model_short,
                            benchmark=bench,
                            intervention=iv,
                            dimension=dim,
                            num_demos=nd,
                            sample_ids=ids,
                            amber_index=amber_index,
                            run_date=args.run_date,
                            output_dir=args.output_dir,
                        )
                        if path is None:
                            skipped += 1
                            continue
                        print(f"Wrote {path}")
                        written.append(str(path))

    print(f"Done: wrote {len(written)} galleries, skipped {skipped} missing cells.")
    if not written:
        raise SystemExit("No galleries written.")


if __name__ == "__main__":
    main()
