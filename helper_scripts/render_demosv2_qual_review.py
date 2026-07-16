#!/usr/bin/env python3
"""HTML galleries for demos_v2 qualitative grid cells.

Discovers result dirs matching ``{iv}__b{beta}__d{dim}__nd{N}`` for one
dimension (plus baseline ``no_intervention``) and writes side-by-side
galleries under ``evaluation/results/{run_date}/_samples/{model}/{bench}/``.

Sample ids come from ``data/vti/qual_subset_chair5_amber25.json`` (same
2026-06-22 LLaVA qualitative bundle).

Usage::

    python helper_scripts/render_demosv2_qual_review.py \\
      --run_date 2026-07-13 --model llava-hf/llava-1.5-7b-hf --dimension all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "helper_scripts"))

from review_lib import (  # noqa: E402
    GalleryPanel,
    collect_response_sets,
    infer_benchmark,
    sample_bundle_dir,
    write_gallery_html,
)
from sample_responses import _build_amber_index, _meta_for_panel  # noqa: E402
from src.model import _normalize_model_name  # noqa: E402

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_SUBSET = _PROJECT_ROOT / "data" / "vti" / "qual_subset_chair5_amber25.json"
IV_ORDER = [
    "vti_textual_additive_mlp",
    "vti_textual_additive_layer",
    "vti_textual_uniform_rotation_mlp",
    "vti_textual_uniform_rotation_layer",
]
BETA_ORDER = ("0.5", "0.2", "0.9")
ND_ORDER = ("50", "100", "200", "500")

_CELL_RE = re.compile(
    r"^(?P<iv>vti_textual_(?:additive|uniform_rotation|gated_rotation)_(?:mlp|layer))"
    r"__b(?P<beta>[0-9.]+)"
    r"__d(?P<dim>[a-z]+)"
    r"__nd(?P<nd>\d+)$"
)


def _friendly_label(name: str) -> str:
    if name == "no_intervention":
        return "baseline · no_intervention"
    m = _CELL_RE.match(name)
    if not m:
        return name
    iv = m.group("iv").replace("vti_textual_", "")
    return f"{iv} · β={m.group('beta')} · nd={m.group('nd')}"


def _cell_sort_key(name: str) -> tuple:
    if name == "no_intervention":
        return (0, 0, 0, 0)
    m = _CELL_RE.match(name)
    if not m:
        return (9, name, "", "")
    iv = m.group("iv")
    try:
        iv_i = IV_ORDER.index(iv)
    except ValueError:
        iv_i = 50
    try:
        b_i = BETA_ORDER.index(m.group("beta"))
    except ValueError:
        b_i = 50
    try:
        n_i = ND_ORDER.index(m.group("nd"))
    except ValueError:
        n_i = 50
    return (1, iv_i, n_i, b_i)


def _discover_dim_cells(
    results_root: Path,
    model_short: str,
    benchmark: str,
    dimension: str,
) -> list[Path]:
    bench_dir = results_root / model_short / benchmark
    if not bench_dir.is_dir():
        return []
    cells: list[Path] = []
    for d in bench_dir.iterdir():
        if not d.is_dir() or not (d / "responses.json").is_file():
            continue
        if d.name == "no_intervention":
            cells.append(d)
            continue
        m = _CELL_RE.match(d.name)
        if m and m.group("dim") == dimension:
            cells.append(d)
    return sorted(cells, key=lambda p: _cell_sort_key(p.name))


def _load_subset(subset_path: Path) -> dict:
    return json.loads(subset_path.read_text())


def _html_path(
    run_date: str,
    model_short: str,
    benchmark: str,
    dimension: str,
    output_dir: str,
) -> Path:
    return (
        sample_bundle_dir(run_date, model_short, benchmark, output_dir)
        / f"{model_short}_{benchmark}_demosv2_{dimension}_review.html"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_date", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dimension", required=True,
                   choices=["all", "existence", "attribute", "counting", "relation"])
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--subset_ids_file", type=Path, default=DEFAULT_SUBSET)
    p.add_argument("--benchmark", choices=("amber", "chair", "both"), default="both")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    results_root = Path(args.output_dir) / args.run_date
    if not results_root.is_dir():
        raise SystemExit(f"No results at {results_root}")
    subset = _load_subset(args.subset_ids_file)
    benchmarks = ("amber", "chair") if args.benchmark == "both" else (args.benchmark,)

    amber_index = None
    if "amber" in benchmarks:
        amber_ids = list(subset.get("amber") or [])
        amber_index = _build_amber_index(amber_ids)

    written = []
    for bench in benchmarks:
        ids = list(subset.get(bench) or [])
        if not ids:
            print(f"[warning] no ids for {bench} in {args.subset_ids_file}", file=sys.stderr)
            continue
        cells = _discover_dim_cells(results_root, model_short, bench, args.dimension)
        if not cells:
            print(f"[warning] no cells for {bench} dim={args.dimension}", file=sys.stderr)
            continue
        result_paths = [c / "responses.json" for c in cells]
        labels = [_friendly_label(c.name) for c in cells]
        panel_specs = [(sid, "") for sid in ids]
        panels: list[GalleryPanel] = []
        for sid, heading in panel_specs:
            b = infer_benchmark(sid, bench)
            meta = _meta_for_panel(sid, b, amber_index)
            resp_sets = collect_response_sets(sid, result_paths, labels)
            panels.append(
                GalleryPanel(
                    sid, b, meta, resp_sets,
                    section_heading=heading or f"{bench.upper()} · demos_v2 · {args.dimension}",
                )
            )
        out_html = _html_path(
            args.run_date, model_short, bench, args.dimension, args.output_dir,
        )
        subtitle = (
            f"demos_v2 qual · {model_short} · {bench.upper()} · dim={args.dimension} · "
            f"{len(panels)} samples · {len(cells)} cells"
        )
        write_gallery_html(
            f"demos_v2 qual review — {bench.upper()} / {args.dimension} ({args.run_date})",
            subtitle,
            panels,
            out_html,
        )
        print(f"Wrote {out_html}")
        written.append(str(out_html))

    if not written:
        raise SystemExit("No galleries written.")


if __name__ == "__main__":
    main()
