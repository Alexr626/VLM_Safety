#!/usr/bin/env python3
"""Build self-contained HTML galleries for a VTI visual-smoke run_date.

Reuses ``review_lib`` (same machinery as ``sample_responses.py`` / ``review_responses.py``).
For each pinned AMBER/CHAIR sample, shows responses from **every smoke cell**
(baseline + textual Stage 0 + visual Stages 1/1b).

Sample ids default to the 2026-06-22 qualitative bundle (same pinned AMBER-25 + CHAIR-5
used by ``diagnostic_experiments/vti_visual_smoke/run_smoke.py``).

Output::

    evaluation/results/{run_date}/_samples/{model_short}/{benchmark}/
      {model_short}_{benchmark}_smoke_review.html

Usage::

    python helper_scripts/render_smoke_review.py --run_date 2026-07-02
    RUN_DATE=2026-07-02 bash helper_scripts/run_render_smoke_review.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "helper_scripts"))

from review_lib import (  # noqa: E402
    GalleryPanel,
    bundle_json_path,
    collect_response_sets,
    infer_benchmark,
    load_sample_meta,
    sample_bundle_dir,
    write_gallery_html,
)
from sample_responses import _build_amber_index, _meta_for_panel  # noqa: E402
from src.model import _normalize_model_name  # noqa: E402

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_SAMPLE_RUN_DATE = "2026-06-22"

# Explicit cell order (matches run_smoke.py stages; unique dirs only).
SMOKE_CELL_ORDER = [
    "no_intervention",
    "vti_textual_uniform_rotation_layer__b0.2",
    "vti_textual_uniform_rotation_layer__b0.4",
    "vti_textual_additive_layer__b0.9",
    "vti_visual_additive_layer__a0.2",
    "vti_visual_additive_layer__a0.4",
    "vti_visual_additive_layer__a0.9",
    "vti_visual_uniform_rotation_mlp__a0.2",
    "vti_visual_uniform_rotation_mlp__a0.4",
    "vti_visual_uniform_rotation_mlp__a0.9",
    "vti_visual_additive_mlp__a0.2",
    "vti_visual_additive_mlp__a0.4",
    "vti_visual_additive_mlp__a0.9",
    "vti_visual_uniform_rotation_layer__a0.2",
    "vti_visual_uniform_rotation_layer__a0.4",
    "vti_visual_uniform_rotation_layer__a0.9",
]


def _friendly_cell_label(cell_name: str) -> str:
    if cell_name == "no_intervention":
        return "baseline · no_intervention"
    if cell_name.startswith("vti_textual_"):
        rest = cell_name[len("vti_textual_"):]
        if "__b" in rest:
            variant_site, beta = rest.rsplit("__b", 1)
            return f"Stage 0 textual · {variant_site} · β={beta}"
    if cell_name.startswith("vti_visual_"):
        rest = cell_name[len("vti_visual_"):]
        if "__a" in rest:
            variant_site, alpha = rest.rsplit("__a", 1)
            return f"Stage 1/1b visual · {variant_site} · α={alpha}"
    return cell_name


def _cell_sort_key(name: str) -> tuple:
    try:
        return (0, SMOKE_CELL_ORDER.index(name))
    except ValueError:
        return (1, name)


def _discover_smoke_cells(results_root: Path, model_short: str, benchmark: str) -> list[Path]:
    bench_dir = results_root / model_short / benchmark
    if not bench_dir.is_dir():
        return []
    cells = [
        d for d in bench_dir.iterdir()
        if d.is_dir() and (d / "responses.json").is_file()
    ]
    return sorted(cells, key=lambda p: _cell_sort_key(p.name))


def _load_sample_spec(
    sample_run_date: str,
    model_short: str,
    benchmark: str,
    output_dir: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (panel_specs as (id, heading), flat ordered ids)."""
    bundle_path = bundle_json_path(sample_run_date, model_short, benchmark, output_dir)
    if not bundle_path.is_file():
        raise SystemExit(
            f"Sample bundle not found: {bundle_path}\n"
            f"Run sample_responses.py --run_date {sample_run_date} first, or pass "
            "--sample-run-date pointing at an existing bundle."
        )
    bundle = json.loads(bundle_path.read_text())
    spec = bundle.get("sample_ids", bundle)
    if benchmark == "amber" and isinstance(spec, dict):
        ordered = list(spec.get("ordered", []))
        panel_specs = [tuple(x) for x in spec.get("panel_specs", [])]
        if not panel_specs:
            panel_specs = [(sid, "") for sid in ordered]
        return panel_specs, ordered
    if isinstance(spec, list):
        return [(sid, "") for sid in spec], list(spec)
    raise SystemExit(f"Unexpected sample_ids shape in {bundle_path}")


def _smoke_html_path(
    run_date: str,
    model_short: str,
    benchmark: str,
    output_dir: str,
) -> Path:
    return (
        sample_bundle_dir(run_date, model_short, benchmark, output_dir)
        / f"{model_short}_{benchmark}_smoke_review.html"
    )


def _build_panels(
    results_root: Path,
    model_short: str,
    benchmark: str,
    panel_specs: list[tuple[str, str]],
    amber_index: dict | None,
) -> list[GalleryPanel]:
    cells = _discover_smoke_cells(results_root, model_short, benchmark)
    if not cells:
        return []
    result_paths = [c / "responses.json" for c in cells]
    labels = [_friendly_cell_label(c.name) for c in cells]

    panels: list[GalleryPanel] = []
    prev_heading = None
    for sid, heading in panel_specs:
        display = heading if heading != prev_heading else ""
        prev_heading = heading
        bench = infer_benchmark(sid, benchmark)
        meta = _meta_for_panel(sid, bench, amber_index)
        resp_sets = collect_response_sets(sid, result_paths, labels)
        section = display or f"{benchmark.upper()} · smoke"
        panels.append(GalleryPanel(sid, bench, meta, resp_sets, section_heading=section))
    return panels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_date", required=True,
                   help="Smoke results date (e.g. 2026-07-02)")
    p.add_argument("--sample-run-date", default=DEFAULT_SAMPLE_RUN_DATE,
                   help="Run date of the pinned sample-id bundle (default: 2026-06-22)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--benchmark", choices=("amber", "chair", "both"), default="both")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    results_root = Path(args.output_dir) / args.run_date
    if not results_root.is_dir():
        raise SystemExit(f"No results at {results_root}")

    benchmarks = ("amber", "chair") if args.benchmark == "both" else (args.benchmark,)
    amber_index = None
    if "amber" in benchmarks:
        bundle = json.loads(
            bundle_json_path(
                args.sample_run_date, model_short, "amber", args.output_dir,
            ).read_text()
        )
        ordered = bundle.get("sample_ids", {}).get("ordered", [])
        amber_index = _build_amber_index(ordered)

    manifest_paths: list[str] = []
    for bench in benchmarks:
        panel_specs, ordered = _load_sample_spec(
            args.sample_run_date, model_short, bench, args.output_dir,
        )
        panels = _build_panels(
            results_root, model_short, bench, panel_specs, amber_index,
        )
        if not panels:
            print(f"[warning] no smoke cells for {bench}", file=sys.stderr)
            continue
        out_html = _smoke_html_path(args.run_date, model_short, bench, args.output_dir)
        n_cells = len(_discover_smoke_cells(results_root, model_short, bench))
        subtitle = (
            f"VTI smoke · {model_short} · {bench.upper()} · "
            f"{len(panels)} samples · {n_cells} intervention cells"
        )
        write_gallery_html(
            f"VTI smoke review — {bench.upper()} ({args.run_date})",
            subtitle,
            panels,
            out_html,
        )
        print(f"Wrote {out_html}")
        manifest_paths.append(str(out_html))

    if not manifest_paths:
        raise SystemExit("No galleries written — check run_date and results tree.")


if __name__ == "__main__":
    main()
