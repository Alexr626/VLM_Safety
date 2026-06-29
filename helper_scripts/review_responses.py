#!/usr/bin/env python3
"""Look up input images + model responses for one or more benchmark sample ids.

With ``--html``, writes a **single self-contained gallery page** (all images
embedded as base64) under the dated run directory::

    evaluation/results/{run_date}/_review/gallery.html

``run_date`` is taken from ``--run-date``, or inferred from results paths.

Examples
--------
One sample, open the image::

    python helper_scripts/review_responses.py pope_random_00166 --open

Several samples, one HTML file::

    python helper_scripts/review_responses.py --ids chair_000000357659 chair_000000256343 \\
        evaluation/results/2026-06-22/llava-1.5-7b-hf/chair/no_intervention/responses.json \\
        --html --run-date 2026-06-22

Ids from a file (one id per line)::

    python helper_scripts/review_responses.py --ids-file my_ids.txt results.json --html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from review_lib import (
    GalleryPanel,
    collect_response_sets,
    default_gallery_path,
    infer_benchmark,
    infer_run_date,
    load_sample_meta,
    open_in_os,
    parse_results_file,
    print_summary,
    write_gallery_html,
)


def _load_ids_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample_id", nargs="?", default=None,
                     help="single sample id (optional if --ids / --ids-file given)")
    p.add_argument("results", nargs="*", type=Path,
                   help="results JSON files (after ids, if any)")
    p.add_argument("--ids", nargs="+", default=[],
                   help="additional sample ids to include in the gallery")
    p.add_argument("--ids-file", type=Path, default=None,
                   help="text file with one sample id per line")
    p.add_argument("--benchmark", default=None,
                   help="override benchmark key for all ids (rare)")
    p.add_argument("--run-date", default=None,
                   help="dated run folder (default: infer from results paths)")
    p.add_argument("--output-dir", default="evaluation/results",
                   help="results root (default: evaluation/results)")
    p.add_argument("--open", action="store_true", dest="open_image",
                   help="open each sample image in the system viewer")
    p.add_argument("--html", action="store_true",
                   help="write one gallery HTML under the run dir")
    p.add_argument("--html-out", type=Path, default=None,
                   help="override gallery HTML path")
    p.add_argument("--no-open", action="store_true",
                   help="with --html, write the file but do not open it")
    p.add_argument("--max-chars", type=int, default=0,
                   help="truncate terminal responses to N chars (0 = no limit)")
    return p.parse_args(argv)


def _collect_ids(args) -> list[str]:
    ids: list[str] = []
    if args.sample_id:
        ids.append(args.sample_id)
    ids.extend(args.ids)
    if args.ids_file:
        if not args.ids_file.exists():
            raise SystemExit(f"ids file not found: {args.ids_file}")
        ids.extend(_load_ids_file(args.ids_file))
    # preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        raise SystemExit("provide at least one sample id (positional, --ids, or --ids-file)")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    sample_ids = _collect_ids(args)

    panels: list[GalleryPanel] = []
    for sid in sample_ids:
        benchmark = infer_benchmark(sid, args.benchmark)
        try:
            meta = load_sample_meta(sid, benchmark)
        except FileNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2
        if meta is None:
            print(f"[warning] id '{sid}' not found in combined.json — "
                  "relying on results files for metadata.", file=sys.stderr)

        resp_sets = []
        for rp in args.results:
            if not rp.exists():
                print(f"[warning] results file not found: {rp}", file=sys.stderr)
                continue
            try:
                rs = parse_results_file(rp, sid)
            except Exception as e:
                print(f"[warning] could not parse {rp}: {e}", file=sys.stderr)
                continue
            if rs is None:
                print(f"[warning] id '{sid}' not found in {rp}", file=sys.stderr)
            else:
                resp_sets.append(rs)

        print_summary(sid, benchmark, meta, resp_sets, args.max_chars)

        if args.open_image:
            image_path = meta.get("image_path") if meta else None
            if image_path and Path(image_path).exists():
                print(f"\nOpening image: {image_path}")
                open_in_os(Path(image_path))
            else:
                print(f"[warning] cannot open image for {sid}", file=sys.stderr)

        panels.append(GalleryPanel(sid, benchmark, meta, resp_sets))

    if args.html:
        run_date = args.run_date or infer_run_date(list(args.results), args.output_dir)
        if args.html_out:
            out = args.html_out
        elif run_date:
            out = default_gallery_path(run_date, args.output_dir)
        else:
            print("[error] --html requires --run-date or a results path under "
                  f"{args.output_dir}/YYYY-MM-DD/", file=sys.stderr)
            return 2
        title = f"Response review — {run_date or 'gallery'}"
        subtitle = f"{len(panels)} sample(s) · {len(args.results)} results file(s)"
        write_gallery_html(title, subtitle, panels, out)
        print(f"\nWrote gallery HTML: {out}")
        if not args.no_open:
            open_in_os(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
