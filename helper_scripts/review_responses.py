#!/usr/bin/env python3
"""Look up the input image + model response(s) for a benchmark sample id.

Given a sample id (e.g. ``pope_random_00166``) this resolves the image that was
fed to the model (from ``data/{benchmark}/combined.json``), the question, and
the ground-truth label, and pulls the model's response(s) out of zero or more
results JSON files. It can:

  * print a terminal summary (image path, question, ground truth, responses),
  * open the input image in the system image viewer (``--open``), and
  * build a self-contained HTML page that shows the image **side-by-side** with
    the responses and open it in the browser (``--html``).

Supported results formats (auto-detected):
  * ``vti_rotation_strength`` sweep JSON
    (``per_sample`` / ``changed_examples_by_beta`` blocks)
  * standard ``run_eval`` ``responses.json`` (a list of ``{id, response, ...}``
    records, or ``{"records": [...]}``)

Examples
--------
Just resolve + open the image the model was fed::

    python helper_scripts/review_responses.py pope_random_00166 --open

Lay the sweep responses next to the image in the browser::

    python helper_scripts/review_responses.py pope_random_00166 \\
        evaluation/vti_rotation_strength/results/2026-06-19/\\
qwen2.5-vl-7b-instruct/sweep_uniform_rotation_layer_random_n200.json --html

Compare several result files for one id in the terminal::

    python helper_scripts/review_responses.py pope_random_00042 fileA.json fileB.json
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Benchmark registry keys whose on-disk dir differs from the key itself.
_DATA_SUBDIR_OVERRIDES = {"mmhal_bench": "mmhal-bench"}


# ── id -> benchmark / combined.json resolution ────────────────────────────────

def _data_subdir(benchmark: str) -> str:
    return _DATA_SUBDIR_OVERRIDES.get(benchmark, benchmark)


def infer_benchmark(sample_id: str, override: Optional[str]) -> str:
    """Best-effort benchmark key for a sample id (prefix before the first '_')."""
    if override:
        return override
    prefix = sample_id.split("_", 1)[0].lower()
    # POPE ids are pope_{split}_{n}; the prefix is already the registry key.
    return prefix


def _combined_path(benchmark: str) -> Path:
    return PROJECT_ROOT / "data" / _data_subdir(benchmark) / "combined.json"


def load_sample_meta(sample_id: str, benchmark: str) -> Optional[Dict[str, Any]]:
    """Return the combined.json entry for ``sample_id`` (image_path/text/label)."""
    path = _combined_path(benchmark)
    if not path.exists():
        raise FileNotFoundError(
            f"No combined.json for benchmark '{benchmark}' at {path}. "
            f"Pass --benchmark, or run data_scripts/download_{benchmark}.py."
        )
    with open(path) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else list(data.values())
    for e in entries:
        if e.get("id") == sample_id:
            return e
    return None


# ── results-file parsing ──────────────────────────────────────────────────────

class ResponseSet:
    """Responses for one sample id extracted from a single results file."""

    def __init__(self, source_label: str, source_path: Path):
        self.source_label = source_label
        self.source_path = source_path
        self.question: Optional[str] = None
        self.ground_truth: Optional[str] = None
        # each item: (label, text, optional_note)
        self.items: List[Tuple[str, str, Optional[str]]] = []

    def add(self, label: str, text: str, note: Optional[str] = None) -> None:
        self.items.append((label, text if text is not None else "", note))

    def __bool__(self) -> bool:
        return bool(self.items)


def _sweep_source_label(obj: Dict[str, Any], path: Path) -> str:
    bits = [obj.get("model", ""), obj.get("variant", "")]
    site = obj.get("hook_site")
    if site:
        bits[-1] = f"{bits[-1]}@{site}"
    split = obj.get("pope_split")
    if split:
        bits.append(split)
    label = " ".join(b for b in bits if b).strip()
    return label or path.name


def _parse_sweep(obj: Dict[str, Any], sample_id: str, path: Path) -> Optional[ResponseSet]:
    """vti_rotation_strength sweep JSON (per_sample preferred)."""
    rs = ResponseSet(_sweep_source_label(obj, path), path)

    per_sample = obj.get("per_sample")
    if isinstance(per_sample, list):
        rec = next((r for r in per_sample if r.get("id") == sample_id), None)
        if rec is None:
            return None
        rs.question = rec.get("question")
        rs.ground_truth = rec.get("ground_truth")
        if "baseline" in rec:
            rs.add("baseline (no hook)", rec["baseline"])
        by_beta = rec.get("by_beta", {})
        for beta in _sorted_beta_keys(by_beta.keys()):
            v = by_beta[beta]
            if isinstance(v, dict):
                rs.add(f"beta={beta}", v.get("response", ""), v.get("class"))
            else:
                rs.add(f"beta={beta}", str(v))
        if "decode_only_beta_max" in rec:
            rs.add("decode-only @ beta_max", rec["decode_only_beta_max"])
        if "skip_pos0_beta_max" in rec:
            rs.add("skip-pos0 @ beta_max", rec["skip_pos0_beta_max"])
        return rs if rs else None

    # Fallback: only the (capped) changed_examples_by_beta block is present.
    changed = obj.get("changed_examples_by_beta")
    if isinstance(changed, dict):
        baseline_added = False
        for beta in _sorted_beta_keys(changed.keys()):
            ex = next((e for e in changed[beta] if e.get("id") == sample_id), None)
            if ex is None:
                continue
            if not baseline_added and "baseline" in ex:
                rs.add("baseline (no hook)", ex["baseline"])
                baseline_added = True
            rs.add(f"beta={beta}", ex.get("steered", ""), "changed")
        return rs if rs else None

    return None


def _sorted_beta_keys(keys) -> List[str]:
    def _key(k: str):
        try:
            return (0, float(k))
        except (TypeError, ValueError):
            return (1, k)
    # Strongest beta first reads naturally for review.
    return sorted(keys, key=_key, reverse=True)


def _parse_responses_json(records: List[dict], sample_id: str, path: Path) -> Optional[ResponseSet]:
    """Standard run_eval responses.json: list of per-sample records."""
    rec = next((r for r in records if r.get("id") == sample_id), None)
    if rec is None:
        return None
    rs = ResponseSet(path.name, path)
    rs.question = rec.get("question")
    rs.ground_truth = rec.get("ground_truth")
    resp = rec.get("response", "")
    note = rec.get("intervention") or rec.get("task")
    rs.add("response", resp, note)
    return rs


def parse_results_file(path: Path, sample_id: str) -> Optional[ResponseSet]:
    with open(path) as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        if "per_sample" in obj or "changed_examples_by_beta" in obj or "metrics_by_beta" in obj:
            return _parse_sweep(obj, sample_id, path)
        if isinstance(obj.get("records"), list):
            return _parse_responses_json(obj["records"], sample_id, path)
        # A dict keyed by id -> record.
        if sample_id in obj and isinstance(obj[sample_id], dict):
            return _parse_responses_json([{"id": sample_id, **obj[sample_id]}], sample_id, path)
    if isinstance(obj, list):
        return _parse_responses_json(obj, sample_id, path)
    return None


# ── yes/no helpers (mirror evaluation/classifiers/metrics._normalize_yes_no) ──

def normalize_yes_no(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if re.match(r"^\s*yes\b", t):
        return "yes"
    if re.match(r"^\s*no\b", t):
        return "no"
    if re.search(r"\bno\b", t):
        return "no"
    if re.search(r"\byes\b", t):
        return "yes"
    return None


# ── terminal rendering ────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + f" ... [+{len(text) - max_chars} chars]"
    return text


def print_summary(sample_id: str, benchmark: str, meta: Optional[Dict[str, Any]],
                  resp_sets: List[ResponseSet], max_chars: int) -> None:
    sep = "=" * 78
    print(sep)
    print(f"  {sample_id}   (benchmark: {benchmark})")
    print(sep)

    question = meta.get("text") if meta else None
    label = meta.get("label") if meta else None
    image_path = meta.get("image_path") if meta else None
    # Fall back to question/gt from the results files if combined.json lacks them.
    if not question:
        question = next((r.question for r in resp_sets if r.question), None)
    if not label:
        label = next((r.ground_truth for r in resp_sets if r.ground_truth), None)

    print(f"  question     : {question or '<unknown>'}")
    print(f"  ground truth : {label or '<unknown>'}")
    print(f"  image        : {image_path or '<unresolved>'}")
    if image_path and not Path(image_path).exists():
        print("                 [WARNING] image file not found on disk")

    if not resp_sets:
        print("\n  (no results files provided — pass one or more results JSON paths")
        print("   to see the model responses, and/or use --open / --html)")
        print(sep)
        return

    for rs in resp_sets:
        print("\n" + "-" * 78)
        print(f"  results: {rs.source_label}")
        print(f"  file   : {rs.source_path}")
        print("-" * 78)
        for lbl, text, note in rs.items:
            yn = normalize_yes_no(text)
            tag = f" [{note}]" if note else ""
            yn_tag = f" -> {yn}" if yn else ""
            print(f"\n  • {lbl}{tag}{yn_tag}")
            body = _truncate(text.strip() or "<empty>", max_chars)
            for line in body.splitlines() or ["<empty>"]:
                print(f"      {line}")
    print("\n" + sep)


# ── HTML rendering ────────────────────────────────────────────────────────────

_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def _img_data_uri(image_path: Path) -> Optional[str]:
    if not image_path.exists():
        return None
    mime = _MIME.get(image_path.suffix.lower(), "application/octet-stream")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _yn_badge(text: str) -> str:
    yn = normalize_yes_no(text)
    if yn == "yes":
        return '<span class="yn yn-yes">yes</span>'
    if yn == "no":
        return '<span class="yn yn-no">no</span>'
    return '<span class="yn yn-unk">?</span>'


def build_html(sample_id: str, benchmark: str, meta: Optional[Dict[str, Any]],
               resp_sets: List[ResponseSet]) -> str:
    question = (meta.get("text") if meta else None) or \
        next((r.question for r in resp_sets if r.question), None) or "<unknown>"
    label = (meta.get("label") if meta else None) or \
        next((r.ground_truth for r in resp_sets if r.ground_truth), None) or "<unknown>"
    image_path = meta.get("image_path") if meta else None

    img_html = "<p class='missing'>image unresolved</p>"
    if image_path:
        uri = _img_data_uri(Path(image_path))
        if uri:
            img_html = f"<img src='{uri}' alt='{html.escape(sample_id)}'>"
        else:
            img_html = (f"<p class='missing'>image file not found:<br>"
                        f"{html.escape(str(image_path))}</p>")

    sections = []
    for rs in resp_sets:
        rows = []
        for lbl, text, note in rs.items:
            note_html = f"<span class='note'>{html.escape(note)}</span>" if note else ""
            body = html.escape(text.strip()) if text.strip() else "<em>&lt;empty&gt;</em>"
            rows.append(
                f"<div class='resp'>"
                f"<div class='resp-head'>{_yn_badge(text)}"
                f"<span class='lbl'>{html.escape(lbl)}</span>{note_html}</div>"
                f"<div class='resp-body'>{body}</div></div>"
            )
        sections.append(
            f"<section class='resultset'>"
            f"<h2>{html.escape(rs.source_label)}</h2>"
            f"<p class='src'>{html.escape(str(rs.source_path))}</p>"
            f"{''.join(rows)}</section>"
        )
    if not sections:
        sections.append("<section class='resultset'><p class='missing'>"
                        "No results files provided.</p></section>")

    gt_badge = _yn_badge(label)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(sample_id)} — response review</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; }}
  .wrap {{ display: flex; align-items: flex-start; gap: 0; }}
  .pane-img {{ position: sticky; top: 0; flex: 0 0 46%; max-width: 46%;
              height: 100vh; padding: 20px; overflow: auto;
              background: #0f1115; color: #e6e6e6; }}
  .pane-img img {{ max-width: 100%; height: auto; border-radius: 8px;
                  box-shadow: 0 4px 24px rgba(0,0,0,.5); }}
  .pane-txt {{ flex: 1 1 54%; padding: 24px 28px; overflow: auto; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .meta {{ margin: 0 0 16px; color: #9aa0a6; font-size: 13px; word-break: break-all; }}
  .qa {{ background: #f4f5f7; border-radius: 8px; padding: 12px 14px; margin: 0 0 20px; }}
  .qa b {{ color: #555; }}
  .resultset {{ border-top: 2px solid #e0e0e0; padding-top: 12px; margin-top: 18px; }}
  .resultset h2 {{ font-size: 15px; margin: 0 0 2px; }}
  .src {{ margin: 0 0 12px; color: #9aa0a6; font-size: 12px; word-break: break-all; }}
  .resp {{ border: 1px solid #e2e2e2; border-radius: 8px; margin: 0 0 10px;
          overflow: hidden; }}
  .resp-head {{ display: flex; align-items: center; gap: 8px;
               background: #f7f8fa; padding: 6px 10px; font-size: 13px; }}
  .lbl {{ font-weight: 600; }}
  .note {{ color: #8a6d3b; background: #fcf3d9; border-radius: 4px;
          padding: 1px 6px; font-size: 11px; }}
  .resp-body {{ padding: 10px 12px; white-space: pre-wrap; }}
  .yn {{ font-size: 11px; font-weight: 700; border-radius: 4px;
        padding: 1px 7px; color: #fff; }}
  .yn-yes {{ background: #1a7f37; }}
  .yn-no {{ background: #b42318; }}
  .yn-unk {{ background: #888; }}
  .missing {{ color: #b42318; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #15171c; color: #e6e6e6; }}
    .qa {{ background: #1f2228; }} .qa b {{ color: #aaa; }}
    .resp {{ border-color: #2c2f36; }} .resp-head {{ background: #1f2228; }}
    .resultset {{ border-color: #2c2f36; }}
  }}
</style></head>
<body><div class="wrap">
  <div class="pane-img">{img_html}</div>
  <div class="pane-txt">
    <h1>{html.escape(sample_id)}</h1>
    <p class="meta">benchmark: {html.escape(benchmark)} &nbsp;·&nbsp; {html.escape(str(image_path or ''))}</p>
    <div class="qa">
      <div><b>Q:</b> {html.escape(question)}</div>
      <div><b>Ground truth:</b> {gt_badge} {html.escape(label)}</div>
    </div>
    {''.join(sections)}
  </div>
</div></body></html>
"""


# ── opening files in the OS ───────────────────────────────────────────────────

def open_in_os(path: Path) -> None:
    p = str(path)
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", p])
        elif sys.platform.startswith("win"):
            import os
            os.startfile(p)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", p],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # No OS opener (e.g. headless host) — fall back to webbrowser for html.
        webbrowser.open(path.as_uri())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sample_id", help="e.g. pope_random_00166")
    p.add_argument("results", nargs="*", type=Path,
                   help="zero or more results JSON files to pull responses from")
    p.add_argument("--benchmark", default=None,
                   help="override benchmark key (default: inferred from id prefix)")
    p.add_argument("--open", action="store_true", dest="open_image",
                   help="open the input image in the system viewer")
    p.add_argument("--html", action="store_true",
                   help="build a side-by-side image+responses HTML page and open it")
    p.add_argument("--html-out", type=Path, default=None,
                   help="where to write the HTML (default: a temp file)")
    p.add_argument("--no-open", action="store_true",
                   help="with --html, write the file but do not open it")
    p.add_argument("--max-chars", type=int, default=0,
                   help="truncate terminal responses to N chars (0 = no limit)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    benchmark = infer_benchmark(args.sample_id, args.benchmark)

    try:
        meta = load_sample_meta(args.sample_id, benchmark)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    if meta is None:
        print(f"[warning] id '{args.sample_id}' not found in "
              f"{_combined_path(benchmark)} — relying on results files for metadata.",
              file=sys.stderr)

    resp_sets: List[ResponseSet] = []
    for rp in args.results:
        if not rp.exists():
            print(f"[warning] results file not found: {rp}", file=sys.stderr)
            continue
        try:
            rs = parse_results_file(rp, args.sample_id)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warning] could not parse {rp}: {e}", file=sys.stderr)
            continue
        if rs is None:
            print(f"[warning] id '{args.sample_id}' not found in {rp}", file=sys.stderr)
        else:
            resp_sets.append(rs)

    print_summary(args.sample_id, benchmark, meta, resp_sets, args.max_chars)

    image_path = meta.get("image_path") if meta else None
    if args.open_image:
        if image_path and Path(image_path).exists():
            print(f"\nOpening image: {image_path}")
            open_in_os(Path(image_path))
        else:
            print("[error] cannot open image — path unresolved or missing.",
                  file=sys.stderr)

    if args.html:
        doc = build_html(args.sample_id, benchmark, meta, resp_sets)
        if args.html_out:
            out = args.html_out
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            fd = tempfile.NamedTemporaryFile(
                prefix=f"review_{args.sample_id}_", suffix=".html", delete=False)
            out = Path(fd.name)
            fd.close()
        out.write_text(doc, encoding="utf-8")
        print(f"\nWrote HTML: {out}")
        if not args.no_open:
            open_in_os(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
