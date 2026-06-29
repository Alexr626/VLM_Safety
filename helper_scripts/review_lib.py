"""Shared helpers for qualitative response review (image + model outputs).

Used by ``review_responses.py`` (single-id CLI) and ``sample_responses.py``
(batch sampling for a run_date).
"""

from __future__ import annotations

import base64
import html
import json
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DATA_SUBDIR_OVERRIDES = {"mmhal_bench": "mmhal-bench"}
_RUN_DATE_RE = re.compile(r"evaluation/results/(\d{4}-\d{2}-\d{2})")


def infer_run_date(paths: List[Path], output_dir: str = "evaluation/results") -> Optional[str]:
    """Infer YYYY-MM-DD from paths under ``evaluation/results/{date}/``."""
    for p in paths:
        s = str(p.as_posix())
        m = _RUN_DATE_RE.search(s)
        if m:
            return m.group(1)
        # Also accept explicit output_dir prefix.
        prefix = f"{output_dir.rstrip('/')}/"
        if prefix in s:
            rest = s.split(prefix, 1)[1]
            date = rest.split("/", 1)[0]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                return date
    return None


def review_dir(run_date: str, output_dir: str = "evaluation/results") -> Path:
    """Ad-hoc per-id HTML from ``review_responses.py``."""
    return Path(output_dir) / run_date / "_review"


def samples_dir(run_date: str, output_dir: str = "evaluation/results") -> Path:
    """Batch sample bundle root from ``sample_responses.py``."""
    return Path(output_dir) / run_date / "_samples"


def sample_bundle_dir(run_date: str, model_short: str, benchmark: str,
                      output_dir: str = "evaluation/results") -> Path:
    """Per-(model, benchmark) sample bundle directory."""
    return samples_dir(run_date, output_dir) / model_short / benchmark


def bundle_stem(model_short: str, benchmark: str) -> str:
    """Filename prefix for a (model, benchmark) sample bundle."""
    return f"{model_short}_{benchmark}"


def bundle_json_path(run_date: str, model_short: str, benchmark: str,
                     output_dir: str = "evaluation/results") -> Path:
    return sample_bundle_dir(run_date, model_short, benchmark, output_dir) / (
        f"{bundle_stem(model_short, benchmark)}_response_samples.json"
    )


def bundle_md_path(run_date: str, model_short: str, benchmark: str,
                   output_dir: str = "evaluation/results") -> Path:
    return sample_bundle_dir(run_date, model_short, benchmark, output_dir) / (
        f"{bundle_stem(model_short, benchmark)}_response_samples.md"
    )


def bundle_html_path(run_date: str, model_short: str, benchmark: str,
                     output_dir: str = "evaluation/results") -> Path:
    return sample_bundle_dir(run_date, model_short, benchmark, output_dir) / (
        f"{bundle_stem(model_short, benchmark)}_review.html"
    )


def default_gallery_path(run_date: str, output_dir: str = "evaluation/results",
                         *, batch: bool = False, model_short: str = "",
                         benchmark: str = "") -> Path:
    if batch and model_short and benchmark:
        return bundle_html_path(run_date, model_short, benchmark, output_dir)
    if batch:
        return samples_dir(run_date, output_dir) / "review.html"
    return review_dir(run_date, output_dir) / "gallery.html"


def default_html_path(sample_id: str, run_date: str, output_dir: str = "evaluation/results",
                      *, batch: bool = False, model_short: str = "",
                      benchmark: str = "") -> Path:
    """Legacy per-sample path (prefer ``default_gallery_path``)."""
    if batch and model_short and benchmark:
        return (samples_dir(run_date, output_dir) / "review"
                / model_short / benchmark / f"{sample_id}.html")
    return review_dir(run_date, output_dir) / f"{sample_id}.html"


def _data_subdir(benchmark: str) -> str:
    return _DATA_SUBDIR_OVERRIDES.get(benchmark, benchmark)


def infer_benchmark(sample_id: str, override: Optional[str]) -> str:
    if override:
        return override
    return sample_id.split("_", 1)[0].lower()


def combined_path(benchmark: str) -> Path:
    return PROJECT_ROOT / "data" / _data_subdir(benchmark) / "combined.json"


def load_sample_meta(sample_id: str, benchmark: str) -> Optional[Dict[str, Any]]:
    path = combined_path(benchmark)
    if not path.exists():
        raise FileNotFoundError(
            f"No combined.json for benchmark '{benchmark}' at {path}. "
            f"Pass --benchmark, or run the benchmark download script."
        )
    with open(path) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else list(data.values())
    for e in entries:
        if e.get("id") == sample_id:
            return e
    return None


class ResponseSet:
    def __init__(self, source_label: str, source_path: Path):
        self.source_label = source_label
        self.source_path = source_path
        self.question: Optional[str] = None
        self.ground_truth: Optional[str] = None
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


def _sorted_beta_keys(keys) -> List[str]:
    def _key(k: str):
        try:
            return (0, float(k))
        except (TypeError, ValueError):
            return (1, k)
    return sorted(keys, key=_key, reverse=True)


def _parse_sweep(obj: Dict[str, Any], sample_id: str, path: Path) -> Optional[ResponseSet]:
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
        for beta in _sorted_beta_keys((rec.get("by_beta") or {}).keys()):
            v = rec["by_beta"][beta]
            if isinstance(v, dict):
                rs.add(f"beta={beta}", v.get("response", ""), v.get("class"))
            else:
                rs.add(f"beta={beta}", str(v))
        if "decode_only_beta_max" in rec:
            rs.add("decode-only @ beta_max", rec["decode_only_beta_max"])
        if "skip_pos0_beta_max" in rec:
            rs.add("skip-pos0 @ beta_max", rec["skip_pos0_beta_max"])
        return rs if rs else None

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


def _parse_responses_json(records: List[dict], sample_id: str, path: Path,
                          label: Optional[str] = None) -> Optional[ResponseSet]:
    rec = next((r for r in records if r.get("id") == sample_id), None)
    if rec is None:
        return None
    rs = ResponseSet(label or path.parent.name, path)
    rs.question = rec.get("question")
    rs.ground_truth = rec.get("ground_truth")
    note = rec.get("intervention") or rec.get("task")
    rs.add("response", rec.get("response", ""), note)
    return rs


def parse_results_file(path: Path, sample_id: str,
                       label: Optional[str] = None) -> Optional[ResponseSet]:
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if "per_sample" in obj or "changed_examples_by_beta" in obj or "metrics_by_beta" in obj:
            return _parse_sweep(obj, sample_id, path)
        if isinstance(obj.get("records"), list):
            return _parse_responses_json(obj["records"], sample_id, path, label)
        if sample_id in obj and isinstance(obj[sample_id], dict):
            return _parse_responses_json([{"id": sample_id, **obj[sample_id]}],
                                         sample_id, path, label)
    if isinstance(obj, list):
        return _parse_responses_json(obj, sample_id, path, label)
    return None


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


def truncate_text(text: str, max_chars: int) -> str:
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
            body = truncate_text(text.strip() or "<empty>", max_chars)
            for line in body.splitlines() or ["<empty>"]:
                print(f"      {line}")
    print("\n" + sep)


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


_SHARED_CSS = """
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; }
  .page-hdr { padding: 20px 28px 8px; border-bottom: 1px solid #e0e0e0; }
  .page-hdr h1 { font-size: 20px; margin: 0 0 4px; }
  .page-sub { color: #9aa0a6; font-size: 13px; margin: 0; }
  nav.toc { padding: 12px 28px 20px; background: #f7f8fa; border-bottom: 1px solid #e0e0e0; }
  nav.toc h2 { font-size: 13px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .04em; color: #666; }
  nav.toc ul { margin: 0; padding-left: 18px; columns: 2; column-gap: 24px; }
  nav.toc a { color: #1a56db; text-decoration: none; font-size: 13px; }
  nav.toc a:hover { text-decoration: underline; }
  .sample-block { border-bottom: 2px solid #e0e0e0; padding: 24px 0; }
  .sample-block .group { font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
                         color: #888; margin: 0 0 8px 28px; }
  .wrap { display: flex; align-items: flex-start; gap: 0; }
  .pane-img { flex: 0 0 42%; max-width: 42%; padding: 0 20px 0 28px; position: sticky; top: 0;
              align-self: flex-start; background: #0f1115; padding-top: 16px; padding-bottom: 16px; }
  .pane-img img { max-width: 100%; height: auto; border-radius: 8px;
                  box-shadow: 0 4px 24px rgba(0,0,0,.5); }
  .pane-txt { flex: 1 1 58%; padding: 0 28px 16px 12px; min-width: 0; }
  h1.sample-id { font-size: 18px; margin: 0 0 4px; }
  .meta { margin: 0 0 12px; color: #9aa0a6; font-size: 13px; word-break: break-all; }
  .qa { background: #f4f5f7; border-radius: 8px; padding: 12px 14px; margin: 0 0 16px; }
  .qa b { color: #555; }
  .resultset { border-top: 2px solid #e0e0e0; padding-top: 12px; margin-top: 16px; }
  .resultset h2 { font-size: 15px; margin: 0 0 2px; }
  .src { margin: 0 0 12px; color: #9aa0a6; font-size: 12px; word-break: break-all; }
  .resp { border: 1px solid #e2e2e2; border-radius: 8px; margin: 0 0 10px; overflow: hidden; }
  .resp-head { display: flex; align-items: center; gap: 8px; background: #f7f8fa;
               padding: 6px 10px; font-size: 13px; }
  .lbl { font-weight: 600; }
  .note { color: #8a6d3b; background: #fcf3d9; border-radius: 4px; padding: 1px 6px; font-size: 11px; }
  .resp-body { padding: 10px 12px; white-space: pre-wrap; }
  .yn { font-size: 11px; font-weight: 700; border-radius: 4px; padding: 1px 7px; color: #fff; }
  .yn-yes { background: #1a7f37; }
  .yn-no { background: #b42318; }
  .yn-unk { background: #888; }
  .missing { color: #b42318; }
  @media (prefers-color-scheme: dark) {
    body { background: #15171c; color: #e6e6e6; }
    .page-hdr, .sample-block { border-color: #2c2f36; }
    nav.toc { background: #1a1d24; border-color: #2c2f36; }
    .qa { background: #1f2228; } .qa b { color: #aaa; }
    .resp { border-color: #2c2f36; } .resp-head { background: #1f2228; }
    .resultset { border-color: #2c2f36; }
  }
  @media (max-width: 900px) {
    .wrap { flex-direction: column; }
    .pane-img { position: static; flex: none; max-width: 100%; width: 100%; }
    nav.toc ul { columns: 1; }
  }
"""


def _img_block(meta: Optional[Dict[str, Any]], sample_id: str) -> str:
    image_path = meta.get("image_path") if meta else None
    if not image_path:
        return "<p class='missing'>image unresolved</p>"
    uri = _img_data_uri(Path(image_path))
    if uri:
        return f"<img src='{uri}' alt='{html.escape(sample_id)}'>"
    return (f"<p class='missing'>image file not found:<br>"
            f"{html.escape(str(image_path))}</p>")


def _qa_block(meta: Optional[Dict[str, Any]], resp_sets: List[ResponseSet]) -> Tuple[str, str, str]:
    question = (meta.get("text") if meta else None) or \
        next((r.question for r in resp_sets if r.question), None) or "<unknown>"
    label = (meta.get("label") if meta else None) or \
        next((r.ground_truth for r in resp_sets if r.ground_truth), None) or "<unknown>"
    gt_badge = _yn_badge(label)
    return question, label, gt_badge


def _resp_sections_html(resp_sets: List[ResponseSet]) -> str:
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
                        "No results for this sample.</p></section>")
    return "".join(sections)


@dataclass
class GalleryPanel:
    sample_id: str
    benchmark: str
    meta: Optional[Dict[str, Any]]
    resp_sets: List[ResponseSet] = field(default_factory=list)
    section_heading: str = ""


def _panel_html(panel: GalleryPanel) -> str:
    anchor = html.escape(panel.sample_id, quote=True)
    question, label, gt_badge = _qa_block(panel.meta, panel.resp_sets)
    image_path = (panel.meta or {}).get("image_path", "")
    group = (f"<p class='group'>{html.escape(panel.section_heading)}</p>"
             if panel.section_heading else "")
    return f"""
<section class="sample-block" id="{anchor}">
  {group}
  <div class="wrap">
    <div class="pane-img">{_img_block(panel.meta, panel.sample_id)}</div>
    <div class="pane-txt">
      <h1 class="sample-id">{html.escape(panel.sample_id)}</h1>
      <p class="meta">benchmark: {html.escape(panel.benchmark)} &nbsp;·&nbsp; {html.escape(str(image_path))}</p>
      <div class="qa">
        <div><b>Q:</b> {html.escape(question)}</div>
        <div><b>Ground truth:</b> {gt_badge} {html.escape(label)}</div>
      </div>
      {_resp_sections_html(panel.resp_sets)}
    </div>
  </div>
</section>"""


def build_gallery_html(title: str, subtitle: str,
                       panels: List[GalleryPanel]) -> str:
    toc_items = "".join(
        f'<li><a href="#{html.escape(p.sample_id, quote=True)}">'
        f'{html.escape(p.section_heading + " — " if p.section_heading else "")}'
        f'{html.escape(p.sample_id)}</a></li>'
        for p in panels
    )
    body = "".join(_panel_html(p) for p in panels)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_SHARED_CSS}</style></head>
<body>
<header class="page-hdr">
  <h1>{html.escape(title)}</h1>
  <p class="page-sub">{html.escape(subtitle)}</p>
</header>
<nav class="toc"><h2>Samples</h2><ul>{toc_items}</ul></nav>
{body}
</body></html>"""


def write_gallery_html(title: str, subtitle: str, panels: List[GalleryPanel],
                       out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_gallery_html(title, subtitle, panels), encoding="utf-8")
    return out_path


def build_html(sample_id: str, benchmark: str, meta: Optional[Dict[str, Any]],
               resp_sets: List[ResponseSet]) -> str:
    """Single-sample page (legacy); prefer ``build_gallery_html``."""
    return build_gallery_html(
        sample_id, f"benchmark: {benchmark}",
        [GalleryPanel(sample_id, benchmark, meta, resp_sets)],
    )


def write_html(sample_id: str, benchmark: str, meta: Optional[Dict[str, Any]],
               resp_sets: List[ResponseSet], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(sample_id, benchmark, meta, resp_sets), encoding="utf-8")
    return out_path


def collect_response_sets(sample_id: str, result_paths: List[Path],
                          labels: Optional[List[str]] = None) -> List[ResponseSet]:
    out: List[ResponseSet] = []
    for i, rp in enumerate(result_paths):
        if not rp.exists():
            continue
        lbl = labels[i] if labels and i < len(labels) else None
        try:
            rs = parse_results_file(rp, sample_id, label=lbl)
        except (json.JSONDecodeError, OSError):
            continue
        if rs:
            if lbl:
                rs.source_label = lbl
            out.append(rs)
    return out


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
        webbrowser.open(path.as_uri())
