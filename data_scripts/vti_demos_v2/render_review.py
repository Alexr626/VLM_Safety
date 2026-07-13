#!/usr/bin/env python3
"""HTML review galleries for demos_v2 pipeline stages."""

from __future__ import annotations

import argparse
import difflib
import html
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG.parent))
sys.path.insert(0, str(_ROOT / "helper_scripts"))

from review_lib import _img_data_uri, open_in_os  # noqa: E402
from vti_demos_v2.images import image_path  # noqa: E402
from vti_demos_v2.io_utils import read_jsonl  # noqa: E402
from src.paths import coco_train2014_dir, vti_demos_v2_dir, vti_demos_v2_path  # noqa: E402

_CSS = """
  body { margin: 0; font: 14px/1.45 -apple-system, Segoe UI, Roboto, sans-serif; }
  .hdr { padding: 16px 24px; border-bottom: 1px solid #ddd; }
  .hdr h1 { margin: 0 0 4px; font-size: 18px; }
  .sub { color: #888; margin: 0; font-size: 13px; }
  nav.toc { padding: 10px 24px; background: #f7f8fa; border-bottom: 1px solid #ddd; }
  nav.toc ul { columns: 3; margin: 0; padding-left: 18px; }
  .block { border-bottom: 2px solid #e0e0e0; padding: 18px 0; }
  .wrap { display: flex; gap: 16px; padding: 0 24px; }
  .img { flex: 0 0 38%; background: #111; padding: 12px; border-radius: 8px; }
  .img img { max-width: 100%; height: auto; border-radius: 6px; }
  .txt { flex: 1; min-width: 0; }
  h2 { font-size: 16px; margin: 0 0 6px; }
  .meta { color: #888; font-size: 12px; word-break: break-all; }
  .box { background: #f4f5f7; border-radius: 8px; padding: 10px 12px; margin: 8px 0; }
  .pass { color: #1a7f37; font-weight: 700; }
  .fail { color: #b42318; font-weight: 700; }
  .cap { white-space: pre-wrap; margin: 6px 0; }
  .hl { background: #fff3bf; padding: 0 2px; border-radius: 2px; }
  .ins { background: #d3f9d8; }
  .del { background: #ffe3e3; text-decoration: line-through; }
  .var { border: 1px solid #e2e2e2; border-radius: 8px; margin: 8px 0; overflow: hidden; }
  .var h3 { margin: 0; padding: 6px 10px; background: #f7f8fa; font-size: 13px; }
  .var .body { padding: 8px 10px; }
  .missing { color: #b42318; }
"""


def _img_html(sid: str, filename: str | None = None) -> str:
    path = coco_train2014_dir() / (filename or f"COCO_train2014_{sid}.jpg")
    if not path.is_file():
        # try via helper
        path = image_path(sid)
    uri = _img_data_uri(path) if path.is_file() else None
    if uri:
        return f"<img src='{uri}' alt='{html.escape(sid)}'>"
    return f"<p class='missing'>image missing: {html.escape(str(path))}</p>"


def _highlight_spans(caption: str, spans: dict) -> str:
    # naive sequential highlight of span substrings
    escaped = html.escape(caption)
    for key, span in (spans or {}).items():
        if not span:
            continue
        esc_span = html.escape(span)
        if esc_span in escaped:
            escaped = escaped.replace(
                esc_span,
                f"<span class='hl' title='{html.escape(key)}'>{esc_span}</span>",
                1,
            )
    return escaped


def _diff_html(a: str, b: str) -> str:
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split())
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(" ".join(a.split()[i1:i2])))
        elif tag == "delete":
            parts.append(
                "<span class='del'>" + html.escape(" ".join(a.split()[i1:i2])) + "</span>"
            )
        elif tag == "insert":
            parts.append(
                "<span class='ins'>" + html.escape(" ".join(b.split()[j1:j2])) + "</span>"
            )
        elif tag == "replace":
            parts.append(
                "<span class='del'>" + html.escape(" ".join(a.split()[i1:i2])) + "</span> "
                "<span class='ins'>" + html.escape(" ".join(b.split()[j1:j2])) + "</span>"
            )
    return " ".join(p for p in parts if p)


def _page(title: str, subtitle: str, blocks: list[str], toc: list[tuple[str, str]]) -> str:
    toc_html = "".join(
        f'<li><a href="#{html.escape(i, quote=True)}">{html.escape(label)}</a></li>'
        for i, label in toc
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head><body>
<header class="hdr"><h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(subtitle)}</p></header>
<nav class="toc"><ul>{toc_html}</ul></nav>
{''.join(blocks)}
</body></html>"""


def render_stage0(rows: list[dict]) -> str:
    blocks, toc = [], []
    for r in rows:
        sid = r["id"]
        toc.append((sid, sid))
        ca, ra = r["counting_anchor"], r["relation_anchor"]
        blocks.append(f"""
<section class="block" id="{html.escape(sid, quote=True)}">
  <div class="wrap">
    <div class="img">{_img_html(sid, r.get('image'))}</div>
    <div class="txt">
      <h2>{html.escape(sid)}</h2>
      <p class="meta">score={r.get('score')} · {html.escape(r.get('image',''))}</p>
      <div class="box">
        <div><b>counting:</b> {html.escape(str(ca))}</div>
        <div><b>relation:</b> {html.escape(str(ra))}</div>
        <div><b>distractors:</b> {html.escape(str(r.get('distractor_candidates')))}</div>
        <div><b>present:</b> {html.escape(str(r.get('present_categories')))}</div>
      </div>
    </div>
  </div>
</section>""")
    return _page("demos_v2 · stage 0 candidates", f"{len(rows)} candidates", blocks, toc)


def render_stage1(rows: list[dict], rejected: list[dict]) -> str:
    blocks, toc = [], []
    for label, group, ok in (("PASS", rows, True), ("REJECT", rejected, False)):
        for r in group:
            sid = r["id"]
            toc.append((f"{'p' if ok else 'r'}-{sid}", f"{'[ok] ' if ok else '[rej] '}{sid}"))
            s1 = r.get("stage1") or {}
            attr = r.get("attribute") or (s1.get("attribute") or {})
            blocks.append(f"""
<section class="block" id="{'p' if ok else 'r'}-{html.escape(sid, quote=True)}">
  <div class="wrap">
    <div class="img">{_img_html(sid, r.get('image'))}</div>
    <div class="txt">
      <h2><span class="{'pass' if ok else 'fail'}">{label}</span> {html.escape(sid)}</h2>
      <div class="box">
        <div><b>distractor:</b> {html.escape(str(r.get('distractor') or (s1.get('distractor') or {}).get('choice')))}</div>
        <div><b>attribute:</b> {html.escape(str(attr))}</div>
        <div><b>reject:</b> {html.escape(str(r.get('reject_reason','')))}</div>
        <pre class="cap">{html.escape(str(s1)[:2000])}</pre>
      </div>
    </div>
  </div>
</section>""")
    return _page("demos_v2 · stage 1", f"{len(rows)} pass / {len(rejected)} reject", blocks, toc)


def render_stage2(rows: list[dict], rejected: list[dict]) -> str:
    blocks, toc = [], []
    for label, group, ok in (("PASS", rows, True), ("REJECT", rejected, False)):
        for r in group:
            sid = r["id"]
            toc.append((f"{'p' if ok else 'r'}-{sid}", f"{'[ok] ' if ok else '[rej] '}{sid}"))
            cap = r.get("value") or ""
            spans = r.get("spans") or {}
            body = _highlight_spans(cap, spans) if cap else html.escape(str(r.get("errors")))
            blocks.append(f"""
<section class="block" id="{'p' if ok else 'r'}-{html.escape(sid, quote=True)}">
  <div class="wrap">
    <div class="img">{_img_html(sid, r.get('image'))}</div>
    <div class="txt">
      <h2><span class="{'pass' if ok else 'fail'}">{label}</span> {html.escape(sid)}</h2>
      <div class="box"><div class="cap">{body}</div>
      <div class="meta">spans={html.escape(str(spans))}</div></div>
    </div>
  </div>
</section>""")
    return _page("demos_v2 · stage 2 captions", f"{len(rows)} pass / {len(rejected)} reject", blocks, toc)


def render_variants(rows: list[dict], title: str) -> str:
    blocks, toc = [], []
    for r in rows:
        sid = r["id"]
        toc.append((sid, sid))
        truthful = r.get("value", "")
        vars_html = []
        for dim, variant in (r.get("h_values") or {}).items():
            vars_html.append(
                f"<div class='var'><h3>{html.escape(dim)}</h3>"
                f"<div class='body'>{_diff_html(truthful, variant)}</div></div>"
            )
        rej = r.get("reject_reason")
        badge = f"<span class='fail'>REJECT {html.escape(str(rej))}</span> " if rej else ""
        blocks.append(f"""
<section class="block" id="{html.escape(sid, quote=True)}">
  <div class="wrap">
    <div class="img">{_img_html(sid, r.get('image'))}</div>
    <div class="txt">
      <h2>{badge}{html.escape(sid)}</h2>
      <div class="box"><b>truthful</b><div class="cap">{html.escape(truthful)}</div></div>
      {''.join(vars_html)}
      <div class="box"><b>anchors</b><pre class="cap">{html.escape(str(r.get('anchors',''))[:1500])}</pre></div>
    </div>
  </div>
</section>""")
    return _page(title, f"{len(rows)} records", blocks, toc)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=["0", "1", "2", "3", "4", "final"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of records shown (useful for large stage-0)")
    p.add_argument("--open", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    review = v2 / "_review"
    review.mkdir(parents=True, exist_ok=True)

    if args.stage == "0":
        rows = read_jsonl(v2 / "stage0_candidates.jsonl")
        if args.limit:
            rows = rows[: args.limit]
        html_doc = render_stage0(rows)
        out = args.out or (review / "stage0_review.html")
    elif args.stage == "1":
        html_doc = render_stage1(
            read_jsonl(v2 / "stage1_verified.jsonl"),
            read_jsonl(v2 / "stage1_rejected.jsonl"),
        )
        out = args.out or (review / "stage1_review.html")
    elif args.stage == "2":
        html_doc = render_stage2(
            read_jsonl(v2 / "stage2_captions.jsonl"),
            read_jsonl(v2 / "stage2_rejected.jsonl"),
        )
        out = args.out or (review / "stage2_review.html")
    elif args.stage == "3":
        rows = read_jsonl(v2 / "stage3_variants.jsonl")
        html_doc = render_variants(rows, "demos_v2 · stage 3 variants")
        out = args.out or (review / "stage3_review.html")
    elif args.stage == "4":
        rows = read_jsonl(v2 / "stage4_verdicts.jsonl") + read_jsonl(
            v2 / "stage4_rejected.jsonl"
        )
        html_doc = render_variants(rows, "demos_v2 · stage 4 faithfulness")
        out = args.out or (review / "stage4_review.html")
    else:
        rows = read_jsonl(vti_demos_v2_path())
        html_doc = render_variants(rows, "demos_v2 · final")
        out = args.out or (review / "final_review.html")

    out.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out}")
    if args.open:
        open_in_os(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
