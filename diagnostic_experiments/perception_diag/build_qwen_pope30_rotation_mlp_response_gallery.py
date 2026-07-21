#!/usr/bin/env python3
"""HTML gallery: Qwen2.5-VL POPE-30 rotation @ mlp responses.

Condition: leading clause toward no (``assertive_toward_no``) only.

Per item: no-intervention baseline, then rotation @ mlp at β∈{0.2,0.5,0.9}
for layer windows 5–14 and 18–27 (Qwen top window; there is no 16–27 cell).

Layout: each POPE item (image + prompt) once; all responses listed under it.

Usage::

    python diagnostic_experiments/perception_diag/build_qwen_pope30_rotation_mlp_response_gallery.py
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "helper_scripts"))

from diagnostic_experiments.perception_diag.build_windowed_steering_summary import (  # noqa: E402
    load_manifest,
)
from review_lib import _img_data_uri, _yn_badge  # noqa: E402
from src.paths import perception_dump_dir, project_root  # noqa: E402

MODEL_SHORT = "qwen2.5-vl-7b-instruct"
CONDITION = "assertive_toward_no"
CONDITION_LABEL = "leading clause toward no"

# Ordered: window 5–14 across betas, then window 18–27 across betas.
CELLS: List[Tuple[str, str]] = [
    ("rotation_mlp_0.2_layers_5_14", "rotation @ mlp · β=0.2 · layers 5–14"),
    ("rotation_mlp_0.5_layers_5_14", "rotation @ mlp · β=0.5 · layers 5–14"),
    ("rotation_mlp_0.9_layers_5_14", "rotation @ mlp · β=0.9 · layers 5–14"),
    ("rotation_mlp_0.2_layers_18_27", "rotation @ mlp · β=0.2 · layers 18–27"),
    ("rotation_mlp_0.5_layers_18_27", "rotation @ mlp · β=0.5 · layers 18–27"),
    ("rotation_mlp_0.9_layers_18_27", "rotation @ mlp · β=0.9 · layers 18–27"),
]

QTYPE_ORDER = ("adversarial", "popular", "random")

CSS = """
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif;
         color: #1a1a1a; background: #fff; }
  .page-hdr { padding: 20px 28px 12px; border-bottom: 1px solid #ddd; }
  .page-hdr h1 { font-size: 20px; margin: 0 0 6px; }
  .page-sub { color: #666; font-size: 13px; margin: 0; max-width: 960px; }
  nav.toc { padding: 14px 28px 18px; background: #f6f7f9; border-bottom: 1px solid #ddd; }
  nav.toc h2 { font-size: 12px; margin: 12px 0 6px; text-transform: uppercase;
               letter-spacing: .04em; color: #666; }
  nav.toc h2:first-child { margin-top: 0; }
  nav.toc ul { margin: 0; padding-left: 18px; columns: 2; column-gap: 24px; }
  nav.toc a { color: #1a56db; text-decoration: none; font-size: 13px; }
  nav.toc a:hover { text-decoration: underline; }
  .qtype-section { border-bottom: 3px solid #222; }
  .qtype-hdr { padding: 18px 28px 8px; background: #eef2f7;
               border-bottom: 1px solid #ccd3dd; position: sticky; top: 0; z-index: 2; }
  .qtype-hdr h2 { margin: 0; font-size: 17px; }
  .item { display: flex; gap: 0; border-bottom: 1px solid #e4e4e4; padding: 18px 0; }
  .pane-img { flex: 0 0 32%; max-width: 32%; padding: 0 18px 0 28px; }
  .pane-img img { max-width: 100%; height: auto; border-radius: 6px;
                  box-shadow: 0 2px 12px rgba(0,0,0,.25); background: #111; }
  .pane-txt { flex: 1 1 68%; padding: 0 28px 0 8px; min-width: 0; }
  .item-id { font-size: 16px; font-weight: 650; margin: 0 0 4px; }
  .meta { margin: 0 0 10px; color: #777; font-size: 12px; word-break: break-all; }
  .qa { background: #f4f5f7; border-radius: 8px; padding: 10px 12px; margin: 0 0 12px; }
  .qa b { color: #555; }
  .resp-stack h4 { margin: 14px 0 8px; font-size: 12px; text-transform: uppercase;
                   letter-spacing: .04em; color: #666; }
  .resp { border: 1px solid #ddd; border-radius: 8px; margin: 0 0 8px; overflow: hidden; }
  .resp.baseline { border-left: 4px solid #6b7280; }
  .resp.steered { border-left: 4px solid #2563eb; }
  .resp.changed { border-left-color: #b42318; }
  .resp-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
               background: #f7f8fa; padding: 6px 10px; font-size: 13px; }
  .lbl { font-weight: 650; }
  .note { color: #8a6d3b; background: #fcf3d9; border-radius: 4px;
          padding: 1px 6px; font-size: 11px; }
  .scores { color: #555; font-size: 12px; margin-left: auto; }
  .resp-body { padding: 10px 12px; white-space: pre-wrap; font-family: ui-monospace,
               SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
  .yn { font-size: 11px; font-weight: 700; border-radius: 4px; padding: 1px 7px; color: #fff; }
  .yn-yes { background: #1a7f37; }
  .yn-no { background: #b42318; }
  .yn-unk { background: #888; }
  .missing { color: #b42318; }
  @media (max-width: 900px) {
    .item { flex-direction: column; }
    .pane-img { max-width: 100%; width: 100%; flex: none; }
    nav.toc ul { columns: 1; }
  }
"""


def _load_augmented(path: Path) -> Dict[str, dict]:
    items: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            items[row["item_id"]] = row
    return items


def _prompt_for(item: dict, condition_id: str) -> str:
    for v in item.get("variants", []):
        if v["condition_id"] == condition_id:
            return v.get("prompt") or ""
    return ""


def _fmt_score(row: dict) -> str:
    parts = []
    py = row.get("score_p_yes_raw")
    pn = row.get("score_p_no_raw")
    if py is not None:
        parts.append(f"P(yes)={float(py):.3f}")
    if pn is not None:
        parts.append(f"P(no)={float(pn):.3f}")
    if row.get("degeneracy_flag"):
        parts.append("degeneracy_flag=True")
    if row.get("truncated"):
        parts.append("truncated")
    return " · ".join(parts)


def _outcome_badge(outcome: Optional[str]) -> str:
    if outcome in ("yes", "no"):
        return _yn_badge(outcome)
    return '<span class="yn yn-unk">unparseable</span>'


def _response_card(
    *,
    kind: str,
    label: str,
    row: dict,
    changed: bool = False,
) -> str:
    classes = f"resp {kind}" + (" changed" if changed else "")
    notes = []
    if changed:
        notes.append("parsed outcome differs from baseline")
    if row.get("degeneracy_flag"):
        notes.append("degeneracy_flag")
    note_html = "".join(
        f'<span class="note">{html.escape(n)}</span>' for n in notes
    )
    body = html.escape((row.get("response") or "").strip()) or "<em>&lt;empty&gt;</em>"
    return f"""
<div class="{classes}">
  <div class="resp-head">
    {_outcome_badge(row.get("parsed_outcome"))}
    <span class="lbl">{html.escape(label)}</span>
    {note_html}
    <span class="scores">{html.escape(_fmt_score(row))}</span>
  </div>
  <div class="resp-body">{body}</div>
</div>"""


def _item_block(
    *,
    item_id: str,
    item: dict,
    prompt: str,
    baseline_row: dict,
    steered_rows: List[Tuple[str, dict]],
    img_cache: Dict[str, str],
) -> str:
    image_path = item.get("image_path") or ""
    if image_path not in img_cache:
        uri = _img_data_uri(Path(image_path)) if image_path else None
        img_cache[image_path] = uri or ""
    uri = img_cache[image_path]
    if uri:
        img_html = f"<img src='{uri}' alt='{html.escape(item_id)}'>"
    else:
        img_html = (
            f"<p class='missing'>image not found:<br>"
            f"{html.escape(str(image_path))}</p>"
        )

    b_out = baseline_row.get("parsed_outcome")
    cards = [
        _response_card(
            kind="baseline",
            label="no-intervention baseline",
            row=baseline_row,
        )
    ]
    for label, srow in steered_rows:
        cards.append(
            _response_card(
                kind="steered",
                label=label,
                row=srow,
                changed=(b_out != srow.get("parsed_outcome")),
            )
        )

    return f"""
<article class="item" id="{html.escape(item_id, quote=True)}">
  <div class="pane-img">{img_html}</div>
  <div class="pane-txt">
    <h3 class="item-id">{html.escape(item_id)}</h3>
    <p class="meta">gold={html.escape(str(item.get('gold')))} ·
      qtype={html.escape(str(item.get('qtype')))} ·
      {html.escape(str(image_path))}</p>
    <div class="qa">
      <div><b>Prompt ({html.escape(CONDITION_LABEL)}):</b>
        {html.escape(prompt)}</div>
      <div><b>Ground truth:</b> {_yn_badge(str(item.get('gold') or ''))}
        {html.escape(str(item.get('gold')))}</div>
    </div>
    <div class="resp-stack">
      <h4>Responses (baseline, then rotation @ mlp · layers 5–14 / 18–27 · all β)</h4>
      {''.join(cards)}
    </div>
  </div>
</article>"""


def build_gallery(
    *,
    items: Dict[str, dict],
    baseline_man: Dict[Tuple[str, str], dict],
    steered_root: Path,
) -> str:
    cell_mans: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for cell_id, _title in CELLS:
        man = load_manifest(steered_root / cell_id)
        if not man:
            raise FileNotFoundError(f"missing steered cell: {steered_root / cell_id}")
        cell_mans[cell_id] = man

    by_qtype: Dict[str, List[str]] = defaultdict(list)
    for iid, item in items.items():
        by_qtype[str(item.get("qtype") or "unknown")].append(iid)
    for qtype in by_qtype:
        by_qtype[qtype].sort()

    qtypes = [q for q in QTYPE_ORDER if q in by_qtype] + sorted(
        q for q in by_qtype if q not in QTYPE_ORDER
    )

    img_cache: Dict[str, str] = {}
    toc_parts: List[str] = []
    body_parts: List[str] = []

    for qtype in qtypes:
        ids = by_qtype[qtype]
        toc_parts.append(f"<h2>POPE qtype: {html.escape(qtype)}</h2><ul>")
        blocks: List[str] = []
        for iid in ids:
            item = items[iid]
            prompt = _prompt_for(item, CONDITION)
            baseline_row = baseline_man[(iid, CONDITION)]
            steered_rows = [
                (title, cell_mans[cell_id][(iid, CONDITION)])
                for cell_id, title in CELLS
            ]
            toc_parts.append(
                f'<li><a href="#{html.escape(iid, quote=True)}">'
                f"{html.escape(iid)}</a></li>"
            )
            blocks.append(
                _item_block(
                    item_id=iid,
                    item=item,
                    prompt=prompt,
                    baseline_row=baseline_row,
                    steered_rows=steered_rows,
                    img_cache=img_cache,
                )
            )
        toc_parts.append("</ul>")
        body_parts.append(
            f"""
<section class="qtype-section" id="qtype-{html.escape(qtype, quote=True)}">
  <div class="qtype-hdr"><h2>POPE qtype: {html.escape(qtype)} ({len(ids)} items)</h2></div>
  {''.join(blocks)}
</section>"""
        )

    title = (
        "Qwen2.5-VL POPE-30: baseline vs rotation @ mlp "
        "(layers 5–14 and 18–27, all β), leading clause toward no"
    )
    subtitle = (
        "Condition fixed to assertive_toward_no. Each POPE item (image + prompt) "
        "is shown once. Under it: no-intervention baseline, then rotation @ mlp "
        "at β=0.2 / 0.5 / 0.9 for layer windows 5–14 and 18–27 "
        "(Qwen’s top sliding window; there is no 16–27 cell in this grid). "
        "Red left border marks a parsed-outcome change vs baseline."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head>
<body>
<header class="page-hdr">
  <h1>{html.escape(title)}</h1>
  <p class="page-sub">{html.escape(subtitle)}</p>
</header>
<nav class="toc">
  {''.join(toc_parts)}
</nav>
{''.join(body_parts)}
</body></html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path")
    args = parser.parse_args(argv)

    root = project_root()
    out = args.out or (
        root
        / "diagnostic_experiments"
        / "perception_diag"
        / "windowed_steering_summary"
        / "qwen_pope30_leading_toward_no_rotation_mlp_layers_5_14_and_18_27_response_gallery.html"
    )

    items = _load_augmented(root / "data/pope/augmented_pope30.jsonl")
    baseline_man = load_manifest(
        perception_dump_dir("pope", MODEL_SHORT, "pope30_existence_yes_baseline")
        / "baseline"
    )
    baseline_man = {k: v for k, v in baseline_man.items() if k[1] == CONDITION}
    steered_root = perception_dump_dir(
        "pope", MODEL_SHORT, "pope30_windowed_steering"
    )

    html_doc = build_gallery(
        items=items,
        baseline_man=baseline_man,
        steered_root=steered_root,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
