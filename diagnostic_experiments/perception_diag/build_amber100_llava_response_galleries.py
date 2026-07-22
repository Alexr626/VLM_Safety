#!/usr/bin/env python3
"""HTML galleries for LLaVA AMBER-100 windowed-steering response review.

Two focused galleries (Alex 2026-07-22):

1) ``relation_yes_leading_toward_no_baseline``
   Baseline only: relation × gold=yes × assertive_toward_no (n=20).
   Motivates the large accuracy drop under a leading clause toward no.

2) ``gold_no_additive_mlp_layers_0_9``
   attribute + relation × gold=no: baseline vs additive @ mlp at layers 0–9
   for β∈{0.2, 0.5, 0.9}, both ``neutral`` and ``assertive_toward_yes``.

Usage::

    python diagnostic_experiments/perception_diag/build_amber100_llava_response_galleries.py
    python diagnostic_experiments/perception_diag/build_amber100_llava_response_galleries.py \\
      --galleries relation_yes_leading_toward_no_baseline
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

MODEL_SHORT = "llava-1.5-7b-hf"

CSS = """
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif;
         color: #1a1a1a; background: #fff; }
  .page-hdr { padding: 20px 28px 12px; border-bottom: 1px solid #ddd; }
  .page-hdr h1 { font-size: 20px; margin: 0 0 6px; }
  .page-sub { color: #666; font-size: 13px; margin: 0; max-width: 980px; }
  nav.toc { padding: 14px 28px 18px; background: #f6f7f9; border-bottom: 1px solid #ddd; }
  nav.toc h2 { font-size: 12px; margin: 12px 0 6px; text-transform: uppercase;
               letter-spacing: .04em; color: #666; }
  nav.toc h2:first-child { margin-top: 0; }
  nav.toc ul { margin: 0; padding-left: 18px; columns: 2; column-gap: 24px; }
  nav.toc a { color: #1a56db; text-decoration: none; font-size: 13px; }
  nav.toc a:hover { text-decoration: underline; }
  .summary { padding: 10px 28px; background: #fff8e8; border-bottom: 1px solid #edd9a3;
             font-size: 13px; color: #5c4a1f; }
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
  .cond-block { margin: 0 0 16px; padding: 0 0 4px; border-bottom: 1px dashed #ddd; }
  .cond-block:last-child { border-bottom: none; }
  .cond-block h4 { margin: 10px 0 8px; font-size: 12px; text-transform: uppercase;
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

CONDITION_LABELS = {
    "neutral": "neutral question",
    "assertive_toward_no": "leading clause toward no",
    "assertive_toward_yes": "leading clause toward yes",
}


def load_pretty_jsonl(path: Path) -> List[dict]:
    """Load either true JSONL or concatenated pretty-printed JSON objects."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].lstrip().startswith("{") and lines[0].rstrip().endswith("}"):
        try:
            return [json.loads(ln) for ln in lines]
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    idx = 0
    items: List[dict] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: expected JSON objects, got {type(obj)}")
        items.append(obj)
        idx = end
    return items


def _load_augmented_by_id(path: Path) -> Dict[str, dict]:
    return {row["item_id"]: row for row in load_pretty_jsonl(path)}


def _prompt_for(item: dict, condition_id: str) -> str:
    for v in item.get("variants", []):
        if v.get("condition_id") == condition_id:
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


def _img_html(item_id: str, image_path: str, img_cache: Dict[str, str]) -> str:
    if image_path not in img_cache:
        uri = _img_data_uri(Path(image_path)) if image_path else None
        img_cache[image_path] = uri or ""
    uri = img_cache[image_path]
    if uri:
        return f"<img src='{uri}' alt='{html.escape(item_id)}'>"
    return (
        f"<p class='missing'>image not found:<br>"
        f"{html.escape(str(image_path))}</p>"
    )


def _wrap_page(
    *,
    title: str,
    subtitle: str,
    summary_html: str,
    toc_html: str,
    body_html: str,
) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head>
<body>
<header class="page-hdr">
  <h1>{html.escape(title)}</h1>
  <p class="page-sub">{html.escape(subtitle)}</p>
</header>
{summary_html}
<nav class="toc">
  {toc_html}
</nav>
{body_html}
</body></html>
"""


def build_relation_yes_leading_baseline_gallery(
    *,
    items: Dict[str, dict],
    baseline_man: Dict[Tuple[str, str], dict],
) -> str:
    condition = "assertive_toward_no"
    condition_label = CONDITION_LABELS[condition]
    selected: List[str] = []
    for iid, item in items.items():
        if item.get("qtype") != "relation" or item.get("gold") != "yes":
            continue
        if (iid, condition) not in baseline_man:
            continue
        selected.append(iid)
    selected.sort()

    outcome_counts: Dict[str, int] = defaultdict(int)
    for iid in selected:
        out = baseline_man[(iid, condition)].get("parsed_outcome") or "unparseable"
        outcome_counts[str(out)] += 1

    img_cache: Dict[str, str] = {}
    toc_parts = ["<h2>Items</h2><ul>"]
    blocks: List[str] = []
    for iid in selected:
        item = items[iid]
        row = baseline_man[(iid, condition)]
        prompt = _prompt_for(item, condition)
        image_path = item.get("image_path") or ""
        toc_parts.append(
            f'<li><a href="#{html.escape(iid, quote=True)}">{html.escape(iid)}</a> '
            f"· parsed={html.escape(str(row.get('parsed_outcome')))}</li>"
        )
        blocks.append(
            f"""
<article class="item" id="{html.escape(iid, quote=True)}">
  <div class="pane-img">{_img_html(iid, image_path, img_cache)}</div>
  <div class="pane-txt">
    <h3 class="item-id">{html.escape(iid)}</h3>
    <p class="meta">gold=yes · qtype=relation · {html.escape(str(image_path))}</p>
    <div class="qa">
      <div><b>Prompt ({html.escape(condition_label)}):</b>
        {html.escape(prompt)}</div>
      <div><b>Ground truth:</b> {_yn_badge('yes')} yes</div>
    </div>
    <div class="resp-stack">
      <h4 style="margin:10px 0 8px;font-size:12px;text-transform:uppercase;
                 letter-spacing:.04em;color:#666;">
        No-intervention baseline
      </h4>
      {_response_card(kind="baseline", label="baseline (no intervention)", row=row)}
    </div>
  </div>
</article>"""
        )
    toc_parts.append("</ul>")

    counts_txt = ", ".join(
        f"{k}={v}"
        for k, v in sorted(outcome_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    title = (
        "LLaVA-1.5 AMBER-100: relation · gold=yes · leading clause toward no "
        "(baseline only)"
    )
    subtitle = (
        "Subset motivating the large accuracy drop under assertive_toward_no on "
        "relation questions with gold=yes. Shows the no-intervention baseline "
        "response and first-token P(yes)/P(no) for each of the n=20 items."
    )
    summary = (
        f'<div class="summary"><b>Baseline parsed outcomes (n={len(selected)}):</b> '
        f"{html.escape(counts_txt)}</div>"
    )
    body = f"""
<section class="qtype-section" id="relation-gold-yes">
  <div class="qtype-hdr">
    <h2>relation · gold=yes · {html.escape(condition_label)} ({len(selected)} items)</h2>
  </div>
  {''.join(blocks)}
</section>"""
    return _wrap_page(
        title=title,
        subtitle=subtitle,
        summary_html=summary,
        toc_html="".join(toc_parts),
        body_html=body,
    )


def build_gold_no_additive_layers_0_9_gallery(
    *,
    items: Dict[str, dict],
    baseline_man: Dict[Tuple[str, str], dict],
    steered_root: Path,
) -> str:
    conditions = ("neutral", "assertive_toward_yes")
    qtypes = ("attribute", "relation")
    cells: List[Tuple[str, str]] = [
        ("additive_mlp_0.2_layers_0_9", "additive @ mlp · β=0.2 · layers 0–9"),
        ("additive_mlp_0.5_layers_0_9", "additive @ mlp · β=0.5 · layers 0–9"),
        ("additive_mlp_0.9_layers_0_9", "additive @ mlp · β=0.9 · layers 0–9"),
    ]
    cell_mans: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for cell_id, _title in cells:
        man = load_manifest(steered_root / cell_id)
        if not man:
            raise FileNotFoundError(f"missing steered cell: {steered_root / cell_id}")
        cell_mans[cell_id] = man

    by_qtype: Dict[str, List[str]] = {q: [] for q in qtypes}
    for iid, item in items.items():
        qtype = item.get("qtype")
        if qtype not in qtypes or item.get("gold") != "no":
            continue
        if all((iid, c) in baseline_man for c in conditions):
            by_qtype[str(qtype)].append(iid)
    for q in qtypes:
        by_qtype[q].sort()

    img_cache: Dict[str, str] = {}
    toc_parts: List[str] = []
    body_parts: List[str] = []
    flip_counts: Dict[str, int] = defaultdict(int)

    for qtype in qtypes:
        ids = by_qtype[qtype]
        toc_parts.append(f"<h2>{html.escape(qtype)} · gold=no</h2><ul>")
        blocks: List[str] = []
        for iid in ids:
            item = items[iid]
            image_path = item.get("image_path") or ""
            toc_parts.append(
                f'<li><a href="#{html.escape(iid, quote=True)}">{html.escape(iid)}</a></li>'
            )
            cond_html: List[str] = []
            for cond in conditions:
                prompt = _prompt_for(item, cond)
                base_row = baseline_man[(iid, cond)]
                b_out = base_row.get("parsed_outcome")
                cards = [
                    _response_card(
                        kind="baseline",
                        label="no-intervention baseline",
                        row=base_row,
                    )
                ]
                for cell_id, label in cells:
                    srow = cell_mans[cell_id][(iid, cond)]
                    changed = b_out != srow.get("parsed_outcome")
                    if changed:
                        flip_counts[f"{qtype}:{cond}"] += 1
                    cards.append(
                        _response_card(
                            kind="steered",
                            label=label,
                            row=srow,
                            changed=changed,
                        )
                    )
                cond_html.append(
                    f"""
<div class="cond-block">
  <h4>{html.escape(CONDITION_LABELS[cond])}</h4>
  <div class="qa"><div><b>Prompt:</b> {html.escape(prompt)}</div></div>
  {''.join(cards)}
</div>"""
                )
            blocks.append(
                f"""
<article class="item" id="{html.escape(iid, quote=True)}">
  <div class="pane-img">{_img_html(iid, image_path, img_cache)}</div>
  <div class="pane-txt">
    <h3 class="item-id">{html.escape(iid)}</h3>
    <p class="meta">gold=no · qtype={html.escape(qtype)} ·
      {html.escape(str(image_path))}</p>
    <div class="qa">
      <div><b>Ground truth:</b> {_yn_badge('no')} no</div>
    </div>
    {''.join(cond_html)}
  </div>
</article>"""
            )
        toc_parts.append("</ul>")
        body_parts.append(
            f"""
<section class="qtype-section" id="qtype-{html.escape(qtype, quote=True)}">
  <div class="qtype-hdr">
    <h2>{html.escape(qtype)} · gold=no ({len(ids)} items)</h2>
  </div>
  {''.join(blocks)}
</section>"""
        )

    flip_txt = ", ".join(
        f"{k} flips-vs-baseline={v}" for k, v in sorted(flip_counts.items())
    ) or "none"
    n_items = sum(len(v) for v in by_qtype.values())
    title = (
        "LLaVA-1.5 AMBER-100: gold=no attribute & relation — "
        "additive @ mlp · layers 0–9"
    )
    subtitle = (
        "For each gold=no attribute/relation item: neutral and leading-toward-yes "
        "prompts, with no-intervention baseline then additive @ mlp at layers 0–9 "
        "for β=0.2 / 0.5 / 0.9. Red left border = parsed outcome differs from baseline."
    )
    summary = (
        f'<div class="summary"><b>n={n_items} items</b> '
        f"(20 attribute + 20 relation). Parsed-outcome changes vs baseline across "
        f"steered cards (item×condition×β counted separately): "
        f"{html.escape(flip_txt)}</div>"
    )
    return _wrap_page(
        title=title,
        subtitle=subtitle,
        summary_html=summary,
        toc_html="".join(toc_parts),
        body_html="".join(body_parts),
    )


GALLERY_BUILDERS = {
    "relation_yes_leading_toward_no_baseline": {
        "filename": (
            "amber100_relation_gold_yes_leading_toward_no_baseline_response_gallery.html"
        ),
        "builder": "baseline",
    },
    "gold_no_additive_mlp_layers_0_9": {
        "filename": (
            "amber100_attribute_relation_gold_no_additive_mlp_layers_0_9_"
            "response_gallery.html"
        ),
        "builder": "steered",
    },
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--galleries",
        nargs="+",
        choices=sorted(GALLERY_BUILDERS.keys()),
        default=list(GALLERY_BUILDERS.keys()),
        help="Which galleries to write (default: both)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: windowed_steering_summary/)",
    )
    args = parser.parse_args(argv)

    root = project_root()
    out_dir = args.out_dir or (
        root
        / "diagnostic_experiments"
        / "perception_diag"
        / "windowed_steering_summary"
    )
    items = _load_augmented_by_id(root / "data/amber/augmented_amber100.jsonl")
    baseline_root = perception_dump_dir("amber", MODEL_SHORT, "amber100_baseline")
    baseline_man = load_manifest(baseline_root / "baseline")
    if not baseline_man:
        raise SystemExit(f"missing baseline manifest under {baseline_root / 'baseline'}")
    steered_root = perception_dump_dir(
        "amber", MODEL_SHORT, "amber100_windowed_steering"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.galleries:
        meta = GALLERY_BUILDERS[key]
        out = out_dir / meta["filename"]
        if meta["builder"] == "baseline":
            doc = build_relation_yes_leading_baseline_gallery(
                items=items,
                baseline_man=baseline_man,
            )
        else:
            doc = build_gold_no_additive_layers_0_9_gallery(
                items=items,
                baseline_man=baseline_man,
                steered_root=steered_root,
            )
        out.write_text(doc, encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
