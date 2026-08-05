#!/usr/bin/env python3
"""Build qualitative HTML galleries for selected AMBER summary plots.

For each configured plot: sample 50 AMBER-disc items (independent seed per
file), render one image + question + gold once, then show every config that
plot covers beside/under that image with parsed yes/no, correct/incorrect,
and full response text.

Writes under::

    evaluation/results/{run_date}/_analysis_steering_visual_reasoning_validation/
      {llava,qwen}_amber_results/qualitative/<plot_stem>.html
"""

from __future__ import annotations

import html
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.classifiers.metrics import _normalize_yes_no  # noqa: E402

RUN_DATE = "2026-07-30"
ANALYSIS = (
    _PROJECT_ROOT
    / "evaluation"
    / "results"
    / RUN_DATE
    / "_analysis_steering_visual_reasoning_validation"
)
N_SAMPLES = 50
NDS = (50, 100, 200, 500)
BETAS = ("0.2", "0.5", "0.9")

LLAVA_MODEL = "llava-1.5-7b-hf"
QWEN_MODEL = "qwen2.5-vl-7b-instruct"
LLAVA_WINDOWS = (
    ("all", "all layers (0–31)"),
    ("5_14", "early–middle (5–14)"),
    ("20_29", "late (20–29)"),
)
QWEN_WINDOWS = (
    ("all", "all layers"),
    ("5_14", "early (5–14)"),
    ("15_24", "late (15–24)"),
)


@dataclass(frozen=True)
class Config:
    key: str
    label: str
    cell_dir: str  # relative under model/amber/


@dataclass(frozen=True)
class PlotSpec:
    model_dir: str
    model_short: str
    results_subdir: str  # llava_amber_results | qwen_amber_results
    plot_relpath: str  # under plots/
    out_stem: str
    title: str
    stratify_gold: bool
    seed: int
    configs: Tuple[Config, ...]
    # Optional table layout: (row_key, col_key) grouping for steered configs.
    # Config.key must be "{row}|{col}" when layout is set; baseline stays free.
    table_rows: Optional[Tuple[Tuple[str, str], ...]] = None  # (key, label)
    table_cols: Optional[Tuple[Tuple[str, str], ...]] = None


def _cell(beta: str, nd: int, layers: str) -> str:
    return (
        f"vti_textual_additive_mlp__b{beta}__dall__nd{nd}"
        f"__meandiff__layers_{layers}"
    )


def _baseline() -> Config:
    return Config("baseline", "baseline (no intervention)", "no_intervention")


def _nd_beta_grid(layers: str, window_label: str) -> Tuple[Config, ...]:
    cfgs: List[Config] = [_baseline()]
    for nd in NDS:
        for beta in BETAS:
            cfgs.append(
                Config(
                    key=f"nd{nd}|b{beta}",
                    label=f"{window_label} · nd={nd} · β={beta}",
                    cell_dir=_cell(beta, nd, layers),
                )
            )
    return tuple(cfgs)


def _window_beta_grid(
    windows: Sequence[Tuple[str, str]], nd: int = 500
) -> Tuple[Config, ...]:
    cfgs: List[Config] = [_baseline()]
    for layers, wlabel in windows:
        for beta in BETAS:
            cfgs.append(
                Config(
                    key=f"{layers}|b{beta}",
                    label=f"{wlabel} · nd={nd} · β={beta}",
                    cell_dir=_cell(beta, nd, layers),
                )
            )
    return tuple(cfgs)


def _gold_window_fixed(
    windows: Sequence[Tuple[str, str]], nd: int = 500, beta: str = "0.9"
) -> Tuple[Config, ...]:
    cfgs: List[Config] = [_baseline()]
    for layers, wlabel in windows:
        cfgs.append(
            Config(
                key=layers,
                label=f"{wlabel} · nd={nd} · β={beta}",
                cell_dir=_cell(beta, nd, layers),
            )
        )
    return tuple(cfgs)


PLOT_SPECS: Tuple[PlotSpec, ...] = (
    PlotSpec(
        model_dir=LLAVA_MODEL,
        model_short="llava",
        results_subdir="llava_amber_results",
        plot_relpath=(
            "accuracy_yes_vs_no_comparisons/"
            "gold_label_accuracy_by_layer_window_nd500_beta0.9.png"
        ),
        out_stem="gold_label_accuracy_by_layer_window_nd500_beta0.9",
        title="LLaVA AMBER — gold-label accuracy by layer window (nd=500, β=0.9)",
        stratify_gold=True,
        seed=3101,
        configs=_gold_window_fixed(LLAVA_WINDOWS),
    ),
    PlotSpec(
        model_dir=LLAVA_MODEL,
        model_short="llava",
        results_subdir="llava_amber_results",
        plot_relpath="layer_windows/all/accuracy_vs_beta_by_steering_vector_sample_size.png",
        out_stem="accuracy_vs_beta_by_steering_vector_sample_size_layers_all",
        title="LLaVA AMBER — accuracy vs β by steering-vector n (all layers)",
        stratify_gold=False,
        seed=3102,
        configs=_nd_beta_grid("all", "all layers"),
        table_rows=tuple((f"nd{nd}", f"nd={nd}") for nd in NDS),
        table_cols=tuple((f"b{b}", f"β={b}") for b in BETAS),
    ),
    PlotSpec(
        model_dir=LLAVA_MODEL,
        model_short="llava",
        results_subdir="llava_amber_results",
        plot_relpath="steering_vector_sample_size/500/accuracy_by_layer_window_and_beta.png",
        out_stem="accuracy_by_layer_window_and_beta_nd500",
        title="LLaVA AMBER — accuracy by layer window and β (nd=500)",
        stratify_gold=False,
        seed=3103,
        configs=_window_beta_grid(LLAVA_WINDOWS, nd=500),
        table_rows=tuple((ly, lab) for ly, lab in LLAVA_WINDOWS),
        table_cols=tuple((f"b{b}", f"β={b}") for b in BETAS),
    ),
    PlotSpec(
        model_dir=QWEN_MODEL,
        model_short="qwen",
        results_subdir="qwen_amber_results",
        plot_relpath="layer_windows/early/accuracy_vs_beta_by_steering_vector_sample_size.png",
        out_stem="accuracy_vs_beta_by_steering_vector_sample_size_layers_early",
        title="Qwen AMBER — accuracy vs β by steering-vector n (early layers 5–14)",
        stratify_gold=False,
        seed=3201,
        configs=_nd_beta_grid("5_14", "early (5–14)"),
        table_rows=tuple((f"nd{nd}", f"nd={nd}") for nd in NDS),
        table_cols=tuple((f"b{b}", f"β={b}") for b in BETAS),
    ),
    PlotSpec(
        model_dir=QWEN_MODEL,
        model_short="qwen",
        results_subdir="qwen_amber_results",
        plot_relpath="layer_windows/late/accuracy_vs_beta_by_steering_vector_sample_size.png",
        out_stem="accuracy_vs_beta_by_steering_vector_sample_size_layers_late",
        title="Qwen AMBER — accuracy vs β by steering-vector n (late layers 15–24)",
        stratify_gold=False,
        seed=3202,
        configs=_nd_beta_grid("15_24", "late (15–24)"),
        table_rows=tuple((f"nd{nd}", f"nd={nd}") for nd in NDS),
        table_cols=tuple((f"b{b}", f"β={b}") for b in BETAS),
    ),
    PlotSpec(
        model_dir=QWEN_MODEL,
        model_short="qwen",
        results_subdir="qwen_amber_results",
        plot_relpath="steering_vector_sample_size/500/accuracy_by_layer_window_and_beta.png",
        out_stem="accuracy_by_layer_window_and_beta_nd500",
        title="Qwen AMBER — accuracy by layer window and β (nd=500)",
        stratify_gold=False,
        seed=3203,
        configs=_window_beta_grid(QWEN_WINDOWS, nd=500),
        table_rows=tuple((ly, lab) for ly, lab in QWEN_WINDOWS),
        table_cols=tuple((f"b{b}", f"β={b}") for b in BETAS),
    ),
)


def _results_root(model_dir: str) -> Path:
    return _PROJECT_ROOT / "evaluation" / "results" / RUN_DATE / model_dir / "amber"


def _load_responses(model_dir: str, cell_dir: str) -> Dict[str, dict]:
    path = _results_root(model_dir) / cell_dir / "responses.json"
    if not path.exists():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text())
    return {r["id"]: r for r in records}


def _sample_ids(
    pool: Sequence[dict], *, n: int, seed: int, stratify_gold: bool
) -> List[str]:
    rng = random.Random(seed)
    if not stratify_gold:
        ids = [r["id"] for r in pool]
        if n > len(ids):
            raise ValueError(f"n={n} > pool={len(ids)}")
        return rng.sample(ids, n)

    yes = [r["id"] for r in pool if _normalize_yes_no(r.get("ground_truth") or "") == "yes"]
    no = [r["id"] for r in pool if _normalize_yes_no(r.get("ground_truth") or "") == "no"]
    n_yes = round(n * len(yes) / max(len(yes) + len(no), 1))
    n_yes = max(1, min(n - 1, n_yes))
    n_no = n - n_yes
    if n_yes > len(yes) or n_no > len(no):
        raise ValueError(
            f"cannot stratify n={n} into yes={n_yes}/{len(yes)} no={n_no}/{len(no)}"
        )
    picked = rng.sample(yes, n_yes) + rng.sample(no, n_no)
    rng.shuffle(picked)
    return picked


def _rel_image(abs_path: str, out_html: Path) -> str:
    img = Path(abs_path)
    try:
        return os.path.relpath(str(img.resolve()), str(out_html.parent.resolve()))
    except Exception:
        return str(img)


def _verdict(pred: Optional[str], gold: Optional[str]) -> Tuple[str, str]:
    if gold is None:
        return "unknown", "gold unparsed"
    if pred is None:
        return "unparsed", "unparsed"
    if pred == gold:
        return "correct", "correct"
    return "incorrect", "incorrect"


CSS = """
  :root {
    --bg: #f4f1ea;
    --card: #fffdf8;
    --ink: #1b1916;
    --muted: #5b564d;
    --line: #d9d1c3;
    --accent: #245c4f;
    --ok: #1a7f37;
    --bad: #b42318;
    --unk: #6b6560;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    background:
      radial-gradient(1200px 500px at 10% -10%, #e9dfcf 0%, transparent 60%),
      var(--bg);
    color: var(--ink);
    line-height: 1.45;
  }
  header.page {
    padding: 2rem 1.5rem 1rem;
    border-bottom: 1px solid var(--line);
  }
  header.page h1 { margin: 0 0 0.4rem; font-size: 1.55rem; font-weight: 650; }
  header.page p { margin: 0.25rem 0; color: var(--muted); max-width: 58rem; }
  header.page code, footer code, .meta code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.82em;
    background: #ebe4d6;
    padding: 0.08em 0.32em;
    border-radius: 3px;
  }
  main { padding: 1rem 1.25rem 2.5rem; display: grid; gap: 1.1rem; }
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.95rem 1rem 1.1rem;
  }
  .card h2 { margin: 0 0 0.35rem; font-size: 1.12rem; color: var(--accent); }
  .meta {
    display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;
    margin-bottom: 0.75rem; color: var(--muted); font-size: 0.92rem;
  }
  .pill {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.72rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    color: var(--muted);
    background: #f3eee3;
  }
  .pill.gold { color: #8a4b12; border-color: #e0c7a4; background: #f8efe3; }
  .body {
    display: grid;
    grid-template-columns: minmax(180px, 280px) 1fr;
    gap: 1rem;
    align-items: start;
  }
  @media (max-width: 900px) { .body { grid-template-columns: 1fr; } }
  .media img {
    display: block; width: 100%; height: auto; max-height: 320px;
    object-fit: contain; background: #efe9dc; border: 1px solid var(--line);
    border-radius: 6px;
  }
  .label {
    display: block; font-size: 0.7rem; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.2rem;
  }
  .question { margin: 0 0 0.75rem; font-size: 1.08rem; }
  .baseline-box, .resp-cell {
    border: 1px solid var(--line);
    border-radius: 7px;
    background: #f8f4eb;
    padding: 0.55rem 0.65rem;
  }
  .baseline-box { margin-bottom: 0.85rem; }
  .resp-head {
    display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;
    margin-bottom: 0.35rem; font-size: 0.82rem;
  }
  .resp-head .cfg {
    font-weight: 650; color: var(--ink);
  }
  .badge {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.7rem; font-weight: 700; border-radius: 4px;
    padding: 0.08rem 0.45rem; color: #fff;
  }
  .yn-yes { background: var(--ok); }
  .yn-no { background: var(--bad); }
  .yn-unk { background: var(--unk); }
  .v-correct { background: var(--ok); }
  .v-incorrect { background: var(--bad); }
  .v-unparsed, .v-unknown { background: var(--unk); }
  pre {
    margin: 0; white-space: pre-wrap; word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem; line-height: 1.4;
  }
  .grid-wrap { overflow-x: auto; }
  table.resp-grid {
    border-collapse: collapse; width: 100%; min-width: 640px;
    font-size: 0.82rem;
  }
  table.resp-grid th, table.resp-grid td {
    border: 1px solid var(--line); vertical-align: top;
    padding: 0.45rem 0.5rem; background: #fffdf8;
  }
  table.resp-grid th {
    background: #efe8da; text-align: left; font-weight: 650;
  }
  table.resp-grid th.corner { background: #e6ddcc; }
  .flat-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.55rem;
  }
  footer {
    padding: 0 1.5rem 2rem; color: var(--muted); font-size: 0.85rem;
    max-width: 64rem;
  }
"""


def _badge_yn(pred: Optional[str]) -> str:
    if pred == "yes":
        return '<span class="badge yn-yes">yes</span>'
    if pred == "no":
        return '<span class="badge yn-no">no</span>'
    return '<span class="badge yn-unk">unparsed</span>'


def _badge_verdict(kind: str, label: str) -> str:
    return f'<span class="badge v-{html.escape(kind)}">{html.escape(label)}</span>'


def _resp_inner(label: str, text: str, gold: Optional[str]) -> str:
    pred = _normalize_yes_no(text or "")
    vkind, vlabel = _verdict(pred, gold)
    body = html.escape((text or "").strip()) or "<em>&lt;empty&gt;</em>"
    return (
        f'<div class="resp-head">'
        f'<span class="cfg">{html.escape(label)}</span>'
        f"{_badge_yn(pred)}"
        f"{_badge_verdict(vkind, vlabel)}"
        f"</div>"
        f"<pre>{body}</pre>"
    )


def _render_responses(
    spec: PlotSpec,
    by_cfg: Dict[str, dict],
    gold: Optional[str],
) -> str:
    baseline = next(c for c in spec.configs if c.key == "baseline")
    base_rec = by_cfg[baseline.key]
    parts = [
        '<div class="baseline-box">',
        _resp_inner(baseline.label, base_rec.get("response") or "", gold),
        "</div>",
    ]

    steered = [c for c in spec.configs if c.key != "baseline"]
    if spec.table_rows and spec.table_cols:
        lookup = {c.key: c for c in steered}
        parts.append('<div class="grid-wrap"><table class="resp-grid">')
        parts.append("<thead><tr><th class='corner'></th>")
        for _ck, clab in spec.table_cols:
            parts.append(f"<th>{html.escape(clab)}</th>")
        parts.append("</tr></thead><tbody>")
        for rkey, rlab in spec.table_rows:
            parts.append(f"<tr><th>{html.escape(rlab)}</th>")
            for ckey, _clab in spec.table_cols:
                cfg = lookup[f"{rkey}|{ckey}"]
                rec = by_cfg[cfg.key]
                parts.append(
                    "<td class='resp-cell'>"
                    + _resp_inner(cfg.label, rec.get("response") or "", gold)
                    + "</td>"
                )
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
    else:
        parts.append('<div class="flat-list">')
        for cfg in steered:
            rec = by_cfg[cfg.key]
            parts.append(
                '<div class="resp-cell">'
                + _resp_inner(cfg.label, rec.get("response") or "", gold)
                + "</div>"
            )
        parts.append("</div>")
    return "\n".join(parts)


def _build_html(spec: PlotSpec, sample_ids: List[str], caches: Dict[str, Dict[str, dict]], out_path: Path) -> str:
    plot_href = html.escape(f"../plots/{spec.plot_relpath}")
    cards = []
    for sid in sample_ids:
        base = caches["baseline"][sid]
        gold_raw = base.get("ground_truth") or ""
        gold = _normalize_yes_no(gold_raw)
        q = base.get("question") or ""
        meta = base.get("metadata") or {}
        img_abs = meta.get("image_path") or ""
        img_rel = _rel_image(img_abs, out_path) if img_abs else ""
        cat = meta.get("category") or "?"
        by_cfg = {c.key: caches[c.key][sid] for c in spec.configs}

        gold_pill = html.escape(gold or gold_raw or "?")
        cards.append(
            f"""
<section class="card" id="{html.escape(sid)}">
  <h2>{html.escape(sid)}</h2>
  <div class="meta">
    <span class="pill gold">gold: {gold_pill}</span>
    <span class="pill">{html.escape(str(cat))}</span>
    <code>{html.escape(Path(img_abs).name if img_abs else "")}</code>
  </div>
  <div class="body">
    <div class="media">
      <span class="label">Image</span>
      {"<a href='" + html.escape(img_rel) + "' target='_blank' rel='noopener'><img src='" + html.escape(img_rel) + "' alt='" + html.escape(sid) + "' loading='lazy'/></a>" if img_rel else "<p>image missing</p>"}
    </div>
    <div class="text">
      <span class="label">Question</span>
      <p class="question">{html.escape(q)}</p>
      <span class="label">Responses (same item across configs)</span>
      {_render_responses(spec, by_cfg, gold)}
    </div>
  </div>
</section>"""
        )

    gold_note = (
        "Sampling stratified by gold yes/no in proportion to the AMBER-450 pool."
        if spec.stratify_gold
        else "Sampling uniform over the AMBER-450 items counted in this plot."
    )
    n_yes = sum(
        1
        for sid in sample_ids
        if _normalize_yes_no(caches["baseline"][sid].get("ground_truth") or "") == "yes"
    )
    n_no = len(sample_ids) - n_yes

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(spec.title)}</title>
<style>{CSS}</style>
</head>
<body>
<header class="page">
  <h1>{html.escape(spec.title)}</h1>
  <p>Model <code>{html.escape(spec.model_dir)}</code> · run date <code>{RUN_DATE}</code> ·
     N={len(sample_ids)} examples (seed={spec.seed}; gold-yes={n_yes}, gold-no={n_no}).</p>
  <p>{html.escape(gold_note)} Independent draw for this file. One image per example;
     all plot configs shown beside it.</p>
  <p>Source plot: <a href="{plot_href}"><code>{html.escape(spec.plot_relpath)}</code></a></p>
  <p>Parser: leading yes/no token with word-boundary fallback
     (<code>evaluation.classifiers.metrics._normalize_yes_no</code>). Unparsed counts as incorrect.</p>
</header>
<main>
{"".join(cards)}
</main>
<footer>
  Responses from <code>evaluation/results/{RUN_DATE}/{html.escape(spec.model_dir)}/amber/*/responses.json</code>.
  Images under <code>data/amber/images/</code>.
</footer>
</body>
</html>
"""


def build_one(spec: PlotSpec) -> Path:
    out_dir = ANALYSIS / spec.results_subdir / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.out_stem}.html"

    caches: Dict[str, Dict[str, dict]] = {}
    for cfg in spec.configs:
        caches[cfg.key] = _load_responses(spec.model_dir, cfg.cell_dir)

    pool = list(caches["baseline"].values())
    # Ensure every config has the same ids.
    id_sets = [set(caches[c.key]) for c in spec.configs]
    common = set.intersection(*id_sets)
    pool = [r for r in pool if r["id"] in common]
    sample_ids = _sample_ids(pool, n=N_SAMPLES, seed=spec.seed, stratify_gold=spec.stratify_gold)

    out_path.write_text(_build_html(spec, sample_ids, caches, out_path), encoding="utf-8")
    return out_path


def main() -> None:
    paths = []
    for spec in PLOT_SPECS:
        path = build_one(spec)
        paths.append(path)
        print(f"wrote {path.relative_to(_PROJECT_ROOT)}")
    print(f"done: {len(paths)} HTML files")


if __name__ == "__main__":
    main()
