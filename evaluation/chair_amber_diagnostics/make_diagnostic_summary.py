#!/usr/bin/env python3
"""Consolidate the CHAIR + AMBER diagnostics into one markdown report.

Reads whatever is on disk for a given run_date and emits a facts-only summary
(no interpretation — that is the analyst's job via Romanus):

  Experiment 1 (reproduction grid): per benchmark, a method x beta table per
  model, with baseline->method deltas. CHAIR shows the four length/grounding
  columns; AMBER shows accuracy / yes_ratio / neg_item_accuracy adjacent (so a
  yes-drift = yes_ratio up + neg_item_accuracy down is readable) plus by_qtype.

  Experiment 2 (sweep): per benchmark, per model, the metric-vs-beta curves from
  the sweep JSONs with empty_fraction / n_unparsed alongside, and the collapse
  onset beta marked (first beta with empty_fraction >= 0.5).

Robust to partial runs (missing cells are shown as '--'). Output:
  evaluation/results/{run_date}/_diagnostic_summary_chair_amber.md
and a pointer line appended to RESEARCH_LOG.md.

Usage:
  python evaluation/chair_amber_diagnostics/make_diagnostic_summary.py --run_date 2026-06-22
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

CHAIR_CAP = 64
SUBSET_SEED = 1234
BETA_GRID = [0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
COLLAPSE_EMPTY_FRAC = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_date", required=True)
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--no_log", action="store_true",
                   help="Do not append the pointer entry to RESEARCH_LOG.md.")
    return p.parse_args()


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(v, pct=False, nd=1):
    if not isinstance(v, (int, float)):
        return "--"
    return f"{v * 100:.{nd}f}" if pct else f"{v:.{nd}f}"


def _iv_dirs(bench_dir: Path):
    """Return {iv_dir_name: summary_dict} for an Exp1 benchmark dir."""
    out = {}
    if not bench_dir.is_dir():
        return out
    for d in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        s = _load(d / "metric_summary.json")
        if s is not None:
            out[d.name] = s
    return out


def _split_iv(name: str):
    """'vti_textual_additive_mlp__b0.4' -> ('vti_textual_additive_mlp', '0.4')."""
    if "__b" in name:
        iv, beta = name.split("__b", 1)
        return iv, beta
    return name, None


def _methods_betas(cells: dict):
    methods, betas = [], []
    for name in cells:
        iv, beta = _split_iv(name)
        if iv != "no_intervention" and iv not in methods:
            methods.append(iv)
        if beta is not None and beta not in betas:
            betas.append(beta)
    betas = sorted(betas, key=lambda b: float(b))
    return sorted(methods), betas


def _by_qtype_cell(bq: dict, dim: str) -> str:
    g = (bq or {}).get(dim)
    if not g:
        return "--"
    return (f"{_fmt(g.get('accuracy'), pct=True)}/"
            f"{_fmt(g.get('yes_ratio'), pct=True)}/"
            f"{_fmt(g.get('neg_item_accuracy'), pct=True)}")


def _exp1_chair(md, model_short, cells):
    base = cells.get("no_intervention")
    methods, betas = _methods_betas(cells)
    md.append(f"\n#### CHAIR — {model_short}\n")
    if base:
        md.append(f"baseline (no_intervention): "
                  f"chair_s={_fmt(base.get('chair_s'), pct=True)} "
                  f"chair_i={_fmt(base.get('chair_i'), pct=True)} "
                  f"avg_obj={_fmt(base.get('avg_objects_mentioned'))} "
                  f"avg_len={_fmt(base.get('avg_caption_len_chars'))} "
                  f"n={base.get('n_total','--')} (empty={base.get('n_empty','--')})\n")
    if not methods:
        md.append("_(no intervention cells found)_\n")
        return
    md.append("| method | beta | chair_s | chair_i | avg_obj | avg_len | "
              "dCHAIR_i | n_empty |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    bi = base.get("chair_i") if base else None
    for m in methods:
        for b in betas:
            s = cells.get(f"{m}__b{b}")
            if not s:
                continue
            dci = (f"{(s.get('chair_i',0)-bi)*100:+.1f}"
                   if isinstance(bi, (int, float)) and isinstance(s.get('chair_i'), (int, float))
                   else "--")
            md.append(f"| {m} | {b} | {_fmt(s.get('chair_s'),pct=True)} | "
                      f"{_fmt(s.get('chair_i'),pct=True)} | "
                      f"{_fmt(s.get('avg_objects_mentioned'))} | "
                      f"{_fmt(s.get('avg_caption_len_chars'))} | {dci} | "
                      f"{s.get('n_empty','--')} |")


def _exp1_amber(md, model_short, cells):
    base = cells.get("no_intervention")
    methods, betas = _methods_betas(cells)
    md.append(f"\n#### AMBER discriminative — {model_short}\n")
    if base:
        md.append(f"baseline (no_intervention): "
                  f"acc={_fmt(base.get('accuracy_overall'),pct=True)} "
                  f"f1={_fmt(base.get('f1_overall'),pct=True)} "
                  f"yes_ratio={_fmt(base.get('yes_ratio'),pct=True)} "
                  f"neg_item_acc={_fmt(base.get('neg_item_accuracy'),pct=True)} "
                  f"pos_item_acc={_fmt(base.get('pos_item_accuracy'),pct=True)} "
                  f"n={base.get('n_total','--')}\n")
    if not methods:
        md.append("_(no intervention cells found)_\n")
        return
    # yes_ratio and neg_item_accuracy adjacent so yes-drift is readable.
    md.append("| method | beta | acc | yes_ratio | neg_item_acc | dyes_ratio | "
              "pos_item_acc | f1 | n_unp |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    byr = base.get("yes_ratio") if base else None
    for m in methods:
        for b in betas:
            s = cells.get(f"{m}__b{b}")
            if not s:
                continue
            dyr = (f"{(s.get('yes_ratio',0)-byr)*100:+.1f}"
                   if isinstance(byr, (int, float)) and isinstance(s.get('yes_ratio'), (int, float))
                   else "--")
            md.append(f"| {m} | {b} | {_fmt(s.get('accuracy_overall'),pct=True)} | "
                      f"{_fmt(s.get('yes_ratio'),pct=True)} | "
                      f"{_fmt(s.get('neg_item_accuracy'),pct=True)} | {dyr} | "
                      f"{_fmt(s.get('pos_item_accuracy'),pct=True)} | "
                      f"{_fmt(s.get('f1_overall'),pct=True)} | "
                      f"{s.get('n_unparsed','--')} |")
    # by_qtype at canonical beta 0.4 for each method (baseline row for comparison).
    md.append("\nby_qtype @ beta=0.4 (acc / yes_ratio / neg_item_acc %):\n")
    md.append("| method | existence | attribute | relation |")
    md.append("| --- | --- | --- | --- |")
    if base:
        bq_base = base.get("by_qtype") or {}
        md.append(f"| no_intervention (baseline) | "
                  f"{_by_qtype_cell(bq_base, 'existence')} | "
                  f"{_by_qtype_cell(bq_base, 'attribute')} | "
                  f"{_by_qtype_cell(bq_base, 'relation')} |")
    for m in methods:
        s = cells.get(f"{m}__b0.4")
        bq = (s or {}).get("by_qtype") or {}
        md.append(f"| {m} | {_by_qtype_cell(bq, 'existence')} | "
                  f"{_by_qtype_cell(bq, 'attribute')} | "
                  f"{_by_qtype_cell(bq, 'relation')} |")


def _exp2_curve(md, model_short, bench, sweep):
    md.append(f"\n#### {bench.upper()} sweep — {model_short} "
              f"(uniform_rotation @ layer, n={sweep.get('n_samples','?')})\n")
    mbb = sweep.get("metrics_by_beta", {})
    betas = [k for k in mbb if k != "baseline"]
    betas = sorted(betas, key=lambda b: float(b))
    order = (["baseline"] + betas) if "baseline" in mbb else betas
    collapse_onset = None
    if bench == "chair":
        md.append("| beta | chair_s | chair_i | avg_obj | avg_len | mlen | "
                  "empty_frac | changed |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for k in order:
            m = mbb[k]
            ef = m.get("empty_fraction", (m.get("empty", 0) / sweep.get("n_samples", 1)
                                          if sweep.get("n_samples") else 0))
            if collapse_onset is None and k != "baseline" and ef >= COLLAPSE_EMPTY_FRAC:
                collapse_onset = k
            md.append(f"| {k} | {_fmt(m.get('chair_s'),pct=True)} | "
                      f"{_fmt(m.get('chair_i'),pct=True)} | "
                      f"{_fmt(m.get('avg_objects_mentioned'))} | "
                      f"{_fmt(m.get('avg_caption_len_chars'))} | "
                      f"{_fmt(m.get('mean_len'))} | {_fmt(ef,pct=True)} | "
                      f"{m.get('changed','--')} |")
    else:
        md.append("| beta | acc | f1 | yes_ratio | neg_item_acc | pos_item_acc | "
                  "n_unp | empty | h+ | h- |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for k in order:
            m = mbb[k]
            md.append(f"| {k} | {_fmt(m.get('accuracy'),pct=True)} | "
                      f"{_fmt(m.get('f1'),pct=True)} | "
                      f"{_fmt(m.get('yes_ratio'),pct=True)} | "
                      f"{_fmt(m.get('neg_item_accuracy'),pct=True)} | "
                      f"{_fmt(m.get('pos_item_accuracy'),pct=True)} | "
                      f"{m.get('n_unparsed','--')} | {m.get('empty','--')} | "
                      f"{m.get('flip_tn_to_fp','--')} | {m.get('flip_fp_to_tn','--')} |")
    if collapse_onset:
        md.append(f"\ncollapse onset (empty_fraction >= {COLLAPSE_EMPTY_FRAC}): "
                  f"beta={collapse_onset}")
    md.append("\nmitigation probes @ beta_max:")
    for k, m in (sweep.get("mitigation_probes_beta_max") or {}).items():
        if bench == "chair":
            md.append(f"  - {k}: chair_i={_fmt(m.get('chair_i'),pct=True)} "
                      f"avg_len={_fmt(m.get('avg_caption_len_chars'))} "
                      f"empty_frac={_fmt(m.get('empty_fraction'),pct=True)}")
        else:
            md.append(f"  - {k}: acc={_fmt(m.get('accuracy'),pct=True)} "
                      f"yes_ratio={_fmt(m.get('yes_ratio'),pct=True)} "
                      f"neg_item_acc={_fmt(m.get('neg_item_accuracy'),pct=True)}")


def main():
    args = parse_args()
    root = Path(args.output_dir) / args.run_date
    if not root.is_dir():
        print(f"No results dir at {root}")
        return
    model_dirs = sorted(p for p in root.iterdir()
                        if p.is_dir() and not p.name.startswith("_"))

    md = [f"# CHAIR + AMBER diagnostics — {args.run_date}\n",
          "_Facts and numbers only; interpretation is the analyst's job._\n",
          "## Header facts\n",
          f"- CHAIR max_new_tokens (frozen, Step 0): **{CHAIR_CAP}**",
          f"- Subset seed: {SUBSET_SEED}",
          "- Subset id files: `data/chair/pinned_chair_500.json`, "
          "`data/amber/pinned_amber_disc_450.json`",
          f"- beta grid: {BETA_GRID}",
          "- CHAIR prompt (verbatim): \"Please Describe this image in detail.\"",
          "- AMBER metric convention: positive=yes (POPE-aligned); "
          "neg_item_accuracy carries the negative-detection signal",
          f"- models discovered: {[p.name for p in model_dirs]}",
          ]

    md.append("\n## Experiment 1 — Reproduction grid (additive_mlp / additive_layer "
              "/ uniform_rotation_mlp)\n")
    for mp in model_dirs:
        ch_cells = _iv_dirs(mp / "chair")
        if ch_cells:
            _exp1_chair(md, mp.name, ch_cells)
    for mp in model_dirs:
        am_cells = _iv_dirs(mp / "amber")
        if am_cells:
            _exp1_amber(md, mp.name, am_cells)

    md.append("\n## Experiment 2 — Rotation-strength sweep "
              "(uniform_rotation @ layer)\n")
    for mp in model_dirs:
        for bench in ("chair", "amber"):
            sweep_dir = mp / f"{bench}_rotation_strength"
            if not sweep_dir.is_dir():
                continue
            for f in sorted(sweep_dir.glob("sweep_*.json")):
                sweep = _load(f)
                if sweep:
                    _exp2_curve(md, mp.name, bench, sweep)

    out_path = root / "_diagnostic_summary_chair_amber.md"
    out_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_path}")

    if not args.no_log:
        log = _PROJECT_ROOT / "RESEARCH_LOG.md"
        ts = datetime.now().strftime("%Y-%m-%d")
        entry = (f"\n### {ts} — CHAIR+AMBER diagnostics summary (pointer)\n\n"
                 f"Consolidated diagnostic report for run_date {args.run_date} "
                 f"written to `{out_path}` (Experiment 1 reproduction grid + "
                 f"Experiment 2 rotation-strength sweep; facts only).\n\n---\n")
        with open(log, "a") as f:
            f.write(entry)
        print(f"Appended pointer to {log}")


if __name__ == "__main__":
    main()
