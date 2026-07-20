#!/usr/bin/env python3
"""Consolidate the CHAIR + AMBER diagnostics into markdown + JSON reports.

Reads whatever is on disk for a given run_date and emits a facts-only summary
(no interpretation — that is the analyst's job via Romanus):

  Experiment 1 (reproduction grid): per benchmark, a method x beta table per
  model, with baseline->method deltas. CHAIR shows the four length/grounding
  columns; AMBER shows accuracy / yes_ratio / neg_item_accuracy adjacent (so a
  yes-drift = yes_ratio up + neg_item_accuracy down is readable) plus by_qtype.

  Experiment 2 (sweep): per benchmark, per model, the metric-vs-beta curves from
  the sweep JSONs with empty_fraction / n_unparsed alongside, and the collapse
  onset beta marked (first beta with empty_fraction >= 0.5).

Robust to partial runs (missing cells are omitted). Outputs:
  evaluation/results/{run_date}/_diagnostic_summary_chair_amber.md   (human)
  evaluation/results/{run_date}/_diagnostic_summary_chair_amber.json (machine)
and a pointer line appended to RESEARCH_LOG.md.

Usage:
  python evaluation/chair_amber_diagnostics/make_diagnostic_summary.py --run_date 2026-06-22
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

CHAIR_CAP = 256
SUBSET_SEED = 1234
BETA_GRID = [0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
COLLAPSE_EMPTY_FRAC = 0.5
SCHEMA_VERSION = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_date", required=True)
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--no_log", action="store_true",
                   help="Do not append the pointer entry to RESEARCH_LOG.md.")
    return p.parse_args()


def _load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(v, pct=False, nd=1):
    if not isinstance(v, (int, float)):
        return "--"
    return f"{v * 100:.{nd}f}" if pct else f"{v:.{nd}f}"


def _iv_dirs(bench_dir: Path) -> Dict[str, dict]:
    """Return {iv_dir_name: summary_dict} for an Exp1 benchmark dir."""
    out: Dict[str, dict] = {}
    if not bench_dir.is_dir():
        return out
    for d in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        s = _load(d / "metric_summary.json")
        if s is not None:
            out[d.name] = s
    return out


def _split_iv(name: str) -> Tuple[str, Optional[str]]:
    """'vti_textual_additive_mlp__b0.4' -> ('vti_textual_additive_mlp', '0.4')."""
    if "__b" in name:
        iv, beta = name.split("__b", 1)
        return iv, beta
    return name, None


def _methods_betas(cells: dict) -> Tuple[List[str], List[str]]:
    methods, betas = [], []
    for name in cells:
        iv, beta = _split_iv(name)
        if iv != "no_intervention" and iv not in methods:
            methods.append(iv)
        if beta is not None and beta not in betas:
            betas.append(beta)
    betas = sorted(betas, key=lambda b: float(b))
    return sorted(methods), betas


def _delta_pp(cur: Optional[float], base: Optional[float]) -> Optional[float]:
    """Delta in percentage points (markdown dCHAIR_i / dyes_ratio convention)."""
    if isinstance(base, (int, float)) and isinstance(cur, (int, float)):
        return round((cur - base) * 100, 4)
    return None


def _by_qtype_struct(bq: Optional[dict], dim: str) -> Optional[dict]:
    g = (bq or {}).get(dim)
    if not g:
        return None
    return {
        "accuracy": g.get("accuracy"),
        "yes_ratio": g.get("yes_ratio"),
        "neg_item_accuracy": g.get("neg_item_accuracy"),
    }


def _by_qtype_cell(bq: dict, dim: str) -> str:
    g = _by_qtype_struct(bq, dim)
    if not g:
        return "--"
    return (f"{_fmt(g.get('accuracy'), pct=True)}/"
            f"{_fmt(g.get('yes_ratio'), pct=True)}/"
            f"{_fmt(g.get('neg_item_accuracy'), pct=True)}")


def _empty_fraction(m: dict, n_samples: Optional[int]) -> Optional[float]:
    if "empty_fraction" in m:
        return m["empty_fraction"]
    if n_samples and "empty" in m:
        return m["empty"] / n_samples
    return None


def _collect_exp1_chair(model_short: str, cells: dict) -> dict:
    base = cells.get("no_intervention")
    methods, betas = _methods_betas(cells)
    bi = base.get("chair_i") if base else None
    grid = []
    for m in methods:
        for b in betas:
            s = cells.get(f"{m}__b{b}")
            if not s:
                continue
            grid.append({
                "method": m,
                "beta": float(b),
                "chair_s": s.get("chair_s"),
                "chair_i": s.get("chair_i"),
                "avg_objects_mentioned": s.get("avg_objects_mentioned"),
                "avg_caption_len_chars": s.get("avg_caption_len_chars"),
                "delta_chair_i_pp": _delta_pp(s.get("chair_i"), bi),
                "n_empty": s.get("n_empty"),
                "n_total": s.get("n_total"),
            })
    return {
        "model": model_short,
        "benchmark": "chair",
        "baseline": base,
        "methods": methods,
        "betas": [float(b) for b in betas],
        "grid": grid,
    }


def _collect_exp1_amber(model_short: str, cells: dict) -> dict:
    base = cells.get("no_intervention")
    methods, betas = _methods_betas(cells)
    byr = base.get("yes_ratio") if base else None
    grid = []
    for m in methods:
        for b in betas:
            s = cells.get(f"{m}__b{b}")
            if not s:
                continue
            grid.append({
                "method": m,
                "beta": float(b),
                "accuracy_overall": s.get("accuracy_overall"),
                "f1_overall": s.get("f1_overall"),
                "yes_ratio": s.get("yes_ratio"),
                "neg_item_accuracy": s.get("neg_item_accuracy"),
                "pos_item_accuracy": s.get("pos_item_accuracy"),
                "delta_yes_ratio_pp": _delta_pp(s.get("yes_ratio"), byr),
                "n_unparsed": s.get("n_unparsed"),
                "n_total": s.get("n_total"),
            })
    by_qtype: Dict[str, dict] = {}
    if base:
        by_qtype["no_intervention"] = {
            dim: _by_qtype_struct(base.get("by_qtype"), dim)
            for dim in ("existence", "attribute", "relation")
        }
    for m in methods:
        s = cells.get(f"{m}__b0.4")
        if not s:
            continue
        by_qtype[m] = {
            dim: _by_qtype_struct(s.get("by_qtype"), dim)
            for dim in ("existence", "attribute", "relation")
        }
    return {
        "model": model_short,
        "benchmark": "amber",
        "baseline": base,
        "methods": methods,
        "betas": [float(b) for b in betas],
        "grid": grid,
        "by_qtype_at_beta_0.4": by_qtype,
    }


def _collect_exp2(model_short: str, bench: str, sweep_path: Path, sweep: dict) -> dict:
    mbb = sweep.get("metrics_by_beta", {})
    betas = sorted((k for k in mbb if k != "baseline"), key=lambda b: float(b))
    order = (["baseline"] + betas) if "baseline" in mbb else betas
    n_samples = sweep.get("n_samples")
    collapse_onset = None
    curve = []
    for k in order:
        m = dict(mbb[k])
        ef = _empty_fraction(m, n_samples)
        if ef is not None:
            m["empty_fraction"] = ef
        if collapse_onset is None and k != "baseline" and ef is not None and ef >= COLLAPSE_EMPTY_FRAC:
            collapse_onset = k
        entry: Dict[str, Any] = {"beta": k, "metrics": m}
        if k != "baseline":
            entry["beta_float"] = float(k)
        curve.append(entry)
    probes = {}
    for k, m in (sweep.get("mitigation_probes_beta_max") or {}).items():
        pm = dict(m)
        ef = _empty_fraction(pm, n_samples)
        if ef is not None:
            pm["empty_fraction"] = ef
        probes[k] = pm
    return {
        "model": model_short,
        "benchmark": bench,
        "variant": sweep.get("variant", "uniform_rotation"),
        "hook_site": sweep.get("hook_site", "layer"),
        "n_samples": n_samples,
        "source_file": str(sweep_path.resolve().relative_to(_PROJECT_ROOT.resolve())),
        "collapse_empty_fraction_threshold": COLLAPSE_EMPTY_FRAC,
        "collapse_onset_beta": collapse_onset,
        "curve": curve,
        "mitigation_probes_beta_max": probes,
    }


def build_report(run_date: str, output_dir: str) -> dict:
    root = Path(output_dir) / run_date
    if not root.is_dir():
        raise FileNotFoundError(f"No results dir at {root}")
    model_dirs = sorted(p for p in root.iterdir()
                        if p.is_dir() and not p.name.startswith("_"))

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "header": {
            "chair_max_new_tokens": CHAIR_CAP,
            "subset_seed": SUBSET_SEED,
            "subset_id_files": {
                "chair": "data/chair/pinned_chair_500.json",
                "amber": "data/amber/pinned_amber_disc_450.json",
            },
            "beta_grid": BETA_GRID,
            "chair_prompt": "Please Describe this image in detail.",
            "amber_metric_convention": (
                "positive=yes (POPE-aligned); neg_item_accuracy carries the "
                "negative-detection signal"
            ),
            "models": [p.name for p in model_dirs],
        },
        "experiment_1": {
            "description": (
                "Reproduction grid: additive_mlp / additive_layer / "
                "uniform_rotation_mlp"
            ),
            "chair": [],
            "amber": [],
        },
        "experiment_2": {
            "description": "Rotation-strength sweep: uniform_rotation @ layer",
            "chair": [],
            "amber": [],
        },
    }

    for mp in model_dirs:
        ch_cells = _iv_dirs(mp / "chair")
        if ch_cells:
            report["experiment_1"]["chair"].append(
                _collect_exp1_chair(mp.name, ch_cells))
    for mp in model_dirs:
        am_cells = _iv_dirs(mp / "amber")
        if am_cells:
            report["experiment_1"]["amber"].append(
                _collect_exp1_amber(mp.name, am_cells))

    for mp in model_dirs:
        for bench in ("chair", "amber"):
            sweep_dir = mp / f"{bench}_rotation_strength"
            if not sweep_dir.is_dir():
                continue
            for f in sorted(sweep_dir.glob("sweep_*.json")):
                sweep = _load(f)
                if sweep:
                    report["experiment_2"][bench].append(
                        _collect_exp2(mp.name, bench, f, sweep))

    return report


def _render_exp1_chair(md: List[str], block: dict) -> None:
    model_short = block["model"]
    base = block.get("baseline")
    md.append(f"\n#### CHAIR — {model_short}\n")
    if base:
        md.append(f"baseline (no_intervention): "
                  f"chair_s={_fmt(base.get('chair_s'), pct=True)} "
                  f"chair_i={_fmt(base.get('chair_i'), pct=True)} "
                  f"avg_obj={_fmt(base.get('avg_objects_mentioned'))} "
                  f"avg_len={_fmt(base.get('avg_caption_len_chars'))} "
                  f"n={base.get('n_total','--')} (empty={base.get('n_empty','--')})\n")
    if not block.get("grid"):
        md.append("_(no intervention cells found)_\n")
        return
    md.append("| method | beta | chair_s | chair_i | avg_obj | avg_len | "
              "dCHAIR_i | n_empty |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in block["grid"]:
        dci = row.get("delta_chair_i_pp")
        dci_s = f"{dci:+.1f}" if isinstance(dci, (int, float)) else "--"
        b = row["beta"]
        b_s = f"{b:g}" if isinstance(b, float) and b == int(b) else str(b)
        md.append(f"| {row['method']} | {b_s} | "
                  f"{_fmt(row.get('chair_s'), pct=True)} | "
                  f"{_fmt(row.get('chair_i'), pct=True)} | "
                  f"{_fmt(row.get('avg_objects_mentioned'))} | "
                  f"{_fmt(row.get('avg_caption_len_chars'))} | {dci_s} | "
                  f"{row.get('n_empty', '--')} |")


def _render_exp1_amber(md: List[str], block: dict) -> None:
    model_short = block["model"]
    base = block.get("baseline")
    md.append(f"\n#### AMBER discriminative — {model_short}\n")
    if base:
        md.append(f"baseline (no_intervention): "
                  f"acc={_fmt(base.get('accuracy_overall'), pct=True)} "
                  f"f1={_fmt(base.get('f1_overall'), pct=True)} "
                  f"yes_ratio={_fmt(base.get('yes_ratio'), pct=True)} "
                  f"neg_item_acc={_fmt(base.get('neg_item_accuracy'), pct=True)} "
                  f"pos_item_acc={_fmt(base.get('pos_item_accuracy'), pct=True)} "
                  f"n={base.get('n_total','--')}\n")
    if not block.get("grid"):
        md.append("_(no intervention cells found)_\n")
        return
    md.append("| method | beta | acc | yes_ratio | neg_item_acc | dyes_ratio | "
              "pos_item_acc | f1 | n_unp |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in block["grid"]:
        dyr = row.get("delta_yes_ratio_pp")
        dyr_s = f"{dyr:+.1f}" if isinstance(dyr, (int, float)) else "--"
        b = row["beta"]
        b_s = f"{b:g}" if isinstance(b, float) and b == int(b) else str(b)
        md.append(f"| {row['method']} | {b_s} | "
                  f"{_fmt(row.get('accuracy_overall'), pct=True)} | "
                  f"{_fmt(row.get('yes_ratio'), pct=True)} | "
                  f"{_fmt(row.get('neg_item_accuracy'), pct=True)} | {dyr_s} | "
                  f"{_fmt(row.get('pos_item_accuracy'), pct=True)} | "
                  f"{_fmt(row.get('f1_overall'), pct=True)} | "
                  f"{row.get('n_unparsed', '--')} |")
    md.append("\nby_qtype @ beta=0.4 (acc / yes_ratio / neg_item_acc %):\n")
    md.append("| method | existence | attribute | relation |")
    md.append("| --- | --- | --- | --- |")
    bq = block.get("by_qtype_at_beta_0.4") or {}
    if "no_intervention" in bq:
        bq_base = {k: v for k, v in bq["no_intervention"].items()}
        md.append(f"| no_intervention (baseline) | "
                  f"{_by_qtype_cell(bq_base, 'existence')} | "
                  f"{_by_qtype_cell(bq_base, 'attribute')} | "
                  f"{_by_qtype_cell(bq_base, 'relation')} |")
    for m in block.get("methods", []):
        if m not in bq:
            continue
        md.append(f"| {m} | {_by_qtype_cell(bq[m], 'existence')} | "
                  f"{_by_qtype_cell(bq[m], 'attribute')} | "
                  f"{_by_qtype_cell(bq[m], 'relation')} |")


def _render_exp2(md: List[str], block: dict) -> None:
    bench = block["benchmark"]
    model_short = block["model"]
    md.append(f"\n#### {bench.upper()} sweep — {model_short} "
              f"({block.get('variant', 'uniform_rotation')} @ "
              f"{block.get('hook_site', 'layer')}, n={block.get('n_samples','?')})\n")
    if not block.get("curve"):
        md.append("| beta | _(no data)_ |")
        md.append("| --- | --- |")
        md.append("\nmitigation probes @ beta_max:\n")
        return
    if bench == "chair":
        md.append("| beta | chair_s | chair_i | avg_obj | avg_len | mlen | "
                  "empty_frac | changed |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for entry in block["curve"]:
            m = entry["metrics"]
            md.append(f"| {entry['beta']} | {_fmt(m.get('chair_s'), pct=True)} | "
                      f"{_fmt(m.get('chair_i'), pct=True)} | "
                      f"{_fmt(m.get('avg_objects_mentioned'))} | "
                      f"{_fmt(m.get('avg_caption_len_chars'))} | "
                      f"{_fmt(m.get('mean_len'))} | "
                      f"{_fmt(m.get('empty_fraction'), pct=True)} | "
                      f"{m.get('changed', '--')} |")
    else:
        md.append("| beta | acc | f1 | yes_ratio | neg_item_acc | pos_item_acc | "
                  "n_unp | empty | h+ | h- |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for entry in block["curve"]:
            m = entry["metrics"]
            md.append(f"| {entry['beta']} | {_fmt(m.get('accuracy'), pct=True)} | "
                      f"{_fmt(m.get('f1'), pct=True)} | "
                      f"{_fmt(m.get('yes_ratio'), pct=True)} | "
                      f"{_fmt(m.get('neg_item_accuracy'), pct=True)} | "
                      f"{_fmt(m.get('pos_item_accuracy'), pct=True)} | "
                      f"{m.get('n_unparsed', '--')} | {m.get('empty', '--')} | "
                      f"{m.get('flip_tn_to_fp', '--')} | "
                      f"{m.get('flip_fp_to_tn', '--')} |")
    onset = block.get("collapse_onset_beta")
    if onset:
        md.append(f"\ncollapse onset (empty_fraction >= "
                  f"{block.get('collapse_empty_fraction_threshold', COLLAPSE_EMPTY_FRAC)}): "
                  f"beta={onset}")
    md.append("\nmitigation probes @ beta_max:")
    for k, m in (block.get("mitigation_probes_beta_max") or {}).items():
        if bench == "chair":
            md.append(f"  - {k}: chair_i={_fmt(m.get('chair_i'), pct=True)} "
                      f"avg_len={_fmt(m.get('avg_caption_len_chars'))} "
                      f"empty_frac={_fmt(m.get('empty_fraction'), pct=True)}")
        else:
            md.append(f"  - {k}: acc={_fmt(m.get('accuracy'), pct=True)} "
                      f"yes_ratio={_fmt(m.get('yes_ratio'), pct=True)} "
                      f"neg_item_acc={_fmt(m.get('neg_item_accuracy'), pct=True)}")


def render_markdown(report: dict) -> str:
    hdr = report["header"]
    md = [
        f"# CHAIR + AMBER diagnostics — {report['run_date']}\n",
        "_Facts and numbers only; interpretation is the analyst's job._\n",
        "## Header facts\n",
        f"- CHAIR max_new_tokens (frozen, Step 0): **{hdr['chair_max_new_tokens']}**",
        f"- Subset seed: {hdr['subset_seed']}",
        "- Subset id files: "
        f"`{hdr['subset_id_files']['chair']}`, "
        f"`{hdr['subset_id_files']['amber']}`",
        f"- beta grid: {hdr['beta_grid']}",
        f"- CHAIR prompt (verbatim): \"{hdr['chair_prompt']}\"",
        f"- AMBER metric convention: {hdr['amber_metric_convention']}",
        f"- models discovered: {hdr['models']}",
    ]
    md.append(f"\n## Experiment 1 — {report['experiment_1']['description']}\n")
    for block in report["experiment_1"]["chair"]:
        _render_exp1_chair(md, block)
    for block in report["experiment_1"]["amber"]:
        _render_exp1_amber(md, block)
    md.append(f"\n## Experiment 2 — {report['experiment_2']['description']}\n")
    for block in report["experiment_2"]["chair"]:
        _render_exp2(md, block)
    for block in report["experiment_2"]["amber"]:
        _render_exp2(md, block)
    return "\n".join(md) + "\n"


def main():
    args = parse_args()
    try:
        report = build_report(args.run_date, args.output_dir)
    except FileNotFoundError as e:
        print(e)
        return

    root = Path(args.output_dir) / args.run_date
    md_path = root / "_diagnostic_summary_chair_amber.md"
    json_path = root / "_diagnostic_summary_chair_amber.json"

    md_path.write_text(render_markdown(report))
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")

    if not args.no_log:
        log = _PROJECT_ROOT / "RESEARCH_LOG.md"
        ts = datetime.now().strftime("%Y-%m-%d")
        entry = (f"\n### {ts} — CHAIR+AMBER diagnostics summary (pointer)\n\n"
                 f"Consolidated diagnostic report for run_date {args.run_date} "
                 f"written to `{md_path}` and `{json_path}` "
                 f"(Experiment 1 reproduction grid + Experiment 2 rotation-strength "
                 f"sweep; facts only).\n\n---\n")
        with open(log, "a") as f:
            f.write(entry)
        print(f"Appended pointer to {log}")


if __name__ == "__main__":
    main()
