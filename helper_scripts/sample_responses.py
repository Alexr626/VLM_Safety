#!/usr/bin/env python3
"""Extract qualitative response samples from a dated evaluation run.

Writes **one bundle per (model, benchmark)** under::

    evaluation/results/{run_date}/_samples/{model_short}/{benchmark}/
      {model_short}_{benchmark}_response_samples.json
      {model_short}_{benchmark}_response_samples.md
      {model_short}_{benchmark}_review.html

CHAIR: ``n_samples`` random ids from the pinned CHAIR-500 subset (default 5).

AMBER: ``n_amber_per_stratum`` ids per (question-type × ground-truth) cell,
ordered existence→attribute→relation, and within each type no then yes
(default 5 per cell). Sampling is drawn from the pinned AMBER-disc-450 pool
so model responses exist for evaluated runs. Note: AMBER existence questions
are hallucination probes — gold is always ``no``; there is no existence/yes
stratum in the benchmark.

Default beta slices (override in code or extend CLI later):

  Exp1 LLaVA  : baseline, 0.4, 0.9
  Exp1 Qwen   : baseline, 0.4
  Exp2 both   : baseline, 0.3, 0.6

Usage::

    python helper_scripts/sample_responses.py --run_date 2026-06-22
    RUN_DATE=2026-06-22 bash helper_scripts/run_sample_responses.sh
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "helper_scripts"))

from review_lib import (  # noqa: E402
    GalleryPanel,
    bundle_json_path,
    bundle_md_path,
    collect_response_sets,
    default_gallery_path,
    infer_benchmark,
    load_sample_meta,
    sample_bundle_dir,
    samples_dir,
    write_gallery_html,
)
from src.model import _normalize_model_name  # noqa: E402

CHAIR_PROMPT = "Please Describe this image in detail."
PINNED = {
    "chair": _PROJECT_ROOT / "data/chair/pinned_chair_500.json",
    "amber": _PROJECT_ROOT / "data/amber/pinned_amber_disc_450.json",
}
DEFAULT_EXP1_IVS = [
    "vti_textual_additive_mlp",
    "vti_textual_additive_layer",
    "vti_textual_uniform_rotation_mlp",
]
MODELS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}
MODEL_LABELS = {
    "llava": "LLaVA-1.5",
    "qwen": "Qwen2.5-VL",
}
EXP1_BETAS = {
    "llava": ["baseline", "0.4", "0.9"],
    "qwen": ["baseline", "0.4"],
}
EXP2_BETAS = {
    "llava": ["baseline", "0.3", "0.6"],
    "qwen": ["baseline", "0.3", "0.6"],
}
AMBER_DIMS = ("existence", "attribute", "relation")
AMBER_GOLD_ORDER = ("no", "yes")
SAMPLE_SEED = 5678
N_SAMPLES = 5
N_AMBER_PER_STRATUM = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_date", required=True)
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--n_samples", type=int, default=N_SAMPLES,
                   help="CHAIR: number of random samples (default: 5)")
    p.add_argument("--n-amber-per-stratum", type=int, default=N_AMBER_PER_STRATUM,
                   help="AMBER: samples per (qtype × gold) cell (default: 5)")
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--exp1-interventions", nargs="+", default=DEFAULT_EXP1_IVS)
    p.add_argument("--no-html", action="store_true",
                   help="Skip writing review.html galleries.")
    p.add_argument("--json-only", action="store_true",
                   help="Write JSON only (skip markdown).")
    return p.parse_args()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _pinned_ids(benchmark: str) -> list[str]:
    spec = _load_json(PINNED[benchmark])
    if isinstance(spec, dict):
        return list(spec[benchmark])
    return list(spec)


def _build_amber_index(ids: list[str] | set[str]) -> dict[str, dict]:
    """Map sample id -> question, gold, qtype, image_path (no PIL load)."""
    from src.dataset import (
        benchmark_data_dir, combined_json_path,
        _load_amber_annotations, _amber_discriminative_qtype,
    )
    root = benchmark_data_dir("amber")
    annotations = _load_amber_annotations(root)
    want = set(ids)
    index: dict[str, dict] = {}
    for row in json.load(open(combined_json_path("amber"))):
        if row.get("task", "discriminative") != "discriminative":
            continue
        sid = row["id"]
        if sid not in want:
            continue
        rid = (row.get("raw") or {}).get("id")
        ann = annotations.get(rid) if rid is not None else None
        qtype = _amber_discriminative_qtype((ann or {}).get("type", ""))
        gold = ((ann or {}).get("truth") or "").lower()
        rel = row.get("image_path") or row.get("image", "")
        img_root = root / "images"
        img_path = img_root / rel if rel and not Path(rel).is_absolute() else Path(rel or "")
        index[sid] = {
            "text": row.get("text", ""),
            "label": gold,
            "qtype": qtype,
            "image_path": str(img_path) if img_path else None,
        }
    return index


def _amber_strata_pools(pinned_ids: list[str]) -> dict[tuple[str, str], list[str]]:
    index = _build_amber_index(pinned_ids)
    pools: dict[tuple[str, str], list[str]] = {
        (dim, gold): [] for dim in AMBER_DIMS for gold in AMBER_GOLD_ORDER
    }
    for sid, info in index.items():
        dim = info.get("qtype") or "unknown"
        gold = (info.get("label") or "").lower()
        if dim in AMBER_DIMS and gold in AMBER_GOLD_ORDER:
            pools[(dim, gold)].append(sid)
    for key in pools:
        pools[key] = sorted(pools[key])
    return pools


def _pick_amber_stratified(pinned_ids: list[str], n_per: int, seed: int) -> dict:
    """Pick n_per ids per (qtype, gold) cell; return structure + ordered flat list."""
    pools = _amber_strata_pools(pinned_ids)
    rng = random.Random(seed)
    stratified: dict[str, dict[str, list[str]]] = {
        dim: {gold: [] for gold in AMBER_GOLD_ORDER} for dim in AMBER_DIMS
    }
    ordered: list[str] = []
    panel_specs: list[tuple[str, str]] = []  # (sample_id, section_heading)
    warnings: list[str] = []
    used: set[str] = set()

    for dim in AMBER_DIMS:
        for gold in AMBER_GOLD_ORDER:
            pool = [i for i in pools.get((dim, gold), []) if i not in used]
            rng.shuffle(pool)
            picked = pool[:n_per]
            if len(picked) < n_per:
                if not pool:
                    warnings.append(
                        f"{dim}/{gold}: no items in pinned eval subset "
                        f"(AMBER existence is gold=no only)" if dim == "existence" and gold == "yes"
                        else f"{dim}/{gold}: no items in pinned eval subset"
                    )
                else:
                    warnings.append(
                        f"{dim}/{gold}: only {len(picked)} available in pinned subset "
                        f"(requested {n_per})"
                    )
            stratified[dim][gold] = picked
            heading = f"{dim.capitalize()} · ground truth: {gold}"
            for sid in picked:
                ordered.append(sid)
                panel_specs.append((sid, heading))
                used.add(sid)

    return {
        "stratified": stratified,
        "ordered": ordered,
        "panel_specs": panel_specs,
        "warnings": warnings,
        "n_per_stratum": n_per,
    }


def _pick_chair_sample_ids(n: int, seed: int) -> list[str]:
    ids = _pinned_ids("chair")
    rng = random.Random(seed)
    return rng.sample(ids, min(n, len(ids)))


def _meta_for_panel(sample_id: str, benchmark: str,
                    amber_index: dict[str, dict] | None) -> dict | None:
    try:
        meta = load_sample_meta(sample_id, benchmark)
    except FileNotFoundError:
        meta = None
    if benchmark != "amber" or not amber_index:
        return meta
    info = amber_index.get(sample_id)
    if not info:
        return meta
    base = dict(meta) if meta else {"id": sample_id}
    base["text"] = info.get("text") or base.get("text", "")
    base["label"] = info.get("label") or base.get("label", "")
    if info.get("image_path"):
        base["image_path"] = info["image_path"]
    base["qtype"] = info.get("qtype")
    return base


def _exp1_cell_dir(results_root: Path, model_short: str, benchmark: str,
                   intervention: str, beta_label: str) -> Path:
    bench_dir = results_root / model_short / benchmark
    if beta_label == "baseline":
        return bench_dir / "no_intervention"
    return bench_dir / f"{intervention}__b{beta_label}"


def _exp2_sweep_path(results_root: Path, model_short: str, benchmark: str) -> Path | None:
    d = results_root / model_short / f"{benchmark}_rotation_strength"
    if not d.is_dir():
        return None
    sweeps = [p for p in sorted(d.glob("sweep_uniform_rotation_layer_n*.json"))
              if ".checkpoint" not in p.name and ".progress" not in p.name]
    return sweeps[-1] if sweeps else None


def _index_exp1(cell_dir: Path) -> dict[str, dict]:
    data = _load_json(cell_dir / "responses.json")
    if not isinstance(data, list):
        return {}
    return {r["id"]: r for r in data if r.get("id")}


def _index_exp2(sweep_path: Path) -> dict[str, dict]:
    data = _load_json(sweep_path)
    if not data:
        return {}
    return {r["id"]: r for r in data.get("per_sample", []) if r.get("id")}


def _meta_line(rec: dict | None, benchmark: str) -> str:
    if not rec:
        return ""
    parts = [f"id={rec.get('id', '?')}"]
    if benchmark == "amber":
        cat = (rec.get("metadata") or {}).get("category")
        gt = rec.get("ground_truth")
        if cat:
            parts.append(f"qtype={cat}")
        if gt:
            parts.append(f"gold={gt}")
    img = (rec.get("metadata") or {}).get("image_path")
    if img:
        parts.append(f"image={Path(img).name}")
    return " | ".join(parts)


def _truncate(text: str | None, max_chars: int = 1200) -> str:
    if text is None:
        return "_(missing — cell not on disk)_"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _response_exp1(rec: dict | None) -> str | None:
    return rec.get("response") if rec else None


def _response_exp2(rec: dict | None, beta_label: str) -> str | None:
    if not rec:
        return None
    if beta_label == "baseline":
        return rec.get("baseline")
    bb = rec.get("by_beta") or {}
    entry = bb.get(beta_label) or bb.get(f"{float(beta_label):g}")
    return entry.get("response") if isinstance(entry, dict) else None


def _prompt_exp1(rec: dict | None) -> str:
    return (rec.get("question") or rec.get("prompt") or "") if rec else ""


def _collect_exp1(results_root, model_key, benchmark, interventions, betas, sample_ids):
    model_short = _normalize_model_name(MODELS[model_key])
    out = {"model": model_short, "benchmark": benchmark, "sample_ids": sample_ids,
           "conditions": {}}
    flat_ids = sample_ids if isinstance(sample_ids, list) else sample_ids.get("ordered", [])
    for beta in betas:
        ivs = ["no_intervention"] if beta == "baseline" else interventions
        for iv in ivs:
            cell = _exp1_cell_dir(results_root, model_short, benchmark, iv, beta)
            label = "baseline" if beta == "baseline" else f"{iv} @ beta={beta}"
            idx = _index_exp1(cell)
            cond = {
                "beta": beta,
                "intervention": iv if beta != "baseline" else "no_intervention",
                "results_file": str(cell / "responses.json"),
                "on_disk": (cell / "responses.json").is_file(),
                "samples": [],
            }
            for sid in flat_ids:
                rec = idx.get(sid)
                cond["samples"].append({
                    "id": sid, "meta": _meta_line(rec, benchmark),
                    "prompt": _prompt_exp1(rec), "response": _response_exp1(rec),
                })
            out["conditions"][label] = cond
    return out


def _collect_exp2(results_root, model_key, benchmark, betas, sample_ids):
    model_short = _normalize_model_name(MODELS[model_key])
    sweep_path = _exp2_sweep_path(results_root, model_short, benchmark)
    out = {"model": model_short, "benchmark": benchmark, "sample_ids": sample_ids,
           "sweep_file": str(sweep_path) if sweep_path else None, "conditions": {}}
    idx = _index_exp2(sweep_path) if sweep_path else {}
    flat_ids = sample_ids if isinstance(sample_ids, list) else sample_ids.get("ordered", [])
    for beta in betas:
        label = f"uniform_rotation_layer @ {beta}"
        cond = {"beta": beta, "intervention": "uniform_rotation_layer",
                "on_disk": sweep_path is not None, "samples": []}
        for sid in flat_ids:
            rec = idx.get(sid)
            cond["samples"].append({
                "id": sid,
                "meta": _meta_line({"id": sid, "ground_truth": (rec or {}).get("ground_truth"),
                                    "metadata": {"category": (rec or {}).get("category")}},
                                   benchmark) if rec else f"id={sid}",
                "prompt": "",
                "response": _response_exp2(rec, beta),
            })
        out["conditions"][label] = cond
    return out


def _enrich_exp2_prompts(exp2, exp1, benchmark):
    by_id = {s["id"]: s for s in exp1.get("conditions", {}).get("baseline", {}).get("samples", [])}
    default = CHAIR_PROMPT if benchmark == "chair" else ""
    for cond in exp2.get("conditions", {}).values():
        for s in cond["samples"]:
            base = by_id.get(s["id"])
            if base:
                s["prompt"] = s.get("prompt") or base.get("prompt") or default
                if base.get("meta"):
                    s["meta"] = base["meta"]
            elif not s.get("prompt"):
                s["prompt"] = default


def _md_section(title, block, betas_order):
    lines = [f"### {title}\n"]
    if block.get("sweep_file"):
        lines.append(f"_sweep: `{block['sweep_file']}`_\n")
    sid = block["sample_ids"]
    if isinstance(sid, dict):
        lines.append("_AMBER stratified sample ids (existence → attribute → relation; "
                     "within each: no then yes):_\n")
        for dim in AMBER_DIMS:
            for gold in AMBER_GOLD_ORDER:
                ids = sid.get("stratified", {}).get(dim, {}).get(gold, [])
                if ids:
                    lines.append(f"- **{dim} / {gold}**: {', '.join(ids)}\n")
        lines.append(f"\n_ordered flat list ({len(sid.get('ordered', []))} ids): "
                     f"{', '.join(sid.get('ordered', []))}_\n")
        for w in sid.get("warnings", []):
            lines.append(f"- _warning: {w}_\n")
    else:
        lines.append(f"_sample ids: {', '.join(sid)}_\n")
    conds = block["conditions"]
    keys = []
    for b in betas_order:
        for k, v in conds.items():
            if v.get("beta") == b and k not in keys:
                keys.append(k)
    for k in conds:
        if k not in keys:
            keys.append(k)
    for k in keys:
        c = conds[k]
        status = "" if c.get("on_disk") else " **(MISSING)**"
        lines.append(f"#### {k}{status}\n")
        if not c.get("on_disk"):
            lines.append("_Not on disk for this run._\n")
            continue
        for s in c["samples"]:
            lines.append(f"**{s['id']}**")
            if s.get("meta"):
                lines.append(f" — {s['meta']}")
            lines.append("\n")
            if s.get("prompt"):
                lines.append(f"> **Q:** {s['prompt']}\n>\n")
            lines.append(f"> **R:** {_truncate(s.get('response'))}\n\n")
    return lines


def _exp1_result_paths(results_root: Path, model_short: str, benchmark: str,
                       interventions: list[str], betas: list[str]) -> list[Path]:
    paths = []
    for beta in betas:
        if beta == "baseline":
            p = _exp1_cell_dir(results_root, model_short, benchmark,
                               "no_intervention", "baseline") / "responses.json"
            if p.is_file():
                paths.append(p)
            continue
        for iv in interventions:
            p = _exp1_cell_dir(results_root, model_short, benchmark, iv, beta) / "responses.json"
            if p.is_file():
                paths.append(p)
    return paths


def _build_gallery_panels(results_root, model_key, benchmark, sample_spec,
                          interventions, amber_index) -> list[GalleryPanel]:
    model_short = _normalize_model_name(MODELS[model_key])
    model_label = MODEL_LABELS[model_key]
    exp1_paths = _exp1_result_paths(results_root, model_short, benchmark,
                                    interventions, EXP1_BETAS[model_key])
    sweep = _exp2_sweep_path(results_root, model_short, benchmark)
    result_paths = exp1_paths + ([sweep] if sweep else [])

    if benchmark == "amber" and isinstance(sample_spec, dict):
        iter_specs = sample_spec.get("panel_specs", [])
    else:
        ids = sample_spec if isinstance(sample_spec, list) else sample_spec.get("ordered", [])
        iter_specs = [(sid, "") for sid in ids]

    panels: list[GalleryPanel] = []
    prev_heading = None
    for sid, heading in iter_specs:
        display = heading if heading != prev_heading else ""
        prev_heading = heading
        bench = infer_benchmark(sid, None)
        meta = _meta_for_panel(sid, bench, amber_index)
        resp_sets = collect_response_sets(sid, result_paths)
        if not resp_sets and benchmark == "chair":
            continue
        section = display or f"{model_label} · {benchmark.upper()}"
        panels.append(GalleryPanel(sid, bench, meta, resp_sets, section_heading=section))
    return panels


def _write_bundle_md(out_md: Path, run_date: str, model_key: str, benchmark: str,
                     e1, e2, html_path: str | None, seed: int, interventions: list[str]) -> None:
    model_label = MODEL_LABELS[model_key]
    md = [
        f"# Response samples — {model_label} · {benchmark.upper()} ({run_date})\n",
        f"_Written to `{out_md.parent}`._\n",
        f"- sample seed: {seed}",
        f"- Exp1 interventions: {', '.join(interventions)}",
        "",
        "## Experiment 1 — Reproduction grid\n",
    ]
    md.extend(_md_section(
        f"{benchmark.upper()} (betas: {', '.join(EXP1_BETAS[model_key])})",
        e1, EXP1_BETAS[model_key]))
    md.append("\n## Experiment 2 — Rotation-strength\n")
    md.extend(_md_section(
        f"{benchmark.upper()} (betas: {', '.join(EXP2_BETAS[model_key])})",
        e2, EXP2_BETAS[model_key]))
    if html_path:
        md.append("\n## HTML gallery\n")
        md.append(f"Self-contained review page: `{html_path}`\n")
        md.append("_Download and open in a local browser._\n")
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_md}")


def main() -> None:
    args = parse_args()
    results_root = Path(args.output_dir) / args.run_date
    if not results_root.is_dir():
        raise SystemExit(f"No results at {results_root}")

    out_root = samples_dir(args.run_date, args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    chair_ids = _pick_chair_sample_ids(args.n_samples, args.seed)
    amber_pick = _pick_amber_stratified(_pinned_ids("amber"), args.n_amber_per_stratum,
                                        args.seed + 1)
    amber_index = _build_amber_index(_pinned_ids("amber"))

    sample_ids = {"chair": chair_ids, "amber": amber_pick}

    manifest = {
        "run_date": args.run_date,
        "output_dir": args.output_dir,
        "samples_dir": str(out_root),
        "sample_seed": args.seed,
        "chair_n_samples": args.n_samples,
        "amber_n_per_stratum": args.n_amber_per_stratum,
        "sample_ids": sample_ids,
        "exp1_interventions": args.exp1_interventions,
        "bundles": [],
    }

    for model_key in ("llava", "qwen"):
        model_short = _normalize_model_name(MODELS[model_key])
        model_label = MODEL_LABELS[model_key]
        for bench in ("chair", "amber"):
            bundle_dir = sample_bundle_dir(args.run_date, model_short, bench, args.output_dir)
            bundle_dir.mkdir(parents=True, exist_ok=True)
            ids_spec = sample_ids[bench]

            e1 = _collect_exp1(results_root, model_key, bench, args.exp1_interventions,
                               EXP1_BETAS[model_key], ids_spec)
            e2 = _collect_exp2(results_root, model_key, bench,
                               EXP2_BETAS[model_key], ids_spec)
            _enrich_exp2_prompts(e2, e1, bench)

            html_path: str | None = None
            if not args.no_html:
                panels = _build_gallery_panels(
                    results_root, model_key, bench, ids_spec,
                    args.exp1_interventions, amber_index if bench == "amber" else None)
                if panels:
                    out_html = default_gallery_path(
                        args.run_date, args.output_dir, batch=True,
                        model_short=model_short, benchmark=bench)
                    n = len(panels)
                    subtitle = (f"{model_label} · {bench.upper()} · {n} samples · "
                                f"Exp1+Exp2 · seed {args.seed}")
                    write_gallery_html(
                        f"Response samples — {model_label} · {bench.upper()} ({args.run_date})",
                        subtitle, panels, out_html)
                    html_path = str(out_html)
                    print(f"Wrote gallery HTML: {out_html}")

            bundle = {
                "run_date": args.run_date,
                "model": model_short,
                "model_label": model_label,
                "benchmark": bench,
                "sample_ids": ids_spec,
                "exp1_interventions": args.exp1_interventions,
                "experiment_1": e1,
                "experiment_2": e2,
                "html_gallery": html_path,
            }
            out_json = bundle_json_path(args.run_date, model_short, bench, args.output_dir)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(bundle, indent=2) + "\n")
            print(f"Wrote {out_json}")

            if not args.json_only:
                out_md = bundle_md_path(args.run_date, model_short, bench, args.output_dir)
                _write_bundle_md(out_md, args.run_date, model_key, bench,
                                 e1, e2, html_path, args.seed, args.exp1_interventions)

            manifest["bundles"].append({
                "model": model_short,
                "benchmark": bench,
                "dir": str(bundle_dir),
                "json": str(out_json),
                "md": str(bundle_md_path(args.run_date, model_short, bench, args.output_dir))
                if not args.json_only else None,
                "html": html_path,
                "n_samples": (len(ids_spec) if bench == "chair"
                              else len(ids_spec.get("ordered", []))),
            })

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
