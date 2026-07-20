#!/usr/bin/env python3
"""VTI visual-arm smoke driver (LLaVA-1.5, pinned AMBER-25 + CHAIR-5)."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.benchmarks import load_amber_eval, load_chair_eval
from evaluation.classifiers.metrics import compute_metric_records
from evaluation.interventions import get_intervention
from evaluation.interventions.no_intervention import NoIntervention
from evaluation.interventions.vti.hooks import vti_hook_ctx
from evaluation.interventions.vti.intervention import VTITextualIntervention
from evaluation.interventions.vti.visual_hooks import vti_visual_hook_ctx
from evaluation.interventions.vti.visual_intervention import VTIVisualIntervention
from src.extraction import cleanup_gpu
from src.mediation import compute_yes_prob, yes_token_ids
from src.model import create_wrapper, _normalize_model_name
from src.vision_dispatch import verify_vision_layout

from diagnostic_experiments.vti_visual_smoke.analyze_gate import (
    summarize_cell,
    write_smoke_gate_plots,
)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CHAIR_PROMPT = "Please Describe this image in detail."
CHAIR_CAP = 256
AMBER_SAMPLE_JSON = (
    _PROJECT_ROOT
    / "evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/amber"
    / "llava-1.5-7b-hf_amber_response_samples.json"
)
CHAIR_SAMPLE_JSON = (
    _PROJECT_ROOT
    / "evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/chair"
    / "llava-1.5-7b-hf_chair_response_samples.json"
)

STAGE0_CELLS = [
    ("no_intervention", {}),
    ("vti_textual_uniform_rotation_layer", {"beta": 0.2}),
    ("vti_textual_uniform_rotation_layer", {"beta": 0.4}),
    ("vti_textual_additive_layer", {"beta": 0.9}),
]

STAGE1_VISUAL_VARIANTS = [
    "vti_visual_additive_layer",
    "vti_visual_uniform_rotation_mlp",
    "vti_visual_additive_mlp",
    "vti_visual_uniform_rotation_layer",
]
STAGE1_ALPHAS = [0.2, 0.4, 0.9]


def _load_sample_ids(path: Path, key: str = "ordered") -> List[str]:
    data = json.loads(path.read_text())
    if key in data:
        return list(data[key])
    sample_ids = data.get("sample_ids")
    if isinstance(sample_ids, dict) and key in sample_ids:
        return list(sample_ids[key])
    if isinstance(sample_ids, list):
        return list(sample_ids)
    return []


def _cell_dir(
    output_dir: Path,
    run_date: str,
    model_short: str,
    benchmark: str,
    intervention_name: str,
    coeff_suffix: str,
) -> Path:
    iv_dir = f"{intervention_name}{coeff_suffix}"
    return output_dir / run_date / model_short / benchmark / iv_dir


def _coeff_suffix(intervention_name: str, kwargs: dict) -> str:
    if "alpha" in kwargs and kwargs["alpha"] is not None:
        return f"__a{kwargs['alpha']}"
    if "beta" in kwargs and kwargs["beta"] is not None:
        return f"__b{kwargs['beta']}"
    return ""


def _hook_context(wrapper, intervention):
    if isinstance(intervention, NoIntervention):
        return nullcontext()
    if isinstance(intervention, VTIVisualIntervention):
        directions = intervention.ensure_directions(wrapper)
        return vti_visual_hook_ctx(
            wrapper,
            directions,
            variant=intervention.config["variant"],
            alpha=intervention.config["alpha"],
            hook_site=intervention.config["hook_site"],
            eps_coeff=intervention.config["eps_coeff"],
            include_cls=intervention.config["include_cls"],
        )
    if isinstance(intervention, VTITextualIntervention):
        directions = intervention.ensure_directions(wrapper)
        return vti_hook_ctx(
            wrapper,
            directions,
            variant=intervention.config["variant"],
            alpha=intervention.config["beta"],
            hook_site=intervention.config["hook_site"],
            eps_coeff=intervention.config["eps_coeff"],
        )
    return nullcontext()


def _p_yes(wrapper, intervention, image, question) -> float:
    ids = yes_token_ids(wrapper)
    with _hook_context(wrapper, intervention):
        prob, _ = compute_yes_prob(wrapper, image, question, ids)
    return prob


def _run_amber_cell(
    wrapper,
    intervention,
    samples,
    out_dir: Path,
    max_new_tokens: int = 256,
    skip_if_exists: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.json"
    pyes_path = out_dir / "per_item_p_yes.json"
    summary_path = out_dir / "metric_summary.json"
    if skip_if_exists and responses_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text())

    records = []
    pyes_records = []
    yes_ids = yes_token_ids(wrapper)
    for sample in samples:
        with _hook_context(wrapper, intervention):
            p_yes, _ = compute_yes_prob(
                wrapper, sample.image, sample.question, yes_ids,
            )
        # generate() applies its own hook context for steered interventions
        response = intervention.generate(
            wrapper, sample.image, sample.question,
            max_new_tokens=max_new_tokens,
        )
        rec = {
            "id": sample.id,
            "question": sample.question,
            "benchmark": sample.benchmark,
            "task": sample.task,
            "ground_truth": sample.ground_truth,
            "metadata": sample.metadata or {},
            "response": response,
        }
        records.append(rec)
        pyes_records.append({
            "id": sample.id,
            "ground_truth": sample.ground_truth,
            "p_yes": p_yes,
        })
        cleanup_gpu()

    responses_path.write_text(json.dumps(records, indent=2) + "\n")
    pyes_path.write_text(json.dumps(pyes_records, indent=2) + "\n")
    metrics = compute_metric_records(records, "amber")
    summary = {
        "model": wrapper.model_name,
        "benchmark": "amber",
        "intervention": intervention.name,
        "intervention_config": intervention.config,
        **metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_chair_cell(
    wrapper,
    intervention,
    samples,
    out_dir: Path,
    max_new_tokens: int = CHAIR_CAP,
    skip_if_exists: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.json"
    summary_path = out_dir / "metric_summary.json"
    if skip_if_exists and responses_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text())

    records = []
    for sample in samples:
        response = intervention.generate(
            wrapper, sample.image, sample.question,
            max_new_tokens=max_new_tokens,
        )
        records.append({
            "id": sample.id,
            "question": sample.question,
            "benchmark": sample.benchmark,
            "task": sample.task,
            "ground_truth": sample.ground_truth,
            "metadata": sample.metadata or {},
            "response": response,
        })
        cleanup_gpu()

    responses_path.write_text(json.dumps(records, indent=2) + "\n")
    metrics = compute_metric_records(records, "chair")
    summary = {
        "model": wrapper.model_name,
        "benchmark": "chair",
        "intervention": intervention.name,
        "intervention_config": intervention.config,
        **metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_cells(
    wrapper,
    cells,
    amber_samples,
    chair_samples,
    output_dir: Path,
    run_date: str,
    model_short: str,
    skip_if_exists: bool,
) -> None:
    for intervention_name, iv_kw in cells:
        intervention = get_intervention(
            intervention_name, model_id=MODEL_ID, **iv_kw,
        )
        suffix = _coeff_suffix(intervention_name, iv_kw)
        print(f"\n=== AMBER | {intervention_name}{suffix} ===")
        adir = _cell_dir(output_dir, run_date, model_short, "amber",
                         intervention_name, suffix)
        _run_amber_cell(
            wrapper, intervention, amber_samples, adir,
            skip_if_exists=skip_if_exists,
        )
        print(f"=== CHAIR | {intervention_name}{suffix} ===")
        cdir = _cell_dir(output_dir, run_date, model_short, "chair",
                         intervention_name, suffix)
        _run_chair_cell(
            wrapper, intervention, chair_samples, cdir,
            skip_if_exists=skip_if_exists,
        )


def _stage1_cells(shakedown_only: bool = False) -> list:
    variants = (
        ["vti_visual_additive_layer", "vti_visual_uniform_rotation_mlp"]
        if shakedown_only
        else STAGE1_VISUAL_VARIANTS
    )
    cells = []
    for v in variants:
        for a in STAGE1_ALPHAS:
            cells.append((v, {"alpha": a}))
    return cells


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def _cell_stage(cell_name: str) -> str:
    if cell_name == "no_intervention":
        return "shared baseline"
    if cell_name.startswith("vti_textual_"):
        return "Stage 0 (textual gate control)"
    if cell_name.startswith("vti_visual_"):
        return "Stage 1/1b (visual arm)"
    return "unknown"


def _load_metric_summary(cell_dir: Path) -> dict:
    path = cell_dir / "metric_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_summary(
    output_dir: Path,
    run_date: str,
    model_short: str,
    path: Path,
) -> None:
    root = output_dir / run_date / model_short
    baseline_amber = root / "amber" / "no_intervention"
    amber_ids = _load_sample_ids(AMBER_SAMPLE_JSON)
    chair_ids = _load_sample_ids(CHAIR_SAMPLE_JSON, key="sample_ids")

    lines = [
        f"# VTI visual-arm smoke report ({run_date})",
        "",
        "## Run overview",
        "",
        f"- **Model:** `{MODEL_ID}` (`{model_short}`)",
        f"- **Purpose:** qualitative first-light for the **visual VTI arm**, with a",
        "  **textual-arm positive control** (Stage 0) to validate the AMBER gate machinery.",
        f"- **AMBER subset:** {len(amber_ids)} pinned discriminative items (stratified",
        "  5-per-(qtype × gold) from the 2026-06-22 LLaVA diagnostic sample draw).",
        f"- **CHAIR subset:** {len(chair_ids)} pinned COCO val2014 images from the same",
        "  diagnostic draw.",
        f"- **CHAIR prompt:** `\"{CHAIR_PROMPT}\"`",
        f"- **CHAIR cap:** `max_new_tokens={CHAIR_CAP}`",
        "- **AMBER scoring:** parsed yes/no accuracy + offline gate readout from stored",
        "  `per_item_p_yes.json` (c-AUC, acc@0.5, h+/h− flip decomposition vs baseline).",
        "- **CHAIR scoring:** `chair_s` / `chair_i` (qualitative at n=5; read captions).",
        "- **Stage 2 discrepancy toggles** (`legacy_pc_plus_mean`, `mask_fill=mean`, etc.)",
        "  were **not** run in this session.",
        "",
        "### Cells executed",
        "",
        "| Stage | Arm | Cells |",
        "|-------|-----|-------|",
        "| 0 | Textual (gate validation) | `no_intervention`; "
        "`vti_textual_uniform_rotation_layer` β∈{0.2,0.4}; "
        "`vti_textual_additive_layer` β=0.9 |",
        "| 1 | Visual (shakedown) | `no_intervention` (re-run); "
        "`vti_visual_additive_layer` + `vti_visual_uniform_rotation_mlp` "
        "× α∈{0.2,0.4,0.9} |",
        "| 1b | Visual (remaining variants) | `vti_visual_additive_mlp` + "
        "`vti_visual_uniform_rotation_layer` × α∈{0.2,0.4,0.9} |",
        "",
        f"**Total unique conditions:** 16 per benchmark "
        f"(1 baseline + 3 textual + 12 visual). "
        f"Results root: `{root.as_posix()}`.",
        "",
        "## AMBER-25 — response metrics + gate readout",
        "",
        "| Cell | Stage | acc | neg_acc | yes_ratio | c-AUC | acc@0.5 | yes_r@0.5 | h+ | h− |",
        "|------|-------|-----|---------|-----------|-------|---------|-----------|----|----|",
    ]

    for cell in sorted((root / "amber").iterdir()):
        if not cell.is_dir():
            continue
        m = _load_metric_summary(cell)
        gate = {}
        if (cell / "per_item_p_yes.json").exists():
            base = baseline_amber if cell.name != "no_intervention" else None
            gate = summarize_cell(cell, baseline_dir=base)
        c_auc = gate.get("c_auc")
        c_auc_s = f"{c_auc:.3f}" if isinstance(c_auc, float) else "—"
        h_plus = gate.get("h_plus", "—")
        h_minus = gate.get("h_minus", "—")
        if h_plus is None:
            h_plus = "—"
        if h_minus is None:
            h_minus = "—"
        lines.append(
            f"| `{cell.name}` | {_cell_stage(cell.name)} | "
            f"{_pct(m.get('accuracy_overall'))} | "
            f"{_pct(m.get('neg_item_accuracy'))} | "
            f"{_pct(m.get('yes_ratio'))} | "
            f"{c_auc_s} | "
            f"{_pct(gate.get('accuracy_at_0.5'))} | "
            f"{_pct(gate.get('yes_ratio_at_0.5'))} | "
            f"{h_plus} | {h_minus} |"
        )

    lines.extend([
        "",
        "## CHAIR-5 — caption metrics (qualitative at n=5)",
        "",
        "| Cell | Stage | chair_s ↓ | chair_i ↓ | n_empty | avg_len_chars |",
        "|------|-------|-----------|-----------|---------|---------------|",
    ])

    for cell in sorted((root / "chair").iterdir()):
        if not cell.is_dir():
            continue
        m = _load_metric_summary(cell)
        avg_len = m.get("avg_caption_len_chars")
        avg_len_s = f"{avg_len:.1f}" if isinstance(avg_len, (int, float)) else "—"
        lines.append(
            f"| `{cell.name}` | {_cell_stage(cell.name)} | "
            f"{_pct(m.get('chair_s'))} | "
            f"{_pct(m.get('chair_i'))} | "
            f"{m.get('n_empty', '—')} | {avg_len_s} |"
        )

    lines.extend([
        "",
        "## Pinned sample IDs",
        "",
        "<details><summary>AMBER-25 (`ordered`)</summary>",
        "",
        "```",
        ", ".join(amber_ids),
        "```",
        "",
        "</details>",
        "",
        "<details><summary>CHAIR-5</summary>",
        "",
        "```",
        ", ".join(chair_ids),
        "```",
        "",
        "</details>",
        "",
        "## Notes",
        "",
        "- **July 2 run included both arms:** Stage 0 exercised **textual** steering as a",
        "  known positive control for the gate; Stages 1/1b exercised **visual** steering.",
        "- The smoke driver is **intervention-agnostic** on the eval harness side: any",
        "  registered `vti_textual_*` or `vti_visual_*` cell can be slotted into the",
        "  stage cell lists. Stage 0 is textual-only by design; Stages 1/1b are",
        "  visual-only by design. Running textual-only or visual-only would mean editing",
        "  which stages you invoke (or which cells each stage lists).",
        "- Gate metrics (`c-AUC`, `acc@0.5`, h+/h−) apply to **AMBER only** (from stored",
        "  `p_yes`); CHAIR has no gate readout in this driver.",
        "- Per-cell artifacts: `responses.json`, `metric_summary.json`; AMBER also has",
        "  `per_item_p_yes.json`.",
        f"- Gate plots (AMBER): `evaluation/results/{run_date}/_diagnostics/gate_plots/`",
        "  — `roc_overlay_{textual,visual,all}.png`, `c_curve_{textual,visual,all}.png`,",
        "  and `p_yes_histograms/p_yes_hist_baseline_vs_<cell>.png`.",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["0", "1", "1b", "analyze", "verify"],
                   required=True,
                   help="0=textual gate; 1=visual shakedown; 1b=rest of stage1; "
                        "analyze=summary; verify=layout only")
    p.add_argument("--run_date", default=None)
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--skip_if_exists", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_date = args.run_date or datetime.now().strftime("%Y-%m-%d")
    model_short = _normalize_model_name(MODEL_ID)

    if args.stage == "verify":
        wrapper = create_wrapper(MODEL_ID).load()
        print(verify_vision_layout(wrapper))
        return

    output_dir = Path(args.output_dir)

    if args.stage == "analyze":
        out = (
            output_dir / run_date / "_diagnostics"
            / "vti_visual_smoke_summary.md"
        )
        _write_summary(output_dir, run_date, model_short, out)
        print(f"Wrote {out}")
        plots_dir = output_dir / run_date / "_diagnostics" / "gate_plots"
        amber_dir = output_dir / run_date / model_short / "amber"
        plot_paths = write_smoke_gate_plots(
            amber_dir, plots_dir, run_date=run_date, model_short=model_short,
        )
        for p in plot_paths:
            print(f"Wrote {p}")
        return

    amber_ids = _load_sample_ids(AMBER_SAMPLE_JSON)
    chair_ids = _load_sample_ids(CHAIR_SAMPLE_JSON, key="sample_ids")
    amber_samples = load_amber_eval(
        task="discriminative", subset_ids=set(amber_ids),
    )
    chair_samples = load_chair_eval(
        subset_ids=set(chair_ids), prompt_override=CHAIR_PROMPT,
    )

    wrapper = create_wrapper(MODEL_ID).load()
    verify_vision_layout(wrapper)

    if args.stage == "0":
        _run_cells(
            wrapper, STAGE0_CELLS, amber_samples, chair_samples,
            output_dir, run_date, model_short, args.skip_if_exists,
        )
    elif args.stage == "1":
        cells = [("no_intervention", {})] + _stage1_cells(shakedown_only=True)
        _run_cells(
            wrapper, cells, amber_samples, chair_samples,
            output_dir, run_date, model_short, args.skip_if_exists,
        )
    elif args.stage == "1b":
        cells = _stage1_cells(shakedown_only=False)
        # drop shakedown duplicates already run in stage 1
        seen = {
            ("vti_visual_additive_layer", 0.2),
            ("vti_visual_additive_layer", 0.4),
            ("vti_visual_additive_layer", 0.9),
            ("vti_visual_uniform_rotation_mlp", 0.2),
            ("vti_visual_uniform_rotation_mlp", 0.4),
            ("vti_visual_uniform_rotation_mlp", 0.9),
        }
        cells = [
            (n, k) for n, k in cells
            if (n, k.get("alpha")) not in seen
        ]
        _run_cells(
            wrapper, cells, amber_samples, chair_samples,
            output_dir, run_date, model_short, args.skip_if_exists,
        )


if __name__ == "__main__":
    main()
