#!/usr/bin/env python3
"""
Plots for Together LLM-judge refusal rates only.

Reads: diagnostic_experiments/refusal_rates_all_models_together_judge.json
(produced by run_refusal_judge_all_models.py after refusal_judge_together.py per model).

Output: VLM_Safety/charts/llm_judge_together/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "diagnostic_experiments" / "refusal_rates_all_models_together_judge.json"
OUT = ROOT / "charts" / "llm_judge_together"
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["response_vl", "response_tt", "response_ct"]
COL_LABELS = ["VL", "TT", "CT"]


def rate(refused: int, total: int) -> float:
    return refused / total if total else 0.0


def load() -> dict:
    with AGG.open() as f:
        return json.load(f)


def plot_overall_comparison(data: dict) -> None:
    models = list(data["models"].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.25
    for i, (col, lab) in enumerate(zip(COLS, COL_LABELS)):
        vals = [rate(data["models"][m]["overall"][col]["refused"], data["models"][m]["overall"][col]["total"]) for m in models]
        ax.bar(x + (i - 1) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Refusal rate (judge)")
    all_v = []
    for m in models:
        for c in COLS:
            all_v.append(rate(data["models"][m]["overall"][c]["refused"], data["models"][m]["overall"][c]["total"]))
    ymax = max(all_v) if all_v else 0.0
    ax.set_ylim(0, 1.05 if ymax > 0.15 else 0.12)
    ax.set_title(
        "Together LLM judge (openai/gpt-oss-120b)\n"
        "HoliSafe refusal rate — overall (all models)"
    )
    ax.legend(title="Response column")
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5,
        0.02,
        "Source: refusal_rates_all_models_together_judge.json",
        ha="center",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(OUT / "judge_overall_all_models.png", dpi=150)
    plt.close(fig)


def plot_per_model_by_category(data: dict, model: str) -> None:
    mdata = data["models"][model]
    by_cat = mdata["by_category"]
    cats = sorted(by_cat.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cats))
    w = 0.25
    for i, (col, lab) in enumerate(zip(COLS, COL_LABELS)):
        vals = [
            rate(by_cat[c][col]["refused"], by_cat[c][col]["total"])
            for c in cats
        ]
        ax.bar(x + (i - 1) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("Refusal rate (judge)")
    ax.set_title(f"Together LLM judge — {model}\nRefusal rate by HoliSafe category")
    ax.legend(title="Column")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    safe = model.replace(".", "_")
    fig.savefig(OUT / f"judge_{safe}_by_category.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not AGG.is_file():
        raise SystemExit(f"Missing aggregate file: {AGG}")
    data = load()
    plot_overall_comparison(data)
    for model in data["models"]:
        plot_per_model_by_category(data, model)
    print(f"LLM judge charts written to {OUT}")
    # Print numeric summary to stdout
    print("\nOverall refusal rates (from aggregate JSON):")
    for m in data["models"]:
        o = data["models"][m]["overall"]
        parts = []
        for col in COLS:
            r, t = o[col]["refused"], o[col]["total"]
            parts.append(f"{col}: {r}/{t} = {100*rate(r,t):.2f}%")
        print(f"  {m}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
