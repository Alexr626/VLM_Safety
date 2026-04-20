#!/usr/bin/env python3
"""Generate individual matplotlib figures from diagnostic + intervention summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "charts"
OUT.mkdir(parents=True, exist_ok=True)

DIAG_MODELS = [
    "internvl2-8b",
    "internvl2.5-8b-mpo",
    "llava-1.5-7b-hf",
    "qwen2.5-vl-7b-instruct",
]


def load_json(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def plot_diagnostic_model(model: str) -> None:
    path = (
        ROOT
        / "diagnostic_experiments"
        / model
        / "behavioral_ground_truth"
        / "outputs"
        / "results"
        / "refusal_summary.json"
    )
    data = load_json(path)
    per = data.get("ssu_per_category") or data.get("SSU_per_category")
    if not per:
        return
    cats = sorted(per.keys())
    vl = [per[c]["vl"]["asr_harmful_content"] for c in cats]
    tt = [per[c]["tt"]["asr_harmful_content"] for c in cats]

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, vl, w, label="VL", color="#2563eb")
    ax.bar(x + w / 2, tt, w, label="TT", color="#f97316")
    ax.set_ylabel("ASR (harmful content)")
    ax.set_xlabel("Category")
    ax.set_title(f"Diagnostic — {model}\nSSU per-category ASR (keyword eval)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    safe = model.replace(".", "_")
    fig.savefig(OUT / f"diagnostic_{safe}_asr_ssu_by_category.png", dpi=150)
    plt.close(fig)


def plot_intervention_overall() -> None:
    methods = [
        ("none", "none_summary.json"),
        ("original", "original_summary.json"),
        ("spherical_t1.00", "spherical_t1.00_summary.json"),
    ]
    base = ROOT / "intervention" / "llava-1.5-7b-hf" / "outputs" / "results"
    labels = []
    sss_rr, ssu_rr, sss_asr, ssu_asr = [], [], [], []
    for name, fname in methods:
        d = load_json(base / fname)
        labels.append(name)
        sss_rr.append(d["SSS"]["refusal_rate"])
        ssu_rr.append(d["SSU"]["refusal_rate"])
        sss_asr.append(d["SSS"]["asr"])
        ssu_asr.append(d["SSU"]["asr"])

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5 * w, sss_rr, w, label="SSS refusal rate", color="#0d9488")
    ax.bar(x - 0.5 * w, ssu_rr, w, label="SSU refusal rate", color="#14b8a6")
    ax.bar(x + 0.5 * w, sss_asr, w, label="SSS ASR", color="#6366f1")
    ax.bar(x + 1.5 * w, ssu_asr, w, label="SSU ASR", color="#a855f7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title("Intervention — LLaVA-1.5-7B\nOverall SSS / SSU (none vs original vs spherical t=1.00)")
    ax.legend(ncol=2, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "intervention_llava_overall_methods.png", dpi=150)
    plt.close(fig)


def plot_intervention_per_category_individual() -> None:
    methods = [
        ("none", "none_summary.json"),
        ("original", "original_summary.json"),
        ("spherical_t1.00", "spherical_t1.00_summary.json"),
    ]
    base = ROOT / "intervention" / "llava-1.5-7b-hf" / "outputs" / "results"
    loaded = [(n, load_json(base / f)) for n, f in methods]
    per_key = "SSU_per_category"
    all_cats = sorted(
        set().union(*[set(d[per_key].keys()) for _, d in loaded])
    )

    for cat in all_cats:
        names = [n for n, _ in loaded]
        rr = [d[per_key][cat]["refusal_rate"] for _, d in loaded]
        asr = [d[per_key][cat]["asr"] for _, d in loaded]
        x = np.arange(len(names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w / 2, rr, w, label="Refusal rate", color="#dc2626")
        ax.bar(x + w / 2, asr, w, label="ASR", color="#16a34a")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Rate")
        ax.set_title(f"Intervention — category: {cat}\nSSU refusal vs ASR by method")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        safe_cat = cat.replace(" ", "_")
        fig.savefig(OUT / f"intervention_llava_category_{safe_cat}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    for m in DIAG_MODELS:
        plot_diagnostic_model(m)
    plot_intervention_overall()
    plot_intervention_per_category_individual()
    print(f"Wrote charts to {OUT}")


if __name__ == "__main__":
    main()
