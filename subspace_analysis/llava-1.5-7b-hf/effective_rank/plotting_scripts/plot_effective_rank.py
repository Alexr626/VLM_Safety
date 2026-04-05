"""
Plot Experiment A: Effective Rank of Modality Shift Spaces
Generates individual plots for effective rank comparison, rank gap,
multi-threshold view, and cumulative variance spectra.

Reads from: ../outputs/results/effective_rank_results.json
            ../outputs/artifacts/singular_spectra.npz
Saves to:   ../outputs/results/plots/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR     = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent                  # effective_rank/
_MODEL_NAME     = _EXPERIMENT_DIR.parent.name         # llava-1.5-7b-hf
_PROJECT_ROOT   = _EXPERIMENT_DIR.parent.parent.parent
_RESULTS_DIR    = _EXPERIMENT_DIR / "outputs" / "results"
_EXP_ARTIFACTS  = _PROJECT_ROOT / "experiment_artifacts" / _MODEL_NAME / "effective_rank"

_RANK_PATH    = _RESULTS_DIR / "effective_rank_results.json"
_SPECTRA_PATH = _EXP_ARTIFACTS / "singular_spectra.npz"
_OUTPUT_DIR   = _RESULTS_DIR / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not _RANK_PATH.exists():
    print(
        f"ERROR: {_RANK_PATH} not found.\n"
        "Please run Experiment A first:\n"
        "  python .../effective_rank/experiment_scripts/experiment_a_effective_rank.py"
    )
    sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────
with open(_RANK_PATH) as f:
    data = json.load(f)

data = [d for d in data if d["layer"] > 0]

layers = np.array([d["layer"] for d in data])
thresholds = [0.4, 0.6, 0.7, 0.8, 0.9]

ranks = {}
for tau in thresholds:
    key = str(tau)
    ranks[key] = {
        "sss": np.array([d[f"rank_sss_tau{tau}"] for d in data]),
        "ssu": np.array([d[f"rank_ssu_tau{tau}"] for d in data]),
        "all": np.array([d[f"rank_all_tau{tau}"] for d in data]),
    }

has_spectra = _SPECTRA_PATH.exists()
if has_spectra:
    spectra = np.load(_SPECTRA_PATH)

# ── Style configuration ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLOR_SSS = "#2563eb"
COLOR_SSU = "#dc2626"
COLOR_ALL = "#6b7280"
COLOR_RATIO = "#059669"
SAFETY_LAYER_COLOR = "#fef3c7"

TAU_COLORS = {
    "0.4": "#94a3b8",
    "0.6": "#60a5fa",
    "0.7": "#a78bfa",
    "0.8": "#f97316",
    "0.9": "#ef4444",
}

bar_width = 0.28
x = np.arange(len(layers))
safety_lo = np.searchsorted(layers, 6) - 0.5
safety_hi = np.searchsorted(layers, 14) + 0.5
safety_center = (np.searchsorted(layers, 6) + np.searchsorted(layers, 14)) / 2
safety_mask = (layers >= 6) & (layers <= 14)

SUBTITLE = (
    r"LLaVA-1.5-7B  ·  Effective rank = min $k$ s.t. top-$k$ singular values"
    r" capture $\geq \tau$ of total variance"
)


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {path}")
    plt.close(fig)


# ── Plot 1: Effective rank at τ=0.9 ──────────────────────────────────────
tau_main = "0.9"
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

ax.bar(x - bar_width, ranks[tau_main]["sss"], bar_width,
       color=COLOR_SSS, alpha=0.8, label="SSS",
       edgecolor="white", linewidth=0.3)
ax.bar(x, ranks[tau_main]["ssu"], bar_width,
       color=COLOR_SSU, alpha=0.8, label="SSU",
       edgecolor="white", linewidth=0.3)
ax.bar(x + bar_width, ranks[tau_main]["all"], bar_width,
       color=COLOR_ALL, alpha=0.5, label="All (combined)",
       edgecolor="white", linewidth=0.3)

ax.set_ylabel(r"Effective Rank ($\tau$ = 0.9)")
ax.set_title(
    r"Effective Rank at $\tau$ = 0.9: SSS vs SSU vs Combined" + "\n" + SUBTITLE
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.text(safety_center, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

fig.tight_layout()
_save(fig, "effective_rank_tau09.png")

# ── Plot 2: Rank difference (SSS − SSU) ──────────────────────────────────
rank_diff = ranks[tau_main]["sss"] - ranks[tau_main]["ssu"]
fig, ax = plt.subplots(figsize=(14, 4))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

bar_colors = [COLOR_RATIO if d >= 0 else "#b91c1c" for d in rank_diff]
ax.bar(x, rank_diff, width=0.6, color=bar_colors, alpha=0.8,
       edgecolor="white", linewidth=0.3)
ax.axhline(y=0, color="#888", linestyle="-", linewidth=0.8, alpha=0.5)

ax.set_ylabel(r"Rank Difference (SSS $-$ SSU)")
ax.set_title(
    r"Rank Gap at $\tau$ = 0.9: SSS $-$ SSU "
    "(positive = SSU more concentrated)\n" + SUBTITLE
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")

mean_diff_safety = rank_diff[safety_mask].mean()
ax.text(safety_center, ax.get_ylim()[1] * 0.88,
        f"mean gap (L6\u201314): {mean_diff_safety:.1f}",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#059669", alpha=0.9))

fig.tight_layout()
_save(fig, "rank_gap.png")

# ── Plot 3: Multi-threshold SSS vs SSU ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

for tau in thresholds:
    key = str(tau)
    sss_vals = ranks[key]["sss"]
    ssu_vals = ranks[key]["ssu"]
    color = TAU_COLORS[key]
    ax.plot(x, sss_vals, "o-", color=color, markersize=3.5, linewidth=1.3,
            alpha=0.85, label=f"SSS $\\tau$={tau}")
    ax.plot(x, ssu_vals, "s--", color=color, markersize=3.5, linewidth=1.3,
            alpha=0.55)

ax.plot([], [], "o-", color="black", markersize=3.5, linewidth=1.3, label="solid = SSS")
ax.plot([], [], "s--", color="black", markersize=3.5, linewidth=1.3, alpha=0.55,
        label="dashed = SSU")
ax.set_ylabel("Effective Rank")
ax.set_title(
    r"Effective Rank Across Energy Thresholds ($\tau$)" + "\n" + SUBTITLE
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)

fig.tight_layout()
_save(fig, "multi_threshold.png")

# ── Plot 4: Cumulative variance spectra ───────────────────────────────────
if has_spectra:
    fig, ax = plt.subplots(figsize=(8, 5))
    selected_layers = [3, 7, 10, 25]
    layer_colors = ["#6b7280", "#dc2626", "#b91c1c", "#2563eb"]
    n_components = 30

    for layer_idx, color in zip(selected_layers, layer_colors):
        sss_key = f"sss_layer_{layer_idx}"
        ssu_key = f"ssu_layer_{layer_idx}"
        if sss_key in spectra and ssu_key in spectra:
            sss_spec = spectra[sss_key][:n_components]
            ssu_spec = spectra[ssu_key][:n_components]
            components = np.arange(1, len(sss_spec) + 1)

            ax.plot(components, np.cumsum(sss_spec), "o-", color=color,
                    markersize=3, linewidth=1.3, alpha=0.85,
                    label=f"L{layer_idx} SSS")
            ax.plot(components, np.cumsum(ssu_spec), "s--", color=color,
                    markersize=3, linewidth=1.3, alpha=0.55,
                    label=f"L{layer_idx} SSU")

    ax.axhline(y=0.9, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(n_components - 0.5, 0.91, r"$\tau$ = 0.9", fontsize=8,
            color="#ef4444", va="bottom", ha="right")
    ax.axhline(y=0.8, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
    ax.text(n_components - 0.5, 0.81, r"$\tau$ = 0.8", fontsize=8,
            color="#f97316", va="bottom", ha="right")

    ax.set_xlabel("Singular Value Component")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative Variance Spectra (first 30 components)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save(fig, "cumulative_variance_spectra.png")
