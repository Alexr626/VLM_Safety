"""
Plot Experiment B: Safety Subspace Decomposition
Generates individual plots analyzing how the SSU-vs-SSS distortion distributes
across the top-10 principal safety components obtained via SVD.

Reads from: ../experiment_b_safety_decomposition/per_component_projections.json
            ../experiment_b_safety_decomposition/dominant_vs_method2_cosine.json
            ../experiment_b_safety_decomposition/safety_subspace_spectra.npz
Saves to:   ../experiment_b_safety_decomposition/plots/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_OUTPUTS_DIR = _SCRIPT_DIR.parent


def _parse_model() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    args, _ = p.parse_known_args()
    return args.model.split("/")[-1]


_MODEL_DIR = _OUTPUTS_DIR / _parse_model()
_PROJ_PATH = _MODEL_DIR / "experiment_b_safety_decomposition" / "per_component_projections.json"
_COSINE_PATH = _MODEL_DIR / "experiment_b_safety_decomposition" / "dominant_vs_method2_cosine.json"
_SPECTRA_PATH = _MODEL_DIR / "experiment_b_safety_decomposition" / "safety_subspace_spectra.npz"
_OUTPUT_DIR = _MODEL_DIR / "experiment_b_safety_decomposition" / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for path, label in [
    (_PROJ_PATH, "per-component projections"),
    (_COSINE_PATH, "dominant vs method-2 cosine"),
    (_SPECTRA_PATH, "safety subspace spectra"),
]:
    if not path.exists():
        print(
            f"ERROR: {path} not found ({label}).\n"
            "Please run Experiment B first:\n"
            "  python diagnostic_exploratory_tests/followup_subspace_analysis/"
            "experiment_b_safety_decomposition.py"
        )
        sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────
with open(_PROJ_PATH) as f:
    proj_data = json.load(f)

with open(_COSINE_PATH) as f:
    cosine_data = json.load(f)

spectra = np.load(_SPECTRA_PATH)

proj_data = [d for d in proj_data if d["layer"] > 0]
cosine_data = [d for d in cosine_data if d["layer"] > 0]

all_layers = sorted(set(d["layer"] for d in proj_data))
all_components = sorted(set(d["component_k"] for d in proj_data))
n_layers = len(all_layers)
n_components = len(all_components)

proj_lookup = {(d["layer"], d["component_k"]): d for d in proj_data}

cos_layers = np.array([d["layer"] for d in cosine_data])
cos_values = np.array([d["cosine_sim_dominant_vs_s_l"] for d in cosine_data])

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
SAFETY_LAYER_COLOR = "#fef3c7"

layer_arr = np.array(all_layers)
x_layers = np.arange(n_layers)
safety_lo_idx = np.searchsorted(layer_arr, 6) - 0.5
safety_hi_idx = np.searchsorted(layer_arr, 14) + 0.5
safety_center_idx = (np.searchsorted(layer_arr, 6) + np.searchsorted(layer_arr, 14)) / 2

SUBTITLE = "LLaVA-1.5-7B  ·  SSU vs SSS Projections onto Top-10 Safety SVD Components"


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {path}")
    plt.close(fig)


# ── Plot 1: Dominant Component vs Method 2 Safety Direction ───────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(safety_lo_idx, safety_hi_idx, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

abs_cos = np.abs(cos_values)
x_cos = np.arange(len(cos_layers))

ax.plot(x_cos, abs_cos, "o-", color="#7c3aed", markersize=5, linewidth=1.5,
        alpha=0.85, label="|cosine similarity|")
ax.axhline(y=0.9, color="#ef4444", linestyle="--", linewidth=1, alpha=0.6)
ax.text(len(x_cos) - 0.5, 0.905, "0.9 reference", fontsize=8, color="#ef4444",
        va="bottom", ha="right")

ax.set_ylabel("|Cosine Similarity|")
ax.set_title("Dominant Component vs Method 2 Safety Direction\n" + SUBTITLE)
ax.set_xticks(x_cos)
ax.set_xticklabels([str(l) for l in cos_layers], fontsize=7)
ax.set_xlabel("Transformer Layer")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

ax.text(safety_center_idx, ax.get_ylim()[1] * 0.07,
        "safety-critical layers (6\u201314)",
        ha="center", va="bottom", fontsize=8.5, color="#b45309", fontstyle="italic")

safety_mask_cos = (cos_layers >= 6) & (cos_layers <= 14)
mean_abs_cos_safety = abs_cos[safety_mask_cos].mean()
ax.text(safety_center_idx, 0.5,
        f"mean |cos| (L6\u201314): {mean_abs_cos_safety:.4f}",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#059669", alpha=0.9))

fig.tight_layout()
_save(fig, "dominant_vs_method2_cosine.png")

# ── Plot 2: Heatmap of SSU-SSS Projection Difference ─────────────────────
diff_matrix = np.zeros((n_components, n_layers))
for i, layer in enumerate(all_layers):
    for j, comp in enumerate(all_components):
        rec = proj_lookup.get((layer, comp))
        if rec is not None:
            diff_matrix[j, i] = rec["SSU_mean_proj"] - rec["SSS_mean_proj"]

vmax = np.max(np.abs(diff_matrix))
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(14, 5))
ax.grid(False)
im = ax.imshow(diff_matrix, aspect="auto", cmap="RdBu_r", norm=norm,
               interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("SSU $-$ SSS Mean Projection", fontsize=9)

ax.set_xticks(np.arange(n_layers))
ax.set_xticklabels([str(l) for l in all_layers], fontsize=7)
ax.set_yticks(np.arange(n_components))
ax.set_yticklabels([str(k) for k in all_components], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Safety Component $k$")
ax.set_title("SSU$-$SSS Projection Difference by Safety Component\n" + SUBTITLE)

safety_layer_indices = [i for i, l in enumerate(all_layers) if 6 <= l <= 14]
if safety_layer_indices:
    lo_col = safety_layer_indices[0] - 0.5
    hi_col = safety_layer_indices[-1] + 0.5
    ax.axvline(x=lo_col, color="#b45309", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=hi_col, color="#b45309", linestyle="--", linewidth=1, alpha=0.7)
    ax.text((lo_col + hi_col) / 2, -0.8, "safety-critical (6\u201314)",
            ha="center", va="bottom", fontsize=7.5, color="#b45309",
            fontstyle="italic")

fig.tight_layout()
_save(fig, "ssu_sss_projection_heatmap.png")

# ── Plot 3: Variance Explained by Top Components ─────────────────────────
selected_layers_c = [3, 7, 10, 14, 20, 28]
n_top_components = 5
bar_width_c = 0.13
component_colors = ["#2563eb", "#dc2626", "#059669", "#f97316", "#7c3aed"]

fig, ax = plt.subplots(figsize=(10, 5))
x_sel = np.arange(len(selected_layers_c))

for k_idx in range(n_top_components):
    var_vals = []
    for layer in selected_layers_c:
        rec = proj_lookup.get((layer, all_components[k_idx]))
        if rec is not None:
            var_vals.append(rec["variance_explained_frac"])
        else:
            var_vals.append(0.0)
    offset = (k_idx - (n_top_components - 1) / 2) * bar_width_c
    ax.bar(x_sel + offset, var_vals, bar_width_c,
           color=component_colors[k_idx], alpha=0.8,
           label=f"Component {all_components[k_idx]}",
           edgecolor="white", linewidth=0.3)

ax.set_xticks(x_sel)
ax.set_xticklabels([f"L{l}" for l in selected_layers_c], fontsize=9)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Variance Explained (fraction)")
ax.set_title("Variance Explained by Top Safety Components\n" + SUBTITLE)
ax.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)

comp0_fracs = []
for layer in selected_layers_c:
    rec = proj_lookup.get((layer, all_components[0]))
    if rec is not None:
        comp0_fracs.append(rec["variance_explained_frac"])
mean_comp0_frac = np.mean(comp0_fracs) if comp0_fracs else 0.0
ax.text(0.02, 0.95,
        f"Component 0 mean: {mean_comp0_frac:.1%} of variance",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#059669", alpha=0.9))

fig.tight_layout()
_save(fig, "variance_explained.png")

# ── Plot 4: P-values per Component for Safety-Critical Layers ────────────
sig_layers = [7, 10, 14]
sig_colors = ["#dc2626", "#2563eb", "#059669"]
sig_markers = ["o", "s", "D"]

fig, ax = plt.subplots(figsize=(10, 5))
components_arr = np.array(all_components)

for layer, color, marker in zip(sig_layers, sig_colors, sig_markers):
    p_vals = []
    for comp in all_components:
        rec = proj_lookup.get((layer, comp))
        if rec is not None and rec["p_value"] is not None and not np.isnan(rec["p_value"]):
            p_vals.append(np.clip(rec["p_value"], 1e-300, 1.0))
        else:
            p_vals.append(1.0)
    ax.plot(components_arr, p_vals, f"{marker}-", color=color,
            markersize=6, linewidth=1.5, alpha=0.85,
            label=f"Layer {layer}")

ax.axhline(y=0.05, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
ax.text(n_components - 0.7, 0.05, "p = 0.05", fontsize=8, color="#ef4444",
        va="bottom", ha="right")
ax.axhline(y=0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
ax.text(n_components - 0.7, 0.001, "p = 0.001", fontsize=8, color="#f97316",
        va="bottom", ha="right")

ax.set_yscale("log")
ax.set_ylabel("p-value (log scale)")
ax.set_xlabel("Safety Component $k$")
ax.set_title("Statistical Significance: SSU vs SSS per Component\n" + SUBTITLE)
ax.set_xticks(components_arr)
ax.set_xticklabels([str(k) for k in all_components], fontsize=9)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

all_p = []
for layer in sig_layers:
    for comp in all_components:
        rec = proj_lookup.get((layer, comp))
        if rec is not None and rec["p_value"] is not None and not np.isnan(rec["p_value"]):
            all_p.append(np.clip(rec["p_value"], 1e-300, 1.0))
if all_p:
    min_p = max(min(all_p), 1e-260)
    ax.set_ylim(min_p / 10, 2)

fig.tight_layout()
_save(fig, "p_values_per_component.png")
