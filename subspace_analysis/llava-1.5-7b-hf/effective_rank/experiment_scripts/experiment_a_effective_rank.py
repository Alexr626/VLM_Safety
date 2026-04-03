#!/usr/bin/env python3
"""
Experiment A: Effective Rank of Modality Shift Spaces
=====================================================
Measure the linear dimensionality of the modality-induced shift space
for SSS and SSU examples separately.

Low effective rank  → distortion is structured, amenable to geometric correction.
High effective rank → distortion is diffuse, needs more complex intervention.

Dependencies:
  - activations/sample_{id}_vl.npz and sample_{id}_tt.npz
  - method2_activation_shift/sample_metadata.json

Outputs (under outputs/{model_name}/experiment_a_effective_rank/):
  - effective_rank_results.json
  - singular_spectra.npz

Usage:
  cd ~/dev/VLM_Safety
  python experiment_a_effective_rank.py
"""

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent                          # effective_rank/
_MODEL_NAME = _EXPERIMENT_DIR.parent.name                     # llava-1.5-7b-hf
_PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent.parent          # VLM_Safety/
_DATA = _PROJECT_ROOT / "data"
_EXPERIMENT_ARTIFACTS = _PROJECT_ROOT / "experiment_artifacts" / _MODEL_NAME / "effective_rank"
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_modality_shift_matrix, effective_rank,
    load_json, save_json, save_npz,
)


def main():
    _HOLISAFE_ACT = _DATA / "holisafe-bench" / "activations" / _MODEL_NAME
    cache = ActivationCache(str(_HOLISAFE_ACT))

    metadata = load_json(str(_HOLISAFE_ACT / "sample_metadata.json"))
    sss_ids = [s["id"] for s in metadata if s["label"] == "SSS"]
    ssu_ids = [s["id"] for s in metadata if s["label"] == "SSU"]

    safety_vecs = np.load(
        _PROJECT_ROOT / "experiment_artifacts" / _MODEL_NAME / "vl_activation_shift" / "safety_direction_vectors.npz")
    layers = sorted(int(k.replace("layer_", "")) for k in safety_vecs.files)

    print(f"SSS: {len(sss_ids)}, SSU: {len(ssu_ids)}, Layers: {len(layers)}")

    results_dir = _EXPERIMENT_DIR / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _EXPERIMENT_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    thresholds = [0.4, 0.6, 0.7, 0.8, 0.9]
    results = []
    spectra = {}

    for l in layers:
        print(f"  Layer {l:2d}...", end=" ", flush=True)

        M_sss = load_modality_shift_matrix(cache, sss_ids, l)
        M_ssu = load_modality_shift_matrix(cache, ssu_ids, l)
        M_all = np.concatenate([M_sss, M_ssu], axis=0)

        layer_result = {"layer": l}
        for tau in thresholds:
            layer_result[f"rank_sss_tau{tau}"] = effective_rank(M_sss, tau)
            layer_result[f"rank_ssu_tau{tau}"] = effective_rank(M_ssu, tau)
            layer_result[f"rank_all_tau{tau}"] = effective_rank(M_all, tau)

        results.append(layer_result)

        # Full singular value spectra (normalized variance explained)
        M_sss_c = M_sss - M_sss.mean(axis=0, keepdims=True)
        M_ssu_c = M_ssu - M_ssu.mean(axis=0, keepdims=True)

        _, s_sss, _ = np.linalg.svd(M_sss_c, full_matrices=False)
        _, s_ssu, _ = np.linalg.svd(M_ssu_c, full_matrices=False)

        spectra[f"sss_layer_{l}"] = (s_sss ** 2 / (s_sss ** 2).sum()).astype(np.float32)
        spectra[f"ssu_layer_{l}"] = (s_ssu ** 2 / (s_ssu ** 2).sum()).astype(np.float32)

        r90_sss = layer_result["rank_sss_tau0.9"]
        r90_ssu = layer_result["rank_ssu_tau0.9"]
        print(f"rank@90%: SSS={r90_sss}, SSU={r90_ssu}")

    save_json(results, str(results_dir / "effective_rank_results.json"))
    save_npz(spectra, str(_EXPERIMENT_ARTIFACTS / "singular_spectra.npz"))

    # Summary for safety-critical layers
    print(f"\nEffective rank at τ=0.9 for safety-critical layers (6–14):")
    for r in results:
        if 6 <= r["layer"] <= 14:
            diff = r["rank_ssu_tau0.9"] - r["rank_sss_tau0.9"]
            print(f"  Layer {r['layer']:2d}: SSS={r['rank_sss_tau0.9']:3d}  "
                  f"SSU={r['rank_ssu_tau0.9']:3d}  diff={diff:+d}")


if __name__ == "__main__":
    main()
