"""
Shared extraction and analysis utilities used by all three methods.

Provides:
  - Activation caching (save / load per-sample .npz files)
  - Last-token activation extraction from hidden_states tuples
  - Cross-modal attention extraction
  - Fisher Discriminant Ratio (FDR) computation
  - General GPU memory cleanup
"""

import gc
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ── GPU helpers ───────────────────────────────────────────────────────────────

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Activation helpers ────────────────────────────────────────────────────────

def get_last_token_activations(hidden_states) -> Dict[int, np.ndarray]:
    """
    Extract last-token hidden state from each layer.

    Args:
        hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden_dim)

    Returns:
        dict mapping layer_idx -> np.ndarray of shape (hidden_dim,), float32
    """
    return {
        l: hidden_states[l][0, -1, :].detach().cpu().float().numpy()
        for l in range(len(hidden_states))
    }


# ── Activation cache (disk-based) ────────────────────────────────────────────

class ActivationCache:
    """
    Simple on-disk cache for per-sample hidden state activations.

    File naming:
        {cache_dir}/{sample_id}_{suffix}.npz
    Each .npz has keys "layer_{l}" for each layer l.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sample_id: int, suffix: str) -> Path:
        return self.cache_dir / f"sample_{sample_id}_{suffix}.npz"

    def exists(self, sample_id: int, suffix: str = "vl") -> bool:
        return self._path(sample_id, suffix).exists()

    def save(self, sample_id: int, activations: Dict[int, np.ndarray], suffix: str = "vl"):
        """Save dict {layer_idx: array} to .npz."""
        np.savez_compressed(
            self._path(sample_id, suffix),
            **{f"layer_{l}": arr for l, arr in activations.items()}
        )

    def load(self, sample_id: int, suffix: str = "vl") -> Dict[int, np.ndarray]:
        """Load dict {layer_idx: array} from .npz."""
        p = self._path(sample_id, suffix)
        if not p.exists():
            raise FileNotFoundError(p)
        data = np.load(p)
        return {int(k.replace("layer_", "")): data[k] for k in data.files}

    def load_or_none(self, sample_id: int, suffix: str = "vl") -> Optional[Dict[int, np.ndarray]]:
        try:
            return self.load(sample_id, suffix)
        except FileNotFoundError:
            return None


# ── Cross-modal attention ─────────────────────────────────────────────────────

def extract_cross_modal_attention(
    attentions,
    text_positions: List[int],
    img_start: int,
    img_end: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    For each attention head (layer, head), compute the cross-modal attention
    weight from text query tokens to each visual token.

    a_j^{l,h} = max_{t in T} A^{l,h}[t, j]

    Args:
        attentions:     tuple of attention tensors, one per transformer layer.
                        Each has shape (1, num_heads, seq_len, seq_len).
        text_positions: list of token positions that are TEXT tokens.
        img_start:      first image token position in expanded sequence.
        img_end:        one-past-last image token position.

    Returns:
        dict mapping (layer, head) -> np.ndarray of shape (num_image_tokens,)
        representing max cross-modal attention to each image token position.
    """
    result: Dict[Tuple[int, int], np.ndarray] = {}
    text_pos = np.array(text_positions, dtype=np.int64)

    for l, attn_l in enumerate(attentions):
        # attn_l: (1, num_heads, seq_len, seq_len) — causal attention
        attn_np = attn_l[0].detach().cpu().float().numpy()  # (num_heads, seq_len, seq_len)
        num_heads, seq_len, _ = attn_np.shape

        # Clamp indices to valid range
        valid_text = text_pos[text_pos < seq_len]
        img_s = min(img_start, seq_len)
        img_e = min(img_end, seq_len)

        if img_s >= img_e or len(valid_text) == 0:
            continue

        for h in range(num_heads):
            # text_to_img[t, j] = attention from text token t to image token j
            text_to_img = attn_np[h][np.ix_(valid_text, np.arange(img_s, img_e))]
            # Take max over text tokens for each image token
            cross_modal = text_to_img.max(axis=0)  # shape: (num_image_tokens,)
            result[(l, h)] = cross_modal

    return result


def rank_heads_by_visual_attention(
    cross_modal_dict: Dict[Tuple[int, int], np.ndarray],
    top_n: int = 3,
) -> List[Tuple[int, int]]:
    """
    Rank attention heads by their total attention mass to visual tokens.

    Returns the top_n (layer, head) tuples with highest summed cross-modal attention.
    """
    scores = {key: float(arr.sum()) for key, arr in cross_modal_dict.items()}
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return ranked[:top_n]


def effective_visual_attention(
    cross_modal_dict: Dict[Tuple[int, int], np.ndarray],
    top_heads: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute effective visual attention by averaging over the top-n heads.

    Returns array of shape (num_image_tokens,).
    """
    arrays = [cross_modal_dict[h] for h in top_heads if h in cross_modal_dict]
    if not arrays:
        return np.zeros(0)
    return np.mean(arrays, axis=0)


# ── Fisher Discriminant Ratio ─────────────────────────────────────────────────

def compute_fdr(
    X_sss: np.ndarray,
    X_ssu: np.ndarray,
    pca_dim: Optional[int] = None,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Fisher Discriminant Ratio between SSS and SSU activation sets.

    FDR = (mu_sss - mu_ssu)^T @ (Sigma_sss + Sigma_ssu)^{-1} @ (mu_sss - mu_ssu)

    High FDR = representations are more separable between the two classes.

    Args:
        X_sss: shape (N_sss, d)
        X_ssu: shape (N_ssu, d)
        pca_dim: if not None, PCA-reduce to this dimension first.
                 Recommended: min(N_sss, N_ssu) - 1  when N < d.
        epsilon: regularisation added to diagonal of covariance sum.

    Returns:
        FDR scalar (float), or float('nan') on failure.
    """
    if len(X_sss) < 2 or len(X_ssu) < 2:
        return float("nan")

    # PCA reduction for high-dimensional / small-sample regime
    if pca_dim is None:
        n_min = min(len(X_sss), len(X_ssu))
        d = X_sss.shape[1]
        if n_min < d:
            pca_dim = max(1, n_min - 1)

    if pca_dim is not None:
        from sklearn.decomposition import PCA
        X_all = np.vstack([X_sss, X_ssu])
        n_components = min(pca_dim, X_all.shape[0] - 1, X_all.shape[1])
        if n_components < 1:
            return float("nan")
        pca = PCA(n_components=n_components)
        pca.fit(X_all)
        X_sss = pca.transform(X_sss)
        X_ssu = pca.transform(X_ssu)

    d = X_sss.shape[1]
    mu_sss = X_sss.mean(axis=0)
    mu_ssu = X_ssu.mean(axis=0)

    Sigma_sss = np.cov(X_sss.T) if len(X_sss) > 1 else np.zeros((d, d))
    Sigma_ssu = np.cov(X_ssu.T) if len(X_ssu) > 1 else np.zeros((d, d))

    if d == 1:
        Sigma_sss = np.array([[float(Sigma_sss)]])
        Sigma_ssu = np.array([[float(Sigma_ssu)]])

    diff = mu_sss - mu_ssu
    Sigma_sum = Sigma_sss + Sigma_ssu + epsilon * np.eye(d)

    try:
        fdr = float(diff @ np.linalg.solve(Sigma_sum, diff))
    except np.linalg.LinAlgError:
        fdr = float("nan")

    return fdr


# ── Result I/O helpers ────────────────────────────────────────────────────────

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy scalars/arrays to native Python for JSON serialisation
    def _convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    import json as _json

    class _NpEncoder(_json.JSONEncoder):
        def default(self, o):
            return _convert(o)

    with open(path, "w") as f:
        _json.dump(obj, f, indent=2, cls=_NpEncoder)
    print(f"  Saved → {path}")


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_pickle(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def save_npz(arrays: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"  Saved → {path}")


# ── Subspace analysis utilities ──────────────────────────────────────────────

def load_activation_matrix(cache: ActivationCache,
                           sample_ids: list,
                           layer: int,
                           suffix: str) -> np.ndarray:
    """Load activations for multiple samples at a single layer, stacked into a matrix.

    Args:
        cache: ActivationCache instance pointing at activations/ directory
        sample_ids: list of sample ID strings
        layer: transformer layer index
        suffix: activation file suffix ('vl', 'tt', etc.)

    Returns:
        np.ndarray of shape (N_samples, hidden_dim)
    """
    rows = []
    for sid in sample_ids:
        acts = cache.load_or_none(sid, suffix=suffix)
        if acts is None:
            raise FileNotFoundError(f"Missing activation: sample_{sid}_{suffix}.npz")
        rows.append(acts[layer])
    return np.stack(rows)


def load_modality_shift_matrix(cache: ActivationCache,
                               sample_ids: list,
                               layer: int) -> np.ndarray:
    """Compute modality shift vectors (VL - TT) for multiple samples at a layer.

    Returns:
        np.ndarray of shape (N_samples, hidden_dim)
    """
    vl = load_activation_matrix(cache, sample_ids, layer, suffix="vl")
    tt = load_activation_matrix(cache, sample_ids, layer, suffix="tt")
    return vl - tt


def effective_rank(matrix: np.ndarray, tau: float = 0.9) -> int:
    """Compute effective rank: minimum number of singular values
    explaining >= tau fraction of total variance.

    Args:
        matrix: (N, d) data matrix (will be mean-centered)
        tau: energy threshold (0 < tau <= 1)

    Returns:
        k: effective rank (int)
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, sigmas, _ = np.linalg.svd(centered, full_matrices=False)
    energy = np.cumsum(sigmas ** 2)
    total = energy[-1]
    if total < 1e-12:
        return 0
    k = int(np.searchsorted(energy / total, tau)) + 1
    return min(k, len(sigmas))


def extract_subspace(matrix: np.ndarray, k: int,
                     center: bool = True) -> np.ndarray:
    """Extract top-k principal directions from a data matrix via SVD.

    Args:
        matrix: (N, d) data matrix
        k: number of components to keep
        center: whether to mean-center before SVD

    Returns:
        V_k: (k, d) orthonormal basis of the top-k subspace (right singular vectors)
    """
    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, sigmas, Vt = np.linalg.svd(matrix, full_matrices=False)
    return Vt[:k]  # (k, d)


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces.

    Args:
        V1: (k1, d) orthonormal basis of subspace 1
        V2: (k2, d) orthonormal basis of subspace 2

    Returns:
        angles: array of min(k1, k2) principal angles in radians
    """
    M = V1 @ V2.T  # (k1, k2)
    _, sigmas, _ = np.linalg.svd(M, full_matrices=False)
    # Clamp to [0, 1] for numerical stability before arccos
    sigmas = np.clip(sigmas, 0.0, 1.0)
    return np.arccos(sigmas)


def subspace_overlap(V1: np.ndarray, V2: np.ndarray) -> float:
    """Compute mean cosine of principal angles between two subspaces.
    Returns 1.0 for identical subspaces, 0.0 for fully orthogonal.

    Args:
        V1: (k1, d) orthonormal basis
        V2: (k2, d) orthonormal basis

    Returns:
        overlap: float in [0, 1]
    """
    angles = principal_angles(V1, V2)
    return float(np.mean(np.cos(angles)))
