"""
Shared extraction and I/O utilities.

Provides:
  - General GPU memory cleanup
  - Last-token activation extraction from hidden_states tuples
  - Activation caching (save / load per-sample .npz files)
  - JSON / pickle / npz result I/O
"""

import gc
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

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
        {cache_dir}/sample_{sample_id}_{suffix}.npz
    Each .npz has keys "layer_{l}" for each layer l. Arrays can be any shape,
    e.g. (hidden_dim,) for one position or (n_steps, hidden_dim) for many.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sample_id, suffix: str) -> Path:
        return self.cache_dir / f"sample_{sample_id}_{suffix}.npz"

    def exists(self, sample_id, suffix: str = "vl") -> bool:
        return self._path(sample_id, suffix).exists()

    def save(self, sample_id, activations: Dict[int, np.ndarray], suffix: str = "vl"):
        """Save dict {layer_idx: array} to .npz."""
        np.savez_compressed(
            self._path(sample_id, suffix),
            **{f"layer_{l}": arr for l, arr in activations.items()}
        )

    def load(self, sample_id, suffix: str = "vl") -> Dict[int, np.ndarray]:
        """Load dict {layer_idx: array} from .npz."""
        p = self._path(sample_id, suffix)
        if not p.exists():
            raise FileNotFoundError(p)
        data = np.load(p)
        return {int(k.replace("layer_", "")): data[k] for k in data.files}

    def load_or_none(self, sample_id, suffix: str = "vl") -> Optional[Dict[int, np.ndarray]]:
        try:
            return self.load(sample_id, suffix)
        except FileNotFoundError:
            return None


def load_activation_matrix(cache: ActivationCache,
                           sample_ids: List,
                           layer: int,
                           suffix: str = "vl") -> np.ndarray:
    """Stack one layer's cached activations across samples -> (N, ...)."""
    rows = []
    for sid in sample_ids:
        acts = cache.load_or_none(sid, suffix=suffix)
        if acts is None:
            raise FileNotFoundError(f"Missing activation: sample_{sid}_{suffix}.npz")
        rows.append(acts[layer])
    return np.stack(rows)


# ── Result I/O helpers ────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)
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
