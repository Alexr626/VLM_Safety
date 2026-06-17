"""Textual VTI direction extraction via wrapper forwards + PCA."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.paths import coco_train2014_dir, experiment_artifacts_dir, vti_demos_path

from .pca import PCA


def load_vti_demos(
    demos_path: Optional[Path] = None,
    num_demos: int = 70,
    seed: int = 42,
) -> List[dict]:
    """Load and subsample paired clean/hallucinated caption demos."""
    path = Path(demos_path) if demos_path is not None else vti_demos_path()
    if not path.exists():
        raise FileNotFoundError(
            f"VTI demo file not found: {path}. "
            "Expected data/vti/demos.jsonl (copied from the authors' VTI repo)."
        )
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    if num_demos < len(data):
        data = rng.sample(data, num_demos)
    return data


def load_demo_image(demo: dict, image_root: Optional[Path] = None) -> Image.Image:
    """Load a demo COCO train2014 image."""
    root = image_root or coco_train2014_dir()
    img_path = root / demo["image"]
    if not img_path.exists():
        raise FileNotFoundError(
            f"Demo image missing: {img_path}. "
            "Run: python data_scripts/download_chair.py --with-train2014"
        )
    return Image.open(img_path).convert("RGB")


def _demo_prompt(demo: dict, caption: str) -> str:
    """Build the VL text for direction extraction (question + caption body)."""
    question = demo.get("question", "")
    if question and not question.endswith((" ", "\n")):
        return f"{question} {caption}"
    return f"{question}{caption}"


def get_hiddenstates(wrapper, demos: List[dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Run paired forwards and collect last-token states per layer.

    Returns a list of ``(hallucinated, clean)`` tensors, each of shape
    ``(num_layers + 1, hidden_dim)`` — index 0 is the embedding row.
    Order matches the reference: style 0 = ``h_value``, style 1 = ``value``.
    """
    h_all: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for demo in demos:
            image = load_demo_image(demo)
            embeddings_for_styles: List[torch.Tensor] = []
            for caption in (demo["h_value"], demo["value"]):
                text = _demo_prompt(demo, caption)
                hidden_states, _, _ = wrapper.forward_vl(image, text)
                n_expected = wrapper.num_layers + 1
                if len(hidden_states) != n_expected:
                    raise RuntimeError(
                        f"hidden_states length {len(hidden_states)} != "
                        f"num_layers+1 ({n_expected}) for {wrapper.model_name}"
                    )
                per_layer = torch.stack(
                    [hidden_states[layer][0, -1, :].detach().cpu().float()
                     for layer in range(len(hidden_states))],
                    dim=0,
                )  # (num_layers + 1, hidden_dim)
                embeddings_for_styles.append(per_layer)
            h_all.append((embeddings_for_styles[0], embeddings_for_styles[1]))
    return h_all


def obtain_textual_vti(
    wrapper,
    demos: List[dict],
    rank: int = 1,
) -> torch.Tensor:
    """Compute per-layer textual steering directions.

    Fits PCA (rank=1) on flattened ``clean - hallucinated`` last-token states
    across demos, then reshapes to ``(num_layers + 1, hidden_dim)``.
    Caller should slice ``[1:]`` to align with decoder layers 0..L-1.
    """
    hidden_states = get_hiddenstates(wrapper, demos)
    hidden_states_all: List[torch.Tensor] = []
    for neg, pos in hidden_states:
        hidden_states_all.append((pos.view(-1) - neg.view(-1)).float())

    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (
        pca.components_.sum(dim=1, keepdim=True) + pca.mean_
    ).mean(0).view(
        hidden_states[-1][0].size(0),
        hidden_states[-1][0].size(1),
    )
    return direction


def direction_cache_path(
    model_short: str,
    num_demos: int,
    rank: int,
    seed: int,
    cache_dir: Optional[Path] = None,
) -> Path:
    root = cache_dir or experiment_artifacts_dir("vti", model_short)
    return root / f"textual_directions_nd{num_demos}_rank{rank}_seed{seed}.npz"


def save_directions(
    directions: np.ndarray,
    path: Path,
) -> None:
    """Save ``(num_layers, hidden_dim)`` directions keyed by ``layer_{i}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"layer_{i}": directions[i] for i in range(directions.shape[0])}
    np.savez(path, **arrays, num_layers=directions.shape[0],
             hidden_dim=directions.shape[1])


def load_directions(path: Path) -> np.ndarray:
    data = np.load(path)
    n = int(data["num_layers"])
    d = int(data["hidden_dim"])
    return np.stack([data[f"layer_{i}"] for i in range(n)], axis=0).astype(np.float32)


def compute_or_load_textual_directions(
    wrapper,
    model_short: str,
    num_demos: int = 70,
    rank: int = 1,
    seed: int = 42,
    demos_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Return decoder-layer directions ``(num_layers, hidden_dim)`` float32."""
    cache_path = direction_cache_path(
        model_short, num_demos, rank, seed, cache_dir=cache_dir,
    )
    if cache_path.exists() and not force_recompute:
        directions = load_directions(cache_path)
        if directions.shape != (wrapper.num_layers, wrapper.hidden_dim):
            raise RuntimeError(
                f"Cached directions shape {directions.shape} != "
                f"({wrapper.num_layers}, {wrapper.hidden_dim})"
            )
        return directions

    demos = load_vti_demos(demos_path=demos_path, num_demos=num_demos, seed=seed)
    full = obtain_textual_vti(wrapper, demos, rank=rank)
    directions = full[1:].detach().cpu().float().numpy()
    if directions.shape != (wrapper.num_layers, wrapper.hidden_dim):
        raise RuntimeError(
            f"Direction shape {directions.shape} != "
            f"({wrapper.num_layers}, {wrapper.hidden_dim})"
        )
    save_directions(directions, cache_path)
    return directions
