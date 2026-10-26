"""Central path helpers for data, experiment outputs, and the LiSeCo clone."""

from pathlib import Path
from typing import Optional

_PROJECT_ROOT: Optional[Path] = None


def project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return _PROJECT_ROOT


# ── Experiments ──────────────────────────────────────────────────────────────

def experiments_root() -> Path:
    return project_root() / "experiments"


def experiment_dir(experiment: str) -> Path:
    return experiments_root() / experiment


def experiment_results_dir(experiment: str, model_short: str) -> Path:
    """``experiments/{experiment}/outputs/{model_short}/results/``."""
    return experiment_dir(experiment) / "outputs" / model_short / "results"


def experiment_plots_dir(experiment: str, model_short: str) -> Path:
    return experiment_results_dir(experiment, model_short) / "plots"


def artifacts_root() -> Path:
    """Large intermediate files (cached states, probe weights); gitignored."""
    return project_root() / "artifacts"


def artifacts_dir(experiment: str, model_short: str) -> Path:
    return artifacts_root() / experiment / model_short


def evaluation_results_dir(run_date: str, model_short: str, benchmark_key: str,
                           intervention: str) -> Path:
    """``evaluation/results/{run_date}/{model_short}/{benchmark_key}/{intervention}/``.

    benchmark_key is ``pope_{split}`` for POPE, else the benchmark name.
    """
    return (project_root() / "evaluation" / "results" / run_date
            / model_short / benchmark_key / intervention)


# ── Data ─────────────────────────────────────────────────────────────────────

def data_root() -> Path:
    return project_root() / "data"


def coco_root() -> Path:
    return data_root() / "coco"


def coco_val2014_dir() -> Path:
    return coco_root() / "val2014"


def coco_annotations_dir() -> Path:
    return coco_root() / "annotations"


def amber_data_dir() -> Path:
    return data_root() / "amber"


def pope_data_dir() -> Path:
    return data_root() / "pope"


def chair_data_dir() -> Path:
    return data_root() / "chair"


# ── Third-party code ─────────────────────────────────────────────────────────

def llm_control_root() -> Path:
    """Local clone of https://github.com/chengemily1/llm-control (gitignored)."""
    return project_root() / "llm-control"
