"""Central path helpers for diagnostic experiments and shared artifacts."""

from pathlib import Path
from typing import Optional

_PROJECT_ROOT: Optional[Path] = None


def project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return _PROJECT_ROOT


def diagnostic_root() -> Path:
    return project_root() / "diagnostic_experiments"


def diagnostic_experiment_dir(experiment: str) -> Path:
    return diagnostic_root() / experiment


def diagnostic_results_dir(experiment: str, model_short: str) -> Path:
    return diagnostic_experiment_dir(experiment) / model_short / "results"


def diagnostic_plots_dir(experiment: str, model_short: str) -> Path:
    return diagnostic_results_dir(experiment, model_short) / "plots"


def experiment_artifacts_root() -> Path:
    return project_root() / "experiment_artifacts"


def experiment_artifacts_dir(experiment: str, model_short: str) -> Path:
    return experiment_artifacts_root() / experiment / model_short


def evaluation_results_dir(model_short: str, benchmark: str,
                           intervention: str) -> Path:
    return (project_root() / "evaluation" / "results"
            / model_short / benchmark / intervention)


def data_root() -> Path:
    return project_root() / "data"


def coco_root() -> Path:
    return data_root() / "coco"


def coco_val2014_dir() -> Path:
    return coco_root() / "val2014"


def coco_train2014_dir() -> Path:
    return coco_root() / "train2014"


def vti_data_dir() -> Path:
    return data_root() / "vti"


def vti_demos_path() -> Path:
    return vti_data_dir() / "demos.jsonl"
