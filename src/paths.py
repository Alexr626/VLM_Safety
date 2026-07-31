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


def coco_annotations_dir() -> Path:
    return coco_root() / "annotations"


def coco_train2014_dir() -> Path:
    return coco_root() / "train2014"


def vti_data_dir() -> Path:
    return data_root() / "vti"


def vti_demos_path() -> Path:
    return vti_data_dir() / "demos.jsonl"


def vti_demos_v2_dir() -> Path:
    """Intermediate artifacts for the demos_v2 generation pipeline."""
    return vti_data_dir() / "v2"


def vti_demos_v2_path() -> Path:
    """Final multi-dimension demos_v2 JSONL (one row per image)."""
    return vti_data_dir() / "demos_v2.jsonl"


def vti_demos_850_path() -> Path:
    """850-row demos pool: 555 demos_v2 rows + 295 top-up (separate file)."""
    return vti_data_dir() / "demos_850.jsonl"


def vti_demos_850_partition_path() -> Path:
    """Disjoint 50/100/200/500 partition over demos_850.jsonl (seed 42)."""
    return vti_data_dir() / "demos_850_partition_s42.json"


def amber_data_dir() -> Path:
    return data_root() / "amber"


def pope_data_dir() -> Path:
    return data_root() / "pope"


def benchmark_data_dir(benchmark: str) -> Path:
    """``data/{amber|pope|…}/`` for a benchmark key."""
    key = benchmark.strip().lower()
    if key == "amber":
        return amber_data_dir()
    if key == "pope":
        return pope_data_dir()
    return data_root() / key


def augmented_jsonl_path(benchmark: str, stem: str) -> Path:
    """Leading-clause augmented JSONL under the benchmark data dir.

    Example: ``augmented_jsonl_path("amber", "amber100")``
    → ``data/amber/augmented_amber100.jsonl``.
    """
    return benchmark_data_dir(benchmark) / f"augmented_{stem}.jsonl"


def perception_dump_dir(
    benchmark: str, model_short: str, run_tag: str
) -> Path:
    """Steered-capture dump root for a (benchmark, model, run_tag).

    Layout: ``data/{benchmark}/dumps/{model_short}/{run_tag}/``.
    """
    return (
        benchmark_data_dir(benchmark) / "dumps" / model_short / run_tag
    )


def infer_benchmark_from_run_tag(run_tag: str) -> str:
    """Map a perception dump ``run_tag`` to ``amber`` or ``pope``."""
    tag = run_tag.strip().lower()
    if tag.startswith("amber"):
        return "amber"
    if tag.startswith("pope"):
        return "pope"
    raise ValueError(
        f"Cannot infer benchmark from run_tag={run_tag!r}; "
        "expected amber* or pope* prefix"
    )
