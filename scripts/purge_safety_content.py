#!/usr/bin/env python3
"""One-shot purge of safety-specific content for hallucination repurpose."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    "diagnostic_experiments/experiment_scripts",
    "diagnostic_experiments/plotting_scripts",
    "diagnostic_experiments/run_scripts",
    "diagnostic_experiments/llava-1.5-7b-hf",
    "diagnostic_experiments/sharegpt4v-7b",
    "diagnostic_experiments/qwen2-vl-7b",
    "diagnostic_experiments/qwen2-vl-7b-instruct",
    "diagnostic_experiments/qwen2.5-vl-7b-instruct",
    "diagnostic_experiments/internvl2-8b",
    "diagnostic_experiments/internvl2.5-8b-mpo",
    "experiment_artifacts",
    "helper_scripts",
    "plans_and_project_descriptions",
    "data/holisafe-bench",
    "data/mssbench",
    "data/mm-safetybench",
    "data/figstep",
    "data/catqa-contrastive",
    "data/siuo",
    "data/llava-instruct-ref",
    "data/mm-safetybench-ref",
    "data/captions/claude_generated",
    "evaluation/scripts",
    "evaluation/results",
]

FILES = [
    "evaluation/benchmarks/mm_safetybench.py",
    "evaluation/benchmarks/figstep.py",
    "evaluation/benchmarks/mssbench.py",
    "evaluation/interventions/comp_safety_shift.py",
    "evaluation/interventions/adashield_s.py",
    "evaluation/interventions/vanilla.py",
    "evaluation/classifiers/keyword.py",
    "data_scripts/extract_ref_activations.py",
    "data_scripts/generate_catqa_harmless_pairs.py",
    "data_scripts/generate_siuo_sss_pairs.py",
    "data_scripts/generate_cohesive_text.py",
    "data_scripts/extract_ct.py",
    "data_scripts/run_fill_artifacts_5080.sh",
    "data_scripts/run_fill_artifacts_5090.sh",
    "data/captions/holisafe.json",
    "data/captions/mssbench.json",
    "data/captions/mm_safetybench.json",
    "data/captions/figstep.json",
    "data/safeunsafeds.jsonl",
]

MKDIRS = [
    "experiment_artifacts",
    "diagnostic_experiments/modality_shift/run_scripts",
    "diagnostic_experiments/causal_mediation/run_scripts",
    "data/captions",
    "data/coco",
    "evaluation/results",
]

KEEP_EMPTY = [
    "experiment_artifacts/.gitkeep",
    "data/captions/.gitkeep",
    "data/coco/.gitkeep",
    "evaluation/results/.gitkeep",
]


def _rmtree_force(path: Path):
    def _onerror(func, p, exc_info):
        import os
        import stat
        if not os.access(p, os.W_OK):
            os.chmod(p, stat.S_IWUSR | stat.S_IREAD)
            func(p)
        else:
            raise exc_info[1]
    shutil.rmtree(path, onerror=_onerror)


def main():
    for rel in DIRS:
        p = ROOT / rel
        if p.exists():
            _rmtree_force(p)
            print(f"removed dir  {rel}")
    for rel in FILES:
        p = ROOT / rel
        if p.exists():
            p.unlink()
            print(f"removed file {rel}")
    for rel in MKDIRS:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    for rel in KEEP_EMPTY:
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
    print("purge complete")


if __name__ == "__main__":
    main()
