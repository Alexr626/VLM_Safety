"""
Run refusal_judge_together.py on every model's holisafe_responses.json and merge summaries.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
JUDGE = SCRIPT_DIR / "refusal_judge_together.py"


def main():
    parser = argparse.ArgumentParser(description="Run HoliSafe LLM judge on all diagnostic models.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write per-model refusal_results.json and aggregate JSON "
        "(default: next to each holisafe_responses.json, aggregate under diagnostic_experiments/).",
    )
    args = parser.parse_args()

    holisafe_paths = sorted(SCRIPT_DIR.glob("*/behavioral_ground_truth/outputs/results/holisafe_responses.json"))
    if not holisafe_paths:
        print("No holisafe_responses.json found under diagnostic_experiments/*/behavioral_ground_truth/outputs/results/")
        sys.exit(1)

    out_root = args.out_dir.resolve() if args.out_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    aggregate = {"models": {}, "judge_script": str(JUDGE), "output_root": str(out_root) if out_root else None}

    for holisafe in holisafe_paths:
        model_dir = holisafe.parent.parent.parent.parent  # .../<model>/behavioral_ground_truth/...
        model_name = model_dir.name
        if out_root is not None:
            model_out_dir = out_root / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = model_out_dir / "refusal_results.json"
        else:
            out_path = holisafe.parent / "refusal_results.json"

        print(f"\n========== {model_name} ==========")
        subprocess.run(
            [sys.executable, str(JUDGE), "--input", str(holisafe), "--output", str(out_path)],
            check=True,
            cwd=str(SCRIPT_DIR),
        )

        with open(out_path) as f:
            per = json.load(f)
        aggregate["models"][model_name] = {
            "holisafe_responses": str(holisafe),
            "refusal_results": str(out_path),
            "overall": per.get("overall"),
            "by_category": per.get("by_category"),
            "num_raw_judgments": len(per.get("raw", [])),
        }

    if out_root is not None:
        aggregate_out = out_root / "refusal_rates_all_models_together_judge.json"
    else:
        aggregate_out = SCRIPT_DIR / "refusal_rates_all_models_together_judge.json"

    with open(aggregate_out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate summary written to {aggregate_out}")


if __name__ == "__main__":
    main()
