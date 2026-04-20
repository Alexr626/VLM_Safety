"""
Run refusal_judge_together.py on every model's holisafe_responses.json and merge summaries.
"""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
JUDGE = SCRIPT_DIR / "refusal_judge_together.py"
AGGREGATE_OUT = SCRIPT_DIR / "refusal_rates_all_models_together_judge.json"


def main():
    holisafe_paths = sorted(SCRIPT_DIR.glob("*/behavioral_ground_truth/outputs/results/holisafe_responses.json"))
    if not holisafe_paths:
        print("No holisafe_responses.json found under diagnostic_experiments/*/behavioral_ground_truth/outputs/results/")
        sys.exit(1)

    aggregate = {"models": {}, "judge_script": str(JUDGE)}

    for holisafe in holisafe_paths:
        model_dir = holisafe.parent.parent.parent.parent  # .../<model>/behavioral_ground_truth/...
        model_name = model_dir.name
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

    with open(AGGREGATE_OUT, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate summary written to {AGGREGATE_OUT}")


if __name__ == "__main__":
    main()
