"""
Generate harmless counterparts for the CatQA (CategoricalHarmfulQA) dataset.

Uses the prompt from SafeSteer (Ghosh et al., EMNLP 2025 / Bhattacharjee et al., 2024)
to create minimal-edit harmless versions of each harmful question, suitable for
contrastive direction extraction in activation space analysis.

Usage:
    # Using Claude (default):
    export ANTHROPIC_API_KEY="sk-ant-..."
    python generate_catqa_harmless_pairs.py --provider anthropic

    # Using OpenAI:
    export OPENAI_API_KEY="sk-..."
    python generate_catqa_harmless_pairs.py --provider openai

    # Custom options:
    python generate_catqa_harmless_pairs.py \
        --provider anthropic \
        --model claude-sonnet-4-20250514 \
        --output catqa_contrastive_pairs.json \
        --batch-size 5 \
        --max-retries 3

Requirements:
    pip install datasets anthropic openai tqdm
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prompt template (from SafeSteer / Bhattacharjee et al.)
# ---------------------------------------------------------------------------
HARMLESS_REWRITE_PROMPT = (
    "The following question is a harmful question. Make minimal changes to it "
    "so that the content of the questions stays the same as much as possible, "
    "but it is now harmless.\n"
    "---\n"
    "Harmful question: {question}\n"
    "---\n"
    "Output only the modified input."
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_catqa(language: str = "en") -> list[dict]:
    """Load CatQA from HuggingFace and return as list of dicts."""
    from datasets import load_dataset

    print(f"Loading CategoricalHarmfulQA (split='{language}') from HuggingFace...")
    ds = load_dataset("declare-lab/CategoricalHarmfulQA", split=language)

    records = []
    for row in ds:
        records.append({
            "category": row["Category"],
            "subcategory": row["Subcategory"],
            "question_harmful": row["Question"],
        })

    print(f"  Loaded {len(records)} harmful questions across "
          f"{len(set(r['category'] for r in records))} categories.")
    return records


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------
def generate_harmless_anthropic(
    questions: list[str],
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 3,
    delay: float = 0.5,
) -> list[str]:
    """Generate harmless counterparts using the Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    results = []

    for q in tqdm(questions, desc="Generating (Anthropic)", unit="q"):
        prompt = HARMLESS_REWRITE_PROMPT.format(question=q)

        for attempt in range(1, max_retries + 1):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                results.append(text)
                break
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                print(f"\n  Rate limited. Waiting {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
            except anthropic.APIError as e:
                if attempt == max_retries:
                    print(f"\n  Failed after {max_retries} attempts for: {q[:60]}...")
                    results.append("")
                else:
                    time.sleep(1)

        time.sleep(delay)  # polite rate limiting

    return results


def generate_harmless_openai(
    questions: list[str],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    delay: float = 0.5,
) -> list[str]:
    """Generate harmless counterparts using the OpenAI API."""
    from openai import OpenAI, RateLimitError, APIError

    client = OpenAI()  # reads OPENAI_API_KEY from env
    results = []

    for q in tqdm(questions, desc="Generating (OpenAI)", unit="q"):
        prompt = HARMLESS_REWRITE_PROMPT.format(question=q)

        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content.strip()
                results.append(text)
                break
            except RateLimitError:
                wait = 2 ** attempt
                print(f"\n  Rate limited. Waiting {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
            except APIError as e:
                if attempt == max_retries:
                    print(f"\n  Failed after {max_retries} attempts for: {q[:60]}...")
                    results.append("")
                else:
                    time.sleep(1)

        time.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# Checkpointing (resume from where we left off)
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path) -> list[dict]:
    """Load partial results from a checkpoint file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_checkpoint(records: list[dict], path: Path):
    """Save current results as a checkpoint."""
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate harmless counterparts for CatQA dataset."
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai"], default="anthropic",
        help="LLM API provider (default: anthropic)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name. Defaults: claude-sonnet-4-20250514 (anthropic) / gpt-4o-mini (openai)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: data/catqa-contrastive/catqa_contrastive_pairs.json)"
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="CatQA language split: en, zh, vi (default: en)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Save checkpoint every N generations (default: 10)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per API call on failure (default: 3)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay in seconds between API calls (default: 0.5)"
    )
    args = parser.parse_args()

    # Resolve model defaults
    if args.model is None:
        args.model = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o-mini",
        }[args.provider]

    # Resolve output path default
    if args.output is None:
        project_root = Path(__file__).resolve().parent.parent
        args.output = str(project_root / "data" / "catqa-contrastive" / "catqa_contrastive_pairs.json")

    # Check API key
    key_env = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key_env):
        print(f"Error: {key_env} environment variable not set.")
        sys.exit(1)

    # Load dataset
    records = load_catqa(args.language)

    # Load checkpoint (resume support)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    completed = load_checkpoint(checkpoint_path)
    start_idx = len(completed)

    if start_idx > 0:
        print(f"Resuming from checkpoint: {start_idx}/{len(records)} already completed.")

    # Select generator
    generate_fn = (
        generate_harmless_anthropic if args.provider == "anthropic"
        else generate_harmless_openai
    )

    # Process in batches with checkpointing
    remaining = records[start_idx:]
    for batch_start in range(0, len(remaining), args.batch_size):
        batch = remaining[batch_start : batch_start + args.batch_size]
        questions = [r["question_harmful"] for r in batch]

        harmless_versions = generate_fn(
            questions,
            model=args.model,
            max_retries=args.max_retries,
            delay=args.delay,
        )

        for record, harmless_q in zip(batch, harmless_versions):
            record["question_harmless"] = harmless_q
            completed.append(record)

        save_checkpoint(completed, checkpoint_path)

    # Final output
    save_checkpoint(completed, output_path)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print summary
    n_success = sum(1 for r in completed if r.get("question_harmless"))
    n_failed = len(completed) - n_success
    print(f"\nDone! Saved {len(completed)} contrastive pairs to: {output_path}")
    print(f"  Successful: {n_success}")
    if n_failed:
        print(f"  Failed (empty): {n_failed}")

    # Show a few examples
    print("\n--- Sample pairs ---")
    for r in completed[:3]:
        print(f"\n  Category:    {r['category']} / {r['subcategory']}")
        print(f"  Harmful:     {r['question_harmful'][:100]}...")
        print(f"  Harmless:    {r.get('question_harmless', 'N/A')[:100]}...")

    print(f"\nProvider: {args.provider} | Model: {args.model}")
    print(f"Estimated cost: ~{len(completed) * 0.001:.2f}–{len(completed) * 0.005:.2f} USD "
          f"(varies by model and response length)")


if __name__ == "__main__":
    main()
