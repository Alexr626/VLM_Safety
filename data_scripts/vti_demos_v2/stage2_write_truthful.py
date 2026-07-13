#!/usr/bin/env python3
"""Stage 2 — MLLM writes the rigid 4-sentence truthful caption + spans."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

from vti_demos_v2 import config  # noqa: E402
from vti_demos_v2.images import ensure_image  # noqa: E402
from vti_demos_v2.io_utils import (  # noqa: E402
    append_call_log,
    append_jsonl,
    load_ids,
    read_jsonl,
    rejection_histogram,
    write_summary,
)
from vti_demos_v2.mllm_client import MLLMClient, extract_json_object  # noqa: E402
from vti_demos_v2.number_words import number_to_word  # noqa: E402
from vti_demos_v2.prompts import STAGE2_SYSTEM, stage2_user  # noqa: E402
from vti_demos_v2.validators import structural_check_truthful  # noqa: E402
from src.paths import vti_demos_v2_dir  # noqa: E402


def _mock_caption(rec: dict) -> dict:
    ca = rec["counting_anchor"]
    ra = rec["relation_anchor"]
    attr = rec["attribute"]
    n_word = number_to_word(int(ca["count"]))
    a, b = ra["a"], ra["b"]
    rel = ra["relation"]
    obj, val = attr["object"], attr["true_value"]
    # Build a rigid 4-sentence caption with unique spans
    hint = f"including {ca['category']}s, {a}s, and {b}s"
    # avoid plural weirdness for person etc. — keep simple for mock
    hint = f"including {ca['category']}, {a}, and {b}"
    s1 = f"The image shows a scene with objects, {hint}."
    s2 = f"The {obj} look {val}."
    # ensure attribute value appears once — if val in s1 somehow, simplify
    s3 = f"There are at least {n_word} {ca['category']}."
    s4 = f"The {a} is to the {rel} of the {b}."
    caption = f"{s1} {s2} {s3} {s4}"
    return {
        "caption": caption,
        "spans": {
            "existence_insertion_hint": hint,
            "attribute": val,
            "counting": n_word,
            "relation": rel,
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default=config.STAGE2_PROVIDER)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--input", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    inp = args.input or (v2 / "stage1_verified.jsonl")
    out_ok = v2 / "stage2_captions.jsonl"
    out_rej = v2 / "stage2_rejected.jsonl"
    calls = v2 / "calls" / "stage2.jsonl"

    done = load_ids(out_ok) | load_ids(out_rej)
    rows = read_jsonl(inp)
    if args.limit is not None:
        rows = rows[: args.limit]
    pending = [r for r in rows if r["id"] not in done]
    print(f"Stage 2: {len(pending)} pending / {len(rows)} input")

    client = MLLMClient(args.provider)
    tok_in = tok_out = 0
    n_ok = n_rej = 0

    for rec in pending:
        img_path = ensure_image(rec["id"])
        ca = rec["counting_anchor"]
        n_word = number_to_word(int(ca["count"]))
        user = stage2_user(rec)
        accepted = None
        last_errs = None
        for attempt in range(2):
            retry_user = user if attempt == 0 else (
                user + "\n\nPrevious caption failed validation:\n- "
                + "\n- ".join(last_errs or [])
                + "\nFix these issues and return JSON only."
            )
            result = client.complete(
                system=STAGE2_SYSTEM,
                user_text=retry_user,
                image=img_path,
                mock_json=_mock_caption(rec),
            )
            tok_in += result.input_tokens
            tok_out += result.output_tokens
            append_call_log(calls, {
                "id": rec["id"], "model": result.model, "attempt": attempt,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "response": result.text[:4000],
            })
            try:
                parsed = extract_json_object(result.text)
            except Exception as e:
                last_errs = [f"unparseable: {e}"]
                continue
            caption = (parsed.get("caption") or "").strip()
            spans = parsed.get("spans") or {}
            errs = structural_check_truthful(
                caption,
                count_word=n_word,
                category=ca["category"],
                relation_word=rec["relation_anchor"]["relation"],
                attribute_value=rec["attribute"]["true_value"],
                distractor=rec["distractor"],
                spans=spans,
            )
            if errs:
                last_errs = errs
                continue
            accepted = {"caption": caption, "spans": spans}
            break

        if accepted is None:
            append_jsonl(out_rej, {
                **rec,
                "reject_reason": "structural_validation_failed",
                "errors": last_errs,
            })
            n_rej += 1
            continue

        out = {
            **rec,
            "value": accepted["caption"],
            "spans": accepted["spans"],
            "stage2_model": client.name,
        }
        append_jsonl(out_ok, out)
        n_ok += 1
        print(f"  pass {rec['id']}")

    write_summary(v2 / "stage2_summary.json", {
        "provider": client.name,
        "n_pass_total": len(load_ids(out_ok)),
        "n_reject_total": len(load_ids(out_rej)),
        "n_pass_this_run": n_ok,
        "n_reject_this_run": n_rej,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "rejection_histogram": rejection_histogram(read_jsonl(out_rej)),
    })
    print(f"Stage 2 done: +{n_ok} pass, +{n_rej} reject")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
