#!/usr/bin/env python3
"""Stage 1 — MLLM verifies anchors / selects distractor + attribute."""

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
from vti_demos_v2.prompts import STAGE1_SYSTEM, stage1_user  # noqa: E402
from src.paths import vti_demos_v2_dir  # noqa: E402


def _mock_payload(rec: dict) -> dict:
    distractor = (rec.get("distractor_candidates") or ["fork"])[0]
    count_cat = rec["counting_anchor"]["category"]
    # pick a present category different from counting if possible
    present = [c for c in rec.get("present_categories", {}) if c != count_cat]
    obj = present[0] if present else count_cat
    return {
        "counting_check": {"verdict": "pass", "reason": "mock: distinguishable"},
        "relation_check": {"verdict": "pass", "reason": "mock: A left of B"},
        "distractor": {
            "verdict": "pass",
            "choice": distractor,
            "reason": "mock: absent and plausible",
        },
        "attribute": {
            "verdict": "pass",
            "object": obj,
            "true_value": "white",
            "false_value": "black",
            "overlaps_counting_category": obj == count_cat,
            "reason": "mock: color attribute",
        },
    }


def _all_pass(parsed: dict) -> bool:
    return (
        parsed.get("counting_check", {}).get("verdict") == "pass"
        and parsed.get("relation_check", {}).get("verdict") == "pass"
        and parsed.get("distractor", {}).get("verdict") == "pass"
        and parsed.get("attribute", {}).get("verdict") == "pass"
        and bool(parsed.get("distractor", {}).get("choice"))
        and bool(parsed.get("attribute", {}).get("object"))
        and bool(parsed.get("attribute", {}).get("true_value"))
        and bool(parsed.get("attribute", {}).get("false_value"))
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default=config.STAGE1_PROVIDER)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--input", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    inp = args.input or (v2 / "stage0_candidates.jsonl")
    out_ok = v2 / "stage1_verified.jsonl"
    out_rej = v2 / "stage1_rejected.jsonl"
    calls = v2 / "calls" / "stage1.jsonl"

    done = load_ids(out_ok) | load_ids(out_rej)
    rows = read_jsonl(inp)
    if args.limit is not None:
        rows = rows[: args.limit]
    pending = [r for r in rows if r["id"] not in done]
    print(f"Stage 1: {len(pending)} pending / {len(rows)} input (skip {len(done)})")

    client = MLLMClient(args.provider)
    tok_in = tok_out = 0
    n_ok = n_rej = 0

    for rec in pending:
        img_path = ensure_image(rec["id"])
        user = stage1_user(rec)
        parsed = None
        last_err = None
        for attempt in range(2):
            retry_user = user if attempt == 0 else (
                user + f"\n\nPrevious reply failed to parse: {last_err}. "
                "Return ONLY valid JSON matching the schema."
            )
            result = client.complete(
                system=STAGE1_SYSTEM,
                user_text=retry_user,
                image=img_path,
                mock_json=_mock_payload(rec),
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
                break
            except Exception as e:
                last_err = str(e)
                parsed = None

        if parsed is None:
            append_jsonl(out_rej, {**rec, "reject_reason": "unparseable",
                                   "parse_error": last_err})
            n_rej += 1
            continue

        if not _all_pass(parsed):
            reasons = []
            for k in ("counting_check", "relation_check", "distractor", "attribute"):
                if parsed.get(k, {}).get("verdict") != "pass":
                    reasons.append(f"{k}:{parsed.get(k, {}).get('reason', 'fail')}")
            append_jsonl(out_rej, {
                **rec, "stage1": parsed,
                "reject_reason": ";".join(reasons) or "checks_failed",
            })
            n_rej += 1
            continue

        out = {
            **rec,
            "distractor": parsed["distractor"]["choice"],
            "attribute": {
                "object": parsed["attribute"]["object"],
                "true_value": parsed["attribute"]["true_value"],
                "false_value": parsed["attribute"]["false_value"],
                "overlaps_counting_category": bool(
                    parsed["attribute"].get("overlaps_counting_category")
                ),
            },
            "stage1": parsed,
            "stage1_model": client.name,
        }
        append_jsonl(out_ok, out)
        n_ok += 1
        print(f"  pass {rec['id']} distractor={out['distractor']} "
              f"attr={out['attribute']['object']}:{out['attribute']['true_value']}")

    write_summary(v2 / "stage1_summary.json", {
        "provider": client.name,
        "n_pass": n_ok + len(load_ids(out_ok)) - n_ok,  # total on disk
        "n_pass_this_run": n_ok,
        "n_reject_this_run": n_rej,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "rejection_histogram": rejection_histogram(read_jsonl(out_rej)),
    })
    # fix n_pass to total
    write_summary(v2 / "stage1_summary.json", {
        "provider": client.name,
        "n_pass_total": len(load_ids(out_ok)),
        "n_reject_total": len(load_ids(out_rej)),
        "n_pass_this_run": n_ok,
        "n_reject_this_run": n_rej,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "rejection_histogram": rejection_histogram(read_jsonl(out_rej)),
    })
    print(f"Stage 1 done: +{n_ok} pass, +{n_rej} reject")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
