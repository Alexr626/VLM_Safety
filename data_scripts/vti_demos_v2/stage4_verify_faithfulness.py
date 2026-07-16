#!/usr/bin/env python3
"""Stage 4 — independent MLLM verifies truthful=true / variants=false."""

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
from vti_demos_v2.mllm_client import MLLMClient, extract_json_object, parse_provider_spec  # noqa: E402
from vti_demos_v2.prompts import STAGE4_SYSTEM, stage4_statement_specs, stage4_user  # noqa: E402
from vti_demos_v2.validators import split_sentences  # noqa: E402
from src.paths import vti_demos_v2_dir  # noqa: E402


def _assert_independent(stage4_spec: str, stage2_spec: str, stage3_spec: str) -> None:
    p4, m4 = parse_provider_spec(stage4_spec)
    if p4 == "mock":
        return
    _, m2 = parse_provider_spec(stage2_spec)
    _, m3 = parse_provider_spec(stage3_spec)
    if m4 == m2 or m4 == m3:
        raise SystemExit(
            f"Stage-4 model must differ from stage-2 and stage-3 "
            f"(got stage4={m4}, stage2={m2}, stage3={m3})."
        )


def _mock_answers(rec: dict) -> dict:
    specs = stage4_statement_specs(rec)
    return {
        "answers": [
            {"idx": x["idx"], "verdict": x["expect"], "reason": "mock"}
            for x in specs
        ]
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default=config.STAGE4_PROVIDER)
    p.add_argument("--stage2-provider", default=config.STAGE2_PROVIDER,
                   help="used only to assert model independence")
    p.add_argument("--stage3-provider", default=config.STAGE3_PROVIDER,
                   help="used only to assert model independence")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--input", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _assert_independent(args.provider, args.stage2_provider, args.stage3_provider)

    v2 = vti_demos_v2_dir()
    inp = args.input or (v2 / "stage3_variants.jsonl")
    out_ok = v2 / "stage4_verdicts.jsonl"
    out_rej = v2 / "stage4_rejected.jsonl"
    calls = v2 / "calls" / "stage4.jsonl"

    done = load_ids(out_ok) | load_ids(out_rej)
    rows = read_jsonl(inp)
    if args.limit is not None:
        rows = rows[: args.limit]
    pending = [r for r in rows if r["id"] not in done]
    print(f"Stage 4: {len(pending)} pending / {len(rows)} input")

    client = MLLMClient(args.provider)
    tok_in = tok_out = 0
    n_ok = n_rej = 0

    for rec in pending:
        img_path = ensure_image(rec["id"])
        rec_for_prompt = {
            **rec,
            "sentences": split_sentences(rec["value"]),
        }
        user = stage4_user(rec_for_prompt)
        result = client.complete(
            system=STAGE4_SYSTEM,
            user_text=user,
            image=img_path,
            mock_json=_mock_answers(rec_for_prompt),
        )
        tok_in += result.input_tokens
        tok_out += result.output_tokens
        append_call_log(calls, {
            "id": rec["id"], "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "response": result.text[:4000],
        })
        try:
            parsed = extract_json_object(result.text)
            answers = parsed.get("answers") or []
        except Exception as e:
            append_jsonl(out_rej, {
                **rec, "reject_reason": "unparseable", "parse_error": str(e),
            })
            n_rej += 1
            continue

        by_idx = {int(a.get("idx", -1)): a for a in answers}
        failures = []
        for spec in stage4_statement_specs(rec_for_prompt):
            i, expected = spec["idx"], spec["expect"]
            a = by_idx.get(i)
            if a is None:
                failures.append({"idx": i, "dimension": spec["dimension"], "reason": "missing", "expected": expected})
                continue
            verdict = str(a.get("verdict", "")).lower()
            if verdict != expected:
                failures.append({
                    "idx": i, "verdict": verdict, "expected": expected,
                    "reason": a.get("reason"), "dimension": spec["dimension"],
                })

        out_base = {
            **rec,
            "stage4": parsed,
            "stage4_model": client.name,
        }
        if failures:
            append_jsonl(out_rej, {
                **out_base,
                "reject_reason": "faithfulness_mismatch",
                "failures": failures,
                "failing_dimensions": sorted({x["dimension"] for x in failures}),
            })
            n_rej += 1
            print(f"  reject {rec['id']}: {failures}")
            continue

        append_jsonl(out_ok, out_base)
        n_ok += 1
        print(f"  pass {rec['id']}")

    write_summary(v2 / "stage4_summary.json", {
        "provider": client.name,
        "n_pass_total": len(load_ids(out_ok)),
        "n_reject_total": len(load_ids(out_rej)),
        "n_pass_this_run": n_ok,
        "n_reject_this_run": n_rej,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "rejection_histogram": rejection_histogram(read_jsonl(out_rej)),
        "note": (
            "Verifier is a VLM filter, not ground truth; counting leans on "
            "COCO annotations. Manual gate 4 is authoritative."
        ),
    })
    print(f"Stage 4 done: +{n_ok} pass, +{n_rej} reject")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
