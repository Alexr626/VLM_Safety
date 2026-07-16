#!/usr/bin/env python3
"""Stage 1 — MLLM verifies v2.1 option sets without choosing assignments."""

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


def _index_key_relation(opt: dict) -> tuple:
    return (
        str(opt.get("a", "")).lower(),
        str(opt.get("b", "")).lower(),
        str(opt.get("type", "")).lower(),
    )


def _merge_relation_verdicts(stage0_opts: list, model_opts: list) -> list:
    """Attach model verdicts onto stage-0 geometry (preserve a_side / gap / geometry)."""
    by_key = {_index_key_relation(x): x for x in (stage0_opts or [])}
    out = []
    for m in model_opts or []:
        if m.get("verdict") != "pass":
            continue
        base = by_key.get(_index_key_relation(m))
        if base is None:
            # Model invented / reordered pair — keep only if it still has a_side.
            if m.get("a_side") or m.get("type") == "support":
                out.append({**m})
            continue
        merged = {**base, **{k: v for k, v in m.items() if k in ("verdict", "reason")}}
        out.append(merged)
    return out


def _merge_counting_verdicts(stage0_opts: list, model_opts: list) -> list:
    by_key = {
        (str(x.get("category", "")).lower(), int(x.get("count", -1))): x
        for x in (stage0_opts or [])
    }
    out = []
    for m in model_opts or []:
        if m.get("verdict") != "pass":
            continue
        base = by_key.get((str(m.get("category", "")).lower(), int(m.get("count", -1))))
        if base is None:
            out.append(dict(m))
            continue
        merged = {
            **base,
            "verdict": "pass",
            "count_complete": bool(m.get("count_complete", False)),
            "reason": m.get("reason"),
        }
        out.append(merged)
    return out


def _merge_distractor_verdicts(stage0_opts: list, model_opts: list) -> list:
    """stage0 distractors may be list[str] or list[{category, score}]."""
    by_cat = {}
    for x in stage0_opts or []:
        if isinstance(x, str):
            by_cat[x.lower()] = {"category": x, "score": 0.0}
        else:
            by_cat[str(x.get("category", "")).lower()] = dict(x)
    out = []
    for m in model_opts or []:
        if m.get("verdict") != "absent":
            continue
        cat = str(m.get("category", "")).lower()
        base = by_cat.get(cat, {"category": m.get("category"), "score": 0.0})
        out.append({**base, "verdict": "absent", "reason": m.get("reason")})
    return out


def _mock_payload(rec: dict) -> dict:
    distractor = (rec.get("distractor_candidates") or [{"category": "fork"}])[0]
    if isinstance(distractor, str):
        distractor = {"category": distractor}
    count_cat = rec["counting_options"][0]["category"]
    # pick a present category different from counting if possible
    present = [c for c in rec.get("present_categories", {}) if c != count_cat]
    obj = present[0] if present else count_cat
    return {
        "counting_options": [{**x, "verdict": "pass", "count_complete": True,
                              "reason": "mock"} for x in rec["counting_options"]],
        "relation_options": [{**x, "verdict": "pass", "reason": "mock"}
                             for x in rec["relation_options"]],
        "distractors": [
            {**(x if isinstance(x, dict) else {"category": x}),
             "verdict": "absent", "reason": "mock"}
            for x in rec["distractor_candidates"]
        ],
        "attributes": [{"type": "color", "object": obj, "true_value": "white",
                        "false_value": "black", "confidence": "high"}],
    }


def _all_pass(parsed: dict) -> bool:
    return (any(x.get("verdict") == "pass" for x in parsed.get("counting_options", []))
            and any(x.get("verdict") == "pass" for x in parsed.get("relation_options", []))
            and any(x.get("verdict") == "absent" for x in parsed.get("distractors", []))
            and any(x.get("type") in config.ATTR_TYPES and x.get("object")
                    and x.get("true_value") and x.get("false_value")
                    for x in parsed.get("attributes", [])))


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
            reasons = [name for name, ok in (
                ("counting_check", any(x.get("verdict") == "pass" for x in parsed.get("counting_options", []))),
                ("relation_check", any(x.get("verdict") == "pass" for x in parsed.get("relation_options", []))),
                ("distractor", any(x.get("verdict") == "absent" for x in parsed.get("distractors", []))),
                ("attribute", bool(parsed.get("attributes")))) if not ok]
            append_jsonl(out_rej, {
                **rec, "stage1": parsed,
                "reject_reason": ";".join(reasons) or "checks_failed",
            })
            n_rej += 1
            continue

        out = {
            **rec,
            "verified_counting_options": _merge_counting_verdicts(
                rec.get("counting_options") or [], parsed.get("counting_options") or [],
            ),
            "verified_relation_options": _merge_relation_verdicts(
                rec.get("relation_options") or [], parsed.get("relation_options") or [],
            ),
            "verified_distractor_options": _merge_distractor_verdicts(
                rec.get("distractor_candidates") or [], parsed.get("distractors") or [],
            ),
            "verified_attribute_options": [x for x in parsed.get("attributes") or []
                                           if x.get("type") in config.ATTR_TYPES and x.get("object")
                                           and x.get("true_value") and x.get("false_value")],
            "stage1": parsed,
            "stage1_model": client.name,
        }
        if not (
            out["verified_counting_options"]
            and out["verified_relation_options"]
            and out["verified_distractor_options"]
            and out["verified_attribute_options"]
        ):
            append_jsonl(out_rej, {
                **rec, "stage1": parsed,
                "reject_reason": "verified_options_empty_after_merge",
            })
            n_rej += 1
            continue
        append_jsonl(out_ok, out)
        n_ok += 1
        print(f"  pass {rec['id']} ({len(out['verified_relation_options'])} relations)")

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
