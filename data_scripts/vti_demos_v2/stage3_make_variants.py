#!/usr/bin/env python3
"""Stage 3 — build four hallucinated variants.

Counting / relation / attribute: pure string edits.
Existence: deterministic insert into ``existence_insertion_hint``, then a
text-only Haiku grammar polish; full LLM insert only if that fails validation.
"""

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
from vti_demos_v2.io_utils import (  # noqa: E402
    append_call_log,
    append_jsonl,
    load_ids,
    read_jsonl,
    rejection_histogram,
    write_summary,
)
from vti_demos_v2.mllm_client import MLLMClient, extract_json_object  # noqa: E402
from vti_demos_v2.number_words import false_count, number_to_word  # noqa: E402
from vti_demos_v2.prompts import (  # noqa: E402
    STAGE3_EXISTENCE_SYSTEM,
    STAGE3_GRAMMAR_SYSTEM,
    stage3_existence_user,
    stage3_grammar_user,
)
from vti_demos_v2.validators import (  # noqa: E402
    deterministic_existence_insert,
    validate_minimal_pair,
    validate_record_variants,
)
from src.paths import vti_demos_v2_dir  # noqa: E402


def _replace_once(text: str, old: str, new: str) -> str:
    idx = text.lower().find(old.lower())
    if idx < 0:
        raise ValueError(f"span {old!r} not found")
    return text[:idx] + new + text[idx + len(old):]


def _mock_existence(caption: str, distractor: str, hint: str) -> dict:
    try:
        return {"caption": deterministic_existence_insert(caption, hint, distractor)}
    except ValueError:
        return {"caption": caption}


def _grammar_polish(
    truthful: str,
    draft: str,
    distractor: str,
    *,
    client: MLLMClient,
    calls: Path,
    rec_id: str,
) -> tuple[str | None, list[str], int, int]:
    """Text-only grammar repair. Returns (caption_or_None, errs, tok_in, tok_out)."""
    result = client.complete(
        system=STAGE3_GRAMMAR_SYSTEM,
        user_text=stage3_grammar_user(truthful, draft, distractor),
        image=None,
        mock_json={"caption": draft},
    )
    append_call_log(calls, {
        "id": rec_id,
        "model": result.model,
        "attempt": 0,
        "phase": "grammar_polish",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "response": result.text[:4000],
    })
    try:
        parsed = extract_json_object(result.text)
        cand = (parsed.get("caption") or "").strip()
    except Exception as e:
        return None, [f"grammar_unparseable: {e}"], result.input_tokens, result.output_tokens
    errs = validate_minimal_pair(
        truthful, cand, "existence", distractor=distractor,
    )
    if errs:
        return None, [f"grammar: {e}" for e in errs], result.input_tokens, result.output_tokens
    return cand, [], result.input_tokens, result.output_tokens


def _try_existence(
    truthful: str,
    hint: str,
    distractor: str,
    *,
    client: MLLMClient,
    calls: Path,
    rec_id: str,
) -> tuple[str | None, str | None, list[str] | None, int, int]:
    """Return (caption, source, errors, tok_in, tok_out)."""
    tok_in = tok_out = 0
    last_errs: list[str] = []

    # 1) Deterministic insert, then text-only grammar polish (Haiku by default).
    draft = None
    try:
        draft = deterministic_existence_insert(truthful, hint, distractor)
    except ValueError as e:
        last_errs = [f"deterministic: {e}"]

    if draft is not None:
        polished, polish_errs, ti, to = _grammar_polish(
            truthful, draft, distractor,
            client=client, calls=calls, rec_id=rec_id,
        )
        tok_in += ti
        tok_out += to
        if polished is not None:
            return polished, f"deterministic+grammar:{client.name}", None, tok_in, tok_out
        last_errs = polish_errs or [
            f"deterministic_raw: {e}"
            for e in validate_minimal_pair(
                truthful, draft, "existence", distractor=distractor,
            )
        ] or ["grammar_polish_failed"]

    # 2) Full LLM existence insert fallback.
    for attempt in range(2):
        user = stage3_existence_user(truthful, distractor, hint)
        if attempt and last_errs:
            user += "\n\nPrevious attempt failed validation:\n- " + "\n- ".join(last_errs)
        result = client.complete(
            system=STAGE3_EXISTENCE_SYSTEM,
            user_text=user,
            image=None,
            mock_json=_mock_existence(truthful, distractor, hint),
        )
        tok_in += result.input_tokens
        tok_out += result.output_tokens
        append_call_log(calls, {
            "id": rec_id, "model": result.model, "attempt": attempt,
            "phase": "existence_llm",
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "response": result.text[:4000],
            "fallback_after": "deterministic+grammar",
        })
        try:
            parsed = extract_json_object(result.text)
            cand = (parsed.get("caption") or "").strip()
        except Exception as e:
            last_errs = [f"unparseable: {e}"]
            continue
        errs = validate_minimal_pair(
            truthful, cand, "existence", distractor=distractor,
        )
        if errs:
            last_errs = errs
            continue
        return cand, f"llm:{result.model}", None, tok_in, tok_out

    return None, None, last_errs, tok_in, tok_out

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default=config.STAGE3_PROVIDER)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--input", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    inp = args.input or (v2 / "stage2_captions.jsonl")
    out_ok = v2 / "stage3_variants.jsonl"
    out_rej = v2 / "stage3_rejected.jsonl"
    calls = v2 / "calls" / "stage3.jsonl"

    done = load_ids(out_ok) | load_ids(out_rej)
    rows = read_jsonl(inp)
    if args.limit is not None:
        rows = rows[: args.limit]
    pending = [r for r in rows if r["id"] not in done]
    print(f"Stage 3: {len(pending)} pending / {len(rows)} input")

    client = MLLMClient(args.provider)
    tok_in = tok_out = 0
    n_ok = n_rej = 0
    n_det = n_llm = 0

    for rec in pending:
        truthful = rec["value"]
        spans = rec["spans"]
        ca = rec["counting_anchor"]
        true_n = int(ca["count"])
        true_word = spans["counting"]
        false_word = number_to_word(false_count(true_n))
        true_rel = spans["relation"]
        false_rel = "right" if true_rel == "left" else "left"
        true_attr = spans["attribute"]
        false_attr = rec["attribute"]["false_value"]
        distractor = rec["distractor"]
        hint = spans["existence_insertion_hint"]

        try:
            h_counting = _replace_once(truthful, true_word, false_word)
            h_relation = _replace_once(truthful, true_rel, false_rel)
            h_attribute = _replace_once(truthful, true_attr, false_attr)
        except ValueError as e:
            append_jsonl(out_rej, {
                **rec, "reject_reason": f"deterministic_edit_failed:{e}",
            })
            n_rej += 1
            print(f"  REJECT {rec['id']} deterministic edit bug: {e}", file=sys.stderr)
            continue

        det_errs = []
        for dim, variant, kw in (
            ("counting", h_counting, dict(
                true_count_word=true_word, false_count_word=false_word)),
            ("relation", h_relation, dict(true_relation=true_rel)),
            ("attribute", h_attribute, dict(
                true_attr=true_attr, false_attr=false_attr)),
        ):
            det_errs.extend(
                f"{dim}: {e}"
                for e in validate_minimal_pair(truthful, variant, dim, **kw)
            )
        if det_errs:
            append_jsonl(out_rej, {
                **rec, "reject_reason": "deterministic_diff_failed",
                "errors": det_errs,
            })
            n_rej += 1
            print(f"  REJECT {rec['id']} span bug: {det_errs}", file=sys.stderr)
            continue

        h_existence, src, last_errs, ti, to = _try_existence(
            truthful, hint, distractor,
            client=client, calls=calls, rec_id=rec["id"],
        )
        tok_in += ti
        tok_out += to
        if h_existence is None:
            append_jsonl(out_rej, {
                **rec, "reject_reason": "existence_edit_failed",
                "errors": last_errs,
            })
            n_rej += 1
            continue
        if src and src.startswith("deterministic"):
            n_det += 1
        else:
            n_llm += 1

        h_values = {
            "existence": h_existence,
            "attribute": h_attribute,
            "counting": h_counting,
            "relation": h_relation,
        }
        all_errs = validate_record_variants(
            truthful, h_values,
            distractor=distractor,
            true_attr=true_attr,
            false_attr=false_attr,
            true_count_word=true_word,
            false_count_word=false_word,
            true_relation=true_rel,
        )
        if all_errs:
            append_jsonl(out_rej, {
                **rec, "reject_reason": "variant_validation_failed",
                "errors": all_errs, "h_values": h_values,
            })
            n_rej += 1
            continue

        out = {
            **rec,
            "h_values": h_values,
            "existence_source": src,
            "anchors": {
                "existence": {"distractor": distractor},
                "attribute": {
                    "object": rec["attribute"]["object"],
                    "true_value": true_attr,
                    "false_value": false_attr,
                },
                "counting": {
                    "category": ca["category"],
                    "annotated_count": true_n,
                    "true_word": true_word,
                    "false_word": false_word,
                },
                "relation": {
                    "a": rec["relation_anchor"]["a"],
                    "b": rec["relation_anchor"]["b"],
                    "true": true_rel,
                    "false": false_rel,
                },
            },
            "stage3_model": src or client.name,
            "diff_report": {
                dim: validate_minimal_pair(
                    truthful, h_values[dim], dim,
                    distractor=distractor,
                    true_attr=true_attr, false_attr=false_attr,
                    true_count_word=true_word, false_count_word=false_word,
                    true_relation=true_rel,
                )
                for dim in h_values
            },
        }
        append_jsonl(out_ok, out)
        n_ok += 1
        print(f"  pass {rec['id']} (existence={src})")

    write_summary(v2 / "stage3_summary.json", {
        "provider": client.name,
        "n_pass_total": len(load_ids(out_ok)),
        "n_reject_total": len(load_ids(out_rej)),
        "n_pass_this_run": n_ok,
        "n_reject_this_run": n_rej,
        "n_existence_deterministic_grammar": n_det,
        "n_existence_llm": n_llm,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "rejection_histogram": rejection_histogram(read_jsonl(out_rej)),
    })
    print(
        f"Stage 3 done: +{n_ok} pass, +{n_rej} reject "
        f"(existence deterministic+grammar={n_det}, llm={n_llm})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
