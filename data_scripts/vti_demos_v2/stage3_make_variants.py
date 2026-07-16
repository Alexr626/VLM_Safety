#!/usr/bin/env python3
"""Stage 3 — build four hallucinated variants.

Counting / relation / attribute: pure string edits.
Existence: deterministic insert into ``existence_insertion_hint``, then a
text-only Haiku grammar polish; full LLM insert only if that fails validation.
"""

from __future__ import annotations

import argparse
import re
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
from vti_demos_v2.number_words import false_count, false_count_at_most, number_to_word  # noqa: E402
from vti_demos_v2.prompts import (  # noqa: E402
    STAGE3_EXISTENCE_SYSTEM,
    STAGE3_GRAMMAR_SYSTEM,
    stage3_existence_user,
    stage3_grammar_user,
)
from vti_demos_v2.validators import (  # noqa: E402
    category_plural_form,
    category_singular_form,
    deterministic_existence_insert,
    replace_span_in_sentence,
    span_sentence_idx,
    span_text,
    split_sentences,
    validate_combined,
    validate_minimal_pair,
    validate_record_variants,
)
from src.paths import vti_demos_v2_dir  # noqa: E402


def _distractor_category(rec: dict) -> str:
    d = rec.get("distractor")
    if isinstance(d, dict):
        return str(d.get("category") or d.get("choice") or "")
    return str(d or "")


def _false_relation_phrase(true_phrase: str) -> str:
    t = true_phrase.lower().strip()
    flips = {
        "left": "right",
        "right": "left",
        "above": "below",
        "below": "above",
        "on top of": "underneath",
        "underneath": "on top of",
        "right next to": "far away from",
        "far away from": "right next to",
    }
    if t not in flips:
        raise ValueError(f"unknown relation phrase {true_phrase!r}")
    return flips[t]


def _counting_edit(
    truthful: str,
    spans: dict,
    *,
    mode: str,
    category: str,
    true_word: str,
    false_word: str,
    false_n: int,
) -> tuple[str, bool]:
    """Apply counting edit; return (variant, used_noun_morph)."""
    sent_idx = span_sentence_idx(spans["counting"])
    morph = mode == "at_most" and false_n == 1
    if morph:
        s3 = split_sentences(truthful)[sent_idx]
        pl = category_plural_form(category)
        sg = category_singular_form(category)
        for form in (pl, sg, category):
            old = f"{true_word} {form}"
            if re.search(rf"\b{re.escape(old)}\b", s3, re.I):
                new = f"{false_word} {sg}"
                return replace_span_in_sentence(truthful, sent_idx, old, new), True
    return replace_span_in_sentence(truthful, sent_idx, true_word, false_word), False


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
        ca = rec["counting"]
        relation = rec["relation"]
        true_n = int(ca["count"])
        true_word = span_text(spans["counting"])
        false_n = (
            false_count_at_most(true_n) if ca["mode"] == "at_most" else false_count(true_n)
        )
        false_word = number_to_word(false_n)
        true_rel = span_text(spans["relation"])
        false_rel = _false_relation_phrase(true_rel)
        true_attr = span_text(spans["attribute"])
        false_attr = rec["attribute"]["false_value"]
        distractor = _distractor_category(rec)
        hint = span_text(spans["existence_insertion_hint"])

        try:
            h_counting, count_morph = _counting_edit(
                truthful, spans,
                mode=ca["mode"], category=ca["category"],
                true_word=true_word, false_word=false_word, false_n=false_n,
            )
            h_relation = replace_span_in_sentence(
                truthful, span_sentence_idx(spans["relation"]), true_rel, false_rel,
            )
            h_attribute = replace_span_in_sentence(
                truthful, span_sentence_idx(spans["attribute"]), true_attr, false_attr,
            )
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
                true_count_word=true_word, false_count_word=false_word,
                count_category=ca["category"],
                allow_count_noun_morph=count_morph)),
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
        # Combined: S1 from existence + S2–S4 sentence-scoped swaps on that base.
        h_all = h_existence
        h_all = replace_span_in_sentence(
            h_all, span_sentence_idx(spans["attribute"]), true_attr, false_attr,
        )
        if count_morph:
            # Re-apply the same contiguous morph on the existence-based caption.
            h_all, _ = _counting_edit(
                h_all, spans,
                mode=ca["mode"], category=ca["category"],
                true_word=true_word, false_word=false_word, false_n=false_n,
            )
        else:
            h_all = replace_span_in_sentence(
                h_all, span_sentence_idx(spans["counting"]), true_word, false_word,
            )
        h_all = replace_span_in_sentence(
            h_all, span_sentence_idx(spans["relation"]), true_rel, false_rel,
        )
        h_values["all"] = h_all
        comb_errs = validate_combined(truthful, h_all)
        all_errs = validate_record_variants(
            truthful, h_values,
            distractor=distractor,
            true_attr=true_attr,
            false_attr=false_attr,
            true_count_word=true_word,
            false_count_word=false_word,
            true_relation=true_rel,
        )
        all_errs.extend(comb_errs)
        if count_morph:
            # Re-check counting with morph flags (record validator uses defaults).
            all_errs = [
                e for e in all_errs
                if not e.startswith("counting:")
            ] + [
                f"counting: {e}"
                for e in validate_minimal_pair(
                    truthful, h_counting, "counting",
                    true_count_word=true_word, false_count_word=false_word,
                    count_category=ca["category"], allow_count_noun_morph=True,
                )
            ]
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
                    "type": rec["attribute"]["type"],
                    "object": rec["attribute"]["object"],
                    "true_value": true_attr,
                    "false_value": false_attr,
                },
                "counting": {
                "category": ca["category"],
                    "annotated_count": true_n,
                    "mode": ca["mode"],
                    "true_word": true_word,
                    "false_word": false_word,
                },
                "relation": {
                    "a": relation["a"],
                    "b": relation["b"],
                    "type": relation["type"],
                    "geometry": relation.get("geometry", {}),
                    "true": true_rel,
                    "false": false_rel,
                    "false_phrase": false_rel,
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
