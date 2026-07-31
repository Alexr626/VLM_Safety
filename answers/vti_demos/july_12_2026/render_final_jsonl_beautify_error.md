# `render_review.py --stage final` JSONDecodeError

**Cause:** `data/vti/demos_v2.jsonl` was pretty-printed in the editor (tabs + multi-line objects). `read_jsonl` expected one JSON object per line, so line 1 `{` failed with `Expecting property name enclosed in double quotes`.

**Fix applied:**
1. Re-serialized `demos_v2.jsonl` to canonical JSONL (3 records).
2. Hardened `data_scripts/vti_demos_v2/io_utils.py` `read_jsonl` to fall back to streaming `JSONDecoder.raw_decode` when line-oriented parse fails (IDE beautify).

**Re-run:**
```bash
python data_scripts/vti_demos_v2/render_review.py --stage final --open
```
Wrote `data/vti/v2/_review/final_review.html` successfully after the fix.

**Note:** Avoid “Format Document” on `*.jsonl` files — keep one object per line.
