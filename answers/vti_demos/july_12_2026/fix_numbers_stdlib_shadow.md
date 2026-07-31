# Fix: `numbers.py` shadowed stdlib during Anthropic import

**Symptom:** Running stage1+ with a real provider failed at `MLLMClient(...); import anthropic` with:

```
ImportError: attempted relative import with no known parent package
```

from `data_scripts/vti_demos_v2/numbers.py` while Python's `decimal` tried `import numbers`.

**Cause:** Stage scripts put `data_scripts/` on `sys.path`. Our module `numbers.py` shadowed the stdlib `numbers` package. Mock mode never imported `anthropic`, so the dry-run hid the bug.

**Fix:** Renamed to `number_words.py` and updated imports. Verified: pytest 16 passed; `MLLMClient("anthropic:claude-sonnet-4-6")` constructs successfully.
