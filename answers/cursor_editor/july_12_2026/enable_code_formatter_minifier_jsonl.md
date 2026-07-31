# Enabling Code Formatter & Minifier for readable JSONL

**Extension:** `lyuwenhan.code-formatter-and-minifier` (Code Formatter & Minifier), already installed under the Cursor remote server extensions.

## How to use it (no extra “enable” toggle)

It activates for `json` / `jsonl` (and js/ts/html/css). You invoke it via commands, not Format Document alone.

1. Open a `.jsonl` file (e.g. `data/vti/v2/stage2_captions.jsonl`).
2. Check the status bar language mode is **JSON Lines** / **jsonl** (bottom-right). If it says Plain Text, click it and pick JSON Lines, or add a file association (below).
3. Command Palette (`Ctrl+Shift+P`) → search **Minifier**:
   - **Beautify current file** — pretty-print
   - **Minify current file** — compress back
   - Selection variants if you only want one line/object

Right-click in the editor may also expose Minifier actions.

## Optional: associate `.jsonl` + format-on-command defaults

In Cursor Settings JSON (`Ctrl+,` → open `settings.json`):

```json
{
  "files.associations": {
    "*.jsonl": "jsonl"
  }
}
```

This extension is **not** typically the VS Code “default formatter” for `Shift+Alt+F`; use its **Minifier: Beautify…** commands.

## Important for this repo’s demos JSONL

Pipeline stages expect **one JSON object per line**. Beautifying a whole `.jsonl` into multi-line indented JSON can break resume/parsers that read line-by-line.

Safer workflows:
- Beautify only to **read**, then **Minify** before saving, or
- Copy one line into a scratch `.json` buffer and format that, or
- Don’t save beautified stage artifacts under `data/vti/v2/`.

`demos.jsonl` / stage outputs should stay single-line JSONL on disk.
