#!/usr/bin/env bash
# Last updated: 2026-07-30
# Verify the spec gate. Run from anywhere; resolves paths itself.
#
#   bash .claude/hooks/verify_harness.sh
#
# Exercises the hook directly with synthetic PreToolUse events, so it does not depend on
# getting an agent to attempt a write. Reruns from a different working directory, which is
# the failure that shipped in the first version.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
HOOK="$HERE/require_design_spec.py"
DSTUB="$ROOT/designs/__verify_stub_design.md"
XSTUB="$ROOT/extractions/__verify_stub_extraction.md"

pass=0; fail=0

check() {  # name want_exit json
  local name="$1" want="$2" payload="$3" err got
  err="$(printf '%s' "$payload" | python3 "$HOOK" 2>&1 >/dev/null)"; got=$?
  if [ "$got" -eq "$want" ]; then
    printf '  ok    %-44s exit=%s\n' "$name" "$got"; pass=$((pass+1))
  else
    printf '  FAIL  %-44s exit=%s want=%s  %s\n' "$name" "$got" "$want" "${err:0:55}"; fail=$((fail+1))
  fi
}

ev() { printf '{"tool_name":"Write","tool_input":{"file_path":"%s","content":"%s"}}' "$1" "$2"; }

fill() { python3 -c '
import sys
open(sys.argv[1], "w").write(
    "# Spec\n\n## The question\nA real filled question with no placeholders.\n"
    + ("Filled specification content carrying real detail. " * 10))' "$1"; }

unfill() { python3 -c '
import sys
open(sys.argv[1], "w").write(
    "# Spec\n\n" + ("padding to clear the length floor. " * 14)
    + "\n\n## The question\n<answer>\n<condition 1>\n<prediction>\n<answer>\n")' "$1"; }

echo "hook: $HOOK"
[ -f "$HOOK" ] || { echo "MISSING - the gate is not installed"; exit 1; }
mkdir -p "$ROOT/designs" "$ROOT/extractions"

echo
echo "must block (exit 2):"
check "no spec declared"              2 "$(ev implementation_plans/x.md '# Plan')"
check "design_spec points nowhere"    2 "$(ev implementation_plans/x.md '# Plan\ndesign_spec: designs/absent_design.md')"
check "extraction_spec points nowhere" 2 "$(ev implementation_plans/x.md '# Plan\nextraction_spec: extractions/absent_extraction.md')"
printf 'too short\n' > "$DSTUB"
check "design_spec is a stub"         2 "$(ev implementation_plans/x.md '# Plan\ndesign_spec: designs/__verify_stub_design.md')"
unfill "$DSTUB"
check "design_spec still a template"  2 "$(ev implementation_plans/x.md '# Plan\ndesign_spec: designs/__verify_stub_design.md')"
unfill "$XSTUB"
check "extraction_spec still template" 2 "$(ev implementation_plans/x.md '# Plan\nextraction_spec: extractions/__verify_stub_extraction.md')"
fill "$DSTUB"; fill "$XSTUB"
check "both spec kinds declared"      2 "$(ev implementation_plans/x.md '# Plan\ndesign_spec: designs/__verify_stub_design.md\nextraction_spec: extractions/__verify_stub_extraction.md')"
check "design_spec in wrong directory" 2 "$(ev implementation_plans/x.md '# Plan\ndesign_spec: extractions/__verify_stub_extraction.md')"

echo
echo "must pass (exit 0):"
check "plan with filled design_spec"     0 "$(ev implementation_plans/x.md '# Plan\ndesign_spec: designs/__verify_stub_design.md')"
check "plan with filled extraction_spec" 0 "$(ev implementation_plans/x.md '# Plan\nextraction_spec: extractions/__verify_stub_extraction.md')"
check "write under answers/"             0 "$(ev answers/vti/july_28_2026/note.md '# Note')"
check "write under analysis/"            0 "$(ev analysis/run_reading.md '# Reading')"

echo
echo "same suite from a different working directory:"
cd "$ROOT/answers" 2>/dev/null || cd /tmp
check "answers/ write after cd"       0 "$(ev answers/vti/july_28_2026/note.md '# Note')"
check "no spec after cd"              2 "$(ev implementation_plans/x.md '# Plan')"

rm -f "$DSTUB" "$XSTUB"
echo
echo "passed=$pass failed=$fail"
[ "$fail" -eq 0 ] || { echo "GATE IS NOT BEHAVING - do not rely on it"; exit 1; }
echo "gate behaving correctly"
