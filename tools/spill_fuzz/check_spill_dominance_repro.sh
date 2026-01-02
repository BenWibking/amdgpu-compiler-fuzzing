#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/spill_fuzz/check_spill_dominance_repro.sh \
  --llc /path/to/llc \
  --mcpu gfx942 \
  --input /path/to/kernel.ll \
  --vgpr 20 \
  --sgpr 108

This script:
  1) Runs llc on the original IR and checks for zero spill dominance issues.
  2) Rewrites the IR with fixed register limits and checks for spill dominance
     issues in the greedy-RA MIR dump.
EOF
}

LLC=""
MCPU=""
INPUT=""
VGPR=""
SGPR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llc) LLC="$2"; shift 2 ;;
    --mcpu) MCPU="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --vgpr) VGPR="$2"; shift 2 ;;
    --sgpr) SGPR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$LLC" || -z "$MCPU" || -z "$INPUT" || -z "$VGPR" || -z "$SGPR" ]]; then
  usage
  exit 1
fi

if [[ ! -x "$LLC" ]]; then
  echo "llc not found or not executable: $LLC" >&2
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "input not found: $INPUT" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

ORIG_ERR="$TMPDIR/llc.orig.err"
MUT_ERR="$TMPDIR/llc.mut.err"
CORPUS_DIR="$TMPDIR/corpus"
OUT_DIR="$TMPDIR/out"

mkdir -p "$CORPUS_DIR" "$OUT_DIR"
cp "$INPUT" "$CORPUS_DIR/"

echo "== Checking original IR =="
"$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu="$MCPU" \
  -stop-after=greedy -print-after=greedy -o /dev/null \
  "$INPUT" > /dev/null 2> "$ORIG_ERR"

python3 - "$ORIG_ERR" <<'PY'
from tools.spill_fuzz.spill_fuzz import check_spill_dominance
import sys

text = open(sys.argv[1], "r").read()
issues = check_spill_dominance(text)
print(f"original issues: {len(issues)}")
for issue in issues:
    print(f"  {issue.function} {issue.block} %stack.{issue.slot}: {issue.reason}")
PY

echo "== Generating mutated IR =="
./tools/spill_fuzz/spill_fuzz.py \
  --corpus "$CORPUS_DIR" \
  --llc "$LLC" \
  --mcpu "$MCPU" \
  --passes greedy \
  --iterations 1 \
  --min-vgpr "$VGPR" --max-vgpr "$VGPR" \
  --min-sgpr "$SGPR" --max-sgpr "$SGPR" \
  --out-dir "$OUT_DIR" > /dev/null

stem="$(basename "$INPUT" .ll)"
MUT_LL="$OUT_DIR/${stem}.vgpr${VGPR}.sgpr${SGPR}.ll"
if [[ ! -f "$MUT_LL" ]]; then
  echo "mutated IR not found: $MUT_LL" >&2
  exit 1
fi

echo "== Checking mutated IR =="
"$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu="$MCPU" \
  -stop-after=greedy -print-after=greedy -o /dev/null \
  "$MUT_LL" > /dev/null 2> "$MUT_ERR"

python3 - "$MUT_ERR" <<'PY'
from tools.spill_fuzz.spill_fuzz import check_spill_dominance
import sys

text = open(sys.argv[1], "r").read()
issues = check_spill_dominance(text)
print(f"mutated issues: {len(issues)}")
for issue in issues:
    print(f"  {issue.function} {issue.block} %stack.{issue.slot}: {issue.reason}")
PY
