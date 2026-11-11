#!/usr/bin/env bash
# Assemble every fixture with NVIDIA's ptxas, so the translator is checked against
# an assembler and not only against our reading of the ISA.
#
#   tools/ptxas_verify.sh            # assemble everything, report rejections
#   tools/ptxas_verify.sh --disasm   # also print the matrix SASS each fixture becomes
#
# Needs Docker (on macOS, colima). Set WEFT_PTXAS_IMAGE to pin the CUDA image and
# DOCKER_CONTEXT to pick a context. Exits 2 if the image cannot be fetched, 1 if a
# fixture is rejected for a reason nobody expected or if one that has an expectation
# now assembles, and 0 when the only rejections are the expected ones.
set -uo pipefail

image=${WEFT_PTXAS_IMAGE:-nvidia/cuda:12.9.1-devel-ubuntu22.04}
disasm=0
for arg in "$@"; do
  case $arg in
    --disasm) disasm=1 ;;
    *) echo "usage: $0 [--disasm]" >&2; exit 2 ;;
  esac
done

root=$(cd "$(dirname "$0")/.." && pwd)

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "pulling $image"
  if ! docker pull -q "$image" >/dev/null; then
    echo "cannot fetch $image" >&2
    exit 2
  fi
fi

docker run --rm -v "$root":/w -w /w "$image" bash -lc '
disasm='$disasm'
ok=0; absent=0; fail=0; skip=0; stale=0
for f in kernels/*.ptx kernels/*/*.ptx; do
  [ -f "$f" ] || continue
  t=$(sed -n "s/^[[:space:]]*\.target[[:space:]]*\([^[:space:]]*\).*/\1/p" "$f" | head -1)
  v=$(sed -n "s/^[[:space:]]*\.version[[:space:]]*\([^[:space:]]*\).*/\1/p" "$f" | head -1)
  if [ -z "$t" ]; then
    printf "skip %-30s no .target\n" "$f"; skip=$((skip+1)); continue
  fi
  # The fixtures whose subject this ISA has no spelling for, each with the refusal it
  # has to produce. Matching the reason is what keeps a fixture from hiding behind its
  # name: a new failure inside one of them is still a failure.
  want=""
  case "$f" in
    kernels/calldrop.ptx|kernels/callmiss.ptx) want="Call has wrong number of parameters" ;;
    kernels/cvtf8x2simd.ptx)                   want="Illegal modifier" ;;
    kernels/cvtqual.ptx)                       want="Arguments mismatch for instruction" ;;
    kernels/ldmatrixsub.ptx)                   want="Unexpected instruction types" ;;
    kernels/tensormap3.ptx)                    want="Arguments mismatch for instruction" ;;
    kernels/tensormap4.ptx)                    want="Illegal modifier" ;;
  esac
  if ptxas -arch="$t" -o /tmp/o.cubin "$f" 2>/tmp/e; then
    if [ -n "$want" ]; then
      stale=$((stale+1))
      printf "STALE %-28s %-8s ptx %s assembles now, drop its expectation\n" "$f" "$t" "$v"
      continue
    fi
    ok=$((ok+1)); printf "ok   %-30s %-8s ptx %s\n" "$f" "$t" "$v"
    if [ "$disasm" = 1 ]; then
      sass=$(nvdisasm /tmp/o.cubin 2>/dev/null | grep -oE "(ST|LD)SM[.0-9A-Z]*" | sort | uniq -c | tr -s " " | tr "\n" " ")
      [ -n "$sass" ] && printf "     %s\n" "$sass"
    fi
  else
    msg=$(head -1 /tmp/e | sed "s/.*error[[:space:]]*: //")
    if [ -n "$want" ] && [ "${msg#*$want}" != "$msg" ]; then
      absent=$((absent+1)); printf "expect %-26s %-8s ptx %s: %s\n" "$f" "$t" "$v" "$msg"
    else
      fail=$((fail+1))
      printf "FAIL %-30s %-8s ptx %s: %s\n" "$f" "$t" "$v" "$msg"
    fi
  fi
done
echo "---- $ok assembled, $absent expected-absent, $fail rejected, $skip skipped"
[ "$fail" -eq 0 ] && [ "$stale" -eq 0 ]
'
