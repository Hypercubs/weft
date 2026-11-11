#!/usr/bin/env bash
# Assemble every fixture under kernels/ with NVIDIA's ptxas, so the translator is
# checked against an assembler and not only against our reading of the ISA.
#
#   tools/ptxas_verify.sh            # assemble everything, report rejections
#   tools/ptxas_verify.sh --disasm   # also print the matrix SASS each fixture becomes
#
# `kernels/` holds only fixtures that have to assemble, so a rejection is a failure and
# there is no list of accepted ones. The files ptxas cannot take are not candidates and
# do not live here: `tests/refused/` holds the two signature tests, whose whole subject
# is a call an assembler must reject, and `tests/absent/` holds the five whose
# instruction this ISA has no spelling for. check.sh still runs all seven and names the
# refusal each has to produce.
#
# Needs Docker (on macOS, colima). Set WEFT_PTXAS_IMAGE to pin the CUDA image and
# DOCKER_CONTEXT to pick a context. Exits 2 if the image cannot be fetched, 1 if any
# fixture is rejected, and 0 when every one assembles.
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
ok=0; fail=0; skip=0
for f in kernels/*.ptx kernels/*/*.ptx; do
  [ -f "$f" ] || continue
  t=$(sed -n "s/^[[:space:]]*\.target[[:space:]]*\([^[:space:]]*\).*/\1/p" "$f" | head -1)
  v=$(sed -n "s/^[[:space:]]*\.version[[:space:]]*\([^[:space:]]*\).*/\1/p" "$f" | head -1)
  if [ -z "$t" ]; then
    printf "skip %-30s no .target\n" "$f"; skip=$((skip+1)); continue
  fi
  if ptxas -arch="$t" -o /tmp/o.cubin "$f" 2>/tmp/e; then
    ok=$((ok+1)); printf "ok   %-30s %-8s ptx %s\n" "$f" "$t" "$v"
    if [ "$disasm" = 1 ]; then
      sass=$(nvdisasm /tmp/o.cubin 2>/dev/null | grep -oE "(ST|LD)SM[.0-9A-Z]*" | sort | uniq -c | tr -s " " | tr "\n" " ")
      [ -n "$sass" ] && printf "     %s\n" "$sass"
    fi
  else
    fail=$((fail+1))
    printf "FAIL %-30s %-8s ptx %s: %s\n" "$f" "$t" "$v" "$(head -1 /tmp/e | sed "s/.*error[[:space:]]*: //")"
  fi
done
echo "---- $ok assembled, $fail rejected, $skip skipped"
[ "$fail" -eq 0 ]
'
