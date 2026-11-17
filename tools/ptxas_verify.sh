#!/usr/bin/env bash
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
