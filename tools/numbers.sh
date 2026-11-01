#!/usr/bin/env bash
# Regenerates the `Numbers` table in the README from the kernels themselves, so
# the table is a measurement rather than eight numbers typed once. Prints the
# markdown rows; `make numbers` runs it and drops the README's paragraph
# unchanged unless the numbers move.
set -euo pipefail

cd "$(dirname "$0")/.."
WEFT=./build/weft
BENCH="${BENCH:-30}"

row() {  # name, grid, buffers, then the extra flags
  local name="$1" grid="$2" bufs="$3"; shift 3
  local out
  out="$("$WEFT" run kernels/basic.metal --kernel "$name" --grid "$grid" \
        --buffers "$bufs" "$@" --bench "$BENCH" 2>/dev/null)"
  local cpu gpu thr
  cpu="$(echo "$out" | sed -n 's/^cpu .*(best \([0-9.]*\)).*/\1/p')"
  gpu="$(echo "$out" | sed -n 's/^gpu .*(best \([0-9.]*\)).*/\1/p')"
  thr="$(echo "$out" | sed -n 's/^cpu *cpu, \([0-9]*\) thread.*/\1/p')"
  [ -n "$cpu" ] && [ -n "$gpu" ] || { echo "weft: no timing for $name" >&2; return 1; }
  awk -v n="$name" -v g="$grid" -v gpu="$gpu" -v cpu="$cpu" -v t="$thr" 'BEGIN{
    r = cpu / gpu;
    if (r < 0.97)      v = sprintf("**CPU faster** %.1fx", gpu / cpu);
    else if (r > 1.03) v = sprintf("%.1fx slower", r);
    else               v = "tie";
    printf "| %s | %d | %.2f ms | %.2f ms | %d | %s |\n", n, g, gpu, cpu, t, v;
  }'
}

echo "| kernel | grid | GPU | CPU | threads | |"
echo "|---|---|---|---|---|---|"
row saxpy        1048576 1048576,1048576,1048576
row saxpy        4194304 4194304,4194304,4194304
row vec4_scale   1048576 4194304,4194304
row stencil5     1048576 1048576,1048576
row atomic_bins  1048576 1048576,8
row reduce_sum   1048576 1048576,4096 --threads 256
row warpscan     1048576 1048576,1048576 --threads 256
row block_reduce 1048576 1048576,4096 --threads 256
