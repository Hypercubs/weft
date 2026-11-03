#!/usr/bin/env bash
# Runs each example kernel twice — once on the GPU, once on the CPU — and fails
# if the two disagree. Then checks the PTX front end against the Metal front end
# by comparing their byte-for-byte output.
set -euo pipefail

cd "$(dirname "$0")"
WEFT=./build/weft
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

# A build with no GPU column — anywhere that is not macOS — still runs every
# kernel here: translation, launch and the byte comparisons between front ends.
# What it has no hardware for is the diff against the GPU, so that line says so
# instead of reporting an agreement nobody checked.
NOSHADOW=""
if "$WEFT" run kernels/basic.metal --kernel saxpy --grid 16 --buffers 16,16,16 \
     --bench 1 2>/dev/null | grep -q '^gpu *unavailable'; then
  CPUMODE="--cpu-only"
  NOSHADOW="  (no gpu on this host, ran on the cpu)"
else
  CPUMODE=""
fi

# The `compare` line exists only when there was a GPU run to compare against.
summary() {
  local out="$1" line
  line="$(echo "$out" | grep '^compare' | sed 's/compare  *//' || true)"
  echo "${line:-${NOSHADOW#  }}"
}

run() {
  local name="$1"; shift
  local out
  out="$("$WEFT" run kernels/basic.metal --kernel "$name" "$@" --bench 5 $CPUMODE)"
  if echo "$out" | grep -q DISAGREE; then
    echo "FAIL  $name"
    echo "$out" | sed 's/^/      /'
    fail=1
  else
    echo "ok    $name  $(summary "$out")"
  fi
}

# The same check for a kernel that lives outside basic.metal.
runf() {
  local file="$1" name="$2"; shift 2
  local out
  out="$("$WEFT" run "$file" --kernel "$name" "$@" --bench 5 $CPUMODE)"
  if echo "$out" | grep -q DISAGREE; then
    echo "FAIL  $name"
    echo "$out" | sed 's/^/      /'
    fail=1
  else
    echo "ok    $name  $(summary "$out")"
  fi
}

run saxpy      --grid 1048576 --buffers 1048576,1048576,1048576
run texmix     --grid 65536   --buffers 65536 --textures 64x64   # texture read + nearest sample
run swzcmp     --grid 65536   --buffers 65536                    # compound swizzle assignment
# A 1-D texture: one row, one coordinate, bound as a Wx1 image.
run tex1ds     --grid 65536   --buffers 65536 --textures 64x1d

# Bilinear filtering: the weights are quantised the way the hardware quantises
# them (round(f * 256) / 256, measured from the GPU) and blended with the fma the
# hardware uses, which reproduces a one-axis sample exactly. Across both axes one
# ulp of the final blend remains — 1.2e-07, where the unquantised weights were
# 1.4e-03. The run's `agree` is that bound; a regression shows up as DISAGREE.
run texlin     --grid 65536   --buffers 65536 --textures 64x64   # bilinear, 8-bit weights + fma
# A 2D array: `--textures 64x64*4` binds four slices, and the kernel reads and
# samples the slice it names. The layer is folded into the value, so reading the
# wrong slice — or the whole array from slice 0 — answers wrong rather than close.
run texarray   --grid 65536   --buffers 65536 --textures '64x64*4'
# A 3D texture: `--textures 64x64x8` binds a volume, and every read and sample
# names a plane. The plane is folded into the answer, so reading plane 0 of the
# volume — or taking the depth for a slice — answers wrong rather than close.
run tex3d      --grid 65536   --buffers 65536 --textures 64x64x8
# A cube: `--textures 64x64c` binds six square faces, and a sample takes a
# direction. Each lane rebuilds the direction that lands on the texel it wants, so
# a wrong face or a flipped axis reads a different texel.
run texcube    --grid 65536   --buffers 65536 --textures 64x64c
run stencil5   --grid 1048576 --buffers 1048576,1048576
run stencil5   --grid 1000    --buffers 1000,1000          # ragged final group
run reverse    --grid 1048576 --threads 256 --buffers 1048576,1048576
run reduce_sum --grid 1048576 --threads 256 --buffers 1048576,4096
run reduce_sum --grid 1536    --threads 96  --buffers 1536,16
run vec4_scale --grid 262144  --buffers 1048576,1048576     # 16-byte vector load/store
run vec3_mix   --grid 262144  --buffers 1048576,1048576     # float3: 16-byte stride, 4 bytes padding
run half_cvt   --grid 1048576 --buffers 1048576,1048576     # half + half2 layout and conversion
run halfarith  --grid 65536   --buffers 9961472,524320     # fp16 arithmetic, library, half2/3/4
run half_tg    --grid 65536   --threads 64 --buffers 65536,1048576  # fp16 in threadgroup memory
run tg_vec3    --grid 65536   --threads 64 --buffers 1048576,1048576  # float3 in threadgroup memory
run int_pack   --grid 262144  --buffers 1048576,1048576     # int3/uint3/uchar4/short2/char4
run vecmath    --grid 4096    --buffers 12288,16384         # dot/cross/normalize/mix/clamp/smoothstep
run swzwrite   --grid 4096    --buffers 4096,32768          # a swizzle on the left of an assignment
run atomic_bins --grid 1048576 --buffers 1048576,8          # integer atomics
run warpscan   --grid 1048576 --threads 256 --buffers 1048576,1048576   # 5 shuffles
run fillscan   --grid 1048576 --threads 256 --buffers 1048576,1048576   # shuffle_and_fill
run prefixscan --grid 1048576 --threads 256 --buffers 1048576,1048576   # simd_prefix sums
# A float prefix sum cannot be bit-identical: the hardware folds the warp in its
# own tree and this folds it the doubling way, so the two differ by a few ulp of
# the running sum. What must hold is that the gap stays that small — a wrong scan
# would be off by the whole warp, not by 1e-7 relative.
if [ -n "$CPUMODE" ]; then
  # The gap being measured is between this engine and the hardware's own fold
  # tree. With no GPU run there is no second answer, so there is nothing to
  # measure, and saying so is better than failing a check that cannot run.
  echo "skip  pfxscan    float prefix sums need the gpu column to measure the gap"
else
pfx_md=$("$WEFT" run kernels/basic.metal --kernel pfxscan --grid 1048576 --threads 256 \
    --buffers 1048576,1048576 --bench 3 | grep '^compare' | sed 's/.*max |diff| //; s/ .*//')
if [ "$pfx_md" = "0" ]; then
  echo "ok    pfxscan    float prefix sums are bit-identical: one fold tree, both engines"
else
  echo "FAIL  pfxscan    float prefix sums differ by $pfx_md absolute"; fail=1
fi
fi
run logic      --grid 1048576 --buffers 1048576,1048576   # lop3 + the class tests
run block_reduce --grid 1048576 --threads 256 --buffers 1048576,4096
run block_reduce --grid 3072  --threads 96  --buffers 3072,32          # 3 warps/group
run votek      --grid 1048576 --threads 256 --buffers 1048576,1048576  # simd_ballot/_all/_any
run bits       --grid 1048576 --buffers 1048576,1048576   # funnel shift, mulhi, prmt, cvt
run roundk     --grid 1048576 --buffers 1048576,1048576   # the four rounding converts
run grpops     --grid 1048576 --threads 256 --buffers 1048576,1048576  # the rest of the warp/quad library
run grpsf      --grid 1048576 --threads 256 --buffers 1048576,1048576  # float sum/max/product, bit-exact
run widevec    --grid 1048576 --buffers 1048576,1048576   # bool/long/ulong/double vectors
run sgmat      --grid 1048576 --threads 256 --buffers 1048576,1048576  # simdgroup 8x8 multiply
run sgmatT     --grid 1048576 --threads 256 --buffers 1048576,1048576  # ... with a transposed load
run wide_shfl  --grid 1048576 --threads 256 --buffers 1048576,1048576  # shuffles of a float2 and a long
run veccmp     --grid 1048576 --buffers 1048576,1048576   # vector comparisons -> bool vectors
# packed_* is a layout, and the kernel checks the layouts by printing sizeof into
# the arithmetic: 12+3+6+4 bytes, and 32 for the struct of three of them. The
# result also sums a large constant, so fast-math reassociation shows up as a
# 1e-7 relative difference and nothing more.
run packedk    --grid 1048576 --buffers 1048576,1048576
# 8- and 16-bit vectors wrap in their own width: 200*2 as uchar is 144, and
# 30000*2 as short is -5536. A wider intermediate would give a different sum.
run narrowv    --grid 1048576 --buffers 1048576,1048576
# A file-scope struct and two helpers: one that only computes is compiled as
# written, one that waits on the threadgroup is expanded into the kernel.
run helperk    --grid 1048576 --threads 256 --buffers 1048576,1048576
# A storage image: each lane writes a texel and reads its neighbour's after a
# mem_texture barrier, so a write that never reached the texture answers wrong.
run texrw      --grid 4096 --threads 256 --buffers 4096,4096 --textures 64x64
# threadgroups_per_grid rounds up for a short last group: 1000 threads at 96 a
# group is 11 groups, not 10.
run counts     --grid 1000 --threads 96 --buffers 1000
run counts     --grid 3072 --threads 96 --buffers 3072
# A struct the kernel body declares for itself, held in locals on both sides of a
# barrier: the type is lifted to file scope so the state machine can carry the
# value, and the values read after the wait are the neighbours'.
run localtype  --grid 1048576 --threads 256 --buffers 1048576,1048576
# Lambdas called where they are written: the first shadows a local with its own
# parameter, the second captures one by reference across the barrier.
run lambdak    --grid 1048576 --threads 256 --buffers 1048576,1048576
# A lambda kept in a local across the barrier, and the three capture-list shapes.
run lamstore   --grid 1048576 --threads 256 --buffers 1048576,1048576
run lamcap     --grid 1048576 --threads 256 --buffers 1048576,1048576
# A body-declared type with a method of its own, its local held across the wait.
run localmeth  --grid 1048576 --threads 256 --buffers 1048576,1048576
# A lambda kept in a local that copies one: the copy has to be the value the lambda
# was made with, while a `[&w]` beside it follows the blob.
run stcopy     --grid 1048576 --threads 256 --buffers 1048576,1048576
run stname     --grid 1048576 --threads 256 --buffers 1048576,1048576
# ... and one that copies an array local, element by element.
run starray    --grid 1048576 --threads 256 --buffers 1048576,1048576
run stinit     --grid 1048576 --threads 256 --buffers 1048576,1048576
run stmut      --grid 1048576 --threads 256 --buffers 1048576,1048576
run stparm     --grid 1048576 --threads 256 --buffers 1048576,1048576
run stnest     --grid 1048576 --threads 256 --buffers 1048576,1048576
run autolocal  --grid 1048576 --threads 256 --buffers 1048576,1048576
run arrinit    --grid 1048576 --threads 256 --buffers 1048576,1048576
run refcap     --grid 1048576 --threads 256 --buffers 1048576,1048576
run shadow     --grid 1048576 --threads 256 --buffers 1048576,1048576
run addrspace  --grid 1048576 --threads 256 --buffers 1048576,1048576,1048576
run arr2d      --grid 1048576 --threads 256 --buffers 1048576,1048576
run refsub     --grid 1048576 --threads 256 --buffers 1048576,1048576
run tgblock    --grid 1048576 --threads 256 --buffers 1048576,1048576
# Packing has one rounding rule and it is a tie that tests it: the values are every
# byte boundary, exact ties, and snorm's negative half, compared byte for byte.
run packround  --grid 65536 --buffers 262144,786432
# A componentwise compare makes a mask, and select/any/all are its three readers.
run choosemask --grid 65536 --buffers 262144,524288
# Swizzles in every shape a kernel writes: a prefix, a reversal, a broadcast, two
# out of order, and a swizzle of a swizzle.
run swizzle    --grid 65536 --buffers 262144,524288

# The two hard fixtures. A raymarcher is nothing but `length`, `normalize`, `dot`,
# `reflect` and `smoothstep` in a 64-step loop, so a single bit of difference in
# the portable library walks the ray somewhere else; the FFT is a permuted tree
# with a `sin`/`cos` at every node and nine barriers per threadgroup.
# A 1-D texture: one coordinate, and a 64x1 binding behind it. The height is
# folded into the answer, so a binding of the wrong shape lands on a different
# number instead of on a sample that quietly agrees.
runf kernels/ptx1d.ptx ptx1d --grid 65536 --buffers 65536 --textures 64x1
runf kernels/hard.metal raymarch --grid 65536 --buffers 65536,65536     # SDF raymarch, 64 steps
runf kernels/hard.metal fft --grid 2048 --threads 256 --buffers 4096,4096  # 256-point radix-2, 9 barriers

# `double` is rejected by Metal on Apple GPUs, so the double vectors are only
# checkable on the CPU: the kernel compares the vector operators against the same
# arithmetic written out scalar by scalar, and every output must be 1.0f.
"$WEFT" run kernels/dblvec.metal --kernel dblvec --cpu-only --grid 4096 --buffers 4096,4096 \
    --dump 1="$TMP/dbl.bin" >/dev/null
ones=$(od -An -tu4 -v "$TMP/dbl.bin" | tr -s ' ' '\n' | grep -c 1065353216)
if [ "$ones" = "4096" ]; then
  echo "ok    dblvec   double2/3/4 match the scalar arithmetic (4096/4096)"
else
  echo "FAIL  dblvec   $ones of 4096 components matched"; fail=1
fi

# Both sides of a twin comparison, with one extra rule: neither dump may be empty.
# `cmp` treats two missing files as equal, so a comparison that silently produced
# nothing would otherwise read as agreement. Defined before its first caller — a
# function called before its definition is just a missing command.
same_dumps() {
  local tag="$1" a="$2" b="$3"
  if [ -s "$a" ] && [ -s "$b" ] && cmp -s "$a" "$b"; then return 0; fi
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then
    local sa sb
    sa=$(wc -c <"$a" 2>/dev/null || echo 0)
    sb=$(wc -c <"$b" 2>/dev/null || echo 0)
    echo "      $tag: a dump is empty or missing ($sa / $sb bytes)"
  fi
  return 1
}

# PTX front end vs the Metal front end: same math, two source languages.
cross() {
  local kernel="$1" idx="$2"; shift 2
  "$WEFT" run kernels/basic.metal --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/msl.bin" "$@" >/dev/null
  "$WEFT" run "kernels/$kernel.ptx"        --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/ptx.bin" "$@" >/dev/null
  if same_dumps "ptx:$kernel" "$TMP/msl.bin" "$TMP/ptx.bin"; then
    echo "ok    ptx:$kernel  matches the metal front end byte for byte"
  else
    echo "FAIL  ptx:$kernel  differs from the metal front end"
    fail=1
  fi
}

cross saxpy   2 --grid 1048576 --buffers 1048576,1048576,1048576
cross reverse 1 --grid 1048576 --threads 256 --buffers 1048576,1048576
# Dynamic threadgroup memory (`extern .shared`) and cp.async.
cross async_copy 1 --grid 1024 --threads 64 --buffers 1024,1024
# Two threadgroup arrays in one kernel: the second must not alias the first.
cross twotile 1 --grid 1024 --threads 64 --buffers 1024,1024

# The same, but the PTX was emitted by LLVM's NVPTX backend (llc), not written
# by hand — so the input is real compiler output, register allocation and all.
nvptx() {
  local kernel="$1" idx="$2"; shift 2
  "$WEFT" run kernels/basic.metal      --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/msl.bin" "$@" >/dev/null
  "$WEFT" run "kernels/cuda/$kernel.ptx" --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/nv.bin" "$@" >/dev/null
  if same_dumps "nvptx:$kernel" "$TMP/msl.bin" "$TMP/nv.bin"; then
    echo "ok    nvptx:$kernel  llc-emitted ptx matches the metal kernel byte for byte"
  else
    echo "FAIL  nvptx:$kernel  differs from the metal kernel"
    fail=1
  fi
  # The warp backend is a second execution mode for the same PTX: a kernel with a
  # warp body runs 32 lanes wide there instead of as a per-lane state machine. Both
  # modes have to land on the same bytes as the twin, or which backend a kernel
  # picks would change its answer.
  env WEFT_WARP=1 "$WEFT" run "kernels/cuda/$kernel.ptx" --kernel "$kernel" --cpu-only \
      --dump "$idx"="$TMP/nvw.bin" "$@" >/dev/null
  if same_dumps "nvptx:$kernel warp" "$TMP/msl.bin" "$TMP/nvw.bin"; then
    echo "ok    nvptx:$kernel:warp  the warp backend lands on the same bytes"
  else
    echo "FAIL  nvptx:$kernel:warp  the warp backend differs from the per-lane one"
    fail=1
  fi
}

nvptx saxpy   2 --grid 1048576 --buffers 1048576,1048576,1048576
nvptx reverse 1 --grid 1048576 --threads 256 --buffers 1048576,1048576
# Real compiler shuffles: shfl.sync.up in a scan, shfl.down + shared memory +
# a barrier in a CUB-shaped block reduction.
nvptx warpscan     1 --grid 1048576 --threads 256 --buffers 1048576,1048576
# Four samples that sat in kernels/cuda/ with nothing diffing them: a shuffle-down
# fold, the same fold with twelve multiplies a step, a predicated store, and an
# indexed rotate. The coverage check at the end of this file is what found them.
nvptx bsum         1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx hsum         1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx pred         1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx rotate       1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx block_reduce 1 --grid 1048576 --threads 256 --buffers 1048576,4096

# Triton goes through LLVM's NVPTX backend too, so its output is PTX in a
# particular shape: `.maxntid` from `num_warps`, the pointers then
# `global_scratch` (declared, never read, so not bound) then the scalars, and for
# `tl.dot` tensor-core instructions. The Metal twin is written from the fragment
# layout the PTX instruction defines — each lane owns rows g and g+8, columns 2t
# and 2t+1 — so a mismatch means the layout translation is wrong, not that the
# two disagree about rounding: the PTX rounds its inputs to integers first, so
# every product and every sum of sixteen is exact.
triton() {
  local twin="$1" kernel="$2" idx="$3"; shift 3
  "$WEFT" run kernels/basic.metal         --kernel "triton_$twin" --cpu-only --dump "$idx"="$TMP/msl.bin" "$@" >/dev/null
  "$WEFT" run "kernels/triton/$twin.ptx"  --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/tr.bin"  "$@" >/dev/null
  if same_dumps "triton:$twin" "$TMP/msl.bin" "$TMP/tr.bin"; then
    echo "ok    triton:$twin  matches the metal twin byte for byte"
  else
    echo "FAIL  triton:$twin  differs from the metal twin"; fail=1
  fi
  # The warp backend runs the same PTX 32 lanes wide; the tensor-core shapes are
  # already warp-only, so this is the comparison that would notice if a kernel
  # with both bodies answered differently depending on which one ran.
  env WEFT_WARP=1 "$WEFT" run "kernels/triton/$twin.ptx" --kernel "$kernel" --cpu-only \
      --dump "$idx"="$TMP/trw.bin" "$@" >/dev/null
  if same_dumps "triton:$twin warp" "$TMP/msl.bin" "$TMP/trw.bin"; then
    echo "ok    triton:$twin:warp  the warp backend lands on the same bytes"
  else
    echo "FAIL  triton:$twin:warp  the warp backend differs"; fail=1
  fi
}

# The GPU column, for a front end that used to have none.
#
# PTX has a second back end: the same decode emitted as Metal (src/weft_msl.hpp is
# the runtime behind it). This runs that module on the GPU and compares it with
# the module the C++ target produced from the same input — so the two sides are
# the same generated code, and a disagreement points at a back end rather than at
# two implementations that were written to agree.
#
# A kernel whose GPU form does not exist on this host keeps its CPU column and
# says which instruction or which compiler failure stopped it, rather than
# passing a hand-written twin off as a GPU run of this translation.
gpucol() {
  local file="$1" threads="$2" grid="$3"; shift 3
  local key out
  # The path under kernels/, not the basename: two fixtures are both named
  # `reduce.ptx`, and one line of output should not stand for two runs.
  key="${file#kernels/}"
  if [ -n "$CPUMODE" ]; then
    echo "skip  gpu:$key  no gpu on this host"
    return
  fi
  local st=0
  # `tr -d '\000'`: a NUL byte ends a shell variable, so a driver that leaves one
  # in the stream would make a whole report look like silence.
  local retried="" attempt=1 tries=""
  # A run that printed nothing did not compare anything, and saying so is better
  # than reporting a disagreement that never happened: this host's Metal compiler
  # occasionally takes the process down on a shader it builds (the same driver
  # that reports XPC_ERROR_CONNECTION_INTERRUPTED), which loses the whole report
  # with the stdout buffer. That is once in a few hundred runs, so the report is
  # worth asking for three times before calling it a gap — a fault that is really
  # there fails every attempt and is still named, with every exit status it gave.
  while [ "$attempt" -le 3 ]; do
    st=0
    if out=$("$WEFT" run "$file" --grid "$grid" --threads "$threads" --bench 2 "$@" 2>&1 |
             LC_ALL=C tr -d '\000'); then :; else st=$?; fi
    if [ -n "$out" ]; then break; fi
    tries="$tries $st"
    attempt=$((attempt + 1))
  done
  if [ -z "$out" ]; then
    echo "skip  gpu:$key  three attempts printed nothing, exiting$tries, so nothing was compared"
    return
  fi
  if [ "$attempt" -gt 1 ]; then
    retried=" (retried after exiting$tries)"
  fi
  # The reason is read off the line itself rather than judged by a search: a
  # driver that writes its own bytes to the same stream must not be able to make
  # a named gap look like an unnamed one.
  local why
  # `LC_ALL=C`: the driver can leave bytes in the stream that are not text, and a
  # UTF-8-aware sed refuses them — which would lose the reason rather than the
  # noise.
  why="$(printf '%s\n' "$out" | LC_ALL=C sed -n 's/.*no column for this kernel *//p' | head -1)"
  if [ -n "$why" ]; then
    echo "skip  gpu:$key  $why"
    return
  fi
  if printf '%s\n' "$out" | grep -aq '^compare .*agree'; then
    echo "ok    gpu:$key  the emitted metal matches the cpu byte for byte$retried"
  else
    echo "FAIL  gpu:$key"
    printf '%s\n' "$out" | tail -4 | LC_ALL=C sed 's/^/      /'
    fail=1
  fi
}

# llc-emitted samples: the input is real compiler output, register allocation
# and all, and every one of them now runs on both engines from one decode.
gpucol kernels/cuda/saxpy.ptx        256 4096
gpucol kernels/cuda/reverse.ptx    256 4096
gpucol kernels/cuda/warpscan.ptx     256 4096
gpucol kernels/cuda/block_reduce.ptx 256 4096
gpucol kernels/cuda/bsum.ptx         256 4096
gpucol kernels/cuda/hsum.ptx         256 4096
gpucol kernels/cuda/pred.ptx         256 4096
gpucol kernels/cuda/rotate.ptx       256 4096
gpucol kernels/cuda/scale.ptx        256 4096
# A v4 load and store is four elements a lane, so the buffers hold 4 x the grid:
# sized from the grid alone, the kernel's own writes run past the end of both.
gpucol kernels/cuda/vecscale.ptx     256 4096 --buffers 16384,16384
# A tree reduction over the whole group: eight barriers, and every thread reads
# slots another thread wrote, so the barrier has to be the hardware's.
gpucol kernels/cuda/reduce.ptx       256 4096
# Triton's shapes, which arrive as PTX: elementwise, a saturating fp16 convert,
# and the wgmma tensor-core form.
gpucol kernels/triton/add.ptx        128 4096
gpucol kernels/triton/cvt8.ptx       128 4096
# The tensor-core kernels tile their buffers by block, so the buffers have to
# hold every block's tile: a tile is 8 KiB of A and B at most, and this grid is
# 128 blocks. Sizing them from the grid alone put writes out of bounds on both
# engines, which is agreement about nothing.
gpucol kernels/triton/wgmma.ptx      128 4096 --buffers 32768,32768,32768
# The same instruction with bf16 operands: one 16-bit element either way, so the
# core-matrix layout is identical and only the decode differs.
gpucol kernels/triton/wgmma_bf16.ptx 128 4096 --buffers 32768,32768,32768
# tf32: 32-bit operands, so a core-matrix row holds four of them and K is 8.
gpucol kernels/triton/wgmma_tf32.ptx 128 4096 --buffers 32768,32768,32768
# fp8: one byte an element, sixteen to a core-matrix row, so K is 32. The buffers
# are sized to what this kernel touches rather than to the grid: nothing past the
# tile is written, so a larger buffer would have both engines compared over memory
# neither of them set.
gpucol kernels/triton/wgmma_fp8.ptx  128 4096 --buffers 2048,256,512 --buffer-bytes 0=1,1=1
# 8-bit integers: the fp8 geometry byte for byte, so the same buffers — and the
# decode alone has to tell signed from unsigned.
gpucol kernels/triton/wgmma_s8.ptx   128 4096 --buffers 2048,256,512 --buffer-bytes 0=1,1=1
# Sixteen columns: two N groups, so B's SBO is a real address and a lane holds
# eight accumulators. A is 1024 halves, B 256, D 1024 floats.
gpucol kernels/triton/wgmma_n16.ptx  128 4096 --buffers 1024,256,1024 --buffer-bytes 0=2,1=2
# MN-major operands, and the same tile issued twice into its own accumulator.
gpucol kernels/triton/wgmma_trans.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
gpucol kernels/triton/wgmma_acc.ptx   128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# The 128-byte swizzle, which is the layout a real Hopper kernel stages.
gpucol kernels/triton/wgmma_swz128.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# The narrow swizzle: a 32-byte atom, where the XOR mask is one bit.
gpucol kernels/triton/wgmma_swz32.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# The middle width, and a tile that starts one row into the pattern.
gpucol kernels/triton/wgmma_swz64.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
gpucol kernels/triton/wgmma_swzbase.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# A swizzled operand that is MN-major: the mirror of the K-major walk.
gpucol kernels/triton/wgmma_swztrans.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# The swizzle on the two ends of the element-width range: four bytes and one.
gpucol kernels/triton/wgmma_swztf32.ptx 128 4096 --buffers 512,64,512 --buffer-bytes 0=4,1=4
gpucol kernels/triton/wgmma_swzf8.ptx   128 4096 --buffers 2048,256,512 --buffer-bytes 0=1,1=1
# The other transpose, and a chain of three issues into one accumulator.
gpucol kernels/triton/wgmma_swztransb.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
gpucol kernels/triton/wgmma_chain.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# An integer accumulator, and a two-immediate form, which PTX allows.
gpucol kernels/triton/wgmma_s32acc.ptx 128 4096 --buffers 2048,256,512 --buffer-bytes 0=1,1=1
gpucol kernels/triton/wgmma_shortimm.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# The same instruction on the 32-lane backend, forced there by a warp-wide fold.
gpucol kernels/triton/wgmma_warp.ptx 128 4096 --buffers 1024,128,512 --buffer-bytes 0=2,1=2
# Hand-written samples: predication, bit tricks, a reduction barrier, two
# threadgroup tiles, and cp.async into threadgroup memory.
gpucol kernels/reverse.ptx           256 4096
gpucol kernels/saxpy.ptx             256 4096
gpucol kernels/logic.ptx           256 4096
gpucol kernels/round.ptx             256 4096
gpucol kernels/bred.ptx              256 4096
gpucol kernels/twotile.ptx            64 4096
gpucol kernels/async_copy.ptx         64 4096
# PTX textures: a sample through the bound sampler state, and a surface written,
# barriered, then read back through a neighbour's texel. The binding order is the
# order the `.tex`/`.surf` declarations appear in.
gpucol kernels/ptxtex.ptx             64 4096 --textures 64x64 --samplers nearest/clamp
gpucol kernels/ptxsrf.ptx             64 4096 --textures 64x64
# One binding written and then sampled: the write needs a read-write view and a
# filter cannot read through one, so the module declares a second, sampled view of
# the same texture and the launcher binds one handle to both. `t0` is the lane's own
# texel, so a sample that read stale storage answers with the image's old contents.
gpucol kernels/ptxsrftmix.ptx         64 4096 --textures 64x64 --samplers nearest/clamp
# 64-bit floating point, which this GPU has no instruction for: the GPU column runs
# the software double over the bits and the CPU column runs the host's own, so the
# two are the same arithmetic and the comparison is byte for byte. The input is
# written as raw doubles — the `0d` literals in the kernel used to have no GPU
# spelling at all — and the answer comes back as one float a lane.
python3 - "$TMP/f64in.bin" <<'PY'
import struct, sys
vals = [0.0, 1.0, -1.0, 0.5, 3.0, 1e-300, 2.0**-1022, 2.0**60, -7.25, 1e300,
        1234.5678, -0.0, 4096.5, 1e-16]
vals = (vals * 300)[:4096]
open(sys.argv[1], 'wb').write(struct.pack('<4096d', *vals))
PY
gpucol kernels/ptxf64.ptx             64 4096 --buffers 4096,4096,4096,4096 --buffer-bytes 0=8,1=4,2=8,3=8 \
    --load "0=$TMP/f64in.bin"
# `tld4` keeps the four texels apart instead of blending them, which is the one
# thing a filter cannot stand in for.
gpucol kernels/ptxgather.ptx          64 4096 --textures 64x64
# The same footprint in the bindings it used to be refused for: a volume gathers in
# the plane its third coordinate names, a one-row binding puts the second row of the
# footprint outside the image (so it answers with the two texels along x, as an edge
# does), and a cube names a face and a place on it instead of a direction.
gpucol kernels/ptxgather3d.ptx        64 4096 --textures 64x64x8
gpucol kernels/ptxgather1d.ptx        64 4096 --textures 64x1
gpucol kernels/ptxgathercube.ptx      64 4096 --textures 64x64c
# `ldmatrix` hands each lane a row address that another lane supplied, so the tile
# is only reachable from a backend that holds a whole warp. The two address forms —
# a named shared symbol and the register a lane computed — must give the same word,
# and the transposed form must give the transpose; both are checked against the
# position each value came from rather than against each other.
python3 -c 'import struct,sys; open(sys.argv[1],"wb").write(struct.pack("<64H",*range(64)))' \
    "$TMP/ldm_in.bin"
if env WEFT_WARP=1 "$WEFT" run kernels/ldmatrix.ptx --kernel ldmatrix --cpu-only --grid 32 \
    --threads 32 --buffers 96,64 --buffer-bytes 1=2 --load 1="$TMP/ldm_in.bin" \
    --dump 0="$TMP/ldm_out.bin" >/dev/null 2>&1; then :; else
  echo "FAIL  ptx:ldmatrix  the fixture did not run"; fail=1
fi
if python3 - "$TMP/ldm_out.bin" <<'PY'
import struct, sys
w = struct.unpack('<96I', open(sys.argv[1], 'rb').read(384))
bad = 0
for l in range(32):
    i = 8 * (l >> 2) + 2 * (l & 3)
    want = i | ((i + 1) << 16)
    if w[l] != want: bad += 1
    if w[32 + l] != want: bad += 1
    t = 8 * (2 * (l & 3)) + (l >> 2)
    if w[64 + l] != (t | ((t + 8) << 16)): bad += 1
if bad:
    print("  %d of 96 words differ" % bad)
sys.exit(0 if bad == 0 else 1)
PY
then
  echo "ok    ptx:ldmatrix  symbol, register and transposed forms all match the layout"
else
  echo "FAIL  ptx:ldmatrix  a form does not match the documented layout"; fail=1
fi

# A quad is four lanes, and a quad operation addresses them by an index or an XOR.
# The fixture writes q0 + 2q1 + 4q2 + 8q3 + 16s0 + 32s1 + 64s2 of the value `lane`,
# so the number it writes is a fingerprint of all seven lookups: check.sh recomputes
# it from the lane id. Agreement between the two columns cannot catch a base taken
# from the warp instead of the quad, because both would take it the same way.
if "$WEFT" run kernels/spirv/quads.spv --kernel quads --cpu-only --grid 32 --threads 32 --buffers 32,32 --dump 1="$TMP/quads.bin" >/dev/null 2>&1; then
  if python3 - "$TMP/quads.bin" <<'PY'
import struct, sys
v = struct.unpack("<32f", open(sys.argv[1], "rb").read(128))
bad = []
for i in range(32):
    b, x = i & ~3, i & 3
    want = ((b + 0) + (b + 1) * 2 + (b + 2) * 4 + (b + 3) * 8
            + (b + (x ^ 1)) * 16 + (b + (x ^ 2)) * 32 + (b + (x ^ 3)) * 64)
    if v[i] != float(want):
        bad.append((i, v[i], want))
for i, got, want in bad[:4]:
    print("  lane %d wrote %g, wanted %d" % (i, got, want))
sys.exit(0 if not bad else 1)
PY
  then
    echo "ok    spvgpu:quads  all four broadcast indices and all three swap directions"
  else
    echo "FAIL  spvgpu:quads  a quad index, a swap direction or the base is wrong"; fail=1
  fi
else
  echo "FAIL  spvgpu:quads  the fixture did not run"; fail=1
fi

# A double is 64 bits, so the relay carries it as two words. `(uint64_t)1.0` is 1 and
# its pattern is 1, so a relay that packed the value instead of the bits wrote zero
# here while both columns compiled — the sum is checked against the lane id instead.
if "$WEFT" run kernels/spirv/dblshfl.spv --kernel dblshfl --cpu-only --grid 32 --threads 32 --buffers 32 --dump 0="$TMP/dblshfl.bin" >/dev/null 2>&1; then
  if python3 - "$TMP/dblshfl.bin" <<'PY'
import struct, sys
v = struct.unpack("<32f", open(sys.argv[1], "rb").read(128))
bad = []
for i in range(32):
    want = 5 + (i ^ 3) + (i - 1 if i >= 1 else i) + (i + 2 if i + 2 < 32 else i)
    if v[i] != float(want):
        bad.append((i, v[i], want))
for i, got, want in bad[:4]:
    print("  lane %d wrote %g, wanted %d" % (i, got, want))
sys.exit(0 if not bad else 1)
PY
  then
    echo "ok    spvgpu:dblshfl  a double through broadcast, xor, up and down"
  else
    echo "FAIL  spvgpu:dblshfl  the 64-bit relay moved the value instead of the bits"; fail=1
  fi
else
  echo "FAIL  spvgpu:dblshfl  the fixture did not run"; fail=1
fi

# A product folds from 1, not from 0: seeded with the addition identity every answer
# is 0 and both columns would still agree on it.
if "$WEFT" run kernels/spirv/submul.spv --kernel submul --cpu-only --grid 32 --threads 32 --buffers 32 --dump 0="$TMP/submul.bin" >/dev/null 2>&1; then
  if python3 - "$TMP/submul.bin" <<'PY'
import struct, sys
v = struct.unpack("<32f", open(sys.argv[1], "rb").read(128))
bad = []
for i in range(32):
    want = 3.5 + 7 + (i + 1 if i < 5 else i + 7)
    if v[i] != want:
        bad.append((i, v[i], want))
for i, got, want in bad[:4]:
    print("  lane %d wrote %g, wanted %g" % (i, got, want))
sys.exit(0 if not bad else 1)
PY
  then
    echo "ok    spvgpu:submul  a product fold over a double and a u32, and its scan"
  else
    echo "FAIL  spvgpu:submul  the product fold is seeded wrong or runs on the bits"; fail=1
  fi
else
  echo "FAIL  spvgpu:submul  the fixture did not run"; fail=1
fi

# A boolean fold has two identities and they are not the same one: `and` starts true,
# `or` and `xor` start false. The sum below is 5 below lane 5 and 13 from it, so a
# swapped identity moves it.
if "$WEFT" run kernels/spirv/subbool.spv --kernel subbool --cpu-only --grid 32 --threads 32 --buffers 32 --dump 0="$TMP/subbool.bin" >/dev/null 2>&1; then
  if python3 - "$TMP/subbool.bin" <<'PY'
import struct, sys
v = struct.unpack("<32f", open(sys.argv[1], "rb").read(128))
bad = [(i, v[i]) for i in range(32) if v[i] != (5.0 if i < 5 else 13.0)]
for i, got in bad[:4]:
    print("  lane %d wrote %g, wanted %g" % (i, got, 5.0 if i < 5 else 13.0))
sys.exit(0 if not bad else 1)
PY
  then
    echo "ok    spvgpu:subbool  and, or and xor folds over a boolean, and an xor scan"
  else
    echo "FAIL  spvgpu:subbool  a boolean fold is seeded with the wrong identity"; fail=1
  fi
else
  echo "FAIL  spvgpu:subbool  the fixture did not run"; fail=1
fi
# `.a2d`: a layered sample whose third coordinate is the slice, and `txq.depth`
# answering with the slice count both readings are folded into the output.
gpucol kernels/ptxarray.ptx           64 4096 --textures '64x64*4' --samplers nearest/clamp
# `.3d`: `txq.depth` reports the plane count and `tex.3d` samples the plane its
# third coordinate names, which is the one thing a 2-D sample cannot express.
gpucol kernels/ptx3d.ptx             256 65536 --textures 64x64x8 --samplers nearest/clamp
# `.cube`: PTX names a face and a position on it, where the sampler takes a
# direction, and `weft_cube_dir` rebuilds the direction that lands back on that
# face at that place. Every lane covers all six faces over the grid, and the face
# is folded into the answer, so picking the wrong one cannot hide behind a texel
# that happens to hold the same number.
gpucol kernels/ptxcube.ptx          256 65536 --textures 64x64c --samplers nearest/clamp
# The bit tricks: funnel shifts, `prmt`, `mul.hi` at 32 and 64 bits, and the
# float-integer conversions. This one used to be a skip — the emitted source
# compiled and the pipeline step then killed the compiler, because a `mul.hi.u64`
# was a `__int128` multiply and this host's back end cannot lower one in a kernel
# this size. The high word is four 32x32 products now, so it has a GPU column of
# its own like any other kernel.
gpucol kernels/cuda/bits.ptx         256 4096
# `mul.hi` at 64 bits with the sign bits forced: the carry chain of the limb
# product, unsigned and signed, and a negative times a negative. Ten words per
# lane out of four words in.
gpucol kernels/mulhi.ptx            256 4096 --buffers 16384,40960
gpucol kernels/dp4a.ptx             256 65536 --buffers 65536
gpucol kernels/mufu.ptx             256 65536 --buffers 65536
# `redux.sync` is one group instruction on the GPU (`simd_sum` and the other
# folds), so a warp-wide integer fold has both columns; a lane outside the member
# mask stands in with the fold's identity, which keeps a partial mask exact.
gpucol kernels/triton/reduce.ptx     128 4096 --buffers 4096,32768
# A 64-bit `red` has no result, so an order-free operation is two 32-bit atomics
# on the halves — the low half's carry into the high one for add/sub, a fold per
# half for and/or/xor. Nothing is read back, so nothing needs the pair as one
# value, and the answer is order-free either way.
gpucol kernels/red64.ptx             256 4096 --buffers 8,8,8,8
# A 64-bit atomic, which this host has no instruction for: `atomic_ulong` is in
# the header, but every entry point that could read, write, exchange or compare one
# is constrained to 32-bit types, and the two that are not (`atomic_max_explicit` /
# `atomic_min_explicit`) compile and are then refused by the device at pipeline
# creation — tools/atom64_probe.mm asks for each answer separately. So the emission
# builds one out of the 32-bit atomics the hardware does have: a lock on the word,
# held while the pair is read and written as two 32-bit atomics. Its 32-bit
# counters are `red`s by the dead-result rewrite, which is what ptxas emits for a
# dropped result. The launch stays parallel — one warp per threadgroup, because a
# thread that spins needs a SIMD group of its own — at 10 ms for 4,096 lanes and
# 150 ms for 262,144, against the CPU's 0.4 ms and 23 ms.
gpucol kernels/atom.ptx              256 4096
# The kernels whose GPU form this host has no answer for. Each is a named skip
# with the reason the translator gave, so the gap has a name on it; the CPU
# column above still runs every one of them.
gpucol kernels/triton/dot32.ptx      128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/dotbf16.ptx    128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/dotfp8.ptx     128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/dote5m2.ptx    128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/matmul.ptx     128 4096 --buffers 32768,32768,32768

# The SPIR-V front end got the same second back end, so its fixtures get the same
# column: the metal emitted from the module against the C++ module from the same
# decode. An image and a sampler are bindings a Metal kernel declares, so those
# two kernels cover the binding side as well.
spvgpu() {
  # The first word names the kernel to run and the row that prints. A row that runs
  # one kernel at a second launch size writes `label=kernel`, so the two results
  # have different names in this output instead of colliding.
  local kernel="$1" file="$2" label; shift 2
  label="$kernel"
  case "$kernel" in *=*) label="${kernel%%=*}"; kernel="${kernel#*=}" ;; esac
  local out
  if [ -n "$CPUMODE" ]; then
    echo "skip  spvgpu:$label  no gpu on this host"
    return
  fi
  local st=0 retried="" attempt=1 tries=""
  # Same retry as the PTX column: silence is the compiler service dying, not an
  # answer, and it is rare enough to be worth asking three times before naming a
  # gap.
  while [ "$attempt" -le 3 ]; do
    st=0
    if out=$("$WEFT" run "kernels/spirv/$file.spv" --kernel "$kernel" --bench 2 "$@" 2>&1 |
             LC_ALL=C tr -d '\000'); then :; else st=$?; fi
    if [ -n "$out" ]; then break; fi
    tries="$tries $st"
    attempt=$((attempt + 1))
  done
  if [ -z "$out" ]; then
    echo "skip  spvgpu:$label  three attempts printed nothing, exiting$tries, so nothing was compared"
    return
  fi
  if [ "$attempt" -gt 1 ]; then
    retried=" (retried after exiting$tries)"
  fi
  local why
  why="$(printf '%s\n' "$out" | LC_ALL=C sed -n 's/.*no column for this kernel *//p' | head -1)"
  if [ -n "$why" ]; then
    echo "skip  spvgpu:$label  $why"
    return
  fi
  if printf '%s\n' "$out" | grep -aq '^compare .*agree'; then
    echo "ok    spvgpu:$label  the emitted metal matches the cpu byte for byte$retried"
  else
    echo "FAIL  spvgpu:$label"
    printf '%s\n' "$out" | tail -4 | LC_ALL=C sed 's/^/      /'
    fail=1
  fi
}
spvgpu saxpy        saxpy     --grid 1048576 --buffers 1048576,1048576,1048576
spvgpu warpscan     warpscan  --grid 1048576 --threads 256 --buffers 1048576,1048576
spvgpu block_reduce blockreduce --grid 1048576 --threads 256 --buffers 1048576,4096
spvgpu votek        vote      --grid 1048576 --threads 256 --buffers 1048576,1048576
spvgpu wide_shfl    wide      --grid 1048576 --threads 256 --buffers 1048576,1048576
# A quad broadcast at each of the four lane indices, and the three quad swaps. The
# fixture writes a weighted sum that check.sh recomputes below.
spvgpu quads       quads      --grid 32 --threads 32 --buffers 32,32
# A double through all four shuffle modes, which the relay moves two words at a
# time: the fixture's sum is recomputed by hand below.
spvgpu dblshfl     dblshfl    --grid 32 --threads 32 --buffers 32
# A product fold over a double, over a u32, and an inclusive scan of the same:
# the value the fold starts from is the point of the fixture.
spvgpu submul      submul     --grid 32 --threads 32 --buffers 32
# A boolean folded three ways, plus an xor scan: the identities are true for `and`
# and false for the others, which is what the fixture's sum pins.
spvgpu subbool     subbool    --grid 32 --threads 32 --buffers 32
spvgpu serial_sum   llvm      --grid 1 --threads 1 --buffers 1000,1 --arg 0=1000
spvgpu globk        global    --grid 1048576 --buffers 1048576,1048576
# An image read and a filtered sample; then a storage image written, barriered,
# and read back through the neighbour's texel.
spvgpu texmix       texture   --grid 65536 --buffers 65536 --textures 64x64 \
  --samplers nearest/clamp
spvgpu texrw        texturerg --grid 4096 --threads 256 --buffers 4096,4096 \
  --textures 64x64
# `Arrayed` in OpTypeImage: every coordinate carries a layer, and
# `OpImageQuerySizeLod` reports the slice count as its third component.
spvgpu texarray     arraytex  --grid 65536 --buffers 65536 --textures '64x64*4' \
  --samplers nearest/clamp
# `Dim = 3D` in OpTypeImage: a coordinate of three components, and
# `OpImageQuerySizeLod` reporting the depth as its third.
spvgpu tex3d        threed    --grid 65536 --buffers 65536 --textures 64x64x8 \
  --samplers nearest/clamp
# Matrices: a 4x4 built from columns, multiplied four ways and transposed, with
# every product weighed differently so a column taken for a row lands on a
# different number.
# The Shader model. A GLCompute module takes no function parameters at all: the
# buffer, the image and the sampler are module-scope variables, which is the shape
# glslang emits for a compute shader. `gl2d` is the control that the model decodes
# at all, `gl1d` samples a plane one row high, and `glcube` builds a direction from
# a face and a position on it so all six faces are sampled.
spvgpu gl1d        gl1d    --grid 65536 --buffers 65536 --textures 64x1 --samplers nearest/clamp
spvgpu gl2d        gl2d    --grid 65536 --buffers 65536 --textures 64x64 --samplers nearest/clamp
spvgpu glcube      glcube  --grid 65536 --buffers 65536 --textures 64x64c --samplers nearest/clamp
# `OpImageGather`: the four texels of a 2x2 footprint, one per channel, which is
# the SPIR-V twin of PTX's `tld4`. It carries a component operand the sample
# instructions do not, so its image operands start a word later.
spvgpu glgather    glgather --grid 65536 --buffers 65536 --textures 64x64 --samplers nearest/clamp
# A gather of one slice of a 2-D array: the footprint is a 2x2 block of the layer
# the third coordinate names.
spvgpu glgatherarray glgatherarray --grid 65536 --buffers 65536 --textures '64x64*4' --samplers nearest/clamp
# A 2-D group. `LocalSize 8 8 1` fixes how a lane id decomposes, so the kernel
# rebuilds its own flat lane from `LocalInvocationId` and writes a 1 where that
# equals the global id modulo the group, a 0 where it does not — which is what a
# wrong decomposition would produce.
spvgpu glgroup     glgroup  --grid 65536 --buffers 65536
spvgpu matk         matk      --grid 65536 --buffers 65536,65536
# OpenCL.std's trigonometry: `sin`, `atan2` and `pow` through the portable
# library rather than the GPU's own approximation, so every lane is bit-identical.
spvgpu mathx        mathx     --grid 65536 --buffers 65536,65536
# The functions that read a float's own bits: `ldexp`, `ilogb`, `logb`,
# `nextafter`, `maxmag`, `minmag`, `pown`, `rootn`, `expm1`.
spvgpu mathy        mathy     --grid 65536 --buffers 65536,65536
spvgpu mathz        mathz     --grid 65536 --buffers 65536,65536
spvgpu atomf        atomf     --grid 1024 --threads 1024 --buffers 1024,1024,2
# `OpImageGather` has no column here, and not because it was skipped: the validator
# refuses it outside a Shader module (`requires one of these capabilities: Shader`),
# and this front end decodes OpenCL/Kernel modules. The PTX front end's `tld4` is the
# only gather there is to translate, and it has a column above.
# The 64-bit counter here reads nothing back — `%old64` is written and never used —
# and an atomic nobody reads the result of is a reduction, so it takes the form
# that exists at 64 bits: two 32-bit atomics on the halves. That is what makes
# this a column rather than a refusal.
spvgpu atomic       atomic    --grid 8 --threads 8 --buffers 8,8,8
# A 64-bit atomic whose old value *is* read back has one writer at a time, so the
# module takes the warp target, which walks every warp in one thread. All six
# counters agree with the CPU byte for byte, compared as integers: a lost increment
# is 7.7e-43 when eight bytes of counter are read as two floats.
spvgpu atom64k      atom64    --grid 4096 --threads 32 --buffers 1,1,1,1,1,1
# The pair read and written as one value with nothing else in the module: a 64-bit
# atomic load and a 64-bit atomic store. Both were refused on this host until the
# warp target could carry them, and they take the same one-thread walk.
spvgpu atom64rw     atom64rw  --grid 64 --threads 32 --buffers 1,1
# A loop around the pair: an entry block, a loop block carrying two OpPhi values,
# and an exit block. This is the shape the refusal used to name — the warp target
# took one block only, so a loop could not reach it and the pair fell through to the
# per-lane back end, which cannot host a contending section. Every lane adds four to
# the counter and writes the sum of its four old values to its own slot.
spvgpu atom64loop  atom64loop --grid 64 --threads 32 --buffers 1,64
# A barrier between the increment and a 64-bit load, which is the shape a
# threadgroup stage is written in. The warp target runs the lanes in step for this:
# one program counter each, lowest first, so a lane parked on the barrier cannot be
# passed by a lane that has not arrived. All 32 lanes of the one warp must therefore
# read the same counter, and every slot must hold it.
spvgpu atom64bar   atom64bar  --grid 32 --threads 32 --buffers 1,32
# The rendezvous family in one module: a ballot of the even lanes, a 64-bit
# broadcast from lane 0, a barrier, and the pair. One thread owns the lanes, so the
# vote is a relay through the warp's slot array and the sweep brings the lanes back
# in order to read it — no hardware ballot is involved and none is available. Two
# of those votes are `all-equal`, over a 64-bit value and over a `ulong2`, each one
# that has to be true and one that has to be false, and each is written as a residual
# so a wrong answer is a nonzero slot.
spvgpu atom64vote  atom64vote --grid 32 --threads 32 --buffers 1,32,32,32,32,32,32
# The OpenCL.std imports rather than the native instructions — what LLVM's SPIR-V
# backend actually emits: `sub_group_ballot`, `sub_group_reduce_add`, a 64-bit
# `sub_group_broadcast`, and a reduce and a scan over 64-bit values, beside a
# barrier and the pair. The answers are worked out by hand in the module: 0x55555555,
# 496, 528, the prefix (l+1)(l+2)/2, and the counter in every slot.
spvgpu atom64sub   atom64sub  --grid 32 --threads 32 --buffers 1,32,32,32,32,32,32,32
# A `double` on a GPU with no fp64. Metal rejects the type, so the runtime carries it
# in software — exact IEEE-754 over the 64-bit integers that do exist — and this row
# is the proof: the cpu column computes with the host's own doubles and the two are
# compared bit for bit, over multiply, add, divide, a catastrophic subtraction, the
# four roundings, a trip through binary32, a comparison, a cast and a subgroup
# reduction. The reduction's residual buffer must be zero in every lane.
spvgpu fp64       fp64       --grid 32 --threads 32 --buffers 32,320,64,32
# The same software double, two components at a time: a `double2` built, added,
# multiplied, divided, subtracted and taken apart. The per-component path has to
# call the fp64 helper rather than apply the operator to the two 64-bit words, and
# an operator there adds significands as integers — a wrong number, not a refusal.
spvgpu dvec       dvec       --grid 32 --threads 32 --buffers 32,128
# A helper function: the entry point calls a defined function twice, so the module has
# two functions rather than one and the call is inlined where it is made. The second
# call goes through the same helper ids again — the case a one-call-site inliner gets
# wrong — and the values are hand-checkable: twice(x) = 2x + 1.
spvgpu helper     helper     --grid 32 --threads 32 --buffers 32,64
# A matrix that is not square: 2 rows by 3 columns, its 3x2 transpose, and a vector
# each way. A square shape cannot tell a row from a column — they are the same size —
# so the oblong ones are what pin the layout.
spvgpu matns      matns      --grid 32 --threads 32 --buffers 32,128
# The fold in floats: reduce add/min/max and an inclusive and an exclusive scan,
# over half-integers so every partial sum is exact and every answer can be worked
# out by hand in the module. The residual slots catch a wrong shift (an exclusive
# scan that folded its own lane in), a mask that counted a lane which never ran,
# and any disagreement between the two engines' trees. Sixteen lanes of the last
# warp are dead in the `--grid 32` case, which is the point of the mask.
spvgpu subfold    subfold    --grid 32 --threads 32 --buffers 256
# The same fold over a short warp, which is where a fold that reads a lane outside
# the grid goes wrong: only 24 of the 32 lanes ran, so the ones past the grid hold
# nothing and a tree that reads them answers with whatever was there. The hand
# numbers no longer fit, so this row compares the two engines rather than a
# residual — that comparison is what caught the 82-off sum.
spvgpu subfold:ragged=subfold subfold --grid 24 --threads 32 --buffers 256
# The kernel row proves the path end to end; this one proves each operation on its
# own, over a value set per operation — every rounding, every comparison against
# both signs and zero, subnormals, the extremes of the exponent range, and the
# conversions in both directions, including the ones the specification leaves
# undefined and which are therefore not compared. The reference is this machine's
# own binary64, and the comparison is of the bits.
if fp64_out="$(python3 tools/f64_check.py 2>&1)"; then
  echo "ok    fp64:exact  $(printf '%s\n' "$fp64_out" | tail -1)"
else
  echo "FAIL  fp64:exact"
  printf '%s\n' "$fp64_out" | tail -12 | sed 's/^/      /'
  fail=1
fi

triton add    add_kernel    2 --grid 1024 --buffers 1024,1024,1024 --arg 0=1000
triton matmul matmul_kernel 2 --grid 256  --buffers 2048,1024,1024
# fp32 `tl.dot` is the tf32 instruction: the operands are reduced to 10 mantissa
# bits, and the twin reduces them the same way rather than computing in f32.
triton dot32  dot32_kernel  2 --grid 256  --buffers 1024,512,1024
# bf16 `tl.dot`: the same fragments, 8 mantissa bits, and the twin does the
# rounding itself rather than being handed the converted values.
triton dotbf16 dotbf16_kernel 2 --grid 256 --buffers 2048,1024,1024
# `redux.sync`, the warp fold Triton emits for a `tl.sum`/`tl.max` over integers:
# six folds, each checked against a shuffle tree of the same shape.
triton reduce reduce_kernel 1 --grid 1024 --buffers 1024,8192
# fp8 `tl.dot` is `mma.sync.m16n8k32`: four bytes a register and twice the K of
# the half form. Its reference is not a Metal twin but `kernels/triton/fp8_ref.py`,
# an e4m3 written from the format's definition (1-4-3, saturated at 448,
# ties-to-even in exact integer arithmetic), and the tile holds exact binary
# fractions, so every product and every sum of 32 is exact: a difference is a
# fragment-layout or conversion bug, never a rounding one.
if python3 kernels/triton/fp8_ref.py gen 512 256 "$TMP/f8a.bin" "$TMP/f8b.bin" 2>"$TMP/f8.log" \
   && "$WEFT" run kernels/triton/dotfp8.ptx --kernel dotfp8_kernel --cpu-only \
        --grid 1 --threads 32 --buffers 512,256,128 --buffer-bytes 0=1,1=1 \
        --load 0="$TMP/f8a.bin" --load 1="$TMP/f8b.bin" --dump 2="$TMP/f8d.bin" >>"$TMP/f8.log" 2>&1 \
   && python3 kernels/triton/fp8_ref.py matmul "$TMP/f8a.bin" "$TMP/f8b.bin" "$TMP/f8d.bin" >>"$TMP/f8.log" 2>&1; then
  echo "ok    triton:dotfp8  m16n8k32 e4m3 matches the e4m3 reference exactly"
else
  echo "FAIL  triton:dotfp8"; sed 's/^/      /' "$TMP/f8.log"; fail=1
fi

# wgmma is a warpgroup (128-lane) instruction whose operands come from shared
# memory through a descriptor — so no lane reads another lane's register, and it
# lowers on the per-lane backend where the threadgroup is the warpgroup. The
# reference multiplies the input tiles directly, which is independent of both the
# descriptor decode and the core-matrix walk the kernel itself uses.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wa.bin" "$TMP/wb.bin" 2>"$TMP/wg.log" \
   && "$WEFT" run kernels/triton/wgmma.ptx --kernel wgmma_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wa.bin" --load 1="$TMP/wb.bin" --dump 2="$TMP/wd.bin" >>"$TMP/wg.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wa.bin" "$TMP/wb.bin" "$TMP/wd.bin" >>"$TMP/wg.log" 2>&1; then
  echo "ok    triton:wgmma  m64n8k16 f16 matches the reference exactly"
else
  echo "FAIL  triton:wgmma"; sed 's/^/      /' "$TMP/wg.log"; fail=1
fi

# The same instruction with bf16 operands. The element is 16 bits either way, so
# the core-matrix layout is the one above and only the decode changes — and the
# reference can tell the two apart, because the same bits are different numbers
# in f16 and in bf16.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wab.bin" "$TMP/wbb.bin" bf16 2>"$TMP/wgb.log" \
   && "$WEFT" run kernels/triton/wgmma_bf16.ptx --kernel wgmma_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wab.bin" --load 1="$TMP/wbb.bin" --dump 2="$TMP/wdb.bin" >>"$TMP/wgb.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wab.bin" "$TMP/wbb.bin" "$TMP/wdb.bin" bf16 >>"$TMP/wgb.log" 2>&1; then
  echo "ok    triton:wgmma_bf16  m64n8k16 bf16 matches the reference exactly"
else
  echo "FAIL  triton:wgmma_bf16"; sed 's/^/      /' "$TMP/wgb.log"; fail=1
fi

# tf32: the same instruction and descriptor, one element size up. A 16-byte
# core-matrix row holds four 32-bit values instead of eight halves, so K is 8 and
# B's core matrix is four columns wide — the two places the geometry could have
# been written into the decode rather than derived from the element size.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wa32.bin" "$TMP/wb32.bin" tf32 2>"$TMP/wg32.log" \
   && "$WEFT" run kernels/triton/wgmma_tf32.ptx --kernel wgmma_tf32_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 512,64,512 --buffer-bytes 0=4,1=4 \
        --load 0="$TMP/wa32.bin" --load 1="$TMP/wb32.bin" --dump 2="$TMP/wd32.bin" >>"$TMP/wg32.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wa32.bin" "$TMP/wb32.bin" "$TMP/wd32.bin" tf32 >>"$TMP/wg32.log" 2>&1; then
  echo "ok    triton:wgmma_tf32  m64n8k8 tf32 matches the reference exactly"
else
  echo "FAIL  triton:wgmma_tf32"; sed 's/^/      /' "$TMP/wg32.log"; fail=1
fi

# fp8: the widest geometry and the two formats. A one-byte element puts sixteen
# to a core-matrix row, so K is 32 and B's whole 8 columns fit inside one core.
# The two formats share that layout exactly and differ only in decode, so the e5m2
# sample is the e4m3 one with the format swapped — the same instruction stream a
# compiler emits for the other format, and a reference that can tell them apart.
for f8 in e4m3 e5m2; do
  if [ "$f8" = e5m2 ]; then
    sed 's/\.e4m3\.e4m3/.e5m2.e5m2/g' kernels/triton/wgmma_fp8.ptx > "$TMP/wg_$f8.ptx"
  else
    cp kernels/triton/wgmma_fp8.ptx "$TMP/wg_$f8.ptx"
  fi
  if python3 kernels/triton/wgmma_ref.py gen "$TMP/a8.bin" "$TMP/b8.bin" "$f8" 2>"$TMP/w8.log" \
     && "$WEFT" run "$TMP/wg_$f8.ptx" --kernel wgmma_fp8_kernel --cpu-only \
          --grid 128 --threads 128 --buffers 2048,256,512 --buffer-bytes 0=1,1=1 \
          --load 0="$TMP/a8.bin" --load 1="$TMP/b8.bin" --dump 2="$TMP/d8.bin" >>"$TMP/w8.log" 2>&1 \
     && python3 kernels/triton/wgmma_ref.py check "$TMP/a8.bin" "$TMP/b8.bin" "$TMP/d8.bin" "$f8" >>"$TMP/w8.log" 2>&1; then
    echo "ok    triton:wgmma_$f8  m64n8k32 $f8 matches the reference exactly"
  else
    echo "FAIL  triton:wgmma_$f8"; sed 's/^/      /' "$TMP/w8.log"; fail=1
  fi
done

# The integer twins: the same bytes read as signed and as unsigned integers. The
# layout is the fp8 one to the byte, so a disagreement here is a decode rather
# than a walk — and the reference reads the same bytes as different numbers, which
# is how a decode wired to the wrong signedness is caught.
for i8 in s8 u8; do
  if [ "$i8" = u8 ]; then
    sed 's/\.s8\.s8/.u8.u8/g' kernels/triton/wgmma_s8.ptx > "$TMP/wgi_$i8.ptx"
  else
    cp kernels/triton/wgmma_s8.ptx "$TMP/wgi_$i8.ptx"
  fi
  if python3 kernels/triton/wgmma_ref.py gen "$TMP/ai8.bin" "$TMP/bi8.bin" "$i8" 2>"$TMP/wgi.log" \
     && "$WEFT" run "$TMP/wgi_$i8.ptx" --kernel wgmma_s8_kernel --cpu-only \
          --grid 128 --threads 128 --buffers 2048,256,512 --buffer-bytes 0=1,1=1 \
          --load 0="$TMP/ai8.bin" --load 1="$TMP/bi8.bin" --dump 2="$TMP/di8.bin" >>"$TMP/wgi.log" 2>&1 \
     && python3 kernels/triton/wgmma_ref.py check "$TMP/ai8.bin" "$TMP/bi8.bin" "$TMP/di8.bin" "$i8" >>"$TMP/wgi.log" 2>&1; then
    echo "ok    triton:wgmma_$i8  m64n8k32 $i8 matches the reference exactly"
  else
    echo "FAIL  triton:wgmma_$i8"; sed 's/^/      /' "$TMP/wgi.log"; fail=1
  fi
done

# Sixteen columns, which is the tile that can tell the two descriptor offsets apart:
# with N = 8 the column index never leaves the first group of eight, so B's SBO term
# is multiplied by zero and a translator that put the K stride in the wrong field
# would still agree. It also needs two groups of accumulator registers a lane, which
# nothing else exercises.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wn16a.bin" "$TMP/wn16b.bin" f16n16 2>"$TMP/wn16.log" \
   && "$WEFT" run kernels/triton/wgmma_n16.ptx --kernel wgmma_n16_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,256,1024 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wn16a.bin" --load 1="$TMP/wn16b.bin" --dump 2="$TMP/wn16d.bin" >>"$TMP/wn16.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wn16a.bin" "$TMP/wn16b.bin" "$TMP/wn16d.bin" f16n16 >>"$TMP/wn16.log" 2>&1; then
  echo "ok    triton:wgmma_n16  m64n16k16 f16 matches the reference exactly"
else
  echo "FAIL  triton:wgmma_n16"; sed 's/^/      /' "$TMP/wn16.log"; fail=1
fi

# MN-major operands: the same logical tiles staged the other way up, so the same
# reference applies — and a walk that read the wrong direction inside the 16-byte
# chunk would multiply the wrong pairs rather than fail to compile.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wta.bin" "$TMP/wtb.bin" f16 2>"$TMP/wt.log" \
   && "$WEFT" run kernels/triton/wgmma_trans.ptx --kernel wgmma_trans_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wta.bin" --load 1="$TMP/wtb.bin" --dump 2="$TMP/wtd.bin" >>"$TMP/wt.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wta.bin" "$TMP/wtb.bin" "$TMP/wtd.bin" f16 >>"$TMP/wt.log" 2>&1; then
  echo "ok    triton:wgmma_trans  m64n8k16 with MN-major operands matches the reference"
else
  echo "FAIL  triton:wgmma_trans"; sed 's/^/      /' "$TMP/wt.log"; fail=1
fi

# scale-d: two issues into the same accumulator must give exactly twice the product.
# Every other fixture writes once, so a translator that overwrote instead of adding
# would agree with all of them.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/waa.bin" "$TMP/wab2.bin" f16acc 2>"$TMP/waa.log" \
   && "$WEFT" run kernels/triton/wgmma_acc.ptx --kernel wgmma_acc_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/waa.bin" --load 1="$TMP/wab2.bin" --dump 2="$TMP/wad.bin" >>"$TMP/waa.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/waa.bin" "$TMP/wab2.bin" "$TMP/wad.bin" f16acc >>"$TMP/waa.log" 2>&1; then
  echo "ok    triton:wgmma_acc  two issues with scale-d double the product, exactly"
else
  echo "FAIL  triton:wgmma_acc"; sed 's/^/      /' "$TMP/waa.log"; fail=1
fi

# The swizzled layout: the descriptor carries mode 1 in bits 62-63 and the tile
# rows are 128 bytes, so the 32 bytes of K an instruction reads are scattered
# across each row by the chunk/row XOR. A walk that ignored the mode, or applied
# the wrong width, reads the wrong halves and the product is wrong.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wza.bin" "$TMP/wzb.bin" f16 2>"$TMP/wz.log" \
   && "$WEFT" run kernels/triton/wgmma_swz128.ptx --kernel wgmma_swz128_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wza.bin" --load 1="$TMP/wzb.bin" --dump 2="$TMP/wzd.bin" >>"$TMP/wz.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wza.bin" "$TMP/wzb.bin" "$TMP/wzd.bin" f16 >>"$TMP/wz.log" 2>&1; then
  echo "ok    triton:wgmma_swz128  a 128-byte-swizzled tile matches the reference"
else
  echo "FAIL  triton:wgmma_swz128"; sed 's/^/      /' "$TMP/wz.log"; fail=1
fi

# The narrow swizzle is the same instruction with a different mode in the
# descriptor, and it separates the two masks: the XOR mask is one bit here and
# seven for the wide atom, while the mask that holds a byte inside an atom is 31
# either way — swapping them reads the wrong halves and only this fixture says so.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/w3a.bin" "$TMP/w3b.bin" f16 2>"$TMP/w3.log" \
   && "$WEFT" run kernels/triton/wgmma_swz32.ptx --kernel wgmma_swz32_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w3a.bin" --load 1="$TMP/w3b.bin" --dump 2="$TMP/w3d.bin" >>"$TMP/w3.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w3a.bin" "$TMP/w3b.bin" "$TMP/w3d.bin" f16 >>"$TMP/w3.log" 2>&1; then
  echo "ok    triton:wgmma_swz32  a 32-byte-swizzled tile matches the reference"
else
  echo "FAIL  triton:wgmma_swz32"; sed 's/^/      /' "$TMP/w3.log"; fail=1
fi

# The middle swizzle width: a 64-byte atom, whose row is twice what an
# instruction reads, so the pattern moves the second chunk into the other half.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/w4a.bin" "$TMP/w4b.bin" f16 2>"$TMP/w4.log" \
   && "$WEFT" run kernels/triton/wgmma_swz64.ptx --kernel wgmma_swz64_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w4a.bin" --load 1="$TMP/w4b.bin" --dump 2="$TMP/w4d.bin" >>"$TMP/w4.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w4a.bin" "$TMP/w4b.bin" "$TMP/w4d.bin" f16 >>"$TMP/w4.log" 2>&1; then
  echo "ok    triton:wgmma_swz64  a 64-byte-swizzled tile matches the reference"
else
  echo "FAIL  triton:wgmma_swz64"; sed 's/^/      /' "$TMP/w4.log"; fail=1
fi

# A swizzled tile that begins one row into the pattern. The descriptor's
# base-offset field (bits 49-51) is what says so; ignoring it puts every element
# one row out in the pattern, which is a wrong product rather than a crash.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/w5a.bin" "$TMP/w5b.bin" f16 2>"$TMP/w5.log" \
   && "$WEFT" run kernels/triton/wgmma_swzbase.ptx --kernel wgmma_swzbase_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w5a.bin" --load 1="$TMP/w5b.bin" --dump 2="$TMP/w5d.bin" >>"$TMP/w5.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w5a.bin" "$TMP/w5b.bin" "$TMP/w5d.bin" f16 >>"$TMP/w5.log" 2>&1; then
  echo "ok    triton:wgmma_swzbase  a tile three rows into the pattern matches the reference"
else
  echo "FAIL  triton:wgmma_swzbase"; sed 's/^/      /' "$TMP/w5.log"; fail=1
fi

# The swizzle on an MN-major operand: M is the contiguous direction, so the walk
# mirrors the K-major one — the byte's place inside an atom is its M offset and the
# K row is one atom wide. B stays K-major and both tiles are mode 1.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/w6a.bin" "$TMP/w6b.bin" f16 2>"$TMP/w6.log" \
   && "$WEFT" run kernels/triton/wgmma_swztrans.ptx --kernel wgmma_swztrans_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w6a.bin" --load 1="$TMP/w6b.bin" --dump 2="$TMP/w6d.bin" >>"$TMP/w6.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w6a.bin" "$TMP/w6b.bin" "$TMP/w6d.bin" f16 >>"$TMP/w6.log" 2>&1; then
  echo "ok    triton:wgmma_swztrans  a swizzled MN-major A matches the reference"
else
  echo "FAIL  triton:wgmma_swztrans"; sed 's/^/      /' "$TMP/w6.log"; fail=1
fi

# The swizzle does not care how wide an element is: it places bytes, and the width
# only decides how many of them a chunk of K holds. tf32 is the wide end — K is 8 and
# a row of the tile is a quarter of the atom — and an atom measured in elements
# rather than bytes lands every row on the wrong half.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/w7a.bin" "$TMP/w7b.bin" tf32 2>"$TMP/w7.log" \
   && "$WEFT" run kernels/triton/wgmma_swztf32.ptx --kernel wgmma_swztf32_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 512,64,512 --buffer-bytes 0=4,1=4 \
        --load 0="$TMP/w7a.bin" --load 1="$TMP/w7b.bin" --dump 2="$TMP/w7d.bin" >>"$TMP/w7.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w7a.bin" "$TMP/w7b.bin" "$TMP/w7d.bin" tf32 >>"$TMP/w7.log" 2>&1; then
  echo "ok    triton:wgmma_swztf32  a swizzled tf32 tile matches the reference"
else
  echo "FAIL  triton:wgmma_swztf32"; sed 's/^/      /' "$TMP/w7.log"; fail=1
fi

# The narrow end: one byte an element, so K is 32 and the byte column runs across
# the atom's chunks — the phase and the column overlap, which no wider element does.
# e5m2 is the same geometry, so the same fixture with the format swapped.
for f8 in e4m3 e5m2; do
  if [ "$f8" = e5m2 ]; then
    sed 's/\.e4m3\.e4m3/.e5m2.e5m2/g' kernels/triton/wgmma_swzf8.ptx > "$TMP/wsw_$f8.ptx"
  else
    cp kernels/triton/wgmma_swzf8.ptx "$TMP/wsw_$f8.ptx"
  fi
  if python3 kernels/triton/wgmma_ref.py gen "$TMP/w9a.bin" "$TMP/w9b.bin" "$f8" 2>"$TMP/w9.log" \
     && "$WEFT" run "$TMP/wsw_$f8.ptx" --kernel wgmma_swzf8_kernel --cpu-only \
          --grid 128 --threads 128 --buffers 2048,256,512 --buffer-bytes 0=1,1=1 \
          --load 0="$TMP/w9a.bin" --load 1="$TMP/w9b.bin" --dump 2="$TMP/w9d.bin" >>"$TMP/w9.log" 2>&1 \
     && python3 kernels/triton/wgmma_ref.py check "$TMP/w9a.bin" "$TMP/w9b.bin" "$TMP/w9d.bin" "$f8" >>"$TMP/w9.log" 2>&1; then
    echo "ok    triton:wgmma_swz$f8  a swizzled $f8 tile matches the reference"
  else
    echo "FAIL  triton:wgmma_swz$f8"; sed 's/^/      /' "$TMP/w9.log"; fail=1
  fi
done

# The other transpose: B is MN-major and A stays K-major, both swizzled. A walk
# that gave the two operands each other's contiguous direction reads the wrong
# operand here, and the transposed-A fixture alone cannot say which of the two is
# right.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wba.bin" "$TMP/wbb.bin" f16 2>"$TMP/wb.log" \
   && "$WEFT" run kernels/triton/wgmma_swztransb.ptx --kernel wgmma_swztransb_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wba.bin" --load 1="$TMP/wbb.bin" --dump 2="$TMP/wbd.bin" >>"$TMP/wb.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wba.bin" "$TMP/wbb.bin" "$TMP/wbd.bin" f16 >>"$TMP/wb.log" 2>&1; then
  echo "ok    triton:wgmma_swztransb  a swizzled MN-major B matches the reference"
else
  echo "FAIL  triton:wgmma_swztransb"; sed 's/^/      /' "$TMP/wb.log"; fail=1
fi

# Three issues into one accumulator: what a K loop looks like. Two issues already
# pin scale-d, but an accumulator read once and then dropped still passes those;
# three is where it has to carry every time, so the answer is three products.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wca.bin" "$TMP/wcb.bin" f16chain 2>"$TMP/wc.log" \
   && "$WEFT" run kernels/triton/wgmma_chain.ptx --kernel wgmma_chain_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wca.bin" --load 1="$TMP/wcb.bin" --dump 2="$TMP/wcd.bin" >>"$TMP/wc.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wca.bin" "$TMP/wcb.bin" "$TMP/wcd.bin" f16chain >>"$TMP/wc.log" 2>&1; then
  echo "ok    triton:wgmma_chain  three issues tripled the product, exactly"
else
  echo "FAIL  triton:wgmma_chain"; sed 's/^/      /' "$TMP/wc.log"; fail=1
fi

# The integer accumulator: the same bytes as the s8 fixture, summed in s32, so the
# answer is an integer and the reference reads it as one. The signedness pair comes
# for free, the file with `.u8.u8` swapped in.
for i8 in s8 u8; do
  if [ "$i8" = u8 ]; then
    sed 's/\.s8\.s8/.u8.u8/g' kernels/triton/wgmma_s32acc.ptx > "$TMP/w32acc_$i8.ptx"
  else
    cp kernels/triton/wgmma_s32acc.ptx "$TMP/w32acc_$i8.ptx"
  fi
  if python3 kernels/triton/wgmma_ref.py gen "$TMP/w3a.bin" "$TMP/w3b.bin" "${i8}acc" 2>"$TMP/w3acc.log" \
     && "$WEFT" run "$TMP/w32acc_$i8.ptx" --kernel wgmma_s32acc_kernel --cpu-only \
          --grid 128 --threads 128 --buffers 2048,256,512 --buffer-bytes 0=1,1=1 \
          --load 0="$TMP/w3a.bin" --load 1="$TMP/w3b.bin" --dump 2="$TMP/w3d.bin" >>"$TMP/w3acc.log" 2>&1 \
     && python3 kernels/triton/wgmma_ref.py check "$TMP/w3a.bin" "$TMP/w3b.bin" "$TMP/w3d.bin" "${i8}acc" >>"$TMP/w3acc.log" 2>&1; then
    echo "ok    triton:wgmma_${i8}acc  a ${i8} tile summed in s32 matches the reference"
  else
    echo "FAIL  triton:wgmma_${i8}acc"; sed 's/^/      /' "$TMP/w3acc.log"; fail=1
  fi
done

# A short immediate list. `1, 1` is scale-d then scale-a: accumulate, and negate A.
# The form was refused as unreadable and is legal PTX; a translator that read it but
# took every flag as zero returns the product instead of its negative.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wia.bin" "$TMP/wib.bin" f16neg 2>"$TMP/wi.log" \
   && "$WEFT" run kernels/triton/wgmma_shortimm.ptx --kernel wgmma_shortimm_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wia.bin" --load 1="$TMP/wib.bin" --dump 2="$TMP/wid.bin" >>"$TMP/wi.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wia.bin" "$TMP/wib.bin" "$TMP/wid.bin" f16neg >>"$TMP/wi.log" 2>&1; then
  echo "ok    triton:wgmma_shortimm  two immediates negate A, exactly"
else
  echo "FAIL  triton:wgmma_shortimm"; sed 's/^/      /' "$TMP/wi.log"; fail=1
fi

# wgmma on the 32-lane backend. The kernel carries a warp fold, so it cannot run on
# the per-lane machine at all; one thread there owns a warp of the warpgroup and a
# lane is `ctx.lid0 + l`. The D tile has to come out the same either way.
if python3 kernels/triton/wgmma_ref.py gen "$TMP/wwa.bin" "$TMP/wwb.bin" f16 2>"$TMP/ww.log" \
   && "$WEFT" run kernels/triton/wgmma_warp.ptx --kernel wgmma_warp_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wwa.bin" --load 1="$TMP/wwb.bin" --dump 2="$TMP/wwd.bin" >>"$TMP/ww.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wwa.bin" "$TMP/wwb.bin" "$TMP/wwd.bin" f16 >>"$TMP/ww.log" 2>&1; then
  echo "ok    triton:wgmma_warp  a wgmma on the warp backend matches the reference"
else
  echo "FAIL  triton:wgmma_warp"; sed 's/^/      /' "$TMP/ww.log"; fail=1
fi

# e5m2 is the other fp8 format and the same instruction: 5 exponent bits, 2
# mantissa bits, infinities and NaN at the top exponent. Same layout, different
# decode — so this catches a decoder wired to the wrong format.
if python3 kernels/triton/fp8_ref.py gen 512 256 "$TMP/e5a.bin" "$TMP/e5b.bin" e5m2 2>"$TMP/e5.log" \
   && "$WEFT" run kernels/triton/dote5m2.ptx --kernel dote5m2_kernel --cpu-only \
        --grid 1 --threads 32 --buffers 512,256,128 --buffer-bytes 0=1,1=1 \
        --load 0="$TMP/e5a.bin" --load 1="$TMP/e5b.bin" --dump 2="$TMP/e5d.bin" >>"$TMP/e5.log" 2>&1 \
   && python3 kernels/triton/fp8_ref.py matmul "$TMP/e5a.bin" "$TMP/e5b.bin" "$TMP/e5d.bin" e5m2 >>"$TMP/e5.log" 2>&1; then
  echo "ok    triton:dote5m2  m16n8k32 e5m2 matches the e5m2 reference exactly"
else
  echo "FAIL  triton:dote5m2"; sed 's/^/      /' "$TMP/e5.log"; fail=1
fi

# `cvt.rn.satfinite.e4m3x2.f32` + `prmt`, the sequence that builds an fp8 operand:
# checked byte for byte on values that exercise saturation, subnormals and both
# kinds of rounding tie.
if python3 kernels/triton/fp8_ref.py gen-cvt "$TMP/f8in.bin" 2>"$TMP/f8c.log" \
   && "$WEFT" run kernels/triton/cvt8.ptx --kernel cvt8_kernel --cpu-only \
        --grid 32 --threads 32 --buffers 128,32 \
        --load 0="$TMP/f8in.bin" --dump 1="$TMP/f8out.bin" >>"$TMP/f8c.log" 2>&1 \
   && python3 kernels/triton/fp8_ref.py cvt "$TMP/f8in.bin" "$TMP/f8out.bin" >>"$TMP/f8c.log" 2>&1; then
  echo "ok    triton:cvt8  packing to e4m3 matches the reference on every branch"
else
  echo "FAIL  triton:cvt8"; sed 's/^/      /' "$TMP/f8c.log"; fail=1
fi

# The geometry Triton compiled for is in the module, not in the caller's head:
# `.maxntid 128` is the launch size, and the `global_scratch` pointer the kernel
# never reads is not one of the buffers it is launched with.
meta=$("$WEFT" run kernels/triton/add.ptx --kernel add_kernel --grid 1024 \
    --buffers 1024,1024,1024 --arg 0=1000 --cpu-only)
if echo "$meta" | grep -q 'threadgroup 128' && echo "$meta" | grep -q '1 declared parameter is not bound'; then
  echo "ok    triton:abi  threads default to .maxntid (128), unread scratch not bound"
else
  echo "FAIL  triton:abi"; echo "$meta" | sed 's/^/      /'; fail=1
fi

# A barrier inside a loop, from real codegen — this is the resumable path.
"$WEFT" run kernels/basic.metal --kernel reduce_sum --cpu-only --grid 1048576 --threads 256 --buffers 1048576,4096 --dump 1="$TMP/red_ref.bin" >/dev/null
"$WEFT" run kernels/cuda/reduce.ptx --kernel reduce --cpu-only --grid 1048576 --threads 256 --buffers 1048576,4096 --dump 1="$TMP/red_ptx.bin" >/dev/null
if cmp -s "$TMP/red_ref.bin" "$TMP/red_ptx.bin"; then
  echo "ok    nvptx:reduce   barrier inside a loop matches the metal reduction"
else
  echo "FAIL  nvptx:reduce"; fail=1
fi

# Scalar kernel parameters and a v4 vector load, both from real codegen.
"$WEFT" run kernels/basic.metal --kernel scale --cpu-only --grid 1048576 --buffers 1048576,1048576 --dump 1="$TMP/scale_ref.bin" >/dev/null
"$WEFT" run kernels/cuda/scale.ptx --kernel scale --cpu-only --grid 1048576 --buffers 1048576,1048576 --arg 0=1.5 --arg 1=1048576 --dump 1="$TMP/scale_ptx.bin" >/dev/null
if cmp -s "$TMP/scale_ref.bin" "$TMP/scale_ptx.bin"; then
  echo "ok    nvptx:scale   scalar parameters and a guarded body match"
else
  echo "FAIL  nvptx:scale"; fail=1
fi

"$WEFT" run kernels/cuda/vecscale.ptx --kernel vecscale --cpu-only --grid 262144 --buffers 1048576,1048576 --arg 0=1.5 --dump 1="$TMP/vec.bin" >/dev/null
if cmp -s "$TMP/scale_ref.bin" "$TMP/vec.bin"; then
  echo "ok    nvptx:vecscale  v4 loads and stores match"
else
  echo "FAIL  nvptx:vecscale"; fail=1
fi

# Bit manipulation: funnel shifts, high-word multiplies, a byte permute, a NaN
# test and the rounding converts. The PTX side is LLVM's own output; the Metal
# side computes the same numbers from the definitions. Neither is the oracle for
# the other, so a match means both are right.
nvptx bits 1 --grid 1048576 --buffers 1048576,1048576

"$WEFT" run kernels/basic.metal --kernel roundk --cpu-only --grid 1048576 \
    --buffers 1048576,1048576 --dump 1="$TMP/round_msl.bin" >/dev/null
"$WEFT" run kernels/round.ptx --kernel roundk --cpu-only --grid 1048576 \
    --buffers 1048576,1048576 --dump 1="$TMP/round_ptx.bin" >/dev/null
if cmp -s "$TMP/round_msl.bin" "$TMP/round_ptx.bin"; then
  echo "ok    ptx:roundk    cvt.rn/.rz/.rm/.rp agree with rint/trunc/floor/ceil"
else
  echo "FAIL  ptx:roundk    rounding modes differ from the metal kernel"; fail=1
fi

# lop3's truth tables and the eight testp classes, against a bit-pattern oracle
# written in the Metal front end. Metal's own isnan/isfinite are unusable here:
# it compiles with fast-math, which is free to fold them away.
"$WEFT" run kernels/basic.metal --kernel logic --cpu-only --grid 1048576 \
    --buffers 1048576,1048576 --dump 1="$TMP/logic_msl.bin" >/dev/null
"$WEFT" run kernels/logic.ptx --kernel logic --cpu-only --grid 1048576 \
    --buffers 1048576,1048576 --dump 1="$TMP/logic_ptx.bin" >/dev/null
if cmp -s "$TMP/logic_msl.bin" "$TMP/logic_ptx.bin"; then
  echo "ok    ptx:logic     lop3 tables and all eight testp classes agree"
else
  echo "FAIL  ptx:logic     differs from the metal kernel"; fail=1
fi

# Atomics, checked against arithmetic rather than against each other: both
# counters rise by exactly the grid, and the two maxes are set to a constant
# larger than anything already in the buffer. A zero grid gives the starting
# value, since the buffers are not zero-filled.
atomcheck() {
  local backend="$1" prefix=""
  [ "$backend" = warp ] && prefix="WEFT_WARP=1"
  local n=262144
  env $prefix "$WEFT" run kernels/atom.ptx --kernel atomk --cpu-only --grid 0 \
      --buffers 8,8,8,8 --dump 0="$TMP/at_base0.bin" --dump 2="$TMP/at_base2.bin" >/dev/null
  env $prefix "$WEFT" run kernels/atom.ptx --kernel atomk --cpu-only --grid "$n" \
      --buffers 8,8,8,8 --bench 1 --dump 0="$TMP/at_out0.bin" --dump 1="$TMP/at_out1.bin" \
      --dump 2="$TMP/at_out2.bin" --dump 3="$TMP/at_out3.bin" >/dev/null
  local b32 o32 b64 o64 m32 m64l m64h
  b32=$(od -An -tu4 -N4 "$TMP/at_base0.bin" | tr -d ' ')
  o32=$(od -An -tu4 -N4 "$TMP/at_out0.bin"  | tr -d ' ')
  b64=$(od -An -tu4 -N4 "$TMP/at_base2.bin" | tr -d ' ')
  o64=$(od -An -tu4 -N4 "$TMP/at_out2.bin"  | tr -d ' ')
  m32=$(od -An -tx4 -N4 "$TMP/at_out1.bin"  | tr -d ' ')
  m64l=$(od -An -tx4 -j0 -N4 "$TMP/at_out3.bin" | tr -d ' ')
  m64h=$(od -An -tx4 -j4 -N4 "$TMP/at_out3.bin" | tr -d ' ')
  if [ "$o32" -eq "$((b32 + n))" ] && [ "$o64" -eq "$((b64 + n))" ] \
     && [ "$m32" = "7fffffff" ] && [ "$m64l" = "ffffffff" ] && [ "$m64h" = "7fffffff" ]; then
    echo "ok    atom:$backend  u32/u64 counters +262144, both maxes exact"
  else
    echo "FAIL  atom:$backend  counters $b32->$o32 $b64->$o64, maxes $m32 $m64h$m64l"; fail=1
  fi
}

atomcheck lanes
atomcheck warp

# bar.red — a threadgroup-wide fold, checked against arithmetic. Each group's
# record depends only on how many threads it actually has, so a grid whose last
# group is short must record a different answer, and no group may see another
# group's slots.
bredcheck() {
  local backend="$1" prefix="" bad=""
  [ "$backend" = warp ] && prefix="WEFT_WARP=1"
  for cfg in "256 256" "100 256" "1024 256" "300 96" "1000 96" "512 128"; do
    local grid="${cfg% *}" tg="${cfg#* }"
    env $prefix "$WEFT" run kernels/bred.ptx --kernel bred --cpu-only --grid "$grid" \
        --threads "$tg" --buffers 4,64 --dump 1="$TMP/bred.bin" >/dev/null
    local ngroups=$(((grid + tg - 1) / tg)) g=0
    while [ "$g" -lt "$ngroups" ]; do
      local live=$((grid - g * tg))
      [ "$live" -gt "$tg" ] && live="$tg"
      local pop=$((live < 100 ? live : 100))
      local and=$((live <= 200 ? 1 : 0))
      local loop=$(((live < 32 ? live : 32) + (live < 16 ? live : 16)))
      local got
      got=$(od -An -tu4 -j $((16 * g)) -N16 "$TMP/bred.bin" | tr -s ' ' | sed 's/^ //; s/ $//')
      [ "$got" = "$pop $and 1 $loop" ] || bad="$bad  $grid/$tg group $g got '$got' want '$pop $and 1 $loop'"
      g=$((g + 1))
    done
  done
  if [ -z "$bad" ]; then
    echo "ok    bred:$backend  bar.red popc/and/or exact, incl. short threadgroups"
  else
    echo "FAIL  bred:$backend $bad"; fail=1
  fi
}

bredcheck lanes
bredcheck warp

# Warp votes, checked against the answer rather than against each other: the
# ballot of the even lanes is 0x55555555, and all(lane<32) + uni(lane<32) +
# any(lane==7) is 3. Both backends, since they compute it completely differently.
for backend in lanes warp; do
  if [ "$backend" = warp ]; then export WEFT_WARP=1; else unset WEFT_WARP; fi
  "$WEFT" run kernels/cuda/vote.ptx --kernel vote --cpu-only --grid 4096 --threads 256 \
      --buffers 4096,4096 --dump 0="$TMP/vb.bin" --dump 1="$TMP/vc.bin" >/dev/null
  bal="$(od -An -tx4 -v "$TMP/vb.bin" | tr -s ' ' '\n' | grep -v '^$' | sort -u | tr -d '\n')"
  cnt="$(od -An -tu4 -v "$TMP/vc.bin" | tr -s ' ' '\n' | grep -v '^$' | sort -u | tr -d '\n')"
  if [ "$bal" = "55555555" ] && [ "$cnt" = "3" ]; then
    echo "ok    vote:$backend  ballot 0x55555555, all/uni/any all hold"
  else
    echo "FAIL  vote:$backend  ballot=$bal count=$cnt"; fail=1
  fi
done
unset WEFT_WARP

# SPIR-V front end vs the Metal front end, byte for byte. The modules are
# assembled from text by spirv-as and validated, so a mistake in the module
# cannot be mistaken for a mistake in the translator.
spvcross() {
  local kernel="$1" file="$2" idx="$3"; shift 3
  "$WEFT" run kernels/basic.metal --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/msl.bin" "$@" >/dev/null
  "$WEFT" run "kernels/spirv/$file.spv" --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/spv.bin" "$@" >/dev/null
  if same_dumps "spirv:$kernel" "$TMP/msl.bin" "$TMP/spv.bin"; then
    echo "ok    spirv:$kernel  matches the metal front end byte for byte"
  else
    echo "FAIL  spirv:$kernel  differs from the metal front end"; fail=1
  fi
}

spvcross saxpy        saxpy       2 --grid 1048576 --buffers 1048576,1048576,1048576
spvcross warpscan     warpscan    1 --grid 1048576 --threads 256 --buffers 1048576,1048576
# Loops with OpPhi, threadgroup memory and a barrier.
spvcross block_reduce blockreduce 1 --grid 1048576 --threads 256 --buffers 1048576,4096
spvcross block_reduce blockreduce 1 --grid 1000    --threads 96  --buffers 1000,11
spvcross votek        vote        1 --grid 1048576 --threads 256 --buffers 1048576,1048576
# Values wider than the 32-bit rendezvous: a float2 and a 64-bit integer.
spvcross wide_shfl    wide        1 --grid 1048576 --threads 256 --buffers 1048576,1048576
# The one module LLVM's own SPIR-V backend emitted, not written by hand.
spvcross serial_sum   llvm        1 --grid 1 --threads 1 --buffers 1000,1 --arg 0=1000
# Images and samplers. In SPIR-V the sampler is host state (--samplers), while the
# Metal twin bakes a constexpr sampler into the kernel; both must land on the same
# texel, so every one of the 65536 outputs has to match.
spvcross texmix       texture     0 --grid 65536 --buffers 65536 --textures 64x64 --samplers nearest/clamp
# A module-scope `__global` buffer, which is bound as the parameter after the
# explicit arguments rather than passed as a pointer.
spvcross globk        global      1 --grid 1048576 --buffers 1048576,1048576
# A storage image, written and read back across lanes: `OpImageWrite` in SPIR-V,
# `texture2d<..., access::read_write>` in Metal.
spvcross texrw        texturerg   1 --grid 4096 --threads 256 --buffers 4096,4096 --textures 64x64
# The array form of both: the SPIR-V module's layered read and sample against
# the Metal kernel's, on the same four slices.
spvcross texarray     arraytex    0 --grid 65536 --buffers 65536 --textures '64x64*4' --samplers nearest/clamp
# The volume, both front ends, on the same eight planes.
spvcross tex3d        threed      0 --grid 65536 --buffers 65536 --textures 64x64x8 --samplers nearest/clamp

# SPIR-V atomics, checked against arithmetic: each counter rises by exactly the
# grid, from a zero-thread baseline.
spvatom() {
  local n=262144
  "$WEFT" run kernels/spirv/atomic.spv --kernel atomic --cpu-only --grid 0 --buffers 8,8,8 \
      --dump 1="$TMP/sa_b1.bin" --dump 2="$TMP/sa_b2.bin" >/dev/null
  "$WEFT" run kernels/spirv/atomic.spv --kernel atomic --cpu-only --grid "$n" --buffers 8,8,8 \
      --bench 1 --dump 1="$TMP/sa_o1.bin" --dump 2="$TMP/sa_o2.bin" >/dev/null
  local b32 o32 b64 o64
  b32=$(od -An -tu4 -N4 "$TMP/sa_b1.bin" | tr -d ' ')
  o32=$(od -An -tu4 -N4 "$TMP/sa_o1.bin" | tr -d ' ')
  b64=$(od -An -tu4 -N4 "$TMP/sa_b2.bin" | tr -d ' ')
  o64=$(od -An -tu4 -N4 "$TMP/sa_o2.bin" | tr -d ' ')
  if [ "$o32" -eq "$((b32 + n))" ] && [ "$o64" -eq "$((b64 + n))" ]; then
    echo "ok    spirv:atomic  u32 and u64 counters +$n exactly"
  else
    echo "FAIL  spirv:atomic  counters $b32->$o32 $b64->$o64"; fail=1
  fi
}
spvatom

# The same pair, but read and written as a whole value: a 64-bit atomic load and a
# 64-bit atomic store, checked against arithmetic rather than against a twin
# because no Metal kernel does this. Every lane reads A and writes A+1 to B, so B
# ends at A+1 from its baseline. The module is named by path here so the inventory
# at the end of this file can see that it is compared.
# The answer is one, so it is checked as bytes: the dump's element width is the
# launcher's business and a byte view does not depend on it. The module takes
# `A - A + 1`, so the answer does not depend on what the buffer held either.
"$WEFT" run kernels/spirv/atom64rw.spv --kernel atom64rw --cpu-only --grid 64 --threads 32 \
    --buffers 1,1 --dump 1="$TMP/rw1.bin" >/dev/null
rw_hex=$(od -An -tx1 -N8 "$TMP/rw1.bin" | tr -d ' \n')
case "$rw_hex" in
  0100000000000000|01000000)
    echo "ok    spirv:atom64rw  64-bit atomic load and store: B = A + 1" ;;
  *)
    echo "FAIL  spirv:atom64rw  B reads $rw_hex, wanted the 64-bit one"; fail=1 ;;
esac

# 64-bit atomics whose old value is read back, checked against arithmetic: the
# olds of `+= 1` are a permutation of 0..n-1, the olds of `max 1` on a counter that
# starts at 0 are one 0 and n-1 ones, and the signed `min -1` takes the counter to
# -1 with the olds summing to -(n-1) — unsigned ordering would leave both at 0,
# which is what makes it the signed check.
spvatom64() {
  local n=4096 f=kernels/spirv/atom64.spv b
  # The counters have to start at zero for the arithmetic below to mean anything,
  # and the launcher's generator fills a buffer with float patterns — so the eight
  # bytes of each counter are supplied from a file instead.
  printf '\0\0\0\0\0\0\0\0' > "$TMP/a64_zero.bin"
  local io=()
  for b in 0 1 2 3 4 5; do
    io+=(--load "$b=$TMP/a64_zero.bin" --dump "$b=$TMP/a64_$b.bin")
  done
  "$WEFT" run "$f" --kernel atom64k --cpu-only --grid "$n" --threads 32 \
      --buffers 1,1,1,1,1,1 "${io[@]}" >/dev/null
  # Two 32-bit words, little-endian, joined into one number. macOS od has no 64-bit
  # unsigned format, and bash arithmetic is 64-bit, so this is exact either way.
  u64of() {
    local w lo hi
    w=$(od -An -tu4 -N8 "$1" | tr -s ' ' '\n' | sed '/^$/d')
    lo=$(printf '%s\n' "$w" | sed -n 1p); hi=$(printf '%s\n' "$w" | sed -n 2p)
    echo $(( (hi << 32) | lo ))
  }
  local c b1 b2 b3 b4 b5 bad=""
  c=$(u64of "$TMP/a64_0.bin"); b1=$(u64of "$TMP/a64_1.bin")
  b2=$(u64of "$TMP/a64_2.bin"); b3=$(u64of "$TMP/a64_3.bin")
  b4=$(u64of "$TMP/a64_4.bin"); b5=$(u64of "$TMP/a64_5.bin")
  [ "$c" -eq "$n" ] || bad="$bad counter=$c"
  [ "$b1" -eq "$(( n * (n - 1) / 2 ))" ] || bad="$bad add_olds=$b1"
  [ "$b2" -eq 1 ] || bad="$bad umax=$b2"
  [ "$b3" -eq "$(( n - 1 ))" ] || bad="$bad umax_olds=$b3"
  [ "$b4" -eq -1 ] || bad="$bad smin=$b4"
  [ "$b5" -eq "$(( 1 - n ))" ] || bad="$bad smin_olds=$b5"
  # The same six counters on the GPU, against the same arithmetic. The launcher's
  # own compare reads these buffers as floats, and a u64 counter that lost 548
  # increments is a 7.7e-43 difference in that view — so it said "agree" about a
  # wrong answer. This one compares integers, and compares them because the answer
  # is known, not because two engines were asked the same question.
  local gbad=""
  if [ -n "$CPUMODE" ]; then
    gbad="no gpu on this host"
  else
    # Both sizes are exact, byte for byte, on every run measured: a wide kernel is
    # one thread walking every warp, so the pair has one writer and nothing spins.
    local gn gc gb1 gb2 gb3 gb4 gb5 go gio
    for gn in 4096 65536; do
      go=""
      gio=()
      for b in 0 1 2 3 4 5; do
        gio+=(--load "$b=$TMP/a64_zero.bin" --dump-gpu "$b=$TMP/a64g_$b.bin")
      done
      "$WEFT" run "$f" --kernel atom64k --grid "$gn" --threads 32 \
          --buffers 1,1,1,1,1,1 "${gio[@]}" >/dev/null 2>&1
      gc=$(u64of "$TMP/a64g_0.bin"); gb1=$(u64of "$TMP/a64g_1.bin")
      gb2=$(u64of "$TMP/a64g_2.bin"); gb3=$(u64of "$TMP/a64g_3.bin")
      gb4=$(u64of "$TMP/a64g_4.bin"); gb5=$(u64of "$TMP/a64g_5.bin")
      [ "$gc" -eq "$gn" ] || go="$go counter=$gc"
      [ "$gb1" -eq "$(( gn * (gn - 1) / 2 ))" ] || go="$go add_olds=$gb1"
      [ "$gb2" -eq 1 ] || go="$go umax=$gb2"
      [ "$gb3" -eq "$(( gn - 1 ))" ] || go="$go umax_olds=$gb3"
      [ "$gb4" -eq -1 ] || go="$go smin=$gb4"
      [ "$gb5" -eq "$(( 1 - gn ))" ] || go="$go smin_olds=$gb5"
      [ -z "$go" ] || gbad="$gbad ${gn}lanes:$go"
    done
  fi
  if [ -z "$bad" ] && [ -z "$gbad" ]; then
    echo "ok    spirv:atom64  $n lanes: add/max olds sum exactly, signed compare included,"
    echo "      and the gpu's six counters are the same integers at 4,096 and 65,536"
  else
    [ -z "$bad" ] || { echo "FAIL  spirv:atom64 $bad"; fail=1; }
    [ -z "$gbad" ] || { echo "FAIL  spirv:atom64 gpu:$gbad"; fail=1; }
    [ -z "$gbad" ] || [ -z "$bad" ] || true
  fi
}
spvatom64

# Every SPIR-V sample has to be a valid module before any of the above means
# anything: spirv-val is the reference the front end is being checked against.
if command -v spirv-val >/dev/null 2>&1; then
  bad=""
  for f in kernels/spirv/*.spv; do
    env=spv1.3
    case "$f" in *llvm.spv) env=spv1.4 ;; esac
    spirv-val --target-env "$env" "$f" >/dev/null 2>&1 || bad="$bad $f"
  done
  if [ -z "$bad" ]; then
    echo "ok    spirv       every module validates with spirv-val"
  else
    echo "FAIL  spirv       invalid modules:$bad"; fail=1
  fi
fi

# Both execution backends, same input, byte for byte: the per-lane state machine
# and the warp-vector backend (WEFT_WARP=1) must agree on every kernel.
# Same kernel twice — once per execution backend — into $3. Kernels differ in how
# many buffers they take, so try the common shapes rather than hard-coding one.
backend_run() {
  local prefix="$1" file="$2" k="$3" dst="$4" n
  for n in 2 3 1; do
    local bufs
    bufs="$(printf '1048576,%.0s' $(seq 1 "$n"))"
    # shellcheck disable=SC2086
    if env $prefix "$WEFT" run "$file" --kernel "$k" --cpu-only --grid 1048576 \
        --threads 256 --buffers "${bufs%,}" --arg 0=1.5 --arg 1=1048576 \
        --dump 1="$dst" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

backends_cmp() {
  local f="$1" k="$2"
  backend_run "" "$f" "$k" "$TMP/lanes.bin" || return 0
  backend_run "WEFT_WARP=1" "$f" "$k" "$TMP/warp.bin" || return 0
  if cmp -s "$TMP/lanes.bin" "$TMP/warp.bin"; then
    echo "ok    backends:$k  per-lane and warp-vector agree byte for byte"
  else
    echo "FAIL  backends:$k  the two backends disagree"; fail=1
  fi
}

for f in kernels/cuda/*.ptx; do
  backends_cmp "$f" "$(basename "$f" .ptx)"
done

# The hand-written kernels too, whose file names do not always match the entry.
backends_cmp kernels/round.ptx roundk
backends_cmp kernels/logic.ptx logic
backends_cmp kernels/bred.ptx  bred

# A render pass. The target is 64x64 pixels and every stage is one invocation per
# pixel, so the CPU's pixel loop and the GPU's rasteriser have to land on the same
# pixel for every pixel — the buffer is `in[y * 64 + x]`, so an off-by-one in the
# position mapping moves the input rather than hiding.
rendertest() {
  local file="$1" name="$2" target="$3"; shift 3
  local out
  out="$("$WEFT" run "$file" --kernel "$name" --render "$target" "$@" --bench 5 $CPUMODE)"
  if echo "$out" | grep -q DISAGREE; then
    echo "FAIL  render:$name"
    echo "$out" | sed 's/^/      /'
    fail=1
  else
    echo "ok    render:$name  $(summary "$out")"
  fi
}

rendertest kernels/basic.metal fragshade 64x64 --buffers 4096 --textures 64x64
# A target that is neither square nor a round number: 100x40, where the row
# stride is the target's width and not the square of anything.
rendertest kernels/basic.metal fragwide  100x40 --buffers 4000
# The tile stage fills tile memory for every pixel before any fragment reads it.
# Both sides lower tile memory to one element per pixel, and the tile buffer is
# compared like any other, so a stage that writes the wrong pixel - or a fragment
# that reads memory the stage never filled - is a difference, not a coincidence.
rendertest kernels/basic.metal fragtile 64x64 --buffers 4096
# The same, with the tile stage filling its group's block and waiting on it: the
# barrier is inside the stage, so this is the tile stage through the coroutine
# path, and the slot it leaves depends on the lane next door.
rendertest kernels/tilegroup.metal fraggroup 64x64 --buffers 4096 --textures 64x64

# Two tile stages in one pass is ambiguous, and a fragment that reads tile memory
# with no stage to fill it would compare two numbers neither side computed.
cat > "$TMP/twotile.metal" <<'EOF'
struct TileVal { float4 c; };
[[tile]] void t1(imageblock<TileVal> blk [[imageblock_data]],
                 device const float* in [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    blk.c = float4(in[gid], 0.0f, 0.0f, 1.0f);
}
[[tile]] void t2(imageblock<TileVal> blk [[imageblock_data]],
                 device const float* in [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    blk.c = float4(in[gid], 1.0f, 0.0f, 1.0f);
}
fragment float4 f(float4 pos [[position]], device const float* in [[buffer(0)]],
                  imageblock<TileVal> blk [[imageblock_data]]) {
    return float4(blk.c.x, 0.0f, 0.0f, 1.0f);
}
EOF
two_msg="$("$WEFT" run "$TMP/twotile.metal" --kernel f --render 64x64 --buffers 4096 2>&1 || true)"
if echo "$two_msg" | grep -q "has one .\[\[tile\]\]. stage"; then
  echo "ok    render:twotile  two tile stages are refused, not picked between"
else
  echo "FAIL  render:twotile  two tile stages were not refused"; fail=1
fi
cat > "$TMP/notile.metal" <<'EOF'
struct TileVal { float4 c; };
fragment float4 f(float4 pos [[position]], device const float* in [[buffer(0)]],
                  imageblock<TileVal> blk [[imageblock_data]]) {
    return float4(blk.c.x, 0.0f, 0.0f, 1.0f);
}
EOF
no_msg="$("$WEFT" run "$TMP/notile.metal" --kernel f --render 64x64 --buffers 4096 2>&1 || true)"
if echo "$no_msg" | grep -q "no .\[\[tile\]\]. stage"; then
  echo "ok    render:notile  a fragment reading tile memory needs a stage to fill it"
else
  echo "FAIL  render:notile  a fragment read tile memory with no tile stage"; fail=1
fi

# A fragment that reads the tile next door. The GPU leaves that read undefined -
# it is another tile's memory, which may not have been written yet - so there is
# no answer to compare and the run is refused. The legal case, a neighbour inside
# your own tile, is fraggroup above, whose slot comes from the lane next door.
hop_msg="$("$WEFT" run kernels/tilegroup.metal --kernel fragtilehop --render 64x64 --textures 64x64 2>&1 || true)"

# A mesh object stage. A mesh pipeline needs MTLGPUFamilyApple9 (M3 or later), but
# a mesh stage's output is only buffers, and the translation of it to plain compute
# is what runs on any Metal device — so this is checked twice: the kernel's two
# blocks against a reference model recomputed independently in Python, and the GPU
# run against the CPU one, on the same machine, bit for bit. One mesh per 32-lane
# threadgroup is the rule both use to find each mesh's block.
rm -f "$TMP/mesh_in.bin" "$TMP/mesh_v.bin" "$TMP/mesh_p.bin"
if "$WEFT" run kernels/mesh.metal --kernel meshmaker --grid 256 --buffers 256 \
     --bench 1 --dump 0="$TMP/mesh_in.bin" --dump 1="$TMP/mesh_v.bin" \
     --dump 2="$TMP/mesh_p.bin" > "$TMP/mesh.log" 2>&1 \
   && python3 - "$TMP/mesh_in.bin" "$TMP/mesh_v.bin" "$TMP/mesh_p.bin" "$TMP/mesh.log" <<'MESHPY'
import struct, sys
# The GPU ran this too: a mesh stage is lowered to plain compute, so there is a
# Metal answer to diff against on a device with no mesh pipeline at all. A host
# with no Metal backend has no GPU column, and says so instead — that is the one
# case where the model alone is the check.
log = open(sys.argv[4], encoding='utf-8').read()
if 'unavailable' not in log and '— agree' not in log:
    sys.exit(1)
inp = open(sys.argv[1], 'rb').read()
v = open(sys.argv[2], 'rb').read()
p = open(sys.argv[3], 'rb').read()
grid, lanes, nv, np_ = 256, 32, 16, 16
groups = grid // lanes

def f(i):
    return struct.unpack_from('<f', inp, 4 * i)[0]

if len(p) != groups * np_ * 16:   # a triangle is a uint3: Metal pads it to 16 bytes
    sys.exit(1)
if len(v) % (groups * nv):
    sys.exit(1)
vs = len(v) // (groups * nv)   # sizeof(Vtx): the compiler's answer, not a guess
bad = 0
for g in range(groups):
    for i in range(nv):
        o = (g * nv + i) * vs
        px, py, pz, pw = struct.unpack_from('<4f', v, o)
        ux, uy = struct.unpack_from('<2f', v, o + 16)
        if i == 4:
            # lane 16 rewrote this slot from vertex 3, after the barrier
            want, ux_w, uy_w = (3.0 + 3.5, f(g * lanes + 3), 6.0, 1.0), 3.0, 3.5
        else:
            want = (float(i), f(g * lanes + i), float(i) * 2.0, 1.0)
            ux_w, uy_w = float(i), float(i) + 0.5
        if (px, py, pz, pw) != want or (ux, uy) != (ux_w, uy_w):
            bad += 1
    for k in range(np_):
        a, b, c = struct.unpack_from('<3I', p, (g * np_ + k) * 16)
        if (a, b, c) != (k, (k + 1) % nv, (k + 2) % nv):
            bad += 1
sys.exit(1 if bad else 0)
MESHPY
then
  echo "ok    mesh        16 vertices and 16 triangles per mesh, matched against a model"
else
  echo "FAIL  mesh         a mesh object stage did not match the reference model"
  tail -4 "$TMP/mesh.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

# A task stage ahead of a mesh stage. The task stage runs first over --grid (96
# threads = three groups of 32), sets the meshgrid to four meshes and fills one
# payload element per task group. The launcher cuts the meshgrid into contiguous
# blocks, one per task group, and copies each group's payload to the meshes of its
# block: meshes 0 and 1 read group 0, mesh 2 group 1, mesh 3 group 2. The
# partition is the launcher's, written down here and nowhere left to guesswork.
rm -f "$TMP/tm_res.bin" "$TMP/tm_v.bin" "$TMP/tm_p.bin" "$TMP/tm_pl.bin"
if "$WEFT" run kernels/taskmesh.metal --kernel meshuse --grid 96 --buffers 8 \
     --bench 1 --dump 0="$TMP/tm_res.bin" --dump 1="$TMP/tm_v.bin" \
     --dump 2="$TMP/tm_p.bin" --dump 3="$TMP/tm_pl.bin" > "$TMP/tm.log" 2>&1 \
   && python3 - "$TMP/tm_res.bin" "$TMP/tm_v.bin" "$TMP/tm_p.bin" "$TMP/tm_pl.bin" \
              "$TMP/tm.log" <<'TASKPY'
import struct, sys
tlog = open(sys.argv[5], encoding='utf-8').read()
if 'unavailable' not in tlog and '— agree' not in tlog:
    sys.exit(1)
res = open(sys.argv[1], 'rb').read()
v = open(sys.argv[2], 'rb').read()
p = open(sys.argv[3], 'rb').read()
pl = open(sys.argv[4], 'rb').read()
meshes, nv = 4, 8
# struct Payload { uint base; float scale; }, one element per mesh: the launcher
# has already copied each task group's payload into the block of meshes it owns.
if len(pl) != meshes * 8:
    sys.exit(1)
base = [struct.unpack_from('<I', pl, 8 * m)[0] for m in range(meshes)]
scale = [struct.unpack_from('<f', pl, 8 * m + 4)[0] for m in range(meshes)]
# Three task groups (base 0, 4, 8) over four meshes: meshes 0 and 1 are group 0's.
if (base, scale) != ([0, 0, 4, 8], [0.5, 0.5, 1.5, 2.5]):
    sys.exit(1)
vs = len(v) // (meshes * nv)
out = struct.unpack('<%df' % (meshes * 2), res)
bad = 0
for m in range(meshes):
    t = m
    for i in range(nv):
        q = struct.unpack_from('<4f', v, (m * nv + i) * vs)
        if q != (float(base[t] + i), scale[t], float(m), 1.0):
            bad += 1
    if struct.unpack_from('<3I', p, m * nv * 16) != (0, 1, 2):
        bad += 1
    if out[2 * m] != float(base[t]) + scale[t] or out[2 * m + 1] != float(m):
        bad += 1
sys.exit(1 if bad else 0)
TASKPY
then
  echo "ok    task:mesh   a task stage set the meshgrid (4 meshes) and its payload reached them"
else
  echo "FAIL  task:mesh   the task and mesh stages did not match the reference model"
  tail -5 "$TMP/tm.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

# The translation cache is keyed by the generated code, so a second run of the
# same kernel must not compile again: that is what makes the CLI usable in a loop.
cached_run="$("$WEFT" run kernels/basic.metal --kernel saxpy --grid 4096 --buffers 4096,4096,4096 --cpu-only 2>&1)"
if echo "$cached_run" | grep -q "cached"; then
  echo "ok    cache       a repeated translation is reused, not recompiled"
else
  echo "FAIL  cache       the second run compiled again"
  echo "$cached_run" | grep translate | sed 's/^/      /'; fail=1
fi
if echo "$hop_msg" | grep -q "outside its own 32x32 tile"; then
  echo "ok    render:tilehop  a read of another tile is refused, not compared"
else
  echo "FAIL  render:tilehop  an out-of-tile tile-memory read was not refused"; fail=1
fi

# A launch that cannot fill the block a module declares leaves lanes the program
# is written against, and their staging slots would be read unfilled. The run
# names the launch instead of answering with those bits.
short_block="$("$WEFT" run kernels/triton/wgmma.ptx --kernel wgmma_kernel --grid 32 \
  --threads 128 --buffers 32768,32768,32768 2>&1 || true)"
if echo "$short_block" | grep -q "at least one full block"; then
  echo "ok    launch:shortblock  a grid under one block is refused, not compared"
else
  echo "FAIL  launch:shortblock  a under-filled block launch was compared anyway"; fail=1
fi

# The linked path: build/embed links the translated kernel into one binary and
# calls it through include/weft/weft.h. It writes its two inputs and its result,
# and the CLI is then fed those same inputs, so the two paths have to produce the
# same bytes - a drift between the library and the tool is a difference here.
if make -s build/embed >/dev/null 2>&1 && ./build/embed "$TMP/embed" >/dev/null 2>&1; then
  "$WEFT" run kernels/basic.metal --kernel saxpy --grid 4096 --buffers 4096,4096,4096 \
      --cpu-only --load 0="$TMP/embed_a.bin" --load 1="$TMP/embed_b.bin" \
      --dump 2="$TMP/cli_out.bin" >/dev/null
  if cmp -s "$TMP/embed_out.bin" "$TMP/cli_out.bin"; then
    echo "ok    embed       the linked kernel and the CLI agree byte for byte"
  else
    echo "FAIL  embed       the linked kernel and the CLI differ"; fail=1
  fi
else
  echo "FAIL  embed       could not build or run examples/embed"; fail=1
fi

# A buffer's width comes from the kernel's declaration: `--buffers 262144` for a
# float4 kernel is a million floats, not a quarter of a million.
typed=$("$WEFT" run kernels/basic.metal --kernel vec4_scale --grid 262144 \
    --buffers 262144,262144 --bench 1 | grep '^kernel')
if echo "$typed" | grep -q "float4 16B"; then
  echo "ok    typed       buffers are elements of the declared type (float4 16B)"
else
  echo "FAIL  typed       the float4 buffer was not reported as 16 bytes"
  echo "$typed" | sed 's/^/      /'; fail=1
fi

# SPIR-V states its own layout (ArrayStride, Offset), so weft-gen reads each
# binding's element width off the pointer's pointee. PTX binds an address, so the
# width comes from the instructions that reach the buffer instead: `ld.global.f32`
# is four bytes, and a kernel whose only access is a byte-block copy (`cp.async`,
# whose width is the copy's, not the element's) still reports none rather than a
# guess. A stated width wins over either.
spv_types=$(./build/weft-gen kernels/spirv/global.spv --describe)
if echo "$spv_types" | grep -q "types=4,4"; then
  echo "ok    spirv:types  element widths come from the module's own pointer types"
else
  echo "FAIL  spirv:types  SPIR-V widths were not reported"; echo "$spv_types" | sed 's/^/      /'; fail=1
fi
ptx_types=$(./build/weft-gen kernels/reverse.ptx --describe)
ptx_stated=$(./build/weft-gen kernels/reverse.ptx --describe --buffer-bytes 0=4,1=16)
ptx_none=$(./build/weft-gen kernels/async_copy.ptx --describe)
if echo "$ptx_types" | grep -q "types=4,4" && echo "$ptx_stated" | grep -q "types=4,16" &&
   echo "$ptx_none" | grep -q "types=0,4"; then
  echo "ok    ptx:types    widths come from the instructions that reach each buffer"
else
  echo "FAIL  ptx:types    the PTX width inference drifted"; fail=1
fi

# weft-gen is what a build system calls, so its manifest has to name the arguments.
desc=$(./build/weft-gen kernels/basic.metal --describe)
if echo "$desc" | grep -q '^kernel 0 saxpy kernel buffers=3 textures=0 sync=0 threads=0 types=float,float,float$'; then
  echo "ok    weft-gen    --describe names each kernel, its buffers and their types"
else
  echo "FAIL  weft-gen    --describe did not describe saxpy"
  echo "$desc" | head -3 | sed 's/^/      /'; fail=1
fi

# ... including the launch size a module states, which is the one thing a caller
# of a Triton kernel cannot infer from the arguments (threads=32 is `.maxntid`).
tdesc=$(./build/weft-gen kernels/triton/matmul.ptx --describe)
if echo "$tdesc" | grep -q '^kernel 0 matmul_kernel kernel buffers=3 textures=0 sync=1 threads=32 types=4,4,4$'; then
  echo "ok    triton:gen  --describe carries the launch size the module declares"
else
  echo "FAIL  triton:gen"; echo "$tdesc" | sed 's/^/      /'; fail=1
fi

# A render pass is a frame, and the colour target is RGBA rather than one
# channel. --out saves it, so both engines write the same image: the files are
# compared byte for byte, which is the `compare` line in a form a viewer opens.
rm -f /tmp/weft_render_cpu.png /tmp/weft_render_gpu.png
# The PNG header and the IHDR width/height, as bytes: 100 x 40 is 00000064
# 00000028. That is readable without an image library, on any host.
png_ok() {
  head -c 8 "$1" | od -An -tx1 | tr -d ' \n' | grep -q '^89504e470d0a1a0a$' \
    && [ "$(od -An -tx1 -j16 -N8 "$1" | tr -d ' \n')" = "0000006400000028" ]
}
if [ -n "$CPUMODE" ]; then
  # No GPU frame to diff, so the claim is the part that is still checkable: the
  # render pass produced a 100x40 RGBA PNG rather than a file that merely exists.
  if "$WEFT" run kernels/basic.metal --kernel fragwide --render 100x40 --buffers 4000 \
       --bench 1 --out /tmp/weft_render_cpu.png > /tmp/weft_render.log 2>&1 \
     && png_ok /tmp/weft_render_cpu.png; then
    echo "ok    render:image  the frame is a PNG, 100x40  (no gpu on this host to diff)"
  else
    echo "FAIL  render:image"; tail -5 /tmp/weft_render.log 2>/dev/null | sed 's/^/      /'; fail=1
  fi
elif "$WEFT" run kernels/basic.metal --kernel fragwide --render 100x40 --buffers 4000 \
     --bench 1 --out /tmp/weft_render_cpu.png --out-gpu /tmp/weft_render_gpu.png \
     > /tmp/weft_render.log 2>&1 \
   && cmp -s /tmp/weft_render_cpu.png /tmp/weft_render_gpu.png \
   && png_ok /tmp/weft_render_cpu.png; then
  echo "ok    render:image  the frame is a PNG, and the cpu and gpu files are identical"
else
  echo "FAIL  render:image"; tail -5 /tmp/weft_render.log 2>/dev/null | sed 's/^/      /'; fail=1
fi

# Portable math: 54 functions over 65,536 lanes, dumped from both engines and
# compared as bytes. Identical is the whole claim — sqrt, rsqrt, log2, exp2, pow,
# mix, dot, length, distance and normalize are the same code on both sides, so a
# single differing bit is a failure rather than a tolerance.
rm -f /tmp/weft_math_cpu.bin /tmp/weft_math_gpu.bin /tmp/weft_math_in.bin
if [ -n "$CPUMODE" ]; then
  # The libm comparison below still runs on this dump, which is the check that
  # catches two engines sharing one wrong function. What is missing here is the
  # second engine, and the line says so.
  if "$WEFT" run kernels/mathfit.metal --kernel fnprobe --grid 65536 \
       --buffers 393216,3538944 --bench 1 --dump 0=/tmp/weft_math_in.bin \
       --dump 1=/tmp/weft_math_cpu.bin > /tmp/weft_math.log 2>&1 \
     && [ -s /tmp/weft_math_cpu.bin ]; then
    echo "ok    math:portable  54 functions x 65,536 lanes ran on the cpu  (no gpu on this host)"
  else
    echo "FAIL  math:portable"; tail -4 /tmp/weft_math.log 2>/dev/null | sed 's/^/      /'; fail=1
  fi
elif "$WEFT" run kernels/mathfit.metal --kernel fnprobe --grid 65536 \
     --buffers 393216,3538944 --bench 1 --dump 0=/tmp/weft_math_in.bin \
     --dump 1=/tmp/weft_math_cpu.bin --dump-gpu 1=/tmp/weft_math_gpu.bin \
     > /tmp/weft_math.log 2>&1 \
   && cmp -s /tmp/weft_math_cpu.bin /tmp/weft_math_gpu.bin; then
  echo "ok    math:portable  54 functions x 65,536 lanes are byte-identical on cpu and gpu"
else
  echo "FAIL  math:portable"; tail -4 /tmp/weft_math.log 2>/dev/null | sed 's/^/      /'; fail=1
fi


# Agreement is not accuracy: two engines running the same wrong code pass the
# byte comparison, which is what the portable sqrt did before this check
# existed. So all 48 computed columns on all 65,536 lanes are also compared
# against libm, each with its own limit; a failure names the column.
if command -v python3 >/dev/null 2>&1; then
  if acc=$(python3 - /tmp/weft_math_in.bin /tmp/weft_math_cpu.bin <<'ACCPY'
import struct, math, sys
inp = open(sys.argv[1], 'rb').read()
cpu = open(sys.argv[2], 'rb').read()
lanes = len(inp) // 24
rows = len(cpu) // 4 // lanes
def at(i):
    return struct.unpack_from('<f', cpu, 4 * i)[0]
def given(i):
    return struct.unpack_from('<f', inp, 4 * i)[0]
def dot3(x, y):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
def dot4(x, y):
    return dot3(x, y) + x[3] * y[3]
def norm3(x):
    l = math.sqrt(dot3(x, x))
    return [float('nan')] * 3 if l == 0.0 else [c / l for c in x]
def cl(x):
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)
# column -> (name, reference of (a, b, u, a3, b3, a4, b4), limit)
# The metric is |got-ref| / max(1, |ref|): absolute for small answers, relative
# for large ones, so one number works for a value of 1e-3 and one of 1e4.
def refs(a, b, u, a3, b3, a4, b4):
    n, nb = norm3(a3), norm3(b3)
    n2 = norm3([a[0], a[1], 0.0])
    d = dot3(nb, n)
    k = 1.0 - 0.5625 * (1.0 - d * d)
    # refract(I, N, eta) = eta*I - (eta*dot(N,I) + sqrt(k))*N, and zero when the
    # angle is past the critical one — the same text Metal's documentation gives.
    rf = [0.0, 0.0, 0.0] if k < 0 else [0.75 * n[i] - (0.75 * d + math.sqrt(k)) * nb[i]
                                        for i in range(3)]
    return {
        0: ('sqrt', math.sqrt(u)),
        1: ('divide', 1.0 / u),
        2: ('rsqrt', 1.0 / math.sqrt(u)),
        3: ('divide2', u / (u + 1.0)),
        4: ('exp2', 2.0 ** u),
        5: ('log2', math.log2(u)),
        6: ('length2', math.sqrt(a[0] ** 2 + a[1] ** 2)),
        7: ('dot2', a[0] * b[0] + a[1] * b[1]),
        8: ('length4', math.sqrt(dot4(a4, a4))),
        9: ('dot4', dot4(a4, b4)),
        10: ('mix', a[0] + (b[0] - a[0]) * a[1]),
        11: ('mix_form1', a[0] + (b[0] - a[0]) * a[1]),
        12: ('mix_form2', a[1] * (b[0] - a[0]) + a[0]),
        13: ('mix_form3', a[0] * (1.0 - a[1]) + b[0] * a[1]),
        14: ('pow3', u ** 3),
        15: ('pow_half', math.sqrt(u)),
        16: ('sqrt_sq', u),
        17: ('fma', u * u + 1.0),
        18: ('recip_sqrt', 1.0 / math.sqrt(u)),
        19: ('u_rsqrt', u / math.sqrt(u)),
        20: ('normalize', n2[0]),
        21: ('distance', math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)),
        22: ('length3', math.sqrt(dot3(a3, a3))),
        23: ('dot3', dot3(a3, a3)),
        24: ('sqrt_chain', math.sqrt(a[0] ** 2 + a[1] ** 2)),
        25: ('fma_chain', a[0] ** 2 + a[1] ** 2),
        26: ('recip_chain', 1.0 / math.sqrt(a[0] ** 2 + 1.0)),
        27: ('log2_scaled', 1.5 * math.log2(u)),
        28: ('exp2_log2', u ** 1.5),   # exp2(1.5*log2(u))
        29: ('mix_vs_fma', 0.0),
        30: ('sqrt_vs_rsqrt', 0.0),
        31: ('fma_exact', 0.0),
        32: ('sin', math.sin(a[0])),
        33: ('cos', math.cos(a[1])),
        34: ('tan', math.tan(a[0] * 0.5)),
        35: ('asin', math.asin(cl(a[0]))),
        36: ('acos', math.acos(cl(a[1]))),
        37: ('atan', math.atan(a[2])),
        38: ('atan2', math.atan2(a[1], a[0])),
        39: ('sinh', math.sinh(a[0])),
        40: ('cosh', math.cosh(a[1])),
        41: ('tanh', math.tanh(a[0])),
        42: ('asinh', math.asinh(a[2])),
        43: ('acosh', math.acosh(abs(a[0]) + 1.0)),
        44: ('atanh', math.atanh(a[0] * 0.5)),
        45: ('exp10', 10.0 ** a[1]),
        46: ('log10', math.log10(abs(a[0]) + 0.5)),
        47: ('powr', (abs(a[0]) + 0.5) ** a[1]),
        48: ('cross', a[1] * b[2] - a[2] * b[1]),
        49: ('reflect', n[0] - 2.0 * dot3(nb, n) * nb[0]),
        50: ('refract', rf[0]),
        51: ('faceforward', a[2] if dot3(a3, b3) < 0 else -a[2]),
    }
# exp10/powr answer on a coarser grid by construction; everything else is within
# a few ulp, so 1e-6 is a tight bound rather than a generous one.
limited = {'exp10': 2e-5, 'powr': 2e-5}
worst = {}
for i in range(lanes):   # every lane: a subsample would miss the lane with the bug
    a = [given(6 * i + t) for t in range(3)]
    b = [given(6 * i + t) for t in range(3, 6)]
    u = abs(a[0]) + 0.5
    try:
        table = refs(a, b, u, a, b, a + [1.0], b + [1.0])
    except (ValueError, OverflowError):
        continue  # an argument outside a function's domain, on this lane
    for col, (name, ref) in table.items():
        if ref != ref:
            continue  # the lane itself is degenerate (normalize of zero)
        got = at(i * rows + col)
        e = 1e9 if got != got else abs(got - ref) / max(1.0, abs(ref))
        if e > worst.get(name, (-1.0, 0))[0]:
            worst[name] = (e, col)
bad = [(n, e, c) for n, (e, c) in worst.items() if e > limited.get(n, 1e-6)]
if bad:
    print('%d column(s) over limit: %s' % (len(bad), ', '.join(
        '%s %.0e' % (n, e) for n, e, c in sorted(bad, key=lambda t: -t[1])[:5])))
    sys.exit(1)
if len(worst) < 48:
    print('only %d of 48 columns compared' % len(worst))
    sys.exit(1)
top = max(worst.items(), key=lambda kv: kv[1][0])
print('%.1e (%s, 48 columns)' % (top[1][0], top[0]))
sys.exit(0)
ACCPY
  ); then
    echo "ok    math:accuracy  worst $acc against libm"
  else
    echo "FAIL  math:accuracy  worst $acc against libm"; fail=1
  fi
else
  echo "FAIL  math:accuracy  no python3 to compare against libm"; fail=1
fi

# ---- every ptx/spv fixture is compared by a check that names it ----
# A `.ptx` or `.spv` sample that nothing diffs is a translation nobody checked, and
# a hand-written twin that no call invokes can drift away from the kernel it
# mirrors with nothing to notice. The inventory is the directory, the coverage is
# the calls above, and both have to agree.
#
# Four llc-emitted samples predate these comparisons and no twin was written for
# them, so they are named here as known-uncompared instead of being skipped in
# silence: that is a gap with a name on it, not one hiding behind a passing run.
known_uncompared=""   # empty: every fixture has a comparison now
self="${BASH_SOURCE[0]}"
uncovered=""
for fx in kernels/*.ptx kernels/cuda/*.ptx kernels/triton/*.ptx kernels/spirv/*.spv; do
  [ -e "$fx" ] || continue
  fxbase="$(basename "$fx")"; fxkey="${fxbase%.*}"
  case "$known_uncompared" in *" $fx "*) continue ;; esac
  if grep -qF "$fx" "$self"; then continue; fi
  case "$fx" in
    kernels/cuda/*)   if grep -qE "^nvptx +$fxkey([[:space:]]|$)" "$self"; then continue; fi ;;
    kernels/triton/*) if grep -qE "^triton +$fxkey([[:space:]]|$)" "$self"; then continue; fi ;;
    kernels/spirv/*)  if grep -qE "^spvcross +[^ ]+ +$fxkey([[:space:]]|$)" "$self"; then continue; fi
                      # An `spvgpu` line is a comparison of the module against
                      # itself: the metal emitted from it against the C++ emitted
                      # from the same decode. It names the fixture the same way.
                      if grep -qE "^spvgpu +$fxkey([[:space:]]|$)" "$self"; then continue; fi ;;
    *)                if grep -qE "^cross +$fxkey([[:space:]]|$)" "$self"; then continue; fi ;;
  esac
  uncovered="$uncovered $fx"
done
if [ -z "$uncovered" ]; then
  echo "ok    twins        every ptx/spv fixture is named by a comparison in this file"
else
  echo "FAIL  twins        fixtures no comparison names:$uncovered"; fail=1
fi

exit "$fail"
