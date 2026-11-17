#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
WEFT=./build/weft
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

NOSHADOW=""
CPUMODE=""
GPUCOL=""
GPUREF=""
host_os="$(uname -s)"
if [ "$host_os" = Darwin ]; then
  probe=""
  if ! probe="$("$WEFT" run kernels/basic.metal --kernel saxpy --grid 16 \
        --buffers 16,16,16 --bench 1 2>&1)"; then
    echo "FAIL  the metal column exited nonzero on the probe kernel"
    printf '%s\n' "$probe" | sed 's/^/      /'
    exit 1
  fi
  if ! printf '%s\n' "$probe" | grep -q '^compare'; then
    echo "FAIL  the metal column did not compare the probe kernel"
    printf '%s\n' "$probe" | sed 's/^/      /'
    exit 1
  fi
else
  CPUMODE="--cpu-only"
  NOSHADOW="  (no gpu on this host, ran on the cpu)"
fi

ptx_state() {
  case "$1" in
    *"no CUDA driver"*|*"no compute entry points"*|*"no CUDA device"*) echo absent;;
    "gpu"*"no column for this kernel"*) echo refused;;
    *) echo unknown;;
  esac
}

probe_bad=""
[ "$(ptx_state 'gpu       no column for this kernel  no CUDA driver (libcuda) on this host')" = absent ] || probe_bad="$probe_bad absent-driver"
[ "$(ptx_state 'gpu       no column for this kernel  the driver refused the module: invalid device function')" = refused ] || probe_bad="$probe_bad refused-module"
[ "$(ptx_state 'gpu       no column for this kernel  the launch failed: out of resources')" = refused ] || probe_bad="$probe_bad failed-launch"
[ "$(ptx_state 'weft: something else')" = unknown ] || probe_bad="$probe_bad unknown"
if [ -n "$probe_bad" ]; then
  echo "FAIL  the ptx probe misreads$probe_bad"
  fail=1
fi
if [ "$host_os" = Darwin ]; then
  probe=""
  if ! probe="$("$WEFT" run kernels/cuda/saxpy.ptx --grid 16 --threads 16 \
        --buffers 16,16,16 --bench 1 2>&1)"; then
    echo "FAIL  the metal ptx column exited nonzero on the probe kernel"
    printf '%s\n' "$probe" | sed 's/^/      /'
    exit 1
  fi
  if ! printf '%s\n' "$probe" | grep -q '^compare'; then
    echo "FAIL  the metal ptx column did not compare the probe kernel"
    printf '%s\n' "$probe" | sed 's/^/      /'
    exit 1
  fi
else
  probe=""
  probe_st=0
  probe="$("$WEFT" run kernels/cuda/saxpy.ptx --grid 16 --threads 16 \
        --buffers 16,16,16 --bench 1 2>&1)" || probe_st=$?
  if [ "$probe_st" -ne 0 ]; then
    echo "FAIL  the ptx column exited $probe_st on the probe kernel"
    printf '%s\n' "$probe" | sed 's/^/      /'
    exit 1
  fi
  if ! printf '%s\n' "$probe" | grep -q '^compare'; then
    gpu_line="$(printf '%s\n' "$probe" | grep '^gpu' | head -1 || true)"
    case "$(ptx_state "$gpu_line")" in
      absent) GPUCOL=1;;
      refused)
        GPUREF="${gpu_line#*no column for this kernel}"
        GPUREF="$(printf '%s' "$GPUREF" | sed 's/^[[:space:]]*//')"
        ;;
      *)
        echo "FAIL  the ptx column printed neither a comparison nor a reason"
        printf '%s\n' "$gpu_line" | sed 's/^/      /'
        exit 1
        ;;
    esac
  fi
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
  local need_cmp=1
  case "$CPUMODE" in *--cpu-only*) need_cmp=0 ;; esac
  if echo "$out" | grep -qE "DISAGREE|^weft: " || { [ "$need_cmp" = "1" ] && ! echo "$out" | grep -q "compare"; }; then
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

run texlin     --grid 65536   --buffers 65536 --textures 64x64   # bilinear, 8-bit weights + fma
run texarray   --grid 65536   --buffers 65536 --textures '64x64*4'
run tex3d      --grid 65536   --buffers 65536 --textures 64x64x8
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
if [ -n "$CPUMODE" ]; then
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
run packedk    --grid 1048576 --buffers 1048576,1048576
run narrowv    --grid 1048576 --buffers 1048576,1048576
run helperk    --grid 1048576 --threads 256 --buffers 1048576,1048576
run texrw      --grid 4096 --threads 256 --buffers 4096,4096 --textures 64x64
run counts     --grid 1000 --threads 96 --buffers 1000
run counts     --grid 3072 --threads 96 --buffers 3072
run localtype  --grid 1048576 --threads 256 --buffers 1048576,1048576
run lambdak    --grid 1048576 --threads 256 --buffers 1048576,1048576
# A lambda kept in a local across the barrier, and the three capture-list shapes.
run lamstore   --grid 1048576 --threads 256 --buffers 1048576,1048576
run lamcap     --grid 1048576 --threads 256 --buffers 1048576,1048576
# A body-declared type with a method of its own, its local held across the wait.
run localmeth  --grid 1048576 --threads 256 --buffers 1048576,1048576
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
run packround  --grid 65536 --buffers 262144,786432
# A componentwise compare makes a mask, and select/any/all are its three readers.
run choosemask --grid 65536 --buffers 262144,524288
run swizzle    --grid 65536 --buffers 262144,524288

runf kernels/ptx1d.ptx ptx1d --grid 65536 --buffers 65536 --textures 64x1
runf kernels/sharedofs.ptx sharedofs --grid 256 --buffers 768
runf kernels/mbar.ptx mbar --grid 32 --threads 32 --buffers 32
runf kernels/mbartx.ptx mbartx --grid 1 --threads 1 --buffers 3

"$WEFT" run kernels/mbartx.ptx --kernel mbartx --grid 1 --threads 1 --buffers 3 \
    --cpu-only --dump 0="$TMP/mbartx.bin" >/dev/null
read -r tx_owed tx_arrived tx_count <<< "$(od -An -tu4 -v "$TMP/mbartx.bin")"
if [ "$tx_owed" = "16" ] && [ "$tx_arrived" = "23" ] && [ "$tx_count" = "8" ]; then
  echo "ok    mbartx   4 + 12 expected bytes is 16, then 8 more and an arrival is 23"
else
  echo "FAIL  mbartx   read $tx_owed, $tx_arrived, $tx_count; wanted 16, 23, 8"; fail=1
fi

printf '\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\x00\x00\x80\x40' > "$TMP/bulk_in.bin"
runf kernels/mbarbulk.ptx mbarbulk --grid 1 --threads 1 --buffers 1,4 --load 1="$TMP/bulk_in.bin"
"$WEFT" run kernels/mbarbulk.ptx --kernel mbarbulk --grid 1 --threads 1 --buffers 1,4 \
    --load 1="$TMP/bulk_in.bin" --cpu-only --dump 0="$TMP/bulk_out.bin" >/dev/null
read -r bulk_word <<< "$(od -An -tu4 -v "$TMP/bulk_out.bin")"
if [ "$bulk_word" = "1065353216" ]; then
  echo "ok    mbarbulk the copy moved the bytes the barrier was expecting (1.0f)"
else
  echo "FAIL  mbarbulk tile[0] was $bulk_word bits; wanted 1.0f"; fail=1
fi

runf kernels/mbarbulkst.ptx mbarbulkst --grid 1 --threads 1 --buffers 4
"$WEFT" run kernels/mbarbulkst.ptx --kernel mbarbulkst --grid 1 --threads 1 --buffers 4 \
    --cpu-only --dump 0="$TMP/bulkst.bin" >/dev/null
read -r st0 st1 st2 st3 <<< "$(od -An -tu4 -v "$TMP/bulkst.bin")"
if [ "$st0" = "11" ] && [ "$st1" = "12" ] && [ "$st2" = "13" ] && [ "$st3" = "14" ]; then
  echo "ok    mbarbulkst the store moved all four words out of threadgroup memory"
else
  echo "FAIL  mbarbulkst read $st0 $st1 $st2 $st3; wanted 11 12 13 14"; fail=1
fi

runf kernels/mbarredf16.ptx mbarredf16 --grid 1 --threads 1 --buffers 4,4
"$WEFT" run kernels/mbarredf16.ptx --kernel mbarredf16 --grid 1 --threads 1 --buffers 4,4 \
    --cpu-only --dump 1="$TMP/redf16.bin" >/dev/null
if [ "$(od -An -tx4 -v "$TMP/redf16.bin" | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "4a804980 4c404b80 4d404cc0 4e404dc0 " ]; then
  echo "ok    mbarredf16  the f16 reduce added 1..8 to 10..17 as halves"
else
  echo "FAIL  mbarredf16  read $(od -An -tx4 -v "$TMP/redf16.bin" | tr -s ' ' ' '); wanted 11 13 15 17 19 21 23 25 as halves"; fail=1
fi

runf kernels/mbarredinc.ptx mbarredinc --grid 1 --threads 1 --buffers 4,4
"$WEFT" run kernels/mbarredinc.ptx --kernel mbarredinc --grid 1 --threads 1 --buffers 4,4 \
    --cpu-only --dump 0="$TMP/inc.bin" --dump 1="$TMP/dec.bin" >/dev/null
if [ "$(od -An -tu4 -v "$TMP/inc.bin" | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = "1 2 3 0 " ] && \
   [ "$(od -An -tu4 -v "$TMP/dec.bin" | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = "2 0 1 0 " ]; then
  echo "ok    mbarredinc  the bounded inc and dec wrapped where the bound said"
else
  echo "FAIL  mbarredinc  read $(od -An -tu4 -v "$TMP/inc.bin" | tr -s ' ' ' ') / $(od -An -tu4 -v "$TMP/dec.bin" | tr -s ' ' ' ')"; fail=1
fi

runf kernels/mbarredf64.ptx mbarredf64 --grid 1 --threads 1 --buffers 4,4
"$WEFT" run kernels/mbarredf64.ptx --kernel mbarredf64 --grid 1 --threads 1 --buffers 4,4 \
    --cpu-only --dump 1="$TMP/redf64.bin" >/dev/null
if [ "$(head -c 16 "$TMP/redf64.bin" | od -An -tx8 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "3ffc000000000000 4006000000000000 " ]; then
  echo "ok    mbarredf64  the f64 reduce added 0.25 to 1.5 and 2.5"
else
  echo "FAIL  mbarredf64  read $(od -An -tx8 -v "$TMP/redf64.bin" | tr -s ' ' ' '); wanted 1.75 and 2.75"; fail=1
fi

runf kernels/mbarredbf16.ptx mbarredbf16 --grid 1 --threads 1 --buffers 4,4
"$WEFT" run kernels/mbarredbf16.ptx --kernel mbarredbf16 --grid 1 --threads 1 --buffers 4,4 \
    --cpu-only --dump 1="$TMP/redbf16.bin" >/dev/null
if [ "$(od -An -tx4 -v "$TMP/redbf16.bin" | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "40403fc0 40c04090 411040f0 41404128 " ]; then
  echo "ok    mbarredbf16  the bf16 reduce added 1..8 to 0.5, 1, ... 4 as bf16"
else
  echo "FAIL  mbarredbf16  read $(od -An -tx4 -v "$TMP/redbf16.bin" | tr -s ' ' ' '); wanted 1.5 3 4.5 6 7.5 9 10.5 12"; fail=1
fi

runf kernels/tf32cvt.ptx tf32cvt --grid 8 --threads 8 --buffers 24
"$WEFT" run kernels/tf32cvt.ptx --kernel tf32cvt --grid 8 --threads 8 --buffers 24 \
    --cpu-only --dump 0="$TMP/tf32.bin" >/dev/null
if [ "$(head -c 96 "$TMP/tf32.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "3f800000 3f800000 3f800000 3f802000 3f800000 3f800000 3f802000 3f802000 3f802000 3f804000 3f804000 3f802000 3f804000 3f804000 3f804000 3f806000 3f804000 3f804000 3f806000 3f806000 3f806000 3f808000 3f808000 3f806000 " ]; then
  echo "ok    tf32cvt  rna, rn and rz round the thirteen dropped bits as the isa says"
else
  echo "FAIL  tf32cvt  read $(head -c 96 "$TMP/tf32.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/tf32cvt.ptx tf32sat --grid 1 --threads 1 --buffers 4
"$WEFT" run kernels/tf32cvt.ptx --kernel tf32sat --grid 1 --threads 1 --buffers 4 \
    --cpu-only --dump 0="$TMP/tf32sat.bin" >/dev/null
if [ "$(head -c 16 "$TMP/tf32sat.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "00000000 7f7fe000 00000000 3f800000 " ]; then
  echo "ok    tf32sat  relu zeroed a negative and satfinite clamped inf and nan"
else
  echo "FAIL  tf32sat  read $(head -c 16 "$TMP/tf32sat.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtf16.ptx cvtf16 --grid 1 --threads 1 --buffers 8
"$WEFT" run kernels/cvtf16.ptx --kernel cvtf16 --grid 1 --threads 1 --buffers 8 \
    --cpu-only --dump 0="$TMP/cvtf16.bin" >/dev/null
if [ "$(head -c 32 "$TMP/cvtf16.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "00003c00 00003c02 00007bff 00000000 00000000 0000c100 00003f80 00007f7f " ]; then
  echo "ok    cvtf16  the 16-bit converts rounded, clamped and zeroed as the isa says"
else
  echo "FAIL  cvtf16  read $(head -c 32 "$TMP/cvtf16.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtf16.ptx cvtf16rz --grid 1 --threads 1 --buffers 5
"$WEFT" run kernels/cvtf16.ptx --kernel cvtf16rz --grid 1 --threads 1 --buffers 5 \
    --cpu-only --dump 0="$TMP/cvtf16rz.bin" >/dev/null
if [ "$(head -c 20 "$TMP/cvtf16rz.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "00003c03 00007bff 00007c00 00003f80 00007fc0 " ]; then
  echo "ok    cvtf16rz  the truncating converts dropped the tail and kept the infinities"
else
  echo "FAIL  cvtf16rz  read $(head -c 20 "$TMP/cvtf16rz.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtf16.ptx cvtsat --grid 1 --threads 1 --buffers 6
"$WEFT" run kernels/cvtf16.ptx --kernel cvtsat --grid 1 --threads 1 --buffers 6 \
    --cpu-only --dump 0="$TMP/cvtsat.bin" >/dev/null
if [ "$(head -c 24 "$TMP/cvtsat.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "3f800000 00000000 00000000 3e800000 00003c00 00000000 " ]; then
  echo "ok    cvtsat  .sat clamped to [0,1] and flushed the nan to positive zero"
else
  echo "FAIL  cvtsat  read $(head -c 24 "$TMP/cvtsat.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

python3 - "$TMP" <<'PYEOF'
import struct, sys
d = sys.argv[1]
def h(x): return struct.pack('<e', x)
vals = [0.5, 1.5, -2.0, 4.0, 0.25, -0.5, 8.0, -0.0]
open(d + '/hin.bin', 'wb').write(b''.join(h(v) for v in vals))
vals2 = vals + [3.0, -3.0, 0.125, 6.0, -0.75, 2.5, 1.0, 16.0]
open(d + '/hin2.bin', 'wb').write(b''.join(h(v) for v in vals2))
v32 = [(i % 7 - 3) * 0.5 for i in range(32)]
open(d + '/hin32.bin', 'wb').write(b''.join(h(v) for v in v32))
def sat(x): return 0.0 if x != x else (0.0 if x < 0 else (1.0 if x > 1 else x))
exp = []
for b in vals:
    # The two `fma` ties: 1 + 2^-11 is halfway between 1 and 1+2^-10 and takes the
    # even one; 1 + 3*2^-11 is halfway one ulp up and takes two ulp. Both are one
    # rounding of the exact product and sum, which a float intermediate is not.
    exp += [1 + b, 1 - b, 0.5 * b, 2 * b + 1, min(1, b), max(1, b), -b, abs(b),
            sat(0.75 + b), sat(b), 1 + 2**-11, 1 + 2**-10 + 2**-11,
            65504.0, 32752.0, 2.0**-23, 0.0]
open(d + '/exp_ha.bin', 'wb').write(b''.join(h(v) for v in exp))
w = []
for i in range(8):
    lo, hi = vals2[2 * i], vals2[2 * i + 1]
    for op in (lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y,
               lambda x, y: y * x + y, min, max):
        w.append(h(op(lo, 0.5)) + h(op(hi, 1.0)))
open(d + '/exp_ha2.bin', 'wb').write(b''.join(w))
open(d + '/exp_hs.bin', 'wb').write(b''.join(h(v32[(i + 1) % 32] + 1.0) for i in range(32)))
open(d + '/exp_spvh.bin', 'wb').write(b''.join(h(v32[(i + 1) % 32] * 2 + 2.0) for i in range(32)))
# bf16 is the top half of a float, so the reference is the same float arithmetic
# rounded once, nearest even, at the 8-significand-bit grid the hardware rounds at.
def bfb(x):
    u = struct.unpack('<I', struct.pack('<f', x))[0]
    if (u & 0x7F800000) != 0x7F800000:
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return struct.pack('<H', (u >> 16) & 0xFFFF)
def bfv(b):
    return struct.unpack('<f', struct.pack('<I', b << 16))[0]
open(d + '/hbin.bin', 'wb').write(b''.join(bfb(v) for v in vals))
open(d + '/hbin2.bin', 'wb').write(b''.join(bfb(v) for v in vals2))
mx, mn = bfv(0x7F7F), bfv(0x0001)
expb = []
for b in vals:
    # The two fused ties: 1 + 2^-9 is halfway between 1 and 1+2^-8 and takes the even
    # one; 1 + 3*2^-9 is halfway one ulp up and takes two ulp.
    expb += [1 + b, 1 - b, 0.5 * b, 2 * b + 1, min(1, b), max(1, b), -b, abs(b),
             sat(0.75 + b), sat(b), 1 + 2**-9, 1 + 2**-8 + 2**-9, mx + 1.0,
             0.5 * mx, 2 * mn, 0.0]
open(d + '/exp_bf.bin', 'wb').write(b''.join(bfb(v) for v in expb))
wb = []
for i in range(8):
    lo, hi = vals2[2 * i], vals2[2 * i + 1]
    for op in (lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y,
               lambda x, y: y * x + y, min, max):
        wb.append(bfb(op(lo, 0.5)) + bfb(op(hi, 1.0)))
open(d + '/exp_bf2.bin', 'wb').write(b''.join(wb))
open(d + '/exp_hmin.bin', 'wb').write(b''.join(struct.pack('<H', v) for v in
    [0x7FFF, 0x3C00, 0xBC00, 0xC000, 0x8000, 0xC200]))
# Division, reciprocal and square root of a narrow float, against an exact grid: the
# expected value is the nearest representable one, found by comparing integers, not a
# second copy of the significand routine the C++ side runs. A half's grid is uniform at
# 2^-24 and a bf16's at 2^-133, so every value is an integer count of that step.
from bisect import bisect_left
def narrow_grid(mbits, ebits):
    lead = 1 << mbits
    xmax = (1 << ebits) - 1
    sube = 1 - ((1 << (ebits - 1)) - 1) - mbits
    ks = []
    for b in range(xmax << mbits):
        e = (b >> mbits) & xmax
        f = b & (lead - 1)
        ks.append(f if e == 0 else (f + lead) << (e - 1))
    return ks, -sube
def nearest_k(K, scale, A, B=1):
    tgt = A << scale
    row = [k * B for k in K]
    i = bisect_left(row, tgt)
    if i < len(row) and row[i] == tgt: return i
    if i == 0: return 0
    if i == len(row): return len(row) - 1
    lo, hi = i - 1, i
    d1, d2 = tgt - row[lo], row[hi] - tgt
    if d1 != d2: return lo if d1 < d2 else hi
    return lo if (lo & 1) == 0 else hi
def nearest_sqrt(K, scale, ka):
    tgt = ka << scale
    row = [k * k for k in K]
    i = bisect_left(row, tgt)
    if i < len(row) and row[i] == tgt: return i
    lo, hi = i - 1, i
    if lo < 0: return 0
    if hi >= len(row): return len(row) - 1
    four, mid = 4 * tgt, (K[lo] + K[hi]) ** 2
    if four != mid: return lo if four < mid else hi
    return lo if (lo & 1) == 0 else hi
HK, HS = narrow_grid(10, 5)
BK, BS = narrow_grid(7, 8)
def q16(a, b): return nearest_k(HK, HS, HK[a], HK[b])
def r16(a): return nearest_sqrt(HK, HS, HK[a])
def qbf(a, b): return nearest_k(BK, BS, BK[a], BK[b])
def rbf(a): return nearest_sqrt(BK, BS, BK[a])
nd = [q16(0x4000, 0x4200), q16(0x3C00, 0x4200), q16(0x3C00, 0x4700),
      q16(0x3C00, 0x3C01), q16(0x3C00, 0x4200), q16(0x3C00, 0x3C02),
      r16(0x4000), r16(0x4200), r16(0x7BFF), r16(0x0001),
      q16(0x0001, 0x4200), q16(0x0001, 0x3E00), q16(0x3C00, 0x7BFF),
      0x7C00, 0x7FFF, 0x7FFF]
open(d + '/exp_nd.bin', 'wb').write(b''.join(struct.pack('<H', v) for v in nd))
open(d + '/exp_nd2.bin', 'wb').write(b''.join(struct.pack('<H', v) for v in
    [0x0000, 0x8000, 0x7FFF, 0xFC00, 0x7C00, 0x8000,
     q16(0x0001, 0x4000), r16(0x3C01)]))
open(d + '/exp_bfd.bin', 'wb').write(b''.join(struct.pack('<H', v) for v in
    [qbf(0x4000, 0x4040), qbf(0x3F80, 0x4040), qbf(0x3F80, 0x3F81),
     qbf(0x3F80, 0x4040), rbf(0x4000), rbf(0x7F7F), qbf(0x0001, 0x4040), 0x7FFF]))
# Comparisons are of the values, so the reference compares the halfs as Python floats
# (exact for all of these) and writes the masks `set` answers: all ones or zero.
cv = [-2.0, -0.5, -0.0, 0.0, 0.25, 1.0, 2.0, 65504.0]
open(d + '/hcmpin.bin', 'wb').write(b''.join(h(v) for v in cv))
cmp = []
for x in cv:
    cmp += [int(x < 1.0), int(x >= -0.5), 0xFFFFFFFF, 0xFFFFFFFF if x != 0.0 else 0,
            int(x != x), int(x == x), 0xFFFFFFFF if x <= -0.0 else 0, int(x > -2.0)]
open(d + '/exp_cmp.bin', 'wb').write(b''.join(struct.pack('<I', v) for v in cmp))
# Four small angles and four a half can hold up to its top: the reduction above 256
# used to hand every one of those lanes to the fp64 text, so the reference has to see them.
tv = [0.5, 2.0, 0.25, 3.0, 1000.0, 4000.0, 16000.0, 65504.0]
open(d + '/htranin.bin', 'wb').write(b''.join(h(v) for v in tv))
# A double's tangent, for the one `tan` the isa has.
tv32 = [0.5, 1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 3.0]
open(d + '/tan32in.bin', 'wb').write(b''.join(struct.pack('<f', v) for v in tv32))
# bf16 comparisons, and the packed forms. A packed `set` answers a 0xFFFF per half
# that is true; a lane equal to the pair (1.0, 0.5) answers all ones.
bv = [-3.0, -1.0, -0.0, 0.5, 1.0, 2.0, 100.0, -7.5]
open(d + '/hbfcmpin.bin', 'wb').write(b''.join(bfb(v) for v in bv))
bc = []
for x in bv:
    bc += [int(x < 1.0), int(x >= -1.0), 0xFFFFFFFF if x != 0.0 else 0, 0xFFFFFFFF]
open(d + '/exp_bfcmp.bin', 'wb').write(b''.join(struct.pack('<I', v) for v in bc))
open(d + '/hpairin.bin', 'wb').write(b''.join(h(v) for v in vals2))
pc = []
for i in range(0, len(vals2), 2):
    lo, hi = vals2[i], vals2[i + 1]
    pc += [(0xFFFF if lo == 0.5 else 0) | (0xFFFF0000 if hi == 1.0 else 0),
           int(lo < 0.5), int(hi < 1.0)]
open(d + '/exp_pcmp.bin', 'wb').write(b''.join(struct.pack('<I', v) for v in pc))
# 64 lanes add 1.0 to the low half of one word and 0.5 to the high one: 64.0 and
# 32.0, and the packed form adds (1.0, 0.5) sixty-four times.
# The atomics run over a word that starts at zero, so the totals are the sum and
# nothing else, and the word beside it has to stay zero: the second word of the
# narrowatom buffer is compared too, which is the neighbour the atom must not touch.
open(d + '/exp_natom.bin', 'wb').write(struct.pack('<HHHH', 0x5400, 0x5000, 0, 0))
open(d + '/exp_ratom.bin', 'wb').write(struct.pack('<I', 0x54005000))
open(d + '/zatom8.bin', 'wb').write(b'\x00' * 8)
open(d + '/zatom4.bin', 'wb').write(b'\x00' * 4)
# slct: the sign of c picks the argument, and -0.0 is not less than zero, so it
# picks the first one. The float pair is 111 and 222, a number that is neither.
open(d + '/slin.bin', 'wb').write(struct.pack('<iiii', 5, -5, 0, 0))
open(d + '/slinf.bin', 'wb').write(struct.pack('<ffff', 1.5, -2.5, 0.0, -0.0))
open(d + '/exp_slcti.bin', 'wb').write(struct.pack('<IIII', 0x3C00, 0x4000, 0x3C00, 0x3C00))
open(d + '/exp_slctf.bin', 'wb').write(struct.pack('<IIII', 111, 222, 111, 111))
# atom.cas.b16: 1.0 above 0.5 in one word. A matching compare swaps the low half to
# 2.0 and leaves 0.5 alone, a miss writes nothing, and either way the half that was
# there comes back, so buffer 1 holds 1.0 in both cases.
open(d + '/casin.bin', 'wb').write(struct.pack('<I', 0x38003C00))
open(d + '/exp_cas16.bin', 'wb').write(struct.pack('<I', 0x38004000))
open(d + '/exp_cas16miss.bin', 'wb').write(struct.pack('<I', 0x38003C00))
open(d + '/exp_cas16ret.bin', 'wb').write(struct.pack('<I', 0x3C00))
# atom.inc and atom.dec count round a cycle bounded by 3: from 0, three steps are 3
# and five wrap back to 1; from 3, three steps down are 0 and from 0, two steps wrap
# up to the bound and then down to 2. A translation that never wrote the word would
# answer with the start every time, so each of these is a different number.
open(d + '/inc0.bin', 'wb').write(struct.pack('<I', 0))
open(d + '/inc3in.bin', 'wb').write(struct.pack('<I', 3))
open(d + '/exp_inc3.bin', 'wb').write(struct.pack('<I', 3))
open(d + '/exp_inc5.bin', 'wb').write(struct.pack('<I', 1))
open(d + '/exp_dec3.bin', 'wb').write(struct.pack('<I', 0))
open(d + '/exp_dec2.bin', 'wb').write(struct.pack('<I', 2))
open(d + '/exp_incgrab.bin', 'wb').write(struct.pack('<I', 0))
# 16-bit integer atomics through SPIR-V: the buffer is the operand half and a 0xABCD
# sentinel above it, so writing the whole 32-bit word shows up as a wrong top half.
# The mix is and, or, xor, a signed min of -2, an unsigned min of 0x0100, an exchange
# to 0x1234, and a compare-exchange that matches it.
def sm(v):
    return v - 0x10000 if v >= 0x8000 else v
open(d + '/a16in.bin', 'wb').write(struct.pack('<HH', 0x0001, 0xABCD))
open(d + '/a16in0.bin', 'wb').write(struct.pack('<HH', 0x0000, 0xABCD))
open(d + '/exp_a16add.bin', 'wb').write(struct.pack('<HH', 8, 0xABCD))
vmix = 0x0001 & 0x00FF
vmix |= 0x3000
vmix ^= 0x0001
vmix = min(sm(vmix), sm(0xFFFE)) & 0xFFFF
vmix = min(vmix, 0x0100)
vmix = 0x1234
vmix = 0xBEEF if vmix == 0x1234 else vmix
open(d + '/exp_a16mix.bin', 'wb').write(struct.pack('<HH', vmix, 0xABCD))
# OpSwitch: the selector picks a label out of a table whose default comes first and
# whose literals are as wide as the selector. `sw64`'s literals have bits above 32
# and one input is 0xFFFFFFFF00000001, whose low half is the case `1`: a front end
# that reads one word per literal lands on the wrong entry and writes its number.
open(d + '/sw32in.bin', 'wb').write(struct.pack('<8I', 0, 7, 1, 99, 2, 7, 0, 5))
open(d + '/exp_sw32.bin', 'wb').write(struct.pack('<8I', 10, 12, 11, 99, 99, 12, 10, 99))
open(d + '/sw64in.bin', 'wb').write(b''.join(struct.pack('<Q', v) for v in
    [1, 4294967298, 4294967295, 7, 0xFFFFFFFF00000001, 0, 1, 4294967298]))
open(d + '/exp_sw64.bin', 'wb').write(b''.join(struct.pack('<Q', v) for v in
    [21, 20, 22, 99, 99, 99, 21, 20]))
# The signed selector: `(int)gid - 3` selects with -3, -2, -1 and 0, and the cases
# are -1 and -2. A literal is 0xFFFFFFFF, so a comparison that sign-extends the
# selector to 64 bits first misses its own case and answers the default: lane 2
# writes 9 where the module asked for 5, and lane 1 writes 9 instead of 6.
open(d + '/exp_sws.bin', 'wb').write(struct.pack('<4i', 9, 6, 5, 9))
# brx.idx: the index picks a label out of the `.branchtargets` table, and an index
# past the end falls through to the instruction after it. `brxpred` puts the same
# table in the compact `M<3>` form behind a guard, so an index of 3 or more is 99
# and 2 is 22: a table that ignored the guard, or a range that expanded to the
# wrong labels, answers a number the reference does not.
open(d + '/brxin.bin', 'wb').write(struct.pack('<8I', 0, 1, 2, 3, 2, 0, 1, 9))
open(d + '/exp_brx.bin', 'wb').write(struct.pack('<8I', 10, 11, 12, 99, 12, 10, 11, 99))
open(d + '/exp_brxp.bin', 'wb').write(struct.pack('<8I', 20, 21, 22, 99, 22, 20, 21, 99))
# Bit scans. `bfind` is the highest set bit, counted from 0, or 0xFFFFFFFF when
# there is none; `.s32` searches for the highest bit that differs from the sign, so
# a negative value is searched inverted and 0 and -1 have none; `fns` is the nth
# set bit at or above a starting position. Both references are computed from the
# inputs rather than written out, so the expectation follows the definition.
def bfind_u(v):
    return 0xFFFFFFFF if v == 0 else v.bit_length() - 1
def bfind_s(v):
    a = (~v) & 0xFFFFFFFF if v >= 0x80000000 else v
    return 0xFFFFFFFF if a == 0 else a.bit_length() - 1
def fns(mask, base, off):
    for p in range(base & 31, 32):
        if (mask >> p) & 1:
            if off == 0:
                return p
            off -= 1
    return 0xFFFFFFFF
bi = [0, 1, 0x80000000, 0x00010000, 0xFFFFFFFF, 0x00000002, 0x40000000, 0x0000000C]
open(d + '/bfindu_in.bin', 'wb').write(struct.pack('<8I', *bi))
open(d + '/exp_bfindu.bin', 'wb').write(struct.pack('<8I', *[bfind_u(v) for v in bi]))
bs = [0xFFFFFFFF, 0, 1, 0xFFFFFFFE, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFD, 5]
open(d + '/bfinds_in.bin', 'wb').write(struct.pack('<8I', *bs))
open(d + '/exp_bfinds.bin', 'wb').write(struct.pack('<8I', *[bfind_s(v) for v in bs]))
fm = [0x0000000B, 0x000000F0, 0, 0xFFFFFFFF, 0x0F0F0F0F, 0x80000000, 0x0000FFFF, 1]
open(d + '/fns_in.bin', 'wb').write(struct.pack('<8I', *fm))
open(d + '/exp_fns3.bin', 'wb').write(struct.pack('<8I', *[fns(m, 0, 2) for m in fm]))
open(d + '/exp_fnsb4.bin', 'wb').write(struct.pack('<8I', *[fns(m, 4, 1) for m in fm]))
# A half register declared `.reg .f16`: doubling is exact until it overflows, so
# 65504 doubles to infinity, a subnormal doubles to the next subnormal, and -0
# stays -0.
open(d + '/h16in.bin', 'wb').write(struct.pack('<8H', 0x3C00, 0x3800, 0x4000, 0xBC00,
                                               0x7BFF, 0x0001, 0x7C00, 0x8000))
open(d + '/exp_h16reg.bin', 'wb').write(struct.pack('<8H', 0x4000, 0x3C00, 0x4400, 0xC000,
                                                    0x7C00, 0x0002, 0x7C00, 0x8000))
# Warp-wide agreement: `elect.sync` picks the lowest live lane of the mask, so the
# elected one adds 1000, and `match` answers the mask of lanes holding this lane's
# value. Lanes 8 to 31 never ran, so the member mask naming them changes nothing.
wm = [3, 3, 5, 3, 5, 7, 7, 3]
open(d + '/wmmix.bin', 'wb').write(struct.pack('<8I', *wm))
open(d + '/wmeq.bin', 'wb').write(struct.pack('<8I', *([4] * 8)))
open(d + '/exp_elect.bin', 'wb').write(struct.pack('<8I', 1000, 0, 0, 0, 0, 0, 0, 0))
same = []
for v in wm:
    m = 0
    for k, w in enumerate(wm):
        if w == v:
            m |= 1 << k
    same.append(m)
open(d + '/exp_matchny.bin', 'wb').write(struct.pack('<8I', *same))
open(d + '/exp_matchall.bin', 'wb').write(struct.pack('<8I', *([0] * 8)))
open(d + '/exp_matchallq.bin', 'wb').write(struct.pack('<8I', *([0xFFFFFFFF] * 8)))
open(d + '/exp_matchallp.bin', 'wb').write(struct.pack('<8I', *([3] * 8)))
open(d + '/exp_matchallpq.bin', 'wb').write(struct.pack('<8I', *([7] * 8)))
# dp2a: two 16-bit lanes of a against two bytes of b, `.lo` taking bytes 0 and 1 and
# `.hi` bytes 2 and 3, each lane folded in at the signedness its suffix names. The
# reference is the definition, so an implementation that used the wrong byte pair,
# or that zero-extended a signed lane, answers a number this does not.
def s16(v):
    return v - 0x10000 if v & 0x8000 else v
def s8(v):
    return v - 0x100 if v & 0x80 else v
def dp2a(a, b, hi, saa, sbb):
    a0, a1 = a & 0xFFFF, (a >> 16) & 0xFFFF
    if saa:
        a0, a1 = s16(a0), s16(a1)
    base = 2 if hi else 0
    b0, b1 = (b >> (8 * base)) & 0xFF, (b >> (8 * (base + 1))) & 0xFF
    if sbb:
        b0, b1 = s8(b0), s8(b1)
    return (a0 * b0 + a1 * b1) & 0xFFFFFFFF
dpa = [0x00020003, 0x0000FFFF, 0xFFFF0001, 0x80007FFF, 0x12345678, 1, 0xFFFFFFFF, 0x00010002]
dpb = [0x04030201, 0xFF0201FF, 0x00000000, 0xFFFFFFFF, 0x807F7F80, 0x01020304, 0x00000005, 0xFE00FF01]
open(d + '/dpin.bin', 'wb').write(struct.pack('<16I', *(dpa + dpb)))
open(d + '/exp_dp2alo.bin', 'wb').write(struct.pack('<8I', *[dp2a(x, y, False, False, False) for x, y in zip(dpa, dpb)]))
open(d + '/exp_dp2ahi.bin', 'wb').write(struct.pack('<8I', *[dp2a(x, y, True, False, False) for x, y in zip(dpa, dpb)]))
open(d + '/exp_dp2aslo.bin', 'wb').write(struct.pack('<8I', *[dp2a(x, y, False, True, True) for x, y in zip(dpa, dpb)]))
open(d + '/exp_dp2ashi.bin', 'wb').write(struct.pack('<8I', *[dp2a(x, y, True, True, True) for x, y in zip(dpa, dpb)]))
# movmatrix: lane l holds row l>>2 at columns 2(l&3) and 2(l&3)+1, and the result is
# that fragment of the transposed matrix. The input is a fragment, so the reference
# transposes the matrix it encodes rather than the words.
def vfrag(m):
    w = []
    for l in range(32):
        r, c = l >> 2, 2 * (l & 3)
        w.append(m[r][c] | (m[r][c + 1] << 16))
    return w
vmi = [[(r * 8 + c + 1) & 0xFFFF for c in range(8)] for r in range(8)]
vmt = [[vmi[j][i] for j in range(8)] for i in range(8)]
open(d + '/mvin.bin', 'wb').write(struct.pack('<32I', *vfrag(vmi)))
open(d + '/exp_movtx.bin', 'wb').write(struct.pack('<32I', *vfrag(vmt)))
# Video instructions: packed lanes that do not carry into their neighbour. Each lane is
# widened at the signedness its suffix names, the operation runs in 32 bits, and the
# answer is cut back to the lane width.
def vlanes(w, n, width, sgn):
    m = (1 << width) - 1
    out = []
    for i in range(n):
        v = (w >> (width * i)) & m
        if sgn and v >> (width - 1):
            v -= 1 << width
        out.append(v)
    return out
def vpacked(ts, width):
    m = (1 << width) - 1
    acc = 0
    for i, t in enumerate(ts):
        acc |= (t & m) << (width * i)
    return acc
def vcal(name, a, b, n, width, sa, sb):
    x, y = vlanes(a, n, width, sa), vlanes(b, n, width, sb)
    if name == 'add':
        ts = [p + q for p, q in zip(x, y)]
    elif name == 'min':
        ts = [min(p, q) for p, q in zip(x, y)]
    elif name == 'absdiff':
        ts = [abs(p - q) for p, q in zip(x, y)]
    else:
        ts = [(p + q + 1) >> 1 if p + q >= 0 else (p + q) >> 1 for p, q in zip(x, y)]
    return vpacked(ts, width)
vva = [0x00010002, 0xFFFF0001, 0x80007FFF, 0x12345678, 0xFFFFFFFF, 0x0000FFFF, 0x7FFF8000, 0x00020005]
vvb = [0x00030004, 0x0002FFFF, 0x7FFF8000, 0xF0E0D0C0, 0x01010101, 0xFFFF0000, 0x80007FFF, 0x00050002]
open(d + '/vidin.bin', 'wb').write(struct.pack('<16I', *(vva + vvb)))
# multimem: this runtime binds one replica, so a reduce across the replicas of a value
# is that value under every operation, and a multicast store writes the one copy there
# is. All three kernels therefore answer the input unchanged.
mmv = [0x10203040, 0x00000001, 0xFFFFFFFE, 0x7FFFFFFF, 0x80000000, 0x0BADF00D, 0x0000FFFF, 0xAAAAAAAA]
open(d + '/mmin.bin', 'wb').write(struct.pack('<8I', *mmv))
open(d + '/mmexp.bin', 'wb').write(struct.pack('<8I', *mmv))
# b128: one 16-byte value, two 64-bit halves, read and written as one access. Four
# threads each move one value, so the dump only matches when every lane's sixteen
# bytes arrive.
bv = [0x1122334455667788, 0x99AABBCCDDEEFF00, 0x0123456789ABCDEF, 0xFEDCBA9876543210,
      0xFFFFFFFFFFFFFFFF, 0x0F0F0F0F0F0F0F0F, 0xDEADBEEFCAFEF00D, 0x0000000000000001]
open(d + '/b128in.bin', 'wb').write(struct.pack('<8Q', *bv))
open(d + '/b128exp.bin', 'wb').write(struct.pack('<8Q', *bv))
# call.uni: the callee adds one through its own pointer argument. The answer is the
# input plus one; a pasted body that shared the caller's register file would lose
# the pointer to the word it had read.
cv = [0x10, 0x20, 0x7FFFFFFF, 0xFFFFFFFF, 0, 1, 0xDEADBEEF, 0x12345678]
open(d + '/callin.bin', 'wb').write(struct.pack('<8I', *cv))
open(d + '/callexp.bin', 'wb').write(struct.pack('<8I', *[(v + 1) & 0xFFFFFFFF for v in cv]))
for vn, vc, vn_lanes, vwd, vsa, vsb in [
    ('vadd2u', 'add', 2, 16, False, False),
    ('vadd2s', 'add', 2, 16, True, True),
    ('vmin4s', 'min', 4, 8, True, True),
    ('vabs4u', 'absdiff', 4, 8, False, False),
    ('vadd4u', 'add', 4, 8, False, False),
    ('vavg4s', 'avg', 4, 8, True, True),
]:
    open(d + '/exp_' + vn + '.bin', 'wb').write(
        struct.pack('<8I', *[vcal(vc, x, y, vn_lanes, vwd, vsa, vsb) for x, y in zip(vva, vvb)]))
PYEOF
narrowref() {  # name, file, kernel, grid, threads, buffers, dump index, input, expected
  local name="$1" file="$2" kern="$3" grid="$4" thr="$5" bufs="$6" idx="$7" in="$8"
  local exp="$9"
  if [ -n "$in" ]; then
    "$WEFT" run "$file" --kernel "$kern" --grid "$grid" --threads "$thr" \
        --buffers "$bufs" --load 0="$TMP/$in" --cpu-only --dump "$idx=$TMP/$name.bin" >/dev/null
  else
    "$WEFT" run "$file" --kernel "$kern" --grid "$grid" --threads "$thr" \
        --buffers "$bufs" --cpu-only --dump "$idx=$TMP/$name.bin" >/dev/null
  fi
  if cmp -s "$TMP/$name.bin" "$TMP/$exp"; then
    echo "ok    $name  the narrow arithmetic is the grid, bit for bit"
  else
    echo "FAIL  $name  $(cmp "$TMP/$name.bin" "$TMP/$exp" 2>&1 | head -1)"; fail=1
  fi
}
slctref() {  # name, file, kernel, grid, threads, buffers, dump index, input, expected
  local name="$1" file="$2" kern="$3" grid="$4" thr="$5" bufs="$6" idx="$7" in="$8"
  local exp="$9"
  "$WEFT" run "$file" --kernel "$kern" --grid "$grid" --threads "$thr" \
      --buffers "$bufs" --load 0="$TMP/$in" --cpu-only --dump "$idx=$TMP/$name.bin" >/dev/null
  if cmp -s "$TMP/$name.bin" "$TMP/$exp"; then
    echo "ok    $name  slct picks the argument the sign of c asks for, bit for bit"
  else
    echo "FAIL  $name  $(cmp "$TMP/$name.bin" "$TMP/$exp" 2>&1 | head -1)"; fail=1
  fi
}
narrowref halfops    kernels/halfops.ptx halfops    8  8  8,128  1 hin.bin   exp_ha.bin
narrowref halfops2   kernels/halfops2.ptx halfops2   8  8  8,48   1 hin2.bin  exp_ha2.bin
narrowref halfops_tg kernels/halfops3.ptx halfops_tg 32 32 32,32  1 hin32.bin exp_hs.bin
narrowref halfmin    kernels/halfops4.ptx halfmin    1  1  6      0 ""        exp_hmin.bin
narrowref bf16ops    kernels/bf16ops.ptx bf16ops    8  8  8,128  1 hbin.bin  exp_bf.bin
narrowref bf16ops2   kernels/bf16ops.ptx bf16ops2   8  8  8,48   1 hbin2.bin exp_bf2.bin
narrowref narrowdiv  kernels/narrowdiv.ptx narrowdiv  1 1 16  0 "" exp_nd.bin
narrowref narrowdiv2 kernels/narrowdiv.ptx narrowdiv2 1 1 8   0 "" exp_nd2.bin
narrowref bf16div    kernels/bf16div.ptx bf16div      1 1 8   0 "" exp_bfd.bin
narrowref narrowcmp  kernels/narrowcmp.ptx narrowcmp 8 8 8,64 1 hcmpin.bin exp_cmp.bin
narrowref narrowcmp_bf16 kernels/narrowcmp2.ptx narrowcmp_bf16 8 8 8,32 1 hbfcmpin.bin exp_bfcmp.bin
narrowref narrowcmp2 kernels/narrowcmp3.ptx narrowcmp2 8 8 8,24 1 hpairin.bin exp_pcmp.bin
narrowref narrowatom kernels/narrowatom.ptx narrowatom 64 64 2 0 zatom8.bin exp_natom.bin
narrowref redatom    kernels/narrowatom2.ptx redatom    64 64 1 0 zatom4.bin exp_ratom.bin
slctref  slcti       kernels/slct.ptx slcti 4 4 4,4 1 slin.bin  exp_slcti.bin
slctref  slctf       kernels/slct.ptx slctf 4 4 4,4 1 slinf.bin exp_slctf.bin
narrowref cas16     kernels/casb16.ptx casb16     1 1 1,1 0 casin.bin exp_cas16.bin
narrowref cas16miss kernels/casb16.ptx casb16miss 1 1 1,1 0 casin.bin exp_cas16miss.bin
narrowref cas16ret  kernels/casb16.ptx casb16     1 1 1,1 1 casin.bin exp_cas16ret.bin
narrowref cas16race kernels/casb16.ptx casb16race 8 8 1   0 casin.bin exp_cas16.bin
python3 - "$TMP" <<'PYH'
import struct, sys, os
d = sys.argv[1]
def w(n, *vals):
    open(os.path.join(d, n), 'wb').write(b''.join(struct.pack('<I', v) for v in vals))
w('hvmax_in.bin', 0x44003C00);  w('hvmax_ex.bin', 0x44003E00)
w('hvmin_in.bin', 0x44003C00);  w('hvmin_ex.bin', 0x41003C00)
w('hvadd_in.bin', 0x00000000);  w('hvadd_ex.bin', 0x44004000)
w('hvbf_in.bin',  0x00000000);  w('hvbf_ex.bin',  0x41804100)
w('hvpack_in.bin', 0x3C003C00, 0x42004200)
w('hvpack_ex.bin', 0x3C004000, 0x42004200)
PYH
narrowref hvmax  kernels/halfvecatom.ptx hvmax  8 8 1 0 hvmax_in.bin  hvmax_ex.bin
narrowref hvmin  kernels/halfvecatom.ptx hvmin  8 8 1 0 hvmin_in.bin  hvmin_ex.bin
narrowref hvadd  kernels/halfvecatom.ptx hvadd  8 8 1 0 hvadd_in.bin  hvadd_ex.bin
narrowref hvbf   kernels/halfvecatom.ptx hvbf   8 8 1 0 hvbf_in.bin   hvbf_ex.bin
narrowref hvpack kernels/halfvecatom.ptx hvpack 8 8 2 0 hvpack_in.bin hvpack_ex.bin
python3 - "$TMP" <<'PYC'
import struct, sys, os
d = sys.argv[1]
out = []
for l in range(8):
    m = (l + 1) % 8
    out += [10 * m + 1, 10 * m + 4, 10 * m + 7, 10 * m + 10]
open(os.path.join(d, 'sclv4_ex.bin'), 'wb').write(b''.join(struct.pack('<I', v) for v in out))
open(os.path.join(d, 'sclmapa_ex.bin'), 'wb').write(
    b''.join(struct.pack('<I', l * 5 + 3) for l in range(8)))
PYC
narrowref sclv4   kernels/scluster.ptx sclv4   8 8 32 0 "" sclv4_ex.bin
narrowref sclmapa kernels/scluster.ptx sclmapa 8 8 8  0 "" sclmapa_ex.bin
python3 - "$TMP" <<'PYA'
import struct, sys, os
d = sys.argv[1]
open(os.path.join(d, 'satadd_ex.bin'), 'wb').write(
    struct.pack('<IIII', 8, 0x40800000, 7, 1))
open(os.path.join(d, 'satb16_ex.bin'), 'wb').write(struct.pack('<I', 0x38004000))
open(os.path.join(d, 'satpack_ex.bin'), 'wb').write(struct.pack('<I', 0x48004800))
open(os.path.join(d, 'satret_ex.bin'), 'wb').write(struct.pack('<I', 28))
PYA
narrowref shatadd  kernels/shatom.ptx satadd  8 8 4 0 "" satadd_ex.bin
narrowref shatb16  kernels/shatom.ptx satb16  8 8 1 0 "" satb16_ex.bin
narrowref shatpack kernels/shatom.ptx satpack 8 8 1 0 "" satpack_ex.bin
narrowref shatret  kernels/shatom.ptx satret  8 8 1 0 "" satret_ex.bin
narrowref inc3     kernels/incdec.ptx inc     3 3 1   0 inc0.bin   exp_inc3.bin
narrowref inc5     kernels/incdec.ptx inc     5 5 1   0 inc0.bin   exp_inc5.bin
narrowref dec3     kernels/incdec.ptx dec     3 3 1   0 inc3in.bin exp_dec3.bin
narrowref dec2     kernels/incdec.ptx dec     2 2 1   0 inc0.bin   exp_dec2.bin
narrowref incgrab  kernels/incdec.ptx incgrab 1 1 1,1 1 inc0.bin   exp_incgrab.bin
narrowref redinc   kernels/incdec.ptx redinc  3 3 1   0 inc0.bin   exp_inc3.bin
narrowref a16add    kernels/spirv/atom16.spv a16add 8 8 2 0 a16in0.bin exp_a16add.bin
narrowref a16mix    kernels/spirv/atom16.spv a16mix 1 1 2 0 a16in.bin exp_a16mix.bin
narrowref sw32      kernels/spirv/switch.spv sw32 8 8 8,8 1 sw32in.bin exp_sw32.bin
narrowref sw64      kernels/spirv/switch.spv sw64 8 8 8,8 1 sw64in.bin exp_sw64.bin
narrowref sws       kernels/spirv/switchs.spv sws 4 4 4 0 "" exp_sws.bin
# An indirect branch through a jump table, on the PTX column.
"$WEFT" run kernels/brxidx.ptx --kernel brxidx --grid 8 --threads 8 --buffers 8,8 \
    --load 0="$TMP/brxin.bin" --cpu-only --dump 1="$TMP/brx.bin" >/dev/null
"$WEFT" run kernels/brxidx.ptx --kernel brxpred --grid 8 --threads 8 --buffers 8,8 \
    --load 0="$TMP/brxin.bin" --cpu-only --dump 1="$TMP/brxp.bin" >/dev/null
if cmp -s "$TMP/brx.bin" "$TMP/exp_brx.bin" && cmp -s "$TMP/brxp.bin" "$TMP/exp_brxp.bin"; then
  echo "ok    brxidx  a jump table picks the label the index names, guard included"
else
  echo "FAIL  brxidx  $(cmp "$TMP/brx.bin" "$TMP/exp_brx.bin" 2>&1 | head -1) $(cmp "$TMP/brxp.bin" "$TMP/exp_brxp.bin" 2>&1 | head -1)"; fail=1
fi
bitok=1
for row in "bfindu:bfindu_in:exp_bfindu" "bfinds:bfinds_in:exp_bfinds" \
           "fns3:fns_in:exp_fns3" "fnsbase:fns_in:exp_fnsb4" "f16reg:h16in:exp_h16reg"; do
  bkn="${row%%:*}"; brest="${row#*:}"; bin="${brest%%:*}"; bexp="${brest##*:}"
  "$WEFT" run kernels/bitfind.ptx --kernel "$bkn" --grid 8 --threads 8 --buffers 8,8 \
      --load 0="$TMP/$bin.bin" --cpu-only --dump 1="$TMP/bf_$bkn.bin" >/dev/null
  if ! cmp -s "$TMP/bf_$bkn.bin" "$TMP/$bexp.bin"; then
    echo "FAIL  bitfind:$bkn  $(cmp "$TMP/bf_$bkn.bin" "$TMP/$bexp.bin" 2>&1 | head -1)"; fail=1; bitok=0
  fi
done
[ "$bitok" = 1 ] && echo "ok    bitfind  bfind, fns and a .reg .f16 declaration are exact"
narrowref electk     kernels/warpmatch.ptx electk    8 8 8    0 ""       exp_elect.bin
narrowref matchny    kernels/warpmatch.ptx matchny   8 8 8,8  1 wmmix.bin exp_matchny.bin
narrowref matchalld  kernels/warpmatch.ptx matchalld 8 8 8,8  1 wmmix.bin exp_matchall.bin
narrowref matchalleq kernels/warpmatch.ptx matchalld 8 8 8,8  1 wmeq.bin  exp_matchallq.bin
narrowref matchallp  kernels/warpmatch.ptx matchallp 8 8 8,8  1 wmmix.bin exp_matchallp.bin
narrowref matchallpe kernels/warpmatch.ptx matchallp 8 8 8,8  1 wmeq.bin  exp_matchallpq.bin
narrowref dp2alo     kernels/dp2a.ptx dp2alo  8 8 16,8 1 dpin.bin exp_dp2alo.bin
narrowref dp2ahi     kernels/dp2a.ptx dp2ahi  8 8 16,8 1 dpin.bin exp_dp2ahi.bin
narrowref dp2aslo    kernels/dp2a.ptx dp2aslo 8 8 16,8 1 dpin.bin exp_dp2aslo.bin
narrowref dp2ashi    kernels/dp2a.ptx dp2ashi 8 8 16,8 1 dpin.bin exp_dp2ashi.bin
narrowref movmtx     kernels/movmatrix.ptx movmtx  32 32 32,32 1 mvin.bin exp_movtx.bin
narrowref movmtx2    kernels/movmatrix.ptx movmtx2 32 32 32,32 1 mvin.bin mvin.bin
narrowref vadd2u     kernels/video.ptx vadd2u  8 8 16,8 1 vidin.bin exp_vadd2u.bin
narrowref vadd2s     kernels/video.ptx vadd2s  8 8 16,8 1 vidin.bin exp_vadd2s.bin
narrowref vmin4s     kernels/video.ptx vmin4s  8 8 16,8 1 vidin.bin exp_vmin4s.bin
narrowref vabs4u     kernels/video.ptx vabs4u  8 8 16,8 1 vidin.bin exp_vabs4u.bin
narrowref vadd4u     kernels/video.ptx vadd4u  8 8 16,8 1 vidin.bin exp_vadd4u.bin
narrowref vavg4s     kernels/video.ptx vavg4s  8 8 16,8 1 vidin.bin exp_vavg4s.bin
narrowref mmld       kernels/multimem.ptx mmld  8 8 8,8 1 mmin.bin mmexp.bin
narrowref mmmin      kernels/multimem.ptx mmmin 8 8 8,8 1 mmin.bin mmexp.bin
narrowref mmst       kernels/multimem.ptx mmst  8 8 8,8 1 mmin.bin mmexp.bin
narrowref b128mv     kernels/b128.ptx b128mv 4 4 4,4 1 b128in.bin b128exp.bin
narrowref callk      kernels/call.ptx callk 8 8 8,8 1 callin.bin callexp.bin
python3 - "$TMP" <<'PYC'
import struct, sys, os
d = sys.argv[1]
vals = [3, 5, 12, 0, 7, 100, 1, 9]
exp = [v + 100 if v >= 10 else v * (v + 1) // 2 for v in vals]
open(os.path.join(d, 'callbrin.bin'), 'wb').write(struct.pack('<8I', *vals))
open(os.path.join(d, 'callbrexp.bin'), 'wb').write(struct.pack('<8I', *exp))
PYC
narrowref callbr     kernels/call.ptx callbr 8 8 8,8 1 callbrin.bin callbrexp.bin
python3 - "$TMP/morein.bin" "$TMP/moreexp.bin" <<'PY'
import struct, sys
ins = [0x7FFFFFFF, 0x00FFFFFF, 0xFFFF0000, 0x00000005,
       0x12345678, 0x800000FF, 0x00010000, 0xFFFFFFFF]
open(sys.argv[1], 'wb').write(struct.pack('<8I', *ins))
M = 0xFFFFFFFF
def s32(v):
    v &= M
    return v - (1 << 32) if v & 0x80000000 else v
def s24(v):
    v &= 0xFFFFFF
    return v - (1 << 24) if v & 0x800000 else v
def clamp(v, lo, hi):
    return max(lo, min(hi, v))
out = []
for x in ins:
    sad = sum(abs(((x >> (8 * k)) & 0xFF) - ((0x01020304 >> (8 * k)) & 0xFF))
              for k in range(4))
    lo24 = (s24(x) * s24(0x1234)) & M
    hi24 = ((s24(x) * s24(0x1234) + s32(x)) >> 32) & M
    packed = (0xFFFF0000
              | (clamp(s32(x), 0, 255) & 0xFF)
              | ((clamp(s32((x >> 8) & M), 0, 255) & 0xFF) << 8)) & M
    out += [sad & M, lo24, hi24, packed]
open(sys.argv[2], 'wb').write(struct.pack('<32I', *out))
PY
narrowref m4         kernels/more.ptx m4 8 8 8,32 1 morein.bin moreexp.bin
python3 - "$TMP/actexp.bin" <<'PY'
import struct, sys
open(sys.argv[1], 'wb').write(struct.pack('<32I', *[0xFFFFFFFF] * 32))
PY
narrowref actm      kernels/warpbits.ptx actm 32 32 32 0 "" actexp.bin
python3 - "$TMP/actexp8.bin" <<'PY'
import struct, sys
open(sys.argv[1], 'wb').write(struct.pack('<8I', *[0xFF] * 8))
PY
narrowref actm8     kernels/warpbits.ptx actm 8 8 8 0 "" actexp8.bin
python3 - "$TMP/wbin.bin" "$TMP/wbexp.bin" <<'PYX'
import struct, sys
words = [0x10000000 + i for i in range(32)]
m = [[0] * 8 for _ in range(8)]
for l in range(32):
    r, c = l >> 2, 2 * (l & 3)
    m[r][c], m[r][c + 1] = words[l] & 0xFFFF, words[l] >> 16
mt = [[m[j][i] for j in range(8)] for i in range(8)]
out = []
for l in range(32):
    r, c = l >> 2, 2 * (l & 3)
    out.append(mt[r][c] | (mt[r][c + 1] << 16))
open(sys.argv[1], 'wb').write(struct.pack('<32I', *words))
open(sys.argv[2], 'wb').write(struct.pack('<32I', *out))
PYX
narrowref mvi       kernels/warpbits.ptx mvi 32 32 32,32 1 wbin.bin wbexp.bin
python3 - "$TMP" <<'PYX'
import struct, math, sys, os
d = sys.argv[1]
vals = [i / 16.0 for i in range(1, 65)]
pack = lambda xs: b''.join(struct.pack('<e', x) for x in xs)
half = lambda v: struct.unpack('<e', struct.pack('<e', v))[0]
open(os.path.join(d, 'p16in.bin'), 'wb').write(pack(vals))
open(os.path.join(d, 'p16ex.bin'), 'wb').write(pack([2.0 ** half(v) for v in vals]))
open(os.path.join(d, 'p16lg.bin'), 'wb').write(pack([math.log2(half(v)) for v in vals]))
open(os.path.join(d, 'p16th.bin'), 'wb').write(pack([math.tanh(half(v)) for v in vals]))
PYX
narrowref p16ex2    kernels/f16x2_trans.ptx p16tx 32 32 32,32,32,32 1 p16in.bin p16ex.bin
narrowref p16lg2    kernels/f16x2_trans.ptx p16tx 32 32 32,32,32,32 2 p16in.bin p16lg.bin
narrowref p16tanh   kernels/f16x2_trans.ptx p16tx 32 32 32,32,32,32 3 p16in.bin p16th.bin

# suq: the surface's own extents, read from the binding the launcher made (64x64).
python3 - "$TMP/surfqexp.bin" <<'PY'
import struct, sys
open(sys.argv[1], 'wb').write(struct.pack('<4I', 64, 64, 1, 129))
PY
"$WEFT" run kernels/surfq.ptx --kernel surfq --grid 1 --threads 1 --buffers 4 \
    --textures 64x64 --cpu-only --dump 0="$TMP/surfq.bin" >/dev/null
if cmp -s "$TMP/surfq.bin" "$TMP/surfqexp.bin"; then
  echo "ok    surfq  a surface answers its own width, height and depth"
else
  echo "FAIL  surfq  $(cmp "$TMP/surfq.bin" "$TMP/surfqexp.bin" 2>&1 | head -1)"; fail=1
fi
"$WEFT" run kernels/narrowtran.ptx --kernel narrowtran --grid 8 --threads 8 --buffers 8,48 \
    --load 0="$TMP/htranin.bin" --cpu-only --dump 1="$TMP/ntran.bin" >/dev/null
if python3 - "$TMP/ntran.bin" <<'PY'
import struct, math, sys
raw = open(sys.argv[1], 'rb').read()
def val(b): return struct.unpack('<e', struct.pack('<H', b))[0]
def bits(x): return struct.unpack('<H', struct.pack('<e', x))[0]
vals = [0.5, 2.0, 0.25, 3.0, 1000.0, 4000.0, 16000.0, 65504.0]
refs = [lambda x: 2.0 ** x, lambda x: math.log2(x), lambda x: 1.0 / math.sqrt(x),
        math.sin, math.cos, math.tanh]
names = ['ex2', 'lg2', 'rsqrt', 'sin', 'cos', 'tanh']
got = struct.unpack('<%dH' % (len(raw) // 2), raw)
bad = 0
for i, x in enumerate(vals):
    for j, f in enumerate(refs):
        b = got[i * 6 + j]
        r = val(b)
        try:
            want = f(x)
        except OverflowError:      # 2**1000 is past a double, let alone a half
            want = float('inf')
        if math.isinf(want) or abs(want) > 65504.0:
            # The half grid ends there, so the reference's answer is the infinite one.
            if not math.isinf(r):
                print('  lane %d %s: got %r want %r' % (i, names[j], r, want))
                bad += 1
            continue
        ulp = abs(val(bits(r) + 1) - r)
        if not (abs(r - want) <= ulp * 1.0001 + 1e-12):
            print('  lane %d %s: got %r want %r (ulp %r)' % (i, names[j], r, want, ulp))
            bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    narrowtran  the approximate forms are within one ulp of the library"
else
  echo "FAIL  narrowtran  an approximate form is off by more than one ulp"; fail=1
fi
if python3 - "$WEFT" "$TMP" <<'PY'
import math, random, struct, subprocess, sys
weft, tmp = sys.argv[1], sys.argv[2]
def hround(v):
    try: return struct.unpack('<e', struct.pack('<e', v))[0]
    except OverflowError: return 65504.0 if v > 0 else -65504.0
def val(b): return struct.unpack('<e', struct.pack('<H', b))[0]
def bits(x): return struct.unpack('<H', struct.pack('<e', x))[0]
random.seed(20261107)
vals = []
while len(vals) < 256:
    v = hround((10.0 ** random.uniform(-22, 4.7)) * random.choice([1, -1]))
    if v != 0 and math.isfinite(v): vals.append(v)
refs = [lambda x: 2.0 ** x, math.log2, lambda x: 1.0 / math.sqrt(x), math.sin, math.cos, math.tanh]
names = ['ex2', 'lg2', 'rsqrt', 'sin', 'cos', 'tanh']
bad = 0
for gi in range(0, len(vals), 8):
    g = vals[gi:gi + 8]
    open(tmp + '/rndin.bin', 'wb').write(b''.join(struct.pack('<e', v) for v in g))
    subprocess.run([weft, 'run', 'kernels/narrowtran.ptx', '--kernel', 'narrowtran', '--grid', '8',
                    '--threads', '8', '--buffers', '8,48', '--load', '0=' + tmp + '/rndin.bin',
                    '--cpu-only', '--dump', '1=' + tmp + '/rndout.bin'], capture_output=True)
    got = struct.unpack('<48H', open(tmp + '/rndout.bin', 'rb').read())
    for i, x in enumerate(g):
        for j, f in enumerate(refs):
            r = val(got[i * 6 + j])
            try: want = f(x)
            except OverflowError: want = float('inf')
            except (ValueError, ZeroDivisionError): want = float('nan')
            if math.isnan(want): ok = math.isnan(r)
            elif math.isinf(want) or abs(want) > 65504.0: ok = math.isinf(r) and ((r > 0) == (want > 0))
            else:
                step = abs(val((bits(r) + 1) & 0xFFFF) - r) if (r == r and 0 < abs(r) < 65504.0) else 1e-3
                ok = abs(r - want) <= step * 1.0001 + 1e-12
            if not ok:
                print('  x=%r %s: got %r want %r' % (x, names[j], r, want)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    narrowrnd  256 random halfs answer a double, six ways each"
else
  echo "FAIL  narrowrnd  a random half disagrees with a double's answer"; fail=1
fi
python3 - "$TMP/lg_in.bin" <<'PY'
import struct, sys
vals = [0.5, 1.0, 2.0, 3.5, 10.0, -0.5, -1.5, 1e5, 1e-3, 1.001,
        -8.0089, 4.2e36, 0.0, -0.0, -1.0, -2.0, -3.0, -1e8, 2.5, -0.25]
open(sys.argv[1], 'wb').write(struct.pack('<20f', *vals))
PY
"$WEFT" run kernels/spirv/lgamma.spv --kernel lgamma --grid 20 --threads 20 --buffers 20,20 \
    --load 0="$TMP/lg_in.bin" --cpu-only --dump 1="$TMP/lg_out.bin" >/dev/null
if python3 - "$TMP/lg_out.bin" <<'PY'
import struct, math, sys
vals = [0.5, 1.0, 2.0, 3.5, 10.0, -0.5, -1.5, 1e5, 1e-3, 1.001,
        -8.0089, 4.2e36, 0.0, -0.0, -1.0, -2.0, -3.0, -1e8, 2.5, -0.25]
got = struct.unpack('<20f', open(sys.argv[1], 'rb').read())
def bits(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def roundf(x): return struct.unpack('<f', struct.pack('<f', x))[0]
bad, worst = 0, 0.0
for v, g in zip(vals, got):
    v = roundf(v)
    if v == 0.0 or (v == math.floor(v) and v < 0.0):
        if not math.isinf(g):
            print('  lgamma(%r) answered %r, not the pole' % (v, g)); bad += 1
        continue
    try:
        w = roundf(math.lgamma(v))
    except OverflowError:
        if not math.isinf(g):
            print('  lgamma(%r) answered %r, not the overflow' % (v, g)); bad += 1
        continue
    u = abs(bits(g) - bits(w))
    worst = max(worst, u)
    if u > 8:
        print('  lgamma(%r) got %r want %r, %d ulp' % (v, g, w, u)); bad += 1
print('  the worst of the rest is %.0f ulp, inside the 8 this holds it to' % worst)
sys.exit(1 if bad else 0)
PY
then
  echo "ok    lgamma  ln|Gamma| within 8 ulp, and the poles answer infinity"
else
  echo "FAIL  lgamma  a value or a pole is wrong"; fail=1
fi
"$WEFT" run kernels/spirv/lgammar.spv --kernel lgammar --grid 20 --threads 20 --buffers 20,20,20 \
    --load 0="$TMP/lg_in.bin" --cpu-only --dump 1="$TMP/lgr_v.bin" --dump 2="$TMP/lgr_s.bin" >/dev/null
if python3 - "$TMP/lgr_v.bin" "$TMP/lgr_s.bin" <<'PY'
import struct, math, sys
vals = [0.5, 1.0, 2.0, 3.5, 10.0, -0.5, -1.5, 1e5, 1e-3, 1.001,
        -8.0089, 4.2e36, 0.0, -0.0, -1.0, -2.0, -3.0, -1e8, 2.5, -0.25]
val = struct.unpack('<20f', open(sys.argv[1], 'rb').read())
sgn = struct.unpack('<20i', open(sys.argv[2], 'rb').read())
def roundf(x): return struct.unpack('<f', struct.pack('<f', x))[0]
bad = 0
for v, a, s in zip(vals, val, sgn):
    v = roundf(v)
    pole = v == 0.0 or (v == math.floor(v) and v < 0.0)
    want = 1 if (pole or v > 0.0) else (-1 if (int(math.floor(v)) & 1) else 1)
    if s != want:
        print('  lgamma_r(%r) sign %d, want %d' % (v, s, want)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    lgammar  the sign is gamma's own on both sides of zero"
else
  echo "FAIL  lgammar  a sign disagrees with the parity of the whole part"; fail=1
fi
runf kernels/spirv/lgamma.spv lgamma --grid 20 --threads 20 --buffers 20,20 --load 0="$TMP/lg_in.bin" --exact
runf kernels/spirv/lgammar.spv lgammar --grid 20 --threads 20 --buffers 20,20,20 --load 0="$TMP/lg_in.bin" --exact
"$WEFT" run kernels/bmsk.ptx --kernel bmsk --grid 1 --threads 1 --buffers 76 \
    --cpu-only --dump 0="$TMP/bmsk.bin" >/dev/null
if python3 - "$TMP/bmsk.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
# Eight .b32 rows, five .b64 rows, then the one .b32 row whose width runs past the type.
want32 = [0xFFFFFFFF, 0x7FFFFFFF, 0x00000FE0, 0, 0, 0x80000000, 1, 0xFF000000]
got32 = list(struct.unpack('<8I', data[:32]))
want64 = [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF00000000, ((1 << 35) - 1) << 5, 1 << 63, 0]
got64 = list(struct.unpack('<5Q', data[32:72]))
tail = struct.unpack('<I', data[72:76])[0]
bad = 0
for i, (g, w) in enumerate(zip(got32, want32)):
    if g != w:
        print('  b32 row %d got 0x%08X want 0x%08X' % (i, g, w)); bad += 1
for i, (g, w) in enumerate(zip(got64, want64)):
    if g != w:
        print('  b64 row %d got 0x%016X want 0x%016X' % (i, g, w)); bad += 1
if tail != 0xFFFFFFFF:
    print('  b32 row 8 got 0x%08X want 0xFFFFFFFF' % tail); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    bmsk        the run of bits is the run, and every edge clamps the isa way"
else
  echo "FAIL  bmsk        a mask came out as a shifted one instead of the run"; fail=1
fi

bo=$("$WEFT" run kernels/bmsk.ptx --kernel bmsk --grid 1 --threads 1 --buffers 2 \
       --cpu-only 2>&1) && br=0 || br=$?
if [ "$br" -eq 0 ] && printf '%s\n' "$bo" | grep -q 'buffer 0 was 2 elements from --buffers and the body writes 76 bytes: using 19' &&
   printf '%s\n' "$bo" | grep -q -- '--buffers 19 to size it yourself'; then
  echo "ok    overrun     a stated --buffers that is short grows to what the body writes, with the count named"
else
  echo "FAIL  overrun     got rc=$br: $(printf '%s\n' "$bo" | tail -1)"; fail=1
fi

bg=$("$WEFT" run kernels/bmsk.ptx --kernel bmsk --grid 1 --threads 1 --cpu-only 2>&1) && bgr=0 || bgr=$?
"$WEFT" run kernels/bmsk.ptx --kernel bmsk --grid 1 --threads 1 --cpu-only --dump 0="$TMP/grow1.bin" >/dev/null 2>&1
"$WEFT" run kernels/bmsk.ptx --kernel bmsk --grid 1 --threads 1 --cpu-only --dump 0="$TMP/grow2.bin" >/dev/null 2>&1
if [ "$bgr" -eq 0 ] && printf '%s\n' "$bg" | grep -q 'the body writes 76 bytes: using 19' &&
   cmp -s "$TMP/grow1.bin" "$TMP/grow2.bin" && [ "$(wc -c < "$TMP/grow1.bin")" -eq 76 ]; then
  echo "ok    overgrow    a default-sized buffer grows to what the body writes, and the run agrees with itself"
else
  echo "FAIL  overgrow    got rc=$bgr: $(printf '%s\n' "$bg" | grep -E 'weft:|note' | head -1)"; fail=1
fi
"$WEFT" run kernels/vsetpack.ptx --kernel vsetpack --grid 1 --threads 1 --buffers 28 \
    --cpu-only --dump 0="$TMP/vsetpack.bin" >/dev/null
if python3 - "$TMP/vsetpack.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()

def vs(n, op, a, at, b, bt):
    width = 32 // n
    field = (1 << width) - 1
    out = 0
    for i in range(n):
        xa = (a >> (width * i)) & field
        xb = (b >> (width * i)) & field
        if at == "s32" and xa >> (width - 1):
            xa -= 1 << width
        if bt == "s32" and xb >> (width - 1):
            xb -= 1 << width
        hit = {"eq": xa == xb, "ne": xa != xb, "lt": xa < xb,
               "le": xa <= xb, "gt": xa > xb, "ge": xa >= xb}[op]
        if hit:
            out |= 1 << (width * i)
    return out

want = [
    vs(2, "eq", 0x00010002, "u32", 0x00010003, "u32"),
    vs(2, "lt", 0xFFFF0001, "s32", 0x00000002, "s32"),
    vs(2, "lt", 0xFFFF0001, "u32", 0x00000002, "u32"),
    vs(2, "ne", 0x00000000, "u32", 0x0000FFFF, "u32"),
    vs(4, "eq", 0x01020304, "u32", 0x01020305, "u32"),
    vs(4, "lt", 0x0A0B0C0D, "u32", 0x00010003, "u32"),
    vs(4, "gt", 0x80FF7F01, "s32", 0x00000000, "s32"),
]
got = list(struct.unpack('<7I', data[:28]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    vsetpack    each half or byte answers 1 or 0, and the signed reading differs"
else
  echo "FAIL  vsetpack    a packed comparison packed or compared the wrong way"; fail=1
fi
runf kernels/vsetpack.ptx vsetpack --grid 1 --threads 1 --buffers 28 --exact
"$WEFT" run kernels/vscalar.ptx --kernel vscalar --grid 1 --threads 1 --buffers 56 \
    --cpu-only --dump 0="$TMP/vscalar.bin" >/dev/null
if python3 - "$TMP/vscalar.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
M32 = 0xFFFFFFFF

def widen(v, t):
    v &= M32
    return v - (1 << 32) if (t == "s32" and v >> 31) else v

def arith(op, a, at, b, bt):
    ta, tb = widen(a, at), widen(b, bt)
    if op == "add":
        r = ta + tb
    elif op == "sub":
        r = ta - tb
    elif op == "min":
        r = min(ta, tb)
    elif op == "max":
        r = max(ta, tb)
    else:
        r = abs(ta - tb)
    return r & M32

def madv(a, b, c):
    return ((a & M32) * (b & M32) + (c & M32)) & M32

def sett(cmp, a, at, b, bt):
    ta, tb = widen(a, at), widen(b, bt)
    return 1 if {"lt": ta < tb, "gt": ta > tb, "eq": ta == tb, "ne": ta != tb}[cmp] else 0

want = [
    arith("add", M32, "u32", 1, "u32"),
    arith("sub", 1, "u32", 5, "u32"),
    arith("min", M32, "s32", 1, "s32"),
    arith("max", M32, "u32", 1, "u32"),
    arith("absdiff", 0xFFFFFFF0, "s32", 1, "s32"),
    arith("absdiff", 1, "u32", 5, "u32"),
    madv(M32, M32, 7),
    madv(0xFFFFFFF0, 5, 3),
    arith("min", 5, "u32", 7, "u32"),
    arith("max", 0xFFFFFFF0, "s32", 5, "s32"),
    sett("lt", M32, "s32", 1, "u32"),
    sett("gt", M32, "u32", M32, "s32"),
    arith("add", 5, "s32", 7, "s32"),
    arith("sub", 5, "u32", M32, "s32"),
]
got = list(struct.unpack('<14I', data[:56]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    vscalar     the types decide the answer, and a mixed vset reads both signs"
else
  echo "FAIL  vscalar     a scalar video operation answered at the wrong signedness"; fail=1
fi
runf kernels/vscalar.ptx vscalar --grid 1 --threads 1 --buffers 56 --exact
"$WEFT" run kernels/vshift.ptx --kernel vshift --grid 1 --threads 1 --buffers 48 \
    --cpu-only --dump 0="$TMP/vshift.bin" >/dev/null
if python3 - "$TMP/vshift.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
M32 = 0xFFFFFFFF

def count(n, mode):
    return (n & 31) if mode == "wrap" else (32 if n > 32 else n)

def shl(a, n, mode):
    return (a << count(n, mode)) & M32

def shr_u(a, n, mode):
    return (a & M32) >> count(n, mode)

def shr_s(a, n, mode):
    sa = a - (1 << 32) if a >> 31 else a
    return (sa >> count(n, mode)) & M32

neg1 = M32
want = [shl(1, 4, "clamp"), shl(1, 36, "wrap"), shl(1, 40, "clamp"),
        shr_u(0x80000000, 4, "clamp"), shr_s(0x80000000, 4, "clamp"),
        shr_s(0x80000000, 40, "clamp"), shl(0x40000000, 1, "clamp"),
        shr_u(0x80000000, 36, "wrap"),   # .s32 destination, .u32 source: zero fill
        1 if -1 < 1 else 0, 1 if neg1 < 1 else 0, 1 if -1 >= -1 else 0,
        1 if neg1 != 1 else 0]
got = list(struct.unpack('<12I', data[:48]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    vshift      a count past the word clamps or wraps, and a signed shift fills"
else
  echo "FAIL  vshift      a video shift answered like a plain C shift"; fail=1
fi
runf kernels/vshift.ptx vshift --grid 1 --threads 1 --buffers 48 --exact
"$WEFT" run kernels/madhi.ptx --kernel madhi --grid 1 --threads 1 --buffers 64 \
    --cpu-only --dump 0="$TMP/madhi.bin" >/dev/null
if python3 - "$TMP/madhi.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF

def lo(a, b, c, cin=0, mask=M32):
    t = a * b + c + cin
    return t & mask, 1 if t > mask else 0

def hi(a, b, c, cin=0, mask=M32):
    t = ((a * b) >> (32 if mask == M32 else 64)) + c + cin
    return t & mask, 1 if t > mask else 0

def hi_s32(a, b, c, cin=0):
    sa = a - (1 << 32) if a >> 31 else a
    sb = b - (1 << 32) if b >> 31 else b
    t = ((sa * sb) >> 32) + c + cin
    return t & M32, 1 if t > M32 else 0

want = [(M32 * M32) >> 32]
s, _ = lo(3, 5, 7); want.append(s)
s, _ = hi(M32, M32, 7); want.append(s)
s, c = lo(1, M32, 1); want.append(s)
s, _ = lo(3, 5, 7, c); want.append(s)
s, _ = hi(M32, M32, 7, c); want.append(s)
s, _ = hi_s32(M32, 0x40000000, 3); want.append(s)
want.append((M32 * 3) & M32)
got = list(struct.unpack('<8I', data[:32]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
want64 = []
s, _ = lo(0xFFFFFFFF00000000, 2, 7, 0, M64); want64.append(s)
s, _ = hi(M64, M64, 7, 0, M64); want64.append(s)
s, c = lo(1, M64, 1, 0, M64); want64.append(s)
s, _ = hi(M64, M64, 7, c, M64); want64.append(s)
got64 = list(struct.unpack('<4Q', data[32:64]))
for i, (g, x) in enumerate(zip(got64, want64)):
    if g != x:
        print('  word64 %d got 0x%016X want 0x%016X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    madhi       the high word is the high one, and its addend carries out on its own"
else
  echo "FAIL  madhi       a high word came out as the low one"; fail=1
fi
runf kernels/madhi.ptx madhi --grid 1 --threads 1 --buffers 64 --exact
"$WEFT" run kernels/carry.ptx --kernel carry --grid 1 --threads 1 --buffers 64 \
    --cpu-only --dump 0="$TMP/carry.bin" >/dev/null
if python3 - "$TMP/carry.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF

def add(a, b, cin=0, mask=M32):
    t = a + b + cin
    return t & mask, 1 if t > mask else 0

def sub(a, b, cin=0, mask=M32):
    t = a - b - cin
    return t & mask, 1 if t < 0 else 0

def madc(a, b, c, cin=0, mask=M32):
    t = a * b + c + cin
    return t & mask, 1 if t > mask else 0

want = []
s, c = add(M32, 1); want.append(s)          # the word wraps, the flag goes to the next
s, c = add(1, 1, c); want.append(s)         # and the next word adds it
s, c = add(5, 7); want.append(s)            # no carry
s, c = add(5, 5, c); want.append(s)
s, c = sub(1, 3); want.append(s)            # a borrow out
s, c = sub(3, 1, c); want.append(s)         # and the borrow taken off the next word
s, c = add(M32, 1)                          # the flag back to one for the product
s, c = madc(3, 5, 7, c); want.append(s)     # 3*5 + 7 + 1
s, c = add(M32, M32)
s, c = add(M32, M32, c); want.append(s)
got = list(struct.unpack('<8I', data[:32]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
want64 = []
s, c = add(M64, 1, 0, M64); want64.append(s)
s, c = add(M64, M64, c, M64); want64.append(s)
s, c = sub(0, 1 << 63, 0, M64); want64.append(s)
s, c = sub(1 << 63, 0, c, M64); want64.append(s)
got64 = list(struct.unpack('<4Q', data[32:64]))
for i, (g, x) in enumerate(zip(got64, want64)):
    if g != x:
        print('  word64 %d got 0x%016X want 0x%016X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    carry       the flag travels between words, and the 64-bit chain is its own"
else
  echo "FAIL  carry       a carry chain came out one word short"; fail=1
fi
runf kernels/carry.ptx carry --grid 1 --threads 1 --buffers 64 --exact
"$WEFT" run kernels/cnot.ptx --kernel cnot --grid 1 --threads 1 --buffers 64 \
    --cpu-only --dump 0="$TMP/cn.bin" >/dev/null
if python3 - "$TMP/cn.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
in32 = [0, 1, 0xFFFFFFFF, 0x80000000, 0, 42, 0, 0x7FFFFFFF]
in64 = [0, 0x0000000100000000, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000]
# The whole definition: one for zero, zero for anything else.
want = [1 if v == 0 else 0 for v in in32 + in64]
got = list(struct.unpack('<8I', data[:32])) + list(struct.unpack('<4Q', data[32:64]))
bad = 0
for i, (g, w) in enumerate(zip(got, want)):
    if g != w:
        print('  row %d got %d want %d' % (i, g, w)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    cnot        zero answers one and every non-zero answers zero, not not's answer"
else
  echo "FAIL  cnot        a logical negation came out as the bitwise one"; fail=1
fi
runf kernels/cnot.ptx cnot --grid 1 --threads 1 --buffers 64 --exact
"$WEFT" run kernels/bswap.ptx --kernel bswap --grid 1 --threads 1 --buffers 72 \
    --cpu-only --dump 0="$TMP/bs.bin" >/dev/null
if python3 - "$TMP/bs.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
in32 = [0x12345678, 0x000000FF, 0xFF000000, 0, 0xFFFFFFFF, 0x0000FF00, 0x00000080,
        0x00FF00FF]
in64 = [0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF, 0x00000000000000FF,
        0xFF00000000000000, 0x1122334455667788]
# The input's bytes read the other way round, which is the whole definition.
flip = lambda v, n: int.from_bytes(v.to_bytes(n, 'little'), 'big')
want32 = [flip(v, 4) for v in in32]
want64 = [flip(v, 8) for v in in64]
got32 = list(struct.unpack('<8I', data[:32]))
got64 = list(struct.unpack('<5Q', data[32:72]))
bad = 0
for i, (g, w) in enumerate(zip(got32, want32)):
    if g != w:
        print('  b32 row %d got 0x%08X want 0x%08X' % (i, g, w)); bad += 1
for i, (g, w) in enumerate(zip(got64, want64)):
    if g != w:
        print('  b64 row %d got 0x%016X want 0x%016X' % (i, g, w)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    bswap       the bytes trade places, and the halves of a word trade with them"
else
  echo "FAIL  bswap       a reversal came out as a shift or a bit reversal"; fail=1
fi
runf kernels/bswap.ptx bswap --grid 1 --threads 1 --buffers 72 --exact
"$WEFT" run kernels/lanemask.ptx --kernel lanemask --grid 64 --threads 64 --buffers 512 \
    --cpu-only --dump 0="$TMP/lm.bin" >/dev/null
if python3 - "$TMP/lm.bin" <<'PY'
import struct, sys
words = struct.unpack('<512I', open(sys.argv[1], 'rb').read())
bad = 0
for lane in range(64):
    w = lane & 31
    lt = ((1 << w) - 1) & 0xFFFFFFFF
    le = ((2 << w) - 1) & 0xFFFFFFFF
    want = [1 << w, lt, le, (~le) & 0xFFFFFFFF, (~lt) & 0xFFFFFFFF, lane // 32, 2, w]
    got = list(words[lane * 8:(lane + 1) * 8])
    if got != want:
        print('  lane %d got %s want %s' % (lane, got, want)); bad += 1
    if bad > 6:
        break
sys.exit(1 if bad else 0)
PY
then
  echo "ok    lanemask  the five masks, warpid, nwarpid and laneid are the warp's own"
else
  echo "FAIL  lanemask  a mask, a warp id or laneid is wrong"; fail=1
fi
runf kernels/lanemask.ptx lanemask --grid 64 --threads 64 --buffers 512 --exact
if python3 - "$WEFT" "$TMP" <<'PY'
import struct, subprocess, sys
weft, tmp = sys.argv[1], sys.argv[2]
sub = [(1e-20, 1e-20), (1e-40, 1e18), (3.0, 3.0), (2.0, 0.5),
       (1e-30, 1e10), (3.0, 2.0), (1e-38, 1e-1), (1.1754944e-38, 2.0)]
scaled = [(1e-14, 1e-14), (1e-30, 1e18), (3.0, 3.0), (2.0, 0.5),
          (1e-20, 1e10), (3.0, 2.0), (1e-28, 1e-1), (1.1754944e-28, 2.0)]
MIN_NORMAL = 1.17549435e-38
def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]
def is_sub(x): return x != 0.0 and abs(x) < MIN_NORMAL
def files(pairs, tag):
    a = '%s/sn_%s_a.bin' % (tmp, tag); b = '%s/sn_%s_b.bin' % (tmp, tag)
    open(a, 'wb').write(struct.pack('<8f', *[p[0] for p in pairs]))
    open(b, 'wb').write(struct.pack('<8f', *[p[1] for p in pairs]))
    return a, b
def cpu(pairs, tag):
    a, b = files(pairs, tag)
    out = '%s/sn_%s.bin' % (tmp, tag)
    subprocess.run([weft, 'run', 'kernels/subnorm.ptx', '--kernel', 'subnorm', '--grid', '8',
                    '--threads', '8', '--buffers', '8,8,8', '--load', '0=' + a, '--load', '1=' + b,
                    '--cpu-only', '--dump', '2=' + out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return struct.unpack('<8f', open(out, 'rb').read())
def parted(tag):
    a = '%s/sn_%s_a.bin' % (tmp, tag); b = '%s/sn_%s_b.bin' % (tmp, tag)
    r = subprocess.run([weft, 'run', 'kernels/subnorm.ptx', '--kernel', 'subnorm', '--grid', '8',
                        '--threads', '8', '--buffers', '8,8,8', '--load', '0=' + a, '--load', '1=' + b,
                        '--exact'], capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith('compare'):
            return int(line.split()[3])
    return -1
bad = 0
for pairs, tag in ((sub, 'sub'), (scaled, 'scaled')):
    got = cpu(pairs, tag)
    expect = 0
    for i, (a, b) in enumerate(pairs):
        a, b = f32(a), f32(b)
        want = f32(a * b if i < 4 else a / b)
        if got[i] != want:
            print('  lane %d cpu %.9g want %.9g' % (i, got[i], want)); bad += 1
        if is_sub(want) or is_sub(a) or is_sub(b): expect += 1
    k = parted(tag)
    if k < 0:
        print('  %s: no gpu column here, so no lane was parted' % tag)
    elif k != expect:
        print('  %s: %d lanes part, the subnormal count is %d' % (tag, k, expect)); bad += 1
    else:
        print('  %s: the cpu column is ieee and %d of 8 lanes part' % (tag, k))
sys.exit(1 if bad else 0)
PY
then
  echo "ok    subnorm  the cpu is ieee and the gpu parts only on the subnormal lanes"
else
  echo "FAIL  subnorm  a lane parted outside the subnormal range, or the cpu is not ieee"; fail=1
fi
python3 - "$TMP/dv_a.bin" "$TMP/dv_b.bin" <<'PY'
import struct, sys
pairs = [(6.0, 3.0), (1.0, 3.0), (-7.5, 2.5), (0.0, 5.0),
         (1e-30, 1e-20), (3.0, 7.0), (2.5, 0.5), (9.75, 0.25)]
open(sys.argv[1], 'wb').write(struct.pack('<8f', *[p[0] for p in pairs]))
open(sys.argv[2], 'wb').write(struct.pack('<8f', *[p[1] for p in pairs]))
PY
"$WEFT" run kernels/spirv/divide2.spv --kernel divide2 --grid 8 --threads 8 --buffers 8,8,8 \
    --load 0="$TMP/dv_a.bin" --load 1="$TMP/dv_b.bin" --cpu-only --dump 2="$TMP/dv_out.bin" >/dev/null
if python3 - "$TMP/dv_out.bin" <<'PY'
import struct, sys
pairs = [(6.0, 3.0), (1.0, 3.0), (-7.5, 2.5), (0.0, 5.0),
         (1e-30, 1e-20), (3.0, 7.0), (2.5, 0.5), (9.75, 0.25)]
def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]
got = struct.unpack('<8f', open(sys.argv[1], 'rb').read())
bad = 0
for i, (a, b) in enumerate(pairs):
    want = f32(f32(a) / f32(b))
    if got[i] != want:
        print('  lane %d got %r want %r' % (i, got[i], want)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    divide2  half_divide and native_divide both answer the ieee quotient"
else
  echo "FAIL  divide2  a spelling answered something other than a / b"; fail=1
fi
runf kernels/spirv/divide2.spv divide2 --grid 8 --threads 8 --buffers 8,8,8 --load 0="$TMP/dv_a.bin" --load 1="$TMP/dv_b.bin" --exact
python3 - "$TMP/mad3in.bin" "$TMP/nanb.bin" <<'PY'
import struct, sys
v = [3.0, 0.3333333432674408, -1.0, 1.5, 2.0, 0.25, 0.5, -3.0, 4.0, 2.0, 2.0, -1.0,
     -1.5, 1.5, 0.5, 0.0, 5.0, 1.25, 7.0, 0.125, -0.5, 1.0, 1.0, 0.0]
open(sys.argv[1], 'wb').write(b''.join(struct.pack('<f', x) for x in v))
words = [0x00000000, 0x3F800001, 0x3FFFFF00, 0x3FFFFFFF, 0x3F000000, 0x41200000,
         0x40480000, 0x501502F9]
open(sys.argv[2], 'wb').write(b''.join(struct.pack('<I', w) for w in words))
PY
"$WEFT" run kernels/spirv/mad3.spv --kernel mad3 --grid 8 --threads 8 --buffers 24,8 \
    --load 0="$TMP/mad3in.bin" --cpu-only --dump 1="$TMP/mad3.bin" >/dev/null
if python3 - "$TMP/mad3.bin" <<'PY'
import struct, sys
from fractions import Fraction
v = [3.0, 0.3333333432674408, -1.0, 1.5, 2.0, 0.25, 0.5, -3.0, 4.0, 2.0, 2.0, -1.0,
     -1.5, 1.5, 0.5, 0.0, 5.0, 1.25, 7.0, 0.125, -0.5, 1.0, 1.0, 0.0]
got = struct.unpack('<8f', open(sys.argv[1], 'rb').read())
bad = 0
for i in range(8):
    a, b, c = v[i * 3:i * 3 + 3]
    want = struct.unpack('<f', struct.pack('<f', float(Fraction(a) * Fraction(b) + Fraction(c))))[0]
    if got[i] != want:
        print('  lane %d %r*%r+%r: got %r want %r' % (i, a, b, c, got[i], want)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    mad3  mad is one rounding of the exact product and sum"
else
  echo "FAIL  mad3  mad is not the fused result"; fail=1
fi
"$WEFT" run kernels/spirv/nanbits.spv --kernel nanbits --grid 8 --threads 8 --buffers 8,8 \
    --load 0="$TMP/nanb.bin" --cpu-only --dump 1="$TMP/nanb.out" >/dev/null
if python3 - "$TMP/nanb.bin" "$TMP/nanb.out" <<'PY'
import struct, sys
inp = struct.unpack('<8I', open(sys.argv[1], 'rb').read())
got = struct.unpack('<8I', open(sys.argv[2], 'rb').read())
bad = 0
for i, w in enumerate(inp):
    want = 0x7FC00000 | (w & 0x3FFFFF)
    if got[i] != want:
        print('  lane %d from 0x%08X got 0x%08X want 0x%08X' % (i, w, got[i], want)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    nanbits  nan carries its operand's payload in a quiet NaN"
else
  echo "FAIL  nanbits  nan did not carry its payload"; fail=1
fi
runf kernels/spirv/mad3.spv mad3 --grid 8 --threads 8 --buffers 24,8 --load 0="$TMP/mad3in.bin" --exact
runf kernels/spirv/nanbits.spv nanbits --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/nanb.bin" --exact
"$WEFT" run kernels/ftan.ptx --kernel ftan --grid 8 --threads 8 --buffers 8 \
    --load 0="$TMP/tan32in.bin" --cpu-only --dump 0="$TMP/tan32.bin" >/dev/null
if python3 - "$TMP/tan32.bin" <<'PY'
import struct, math, sys
vals = [0.5, 1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 3.0]
raw = open(sys.argv[1], 'rb').read()
got = struct.unpack('<%df' % (len(raw) // 4), raw)
bad = 0
for i, x in enumerate(vals):
    r = got[i]
    step = abs(struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', abs(r)))[0] + 1))[0] - abs(r))
    if not (abs(r - math.tan(x)) <= step * 1.0001 + 1e-12):
        print('  lane %d tan(%r): got %r want %r' % (i, x, r, math.tan(x))); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    ftan  tan.approx.f32 is within an ulp of a double's tangent"
else
  echo "FAIL  ftan  tan.approx.f32 is off by more than an ulp"; fail=1
fi
runf kernels/ftan.ptx ftan --grid 8 --threads 8 --buffers 8
runf kernels/halfops.ptx halfops --grid 8 --threads 8 --buffers 8,128
runf kernels/halfops3.ptx halfops_tg --grid 32 --threads 32 --buffers 32,32
runf kernels/halfops4.ptx halfmin --grid 1 --threads 1 --buffers 6
runf kernels/bf16ops.ptx bf16ops --grid 8 --threads 8 --buffers 8,128
runf kernels/narrowdiv.ptx narrowdiv --grid 1 --threads 1 --buffers 16
runf kernels/narrowdiv.ptx narrowdiv2 --grid 1 --threads 1 --buffers 8
runf kernels/narrowcmp.ptx narrowcmp --grid 8 --threads 8 --buffers 8,64
runf kernels/narrowcmp2.ptx narrowcmp_bf16 --grid 8 --threads 8 --buffers 8,32
runf kernels/narrowcmp3.ptx narrowcmp2 --grid 8 --threads 8 --buffers 8,24
runf kernels/narrowatom.ptx narrowatom --grid 64 --threads 64 --buffers 2 --load 0="$TMP/zatom8.bin"
runf kernels/narrowatom2.ptx redatom --grid 64 --threads 64 --buffers 1 --load 0="$TMP/zatom4.bin"
runf kernels/slct.ptx slcti --grid 4 --threads 4 --buffers 4,4 --load 0="$TMP/slin.bin"
runf kernels/slct.ptx slctf --grid 4 --threads 4 --buffers 4,4 --load 0="$TMP/slinf.bin"
runf kernels/casb16.ptx casb16 --grid 1 --threads 1 --buffers 1,1 --load 0="$TMP/casin.bin"
runf kernels/casb16.ptx casb16miss --grid 1 --threads 1 --buffers 1,1 --load 0="$TMP/casin.bin"
runf kernels/casb16.ptx casb16race --grid 8 --threads 8 --buffers 1 --load 0="$TMP/casin.bin"
runf kernels/halfvecatom.ptx hvmax --grid 8 --threads 8 --buffers 1 --load 0="$TMP/hvmax_in.bin"
runf kernels/halfvecatom.ptx hvmin --grid 8 --threads 8 --buffers 1 --load 0="$TMP/hvmin_in.bin"
runf kernels/halfvecatom.ptx hvadd --grid 8 --threads 8 --buffers 1 --load 0="$TMP/hvadd_in.bin"
runf kernels/halfvecatom.ptx hvbf --grid 8 --threads 8 --buffers 1 --load 0="$TMP/hvbf_in.bin"
runf kernels/halfvecatom.ptx hvpack --grid 8 --threads 8 --buffers 2 --load 0="$TMP/hvpack_in.bin"
runf kernels/scluster.ptx sclv4 --grid 8 --threads 8 --buffers 32
runf kernels/scluster.ptx sclmapa --grid 8 --threads 8 --buffers 8
runf kernels/shatom.ptx satadd --grid 8 --threads 8 --buffers 4
runf kernels/shatom.ptx satb16 --grid 8 --threads 8 --buffers 1
runf kernels/shatom.ptx satpack --grid 8 --threads 8 --buffers 1
runf kernels/shatom.ptx satret --grid 8 --threads 8 --buffers 1
runf kernels/shatom64.ptx satom64 --grid 8 --threads 8 --buffers 1
"$WEFT" run kernels/shatom64.ptx --kernel satom64 --grid 8 --threads 8 --buffers 1 \
    --cpu-only --dump 0="$TMP/satom64.bin" >/dev/null
if [ "$(od -An -tx1 -v "$TMP/satom64.bin" | tr -d ' \n')" = "0800000000000000" ]; then
  echo "ok    satom64  a 64-bit add in the window, eight lanes, lands on eight"
else
  echo "FAIL  satom64  read $(od -An -tx1 -v "$TMP/satom64.bin" | tr -d ' \n')"; fail=1
fi
runf kernels/satom64ret.ptx satom64ret --grid 8 --threads 8 --buffers 1
"$WEFT" run kernels/satom64ret.ptx --kernel satom64ret --grid 8 --threads 8 --buffers 1 \
    --cpu-only --dump 0="$TMP/satom64ret.bin" >/dev/null
if [ "$(od -An -tx1 -v "$TMP/satom64ret.bin" | tr -d ' \n')" = "0800000000000000" ]; then
  echo "ok    satom64ret a 64-bit window add that reads its old value lands on eight"
else
  echo "FAIL  satom64ret read $(od -An -tx1 -v "$TMP/satom64ret.bin" | tr -d ' \n')"; fail=1
fi

python3 - "$TMP" <<'PYSA'
import struct, sys, os
d = sys.argv[1]
open(os.path.join(d, 'satom64z.bin'), 'wb').write(b'\x00' * 32)
open(os.path.join(d, 'satom64oldexp.bin'), 'wb').write(struct.pack('<8I', *range(8)))
open(os.path.join(d, 'satom64casexp.bin'), 'wb').write(struct.pack('<8I', 0, 1, 1, 1, 1, 1, 1, 1))
PYSA
runf kernels/satom64old.ptx satom64old --grid 8 --threads 8 --buffers 1 \
    --buffer-bytes 0=32 --load 0="$TMP/satom64z.bin"
"$WEFT" run kernels/satom64old.ptx --kernel satom64old --grid 8 --threads 8 \
    --buffers 1 --buffer-bytes 0=32 --load 0="$TMP/satom64z.bin" --cpu-only \
    --dump 0="$TMP/satom64old.bin" >/dev/null
if cmp -s "$TMP/satom64old.bin" "$TMP/satom64oldexp.bin"; then
  echo "ok    satom64old the window hands the lanes out in lane order"
else
  echo "FAIL  satom64old $(cmp "$TMP/satom64old.bin" "$TMP/satom64oldexp.bin" 2>&1 | head -1)"
  fail=1
fi
runf kernels/satom64cas.ptx satom64cas --grid 8 --threads 8 --buffers 1 \
    --buffer-bytes 0=32 --load 0="$TMP/satom64z.bin"
"$WEFT" run kernels/satom64cas.ptx --kernel satom64cas --grid 8 --threads 8 \
    --buffers 1 --buffer-bytes 0=32 --load 0="$TMP/satom64z.bin" --cpu-only \
    --dump 0="$TMP/satom64cas.bin" >/dev/null
if cmp -s "$TMP/satom64cas.bin" "$TMP/satom64casexp.bin"; then
  echo "ok    satom64cas the window hands the lanes out in lane order"
else
  echo "FAIL  satom64cas $(cmp "$TMP/satom64cas.bin" "$TMP/satom64casexp.bin" 2>&1 | head -1)"
  fail=1
fi

python3 - "$TMP" <<'PYX'
import struct, sys, os
d = sys.argv[1]
A = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
B = (0x0F0E0D0C0B0A0908, 0x0706050403020100)
files = {
    'c128_a.bin': A,
    'c128_b.bin': B,
    'c128_cl.bin': (0x0123456789ABCDEE, A[1]),      # one low byte off
    'c128_ch.bin': (A[0], 0xFEDCBA9876543211),      # one high byte off
    'c128_ok.bin': B,
    'c128_stay.bin': A,
}
for name, w in files.items():
    open(os.path.join(d, name), 'wb').write(struct.pack('<QQ', *w))
PYX
cas128_ref() {  # name, compare file, expected file, message
  "$WEFT" run kernels/cas128.ptx --kernel cas128 --grid 8 --threads 8 --buffers 1,1,1 \
      --load 0="$TMP/c128_a.bin" --load 1="$TMP/$2" --load 2="$TMP/c128_b.bin" \
      --cpu-only --dump 0="$TMP/$1.bin" >/dev/null
  if cmp -s "$TMP/$1.bin" "$TMP/$3"; then
    echo "ok    $1  $4"
  else
    echo "FAIL  $1  $(cmp "$TMP/$1.bin" "$TMP/$3" 2>&1 | head -1)"; fail=1
  fi
}
cas128_ref cas128   c128_a.bin  c128_ok.bin   "sixteen bytes compared and written as one value"
cas128_ref cas128lo c128_cl.bin c128_stay.bin "a compare off in the low half writes nothing"
cas128_ref cas128hi c128_ch.bin c128_stay.bin "a compare off in the high half writes nothing"
runf kernels/cas128.ptx cas128 --grid 8 --threads 8 --buffers 1,1,1 \
    --load 0="$TMP/c128_a.bin" --load 1="$TMP/c128_a.bin" --load 2="$TMP/c128_b.bin"
python3 - "$TMP" <<'PY128S'
import struct, sys, os
d = sys.argv[1]
open(os.path.join(d, 'c128sz.bin'), 'wb').write(b'\x00' * 16)
open(os.path.join(d, 'c128so.bin'), 'wb').write(b'\x00' * 8)
open(os.path.join(d, 'c128scmp5.bin'), 'wb').write(struct.pack('<QQ', 0, 5))
open(os.path.join(d, 'c128snew.bin'), 'wb').write(struct.pack('<QQ', 1, 0))
open(os.path.join(d, 'c128sexp1.bin'), 'wb').write(struct.pack('<II', 1, 0))
open(os.path.join(d, 'c128sexp2.bin'), 'wb').write(struct.pack('<II', 0, 0))
PY128S
cas128s_ref() {  # name, compare file, expected file, message
  "$WEFT" run kernels/cas128s.ptx --kernel cas128s --grid 8 --threads 8 \
      --buffers 1,1,1 --buffer-bytes 0=8,1=16,2=16 \
      --load 0="$TMP/c128so.bin" --load 1="$TMP/$2" --load 2="$TMP/c128snew.bin" \
      --cpu-only --dump 0="$TMP/$1.bin" >/dev/null
  if cmp -s "$TMP/$1.bin" "$TMP/$3"; then
    echo "ok    $1  $4"
  else
    echo "FAIL  $1  $(cmp "$TMP/$1.bin" "$TMP/$3" 2>&1 | head -1)"; fail=1
  fi
}
cas128s_ref cas128s  c128sz.bin   c128sexp1.bin "a 128-bit cas in the window swaps all sixteen bytes"
cas128s_ref cas128sh c128scmp5.bin c128sexp2.bin "a compare off in the high half writes nothing"
runf kernels/cas128s.ptx cas128s --grid 8 --threads 8 --buffers 1,1,1 \
    --buffer-bytes 0=8,1=16,2=16 --load 0="$TMP/c128so.bin" \
    --load 1="$TMP/c128sz.bin" --load 2="$TMP/c128snew.bin"
python3 - "$TMP" <<'PYEX'
import struct, sys, os
d = sys.argv[1]
open(os.path.join(d, 'ex128src.bin'), 'wb').write(struct.pack('<QQ', 7, 9))
open(os.path.join(d, 'ex128in.bin'), 'wb').write(struct.pack('<QQ', 1, 2))
open(os.path.join(d, 'ex8.bin'), 'wb').write(b'\x00' * 8)
open(os.path.join(d, 'ex128exp.bin'), 'wb').write(struct.pack('<QQ', 7, 9))
open(os.path.join(d, 'ex128sexp.bin'), 'wb').write(struct.pack('<Q', 7))
PYEX
ex128_ref() {  # name, buffer bytes, input for buffer 0, expected file, message
  "$WEFT" run "kernels/$1.ptx" --kernel "$1" --grid 8 --threads 8 --buffers 1,1 \
      --buffer-bytes "$2" --load 0="$TMP/$3" --load 1="$TMP/ex128src.bin" \
      --cpu-only --dump 0="$TMP/$1.bin" >/dev/null
  if cmp -s "$TMP/$1.bin" "$TMP/$4"; then
    echo "ok    $1  $5"
  else
    echo "FAIL  $1  $(cmp "$TMP/$1.bin" "$TMP/$4" 2>&1 | head -1)"; fail=1
  fi
}
runf kernels/exch128.ptx exch128 --grid 8 --threads 8 --buffers 1,1 \
    --buffer-bytes 0=16,1=16 --load 0="$TMP/ex128in.bin" --load 1="$TMP/ex128src.bin"
ex128_ref exch128 0=16,1=16 ex128in.bin ex128exp.bin \
    "a 128-bit exch in the heap swaps all sixteen bytes"
runf kernels/exch128s.ptx exch128s --grid 8 --threads 8 --buffers 1,1 \
    --buffer-bytes 0=8,1=16 --load 0="$TMP/ex8.bin" --load 1="$TMP/ex128src.bin"
ex128_ref exch128s 0=8,1=16 ex8.bin ex128sexp.bin \
    "a 128-bit exch in the window swaps all sixteen bytes"
shalfvec_rc=0
shalfvec_out=$("$WEFT" run tests/refused/shalfvec.ptx --kernel shvmin --grid 8 --threads 8 \
    --buffers 1 --cpu-only 2>&1) || shalfvec_rc=$?
if [ $shalfvec_rc -ne 0 ] && [ "${shalfvec_out#*only with .global space}" != "$shalfvec_out" ]; then
  echo "ok    shalfvec   a vector atomic in the window is refused, at the space the isa names"
else
  echo "FAIL  shalfvec   a shared vector atomic was not refused by name (rc=$shalfvec_rc)"; fail=1
fi
runf kernels/incdec.ptx inc --grid 5 --threads 5 --buffers 1 --load 0="$TMP/inc0.bin"
runf kernels/incdec.ptx dec --grid 2 --threads 2 --buffers 1 --load 0="$TMP/inc0.bin"
runf kernels/incdec.ptx incgrab --grid 1 --threads 1 --buffers 1,1 --load 0="$TMP/inc0.bin"
runf kernels/incdec.ptx redinc --grid 3 --threads 3 --buffers 1 --load 0="$TMP/inc0.bin"
runf kernels/spirv/atom16.spv a16add --grid 8 --threads 8 --buffers 2 --load 0="$TMP/a16in0.bin"
runf kernels/spirv/atom16.spv a16mix --grid 1 --threads 1 --buffers 2 --load 0="$TMP/a16in.bin"
runf kernels/spirv/switch.spv sw32 --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/sw32in.bin"
runf kernels/spirv/switch.spv sw64 --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/sw64in.bin"
runf kernels/spirv/switchs.spv sws --grid 4 --threads 4 --buffers 4
runf kernels/brxidx.ptx brxidx --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/brxin.bin"
runf kernels/brxidx.ptx brxpred --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/brxin.bin"
runf kernels/bitfind.ptx bfindu --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/bfindu_in.bin"
runf kernels/bitfind.ptx bfinds --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/bfinds_in.bin"
runf kernels/bitfind.ptx fns3 --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/fns_in.bin"
runf kernels/bitfind.ptx fnsbase --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/fns_in.bin"
runf kernels/bitfind.ptx f16reg --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/h16in.bin"
runf kernels/warpmatch.ptx electk --grid 8 --threads 8 --buffers 8
runf kernels/warpmatch.ptx matchny --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/wmmix.bin"
runf kernels/warpmatch.ptx matchalld --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/wmmix.bin"
runf kernels/warpmatch.ptx matchallp --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/wmeq.bin"
runf kernels/dp2a.ptx dp2alo --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/dpin.bin"
runf kernels/dp2a.ptx dp2ahi --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/dpin.bin"
runf kernels/dp2a.ptx dp2aslo --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/dpin.bin"
runf kernels/dp2a.ptx dp2ashi --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/dpin.bin"
runf kernels/movmatrix.ptx movmtx --grid 32 --threads 32 --buffers 32,32 --load 0="$TMP/mvin.bin"
runf kernels/movmatrix.ptx movmtx2 --grid 32 --threads 32 --buffers 32,32 --load 0="$TMP/mvin.bin"
runf kernels/video.ptx vadd2u --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/vidin.bin"
runf kernels/video.ptx vadd2s --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/vidin.bin"
runf kernels/video.ptx vmin4s --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/vidin.bin"
runf kernels/video.ptx vabs4u --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/vidin.bin"
runf kernels/video.ptx vadd4u --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/vidin.bin"
runf kernels/video.ptx vavg4s --grid 8 --threads 8 --buffers 16,8 --load 0="$TMP/vidin.bin"
runf kernels/multimem.ptx mmld --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/mmin.bin"
runf kernels/multimem.ptx mmmin --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/mmin.bin"
runf kernels/multimem.ptx mmst --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/mmin.bin"
runf kernels/b128.ptx b128mv --grid 4 --threads 4 --buffers 4,4 --load 0="$TMP/b128in.bin"
runf kernels/call.ptx callk --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callin.bin"
runf kernels/call.ptx callbr --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callbrin.bin"
# Nested calls: the kernel calls a function that calls two others, one of which loops.
python3 - "$TMP" <<'PYC'
import struct, sys, os
d = sys.argv[1]
vals = [3, 5, 12, 0, 7, 100, 1, 9]
sums = [v + 100 if v >= 10 else v * (v + 1) // 2 for v in vals]
open(os.path.join(d, 'callnestin.bin'), 'wb').write(struct.pack('<8I', *vals))
open(os.path.join(d, 'callnestexp.bin'), 'wb').write(struct.pack('<8I', *[s + 1 for s in sums]))
PYC
narrowref callnest   kernels/call.ptx callnest 8 8 8,8 1 callnestin.bin callnestexp.bin
runf kernels/call.ptx callnest --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callnestin.bin"

runf kernels/callrec.ptx callrec --grid 1 --threads 1 --buffers 1
"$WEFT" run kernels/callrec.ptx --kernel callrec --grid 1 --threads 1 --buffers 1 \
    --cpu-only --dump 0="$TMP/callrec.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/callrec.bin" | tr -d ' ')" = "12" ]; then
  echo "ok    callrec   two functions that call each other run through one driver, 12"
else
  echo "FAIL  callrec   read $(od -An -tu4 -N4 "$TMP/callrec.bin" | tr -d ' ')"; fail=1
fi
cp tests/refused/callself2.invalid "$TMP/callself2.ptx"
callself2_rc=0
callself2_out=$("$WEFT" run "$TMP/callself2.ptx" --kernel callself2 --grid 1 --threads 1 \
    --buffers 1 --cpu-only 2>&1) || callself2_rc=$?
if [ $callself2_rc -ne 0 ] && [ "${callself2_out#*ran past the 8 frames}" != "$callself2_out" ]; then
  echo "ok    callself2 a self-call with no base case stops at the eighth frame and says so"
else
  echo "FAIL  callself2 a runaway self-call was not stopped by name (rc=$callself2_rc)"; fail=1
fi
runf kernels/calldeep.ptx calldeep --grid 1 --threads 1 --buffers 1
"$WEFT" run kernels/calldeep.ptx --kernel calldeep --grid 1 --threads 1 --buffers 1 \
    --cpu-only --dump 0="$TMP/calldeep.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/calldeep.bin" | tr -d ' ')" = "10" ]; then
  echo "ok    calldeep  a self-call with work after it keeps four frames and sums to 10"
else
  echo "FAIL  calldeep  read $(od -An -tu4 -N4 "$TMP/calldeep.bin" | tr -d ' ')"; fail=1
fi
runf kernels/callvalu.ptx callvalu --grid 1 --threads 1 --buffers 1
"$WEFT" run kernels/callvalu.ptx --kernel callvalu --grid 1 --threads 1 --buffers 1 \
    --cpu-only --dump 0="$TMP/callvalu.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/callvalu.bin" | tr -d ' ')" = "10" ]; then
  echo "ok    callvalu  a recursive call that returns a value reads it back from its frame"
else
  echo "FAIL  callvalu  read $(od -An -tu4 -N4 "$TMP/callvalu.bin" | tr -d ' ')"; fail=1
fi
runf kernels/callpair.ptx callpair --grid 1 --threads 1 --buffers 1,1
"$WEFT" run kernels/callpair.ptx --kernel callpair --grid 1 --threads 1 --buffers 1,1 \
    --cpu-only --dump 0="$TMP/callpair0.bin" --dump 1="$TMP/callpair1.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/callpair0.bin" | tr -d ' ')" = "10" ] &&
   [ "$(od -An -tu4 -N4 "$TMP/callpair1.bin" | tr -d ' ')" = "4" ]; then
  echo "ok    callpair  a recursive call returns two values into two buffers"
else
  echo "FAIL  callpair  read $(od -An -tu4 -N4 "$TMP/callpair0.bin" | tr -d ' ') and $(od -An -tu4 -N4 "$TMP/callpair1.bin" | tr -d ' ')"; fail=1
fi
runf kernels/callmix.ptx callmix --grid 1 --threads 1 --buffers 2,1
"$WEFT" run kernels/callmix.ptx --kernel callmix --grid 1 --threads 1 --buffers 2,1 \
    --cpu-only --dump 0="$TMP/callmix0.bin" --dump 1="$TMP/callmix1.bin" >/dev/null
if [ "$(od -An -tu8 -N8 "$TMP/callmix0.bin" | tr -d ' ')" = "6" ] &&
   [ "$(od -An -tu4 -N4 "$TMP/callmix1.bin" | tr -d ' ')" = "1086324736" ]; then
  echo "ok    callmix   a recursive return list carries a 64-bit value and a float"
else   echo "FAIL  callmix   read $(od -An -tu8 -N8 "$TMP/callmix0.bin" | tr -d ' ') and $(od -An -tu4 -N4 "$TMP/callmix1.bin" | tr -d ' ')"; fail=1
fi
cp tests/refused/trap.invalid "$TMP/trap.ptx"
trap_rc=0
trap_out=$("$WEFT" run "$TMP/trap.ptx" --kernel trapstop --grid 1 --threads 1 --buffers 1 \
    --cpu-only 2>&1) || trap_rc=$?
if [ $trap_rc -ne 0 ] && [ "${trap_out#*a lane trapped}" != "$trap_out" ]; then
  echo "ok    trap      a trap stops the lane and the host reports the stop by name"
else
  echo "FAIL  trap      a trap did not report a stopped lane (rc=$trap_rc)"; fail=1
fi
cp tests/refused/callentry.invalid "$TMP/callentry.ptx"
callentry_rc=0
callentry_out=$("$WEFT" run "$TMP/callentry.ptx" --kernel outer --grid 1 --threads 1 \
    --buffers 1 --cpu-only 2>&1) || callentry_rc=$?
if [ $callentry_rc -ne 0 ] && [ "${callentry_out#*an entry is launched, not called}" != "$callentry_out" ]; then
  echo "ok    callentry a call that names an entry is refused as an entry, not a missing body"
else
  echo "FAIL  callentry a call to an entry was not named as one (rc=$callentry_rc)"; fail=1
fi
cp tests/refused/trap3.invalid "$TMP/trap3.ptx"
trap3_rc=0
trap3_out=$("$WEFT" run "$TMP/trap3.ptx" --kernel trap3 --grid 4 --threads 4 --buffers 4 \
    --cpu-only 2>&1) || trap3_rc=$?
if [ $trap3_rc -ne 0 ] && [ "${trap3_out#*(lane 3)}" != "$trap3_out" ]; then
  echo "ok    trap3     a trap in one lane of four reports that lane, not the first"
else
  echo "FAIL  trap3     the trap did not name lane 3 (rc=$trap3_rc)"; fail=1
fi
runf kernels/callstop.ptx callstop --grid 1 --threads 1 --buffers 1
"$WEFT" run kernels/callstop.ptx --kernel callstop --grid 1 --threads 1 --buffers 1 \
    --cpu-only --dump 0="$TMP/callstop.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/callstop.bin" | tr -d ' ')" = "7" ]; then
  echo "ok    callstop  a callee that stops at a barrier resumes at a case, not a label"
else
  echo "FAIL  callstop  read $(od -An -tu4 -N4 "$TMP/callstop.bin" | tr -d ' ')"; fail=1
fi
python3 - "$TMP" <<'PYCBR'
import sys, os
open(os.path.join(sys.argv[1], 'cbrz.bin'), 'wb').write(b'\x00' * 4)
PYCBR
runf kernels/callbarrec.ptx callbarrec --grid 1 --threads 1 --buffers 1 \
    --load 0="$TMP/cbrz.bin"
"$WEFT" run kernels/callbarrec.ptx --kernel callbarrec --grid 1 --threads 1 --buffers 1 \
    --load 0="$TMP/cbrz.bin" --cpu-only --dump 0="$TMP/cbr.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/cbr.bin" | tr -d ' ')" = "4" ]; then
  echo "ok    callbarrec a recursion that stops keeps every frame: four frames, four adds"
else
  echo "FAIL  callbarrec read $(od -An -tu4 -N4 "$TMP/cbr.bin" | tr -d ' ')"; fail=1
fi
python3 - "$TMP" <<'PYCY'
import sys, os
open(os.path.join(sys.argv[1], 'cycz.bin'), 'wb').write(b'\x00' * 4)
PYCY
runf kernels/cycbar.ptx cycbar --grid 1 --threads 1 --buffers 1 --load 0="$TMP/cycz.bin"
"$WEFT" run kernels/cycbar.ptx --kernel cycbar --grid 1 --threads 1 --buffers 1 \
    --load 0="$TMP/cycz.bin" --cpu-only --dump 0="$TMP/cyc.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/cyc.bin" | tr -d ' ')" = "3" ]; then
  echo "ok    cycbar    a cycle that stops at a barrier keeps its frames and its place"
else
  echo "FAIL  cycbar    read $(od -An -tu4 -N4 "$TMP/cyc.bin" | tr -d ' ')"; fail=1
fi
runf kernels/callself.ptx callself --grid 1 --threads 1 --buffers 1
"$WEFT" run kernels/callself.ptx --kernel callself --grid 1 --threads 1 --buffers 1 \
    --cpu-only --dump 0="$TMP/callself.bin" >/dev/null
if [ "$(od -An -tu4 -N4 "$TMP/callself.bin" | tr -d ' ')" = "21" ]; then
  echo "ok    callself  a tail self-call loops in place: six down to zero sums to 21"
else
  echo "FAIL  callself  read $(od -An -tu4 -N4 "$TMP/callself.bin" | tr -d ' ')"; fail=1
fi

python3 - "$TMP" <<'PYR'
import struct, sys, os
d = sys.argv[1]
vals = [0, 1, 2, 5, 7, 10, 255, 4096]
tri = [0 if v == 0 else 3 * v + 1 for v in vals]
open(os.path.join(d, 'callretin.bin'), 'wb').write(struct.pack('<8I', *vals))
open(os.path.join(d, 'callretexp.bin'), 'wb').write(struct.pack('<8I', *tri))
open(os.path.join(d, 'callret2exp.bin'), 'wb').write(struct.pack('<8I', *[2 * t for t in tri]))
PYR
narrowref callret    kernels/callret.ptx callret  8 8 8,8 1 callretin.bin callretexp.bin
runf kernels/callret.ptx callret  --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callretin.bin"
narrowref callret2   kernels/callret.ptx callret2 8 8 8,8 1 callretin.bin callret2exp.bin
runf kernels/callret.ptx callret2 --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callretin.bin"

python3 - "$TMP" <<'PYB'
import struct, sys, os
d = sys.argv[1]
vals = [3, 11, 7, 0, 19, 5, 24, 2]
n = len(vals)
out = [(vals[i] + 1) + (vals[(i + 1) % n] + 1) for i in range(n)]
open(os.path.join(d, 'callbarin.bin'), 'wb').write(struct.pack('<8I', *vals))
open(os.path.join(d, 'callbarexp.bin'), 'wb').write(struct.pack('<8I', *out))
PYB
narrowref callbar    kernels/callbar.ptx callbar 8 8 8,8 1 callbarin.bin callbarexp.bin
runf kernels/callbar.ptx callbar --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callbarin.bin"

python3 - "$TMP" <<'PYM'
import struct, sys, os
d = sys.argv[1]
vals = [0, 1, 2, 5, 7, 10, 255, 4096]
open(os.path.join(d, 'callmultiin.bin'), 'wb').write(struct.pack('<8I', *vals))
open(os.path.join(d, 'callmultiexp.bin'), 'wb').write(struct.pack('<8I', *[2 * v + 1 for v in vals]))
PYM
narrowref callmulti kernels/callmulti.ptx callmulti 8 8 8,8 1 callmultiin.bin callmultiexp.bin
runf kernels/callmulti.ptx callmulti --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callmultiin.bin"
cp tests/refused/callmulti3.invalid "$TMP/callmulti3.ptx"
callmulti3_rc=0
callmulti3_out=$("$WEFT" run "$TMP/callmulti3.ptx" --kernel callmulti3 --grid 1 --threads 1 --buffers 1,1 --cpu-only 2>&1) || callmulti3_rc=$?
if [ $callmulti3_rc -ne 0 ] && [ "${callmulti3_out#*names 3 destinations}" != "$callmulti3_out" ]; then
  echo "ok    callmulti3 a return list with more destinations than the callee returns is refused"
else
  echo "FAIL  callmulti3 the count mismatch was not refused by name (rc=$callmulti3_rc)"; fail=1
fi
python3 - "$TMP" <<'PYCAL'
import struct, sys
tmp = sys.argv[1]
vals = [1, 2, 3, 0x7fffffff, 0x80000000, 0xffffffff, 0x12345678, 0xdeadbeef,
        0x0f0f0f0f, 0xf0f0f0f0, 0x00ff00ff, 0xff00ff00, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
open(tmp + "/callin.bin", "wb").write(struct.pack("<32I", *vals))
open(tmp + "/calldropexp.bin", "wb").write(struct.pack("<32I", *[(7 * (i + 1)) & 0xffffffff for i in range(32)]))
open(tmp + "/callmissexp.bin", "wb").write(struct.pack("<32I", *[(3 * v) & 0xffffffff for v in vals]))
PYCAL
"$WEFT" run kernels/calldrop.ptx --kernel calldrop --grid 32 --threads 32 --buffers 32 \
    --cpu-only --dump 0="$TMP/calldrop.bin" >/dev/null
if cmp -s "$TMP/calldrop.bin" "$TMP/calldropexp.bin"; then
  echo "ok    calldrop  a call with no return list stored what the callee wrote"
else
  echo "FAIL  calldrop  $(cmp "$TMP/calldrop.bin" "$TMP/calldropexp.bin" 2>&1 | head -1)"; fail=1
fi
"$WEFT" run kernels/callmiss.ptx --kernel callmiss --grid 32 --threads 32 --buffers 32,32 \
    --load 0="$TMP/callin.bin" --cpu-only --dump 1="$TMP/callmiss.bin" >/dev/null
if cmp -s "$TMP/callmiss.bin" "$TMP/callmissexp.bin"; then
  echo "ok    callmiss  the one-slot return list carried the triple the callee computed"
else
  echo "FAIL  callmiss  $(cmp "$TMP/callmiss.bin" "$TMP/callmissexp.bin" 2>&1 | head -1)"; fail=1
fi

cp tests/refused/calldrop.invalid "$TMP/calldrop_bad.ptx"
calldrop_rc=0
calldrop_out=$("$WEFT" run "$TMP/calldrop_bad.ptx" --kernel calldrop --grid 1 --threads 1 --buffers 1,1 --cpu-only 2>&1) || calldrop_rc=$?
if [ $calldrop_rc -ne 0 ] && [ "${calldrop_out#*returns nothing}" != "$calldrop_out" ]; then
  echo "ok    calldrop  a return list around a function that returns nothing is refused"
else
  echo "FAIL  calldrop  the mismatch was not refused by name (rc=$calldrop_rc)"; fail=1
fi
cp tests/refused/callmiss.invalid "$TMP/callmiss_bad.ptx"
callmiss_rc=0
callmiss_out=$("$WEFT" run "$TMP/callmiss_bad.ptx" --kernel callmiss --grid 1 --threads 1 --buffers 1,1 --cpu-only 2>&1) || callmiss_rc=$?
if [ $callmiss_rc -ne 0 ] && [ "${callmiss_out#*no list to put it in}" != "$callmiss_out" ]; then
  echo "ok    callmiss  a returned value the call ignores is refused, not dropped"
else
  echo "FAIL  callmiss  an ignored return value was not refused by name (rc=$callmiss_rc)"; fail=1
fi

python3 - "$TMP" <<'PYB'
import struct, sys, os
d = sys.argv[1]
vals = [3, 11, 7, 0, 19, 5, 24, 2]
n = len(vals)
out = [(vals[i] + 1) + (vals[(i + 1) % n] + 1) for i in range(n)]
open(os.path.join(d, 'callpredin.bin'), 'wb').write(struct.pack('<8I', *vals))
open(os.path.join(d, 'callpredexp.bin'), 'wb').write(struct.pack('<8I', *out))
PYB
narrowref callpred  kernels/callpred.ptx callpred 8 8 8,8 1 callpredin.bin callpredexp.bin
runf kernels/callpred.ptx callpred --grid 8 --threads 8 --buffers 8,8 --load 0="$TMP/callpredin.bin"
runf kernels/surfq.ptx surfq --grid 1 --threads 1 --buffers 4 --textures 64x64
runf kernels/more.ptx m4 --grid 8 --threads 8 --buffers 8,32 --load 0="$TMP/morein.bin"
runf kernels/warpbits.ptx actm --grid 32 --threads 32 --buffers 32
runf kernels/warpbits.ptx actm --grid 8 --threads 8 --buffers 8
runf kernels/warpbits.ptx mvi --grid 32 --threads 32 --buffers 32,32 --load 0="$TMP/wbin.bin"
runf kernels/f16x2_trans.ptx p16tx --grid 32 --threads 32 --buffers 32,32,32,32 --load 0="$TMP/p16in.bin"

runf kernels/narrowtran.ptx narrowtran --grid 8 --threads 8 --buffers 8,48

if [ -f kernels/spirv/halfmath.spv ]; then
  "$WEFT" run kernels/spirv/halfmath.spv --kernel halfmath --grid 32 --threads 32 \
      --buffers 32,32 --load 0="$TMP/hin32.bin" --cpu-only --dump 1="$TMP/halfmath.bin" >/dev/null
  if cmp -s "$TMP/halfmath.bin" "$TMP/exp_spvh.bin"; then
    echo "ok    spirv:halfmath  half arithmetic and a workgroup half array are exact"
  else
    echo "FAIL  spirv:halfmath  $(cmp "$TMP/halfmath.bin" "$TMP/exp_spvh.bin" 2>&1 | head -1)"; fail=1
  fi
else
  echo "skip  spirv:halfmath  no kernels/spirv/halfmath.spv on this host"
fi

python3 -c 'import struct,sys; sys.stdout.buffer.write(b"".join(struct.pack("<I", (((i+1)%16)+1)*1000 + ((i+1)%16) + 100) for i in range(16)))' > "$TMP/exp_ms.bin"
"$WEFT" run kernels/modshared.ptx --kernel modshared --grid 16 --threads 16 --buffers 16 \
    --cpu-only --dump 0="$TMP/ms.bin" >/dev/null
if cmp -s "$TMP/ms.bin" "$TMP/exp_ms.bin"; then
  echo "ok    modshared  a module-scope .shared array is laid out and addressed"
else
  echo "FAIL  modshared  $(cmp "$TMP/ms.bin" "$TMP/exp_ms.bin" 2>&1 | head -1)"; fail=1
fi
# The same kernel on the GPU column, which is where the missing arena showed up.
runf kernels/modshared.ptx modshared --grid 16 --threads 16 --buffers 16

runf kernels/cvtmx.ptx cvtmx --grid 1 --threads 1 --buffers 6
"$WEFT" run kernels/cvtmx.ptx --kernel cvtmx --grid 1 --threads 1 --buffers 6 \
    --cpu-only --dump 0="$TMP/cvtmx.bin" >/dev/null
if [ "$(head -c 24 "$TMP/cvtmx.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "00000055 00000000 00000072 00001f02 00001f01 00001f3f " ]; then
  echo "ok    cvtmx  fp4 and fp6 rounded onto their own grids and saturated by sign"
else
  echo "FAIL  cvtmx  read $(head -c 24 "$TMP/cvtmx.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtmx4.ptx cvtmx4 --grid 1 --threads 1 --buffers 3
"$WEFT" run kernels/cvtmx4.ptx --kernel cvtmx4 --grid 1 --threads 1 --buffers 3 \
    --cpu-only --dump 0="$TMP/cvtmx4.bin" >/dev/null
if [ "$(head -c 12 "$TMP/cvtmx4.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "0000250f 0c00001f 1f040f1f " ]; then
  echo "ok    cvtmx4  four sources packed high first, with relu ahead of the convert"
else
  echo "FAIL  cvtmx4  read $(head -c 12 "$TMP/cvtmx4.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtmx.ptx cvtrs --grid 1 --threads 1 --buffers 4
"$WEFT" run kernels/cvtmx.ptx --kernel cvtrs --grid 1 --threads 1 --buffers 4 \
    --cpu-only --dump 0="$TMP/cvtrs.bin" >/dev/null
if [ "$(head -c 16 "$TMP/cvtrs.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "3c01c401 3c02c402 3f827f7f 3c023c01 " ]; then
  echo "ok    cvtrs  the random bits carried into the discarded mantissa as the isa splits them"
else
  echo "FAIL  cvtrs  read $(head -c 16 "$TMP/cvtrs.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/f64sat.ptx f64sat --grid 1 --threads 1 --buffers 9
"$WEFT" run kernels/f64sat.ptx --kernel f64sat --grid 1 --threads 1 --buffers 18 \
    --cpu-only --dump 0="$TMP/f64sat.bin" >/dev/null
if [ "$(head -c 72 "$TMP/f64sat.bin" | od -An -tx8 -v | tr -s ' ' '\n' | sed '/^$/d; s/^0$/0000000000000000/' | tr '\n' ' ')" = \
    "3ff0000000000000 0000000000000000 0000000000000000 3fd0000000000000 8000000000000000 3ff0000000000000 0000000000000000 3ff0000000000000 3ff0000000000000 " ]; then
  echo "ok    f64sat   the 64-bit clamp held [0,1], flushed the nan and kept the -0.0"
else
  echo "FAIL  f64sat   read $(head -c 72 "$TMP/f64sat.bin" | od -An -tx8 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtqual.ptx cvtrsx4 --grid 1 --threads 1 --buffers 5
"$WEFT" run kernels/cvtqual.ptx --kernel cvtrsx4 --grid 1 --threads 1 --buffers 5 \
    --cpu-only --dump 0="$TMP/cvtrsx4.bin" >/dev/null
if [ "$(head -c 20 "$TMP/cvtrsx4.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "393840b0 0000324a 0d0c0c34 3d3cc034 00000651 " ]; then
  echo "ok    cvtrsx4  the random word carried each of the four values on its own"
else
  echo "FAIL  cvtrsx4  read $(head -c 20 "$TMP/cvtrsx4.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtqual.ptx cvtrz2 --grid 1 --threads 1 --buffers 3
"$WEFT" run kernels/cvtqual.ptx --kernel cvtrz2 --grid 1 --threads 1 --buffers 3 \
    --cpu-only --dump 0="$TMP/cvtrz2.bin" >/dev/null
if [ "$(head -c 12 "$TMP/cvtrz2.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "000039b9 0000003b 3c03bc03 " ]; then
  echo "ok    cvtrz2  the truncating pair converts dropped the tail on both signs"
else
  echo "FAIL  cvtrz2  read $(head -c 12 "$TMP/cvtrz2.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtqual.ptx cvtscaled --grid 1 --threads 1 --buffers 5
"$WEFT" run kernels/cvtqual.ptx --kernel cvtscaled --grid 1 --threads 1 --buffers 5 \
    --cpu-only --dump 0="$TMP/cvtscaled.bin" >/dev/null
if [ "$(head -c 20 "$TMP/cvtscaled.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "00004840 00000080 00003830 00000000 40804000 " ]; then
  echo "ok    cvtscaled  the scale divided, flushed, halved and multiplied, and pzo folded a zero"
else
  echo "FAIL  cvtscaled  read $(head -c 20 "$TMP/cvtscaled.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtf8x2.ptx cvtdown --grid 1 --threads 1 --buffers 6
"$WEFT" run kernels/cvtf8x2.ptx --kernel cvtdown --grid 1 --threads 1 --buffers 6 \
    --cpu-only --dump 0="$TMP/cvtdown.bin" >/dev/null
if [ "$(head -c 24 "$TMP/cvtdown.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "3e003c00 00003c00 47803400 4f002c00 3c003c00 40000000 " ]; then
  echo "ok    cvtdown  five packed formats read out into halves, relu zeroing one of them"
else
  echo "FAIL  cvtdown  read $(head -c 24 "$TMP/cvtdown.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtf8x2dn.ptx cvtup --grid 1 --threads 1 --buffers 6
"$WEFT" run kernels/cvtf8x2dn.ptx --kernel cvtup --grid 1 --threads 1 --buffers 6 \
    --cpu-only --dump 0="$TMP/cvtup.bin" >/dev/null
if [ "$(head -c 24 "$TMP/cvtup.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "00000042 00000002 00001f0c 00001f04 00003c38 00000000 " ]; then
  echo "ok    cvtup  two halves packed down, relu and pzo doing what they say"
else
  echo "FAIL  cvtup  read $(head -c 24 "$TMP/cvtup.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/cvtf8x2simd.ptx cvtsimd --grid 1 --threads 1 --buffers 7
"$WEFT" run kernels/cvtf8x2simd.ptx --kernel cvtsimd --grid 1 --threads 1 --buffers 7 \
    --cpu-only --dump 0="$TMP/cvtsimd.bin" >/dev/null
if [ "$(head -c 28 "$TMP/cvtsimd.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "44444444 38383838 41414141 4b4b4b4b 7e7e7e7e 44444444 44444444 " ]; then
  echo "ok    cvtsimd  four packed lanes added, subtracted, multiplied and fused"
else
  echo "FAIL  cvtsimd  read $(head -c 28 "$TMP/cvtsimd.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

runf kernels/mbarred.ptx mbarred --grid 1 --threads 1 --buffers 4
"$WEFT" run kernels/mbarred.ptx --kernel mbarred --grid 1 --threads 1 --buffers 4 \
    --cpu-only --dump 0="$TMP/reduced.bin" >/dev/null
read -r red0 red1 red2 red3 <<< "$(od -An -tu4 -v "$TMP/reduced.bin")"
if [ "$red0" = "11" ] && [ "$red1" = "22" ] && [ "$red2" = "33" ] && [ "$red3" = "44" ]; then
  echo "ok    mbarred  the reduce copy added 10, 20, 30, 40 to 1, 2, 3, 4"
else
  echo "FAIL  mbarred  read $red0 $red1 $red2 $red3; wanted 11 22 33 44"; fail=1
fi

runf kernels/tensormap.ptx tensormap --grid 1 --threads 1 --buffers 21,12,32
"$WEFT" run kernels/tensormap.ptx --kernel tensormap --grid 1 --threads 1 --buffers 21,12,32 \
    --cpu-only --dump 0="$TMP/tensormap.bin" >/dev/null
tm_got="$(od -An -tu4 -v "$TMP/tensormap.bin" | tr '\n' ' ' | tr -s ' ' | sed 's/^ //; s/ $//')"
tm_want="6 7 0 10 11 0 0 0 0 0 1 2 3 4 101 102 7 8 103 104 11"
if [ "$tm_got" = "$tm_want" ]; then
  echo "ok    tensormap the box loaded zero where it hung off and the store landed on 5, 6, 9, 10"
else
  echo "FAIL  tensormap read $tm_got; wanted $tm_want"; fail=1
fi

runf kernels/tensormap2.ptx tensorswz --grid 1 --threads 1 --buffers 3,64,32
"$WEFT" run kernels/tensormap2.ptx --kernel tensorswz --grid 1 --threads 1 --buffers 3,64,32 \
    --cpu-only --dump 0="$TMP/tensorswz.bin" >/dev/null
sw_got="$(od -An -tu4 -v "$TMP/tensorswz.bin" | tr '\n' ' ' | tr -s ' ' | sed 's/^ //; s/ $//')"
if [ "$sw_got" = "0 32 33" ]; then
  echo "ok    tensorswz the 128-byte swizzle moved row 1 sixteen bytes along"
else
  echo "FAIL  tensorswz read $sw_got; wanted 0 32 33"; fail=1
fi

runf kernels/tensormap3.ptx tensorim2col --grid 1 --threads 1 --buffers 12,18,32
"$WEFT" run kernels/tensormap3.ptx --kernel tensorim2col --grid 1 --threads 1 --buffers 12,18,32 \
    --cpu-only --dump 0="$TMP/tensorim2col.bin" >/dev/null
im_got="$(od -An -tu4 -v "$TMP/tensorim2col.bin" | tr '\n' ' ' | tr -s ' ' | sed 's/^ //; s/ $//')"
if [ "$im_got" = "0 1 2 3 4 5 2 3 4 5 0 0" ]; then
  echo "ok    tensorim2col the column walked the width and the padded edge came back zero"
else
  echo "FAIL  tensorim2col read $im_got; wanted 0 1 2 3 4 5 2 3 4 5 0 0"; fail=1
fi

runf kernels/tensormap4.ptx tensorim2st --grid 1 --threads 1 --buffers 18,18,32
"$WEFT" run kernels/tensormap4.ptx --kernel tensorim2st --grid 1 --threads 1 --buffers 18,18,32 \
    --cpu-only --dump 0="$TMP/tensorim2st.bin" >/dev/null
is_got="$(od -An -tu4 -v "$TMP/tensorim2st.bin" | tr '\n' ' ' | tr -s ' ' | sed 's/^ //; s/ $//')"
is_want="101 102 201 202 203 204 6 7 8 9 10 11 12 13 14 15 16 17"
if [ "$is_got" = "$is_want" ]; then
  echo "ok    tensorim2st the scatter landed where the column said and stopped at the edge"
else
  echo "FAIL  tensorim2st read $is_got; wanted $is_want"; fail=1
fi

runf kernels/prefetch.ptx prefetch --grid 1024 --threads 64 --buffers 1024,1024
"$WEFT" run kernels/prefetch.ptx --kernel prefetch --grid 1024 --threads 64 --buffers 1024,1024 \
    --cpu-only --dump 0="$TMP/pf0.bin" --dump 1="$TMP/pf1.bin" >/dev/null
if cmp -s "$TMP/pf0.bin" "$TMP/pf1.bin"; then
  echo "ok    prefetch   the three hints moved no data and changed no answer"
else
  echo "FAIL  prefetch   a hint changed what the kernel wrote"; fail=1
fi

runf kernels/ldmatrixb8.ptx ldmatrixb8 --grid 32 --threads 32 --buffers 64,64
"$WEFT" run kernels/ldmatrixb8.ptx --kernel ldmatrixb8 --grid 32 --threads 32 --buffers 64,64 \
    --cpu-only --dump 0="$TMP/lb8.bin" --dump 1="$TMP/lb8b.bin" >/dev/null
if od -An -tu4 -v "$TMP/lb8.bin" | awk '{ for (i = 1; i <= NF; i++) { n = n0++;
      l = int(n / 2); r = n % 2; w = 0;
      for (q = 0; q < 4; q++) { s = 2 * r + (q % 2) + 4 * int(q / 2);
        a = s % 2; b = int(s / 2) % 2; c = int(s / 4);
        row = 2 * (l % 4) + a + 8 * c; col = int(l / 4) + 8 * b;
        w += (16 * row + col) * 256 ^ q }
      if ($i != w) bad++ } } END { exit(bad ? 1 : 0) }'; then
  echo "ok    ldmatrixb8  two registers a lane hold the bytes that map names"
else
  echo "FAIL  ldmatrixb8  the byte fragment is not that map"; fail=1
fi
if [ "$(od -An -tu1 -v "$TMP/lb8b.bin" | tr -s ' ' '\n' | sed '/^$/d' | sort -n | tr '\n' ' ')" = \
    "$(seq 0 255 | tr '\n' ' ')" ]; then
  echo "ok    stmatrixb8  the two 16x8 stores cover the tile exactly once"
else
  echo "FAIL  stmatrixb8  the store left a duplicate or a gap"; fail=1
fi

runf kernels/stmatrixb8map.ptx stmb8 --grid 32 --threads 32 --buffers 384
env WEFT_WARP=1 "$WEFT" run kernels/stmatrixb8map.ptx --kernel stmb8 --grid 32 \
    --threads 32 --buffers 384 --cpu-only --dump 0="$TMP/stmatrixb8map.bin" >/dev/null
if python3 - "$TMP/stmatrixb8map.bin" <<'PY'
import struct, sys
w = struct.unpack('<384I', open(sys.argv[1], 'rb').read(1536))
def byte(r, o):
    if r == 1:                            # the single-matrix store, 256 bytes in
        return (o - 256) if 256 <= o < 384 else 0
    return (o % 128) ^ (64 * (o // 128))  # offset XOR matrix number
bad = 0
for n, x in enumerate(w):
    e = sum(byte(n // 128, (n % 128) * 4 + k) << (8 * k) for k in range(4))
    if x != e: bad += 1
if bad:
    print("  %d of 384 words differ" % bad)
sys.exit(0 if bad == 0 else 1)
PY
then
  echo "ok    stmatrixb8map  byte tiles, a constant offset, and lane-supplied addresses"
else
  echo "FAIL  stmatrixb8map  a byte landed where the map does not put it"; fail=1
fi

runf kernels/prmtmodes.ptx prmtmodes --grid 256 --threads 256 --buffers 1792
"$WEFT" run kernels/prmtmodes.ptx --kernel prmtmodes --grid 256 --threads 256 --buffers 1792 \
    --cpu-only --dump 0="$TMP/prmt.bin" >/dev/null
if od -An -tu1 -v "$TMP/prmt.bin" | awk '{ for (i = 1; i <= NF; i++) { n = n0++; j = n % 4
      lane = int(n / 28); w = int(n / 4) % 7; l = lane % 64; k = lane % 4
      for (q = 0; q < 8; q++) src[q] = (4 * l + q) % 256
      if (w == 0) idx = (k + j) % 8
      else if (w == 1) idx = (k - j + 8) % 8
      else if (w == 2) idx = k
      else if (w == 3) idx = (k > j) ? k : j
      else if (w == 4) idx = (k < j) ? k : j
      else idx = 2 * (k % 2) + j % 2
      if (w == 6) { sel = int(lane / 16 ^ j) % 16; byte = src[sel % 8]
        want = (sel >= 8) ? ((byte >= 128) ? 255 : 0) : byte }
      else want = src[idx]
      if ($i != want) bad++ } } END { exit(bad ? 1 : 0) }'; then
  echo "ok    prmtmodes  every named mode and the nibble form follow the isa table"
else
  echo "FAIL  prmtmodes  a mode's bytes are not the ones the table names"; fail=1
fi

runf kernels/ldmatrixsub.ptx ldmsub --grid 32 --threads 32 --buffers 160
"$WEFT" run kernels/ldmatrixsub.ptx --kernel ldmsub --grid 32 --threads 32 --buffers 160 \
    --cpu-only --dump 0="$TMP/lsub.bin" >/dev/null
if od -An -tu4 -v "$TMP/lsub.bin" | awk '{ for (i = 1; i <= NF; i++) { n = n0++; w = 0;
      if (n < 128) { m = n % 64; l = int(m / 2); c0 = 8 * (m % 2); row = int(l / 4)
        for (q = 0; q < 4; q++) { col = 4 * (l % 4) + q; c = c0 + row
          if (n < 64) v = (c < 8) ? (13 * c + 7 * col) % 64 : (29 * (c - 8) + 11 * col + 5) % 64
          else v = (c < 8) ? (5 * c + 3 * col) % 16 : (11 * (c - 8) + 7 * col + 3) % 16
          w += v * 256 ^ q } }
      else { l = n - 128; row = int(l / 4)
        for (q = 0; q < 4; q++) { col = 4 * (l % 4) + q
          v = (5 * row + 3 * col) % 16 - 8; w += ((v + 256) % 256) * 256 ^ q } }
      if ($i != w) bad++ } } END { exit(bad ? 1 : 0) }'; then
  echo "ok    ldmatrixsub  the 6- and 4-bit containers widen to the bytes the map names"
else
  echo "FAIL  ldmatrixsub  a packed fragment is not the map the container describes"; fail=1
fi

runf kernels/ldmatrixx4.ptx ldmx4 --grid 32 --threads 32 --buffers 384
env WEFT_WARP=1 "$WEFT" run kernels/ldmatrixx4.ptx --kernel ldmx4 --grid 32 --threads 32 \
    --buffers 384 --cpu-only --dump 0="$TMP/ldmx4.bin" >/dev/null
if python3 - "$TMP/ldmx4.bin" <<'PY'
import struct, sys
w = struct.unpack('<384I', open(sys.argv[1], 'rb').read(1536))
bad = 0
for l in range(32):
    r, c = l >> 2, l & 3
    for k in range(4):
        p = 64 * k + 8 * r + 2 * c
        t = 64 * k + 8 * (2 * c) + r
        if w[4 * l + k] != p | ((p + 1) << 16): bad += 1
        if w[128 + 4 * l + k] != t | ((t + 8) << 16): bad += 1
        if w[256 + 4 * l + k] != p | ((p + 1) << 16): bad += 1
if bad:
    print("  %d of 384 words differ" % bad)
sys.exit(0 if bad == 0 else 1)
PY
then
  echo "ok    ldmatrixx4  four tiles, named and per-lane, plain and transposed, match the map"
else
  echo "FAIL  ldmatrixx4  a tile at a non-zero offset is not the map"; fail=1
fi

runf kernels/ldmatrixmulti.ptx ldmpack4 --grid 32 --threads 32 --buffers 384
env WEFT_WARP=1 "$WEFT" run kernels/ldmatrixmulti.ptx --kernel ldmpack4 --grid 32 \
    --threads 32 --buffers 384 --cpu-only --dump 0="$TMP/ldmpack4.bin" >/dev/null
if python3 - "$TMP/ldmpack4.bin" <<'PY'
import struct, sys
w = struct.unpack('<384I', open(sys.argv[1], 'rb').read(1536))
bad = 0
for b, bits in ((0, 6), (1, 6), (2, 4)):
    for l in range(32):
        row, c0 = l >> 2, 4 * (l & 3)
        for k in range(4):
            n = 0
            for q in range(4):
                col = c0 + q
                v = (7 * (8 * k + row) + 13 * col) if bits == 6 else (5 * (8 * k + row) + 3 * col)
                n |= ((v & 63) if bits == 6 else (v & 15)) << (8 * q)
            if w[128 * b + 4 * l + k] != n: bad += 1
if bad:
    print("  %d of 384 words differ" % bad)
sys.exit(0 if bad == 0 else 1)
PY
then
  echo "ok    ldmatrixmulti  four 6-bit containers, named and per-lane, and four 4-bit"
else
  echo "FAIL  ldmatrixmulti  a tile at a non-zero offset is not the container map"; fail=1
fi

runf kernels/ldmatrixmulti.ptx ldmb8x2 --grid 32 --threads 32 --buffers 128
env WEFT_WARP=1 "$WEFT" run kernels/ldmatrixmulti.ptx --kernel ldmb8x2 --grid 32 \
    --threads 32 --buffers 128 --cpu-only --dump 0="$TMP/ldmb8x2.bin" >/dev/null
if python3 - "$TMP/ldmb8x2.bin" <<'PY'
import struct, sys
w = struct.unpack('<128I', open(sys.argv[1], 'rb').read(512))
bad = 0
for m in range(2):
    for l in range(32):
        for r in range(2):
            n = 0
            for q in range(4):
                s = 2 * r + (q & 1) + 4 * (q >> 1)
                row = 2 * (l & 3) + (s & 1) + 8 * (s >> 2)
                col = (l >> 2) + 8 * ((s >> 1) & 1)
                n |= ((16 * row + col) ^ (0x5A * m)) << (8 * q)
            if w[64 * m + 2 * l + r] != n: bad += 1
if bad:
    print("  %d of 128 words differ" % bad)
sys.exit(0 if bad == 0 else 1)
PY
then
  echo "ok    ldmatrixmulti  the second byte tile is its own 16x16 map, 256 bytes on"
else
  echo "FAIL  ldmatrixmulti  the second byte tile is the first one's bytes"; fail=1
fi

runf kernels/stmatrix.ptx stmatrix --grid 32 --threads 32 --buffers 64
"$WEFT" run kernels/stmatrix.ptx --kernel stmatrix --grid 32 --threads 32 --buffers 64 \
    --cpu-only --dump 0="$TMP/stmatrix.bin" >/dev/null
if od -An -tu4 -v "$TMP/stmatrix.bin" | awk '{ for (i = 1; i <= NF; i++) { n = n0++; l = n % 32;
      if (n < 32) { want = 2 * l + (2 * l + 1) * 65536 }
      else { a = 8 * (2 * (l % 4)) + int(l / 4); b = 8 * (2 * (l % 4) + 1) + int(l / 4);
             want = a + b * 65536 }
      if ($i != want) bad++ } } END { exit(bad ? 1 : 0) }'; then
  echo "ok    stmatrix  the fragment went back where ldmatrix found it, and the transpose too"
else
  echo "FAIL  stmatrix  the round trip is not the tile that was loaded"; fail=1
fi

runf kernels/ldstoreloop.ptx ldstoreloop --grid 32 --threads 32 --buffers 256
env WEFT_WARP=1 "$WEFT" run kernels/ldstoreloop.ptx --kernel ldstoreloop --grid 32 \
    --threads 32 --buffers 256 --cpu-only --dump 0="$TMP/ldstoreloop.bin" >/dev/null
if python3 - "$TMP/ldstoreloop.bin" <<'PY'
import struct, sys
w = struct.unpack('<256I', open(sys.argv[1], 'rb').read(1024))
bad = 0
for n, x in enumerate(w):
    l, i = (n % 128) // 4, n % 4
    h = 8 * l + 2 * i
    if x != (h | (h + 1) << 16): bad += 1
if bad:
    print("  %d of 256 words differ" % bad)
sys.exit(0 if bad == 0 else 1)
PY
then
  echo "ok    ldstoreloop  a walked store address is the same on every round"
else
  echo "FAIL  ldstoreloop  the loop wrote a round's word to a fixed slot"; fail=1
fi

# `stmatrix.x4` is four tiles at 128 bytes apiece, and `.trans` of four tiles at once.
runf kernels/stmatrixmulti.ptx stmatrixmulti --grid 32 --threads 32 --buffers 256
env WEFT_WARP=1 "$WEFT" run kernels/stmatrixmulti.ptx --kernel stmatrixmulti --grid 32 \
    --threads 32 --buffers 256 --cpu-only --dump 0="$TMP/stmatrixmulti.bin" >/dev/null
if python3 - "$TMP/stmatrixmulti.bin" <<'PY'
import struct, sys
w = struct.unpack('<256I', open(sys.argv[1], 'rb').read(1024))
def trans(h):                      # tile k's (r, c) holds what (c, r) held
    return (h & ~63) | (8 * (h & 7)) | ((h >> 3) & 7)
bad = 0
for n, x in enumerate(w):
    h = 2 * (n % 128)
    e = (h | (h + 1) << 16) if n < 128 else (trans(h) | (trans(h + 1) << 16))
    if x != e: bad += 1
if bad:
    print("  %d of 256 words differ" % bad)
sys.exit(0 if bad == 0 else 1)
PY
then
  echo "ok    stmatrixmulti  four tiles back in place, and the four-tile transpose"
else
  echo "FAIL  stmatrixmulti  a tile past the first is not the tile that was loaded"; fail=1
fi

runf kernels/triton/dots8.ptx dots8_kernel --grid 32 --threads 32 --buffers 128
"$WEFT" run kernels/triton/dots8.ptx --kernel dots8_kernel --grid 32 --threads 32 --buffers 128 \
    --cpu-only --dump 0="$TMP/dots8.bin" >/dev/null
if od -An -tu4 -v "$TMP/dots8.bin" | awk '{ for (i = 1; i <= NF; i++) { n = n0++; r = int(n / 8); c = n % 8;
      if ($i != 80 * (r + 1) * (c + 1)) bad++ } } END { exit(bad ? 1 : 0) }'; then
  echo "ok    dots8  the integer dot is 80 (r + 1) (n + 1) at all 128 outputs"
else
  echo "FAIL  dots8  the integer product is not the one the fragments describe"; fail=1
fi

MT="$TMP/mma"
mkdir -p "$MT"
if python3 - "$MT" <<'MMAIN'
import random, struct, sys
out = sys.argv[1]
random.seed(7)
half = {0: 0x0000, 1: 0x3C00, 2: 0x4000, -1: 0xBC00, -2: 0xC000}
def write(name, vals, fmt):
    open(out + '/' + name, 'wb').write(struct.pack('<%d%s' % (len(vals), fmt), *vals))
write('k8a.bin',  [half[random.randint(-2, 2)] for _ in range(16 * 8)], 'H')
write('k8bt.bin', [half[random.randint(-2, 2)] for _ in range(8 * 8)], 'H')
write('k4a.bin',  [float(random.randint(-2, 2)) for _ in range(16 * 4)], 'f')
write('k4b.bin',  [float(random.randint(-2, 2)) for _ in range(4 * 8)], 'f')
write('s8a.bin',  [random.randint(-4, 4) for _ in range(8 * 16)], 'b')
write('s8bt.bin', [random.randint(-4, 4) for _ in range(8 * 16)], 'b')
def pack(vals):
    return bytes(((vals[i] & 0xF) | ((vals[i + 1] & 0xF) << 4)) for i in range(0, len(vals), 2))
open(out + '/s4a.bin', 'wb').write(pack([random.randint(-8, 7) for _ in range(16 * 64)]))
open(out + '/s4bt.bin', 'wb').write(pack([random.randint(-8, 7) for _ in range(8 * 64)]))
open(out + '/f4a.bin', 'wb').write(pack([random.randint(0, 15) for _ in range(16 * 32)]))
open(out + '/f4bt.bin', 'wb').write(pack([random.randint(0, 15) for _ in range(8 * 32)]))
open(out + '/f6a.bin', 'wb').write(bytes(random.randint(0, 63) for _ in range(16 * 32)))
open(out + '/f6bt.bin', 'wb').write(bytes(random.randint(0, 63) for _ in range(8 * 32)))
open(out + '/f4ka.bin', 'wb').write(pack([random.randint(0, 15) for _ in range(16 * 64)]))
open(out + '/f4kbt.bin', 'wb').write(pack([random.randint(0, 15) for _ in range(8 * 64)]))
MMAIN
then
  "$WEFT" run kernels/mma8.ptx --kernel mmak8f16 --grid 32 --threads 32 --buffers 64,32,128 \
      --load 0="$MT/k8a.bin" --load 1="$MT/k8bt.bin" --cpu-only --dump 2="$MT/k8d.bin" >/dev/null
  if python3 tools/mma_ref.py k8f16 "$MT/k8a.bin" "$MT/k8bt.bin" "$MT/k8d.bin" >/dev/null; then
    echo "ok    mma8        m16n8k8 halves: every lane's A row and B column, both halves"
  else
    echo "FAIL  mma8        m16n8k8 halves are not the product the fragments describe"; fail=1
  fi
  "$WEFT" run kernels/mma8.ptx --kernel mmak4tf32 --grid 32 --threads 32 --buffers 64,32,128 \
      --load 0="$MT/k4a.bin" --load 1="$MT/k4b.bin" --cpu-only --dump 2="$MT/k4d.bin" >/dev/null
  if python3 tools/mma_ref.py k4tf32 "$MT/k4a.bin" "$MT/k4b.bin" "$MT/k4d.bin" >/dev/null; then
    echo "ok    mma8        m16n8k4 tf32: one element a register, K of four"
  else
    echo "FAIL  mma8        m16n8k4 tf32 is not the product the fragments describe"; fail=1
  fi
  "$WEFT" run kernels/mma8.ptx --kernel mmas8 --grid 32 --threads 32 --buffers 32,32,64 \
      --load 0="$MT/s8a.bin" --load 1="$MT/s8bt.bin" --cpu-only --dump 2="$MT/s8d.bin" >/dev/null
  if python3 tools/mma_ref.py s8 "$MT/s8a.bin" "$MT/s8bt.bin" "$MT/s8d.bin" >/dev/null; then
    echo "ok    mma8        m8n8k16 bytes: four bytes a register, s32 accumulators"
  else
    echo "FAIL  mma8        m8n8k16 bytes are not the product the fragments describe"; fail=1
  fi
  "$WEFT" run kernels/mma8.ptx --kernel mmak64s4 --grid 32 --threads 32 --buffers 128,64,128 \
      --load 0="$MT/s4a.bin" --load 1="$MT/s4bt.bin" --cpu-only --dump 2="$MT/s4d.bin" >/dev/null
  if python3 tools/mma_ref.py k64s4 "$MT/s4a.bin" "$MT/s4bt.bin" "$MT/s4d.bin" >/dev/null; then
    echo "ok    mma8        m16n8k64 s4: eight nibbles a register, sign kept"
  else
    echo "FAIL  mma8        m16n8k64 s4 is not the product the fragments describe"; fail=1
  fi
  "$WEFT" run kernels/mma8fp4.ptx --kernel mmak32e2m1 --grid 32 --threads 32 --buffers 128,64,128 \
      --buffer-bytes 0=2,1=2 --load 0="$MT/f4a.bin" --load 1="$MT/f4bt.bin" \
      --cpu-only --dump 2="$MT/f4d.bin" >/dev/null
  if python3 tools/mma_ref.py k32e2m1 "$MT/f4a.bin" "$MT/f4bt.bin" "$MT/f4d.bin" >/dev/null; then
    echo "ok    mma8        m16n8k32 e2m1: four fp4 codes a register, f32 accumulate"
  else
    echo "FAIL  mma8        m16n8k32 e2m1 is not the product the fragments describe"; fail=1
  fi
  "$WEFT" run kernels/mma8fp4.ptx --kernel mmak32e3m2 --grid 32 --threads 32 --buffers 128,64,128 \
      --load 0="$MT/f6a.bin" --load 1="$MT/f6bt.bin" --cpu-only --dump 2="$MT/f6d.bin" >/dev/null
  if python3 tools/mma_ref.py k32e3m2 "$MT/f6a.bin" "$MT/f6bt.bin" "$MT/f6d.bin" >/dev/null; then
    echo "ok    mma8        m16n8k32 e3m2: four fp6 codes a register, f32 accumulate"
  else
    echo "FAIL  mma8        m16n8k32 e3m2 is not the product the fragments describe"; fail=1
  fi
  "$WEFT" run kernels/mma8fp4k64.ptx --kernel mmak64e2m1 --grid 32 --threads 32 --buffers 256,128,128 \
      --load 0="$MT/f4ka.bin" --load 1="$MT/f4kbt.bin" --cpu-only --dump 2="$MT/f4kd.bin" >/dev/null
  if python3 tools/mma_ref.py k64e2m1 "$MT/f4ka.bin" "$MT/f4kbt.bin" "$MT/f4kd.bin" >/dev/null; then
    echo "ok    mma8        m16n8k64 e2m1: eight fp4 codes a register, k of 64"
  else
    echo "FAIL  mma8        m16n8k64 e2m1 is not the product the fragments describe"; fail=1
  fi
else
  echo "FAIL  mma8        no python3 to generate the operands"; fail=1
fi

upwant="3f800000 3f804000 477fe000 c0200000 7f800000 33800000 3f800000 bf800000 7f7f0000"
printf '\x00\x3C\x02\x3C\xFF\x7B\x00\xC1\x00\x7C\x01\x00\x80\x3F\x80\xBF\x7F\x7F\x00\x00' \
    > "$TMP/upin.bin"
"$WEFT" run kernels/cvtf16up.ptx --kernel cvtf16up --grid 1 --buffers 9,9 \
    --buffer-bytes 0=2 --load 0="$TMP/upin.bin" --cpu-only --dump 1="$TMP/upout.bin" >/dev/null
upgot=$(od -An -tx4 -v "$TMP/upout.bin" | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')
if [ "$upgot" = "$upwant " ]; then
  echo "ok    cvtf16up  a half and a bf16 widen to the float those bits name"
else
  echo "FAIL  cvtf16up  got[$upgot] wanted[$upwant ]"; fail=1
fi

runf kernels/tmadot.ptx tmadot --grid 32 --threads 32 --buffers 128,256,128,64 \
    --buffer-bytes 1=2,2=2
"$WEFT" run kernels/tmadot.ptx --kernel tmadot --grid 32 --threads 32 --buffers 128,256,128,64 \
    --buffer-bytes 1=2,2=2 --cpu-only --dump 0="$TMP/tmadot.bin" >/dev/null
if od -An -tf4 -v "$TMP/tmadot.bin" | awk '{ for (i = 1; i <= NF; i++) { n++; v = $i + 0;
      if ((n - 1) % 8 == 0) { if (v != int((n - 1) / 8) + 1) bad++ }
      else if (v != 0) bad++ } } END { exit(bad ? 1 : 0) }'; then
  echo "ok    tmadot  row r of the product is r+1, from tiles that came through the map"
else
  echo "FAIL  tmadot  the product is not the lower-triangular one"; fail=1
fi

"$WEFT" run kernels/mbar.ptx --kernel mbar --grid 32 --threads 32 --buffers 32 \
    --dump 0="$TMP/mbar.bin" >/dev/null
arrived=$(od -An -tu4 -v "$TMP/mbar.bin" | tr -s ' ' '\n' | grep -c '^42$' || true)
if [ "$arrived" = "32" ]; then
  echo "ok    mbar     every lane was released by the last arrival (32/32)"
else
  echo "FAIL  mbar     $arrived of 32 lanes read what the group wrote"; fail=1
fi

"$WEFT" run kernels/sharedofs.ptx --kernel sharedofs --grid 256 --buffers 768 \
    --dump 0="$TMP/sharedofs.bin" >/dev/null
words=$(od -An -tu4 -v "$TMP/sharedofs.bin" | tr -s ' ' '\n')
sevens=$(echo "$words" | grep -c '^7$' || true)
nines=$(echo "$words" | grep -c '^9$' || true)
fives=$(echo "$words" | grep -c '^5$' || true)
if [ "$sevens" = "256" ] && [ "$nines" = "256" ] && [ "$fives" = "256" ]; then
  echo "ok    sharedofs  [buf], [buf+4], [buf+8] read 7, 9, 5 (256 each)"
else
  echo "FAIL  sharedofs  $sevens sevens, $nines nines, $fives fives; wanted 256 of each"; fail=1
fi
runf kernels/hard.metal raymarch --grid 65536 --buffers 65536,65536     # SDF raymarch, 64 steps
runf kernels/hard.metal fft --grid 2048 --threads 256 --buffers 4096,4096  # 256-point radix-2, 9 barriers

"$WEFT" run kernels/dblvec.metal --kernel dblvec --cpu-only --grid 4096 --buffers 4096,4096 \
    --dump 1="$TMP/dbl.bin" >/dev/null
ones=$(od -An -tu4 -v "$TMP/dbl.bin" | tr -s ' ' '\n' | grep -c 1065353216)
if [ "$ones" = "4096" ]; then
  echo "ok    dblvec   double2/3/4 match the scalar arithmetic (4096/4096)"
else
  echo "FAIL  dblvec   $ones of 4096 components matched"; fail=1
fi

# Both sides of a twin comparison, with one extra rule: neither dump may be empty.
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
nvptx warpscan     1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx bsum         1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx hsum         1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx pred         1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx rotate       1 --grid 1048576 --threads 256 --buffers 1048576,1048576
nvptx block_reduce 1 --grid 1048576 --threads 256 --buffers 1048576,4096

triton() {
  local twin="$1" kernel="$2" idx="$3"; shift 3
  "$WEFT" run kernels/basic.metal         --kernel "triton_$twin" --cpu-only --dump "$idx"="$TMP/msl.bin" "$@" >/dev/null
  "$WEFT" run "kernels/triton/$twin.ptx"  --kernel "$kernel" --cpu-only --dump "$idx"="$TMP/tr.bin"  "$@" >/dev/null
  if same_dumps "triton:$twin" "$TMP/msl.bin" "$TMP/tr.bin"; then
    echo "ok    triton:$twin  matches the metal twin byte for byte"
  else
    echo "FAIL  triton:$twin  differs from the metal twin"; fail=1
  fi
  env WEFT_WARP=1 "$WEFT" run "kernels/triton/$twin.ptx" --kernel "$kernel" --cpu-only \
      --dump "$idx"="$TMP/trw.bin" "$@" >/dev/null
  if same_dumps "triton:$twin warp" "$TMP/msl.bin" "$TMP/trw.bin"; then
    echo "ok    triton:$twin:warp  the warp backend lands on the same bytes"
  else
    echo "FAIL  triton:$twin:warp  the warp backend differs"; fail=1
  fi
}

# The GPU column, for a front end that used to have none.
gpucol() {
  local file="$1" threads="$2" grid="$3"; shift 3
  local key out
  key="${file#kernels/}"
  if [ -n "$GPUCOL" ]; then
    echo "skip  gpu:$key  no ptx column on this host"
    return
  fi
  if [ -n "$GPUREF" ]; then
    echo "skip  gpu:$key  ptx column refused: $GPUREF"
    return
  fi
  local st=0
  local retried="" attempt=1 tries=""
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
  local why
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

gpucol kernels/cuda/saxpy.ptx        256 4096
gpucol kernels/cuda/reverse.ptx    256 4096
gpucol kernels/cuda/warpscan.ptx     256 4096
gpucol kernels/cuda/block_reduce.ptx 256 4096
gpucol kernels/cuda/bsum.ptx         256 4096
gpucol kernels/cuda/hsum.ptx         256 4096
gpucol kernels/cuda/pred.ptx         256 4096
gpucol kernels/cuda/rotate.ptx       256 4096
gpucol kernels/cuda/scale.ptx        256 4096
gpucol kernels/cuda/vecscale.ptx     256 4096 --buffers 16384,16384
gpucol kernels/cuda/reduce.ptx       256 4096
gpucol kernels/triton/add.ptx        128 4096
gpucol kernels/triton/cvt8.ptx       128 4096
gpucol kernels/triton/wgmma.ptx      128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/wgmma_bf16.ptx 128 4096 --buffers 32768,32768,32768
# tf32: 32-bit operands, so a core-matrix row holds four of them and K is 8.
gpucol kernels/triton/wgmma_tf32.ptx 128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/wgmma_fp8.ptx  128 4096 --buffers 2048,256,512 --buffer-bytes 0=1,1=1
gpucol kernels/triton/wgmma_s8.ptx   128 4096 --buffers 2048,256,512 --buffer-bytes 0=1,1=1
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
gpucol kernels/reverse.ptx           256 4096
gpucol kernels/saxpy.ptx             256 4096
gpucol kernels/logic.ptx           256 4096
gpucol kernels/round.ptx             256 4096
gpucol kernels/bred.ptx              256 4096
gpucol kernels/twotile.ptx            64 4096
gpucol kernels/async_copy.ptx         64 4096
gpucol kernels/ptxtex.ptx             64 4096 --textures 64x64 --samplers nearest/clamp
gpucol kernels/ptxsrf.ptx             64 4096 --textures 64x64
gpucol kernels/ptxsrflayer.ptx         64 4096 --textures '64x64*4'
gpucol kernels/ptxsrftmix.ptx         64 4096 --buffers 4096,4096 --textures 64x64,64x64 \
  --samplers nearest/clamp,nearest/clamp
python3 - "$TMP/f64in.bin" <<'PY'
import struct, sys
vals = [0.0, 1.0, -1.0, 0.5, 3.0, 1e-300, 2.0**-1022, 2.0**60, -7.25, 1e300,
        1234.5678, -0.0, 4096.5, 1e-16]
vals = (vals * 300)[:4096]
open(sys.argv[1], 'wb').write(struct.pack('<4096d', *vals))
PY
gpucol kernels/ptxf64.ptx             64 4096 --buffers 4096,4096,4096,4096 --buffer-bytes 0=8,1=4,2=8,3=8 \
    --load "0=$TMP/f64in.bin"
gpucol kernels/ptxgather.ptx          64 4096 --textures 64x64
gpucol kernels/ptxgather3d.ptx        64 4096 --textures 64x64x8
gpucol kernels/ptxgather1d.ptx        64 4096 --textures 64x1
gpucol kernels/ptxgathercube.ptx      64 4096 --textures 64x64c
gpucol kernels/ptxgathercubearray.ptx 64 4096 --textures '8x8c*2' --samplers nearest/clamp
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

if "$WEFT" run kernels/spirv/ballot.spv --kernel ballot --cpu-only --grid 32 --threads 32 --buffers 32 --dump 0="$TMP/ballot.bin" >/dev/null 2>&1; then
  if python3 - "$TMP/ballot.bin" <<'PY'
import struct, sys
v = struct.unpack("<32f", open(sys.argv[1], "rb").read(128))
bad = []
for i in range(32):
    want = (0 + 2 * 4 + 4 * 0 + 8 * 1 + 16 * (1 if i < 5 else 0)
            + 32 * ((i + 1) & 31) + 64 * ((i & ~7) + ((i + 1) & 7)))
    if v[i] != float(want):
        bad.append((i, v[i], want))
for i, got, want in bad[:4]:
    print("  lane %d wrote %g, wanted %d" % (i, got, want))
sys.exit(0 if not bad else 1)
PY
  then
    echo "ok    spvgpu:ballot  lsb, msb, bit extract, inverse, and a rotation of 32 and of 8"
  else
    echo "FAIL  spvgpu:ballot  a ballot question or a rotation direction is wrong"; fail=1
  fi
else
  echo "FAIL  spvgpu:ballot  the fixture did not run"; fail=1
fi
gpucol kernels/ptxarray.ptx           64 4096 --textures '64x64*4' --samplers nearest/clamp
gpucol kernels/ptxarray1d.ptx         64 4096 --textures '64x1*4' --samplers nearest/clamp
gpucol kernels/ptxcubearray.ptx       64 4096 --textures '8x8c*2' --samplers nearest/clamp
gpucol kernels/ptx3d.ptx             256 65536 --textures 64x64x8 --samplers nearest/clamp
gpucol kernels/ptxcube.ptx          256 65536 --textures 64x64c --samplers nearest/clamp
gpucol kernels/cuda/bits.ptx         256 4096
gpucol kernels/mulhi.ptx            256 4096 --buffers 16384,40960
gpucol kernels/dp4a.ptx             256 65536 --buffers 65536
gpucol kernels/mufu.ptx             256 65536 --buffers 65536
gpucol kernels/triton/reduce.ptx     128 4096 --buffers 4096,32768
gpucol kernels/red64.ptx             256 4096 --buffers 8,8,8,8
gpucol kernels/atom.ptx              256 4096
gpucol kernels/triton/dot32.ptx      128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/dotbf16.ptx    128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/dotfp8.ptx     128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/dote5m2.ptx    128 4096 --buffers 32768,32768,32768
gpucol kernels/triton/matmul.ptx     128 4096 --buffers 32768,32768,32768

spvgpu() {
  local kernel="$1" file="$2" label; shift 2
  label="$kernel"
  case "$kernel" in *=*) label="${kernel%%=*}"; kernel="${kernel#*=}" ;; esac
  local out
  if [ -n "$CPUMODE" ]; then
    echo "skip  spvgpu:$label  no gpu on this host"
    return
  fi
  local st=0 retried="" attempt=1 tries=""
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
spvgpu halfmath      halfmath  --grid 32 --threads 32 --buffers 32,32
spvgpu saxpy        saxpy     --grid 1048576 --buffers 1048576,1048576,1048576
spvgpu warpscan     warpscan  --grid 1048576 --threads 256 --buffers 1048576,1048576
spvgpu block_reduce blockreduce --grid 1048576 --threads 256 --buffers 1048576,4096
spvgpu votek        vote      --grid 1048576 --threads 256 --buffers 1048576,1048576
spvgpu wide_shfl    wide      --grid 1048576 --threads 256 --buffers 1048576,1048576
spvgpu quads       quads      --grid 32 --threads 32 --buffers 32,32
spvgpu dblshfl     dblshfl    --grid 32 --threads 32 --buffers 32
spvgpu submul      submul     --grid 32 --threads 32 --buffers 32
spvgpu subbool     subbool    --grid 32 --threads 32 --buffers 32
# The four questions about a ballot, and a rotation with and without a cluster.
spvgpu ballot      ballot     --grid 32 --threads 32 --buffers 32
spvgpu serial_sum   llvm      --grid 1 --threads 1 --buffers 1000,1 --arg 0=1000
spvgpu globk        global    --grid 1048576 --buffers 1048576,1048576
spvgpu texmix       texture   --grid 65536 --buffers 65536 --textures 64x64 \
  --samplers nearest/clamp
spvgpu texrw        texturerg --grid 4096 --threads 256 --buffers 4096,4096 \
  --textures 64x64
spvgpu texarray     arraytex  --grid 65536 --buffers 65536 --textures '64x64*4' \
  --samplers nearest/clamp
spvgpu tex3d        threed    --grid 65536 --buffers 65536 --textures 64x64x8 \
  --samplers nearest/clamp
spvgpu gl1d        gl1d    --grid 65536 --buffers 65536 --textures 64x1 --samplers nearest/clamp
spvgpu gl2d        gl2d    --grid 65536 --buffers 65536 --textures 64x64 --samplers nearest/clamp
spvgpu glcube      glcube  --grid 65536 --buffers 65536 --textures 64x64c --samplers nearest/clamp
spvgpu glcubegather glcubegather --grid 65536 --buffers 262144 --textures 64x64c \
  --samplers nearest/clamp
spvgpu glgather    glgather --grid 65536 --buffers 65536 --textures 64x64 --samplers nearest/clamp
spvgpu glgatherarray glgatherarray --grid 65536 --buffers 65536 --textures '64x64*4' --samplers nearest/clamp
spvgpu glgroup     glgroup  --grid 65536 --buffers 65536
spvgpu matk         matk      --grid 65536 --buffers 65536,65536
spvgpu mathx        mathx     --grid 65536 --buffers 65536,65536
spvgpu mathy        mathy     --grid 65536 --buffers 65536,65536
spvgpu mathz        mathz     --grid 65536 --buffers 65536,65536
spvgpu atomf        atomf     --grid 1024 --threads 1024 --buffers 1024,1024,2
spvgpu atomic       atomic    --grid 8 --threads 8 --buffers 8,8,8
spvgpu atom64k      atom64    --grid 4096 --threads 32 --buffers 1,1,1,1,1,1
spvgpu atom64rw     atom64rw  --grid 64 --threads 32 --buffers 1,1
spvgpu atom64loop  atom64loop --grid 64 --threads 32 --buffers 1,64
spvgpu atom64bar   atom64bar  --grid 32 --threads 32 --buffers 1,32
spvgpu atom64vote  atom64vote --grid 32 --threads 32 --buffers 1,32,32,32,32,32,32
spvgpu atom64sub   atom64sub  --grid 32 --threads 32 --buffers 1,32,32,32,32,32,32,32
spvgpu fp64       fp64       --grid 32 --threads 32 --buffers 32,320,64,32
spvgpu dvec       dvec       --grid 32 --threads 32 --buffers 32,128
spvgpu f64math    f64math    --grid 64 --threads 32 --buffers 64,2048,192,128
spvgpu f64edge   f64edge   --grid 32 --threads 32 --buffers 32,1408,320,320 --exact
spvgpu f32math    f32math    --grid 64 --threads 32 --buffers 64,2496,256,256 --exact
spvgpu f32edge   f32edge   --grid 32 --threads 32 --buffers 32,1696,320,320 --exact
if f32math_out="$(python3 tools/f32_math.py 2>&1)"; then
  echo "ok    f32math:ulp  $(printf '%s\n' "$f32math_out" | grep 'functions swept')"
else
  echo "FAIL  f32math:ulp"
  printf '%s\n' "$f32math_out" | tail -14 | sed 's/^/      /'
  fail=1
fi
if f64math_out="$(python3 tools/f64_math.py 2>&1)"; then
  echo "ok    f64math:ulp  $(printf '%s\n' "$f64math_out" | grep 'functions swept')"
else
  echo "FAIL  f64math:ulp"
  printf '%s\n' "$f64math_out" | tail -12 | sed 's/^/      /'
  fail=1
fi
spvgpu helper     helper     --grid 32 --threads 32 --buffers 32,64
spvgpu matns      matns      --grid 32 --threads 32 --buffers 32,128
spvgpu subfold    subfold    --grid 32 --threads 32 --buffers 256
spvgpu subfold:ragged=subfold subfold --grid 24 --threads 32 --buffers 256
if [ -n "$CPUMODE" ]; then
  echo "skip  fp64:exact  no metal on this host"
elif fp64_out="$(python3 tools/f64_check.py 2>&1)"; then
  echo "ok    fp64:exact  $(printf '%s\n' "$fp64_out" | tail -1)"
else
  echo "FAIL  fp64:exact"
  printf '%s\n' "$fp64_out" | tail -12 | sed 's/^/      /'
  fail=1
fi

triton add    add_kernel    2 --grid 1024 --buffers 1024,1024,1024 --arg 0=1000
triton matmul matmul_kernel 2 --grid 256  --buffers 2048,1024,1024
triton dot32  dot32_kernel  2 --grid 256  --buffers 1024,512,1024
triton dotbf16 dotbf16_kernel 2 --grid 256 --buffers 2048,1024,1024
triton reduce reduce_kernel 1 --grid 1024 --buffers 1024,8192
if python3 kernels/triton/fp8_ref.py gen 512 256 "$TMP/f8a.bin" "$TMP/f8b.bin" 2>"$TMP/f8.log" \
   && "$WEFT" run kernels/triton/dotfp8.ptx --kernel dotfp8_kernel --cpu-only \
        --grid 32 --threads 32 --buffers 512,256,128 --buffer-bytes 0=1,1=1 \
        --load 0="$TMP/f8a.bin" --load 1="$TMP/f8b.bin" --dump 2="$TMP/f8d.bin" >>"$TMP/f8.log" 2>&1 \
   && python3 kernels/triton/fp8_ref.py matmul "$TMP/f8a.bin" "$TMP/f8b.bin" "$TMP/f8d.bin" >>"$TMP/f8.log" 2>&1; then
  echo "ok    triton:dotfp8  m16n8k32 e4m3 matches the e4m3 reference exactly"
else
  echo "FAIL  triton:dotfp8"; sed 's/^/      /' "$TMP/f8.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wa.bin" "$TMP/wb.bin" 2>"$TMP/wg.log" \
   && "$WEFT" run kernels/triton/wgmma.ptx --kernel wgmma_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wa.bin" --load 1="$TMP/wb.bin" --dump 2="$TMP/wd.bin" >>"$TMP/wg.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wa.bin" "$TMP/wb.bin" "$TMP/wd.bin" >>"$TMP/wg.log" 2>&1; then
  echo "ok    triton:wgmma  m64n8k16 f16 matches the reference exactly"
else
  echo "FAIL  triton:wgmma"; sed 's/^/      /' "$TMP/wg.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wab.bin" "$TMP/wbb.bin" bf16 2>"$TMP/wgb.log" \
   && "$WEFT" run kernels/triton/wgmma_bf16.ptx --kernel wgmma_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wab.bin" --load 1="$TMP/wbb.bin" --dump 2="$TMP/wdb.bin" >>"$TMP/wgb.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wab.bin" "$TMP/wbb.bin" "$TMP/wdb.bin" bf16 >>"$TMP/wgb.log" 2>&1; then
  echo "ok    triton:wgmma_bf16  m64n8k16 bf16 matches the reference exactly"
else
  echo "FAIL  triton:wgmma_bf16"; sed 's/^/      /' "$TMP/wgb.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wa32.bin" "$TMP/wb32.bin" tf32 2>"$TMP/wg32.log" \
   && "$WEFT" run kernels/triton/wgmma_tf32.ptx --kernel wgmma_tf32_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 512,64,512 --buffer-bytes 0=4,1=4 \
        --load 0="$TMP/wa32.bin" --load 1="$TMP/wb32.bin" --dump 2="$TMP/wd32.bin" >>"$TMP/wg32.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wa32.bin" "$TMP/wb32.bin" "$TMP/wd32.bin" tf32 >>"$TMP/wg32.log" 2>&1; then
  echo "ok    triton:wgmma_tf32  m64n8k8 tf32 matches the reference exactly"
else
  echo "FAIL  triton:wgmma_tf32"; sed 's/^/      /' "$TMP/wg32.log"; fail=1
fi

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

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wn16a.bin" "$TMP/wn16b.bin" f16n16 2>"$TMP/wn16.log" \
   && "$WEFT" run kernels/triton/wgmma_n16.ptx --kernel wgmma_n16_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,256,1024 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wn16a.bin" --load 1="$TMP/wn16b.bin" --dump 2="$TMP/wn16d.bin" >>"$TMP/wn16.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wn16a.bin" "$TMP/wn16b.bin" "$TMP/wn16d.bin" f16n16 >>"$TMP/wn16.log" 2>&1; then
  echo "ok    triton:wgmma_n16  m64n16k16 f16 matches the reference exactly"
else
  echo "FAIL  triton:wgmma_n16"; sed 's/^/      /' "$TMP/wn16.log"; fail=1
fi

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
if python3 kernels/triton/wgmma_ref.py gen "$TMP/waa.bin" "$TMP/wab2.bin" f16acc 2>"$TMP/waa.log" \
   && "$WEFT" run kernels/triton/wgmma_acc.ptx --kernel wgmma_acc_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/waa.bin" --load 1="$TMP/wab2.bin" --dump 2="$TMP/wad.bin" >>"$TMP/waa.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/waa.bin" "$TMP/wab2.bin" "$TMP/wad.bin" f16acc >>"$TMP/waa.log" 2>&1; then
  echo "ok    triton:wgmma_acc  two issues with scale-d double the product, exactly"
else
  echo "FAIL  triton:wgmma_acc"; sed 's/^/      /' "$TMP/waa.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wza.bin" "$TMP/wzb.bin" f16 2>"$TMP/wz.log" \
   && "$WEFT" run kernels/triton/wgmma_swz128.ptx --kernel wgmma_swz128_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wza.bin" --load 1="$TMP/wzb.bin" --dump 2="$TMP/wzd.bin" >>"$TMP/wz.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wza.bin" "$TMP/wzb.bin" "$TMP/wzd.bin" f16 >>"$TMP/wz.log" 2>&1; then
  echo "ok    triton:wgmma_swz128  a 128-byte-swizzled tile matches the reference"
else
  echo "FAIL  triton:wgmma_swz128"; sed 's/^/      /' "$TMP/wz.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/w3a.bin" "$TMP/w3b.bin" f16 2>"$TMP/w3.log" \
   && "$WEFT" run kernels/triton/wgmma_swz32.ptx --kernel wgmma_swz32_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w3a.bin" --load 1="$TMP/w3b.bin" --dump 2="$TMP/w3d.bin" >>"$TMP/w3.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w3a.bin" "$TMP/w3b.bin" "$TMP/w3d.bin" f16 >>"$TMP/w3.log" 2>&1; then
  echo "ok    triton:wgmma_swz32  a 32-byte-swizzled tile matches the reference"
else
  echo "FAIL  triton:wgmma_swz32"; sed 's/^/      /' "$TMP/w3.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/w4a.bin" "$TMP/w4b.bin" f16 2>"$TMP/w4.log" \
   && "$WEFT" run kernels/triton/wgmma_swz64.ptx --kernel wgmma_swz64_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w4a.bin" --load 1="$TMP/w4b.bin" --dump 2="$TMP/w4d.bin" >>"$TMP/w4.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w4a.bin" "$TMP/w4b.bin" "$TMP/w4d.bin" f16 >>"$TMP/w4.log" 2>&1; then
  echo "ok    triton:wgmma_swz64  a 64-byte-swizzled tile matches the reference"
else
  echo "FAIL  triton:wgmma_swz64"; sed 's/^/      /' "$TMP/w4.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/w5a.bin" "$TMP/w5b.bin" f16 2>"$TMP/w5.log" \
   && "$WEFT" run kernels/triton/wgmma_swzbase.ptx --kernel wgmma_swzbase_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w5a.bin" --load 1="$TMP/w5b.bin" --dump 2="$TMP/w5d.bin" >>"$TMP/w5.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w5a.bin" "$TMP/w5b.bin" "$TMP/w5d.bin" f16 >>"$TMP/w5.log" 2>&1; then
  echo "ok    triton:wgmma_swzbase  a tile three rows into the pattern matches the reference"
else
  echo "FAIL  triton:wgmma_swzbase"; sed 's/^/      /' "$TMP/w5.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/w6a.bin" "$TMP/w6b.bin" f16 2>"$TMP/w6.log" \
   && "$WEFT" run kernels/triton/wgmma_swztrans.ptx --kernel wgmma_swztrans_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/w6a.bin" --load 1="$TMP/w6b.bin" --dump 2="$TMP/w6d.bin" >>"$TMP/w6.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w6a.bin" "$TMP/w6b.bin" "$TMP/w6d.bin" f16 >>"$TMP/w6.log" 2>&1; then
  echo "ok    triton:wgmma_swztrans  a swizzled MN-major A matches the reference"
else
  echo "FAIL  triton:wgmma_swztrans"; sed 's/^/      /' "$TMP/w6.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/w7a.bin" "$TMP/w7b.bin" tf32 2>"$TMP/w7.log" \
   && "$WEFT" run kernels/triton/wgmma_swztf32.ptx --kernel wgmma_swztf32_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 512,64,512 --buffer-bytes 0=4,1=4 \
        --load 0="$TMP/w7a.bin" --load 1="$TMP/w7b.bin" --dump 2="$TMP/w7d.bin" >>"$TMP/w7.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/w7a.bin" "$TMP/w7b.bin" "$TMP/w7d.bin" tf32 >>"$TMP/w7.log" 2>&1; then
  echo "ok    triton:wgmma_swztf32  a swizzled tf32 tile matches the reference"
else
  echo "FAIL  triton:wgmma_swztf32"; sed 's/^/      /' "$TMP/w7.log"; fail=1
fi

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

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wba.bin" "$TMP/wbb.bin" f16 2>"$TMP/wb.log" \
   && "$WEFT" run kernels/triton/wgmma_swztransb.ptx --kernel wgmma_swztransb_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wba.bin" --load 1="$TMP/wbb.bin" --dump 2="$TMP/wbd.bin" >>"$TMP/wb.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wba.bin" "$TMP/wbb.bin" "$TMP/wbd.bin" f16 >>"$TMP/wb.log" 2>&1; then
  echo "ok    triton:wgmma_swztransb  a swizzled MN-major B matches the reference"
else
  echo "FAIL  triton:wgmma_swztransb"; sed 's/^/      /' "$TMP/wb.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wca.bin" "$TMP/wcb.bin" f16chain 2>"$TMP/wc.log" \
   && "$WEFT" run kernels/triton/wgmma_chain.ptx --kernel wgmma_chain_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wca.bin" --load 1="$TMP/wcb.bin" --dump 2="$TMP/wcd.bin" >>"$TMP/wc.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wca.bin" "$TMP/wcb.bin" "$TMP/wcd.bin" f16chain >>"$TMP/wc.log" 2>&1; then
  echo "ok    triton:wgmma_chain  three issues tripled the product, exactly"
else
  echo "FAIL  triton:wgmma_chain"; sed 's/^/      /' "$TMP/wc.log"; fail=1
fi

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

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wia.bin" "$TMP/wib.bin" f16neg 2>"$TMP/wi.log" \
   && "$WEFT" run kernels/triton/wgmma_shortimm.ptx --kernel wgmma_shortimm_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wia.bin" --load 1="$TMP/wib.bin" --dump 2="$TMP/wid.bin" >>"$TMP/wi.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wia.bin" "$TMP/wib.bin" "$TMP/wid.bin" f16neg >>"$TMP/wi.log" 2>&1; then
  echo "ok    triton:wgmma_shortimm  a negative scale negates A, exactly"
else
  echo "FAIL  triton:wgmma_shortimm"; sed 's/^/      /' "$TMP/wi.log"; fail=1
fi

if python3 kernels/triton/wgmma_ref.py gen "$TMP/wwa.bin" "$TMP/wwb.bin" f16 2>"$TMP/ww.log" \
   && "$WEFT" run kernels/triton/wgmma_warp.ptx --kernel wgmma_warp_kernel --cpu-only \
        --grid 128 --threads 128 --buffers 1024,128,512 --buffer-bytes 0=2,1=2 \
        --load 0="$TMP/wwa.bin" --load 1="$TMP/wwb.bin" --dump 2="$TMP/wwd.bin" >>"$TMP/ww.log" 2>&1 \
   && python3 kernels/triton/wgmma_ref.py check "$TMP/wwa.bin" "$TMP/wwb.bin" "$TMP/wwd.bin" f16 >>"$TMP/ww.log" 2>&1; then
  echo "ok    triton:wgmma_warp  a wgmma on the warp backend matches the reference"
else
  echo "FAIL  triton:wgmma_warp"; sed 's/^/      /' "$TMP/ww.log"; fail=1
fi

if python3 kernels/triton/fp8_ref.py gen 512 256 "$TMP/e5a.bin" "$TMP/e5b.bin" e5m2 2>"$TMP/e5.log" \
   && "$WEFT" run kernels/triton/dote5m2.ptx --kernel dote5m2_kernel --cpu-only \
        --grid 32 --threads 32 --buffers 512,256,128 --buffer-bytes 0=1,1=1 \
        --load 0="$TMP/e5a.bin" --load 1="$TMP/e5b.bin" --dump 2="$TMP/e5d.bin" >>"$TMP/e5.log" 2>&1 \
   && python3 kernels/triton/fp8_ref.py matmul "$TMP/e5a.bin" "$TMP/e5b.bin" "$TMP/e5d.bin" e5m2 >>"$TMP/e5.log" 2>&1; then
  echo "ok    triton:dote5m2  m16n8k32 e5m2 matches the e5m2 reference exactly"
else
  echo "FAIL  triton:dote5m2"; sed 's/^/      /' "$TMP/e5.log"; fail=1
fi

if python3 kernels/triton/fp8_ref.py gen-cvt "$TMP/f8in.bin" 2>"$TMP/f8c.log" \
   && "$WEFT" run kernels/triton/cvt8.ptx --kernel cvt8_kernel --cpu-only \
        --grid 32 --threads 32 --buffers 128,32 \
        --load 0="$TMP/f8in.bin" --dump 1="$TMP/f8out.bin" >>"$TMP/f8c.log" 2>&1 \
   && python3 kernels/triton/fp8_ref.py cvt "$TMP/f8in.bin" "$TMP/f8out.bin" >>"$TMP/f8c.log" 2>&1; then
  echo "ok    triton:cvt8  packing to e4m3 matches the reference on every branch"
else
  echo "FAIL  triton:cvt8"; sed 's/^/      /' "$TMP/f8c.log"; fail=1
fi

meta=$("$WEFT" run kernels/triton/add.ptx --kernel add_kernel --grid 1024 \
    --buffers 1024,1024,1024 --arg 0=1000 --cpu-only)
if echo "$meta" | grep -q 'threadgroup 128' && echo "$meta" | grep -q '1 declared parameter is not bound'; then
  echo "ok    triton:abi  threads default to .maxntid (128), unread scratch not bound"
else
  echo "FAIL  triton:abi"; echo "$meta" | sed 's/^/      /'; fail=1
fi

# A barrier inside a loop, from real codegen, this is the resumable path.
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

"$WEFT" run kernels/basic.metal --kernel logic --cpu-only --grid 1048576 \
    --buffers 1048576,1048576 --dump 1="$TMP/logic_msl.bin" >/dev/null
"$WEFT" run kernels/logic.ptx --kernel logic --cpu-only --grid 1048576 \
    --buffers 1048576,1048576 --dump 1="$TMP/logic_ptx.bin" >/dev/null
if cmp -s "$TMP/logic_msl.bin" "$TMP/logic_ptx.bin"; then
  echo "ok    ptx:logic     lop3 tables, every class and the two zeros agree"
else
  echo "FAIL  ptx:logic     differs from the metal kernel"; fail=1
fi

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

"$WEFT" run kernels/fxorabs.ptx --kernel fxorabs --grid 1 --threads 1 --buffers 136 \
    --cpu-only --dump 0="$TMP/fxorabs.bin" >/dev/null
if python3 - "$TMP/fxorabs.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()

def xorabs(op, ab, bb):
    ma, mb = ab & 0x7FFFFFFF, bb & 0x7FFFFFFF
    mag = (ma if ma < mb else mb) if op == 'min' else (ma if ma > mb else mb)
    # the magnitude's nan is caught before the sign goes on, and answers the float's
    # canonical nan, the all-ones word
    if (mag & 0x7F800000) == 0x7F800000 and (mag & 0x007FFFFF) != 0:
        return 0x7FFFFFFF
    return mag | ((ab ^ bb) & 0x80000000)

NEG_ONE, ONE, TWO, NEG_TWO, NEG_THREE, THREE = 0xBF800000, 0x3F800000, 0x40000000, 0xC0000000, 0xC0400000, 0x40400000
FOUR, NEG_FOUR, NAN, PZERO, NZERO, INF = 0x40800000, 0xC0800000, 0x7FC00000, 0, 0x80000000, 0x7F800000
want32 = [
    xorabs('min', NEG_ONE, TWO), xorabs('min', ONE, NEG_TWO),
    xorabs('min', NEG_THREE, NEG_TWO), xorabs('max', NEG_THREE, TWO),
    xorabs('max', ONE, NEG_FOUR), xorabs('min', NAN, TWO), xorabs('min', TWO, NAN),
    xorabs('min', NZERO, PZERO), xorabs('max', PZERO, NZERO), xorabs('min', INF, ONE),
]
got32 = list(struct.unpack('<10I', data[:40]))
# A double's plain min/max is minNum/maxNum with -0.0 below +0.0, and two NaNs answer the
# canonical NaN of that format, which is 0x7ff8..., the positive default, not all ones.
# The last two are a zero immediate through a double register: an `int` literal under a
# 64-bit destination is the width the bitcast used to read past.
# The last five are the double's sign operations: copysign takes the sign and keeps the
# payload, abs clears the sign bit and keeps the payload.
want64 = [0x3FF0000000000000, 0x3FF0000000000000, 0x7FF8000000000000, 0,
          0x8000000000000000, 0, 0,
          0xBFF0000000000000, 0xFFF8000000000001, 0, 0x8000000000000000,
          0x7FF8000000000001]
got64 = list(struct.unpack('<12Q', data[40:136]))
bad = 0
for i, (g, x) in enumerate(zip(got32, want32)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
for i, (g, x) in enumerate(zip(got64, want64)):
    if g != x:
        print('  double %d got 0x%016X want 0x%016X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    fxorabs     .xorsign.abs on a float, and the plain min/max of a double"
else
  echo "FAIL  fxorabs     a sign, a magnitude or a NaN answered the wrong way"; fail=1
fi
runf kernels/fxorabs.ptx fxorabs --grid 1 --threads 1 --buffers 136 --exact
cat > "$TMP/xorabsbad.ptx" <<'PTX'
.version 7.0
.target sm_86
.address_size 64
.visible .entry bad(.param .u64 p0)
{
    .reg .b64 %rd<3>;
    .reg .f64 %fd<3>;
    ld.param.u64 %rd1, [p0];
    mov.b64 %fd1, 0x7FF8000000000000;
    mov.b64 %fd2, 0x3FF0000000000000;
    min.NaN.f64 %fd3, %fd1, %fd2;
    mov.b64 %rd2, %fd3;
    st.u64 [%rd1+0], %rd2;
    ret;
}
PTX
"$WEFT" run "$TMP/xorabsbad.ptx" --kernel bad --grid 1 --threads 1 --buffers 8 --cpu-only \
    > "$TMP/xorabsbad.out" 2>&1 || true
if grep -q "takes no modifier" "$TMP/xorabsbad.out"
then
  echo "ok    fxorabs64   a modifier the double form does not have is refused by name"
else
  echo "FAIL  fxorabs64   min.NaN.f64 was not refused by name"; fail=1
fi
"$WEFT" run kernels/fedge.ptx --kernel fedge --grid 1 --threads 1 --buffers 84 \
    --cpu-only --dump 0="$TMP/fedge.bin" >/dev/null
if python3 - "$TMP/fedge.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()

def f(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]

def b32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def is_nan(b):
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0

def minmax(op, ab, bb):
    if is_nan(ab) and is_nan(bb):
        return 0x7FFFFFFF
    if is_nan(ab):
        return bb
    if is_nan(bb):
        return ab
    if (ab & 0x7FFFFFFF) == 0 and (bb & 0x7FFFFFFF) == 0:
        # -0.0 < +0.0 to the isa; a C comparison calls them equal
        if op == 'min':
            return ab if (ab >> 31) else bb
        return ab if (bb >> 31) else bb
    a, b = f(ab), f(bb)
    if op == 'min':
        return ab if a < b else bb
    return ab if a > b else bb

def copysign(ab, bb):
    return (ab & 0x7FFFFFFF) | (bb & 0x80000000)

ONE, NEG_ONE, NAN, NANP, PZERO, NZERO = 0x3F800000, 0xBF800000, 0x7FC00000, 0x7FC00001, 0, 0x80000000
want = [
    copysign(ONE, NZERO), copysign(NANP, NZERO), copysign(NEG_ONE, PZERO),
    minmax('min', NAN, ONE), minmax('min', ONE, NAN),
    minmax('max', NAN, ONE), minmax('max', ONE, NAN),
    minmax('min', NZERO, PZERO), minmax('max', NZERO, PZERO),
    0x7FFFFFFF, minmax('min', ONE, 0x40000000), minmax('max', 0x40000000, ONE),
    0, 0, 0xFFFFFFFF, 0, 0x00FFFFFF, 0, 0, 0x80000000,
    # two NaNs answer the canonical NaN, the all-ones word for a float
    minmax('min', NAN, 0x7FF80001),
]
got = list(struct.unpack('<21I', data[:84]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    fedge       copysign, minNum/maxNum with the zeros, and the shift clamp"
else
  echo "FAIL  fedge       a sign, a NaN or a clamped count answered the wrong way"; fail=1
fi
runf kernels/fedge.ptx fedge --grid 1 --threads 1 --buffers 84 --exact
"$WEFT" run kernels/cvtint.ptx --kernel cvtint --grid 1 --threads 1 --buffers 56 \
    --cpu-only --dump 0="$TMP/cvtint.bin" >/dev/null
if python3 - "$TMP/cvtint.bin" <<'PY'
import math, struct, sys
data = open(sys.argv[1], 'rb').read()

def val(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]

def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def f2i(x, signed):
    # the isa's rule, written as arithmetic rather than as a cast
    if x != x:
        return 0
    if signed:
        if x >= 2147483648.0:
            return 0x7FFFFFFF
        if x <= -2147483649.0:
            return 0x80000000
        return int(x) & 0xFFFFFFFF
    if x >= 4294967296.0:
        return 0xFFFFFFFF
    if x <= -1.0:
        return 0
    return int(x) & 0xFFFFFFFF

NAN, INF, NINF = 0x7FC00000, 0x7F800000, 0xFF800000
HALF5, NEG15, HALF35, NEG25, X21 = 0x40200000, 0xBFC00000, 0x40600000, 0xC0200000, 0x40066666
ONE, NEG_ONE = 0x3F800000, 0xBF800000
want = [
    f2i(val(NAN), True), f2i(val(INF), True), f2i(val(NINF), True),
    f2i(float(math.trunc(val(HALF5))), True),
    f2i(float(math.trunc(val(NEG15))), False), f2i(float(math.trunc(val(NEG15))), True),
    f2i(val(NAN), False),
    # .rni is ties-to-even, so 2.5 answers 2 and 3.5 answers 4
    f2i(float(round(val(HALF5))), True), f2i(float(round(val(HALF35))), True),
    f2i(float(math.floor(val(NEG25))), True), f2i(float(math.ceil(val(X21))), True),
    # `.sat` on a float destination is the [0, 1] clamp, with a NaN flushed to +0
    bits(1.0), bits(0.0), bits(0.0),
]
got = list(struct.unpack('<14I', data[:56]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    cvtint      float to int clamps at the ends of the range and zero for a NaN"
else
  echo "FAIL  cvtint      an out-of-range value or a NaN was not answered the isa's way"; fail=1
fi
runf kernels/cvtint.ptx cvtint --grid 1 --threads 1 --buffers 56 --exact
python3 - "$TMP/spvcvtint_in.bin" <<'PY'
import struct, sys
vals = [float('nan'), float('inf'), float('-inf'), -1.5, -1.5, float('inf'), 2.5, float('nan')]
open(sys.argv[1], 'wb').write(struct.pack('<8f', *vals))
PY
"$WEFT" run kernels/spirv/cvtint.spv --kernel cvtint --grid 1 --threads 1 --buffers 8,8 \
    --buffer-bytes 0=4,1=4 --load 0="$TMP/spvcvtint_in.bin" --cpu-only \
    --dump 1="$TMP/spvcvtint.bin" >/dev/null
if python3 - "$TMP/spvcvtint.bin" <<'PY'
import struct, sys

def f2i(x):
    if x != x:
        return 0
    if x >= 2147483648.0:
        return 0x7FFFFFFF
    if x <= -2147483649.0:
        return 0x80000000
    return int(x) & 0xFFFFFFFF

def f2u(x):
    if x != x:
        return 0
    if x >= 4294967296.0:
        return 0xFFFFFFFF
    if x <= -1.0:
        return 0
    return int(x) & 0xFFFFFFFF

nan, inf = float('nan'), float('inf')
want = [f2i(nan), f2i(inf), f2i(-inf), f2i(-1.5), f2u(-1.5), f2u(inf), f2i(2.5), f2u(nan)]
got = list(struct.unpack('<8I', open(sys.argv[1], 'rb').read(32)))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    spvcvtint   float to int clamps in the spir-v front end too"
else
  echo "FAIL  spvcvtint   an out-of-range conversion was not answered the isa's way"; fail=1
fi
runf kernels/spirv/cvtint.spv cvtint --grid 1 --threads 1 --buffers 8,8 --buffer-bytes 0=4,1=4 --load 0="$TMP/spvcvtint_in.bin" --exact
"$WEFT" run kernels/f64cvt.ptx --kernel f64cvt --grid 1 --threads 1 --buffers 100 \
    --cpu-only --dump 0="$TMP/f64cvt.bin" >/dev/null
if python3 - "$TMP/f64cvt.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()

def s64(x):
    if x != x:
        return 0
    if x >= 9223372036854775808.0:
        return 0x7FFFFFFFFFFFFFFF
    if x <= -9223372036854775808.0:
        return 0x8000000000000000
    return int(x) & 0xFFFFFFFFFFFFFFFF

def u64(x):
    if x != x:
        return 0
    if x >= 18446744073709551616.0:
        return 0xFFFFFFFFFFFFFFFF
    if x <= -1.0:
        return 0
    return int(x)

def s32(x):
    if x != x:
        return 0
    if x >= 2147483648.0:
        return 0x7FFFFFFF
    if x <= -2147483649.0:
        return 0x80000000
    return int(x) & 0xFFFFFFFF

def u32(x):
    if x != x:
        return 0
    if x >= 4294967296.0:
        return 0xFFFFFFFF
    if x <= -1.0:
        return 0
    return int(x)

inf = float('inf')
want = [s64(1e20), s64(18446744073709551616.0), s64(9223372036854775808.0),
        s64(-9223372036854775808.0), s64(-1e20), s64(4611686018427387904.0),
        u64(1e20), u64(-1.5), 0, s64(inf), u64(inf),
        s32(1e20), s32(1099511627776.0), u32(-1.5)]
got = list(struct.unpack('<11Q', data[:88])) + list(struct.unpack('<3I', data[88:100]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%016X want 0x%016X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    f64cvt      a double read as an integer clamps on the software path"
else
  echo "FAIL  f64cvt      an out-of-range double was not answered the isa's way"; fail=1
fi
runf kernels/f64cvt.ptx f64cvt --grid 1 --threads 1 --buffers 100 --exact
cat > "$TMP/undeclared.ptx" <<'PTX'
.version 7.0
.target sm_86
.address_size 64
.visible .entry undeclared(.param .u64 p0)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [p0];
    mov.b64 %fd1, 0x3FF0000000000000;
    mov.b64 %rd2, %fd1;
    st.u64 [%rd1+0], %rd2;
    ret;
}
PTX
"$WEFT" run "$TMP/undeclared.ptx" --kernel undeclared --grid 1 --threads 1 --buffers 8 --cpu-only \
    > "$TMP/undeclared.out" 2>&1 || true
if grep -q "register %fd1 is used but never declared" "$TMP/undeclared.out"
then
  echo "ok    undeclared   a register that was never declared is refused by name"
else
  echo "FAIL  undeclared   %fd1 was not refused by name"; fail=1
fi
"$WEFT" run kernels/warpops.ptx --kernel warpv --grid 32 --threads 32 --buffers 256 \
    --cpu-only --dump 0="$TMP/warpops.bin" >/dev/null
if python3 - "$TMP/warpops.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
got = list(struct.unpack('<256I', data[:1024]))
rows = []
for i in range(32):
    w = got[i * 8:i * 8 + 8]
    rows.append(w)
bad = 0
for i in range(32):
    want = [0xFFFFFFFF, 0xAAAAAAAA, 30,
            i if i == 0 else i - 1,
            i if i == 31 else i + 1,
            i ^ 2, 0, 1]
    for k, (g, x) in enumerate(zip(rows[i], want)):
        if g != x:
            print('  lane %d row %d got 0x%08X want 0x%08X' % (i, k, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    warpops      every lane of activemask, ballot, vote and shfl answers the rule"
else
  echo "FAIL  warpops      a lane read the wrong neighbour or the wrong bit"; fail=1
fi
runf kernels/warpops.ptx warpv --grid 32 --threads 32 --buffers 256 --exact
"$WEFT" run kernels/warpseg.ptx --kernel warpseg --grid 32 --threads 32 --buffers 544 \
    --cpu-only --dump 0="$TMP/warpseg.bin" >/dev/null
if python3 - "$TMP/warpseg.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
got = struct.unpack('<544I', data[:2176])
bad = 0
for x in range(24):
    dl = (x & 7) + 1
    j = x ^ 8
    want = [0x00FFFFFF, 0x00AAAAAA, 1, 0, 30, x ^ 2,
            x - 1 if x >= 9 else x,
            x + 1 if x + 1 <= 15 else x,
            x + dl if x + dl <= 31 else x,
            x - dl if x >= dl else x,
            x ^ 5, 8,
            j if j <= x else x,
            x - 2 if x >= 2 else x,
            x + 2 if x + 2 <= 31 else x,
            0,
            x & 15]
    for k, w in enumerate(want):
        g = got[x * 17 + k]
        if g != w:
            print('  lane %d row %d got 0x%08X want 0x%08X' % (x, k, g, w)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    warpseg      a partial warp, a segment clamp and a register distance all land"
else
  echo "FAIL  warpseg      a lane read outside its segment or an inactive lane was counted"; fail=1
fi
runf kernels/warpseg.ptx warpseg --grid 32 --threads 32 --buffers 544 --exact
"$WEFT" run kernels/warpreg.ptx --kernel warpreg --grid 32 --threads 32 --buffers 128 \
    --cpu-only --dump 0="$TMP/warpreg.bin" >/dev/null
if python3 - "$TMP/warpreg.bin" <<'PY'
import struct, sys
got = struct.unpack('<128I', open(sys.argv[1], 'rb').read()[:512])
bad = 0
for x in range(32):
    want = [x - 1 if x & 7 else x,
            x + 1 if (x & 7) != 7 else x,
            x ^ 4,
            x ^ 2]
    for k, w in enumerate(want):
        g = got[x * 4 + k]
        if g != w:
            print('  lane %d row %d got %d want %d' % (x, k, g, w)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    warpreg      a per-lane segment end held in a register clamps where it should"
else
  echo "FAIL  warpreg      a shuffle crossed its segment with a register clamp"; fail=1
fi
runf kernels/warpreg.ptx warpreg --grid 32 --threads 32 --buffers 128 --exact
"$WEFT" run kernels/warpmask.ptx --kernel warpmask --grid 32 --threads 32 --buffers 416 \
    --cpu-only --dump 0="$TMP/warpmask.bin" >/dev/null
if python3 - "$TMP/warpmask.bin" <<'PY'
import struct, sys
got = struct.unpack('<416I', open(sys.argv[1], 'rb').read()[:1664])
bad = 0
for x in range(32):
    want = [(0x11111111 << (x & 3)) & 0xFFFFFFFF,
            (1 << x) & 0xFFFFFFFF,
            0xFFFFFFFF, 1,
            0, 0,
            (0x11111111 << (x & 3)) & 0x00FFFFFF,
            0x00FFFFFF, 1,
            0,
            1 if x == 0 else 0,
            2,
            1 if x == 2 else 0]
    for k, w in enumerate(want):
        g = got[x * 13 + k]
        if g != w:
            print('  lane %d row %d got 0x%08X want 0x%08X' % (x, k, g, w)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    warpmask     match.any, match.all and elect answer the mask they should"
else
  echo "FAIL  warpmask     a match mask or an elected lane came back wrong"; fail=1
fi
runf kernels/warpmask.ptx warpmask --grid 32 --threads 32 --buffers 416 --exact
"$WEFT" run kernels/redux.ptx --kernel reduxk --grid 32 --threads 32 --buffers 352 \
    --cpu-only --dump 0="$TMP/redux.bin" >/dev/null
if python3 - "$TMP/redux.bin" <<'PY'
import struct, sys
got = struct.unpack('<352I', open(sys.argv[1], 'rb').read()[:1408])
want = [496, 0, 31, 0, 31, 0, 0xFFFFFFF0, 0xFFFFFFF0, 15, 120, 3]
bad = 0
for x in range(32):
    for k, w in enumerate(want):
        g = got[x * 11 + k]
        if g != w:
            print('  lane %d row %d got 0x%08X want 0x%08X' % (x, k, g, w)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    redux        every integer fold, signed and masked, answers the warp's value"
else
  echo "FAIL  redux       a fold dropped a lane or used the wrong identity"; fail=1
fi
runf kernels/redux.ptx reduxk --grid 32 --threads 32 --buffers 352 --exact
"$WEFT" run kernels/isaedge.ptx --kernel isaedge --grid 1 --threads 1 --buffers 124 \
    --cpu-only --dump 0="$TMP/isaedge.bin" >/dev/null
if python3 - "$TMP/isaedge.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
A, B = 0x11223344, 0x55667788

def prmt_generic(a, b, c):
    tmp, d = (b << 32) | a, 0
    for i in range(4):
        v = (c >> (4 * i)) & 0xF
        byte = (tmp >> (8 * (v & 7))) & 0xFF
        if v & 8:
            byte = 0xFF if byte & 0x80 else 0x00
        d |= byte << (8 * i)
    return d

def prmt_f4e(a, b, c):
    tmp, base, d = (b << 32) | a, c & 3, 0
    for i in range(4):
        d |= ((tmp >> (8 * (base + i))) & 0xFF) << (8 * i)
    return d

def prmt_b4e(a, b, c):
    tmp, base, d = (b << 32) | a, c & 3, 0
    for i in range(4):
        d |= ((tmp >> (8 * ((base - i) % 8))) & 0xFF) << (8 * i)
    return d

def shf_l(a, b, c, clamp):
    n = min(c, 32) if clamp else (c & 31)
    if n == 0:
        return b
    if n == 32:
        return a
    return ((b << n) | (a >> (32 - n))) & 0xFFFFFFFF

def shf_r(a, b, c, clamp):
    n = min(c, 32) if clamp else (c & 31)
    if n == 0:
        return a
    if n == 32:
        return b
    return ((a >> n) | (b << (32 - n))) & 0xFFFFFFFF

def lop3(a, b, c, imm):
    d = 0
    for bit in range(32):
        idx = ((a >> bit) & 1) | (((b >> bit) & 1) << 1) | (((c >> bit) & 1) << 2)
        if (imm >> idx) & 1:
            d |= 1 << bit
    return d

L1, L2, L3 = 0xF0F0F0F0, 0x00FF00FF, 0x0F0F0F0F
want = [
    prmt_generic(A, B, 0x3210), prmt_generic(A, B, 0x7654), prmt_generic(A, B, 0x4C80),
    prmt_f4e(A, B, 0), prmt_f4e(A, B, 2), prmt_b4e(A, B, 0),
    shf_l(0x80000001, 2, 1, False), shf_l(0x80000001, 2, 32, True),
    shf_r(1, 0x80000000, 32, True), shf_r(1, 1, 33, False),
    8, 23, 0, lop3(L1, L2, L3, 0x96), lop3(L1, L2, L3, 0xE8),
    # finite on inf, nan, min-normal; normal on subnormal, min-normal, zero
    0, 0, 1, 0, 1, 1,
    # subnormal on subnormal, min-normal, zero
    1, 0, 0,
    # number on nan, inf
    0, 1,
    # infinite on inf, -inf, zero
    1, 1, 0,
    # notanumber on nan, min-normal
    1, 0,
]
got = list(struct.unpack('<31I', data[:124]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    isaedge     prmt signs, funnel shift ends, bfind and every testp class"
else
  echo "FAIL  isaedge     an edge of a bit instruction read the wrong way"; fail=1
fi
"$WEFT" run kernels/spirv/carry.spv --kernel spvcarry --grid 1 --threads 1 --buffers 48 \
    --cpu-only --dump 0="$TMP/spvcarry.bin" >/dev/null
if python3 - "$TMP/spvcarry.bin" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
M32, M64 = 0xFFFFFFFF, (1 << 64) - 1

def add32(a, b):
    s = a + b
    return [s & M32, 1 if s > M32 else 0]

def sub32(a, b):
    return [(a - b) & M32, 1 if a < b else 0]

def umul(a, b):
    p = (a * b) & M64
    return [p & M32, (p >> 32) & M32]

def smul(a, b):
    sa, sb = (a - (1 << 32) if a >> 31 else a), (b - (1 << 32) if b >> 31 else b)
    p = (sa * sb) & M64
    return [p & M32, (p >> 32) & M32]

want = []
want += add32(M32, 1)
want += sub32(5, 7)
want += [sub32(7, 5)[1]]
want += umul(M32, M32)
want += smul(M32, M32)
want += umul(1 << 16, 1 << 16)
want += [1]  # the 64-bit sum of all ones and one carries out
got = list(struct.unpack('<12I', data[:48]))
bad = 0
for i, (g, x) in enumerate(zip(got, want)):
    if g != x:
        print('  word %d got 0x%08X want 0x%08X' % (i, g, x)); bad += 1
sys.exit(1 if bad else 0)
PY
then
  echo "ok    spvcarry    carry, borrow and the high word of a product, each its own word"
else
  echo "FAIL  spvcarry    an extended integer operation answered the wrong word"; fail=1
fi
spvcross() {
  local kernel="$1" file="$2" idx="$3"; shift 3
  if [ ! -f "kernels/spirv/$file.spv" ]; then
    echo "skip  spvcross:$kernel  no kernels/spirv/$file.spv on this host"
    return
  fi
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
spvcross texmix       texture     0 --grid 65536 --buffers 65536 --textures 64x64 --samplers nearest/clamp
spvcross globk        global      1 --grid 1048576 --buffers 1048576,1048576
spvcross texrw        texturerg   1 --grid 4096 --threads 256 --buffers 4096,4096 --textures 64x64
spvcross texarray     arraytex    0 --grid 65536 --buffers 65536 --textures '64x64*4' --samplers nearest/clamp
# The volume, both front ends, on the same eight planes.
spvcross tex3d        threed      0 --grid 65536 --buffers 65536 --textures 64x64x8 --samplers nearest/clamp

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

"$WEFT" run kernels/spirv/atom64rw.spv --kernel atom64rw --cpu-only --grid 64 --threads 32 \
    --buffers 1,1 --dump 1="$TMP/rw1.bin" >/dev/null
rw_hex=$(od -An -tx1 -N8 "$TMP/rw1.bin" | tr -d ' \n')
case "$rw_hex" in
  0100000000000000|01000000)
    echo "ok    spirv:atom64rw  64-bit atomic load and store: B = A + 1" ;;
  *)
    echo "FAIL  spirv:atom64rw  B reads $rw_hex, wanted the 64-bit one"; fail=1 ;;
esac

spvatom64() {
  local n=4096 f=kernels/spirv/atom64.spv b
  printf '\0\0\0\0\0\0\0\0' > "$TMP/a64_zero.bin"
  local io=()
  for b in 0 1 2 3 4 5; do
    io+=(--load "$b=$TMP/a64_zero.bin" --dump "$b=$TMP/a64_$b.bin")
  done
  "$WEFT" run "$f" --kernel atom64k --cpu-only --grid "$n" --threads 32 \
      --buffers 1,1,1,1,1,1 "${io[@]}" >/dev/null
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
  local gbad="" gskip=""
  # PTX, so this is the CUDA column's answer, not Metal's.
  if [ -n "$GPUCOL" ]; then
    gskip="no ptx column on this host"
  elif [ -n "$GPUREF" ]; then
    gskip="ptx column refused: $GPUREF"
  else
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
  if [ -z "$bad" ] && [ -z "$gbad" ] && [ -n "$gskip" ]; then
    echo "ok    spirv:atom64  $n lanes: add/max olds sum exactly, signed compare included"
    echo "      ($gskip, so the gpu's six counters were not read)"
  elif [ -z "$bad" ] && [ -z "$gbad" ]; then
    echo "ok    spirv:atom64  $n lanes: add/max olds sum exactly, signed compare included,"
    echo "      and the gpu's six counters are the same integers at 4,096 and 65,536"
  else
    [ -z "$bad" ] || { echo "FAIL  spirv:atom64 $bad"; fail=1; }
    [ -z "$gbad" ] || { echo "FAIL  spirv:atom64 gpu:$gbad"; fail=1; }
    [ -z "$gbad" ] || [ -z "$bad" ] || true
  fi
}
spvatom64

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
rendertest kernels/basic.metal fragwide  100x40 --buffers 4000
rendertest kernels/basic.metal fragtile 64x64 --buffers 4096
rendertest kernels/tilegroup.metal fraggroup 64x64 --buffers 4096 --textures 64x64

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

hop=$("$WEFT" run kernels/tilegroup.metal --kernel fragtilehop --render 64x64 \
        --buffers 4096 --textures 64x64 --virtual-gpu --exact 2>&1) && hoprc=0 || hoprc=$?
if [ "$hoprc" -eq 0 ] &&
   printf '%s\n' "$hop" | grep -q 'reads tile memory outside its own 32x32 tile' &&
   printf '%s\n' "$hop" | grep -q 'not compared' &&
   ! printf '%s\n' "$hop" | grep -q 'the words agree'; then
  echo "ok    render:tilehop  a read the GPU leaves undefined runs, and is not compared"
else
  echo "FAIL  render:tilehop  got rc=$hoprc: $(printf '%s\n' "$hop" | grep -E 'weft:|compare|note' | head -1)"; fail=1
fi

rm -f "$TMP/mesh_in.bin" "$TMP/mesh_v.bin" "$TMP/mesh_p.bin"
if "$WEFT" run kernels/mesh.metal --kernel meshmaker --grid 256 --buffers 256 \
     --bench 1 --dump 0="$TMP/mesh_in.bin" --dump 1="$TMP/mesh_v.bin" \
     --dump 2="$TMP/mesh_p.bin" > "$TMP/mesh.log" 2>&1 \
   && python3 - "$TMP/mesh_in.bin" "$TMP/mesh_v.bin" "$TMP/mesh_p.bin" "$TMP/mesh.log" <<'MESHPY'
import struct, sys
# The GPU ran this too: a mesh stage is lowered to plain compute, so there is a
# Metal answer to diff against on a device with no mesh pipeline at all. A host
# with no Metal backend has no GPU column, and says so instead, that is the one
# case where the model alone is the check.
log = open(sys.argv[4], encoding='utf-8').read()
if ('unavailable' not in log and ': agree' not in log
        and 'no column for this kernel' not in log):
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

rm -f "$TMP/tm_res.bin" "$TMP/tm_v.bin" "$TMP/tm_p.bin" "$TMP/tm_pl.bin"
if "$WEFT" run kernels/taskmesh.metal --kernel meshuse --grid 96 --buffers 8 \
     --bench 1 --dump 0="$TMP/tm_res.bin" --dump 1="$TMP/tm_v.bin" \
     --dump 2="$TMP/tm_p.bin" --dump 3="$TMP/tm_pl.bin" > "$TMP/tm.log" 2>&1 \
   && python3 - "$TMP/tm_res.bin" "$TMP/tm_v.bin" "$TMP/tm_p.bin" "$TMP/tm_pl.bin" \
              "$TMP/tm.log" <<'TASKPY'
import struct, sys
tlog = open(sys.argv[5], encoding='utf-8').read()
if ('unavailable' not in tlog and ': agree' not in tlog
        and 'no column for this kernel' not in tlog):
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

cached_run="$("$WEFT" run kernels/basic.metal --kernel saxpy --grid 4096 --buffers 4096,4096,4096 --cpu-only 2>&1)"
if echo "$cached_run" | grep -q "cached"; then
  echo "ok    cache       a repeated translation is reused, not recompiled"
else
  echo "FAIL  cache       the second run compiled again"
  echo "$cached_run" | grep translate | sed 's/^/      /'; fail=1
fi

short_block="$("$WEFT" run kernels/triton/wgmma.ptx --kernel wgmma_kernel --grid 32 \
  --threads 128 --buffers 32768,32768,32768 2>&1 || true)"
if echo "$short_block" | grep -q "at least one full block"; then
  echo "ok    launch:shortblock  a grid under one block is refused, not compared"
else
  echo "FAIL  launch:shortblock  a under-filled block launch was compared anyway"; fail=1
fi

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

typed=$("$WEFT" run kernels/basic.metal --kernel vec4_scale --grid 262144 \
    --buffers 262144,262144 --bench 1 | grep '^kernel')
if echo "$typed" | grep -q "float4 16B"; then
  echo "ok    typed       buffers are elements of the declared type (float4 16B)"
else
  echo "FAIL  typed       the float4 buffer was not reported as 16 bytes"
  echo "$typed" | sed 's/^/      /'; fail=1
fi

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

tdesc=$(./build/weft-gen kernels/triton/matmul.ptx --describe)
if echo "$tdesc" | grep -q '^kernel 0 matmul_kernel kernel buffers=3 textures=0 sync=1 threads=32 types=4,4,4$'; then
  echo "ok    triton:gen  --describe carries the launch size the module declares"
else
  echo "FAIL  triton:gen"; echo "$tdesc" | sed 's/^/      /'; fail=1
fi

rm -f /tmp/weft_render_cpu.png /tmp/weft_render_gpu.png
png_ok() {
  head -c 8 "$1" | od -An -tx1 | tr -d ' \n' | grep -q '^89504e470d0a1a0a$' \
    && [ "$(od -An -tx1 -j16 -N8 "$1" | tr -d ' \n')" = "0000006400000028" ]
}
if [ -n "$CPUMODE" ]; then
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

if command -v python3 >/dev/null 2>&1; then
  cat > "$TMP/renderfma.metal" <<'METAL'
#include <metal_stdlib>
using namespace metal;

// The shapes the rewrite touches, one per channel: a subtraction with the addend
// first, a sum of two products, a lerp, and an accumulating product loop.
fragment float4 fragment_main(float4 pos [[position]]) {
  float x = pos.x;
  float y = pos.y;
  float t = x * 0.01f;
  float a = 100.0f - 2.0f * x * x;
  float b = 3.0f * x + 5.0f * (0.5f * y);
  float c = x * (1.0f - t) + y * t;
  float s = 0.0f;
  for (int j = 0; j < 8; ++j) s += x * float(j);
  return float4(a, b, c, s);
}
METAL
  rf=""
  if "$WEFT" run "$TMP/renderfma.metal" --kernel fragment_main --render 16x16 \
       --dump 0="$TMP/renderfma.bin" > "$TMP/renderfma.log" 2>&1 \
     && rf=$(python3 - "$TMP/renderfma.bin" <<'RFPY'
import struct, sys
d = open(sys.argv[1], 'rb').read()
if len(d) != 16 * 16 * 16:
    print('dump is %d bytes, not one RGBA pixel per pixel' % len(d))
    sys.exit(1)
worst = 0.0
for j in range(16):
    for i in range(16):
        x, y = i + 0.5, j + 0.5
        t = x * 0.01
        want = [100.0 - 2.0 * x * x, 3.0 * x + 5.0 * (0.5 * y),
                x * (1.0 - t) + y * t, sum(x * k for k in range(8))]
        got = struct.unpack_from('<4f', d, 16 * (j * 16 + i))
        for g, w in zip(got, want):
            worst = max(worst, abs(g - w) / max(1.0, abs(w)))
print('%.1e' % worst)
sys.exit(0 if worst < 1e-5 else 1)
RFPY
     ); then
    echo "ok    render:fma   the rewritten shapes answer the arithmetic the shader states ($rf)"
  else
    echo "FAIL  render:fma   a rewritten shape is not the arithmetic the shader states"
    tail -4 "$TMP/renderfma.log" 2>/dev/null | sed 's/^/      /'; fail=1
  fi
else
  echo "skip  render:fma   no python3 to compare against"
fi

rm -f /tmp/weft_math_cpu.bin /tmp/weft_math_gpu.bin /tmp/weft_math_in.bin
if [ -n "$CPUMODE" ]; then
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
    # angle is past the critical one, the same text Metal's documentation gives.
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

if [ -n "$CPUMODE" ]; then
  echo "skip  texel:halfimg  no gpu on this host"
else
  "$WEFT" run kernels/spirv/halftex.spv --kernel halftex --grid 256 --threads 256 \
      --textures 16x16 --texel-bytes 2 --cpu-only --dump 0="$TMP/ht.bin" >/dev/null 2>&1
  if python3 - "$TMP/ht.bin" <<'HALFIMG'
import struct, sys
raw = open(sys.argv[1], 'rb').read()
if len(raw) != 256 * 4:
    sys.exit(1)
vals = struct.unpack('<256f', raw)
for v in vals:
    if not (0.0 <= v < 4.0):
        sys.exit(1)
    if struct.unpack('<e', struct.pack('<e', v))[0] != v:
        sys.exit(1)
sys.exit(0)
HALFIMG
  then
    echo "ok    texel:halfimg  a 16x16 half image reads back as the halves it holds"
  else
    echo "FAIL  texel:halfimg  a two-byte texel did not read as a value in range"; fail=1
  fi
fi

if [ -z "${CXX:-}" ]; then
  if command -v clang++ >/dev/null 2>&1; then CXX=clang++; else CXX=c++; fi
fi
if "$CXX" -std=c++20 -O1 -Iinclude -Iruntime tools/half_texel.cpp -o "$TMP/half_texel" \
    2>"$TMP/halftex.log"; then
  ht=$("$TMP/half_texel")
  if [ "$ht" = "1 65504 5.96046e-08 2" ]; then
    echo "ok    texel:half  a two-byte texel reads back as the half it holds"
  else
    echo "FAIL  texel:half  got[$ht]"; fail=1
  fi
else
  echo "FAIL  texel:half  the probe did not compile: $(head -2 "$TMP/halftex.log")\n"; fail=1
fi

runf kernels/movhilo.ptx movhilo --grid 1 --threads 1 --buffers 2
"$WEFT" run kernels/movhilo.ptx --kernel movhilo --grid 1 --threads 1 --buffers 2 \
    --cpu-only --dump 0="$TMP/movhilo.bin" >/dev/null
if [ "$(head -c 8 "$TMP/movhilo.bin" | od -An -tx4 -v | tr -s ' ' '\n' | sed '/^$/d' | tr '\n' ' ')" = \
    "00005678 00001234 " ]; then
  echo "ok    movhilo  the 32-bit word split at 16 bits, low half then high"
else
  echo "FAIL  movhilo  read $(head -c 8 "$TMP/movhilo.bin" | od -An -tx4 -v | tr -s ' ' ' ')"; fail=1
fi

cat > "$TMP/bscale.ptx" <<'BSEOF'
.version 8.8
.target sm_120a
.address_size 64
.visible .entry bscale(.param .u64 p)
{
    .reg .b32 %r<8>;
    .reg .f32 %f<10>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [p];
    mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32.kind::block_scale {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f5,%f6,%f7,%f8}, {%r7};
    ret;
}
BSEOF
bscale_rc=0
bscale_out=$("$WEFT" run "$TMP/bscale.ptx" --kernel bscale --grid 32 --threads 32 \
    --buffers 8 --cpu-only 2>&1) || bscale_rc=$?
if [ $bscale_rc -ne 0 ] && [ "${bscale_out#*block-scaled}" != "$bscale_out" ]; then
  echo "ok    bscale    a block-scaled mma is still refused by name, not read as dense"
else
  echo "FAIL  bscale    a block-scaled mma was not refused by name (rc=$bscale_rc)"; fail=1
fi

known_uncompared=""   # empty: every fixture has a comparison now
self="${BASH_SOURCE[0]}"
uncovered=""
for fx in kernels/*.ptx kernels/cuda/*.ptx kernels/triton/*.ptx kernels/spirv/*.spv tests/*/*.ptx; do
  [ -e "$fx" ] || continue
  fxbase="$(basename "$fx")"; fxkey="${fxbase%.*}"
  case "$known_uncompared" in *" $fx "*) continue ;; esac
  if grep -qF "$fx" "$self"; then continue; fi
  case "$fx" in
    kernels/cuda/*)   if grep -qE "^nvptx +$fxkey([[:space:]]|$)" "$self"; then continue; fi ;;
    kernels/triton/*) if grep -qE "^triton +$fxkey([[:space:]]|$)" "$self"; then continue; fi ;;
    kernels/spirv/*)  if grep -qE "^spvcross +[^ ]+ +$fxkey([[:space:]]|$)" "$self"; then continue; fi
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

if python3 - "$WEFT" <<'PYIR'
import json, subprocess, sys
weft = sys.argv[1]
bad = 0
for fx in ["kernels/basic.metal", "kernels/mesh.metal"]:
    txt = subprocess.run([weft, "run", fx, "--emit-ir"], capture_output=True, text=True).stdout
    js = subprocess.run([weft, "run", fx, "--emit-ir-json"], capture_output=True, text=True).stdout
    a = [l.split()[1] for l in txt.splitlines() if l.startswith("kernel ")]
    b = [k["name"] for k in json.loads(js)["kernels"]]
    if a != b:
        print("text and json disagree on %s: %d vs %d kernels" % (fx, len(a), len(b)))
        bad = 1
d = json.loads(subprocess.run([weft, "run", "kernels/basic.metal", "--emit-ir-json"],
                              capture_output=True, text=True).stdout)
texmix = [k for k in d["kernels"] if k["name"] == "texmix"][0]
if not texmix["textures"] or not texmix["textures"][0]["shape"]:
    print("texmix has no texture binding in the IR")
    bad = 1
mesh = json.loads(subprocess.run([weft, "run", "kernels/mesh.metal", "--emit-ir-json"],
                                 capture_output=True, text=True).stdout)
if not mesh["kernels"][0]["mesh"]:
    print("meshmaker has no mesh sizes in the IR")
    bad = 1
sys.exit(bad)
PYIR
then
  echo "ok    ir           text and json name the same kernels, textures and mesh included"
else
  echo "FAIL  ir           the two IR forms disagree"; fail=1
fi

if vout=$("$WEFT" run kernels/basic.metal --kernel scale --grid 1024 --threads 64 \
            --buffers 1024,1024 --virtual-gpu --exact 2>&1) &&
   printf '%s\n' "$vout" | grep -q 'weft virtual Metal' &&
   printf '%s\n' "$vout" | grep -q 'the words agree' &&
   rout=$("$WEFT" run kernels/tilegroup.metal --kernel fraggroup --render 64x64 --grid 4096 \
            --buffers 4096 --textures 64x64 --virtual-gpu --exact 2>&1) &&
   printf '%s\n' "$rout" | grep -q 'the words agree' &&
   mout=$("$WEFT" run kernels/mesh.metal --virtual-gpu --grid 64 --threads 32 --exact 2>&1) &&
   printf '%s\n' "$mout" | grep -q 'the words agree' &&
   hout=$("$WEFT" run kernels/spirv/halftex.spv --kernel halftex --grid 256 --threads 256 \
            --textures 16x16 --texel-bytes 2 --virtual-gpu --exact 2>&1) &&
   printf '%s\n' "$hout" | grep -q 'the words agree'; then
  echo "ok    virtual      compute, render and mesh ran on the virtual device"
else
  echo "FAIL  virtual      the virtual device did not run the kernel"; fail=1
fi

tw=$("$WEFT" run kernels/tilewide.metal --kernel widefrag --render 32x32 --grid 1024 \
       --buffers 1024 --virtual-gpu --exact 2>&1) && trc=0 || trc=$?
if [ "$trc" -eq 0 ] && printf '%s\n' "$tw" | grep -q 'the words agree'; then
  echo "ok    tilewide    tile memory is sized from the imageblock's own width"
else
  echo "FAIL  tilewide    $(printf '%s\n' "$tw" | grep -E 'weft:|FAIL' | head -1)"; fail=1
fi

to=$("$WEFT" run kernels/tilewide.metal --kernel wideover --render 32x32 --grid 1024 \
       --buffers 1024 --virtual-gpu --exact 2>&1) && torc=0 || torc=$?
if [ "$torc" -eq 2 ] &&
   printf '%s\n' "$to" | grep -q 'touched byte 32768 of buffer 1, which is 32768 bytes long' &&
   printf '%s\n' "$to" | grep -q 'sized by the launcher'; then
  echo "ok    tileover    a body that overruns tile memory is refused by name, with the byte"
else
  echo "FAIL  tileover    $(printf '%s\n' "$to" | grep -E 'weft:|FAIL' | head -1)"; fail=1
fi

te=$("$WEFT" run kernels/tilegroup.metal --kernel tilegroup --render 64x64 --grid 4096 \
       --textures 64x64 2>&1) && terc=0 || terc=$?
tv=$("$WEFT" run kernels/tilegroup.metal --kernel tilegroup --render 64x64 --grid 4096 \
       --textures 64x64 --virtual-gpu 2>&1) && tvrc=0 || tvrc=$?
if [ "$terc" -eq 2 ] && [ "$tvrc" -eq 2 ] &&
   printf '%s\n' "$te" | grep -q "is the pass's tile stage, not its fragment" &&
   printf '%s\n' "$tv" | grep -q "is the pass's tile stage, not its fragment"; then
  echo "ok    tileentry   a tile stage named as the pass is refused by name"
else
  echo "FAIL  tileentry   $(printf '%s\n' "$te" | grep -E 'weft:|FAIL' | head -1)"; fail=1
fi

if "$CXX" -std=c++20 -O1 -Iinclude -Iruntime tools/vmetal_probe.cpp -o "$TMP/vmetal_probe" \
    2>"$TMP/vmetal_probe.log"; then
  vp=$("$TMP/vmetal_probe" 2>&1) && vr=0 || vr=$?
  vc=$("$TMP/vmetal_probe" "$(pwd)" 2>&1) && vcr=0 || vcr=$?
  if [ $vr -eq 0 ] && [ $vcr -eq 0 ] && printf '%s\n' "$vc" | grep -q 'compile.*0 differ'; then
    echo "ok    virtual:rules  dispatches and blits run in order, bad bindings refused, a binding smaller than the body writes refused by name, and Metal, PTX and SPIR-V compile at run time, with widths stated where the source cannot name them"
  else
    echo "FAIL  virtual:rules  (exit $vr/$vcr) $(printf '%s\n' "$vp" | head -2 | tr '\n' ' ') $(printf '%s\n' "$vc" | head -2 | tr '\n' ' ')"; fail=1
  fi
else
  echo "FAIL  virtual:rules  the probe did not compile: $(head -2 "$TMP/vmetal_probe.log")"; fail=1
fi

if [ -n "$CPUMODE" ]; then
  echo "skip  halfgpu      no gpu on this host"
elif hg=$("$WEFT" run kernels/spirv/halftex.spv --kernel halftex --grid 256 --threads 256 \
            --textures 16x16 --texel-bytes 2 --exact 2>&1) &&
   printf '%s\n' "$hg" | grep -q 'the words agree'; then
  echo "ok    halfgpu      a two-byte texel compared on the gpu"
else
  echo "FAIL  halfgpu      the two-byte texel disagreed on the gpu"; fail=1
fi

cat > "$TMP/lowinterp.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    half4 tint;
    float2 uv [[center_no_perspective]];
};

vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.tint = half4(half(p.x), half(p.y), half(0.5), half(1.0));
    o.uv = p;
    return o;
}

[[fragment]]
float4 fragment_main(VOut in [[stage_in]]) {
    return float4(float(in.tint.x) * 0.5f, float(in.tint.y) * 0.5f, in.uv.x * 0.5f, 1.0f);
}
EOF
low="$("$WEFT" run "$TMP/lowinterp.metal" --kernel fragment_main --render 64x64 --bench 3 $CPUMODE 2>&1 || true)"
if printf '%s\n' "$low" | grep -q DISAGREE; then
  echo "FAIL  render:interp  the two engines interpolated differently"; fail=1
elif ! printf '%s\n' "$low" | grep -q 'interp    weft'; then
  echo "FAIL  render:interp  the pass did not interpolate with weft's code"; fail=1
else
  echo "ok    render:interp  a vertex stage's stage_in is blended by the same code on both engines"
fi

cat > "$TMP/lowpoint.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float2 uv;
    float size [[point_size]];
};

vertex VOut vmain(uint vid [[vertex_id]]) {
    const float2 p[3] = { float2(2.3, 2.7), float2(4.5, 4.5), float2(6.0, 1.0) };
    VOut o;
    o.position = float4(p[vid].x / 8.0 * 2.0 - 1.0, 1.0 - p[vid].y / 8.0 * 2.0, 0.0, 1.0);
    o.uv = float2(float(vid) * 0.5, 0.25);
    o.size = 1.0;
    return o;
}

[[fragment]]
float4 fragment_main(VOut in [[stage_in]], float2 pc [[point_coord]]) {
    return float4(in.uv.x, pc.x, pc.y, 1.0);
}
EOF
pt="$("$WEFT" run "$TMP/lowpoint.metal" --kernel fragment_main --render 8x8 --bench 3 $CPUMODE 2>&1 || true)"
if printf '%s\n' "$pt" | grep -q DISAGREE; then
  echo "FAIL  render:point  the point primitive landed on different pixels"; fail=1
elif ! printf '%s\n' "$pt" | grep -q 'interp    weft'; then
  echo "FAIL  render:point  the point pass did not use weft's coverage"; fail=1
else
  echo "ok    render:point  three points placed by a vertex stage, covered by weft's own rule"
fi

cat > "$TMP/lowdiscard.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;

[[fragment]]
float4 fragment_main(float4 pos [[position]]) {
    float2 p = pos.xy / 64.0;
    if (p.x + p.y > 1.0) discard_fragment();
    return float4(p.x, p.y, 0.25, 1.0);
}
EOF
ds="$("$WEFT" run "$TMP/lowdiscard.metal" --kernel fragment_main --render 64x64 --bench 3 $CPUMODE 2>&1 || true)"
if printf '%s\n' "$ds" | grep -q DISAGREE; then
  echo "FAIL  render:discard  the discarded pixels differ"; fail=1
else
  echo "ok    render:discard  discard_fragment leaves the pixel as it was on both engines"
fi

cat > "$TMP/lowdepth.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;

struct Frag {
    float4 colour [[color(0)]];
    float depth [[depth(less)]];
};

[[fragment]]
Frag fragment_main(float4 pos [[position]]) {
    Frag o;
    o.colour = float4(pos.y / 64.0, 0.5, pos.x / 64.0, 1.0);
    o.depth = pos.x < 32.0 ? 0.25 : 2.0;
    return o;
}
EOF
dq="$("$WEFT" run "$TMP/lowdepth.metal" --kernel fragment_main --render 64x64 --bench 3 $CPUMODE 2>&1 || true)"
if printf '%s\n' "$dq" | grep -q DISAGREE; then
  echo "FAIL  render:depth  the rejected fragments wrote something"; fail=1
else
  echo "ok    render:depth  depth(less) rejects the same pixels on both engines"
fi

cat > "$TMP/lowmrt.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;

struct Frag {
    float4 a [[color(0)]];
    float4 b [[color(1)]];
};

[[fragment]]
Frag fragment_main(float4 pos [[position]]) {
    Frag o;
    o.a = float4(pos.x / 64.0, pos.y / 64.0, 0.0, 1.0);
    o.b = float4(1.0 - pos.x / 64.0, 0.25, 1.0 - pos.y / 64.0, 1.0);
    return o;
}
EOF
mr="$("$WEFT" run "$TMP/lowmrt.metal" --kernel fragment_main --render 64x64 --bench 3 $CPUMODE 2>&1 || true)"
if printf '%s\n' "$mr" | grep -q DISAGREE; then
  echo "FAIL  render:mrt  the two colour targets disagree"; fail=1
else
  echo "ok    render:mrt  two colour targets written and compared"
fi

sep_word() { od -An -tx4 -j $((16 * (32 * 64 + 62))) -N4 "$1" 2>/dev/null | tr -d ' \n'; }
cat > "$TMP/sepstruct.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float2 uv;
    float4 extra;
};
struct FIn {
    float4 position [[position]];
    float2 uv;
};
vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = p * 0.5f;
    o.extra = float4(1.0f);
    return o;
}
[[fragment]] float4 fragment_main(FIn in [[stage_in]]) {
    return float4(in.uv.x, 0.0f, 0.0f, 1.0f);
}
EOF
if [ -n "$CPUMODE" ]; then
  if "$WEFT" run "$TMP/sepstruct.metal" --kernel fragment_main --render 64x64 \
       --dump 0="$TMP/sep_cpu.bin" --bench 1 $CPUMODE > "$TMP/sep.log" 2>&1 \
     && [ "$(sep_word "$TMP/sep_cpu.bin")" = 3efa0000 ]; then
    echo "ok    render:sepstruct  the fragment's struct is filled from the stage by field name  (no gpu on this host to diff)"
  else
    echo "FAIL  render:sepstruct"; tail -4 "$TMP/sep.log" 2>/dev/null | sed 's/^/      /'; fail=1
  fi
elif "$WEFT" run "$TMP/sepstruct.metal" --kernel fragment_main --render 64x64 \
       --dump 0="$TMP/sep_cpu.bin" --dump-gpu 0="$TMP/sep_gpu.bin" --bench 1 \
       > "$TMP/sep.log" 2>&1 \
     && cmp -s "$TMP/sep_cpu.bin" "$TMP/sep_gpu.bin" \
     && [ "$(sep_word "$TMP/sep_cpu.bin")" = 3efa0000 ]; then
  echo "ok    render:sepstruct  a fragment struct the stage does not name, filled by field name"
else
  echo "FAIL  render:sepstruct"; tail -4 "$TMP/sep.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

cat > "$TMP/instance.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float2 uv;
};
vertex VOut vmain(uint vid [[vertex_id]], uint iid [[instance_id]],
                  uint base [[base_vertex]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = p + float2(float(iid + base) * 0.25f);
    return o;
}
[[fragment]] float4 fragment_main(VOut in [[stage_in]]) {
    return float4(in.uv.x * 0.5f, 0.0f, 0.0f, 1.0f);
}
EOF
if [ -n "$CPUMODE" ]; then
  if "$WEFT" run "$TMP/instance.metal" --kernel fragment_main --render 64x64 \
       --dump 0="$TMP/inst_cpu.bin" --bench 1 $CPUMODE > "$TMP/inst.log" 2>&1 \
     && [ "$(sep_word "$TMP/inst_cpu.bin")" = 3efa0000 ]; then
    echo "ok    render:instance  instance_id and base_vertex are the zero the pass draws  (no gpu on this host to diff)"
  else
    echo "FAIL  render:instance"; tail -4 "$TMP/inst.log" 2>/dev/null | sed 's/^/      /'; fail=1
  fi
elif "$WEFT" run "$TMP/instance.metal" --kernel fragment_main --render 64x64 \
       --dump 0="$TMP/inst_cpu.bin" --dump-gpu 0="$TMP/inst_gpu.bin" --bench 1 \
       > "$TMP/inst.log" 2>&1 \
     && cmp -s "$TMP/inst_cpu.bin" "$TMP/inst_gpu.bin" \
     && [ "$(sep_word "$TMP/inst_cpu.bin")" = 3efa0000 ]; then
  echo "ok    render:instance  instance_id and base_vertex are the zero both engines draw"
else
  echo "FAIL  render:instance"; tail -4 "$TMP/inst.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

clip_word() { od -An -tx4 -j $((16 * (32 * 64 + $1))) -N4 "$2" 2>/dev/null | tr -d ' \n'; }
cat > "$TMP/clipdist.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float2 uv;
    float cut [[clip_distance]];
};
struct FIn {
    float4 position [[position]];
};
vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = p;
    o.cut = 0.6f - p.x;
    return o;
}
[[fragment]] float4 fragment_main(FIn in [[stage_in]]) {
    return float4(0.25f, 0.5f, 0.75f, 1.0f);
}
EOF
if [ -n "$CPUMODE" ]; then
  if "$WEFT" run "$TMP/clipdist.metal" --kernel fragment_main --render 64x64 \
       --dump 0="$TMP/clip_cpu.bin" --bench 1 $CPUMODE > "$TMP/clip.log" 2>&1 \
     && [ "$(clip_word 37 "$TMP/clip_cpu.bin")" = 3e800000 ] \
     && [ "$(clip_word 38 "$TMP/clip_cpu.bin")" = 00000000 ]; then
    echo "ok    render:clipdistance  the plane cuts between column 37 and 38  (no gpu on this host to diff)"
  else
    echo "FAIL  render:clipdistance"; tail -4 "$TMP/clip.log" 2>/dev/null | sed 's/^/      /'; fail=1
  fi
elif "$WEFT" run "$TMP/clipdist.metal" --kernel fragment_main --render 64x64 \
       --dump 0="$TMP/clip_cpu.bin" --dump-gpu 0="$TMP/clip_gpu.bin" --bench 1 \
       > "$TMP/clip.log" 2>&1 \
     && cmp -s "$TMP/clip_cpu.bin" "$TMP/clip_gpu.bin" \
     && [ "$(clip_word 37 "$TMP/clip_cpu.bin")" = 3e800000 ] \
     && [ "$(clip_word 38 "$TMP/clip_cpu.bin")" = 00000000 ]; then
  echo "ok    render:clipdistance  a clip plane cuts the same pixel on both engines"
else
  echo "FAIL  render:clipdistance"; tail -4 "$TMP/clip.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi
clip_run() {
  if [ -n "$CPUMODE" ]; then
    "$WEFT" run "$1" --kernel fragment_main --render 64x64 --dump 0="$3" --bench 1 $CPUMODE > "$2" 2>&1
  else
    "$WEFT" run "$1" --kernel fragment_main --render 64x64 --dump 0="$3" \
      --dump-gpu 0="${3%.bin}.gpu.bin" --bench 1 > "$2" 2>&1 \
      && cmp -s "$3" "${3%.bin}.gpu.bin"
  fi
}
# The four bytes at byte offset $3 of pixel ($1, $2) of a 64-wide render.
clip_at() { od -An -tx4 -j $((16 * ($2 * 64 + $1) + $3)) -N4 "$4" 2>/dev/null | tr -d ' \n'; }

cat > "$TMP/cliparr.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float2 uv;
    float cut [[clip_distance]] [2];
};
struct FIn {
    float4 position [[position]];
};
vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = p;
    o.cut[0] = 0.6f - p.x;
    o.cut[1] = p.x - 0.2f;
    return o;
}
[[fragment]] float4 fragment_main(FIn in [[stage_in]]) {
    return float4(0.25f, 0.5f, 0.75f, 1.0f);
}
EOF
if clip_run "$TMP/cliparr.metal" "$TMP/cliparr.log" "$TMP/cliparr.bin" \
     && [ "$(clip_at 12 32 0 "$TMP/cliparr.bin")" = 00000000 ] \
     && [ "$(clip_at 13 32 0 "$TMP/cliparr.bin")" = 3e800000 ] \
     && [ "$(clip_at 37 32 0 "$TMP/cliparr.bin")" = 3e800000 ] \
     && [ "$(clip_at 38 32 0 "$TMP/cliparr.bin")" = 00000000 ]; then
  echo "ok    render:cliparray  two planes cut one band, and both engines agree on it"
else
  echo "FAIL  render:cliparray"; tail -4 "$TMP/cliparr.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

cat > "$TMP/clippoint.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float size [[point_size]];
    float cut [[clip_distance]];
};
struct FIn {
    float4 position [[position]];
};
vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 q = vid == 0 ? float2(-0.5f, -0.5f) : (vid == 1 ? float2(0.5f, -0.5f)
                                                           : float2(0.5f, 0.5f));
    VOut o;
    o.position = float4(q, 0.0f, 1.0f);
    o.size = 12.0f;
    o.cut = vid == 1 ? -1.0f : 1.0f;
    return o;
}
[[fragment]] float4 fragment_main(float2 pc [[point_coord]], FIn in [[stage_in]]) {
    return float4(pc, 0.0f, 1.0f);
}
EOF
if clip_run "$TMP/clippoint.metal" "$TMP/clippoint.log" "$TMP/clippoint.bin" \
     && [ "$(clip_at 16 48 12 "$TMP/clippoint.bin")" = 3f800000 ] \
     && [ "$(clip_at 48 16 12 "$TMP/clippoint.bin")" = 3f800000 ] \
     && [ "$(clip_at 48 48 12 "$TMP/clippoint.bin")" = 00000000 ]; then
  echo "ok    render:clippoint  a clipped point draws nothing on either engine"
else
  echo "FAIL  render:clippoint"; tail -4 "$TMP/clippoint.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

cat > "$TMP/clippoint2.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float size [[point_size]];
    float cut [[clip_distance]];
};
vertex VOut vmain(uint vid [[vertex_id]]) {
    VOut o;
    o.position = float4(vid == 0 ? float2(-0.5f, -0.5f) : float2(0.5f, 0.5f), 0.0f, 1.0f);
    o.size = 12.0f;
    o.cut = vid == 0 ? -1.0f : 1.0f;
    return o;
}
[[fragment]] float4 fragment_main(float2 pc [[point_coord]]) {
    return float4(pc, 0.0f, 1.0f);
}
EOF
if clip_run "$TMP/clippoint2.metal" "$TMP/clippoint2.log" "$TMP/clippoint2.bin" \
     && [ "$(clip_at 16 48 12 "$TMP/clippoint2.bin")" = 00000000 ] \
     && [ "$(clip_at 48 16 12 "$TMP/clippoint2.bin")" = 3f800000 ]; then
  echo "ok    render:clippoint2  a point pass with no stage_in drops its clipped point"
else
  echo "FAIL  render:clippoint2"; tail -4 "$TMP/clippoint2.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

cat > "$TMP/clipnostage.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float2 uv;
    float cut [[clip_distance]];
};
vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = p;
    o.cut = 0.6f - p.x;
    return o;
}
[[fragment]] float4 fragment_main(float4 pos [[position]]) {
    return float4(0.25f, 0.5f, 0.75f, 1.0f);
}
EOF
if clip_run "$TMP/clipnostage.metal" "$TMP/clipnostage.log" "$TMP/clipnostage.bin" \
     && [ "$(clip_at 37 32 0 "$TMP/clipnostage.bin")" = 3e800000 ] \
     && [ "$(clip_at 38 32 0 "$TMP/clipnostage.bin")" = 00000000 ]; then
  echo "ok    render:clipnostage  a fragment with no stage_in still cuts, on both engines"
else
  echo "FAIL  render:clipnostage"; tail -4 "$TMP/clipnostage.log" 2>/dev/null | sed 's/^/      /'; fail=1
fi

cat > "$TMP/unreadable.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float2 uv;
};
vertex VOut vmain(uint vid [[vertex_id]], uint amp [[amplification_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = p + float(amp);
    return o;
}
[[fragment]] float4 fragment_main(VOut in [[stage_in]]) {
    return float4(in.uv.x * 0.5f, 0.0f, 0.0f, 1.0f);
}
EOF
uns_msg="$("$WEFT" run "$TMP/unreadable.metal" --kernel fragment_main --render 64x64 2>&1 || true)"
if echo "$uns_msg" | grep -q 'vertex stage names ..amplification_id.., which weft cannot supply'; then
  echo "ok    render:unstaged  a vertex stage the pass cannot supply is refused, not swapped out"
else
  echo "FAIL  render:unstaged  the unreadable vertex stage was not refused"; fail=1
fi

exit "$fail"
