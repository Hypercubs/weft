#!/bin/sh
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1

out=build/fake_cuda
mkdir -p "$out"

weft=build/weft-nometal
if [ ! -x "$weft" ]; then
  echo "FAIL  build/weft-nometal is missing: run make nometal"
  exit 1
fi

case "$(uname -s)" in
  Darwin) cxx="${CXX:-clang++}"; lib="-dynamiclib" ;;
  *) cxx="${CXX:-cc}"; lib="-shared -fPIC" ;;
esac
$cxx -std=c++20 -O1 $lib -o "$out/libcuda.so.1" tools/fake_cuda.cpp || {
  echo "FAIL  the stand-in driver did not build"
  exit 1
}

DYLD_LIBRARY_PATH="$out${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
LD_LIBRARY_PATH="$out${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH LD_LIBRARY_PATH

passes=0
fails=0

ok() {
  passes=$((passes + 1))
  echo "ok    $1"
}

bad() {
  fails=$((fails + 1))
  echo "FAIL  $1"
}

# One launch, its calls in $out/$1.log and its report in $out/$1.out.
run_case() {
  name=$1
  shift
  WEFT_FAKE_LOG="$out/$name.log" "$weft" run "$@" > "$out/$name.out" 2>&1
  st=$?
  return $st
}

# Every pattern has to be in the file.
has() {
  for pat in "$@"; do
    grep -qF -- "$pat" "$file" || return 1
  done
  return 0
}

case_rows() {
  case_name=$1
  file=$2
  shift 2
  if has "$@"; then
    ok "$case_name"
  else
    bad "$case_name"
    for pat in "$@"; do
      grep -qF -- "$pat" "$file" || echo "      missing: $pat"
    done
  fi
}

if run_case tex kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 \
   --samplers nearest/clamp --bench 1; then
  case_rows "texture 2-D float" "$out/tex.log" \
    "cuArrayCreate w=64 h=64 fmt=0x20 ch=1" \
    "cuMemcpy3D srcType=1 srcPitch=256 srcHeight=64 widthBytes=256 height=64 depth=1" \
    "cuModuleGetTexRef tex" \
    "cuTexRefSetArray flags=0" \
    "cuTexRefSetFlags 0x2" \
    "cuMemAlloc at=0x1000 bytes=16384" \
    "cuMemcpyHtoD at=0x1000 bytes=16384" \
    "cuMemcpyDtoH at=0x1000 bytes=16384" \
    "cuLaunchKernel grid=16x1x1 block=256x1x1 shared=0" \
    "cuCtxDestroy"
  case_rows "texture upload of a texture the module declares" "$out/tex.log" "declared .tex tex"
else
  bad "texture 2-D float (exit $st)"
  sed 's/^/      /' "$out/tex.out" | head -6
fi

if run_case half kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 \
   --texel-bytes 2 --bench 1; then
  case_rows "texture 2-D half" "$out/half.log" \
    "cuArrayCreate w=64 h=64 fmt=0x10 ch=1" \
    "cuMemcpy3D srcType=1 srcPitch=128 srcHeight=64 widthBytes=128 height=64 depth=1"
else
  bad "texture 2-D half (exit $st)"
fi

if run_case layers kernels/ptxarray.ptx --grid 4096 --buffers 4096 \
   --textures '64x64*4' --bench 1; then
  case_rows "texture 2-D array, four layers" "$out/layers.log" \
    "cuArray3DCreate w=64 h=64 d=4 fmt=0x20 ch=1 flags=0x1" \
    "depth=4"
else
  bad "texture 2-D array (exit $st)"
fi

if run_case vol kernels/ptx3d.ptx --grid 4096 --buffers 4096 --textures 64x64x8 --bench 1; then
  # A volume is its depth and no flag: only a 2-D array is designated layered.
  case_rows "texture 3-D, eight slices" "$out/vol.log" \
    "cuArray3DCreate w=64 h=64 d=8 fmt=0x20 ch=1 flags=0x0" \
    "depth=8"
else
  bad "texture 3-D (exit $st)"
fi

if run_case cube kernels/ptxcube.ptx --grid 4096 --buffers 4096 --textures 64x64c --bench 1; then
  case_rows "texture cube, six faces" "$out/cube.log" \
    "cuArray3DCreate w=64 h=64 d=6 fmt=0x20 ch=1 flags=0x4" \
    "depth=6"
else
  bad "texture cube (exit $st)"
fi

if run_case row kernels/ptx1d.ptx --grid 65536 --buffers 65536 --textures 64x1 --bench 1; then
  case_rows "texture one row" "$out/row.log" \
    "cuArrayCreate w=64 h=1 fmt=0x20 ch=1" \
    "cuMemcpy3D srcType=1 srcPitch=256 srcHeight=1 widthBytes=256 height=1 depth=1"
else
  bad "texture one row (exit $st)"
fi

if run_case repeat kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 \
   --samplers nearest/repeat --bench 1; then
  case_rows "sampler nearest, repeat" "$out/repeat.log" \
    "cuTexRefSetFilterMode 0" "cuTexRefSetAddressMode dim=0 mode=0"
else
  bad "sampler nearest, repeat (exit $st)"
fi

if run_case linear kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 \
   --samplers linear/clamp --bench 1; then
  case_rows "sampler linear, clamp" "$out/linear.log" \
    "cuTexRefSetFilterMode 1" "cuTexRefSetAddressMode dim=0 mode=1"
else
  bad "sampler linear, clamp (exit $st)"
fi

if run_case zero kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 \
   --samplers linear/zero --bench 1; then
  case_rows "sampler linear, zero border" "$out/zero.log" \
    "cuTexRefSetFilterMode 1" \
    "cuTexRefSetAddressMode dim=0 mode=3" \
    "cuTexRefSetBorderColor 0 0 0 0"
else
  bad "sampler linear, zero border (exit $st)"
fi

if run_case surf kernels/ptxsrftmix.ptx --grid 4096 --buffers 4096,4096 --textures 64x64,64x64 --bench 1; then
  # The stand-in driver exports no cuSurfRef entry points, so the column has to say
  # so rather than answer with numbers it did not compute.
  grep -qF "no surface reference entry points" "$out/surf.out" &&
    ok "surface with no driver entry points is reported, not faked" ||
    bad "surface with no driver entry points is reported, not faked"
  if grep -qF "cuArrayCreate" "$out/surf.log"; then
    bad "surface made no array"
  else
    ok "surface made no array"
  fi
else
  bad "surface (exit $st)"
fi

cat > "$out/params.ptx" <<'PTX'
.version 7.0
.target sm_70
.address_size 64

.visible .entry params(
    .param .u64 params_param_0,
    .param .f32 params_param_1,
    .param .u32 params_param_2,
    .param .u8  params_param_3,
    .param .f64 params_param_4,
    .param .u16 params_param_5,
    .param .f16 params_param_6
)
{
    .reg .b64 %rd<7>;
    .reg .f32 %f<2>;
    .reg .f64 %fd<2>;
    .reg .b32 %r<2>;
    .reg .b16 %rs<4>;

    ld.param.u64 %rd1, [params_param_0];
    ld.param.f32 %f1, [params_param_1];
    st.global.f32 [%rd1], %f1;
    ld.param.u32 %r1, [params_param_2];
    add.s64 %rd2, %rd1, 4;
    st.global.u32 [%rd2], %r1;
    ld.param.u8 %rs1, [params_param_3];
    add.s64 %rd3, %rd1, 8;
    st.global.u8 [%rd3], %rs1;
    ld.param.f64 %fd1, [params_param_4];
    add.s64 %rd4, %rd1, 16;
    st.global.f64 [%rd4], %fd1;
    ld.param.u16 %rs2, [params_param_5];
    add.s64 %rd5, %rd1, 24;
    st.global.u16 [%rd5], %rs2;
    ld.param.u16 %rs3, [params_param_6];
    add.s64 %rd6, %rd1, 32;
    st.global.u16 [%rd6], %rs3;
    ret;
}
PTX

if run_case params "$out/params.ptx" --grid 256 --buffers 32 \
   --arg 0=2 --arg 1=7 --arg 2=9 --arg 3=3 --arg 4=513 --arg 5=2 --bench 1 --dump 0="$out/f16.bin"; then
  case_rows "scalar cells at their declared widths" "$out/params.log" \
    "cuModuleGetFunction params params=7" \
    "param0 cell=0x0000000000001000" \
    "param1 cell=0x0000000040000000" \
    "param2 cell=0x0000000000000007" \
    "param3 cell=0x0000000000000009" \
    "param4 cell=0x4008000000000000" \
    "param5 cell=0x0000000000000201" \
    "param6 cell=0x0000000000000002" \
    "block=256x1x1"
else
  bad "scalar cells (exit $st)"
  sed 's/^/      /' "$out/params.out" | head -6
fi
if [ "$(od -An -tx1 -j32 -N2 "$out/f16.bin" 2>/dev/null | tr -d ' \n')" = "0200" ]; then
  ok "the half scalar stores what the driver was handed"
else
  bad "the half scalar stores what the driver was handed"
fi

if run_case short kernels/ptxtex.ptx --grid 32 --buffers 32 --textures 64x64 --bench 1; then
  case_rows "grid smaller than a block" "$out/short.log" "cuLaunchKernel grid=1x1x1 block=32x1x1 shared=0"
else
  bad "grid smaller than a block (exit $st)"
fi

WEFT_FAKE_REFUSE=1
export WEFT_FAKE_REFUSE
if run_case refuse kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 --bench 1; then
  case_rows "driver refusal is named" "$out/refuse.out" \
    "no column for this kernel" \
    "the driver refused the module: the stand-in driver refused the module"
  case_rows "driver refusal reached the driver" "$out/refuse.log" "cuModuleLoadData REFUSED"
  if grep -qE '^cpu ' "$out/refuse.out"; then
    ok "driver refusal still runs the cpu column"
  else
    bad "driver refusal still runs the cpu column"
  fi
else
  bad "driver refusal (exit $st)"
  sed 's/^/      /' "$out/refuse.out" | head -6
fi
unset WEFT_FAKE_REFUSE

dd if=/dev/zero of="$out/eleven.bin" bs=128 count=1 2>/dev/null
tr '\000' '\021' < "$out/eleven.bin" > "$out/eleven.tmp" && mv "$out/eleven.tmp" "$out/eleven.bin"
if run_case feed "$out/params.ptx" --grid 256 --buffers 32 --arg 0=2 \
   --load 0="$out/eleven.bin" --dump-gpu 0="$out/feed.bin" --bench 1; then
  head_bytes=$(od -An -tx1 -N4 "$out/feed.bin" | tr -d ' \n')
  if [ "$head_bytes" = "11111111" ]; then
    ok "the launch is fed the pristine input"
  else
    bad "the launch is fed the pristine input (gpu bytes $head_bytes)"
  fi
  # The stand-in does not compute, so a live comparison is one that disagrees.
  grep -qF "DISAGREE" "$out/feed.out" &&
    ok "the comparison runs both columns" || bad "the comparison runs both columns"
else
  bad "pristine input (exit $st)"
fi

rm -f "$out/cpuonly.log"
if run_case cpuonly kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 \
   --cpu-only --bench 1; then
  if [ -f "$out/cpuonly.log" ]; then
    bad "cpu only makes no driver calls"
  else
    ok "cpu only makes no driver calls"
  fi
else
  bad "cpu only (exit $st)"
fi

echo
echo "$passes ok, $fails fail"
[ "$fails" -eq 0 ]
