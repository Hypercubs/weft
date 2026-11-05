#!/bin/sh
# Runs the CUDA column against the stand-in driver in tools/fake_cuda.cpp on this
# host, whatever it has: the launcher is the non-Metal build, and the loader is
# pointed at the stand-in instead of a driver. Every call the column makes is
# recorded and checked here, so the binding, the uploads and the launch are
# tested without CUDA. Run it with `make cuda-calls`.
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
# The file is named for what the loader asks for, not for the host's convention:
# dlopen("libcuda.so.1") finds it by name under the library path below.
# shellcheck disable=SC2086
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

# A kernel whose result is only ever a texture sample: the array shape, the
# upload and the reference are what a driver sees.
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

# A half image is a different array format and half the row: the launcher says
# the texel is two bytes, and nothing is widened before the upload.
if run_case half kernels/ptxtex.ptx --grid 4096 --buffers 4096 --textures 64x64 \
   --texel-bytes 2 --bench 1; then
  case_rows "texture 2-D half" "$out/half.log" \
    "cuArrayCreate w=64 h=64 fmt=0x10 ch=1" \
    "cuMemcpy3D srcType=1 srcPitch=128 srcHeight=64 widthBytes=128 height=64 depth=1"
else
  bad "texture 2-D half (exit $st)"
fi

# Layers, a volume and a cube differ only in the descriptor: depth 4 and 8 are
# layered, six faces are a cubemap.
if run_case layers kernels/ptxarray.ptx --grid 4096 --buffers 4096 \
   --textures '64x64*4' --bench 1; then
  case_rows "texture 2-D array, four layers" "$out/layers.log" \
    "cuArray3DCreate w=64 h=64 d=4 fmt=0x20 ch=1 flags=0x1" \
    "depth=4"
else
  bad "texture 2-D array (exit $st)"
fi

if run_case vol kernels/ptx3d.ptx --grid 4096 --buffers 4096 --textures 64x64x8 --bench 1; then
  case_rows "texture 3-D, eight slices" "$out/vol.log" \
    "cuArray3DCreate w=64 h=64 d=8 fmt=0x20 ch=1 flags=0x1" \
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

# weft lowers `tex1D` to a two-dimensional array with one row, so that is the
# binding the driver is handed.
if run_case row kernels/ptx1d.ptx --grid 65536 --buffers 65536 --textures 64x1 --bench 1; then
  case_rows "texture one row" "$out/row.log" \
    "cuArrayCreate w=64 h=1 fmt=0x20 ch=1" \
    "cuMemcpy3D srcType=1 srcPitch=256 srcHeight=1 widthBytes=256 height=1 depth=1"
else
  bad "texture one row (exit $st)"
fi

# The sampler's two choices reach the driver as a filter and an address mode per
# axis, and zero addressing also sets the border colour it reads.
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

# A `.surf` is not a sampled texture, and asking the driver for a texture
# reference it never declared would be a worse answer than saying so.
if run_case surf kernels/ptxsrftmix.ptx --grid 4096 --buffers 4096 --textures 64x64 --bench 1; then
  grep -qF "'img' is a surface, and the cuda column binds sampled textures" "$out/surf.out" &&
    ok "surface refused by name" || bad "surface refused by name"
  if grep -qF "cuArrayCreate" "$out/surf.log"; then
    bad "surface made no array"
  else
    ok "surface made no array"
  fi
else
  bad "surface (exit $st)"
fi

# One parameter per width the column encodes, each stored at its own width by
# the kernel, so a cell holds the same bytes whichever column wrote it.
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
    .param .u16 params_param_5
)
{
    .reg .b64 %rd<6>;
    .reg .f32 %f<2>;
    .reg .f64 %fd<2>;
    .reg .b32 %r<2>;
    .reg .b16 %rs<3>;

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
    ret;
}
PTX

if run_case params "$out/params.ptx" --grid 256 --buffers 32 \
   --arg 0=2 --arg 1=7 --arg 2=9 --arg 3=3 --arg 4=513 --bench 1; then
  case_rows "scalar cells at their declared widths" "$out/params.log" \
    "cuModuleGetFunction params params=6" \
    "param0 cell=0x0000000000001000" \
    "param1 cell=0x0000000040000000" \
    "param2 cell=0x0000000000000007" \
    "param3 cell=0x0000000000000009" \
    "param4 cell=0x4008000000000000" \
    "param5 cell=0x0000000000000201" \
    "block=256x1x1"
else
  bad "scalar cells (exit $st)"
  sed 's/^/      /' "$out/params.out" | head -6
fi

# The launch is fed the kernel's input, not the buffer the CPU column has just
# finished writing. A loaded 0x11 has to come back as 0x11.
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

# A CPU-only run never opens the driver at all.
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
