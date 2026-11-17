# weft

Runs GPU compute kernels on the CPU by translating them to C++. Front ends: MSL, PTX,
Triton's PTX (`tl.dot` included), SPIR-V. `./build/weft run` diffs GPU against CPU;
`weft-gen` with `include/weft/weft.h` links a kernel into your program.

## What runs

| | Metal | PTX | SPIR-V |
|---|---|---|---|
| buffers, scalar parameters, vector types | yes | yes | yes |
| `threadgroup` / `.shared` memory, barriers | yes | yes | yes |
| `mbarrier`: arrivals, phases, expected bytes | no | `init`, `arrive`, `expect_tx`, parity waits | no |
| bulk copies in, back out, reduced in; prefetch, discard | no | `cp.async.bulk`, `cp.reduce.async.bulk`, `prefetch`, `discard` | no |
| tensor-map copies, 1d to 5d, and `.im2col` | no | `cp.async.bulk.tensor` | no |
| warp shuffles, votes, folds, quads | `simd_*` | `shfl`, `vote`, `redux` | subgroup ops |
| atomics: 32- and 64-bit, `.cas` at 16, `.inc`/`.dec` | yes | yes | yes |
| 1-D, array, 3D, cube textures and samplers | `texture1d`, `texture2d_array`, `texture3d`, `texturecube` | `tex.1d`, `tex.a2d`, `tex.3d`, `tex.cube`, `tld4` | `1D`, `Arrayed`, `3D`, `Cube` |
| render passes, mesh and task stages, `[[tile]]` | yes | no | no |
| `half` and `bf16`: all arithmetic, `set`/`setp`, atomics, `slct`, `.approx` | yes | the same, packed and scalar | 16-bit floats |
| 64-bit floats: arithmetic, comparisons, conversions | yes | yes | yes |
| matrices, transpose, scale | `simdgroup_matrix` | `mma`, `wgmma`, `ldmatrix`, `stmatrix` | any 2 to 4 by 2 to 4 |
| Triton `tl.dot`: f16, tf32, bf16, fp8, int8 | no | yes | no |
| portable math (`sqrt`, `pow`, `sin`, `tan`) | yes | yes | yes |

Tensor maps are the one format weft defines rather than reads: NVIDIA publishes no layout
for the `cp.async.bulk.tensor` descriptor, so weft's layout lives in
`runtime/weft_runtime.hpp`. An object without weft's tag at offset 124 is reported, not
decoded. A kernel weft cannot express is refused by name.

## Use

```
make
./build/weft run kernels/basic.metal --kernel saxpy --grid 1048576
./build/weft run kernels/cuda/scale.ptx --kernel scale --grid 1048576 \
    --buffers 1048576,1048576 --arg 0=1.5
./build/weft run kernels/triton/add.ptx --kernel add_kernel --grid 1024 \
    --buffers 1024,1024,1024 --arg 0=1000
./build/weft run kernels/basic.metal --kernel texmix --grid 65536 \
    --buffers 65536 --textures 64x64        # also 64x64*4 arrays, 64x64x8 volumes
./build/weft run kernels/basic.metal --kernel fragshade --render 64x64 --buffers 4096
```

`--buffers a,b,c` counts elements of each buffer's own type. Other flags: `--render WxH`,
`--textures WxH[*L|D],...`, `--samplers filter/address,...`, `--arg N=V`, `--bench K`,
`--load B=path`, `--dump B=path`, `--dump-gpu B=path`, `--emit`, `--emit-gpu`, `--emit-ir`
(`-json`), `--cpu-only`, `--virtual-gpu`. `WEFT_THREADS` sets CPU workers, `WEFT_WARP=1`
the warp backend for PTX, `WEFT_CXX` the compiler. A buffer reached past its end is
reported, not fatal: storage ends at a page the kernel cannot touch, so the first bad
access names the buffer, its length and the launch, then exits 2.

`--emit-ir` is the contract a back end launches from: binding count, element widths,
parameter slots, textures and read-write flags, threadgroup size, tile, colour, payload and
props indices, mesh sizes, engine flags; `make nometal` builds a launcher with no Metal
column. `--virtual-gpu` runs the GPU column against a software device (buffers, textures, a
pipeline named by a kernel, a queue, an encoder, `dispatch_threads`) whose body is the
translated kernel, so its numbers are the CPU column's; `runtime/vmetal.hpp` is that device
for your own program (`make vmetal` runs a demo), and it is not an MSL interpreter.

## Link a kernel in

```
make gen
./build/weft-gen kernels/basic.metal --kernel saxpy -o build/saxpy.cpp
clang++ -std=c++20 -O2 -fno-strict-aliasing -I include app.cpp build/saxpy.cpp runtime/weft_runtime.cpp
```

```cpp
#include <weft/weft.h>
int k = weft_kernel_find("saxpy");
std::vector<float> a(n), b(n), out(n);
void* args[] = {a.data(), b.data(), out.data()};
weft_run(k, args, (uint32_t)n, 256);
```

No subprocess, no dylib, no device, no compiler at run time. `-fno-strict-aliasing` is
required: a module casts a buffer between the element types its kernel accesses.
`examples/embed/` shows it.

## How it works

Source becomes portable C++: buffers are pointers, thread indices context fields,
threadgroup memory a handle into the group's block. A barrier never switches stacks: a
suspending kernel is a resumable state machine (`s->pc = N; return;`, then a `goto`) with
its locals in a caller-owned blob. Each front end also has a thread-per-32-lane-warp back
end, which is what makes tensor-core fragments and 64-bit atomics expressible.

## Numbers

M1, 8 cores (4P + 4E), 8 GB, against its 7-core GPU. Best of 30, `clang++ -O2`.

| kernel | grid | GPU | CPU | threads | |
|---|---|---|---|---|---|
| `saxpy`, 3 buffers | 1M | 0.26 ms | 0.11 ms | 8 | 2.4x faster |
| `saxpy`, 3 buffers | 4M | 0.92 ms | 0.91 ms | 4 | tie |
| `vec4_scale`, 16-byte load/store | 1M | 0.64 ms | 0.54 ms | 8 | 1.2x faster |
| `stencil5` | 1M | 0.12 ms | 0.49 ms | 8 | 4.1x slower |
| `atomic_bins`, 1M atomics | 1M | 0.67 ms | 2.29 ms | 1 | 3.4x slower |
| `reduce_sum`, 9 barriers | 1M | 0.42 ms | 2.98 ms | 8 | 7.1x slower |
| `warpscan`, 5 shuffles | 1M | 0.14 ms | 2.58 ms | 8 | 18.4x slower |
| `block_reduce`, shuffles plus barrier | 1M | 0.15 ms | 3.82 ms | 8 | 25.5x slower |

`make numbers` regenerates it. A hand-written C loop over 1M elements is 0.39 ms, so the
translated kernel is about 3.5x faster than the obvious loop.

## Exactness

Fixtures compare bytes, not tolerances.

- `src/soft_math.hpp` is one text compiled twice, as MSL and as C++. All 46 float functions
  are correctly rounded at 0 ulp; the worst error against `libm` is `exp10` at 1.1e-05.
  `.approx` forms rewrite to the same library, so both columns run one expression.
- Subnormals part the columns: an fp32 multiply or divide flushes a subnormal operand and
  result to zero where `add` keeps both. `1e-20 * 1e-20` is `0x00000000` on the GPU,
  `0x000116C2` on the CPU.
- `min`/`max` are minNum/maxNum, `-0.0` below `+0.0`. A float read as an integer saturates,
  clamped after rounding: `cvt.rpi.s32.f32` of 2.5 is 3.
- The software double is exact binary64: `fma.rn.f64` uses a 192-bit accumulator, and the 51
  fp64 transcendentals are rounded against a 1,200-digit reference. `half` and `bf16` are
  single-rounded on the integer significands.
- Warp operations answer the ISA's lane-by-lane values, checked against VGRE.

## Limits

- Translation is textual, not a compiler: an undeclared register and a callee outside the
  file are refused by name. A warp kernel with fewer than 32 lanes runs only the lanes that
  exist.
- Shared-memory atomics work at 32, 16 and 64 bits under the window's lock, always on the
  warp back end. A 64-bit atomic that returns its old value, or a `cas`, walks the lanes in
  order: exact, but 585 ms for 1,048,576 lanes against the CPU's 10.5 ms. A half `min`,
  `max` or `exch` is refused, and so is a vector atomic in the window.
- `call.uni` is a real call; a callee that waits makes the call site a resume point. A
  self call in last position loops in place; with work after it, it is eight copies of the
  body, a ninth stopping the lane. Mutual tail calls share one driver.
- `trap` stops the lane and reports its number with a non-zero exit. `multimem` binds one
  replica, so `multimem.red` is refused.
- A reduce or scan is one fold tree on both engines, not `simd_sum`: the hardware folds in
  its own order, and a different order of additions is a different answer. Barrier and
  shuffle kernels are 7 to 25x the GPU.
- The CUDA column is checked against a stand-in driver (`make cuda-calls`); its binding
  is unproven until it runs on a machine with an NVIDIA driver. A real card is the
  only proof.

## Platforms

| host | build | GPU column |
|---|---|---|
| macOS, Apple silicon | `make` | yes, Metal |
| macOS, x86-64 | `make` | yes, Metal, under Rosetta |
| Linux, aarch64 or x86-64 | gcc or clang | PTX through the CUDA driver |
| Windows, x86-64 | MSYS2, clang 16+ or gcc 13+ | PTX through the CUDA driver |

The module needs `_Float16` and parenthesized aggregate initialisation, so clang 16+ or gcc
13+; the launcher probes PATH for the first compiler that accepts it. PTX goes to the CUDA
driver as written, so no toolkit is needed at build time: textures bind through CUDA arrays
with the CPU's texel bytes, filter and address mode, surfaces through a surface reference,
addressed in bytes. Render passes and SPIR-V have no CUDA run and say so; the overrun guard
page is POSIX-only.

## License

MIT. See `LICENSE`.
