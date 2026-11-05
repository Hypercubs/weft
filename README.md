# weft

Runs GPU compute kernels on the CPU. Not another implementation of the algorithm: the same
kernel, translated to C++.

Four front ends: Metal Shading Language, PTX, Triton (the PTX a `@triton.jit` kernel
becomes, `tl.dot` included), and SPIR-V (OpenCL kernel modules). Two ways to use it: the CLI,
which runs a kernel on the GPU and the CPU and diffs them, and `weft-gen` plus
`include/weft/weft.h`, which turns a kernel into a translation unit you link into your own
program.

## What runs

| | Metal | PTX | SPIR-V |
|---|---|---|---|
| buffers, scalar parameters, vector types | yes | yes | yes |
| `threadgroup` / `.shared` memory, barriers | yes | yes | yes |
| `mbarrier`: arrivals, phases, expected bytes | no | `init`, `arrive`, `expect_tx`, parity waits | no |
| bulk copies: into threadgroup, back out, reduced in | no | `cp.async.bulk`, `cp.reduce.async.bulk` | no |
| prefetches and discards | no | `prefetch`, `prefetchu`, `discard` | no |
| tensor-map copies: a box of a laid-out tensor, in and out | no | `cp.async.bulk.tensor`, 1d to 5d | no |
| tensor-map columns: a convolution's channels by pixels | no | `cp.async.bulk.tensor` `.im2col` | no |
| warp shuffles, votes, folds, quads | `simd_*` | `shfl`, `vote`, `redux` | subgroup ops |
| 32-bit and 64-bit atomics | yes | yes | yes |
| textures and samplers | yes | `tld`, `tld4` | images and samplers |
| 1-D, array, 3D, cube textures | `texture1d`, `texture2d_array`, `texture3d`, `texturecube` | `tex.1d`, `tex.a2d`, `tex.3d`, `tex.cube` | `1D`, `Arrayed`, `3D`, `Cube` |
| render passes, mesh and task stages, `[[tile]]` | yes | no | no |
| `half` arithmetic, and `half` in threadgroup memory | yes | no | no |
| 64-bit floats: arithmetic, comparisons, conversions | yes | yes | yes |
| matrices: products, transpose, scale | `simdgroup_matrix` | `mma`, `wgmma`, `ldmatrix`, `stmatrix`, in 8x8 half and 16x16, 16x8 byte shapes and the 6- and 4-bit ones that widen as they load | every shape, 2 to 4 columns of 2 to 4 rows |
| Triton `tl.dot`: f16, tf32, bf16, fp8, int8 | no | yes | no |
| portable math (`sqrt`, `pow`, `sin`) | yes | yes | yes |

The tensor-map row is the one place weft defines a format rather than reads one. NVIDIA does
not publish the byte layout of the descriptor `cp.async.bulk.tensor` reads, so it is
documented in `runtime/weft_runtime.hpp` and both engines decode the same bytes. An object
without weft's tag at offset 124 is reported instead of decoded.

A kernel that cannot be expressed in a form this engine runs is refused by name at
translation time.

## Use

```
make
./build/weft run kernels/basic.metal --kernel saxpy --grid 1048576
./build/weft run kernels/basic.metal --kernel block_reduce --grid 1048576 \
    --threads 256 --buffers 1048576,4096
./build/weft run kernels/cuda/scale.ptx --kernel scale --grid 1048576 \
    --buffers 1048576,1048576 --arg 0=1.5
./build/weft run kernels/spirv/blockreduce.spv --kernel block_reduce --grid 1048576 \
    --threads 256 --buffers 1048576,4096
./build/weft run kernels/triton/add.ptx --kernel add_kernel --grid 1024 \
    --buffers 1024,1024,1024 --arg 0=1000
./build/weft run kernels/basic.metal --kernel texmix --grid 65536 --buffers 65536 \
    --textures 64x64                  # also 64x64*4 arrays, 64x64x8 volumes
./build/weft run kernels/basic.metal --kernel fragshade --render 64x64 --buffers 4096 \
    --textures 64x64
./build/weft run my.metal --emit      # the generated C++
./build/weft run my.metal --emit-gpu  # the Metal source the GPU compiles
./build/weft run my.metal --cpu-only  # no Metal device needed
```

Options: `--grid`, `--threads`, `--buffers a,b,c` (elements of each buffer's own type),
`--render WxH`, `--textures WxH[*L|D],...`, `--samplers filter/address,...`, `--arg N=V`,
`--bench K`, `--cpu-only`, `--emit`, `--emit-gpu`, `--load B=path`, `--dump B=path`,
`--dump-gpu B=path`. `WEFT_THREADS` sets CPU workers, `WEFT_WARP=1` selects the warp-vector
backend for PTX, `WEFT_CXX` the compiler.

A buffer the kernel indexes past the end of is reported instead of taking the process down:
the storage is followed by a page the kernel cannot touch, so the first bad access names the
buffer, its length, and the launch, then exits 2.

## Use it from your code

```
make gen
./build/weft-gen kernels/basic.metal --kernel saxpy -o build/saxpy.cpp
clang++ -std=c++20 -O2 -fno-strict-aliasing -I include app.cpp build/saxpy.cpp runtime/weft_runtime.cpp
```

Compile with `-fno-strict-aliasing`: a generated module casts a buffer between the element
types its kernel accesses, which is what the GPU does, and a compiler that assumes those
casts cannot alias can drop the writes. clang 18 on Linux did exactly that to the half and
bf16 reduction kernels; clang 17 on macOS did not.

```cpp
#include <weft/weft.h>

int k = weft_kernel_find("saxpy");
std::vector<float> a(n), b(n), out(n);
void* args[] = {a.data(), b.data(), out.data()};
weft_run(k, args, (uint32_t)n, 256);
```

No subprocess, no dylib, no Metal device, no `clang++` at run time. `examples/embed/` is a
worked example.

## How it works

Source becomes portable C++: buffers become pointers, thread indices become context fields,
threadgroup memory becomes a handle into the group's block, barriers become the runtime's
barrier, and the math library is rewritten to the portable one.

A barrier never switches stacks. A kernel that suspends compiles to a resumable state
machine: registers and locals live in a caller-owned blob, so a barrier is
`s->pc = N; return;` and the next segment resumes with a `goto`. PTX and SPIR-V always take
that shape. For Metal the body is split at its suspensions and every local it declares moves
into the blob: an array, a `switch`, a `struct` the body declares, a lambda stored in a local.
An `auto` local becomes `std::decay_t<decltype(init)>`, a name reused in a nested scope is
renamed, and a body-declared type with methods moves to file scope. The body is emitted inside
the lane loop, so a lane's `return` leaves the lane. A scan or a fold is cheaper than a
shuffle, because the 32 lanes walk one tree over the same published words.

A Metal kernel that only indexes by `thread_position_in_grid` is also emitted as a `for` loop
over a contiguous range, so pointer arithmetic hoists and the compiler vectorises.

The GPU side has two back ends per front end: one thread per lane, and one thread per 32-lane
warp. The second makes tensor-core fragments and 64-bit atomics expressible, because a
peer's register is `r[l]` inside one thread.

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

`make numbers` regenerates it. Elementwise kernels sit at the memory roof, and a hand-written
C loop over the same 1M elements is 0.39 ms in the same compiler, so the translated kernel is
about 3.5x faster than the obvious loop. Compute-bound rows drift with the machine's thermal
state: `stencil5` has read 0.20, 0.49, and 0.83 ms for the same binary.

## Exactness

Fixtures compare bytes, not tolerances.

- The math library is one text compiled twice: `src/soft_math.hpp` builds as MSL for the GPU
  and as C++ for the CPU from `+ - * /`, `fma`, and fixed trip counts, and a shader's `sin`,
  `atan2`, or `pow` is rewritten to it. Metal's own `sqrt` is bit-for-bit `x * rsqrt(x)`, so
  no CPU can reproduce its bits. Worst absolute error against `libm` over 65,536 inputs:
  `sin` 1.8e-07, `tan` 3.2e-07, `atan2` 3.9e-07, `exp10` 1.1e-05, `powr` 7.7e-07.
- Every float function is correctly rounded, all 46 at 0 ulp. Each is evaluated in a float
  pair carrying 48 bits, and Ziv's test compares the residual with the gap to the nearest
  midpoint; only when it does not clear does the fp64 text run, at one `exp` call in 1,300,
  one `log` in 20,000, one `atan` in 50,000. On this M1, 65,536 lanes taking eight draws
  each: `sin` 33 ns on the CPU and 5.8 on the GPU, `erf` 92 and 3.4, `erfc` 102 and 3.2,
  `tgamma` 200 and 6.8, where the software double cost 1.68 µs and 21, 31, and 50 µs.
- `double` on a GPU that has no fp64: a double is the `uint64_t` its bits live in, and the
  arithmetic is exact IEEE-754 binary64 over the 64-bit integers the hardware has.
  `fma.rn.f64` needs a 192-bit accumulator, since two 53-bit significands span 106 bits. The
  fp64 transcendentals are a double-double over those primitives, 51 functions, correctly
  rounded against a 1,200-digit reference.
- `half` is exact, arithmetic and storage, including inside threadgroup memory.
- The packed formats are one formula run by both engines: `round(clamp(x, 0, 1) * 255)` for
  a unorm, the same in two's complement for a snorm. Only an exact tie is answered
  differently by the hardware's builtin, and `tools/pack_probe.mm` counts 0 differences away
  from a tie.
- A fused multiply-add has to be written: `a + b * c` is one rounding on the GPU and two
  across a vector operator on the CPU, so `fma(rd, t, ro)` makes both run one instruction.

## Limits

- Barrier and shuffle kernels are 7 to 25x the GPU. Each barrier or shuffle is one pass over
  the lanes, so those rows are close to the pass count: `warpscan`'s five shuffles are six.
  The warp-vector backend that would fold the rest away (`WEFT_WARP=1`) measures the same
  rather than better.
- A reduce or a scan is one fold tree, written out on both engines, not `simd_sum`: the
  hardware folds the warp in its own order, and a different order of additions is a different
  answer. It is a tree downwards, not an XOR butterfly, because the lanes past the grid never
  ran; at 24 lanes a butterfly read a sum 82 off.
- A 64-bit atomic that returns the old value is exact but serial on the GPU, since the
  comparison is against the order the CPU's lanes ran in. It is two 32-bit words dispatched
  as one thread walking every warp: 585 ms for 1,048,576 lanes against the CPU's 10.5 ms.
  Everything else does its subgroup work at full width in hardware.
- A cube and a 1-D image need a Shader module on SPIR-V, since `spirv-val` rejects a cube
  under the OpenCL and Kernel capabilities. A GLCompute module can declare them, and then its
  entry point takes no parameters, so buffers, images, and samplers are module-scope
  variables bound by declaration order.
- The CUDA column has been checked against a stand-in driver (`make cuda-calls`), which
  records the calls it makes, but not against a real one: its timings and agreements are
  unproven until a machine with a driver has run it.
- Translation is textual, not a compiler. It handles the shapes it was built for and refuses
  the rest by name.

## Platforms

| host | build | GPU column |
|---|---|---|
| macOS, Apple silicon | `make` | yes, Metal |
| macOS, x86-64 | `make` | yes, Metal, under Rosetta |
| Linux, aarch64 | gcc or clang | PTX through the CUDA driver |
| Linux, x86-64 | same | PTX through the CUDA driver |
| Windows, x86-64 | MSYS2, clang 16+ or gcc 13+ | PTX through the CUDA driver |

The generated module needs `_Float16` and parenthesized aggregate initialisation, so clang 16
or newer, or gcc 12 or newer. The launcher probes the compilers on PATH and uses the first one
that accepts that module, so an older clang++ ahead of a newer gcc is not a problem. On macOS the GPU column is Metal's. Elsewhere the PTX a front
end was given goes to the CUDA driver as written, buffers and scalars included, so no toolkit
is needed at build time: the driver is opened at run time and a host without one runs the CPU
column. Textures are bound through CUDA arrays, with the same texel bytes, filter and
address mode the CPU column reads; render passes and SPIR-V kernels have no CUDA run and say
so. The guard page that names an overrun is POSIX-only, so on Windows the host reports the
fault.

## License

MIT. See `LICENSE`.
