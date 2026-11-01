# weft — Metal, PTX, Triton and SPIR-V kernels on CPU cores

Takes a GPU compute kernel, unmodified, and runs it on the CPU. Not a different
implementation of the algorithm — the same kernel, translated to C++ for the
cores.

The name is the crosswise thread in a weave: a GPU calls its threads a warp, and
this is the other one.

Four front ends feed one engine:

- **Metal Shading Language** (`.metal`) — compute, fragment, `[[tile]]`, mesh and
  task+mesh stages.
- **PTX** (`.ptx`) — the virtual ISA CUDA compiles to.
- **Triton** — the `.ptx` a `@triton.jit` kernel becomes: elementwise work,
  reductions, and `tl.dot` in fp16, tf32, bf16 and fp8.
- **SPIR-V** (`.spv`) — OpenCL-Kernel modules: buffers, images, samplers,
  subgroups, atomics.

Two ways to use it, sharing every line of the engine: the **CLI**, which runs a
kernel on both the GPU and the CPU and diffs them, and **`weft-gen` +
`include/weft/weft.h`**, which turns a kernel into a translation unit you link
into your own program.

```
$ make && ./check.sh
ok    saxpy            1,048,576 values  ·  max |diff| 0  — agree
ok    reduce_sum       1,048,576 values  ·  max |diff| 0  — agree
ok    warpscan         2,097,152 values  ·  max |diff| 0  — agree
ok    raymarch         65,536 values     ·  max |diff| 0  — agree
ok    spirv:atom64     4,096 lanes: add/max olds sum exactly, signed compare included,
ok    spvgpu:atom64k   the emitted metal matches the cpu byte for byte
...
186 ok, 0 fail, 0 skip
```

## What runs

| | Metal | PTX | SPIR-V |
|---|---|---|---|
| buffers, scalar parameters | yes | yes | yes |
| `threadgroup` / `.shared` memory, barriers | yes | yes | yes |
| warp shuffles and votes | `simd_*` | `shfl`, `vote` | subgroup ops |
| 32-bit atomics | yes | yes | yes |
| 64-bit atomics, including the old value | yes | yes | yes |
| textures and samplers | yes | `tld`, `tld4` | images and samplers |
| render passes: fragment, `[[tile]]` stage | yes | — | — |
| mesh and task+mesh stages | yes | — | — |
| `half` arithmetic, and `half` in threadgroup memory | yes | — | — |
| packed and vector types (`float3`, `uchar4`, …) | yes | yes | yes |
| `simdgroup_matrix` | yes | `mma`, `wgmma` | — |
| Triton `tl.dot`: f16, tf32, bf16, fp8 | — | yes | — |
| portable math (`sqrt`, `pow`, `sin`, …) | yes | yes | yes |

A kernel that cannot be expressed in a form this engine runs is **refused by
name** at translation time. A wrong answer is worse than a refusal, and the
message says which shape was not understood.

## Use

```
make

# Metal
./build/weft run kernels/basic.metal --kernel saxpy --grid 1048576
./build/weft run kernels/basic.metal --kernel block_reduce --grid 1048576 \
    --threads 256 --buffers 1048576,4096

# PTX, and scalar parameters
./build/weft run kernels/saxpy.ptx --kernel saxpy --grid 1048576
./build/weft run kernels/cuda/scale.ptx --kernel scale --grid 1048576 \
    --buffers 1048576,1048576 --arg 0=1.5 --arg 1=1048576

# SPIR-V
./build/weft run kernels/spirv/blockreduce.spv --kernel block_reduce --grid 1048576 \
    --threads 256 --buffers 1048576,4096

# Triton: the PTX its compiler emits runs directly, `.maxntid` supplies --threads
./build/weft run kernels/triton/add.ptx --kernel add_kernel --grid 1024 \
    --buffers 1024,1024,1024 --arg 0=1000

# textures, bound at index 0, 1, ...
./build/weft run kernels/basic.metal --kernel texmix --grid 65536 \
    --buffers 65536 --textures 64x64

# a fragment stage: one invocation per pixel, colour compared like any buffer
./build/weft run kernels/basic.metal --kernel fragshade --render 64x64 \
    --buffers 4096 --textures 64x64

# a mesh stage, and a task stage in front of one
./build/weft run kernels/mesh.metal     --kernel meshmaker --grid 256 --buffers 256
./build/weft run kernels/taskmesh.metal --kernel meshuse   --grid 64  --buffers 8

# inspect what was produced
./build/weft run my.metal --emit        # the generated C++
./build/weft run my.metal --emit-gpu    # the Metal source the GPU compiles
./build/weft run my.metal --cpu-only    # no Metal device needed
./build/weft run my.metal --kernel k --dump 2=out.bin --dump-gpu 2=gpu.bin
```

Options: `--grid N`, `--threads N`, `--buffers a,b,c` (elements of each buffer's
own type), `--render WxH`, `--textures WxH,...`, `--samplers filter/address,...`,
`--arg N=V`, `--bench K`, `--cpu-only`, `--emit`, `--emit-gpu`, `--load B=path`,
`--dump B=path`, `--dump-gpu B=path`. The front end is chosen by extension.
`WEFT_THREADS=N` sets CPU workers, `WEFT_WARP=1` selects the warp-vector CPU
backend for PTX, `WEFT_CXX` names the compiler the generated module is built with.

On a host with no Metal (anything that is not macOS) the run is CPU-only and says
so on the `gpu` line — it never prints an agreement it did not check. `--dump`
plus `cmp` against a second front end is how the two are compared there.

## Use it from your code

```
make gen
./build/weft-gen kernels/basic.metal --kernel saxpy -o build/saxpy.cpp
clang++ -std=c++20 -O2 -I include app.cpp build/saxpy.cpp runtime/weft_runtime.cpp
```

```cpp
#include <weft/weft.h>

int k = weft_kernel_find("saxpy");
size_t n = 1 << 20;
std::vector<float> a(n), b(n), out(n);
void* args[] = {a.data(), b.data(), out.data()};
weft_run(k, args, (uint32_t)n, 256);
```

The kernel is a translation unit in your binary: no subprocess, no dylib, no
Metal device and no `clang++` at run time. `examples/embed/` is a worked example.
The CLI compiles each kernel once and caches the library by content hash, so a
second run of the same kernel skips the compiler, and an edit to the kernel or
the runtime cannot pick up a stale one.

## How it works

1. **Translate.** Source becomes portable C++: buffers become pointers, thread
   indices become context fields, threadgroup memory becomes a handle into the
   group's block, barriers become the runtime's barrier, and the math library is
   rewritten to the portable one below.

2. **Execute.** A barrier never switches stacks. PTX and SPIR-V kernels compile to
   a resumable state machine — every register or SSA value lives in a
   caller-owned blob, so a barrier is `s->pc = N; return;` and the next segment
   resumes with a `goto`. Metal kernels compile to C++20 coroutines. Either way
   the dispatcher calls each thread once per barrier segment: a call and a branch,
   not a context switch. A warp shuffle uses the same mechanism — publish, return,
   read the peer on the next pass.

3. **Flatten when possible.** A Metal kernel that only indexes by
   `thread_position_in_grid` is also emitted as a `for` loop over a contiguous
   range, so pointer arithmetic hoists and the compiler vectorises.

On the GPU side there are two back ends per front end: one thread per lane, and
one thread per 32-lane **warp** — the second is what makes tensor-core fragments
and 64-bit atomics expressible, because a peer's register is then `r[l]` inside
one thread instead of a shuffle across lanes.

## Numbers

Fanless 2020 MacBook Air, M1, 8 cores (4P + 4E), 8 GB, against the same kernel on
its 7-core GPU.

| kernel | grid | GPU | CPU | threads | |
|---|---|---|---|---|---|
| `saxpy` (3 buffers) | 1M | 0.28 ms | 0.11 ms | 4 | **CPU faster** 2.4x |
| `saxpy` (3 buffers) | 4M | 1.00 ms | 0.99 ms | 1 | tie |
| `vec4_scale` (16-byte load/store, 4M floats) | 1M | 0.66 ms | 0.54 ms | 8 | **CPU faster** 1.2x |
| `stencil5` | 1M | 0.13 ms | 0.20 ms | 8 | 1.6x slower |
| `atomic_bins` (1M integer atomics) | 1M | 0.67 ms | 2.35 ms | 1 | 3.5x slower |
| `reduce_sum` (9 barriers) | 1M | 0.42 ms | 5.76 ms | 8 | 13.8x slower |
| `warpscan` (5 shuffles) | 1M | 0.14 ms | 6.36 ms | 8 | 46x slower |
| `block_reduce` (shuffles + barrier) | 1M | 0.16 ms | 8.49 ms | 8 | 54x slower |

Best of 30 (`--bench 30`), `/usr/bin/clang++ -O2`, CPU column on the per-lane
backend. `make numbers` regenerates the table by running the same eight launches
(`BENCH=5 make numbers` for a quick pass).

Elementwise kernels sit at the memory roof and win or tie; a hand-written C loop
over the same 1M elements is 0.39 ms in the same compiler, so the translated
kernel is ~3.5x faster than the obvious loop. The barrier and shuffle rows are the
structural losses: each barrier or shuffle costs one resume per lane, where a GPU
does it in hardware.

`weft` also picks the thread count by running the kernel at `{1, 2, 4, 8}` and
keeping the winner — `atomic_bins` is 13x faster on one thread, because its 1,024
counters are 64 cache lines and eight cores ping-pong exactly those. `--bench 1`
skips the search.

## Exactness

The CPU path is exact, and the suites compare bytes rather than tolerances
wherever the answer is deterministic.

- **Every fixture in `check.sh` is a byte comparison** between the CPU and the GPU
  where a GPU is present, and between two front ends where it is not: 186 checks,
  none skipped on this host.
- **The math library is one text compiled twice.** Metal's `sqrt`, `log2`,
  `exp2`, `pow` and `mix` are hardware approximations (`sqrt` is bit-for-bit
  `x * rsqrt(x)`), so no CPU reproduces their bits, and `MTLMathModeSafe` does not
  change that. Both engines run `src/soft_math.hpp` instead — a single source
  compiled as MSL for the GPU and C++ for the CPU, built from `+ - * /`, `fma` and
  fixed trip counts — and the shader's own calls are rewritten to it. Worst
  absolute error against `libm` over 65,536 inputs: `sin` 1.8e-07, `tan` 3.2e-07,
  `atan2` 3.9e-07, `exp10` 1.1e-05, `powr` 7.7e-07. `kernels/mathfit.metal`
  computes 54 functions per lane and the dumps must be identical.
- **`half` is exact**, arithmetic and storage: under `MTLMathModeSafe`,
  `half + half`, `half * half`, `half / half` and a fusable multiply-add are
  bit-identical to `_Float16`, including inside threadgroup memory.
- **A fused multiply-add has to be written.** `a + b * c` is one rounding on the
  GPU and two across a vector operator on the CPU. `fma(rd, t, ro)` makes both
  engines run the same instruction, which is why the `raymarch` and `fft`
  fixtures are exact.
- **A ragged final threadgroup is defined by the launcher**, not by the GPU: the
  absent lanes' shuffle slots and threadgroup arena are zeroed before the run, so
  the same kernel at the same grid gives the same answer at 1, 2, 4 or 8 threads.
- **64-bit atomics on the GPU are exact at 4,096, 65,536 and 262,144 lanes**,
  compared as integers against the arithmetic rather than through a float view.

## Limits

- **Barrier- and shuffle-heavy kernels are 14–54x the GPU** — the three rows
  above. Each barrier or shuffle costs one resume per lane.
- **A 64-bit atomic that returns the old value is serial by construction on the
  GPU**: it is built from two 32-bit words, so a kernel that needs it is
  dispatched as one thread walking every warp. That is exact, and it is slower
  than the CPU — at 1,048,576 lanes it is slow enough that macOS's watchdog ends
  the dispatch.
- **Translation is textual, not a compiler.** It handles the shapes it was built
  for and refuses the rest by name.
- **Exactness is across runs of this build on this machine**, measured on the
  hardware above; the numbers and the GPU column are that machine's.

## Platforms

The engine is portable C++20 and the CPU path is the product, so it builds
wherever a C++ compiler does. The GPU column is the Apple-only part: the
reference it diffs against is Metal.

| host | build | GPU column |
|---|---|---|
| macOS, Apple silicon | `make` | yes — the whole suite and every number in *Numbers* |
| macOS, x86-64 | `make` | yes — under Rosetta, including a render pass and its PNG |
| Linux, aarch64 | `make`, gcc or clang, no frameworks | no: `--cpu-only` is the default |
| Linux, x86-64 | `make` | no |

```
make                              # macOS: clang++ with the frameworks
make CXX=g++                      # or name the compiler for the launcher
WEFT_CXX=g++ ./build/weft run ... # and for the module it generates
```

The generated module needs `_Float16` and parenthesized aggregate
initialisation, so **clang 16+ or gcc 13+**; older compilers fail with their own
message about one of those. On gcc the launcher adds `-fcoroutines` by itself.
Windows is not supported: the launcher uses `dlopen` and `mkdir` and has no shim.

## License

MIT. See `LICENSE`.
