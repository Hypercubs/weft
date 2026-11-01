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
190 ok, 0 fail, 0 skip
```

## What runs

| | Metal | PTX | SPIR-V |
|---|---|---|---|
| buffers, scalar parameters | yes | yes | yes |
| `threadgroup` / `.shared` memory, barriers | yes | yes | yes |
| warp shuffles and votes | `simd_*` | `shfl`, `vote` | subgroup ops |
| 32-bit atomics | yes | yes | yes |
| 64-bit atomics: fetch, compare-exchange, load, store | yes | yes | yes |
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
   read the peer on the next pass. A scan or a fold is cheaper than that: the 32
   lanes of a warp walk the same tree over the same published words, so the first
   lane to reach a call site computes it and the other 31 read their entry.

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
| `saxpy` (3 buffers) | 1M | 0.25 ms | 0.11 ms | 8 | **CPU faster** 2.4x |
| `saxpy` (3 buffers) | 4M | 0.93 ms | 0.92 ms | 1 | tie |
| `vec4_scale` (16-byte load/store, 4M floats) | 1M | 0.65 ms | 0.53 ms | 8 | **CPU faster** 1.2x |
| `stencil5` | 1M | 0.11 ms | 0.19 ms | 8 | 1.8x slower |
| `atomic_bins` (1M integer atomics) | 1M | 0.67 ms | 2.29 ms | 1 | 3.4x slower |
| `reduce_sum` (9 barriers) | 1M | 0.42 ms | 5.47 ms | 8 | 13.1x slower |
| `warpscan` (5 shuffles) | 1M | 0.14 ms | 5.81 ms | 8 | 42.4x slower |
| `block_reduce` (shuffles + barrier) | 1M | 0.15 ms | 7.61 ms | 8 | 49.1x slower |

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
  where a GPU is present, and between two front ends where it is not: 190 checks,
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
- **64-bit atomics on the GPU are exact at 4,096, 65,536, 262,144 and 1,048,576
  lanes**, compared as integers against the arithmetic rather than through a float
  view.

## Limits

- **Barrier- and shuffle-heavy kernels are 13–49x the GPU** — the three rows
  above. Each barrier or shuffle costs one resume per lane, and a sweep over a
  threadgroup measures about a nanosecond per lane, so those rows are close to the
  count of sweeps they need: `warpscan`'s five shuffles are six passes over the
  grid. Going faster means fewer passes, not a cheaper pass — lanes held in the
  registers of one thread (the warp back end) instead of suspended per lane.
  A scan or a fold is the exception: it is one tree over the warp's words, so it
  is computed once per warp rather than once per lane, and `prefixscan` costs
  5.98 ms against 42.4 ms before that shared cache (GPU 0.20 ms).
- **A 64-bit atomic that returns the old value is exact but serial on the GPU**,
  and that is an exactness decision rather than a hardware one: the comparison is
  against the order the CPU's lanes ran in, so the old values have to arrive in
  that order, and no hardware atomic delivers a chosen order. It is built from two
  32-bit words and dispatched as one thread walking every warp, where the words
  it touches are kept in locals — the counters nobody reads back are folded and
  flushed once, and a word nothing else in the kernel reads is loaded once before
  the walk and stored once after it. Exact at 1,048,576 lanes in **61 ms**, against
  the CPU's 10.3 ms, down from 825 ms when every lane paid two device round trips
  per word. A parallel form would be a different answer, not a faster one, so it
  would have to be a second row rather than a fix.
  The pair is never *locked*, and the reason is measured rather than assumed:
  `tools/lock_probe.mm` runs the same kernel with the lock taken and dropped by
  one thread (exact) and with threads contending on that word (never completes,
  in every shape tried, down to two threadgroups of 32). Ordering is not the
  obstacle — a device-scope `atomic_thread_fence` exists and a relaxed store in
  one threadgroup is seen by 1,024 waiters in others. Progress is: a lane that
  wins a section cannot run it until the other lanes in its warp leave the spin
  loop, and they cannot leave until it releases, because a warp is one program
  counter. A threadgroup of a single thread does not dodge that either; several
  are co-scheduled into one warp and deadlock across each other. So a per-lane
  section is refused by name and the same kernel runs on the warp target, where
  one thread owns all 32 lanes and there is nothing to wait for.
  What reaches that target is any module with a 64-bit atomic: loops, `OpPhi` and
  function-scope locals all take it, because the lane loop finishes one lane's whole
  control flow before it starts the next, so one program counter is enough. A
  barrier alongside the pair takes it too, and that is a second dispatch mode: one
  program counter **per lane**, lowest run first, so a lane parked on a barrier
  cannot be passed by a lane that has not arrived — the rendezvous the lanes need
  when they all live in one thread. A subgroup vote or shuffle goes the same way:
  every lane publishes its word into the warp's slot array, the sweep brings the
  lanes back in order, and the answer is folded over the slots — a ballot, an
  `all`/`any`, an all-equal, a broadcast or a shuffle, with a 64-bit value moved one
  word per pass. The `OpenCL.std` subgroup *imports* — what LLVM's SPIR-V backend
  emits — take the same relay, so `sub_group_ballot`, `sub_group_reduce_*`,
  `sub_group_scan_*`, the shuffles and a 64-bit `sub_group_broadcast` all work on
  the GPU now, and so do reductions and scans over 64-bit values: a word is what the
  rendezvous carries, so those move two at a time through a pair of windows.
- **`double`, on a GPU that has no fp64.** Metal rejects the type outright, so the
  runtime carries one in software: a double is the `uint64_t` its bits live in, and
  the arithmetic is exact IEEE-754 binary64 — round-to-nearest-even, subnormals
  included — over the 64-bit integers the hardware does have. Add, subtract, multiply,
  divide, the twelve comparisons, the conversions to and from integers and binary32,
  `floor`/`ceil`/`trunc`/`round`, `fabs`, `min`/`max`, and the subgroup reduce, scan,
  broadcast and vote forms all run on the GPU. Bit-exactness is checked the only way
  that means anything: the CPU column computes the same expressions with the host's own
  doubles and the two are compared byte for byte, and `tools/f64_check.py` sweeps each
  operation over its own value set — both signs, zero, subnormals, the ends of the
  exponent range, the conversions in both directions — and compares the bits. The
  transcendental library over a double (`sqrt`, `pow`, `exp`, …) stays on the CPU, and
  so does a vector of doubles.
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
