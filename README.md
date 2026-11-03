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
  subgroups, atomics, matrices.

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
294 ok, 0 fail, 0 skip
```

## What runs

| | Metal | PTX | SPIR-V |
|---|---|---|---|
| buffers, scalar parameters | yes | yes | yes |
| `threadgroup` / `.shared` memory, barriers | yes | yes | yes |
| `mbarrier`: arrivals, phases, expected bytes | — | `init`, `arrive`, `expect_tx`, parity waits | — |
| bulk copies that complete an `mbarrier` | — | `cp.async.bulk` | — |
| warp shuffles, votes, folds and quads | `simd_*` | `shfl`, `vote`, `redux` | subgroup ops, folds and quads |
| 32-bit atomics | yes | yes | yes |
| 64-bit atomics: fetch, compare-exchange, load, store | yes | yes | yes |
| textures and samplers | yes | `tld`, `tld4` on every binding | images and samplers |
| 1-D textures: one row, one coordinate | `texture1d` | `tex.1d` | `Dim = 1D`, on a `Wx1` binding |
| 2D array textures: a layer on every read and sample | `texture2d_array` | `tex.a2d` | `Arrayed` images |
| 3D textures: a plane on every read, sample and write | `texture3d` | `tex.3d` | `Dim = 3D` |
| cube textures: six faces, sampled by direction | `texturecube` | `tex.cube` | `Dim = Cube` |
| render passes: fragment, `[[tile]]` stage | yes | — | — |
| mesh and task+mesh stages | yes | — | — |
| `half` arithmetic, and `half` in threadgroup memory | yes | — | — |
| packed and vector types (`float3`, `uchar4`, …) | yes | yes | yes |
| 64-bit floats: arithmetic, comparisons, conversions | yes | yes | yes |
| matrices: products, transpose, scale | `simdgroup_matrix` | `mma`, `wgmma` | every shape: 2–4 columns of 2–4 rows |
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

# a 2D array: four slices, and the kernel names the one it reads
./build/weft run kernels/basic.metal --kernel texarray --grid 65536 \
    --buffers 65536 --textures 64x64*4

# a volume: eight planes, and every read and sample names one
./build/weft run kernels/basic.metal --kernel tex3d --grid 65536 \
    --buffers 65536 --textures 64x64x8

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
own type), `--render WxH`, `--textures WxH[*L|D],...`, `--samplers filter/address,...`,
`--arg N=V`, `--bench K`, `--cpu-only`, `--emit`, `--emit-gpu`, `--load B=path`,
`--dump B=path`, `--dump-gpu B=path`. The front end is chosen by extension.
`WEFT_THREADS=N` sets CPU workers, `WEFT_WARP=1` selects the warp-vector CPU
backend for PTX, `WEFT_CXX` names the compiler the generated module is built with.

A buffer the kernel indexes past the end of is reported instead of taking the
process down: the storage a launch hands out is followed by a page the kernel
cannot touch, so the first access past the end names the buffer, its length and
the launch — `--buffers 1 is too small for this launch` — and exits 2.

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

2. **Execute.** A barrier never switches stacks. Every kernel that suspends
   compiles to a resumable state machine wherever the front end can split it: the
   registers and locals live in a caller-owned blob, so a barrier is
   `s->pc = N; return;` and the next segment resumes with a `goto`. PTX and SPIR-V
   take that shape always; for Metal the body is split at its suspensions and every
   local it declares moves into the blob, which covers a local of any type, a
   declaration naming several (`uint x = a, y = b;`), a `for` induction variable and
   a name reused in a later scope, an array local — `float taps[4]` and `float4
   v[2]` are fields like any other — a type the body declares for itself — `struct Tap
   { float w, v; };` — lifted to file scope under a name keyed to that kernel so a
   local of it is a field too, and a `switch`, lowered to a compare chain
   with a label per arm so fallthrough, `break` and a `continue` out to the loop all
   keep their meaning. A lambda takes it too, in both shapes. One called where it is
   written keeps its parameters apart from the locals whose names they shadow, and
   every capture form (`[&]`, `[=]`, `[&, w]`, `[v]`) becomes an init-capture that
   reads the blob where the lambda stands. One kept in a local becomes a struct with
   a call operator and its captures as members — a copy for `[=]`/`[v]`, a pointer
   into the blob for `[&]`/`[&v]`, which is what a capture means — so the closure is
   a value the blob holds, carrying the copies the lane made rather than the blob's
   later values. (Two lambda *expressions* have two types, which is why a stored
   closure cannot be a lambda in the blob; a struct has a name.) A body-declared
   type may carry methods as well: C++ already forbids a local class's methods from
   naming the enclosing function's variables, so the declaration moves to file scope
   with them intact. A stored closure also keeps its own copies, its `mutable`
   writes, its initialised captures (`[t = v * 2.0f]`), a copy of any kernel
   parameter it reads, a name bound to one (`[&q = v]`), and a lambda written
   inside it. A local declared `auto` has no written type for the blob to name, so
   its field is `std::decay_t<decltype(init)>` — the type `auto` itself deduced —
   and an array with a brace initialiser is filled one element per statement,
   leaving the rest zero. A name declared again inside a nested scope is renamed,
   with every use of it inside that scope, so two live variables are two fields.
   An address space (`device`, `constant`, `thread`) is rewritten out of a body
   before a type is read from it: on the host they are the same memory, and
   `constant` is `const`. An array may have more than one extent, and a brace
   initialiser is placed by position at every level. A `threadgroup` array may be
   declared inside a block as well as at the top, since its slot in the group's
   shared block is the declaration's own ordinal rather than a running count — which
   is what a resumable kernel needs, because it jumps past declarations it already
   ran. All 77 kernels in `kernels/basic.metal` take it.
   A helper that waits on the block is inlined at its call site as a plain block
   before the calling statement, so the wait stays at statement level and an early
   `return` leaves through a label after the block. Either way the
   dispatcher enters each threadgroup once per barrier segment — a direct call
   over the lanes in the module's own function, not a call per thread and not a
   context switch. A warp shuffle uses the same mechanism — publish, return,
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
| `saxpy` (3 buffers) | 1M | 0.26 ms | 0.11 ms | 8 | **CPU faster** 2.4x |
| `saxpy` (3 buffers) | 4M | 0.92 ms | 0.91 ms | 4 | tie |
| `vec4_scale` (16-byte load/store, 4M floats) | 1M | 0.64 ms | 0.54 ms | 8 | **CPU faster** 1.2x |
| `stencil5` | 1M | 0.12 ms | 0.49 ms | 8 | 4.1x slower |
| `atomic_bins` (1M integer atomics) | 1M | 0.67 ms | 2.29 ms | 1 | 3.4x slower |
| `reduce_sum` (9 barriers) | 1M | 0.42 ms | 2.98 ms | 8 | 7.1x slower |
| `warpscan` (5 shuffles) | 1M | 0.14 ms | 2.58 ms | 8 | 18.4x slower |
| `block_reduce` (shuffles + barrier) | 1M | 0.15 ms | 3.82 ms | 8 | 25.5x slower |

Best of 30 (`--bench 30`), `/usr/bin/clang++ -O2`, CPU column on the state-machine
backend. `make numbers` regenerates the table by running the same eight launches
(`BENCH=5 make numbers` for a quick pass). This machine is fanless and its
compute-bound rows drift with its thermal state: `stencil5`'s CPU column has read
0.20, 0.49 and 0.83 ms for the same binary in one sitting, so that row is the one
to read loosely. The memory-bound rows are steady to a hundredth, and the barrier
rows sit within 7% of each other run to run.

Elementwise kernels sit at the memory roof and win or tie; a hand-written C loop
over the same 1M elements is 0.39 ms in the same compiler, so the translated
kernel is ~3.5x faster than the obvious loop. The barrier and shuffle rows are the
structural losses: each barrier or shuffle still costs one pass over the lanes,
where a GPU does it in hardware. A pass used to be an indirect call per lane; then
one call per threadgroup, with the lane loop, the shuffle pass number and the
count of lanes still running all inside the module; and now no call at all — the
body is emitted inside the lane loop, so a lane's `return` leaves the lane and not
the group. A visit was fifteen instructions in the generated assembly and six of
them were the call and its prologue; that is what moved the rows last:
`block_reduce` 7.7 → 4.6 → 3.8 ms, `warpscan` 5.9 → 2.7 → 2.6, `fillscan` 3.5 → 1.6,
`prefixscan` 6.0 → 4.5, `reduce_sum` 5.5 → 4.4 → 3.0 (see Limits).

`weft` also picks the thread count by running the kernel at `{1, 2, 4, 8}` and
keeping the winner — `atomic_bins` is 13x faster on one thread, because its 1,024
counters are 64 cache lines and eight cores ping-pong exactly those. `--bench 1`
skips the search.

## Exactness

The CPU path is exact, and the suites compare bytes rather than tolerances
wherever the answer is deterministic.

- **Every fixture in `check.sh` is a byte comparison** between the CPU and the GPU
  where a GPU is present, and between two front ends where it is not: 294 checks,
  none skipped on this host.
- **The math library is one text compiled twice.** A SPIR-V module's `sin`,
  `atan2` or `pow` is routed through it as well — the whole OpenCL.std
  trigonometry, inverse trigonometry and the pi-scaled forms — so a module's
  transcendental answers to the bit on both targets instead of the GPU's
  approximation against the host's libm. `mathx` checks 65,536 lanes of `sin`,
  `atan2` and `pow` and they match to the last bit, and `mathy` does the same for
  the nine that read a float's own bits — `ldexp`, `ilogb`, `logb`, `nextafter`,
  `maxmag`, `minmag`, `pown`, `rootn`, `expm1`. `mathz` covers the four that
  compute two numbers — `frexp`, `modf`, `sincos`, `remquo` — where the second
  value travels through the pointer the instruction takes as its last operand;
  both halves of all four agree, so nothing in this family is refused. `atomf`
  covers the float atomics: `min`, `max` and `exchange` (`OpAtomicFMinEXT`,
  `OpAtomicFMaxEXT`, and the bit-exact exchange, which needs no float form
  because the bits are the value), folded over one threadgroup so the rendezvous
  is total. A float `OpAtomicCompareExchange` is not translated because the model
  has no such instruction — the assembler refuses it: `expected Result Type to be
  integer scalar type`. On the PTX column `dp4a` is checked the same way: four
  byte products in both sign combinations, which is integer arithmetic and so
  exact on both sides rather than to a tolerance. The portable library now runs on
  that column's `.approx` forms too, and on `mufu` — the multifunction unit's
  seven approximations, which have only that one mode. Before it did, the same
  kernel read 3.4e-07 apart relative: the GPU's `sqrt` is `x * rsqrt(x)` and its
  `ex2` and `lg2` are the unit's own polynomials, so no form of those left to the
  hardware can agree with a host's libm. `mufu` is the check that they now do.
- **Every engine runs the same text** for a function the library defines. Metal's `sqrt`, `log2`,
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
- **The packed formats are one formula, run by both engines.** `pack_float_to_unorm4x8`
  and its seven siblings are arithmetic, not an approximation: `round(clamp(x, 0, 1) * 255)`
  for an unsigned normal, `round(clamp(x, -1, 1) * 127)` in two's complement for a
  signed one, and the inverse division to unpack, with `-128` clamped to -1. Both
  columns run `src/soft_math.hpp`'s version, so they agree exactly, and `packround`
  checks it over 1,048,576 values drawn from every byte boundary, exact ties and
  snorm's negative half: **max |diff| 0**. The one input the hardware's own builtin
  answers differently is an exact tie, measured rather than assumed:
  `tools/pack_probe.mm` runs the GPU builtin against the formula over 1,794 values
  and reports **0 differences away from a tie**, with 383 ties rounding differently
  because the hardware's intermediate precision is not this text's — 128.5/255 packs
  to 128 on the GPU and to 129 here. The formula is the documented rule and both
  engines run it, which is what makes the columns agree with each other.
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

- **Barrier- and shuffle-heavy kernels are 7–25x the GPU** — the three rows
  above. Each barrier or shuffle is one pass over the lanes, so those rows are
  close to the count of passes they need: `warpscan`'s five shuffles are six of
  them. A pass has been through four shapes and every one moved these rows: a
  coroutine frame resume, then a `switch` on a resume point called once per lane,
  then that `switch` inside a lane loop called once per threadgroup, and now no
  call at all — the body is emitted inside the lane loop as a lambda, so `return`
  unwinds the lane rather than the group. That last step was found by reading the
  generated assembly rather than by guessing: a visit was fifteen instructions
  with six of them the call and its prologue. It applies to both state-machine
  front ends (Metal and SPIR-V), which share the shape. `block_reduce`
  7.7 → 4.6 → 3.8 ms, `warpscan` 5.9 → 2.7 → 2.6, `fillscan` 3.5 → 1.6,
  `prefixscan` 6.0 → 4.5, `reduce_sum` 5.5 → 4.4 → 3.0, the last step measured
  either side of the change in one sitting, as close as a fanless machine allows.
  What is left is the pass count, and the backend that folds it away already
  exists: the warp-vector one, opt-in as `WEFT_WARP=1`, holds 32 lanes in one
  thread's registers and gives each lane its own program counter, so divergence is
  a split rather than a mask. It measures the same as the per-lane machine instead
  of beating it — 3.4 against 3.0 ms on `warpscan`, 5.2 against 6.0 on
  `block_reduce` — and two attempts to make it faster came back with numbers rather
  than a win: eliding the mask on a register store was flat on every kernel it was
  tried and made three rows of this suite disagree, and folding a 32-lane reduce
  with one running sum instead of five butterflies measured 1.92 ms against 1.09,
  because a butterfly stage is 32 independent combines and an accumulator is one
  dependency chain.
  A scan or a fold is the exception even to that: it is one tree over the warp's
  words, so it is computed once per warp rather than once per lane, which is what
  `prefixscan` was before its state machine (42.4 ms before the shared cache).
- **A reduce or a scan is one fold tree, written out on both engines.** Not
  `simd_sum`: the hardware folds the warp in its own order, and for floats a
  different order of additions is a different answer, so the tree is spelled out and
  both columns perform the identical sequence — one `simd_shuffle` a step on the GPU,
  one array read a step of the published words on the CPU, which keeps the whole
  tree at one rendezvous instead of one per step. It is a tree *downwards* and not
  an XOR butterfly for a reason a short warp measured: a butterfly folds lane `i`
  with lane `i^d`, the lanes past the grid never ran, and the hardware hands back
  whatever their registers hold — at 24 lanes that read a sum that was 82 off. A
  downward step only ever uses lanes that exist, and the ones that did not stand in
  with the fold's identity. `spvgpu:subfold` and `spvgpu:subfold:ragged` are the
  fixtures: half-integer lanes, so every hand-worked prefix, reduce and exclusive
  scan is exact and the residual slots must be zero.
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
  when they all live in one thread. A subgroup operation only takes that target
  when it is in a body with more than one block, where a lane runs its own program
  counter and the lanes diverge — a hardware rendezvous among lanes that have
  diverged has no answer — and then it is the same relay: every lane publishes its
  word into the warp's slot array, the sweep brings the lanes back in order, and the
  answer is folded over the slots, with a 64-bit value moved one word per pass.
  Everything else does its subgroup work at **full width, in hardware**: a shuffle
  is one `simd_shuffle`, a vote or an all-equal is one `simd_ballot` (for all-equal,
  lane 0's value is shuffled out and compared), and a reduce or a scan is the fold
  tree below. Measured on the SPIR-V column at 1,048,576 lanes, all three of these
  used to take the relay and now do not: `warpscan` **13.2 → 0.37 ms** (36x),
  `votek` **14.4 → 0.17 ms** (85x), `wide_shfl` **14.3 → 0.31 ms** (46x), each still
  byte-identical to the CPU. The `OpenCL.std` subgroup *imports* — what LLVM's
  SPIR-V backend emits — take the same route, so `sub_group_ballot`,
  `sub_group_reduce_*`, `sub_group_scan_*`, the shuffles and a 64-bit
  `sub_group_broadcast` all have a hardware form, over 64-bit values included.
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
  exponent range, the conversions in both directions — and compares the bits. The PTX
  column reaches the same software double from its own `0d…` literals down: `add`,
  `sub`, `mul`, `div`, `mad`, `neg`, `abs`, `min`, `max`, every comparison, every
  conversion between f64 and binary32 or an integer, and `ld`/`st.f64` as the eight
  bytes they are — `ptxf64` compares 4,096 lanes against the host's arithmetic at
  **max |diff| 0**. The fused multiply-add is exact as well — `fma.rn.f64` from PTX and
  OpenCL's `fma` from SPIR-V, one rounding either way — and that needed a 192-bit
  accumulator rather than the 56-bit working form the rest of the arithmetic uses: a
  product of two 53-bit significands spans 106 bits and adding a third value can cancel
  almost all of it. `tools/f64_check.py` checks it against an exact rational reference —
  ties, cancellation, both ends of the range — and `F64_EXTRA=6000` runs the longer
  sweep. A vector of doubles is carried the same way, 2, 3 or 4 components at a time:
  `weft_d2` is one struct on both engines — `uint64_t` components on the GPU, `double`
  on the CPU — and every operation on it is the helper applied per component, because
  an operator there would add two significands as integers and answer with a wrong
  number rather than refuse. `dvec` builds, adds, multiplies, divides, subtracts and
  takes one apart against the host's doubles, **max |diff| 0**. The square root is
  carried as well, and it is the one function in that library IEEE pins to the
  correctly rounded answer: an integer restoring root over the significand, so the
  software value is the host's own number rather than an approximation of it, and
  `tools/f64_check.py sqrt` sweeps it against `math.sqrt` bit for bit. What stays on
  the CPU is the rest of the transcendental library over a double (`pow`, `exp`,
  `log`, the trigonometry): those round through whatever the host's libm chose, so
  there is no bit-exact software form to compare against.
- **A ceiling says so.** `--threads` above 1024 is refused instead of quietly
  running at 256; a kernel whose `threadgroup` declarations need more than the
  48 KiB arena stops with the number it needed and the cap; more than 64
  threadgroup arrays is named. None of them dies without a word any more, and a
  kernel that indexes past the end of a buffer names the buffer, its length and
  the launch.
- **A cube runs in all three columns.** `texturecube` runs
  on the Metal front end and on the CPU behind it: six square faces stored face
  after face, a sample that takes a direction, and the face picked by the axis the
  direction points along most strongly with ties going to X then Y. `texcube`
  compares 65,536 directions exactly, each lane rebuilding the direction that
  lands on the texel it wants, so a wrong face or a flipped axis answers on a
  different texel. PTX's cube form is `(s, t, face)` — a face *number* and a place
  on it — and that is the same information a direction carries, so `weft_cube_dir`
  inverts the sampling rule: it rebuilds the direction that lands back on that face
  at that place, and the one text is compiled by both engines, so both build the
  same bits. `ptxcube` is the fixture, 65,536 samples with the face folded into the
  answer, **max |diff| 0**. SPIR-V has both this and `Dim = 1D`, and getting them
  there needed the model they are legal in: `spirv-val` rejects them under the
  OpenCL/Kernel capabilities, and it is right to — a cube needs `Shader` and a
  1-D image needs `Sampled1D`/`Image1D`, which a Kernel module cannot declare. A
  **GLCompute** module can, and that is a different shape rather than a different
  word: its entry point takes no parameters at all, so the buffers, images and
  samplers are module-scope variables and the launcher binds them by declaration
  order. `glcube`, `gl1d` and `gl2d` are the fixtures — all three validate, and
  all three compare 65,536 values at **max |diff| 0**. The other reading of a 1-D
  image is a plane one row high, which is how PTX's `tex.1d` is lowered and how
  this front end reads `Dim = 1D`: bound as `Wx1`, sampled at 0.5, the texel
  itself. The Metal front end is the column that binds a real `texture1d`.
  `glgather` gathers a plain image and `glgatherarray` a slice of a 2-D array,
  both at **max |diff| 0** — the two shapes the validator allows a Kernel module's
  gather, `Expected Image 'Dim' to be 2D, Cube, or Rect` being what it says about a
  volume. The other forms are the PTX column's, where `tld4` covers a 1-D row, a
  volume's plane and a cube face as well as a 2-D footprint. `glgroup` is the
  group-shape fixture: `LocalSize 8 8 1`, and a kernel that rebuilds its own flat
  lane from `LocalInvocationId`, so
  a decomposition that is off writes a zero where it writes a one.
- **Translation is textual, not a compiler.** It handles the shapes it was built
  for and refuses the rest by name. A swizzle works on either side of an
  assignment: reading `.xy`, `.zyx`, `.xxx`, `.wz`, a swizzle of a swizzle, or one
  applied to a call (`t.sample(...).xy`) builds a value, and writing `.yw = ...`
  stores into the members it names, as one call whose body takes the right-hand
  side into locals first — so `v.xz = v.wy` reorders in place instead of reading a
  member it has just overwritten. Componentwise comparisons with `select`, `any`
  and `all` work too. A *compound* assignment to a swizzle (`v.xy += ...`) is the
  read in the argument of that same store, so `v.yx *= v.zw` multiplies the two
  old components rather than a mix of old and new — the helper takes what it is
  given into locals before it stores anything.
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
