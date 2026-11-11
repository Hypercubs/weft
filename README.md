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
| atomics: 32-bit and 64-bit, and `.cas` at 16 bits, `.inc`/`.dec` on u32 | yes | yes | yes |
| textures and samplers | yes | `tld`, `tld4` | images and samplers |
| 1-D, array, 3D, cube textures | `texture1d`, `texture2d_array`, `texture3d`, `texturecube` | `tex.1d`, `tex.a2d`, `tex.3d`, `tex.cube` | `1D`, `Arrayed`, `3D`, `Cube` |
| render passes, mesh and task stages, `[[tile]]` | yes | no | no |
| `half` and `bf16` arithmetic, and `half` in threadgroup memory | yes | `.f16`, `.f16x2`, `.bf16`, `.bf16x2`: add, sub, mul, fma, div, rcp, sqrt, min, max (`.NaN`, `.xorsign.abs`), `set`/`setp` comparisons (scalar and packed, `p\u007cq` included), neg, abs, `atom.add`/`red.add` (scalar and packed), `slct`, and `.approx` `ex2`, `lg2`, `rsqrt`, `sin`, `cos`, `tanh` | 16-bit floats, workgroup arrays included |
| 64-bit floats: arithmetic, comparisons, conversions | yes | yes | yes |
| matrices: products, transpose, scale | `simdgroup_matrix` | `mma`, `wgmma`, `ldmatrix`, `stmatrix`, in 8x8 half and 16x16, 16x8 byte shapes and the 6- and 4-bit ones that widen as they load | every shape, 2 to 4 columns of 2 to 4 rows |
| Triton `tl.dot`: f16, tf32, bf16, fp8, int8 | no | yes | no |
| portable math (`sqrt`, `pow`, `sin`, `tan`) | yes | yes | yes |

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
- The `.approx` forms on a float are the library's, not the unit's polynomial: `sin`,
  `cos`, `tanh` and `tan`, with the `.f32` suffix or the `mufu.` spelling, all rewrite to
  it, so both columns run one expression and are compared byte for byte. A half has no
  `tan.approx` in the isa, which is why the 16-bit list above does not name one.
- Subnormals are the one place the hardware itself parts the columns, and it is the
  GPU's doing: on this M1 an fp32 multiply or divide flushes a subnormal operand to
  zero and a subnormal result to zero, while `add` keeps both. Measured over eight
  lanes: `1e-20 * 1e-20` answers `0x00000000` on the GPU where the CPU answers
  `0x000116C2`, `1e-40 * 1e18` loses its operand, `1e-30 / 1e10` loses its result, and
  `1e-30 + 1e-30` agrees. Safe math does not take this back because it is not a
  compiler setting, and `fma` inherits it from the multiply. So a kernel that lands in
  the subnormal range is the one place the two columns can differ; everything else is
  compared as words.
- `lgamma` and `lgamma_r` answer ln|Gamma|, which stays finite where `tgamma` itself
  has overflowed. The positive side is gamma's own climb with the logarithm of the
  climbed product taken off the end; the negative side reflects through
  `ln(pi) - ln|sin(pi x)| - ln Gamma(1 - x)`, and the sine there comes from the
  argument's own fraction held in a pair, because a logarithm of a sine near a pole
  needs the sine's relative accuracy. A 2,048-value sweep against the double
  reference peaked at 7 ulp, which is the bound the suite holds it to.
- `min`/`max` are minNum/maxNum with `-0.0` below `+0.0`: a NaN operand hands back the
  other one, and the two zeros are not a tie, so `max(-0.0, +0.0)` is `+0.0`. Delegating
  to `std::min`/`std::max` got both wrong (a NaN in the first operand came back out, and
  `max` returned `-0.0`). `.NaN` answers the canonical NaN, which is `0x7FFFFFFF` for a
  float and `0x7FF8000000000000` for a double -- the all-ones word is the float's only --
  and so does the plain form when both operands are NaN, as the device's own min/max
  does. An independent CPU-side NVIDIA replica, VGRE, agrees with all of these except
  that last payload, where its `fminf` returns the quieted input instead.
- `min`/`max.xorsign.abs` are implemented for `.f32`: the magnitude is the comparison of
  the two magnitudes and the sign is the exclusive or of both, so `max.xorsign.abs(1, -4)`
  is `-4`. The comparison runs on the magnitude's bit pattern, where a non-negative
  float's order is its integer order. The ISA spells these modifiers on the float form
  only, so `.NaN` and `.xorsign.abs` on a double, and a two-input `.abs` without
  `.xorsign`, are refused by name rather than answered with the plain comparison.
- `copysign` is lowered as the bit operation rather than a library call, so a NaN's
  payload rides through: `copysign(NaN(0x7FC00001), -0.0)` is `0xFFC00001`.
- A float read as an integer saturates. `F2I` clamps a value past the destination's range
  to its end and answers zero for a NaN (its `.NTZ` bit), and the clamp follows the
  rounding: `cvt.rpi.s32.f32` of 2.5 is 3. A C++ cast is undefined for every one of those
  and answered 0 for both infinities, so the CPU column and the GPU column disagreed on
  exactly the five rows that step outside the range. Both now run the same clamp, and the
  fourteen rows agree with VGRE's `__float2int_rz` / `__float2uint_rz` and `__saturatef`.
  The SPIR-V front end's `OpConvertFToS`/`OpConvertFToU` was the same bare cast and now
  runs the same helpers; the spec leaves those results undefined, the device clamps.
- The software double's own conversion saturates too, and three of its answers were wrong:
  an unsigned destination returned the two's complement of a negative magnitude instead of
  zero, the 64-bit clamp only began at `2^119` so `1e20` wrapped to `0x6BC75E2D63100000`,
  and a 32-bit destination narrowed the 64-bit result instead of clamping (`1e20` came back
  as `0x63100000`). All three clamp now, so the native double and the software one answer
  the same words.
- The three-input `min`/`max` form (`{min,max}{.NaN}{.abs}.f32 d, a, b, c`, sm_100) is
  refused by name: it asks for an answer over three magnitudes, and reading two of its
  operands was emitting C++ that did not compile.
- `shfl.sync`'s `c` operand is two fields, not one: a clamp value in bits 4:0 and a
  sub-segment mask in bits 12:8. Read as a single number, the wrong lane was chosen
  wherever a segment edge moved where the shift stopped, and the shifted-vector fast path
  ignored `c` outright. `activemask` also counted a predicated-off lane as active. Both
  follow the spec's model now, checked on a 24-lane warp with a partial member mask, a
  segment clamp, the packed segment mask and a distance held in a register. The clamp
  itself may be a register, which is how a front end writes a `width` below 32: the
  per-lane segment ends of an 8-wide `up`, `down` and `xor` agree with VGRE.
- `match.any`, `match.all` and `elect.sync` answer the mask the spec asks for: the lanes
  holding the same value, the whole member mask or nothing with the matching predicate,
  and the lowest lane of the mask. The nine `match` words agree with VGRE lane by lane;
  `elect` has no CUDA-C spelling, so it is checked against the spec's rule.
- `redux.sync` folds an integer across the warp — `add`, `min`, `max`, `and`, `or` and
  `xor`, in the signed and unsigned forms, over the full warp and over a partial member
  mask where the lanes outside it contribute the fold's identity. All eleven answers
  agree with VGRE lane by lane.
- The warp operations answer the ISA's lane-by-lane values, checked against a separate
  engine: `activemask`, `vote.ballot`, `vote.all`, `vote.any` and `shfl.sync` in all four
  modes (`idx`, `up`, `down`, `bfly`) agree with VGRE on all 256 words of a full warp.
- A register that is used but never declared is refused by name. `%fd1` with no `.reg .f64`
  was emitted as `s->fd[1]`, a field the state struct does not have, so the failure surfaced
  as a C++ error about generated code rather than as a report about the fixture.
- `weft_bitcast` fills the low bytes of a zeroed destination when the source is narrower,
  which is what keeps `mov.b64 %fd1, 0x0` at zero: `0x0` is an `int`, and copying a
  `double`'s eight bytes out of it read the four bytes that followed, answering
  `0x0000000100000000`.
- A shift count at or past the register width is clamped to it, as the ISA says: an
  unsigned right shift fills with zero and a signed one with the sign. The guard filled
  an unsigned shift with the sign bit, so `shr.u32 0xFFFFFFFF, 32` answered 1.
- `testp.normal` counts positive and negative zero as normal, which is what the ISA says
  and not what `isnormal` says: zero is neither normal nor subnormal there, so the test is
  "finite and not subnormal, or zero". Measured before the fix: 0.
- SPIR-V's four extended integer operations (`OpIAddCarry`, `OpISubBorrow`,
  `OpUMulExtended`, `OpSMulExtended`) answer a struct of two words, and a kernel takes
  them apart one member at a time. The carry is the sum below an addend when it wraps, the
  borrow is Operand 1 below Operand 2 read unsigned, and the high word of a product is the
  product above its low word -- for a signed multiply that is the product of the
  sign-extended operands, not the unsigned one. The 64-bit multiplies are refused by name,
  because their high word needs a 128-bit product neither target can name.
- The packed comparisons `vset2`/`vset4` answer 1 or 0 in each half or byte rather than
  all ones, so a lane of `0xFFFF0001` read as two signed halves is -1 against 0 while the
  same bits read as two unsigned halves are 65535 against it, and the two rows disagree.
- The scalar video instructions (`vadd`, `vsub`, `vmin`, `vmax`, `vabsdiff`, `vmad`) read
  the same bits at the type their suffixes name, so `vmin.u32.u32.u32` on `0xFFFFFFFF`
  takes the unsigned maximum while `vmin.s32.s32.s32` on those bits takes -1 and is the
  minimum. The answer is cut back to the destination word, and `.sat`, the part selectors
  and the merge form are refused rather than dropped.
- The scalar video shifts `vshl`/`vshr` clamp a count past the word to 32 or wrap it to
  five bits, which is the difference between a zero and a shift for the same count of 36,
  and a count of exactly 32 is a real answer; the fill comes from the source type. `vset`
  puts a comparison in a register, and its two source types may differ: both are widened
  to 64 bits first, where an unsigned word is a positive value and a signed one keeps its
  sign, so one comparison answers both.
- A carry chain (`add.cc` then `addc`, `sub.cc` then `subc`, `madc`) is how a wide
  integer addition is spelled, and its flag now lives in the per-thread state instead
  of being dropped: `d = a + b + cc` and `d = a - (b + cc)`, with the carry out that a
  wrapping sum leaves and the borrow out that a difference takes. A kernel holding one
  stays on the per-lane back end, because the vector one has a register per lane and no
  place for a flag that is not a register. `mad.hi` and `madc.hi` pick the high word of the product and `mad.lo`
  the low one, exact at both widths, signed included.

- `cnot` is the C logical negation and `not` is the bitwise one, so a word of all ones
  answers 0 under `cnot` and all zeros under `not`. The width only picks the register that
  is tested, which is why `.b16`, `.b32` and `.b64` all answer 1 for zero and 0 otherwise.
- `bswap` reverses bytes, not bits, so `bswap.b32` of `0x000000FF` is `0xFF000000`, which
  is the opposite end from what `brev` would give. The 64-bit form reverses all eight bytes,
  so its two halves trade places as well. One row pins both widths, including the registers
  where a single byte has to cross the whole width, and the check reads each input's bytes
  backwards rather than comparing against a table.
- `bmsk` is the run of bits from `a` up to `b`, and its edges are the isa's: a width of the
  type's own size is all ones rather than a shift by the size, a width of zero or less sets
  nothing, and a width past the type is all ones too. `.b32` and `.b64` both exist, and a run
  that crosses the top of a `.b32` keeps only the bits that fit, so `bmsk.b32` from 24 to 40
  is `0xFF000000`. One row pins eight edges of the 32-bit form, five of the 64-bit form, and
  the row that crosses the top.
- The lane masks and the warp's own names are registers, not instructions: `%lanemask_eq`,
  `%lanemask_lt`, `%lanemask_le`, `%lanemask_gt`, `%lanemask_ge`, `%warpid`, `%nwarpid`
  and `%laneid` resolve to shifts of the lane, so a 64-lane block reads `warpid` 0 and
  then 1, `nwarpid` 2, and `laneid` restarting at zero where `%tid.x` keeps counting. All
  of them were refused by name, and `%laneid` was refused in `mov` even though the
  arithmetic path knew it: the register scanner treated only a dotted name as special, so
  the undotted ones fell through as ordinary registers. A fixture pins the five masks as
  the shifts they are, and the warp split alongside them.
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
- `half` and `bf16` are exact, arithmetic and storage, including inside threadgroup
  memory, on all three front ends. A `fma` on either is one rounding of the exact
  product and sum, computed on the integer significands rather than by trusting an
  intermediate: on a host without fp16 arithmetic a half product widens to float, and
  one rounding there plus one on the way back is one ulp off at a tie. One integer
  routine rounds both widths, so the 8-bit significand of a bf16 is single-rounded the
  same way. A quotient and a root land on the nearest value of the width's own grid,
  taken from the significands with the remainder as sticky, so a subnormal result
  rounds on the subnormal grid instead of being flushed.
- An `atom.add` on a 16-bit float is a compare-exchange on the 32-bit word that
  holds the half, so the value beside it in the same word keeps its own total: 64
  lanes adding 1.0 and 0.5 to the two halves of one word leave 64.0 and 32.0,
  whatever order they land in, and a packed add rounds each half once through the
  same integer routine. A 16-bit `min` or `max` is refused by name, because the isa
  names only `.add` at that width. `atom.cas.b16` is the same word trick with no
  arithmetic in it: compare the half addressed, write it only on a match, and leave
  its neighbour alone. A `.b16` atomic that is not `.cas` is refused by name, and so
  is `.exch`, which table 41 of the isa lists only at 32 bits and above.
- `atom.inc`/`atom.dec` exist at one width and have no instruction: each is one step
  round a cycle bounded by its operand, so both are a compare-exchange loop over the
  isa's own definitions, `(r >= b) ? 0 : r + 1` and `(r == 0 || r > b) ? b : r - 1`.
  The step depends only on the value it reads, so N lanes leave the counter advanced
  N steps whatever order they land in, and the wrap is what a histogram bound uses.
- `slct` compares c as the sign its suffix names. A `.b32` register carries no sign
  of its own, so reading c as unsigned sends every negative value to the first
  argument: the fixture's `-5` has to pick the second one, and `-0.0` the first.
- A `switch` in SPIR-V is a table whose default label comes first, right after the
  selector, with a literal and a label after that, in pairs. A literal is as wide as
  the selector, so a 64-bit one takes two words and the pairs sit three words apart.
  The generated code compares the selector's bit pattern against each literal in
  turn: a signed selector matches the two's-complement literal the module holds, and
  a 64-bit `0xFFFFFFFF00000001` does not match the 32-bit case `1`.
- `brx.idx` indexes a jump table that a `.branchtargets` directive declares at a
  label: the index comes first and the table label second, and the list is
  destinations rather than storage. An index past the end is undefined in the isa
  and every producer guards it itself, llc with a `setp`/`bra` pair before the
  instruction, so the translation falls through to the instruction after the table,
  which is where a hardware jump table lands. The compact `NAME<3>` form of the
  list names three labels, and a `@%p` guard on the branch is the same guard every
  other predicated instruction carries.
- `bfind`, `bfind.s32` and `fns` are one helper each on both columns, so the two
  cannot answer differently: the highest set bit counted from the least significant
  end, that bit searched past the sign for a signed value, and the nth set bit at or
  above a starting position, each answering `0xFFFFFFFF` when there is none. A
  `.reg .f16` register is the same 16 bits `.b16` names, which is what every half
  instruction here reads, so a module that declares halves that way is not refused.
- A cache priority applied to a line, `griddepcontrol`, `pmevent`, `setmaxnreg` and a
  `createpolicy` descriptor are dropped, and `mapa` and `barrier.cluster` are the
  identity and nothing: there is no cache here to hint, no scheduler to steer, and a
  cluster is one CTA wide, so none of them can change a value a kernel observes.
- `dp2a` is the 16x8 sibling of `dp4a`: both of `a`'s 16-bit lanes always take part,
  `.lo`/`.hi` picks which byte pair of `b` does, and each lane is folded in at the
  signedness its suffix names. The sum is taken in 64 bits and cut to 32 at the end,
  so the wrap is the destination's own.
- `elect.sync` and `match.any`/`match.all` are the warp agreeing on one thing. The
  election is the lowest live lane of the mask on every engine that can run one, and
  the match counts only the lanes that ran: a member mask naming lanes 8 to 31 of a
  short warp changes nothing, on the host because the thread count is known and on the
  GPU because the ballot of the lanes that ran is what it folds.
- `movmatrix` is the transpose of the fragment `ldmatrix` distributes: lane l holds
  row l>>2 at columns 2(l&3) and 2(l&3)+1, so the transposed tile puts original element
  (2(l&3), l>>2) in the low half and (2(l&3)+1, l>>2) in the high one, both read out
  of peers four lanes apart. The fixture runs the transpose twice -- the identity -- so
  a version that moved the right elements to the wrong halves answers the wrong bytes
  instead of passing as some other permutation.
- `ldmatrix.x4` is four tiles in one instruction, and each tile sits its own 128
  bytes on: the `.x4` fixture has half h hold h, so a register's value is the position
  it came from, and the third block loads the same tiles through a per-lane address,
  lane a naming matrix a>>3 at row a&7. Writing it found a warp-store bug: the batched
  path folded a loop counter into the address as the number it held on the first pass,
  so every iteration stored to the first iteration's address. The emitter now counts
  each register's definitions before the body is emitted and follows only a register
  written once, which leaves the batched path for the straight-line addresses it was
  written for and sends a loop-carried one to the per-lane store.
- The multi-matrix forms of the wider shapes have fixtures of their own: `.x4` of the
  six- and four-bit containers, named and through a per-lane address, and `.x2` of the
  16x16 byte tile. The container values are chosen so a wrong tile, row or column
  always shows -- `(7c + 13col) & 63` and `(5c + 3col) & 15` invert along each axis in
  their own modulus -- and the second byte tile is the first XOR 0x5A, so a stride of
  zero cannot pass either.
- A store whose address moves from one round to the next is a fixture of its own
  (`ldstoreloop`), because the warp backend's batched store is only sound when the
  address is the same expression every time the statement runs. The same four stores
  are written out and then run from a loop that walks the pointer; the loop can only
  match by not folding a first value into the address.
- `stmatrix.x4` is checked as a round trip rather than against the store map alone:
  `ldmatrix.x4` of a four-tile tile, `stmatrix.x4` back into another buffer, and the
  transpose of the same four registers. The plain store has to rebuild the tile byte
  for byte, and the transposed one has to give the tile read down its columns.
- The value model refuses two more things than it used to: a register written more
  than once cannot be folded into an address, and neither can one written under a
  predicate, since the lanes that failed it still hold the old value.
- `stmatrix.m16n8.b8` is pinned by position, not by coverage: every byte holds the
  offset it belongs at XORed with its matrix number, so a row taken for a column, a
  matrix at the wrong stride or another lane's address all read back wrong. The
  lane-supplied form takes eight addresses a matrix, from lanes `8m` to `8m+7`, each
  the 16-byte block of two rows the PTX ISA assigns it.
- `tools/ptxas_verify.sh` assembles every fixture with NVIDIA's `ptxas` in a container
  and reports each rejection or the SASS it produced (`--disasm`), so a fixture's own
  declaration and the instruction it stands for are both checked by the vendor's
  assembler. It needs Docker and a couple of gigabytes for the CUDA image. It assembles
  166 of the 169 fixtures assemble, and they are every file in `kernels/`. Three are not
  assembly candidates and live under `tests/`: `cvtqual`, whose `.rz`, `.scaled` and
  four-value `.rs` qualifiers this PTX has no spelling for, and the two signature tests.
  Five of the seven that used to sit outside are written out now, each as the same
  arithmetic the instruction names: an fp4 multiply at a k of 64 is two k32 products
  summed, a signed nibble widen is the unsigned one with bit 3 of every nibble copied
  upward, an `im2col` column is the loop the map already describes, and the four-lane
  narrow arithmetic is a decode to binary32, the one operation there, and the pair
  convert back. `calldrop` and `callmiss` are the other two and are
  negative tests. A return list around a function that returns nothing, and a returned
  value with no list to put it in, are a signature mismatch on purpose, and `ptxas`
  counts return slots in that list: an assembler that took either would have stopped
  checking signatures, so neither can ever be assembled. That is why they are not
  candidates, and why `kernels/` holds only what assembles -- the gate is every file in
  it with no exception list, and a rejection there is a failure. Two more versions were
  swept before settling on
  that image: CUDA 13.1's `ptxas` (PTX 9.1, the newest ISA there is) refuses the same
  ones for the same reasons and drops about twenty fixtures the 12.9 assembler takes, so
  it is not a drop-in oracle, and the arch-specific targets (`sm_90a`, `sm_103a`,
  `sm_121a`) were tried in case the missing forms were gated on one of them rather than
  absent. They are absent. The qualifiers are gone too, but the destination register's
  width is not: a narrow pair convert names a `.b8` register for fp4 and a `.b16` one
  for f6, and a `.b32` one is the `Arguments mismatch` those files hit. With that width
  stated, `cvtmx4` and `cvtf8x2dn` are each the same bits written as two pair converts
  and a shift.
- Most of the fixtures that used to be refused were a missing spelling rather than a
  missing instruction, and each of these now writes one ptxas takes: integer `wgmma`
  accumulates into `.s32` (with the results converted once on the way out), a `tex.a2d`
  array index is a register, a `red` of -1 is an `add`, a bf16 clamp spells `max` and
  `min`, `mufu` is the bare `.approx` form, `bswap` is `prmt` with the reversing
  selector, `bmsk.b64` is its own mask formula on two shifts, a half or bf16 `div`,
  `rcp` and `sqrt` widen to a double and round once, a half `.approx` answers in float,
  `tan` is `sin/cos`, `lg2.approx.f16x2` answers each half in float, and a 1-D or 3-D
  `tld4` is the four taps of its footprint, and an fp4 or fp6 `mma` is
  `.kind::f8f6f4`, the same dense product with no block scales and, at a k of 64,
  two of them over the two halves of the reduction, a four-value MX convert
  is two pair converts and a shift, an `x2.f16x2` down-convert is the two halves
  unpacked and widened to floats, and `.pzo` is an added `+0.0`. A second `.version` and `.target` in the
  middle of a file is illegal however many entries it holds, so `bf16ops` and
  `narrowdiv` are one module each with two of them.
- Splitting those files exposed instructions whose operand list or target was wrong
  in ways the packed file hid: `red.global.add.f16x2` needs `.noftz`, a packed `set`
  states its type twice (`set.eq.f16x2.f16x2`), a shared address is a `.u32` offset
  materialised from the symbol (`mov.u32 %r, tile` then `add`), `min.xorsign.abs`
  needs `sm_86` and PTX 7.2, and `setp.bf16` needs PTX 7.8.
- A legal instruction often needed a qualifier the fixture had left out: `testp`
  spells its type (`testp.finite.f32`) and the NaN class is `.notanumber`;
  `suld`/`sust` require `.clamp` or `.trap`; `bmsk` takes register bounds, not
  immediates; a convert to an integer takes `.rni`/`.rzi`/`.rmi`/`.rpi`, not
  `.rn`; a convert from an integer to an `f64` requires `.rn` while `f32` to `f64`
  must not carry one; a `bar.red.and/or.pred`
  fold writes a predicate, so the fixture turns it into a word with `selp`;
  `cp.async` needs `sm_80`; and `dp4a` names two types, `dp4a.atype.btype`.
- Two shorthands were not legal PTX at all. `vset4` has no three-operand spelling: the
  merge operand is required, and ptxas answers `Arguments mismatch` without it. The
  16-bit float atomics take `.min` and `.max` only with `.noftz` *and* a vector modifier
  (`.v2.f16.min.noftz`), and ptxas allows a vector atomic only in global memory — so a
  shared half min exists at no spelling, `atom.min.f32` is not a form at all, and the
  shared row that wanted a float min is a `u32` min now. The packed half add wants
  `.noftz` too.
- An operand's width is part of its spelling. A narrow convert's destination is a
  `.b16` register — `.b8` when it holds an fp4 pair — and a word-sized one there is
  `Arguments mismatch`; its source is the narrow width too, so a pair that arrives in a
  word is copied down with `cvt.u16.u32` / `cvt.u8.u32` first. The `tf32` operands of
  `mma` are `.b32` registers holding bit patterns, not `.f32` ones, and the `.f32`
  sources of a four-lane `.rs` convert must really be `.f32` registers — `mov.b32`
  fills one without converting. `.rs` also takes its random word as a register: a
  literal there is `Arguments mismatch`.
- A register name exists only when its index is inside the declared count: `%rd<2>`
  declares `%rd0` and `%rd1`, so a use of `%rd2` is `Unknown symbol '%rd2'` and the
  instruction that used it draws `Arguments mismatch` — a type error in disguise.
  `callmulti` wrote `%rd3` under a `%rd<3>` declaration; widening the count to four
  is what let its `mul.wide` assemble.
- A call's return list is `.reg`, not `.param`: a `.param` list answers `Multiple
  return parameters require .register state space`, so two returned values arrive by
  `mov.b32` into the registers the signature names. A forward call also needs its
  prototype, `.visible .func () ping(.param .u64 src, .param .u64 dst);`, or ptxas
  answers `Call to 'ping' requires call prototype`; with it `callrec` is legal PTX.
- Some of these forms are version-gated as well as target-gated: the fp4 and fp6
  converts need `.version 8.6` on `sm_100a`, `.rs` needs 8.7, and the same file at
  `.version 7.0 / .target sm_80` answers `Feature 'cvt with .e2m1x2' requires PTX ISA
  .version 8.6 or later`. ptxas also refuses a byte above 126 anywhere in the file,
  comments included, so a typed em dash is `Unexpected non-ASCII character`.
- Spellings the fixtures name because the ISA does not have them: the plain
  `.e2m1`/`.e3m2` `mma`, which ptxas carries only as the block-scaled `.kind::f8f6f4`;
  the `.rn` four-value narrow converts; `cvt.rz` onto an fp8 or fp4 grid, which has no
  rounding to name; `.scaled::n1::ue8m0`, `Unknown modifier` at every version this
  assembler knows; `cp.async.bulk.tensor` with `.im2col` in the store direction; a
  vector `atom` outside global memory; and `.f32` accumulating `.s8` in `wgmma`.
- A second assembler was tried as a check on these limits and adds nothing: CUDA 13.0's
  `ptxas` (PTX 9.0) still refuses `.im2col`, `.scaled::n1::ue8m0`, `cvt.rz` on an fp8
  grid, the plain fp4/fp6 `mma`, `bswap`, `tan`, `tld4.1d/.3d` and `red.sub`, and it has
  dropped `sm_70` outright, so 12.9 stays the oracle.
- A fixture file holds one module. `halfops`, `narrowatom`, `narrowcmp` and
  `tensormap` packed thirteen modules between them into four files, which PTX does not
  allow, so they are thirteen files now, each with its own leading comment and its own
  check rows, and the assembler sees every one of them.
- A multicast reduce carries no scope modifier: `multimem.ld_reduce.global.add.u32`
  assembles and the `.gpu` and `.sys` spellings do not, so the fixture states the
  scope-less form.
- The `wgmma` operand scales are multipliers, not flags: `1` leaves the operand and `-1`
  negates it, and the transposes stay 0 or 1. weft read the field as a boolean, so a
  legal `1` negated A while a dump's `0` meant identity; it is read as a multiplier now,
  and a missing immediate means identity. How many immediates a shape takes is part of
  its spelling: five for `k16` (`m64n8k16`, `m64n16k16`), three for `k8` and `k32`
  (`tf32`, `e4m3`, `e5m2`), and one for an integer accumulator, where
  `1, 1, 1, 0, 0` is `Arguments mismatch` and `1` assembles.
- The 3-D, cube and array textures take four coordinates, `q = 1` being the projective
  identity, and the `.a2d` slice index is an integer inside that vector rather than a
  float. Both are what ptxas asks for, so the fixtures state them that way.
- A texture or surface is declared either way: `.global .texref t;`, which is what PTX
  accepts today, or the older `.tex .u32 t, 1, 1, 1, 1;` the dumps use. Both name the
  same binding in `--textures` order, so the fixtures state their bindings legally and
  the caller is still handed no index.
- The two spaces cannot share a symbol: `sust` and `suld` take a `.surfref` where `tex`
  wants a `.texref`, and declaring one name as both is `Inconsistent redefinition`, so
  `ptxsrftmix` declares one of each and binds `--textures 64x64,64x64`. Its surface read
  crosses lanes through a barrier and lands in a second buffer, which keeps the write
  inside the comparison instead of leaving it dead. A `tex` operand may also be a `.u64`
  handle rather than a declaration; ptxas takes that spelling, and weft does not read
  it, so a dump that passes a handle stays outside the bindings the front end fills.
- The fixtures carry declarations ptxas accepts now. `.version` and `.target` were raised
  to what each instruction actually needs — `wgmma` exists only on `sm_90a`, the 16x8
  and 8x16 matrix shapes on `sm_100a`, `mbarrier.expect_tx` needs PTX 8.0 — and
  `.maxntid` moved from module scope, where it is not legal PTX, to the slot between
  the parameter list and the body. The translator looks for the directive in both
  places, since the dumps that put it above the entry mean the same thing.
- `movmatrix` has no `.trans`-less form, and our own oracle assumed one: the fixture and
  its Python reference now transpose the fragment, which is what the instruction does.
- The matrix fixtures are assembled by NVIDIA's own `ptxas` and read back with
  `nvdisasm`, so each map is checked against the instruction the hardware runs and not
  only against our reading of the ISA. `stmatrixb8map` comes out as two
  `STSM.8.MT168.4` and one `STSM.8.MT168`, `ldmatrixb8` as `LDSM.8.MT1616` with
  `STSM.8.MT168.2`, `stmatrixmulti` and `ldmatrixx4` as `STSM.16.M88.4`,
  `STSM.16.MT88.4`, `LDSM.16.M88.4` and `LDSM.16.MT88.4`, and `ldmatrixmulti` as
  `LDSM.8.MT1616.2` beside `LDSM.U4` and `LDSM.U6`, the two widening loads. Reaching
  that meant correcting the fixtures' own declarations: the 16x8 and 8x16 shapes need
  `.version 8.6` or later with `.target sm_100a`, and the `.s8.s4` widening load needs
  `.version 9.4` on `sm_100f`, which is newer than the release here assembles, so that
  one rests on the ISA's own example of the same syntax.
- A shared operand's constant survives into the address. `[tile+256]` used to store
  at `tile`, in `stmatrix` and in `ldmatrix` both: the handlers rebuilt the base from
  the bare symbol name after `mem_operand` had already folded the constant in.
- The plain forms of the pack-of-lanes video instructions (`vadd2`, `vmin4`, `vabsdiff4`,
  `vavrg4` and the rest) widen each lane at the signedness its own suffix names, add in
  32 bits and cut each answer back to the lane width. The merge form is refused by
  name: its destination mask and `.asel`/`.bsel` selectors pick operands out of the
  pair, and the fourth operand it reads is not a value this subset has.
- The packed formats are one formula run by both engines: `round(clamp(x, 0, 1) * 255)` for
  a unorm, the same in two's complement for a snorm. Only an exact tie is answered
  differently by the hardware's builtin, and `tools/pack_probe.mm` counts 0 differences away
  from a tie.
- A fused multiply-add has to be written: `a + b * c` is one rounding on the GPU and two
  across a vector operator on the CPU, so `fma(rd, t, ro)` makes both run one instruction.

## Limits

- `activemask.b32` folds the lane predicate the warp body dispatches on: 32 bits for a
  full warp, and only the live lanes of a short one.
- A warp kernel launched with fewer than 32 lanes runs only the lanes that exist. Both
  warp bodies mark the lanes past the launch finished before the body runs, which is
  also what keeps a masked store from running past its buffer. Ten fixtures had been
  launching one thread and getting a whole warp for free; they now name the 32 lanes
  the instruction needs.
- `movmatrix` covers both forms: `.trans` transposes the fragment, and the form
  without it is the identity, because both operands are the same matrix in the same
  layout.
- The packed half transcendentals -- `ex2`, `lg2`, `rsqrt`, `sin`, `cos` and `tanh` on
  `.f16x2` and `.bf16x2` -- are the scalar answer per half: each half is widened,
  answered and rounded on its own, so a pair is two roundings and the low one cannot
  move the high one.
- The vector half atomics -- `atom.global.v2/v4/v8` of `.f16` or `.bf16` with `.add`,
  `.min` or `.max`, and the packed pairs at `.v2` and `.v4` -- are per element, as the
  isa says: each element is one compare-exchange on the word that holds it, so the
  element beside it keeps its value. This is also the only spelling a half has for
  `.min` and `.max`, which the scalar table does not name at all.
- Atomics in shared memory are translated at every width the window has an instruction
  for: the integer, float and 16-bit forms, the packed pairs included, with the window's
  address resolved as an offset (`sh + off`) rather than a pointer. `src/weft_msl.hpp`
  carries the family twice because MSL keeps the window and the heap as different
  pointer types, and the narrow helpers in `src/soft_math.hpp` are templates over their
  pointer so one body serves both. A 64-bit atomic in the window is refused by name:
  this host's GPU has no atomic at that width in threadgroup memory, and the lock the
  device form takes orders the heap.
- `ld`/`st.shared::cluster` and `.shared::cta` name the one window this launch owns, so
  they are the same access as `.shared`, the vector forms included: their address is
  resolved the same way, which is what `kernels/scluster.ptx` checks against a
  neighbour's words. `mapa` of rank 0 is the identity.
- `sad`, `mul24`/`mad24` (`.lo` and `.hi`, signed and unsigned) and `cvt.pack.sat`
  behave the way the ISA defines them: the 24-bit multiply widens the low 24 bits at
  the suffix's signedness, so a lane whose low 24 bits are negative multiplies as
  one, and the pack saturates each source before shifting it into place.
  `nanosleep` is a scheduling hint and emits nothing.
- `suq.width`, `suq.height` and `suq.depth` answer from the binding the launcher
  made, not from the kernel's geometry. `suq.format` and `suq.channel_data_type` are
  refused by name: a surface here is texel storage in the one format the arena is
  laid out as, so the binding carries neither field to answer with.
- `call.uni` to a `.func` in the same module is a real call: the callee is translated
  once with a register file of its own, because PTX gives a function one and pasting
  its body into the caller would make its `%rd1` the caller's `%rd1`. A callee that
  branches is the same switch a kernel body is: a branch is a write to the state's
  program counter, so a forward jump and a loop both land on the case the label defined,
  and the counter is reset on entry so a second call starts at the top rather than
  wherever the first left. A callee that calls another function carries that
  function's state as a member of its own, so a call reaches as many levels deep as
  the module says and a nested callee's loop cannot land in the caller's case; the
  functions it calls are translated first, whichever order the file writes them in.
  A value a callee hands back through its parenthesised list is a field of its own
  state, which the call site then assigns: the generated body is a switch over a
  program counter, so the `ret` that ends it returns from the middle of that switch
  and a C++ return type could not carry an answer across the jump. A callee that
  waits -- `bar.sync`, a reduction barrier, a phased mbarrier -- makes the call site
  a resume point as well: the lane gives the worker back with the callee's own counter
  naming where it stopped, and the caller parks until that counter is the finished
  sentinel, one level up per call. A call behind a predicate records that predicate in
  a bit of the call site's own state, so a lane coming back from a park re-enters the
  call exactly when it was made rather than re-testing a register. What is refused by
  name is a callee whose body is not in this file, one that calls itself, directly or
  through a chain (a cycle has no finite state struct, and a depth limit would be a
  number this project would have to invent), and a return list with more than one
  slot, which the state has one field to hold.
- A 16-byte value is `weft_u128`, a pair of 64-bit halves read and written in one
  access, so `ld`/`st`.b128 and the tensor-map load move all sixteen bytes: measured
  byte-exact across four threads, one sixteen-byte value each. `atom.cas.b128` compares
  and writes all sixteen as one value under the same lock the 64-bit pair takes, so the
  compare covers every byte: an expected value that differs only in the high half
  writes nothing, which is the case a compare on a prefix would get wrong, and a swap
  moves both halves together. A 64-bit or 128-bit atomic in the shared window is refused
  by name, because neither width has an instruction there and the lock orders the heap.

- `multimem` is a multicast address and this runtime binds one replica, so a reduce
  across the replicas is the value itself under every operation and a multicast store
  writes the one copy there is. `multimem.red` is refused by name: an atomic operation
  at that address would become a plain store.

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
  A 64-bit max or min whose result nobody reads is cheaper: the lock goes around the warp
  instead of inside it, and a value the word already holds skips it entirely, so a
  reduction onto one constant went from 481 ms to 28 ms at 1,048,576 lanes; a reduction
  whose value keeps changing still takes one lock per warp.
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
