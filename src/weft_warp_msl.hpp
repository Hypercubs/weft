#pragma once

// The warp side of the Metal runtime: one Metal thread runs one whole 32-lane
// warp.
//
// The per-lane target gives each thread one lane, which is why a tensor-core
// fragment has no form there, `mma` reads registers other lanes hold, and a
// lane cannot read a peer's register without a shuffle that a diverging state
// machine is not allowed to make. In this target the 32 lanes are data inside
// one thread: a peer's register is `r[l]`, the mask is `am[l]`, and nothing
// crosses a hardware lane at all.
//
// What that costs: 32 times fewer threads (one per warp, not per lane), and the
// warp's register file lives in thread memory, `r[24][32]` is 3 KiB before the
// floats and the 64-bit words are counted. So this is the target for the kernels
// that *need* a warp, not the one that is fastest for the ones that do not.
static const char* kWWeftWarpMsl = R"WEFWARP(
#include <metal_stdlib>
using namespace metal;

namespace weft {

// Lanes in a warp. The dispatch is one thread per warp, so the launcher divides
// the launch by this and the kernel multiplies it back.
static constant uint32_t kWarpLanes = 32u;

// One warp's view of the launch. `lid0` is the first lane this thread owns, so
// lane `l` of the warp is global thread `ctx.lid0 + l`, which is the whole of
// what `%tid.x` means in the warp backend.
struct WCtx {
  uint32_t lid0;
  uint32_t tg_size;  // lanes in this threadgroup
  uint32_t tgid;
  uint32_t grid;     // lanes in the launch
  device uint8_t* b[32];
  device uint64_t* params;
  threadgroup uint32_t* red;  // this threadgroup's reduction slots
  // The words a 64-bit read-modify-write locks on. The launcher zeroes them
  // before every dispatch, so a lock left behind by a finished launch is not
  // seen by the next one.
  device uint32_t* locks;
};

// How a warp left the body: every lane finished, or a barrier ended the segment
// and the body has to be called again. The kernel loops on this, which is what
// the CPU scheduler does with the same two answers.
static constant uint32_t kWarpDone = 0xFFFFFFFFu;
static constant uint32_t kWarpBarrier = 0xFFFFFFFEu;

//
// One thread runs the whole warp, so a slot another lane writes is a slot this
// thread reads: a barrier inside the warp is satisfied the moment it is reached,
// and the reduction barrier's publish slots are threadgroup memory rather than
// anything that has to cross a hardware lane.
enum RedMode { kRedPopc = 0, kRedAnd = 1, kRedOr = 2 };

static inline uint32_t red_fold_bits(int mode, uint32_t acc, uint32_t v) {
  if (mode == kRedPopc) return acc + v;
  if (mode == kRedAnd) return acc & (v ? 1u : 0u);
  return acc | (v ? 1u : 0u);
}
static inline uint32_t red_empty(int mode) { return mode == kRedAnd ? 1u : 0u; }

// `live_threads` is the per-lane header's, so the fold stops at the same count
// on both targets.

static inline void red_publish(thread WCtx& c, uint32_t lane, uint32_t v) {
  const uint32_t at = c.lid0 + lane;
  if (at < live_threads(c.tgid, c.tg_size, c.grid)) c.red[at] = v;
}
static inline uint32_t red_warp(thread WCtx& c, int mode) {
  const uint32_t live = live_threads(c.tgid, c.tg_size, c.grid);
  uint32_t n = live > c.lid0 ? live - c.lid0 : 0u;
  if (n > 32u) n = 32u;
  uint32_t acc = red_empty(mode);
  for (uint32_t i = 0; i < n; i++) acc = red_fold_bits(mode, acc, c.red[c.lid0 + i]);
  if ((c.lid0 & 31u) == 0u) c.red[c.tg_size + (c.lid0 >> 5)] = acc;
  return acc;
}
static inline uint32_t red_group(thread WCtx& c, int mode) {
  const uint32_t nw = (live_threads(c.tgid, c.tg_size, c.grid) + 31u) / 32u;
  uint32_t acc = red_empty(mode);
  for (uint32_t i = 0; i < nw; i++) acc = red_fold_bits(mode, acc, c.red[c.tg_size + i]);
  return acc;
}

static inline device uint8_t* buffer(thread const WCtx& c, int i) { return c.b[i]; }
// A binding as a number, the way the per-lane runtime hands one out: a warp's
// address arithmetic is the same whether the lanes are threads or data.
static inline uint64_t buf_addr(thread const WCtx& c, int i) {
  return (uint64_t)(uintptr_t)c.b[i];
}
static inline uint64_t param(thread const WCtx& c, int i) { return c.params[i]; }

// The 64-bit read-modify-write helpers, the lock they take and the pair they
// move live in src/weft_msl.hpp: that preamble is emitted first, and the
// per-lane back end needs the same ones.

}  // namespace weft
)WEFWARP";
