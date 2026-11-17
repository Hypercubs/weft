#pragma once

static const char* kWWeftWarpMsl = R"WEFWARP(
#include <metal_stdlib>
using namespace metal;

namespace weft {

// Lanes in a warp.
static constant uint32_t kWarpLanes = 32u;

// One warp's view of the launch.
struct WCtx {
  uint32_t lid0;
  uint32_t tg_size;  // lanes in this threadgroup
  uint32_t tgid;
  uint32_t grid;     // lanes in the launch
  device uint8_t* b[32];
  device uint64_t* params;
  threadgroup uint32_t* red;  // this threadgroup's reduction slots
  device uint32_t* locks;
  device uint32_t* fault;  // the word a stopped lane reports through
};

// and the body has to be called again.
static constant uint32_t kWarpDone = 0xFFFFFFFFu;
static constant uint32_t kWarpBarrier = 0xFFFFFFFEu;

// anything that has to cross a hardware lane.
enum RedMode { kRedPopc = 0, kRedAnd = 1, kRedOr = 2 };

static inline uint32_t red_fold_bits(int mode, uint32_t acc, uint32_t v) {
  if (mode == kRedPopc) return acc + v;
  if (mode == kRedAnd) return acc & (v ? 1u : 0u);
  return acc | (v ? 1u : 0u);
}
static inline uint32_t red_empty(int mode) { return mode == kRedAnd ? 1u : 0u; }

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
// address arithmetic is the same whether the lanes are threads or data.
static inline uint64_t buf_addr(thread const WCtx& c, int i) {
  return (uint64_t)(uintptr_t)c.b[i];
}
static inline uint64_t param(thread const WCtx& c, int i) { return c.params[i]; }

// per-lane back end needs the same ones.

}  // namespace weft
)WEFWARP";
