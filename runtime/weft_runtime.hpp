// weft — the GPU's execution model, implemented on CPU cores.
//
// A Metal kernel sees: a grid of threads, grouped into threadgroups, each group
// sharing a block of fast memory, with a barrier that all threads in the group
// must reach before any of them continues. That is the whole contract.
//
// Here it is on a CPU: threadgroups are handed to worker threads, threads inside
// a group are stackful fibers (so a barrier can suspend anywhere, including deep
// inside a function call, which coroutines cannot do), shared memory is a bump
// arena per group, and the barrier is a fiber switch of about 40 cycles.
//
// Nothing on the per-thread path touches thread-local storage: on macOS that
// compiles to a call, and a million threads times eight barriers is a million
// calls. Worker state and buffer pointers ride in Ctx instead.
#pragma once

// The module ABI a translated file implements: the kernel table and the entry
// points a caller uses (see include/weft/weft.h).
#include <weft/weft.h>

#include <atomic>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace weft {

struct WorkerState;

// Per-threadgroup shared-memory bookkeeping. Threads in one group run on one
// worker thread and only hand off at barriers, so this needs no locking.
struct SlotState {
  size_t off[64];
  size_t bump;
  int count;
};

// Field order is not arbitrary: the dispatcher touches `lid`, `tg_size`, `pass`,
// `warp` and `fold` for every lane of every sweep, and those five share the first
// cache line so a sweep reads one line per lane instead of two.
struct Ctx {
  uint32_t lid = 0;       // thread_position_in_threadgroup
  uint32_t tg_size = 0;   // threads_per_threadgroup
  uint32_t pass = 0;      // dispatcher pass number, for shuffle double-buffering
  uint32_t* warp = nullptr;  // per-lane scratch for shfl
  // A warp holds 32 lanes of one threadgroup on one worker thread, so the scan
  // and fold trees those lanes run are the same tree: the first lane to reach a
  // call site computes it and the other 31 read its answer. `fold` points at this
  // lane's warp's slot — [0] site, [1] pass, [2] base, [3..34] the tree.
  uint32_t* fold = nullptr;
  unsigned char* warp16 = nullptr;  // per-lane scratch for values wider than a word
  uint32_t* red = nullptr;   // per-threadgroup scratch for bar.red (tg + 32 slots)
  uint32_t gid = 0;       // thread_position_in_grid
  uint32_t tgid = 0;      // threadgroup_position_in_grid
  uint32_t grid = 0;      // threads_per_grid
  const unsigned char* shared = nullptr;
  SlotState* slots = nullptr;
  void* const* bufs = nullptr;
  WorkerState* ws = nullptr;
  int slot = 0;
};

// Words in one warp's scan cache: the three key words and the 32 results.
constexpr uint32_t kFoldWords = 35u;

using Body = void (*)(Ctx&);

// Kernels that only index by thread_position_in_grid also get a loop form:
// the body is emitted inside a `for` over a contiguous range, so the buffer
// pointers hoist out and the compiler can vectorise. That form is what keeps
// elementwise kernels close to the GPU; the per-thread function is the
// fallback when a kernel needs threadgroup-relative indices.
using FlatBody = void (*)(Ctx&, uint32_t begin, uint32_t end);

// A resumable kernel: the thread's whole state lives in a caller-owned blob, so
// a barrier is just "save the resume point and return" — no new stack, no
// context switch. The dispatcher calls every thread once per barrier segment.
// `pc == kThreadDone` means the thread finished.
using StepFn = void (*)(Ctx&, void* state, void* const* bufs, unsigned char* sh);

// The same state machine with its lane loop inside the function. A barrier
// segment then costs one direct call per threadgroup instead of an indirect one
// per lane, which is the whole of the cost between two barriers. It also returns
// how many lanes are still running, so the dispatcher does not have to walk the
// states again to find out.
using GroupStepFn = uint32_t (*)(Ctx* ctxs, unsigned char* states, size_t stride, uint32_t live,
                                 uint32_t pass, void* const* bufs, unsigned char* sh);
constexpr uint32_t kThreadDone = 0xFFFFFFFFu;

// A warp executed as one vector: the 32 lanes of a warp advance together, and a
// shuffle is a gather inside the register file instead of 32 suspensions. Each
// lane keeps its own program counter, so divergence is handled by running the
// lanes that share a program counter as one group; when two groups reach the
// same instruction they naturally merge again.
//
// The function runs until every lane has finished (`kWarpDone`) or the warp
// reaches a barrier (`kWarpBarrier`), in which case the scheduler re-enters it
// once the whole threadgroup has arrived.
struct WCtx {
  uint32_t lid0 = 0;      // index of this warp's first lane in the threadgroup
  uint32_t tg_size = 0;
  uint32_t tgid = 0;
  uint32_t grid = 0;
  void* const* bufs = nullptr;
  unsigned char* shared = nullptr;
  uint32_t* red = nullptr;  // per-threadgroup scratch for bar.red
};

constexpr uint32_t kWarpDone = 0;
constexpr uint32_t kWarpBarrier = 1;

using WarpFn = uint32_t (*)(WCtx&, void* state);

// A kernel body that can suspend at a barrier and be resumed later. The C++
// compiler keeps the frame alive, locals included, so a Metal kernel with
// arbitrary locals can be resumable too — this is what replaces fibers.
struct Task {
  struct promise_type {
    Task get_return_object() { return Task{handle_type::from_promise(*this)}; }
    std::suspend_never initial_suspend() noexcept { return {}; }  // run to 1st barrier
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() { __builtin_trap(); }
    static void* operator new(std::size_t n);
    static void operator delete(void* p, std::size_t n) noexcept;
  };
  using handle_type = std::coroutine_handle<promise_type>;

  explicit Task(handle_type h) : h_(h) {}
  Task(Task&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;
  ~Task() {
    if (h_) h_.destroy();
  }

  bool done() const { return !h_ || h_.done(); }
  void resume() {
    if (h_ && !h_.done()) h_.resume();
  }
  // The same call for a caller that has just checked `done()` — the dispatcher
  // does, once per lane per pass, and a 1M-lane scan runs it five million times.
  void resume_ready() { h_.resume(); }

 private:
  handle_type h_;
};

struct BarrierAwaitable {
  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) const noexcept {}
  void await_resume() const noexcept {}
};

inline BarrierAwaitable barrier_async() { return {}; }

// Warp shuffles. A shuffle is a rendezvous among 32 lanes, and the only
// rendezvous this runtime has is "hand the worker back and resume later" — so a
// shuffle suspends exactly like a barrier and costs the same.
enum ShflMode { kShflIdx = 0, kShflUp = 1, kShflDown = 2, kShflBfly = 3 };

// A rendezvous slot is one word, and a value narrower than a word only fills
// part of it: the copy is `sizeof(T)` bytes, so a half or a char travels
// zero-extended and comes back whole.
template <class T>
inline uint32_t to_bits(T v) {
  static_assert(sizeof(T) <= 4, "a value wider than a word takes the wide slot");
  uint32_t b = 0;
  std::memcpy(&b, &v, sizeof(T));
  return b;
}

template <class T>
inline T from_bits(uint32_t b) {
  static_assert(sizeof(T) <= 4, "a value wider than a word takes the wide slot");
  T v{};
  std::memcpy(&v, &b, sizeof(T));
  return v;
}

// Lanes do not stop together. One dispatcher pass moves every lane on by
// exactly one shuffle: it publishes, returns, and reads the peer on the pass
// after. So the slots are double-buffered by *pass* — a lane publishes into the
// half for its own pass and reads back the half it published into, so the next
// pass writing the other half cannot corrupt a lane that has not caught up.
inline const uint32_t* shfl_pub_buf(const Ctx& c) {
  return (const uint32_t*)c.warp + (size_t)(c.pass & 1u) * c.tg_size;
}
inline const uint32_t* shfl_read_buf(const Ctx& c) {
  return (const uint32_t*)c.warp + (size_t)((c.pass - 1u) & 1u) * c.tg_size;
}

// A warp vote: which lanes of this warp published a non-zero value. Every lane
// asks the same question in the same pass, so the first lane to ask computes it
// and the rest read the answer (32 loads a warp instead of 32 a lane). Defined
// in the runtime .cpp, where the worker's scratch is a complete type.
uint32_t vote_ballot(Ctx& c);

// The peer a lane will read from. Metal's shuffles are full-warp, and a lane
// whose source falls outside it keeps its own value — which is what PTX does
// with the clamps CUDA emits (0x1f for down, 0 for up).
//
// `width` is the size of the group the shuffle is relative to: 32 for a warp,
// 4 for a quad. A group is a run of consecutive lanes inside a warp, so the same
// arithmetic works for both — only the base and the mask change.
inline uint32_t shfl_peer_lane(const Ctx& c, uint32_t b, int mode, uint32_t width = 32u) {
  const uint32_t base = c.lid & ~(width - 1u);
  const uint32_t lane = c.lid - base;
  const uint32_t top = width - 1u;
  uint32_t src_lane;
  if (mode == kShflIdx)
    src_lane = b & top;
  else if (mode == kShflBfly)
    src_lane = lane ^ (b & top);
  else if (mode == kShflUp)
    src_lane = (lane >= b) ? lane - b : lane;
  else
    src_lane = (lane + b < width) ? lane + b : lane;
  return base + src_lane;
}

inline uint32_t shfl_peer(const Ctx& c, uint32_t b, int mode, uint32_t width = 32u) {
  return shfl_read_buf(c)[shfl_peer_lane(c, b, mode, width)];
}

// A rendezvous slot for a value wider than a word: 16 bytes takes every type
// simd_shuffle accepts — a float4, a long2, a double2 — so one of those is still
// a single suspension. The slots are double-buffered by pass exactly like the
// word-wide ones, and live in their own scratch so that indexing the word-wide
// array (every lane reading every peer) still means what it says.
constexpr size_t kWideSlotBytes = 16;

// A partial threadgroup's lanes past `live` never run, so their shuffle slots
// still hold what the previous group on this worker published. Zero them once
// per group: without that, the same kernel at the same grid answers differently
// for every worker count, and the ragged last group's reduction depends on what
// ran before it. A dead lane reads as 0, which is also the value that makes a
// sum over the lanes that do exist the sum of exactly those lanes.
// The most lanes a threadgroup can have; the scratch below is sized for it, so
// the wide half always starts at the same offset.
constexpr uint32_t kMaxLanes = 1024;

inline void clear_dead_lanes(unsigned char* warp, uint32_t live, uint32_t tg) {
  if (live >= tg) return;
  uint32_t* w = reinterpret_cast<uint32_t*>(warp);
  unsigned char* wide = warp + (size_t)2 * kMaxLanes * sizeof(uint32_t);
  for (size_t half = 0; half < 2; half++) {
    std::memset(w + half * tg + live, 0, (size_t)(tg - live) * sizeof(uint32_t));
    std::memset(wide + (half * tg + live) * kWideSlotBytes, 0,
                (size_t)(tg - live) * kWideSlotBytes);
  }
}

inline unsigned char* wide_pub_slot(const Ctx& c) {
  return c.warp16 + ((size_t)(c.pass & 1u) * c.tg_size + c.lid) * kWideSlotBytes;
}
inline const unsigned char* wide_read_slot(const Ctx& c, uint32_t lane) {
  return c.warp16 + ((size_t)((c.pass - 1u) & 1u) * c.tg_size + lane) * kWideSlotBytes;
}

// ---- the two halves of a rendezvous -----------------------------------------
//
// A coroutine spells a rendezvous as one suspension: publish before it, read
// after. A state machine has to name the boundary, so the same two steps are
// separate calls here — publish at the end of one segment, take at the start of
// the next. Both shapes go through these, so they cannot disagree about what a
// rendezvous carries.
template <class T>
inline void shfl_publish(Ctx& c, T value) {
  if constexpr (sizeof(T) <= 4) {
    ((uint32_t*)shfl_pub_buf(c))[c.lid] = to_bits(value);
  } else {
    static_assert(sizeof(T) <= kWideSlotBytes, "a shuffle is at most 16 bytes wide");
    std::memcpy(wide_pub_slot(c), &value, sizeof(T));
  }
}

// The peer's published word, or the fill value when the peer falls outside the
// group (which is what `simd_shuffle_and_fill_*` promises).
template <class T>
inline T shfl_take(Ctx& c, uint32_t b, int mode, uint32_t width = 32u) {
  if constexpr (sizeof(T) <= 4) {
    return from_bits<T>(shfl_peer(c, b, mode, width));
  } else {
    T v{};
    std::memcpy(&v, wide_read_slot(c, shfl_peer_lane(c, b, mode, width)), sizeof(T));
    return v;
  }
}

template <class T>
inline T shfl_take_fill(Ctx& c, uint32_t b, int mode, T fill, uint32_t width = 32u) {
  const uint32_t lane = c.lid & (width - 1u);
  const bool in = (mode == kShflUp) ? (lane >= b) : (lane + b < width);
  return in ? shfl_take<T>(c, b, mode, width) : fill;
}

template <class T>
struct ShflAwaitable {
  Ctx* c;
  T value;
  uint32_t b;
  int mode;
  uint32_t width = 32u;

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { shfl_publish(*c, value); }
  T await_resume() const noexcept { return shfl_take<T>(*c, b, mode, width); }
};

template <class T>
inline ShflAwaitable<T> shfl(Ctx& c, T value, uint32_t b, int mode) {
  return ShflAwaitable<T>{&c, value, b, mode, 32u};
}

// The same rendezvous inside a smaller group of a warp: Metal's quad (width 4).
template <class T, std::enable_if_t<sizeof(T) <= 4, int> = 0>
inline ShflAwaitable<T> grp_shfl(Ctx& c, T value, uint32_t b, int mode, uint32_t width) {
  return ShflAwaitable<T>{&c, value, b, mode, width};
}

template <class T>
struct WideShflAwaitable {
  Ctx* c;
  T value;
  T result;
  uint32_t b;
  int mode;
  uint32_t width;

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { shfl_publish(*c, value); }
  T await_resume() noexcept {
    result = shfl_take<T>(*c, b, mode, width);
    return result;
  }
};

// A value that is not one word takes the wide slot: same rendezvous, same
// semantics, sixteen bytes of payload instead of four.
template <class T, std::enable_if_t<(sizeof(T) > 4), int> = 0>
inline WideShflAwaitable<T> grp_shfl(Ctx& c, T value, uint32_t b, int mode, uint32_t width) {
  static_assert(sizeof(T) <= kWideSlotBytes, "a shuffle is at most 16 bytes wide");
  return WideShflAwaitable<T>{&c, value, T{}, b, mode, width};
}

// simd_shuffle_and_fill_up/down: the same shuffle, but a lane whose source is
// outside the warp takes the fill value instead of its own. Every lane still
// publishes — the select happens when the lane is resumed, not before the
// rendezvous — because a lane that skipped the publish would leave a stale slot
// for the peer that reads it.
template <class T>
struct ShflFillAwaitable {
  Ctx* c;
  T value;
  uint32_t b;
  int mode;
  T fill;
  bool in_range;

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { shfl_publish(*c, value); }
  T await_resume() const noexcept { return in_range ? shfl_take<T>(*c, b, mode) : fill; }
};

// simd_prefix_inclusive_sum / exclusive_sum: a warp-wide scalar scan. One
// rendezvous is enough — the published array holds every lane's input, so each
// lane folds the slots up to its own index after it is resumed. The GPU does it
// with five shuffles; this does it with one suspension and 32 loads, which is
// cheaper here because a suspension is the expensive part.
template <class T>
struct PrefixAwaitable {
  Ctx* c;
  T value;
  bool inclusive;
  bool product = false;
  uint32_t width = 32u;
  uint32_t site = 0u;  // the emitted call site, 0 for "do not cache"

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { shfl_publish(*c, value); }
  // One implementation, both shapes: a coroutine resumes here, a state machine
  // calls it once the next pass has begun.
  T await_resume() const noexcept { return take(); }
  T take() const noexcept {
    const uint32_t base = c->lid & ~(width - 1u);
    const uint32_t lane = c->lid - base;
    uint32_t* const slot = c->fold;
    // Every lane of a warp walks the same tree over the same published words, so
    // one lane computes it and the rest read their entry. The key has to name the
    // call site as well as the pass: two lanes of one warp can be at different
    // scans in the same pass, and a cached tree must never answer for another one.
    if (slot && site != 0u && slot[0] == site && slot[1] == c->pass && slot[2] == base) {
      // The cache holds the whole tree, so the inclusive/exclusive choice is the
      // caller's: exclusive reads the entry before this lane's.
      if (inclusive) return lane < width ? from_bits<T>(slot[3 + lane]) : T(0);
      return lane ? from_bits<T>(slot[3 + lane - 1u]) : T(product ? 1 : 0);
    }
    const uint32_t* slots = shfl_read_buf(*c) + base;
    // The scan is run in the doubling (Kogge-Stone) order the hardware uses, not
    // as a running sum: for floats the two orders round differently, and the
    // point is to land on the same bits as the GPU. Integer addition is
    // associative, so this is exact for the integer types either way.
    T v[32];
    for (uint32_t i = 0; i < width && i < 32u; i++) v[i] = from_bits<T>(slots[i]);
    for (uint32_t d = 1; d < width; d <<= 1)
      for (uint32_t i = width - 1u; i >= d; i--)
        v[i] = product ? T(v[i] * v[i - d]) : T(v[i] + v[i - d]);
    if (slot && site != 0u) {
      for (uint32_t i = 0; i < width && i < 32u; i++) slot[3 + i] = to_bits<T>(v[i]);
      slot[0] = site;
      slot[1] = c->pass;
      slot[2] = base;
    }
    if (inclusive) return v[lane];
    return lane ? v[lane - 1u] : T(product ? 1 : 0);
  }
};

template <class T>
inline T prefix_take(Ctx& c, bool inclusive, bool product, uint32_t width, uint32_t site) {
  return PrefixAwaitable<T>{&c, T{}, inclusive, product, width, site}.take();
}

template <class T>
inline PrefixAwaitable<T> prefix(Ctx& c, T value, bool inclusive, bool product = false,
                                 uint32_t width = 32u, uint32_t site = 0u) {
  return PrefixAwaitable<T>{&c, value, inclusive, product, width, site};
}

// ---- bar.red: a fold over the whole threadgroup ---------------------------------
//
// `bar.red.popc/or/and` is a barrier whose result is a reduction across every
// thread in the group — `__syncthreads_count` and its two cousins. A barrier is
// the only rendezvous this runtime has, so this is two of them: publish my value
// and arrive, then the group has everything and each warp folds its own 32 slots
// into one total, arrive again, and the per-warp totals are folded. Every thread
// ends up with the same answer, which is what the instruction promises.
//
// Two rendezvous instead of one because the alternative is each thread reading
// all tg_size slots, and a barrier is cheaper than 256 loads per thread.
enum RedMode { kRedPopc = 0, kRedAnd = 1, kRedOr = 2 };

inline uint32_t red_fold_bits(int mode, uint32_t acc, uint32_t v) {
  if (mode == kRedPopc) return acc + v;
  if (mode == kRedAnd) return acc & (v ? 1u : 0u);
  return acc | (v ? 1u : 0u);
}

inline uint32_t red_empty(int mode) { return mode == kRedAnd ? 1u : 0u; }

// The last threadgroup of a grid can be short: 1000 threads at 96 a group leaves
// 40 in the last one. The slots of the threads that do not exist hold whatever
// the previous group left there, so the fold has to stop at the live count —
// which is derivable from the launch geometry, no extra state.
inline uint32_t live_threads(uint32_t tgid, uint32_t tg_size, uint32_t grid) {
  const uint64_t base = (uint64_t)tgid * tg_size;
  if (base >= grid || tg_size == 0) return 0;
  const uint64_t n = (uint64_t)grid - base;
  return n < tg_size ? (uint32_t)n : tg_size;
}

inline void red_publish(Ctx& c, uint32_t v) { c.red[c.lid] = v; }
inline void red_publish(WCtx& c, uint32_t lane, uint32_t v) {
  const uint32_t at = c.lid0 + lane;
  if (at < live_threads(c.tgid, c.tg_size, c.grid)) c.red[at] = v;
}

inline uint32_t red_warp(Ctx& c, int mode) {
  const uint32_t live = live_threads(c.tgid, c.tg_size, c.grid);
  const uint32_t base = c.lid & ~31u;
  uint32_t n = live > base ? live - base : 0u;
  if (n > 32u) n = 32u;
  uint32_t acc = red_empty(mode);
  for (uint32_t i = 0; i < n; i++) acc = red_fold_bits(mode, acc, c.red[base + i]);
  if ((c.lid & 31u) == 0u) c.red[c.tg_size + (c.lid >> 5)] = acc;
  return acc;
}

inline uint32_t red_warp(WCtx& c, int mode) {
  const uint32_t live = live_threads(c.tgid, c.tg_size, c.grid);
  uint32_t n = live > c.lid0 ? live - c.lid0 : 0u;
  if (n > 32u) n = 32u;
  uint32_t acc = red_empty(mode);
  for (uint32_t i = 0; i < n; i++) acc = red_fold_bits(mode, acc, c.red[c.lid0 + i]);
  if ((c.lid0 & 31u) == 0u) c.red[c.tg_size + (c.lid0 >> 5)] = acc;
  return acc;
}

// The warp-vector backend's WCtx carries the same geometry, so both overloads
// are the same fold over the same slots.
inline uint32_t red_group(Ctx& c, int mode) {
  const uint32_t nw = (live_threads(c.tgid, c.tg_size, c.grid) + 31u) / 32u;
  uint32_t acc = red_empty(mode);
  for (uint32_t i = 0; i < nw; i++) acc = red_fold_bits(mode, acc, c.red[c.tg_size + i]);
  return acc;
}

inline uint32_t red_group(WCtx& c, int mode) {
  const uint32_t nw = (live_threads(c.tgid, c.tg_size, c.grid) + 31u) / 32u;
  uint32_t acc = red_empty(mode);
  for (uint32_t i = 0; i < nw; i++) acc = red_fold_bits(mode, acc, c.red[c.tg_size + i]);
  return acc;
}

template <class T, std::enable_if_t<sizeof(T) <= 4, int> = 0>
inline ShflFillAwaitable<T> shfl_fill(Ctx& c, T value, uint32_t b, int mode, T fill,
                                     uint32_t width = 32u) {
  const uint32_t lane = c.lid & (width - 1u);
  const bool in = (mode == kShflUp) ? (lane >= b) : (lane + b < width);
  return ShflFillAwaitable<T>{&c, value, b, mode, fill, in};
}

// The wide form of a filling shuffle: a lane whose source is outside the group
// answers with the fill value instead of its peer's slot.
template <class T>
struct WideShflFillAwaitable {
  Ctx* c;
  T value;
  T fill;
  T result;
  uint32_t b;
  int mode;
  uint32_t width;
  bool in_range;

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { shfl_publish(*c, value); }
  T await_resume() noexcept {
    if (!in_range) return fill;
    result = shfl_take<T>(*c, b, mode, width);
    return result;
  }
};

template <class T, std::enable_if_t<(sizeof(T) > 4), int> = 0>
inline WideShflFillAwaitable<T> shfl_fill(Ctx& c, T value, uint32_t b, int mode, T fill,
                                          uint32_t width = 32u) {
  static_assert(sizeof(T) <= kWideSlotBytes, "a shuffle is at most 16 bytes wide");
  const uint32_t lane = c.lid & (width - 1u);
  const bool in = (mode == kShflUp) ? (lane >= b) : (lane + b < width);
  return WideShflFillAwaitable<T>{&c, value, fill, T{}, b, mode, width, in};
}

// ---- group reductions -------------------------------------------------------
//
// simd_sum / simd_max / quad_sum and their cousins are a fold over the group's
// published values, done in the butterfly order the hardware uses: five minutes
// of xor-shuffles becomes, here, one rendezvous and five folds. The order is the
// same one, so a float sum lands on the same bits.
enum GrpReduce { kGrpSum = 0, kGrpMin = 1, kGrpMax = 2, kGrpProd = 3 };

template <class T>
inline T grp_fold(int mode, T a, T b) {
  if (mode == kGrpSum) return T(a + b);
  if (mode == kGrpProd) return T(a * b);
  if (mode == kGrpMin) return b < a ? b : a;
  return a < b ? b : a;
}

template <class T>
struct ReduceAwaitable {
  Ctx* c;
  T value;
  int mode;
  uint32_t width;
  uint32_t site = 0u;  // the emitted call site, 0 for "do not cache"

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { shfl_publish(*c, value); }
  T await_resume() const noexcept { return take(); }
  T take() const noexcept {
    const uint32_t base = c->lid & ~(width - 1u);
    const uint32_t* slots = shfl_read_buf(*c) + base;
    uint32_t* const slot = c->fold;
    // Same cache as a scan: the butterfly is the same tree for all 32 lanes, and
    // the weights it folds are the warp's published words, not this lane's.
    if (slot && site != 0u && slot[0] == site && slot[1] == c->pass && slot[2] == base) {
      const uint32_t at = c->lid - base;
      return at < width && at < 32u ? from_bits<T>(slot[3 + at]) : T(0);
    }
    // The hardware's fold is a butterfly: every lane folds its partner's
    // *running* value, step by step. Reading the published slots once and
    // folding them together would be a different tree — and for floats a
    // different answer — so the same three-step shape is kept here, one array
    // for the values and one for the step's results.
    T v[32], next[32];
    for (uint32_t i = 0; i < width && i < 32u; i++) v[i] = from_bits<T>(slots[i]);
    for (uint32_t d = 1; d < width; d <<= 1) {
      for (uint32_t i = 0; i < width && i < 32u; i++)
        next[i] = grp_fold<T>(mode, v[i], v[i ^ d]);
      for (uint32_t i = 0; i < width && i < 32u; i++) v[i] = next[i];
    }
    if (slot && site != 0u) {
      for (uint32_t i = 0; i < width && i < 32u; i++) slot[3 + i] = to_bits<T>(v[i]);
      slot[0] = site;
      slot[1] = c->pass;
      slot[2] = base;
    }
    return v[c->lid - base];
  }
};

template <class T>
inline T reduce_take(Ctx& c, int mode, uint32_t width, uint32_t site) {
  return ReduceAwaitable<T>{&c, T{}, mode, width, site}.take();
}

template <class T>
inline ReduceAwaitable<T> grp_reduce(Ctx& c, T value, int mode, uint32_t width = 32u,
                                     uint32_t site = 0u) {
  return ReduceAwaitable<T>{&c, value, mode, width, site};
}

// simd_ballot / simd_all / simd_any: the same rendezvous, one bit wide. The
// answer is the same in every lane, so the warp computes it once per pass.
struct VoteAwaitable {
  Ctx* c;
  bool value;
  int mode;  // 0 ballot, 1 all, 2 any
  uint32_t width = 32u;

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept { shfl_publish(*c, value); }
  uint32_t await_resume() const noexcept { return take(); }
  uint32_t take() const noexcept {
    uint32_t m = vote_ballot(*c);
    uint32_t group_mask = 0xFFFFFFFFu;
    if (width < 32u) {
      // A narrow vote is reported inside its own group, not at its position in
      // the warp: quad_ballot gives every quad the same four-bit answer.
      const uint32_t base = c->lid & ~(width - 1u);
      group_mask = (1u << width) - 1u;
      m = (m >> base) & group_mask;
    }
    if (mode == 1) return m == group_mask ? 1u : 0u;
    if (mode == 2) return m != 0u ? 1u : 0u;
    return m;
  }
};

inline uint32_t vote_take(Ctx& c, int mode, uint32_t width) {
  return VoteAwaitable{&c, false, mode, width}.take();
}

inline VoteAwaitable vote(Ctx& c, bool value, int mode) { return VoteAwaitable{&c, value, mode, 32u}; }

// The same vote inside a group: the mask carries the whole warp, and a quad only
// owns its four bits of it, which is what quad_ballot hands back.
inline VoteAwaitable grp_vote(Ctx& c, bool value, int mode, uint32_t width) {
  return VoteAwaitable{&c, value, mode, width};
}

void* coro_alloc(std::size_t n);
void coro_arena_reset();

// Both of these used to trap, which says nothing to anyone reading a terminal.
// They name what is wrong and stop, once, even when several lanes reach it in
// the same instruction.
[[noreturn]] void shared_overflow(std::size_t need, std::size_t cap);
[[noreturn]] void shared_too_many();

using TaskFn = Task (*)(Ctx&);

// sync == the kernel declares threadgroup memory or calls a barrier.
struct KernelDef {
  const char* name;
  Body body;
  int nargs;
  // Whether the kernel synchronises (shared memory, a barrier or a shuffle).
  // Module metadata: `weft-gen --describe` reports it, and nothing dispatches on
  // it — the fiber backend that used to was deleted, because every kernel with a
  // barrier is routed to its coroutine or state-machine body first.
  bool sync;
  FlatBody flat;    // nullptr when the kernel cannot be flattened
  StepFn step;      // resumable via a state blob (PTX)
  size_t state_size;
  TaskFn coro;      // resumable via a coroutine frame (Metal with barriers)
  WarpFn warp;        // 32 lanes at a time (PTX, when the body allows it)
  size_t warp_size;   // bytes of state for one warp
  GroupStepFn group_step;  // set when the module emits one
};

// ---- textures ----
//
// A texture is an image the launcher fills before the run, so the GPU and the
// CPU sides read the same texels. Only the shapes a compute kernel uses are here:
// no mip levels. It is 2D, a 2D *array*, or 3D, and the last two differ only in
// what the outermost index means — a slice or a plane — so one number and one
// layout serve both, which is `MTLTextureType2DArray`'s layout and
// `MTLTextureType3D`'s. `read` is unnormalised and exact; `sample` filters.
struct Texture {
  // Writable, because a storage image is written: `texture2d<T, access::write>`
  // and `OpImageWrite` land in the same storage the reads come from, which is
  // what the GPU does with a texture flagged for shader writes.
  unsigned char* data = nullptr;
  uint32_t width = 0, height = 0;
  uint32_t elem = 4;    // bytes per texel (4 for float/uint, 2 for half)
  // The outermost extent: the slices of a 2D array, or a 3D texture's depth. One
  // number because the storage is the same either way — slice after slice, plane
  // after plane — so a 3D read is the same index arithmetic as a layered one and
  // neither needs its own path through this file.
  uint32_t layers = 1;
};

Texture texture(int index);
void set_texture(int index, void* data, uint32_t width, uint32_t height, uint32_t elem);
// The array form: `layers` slices, laid out one after another.
void set_texture_layers(int index, void* data, uint32_t width, uint32_t height, uint32_t elem,
                        uint32_t layers);

// A SPIR-V sampler is a value the host creates, not something a kernel declares,
// so its two meaningful choices come from the command line and arrive here.
struct Sampler {
  int filter = 0;  // kTexNearest / kTexLinear
  int addr = 0;    // kTexClampToEdge / kTexClampToZero / kTexRepeat
};

Sampler sampler(int index);
void set_sampler(int index, int filter, int addr);

// The image a sampler reads: `OpTypeImage` is a handle, so a kernel value is a
// pointer to the bound texture — the texture itself is host storage, exactly as
// it is on a GPU, where an image handle names a resource rather than copying it.
const Texture* texture_ptr(int index);

// `OpSampledImage`: an image combined with the sampler it will be read through.
struct SampledImage {
  const Texture* tex = nullptr;
  Sampler s{};
};

template <class T>
inline T tex_read(const Texture& t, uint32_t x, uint32_t y, uint32_t layer) {
  if (layer >= t.layers || x >= t.width || y >= t.height || !t.data) return T(0);
  T v;
  std::memcpy(&v, t.data + (((size_t)layer * t.height + y) * t.width + x) * t.elem, sizeof(T));
  return v;
}

template <class T>
inline T tex_read(const Texture& t, uint32_t x, uint32_t y) {
  return tex_read<T>(t, x, y, 0u);
}

// A write to a texture: `OpImageWrite` and `texture2d::write`. The launcher
// creates single-channel images, so the first channel is the texel and the rest
// of the vector has nowhere to go — the same thing the GPU does with an R32
// format.
// A texel is one value or a vector whose first member is channel 0, so the
// address of either is the address of the texel being written. That keeps this
// one template independent of the vector types, which do not exist yet here —
// this header is compiled before the preamble that defines them.
template <class T>
inline void tex_write(const Texture& t, uint32_t x, uint32_t y, uint32_t layer, T v) {
  if (layer >= t.layers || x >= t.width || y >= t.height || !t.data || t.elem > sizeof(T)) return;
  std::memcpy(t.data + (((size_t)layer * t.height + y) * t.width + x) * t.elem, &v, t.elem);
}

template <class T>
inline void tex_write(const Texture& t, uint32_t x, uint32_t y, T v) {
  tex_write<T>(t, x, y, 0u, v);
}

// Metal's sampler state, reduced to the two choices that change an answer.
enum TexFilter { kTexNearest = 0, kTexLinear = 1 };
enum TexAddr { kTexClampToEdge = 0, kTexClampToZero = 1, kTexRepeat = 2 };

inline uint32_t tex_wrap(int64_t i, uint32_t n, int addr) {
  if (i >= 0 && (uint32_t)i < n) return (uint32_t)i;
  if (addr == kTexRepeat) {
    const int64_t m = (int64_t)n;
    int64_t r = i % m;
    return (uint32_t)(r < 0 ? r + m : r);
  }
  // Clamp to zero answers with nothing; clamp to edge answers with the nearest
  // texel, which is what a filter footprint hanging off the image must do —
  // otherwise every sample near an edge is darkened towards black.
  if (addr == kTexClampToZero || n == 0) return (uint32_t)-1;
  return i < 0 ? 0u : n - 1u;
}

template <class T>
inline float tex_at_f(const Texture& t, int64_t x, int64_t y, uint32_t layer, int addr) {
  if (layer >= t.layers) return 0.0f;
  const uint32_t cx = tex_wrap(x, t.width, addr), cy = tex_wrap(y, t.height, addr);
  if (cx == (uint32_t)-1 || cy == (uint32_t)-1) return 0.0f;
  return (float)tex_read<T>(t, cx, cy, layer);
}

template <class T>
inline float tex_at_f(const Texture& t, int64_t x, int64_t y, int addr) {
  return tex_at_f<T>(t, x, y, 0u, addr);
}

// A `tex` instruction hands back four channels. The launcher binds one-channel
// images, so the first is the texel and the rest are the shape the hardware
// returns for an R32 format: nothing, nothing, one.
template <class T>
inline void tex_sample4(const Texture& t, float u, float v, uint32_t layer, int filter, int addr,
                        float out[4]) {
  out[0] = tex_sample<T>(t, u, v, layer, filter, addr);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

template <class T>
inline void tex_sample4(const Texture& t, float u, float v, int filter, int addr, float out[4]) {
  tex_sample4<T>(t, u, v, 0u, filter, addr, out);
}

// The 3D form of a `tex` instruction's four channels: one filtered sample of the
// plane the coordinate names, and the same (value, 0, 0, 1) shape.
template <class T>
inline void tex_sample4(const Texture& t, float u, float v, float w, int filter, int addr,
                        float out[4]) {
  out[0] = tex_sample<T>(t, u, v, w, filter, addr);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

// A face and a position on it, back into the direction a sampler takes. The six
// faces are +X, -X, +Y, -Y, +Z, -Z, and `s`/`t` run from -1 to 1 across the face
// with t pointing up, which is what a cube coordinate means. This inverts the rule
// in `tex_cube_uv` above, so a direction built here lands back on the same face at
// the same place — that inverse is what makes PTX's `(s, t, face)` form
// translatable at all. Both engines run this text, so both build the same bits.
inline void weft_cube_dir(uint32_t face, float s, float t, float& dx, float& dy, float& dz) {
  switch (face % 6u) {
    case 0: dx = 1.0f; dy = -t; dz = -s; break;
    case 1: dx = -1.0f; dy = -t; dz = s; break;
    case 2: dx = s; dy = 1.0f; dz = t; break;
    case 3: dx = s; dy = -1.0f; dz = -t; break;
    case 4: dx = s; dy = -t; dz = 1.0f; break;
    default: dx = -s; dy = -t; dz = -1.0f; break;
  }
}

// A cube `tex` instruction's four channels: a direction instead of a coordinate.
template <class T>
inline void tex_sample4_cube(const Texture& t, float dx, float dy, float dz, int filter, int addr,
                             float out[4]) {
  out[0] = tex_sample_cube<T>(t, dx, dy, dz, filter, addr);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

// `tld4`: the four texels around the coordinate, one per channel, instead of the
// blend a filter would produce. The footprint is the 2x2 block based at the texel
// under the coordinate.
template <class T>
inline void tex_gather4(const Texture& t, float u, float v, uint32_t layer, float out[4]) {
  const int64_t x0 = (int64_t)std::floor(u * (float)t.width - 0.5f);
  const int64_t y0 = (int64_t)std::floor(v * (float)t.height - 0.5f);
  auto at = [&](int64_t x, int64_t y) -> float {
    if (x < 0 || y < 0 || (uint64_t)x >= t.width || (uint64_t)y >= t.height || layer >= t.layers)
      return 0.0f;
    return (float)tex_read<T>(t, (uint32_t)x, (uint32_t)y, layer);
  };
  out[0] = at(x0, y0);
  out[1] = at(x0 + 1, y0);
  out[2] = at(x0, y0 + 1);
  out[3] = at(x0 + 1, y0 + 1);
}

template <class T>
inline void tex_gather4(const Texture& t, float u, float v, float out[4]) {
  tex_gather4<T>(t, u, v, 0u, out);
}

// A 3-D gather: the same 2x2 footprint, in the plane the coordinate names. A
// gather has no filter, so there is nothing to blend along z — the plane is the
// nearest one under w, which is the plane a nearest sample of the same coordinate
// reads. A footprint that leaves the binding answers with nothing, exactly as the
// 2-D form does.
template <class T>
inline void tex_gather4(const Texture& t, float u, float v, float w, float out[4]) {
  const int64_t x0 = (int64_t)std::floor(u * (float)t.width - 0.5f);
  const int64_t y0 = (int64_t)std::floor(v * (float)t.height - 0.5f);
  const int64_t z = (int64_t)std::floor(w * (float)t.layers);
  auto at = [&](int64_t x, int64_t y) -> float {
    if (x < 0 || y < 0 || z < 0 || (uint64_t)x >= t.width || (uint64_t)y >= t.height ||
        (uint64_t)z >= t.layers)
      return 0.0f;
    return (float)tex_read<T>(t, (uint32_t)x, (uint32_t)y, (uint32_t)z);
  };
  out[0] = at(x0, y0);
  out[1] = at(x0 + 1, y0);
  out[2] = at(x0, y0 + 1);
  out[3] = at(x0 + 1, y0 + 1);
}

// A cube gather, which is a face and a place on it rather than a direction: PTX
// names `(s, t, face)` with s and t running -1 to 1 across the face, so the same
// rebase that `weft_cube_dir` inverts for a sample turns it into a layer and a 2-D
// coordinate. The faces are stored one after another, so the face is the layer.
template <class T>
inline void tex_gather4_cube(const Texture& t, float s, float v, uint32_t face, float out[4]) {
  if (t.layers == 0) {
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    return;
  }
  tex_gather4<T>(t, (s + 1.0f) * 0.5f, (v + 1.0f) * 0.5f, face % t.layers, out);
}



// The xy half of a filtered sample at one slice: the whole of a 2D sample, and one
// plane of a 3D one. The weights are the hardware's 8-bit fixed point. Measured on
// this GPU by sampling between two known texels, the weight it applies is exactly
// round(f * 256) / 256 — which is what makes filtered samples identical to the GPU
// instead of close.
template <class T>
inline float tex_bilinear(const Texture& t, float u, float v, uint32_t layer, int addr) {
  if (!t.width || !t.height) return 0.0f;
  const float fx = u * (float)t.width - 0.5f;
  const float fy = v * (float)t.height - 0.5f;
  const int64_t x0 = (int64_t)std::floor(fx), y0 = (int64_t)std::floor(fy);
  auto weight = [](float f) { return std::nearbyint(f * 256.0f) * (1.0f / 256.0f); };
  const float ax = weight(fx - (float)x0), ay = weight(fy - (float)y0);
  const float a00 = tex_at_f<T>(t, x0, y0, layer, addr);
  const float a10 = tex_at_f<T>(t, x0 + 1, y0, layer, addr);
  const float a01 = tex_at_f<T>(t, x0, y0 + 1, layer, addr);
  const float a11 = tex_at_f<T>(t, x0 + 1, y0 + 1, layer, addr);
  // The blend is a fused multiply-add on the hardware, not a separate multiply
  // and add: measured over 32 samples between known texels, `fma(t1-t0, w, t0)`
  // reproduces the GPU's answer exactly where `t0 + (t1-t0)*w` is one ulp out.
  // So an fma here is not an optimisation, it is the definition.
  const float top = std::fma(a10 - a00, ax, a00), bot = std::fma(a11 - a01, ax, a01);
  return std::fma(bot - top, ay, top);
}

template <class T>
inline float tex_sample(const Texture& t, float u, float v, uint32_t layer, int filter, int addr) {
  if (layer >= t.layers) return 0.0f;
  if (filter == kTexNearest) {
    // The texel under the coordinate, clamped to the image, which is what
    // nearest filtering is on every GPU.
    const int64_t x = (int64_t)std::floor(u * (float)t.width);
    const int64_t y = (int64_t)std::floor(v * (float)t.height);
    return tex_at_f<T>(t, x, y, layer, addr);
  }
  return tex_bilinear<T>(t, u, v, layer, addr);
}

// A 3D sample: the plane the coordinate names, and for a filter the plane behind
// it blended in. Nearest takes the plane under w; linear is trilinear, with the z
// weight quantised and applied with an fma exactly as x and y are. A plane that
// clamp-to-zero pushes out of the image contributes nothing, so an edge behaves the
// way a 2D edge does.
template <class T>
inline float tex_sample(const Texture& t, float u, float v, float w, int filter, int addr) {
  const uint32_t n = t.layers;
  if (n == 0) return 0.0f;
  if (filter == kTexNearest) {
    const int64_t x = (int64_t)std::floor(u * (float)t.width);
    const int64_t y = (int64_t)std::floor(v * (float)t.height);
    const uint32_t z = tex_wrap((int64_t)std::floor(w * (float)n), n, addr);
    if (z == (uint32_t)-1) return 0.0f;
    return tex_at_f<T>(t, x, y, z, addr);
  }
  const float fz = w * (float)n - 0.5f;
  const int64_t z0 = (int64_t)std::floor(fz);
  auto weight = [](float f) { return std::nearbyint(f * 256.0f) * (1.0f / 256.0f); };
  const float az = weight(fz - (float)z0);
  const uint32_t a = tex_wrap(z0, n, addr), b = tex_wrap(z0 + 1, n, addr);
  const float s0 = a == (uint32_t)-1 ? 0.0f : tex_bilinear<T>(t, u, v, a, addr);
  const float s1 = b == (uint32_t)-1 ? 0.0f : tex_bilinear<T>(t, u, v, b, addr);
  return std::fma(s1 - s0, az, s0);
}

// ---- cubes ----
//
// A cube is six square faces stored face after face, which is the same slice
// layout a 2D array has, so a cube sample is a 2D sample of one of them. What
// picks the face is the direction: the axis it points along most strongly, and the
// other two components over that length place the texel on the face. Ties go to X
// then Y, and the six faces are +X, -X, +Y, -Y, +Z, -Z — `MTLTextureTypeCube`'s
// order, which is also every other GPU's.
inline void tex_cube_uv(float dx, float dy, float dz, uint32_t& face, float& u, float& v) {
  const float ax = std::fabs(dx), ay = std::fabs(dy), az = std::fabs(dz);
  float sc = 0.0f, tc = 0.0f, ma = 1.0f;
  if (ax >= ay && ax >= az) {
    ma = ax;
    if (dx >= 0.0f) { face = 0; sc = -dz; tc = -dy; }
    else { face = 1; sc = dz; tc = -dy; }
  } else if (ay >= az) {
    ma = ay;
    if (dy >= 0.0f) { face = 2; sc = dx; tc = dz; }
    else { face = 3; sc = dx; tc = -dz; }
  } else {
    ma = az;
    if (dz >= 0.0f) { face = 4; sc = dx; tc = -dy; }
    else { face = 5; sc = -dx; tc = -dy; }
  }
  if (ma == 0.0f) { u = 0.5f; v = 0.5f; return; }
  u = (sc / ma + 1.0f) * 0.5f;
  v = (tc / ma + 1.0f) * 0.5f;
}

template <class T>
inline float tex_sample_cube(const Texture& t, float dx, float dy, float dz, int filter, int addr) {
  uint32_t face = 0;
  float u = 0.5f, v = 0.5f;
  tex_cube_uv(dx, dy, dz, face, u, v);
  return tex_sample<T>(t, u, v, face, filter, addr);
}


// A plain texture is layer 0 of itself, so every call written before arrays
// existed keeps its meaning.
template <class T>
inline float tex_sample(const Texture& t, float u, float v, int filter, int addr) {
  return tex_sample<T>(t, u, v, 0u, filter, addr);
}

// ---- a mesh stage's output ----
//
// A `[[mesh]]` function produces one mesh per threadgroup: NV vertices and NP
// primitives, written by the 32 lanes of that group together. The launcher
// creates a vertex block and a triangle block per mesh, and the view below is
// this invocation's slice of both — so `set_vertex(i, v)` writes slot i of
// *this* mesh and cannot reach another group's. A primitive is three indices,
// 12 bytes: dense, and the layout a primitive buffer has on the hardware.
// A triangle is a `uint3` on the GPU, and Metal pads one to 16 bytes: a block of
// primitives is 16 bytes apart per primitive, so the mirror here has to be too or
// every triangle after the first lands at the wrong offset. The fourth word is
// written as zero rather than left as uninitialised, which is what makes the two
// blocks comparable byte for byte instead of only in their first three words.
struct alignas(16) MeshPrim {
  uint32_t x, y, z, _pad;
};

template <class V, int NV, int NP>
struct mesh_view {
  // `out.vertices[i]` and `out.primitives[i]` are direct slot access with no
  // bounds check, because a slot out of range is undefined on the hardware too.
  struct vrow {
    V* p;
    V& operator[](uint32_t i) const { return p[i]; }
  };
  struct prow {
    MeshPrim* p;
    MeshPrim& operator[](uint32_t i) const { return p[i]; }
  };
  vrow vertices;    // this group's vertex block, already offset to it
  prow primitives;  // this group's triangle block

  void set_vertex(uint32_t i, V v) {
    if (i < (uint32_t)NV) vertices.p[i] = v;
  }
  V get_vertex(uint32_t i) const { return vertices.p[i]; }
  // `uint3(a, b, c)` and `uint3{...}` both arrive here, and the vector types do
  // not exist yet at this point in the translation unit, so the operand is a
  // template and only its three members are used.
  template <class P>
  void set_primitive(uint32_t i, P t) {
    if (i >= (uint32_t)NP) return;
    primitives.p[i] = MeshPrim{(uint32_t)t.x, (uint32_t)t.y, (uint32_t)t.z, 0u};
  }
};

// ---- a task stage's say in the dispatch ----
//
// A task stage (`[[object]]`) runs first and decides how many meshes run after
// it: `mgp.set_threadgroups_per_grid(n)` writes the meshgrid the launcher then
// dispatches. It is a parameter with no binding — `mesh_grid_properties mgp` —
// because it is not memory the caller supplies but the stage's own output about
// the shape of the next stage.
struct mesh_grid_properties {
  uint32_t* out;  // x, y, z
  template <class P>
  void set_threadgroups_per_grid(P p) {
    if (!out) return;
    if constexpr (requires { p.x; }) {
      out[0] = (uint32_t)p.x;
      out[1] = (uint32_t)p.y;
      out[2] = (uint32_t)p.z;
    } else {
      out[0] = (uint32_t)p;
      out[1] = 1u;
      out[2] = 1u;
    }
  }
};

// Inline on purpose: a generated kernel body calls this once per buffer on
// entry, which for a resumable kernel means every lane on every pass. Out of
// line, that was a cross-image call for a single pointer load.
inline void* buffer(const Ctx& c, int index) { return c.bufs[index]; }

// ---- tile memory's boundary ----
//
// A tile stage and the fragments of one tile share that tile's memory and
// nothing else. This runtime lowers tile memory to one slot per pixel of the
// target, which keeps every *pixel* addressable — so an access outside the
// invocation's own 32x32 tile is representable here in a way the hardware
// leaves undefined and order-dependent (it is the other tile's memory, which
// may not have been written yet).
//
// Rather than let the two sides agree on a number neither of them may legally
// compute, `tile_px` counts those accesses and the launcher reports them. The
// index still comes through unchanged, so a kernel that stays inside its tile
// is unaffected and a kernel that does not is named instead of passing.
inline std::atomic<uint64_t>& tile_hits() {
  static std::atomic<uint64_t> n{0};
  return n;
}
inline void tile_reset() { tile_hits().store(0, std::memory_order_relaxed); }
inline uint64_t tile_violations() { return tile_hits().load(std::memory_order_relaxed); }
// Tile memory is a 32x32 block per tile, and a tile stage and the fragments of a
// tile share that block and nothing else. The slot is computed here for both
// sides' convenience, and the tile of the pixel read is compared with the tile
// this invocation is in: the hardware leaves that read undefined (it is another
// tile's block, which may not have been written yet), so it is counted and the
// launcher refuses the run instead of comparing a number neither side computed.
//
// A subscript may be a pixel coordinate (`blk[uint2(x, y)]`) or an index already.
template <class P>
inline long tile_slot(const Ctx& c, const P& p, uint32_t width, uint32_t tiles_x) {
  long pixel;
  if constexpr (requires { p.x; p.y; })
    pixel = (long)p.y * (long)width + (long)p.x;
  else
    pixel = (long)p;
  const long w = width ? (long)width : 1;
  const long x = pixel % w, y = pixel / w;
  const long tx = x >> 5, ty = y >> 5;
  const long gx = (((long)c.gid) % w) >> 5, gy = (((long)c.gid) / w) >> 5;
  if (tx != gx || ty != gy) tile_hits().fetch_add(1, std::memory_order_relaxed);
  return ((ty * (long)tiles_x) + tx) * 1024 + ((y & 31) << 5) + (x & 31);
}
inline void* buffer(const WCtx& c, int index) { return c.bufs[index]; }
void barrier(Ctx& c);
void dispatch(const KernelDef& k, void* const* bufs, uint32_t grid, uint32_t tg);

// Scalar kernel parameters (the `int n` a CUDA kernel carries). The host sets
// them before a run; kernels read them as ordinary values.
void set_param(int index, uint64_t value);
uint64_t param(int index);

// Atomics. These are the only places where two CPU threads touch the same
// address, so they are real atomics, not a lock-free pretence.
float atom_add_f32(float* p, float v);
uint32_t atom_add_u32(uint32_t* p, uint32_t v);
int32_t atom_add_s32(int32_t* p, int32_t v);
uint32_t atom_sub_u32(uint32_t* p, uint32_t v);
uint32_t atom_min_u32(uint32_t* p, uint32_t v);
uint32_t atom_max_u32(uint32_t* p, uint32_t v);
int32_t atom_min_s32(int32_t* p, int32_t v);
int32_t atom_max_s32(int32_t* p, int32_t v);
uint32_t atom_and_u32(uint32_t* p, uint32_t v);
uint32_t atom_or_u32(uint32_t* p, uint32_t v);
uint32_t atom_xor_u32(uint32_t* p, uint32_t v);
uint32_t atom_exch_u32(uint32_t* p, uint32_t v);
uint32_t atom_cas_u32(uint32_t* p, uint32_t cmp, uint32_t v);
uint64_t atom_add_u64(uint64_t* p, uint64_t v);
uint64_t atom_sub_u64(uint64_t* p, uint64_t v);
uint64_t atom_min_u64(uint64_t* p, uint64_t v);
uint64_t atom_max_u64(uint64_t* p, uint64_t v);
uint64_t atom_and_u64(uint64_t* p, uint64_t v);
uint64_t atom_or_u64(uint64_t* p, uint64_t v);
uint64_t atom_xor_u64(uint64_t* p, uint64_t v);
uint64_t atom_exch_u64(uint64_t* p, uint64_t v);
uint64_t atom_cas_u64(uint64_t* p, uint64_t cmp, uint64_t v);
int64_t atom_add_s64(int64_t* p, int64_t v);
int64_t atom_min_s64(int64_t* p, int64_t v);
int64_t atom_max_s64(int64_t* p, int64_t v);

// SPIR-V has a relaxed atomic load and store as instructions of their own, so
// they need the matching pair here rather than a plain dereference.
template <class T>
inline T atomic_load(void* p) {
  return __atomic_load_n(reinterpret_cast<T*>(p), __ATOMIC_RELAXED);
}

template <class T>
inline void atomic_store(void* p, T v) {
  __atomic_store_n(reinterpret_cast<T*>(p), v, __ATOMIC_RELAXED);
}
float atom_max_f32(float* p, float v);
float atom_min_f32(float* p, float v);

// Each distinct `threadgroup T name[N]` declaration in a kernel maps to one slot,
// named by its ordinal in the source. Not by a running count: a resumable kernel
// jumps into the middle of its body, so the declarations it skipped on the way in do
// not run — a counter would hand the next declaration the previous one's storage.
// The ordinal is fixed at translation, so every entry, every lane and every pass
// finds the same object at the same index.
template <class T, size_t N>
T* shared_mem(Ctx& c, int k) {
  if (k < 0 || k >= 64) shared_too_many();
  if (k >= c.slots->count) {
    // Each declaration starts at its own type's alignment, as a threadgroup
    // declaration does on the GPU: a half[3] in front of a half4 must not leave
    // the half4 on a two-byte boundary.
    const size_t al = alignof(T);
    if (al > 1) c.slots->bump = (c.slots->bump + al - 1) & ~(al - 1);
    c.slots->off[k] = c.slots->bump;
    c.slots->bump += sizeof(T) * N;
    c.slots->count = k + 1;
  }
  if (c.slots->bump > (size_t)48 * 1024) shared_overflow(c.slots->bump, (size_t)48 * 1024);
  return reinterpret_cast<T*>(const_cast<unsigned char*>(c.shared) + c.slots->off[k]);
}

// ---- tensor maps ----
//
// `cp.async.bulk.tensor` takes its description of the tensor from a 128-byte object
// rather than from its operands: the instruction names a box and the corner that box
// starts at, and the object says where the tensor begins and how it is laid out.
// NVIDIA's driver encodes that object and does not publish the byte layout — under
// CUDA 12 the type is opaque, and CUTLASS only ever edits fields through
// `tensormap.replace`, which indexes them instead of giving offsets — so weft defines
// the layout, and both engines read it with the decode below (`src/weft_msl.hpp` is the
// same code in Metal). The tag word is what keeps that honest: an object without it is
// not decoded at all, so a blob from `cuTensorMapEncodeTiled` ends on a message instead
// of on a tile copied from wherever its bytes happened to point.
//
//   off   0  u64  address of element (0, 0)
//   off   8  u8   rank, 1 to 5; then the kind, then the swizzle, then a spare byte
//   off  12  u32  element size in bytes: 1, 2, 4, 8 or 16
//   off  16  u32  extent[5]         the tensor's length along each dimension
//   off  40  u64  stride[5]         bytes between neighbours; [0] is one element
//   off 124  u32  tag               0x41544657, 'WFTA'
//
// What the rest means depends on the kind. A box (kind 0) is a rectangular tile:
//
//   off  80  u32  box[5]            how many elements a copy moves
//   off 100  u32  elementStride[5]  the step through the box; 0 or 1 is contiguous
//
// A column (kind 1) is what `.im2col` copies: `channelsPerPixel` channels of each of
// `pixelsPerColumn` pixels, which is the tile a convolution's implicit GEMM wants. The
// instruction names the corner and the filter window's position; the object says how a
// pixel index walks the tensor, for as many spatial dimensions as the rank leaves
// between the channels and the batch:
//
//   off  80  u32  channelsPerPixel
//   off  84  u32  pixelsPerColumn
//   off  88  u32  traversal[3]      elements one pixel step moves along W, H, D
//   off 100  i32  lowerCorner[3]    the window's base offset along W, H, D
//   off 112  u32  dilation[3]       elements one filter step moves along W, H, D
//
// The swizzle is 0, 32, 64 or 128 and applies to the threadgroup destination either
// way, so it lives in the header byte both kinds share.
//
// Dimension 0 is the contiguous one, so an element sits at
//
//   base + coordinate[0] * element size + the rest as coordinate * stride
//
// An element whose coordinate falls outside `extent` is written as zero rather than
// skipped: that is what the hardware's out-of-bounds fill does, and it is why a box may
// hang off the edge of a tensor.
constexpr uint32_t kTmaTag = 0x41544657u;

// The object is 64-byte aligned on the GPU and nothing here needs that: every field is
// read a byte at a time, so a descriptor at the front of a buffer of any alignment is
// the same one.
inline uint32_t tma_word(const unsigned char* d, uint32_t off) {
  return (uint32_t)d[off] | ((uint32_t)d[off + 1u] << 8) | ((uint32_t)d[off + 2u] << 16) |
         ((uint32_t)d[off + 3u] << 24);
}

inline uint64_t tma_dword(const unsigned char* d, uint32_t off) {
  return (uint64_t)tma_word(d, off) | ((uint64_t)tma_word(d, off + 4u) << 32);
}

// The swizzle XORs the sixteen-byte unit an element lands in with its row's index inside
// a block, which is the order a `wgmma` fragment read expects to find.
inline uint32_t tma_swizzled(uint32_t o, uint32_t mode) {
  if (mode != 32u && mode != 64u && mode != 128u) return o;
  const uint32_t shift = (mode == 128u) ? 7u : (mode == 64u) ? 6u : 5u;
  return o ^ (((o >> shift) & ((mode >> 4) - 1u)) << 4);
}

constexpr uint32_t kTmaBox = 0u;     // a rectangular tile
constexpr uint32_t kTmaColumn = 1u;  // a convolution column, for `.im2col`

struct TmaBox {
  uint64_t base = 0;
  uint32_t rank = 0, kind = 0, elem = 0, total = 1, swizzle = 0;
  uint64_t stride[5] = {0, 0, 0, 0, 0};
  uint32_t extent[5] = {0, 0, 0, 0, 0};
  // kind 0: how many elements the box moves, and the step through it.
  uint32_t box[5] = {0, 0, 0, 0, 0};
  uint32_t step[5] = {0, 0, 0, 0, 0};
  // kind 1: channels by pixels, and how a pixel and a filter step move.
  uint32_t channels = 0, pixels = 0;
  uint32_t traversal[3] = {1, 1, 1};
  uint32_t dilation[3] = {1, 1, 1};
  int32_t corner[3] = {0, 0, 0};
};

inline bool tma_decode(const unsigned char* d, TmaBox& m) {
  if (tma_word(d, 124u) != kTmaTag) return false;
  const uint32_t head = tma_word(d, 8u);
  m.rank = head & 0xFFu;
  m.kind = (head >> 8) & 0xFFu;
  m.swizzle = (head >> 16) & 0xFFu;
  m.elem = tma_word(d, 12u);
  if (m.rank == 0u || m.rank > 5u || m.elem == 0u || m.elem > 16u) return false;
  m.base = tma_dword(d, 0u);
  m.total = 1u;
  m.stride[0] = m.elem;
  for (uint32_t i = 0; i < m.rank; i++) {
    m.extent[i] = tma_word(d, 16u + 4u * i);
    if (m.extent[i] == 0u) return false;
    if (i > 0u) m.stride[i] = tma_dword(d, 40u + 8u * i);
  }
  if (m.kind == kTmaBox) {
    for (uint32_t i = 0; i < m.rank; i++) {
      m.box[i] = tma_word(d, 80u + 4u * i);
      m.step[i] = tma_word(d, 100u + 4u * i);
      if (m.box[i] == 0u) return false;
      m.total *= m.box[i];
    }
    return true;
  }
  if (m.kind == kTmaColumn) {
    m.channels = tma_word(d, 80u);
    m.pixels = tma_word(d, 84u);
    if (m.channels == 0u || m.pixels == 0u) return false;
    for (uint32_t i = 0; i < 3u; i++) {
      m.traversal[i] = tma_word(d, 88u + 4u * i);
      m.corner[i] = (int32_t)tma_word(d, 100u + 4u * i);
      m.dilation[i] = tma_word(d, 112u + 4u * i);
      if (m.traversal[i] == 0u) m.traversal[i] = 1u;
      if (m.dilation[i] == 0u) m.dilation[i] = 1u;
    }
    m.total = m.channels * m.pixels;
    return true;
  }
  return false;
}

// The byte a box element starts at, and whether that element is inside the tensor.
inline bool tma_element(const TmaBox& m, const int32_t* co, uint32_t k, uint64_t& addr) {
  uint64_t off = 0;
  uint32_t rest = k;
  bool inside = true;
  for (uint32_t i = 0; i < m.rank; i++) {
    const uint32_t e = rest % m.box[i];
    rest /= m.box[i];
    const int64_t c = (int64_t)co[i] + (int64_t)e * (int64_t)(m.step[i] ? m.step[i] : 1u);
    if (c < 0 || c >= (int64_t)m.extent[i]) {
      inside = false;
      break;
    }
    off += (uint64_t)c * m.stride[i];
  }
  addr = m.base + off;
  return inside;
}

// An object that is not one of ours: the tile is left alone and the transaction reports
// the same "nothing moved" sentinel on both engines, so a descriptor of the wrong kind
// shows up as a message and a number instead of as a copy of whatever its bytes said.
inline void tma_not_ours() {
  static bool said = false;
  if (said) return;
  said = true;
  std::fprintf(stderr,
               "weft: a tensor copy was handed an object with no weft tag at offset 124. "
               "The layout the driver encodes is not published, so weft defines its own and "
               "decodes only that; the tile was left as it was\n");
}

// threadgroup memory <- the tensor. Returns the bytes the box covers, which is what the
// barrier an `expect_tx` sized on is completed with, outside-of-bounds elements and all.
inline uint32_t tma_load(unsigned char* dst, const void* map, int32_t c0, int32_t c1,
                         int32_t c2, int32_t c3, int32_t c4) {
  const unsigned char* d = (const unsigned char*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != kTmaBox) {
    tma_not_ours();
    return 0xFFFFFFFFu;
  }
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  for (uint32_t k = 0; k < m.total; k++) {
    uint64_t addr = 0;
    const bool inside = tma_element(m, co, k, addr);
    const unsigned char* s = (const unsigned char*)(uintptr_t)addr;
    const uint32_t o = tma_swizzled(k * m.elem, m.swizzle);
    for (uint32_t b = 0; b < m.elem; b++) dst[o + b] = inside ? s[b] : (unsigned char)0;
  }
  return m.total * m.elem;
}

// The tensor <- threadgroup memory. A store writes nothing outside the tensor, so an
// element whose coordinate is out of bounds is skipped rather than zero-filled.
inline void tma_store(const void* map, int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                      int32_t c4, const unsigned char* src) {
  const unsigned char* d = (const unsigned char*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != kTmaBox) {
    tma_not_ours();
    return;
  }
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  for (uint32_t k = 0; k < m.total; k++) {
    uint64_t addr = 0;
    if (!tma_element(m, co, k, addr)) continue;
    unsigned char* t = (unsigned char*)(uintptr_t)addr;
    const uint32_t o = tma_swizzled(k * m.elem, m.swizzle);
    for (uint32_t b = 0; b < m.elem; b++) t[b] = src[o + b];
  }
}

// Where pixel `p` of a column lands: the corner advanced by `traversal` in each spatial
// dimension, the first fastest, plus the filter step's `dilation`. False means the pixel
// fell outside the tensor, which is the edge a padded convolution is made of.
inline bool tma_column_at(const TmaBox& m, const int32_t* co, const int32_t* of,
                          const uint32_t* fit, uint32_t p, uint64_t& at) {
  const uint32_t spatial = m.rank - 2u;  // the batch is the last dimension
  const int64_t batch = co[spatial + 1u];
  if (batch < 0 || batch >= (int64_t)m.extent[spatial + 1u]) return false;
  at = (uint64_t)batch * m.stride[spatial + 1u];
  uint32_t q = p;
  for (uint32_t i = 0; i < spatial; i++) {
    const uint32_t step = q % fit[i];
    q /= fit[i];
    const int64_t at_i = (int64_t)co[i + 1u] * (int64_t)m.traversal[i] + m.corner[i] +
                         (int64_t)step * (int64_t)m.traversal[i] +
                         (int64_t)of[i] * (int64_t)m.dilation[i];
    if (at_i < 0 || at_i >= (int64_t)m.extent[i + 1u]) return false;
    at += (uint64_t)at_i * m.stride[i + 1u];
  }
  return true;
}

// The convolution form: threadgroup memory <- a column of the tensor. The coordinate
// names the corner and `off` the filter window's position, one per spatial dimension.
// A pixel that lands outside the tensor is zero, which is what makes padding work.
inline uint32_t tma_im2col(unsigned char* dst, const void* map, int32_t c0, int32_t c1,
                           int32_t c2, int32_t c3, int32_t c4, int32_t o0, int32_t o1,
                           int32_t o2) {
  const unsigned char* d = (const unsigned char*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != kTmaColumn) {
    tma_not_ours();
    return 0xFFFFFFFFu;
  }
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  const int32_t of[3] = {o0, o1, o2};
  const uint32_t spatial = m.rank - 2u;  // the batch is the last dimension
  // How many pixels fit along each spatial dimension at this traversal.
  uint32_t fit[3] = {1u, 1u, 1u};
  for (uint32_t i = 0; i < spatial; i++)
    fit[i] = (m.extent[i + 1u] + m.traversal[i] - 1u) / m.traversal[i];
  for (uint32_t p = 0; p < m.pixels; p++) {
    uint64_t at = 0;
    const bool inside = tma_column_at(m, co, of, fit, p, at);
    for (uint32_t e = 0; e < m.channels; e++) {
      const uint32_t o = tma_swizzled((p * m.channels + e) * m.elem, m.swizzle);
      const int64_t c = (int64_t)co[0] + (int64_t)e;
      if (!inside || c < 0 || c >= (int64_t)m.extent[0]) {
        for (uint32_t b = 0; b < m.elem; b++) dst[o + b] = 0;
        continue;
      }
      const unsigned char* s =
          (const unsigned char*)(uintptr_t)(m.base + at + (uint64_t)c * m.stride[0]);
      for (uint32_t b = 0; b < m.elem; b++) dst[o + b] = s[b];
    }
  }
  return m.channels * m.pixels * m.elem;
}

// The same column going back out: threadgroup memory is scattered into the window. An
// element whose place fell outside the tensor writes nothing — a store has nothing to
// zero-fill — so a column that hangs off the edge changes only the part that fits.
inline void tma_im2col_store(const void* map, int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                             int32_t c4, int32_t o0, int32_t o1, int32_t o2,
                             const unsigned char* src) {
  const unsigned char* d = (const unsigned char*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != kTmaColumn) {
    tma_not_ours();
    return;
  }
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  const int32_t of[3] = {o0, o1, o2};
  const uint32_t spatial = m.rank - 2u;
  uint32_t fit[3] = {1u, 1u, 1u};
  for (uint32_t i = 0; i < spatial; i++)
    fit[i] = (m.extent[i + 1u] + m.traversal[i] - 1u) / m.traversal[i];
  for (uint32_t p = 0; p < m.pixels; p++) {
    uint64_t at = 0;
    if (!tma_column_at(m, co, of, fit, p, at)) continue;
    for (uint32_t e = 0; e < m.channels; e++) {
      const int64_t c = (int64_t)co[0] + (int64_t)e;
      if (c < 0 || c >= (int64_t)m.extent[0]) continue;
      unsigned char* t = (unsigned char*)(uintptr_t)(m.base + at + (uint64_t)c * m.stride[0]);
      const uint32_t o = tma_swizzled((p * m.channels + e) * m.elem, m.swizzle);
      for (uint32_t b = 0; b < m.elem; b++) t[b] = src[o + b];
    }
  }
}

}  // namespace weft
