// weft, the GPU's execution model, implemented on CPU cores.
#pragma once

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

inline uint64_t g_pass_gen = 0;
inline uint64_t pass_generation() { return g_pass_gen; }

struct WorkerState;

// Per-threadgroup shared-memory bookkeeping.
struct SlotState {
  size_t off[64];
  size_t bump;
  int count;
};

// Two words per launch: what stopped a lane, and which lane it was.
inline uint32_t* fault_words() {
  static uint32_t w[2] = {0u, 0u};
  return w;
}

struct Ctx {
  uint32_t lid = 0;       // thread_position_in_threadgroup
  uint32_t tg_size = 0;   // threads_per_threadgroup
  uint32_t pass = 0;      // dispatcher pass number, for shuffle double-buffering
  uint32_t* warp = nullptr;  // per-lane scratch for shfl
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
  uint32_t* fault = fault_words();
};

// Words in one warp's scan cache: the three key words and the 32 results.
constexpr uint32_t kFoldWords = 35u;

using Body = void (*)(Ctx&);

using FlatBody = void (*)(Ctx&, uint32_t begin, uint32_t end);

using StepFn = void (*)(Ctx&, void* state, void* const* bufs, unsigned char* sh);

using GroupStepFn = uint32_t (*)(Ctx* ctxs, unsigned char* states, size_t stride, uint32_t live,
                                 uint32_t pass, void* const* bufs, unsigned char* sh);
constexpr uint32_t kThreadDone = 0xFFFFFFFFu;

struct WCtx {
  uint32_t lid0 = 0;      // index of this warp's first lane in the threadgroup
  uint32_t* fault = fault_words();
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

// A kernel body that can suspend at a barrier and be resumed later.
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

// Warp shuffles.
enum ShflMode { kShflIdx = 0, kShflUp = 1, kShflDown = 2, kShflBfly = 3 };

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

inline const uint32_t* shfl_pub_buf(const Ctx& c) {
  return (const uint32_t*)c.warp + (size_t)(c.pass & 1u) * c.tg_size;
}
inline const uint32_t* shfl_read_buf(const Ctx& c) {
  return (const uint32_t*)c.warp + (size_t)((c.pass - 1u) & 1u) * c.tg_size;
}

// A warp vote: which lanes of this warp published a non-zero value.
uint32_t vote_ballot(Ctx& c);

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

constexpr size_t kWideSlotBytes = 16;

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

template <class T>
inline void shfl_publish(Ctx& c, T value) {
  if constexpr (sizeof(T) <= 4) {
    ((uint32_t*)shfl_pub_buf(c))[c.lid] = to_bits(value);
  } else {
    static_assert(sizeof(T) <= kWideSlotBytes, "a shuffle is at most 16 bytes wide");
    std::memcpy(wide_pub_slot(c), &value, sizeof(T));
  }
}

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

template <class T, std::enable_if_t<(sizeof(T) > 4), int> = 0>
inline WideShflAwaitable<T> grp_shfl(Ctx& c, T value, uint32_t b, int mode, uint32_t width) {
  static_assert(sizeof(T) <= kWideSlotBytes, "a shuffle is at most 16 bytes wide");
  return WideShflAwaitable<T>{&c, value, T{}, b, mode, width};
}

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
  T await_resume() const noexcept { return take(); }
  T take() const noexcept {
    const uint32_t base = c->lid & ~(width - 1u);
    const uint32_t lane = c->lid - base;
    uint32_t* const slot = c->fold;
    if (slot && site != 0u && slot[0] == site && slot[1] == c->pass && slot[2] == base) {
      if (inclusive) return lane < width ? from_bits<T>(slot[3 + lane]) : T(0);
      return lane ? from_bits<T>(slot[3 + lane - 1u]) : T(product ? 1 : 0);
    }
    const uint32_t* slots = shfl_read_buf(*c) + base;
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

enum RedMode { kRedPopc = 0, kRedAnd = 1, kRedOr = 2 };

inline uint32_t red_fold_bits(int mode, uint32_t acc, uint32_t v) {
  if (mode == kRedPopc) return acc + v;
  if (mode == kRedAnd) return acc & (v ? 1u : 0u);
  return acc | (v ? 1u : 0u);
}

inline uint32_t red_empty(int mode) { return mode == kRedAnd ? 1u : 0u; }

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
    if (slot && site != 0u && slot[0] == site && slot[1] == c->pass && slot[2] == base) {
      const uint32_t at = c->lid - base;
      return at < width && at < 32u ? from_bits<T>(slot[3 + at]) : T(0);
    }
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

// simd_ballot / simd_all / simd_any: the same rendezvous, one bit wide.
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

inline VoteAwaitable grp_vote(Ctx& c, bool value, int mode, uint32_t width) {
  return VoteAwaitable{&c, value, mode, width};
}

void* coro_alloc(std::size_t n);
void coro_arena_reset();

// Both of these used to trap, which says nothing to anyone reading a terminal.
[[noreturn]] void shared_overflow(std::size_t need, std::size_t cap);
[[noreturn]] void shared_too_many();

using TaskFn = Task (*)(Ctx&);

// sync == the kernel declares threadgroup memory or calls a barrier.
struct KernelDef {
  const char* name;
  Body body;
  int nargs;
  // Whether the kernel synchronises (shared memory, a barrier or a shuffle).
  bool sync;
  FlatBody flat;    // nullptr when the kernel cannot be flattened
  StepFn step;      // resumable via a state blob (PTX)
  size_t state_size;
  TaskFn coro;      // resumable via a coroutine frame (Metal with barriers)
  WarpFn warp;        // 32 lanes at a time (PTX, when the body allows it)
  size_t warp_size;   // bytes of state for one warp
  GroupStepFn group_step;  // set when the module emits one
};

struct Texture {
  // Writable, because a storage image is written.
  unsigned char* data = nullptr;
  uint32_t width = 0, height = 0;
  uint32_t elem = 4;    // bytes per texel (4 for float/uint, 2 for half)
  // The outermost extent: the slices of a 2D array, or a 3D texture's depth.
  uint32_t layers = 1;
};

Texture texture(int index);
void set_texture(int index, void* data, uint32_t width, uint32_t height, uint32_t elem);
// The array form: `layers` slices, laid out one after another.
void set_texture_layers(int index, void* data, uint32_t width, uint32_t height, uint32_t elem,
                        uint32_t layers);

struct Sampler {
  int filter = 0;  // kTexNearest / kTexLinear
  int addr = 0;    // kTexClampToEdge / kTexClampToZero / kTexRepeat
};

Sampler sampler(int index);
void set_sampler(int index, int filter, int addr);

// The image a sampler reads.
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

// A write to a texture.
template <class T>
inline void tex_write(const Texture& t, uint32_t x, uint32_t y, uint32_t layer, T v) {
  if (layer >= t.layers || x >= t.width || y >= t.height || !t.data || t.elem > sizeof(T)) return;
  std::memcpy(t.data + (((size_t)layer * t.height + y) * t.width + x) * t.elem, &v, t.elem);
}

template <class T>
inline void tex_write(const Texture& t, uint32_t x, uint32_t y, T v) {
  tex_write<T>(t, x, y, 0u, v);
}

// A surface addresses bytes: x is a byte offset within the row, y a row, and a
// `.b32` access moves the four bytes there. An offset off a 4-byte boundary is
// one the card faults on, and reads zero here.
inline uint32_t tex_byte_read(const Texture& t, uint32_t x, uint32_t y, uint32_t layer) {
  if (layer >= t.layers || !t.data || (x & 3u)) return 0u;
  const size_t row = (size_t)t.width * t.elem;
  if (y >= t.height || (size_t)x + 4 > row) return 0u;
  uint32_t v = 0;
  std::memcpy(&v, t.data + ((size_t)layer * t.height + y) * row + x, 4);
  return v;
}

inline void tex_byte_write(const Texture& t, uint32_t x, uint32_t y, uint32_t layer, float v) {
  if (layer >= t.layers || !t.data || (x & 3u)) return;
  const size_t row = (size_t)t.width * t.elem;
  if (y >= t.height || (size_t)x + 4 > row) return;
  std::memcpy(t.data + ((size_t)layer * t.height + y) * row + x, &v, 4);
}

inline uint32_t tex_byte_read(const Texture& t, uint32_t x, uint32_t y) {
  return tex_byte_read(t, x, y, 0u);
}

inline void tex_byte_write(const Texture& t, uint32_t x, uint32_t y, float v) {
  tex_byte_write(t, x, y, 0u, v);
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

// A `tex` instruction hands back four channels.
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

template <class T>
inline void tex_sample4(const Texture& t, float u, float v, float w, int filter, int addr,
                        float out[4]) {
  out[0] = tex_sample<T>(t, u, v, w, filter, addr);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
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

template <class T>
inline void tex_sample4_cube(const Texture& t, float dx, float dy, float dz, uint32_t layer,
                             int filter, int addr, float out[4]) {
  out[0] = tex_sample_cube<T>(t, dx, dy, dz, layer, filter, addr);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

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
  const float top = std::fma(a10 - a00, ax, a00), bot = std::fma(a11 - a01, ax, a01);
  return std::fma(bot - top, ay, top);
}

template <class T>
inline float tex_sample(const Texture& t, float u, float v, uint32_t layer, int filter, int addr) {
  if (layer >= t.layers) return 0.0f;
  if (filter == kTexNearest) {
    const int64_t x = (int64_t)std::floor(u * (float)t.width);
    const int64_t y = (int64_t)std::floor(v * (float)t.height);
    return tex_at_f<T>(t, x, y, layer, addr);
  }
  return tex_bilinear<T>(t, u, v, layer, addr);
}

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

// A layer of a cube array: six faces to a layer, one after another.
template <class T>
inline float tex_sample_cube(const Texture& t, float dx, float dy, float dz, uint32_t layer,
                             int filter, int addr) {
  uint32_t face = 0;
  float u = 0.5f, v = 0.5f;
  tex_cube_uv(dx, dy, dz, face, u, v);
  return tex_sample<T>(t, u, v, layer * 6u + face, filter, addr);
}

template <class T>
inline void tex_gather4_cube_dir(const Texture& t, float dx, float dy, float dz, float du,
                                 float dv, float out[4]) {
  if (t.layers == 0) {
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    return;
  }
  uint32_t face = 0;
  float u = 0.5f, v = 0.5f;
  tex_cube_uv(dx, dy, dz, face, u, v);
  tex_gather4<T>(t, u + du, v + dv, face % t.layers, out);
}

template <class T>
inline void tex_gather4_cube_array_dir(const Texture& t, float dx, float dy, float dz,
                                       uint32_t layer, float du, float dv, float out[4]) {
  uint32_t face = 0;
  float u = 0.5f, v = 0.5f;
  tex_cube_uv(dx, dy, dz, face, u, v);
  tex_gather4<T>(t, u + du, v + dv, layer * 6u + face % 6u, out);
}

template <class T>
inline float tex_sample(const Texture& t, float u, float v, int filter, int addr) {
  return tex_sample<T>(t, u, v, 0u, filter, addr);
}

struct alignas(16) MeshPrim {
  uint32_t x, y, z, _pad;
};

template <class V, int NV, int NP>
struct mesh_view {
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
  template <class P>
  void set_primitive(uint32_t i, P t) {
    if (i >= (uint32_t)NP) return;
    primitives.p[i] = MeshPrim{(uint32_t)t.x, (uint32_t)t.y, (uint32_t)t.z, 0u};
  }
};

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

inline void* buffer(const Ctx& c, int index) { return c.bufs[index]; }

inline std::atomic<uint64_t>& tile_hits() {
  static std::atomic<uint64_t> n{0};
  return n;
}
inline void tile_reset() { tile_hits().store(0, std::memory_order_relaxed); }
inline uint64_t tile_violations() { return tile_hits().load(std::memory_order_relaxed); }
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

// Scalar kernel parameters (the `int n` a CUDA kernel carries).
void set_param(int index, uint64_t value);
uint64_t param(int index);

// Atomics.
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

template <class T, size_t N>
T* shared_mem(Ctx& c, int k) {
  if (k < 0 || k >= 64) shared_too_many();
  if (k >= c.slots->count) {
    const size_t al = alignof(T);
    if (al > 1) c.slots->bump = (c.slots->bump + al - 1) & ~(al - 1);
    c.slots->off[k] = c.slots->bump;
    c.slots->bump += sizeof(T) * N;
    c.slots->count = k + 1;
  }
  if (c.slots->bump > (size_t)48 * 1024) shared_overflow(c.slots->bump, (size_t)48 * 1024);
  return reinterpret_cast<T*>(const_cast<unsigned char*>(c.shared) + c.slots->off[k]);
}

constexpr uint32_t kTmaTag = 0x41544657u;

inline uint32_t tma_word(const unsigned char* d, uint32_t off) {
  return (uint32_t)d[off] | ((uint32_t)d[off + 1u] << 8) | ((uint32_t)d[off + 2u] << 16) |
         ((uint32_t)d[off + 3u] << 24);
}

inline uint64_t tma_dword(const unsigned char* d, uint32_t off) {
  return (uint64_t)tma_word(d, off) | ((uint64_t)tma_word(d, off + 4u) << 32);
}

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

inline void tma_not_ours() {
  static bool said = false;
  if (said) return;
  said = true;
  std::fprintf(stderr,
               "weft: a tensor copy was handed an object with no weft tag at offset 124. "
               "The layout the driver encodes is not published, so weft defines its own and "
               "decodes only that; the tile was left as it was\n");
}

// threadgroup memory <- the tensor.
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

// The tensor <- threadgroup memory.
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

// The convolution form: threadgroup memory <- a column of the tensor.
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
