#include "weft_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------

namespace {

;

constexpr size_t kSharedBytes = 48 * 1024;
constexpr uint32_t kMaxThreadgroup = 1024;

// One worker's reusable fiber pool: a single block of stacks, split on demand.
struct Pool {
  unsigned char* shared = nullptr;

  void* states = nullptr;
  size_t states_cap = 0;

  uint32_t* warp = nullptr;  // one slot per lane, for shuffles
  uint32_t* red = nullptr;   // one slot per thread, plus one per warp, for bar.red
  uint32_t* fold = nullptr;  // one scan/fold cache per warp, for the warp library

  unsigned char* ensure_shared() {
    if (!shared) shared = (unsigned char*)std::aligned_alloc(64, kSharedBytes);
    return shared;
  }

  // bar.red needs tg_size slots for the per-thread values and 32 for the per-warp
  // totals. It is a separate buffer from the shuffle scratch because both can be
  // live in the same kernel and neither may clobber the other.
  uint32_t* ensure_red() {
    if (!red) {
      red = (uint32_t*)std::malloc((size_t)(kMaxThreadgroup + 32) * sizeof(uint32_t));
      if (!red) {
        std::fprintf(stderr, "weft: out of memory for reduction scratch\n");
        std::abort();
      }
    }
    return red;
  }

  // A warp's scan and fold trees are the same for its 32 lanes, so one cache
  // slot per warp is enough: three key words and the 32 results.
  uint32_t* ensure_fold() {
    if (!fold) {
      const size_t n = (size_t)(kMaxThreadgroup / 32u) * weft::kFoldWords * sizeof(uint32_t);
      fold = (uint32_t*)std::calloc(n, 1);
      if (!fold) {
        std::fprintf(stderr, "weft: out of memory for warp scan cache\n");
        std::abort();
      }
    }
    return fold;
  }

  uint32_t* ensure_warp() {
    if (!warp) {
      // Two halves, so site N+1 cannot clobber the slot site N is being read from.
      // The wide (16-byte) slots follow the word-wide ones in the same block, so
      // a kernel that shuffles both a float and a float4 needs no extra scratch;
      // Ctx::warp16 is the tail half of this allocation.
      const size_t n = (size_t)2 * kMaxThreadgroup * (sizeof(uint32_t) + weft::kWideSlotBytes);
      warp = (uint32_t*)std::malloc(n);
      if (!warp) {
        std::fprintf(stderr, "weft: out of memory for warp scratch\n");
        std::abort();
      }
    }
    return warp;
  }

  void* ensure_states(size_t bytes) {
    if (states_cap >= bytes) return states;
    std::free(states);
    states = std::malloc(bytes);
    if (!states) {
      std::fprintf(stderr, "weft: out of memory for %zu bytes of thread state\n", bytes);
      std::abort();
    }
    states_cap = bytes;
    return states;
  }

};

}  // namespace

namespace weft {

struct WorkerState {
  // A warp vote reads what the whole warp published, and every lane of the warp
  // asks the same question in the same pass, so one answer serves all 32. The
  // pass number is the key: within a pass the read buffer cannot change.
  uint32_t vote_pass = 0xFFFFFFFFu;
  uint32_t vote_mask = 0;
};

uint32_t vote_ballot(Ctx& c) {
  if (c.ws->vote_pass != c.pass) {
    const uint32_t* b = shfl_read_buf(c);
    const uint32_t base = c.lid & ~31u;
    uint32_t m = 0;
    for (uint32_t l = 0; l < 32u; l++)
      if (b[base + l]) m |= 1u << l;
    c.ws->vote_pass = c.pass;
    c.ws->vote_mask = m;
  }
  return c.ws->vote_mask;
}

}  // namespace weft

using weft::WorkerState;

namespace {

// Per-group thread contexts, built once and then reused for every pass.
// Rebuilding a Ctx per thread per pass was twelve stores on the hot path; a
// barrier- or shuffle-heavy kernel does six passes over a threadgroup, so that
// was most of the per-lane cost.
struct StepGroup {
  std::vector<weft::Ctx> ctxs;
  uint32_t built = 0;  // lanes whose launch-invariant fields are already written
};

// Resumable kernels: one call per thread per barrier segment. No stacks, no
// switches — the thread's registers are already in the state blob.
void run_group_step(const weft::KernelDef& k, uint32_t tgid, uint32_t tg, uint32_t grid,
                    WorkerState& ws, void* const* bufs, unsigned char* shared,
                    unsigned char* states, uint32_t* warp, uint32_t* red, uint32_t* fold,
                    StepGroup& g) {
  const uint32_t base = tgid * tg;
  const uint32_t live = (base + tg <= grid) ? tg : (grid > base ? grid - base : 0);
  const size_t stride = k.state_size;
  if (live < tg) {
    weft::clear_dead_lanes((unsigned char*)warp, live, tg);
    std::memset(shared, 0, kSharedBytes);
  }

  g.ctxs.resize(live);
  // A new group's scan caches start empty: the pass number counts from zero in
  // every group, so a key left by the previous group could match this one's.
  for (uint32_t w = 0; w < (live + 31u) / 32u; w++) fold[(size_t)w * weft::kFoldWords] = 0u;
  // Nine of the twelve fields depend on the launch, not on the group, so they are
  // written once per lane instead of once per lane per group: a barrier-heavy
  // kernel sets this up 4,096 times for a 1M-lane grid, and every write past the
  // first group is the same value again.
  if (g.built < live) {
    unsigned char* w16 = (unsigned char*)warp + (size_t)2 * kMaxThreadgroup * sizeof(uint32_t);
    for (uint32_t i = g.built; i < live; i++) {
      weft::Ctx& c = g.ctxs[i];
      c.lid = i;
      c.tg_size = tg;
      c.grid = grid;
      c.slots = nullptr;
      c.bufs = bufs;
      c.ws = &ws;
      c.warp = warp;
      c.warp16 = w16;
      c.red = red;
      c.fold = fold + (size_t)(i / 32u) * weft::kFoldWords;
    }
    g.built = live;
  }
  for (uint32_t i = 0; i < live; i++) {
    *reinterpret_cast<uint32_t*>(states + i * stride) = 0;
    weft::Ctx& c = g.ctxs[i];
    c.gid = base + i;
    c.tgid = tgid;
    c.shared = shared;
    c.pass = 0;
    // The shared-memory cursor is per group: a lane that allocated a slot in the
    // previous group must not carry its index into this one.
    c.slot = 0;
  }

  uint32_t remaining = live;
  uint32_t pass = 1;
  while (remaining > 0) {
    // A whole-group call needs every pass number written first; the per-lane call
    // writes its own just before it runs, which keeps the store on a line the
    // call is about to touch anyway.
    if (k.group_step)
      for (uint32_t i = 0; i < live; i++) g.ctxs[i].pass = pass;  // shuffles key on this
    if (k.group_step) {
      // Whole group, one call: the lane loop is in the module's own function.
      k.group_step(g.ctxs.data(), states, stride, live, bufs, shared);
      remaining = 0;
      unsigned char* st = states;
      for (uint32_t i = 0; i < live; i++, st += stride)
        if (*reinterpret_cast<uint32_t*>(st) != weft::kThreadDone) remaining++;
    } else {
      unsigned char* st = states;
      for (uint32_t i = 0; i < live; i++, st += stride) {
        uint32_t* pc = reinterpret_cast<uint32_t*>(st);
        if (*pc == weft::kThreadDone) continue;
        weft::Ctx& c = g.ctxs[i];
        c.pass = pass;
        k.step(c, st, bufs, shared);
        if (*pc == weft::kThreadDone) remaining--;
      }
    }
    pass++;
  }
}

// Warp kernels: 32 lanes execute together, one program counter per lane so that
// divergence is just a split into groups. A barrier is the whole threadgroup
// arriving, which the scheduler below enforces by running every warp until it
// stops and only then releasing them all.
void run_group_warp(const weft::KernelDef& k, uint32_t tgid, uint32_t tg, uint32_t grid,
                    void* const* bufs, unsigned char* shared, unsigned char* states,
                    std::vector<uint8_t>& status, std::vector<weft::WCtx>& ctxs,
                    uint32_t* red) {
  const uint32_t base = tgid * tg;
  const uint32_t live = (base + tg <= grid) ? tg : (grid > base ? grid - base : 0);  const uint32_t warps = (live + 31) / 32;
  // A last group that is short has lanes which do not exist: a kernel that reads
  // a slot only they would have written reads the arena instead. The step and
  // coroutine paths zero it there, so this one does too — the same rule, and it
  // only touches a group the launch did not fill.
  if (live < tg) std::memset(shared, 0, kSharedBytes);

  status.assign(warps, 0);  // 0 running, 1 at a barrier, 2 finished
  ctxs.resize(warps);
  for (uint32_t w = 0; w < warps; w++) {
    weft::WCtx& c = ctxs[w];
    c.lid0 = w * 32;
    c.tg_size = tg;
    c.tgid = tgid;
    c.grid = grid;
    c.bufs = bufs;
    c.shared = shared;
    c.red = red;
    std::memset(states + (size_t)w * k.warp_size, 0, k.warp_size);
  }

  for (;;) {
    bool ran = false;
    for (uint32_t w = 0; w < warps; w++) {
      if (status[w] != 0) continue;
      const uint32_t st = k.warp(ctxs[w], states + (size_t)w * k.warp_size);
      status[w] = (st == weft::kWarpDone) ? 2 : 1;
      ran = true;
    }
    if (ran) continue;

    // Nothing left to run: either everyone is done, or every warp still going is
    // waiting at a barrier. If any are waiting, they all leave together.
    bool released = false;
    for (uint32_t w = 0; w < warps; w++) {
      if (status[w] == 1) { status[w] = 0; released = true; }
    }
    if (!released) return;
  }
}

// Coroutine kernels: the compiler owns the frame, so locals survive a barrier.
struct CoroGroup {
  std::vector<weft::Ctx> ctxs;
  std::vector<weft::Task> tasks;
  weft::SlotState slots{};
  uint32_t built = 0;  // lanes whose launch-invariant fields are already written
};

void run_group_coro(const weft::KernelDef& k, uint32_t tgid, uint32_t tg, uint32_t grid,
                    WorkerState& ws, void* const* bufs, unsigned char* shared, CoroGroup& g,
                    uint32_t* warp, uint32_t* fold) {
  const uint32_t base = tgid * tg;
  const uint32_t live = (base + tg <= grid) ? tg : (grid > base ? grid - base : 0);
  if (live < tg) {
    weft::clear_dead_lanes((unsigned char*)warp, live, tg);
    std::memset(shared, 0, kSharedBytes);
  }
  weft::coro_arena_reset();  // frames from the previous group were destroyed
  g.tasks.clear();
  g.ctxs.resize(live);
  g.slots.bump = 0;
  g.slots.count = 0;
  for (int k = 0; k < 64; k++) g.slots.off[k] = 0;
  // The scan caches are per warp and keyed on the pass number, which counts from
  // zero in every group: clear the keys so the previous group's cannot match.
  for (uint32_t w = 0; w < (live + 31u) / 32u; w++) fold[(size_t)w * weft::kFoldWords] = 0u;

  // Same split as the step path: the launch-invariant fields once per lane, the
  // per-group ones every group. `shared` is here because each group's arena is
  // its own window and it is the one value that changes group to group.
  if (g.built < live) {
    unsigned char* w16 = (unsigned char*)warp + (size_t)2 * kMaxThreadgroup * sizeof(uint32_t);
    for (uint32_t i = g.built; i < live; i++) {
      weft::Ctx& c = g.ctxs[i];
      c.lid = i;
      c.tg_size = tg;
      c.grid = grid;
      c.slots = &g.slots;
      c.bufs = bufs;
      c.ws = &ws;
      c.warp = warp;
      c.warp16 = w16;
      c.fold = fold + (size_t)(i / 32u) * weft::kFoldWords;
    }
    g.built = live;
  }
  for (uint32_t i = 0; i < live; i++) {
    weft::Ctx& c = g.ctxs[i];
    c.gid = base + i;
    c.tgid = tgid;
    c.shared = shared;
    // Per group, like the pass: a stale cursor would point a lane's shared array
    // into the previous group's offsets.
    c.slot = 0;
    c.pass = 0;  // the run up to the first yield is pass 0
    g.tasks.emplace_back(k.coro(c));  // runs up to the first barrier
  }

  // A coroutine with no suspension point (a kernel that only asks for its lane
  // id, say) is already finished when it is built, and the loop below only ever
  // looks at tasks it resumes — so the live count has to come from the tasks
  // themselves, or it would wait forever for a thread with nothing left to do.
  uint32_t remaining = 0;
  for (uint32_t i = 0; i < live; i++)
    if (!g.tasks[i].done()) remaining++;
  uint32_t pass = 1;
  while (remaining > 0) {
    for (uint32_t i = 0; i < live; i++) {
      if (g.tasks[i].done()) continue;
      g.ctxs[i].pass = pass;
      // `resume()` re-checks what the loop just checked; this is the one call
      // that runs 5M times in a 1M-lane scan.
      g.tasks[i].resume_ready();
      if (g.tasks[i].done()) remaining--;
    }
    pass++;
  }
  g.tasks.clear();
}

void run_flat(const weft::KernelDef& k, uint32_t first, uint32_t stride, uint32_t ngroups,
              uint32_t tg, uint32_t grid, WorkerState& ws, void* const* bufs,
              uint32_t* warp, uint32_t* fold) {
  weft::SlotState dummy;
  dummy.bump = 0;
  dummy.count = 0;
  weft::Ctx c;
  c.shared = nullptr;
  c.slots = &dummy;
  c.bufs = bufs;
  c.ws = &ws;
  c.warp = warp;
  c.warp16 = (unsigned char*)warp + (size_t)2 * kMaxThreadgroup * sizeof(uint32_t);
  c.tg_size = tg;
  c.grid = grid;

  for (uint32_t g = first; g < ngroups; g += stride) {
    const uint32_t base = g * tg;
    const uint32_t live = (base + tg <= grid) ? tg : (grid > base ? grid - base : 0);
    for (uint32_t w = 0; w < (live + 31u) / 32u; w++) fold[(size_t)w * weft::kFoldWords] = 0u;
    for (uint32_t i = 0; i < live; i++) {
      c.gid = base + i;
      c.lid = i;
      c.tgid = g;
      c.slot = 0;
      c.fold = fold + (size_t)(i / 32u) * weft::kFoldWords;
      dummy.bump = 0;
      dummy.count = 0;
      k.body(c);
    }
  }
}

}  // namespace

namespace weft {

namespace {
constexpr size_t kCoroArenaBytes = 4 * 1024 * 1024;
thread_local unsigned char* t_coro_arena = nullptr;
thread_local size_t t_coro_used = 0;
}  // namespace

void* coro_alloc(std::size_t n) {
  n = (n + 15) & ~(std::size_t)15;
  if (!t_coro_arena) {
    t_coro_arena = (unsigned char*)std::aligned_alloc(64, kCoroArenaBytes);
    if (!t_coro_arena) {
      std::fprintf(stderr, "weft: out of memory for coroutine frames\n");
      std::abort();
    }
  }
  if (t_coro_used + n > kCoroArenaBytes) {
    std::fprintf(stderr, "weft: coroutine frames exceeded %zu bytes for one threadgroup\n",
                 kCoroArenaBytes);
    std::abort();
  }
  void* p = t_coro_arena + t_coro_used;
  t_coro_used += n;
  return p;
}

void coro_arena_reset() { t_coro_used = 0; }

void* Task::promise_type::operator new(std::size_t n) { return coro_alloc(n); }

void Task::promise_type::operator delete(void* p, std::size_t n) noexcept {
  (void)p;
  (void)n;  // the whole arena is released with the threadgroup
}


static uint64_t g_params[64];

void set_param(int index, uint64_t value) {
  if (index >= 0 && index < 64) g_params[index] = value;
}

uint64_t param(int index) { return (index >= 0 && index < 64) ? g_params[index] : 0ull; }

// Textures belong to a launch, like the scalar parameters: the host fills them
// once before dispatch and every threadgroup reads the same image.
static Texture g_textures[16];

void set_texture(int index, void* data, uint32_t width, uint32_t height, uint32_t elem) {
  if (index < 0 || index >= 16) return;
  g_textures[index].data = (unsigned char*)data;
  g_textures[index].width = width;
  g_textures[index].height = height;
  g_textures[index].elem = elem ? elem : 4u;
}

Texture texture(int index) {
  return (index >= 0 && index < 16) ? g_textures[index] : Texture{};
}

const Texture* texture_ptr(int index) {
  static const Texture kNone{};
  return (index >= 0 && index < 16) ? &g_textures[index] : &kNone;
}

// Samplers are launch state for the same reason textures are: on a GPU they are
// objects the host creates and binds, and a kernel cannot build one.
static Sampler g_samplers[16];

void set_sampler(int index, int filter, int addr) {
  if (index < 0 || index >= 16) return;
  g_samplers[index].filter = filter;
  g_samplers[index].addr = addr;
}

Sampler sampler(int index) { return (index >= 0 && index < 16) ? g_samplers[index] : Sampler{}; }

// A floating-point add has no atomic instruction on any of these cores, so it is
// a compare-and-swap loop. Everything below that *does* have one uses the
// builtin form: `__atomic_fetch_add` on arm64 is a single ldadd, while the
// obvious load/CAS loop is a load, a compare, a store, and a retry under
// contention — three times the cost for the same answer.
float atom_add_f32(float* p, float v) {
  uint32_t* bits = reinterpret_cast<uint32_t*>(p);
  uint32_t old = __atomic_load_n(bits, __ATOMIC_RELAXED);
  for (;;) {
    float of;
    std::memcpy(&of, &old, 4);
    float want = of + v;
    uint32_t wb;
    std::memcpy(&wb, &want, 4);
    if (__atomic_compare_exchange_n(bits, &old, wb, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return of;
  }
}

// min/max have no instruction either: a load, a compare and a conditional store
// is a read-modify-write that can lose an update, so they stay a CAS loop.

#define WEFT_ATOM_U32(name, expr)                                                       \
  uint32_t name(uint32_t* p, uint32_t v) {                                             \
    uint32_t old = __atomic_load_n(p, __ATOMIC_RELAXED);                               \
    for (;;) {                                                                         \
      uint32_t want = (expr);                                                          \
      if (__atomic_compare_exchange_n(p, &old, want, false, __ATOMIC_RELAXED,           \
                                      __ATOMIC_RELAXED))                               \
        return old;                                                                    \
    }                                                                                  \
  }

#define WEFT_ATOM_FETCH_U32(name, op)                                                   \
  uint32_t name(uint32_t* p, uint32_t v) {                                             \
    return __atomic_fetch_##op(p, v, __ATOMIC_RELAXED);                                \
  }

WEFT_ATOM_FETCH_U32(atom_add_u32, add)
WEFT_ATOM_FETCH_U32(atom_sub_u32, sub)
WEFT_ATOM_FETCH_U32(atom_and_u32, and)
WEFT_ATOM_FETCH_U32(atom_or_u32, or)
WEFT_ATOM_FETCH_U32(atom_xor_u32, xor)

uint32_t atom_exch_u32(uint32_t* p, uint32_t v) {
  return __atomic_exchange_n(p, v, __ATOMIC_RELAXED);
}

WEFT_ATOM_U32(atom_min_u32, old < v ? old : v)
WEFT_ATOM_U32(atom_max_u32, old > v ? old : v)

uint32_t atom_cas_u32(uint32_t* p, uint32_t cmp, uint32_t v) {
  __atomic_compare_exchange_n(p, &cmp, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return cmp;  // the value that was there
}

// 64-bit atomics. Same argument as the 32-bit ones: these are the only shared
// addresses two CPU threads touch at once, so they are real atomics. add/sub/
// and/or/xor/exch are a single instruction on arm64; min/max are a CAS loop.
#define WEFT_ATOM_U64(name, expr)                                                       \
  uint64_t name(uint64_t* p, uint64_t v) {                                             \
    uint64_t old = __atomic_load_n(p, __ATOMIC_RELAXED);                               \
    for (;;) {                                                                         \
      uint64_t want = (expr);                                                          \
      if (__atomic_compare_exchange_n(p, &old, want, false, __ATOMIC_RELAXED,           \
                                      __ATOMIC_RELAXED))                               \
        return old;                                                                    \
    }                                                                                  \
  }

#define WEFT_ATOM_FETCH_U64(name, op)                                                   \
  uint64_t name(uint64_t* p, uint64_t v) {                                             \
    return __atomic_fetch_##op(p, v, __ATOMIC_RELAXED);                                \
  }

WEFT_ATOM_FETCH_U64(atom_add_u64, add)
WEFT_ATOM_FETCH_U64(atom_sub_u64, sub)
WEFT_ATOM_FETCH_U64(atom_and_u64, and)
WEFT_ATOM_FETCH_U64(atom_or_u64, or)
WEFT_ATOM_FETCH_U64(atom_xor_u64, xor)

uint64_t atom_exch_u64(uint64_t* p, uint64_t v) {
  return __atomic_exchange_n(p, v, __ATOMIC_RELAXED);
}

WEFT_ATOM_U64(atom_min_u64, old < v ? old : v)
WEFT_ATOM_U64(atom_max_u64, old > v ? old : v)

uint64_t atom_cas_u64(uint64_t* p, uint64_t cmp, uint64_t v) {
  __atomic_compare_exchange_n(p, &cmp, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return cmp;
}

int64_t atom_add_s64(int64_t* p, int64_t v) {
  return (int64_t)atom_add_u64(reinterpret_cast<uint64_t*>(p), (uint64_t)v);
}

int64_t atom_min_s64(int64_t* p, int64_t v) {
  int64_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
  for (;;) {
    int64_t want = old < v ? old : v;
    if (__atomic_compare_exchange_n(p, &old, want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return old;
  }
}

int64_t atom_max_s64(int64_t* p, int64_t v) {
  int64_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
  for (;;) {
    int64_t want = old > v ? old : v;
    if (__atomic_compare_exchange_n(p, &old, want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return old;
  }
}

int32_t atom_add_s32(int32_t* p, int32_t v) {
  return (int32_t)atom_add_u32(reinterpret_cast<uint32_t*>(p), (uint32_t)v);
}

int32_t atom_min_s32(int32_t* p, int32_t v) {
  int32_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
  for (;;) {
    int32_t want = old < v ? old : v;
    if (__atomic_compare_exchange_n(p, &old, want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return old;
  }
}

int32_t atom_max_s32(int32_t* p, int32_t v) {
  int32_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
  for (;;) {
    int32_t want = old > v ? old : v;
    if (__atomic_compare_exchange_n(p, &old, want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return old;
  }
}

float atom_min_f32(float* p, float v) {
  uint32_t* bits = reinterpret_cast<uint32_t*>(p);
  uint32_t old = __atomic_load_n(bits, __ATOMIC_RELAXED);
  for (;;) {
    float of;
    std::memcpy(&of, &old, 4);
    float want_f = of < v ? of : v;
    uint32_t want;
    std::memcpy(&want, &want_f, 4);
    if (__atomic_compare_exchange_n(bits, &old, want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return of;
  }
}

float atom_max_f32(float* p, float v) {
  uint32_t* bits = reinterpret_cast<uint32_t*>(p);
  uint32_t old = __atomic_load_n(bits, __ATOMIC_RELAXED);
  for (;;) {
    float of;
    std::memcpy(&of, &old, 4);
    float want_f = of > v ? of : v;
    uint32_t want;
    std::memcpy(&want, &want_f, 4);
    if (__atomic_compare_exchange_n(bits, &old, want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return of;
  }
}

namespace {

// Dispatch threads that outlive a launch: creating and joining seven pthreads
// costs about a third of a 0.09 ms kernel, which is why `scale` measures faster
// on one thread than on eight.
struct DispPool {
  void (*fn)(void*, uint32_t) = nullptr;
  void* arg = nullptr;
  std::mutex m;
  std::condition_variable cv, done_cv;
  uint32_t next = 0, total = 0, finished = 0, gen = 0;
  std::vector<std::thread> threads;

  void ensure(uint32_t want) {
    while (threads.size() + 1 < want) threads.emplace_back([this] { worker(); });
  }

  void worker() {
    uint32_t seen = 0;
    for (;;) {
      void (*f)(void*, uint32_t);
      void* a;
      uint32_t id;
      {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return gen != seen; });
        seen = gen;
        if (next >= total) continue;
        id = next++;
        f = fn;
        a = arg;
      }
      f(a, id);
      std::lock_guard<std::mutex> lk(m);
      if (++finished == total) done_cv.notify_all();
    }
  }

  void run(void (*f)(void*, uint32_t), void* a, uint32_t n) {
    if (n == 0) return;
    if (n > 1) ensure(n);
    {
      std::lock_guard<std::mutex> lk(m);
      fn = f; arg = a; next = 0; total = n; finished = 0; gen++;
    }
    cv.notify_all();
    for (;;) {
      uint32_t id;
      {
        std::lock_guard<std::mutex> lk(m);
        if (next >= total) break;
        id = next++;
      }
      f(a, id);
      std::lock_guard<std::mutex> lk(m);
      if (++finished == n) break;
    }
    std::unique_lock<std::mutex> lk(m);
    done_cv.wait(lk, [&] { return finished == n; });
  }
};

// The work item is a lambda, so the job carries its exact type rather than
// forcing a std::function copy on every launch.
template <class F>
struct DispJob {
  F* fn;
};

template <class F>
void disp_thunk(void* p, uint32_t id) {
  (*static_cast<DispJob<F>*>(p)->fn)(id);
}

}  // namespace


void dispatch(const KernelDef& k, void* const* bufs, uint32_t grid, uint32_t tg) {
  if (grid == 0) return;
  if (tg == 0 || tg > kMaxThreadgroup) tg = 256;

  const uint32_t ngroups = (grid + tg - 1) / tg;

  uint32_t workers = std::thread::hardware_concurrency();
  if (const char* e = std::getenv("WEFT_THREADS")) {
    int v = std::atoi(e);
    if (v > 0) workers = (uint32_t)v;
  }
  if (workers == 0) workers = 1;
  if (workers > ngroups) workers = ngroups;

  auto work = [&](uint32_t id) {
    thread_local Pool pool;  // touched once per dispatch, not per thread
    WorkerState ws;
    // The warp-vector backend is opt-in: it removes the resume a shuffle costs,
    // but pays mask maintenance on every instruction, and on this machine that
    // only comes out ahead for arithmetic-heavy warps (see README).
    static const bool warp_ok = std::getenv("WEFT_WARP") != nullptr;
    // A kernel with no per-lane body has only the warp one: a tensor-core
    // instruction reads registers that other lanes hold, which a per-lane state
    // machine cannot express. It runs on the warp backend whatever the
    // environment says.
    if (k.warp && (warp_ok || !k.step)) {
      unsigned char* shared = pool.ensure_shared();
      unsigned char* states =
          (unsigned char*)pool.ensure_states((size_t)((tg + 31) / 32) * k.warp_size);
      std::vector<uint8_t> status;
      std::vector<weft::WCtx> ctxs;
      if (workers == 1) {
        for (uint32_t g = 0; g < ngroups; g++)
          run_group_warp(k, g, tg, grid, bufs, shared, states, status, ctxs, pool.ensure_red());
      } else {
        for (uint32_t g = id; g < ngroups; g += workers)
          run_group_warp(k, g, tg, grid, bufs, shared, states, status, ctxs, pool.ensure_red());
      }
    } else if (k.step) {
      unsigned char* shared = pool.ensure_shared();
      unsigned char* states = (unsigned char*)pool.ensure_states((size_t)tg * k.state_size);
      uint32_t* warp = pool.ensure_warp();
      uint32_t* red = pool.ensure_red();
      StepGroup g;
      g.ctxs.reserve(tg);
      uint32_t* const fold = pool.ensure_fold();
      if (workers == 1) {
        for (uint32_t grp = 0; grp < ngroups; grp++)
          run_group_step(k, grp, tg, grid, ws, bufs, shared, states, warp, red, fold, g);
      } else {
        for (uint32_t grp = id; grp < ngroups; grp += workers)
          run_group_step(k, grp, tg, grid, ws, bufs, shared, states, warp, red, fold, g);
      }
    } else if (k.coro) {
      unsigned char* shared = pool.ensure_shared();
      CoroGroup g;
      g.tasks.reserve(tg);
      uint32_t* const fold = pool.ensure_fold();
      if (workers == 1) {
        for (uint32_t grp = 0; grp < ngroups; grp++)
          run_group_coro(k, grp, tg, grid, ws, bufs, shared, g, pool.ensure_warp(), fold);
      } else {
        for (uint32_t grp = id; grp < ngroups; grp += workers)
          run_group_coro(k, grp, tg, grid, ws, bufs, shared, g, pool.ensure_warp(), fold);
      }
    } else if (k.flat) {
      // Contiguous slices: one loop per worker, no per-thread call.
      weft::SlotState dummy;
      dummy.bump = 0;
      dummy.count = 0;
      weft::Ctx c;
      c.slots = &dummy;
      c.bufs = bufs;
      c.ws = &ws;
      c.warp = pool.ensure_warp();
      c.warp16 = (unsigned char*)c.warp + (size_t)2 * kMaxThreadgroup * sizeof(uint32_t);
      c.tg_size = tg;
      c.grid = grid;
      const uint32_t per = (grid + workers - 1) / workers;
      const uint32_t b = id * per < grid ? id * per : grid;
      const uint32_t e = (b + per < grid) ? b + per : grid;
      if (b < e) k.flat(c, b, e);
    } else {
      run_flat(k, id, workers, ngroups, tg, grid, ws, bufs, pool.ensure_warp(), pool.ensure_fold());
    }
  };

  if (workers == 1) {
    work(0);
    return;
  }

  static DispPool* pool = new DispPool();
  DispJob<decltype(work)> job{&work};
  pool->run(&disp_thunk<decltype(work)>, &job, workers);
}

}  // namespace weft
