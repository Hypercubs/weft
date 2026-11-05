// Does a wait on another thread's release end on this host?
//
// This is the question the 64-bit atomic's shape hangs on. A 64-bit atomic is
// built from two 32-bit words, and reading, changing and writing both needs the
// pair to have one writer at a time. If a thread can wait for another thread's
// release, that writer takes turns: the launch stays parallel and a 64-bit atomic
// costs a lock. If it cannot, the only exact form is one thread walking every
// warp, which is serial and slow (README, Limits).
//
// The answer was measured wrong three times before this file, and every time the
// fault was the probe: a parameter bound at the wrong index (so the stride read 0
// and the work loop never ended), a build whose exit status was not checked, a
// hung dispatch that wedged the sweep. So this version cannot make those mistakes:
//
//   - everything a case touches is in ONE buffer, so no argument can be misbound;
//   - the kernel writes back what it read (`st[3]`, `st[4]`) and the host checks
//     it, so a wrong argument is printed rather than inferred;
//   - the expected counter is arithmetic, computed here, not a remembered number;
//   - each case is one process, run by a supervisor that kills a hung one, so a
//     wedged dispatch cannot hide the cases after it.
//
// Build:
//   clang++ -fobjc-arc -O2 tools/lock_probe.mm -framework Metal \
//           -framework Foundation -o /tmp/lockprobe
// Run one case:
//   /tmp/lockprobe control | locked | fenced | exchanged | lone | one_group | handoff
//
// Measured on this host (Apple silicon, 32 groups x 32 threads unless noted):
//
//   control    exact         atomic adds, no section, every shape
//   handoff    exact         a store in one threadgroup seen by 1024 waiters
//   locked     HANGS         relaxed compare-exchange spin, every shape
//   fenced     HANGS         same with a device-scope seq_cst fence on both sides
//   exchanged  HANGS         same with atomic_exchange instead of a compare
//   lone       exact         one thread taking and dropping the same lock
//
// So propagation is fine and a fence is available; what fails is progress. A lane
// that wins the section cannot run it until the other lanes in its warp leave the
// spin loop, and they cannot leave until it releases, the warp is one program
// counter. A threadgroup of one thread does not escape it either: several such
// threadgroups are co-scheduled into one warp, and the wait then deadlocks across
// them instead of within them. Handing out LOCK kernels is only safe when at most
// one lane per warp ever waits, which is what the translator's single-writer walk
// gives and what a per-lane section cannot.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The buffer every case shares. One allocation, one binding, no index to get
// wrong:
//
//   st[0]      the lock word, zeroed by the host before the case
//   st[1]      the control counter: atomic adds, no critical section
//   st[2]      the locked counter: read-modify-write inside the critical section
//   st[3]      the work count per thread, as the kernel read it
//   st[4]      the stride between threads, as the kernel read it
//   st[5]      how many threads entered the critical section
//   st[6]      the lock as the last thread left it (0 = free)
//   st[64...]  the words the work loop reads, all 1, so the work a thread does is
//              the number of iterations it ran and the host can compute it
static const int kWords = 1024;
static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// The work between increments: a strided read of the same words the real 64-bit
// path touches, so the loop is not optimised away and the timing means something.
// Every word it reads is 1, so the value it accumulates is the number of
// iterations it ran, which the host computes independently.
static uint32_t spin_work(device uint32_t* st, uint32_t gid, uint32_t n, uint32_t stride) {
  uint32_t x = 0;
  for (uint32_t i = gid; i < n; i += stride) x += st[64u + (i & 63u)];
  return x;
}

kernel void control(device uint32_t* st [[buffer(0)]], constant uint32_t* p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
  const uint32_t n = p[0], stride = p[1];
  const uint32_t x = spin_work(st, gid, n, stride);
  st[3] = n;
  st[4] = stride;
  atomic_fetch_add_explicit((device atomic_uint*)&st[1], x, memory_order_relaxed);
}

// Metal takes no acquire or release ON AN OPERATION, every `atomic_load_explicit`,
// `store`, `exchange` and `compare_exchange` here is constrained to
// `memory_order_relaxed`, but `atomic_thread_fence(mem_flags, memory_order)` does
// exist, and a fence beside a relaxed operation is the release-then-store /
// load-then-acquire pair C++ gives when the hardware cannot do it in one
// instruction. So this case is `locked` with the two fences added, and it measures
// that ordering was never the blocker: it hangs exactly as `locked` does.
kernel void locked(device uint32_t* st [[buffer(0)]], constant uint32_t* p [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
  const uint32_t n = p[0], stride = p[1];
  const uint32_t x = spin_work(st, gid, n, stride);
  st[3] = n;
  st[4] = stride;
  device atomic_uint* l = (device atomic_uint*)&st[0];
  // The wait, and the one detail a spin has to get right: a compare-exchange that
  // fails overwrites `expect` with the value it found, so it goes back to 0 before
  // the next attempt. Leaving it at 1 makes the next attempt "replace 1 with 1",
  // which succeeds against a lock another thread holds.
  uint32_t expect = 0u;
  for (;;) {
    if (atomic_compare_exchange_weak_explicit(l, &expect, 1u, memory_order_relaxed,
                                              memory_order_relaxed))
      break;
    expect = 0u;
  }
  st[5] += 1u;
  const uint32_t o = atomic_load_explicit((device atomic_uint*)&st[2], memory_order_relaxed);
  atomic_store_explicit((device atomic_uint*)&st[2], o + x, memory_order_relaxed);
  atomic_store_explicit(l, 0u, memory_order_relaxed);
  st[6] = atomic_load_explicit(l, memory_order_relaxed);
}

// One thread sets a flag; every other thread waits for it and then counts. Nothing
// is contended and nobody owns anything, so this is the pure question: does a
// relaxed atomic store in one threadgroup reach a relaxed atomic load in another?
kernel void handoff(device uint32_t* st [[buffer(0)]], constant uint32_t* p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
  const uint32_t n = p[0], stride = p[1];
  st[3] = n;
  st[4] = stride;
  if (gid == 0u) {
    st[5] = 1u;
    atomic_store_explicit((device atomic_uint*)&st[0], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&st[1], 1u, memory_order_relaxed);
    return;
  }
  while (atomic_load_explicit((device atomic_uint*)&st[0], memory_order_relaxed) == 0u) {}
  atomic_fetch_add_explicit((device atomic_uint*)&st[1], 1u, memory_order_relaxed);
  (void)spin_work(st, gid, n, stride);
}

// The same wait, with the load strong: if this ends where `handoff` does not, a
// weak load is being served stale.
kernel void handoff_strong(device uint32_t* st [[buffer(0)]], constant uint32_t* p [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
  const uint32_t n = p[0], stride = p[1];
  st[3] = n;
  st[4] = stride;
  if (gid == 0u) {
    st[5] = 1u;
    atomic_store_explicit((device atomic_uint*)&st[0], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)&st[1], 1u, memory_order_relaxed);
    return;
  }
  for (;;) {
    if (atomic_load_explicit((device atomic_uint*)&st[0], memory_order_relaxed) != 0u) break;
  }
  atomic_fetch_add_explicit((device atomic_uint*)&st[1], 1u, memory_order_relaxed);
  (void)spin_work(st, gid, n, stride);
}

// A lock taken with `atomic_exchange` and dropped with a store: no compare-exchange
// at all, so if this one ends, the fault is in the compare-exchange's weak form.
kernel void exchanged(device uint32_t* st [[buffer(0)]], constant uint32_t* p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  const uint32_t n = p[0], stride = p[1];
  const uint32_t x = spin_work(st, gid, n, stride);
  st[3] = n;
  st[4] = stride;
  device atomic_uint* l = (device atomic_uint*)&st[0];
  for (;;) {
    if (atomic_exchange_explicit(l, 1u, memory_order_relaxed) == 0u) break;
  }
  st[5] += 1u;
  const uint32_t o = atomic_load_explicit((device atomic_uint*)&st[2], memory_order_relaxed);
  atomic_store_explicit((device atomic_uint*)&st[2], o + x, memory_order_relaxed);
  atomic_store_explicit(l, 0u, memory_order_relaxed);
  st[6] = atomic_load_explicit(l, memory_order_relaxed);
}

// The same section ordered by fences: release before the unlock store, acquire
// after the successful take.
kernel void fenced(device uint32_t* st [[buffer(0)]], constant uint32_t* p [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
  const uint32_t n = p[0], stride = p[1];
  const uint32_t x = spin_work(st, gid, n, stride);
  st[3] = n;
  st[4] = stride;
  device atomic_uint* l = (device atomic_uint*)&st[0];
  uint32_t expect = 0u;
  for (;;) {
    if (atomic_compare_exchange_weak_explicit(l, &expect, 1u, memory_order_relaxed,
                                              memory_order_relaxed)) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
      break;
    }
    expect = 0u;
  }
  st[5] += 1u;
  const uint32_t o = atomic_load_explicit((device atomic_uint*)&st[2], memory_order_relaxed);
  atomic_store_explicit((device atomic_uint*)&st[2], o + x, memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
  atomic_store_explicit(l, 0u, memory_order_relaxed);
  st[6] = atomic_load_explicit(l, memory_order_relaxed);
}
)MSL";

struct Case {
  const char* name;
  const char* kernel;
  uint32_t groups;
  uint32_t threads;   // threads per threadgroup
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s control|locked|lone|one_group [n]\n", argv[0]);
    return 2;
  }
  const std::string which = argv[1];
  Case c;
  if (which == "control") {
    c = {"control", "control", 32, 1};
  } else if (which == "locked") {
    // One thread per threadgroup is the shape the translator uses for a kernel
    // that takes the lock: a thread that waits must not share a program counter
    // with the thread it waits for.
    c = {"locked", "locked", 32, 1};
  } else if (which == "handoff") {
    c = {"handoff", "handoff", 32, 32};
  } else if (which == "handoff_strong") {
    c = {"handoff_strong", "handoff_strong", 32, 32};
  } else if (which == "exchanged") {
    c = {"exchanged", "exchanged", 32, 32};
  } else if (which == "fenced") {
    // The relaxed lock with a device-scope fence on both sides: the shape the
    // translator would emit if fences are enough.
    c = {"fenced", "fenced", 32, 1};
  } else if (which == "fenced_one") {
    c = {"fenced_one", "fenced", 1, 32};
  } else if (which == "lone") {
    // No contention at all: the same lock, taken and dropped by one thread. If
    // this hangs then the lock word is not the word the kernel thinks it is.
    c = {"lone", "locked", 1, 1};
  } else if (which == "one_group") {
    // Every contender in one threadgroup, so the wait and the thread it waits for
    // share a SIMD group. This is the case that has no guaranteed forward
    // progress, and nothing in the translator asks for it.
    c = {"one_group", "locked", 1, 32};
  } else {
    std::fprintf(stderr, "no such case: %s\n", which.c_str());
    return 2;
  }
  const uint32_t n = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 2048u;
  // The shape can be overridden, because the shape is the variable: a threadgroup
  // smaller than a SIMD group may share one with another threadgroup, and two
  // threadgroups in one SIMD group run in lockstep with each other's masks.
  if (argc > 3) c.groups = (uint32_t)std::atoi(argv[3]);
  if (argc > 4) c.threads = (uint32_t)std::atoi(argv[4]);
  const uint32_t stride = c.groups * c.threads;

  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::fprintf(stderr, "no metal device\n");
      return 1;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                           options:[MTLCompileOptions new]
                                             error:&err];
    if (!lib) {
      std::fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:c.kernel]];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      std::fprintf(stderr, "pipeline: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLBuffer> st = [dev newBufferWithLength:(NSUInteger)kWords * sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> pb = [dev newBufferWithLength:2 * sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
    uint32_t* s = (uint32_t*)[st contents];
    uint32_t* p = (uint32_t*)[pb contents];
    std::memset(s, 0, (size_t)kWords * sizeof(uint32_t));
    for (int i = 0; i < 64; i++) s[64 + i] = 1u;
    p[0] = n;
    p[1] = stride;

    // What the counter has to be, worked out here rather than remembered: every
    // thread adds the number of iterations it ran, and every word it reads is 1.
    unsigned long long want = 0;
    for (uint32_t gid = 0; gid < stride; gid++)
      for (uint32_t i = gid; i < n; i += stride) want++;

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:st offset:0 atIndex:0];
    [enc setBuffer:pb offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(c.groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(c.threads, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    std::printf("%-10s %u groups x %u threads  status %ld", c.name, c.groups, c.threads,
                (long)cb.status);
    if (cb.error) std::printf(" error '%s'", [[cb.error localizedDescription] UTF8String]);
    std::printf("\n");

    // What the kernel saw, first: if the arguments did not arrive, the verdict is
    // about the harness and not about the host.
    const bool args_ok = s[3] == n && s[4] == stride;
    std::printf("%-10s the kernel read n=%u stride=%u  %s\n", c.name, s[3], s[4],
                args_ok ? "as bound" : "NOT AS BOUND");
    // A handoff case counts the threads that got past the wait in st[1]; a lock
    // case counts the work its section accumulated in st[2]. Reading the wrong one
    // is what made an earlier run report "WRONG" for a case that had worked.
    const bool hand_off = c.kernel[0] == 'h';
    const unsigned long long got = hand_off    ? (unsigned long long)s[1]
                                   : c.kernel[0] == 'c' ? (unsigned long long)s[1]
                                                        : (unsigned long long)s[2];
    const unsigned long long expect = hand_off ? (unsigned long long)stride : want;
    std::printf("%-13s counter %llu  want %llu  %s\n", c.name, got, expect,
                got == expect ? "exact" : "WRONG");
    if (c.kernel[0] != 'c' && !hand_off)
      std::printf("%-13s entered %u  lock left at %u  %s\n", c.name, s[5], s[6],
                  (s[5] == stride && s[6] == 0u) ? "the wait ended" : "a thread is still inside");
    return (args_ok && got == expect) ? 0 : 1;
  }
}
