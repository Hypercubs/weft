// Does a wait on another thread's release end on this host?
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int kWords = 1024;
static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// path touches, so the loop is not optimised away and the timing means something.
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

kernel void locked(device uint32_t* st [[buffer(0)]], constant uint32_t* p [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
  const uint32_t n = p[0], stride = p[1];
  const uint32_t x = spin_work(st, gid, n, stride);
  st[3] = n;
  st[4] = stride;
  device atomic_uint* l = (device atomic_uint*)&st[0];
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
    c = {"locked", "locked", 32, 1};
  } else if (which == "handoff") {
    c = {"handoff", "handoff", 32, 32};
  } else if (which == "handoff_strong") {
    c = {"handoff_strong", "handoff_strong", 32, 32};
  } else if (which == "exchanged") {
    c = {"exchanged", "exchanged", 32, 32};
  } else if (which == "fenced") {
    c = {"fenced", "fenced", 32, 1};
  } else if (which == "fenced_one") {
    c = {"fenced_one", "fenced", 1, 32};
  } else if (which == "lone") {
    c = {"lone", "locked", 1, 1};
  } else if (which == "one_group") {
    c = {"one_group", "locked", 1, 32};
  } else {
    std::fprintf(stderr, "no such case: %s\n", which.c_str());
    return 2;
  }
  const uint32_t n = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 2048u;
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

    const bool args_ok = s[3] == n && s[4] == stride;
    std::printf("%-10s the kernel read n=%u stride=%u  %s\n", c.name, s[3], s[4],
                args_ok ? "as bound" : "NOT AS BOUND");
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
