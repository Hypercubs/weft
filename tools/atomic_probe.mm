// Does a spinlock made of this host's Metal atomics actually make progress?
//
// The warp target wants a lock for a 64-bit atomic: one thread holds all 32
// lanes, so the lanes never contend, and what is left is another warp on the same
// counter. That is only sound if a spinning thread can observe the holder's
// release, so this asks the hardware directly, with a bounded spin so a broken
// primitive reports rather than hangs.
//
// Build: clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//          tools/atomic_probe.mm -o /tmp/atomicprobe && /tmp/atomicprobe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

// A compare-exchange lock, with the spin counted instead of unbounded.
kernel void cas_lock(device uint* locks [[buffer(0)]], device uint* out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  device atomic_uint* s = (device atomic_uint*)&locks[0];
  bool got = false;
  uint spins = 0;
  while (!got && spins < 4000000u) {
    uint expect = 0u;
    got = atomic_compare_exchange_weak_explicit(s, &expect, 1u, memory_order_relaxed,
                                                memory_order_relaxed);
    spins++;
  }
  if (got) {
    const uint seen = atomic_load_explicit(s, memory_order_relaxed);
    out[gid] = seen;                       // must be 1: the lock was ours
    atomic_store_explicit(s, 0u, memory_order_relaxed);
  } else {
    out[gid] = 0xDEADu;
  }
  out[256u + gid] = spins;
}

// The same lock built from fetch_add, in case compare-exchange is the part this
// host gets wrong.
kernel void add_lock(device uint* locks [[buffer(0)]], device uint* out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  device atomic_uint* s = (device atomic_uint*)&locks[0];
  bool got = false;
  uint spins = 0;
  while (!got && spins < 4000000u) {
    const uint v = atomic_fetch_add_explicit(s, 1u, memory_order_relaxed);
    spins++;
    if (v == 0u) got = true;
    else (void)atomic_fetch_sub_explicit(s, 1u, memory_order_relaxed);
  }
  if (got) {
    out[gid] = 1u;
    atomic_store_explicit(s, 0u, memory_order_relaxed);
  } else {
    out[gid] = 0xDEADu;
  }
  out[256u + gid] = spins;
}
)METAL";

static void run(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                const char* label, uint32_t grid, uint32_t threads) {
  id<MTLBuffer> locks = [dev newBufferWithLength:256 * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> out = [dev newBufferWithLength:512 * sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
  memset(locks.contents, 0, 256 * sizeof(uint32_t));
  memset(out.contents, 0, 512 * sizeof(uint32_t));
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:locks offset:0 atIndex:0];
  [e setBuffer:out offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake((grid + threads - 1) / threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  const uint32_t* o = (const uint32_t*)out.contents;
  uint32_t ok = 0, dead = 0, maxspins = 0;
  for (uint32_t i = 0; i < grid; i++) {
    if (o[i] == 1u) ok++;
    else if (o[i] == 0xDEADu) dead++;
    if (o[256 + i] > maxspins) maxspins = o[256 + i];
  }
  std::printf("%-22s grid %-4u threads %-3u  acquired %u  gave up %u  max spins %u\n", label,
              grid, threads, ok, dead, maxspins);
}

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("no device\n"); return 1; }
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:opts error:&err];
    if (!lib) {
      std::printf("compile: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    const char* names[2] = {"cas_lock", "add_lock"};
    for (int n = 0; n < 2; n++) {
      id<MTLComputePipelineState> pso =
          [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(names[n])]
                                             error:&err];
      if (!pso) { std::printf("%s: %s\n", names[n], err.localizedDescription.UTF8String); continue; }
      run(dev, q, pso, names[n], 8, 8);
      run(dev, q, pso, names[n], 64, 64);
      run(dev, q, pso, names[n], 1024, 256);
    }
  }
  return 0;
}
