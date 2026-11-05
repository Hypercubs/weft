// Does this host's Metal compiler have the runtime calls an emitted PTX kernel
// needs?
//
// The PTX front end targets a small runtime: buffer addresses, atomics,
// shuffles, ballots, threadgroup memory. Emitting MSL means those calls have to
// exist with the same names and the same answers, so this probe asks the
// compiler rather than assuming, the same way tools/imageblock_probe.mm asks
// about imageblocks.
//
// Build: clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//          tools/msl_runtime_probe.mm -o /tmp/rtprobe && /tmp/rtprobe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

static void probe(const char* name, const char* body) {
  NSString* src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%s", body];
  NSError* err = nil;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  MTLCompileOptions* o = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
  std::printf("%-30s %s\n", name, lib ? "ACCEPTED" : "refused");
  if (!lib) {
    const char* msg = [[err localizedDescription] UTF8String];
    int n = 0;
    for (const char* p = msg; *p && n < 230; p++, n++) std::putchar(*p);
    std::printf("\n");
  }
}

int main() {
  @autoreleasepool {
    probe("atomic_uint add/sub/min/max", R"(
kernel void k(device uint* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_uint* a = (device atomic_uint*)&p[gid];
  uint x = atomic_fetch_add_explicit(a, 1u, memory_order_relaxed);
  uint y = atomic_fetch_and_explicit(a, 3u, memory_order_relaxed);
  p[gid] = x + y;
}
)");

    probe("atomic_uint cas loop", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_uint* a = (device atomic_uint*)&p[gid];
  uint old = atomic_load_explicit(a, memory_order_relaxed);
  for (;;) { float nf = as_type<float>(old) + 1.0f;
    if (atomic_compare_exchange_weak_explicit(a, &old, as_type<uint>(nf),
        memory_order_relaxed, memory_order_relaxed)) break; }
}
)");

    probe("atomic_ulong 64-bit", R"(
kernel void k(device ulong* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[gid];
  ulong x = atomic_fetch_add_explicit(a, 1ul, memory_order_relaxed);
  p[gid] = x;
}
)");

    probe("ulong cas only", R"(
kernel void k(device ulong* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[gid];
  ulong old = atomic_load_explicit(a, memory_order_relaxed);
  (void)atomic_compare_exchange_weak_explicit(a, &old, old + 1ul,
      memory_order_relaxed, memory_order_relaxed);
}
)");

    probe("simd_shuffle(uint) + ballot", R"(
kernel void k(device uint* p [[buffer(0)]], uint gid [[thread_position_in_grid]],
              uint lid [[thread_index_in_threadgroup]]) {
  uint v = p[gid];
  uint s = simd_shuffle(v, (ushort)((lid + 1u) & 31u));
  uint m = simd_ballot(v != 0u).to_uint();
  bool any = simd_any(v != 0u);
  bool all = simd_all(v != 0u);
  p[gid] = s + m + (any ? 1u : 0u) + (all ? 2u : 0u);
}
)");

    probe("threadgroup array + dynamic index", R"(
struct WeftCtx { uint lid, tgid, tg_size; device ulong* addrs; };
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]],
              uint lid [[thread_index_in_threadgroup]], uint3 tgsz [[threads_per_threadgroup]]) {
  threadgroup uint8_t arena[4096];
  thread WeftCtx c;
  c.lid = lid; c.tgid = 0u; c.tg_size = tgsz.x; c.addrs = (device ulong*)p;
  ((threadgroup uint*)arena)[lid] = (uint)c.addrs[0];
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  p[gid] = (float)((threadgroup uint*)arena)[63u - lid];
}
)");

    probe("scalar math: rint ldexp isnan", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  float x = p[gid];
  p[gid] = rint(x) + ldexp(1.0f, 3) + (isnan(x) ? 1.0f : 0.0f) + INFINITY * 0.0f;
}
)");

    probe("_Float16 as_type round trip", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  _Float16 h = (_Float16)p[gid];
  ushort u = as_type<ushort>(h);
  _Float16 back = as_type<_Float16>(u);
  p[gid] = (float)back;
}
)");

    probe("uint64 device address cast", R"(
static inline float load_at(ulong a) { return *((device float*)(uintptr_t)a); }
kernel void k(device uint64_t* addrs [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  ulong a = addrs[0] + (ulong)gid * 4ul;
  ((device float*)(uintptr_t)(a))[0] = load_at(a) + 1.0f;
}
)");
  }
  return 0;
}
