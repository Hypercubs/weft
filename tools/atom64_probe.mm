// What does this host's Metal actually have for a 64-bit atomic?
//   - `atomic_ulong` is *declared* (metal_atomic, behind `__HAVE_ATOMIC_ULONG__`).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>

static void probe(id<MTLDevice> dev, const char* label, const char* body, bool make_pipeline,
                  unsigned threads = 0) {
  NSString* src = [NSString stringWithFormat:
                                @"#include <metal_stdlib>\nusing namespace metal;\n%s", body];
  NSError* err = nil;
  id<MTLLibrary> lib = nil;
  for (int attempt = 0; attempt < 4 && !lib; attempt++) {
    err = nil;
    lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if (lib) break;
    NSString* d = [err localizedDescription];
    if (d && ([d containsString:@"XPC"] || [d containsString:@"interrupted"])) continue;
    break;
  }
  if (!lib) {
    NSString* d = [err localizedDescription];
    const char* m = d ? [d UTF8String] : "(no message)";
    const bool flake = [d containsString:@"XPC"] || [d containsString:@"interrupted"];
    std::printf("%-34s compile   %s: ", label, flake ? "no answer (compiler died)" : "refused");
    int n = 0;
    for (const char* p = m; *p && n < 150; p++, n++) std::putchar(*p == '\n' ? '|' : *p);
    std::printf("\n");
    return;
  }
  if (!make_pipeline) {
    std::printf("%-34s compile   accepted\n", label);
    return;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:@"k"];
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    NSString* d = [err localizedDescription];
    std::printf("%-34s pipeline  refused: %s\n", label, d ? [d UTF8String] : "(no message)");
    return;
  }
  if (threads == 0) {
    std::printf("%-34s pipeline  accepted\n", label);
    return;
  }

  // Accepted, so run it and print what the buffer holds.
  const size_t len = 1u << 18;
  id<MTLBuffer> buf = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
  unsigned char* base = (unsigned char*)[buf contents];
  for (size_t i = 0; i < len; i++) base[i] = 0;
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:buf offset:0 atIndex:0];
  const MTLSize tg = MTLSizeMake(32, 1, 1);
  [e dispatchThreadgroups:MTLSizeMake((threads + 31) / 32, 1, 1) threadsPerThreadgroup:tg];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if ([cb error]) {
    std::printf("%-34s run       error: %s\n", label,
                [[[cb error] localizedDescription] UTF8String]);
    return;
  }
  unsigned long long w0 = 0;
  std::memcpy(&w0, base, 8);
  std::printf("%-34s runs:     counter = %llu  (expected %u)  %s\n", label, w0, threads,
              w0 == (unsigned long long)threads ? "exact" : "WRONG");
}

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::printf("no device\n");
      return 1;
    }
    std::printf("device: %s\n\n", [[dev name] UTF8String]);

    probe(dev, "atomic_ulong fetch_add", R"(
kernel void k(device ulong* p [[buffer(0)]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  ulong old = atomic_fetch_add_explicit(a, 1ul, memory_order_relaxed);
  p[1] = old;
}
)", true);

    probe(dev, "atomic_ulong max", R"(
kernel void k(device ulong* p [[buffer(0)]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  atomic_max_explicit(a, 5ul, memory_order_relaxed);
}
)", true);

    probe(dev, "atomic_ulong load", R"(
kernel void k(device ulong* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  p[1 + gid] = atomic_load_explicit(a, memory_order_relaxed);
}
)", true, 64);

    probe(dev, "atomic_ulong store", R"(
kernel void k(device ulong* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  atomic_store_explicit(a, 1ul, memory_order_relaxed);
}
)", true);

    probe(dev, "atomic_ulong exchange", R"(
kernel void k(device ulong* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  ulong old = atomic_exchange_explicit(a, 1ul, memory_order_relaxed);
  p[1 + gid] = old;
}
)", true);

    probe(dev, "atomic_ulong cas (compile)", R"(
kernel void k(device ulong* p [[buffer(0)]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  ulong expect = 0ul;
  bool ok = atomic_compare_exchange_weak_explicit(a, &expect, 1ul,
                     memory_order_relaxed, memory_order_relaxed);
  p[1] = ok ? 1ul : 0ul;
}
)", true);

    probe(dev, "atomic_ulong cas loop count", R"(
kernel void k(device ulong* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  ulong old = atomic_load_explicit(a, memory_order_relaxed);
  for (;;) {
    ulong next = old + 1ul;
    if (atomic_compare_exchange_weak_explicit(a, &old, next,
          memory_order_relaxed, memory_order_relaxed)) break;
  }
}
)", true, 4096);

    // A threadgroup-scope copy, in case the device scope is the part refused.
    probe(dev, "ulong cas threadgroup scope", R"(
kernel void k(device ulong* p [[buffer(0)]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_ulong t;
  if (lid == 0u) atomic_store_explicit(&t, 0ul, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong expect = atomic_load_explicit(&t, memory_order_relaxed);
  for (;;) {
    if (atomic_compare_exchange_weak_explicit(&t, &expect, expect + 1ul,
          memory_order_relaxed, memory_order_relaxed)) break;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) p[0] = atomic_load_explicit(&t, memory_order_relaxed);
}
)", true, 32);

    probe(dev, "atomic_uint fetch_add (contrast)", R"(
kernel void k(device uint* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_uint* a = (device atomic_uint*)&p[0];
  atomic_fetch_add_explicit(a, 1u, memory_order_relaxed);
}
)", true, 4096);
  }
  return 0;
}
