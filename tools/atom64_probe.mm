// What does this host's Metal actually have for a 64-bit atomic?
//
// The header is misleading in both directions, so this asks the compiler and the
// device rather than reading it. What it asks, one entry point at a time:
//
//   - `atomic_ulong` is *declared* (metal_atomic, behind `__HAVE_ATOMIC_ULONG__`).
//   - `atomic_fetch_add_explicit` and the other fetching forms are SFINAE'd to
//     32-bit pointer types (`_valid_fetch_*_type`), so they do not exist for
//     `ulong` at all — a compile error, not a hardware answer.
//   - `atomic_max_explicit` / `atomic_min_explicit` on `ulong` *do* compile and
//     the pipeline for them is refused.
//   - `atomic_load` / `store` / `exchange` / `compare_exchange` look generic over
//     `_atomic<T>`, but each is constrained to 32-bit pointer types as well, so a
//     `ulong` atomic has no operation at all except max/min. CAS is what every
//     other 64-bit atomic would be built out of, so its absence is the answer to
//     "can this be done with an instruction": no. The case that does run at the
//     end is the 32-bit `fetch_add` over the same 4096 threads, printing its
//     total, so an accepted pipeline is not mistaken for a correct one.
//
// A refusal that only exists at pipeline creation is exactly the kind that looks
// like a compile error from a distance, which is why the answer is measured here
// and not inferred.
//
// Build: clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//          tools/atom64_probe.mm -o /tmp/atom64probe && /tmp/atom64probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>

// Compiles `body` and, when it compiles, builds its pipeline too — the step where
// a 64-bit atomic is actually refused.
//
// `threads` > 0 also runs the pipeline that many threads wide and prints the
// first word of the buffer afterwards, so a case that is accepted can be checked
// against the answer it is supposed to produce.
static void probe(id<MTLDevice> dev, const char* label, const char* body, bool make_pipeline,
                  unsigned threads = 0) {
  NSString* src = [NSString stringWithFormat:
                                @"#include <metal_stdlib>\nusing namespace metal;\n%s", body];
  NSError* err = nil;
  id<MTLLibrary> lib = nil;
  // The compiler is a service, and it dies under load often enough that a single
  // failure is not an answer: `weft` retries it three times for the same reason.
  // Only a refusal the service actually *said* is printed as one.
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
  // The counter the CAS loop builds should equal the number of threads that ran
  // past its guard, which is `threads` exactly.
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

    // ---- the fetching forms: SFINAE'd to 32-bit, so these should not compile ----
    probe(dev, "atomic_ulong fetch_add", R"(
kernel void k(device ulong* p [[buffer(0)]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  ulong old = atomic_fetch_add_explicit(a, 1ul, memory_order_relaxed);
  p[1] = old;
}
)", true);

    // ---- min/max on ulong: declared, compiled, and refused by the device ----
    probe(dev, "atomic_ulong max", R"(
kernel void k(device ulong* p [[buffer(0)]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  atomic_max_explicit(a, 5ul, memory_order_relaxed);
}
)", true);

    // ---- the generic-over-_atomic<T> forms ----
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

    // ---- the decisive one: a public 64-bit CAS ----
    probe(dev, "atomic_ulong cas (compile)", R"(
kernel void k(device ulong* p [[buffer(0)]]) {
  device atomic_ulong* a = (device atomic_ulong*)&p[0];
  ulong expect = 0ul;
  bool ok = atomic_compare_exchange_weak_explicit(a, &expect, 1ul,
                     memory_order_relaxed, memory_order_relaxed);
  p[1] = ok ? 1ul : 0ul;
}
)", true);

    // ...and a CAS loop that has to give the exact total, which is what every
    // unsigned 64-bit atomic operation can be written as.
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

    // The 32-bit form, for contrast: it has a pipeline, which is why a 64-bit one
    // being accepted by the compiler is not evidence of much.
    probe(dev, "atomic_uint fetch_add (contrast)", R"(
kernel void k(device uint* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  device atomic_uint* a = (device atomic_uint*)&p[0];
  atomic_fetch_add_explicit(a, 1u, memory_order_relaxed);
}
)", true, 4096);
  }
  return 0;
}
