// Which MSL constructs can a generated kernel use?
//
// The PTX and SPIR-V front ends emit a resumable state machine — `switch (pc)`,
// labels, `goto`, and untyped temporaries — and a Metal emitter has to keep the
// same shape to stay a mechanical rewrite instead of a second backend. That is
// only possible if the compiler accepts those constructs, so this probe asks it
// directly with the runtime compiler weft hands its source to.
//
// Build: clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//          tools/msl_probe.mm -o /tmp/mslprobe && /tmp/mslprobe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

static const char* kHead = "#include <metal_stdlib>\nusing namespace metal;\n";

static void probe(const char* name, const char* body) {
  NSString* src = [NSString stringWithFormat:@"%s%s", kHead, body];
  NSError* err = nil;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  MTLCompileOptions* o = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
  std::printf("%-34s %s\n", name, lib ? "ACCEPTED" : "refused");
  if (!lib) {
    const char* msg = [[err localizedDescription] UTF8String];
    int n = 0;
    for (const char* p = msg; *p && n < 260; p++, n++) std::putchar(*p);
    std::printf("\n");
  }
}

int main() {
  @autoreleasepool {
    probe("switch + goto + label", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  uint pc = 0;
  switch (pc) { case 1: goto R1; case 2: goto R2; }
  p[gid] = 1.0f;
  pc = 1; goto R1;
R1:
  p[gid] += 2.0f;
  if (p[gid] > 3.0f) { pc = 2; goto R2; }
R2:
  p[gid] += 4.0f;
}
)");

    probe("address-space pointer stored in struct", R"(
struct S { uint64_t a; uint32_t pc; float v[4]; };
static inline device float* as_dev(uint64_t a) { return (device float*)(uintptr_t)a; }
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  S s;
  s.a = (uint64_t)(uintptr_t)p;
  device float* q = as_dev(s.a);
  q[gid] = 7.0f;
}
)");

    probe("as_type + bit ops", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  uint b = as_type<uint>(p[gid]);
  float f = as_type<float>(b + 0x1000u);
  p[gid] = f;
}
)");

    probe("simdgroup shuffle + ballot", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  float v = p[gid];
  float x = simd_shuffle_down(v, 16u);
  float y = simd_shuffle_xor(v, 4u);
  bool w = simd_any(v > 0.0f);
  uint m = simd_ballot(v > 0.0f);
  p[gid] = x + y + (w ? 1.0f : 0.0f) + float(popcount(m));
}
)");

    probe("threadgroup barrier + threadgroup mem", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]],
              uint lid [[thread_index_in_threadgroup]]) {
  threadgroup float sh[64];
  sh[lid] = p[gid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  p[gid] = sh[63u - lid];
}
)");

    probe("_Float16 arithmetic", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  _Float16 a = (_Float16)p[gid];
  _Float16 b = a * (_Float16)3.0f;
  p[gid] = (float)b;
}
)");

    probe("function pointer / lambda capture", R"(
kernel void k(device float* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  float t = p[gid];
  auto f = [&](float x) -> float { return x * t; };
  p[gid] = f(2.0f);
}
)");
  }
  return 0;
}
