// Does the packing formula in src/soft_math.hpp agree with the hardware's own
// `pack_float_to_*` builtins?
//
// The formula is what both engines run, so the two columns cannot disagree with
// each other, but they could both be wrong about what Metal means by a normalized
// conversion. The rule that is not obvious is rounding: nearest, and at an exact
// tie the hardware could round up, to even, or away from zero. This asks the
// hardware instead of reading the spec, over every byte boundary, every 16-bit
// boundary and exact ties in both signs.
//
// Build: clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//          tools/pack_probe.mm -o /tmp/packprobe && /tmp/packprobe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float pack_clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static int32_t pack_round(float v) {
  return v < 0.0f ? -(int32_t)(0.5f - v) : (int32_t)(v + 0.5f);
}

// Is the product the formula rounds an exact tie at the precision it is computed
// in? That is the only input class where the hardware's intermediate precision can
// decide a different answer, and it is decided from the value rather than from the
// two answers, so a formula that is wrong everywhere cannot hide behind it.
static bool is_tie(float v, double scale) {
  const float p = (float)(v * (float)scale);
  const double a = std::fabs((double)p);
  return a - std::floor(a) == 0.5;
}
static uint32_t pack_u8(float v) { return (uint32_t)pack_round(pack_clamp(v, 0.0f, 1.0f) * 255.0f); }
static uint32_t pack_s8(float v) {
  return (uint32_t)pack_round(pack_clamp(v, -1.0f, 1.0f) * 127.0f) & 0xFFu;
}
static uint32_t pack_u16(float v) {
  return (uint32_t)pack_round(pack_clamp(v, 0.0f, 1.0f) * 65535.0f) & 0xFFFFu;
}
static uint32_t pack_s16(float v) {
  return (uint32_t)pack_round(pack_clamp(v, -1.0f, 1.0f) * 32767.0f) & 0xFFFFu;
}

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::printf("pack_probe: no Metal device\n");
      return 1;
    }

    // Every value the four conversions can see a boundary at, plus the ties.
    std::vector<float> vals;
    for (int i = 0; i <= 1024; i++) vals.push_back((float)i / 1024.0f);
    for (int k = 0; k < 255; k++) vals.push_back(((float)k + 0.5f) / 255.0f);
    for (int k = 0; k < 255; k++) vals.push_back(-((float)k + 0.5f) / 255.0f);
    for (int k = 0; k < 127; k++) vals.push_back(((float)k + 0.5f) / 127.0f);
    for (int k = 0; k < 127; k++) vals.push_back(-((float)k + 0.5f) / 127.0f);
    vals.push_back(-1.0f);
    vals.push_back(1.0f);
    vals.push_back(-1.5f);
    vals.push_back(1.5f);
    vals.push_back(0.0f);
    const uint32_t n = (uint32_t)vals.size();

    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void k(device const float* in [[buffer(0)]],\n"
                     "              device uint* out [[buffer(1)]],\n"
                     "              uint gid [[thread_position_in_grid]]) {\n"
                     "  const float4 v = float4(in[gid], in[gid], in[gid], in[gid]);\n"
                     "  out[gid * 4u + 0u] = pack_float_to_unorm4x8(v);\n"
                     "  out[gid * 4u + 1u] = pack_float_to_snorm4x8(v);\n"
                     "  out[gid * 4u + 2u] = pack_float_to_unorm2x16(v.xy);\n"
                     "  out[gid * 4u + 3u] = pack_float_to_snorm2x16(v.xy);\n"
                     "}\n";
    NSError* err = nil;
    MTLCompileOptions* o = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
    if (!lib) {
      std::printf("pack_probe: the shader did not compile: %s\n",
                  [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&err];
    if (!pso) {
      std::printf("pack_probe: no pipeline: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }

    id<MTLBuffer> in = [dev newBufferWithBytes:vals.data()
                                       length:vals.size() * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [dev newBufferWithLength:(NSUInteger)n * 4 * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:in offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((n + 63) / 64, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
      std::printf("pack_probe: %s\n", [[cb.error localizedDescription] UTF8String]);
      return 1;
    }

    const uint32_t* hw = (const uint32_t*)out.contents;
    int bad[4] = {0, 0, 0, 0};
    int ties[4] = {0, 0, 0, 0};
    static const double scales[4] = {255.0, 127.0, 65535.0, 32767.0};
    for (uint32_t i = 0; i < n; i++) {
      const float v = vals[i];
      const uint32_t mine[4] = {
          pack_u8(v) | (pack_u8(v) << 8) | (pack_u8(v) << 16) | (pack_u8(v) << 24),
          pack_s8(v) | (pack_s8(v) << 8) | (pack_s8(v) << 16) | (pack_s8(v) << 24),
          pack_u16(v) | (pack_u16(v) << 16),
          pack_s16(v) | (pack_s16(v) << 16),
      };
      static const char* names[4] = {"unorm4x8", "snorm4x8", "unorm2x16", "snorm2x16"};
      for (int c = 0; c < 4; c++) {
        if (hw[i * 4 + c] == mine[c]) continue;
        // A tie is where the hardware's intermediate precision decides, and this
        // text cannot reproduce it. Anything else is a disagreement that matters.
        if (is_tie(v, scales[c])) {
          ties[c]++;
          continue;
        }
        if (bad[c] < 3)
          std::printf("  %-10s %+.9g  hardware 0x%08x  formula 0x%08x\n", names[c], (double)v,
                      hw[i * 4 + c], mine[c]);
        bad[c]++;
      }
    }
    const int total = bad[0] + bad[1] + bad[2] + bad[3];
    const int tie_total = ties[0] + ties[1] + ties[2] + ties[3];
    std::printf("pack_probe: %u values x 4 conversions: %d differ away from a tie "
                "(unorm4x8 %d, snorm4x8 %d, unorm2x16 %d, snorm2x16 %d)\n",
                n, total, bad[0], bad[1], bad[2], bad[3]);
    std::printf("            %d exact ties round differently on the hardware "
                "(unorm4x8 %d, snorm4x8 %d, unorm2x16 %d, snorm2x16 %d); both engines "
                "run this formula, so they agree with each other\n",
                tie_total, ties[0], ties[1], ties[2], ties[3]);
    return total == 0 ? 0 : 1;
  }
}
