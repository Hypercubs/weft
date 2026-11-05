// Which imageblock spelling does this host's Metal compiler accept?
//
// The `metal` command-line tool is not installed on this machine, but the runtime
// compiler is, and that is the one weft hands its source to. So the question is
// answered by compiling each spelling with Metal.framework and printing the real
// diagnostic, rather than by guessing.
//
// Build: clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//          tools/imageblock_probe.mm -o /tmp/ibprobe && /tmp/ibprobe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

static const char* kHead =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct TileVal { float4 c[4]; };\n"
    "struct Vary { float4 pos [[position]]; };\n";

static void probe(const char* name, const char* body) {
  NSString* src = [NSString stringWithFormat:@"%s%s", kHead, body];
  NSError* err = nil;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  MTLCompileOptions* o = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
  std::printf("%-26s %s\n", name, lib ? "ACCEPTED" : "refused");
  if (!lib) {
    const char* msg = [[err localizedDescription] UTF8String];
    int n = 0;
    for (const char* p = msg; *p && n < 300; p++, n++) std::putchar(*p);
    std::printf("\n");
  } else {
    for (NSString* fn in @[ @"t", @"f" ]) {
      id<MTLFunction> f = [lib newFunctionWithName:fn];
      if (f) std::printf("    function %s found\n", [fn UTF8String]);
    }
  }
}

int main() {
  @autoreleasepool {
    probe("fragment + tile, plain",
          "[[tile]] void t(imageblock<TileVal> b [[imageblock_data]]) { b.c[0] = float4(1); }\n"
          "fragment float4 f(Vary v [[stage_in]], imageblock<TileVal> b [[imageblock_data]]) {\n"
          "  return b.c[0];\n}\n");
    probe("fragment + tile, implicit",
          "[[tile]] void t(imageblock<TileVal, layout::metal::implicit> b [[imageblock_data]]) {\n"
          "  b.c[0] = float4(1);\n}\n"
          "fragment float4 f(Vary v [[stage_in]],\n"
          "                  imageblock<TileVal, layout::metal::implicit> b [[imageblock_data]]) {\n"
          "  return b.c[0];\n}\n");
    probe("tile only, plain",
          "[[tile]] void t(imageblock<TileVal> b [[imageblock_data]]) { b.c[0] = float4(1); }\n");
    probe("tile only, implicit",
          "[[tile]] void t(imageblock<TileVal, layout::metal::implicit> b [[imageblock_data]]) {\n"
          "  b.c[0] = float4(1);\n}\n");
    probe("fragment only, plain",
          "fragment float4 f(Vary v [[stage_in]], imageblock<TileVal> b [[imageblock_data]]) {\n"
          "  return b.c[0];\n}\n");
    probe("tile only, no imageblock",
          "[[tile]] void t(uint2 tp [[thread_position_in_grid]]) { (void)tp; }\n");
  }
  return 0;
}
