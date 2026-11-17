// Run one probe kernel and print what it wrote.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: f64_probe <file.metal> <in.bin>\n");
    return 2;
  }
  @autoreleasepool {
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
      std::fprintf(stderr, "read: %s\n", [[err localizedDescription] UTF8String]);
      return 2;
    }
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", argv[2]);
      return 2;
    }
    std::vector<unsigned long long> in;
    unsigned long long w = 0;
    while (std::fread(&w, sizeof w, 1, f) == 1) in.push_back(w);
    std::fclose(f);
    if (in.empty()) {
      std::fprintf(stderr, "no input words\n");
      return 2;
    }
    const unsigned n = (unsigned)(in.size() / 2);
    if (n == 0) {
      std::fprintf(stderr, "the input needs a pair of words per case\n");
      return 2;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::fprintf(stderr, "no Metal device\n");
      return 2;
    }
    MTLCompileOptions* opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) {
      std::fprintf(stderr, "%s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"probe"];
    if (!fn) {
      std::fprintf(stderr, "no probe function\n");
      return 1;
    }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      std::fprintf(stderr, "%s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    // The input buffer holds both words of every case; the output holds one per case.
    id<MTLBuffer> bin = [dev newBufferWithLength:in.size() * sizeof(unsigned long long)
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> bout = [dev newBufferWithLength:n * sizeof(unsigned long long)
                                          options:MTLResourceStorageModeShared];
    std::memcpy(bin.contents, in.data(), in.size() * sizeof(unsigned long long));
    std::memset(bout.contents, 0, n * sizeof(unsigned long long));

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bin offset:0 atIndex:0];
    [enc setBuffer:bout offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((n + 31) / 32, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
      std::fprintf(stderr, "dispatch failed: %s\n", [[cb.error localizedDescription] UTF8String]);
      return 1;
    }
    const unsigned long long* r = (const unsigned long long*)bout.contents;
    for (unsigned i = 0; i < n; i++) std::printf("%016llx\n", r[i]);
    return 0;
  }
}
