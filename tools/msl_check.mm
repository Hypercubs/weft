// Compile a file of Metal Shading Language with the runtime compiler.
//
//     clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//       tools/msl_check.mm -o /tmp/mslcheck
//     /tmp/mslcheck out.metal
//
// It is the same compiler the launcher hands its source to, so a file that
// passes here is one the launcher will take, which is what makes it usable as a
// build check for the GPU back end without a GPU present in the loop.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mslcheck <file.metal>\n");
    return 2;
  }
  @autoreleasepool {
    NSString* path = [NSString stringWithUTF8String:argv[1]];
    NSError* read = nil;
    NSString* src = [NSString stringWithContentsOfFile:path
                                             encoding:NSUTF8StringEncoding
                                                error:&read];
    if (!src) {
      std::fprintf(stderr, "mslcheck: %s\n", [[read localizedDescription] UTF8String]);
      return 2;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::fprintf(stderr, "mslcheck: no Metal device\n");
      return 2;
    }
    MTLCompileOptions* o = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
    if (!lib) {
      std::fprintf(stderr, "%s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    // A library that compiles is not the same thing as a kernel the GPU will
    // accept: the pipeline step is where the Metal back end actually generates
    // machine code, so a source that only fails there looks fine here. Build a
    // pipeline for every function when asked with --pipeline.
    if (argc > 2 && std::string(argv[2]) == "--pipeline") {
      int built = 0;
      for (NSUInteger i = 0; i < lib.functionNames.count; i++) {
        NSString* name = lib.functionNames[i];
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) continue;
        NSError* perr = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&perr];
        if (!pso) {
          std::fprintf(stderr, "%s: pipeline failed: %s\n", [name UTF8String],
                       [[perr localizedDescription] UTF8String]);
          return 1;
        }
        built++;
      }
      std::printf("ok  %s compiles and %d pipeline(s) build\n", argv[1], built);
      return 0;
    }
    std::printf("ok  %s compiles\n", argv[1]);
    return 0;
  }
}
