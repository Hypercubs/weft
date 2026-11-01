// weft — a GPU compute kernel, running on CPU cores.
//
// A kernel file is translated once, at build time, into a C++ translation unit
// (`weft-gen my.metal -o my_kernel.cpp`). Compile that file and
// runtime/weft_runtime.cpp into your program, include this header, and the
// kernel is a function call: no GPU, no Metal device, no subprocess, no dylib.
//
//     #include <weft/weft.h>
//
//     int k = weft_kernel_find("saxpy");
//     const weft_kernel_info* info = weft_kernel_at(k);
//     size_t n = 1 << 20;
//     std::vector<float> a(n), b(n), out(n);
//     void* args[] = {a.data(), b.data(), out.data()};
//     weft_run(k, args, (uint32_t)n, 256);
//
// The values that come out are the ones the same kernel produces on the GPU:
// weft is exact by construction (same operations, same order, no reassociation),
// and the CLI checks that claim against real hardware for every sample.
//
// The order of arguments is the order the kernel declares `[[buffer(N)]]`, and a
// buffer's element width is `buf_bytes[i]` — the size of the C++ type the kernel
// declared, so a `float4*` binding is 16 bytes and a `packed_float3*` is 12.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One kernel in a translated module. `nbufs` counts the `[[buffer(N)]]`
// parameters the kernel declares — the entries of the `void*` array `weft_run`
// takes — and not the tile memory or colour attachment a render pass adds, which
// the module allocates for itself.
typedef struct weft_kernel_info {
  const char* name;
  int nbufs;
  int ntex;                   // textures it reads, set with weft_set_texture
  const uint32_t* buf_bytes;  // sizeof of each buffer's element; NULL when untyped
  int sync;                   // it has a barrier, a shuffle or threadgroup memory
  // The threadgroup size the module declares, or 0 when it declares none. Triton
  // records it as `.maxntid` from `num_warps`; it is the geometry the kernel was
  // compiled for, so a launcher that has no other source should use it.
  int threads;
} weft_kernel_info;

int weft_kernel_count(void);
const weft_kernel_info* weft_kernel_at(int index);
int weft_kernel_find(const char* name);  // -1 when the module has no such kernel

// Tile memory: a kernel whose tile stage or fragments read a pixel outside the
// invocation's own 32x32 tile has read memory the GPU leaves undefined. The
// module counts those accesses, so a caller can report them instead of trusting
// the result. `weft_tile_reset` starts a launch with a clean count.
void weft_tile_reset(void);
uint64_t weft_tile_violations(void);

// Bytes a buffer of `elems` elements needs, from the kernel's own declaration.
// Falls back to float when the front end cannot type its pointers (PTX and
// SPIR-V bindings are addresses, not types).
static inline uint64_t weft_arg_bytes(const weft_kernel_info* k, int i, uint32_t elems) {
  uint32_t e = 4;
  if (k && k->buf_bytes && i >= 0 && i < k->nbufs && k->buf_bytes[i]) e = k->buf_bytes[i];
  return (uint64_t)e * (uint64_t)elems;
}

// Scalar kernel parameters (the `int n` a CUDA kernel takes), by index.
void weft_set_param(int index, uint64_t value);

// A texture is an RGBA image of `comps` 4-byte channels per texel.
void weft_set_texture(int index, void* data, uint32_t width, uint32_t height, uint32_t comps);

// filter: 0 nearest, 1 linear.  address: 0 clamp to edge, 1 clamp to zero, 2 repeat.
void weft_set_sampler(int index, int filter, int address);

// Run the kernel over `grid` threads, `threads` per threadgroup. The buffers are
// read and written in place; the call returns when every thread has finished.
void weft_run(int index, void** buffers, uint32_t grid, uint32_t threads);

#ifdef __cplusplus
}
#endif
