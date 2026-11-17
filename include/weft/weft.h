// weft, a GPU compute kernel, running on CPU cores.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One kernel in a translated module.
typedef struct weft_kernel_info {
  const char* name;
  int nbufs;
  int ntex;                   // textures it reads, set with weft_set_texture
  const uint32_t* buf_bytes;  // sizeof of each buffer's element; NULL when untyped
  int sync;                   // it has a barrier, a shuffle or threadgroup memory
  // The threadgroup size the module declares, or 0 when it declares none.
  int threads;
} weft_kernel_info;

int weft_kernel_count(void);
const weft_kernel_info* weft_kernel_at(int index);
int weft_kernel_find(const char* name);  // -1 when the module has no such kernel

void weft_tile_reset(void);
uint64_t weft_tile_violations(void);

// Bytes a buffer of `elems` elements needs, from the kernel's own declaration.
static inline uint64_t weft_arg_bytes(const weft_kernel_info* k, int i, uint32_t elems) {
  uint32_t e = 4;
  if (k && k->buf_bytes && i >= 0 && i < k->nbufs && k->buf_bytes[i]) e = k->buf_bytes[i];
  return (uint64_t)e * (uint64_t)elems;
}

// Scalar kernel parameters (the `int n` a CUDA kernel takes), by index.
void weft_set_param(int index, uint64_t value);

// A texture is an RGBA image of `comps` 4-byte channels per texel.
void weft_set_texture(int index, void* data, uint32_t width, uint32_t height, uint32_t comps);

// A 2D *array* texture.
void weft_set_texture_layers(int index, void* data, uint32_t width, uint32_t height, uint32_t comps,
                             uint32_t layers);

// filter: 0 nearest, 1 linear.  address: 0 clamp to edge, 1 clamp to zero, 2 repeat.
void weft_set_sampler(int index, int filter, int address);

// Run the kernel over `grid` threads, `threads` per threadgroup.
void weft_run(int index, void** buffers, uint32_t grid, uint32_t threads);

#ifdef __cplusplus
}
#endif
