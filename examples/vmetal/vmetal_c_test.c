/* The C ABI, end to end: open a module, dispatch through the device, then run
 * the same kernel through weft_run from fresh inputs and compare the words.
 * Compiles as C11 against runtime/vmetal.h and build/libvmetal.
 */
#include <vmetal.h>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*run_fn)(int, void**, uint32_t, uint32_t);

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s <module> <kernel> <threads> <threads-per-group>\n", argv[0]);
    return 2;
  }
  const uint32_t grid = (uint32_t)strtoul(argv[3], NULL, 10);
  const uint32_t tg = (uint32_t)strtoul(argv[4], NULL, 10);

  vmetal_device* dev = vmetal_device_open(argv[1]);
  if (!dev) {
    fprintf(stderr, "vmetal: %s\n", vmetal_last_error());
    return 1;
  }
  const int k = vmetal_kernel_find(dev, argv[2]);
  if (k < 0) {
    fprintf(stderr, "vmetal: %s\n", vmetal_last_error());
    return 1;
  }
  const int nbuf = vmetal_kernel_buffers(dev, k);
  if (nbuf <= 0) {
    fprintf(stderr, "vmetal: %s declares no buffers\n", argv[2]);
    return 1;
  }
  printf("device  %s\n", vmetal_device_name());
  printf("kernel  %s  %u threads  %u per group  %d buffers\n", argv[2], grid, tg, nbuf);

  /* One element per thread, four bytes each, which is `saxpy`'s shape. */
  const size_t bytes = (size_t)grid * sizeof(float);
  vmetal_buffer** bufs = (vmetal_buffer**)calloc((size_t)nbuf, sizeof *bufs);
  float** seed = (float**)calloc((size_t)nbuf, sizeof *seed);
  for (int b = 0; b < nbuf; b++) {
    seed[b] = (float*)malloc(bytes);
    for (size_t i = 0; i < grid; i++)
      seed[b][i] = 1.5f * (float)((i + 1u) % 17u) + 0.25f * (float)b;
    bufs[b] = vmetal_new_buffer(dev, bytes, seed[b]);
    if (!bufs[b]) {
      fprintf(stderr, "vmetal: %s\n", vmetal_last_error());
      return 1;
    }
  }

  vmetal_queue* q = vmetal_new_queue(dev);
  vmetal_encoder* e = vmetal_encoder_create(q, k);
  if (!q || !e) {
    fprintf(stderr, "vmetal: %s\n", vmetal_last_error());
    return 1;
  }
  for (int b = 0; b < nbuf; b++) vmetal_encoder_set_buffer(e, b, bufs[b], 0);
  vmetal_encoder_dispatch(e, grid, tg);
  double ms = 0.0;
  if (vmetal_encoder_commit(e, &ms) != 0) {
    fprintf(stderr, "vmetal: %s\n", vmetal_last_error());
    return 1;
  }
  vmetal_encoder_release(e);

  void* h = dlopen(argv[1], RTLD_NOW);
  run_fn run = h ? (run_fn)dlsym(h, "weft_run") : NULL;
  if (!run) {
    fprintf(stderr, "vmetal: %s does not export weft_run\n", argv[1]);
    return 1;
  }
  float** want = (float**)calloc((size_t)nbuf, sizeof *want);
  void** ptrs = (void**)calloc((size_t)nbuf, sizeof *ptrs);
  for (int b = 0; b < nbuf; b++) {
    want[b] = (float*)malloc(bytes);
    memcpy(want[b], seed[b], bytes);
    ptrs[b] = want[b];
  }
  run(k, ptrs, grid, tg);

  size_t values = 0, differ = 0;
  for (int b = 0; b < nbuf; b++) {
    const unsigned char* got = vmetal_buffer_contents(bufs[b]);
    for (size_t i = 0; i < grid; i++) {
      uint32_t a = 0, c = 0;
      memcpy(&a, got + 4 * i, 4);
      memcpy(&c, &want[b][i], 4);
      if (a != c) differ++;
      values++;
    }
  }
  printf("c api   %zu values  %zu differ: %s\n", values, differ,
         differ == 0 ? "the words agree" : "DISAGREE");
  return differ == 0 ? 0 : 1;
}
