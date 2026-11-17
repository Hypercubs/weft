// One kernel through the virtual device, with no GPU.
//
// The module is the shared library `weft-gen` produced and this host's compiler
// built. The demo opens it, allocates a buffer per binding, dispatches, then runs
// the same kernel through the module's own `weft_run` from a second copy of the
// inputs. The results are compared as words, which is the device's claim: it
// executes the translated body, so it cannot drift from a direct call.
#include <vmetal.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

struct Module {
  void* handle = nullptr;
  int (*count)() = nullptr;
  const weft_kernel_info* (*at)(int) = nullptr;
  void (*run)(int, void**, uint32_t, uint32_t) = nullptr;
};

Module open_module(const char* path) {
  Module m;
#if defined(_WIN32)
  m.handle = (void*)LoadLibraryA(path);
  if (!m.handle) return m;
  m.count = (int (*)())GetProcAddress((HMODULE)m.handle, "weft_kernel_count");
  m.at = (const weft_kernel_info* (*)(int))GetProcAddress((HMODULE)m.handle, "weft_kernel_at");
  m.run = (void (*)(int, void**, uint32_t, uint32_t))GetProcAddress((HMODULE)m.handle, "weft_run");
#else
  m.handle = dlopen(path, RTLD_NOW);
  if (!m.handle) return m;
  m.count = (int (*)())dlsym(m.handle, "weft_kernel_count");
  m.at = (const weft_kernel_info* (*)(int))dlsym(m.handle, "weft_kernel_at");
  m.run = (void (*)(int, void**, uint32_t, uint32_t))dlsym(m.handle, "weft_run");
#endif
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <module> <kernel> <threads> <threads-per-group>\n", argv[0]);
    return 2;
  }
  const uint32_t grid = (uint32_t)std::strtoul(argv[3], nullptr, 10);
  const uint32_t tg = (uint32_t)std::strtoul(argv[4], nullptr, 10);

  std::string err;
  vmetal::Device dev = vmetal::Device::open(argv[1], &err);
  if (!dev.valid()) {
    std::fprintf(stderr, "vmetal: %s\n", err.c_str());
    return 1;
  }
  vmetal::PipelineState pipe = dev.new_pipeline(argv[2], &err);
  if (!pipe.valid()) {
    std::fprintf(stderr, "vmetal: %s\n", err.c_str());
    return 1;
  }

  Module ref = open_module(argv[1]);
  if (!ref.handle || !ref.count || !ref.at || !ref.run) {
    std::fprintf(stderr, "vmetal: %s does not export the weft entry points\n", argv[1]);
    return 1;
  }
  int kidx = -1;
  for (int i = 0; i < ref.count(); i++) {
    const weft_kernel_info* k = ref.at(i);
    if (k && k->name && std::strcmp(k->name, argv[2]) == 0) kidx = i;
  }
  if (kidx < 0) {
    std::fprintf(stderr, "vmetal: %s has no kernel named %s\n", argv[1], argv[2]);
    return 1;
  }
  const weft_kernel_info* info = ref.at(kidx);
  const int nbuf = info->nbufs > 0 ? info->nbufs : 1;

  std::printf("device  %s\n", dev.name());
  std::printf("kernel  %s  %u threads  %u per group  %d buffers\n", argv[2], grid, tg, nbuf);

  std::vector<std::vector<float>> seed((size_t)nbuf);
  std::vector<vmetal::Buffer> bufs((size_t)nbuf);
  for (int b = 0; b < nbuf; b++) {
    const uint32_t elem = info->buf_bytes && info->buf_bytes[b] ? info->buf_bytes[b] : 4u;
    const size_t words = (size_t)grid * elem / sizeof(float);
    seed[(size_t)b].resize(words);
    for (size_t i = 0; i < words; i++)
      seed[(size_t)b][i] = 1.5f * (float)((i + 1u) % 17u) + 0.25f * (float)b;
    bufs[(size_t)b] = dev.new_buffer(words * sizeof(float), seed[(size_t)b].data());
  }

  vmetal::CommandQueue q = dev.new_queue();
  vmetal::CommandBuffer cb = q.command_buffer();
  vmetal::Encoder enc = cb.compute_encoder();
  enc.set_pipeline(pipe);
  for (int b = 0; b < nbuf; b++) enc.set_buffer(b, bufs[(size_t)b]);
  enc.dispatch_threads(grid, tg);
  enc.end_encoding();
  cb.commit();
  cb.wait_until_completed();
  if (!cb.ok()) {
    std::fprintf(stderr, "vmetal: %s\n", cb.error().c_str());
    return 1;
  }

  // The same launch through the module's own entry point, from a fresh copy of
  // the inputs, so a kernel that writes what it reads cannot agree by accident.
  std::vector<std::vector<float>> direct = seed;
  std::vector<void*> ptrs((size_t)nbuf);
  for (int b = 0; b < nbuf; b++) ptrs[(size_t)b] = direct[(size_t)b].data();
  ref.run(kidx, ptrs.data(), grid, tg);

  size_t values = 0, differ = 0;
  for (int b = 0; b < nbuf; b++) {
    const float* got = (const float*)bufs[(size_t)b].contents();
    const float* want = direct[(size_t)b].data();
    const size_t words = seed[(size_t)b].size();
    for (size_t i = 0; i < words; i++) {
      uint32_t a = 0, c = 0;
      std::memcpy(&a, &got[i], 4);
      std::memcpy(&c, &want[i], 4);
      if (a != c) differ++;
      values++;
    }
  }
  std::printf("vmetal  %zu values  %zu differ: %s\n", values, differ,
              differ == 0 ? "the words agree" : "DISAGREE");
  return differ == 0 ? 0 : 1;
}
