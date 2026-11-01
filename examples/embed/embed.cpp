// The linked path: a translated kernel as a plain function call.
//
// No CLI, no dylib, no dlopen, no Metal device. `make embed` generates
// build/embed_kernel.cpp from kernels/basic.metal with weft-gen, compiles it into
// this binary next to runtime/weft_runtime.cpp, and this program calls it.
//
//     ./build/embed build/embed
//
// It writes the two inputs and the result it computed, so check.sh can feed the
// same inputs to the CLI and compare the bytes. That is the claim being tested:
// the tool you link and the tool you run agree exactly.
#include <weft/weft.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

bool write_floats(const char* path, const std::vector<float>& v) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  const size_t n = std::fwrite(v.data(), sizeof(float), v.size(), f);
  std::fclose(f);
  return n == v.size();
}

}  // namespace

int main(int argc, char** argv) {
  const char* prefix = argc > 1 ? argv[1] : "build/embed";
  const uint32_t n = 4096;

  const int k = weft_kernel_find("saxpy");
  if (k < 0) {
    std::fprintf(stderr, "embed: the module has no kernel called 'saxpy'\n");
    return 1;
  }
  const weft_kernel_info* info = weft_kernel_at(k);
  if (!info || info->nbufs != 3) {
    std::fprintf(stderr, "embed: saxpy should take 3 buffers, the module says %d\n",
                 info ? info->nbufs : -1);
    return 1;
  }
  // The module types its own buffers: the caller can size storage from the
  // kernel's declaration instead of guessing that everything is a float.
  if (weft_arg_bytes(info, 0, 1) != sizeof(float) || weft_arg_bytes(info, 2, n) != n * sizeof(float)) {
    std::fprintf(stderr, "embed: saxpy's buffers should be %zu bytes each, the module says %llu\n",
                 sizeof(float) * (size_t)n, (unsigned long long)weft_arg_bytes(info, 2, n));
    return 1;
  }

  std::vector<float> a(n), b(n), out(n, 0.0f);
  for (uint32_t i = 0; i < n; i++) {
    a[i] = (float)(i % 97u) * 0.5f + 1.0f;
    b[i] = (float)(i % 31u) * 0.25f - 2.0f;
  }

  void* args[] = {a.data(), b.data(), out.data()};
  weft_run(k, args, n, 256);

  char pa[512], pb[512], po[512];
  std::snprintf(pa, sizeof pa, "%s_a.bin", prefix);
  std::snprintf(pb, sizeof pb, "%s_b.bin", prefix);
  std::snprintf(po, sizeof po, "%s_out.bin", prefix);
  if (!write_floats(pa, a) || !write_floats(pb, b) || !write_floats(po, out)) {
    std::fprintf(stderr, "embed: cannot write the buffers beside '%s'\n", prefix);
    return 1;
  }
  std::printf("embed     %s  ·  %u threads  ·  3 buffers (%.0f B each)  ·  wrote %s\n",
              info->name, n, (double)weft_arg_bytes(info, 0, 1) * n, po);
  return 0;
}
