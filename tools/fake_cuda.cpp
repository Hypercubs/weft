// A stand-in for libcuda.so.1: enough of the driver to run weft's CUDA column on
// a host with no CUDA. Every call is recorded and nothing is executed, so the
// binding, the uploads and the launch can be read back as calls. tools/cuda_calls.sh
// builds it, points the loader at it and checks what it recorded. It is a test
// double, not a driver: a launch returns success without running anything, and a
// readback returns what was last written into the buffer.

#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using I = int;
using U64 = uint64_t;
using P = void*;

// The driver's structs, field for field, because the offsets are the ABI.
struct AD {
  size_t w, h;
  I fmt;
  unsigned ch;
};
struct AD3 {
  size_t w, h, d;
  I fmt;
  unsigned ch, flags;
};
struct C2 {
  size_t sx, sy;
  I st;
  const void* sh;
  U64 sd;
  P sa;
  size_t sp;
  size_t dx, dy;
  I dt;
  void* dh;
  U64 dd;
  P da;
  size_t dp;
  size_t wb, h;
};
struct C3 {
  size_t sx, sy, sz, slod;
  I st;
  const void* sh;
  U64 sd;
  P sa;
  P r0;
  size_t sp, shh;
  size_t dx, dy, dz, dlod;
  I dt;
  void* dh;
  U64 dd;
  P da;
  P r1;
  size_t dp, dhh;
  size_t wb, h, d;
};

namespace {

FILE* g_log;
std::string g_ptx;
std::map<U64, std::vector<unsigned char>> g_mem;
std::map<P, AD> g_arr;
std::map<P, AD3> g_arr3;
std::vector<std::string> g_tex;  // the `.tex` names the module declares
int g_params;                    // the parameters the named entry declares
unsigned long long g_next = 0x10000;

void lg(const char* f, ...) {
  if (!g_log) return;
  va_list a;
  va_start(a, f);
  std::vfprintf(g_log, f, a);
  va_end(a);
  std::fputc('\n', g_log);
  std::fflush(g_log);
}

P new_handle(const char* what) {
  g_next += 0x100;
  lg("handle %s 0x%llx", what, g_next);
  return (P)(uintptr_t)g_next;
}

// A name is declared when the module has an entry of that name: the column is
// checked against the module, not against what the launcher asked for.
bool declares(const char* word, const char* name) {
  const std::string want = std::string(word) + " " + name;
  return g_ptx.find(want) != std::string::npos;
}

// The names a `.tex` line declares. The type comes first, then the name, then
// the shape, so the name is the token after the type.
void scan_textures() {
  g_tex.clear();
  size_t at = 0;
  while (at <= g_ptx.size()) {
    size_t eol = g_ptx.find('\n', at);
    std::string line = g_ptx.substr(at, eol == std::string::npos ? std::string::npos : eol - at);
    size_t k = 0;
    while (k < line.size() && (line[k] == ' ' || line[k] == '\t')) k++;
    if (line.compare(k, 5, ".tex ") == 0) {
      size_t i = k + 5;
      if (i < line.size() && line[i] == '.') {
        while (i < line.size() && !std::isspace((unsigned char)line[i])) i++;
      }
      while (i < line.size() && std::isspace((unsigned char)line[i])) i++;
      size_t j = i;
      while (j < line.size() && (std::isalnum((unsigned char)line[j]) || line[j] == '_')) j++;
      if (j > i) g_tex.push_back(line.substr(i, j - i));
    }
    if (eol == std::string::npos) break;
    at = eol + 1;
  }
}

// The parameter count of one entry, read from its own signature rather than the
// whole module, so a file with several kernels is not miscounted.
void count_params(const char* name) {
  g_params = 0;
  size_t at = g_ptx.find(std::string("entry ") + name);
  if (at == std::string::npos) return;
  size_t open = g_ptx.find('(', at);
  size_t close = open == std::string::npos ? std::string::npos : g_ptx.find(')', open);
  if (close == std::string::npos) return;
  for (size_t p = open; (p = g_ptx.find(".param", p)) != std::string::npos && p < close; p += 6)
    g_params++;
}

}  // namespace

extern "C" {

I cuInit(unsigned) {
  const char* path = std::getenv("WEFT_FAKE_LOG");
  g_log = path ? std::fopen(path, "w") : stderr;
  lg("cuInit version=0");
  return 0;
}

I cuDeviceGetCount(int* n) {
  *n = 1;
  lg("cuDeviceGetCount = 1");
  return 0;
}

I cuDeviceGet(int* d, int) {
  *d = 0;
  return 0;
}

I cuDeviceGetName(char* out, int len, int) {
  std::snprintf(out, (size_t)len, "fake cuda device");
  lg("cuDeviceGetName -> %s", out);
  return 0;
}

I cuCtxCreate_v2(P* ctx, unsigned, int) {
  *ctx = new_handle("ctx");
  return 0;
}

I cuCtxDestroy_v2(P) {
  lg("cuCtxDestroy");
  return 0;
}

I cuMemAlloc_v2(U64* at, size_t bytes) {
  *at = (U64)0x1000 + g_mem.size() * 0x100000;
  g_mem[*at] = std::vector<unsigned char>(bytes, 0);
  lg("cuMemAlloc at=0x%llx bytes=%zu", (unsigned long long)*at, bytes);
  return 0;
}

I cuMemFree_v2(U64 at) {
  g_mem.erase(at);
  lg("cuMemFree at=0x%llx", (unsigned long long)at);
  return 0;
}

I cuMemcpyHtoD_v2(U64 at, const void* src, size_t bytes) {
  auto it = g_mem.find(at);
  if (it == g_mem.end() || bytes > it->second.size()) {
    lg("cuMemcpyHtoD BAD at=0x%llx bytes=%zu", (unsigned long long)at, bytes);
    return 500;
  }
  std::memcpy(it->second.data(), src, bytes);
  lg("cuMemcpyHtoD at=0x%llx bytes=%zu", (unsigned long long)at, bytes);
  return 0;
}

I cuMemcpyDtoH_v2(void* dst, U64 at, size_t bytes) {
  auto it = g_mem.find(at);
  if (it == g_mem.end() || bytes > it->second.size()) {
    lg("cuMemcpyDtoH BAD at=0x%llx bytes=%zu", (unsigned long long)at, bytes);
    return 500;
  }
  std::memcpy(dst, it->second.data(), bytes);
  lg("cuMemcpyDtoH at=0x%llx bytes=%zu", (unsigned long long)at, bytes);
  return 0;
}

I cuModuleLoadData(P* m, const void* text) {
  // A driver that loaded but will not take the module, which is the shape of an
  // sm mismatch on a real card. The launcher turns this into a named refusal.
  if (std::getenv("WEFT_FAKE_REFUSE")) {
    lg("cuModuleLoadData REFUSED");
    return 1;
  }
  g_ptx = (const char*)text;
  g_params = 0;
  scan_textures();
  *m = new_handle("module");
  lg("cuModuleLoadData bytes=%zu tex=%zu", g_ptx.size(), g_tex.size());
  for (const std::string& t : g_tex) lg("declared .tex %s", t.c_str());
  return 0;
}

I cuModuleUnload(P) {
  lg("cuModuleUnload");
  return 0;
}

I cuModuleGetFunction(P* fn, P, const char* name) {
  if (!declares(".entry", name) && !declares(".visible .entry", name)) {
    lg("cuModuleGetFunction %s NOT DECLARED", name);
    return 500;
  }
  count_params(name);
  *fn = new_handle("function");
  lg("cuModuleGetFunction %s params=%d", name, g_params);
  return 0;
}

I cuFuncGetAttribute(int* v, int attrib, P) {
  *v = attrib == 0 ? 1024 : 0;
  lg("cuFuncGetAttribute %d -> %d", attrib, *v);
  return 0;
}

I cuLaunchKernel(P, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                 unsigned sh, void*, void** kp, void**) {
  lg("cuLaunchKernel grid=%ux%ux%u block=%ux%ux%u shared=%u", gx, gy, gz, bx, by, bz, sh);
  if (kp)
    for (int i = 0; i < g_params; i++) {
      U64 v = 0;
      std::memcpy(&v, kp[i], 8);
      lg("param%d cell=0x%016llx", i, (unsigned long long)v);
    }
  return 0;
}

I cuCtxSynchronize() {
  return 0;
}

I cuGetErrorString(I r, const char** s) {
  if (r == 1) {
    *s = "the stand-in driver refused the module";
    return 0;
  }
  if (r == 500) {
    *s = "the stand-in driver: the module does not declare that name";
    return 0;
  }
  *s = "the stand-in driver refused the call";
  return 0;
}

I cuModuleGetTexRef(P* tr, P, const char* name) {
  bool found = false;
  for (const std::string& t : g_tex)
    if (t == name) found = true;
  if (!found) {
    lg("cuModuleGetTexRef %s NOT DECLARED", name);
    return 500;
  }
  *tr = new_handle("texref");
  lg("cuModuleGetTexRef %s", name);
  return 0;
}

I cuTexRefSetArray(P, P arr, unsigned flags) {
  if (!g_arr.count(arr) && !g_arr3.count(arr)) {
    lg("cuTexRefSetArray UNKNOWN ARRAY 0x%llx", (unsigned long long)(uintptr_t)arr);
    return 500;
  }
  lg("cuTexRefSetArray flags=%u", flags);
  return 0;
}

I cuTexRefSetFilterMode(P, int mode) {
  lg("cuTexRefSetFilterMode %d", mode);
  return 0;
}

I cuTexRefSetAddressMode(P, int dim, int mode) {
  lg("cuTexRefSetAddressMode dim=%d mode=%d", dim, mode);
  return 0;
}

I cuTexRefSetFlags(P, unsigned flags) {
  lg("cuTexRefSetFlags 0x%x", flags);
  return 0;
}

I cuTexRefSetBorderColor(P, float* rgba) {
  lg("cuTexRefSetBorderColor %g %g %g %g", rgba[0], rgba[1], rgba[2], rgba[3]);
  return 0;
}

I cuArrayCreate_v2(P* out, const AD* d) {
  P h = new_handle("array");
  g_arr[h] = *d;
  lg("cuArrayCreate w=%zu h=%zu fmt=0x%x ch=%u", d->w, d->h, d->fmt, d->ch);
  *out = h;
  return 0;
}

I cuArray3DCreate_v2(P* out, const AD3* d) {
  P h = new_handle("array3d");
  g_arr3[h] = *d;
  lg("cuArray3DCreate w=%zu h=%zu d=%zu fmt=0x%x ch=%u flags=0x%x", d->w, d->h, d->d, d->fmt,
     d->ch, d->flags);
  *out = h;
  return 0;
}

I cuArrayDestroy(P a) {
  g_arr.erase(a);
  g_arr3.erase(a);
  lg("cuArrayDestroy");
  return 0;
}

I cuMemcpy2D_v2(const C2* c) {
  if (!g_arr.count(c->da) || c->st != 1 || c->dt != 3) {
    lg("cuMemcpy2D BAD dst=%p srcType=%d dstType=%d", c->da, c->st, c->dt);
    return 500;
  }
  lg("cuMemcpy2D srcType=%d srcPitch=%zu dstType=%d widthBytes=%zu height=%zu", c->st, c->sp,
     c->dt, c->wb, c->h);
  return 0;
}

I cuMemcpy3D_v2(const C3* c) {
  if (!g_arr.count(c->da) && !g_arr3.count(c->da)) {
    lg("cuMemcpy3D BAD dst=%p srcType=%d dstType=%d", c->da, c->st, c->dt);
    return 500;
  }
  lg("cuMemcpy3D srcType=%d srcPitch=%zu srcHeight=%zu widthBytes=%zu height=%zu depth=%zu",
     c->st, c->sp, c->shh, c->wb, c->h, c->d);
  return 0;
}

}  // extern "C"
