// The GPU column is Metal's, so only the Apple build has one. Everything else in
// this file — the three front ends, the generated module, the CPU launch, the
// comparison and the image writer — is plain C++ and compiles anywhere that has
// dlopen and a C++20 compiler.
#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#include <dlfcn.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "translate.hpp"
#include "soft_math.hpp"
#include "translate_ptx.hpp"
#include "translate_spirv.hpp"
#include <weft/weft.h>

namespace {

// ---- a buffer's end is a page ----
//
// A kernel that indexes past the end of a buffer used to take the whole process
// down with it, and a bare `Segmentation fault` reads as a broken tool rather
// than a launch that was too small. The storage a kernel is handed is therefore
// mapped with a page of PROT_NONE after it, so the first access past the end
// lands in a handler that says which buffer, by how much, and over what grid.
// The page size is the host's, not an assumption: Apple silicon maps 16 KiB pages,
// and a guard page placed at a 4 KiB offset there is not page-aligned at all —
// `mprotect` refuses it. The value is read once, before the first allocation, and
// cached so the fault handler can use it without calling anything.
size_t g_page_bytes = 0;

size_t guard_page() {
  if (!g_page_bytes) g_page_bytes = (size_t)getpagesize();
  return g_page_bytes;
}

struct GuardTag {
  const char* base = nullptr;
  size_t bytes = 0;
  int index = -1;
};

GuardTag g_guards[64];
int g_nguards = 0;
// Two lanes of the same launch can overrun the same buffer in the same
// instruction, so only the first fault gets to speak.
volatile sig_atomic_t g_guard_reported = 0;
uint64_t g_guard_grid = 0;
uint64_t g_guard_threads = 0;

void* guard_alloc(size_t bytes) {
  const size_t page = guard_page();
  size_t pages = (bytes + page - 1) / page;
  if (pages == 0) pages = 1;
  void* p = mmap(nullptr, (pages + 1) * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
  if (p == MAP_FAILED) {
    std::fprintf(stderr, "weft: out of memory for %zu bytes of buffer\n", bytes);
    std::abort();
  }
  if (mprotect((char*)p + pages * page, page, PROT_NONE) != 0) {
    std::fprintf(stderr, "weft: cannot put a guard page after a %zu byte buffer\n", bytes);
    std::abort();
  }
  return p;
}

void guard_free(void* p, size_t bytes) {
  if (!p) return;
  const size_t page = guard_page();
  size_t pages = (bytes + page - 1) / page;
  if (pages == 0) pages = 1;
  munmap(p, (pages + 1) * page);
}

// Formatting that is safe in a signal handler: no stdio, no allocation.
size_t guard_put(char* out, const char* s) {
  size_t n = 0;
  while (s[n]) out[n] = s[n], n++;
  return n;
}

size_t guard_num(char* out, uint64_t v) {
  char tmp[24];
  size_t n = 0;
  do {
    tmp[n++] = (char)('0' + (int)(v % 10));
    v /= 10;
  } while (v);
  for (size_t i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
  return n;
}

void guard_fault(int sig, siginfo_t* si, void*) {
  const char* at = (const char*)si->si_addr;
  for (int i = 0; i < g_nguards; i++) {
    const char* base = g_guards[i].base;
    if (!base) continue;
    const size_t page = g_page_bytes ? g_page_bytes : 4096;
    const size_t len = g_guards[i].bytes;
    const size_t guard = ((len + page - 1) / page) * page;
    if (at < base + guard || at >= base + guard + page) continue;
    if (g_guard_reported) _exit(2);
    g_guard_reported = 1;
    char msg[320];
    size_t n = 0;
    n += guard_put(msg + n, "weft: the kernel touched byte ");
    n += guard_num(msg + n, (uint64_t)(at - base));
    n += guard_put(msg + n, " of buffer ");
    n += guard_num(msg + n, (uint64_t)g_guards[i].index);
    n += guard_put(msg + n, ", which is ");
    n += guard_num(msg + n, len);
    n += guard_put(msg + n, " bytes long");
    if (g_guard_grid) {
      n += guard_put(msg + n, " (the launch was ");
      n += guard_num(msg + n, g_guard_grid);
      n += guard_put(msg + n, " threads of ");
      n += guard_num(msg + n, g_guard_threads);
      n += guard_put(msg + n, ")");
    }
    n += guard_put(msg + n, ". --buffers ");
    n += guard_num(msg + n, (uint64_t)g_guards[i].index);
    n += guard_put(msg + n, " is too small for this launch.\n");
    const ssize_t w = write(STDERR_FILENO, msg, n);
    (void)w;
    _exit(2);
  }
  // Not one of ours: put the default action back and take the same fault again.
  signal(sig, SIG_DFL);
  raise(sig);
}

void guard_install() {
  static bool done = false;
  if (done) return;
  done = true;
  struct sigaction sa;
  std::memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = guard_fault;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
}

void guard_tag(const void* base, size_t bytes, int index) {
  if (!base || index < 0 || index >= 64) return;
  guard_install();
  g_guards[index].base = (const char*)base;
  g_guards[index].bytes = bytes;
  g_guards[index].index = index;
  if (index >= g_nguards) g_nguards = index + 1;
}

// A std::vector's storage, with the guard page after it. Everything the rest of
// this file does to a buffer — resize, assign, data, size, copy — is unchanged.
template <class T>
struct GuardAlloc {
  using value_type = T;
  GuardAlloc() = default;
  template <class U>
  GuardAlloc(const GuardAlloc<U>&) {}
  T* allocate(size_t n) { return (T*)guard_alloc(n * sizeof(T)); }
  void deallocate(T* p, size_t n) { guard_free(p, n * sizeof(T)); }
  template <class U>
  bool operator==(const GuardAlloc<U>&) const { return true; }
  template <class U>
  bool operator!=(const GuardAlloc<U>&) const { return false; }
};

using GuardedFloats = std::vector<float, GuardAlloc<float>>;

const char* kUsage =
    "weft — run GPU compute kernels on CPU cores\n"
    "\n"
    "  weft run <file.metal|file.ptx|file.spv> [options]\n"
    "\n"
    "  --kernel NAME     kernel to run (default: the only kernel in the file)\n"
    "  --grid N          total threads (default 1048576)\n"
    "  --threads N       threads per threadgroup (default 256)\n"
    "  --buffers a,b,c   element counts, one per buffer, in each buffer's own type\n"
    "                    (default: one element per thread)\n"
    "  --buffer-bytes I=N[,J=N]  the element width of buffer I, for a front end that\n"
    "                    binds addresses and cannot type its pointers (PTX, SPIR-V)\n"
    "  --out PATH        write the colour target to a PNG (RGBA8, sRGB)\n"
    "  --out-gpu PATH    the same frame from the gpu run, for a pixel diff\n"
    "  --render WxH      a render pass instead of a compute one: the fragment\n"
    "                    named by --kernel runs once per pixel of a WxH target\n"
  "  --textures WxH[*L][,...]  textures to bind at index 0, 1, ... (default: none);\n"
    "                    *L makes it a 2D array of L slices\n"
    "  --samplers F/A[,F/A]  sampler state per index: nearest|linear / clamp|zero|repeat\n"
    "  --bench K         iterations to time (default 20); 1 also turns off the\n"
    "                    thread search described under --cpu-only\n"
    "  --arg N=V         scalar kernel parameter N (PTX scalars; V may be 1.5 or 42)\n"
    "  --cpu-only        skip the GPU path (no Metal device needed)\n"
    "                    the cpu path times 1/2/4/8 threads on this kernel and keeps\n"
    "                    the fastest, because the right count is a property of the\n"
    "                    kernel: atomics want one, elementwise wants two to four. The\n"
    "                    count it kept is printed on the cpu line.\n"
    "  --emit            print the generated C++ and exit\n"
    "  --emit-gpu        print the Metal source the GPU is handed, and exit\n"
    "  --load B=path     fill buffer B from a file of float32 values\n"
    "  --dump B=path     write the CPU's buffer B out as floats\n"
    "  --dump-gpu B=path write the GPU's buffer B out as floats\n"
    "\n"
    "The same kernel source is compiled twice: once by Metal for the GPU, once\n"
    "translated to C++ for the CPU. Both outputs are compared and both are timed.\n";

std::string read_file(const std::string& path, bool& ok) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { ok = false; return {}; }
  std::string out;
  char buf[8192];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
  std::fclose(f);
  ok = true;
  return out;
}

bool write_file(const std::string& path, const std::string& data) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t n = std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return n == data.size();
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
// The runtime reads its thread count from the environment, and the launcher is
// what sets it.
void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
  _putenv_s(k, v.c_str());
#else
  setenv(k, v.c_str(), 1);
#endif
}

#if defined(__APPLE__)
// The vertex stage of every render pass: one triangle that covers the whole
// target, so each pixel gets exactly one fragment invocation. It is appended to
// the source the GPU compiles and never reaches the CPU translation.
const char* kFullScreenVertex = R"(
struct WeftVertexOut { float4 pos [[position]]; };
vertex WeftVertexOut weft_vertex(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    WeftVertexOut o;
    o.pos = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    return o;
}
)";
#endif

float det_value(uint32_t i, uint32_t buf) {
  uint32_t h = i * 2654435761u + buf * 40503u;
  h ^= h >> 13;
  h *= 2246822519u;
  h ^= h >> 16;
  return 0.5f + (float)(h & 0xFFFF) / 65535.0f;  // [0.5, 1.5): sums stay large
}

struct Options {
  std::string file, kernel, emitted;
  uint32_t grid = 1048576, threads = 256, bench = 20;
  bool threads_set = false;  // --threads was given, so no module default applies
  std::vector<size_t> bufs;
  std::vector<std::pair<int, uint32_t>> widths;  // --buffer-bytes overrides
  // One binding per --textures entry: its width x height, the slice count when it
  // is a 2D array (1 for a plain texture), and the depth when it is a volume —
  // `WxHxD`, which is 0 for the other two shapes.
  struct TexBind { uint32_t w, h, layers, d; bool cube; bool one_d; };
  std::vector<TexBind> textures;  // bound at index 0, 1, ...
  std::vector<std::pair<int, int>> samplers;            // filter, address mode
  std::vector<std::pair<int, uint64_t>> args;  // scalar kernel parameters
  std::vector<std::pair<int, std::string>> dumps, gpu_dumps;
  std::vector<std::pair<int, std::string>> loads;  // raw float32 in
  bool cpu_only = false, emit = false, emit_gpu = false, ptx = false;
  bool render = false;
  uint32_t rw = 0, rh = 0;  // the render target
  std::string out, out_gpu;  // --out, --out-gpu: the colour target as a PNG
  std::string front;  // "metal", "ptx" or "spirv"
};

bool write_bin(const std::string& path, const void* data, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t n = std::fwrite(data, 1, bytes, f);
  std::fclose(f);
  return n == bytes;
}

// ---- PNG out ---------------------------------------------------------------
// A renderer that cannot save its frame has no output. This writes a PNG with
// stored deflate blocks: no zlib, no dependency, and every viewer opens it. The
// colour is clamped to [0,1] and sRGB-encoded because the fragment writes linear
// light and a display expects the curve; alpha is linear and stays linear.
uint32_t crc32_of(const unsigned char* p, size_t n) {
  static uint32_t tab[256];
  static bool ready = false;
  if (!ready) {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int k = 0; k < 8; k++) c = (c & 1u) ? 0xedb88320u ^ (c >> 1) : c >> 1;
      tab[i] = c;
    }
    ready = true;
  }
  uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < n; i++) crc = tab[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
  return crc ^ 0xffffffffu;
}

void put_be32(std::string& s, uint32_t v) {
  s.push_back((char)(v >> 24));
  s.push_back((char)(v >> 16));
  s.push_back((char)(v >> 8));
  s.push_back((char)v);
}

void png_chunk(std::string& out, const char* type, const std::string& data) {
  put_be32(out, (uint32_t)data.size());
  const size_t at = out.size();
  out.append(type, 4);
  out += data;
  put_be32(out, crc32_of((const unsigned char*)out.data() + at, out.size() - at));
}

std::string zlib_stored(const std::string& raw) {
  std::string z("\x78\x01", 2);
  size_t pos = 0;
  for (;;) {
    const size_t n = std::min<size_t>(65535, raw.size() - pos);
    const bool last = pos + n >= raw.size();
    z.push_back((char)(last ? 1 : 0));
    z.push_back((char)(n & 0xffu));
    z.push_back((char)((n >> 8) & 0xffu));
    z.push_back((char)(~n & 0xffu));
    z.push_back((char)((~n >> 8) & 0xffu));
    z.append(raw, pos, n);
    pos += n;
    if (last) break;
  }
  uint32_t a = 1, b = 0;
  for (unsigned char c : raw) {
    a = (a + c) % 65521u;
    b = (b + a) % 65521u;
  }
  put_be32(z, (b << 16) | a);
  return z;
}

unsigned char srgb8(float v) {
  if (!(v > 0.0f)) return 0;  // negative and NaN are black
  if (v >= 1.0f) return 255;
  const float s = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
  return (unsigned char)std::lround(std::min(1.0f, std::max(0.0f, s)) * 255.0f);
}

bool write_png(const std::string& path, const std::vector<float>& px, uint32_t w, uint32_t h) {
  if (px.size() < (size_t)w * h * 4) return false;
  std::string raw;
  raw.reserve((size_t)w * h * 4 + h);
  for (uint32_t y = 0; y < h; y++) {
    raw.push_back(0);  // filter: none
    for (uint32_t x = 0; x < w; x++) {
      const float* p = &px[((size_t)y * w + x) * 4];
      for (int c = 0; c < 3; c++) raw.push_back((char)srgb8(p[c]));
      raw.push_back((char)(unsigned char)std::lround(
          std::min(1.0f, std::max(0.0f, p[3])) * 255.0f));
    }
  }
  std::string out("\x89PNG\r\n\x1a\n", 8);
  std::string ihdr;
  put_be32(ihdr, w);
  put_be32(ihdr, h);
  ihdr.push_back(8);  // bits per channel
  ihdr.push_back(6);  // RGBA
  ihdr.push_back(0);  // deflate
  ihdr.push_back(0);  // adaptive filtering
  ihdr.push_back(0);  // not interlaced
  png_chunk(out, "IHDR", ihdr);
  png_chunk(out, "IDAT", zlib_stored(raw));
  png_chunk(out, "IEND", std::string());
  return write_bin(path, out.data(), out.size());
}

std::string fmt_uint(size_t v) {
  std::string s = std::to_string(v);
  std::string out;
  int c = 0;
  for (int i = (int)s.size() - 1; i >= 0; i--) {
    out.push_back(s[i]);
    if (++c % 3 == 0 && i > 0) out.push_back(',');
  }
  std::string rev(out.rbegin(), out.rend());
  return rev;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(__APPLE__)
  @autoreleasepool {
#endif
    if (argc < 2) { std::fputs(kUsage, stderr); return 2; }
    if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
      std::fputs(kUsage, stdout);
      return 0;
    }
    if (std::strcmp(argv[1], "run") != 0) { std::fputs(kUsage, stderr); return 2; }

    Options o;
    for (int i = 2; i < argc; i++) {
      std::string a = argv[i];
      auto next = [&](const char* what) -> std::string {
        if (i + 1 >= argc) { std::fprintf(stderr, "weft: %s needs a value\n", what); std::exit(2); }
        return argv[++i];
      };
      if (a == "--kernel") o.kernel = next("--kernel");
      else if (a == "--grid") o.grid = (uint32_t)std::strtoul(next("--grid").c_str(), nullptr, 10);
      else if (a == "--threads") {
        const std::string v = next("--threads");
        o.threads = (uint32_t)std::strtoul(v.c_str(), nullptr, 10);
        // The runtime's cap, and the hardware's: a group of more than 1024 is not a
        // group on the GPU either. Saying so here is the difference between a
        // refusal and a launch that quietly ran at a size nobody asked for.
        if (o.threads == 0 || o.threads > 1024) {
          std::fprintf(stderr,
                       "weft: --threads %s is not a threadgroup size; the most a group can "
                       "have is 1024\n",
                       v.c_str());
          return 2;
        }
        o.threads_set = true;
      }
      else if (a == "--bench") o.bench = (uint32_t)std::strtoul(next("--bench").c_str(), nullptr, 10);
      else if (a == "--cpu-only") o.cpu_only = true;
      else if (a == "--buffer-bytes") {
        const std::string v = next("--buffer-bytes");
        size_t pos = 0;
        for (;;) {
          const size_t comma = v.find(',', pos);
          const std::string one =
              v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
          const size_t eq = one.find('=');
          if (eq == std::string::npos) {
            std::fprintf(stderr, "weft: --buffer-bytes wants I=N[,J=N], got '%s'\n", v.c_str());
            std::exit(2);
          }
          const long idx = std::strtol(one.substr(0, eq).c_str(), nullptr, 10);
          const unsigned long n = std::strtoul(one.substr(eq + 1).c_str(), nullptr, 10);
          if (idx < 0 || n == 0) {
            std::fprintf(stderr,
                         "weft: --buffer-bytes wants a non-negative index and a non-zero width\n");
            std::exit(2);
          }
          o.widths.push_back({(int)idx, (uint32_t)n});
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
      }
      else if (a == "--emit") o.emit = true;
      else if (a == "--emit-gpu") o.emit_gpu = true;
      else if (a == "--arg") {
        std::string v = next("--arg");
        size_t eq = v.find('=');
        if (eq == std::string::npos) {
          std::fprintf(stderr, "weft: --arg needs INDEX=VALUE\n");
          return 2;
        }
        const int idx = std::atoi(v.substr(0, eq).c_str());
        const std::string val = v.substr(eq + 1);
        uint64_t bits = 0;
        if (val.find('.') != std::string::npos || val.find('e') != std::string::npos) {
          float f = std::strtof(val.c_str(), nullptr);
          uint32_t b;
          std::memcpy(&b, &f, 4);
          bits = b;
        } else {
          bits = std::strtoull(val.c_str(), nullptr, 0);
        }
        o.args.emplace_back(idx, bits);
      }
      else if (a == "--dump-gpu") {
        std::string v = next("--dump-gpu");
        size_t eq = v.find('=');
        if (eq == std::string::npos) {
          std::fprintf(stderr, "weft: --dump-gpu needs BUFFER=path\n");
          return 2;
        }
        o.gpu_dumps.emplace_back(std::atoi(v.substr(0, eq).c_str()), v.substr(eq + 1));
      }
      else if (a == "--load") {
        std::string v = next("--load");
        size_t eq = v.find('=');
        if (eq == std::string::npos) {
          std::fprintf(stderr, "weft: --load needs BUFFER=path\n");
          return 2;
        }
        o.loads.emplace_back(std::atoi(v.substr(0, eq).c_str()), v.substr(eq + 1));
      }
      else if (a == "--dump") {
        std::string v = next("--dump");
        size_t eq = v.find('=');
        if (eq == std::string::npos) {
          std::fprintf(stderr, "weft: --dump needs BUFFER=path\n");
          return 2;
        }
        o.dumps.emplace_back(std::atoi(v.substr(0, eq).c_str()), v.substr(eq + 1));
      }
      else if (a == "--out") {
        o.out = next("--out");
      } else if (a == "--out-gpu") {
        o.out_gpu = next("--out-gpu");
      } else if (a == "--render") {
        std::string v = next("--render");
        size_t x = v.find('x');
        if (x == std::string::npos) {
          std::fprintf(stderr, "weft: --render needs WxH, got '%s'\n", v.c_str());
          return 2;
        }
        o.render = true;
        o.rw = (uint32_t)std::strtoul(v.substr(0, x).c_str(), nullptr, 10);
        o.rh = (uint32_t)std::strtoul(v.substr(x + 1).c_str(), nullptr, 10);
        if (!o.rw || !o.rh) { std::fprintf(stderr, "weft: --render needs a non-zero WxH\n"); return 2; }
      }
      else if (a == "--textures") {
        std::string v = next("--textures");
        size_t start = 0;
        while (start <= v.size()) {
          size_t comma = v.find(',', start);
          std::string part = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
          size_t x = part.find('x');
          if (x == std::string::npos) {
            std::fprintf(stderr, "weft: --textures needs WxH entries, got '%s'\n", part.c_str());
            return 2;
          }
          // `WxH*L` is a 2D array of L slices, `WxHxD` is a volume of D planes,
          // `WxHc` is a cube of six square faces, `WxHd` is a 1D texture — one row,
          // which is what `texture1d` and PTX's `tex.1d` both read — and without any
          // of them it is one 2D image.
          const std::string rest = part.substr(x + 1);
          const size_t star = rest.find('*');
          const size_t dep = rest.find('x');
          const size_t cub = rest.find('c');
          const size_t ond = rest.find('d');
          size_t stop = star < dep ? star : dep;
          if (cub < stop) stop = cub;
          if (ond < stop) stop = ond;
          const uint32_t tw = (uint32_t)std::strtoul(part.substr(0, x).c_str(), nullptr, 10);
          const uint32_t th = (uint32_t)std::strtoul(rest.substr(0, stop).c_str(), nullptr, 10);
          const bool cube = cub != std::string::npos;
          const bool one_d = ond != std::string::npos;
          const uint32_t tl =
              cube ? 6u
                   : star == std::string::npos
                         ? 1u
                         : (uint32_t)std::strtoul(rest.substr(star + 1).c_str(), nullptr, 10);
          const uint32_t td = dep == std::string::npos
                                  ? 0u
                                  : (uint32_t)std::strtoul(rest.substr(dep + 1).c_str(), nullptr, 10);
          if (!tw || !th || !tl || (dep != std::string::npos && !td) || (cube && tw != th) ||
              (one_d && th != 1u)) {
            std::fprintf(stderr, "weft: --textures entry '%s' needs WxH and a non-zero count%s\n",
                         part.c_str(),
                         cube    ? " (a cube is square)"
                         : one_d ? " (a 1D texture is one row: use Wx1d)"
                                 : "");
            return 2;
          }
          o.textures.push_back(Options::TexBind{tw, th, tl, td, cube, one_d});
          if (comma == std::string::npos) break;
          start = comma + 1;
        }
      }
      else if (a == "--samplers") {
        std::string v = next("--samplers");
        size_t start = 0;
        while (start <= v.size()) {
          size_t comma = v.find(',', start);
          std::string part = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
          int filter = part.find("linear") != std::string::npos ? 1 : 0;
          int addr = 0;
          if (part.find("zero") != std::string::npos) addr = 1;
          else if (part.find("repeat") != std::string::npos) addr = 2;
          o.samplers.emplace_back(filter, addr);
          if (comma == std::string::npos) break;
          start = comma + 1;
        }
      }
      else if (a == "--buffers") {
        std::string v = next("--buffers");
        size_t start = 0;
        while (start <= v.size()) {
          size_t comma = v.find(',', start);
          std::string part = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
          if (!part.empty()) o.bufs.push_back((size_t)std::strtoull(part.c_str(), nullptr, 10));
          if (comma == std::string::npos) break;
          start = comma + 1;
        }
      } else if (!a.empty() && a[0] != '-') {
        o.file = a;
      } else {
        std::fprintf(stderr, "weft: unknown option %s\n", a.c_str());
        return 2;
      }
    }
    if (o.file.empty()) { std::fputs(kUsage, stderr); return 2; }
    if (o.bench == 0) o.bench = 1;

    bool ok = false;
    std::string src = read_file(o.file, ok);
    if (!ok) { std::fprintf(stderr, "weft: cannot read %s\n", o.file.c_str()); return 1; }

    o.ptx = !o.render && o.file.size() > 4 && o.file.compare(o.file.size() - 4, 4, ".ptx") == 0;
    // --out is a render pass's frame: a compute run has no image to write, so
    // say that here rather than translating the file and failing at the end.
    if ((!o.out.empty() || !o.out_gpu.empty()) && !o.render) {
      std::fprintf(stderr, "weft: --out needs a render pass (--render WxH)\n");
      return 2;
    }
    const bool spv = !o.render && o.file.size() > 4 && o.file.compare(o.file.size() - 4, 4, ".spv") == 0;
    // Both binary front ends have a Metal back end now — the same decode emitted
    // for the GPU — so both get a GPU column, which is set up below.
    o.front = spv ? "spir-v" : (o.ptx ? "ptx" : "metal");

    // A render pass translates `fragment` functions and needs the target's
    // width, since `[[position]]` is derived from the pixel index.
    RenderMode rm;
    if (o.render) {
      rm.fragment = o.kernel;
      rm.width = o.rw;
      rm.height = o.rh;
      // A fragment's grid is the target: one invocation per pixel.
      o.grid = o.rw * o.rh;
    }
    TranslateResult tr = spv ? translate_spirv(src, o.widths)
                             : o.ptx ? translate_ptx(src, o.widths)
                                     : translate_msl(src, o.render ? &rm : nullptr);
    if (!tr.ok) { std::fprintf(stderr, "weft: %s\n", tr.error.c_str()); return 1; }

    // ---- the GPU column for PTX ----
    // The PTX front end has a second back end: the same decode emitted as Metal
    // (src/weft_msl.hpp is the runtime behind it). When it produces kernels the
    // GPU run compiles and executes those, so the comparison below is the CPU
    // and the GPU running the same generated code rather than two
    // implementations that were written to agree. When it cannot — a 64-bit
    // atomic, a 64-bit float, which this host's Metal has no form for — the run
    // names the instruction and keeps the CPU column instead of passing a twin
    // off as one.
    std::string gpu_refusal;
    if ((o.ptx || spv) && !o.cpu_only) {
      TranslateResult g = o.ptx ? translate_ptx_gpu(src, o.widths)
                                : translate_spirv_gpu(src, o.widths);
      if (g.ok) {
        tr.gpu_src = g.gpu_src;
        // The GPU translation is the one that decides a kernel has no per-lane
        // body and needs the warp back end, and that decision is the launcher's
        // dispatch geometry — one thread per warp instead of one per lane. The
        // kernel list comes from the CPU translation, so the flag travels across
        // by name rather than being guessed from the source.
        for (size_t i = 0; i < tr.kernels.size() && i < g.kernels.size(); i++)
          if (tr.kernels[i].name == g.kernels[i].name) {
            tr.kernels[i].gpu_warp = g.kernels[i].gpu_warp;
            tr.kernels[i].wide = g.kernels[i].wide;
          }
      } else {
        gpu_refusal = g.error;
        o.cpu_only = true;
      }
    }

    if (o.emit) { std::fputs(tr.code.c_str(), stdout); return 0; }

    // Pick the kernel.
    int kidx = -1;
    if (o.kernel.empty()) {
      if (tr.kernels.size() != 1) {
        std::fprintf(stderr, "weft: %s has %zu kernels; pick one with --kernel\n", o.file.c_str(),
                     tr.kernels.size());
        for (const auto& k : tr.kernels) std::fprintf(stderr, "     %s\n", k.name.c_str());
        return 2;
      }
      kidx = 0;
    } else {
      for (size_t i = 0; i < tr.kernels.size(); i++)
        if (tr.kernels[i].name == o.kernel) kidx = (int)i;
      if (kidx < 0) {
        std::fprintf(stderr, "weft: no kernel named '%s'\n", o.kernel.c_str());
        return 2;
      }
    }
    const TranslatedKernel& kern = tr.kernels[kidx];
    // A module that states the launch size it was compiled for (Triton's
    // `.maxntid`, from `num_warps`) gets it as the default. The threadgroup size
    // decides which elements a block covers, so guessing a different one runs a
    // different program. `--threads` always wins.
    if (!o.threads_set && kern.tg_hint) o.threads = kern.tg_hint;
    // A module that states the group it was written to run in without depending on
    // it — SPIR-V's `LocalSize` — gets it as the default too, but not as a
    // constraint: the kernel is indexed by its global invocation id.
    if (!o.threads_set && !kern.tg_hint && kern.def_threads) o.threads = kern.def_threads;
    // A group wider than a row is not a suggestion: the lane identity is
    // decomposed by its shape, so the launch is that size or every lane holds a
    // different identity than the module wrote about.
    if (kern.fixed_threads && o.threads != kern.fixed_threads) {
      std::printf("threads   %u given, this kernel declares a %u-lane group: using %u\n",
                  o.threads, kern.fixed_threads, kern.fixed_threads);
      o.threads = kern.fixed_threads;
    }
    // What each texture binding has to be. A cube kernel handed a 2-D texture is
    // not a slow run — the sampler reads a texel nobody asked for — so the shape
    // the command line gave is checked against the shape the module declared, in
    // binding order.
    for (size_t t = 0; t < kern.tex_shape.size() && t < o.textures.size(); t++) {
      const std::string& want = kern.tex_shape[t];
      if (want.empty()) continue;
      const Options::TexBind& tb = o.textures[t];
      const std::string got = tb.cube  ? "cube"
                              : tb.one_d ? "1d"
                              : tb.d > 0 ? "3d"
                              : tb.layers > 1 ? "array"
                                              : "2d";
      if (want != got) {
        std::fprintf(stderr,
                     "weft: kernel '%s' binds texture %zu as a %s image, but --textures gave a "
                     "%s one: a %s binding needs a %s entry\n",
                     kern.name.c_str(), t, want.c_str(), got.c_str(), want.c_str(),
                     want == "cube"   ? "WxHc"
                     : want == "1d"   ? "Wx1d"
                     : want == "3d"   ? "WxHxD"
                     : want == "array" ? "WxH*L"
                                      : "WxH");
        return 2;
      }
    }
    // A mesh stage produces one mesh per threadgroup, and the hardware caps a
    // mesh threadgroup at 32 lanes, so that is the group size unless one is given.
    if (!o.threads_set && kern.mesh) o.threads = 32;
    // A kernel with only a warp body — a tensor-core fragment — is one warp wide
    // by construction: its fragment map is derived from `%tid.x` inside the
    // block, so the block *is* that many lanes and a wider group is not a launch
    // the hardware would accept either. The module's declared size wins (32 when
    // it declares none), and the run says so rather than quietly computing a
    // different program.
    if (kern.warp_only) {
      const uint32_t block = kern.tg_hint ? kern.tg_hint : 32u;
      if (o.threads != block) {
        std::printf("threads   %u given, this kernel is a %u-lane warp block: using %u\n",
                    o.threads, block, block);
        o.threads = block;
      }
    }
    // A grid smaller than the block the module declares leaves lanes that do
    // not exist, and the program is written for the whole block: a tensor-core
    // fragment map would stage rows nobody fills, and the answer would be
    // whatever those slots held. Real hardware runs that launch, so this names it
    // as a launch the program was not written for rather than reporting bits.
    if (kern.tg_hint && o.grid < kern.tg_hint) {
      std::fprintf(stderr,
                   "weft: kernel '%s' is a %u-lane block and --grid gave %u: give it at "
                   "least one full block (--grid %u or more)\n",
                   kern.name.c_str(), kern.tg_hint, o.grid, kern.tg_hint);
      return 2;
    }
    // A mesh or task stage runs on the GPU too: the translator lowers it to a
    // compute kernel over the same buffers (see lower_gpu_stages), so the two
    // engines compute the same thing and the comparison is a real one. A mesh
    // *pipeline* is M3 hardware; a mesh stage's output is not.
    // A render pass may declare a `[[tile]]` stage; it runs before the fragment
    // over the same pixels, because the launcher gives it the same grid.
    int tile_kidx = -1;
    if (o.render)
      for (size_t i = 0; i < tr.kernels.size(); i++)
        if (tr.kernels[i].tile) tile_kidx = (int)i;
    // A task stage ahead of the mesh stage it feeds, if the file has one.
    int task_kidx = -1;
    for (size_t i = 0; i < tr.kernels.size(); i++)
      if (tr.kernels[i].task) task_kidx = (int)i;
    if (o.bufs.empty()) o.bufs.assign((size_t)kern.nbufs, (size_t)o.grid);

    // ---- translate -> C++ -> dylib ----
    const char* root = WEFT_ROOT;
    std::string build = std::string(root) + "/build";
    ::mkdir(build.c_str(), 0755);
    // The generated translation unit is a pure function of the kernel source, so
    // its bytes name the compiled library: hash the code, the runtime it links
    // against and the flags, and a second run with the same kernel skips clang++
    // entirely. An edit to the kernel changes the hash, so a stale library cannot
    // be picked up — the cache is keyed by content, not by file name.
    auto fnv = [](const std::string& s, uint64_t h) {
      for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
      return h;
    };
    uint64_t hash = 1469598103934665603ull;
    hash = fnv(tr.code, hash);
    {
      bool ok = false;
      hash = fnv(read_file(std::string(root) + "/runtime/weft_runtime.hpp", ok), hash);
      hash = fnv(read_file(std::string(root) + "/runtime/weft_runtime.cpp", ok), hash);
    }
    // The generated module is compiled by this host's C++ compiler. `WEFT_CXX`
    // overrides it; on Linux the usual `c++`, which is gcc or clang, is enough.
    // The flags are part of the cache key, because a different compiler or a
    // different `-O` level builds a different library.
    // `WEFT_CXX` names the compiler. Without it: clang++ if this host has one,
    // the system C++ compiler otherwise — which on a Linux box is often the only
    // one there is, and gcc is what `c++` means there.
    const char* cxx_env = std::getenv("WEFT_CXX");
    const std::string cxx =
        cxx_env ? std::string(cxx_env)
                : (std::system("command -v clang++ >/dev/null 2>&1") == 0 ? "clang++" : "c++");
#if defined(__APPLE__)
    const char* kSharedFlag = "-dynamiclib";
    const char* kLibExt = ".dylib";
#else
    const char* kSharedFlag = "-shared";
    const char* kLibExt = ".so";
#endif
    hash = fnv(cxx + " -std=c++20 -O2 -fsigned-char " + kSharedFlag + " -fPIC " + root, hash);
    char hexkey[32];
    std::snprintf(hexkey, sizeof hexkey, "%016llx", (unsigned long long)hash);
    const std::string cache = build + "/cache";
    ::mkdir(cache.c_str(), 0755);
    const std::string gen = build + "/generated.cpp";
    const std::string lib = cache + "/" + hexkey + kLibExt;
    const bool cached = ::access(lib.c_str(), R_OK) == 0;

    double t0 = now_ms();
    if (!cached) {
      // Written even on a cache hit: it is the last translation, which is what
      // you read when you want to see what the kernel became.
      if (!write_file(gen, tr.code)) {
        std::fprintf(stderr, "weft: cannot write %s\n", gen.c_str());
        return 1;
      }
      // Compiled to a temporary name and renamed: two runs of the same kernel at
      // once must not read a half-written library.
      const std::string tmp = lib + ".tmp." + std::to_string(::getpid());
      // `-fsigned-char`: Metal's `char` is a signed 8-bit integer. Where the
      // platform's plain `char` is unsigned — aarch64 Linux — a kernel's
      // `(char)-2` would be 254 and would not fit an `int8_t`, so the module is
      // built with the signedness the shader means. It is a no-op where `char`
      // is already signed.
      const std::string base = cxx + " -std=c++20 -O2 -fsigned-char " + kSharedFlag + " -fPIC -I\"" + root +
                               "/runtime\" -I\"" + root + "/include\" \"" + root +
                               "/runtime/weft_runtime.cpp\" \"" + gen + "\" -o \"" + tmp + "\"";
      const std::string logpath = build + "/clang.log";
      int rc = std::system((base + " 2> \"" + logpath + "\"").c_str());
      std::string log;
      {
        bool lok = false;
        log = read_file(logpath, lok);
      }
      // gcc makes C++20 coroutines an opt-in flag and clang does not, so the
      // compiler is asked rather than guessed: a failure whose message is about
      // `co_await` means this one wants `-fcoroutines`, and nothing else does.
      if (rc != 0 && log.find("co_await") != std::string::npos) {
        rc = std::system((base + " -fcoroutines 2> \"" + logpath + "\"").c_str());
        bool lok = false;
        log = read_file(logpath, lok);
      }
      if (rc != 0) {
        std::remove(tmp.c_str());
        std::fprintf(stderr, "weft: compiling the translated kernel failed (%s)\n%s\n",
                     cxx.c_str(), log.c_str());
        return 1;
      }
      std::rename(tmp.c_str(), lib.c_str());
    }
    const double compile_ms = cached ? 0.0 : now_ms() - t0;

    void* handle = dlopen(lib.c_str(), RTLD_NOW);
    if (!handle) { std::fprintf(stderr, "weft: %s\n", dlerror()); return 1; }
    auto run_cpu = (void (*)(int, void**, uint32_t, uint32_t))dlsym(handle, "weft_run");
    // Tile memory is one slot per pixel here, so a read of another tile is
    // representable — on the GPU it is undefined and order-dependent. The
    // module counts those reads; if any happened there is nothing to compare.
    auto tile_reset = (void (*)())dlsym(handle, "weft_tile_reset");
    auto tile_violations = (uint64_t (*)())dlsym(handle, "weft_tile_violations");
    auto kernel_count = (int (*)())dlsym(handle, "weft_kernel_count");
    auto kernel_at = (const weft_kernel_info* (*)(int))dlsym(handle, "weft_kernel_at");
    auto set_param = (void (*)(int, uint64_t))dlsym(handle, "weft_set_param");
    if (set_param)
      for (const auto& a : o.args) set_param(a.first, a.second);

    // Textures are inputs: the same pixels from the same generator feed the GPU
    // and the CPU, so a texture kernel compares like any other.
    // A binding may be a 2D array: its slices are filled from the same generator,
    // one after another, so slice L of the GPU's texture and slice L here hold
    // the same texels and a layer index has to mean the same thing on both.
    std::vector<std::vector<float>> texdata(o.textures.size());
    for (size_t t = 0; t < o.textures.size(); t++) {
      const uint32_t outer = o.textures[t].d > 0 ? o.textures[t].d : o.textures[t].layers;
      const size_t n = (size_t)o.textures[t].w * o.textures[t].h * outer;
      texdata[t].resize(n);
      for (size_t i = 0; i < n; i++) texdata[t][i] = det_value((uint32_t)i, (uint32_t)(100 + t));
    }
    auto set_texture =
        (void (*)(int, void*, uint32_t, uint32_t, uint32_t))dlsym(handle, "weft_set_texture");
    auto set_texture_layers = (void (*)(int, void*, uint32_t, uint32_t, uint32_t, uint32_t))
        dlsym(handle, "weft_set_texture_layers");
    if (set_texture_layers)
      for (size_t t = 0; t < o.textures.size(); t++)
        set_texture_layers((int)t, texdata[t].data(), o.textures[t].w, o.textures[t].h, 4,
                           o.textures[t].d > 0 ? o.textures[t].d : o.textures[t].layers);
    else if (set_texture)
      for (size_t t = 0; t < o.textures.size(); t++) {
        if (o.textures[t].layers != 1) {
          std::fprintf(stderr, "weft: this module cannot bind an array texture\n");
          return 2;
        }
        set_texture((int)t, texdata[t].data(), o.textures[t].w, o.textures[t].h, 4);
      }
    // Samplers are host state for a SPIR-V kernel, the way a descriptor is on a
    // GPU: one is created per index and the kernel reads through it.
    if (kern.ntex > (int)o.textures.size()) {
      std::fprintf(stderr, "weft: kernel '%s' reads %d texture%s, --textures gave %zu\n",
                   kern.name.c_str(), kern.ntex, kern.ntex == 1 ? "" : "s", o.textures.size());
      return 2;
    }
    auto set_sampler = (void (*)(int, int, int))dlsym(handle, "weft_set_sampler");
    if (set_sampler) {
      const size_t want = (size_t)kern.nsamp > o.samplers.size() ? (size_t)kern.nsamp : o.samplers.size();
      for (size_t s = 0; s < want; s++) {
        const std::pair<int, int> v = s < o.samplers.size() ? o.samplers[s] : std::pair<int, int>{0, 0};
        set_sampler((int)s, v.first, v.second);
      }
    }
    if (!run_cpu || !kernel_count) { std::fprintf(stderr, "weft: bad kernel library\n"); return 1; }
    if (kernel_count() != (int)tr.kernels.size()) { std::fprintf(stderr, "weft: kernel table mismatch\n"); return 1; }

    // ---- task stage, then the meshgrid it asked for ----
    // A task stage runs before the meshes it feeds: it fills the payload and says
    // how many meshes to dispatch. It runs over the grid given on the command
    // line, and what it asks for becomes the mesh stage's grid — that is the whole
    // job of the stage. A mesh reads the payload element of the task group that
    // owns its block of the meshgrid, which the launcher fills in below.
    std::vector<uint32_t> task_payload;  // one element per task threadgroup
    std::vector<uint32_t> mesh_payload;  // one element per mesh, after the copy
    uint32_t task_groups = 0;
    if (kern.mesh && task_kidx >= 0) {
      const TranslatedKernel& tk = tr.kernels[task_kidx];
      task_groups = (o.grid + o.threads - 1) / o.threads;
      if (task_groups == 0) task_groups = 1;
      const weft_kernel_info* ti = kernel_at ? kernel_at(task_kidx) : nullptr;
      const uint32_t pbytes =
          (ti && ti->buf_bytes && tk.payload_idx >= 0) ? ti->buf_bytes[tk.payload_idx] : 4u;
      if (tk.payload_idx >= 0)
        task_payload.assign((size_t)task_groups * (pbytes / 4u) +
                                (pbytes % 4u ? 1u : 0u),
                            0u);
      // The task stage's own buffers are the caller's: whatever it declares, it
      // gets the same deterministic input the mesh stage would.
      std::vector<std::vector<float>> thost((size_t)tk.nbufs);
      for (size_t b = 0; b < thost.size(); b++) {
        const size_t n = b < o.bufs.size() ? o.bufs[b] : (size_t)o.grid;
        thost[b].assign(n, 0.0f);
        for (size_t i = 0; i < n; i++) thost[b][i] = det_value((uint32_t)i, (uint32_t)b);
      }
      std::vector<void*> tptr((size_t)tk.nargs, nullptr);
      for (size_t b = 0; b < thost.size(); b++) tptr[b] = thost[b].data();
      if (tk.payload_idx >= 0 && !task_payload.empty())
        tptr[(size_t)tk.payload_idx] = task_payload.data();
      std::vector<uint32_t> props(3, 0u);
      if (tk.props_idx >= 0) tptr[(size_t)tk.props_idx] = props.data();
      if (tile_reset) tile_reset();
      run_cpu(task_kidx, tptr.data(), o.grid, o.threads);
      const uint32_t set = props[0] ? props[0] : task_groups;
      // The meshgrid is cut into contiguous blocks, one per task threadgroup, and
      // every mesh in a block gets that group's payload. A mesh then reads its
      // own element — the partition is defined here, in the launcher, so nothing
      // a mesh computes depends on an assignment left to the hardware.
      if (tk.payload_idx >= 0 && pbytes >= 4) {
        const size_t words = pbytes / 4u;
        mesh_payload.assign((size_t)set * words, 0u);
        for (uint32_t m = 0; m < set; m++) {
          const uint32_t owner = (uint32_t)((uint64_t)m * task_groups / set);
          const uint32_t at = owner < task_groups ? owner : task_groups - 1u;
          for (size_t w = 0; w < words; w++)
            mesh_payload[(size_t)m * words + w] = task_payload[(size_t)at * words + w];
        }
      }
      std::printf("task      %u threadgroup%s of %u lanes  ·  meshgrid it set: %u mesh%s\n",
                  task_groups, task_groups == 1 ? "" : "s", o.threads, set,
                  set == 1 ? "" : "es");
      // From here the mesh stage runs over what the task stage asked for.
      o.grid = set * 32u;
      o.threads = 32u;
      o.threads_set = true;
    }

    // ---- buffers ----
    const size_t nb = (size_t)kern.nbufs;  // buffers the stage declares itself
    // A render pass has up to two more buffers than the fragment declares: tile
    // memory, one float4 per pixel, if the pass has an imageblock, and the
    // colour target, which the launcher creates and the fragment's `return`
    // writes.
    // Tile memory belongs to a pass whose fragment reads it: the tile stage
    // fills it, and a pass with no reader has nothing to fill.
    const bool tile_mem = o.render && tile_kidx >= 0 && kern.tile_idx >= 0;
    const size_t ntile = tile_mem ? 1 : 0;
    // A mesh stage brings its own two outputs, bound after the buffers the kernel
    // declares and before tile memory: one vertex block and one triangle block
    // per mesh, where a mesh is a threadgroup.
    const uint64_t meshes = kern.mesh ? ((uint64_t)o.grid + o.threads - 1) / o.threads : 0;
    // Every binding the module asked for, in binding order: the caller's buffers,
    // then what the launcher creates (mesh blocks, a payload, the meshgrid
    // handle, tile memory) and, for a render pass, the colour target.
    const size_t nbuf = (size_t)kern.nargs;
    const size_t nbind = nbuf - (o.render ? 1u : 0u);
    const size_t colour_idx = nbind;
    if (o.bufs.size() != nb) {
      std::fprintf(stderr, "weft: kernel '%s' takes %zu buffers, --buffers gave %zu\n",
                   kern.name.c_str(), nb, o.bufs.size());
      return 2;
    }
    const int tile_run = tile_mem ? tile_kidx : -1;
    // A tile is 32x32 pixels and a tile function is dispatched one threadgroup per
    // tile, so its group is the tile — not whatever --threads says. That is the
    // size `threads_per_threadgroup` has to report inside it, and it is what makes
    // threadgroup memory inside a tile stage mean what it means on a GPU.
    const uint32_t tile_threads = 32 * 32;
    if (tile_run >= 0 && tr.kernels[tile_run].ntex > (int)o.textures.size()) {
      std::fprintf(stderr, "weft: the tile stage '%s' reads %d texture%s, --textures gave %zu\n",
                   tr.kernels[tile_run].name.c_str(), tr.kernels[tile_run].ntex,
                   tr.kernels[tile_run].ntex == 1 ? "" : "s", o.textures.size());
      return 2;
    }
    if (tile_kidx >= 0 && (size_t)tr.kernels[tile_kidx].nbufs != nb) {
      std::fprintf(stderr, "weft: the tile stage '%s' takes %zu buffers, the fragment %zu\n",
                   tr.kernels[tile_kidx].name.c_str(), (size_t)tr.kernels[tile_kidx].nbufs, nb);
      return 2;
    }
    // Element widths come from the module: `sizeof` of the type each buffer was
    // declared with, so `--buffers N` is N elements of *that* type — 4 bytes for a
    // float*, 16 for a float4*, 12 for a packed_float3*. The front ends that bind
    // addresses report the width their own loads and stores address the buffer at
    // (or the one `--buffer-bytes` stated), and 4 bytes is the fallback only where
    // the module has nothing to say.
    // Every bound buffer gets the width the module reports, mesh outputs included:
    // the vertex block is `sizeof` of the shader's own vertex struct, and a
    // primitive is three indices. The colour target has no entry and is not
    // sized from here.
    std::vector<uint32_t> elem(nbuf, 4);
    if (kernel_at) {
      const weft_kernel_info* ki = kernel_at(kidx);
      for (size_t b = 0; ki && ki->buf_bytes && b < nbind; b++)
        if (ki->buf_bytes[b]) elem[b] = ki->buf_bytes[b];
    }
    // No override here on purpose: a width the caller states is already in the
    // module's ABI (the front ends take it as an argument), so `elem` above is the
    // one answer the run and the module both read. A second path would drift.
    // Tile memory is a 32x32 block per tile, so its size is the number of tiles
    // the target covers — including the short last row and column a target that
    // is not a multiple of 32 still pays for, exactly as the hardware does.
    const uint64_t tiles_x = (o.rw + 31u) / 32u, tiles_y = (o.rh + 31u) / 32u;
    std::vector<GuardedFloats> host(nbuf);
    for (size_t b = 0; b < nbuf; b++) {
      uint64_t bytes;
      if (b < nb)
        bytes = (uint64_t)o.bufs[b] * elem[b];
      else if (kern.mesh && b == nb)
        bytes = meshes * (uint64_t)kern.mesh_nv * elem[b];
      else if (kern.mesh && b == nb + 1)
        bytes = meshes * (uint64_t)kern.mesh_np * elem[b];
      else if ((int)b == kern.payload_idx)
        // One element per mesh, and one per task threadgroup when the task stage
        // is run on its own (`--kernel taskgen`): the stage indexes the payload
        // by its own group id, so the block is as wide as the grid it runs over.
        bytes = (kern.mesh ? meshes : std::max<uint64_t>(1, (o.grid + o.threads - 1) / o.threads)) *
                (uint64_t)elem[b];
      else if ((int)b == kern.props_idx)
        bytes = 12ull;  // x, y, z
      else if (ntile)
        bytes = tiles_x * tiles_y * 1024ull * 16ull;
      else
        bytes = (uint64_t)o.rw * o.rh * 16;
      const size_t n = (size_t)((bytes + 3) / 4);
      host[b].resize(n);
      for (size_t i = 0; i < n; i++) host[b][i] = det_value((uint32_t)i, (uint32_t)b);
      // The payload is the task stage's output, not an input the generator makes
      // up: every mesh reads the element of the task group that owns its block,
      // and a mesh whose file has no task stage reads a defined zero.
      if ((int)b == kern.payload_idx) {
        std::memset(host[b].data(), 0, host[b].size() * sizeof(float));
        if (!mesh_payload.empty())
          std::memcpy(host[b].data(), mesh_payload.data(),
                      std::min(host[b].size(), mesh_payload.size()) * sizeof(uint32_t));
      }
    }
    if (!o.bufs.empty() && !nb) {
      std::fprintf(stderr, "weft: kernel '%s' takes no buffers\n", kern.name.c_str());
      return 2;
    }
    // The colour target starts cleared, as the GPU's does.
    // RGBA: four floats per pixel, cleared to the GPU attachment's clear colour.
    if (o.render) host[colour_idx].assign((size_t)o.rw * o.rh * 4, 0.0f);
    // A buffer can be fed from a file instead of the deterministic generator,
    // which is what makes a run reproducible from outside: the same bytes in, the
    // same bytes out, whoever produced them.
    for (const auto& l : o.loads) {
      if (l.first < 0 || (size_t)l.first >= nb) {
        std::fprintf(stderr, "weft: --load %d: no such buffer\n", l.first);
        return 2;
      }
      bool lok = false;
      const std::string data = read_file(l.second, lok);
      if (!lok) {
        std::fprintf(stderr, "weft: cannot read %s\n", l.second.c_str());
        return 1;
      }
      if (data.size() != host[l.first].size() * sizeof(float)) {
        std::fprintf(stderr, "weft: %s is %zu bytes, buffer %d is %zu\n", l.second.c_str(),
                     data.size(), l.first, host[l.first].size() * sizeof(float));
        return 1;
      }
      std::memcpy(host[l.first].data(), data.data(), data.size());
    }
    const std::vector<GuardedFloats> pristine = host;
    std::vector<void*> ptrs(nbuf);
    for (size_t b = 0; b < nbuf; b++) ptrs[b] = host[b].data();
    // The guard pages are only useful if the fault can be named, so every buffer a
    // kernel is handed is tagged with who it is and how big it is.
    for (size_t b = 0; b < nbuf; b++)
      guard_tag(host[b].data(), host[b].size() * sizeof(float), (int)b);
    g_guard_grid = o.grid;
    g_guard_threads = o.threads;

    // `--emit-gpu` sends the source to stdout so it can be written to a .metal
    // file, and a report in front of it is a syntax error rather than a
    // preamble: in that mode the source is the only thing that goes there.
    const bool report = !o.emit_gpu;
    if (report) std::printf("weft — gpu on cpu\n\n");
    if (kern.mesh && report)
      std::printf("mesh      %d vertices and %d primitives per mesh  ·  %llu mesh%s for this grid\n",
                  kern.mesh_nv, kern.mesh_np, (unsigned long long)meshes,
                  meshes == 1 ? "" : "es");
    if (o.render) {
      if (report)
        std::printf("fragment  %s  ·  target %ux%u  ·  %zu buffer%s\n", kern.name.c_str(), o.rw,
                    o.rh, nb, nb == 1 ? "" : "s");
    }
    else {
      // A front end that types its pointers names the element (`float4 16B`). One
      // that binds an address has no name to give, and calling it `float` would be
      // the tool inventing a layout — so it says `addr 8B`, a byte count without a
      // type. That count is measured (the width the instructions that reach the
      // buffer address it at), or stated with `--buffer-bytes`; only a buffer
      // nothing typed reaches is left at the 4-byte fallback, marked `*`.
      std::string types;
      bool unknown_width = false;
      for (size_t b = 0; b < nb; b++) {
        const std::string wid = b < kern.buf_type.size() ? kern.buf_type[b] : std::string();
        const bool literal = !wid.empty() && wid != "0" &&
                             wid.find_first_not_of("0123456789") == std::string::npos;
        const bool known = !wid.empty() && wid != "0" && !literal;
        types += (b ? ", " : "");
        types += known ? wid : std::string("addr");
        if ((known || literal) && b < elem.size() && elem[b]) {
          types += " " + std::to_string(elem[b]) + "B";
        } else {
          types += " " + std::to_string(b < elem.size() ? elem[b] : 4u) + "B*";
          unknown_width = true;
        }
      }
      if (report)
        std::printf("kernel    %s  ·  grid %s  ·  threadgroup %u  ·  %zu buffer%s (%s)\n",
                    kern.name.c_str(), fmt_uint(o.grid).c_str(), o.threads, nb,
                    nb == 1 ? "" : "s", types.c_str());
      if (report && unknown_width)
        std::printf("note      * nothing typed reaches the buffers marked *, so the element "
                    "width is\n"
                    "           unknown: 4 bytes is assumed. State one with "
                    "`--buffer-bytes I=N[,J=N]`.\n");
      // A parameter the kernel declares and never reads is not bound — Triton's
      // `global_scratch` is the common case — so the buffer count can be one
      // smaller than the signature suggests. Say which, rather than leaving the
      // caller to work out why --buffers wanted fewer.
      if (report && !kern.unused.empty()) {
        std::printf("note      * %zu declared parameter%s not bound (never read:", kern.unused.size(),
                    kern.unused.size() == 1 ? " is" : "s are");
        for (const auto& u : kern.unused) std::printf(" %s", u.c_str());
        std::printf(")\n");
      }
    }
    char timing[96];
    if (cached)
      std::snprintf(timing, sizeof timing, "cached");
    else
      std::snprintf(timing, sizeof timing, "%s %.0f ms", cxx.c_str(), compile_ms);
    if (report)
      std::printf("translate %zu kernel%s, %d with barriers  ·  %s  ·  %s\n",
                  tr.kernels.size(), tr.kernels.size() == 1 ? "" : "s", kern.sync ? 1 : 0,
                  timing, o.front.c_str());

    // ---- CPU path ----
    std::vector<std::vector<float>> cpu_out(nbuf);
    for (size_t b = 0; b < nbuf; b++) cpu_out[b].assign(host[b].size(), 0.0f);

    // One timed launch: restore the inputs, then the tile stage and the
    // fragment stage, which is the order the GPU runs them in and the only
    // order in which tile memory means anything.
    auto launch_once = [&]() -> double {
      for (size_t b = 0; b < nbuf; b++)
        std::memcpy(host[b].data(), pristine[b].data(), host[b].size() * sizeof(float));
      double s = now_ms();
      if (tile_reset) tile_reset();
      if (tile_run >= 0) run_cpu(tile_run, ptrs.data(), o.grid, tile_threads);
      run_cpu(kidx, ptrs.data(), o.grid, o.threads);
      return now_ms() - s;
    };

    // ---- how many cores? ----
    // A launch is cheap now that the pool keeps its threads parked, so the
    // answer belongs to the kernel rather than to the machine: elementwise work
    // saturates on two cores and loses the rest to bandwidth, while an
    // atomic-heavy kernel is faster on one thread because every worker fights
    // for the same cache lines. So time the candidates on this kernel and keep
    // the winner.
    const uint32_t hw = std::max(1u, (uint32_t)std::thread::hardware_concurrency());
    std::vector<uint32_t> cands{1};
    if (hw > 1) cands.push_back(2);
    if (hw > 5) cands.push_back(hw / 2);
    if (hw > 2) cands.push_back(hw);
    uint32_t threads = hw;
    // `--emit-gpu` asks for the shader source, not a run: nothing is launched, so a
    // kernel whose buffers were sized for another path cannot take the process
    // down before the source gets printed.
    if (cands.size() > 1 && o.bench >= 4 && !o.emit_gpu) {
      const uint32_t trials = std::min(o.bench, 5u);
      double best_trial = 1e18;
      for (uint32_t w : cands) {
        if (w > hw) continue;
        set_env("WEFT_THREADS", std::to_string(w));
        launch_once();  // warm this schedule's worker state
        double b = 1e18;
        for (uint32_t t = 0; t < trials; t++) b = std::min(b, launch_once());
        if (b < best_trial) { best_trial = b; threads = w; }
      }
      set_env("WEFT_THREADS", std::to_string(threads));
    }

    double best_cpu = 1e18, sum_cpu = 0.0;
    for (uint32_t it = 0; it < (o.emit_gpu ? 0u : o.bench + 1); it++) {
      for (size_t b = 0; b < nbuf; b++)
        std::memcpy(host[b].data(), pristine[b].data(), host[b].size() * sizeof(float));
      double s = now_ms();
      if (tile_reset) tile_reset();
      if (tile_run >= 0) run_cpu(tile_run, ptrs.data(), o.grid, tile_threads);
      run_cpu(kidx, ptrs.data(), o.grid, o.threads);
      double d = now_ms() - s;
      for (size_t b = 0; b < nbuf; b++) std::memcpy(cpu_out[b].data(), host[b].data(), host[b].size() * sizeof(float));
      if (it > 0) {  // first run is warm-up
        sum_cpu += d;
        if (d < best_cpu) best_cpu = d;
      }
    }

    if (tile_violations) {
      const unsigned long long hits = (unsigned long long)tile_violations();
      if (hits) {
        std::fprintf(stderr,
                     "weft: %s reads tile memory outside its own 32x32 tile "
                     "(%llu access%s).\n"
                     "      A tile stage and the fragments of a tile share that tile's "
                     "memory and nothing\n"
                     "      else: on the GPU the read is another tile's, which may not "
                     "have been written\n"
                     "      yet. There is no answer to compare, so this run is refused.\n",
                     kern.name.c_str(), hits, hits == 1 ? "" : "es");
        return 1;
      }
    }

    // ---- GPU path ----
    // The GPU column is Metal's, so it exists only in the Apple build. Every
    // other platform translates and launches the same kernel the same way; what
    // it does not have is hardware to diff against, and it says so rather than
    // printing an agreement it did not check.
    double gpu_avg = -1.0, gpu_best = -1.0;
    std::vector<std::vector<float>> gpu_out(nbuf);
    std::string gpu_name = "unavailable";
    bool gpu_ran = false;

#if defined(__APPLE__)
    // A kernel the Metal back end cannot build has no GPU column, and saying so
    // here — before the launch is set up — keeps the run a CPU one instead of
    // aborting it. Two things land here: a kernel with no GPU form at all (a
    // 64-bit atomic, a 64-bit float, a body only the warp back end has), which
    // the translator refuses by name, and a shader this host's compiler itself
    // fails on. Both are worth a line and neither is worth a wrong answer.
    // Why the column is missing, when it is, and where the time went before it
    // was. `WEFT_GPU_TRACE=1` writes one line per stage to stderr: this host's
    // compiler service and its driver both fail in ways where only the stage names
    // the fault, and a launch that never completes is not an answer either way.
    const bool gpu_trace = std::getenv("WEFT_GPU_TRACE") != nullptr;
    const double gpu_t0 = now_ms();
    auto gpu_trace_at = [&](const char* what) {
      if (gpu_trace) std::fprintf(stderr, "weft: gpu %8.3f ms  %s\n", now_ms() - gpu_t0, what);
    };

    // What the build check below already produced. It is the same source on the
    // same device, so the launch reuses it rather than compiling it a second time:
    // this host's compiler service dies on a shader now and then
    // (XPC_ERROR_CONNECTION_INTERRUPTED), and asking it for the identical build
    // twice per run is asking twice for the same chance to happen.
    id<MTLDevice> probed_dev = nil;
    id<MTLLibrary> probed_lib = nil;
    id<MTLComputePipelineState> probed_pso = nil;
    std::string probed_src;
    if (!o.cpu_only && !o.emit_gpu && (o.ptx || spv) && !tr.gpu_src.empty()) {
      gpu_trace_at("build check: compiling the emitted source");
      id<MTLDevice> chk = MTLCreateSystemDefaultDevice();
      if (chk) {
        NSString* cs = [[NSString alloc] initWithBytes:tr.gpu_src.data()
                                                length:tr.gpu_src.size()
                                              encoding:NSUTF8StringEncoding];
        MTLCompileOptions* co = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
          co.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
          co.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* cerr = nil;
        id<MTLLibrary> clib = [chk newLibraryWithSource:cs options:co error:&cerr];
        id<MTLFunction> cfn = clib ? [clib newFunctionWithName:[NSString stringWithUTF8String:
                                                               kern.name.c_str()]]
                                   : nil;
        id<MTLComputePipelineState> cpso =
            cfn ? [chk newComputePipelineStateWithFunction:cfn error:&cerr] : nil;
        if (cpso) {
          probed_dev = chk;
          probed_lib = clib;
          probed_pso = cpso;
          probed_src = tr.gpu_src;
        }
        gpu_trace_at(cpso ? "build check: the pipeline built" : "build check: the pipeline was refused");
        if (!cpso) {
          gpu_refusal = std::string("the metal backend cannot build '") + kern.name.c_str() +
                        "': " +
                        (cerr ? [[cerr localizedDescription] UTF8String] : "no such function");
          o.cpu_only = true;
        }
      }
    }
    if (!o.cpu_only) {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev) {
        std::fprintf(stderr, "weft: no Metal device (use --cpu-only)\n");
        return 1;
      }
      MTLSize max_tg = [dev maxThreadsPerThreadgroup];
      gpu_name = std::string([[dev name] UTF8String]) + " (" + std::to_string(max_tg.width) +
                 " threads/group)";
      // A render pass needs a vertex function the file does not have: the
      // launcher appends one that covers the target with a single triangle, so
      // every pixel gets exactly one fragment invocation.
      // The file as written, with tile memory lowered to a buffer: this host's
      // Metal compiler cannot compile an imageblock, so the translator hands
      // back the same file with that one thing spelled as the launcher binds it.
      std::string gpu_src = tr.gpu_src;
      if (o.render) gpu_src += kFullScreenVertex;
      // The shader's math goes through the portable implementation, the same one the
      // CPU path runs, so sqrt/rsqrt/pow are identical on both engines rather than
      // within an ulp of Metal's hardware approximations. Placed after the
      // includes, where float2/float3/float4 exist.
      // The emitted PTX module calls the language's own functions and the build
      // compiles with safe math, so the portable-math rewrite — which exists to
      // replace Metal's *approximations* in a .metal file — has nothing to do
      // here.
      if (o.ptx || spv) gpu_src = std::string();
      {
        if (gpu_src.empty()) gpu_src = tr.gpu_src;
      }
      if (!o.ptx && !spv) gpu_src = rewrite_math_calls(gpu_src);
      if (!o.ptx && !spv) {  // weft portable math
        size_t at = 0, p = 0;
        while ((p = gpu_src.find("\n#include", at)) != std::string::npos) at = p + 1;
        if (at == 0) {
          at = gpu_src.compare(0, 8, "#include") == 0 ? gpu_src.find('\n') : std::string::npos;
        } else {
          at = gpu_src.find('\n', at);
        }
        at = (at == std::string::npos) ? 0 : at + 1;
        gpu_src.insert(at, kWSoftMath);
      }
      if (o.emit_gpu) {
        std::fputs(gpu_src.c_str(), stdout);
        return 0;
      }
      NSString* ns = [[NSString alloc] initWithBytes:gpu_src.data()
                                              length:gpu_src.size()
                                            encoding:NSUTF8StringEncoding];
      NSError* err = nil;
      // Metal compiles with fast math by default: `sqrt` is `u * rsqrt.approx(u)`,
      // divide and rsqrt are approximations, and the compiler is free to fuse and
      // reassociate. That makes the GPU the imprecise side of an exactness claim,
      // and no CPU can reproduce an undocumented hardware approximation. Safe math
      // is IEEE — the same arithmetic the translated C++ performs — so the two
      // engines can be compared bit for bit instead of to a tolerance.
      MTLCompileOptions* copts = [MTLCompileOptions new];
      if (@available(macOS 15.0, *)) {
        copts.mathMode = MTLMathModeSafe;  // IEEE: no approximations, no reassociation
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        copts.fastMathEnabled = NO;       // the same setting, before mathMode existed
#pragma clang diagnostic pop
      }
      // The build check above compiled this exact text on this exact device, so
      // its library is the answer and there is nothing to ask the compiler for.
      // Anything else — a `.metal` file, whose source is rewritten for the target,
      // or a render pass, which appends a vertex stage — is compiled here.
      const bool reuse_build = probed_pso != nil && probed_dev == dev && probed_src == gpu_src;
      id<MTLLibrary> mtllib = reuse_build ? probed_lib : nil;
      if (!mtllib) {
        gpu_trace_at("compiling the source");
        mtllib = [dev newLibraryWithSource:ns options:copts error:&err];
        gpu_trace_at(mtllib ? "the library compiled" : "the library was refused");
      } else {
        gpu_trace_at("the library is the one the build check made");
      }
      if (!mtllib) {
        std::fprintf(stderr, "weft: Metal could not compile the source: %s\n",
                     [[err localizedDescription] UTF8String]);
        return 1;
      }
      id<MTLComputePipelineState> pso = nil;
      id<MTLComputePipelineState> tpso = nil;
      id<MTLRenderPipelineState> rpso = nil;
      if (o.render) {
        id<MTLFunction> vf = [mtllib newFunctionWithName:@"weft_vertex"];
        id<MTLFunction> ff = [mtllib newFunctionWithName:[NSString stringWithUTF8String:kern.name.c_str()]];
        if (!vf || !ff) {
          std::fprintf(stderr, "weft: the library has no %s\n", vf ? kern.name.c_str() : "weft_vertex");
          return 1;
        }
        // The tile stage runs as its own pass before the draw. Tile memory is a
        // buffer both sides share, so it needs a compute pipeline over the same
        // pixels and a dispatch ahead of the render pass — no tile pipeline.
        if (tile_run >= 0) {
          id<MTLFunction> tf = [mtllib newFunctionWithName:[NSString stringWithUTF8String:
                                                      tr.kernels[tile_run].name.c_str()]];
          if (!tf) {
            std::fprintf(stderr, "weft: the library has no %s\n",
                         tr.kernels[tile_run].name.c_str());
            return 1;
          }
          tpso = [dev newComputePipelineStateWithFunction:tf error:&err];
          if (!tpso) {
            std::fprintf(stderr, "weft: the tile stage failed: %s\n",
                         [[err localizedDescription] UTF8String]);
            return 1;
          }
        }
        MTLRenderPipelineDescriptor* rpd = [MTLRenderPipelineDescriptor new];
        rpd.vertexFunction = vf;
        rpd.fragmentFunction = ff;
        rpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
        rpso = [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
        if (!rpso) {
          std::fprintf(stderr, "weft: render pipeline failed: %s\n",
                       [[err localizedDescription] UTF8String]);
          return 1;
        }
      } else if (reuse_build) {
        pso = probed_pso;
        gpu_trace_at("the pipeline is the one the build check made");
      } else {
        id<MTLFunction> fn = [mtllib newFunctionWithName:[NSString stringWithUTF8String:kern.name.c_str()]];
        gpu_trace_at("creating the pipeline");
        pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        gpu_trace_at(pso ? "the pipeline was created" : "the pipeline was refused");
        if (!pso) {
          std::fprintf(stderr, "weft: pipeline failed: %s\n", [[err localizedDescription] UTF8String]);
          return 1;
        }
      }

      // The fragment's own buffers. The colour target is an attachment, not a
      // buffer, so it is created as a texture below.
      std::vector<id<MTLBuffer>> mbuf(nbind);
      for (size_t b = 0; b < nbind; b++) {
        mbuf[b] = [dev newBufferWithLength:host[b].size() * sizeof(float)
                                   options:MTLResourceStorageModeShared];
      }
      // An emitted PTX kernel binds its buffers the way any Metal kernel does —
      // one resource per index, because a GPU address is not something the host
      // can name (tools/ptrprobe). What the C++ ABI passes as a `void*` array is
      // therefore a set of bound resources here, and the scalar parameters are
      // one more buffer at the index after the last of them.
      id<MTLBuffer> par_buf = nil;
      int par_n = 1;
      if (o.ptx || spv) {
        for (const auto& a : o.args) par_n = std::max(par_n, a.first + 1);
        par_buf = [dev newBufferWithLength:(size_t)par_n * sizeof(uint64_t)
                                   options:MTLResourceStorageModeShared];
      }
      // A warp-backend kernel needs the launch it was given. The grid it could
      // derive from `threads_per_grid` is the warp count with the group padding
      // folded in, and the group size is not derivable at all, so both travel as
      // one extra binding after the scalar parameters.
      id<MTLBuffer> geom_buf = nil;
      if (kern.gpu_warp)
        geom_buf = [dev newBufferWithLength:4 * sizeof(uint64_t)
                                   options:MTLResourceStorageModeShared];
      // The words a 64-bit atomic locks on. The shape is `kLockWords` in
      // src/weft_warp_msl.hpp — one word per slot, hashed from the address — so
      // the host only has to make the table exist and hand it over zeroed.
      id<MTLBuffer> lock_buf = nil;
      if (kern.wide)
        lock_buf = [dev newBufferWithLength:1024 * sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];
      id<MTLTexture> target = nil;
      if (o.render) {
        MTLTextureDescriptor* td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                         width:o.rw
                                        height:o.rh
                                     mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        target = [dev newTextureWithDescriptor:td];
      }
      std::vector<id<MTLTexture>> mtex(o.textures.size());
      for (size_t t = 0; t < o.textures.size(); t++) {
        const Options::TexBind& tb = o.textures[t];
        // A volume is its own descriptor: `texture3d` is `MTLTextureType3D`, whose
        // planes are stored one after another — the order the host data was filled
        // in, and the order the CPU runtime indexes.
        MTLTextureDescriptor* td;
        if (tb.d > 0) {
          // This SDK has no `texture3DDescriptor…` class method, so the descriptor
          // is built field by field — which is what the convenience constructors
          // do anyway.
          td = [[MTLTextureDescriptor alloc] init];
          td.pixelFormat = MTLPixelFormatR32Float;
          td.textureType = MTLTextureType3D;
          td.width = tb.w;
          td.height = tb.h;
          td.depth = tb.d;
          td.mipmapLevelCount = 1;
        } else {
          td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                  width:tb.w
                                                                 height:tb.h
                                                              mipmapped:NO];
        }
        // An array binding is the texture type the kernel declares:
        // `texture2d_array<float>` is an MTLTextureType2DArray of L slices, and
        // its slices are stored one after another, which is the order the host
        // data was filled in. A cube is six faces of the same descriptor, in the
        // same order — the two types differ, the layout does not.
        if (tb.one_d) {
          // One row, and the GPU type that says so: `texture1d` is
          // `MTLTextureType1D`, which has a width and no height at all.
          td.textureType = MTLTextureType1D;
        } else if (tb.cube) {
          td.textureType = MTLTextureTypeCube;
        } else if (tb.layers > 1) {
          td.textureType = MTLTextureType2DArray;
          td.arrayLength = tb.layers;
        }
        // Writable, because a kernel may hold a storage image: `access::write`
        // needs the GPU's permission to write the resource back.
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        mtex[t] = [dev newTextureWithDescriptor:td];
        // The slice form uploads one slice per call — its region is a single
        // slice, not the whole array — so the slices are copied in one at a time
        // from the same flat buffer the CPU runtime indexes.
        if (tb.d > 0) {
          // A volume goes up through a blit: this SDK's `replaceRegion` has no 3D
          // form, and the blit's source is the same flat buffer the CPU runtime
          // indexes, one plane after another.
          id<MTLBuffer> src = [dev newBufferWithBytes:texdata[t].data()
                                               length:texdata[t].size() * sizeof(float)
                                              options:MTLResourceStorageModeShared];
          id<MTLCommandQueue> q = [dev newCommandQueue];
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
          [bl copyFromBuffer:src
                  sourceOffset:0
              sourceBytesPerRow:(NSUInteger)tb.w * sizeof(float)
            sourceBytesPerImage:(NSUInteger)tb.w * tb.h * sizeof(float)
                    sourceSize:MTLSizeMake(tb.w, tb.h, tb.d)
                     toTexture:mtex[t]
              destinationSlice:0
              destinationLevel:0
             destinationOrigin:MTLOriginMake(0, 0, 0)];
          [bl endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
        } else if (tb.layers > 1) {
          const size_t slice = (size_t)tb.w * tb.h * sizeof(float);
          for (uint32_t l = 0; l < tb.layers; l++)
            [mtex[t] replaceRegion:MTLRegionMake3D(0, 0, 0, tb.w, tb.h, 1)
                       mipmapLevel:0
                             slice:l
                         withBytes:texdata[t].data() + (size_t)l * tb.w * tb.h
                       bytesPerRow:(NSUInteger)tb.w * sizeof(float)
                      bytesPerImage:slice];
        }
        else
          [mtex[t] replaceRegion:MTLRegionMake2D(0, 0, tb.w, tb.h)
                     mipmapLevel:0
                       withBytes:texdata[t].data()
                     bytesPerRow:(NSUInteger)tb.w * sizeof(float)];
      }

      // The sampler state the CPU runtime models is the GPU's descriptor, so the
      // same two choices go to the encoder: the filter, and what a coordinate
      // outside the image means. A sampler the command line did not name gets the
      // CPU runtime's default — nearest, clamped — and binding it explicitly is
      // what makes that a decision rather than whatever the driver does.
      auto make_sampler = [&](int filter, int addr) -> id<MTLSamplerState> {
        MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
        const MTLSamplerMinMagFilter f =
            filter == 0 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        sd.minFilter = f;
        sd.magFilter = f;
        const MTLSamplerAddressMode am =
            addr == 1 ? MTLSamplerAddressModeClampToZero
                      : addr == 2 ? MTLSamplerAddressModeRepeat
                                  : MTLSamplerAddressModeClampToEdge;
        sd.sAddressMode = am;
        sd.rAddressMode = am;
        return [dev newSamplerStateWithDescriptor:sd];
      };
      std::vector<id<MTLSamplerState>> msamp;
      for (size_t s = 0; s < o.samplers.size(); s++)
        msamp.push_back(make_sampler(o.samplers[s].first, o.samplers[s].second));

      id<MTLCommandQueue> q = [dev newCommandQueue];

      double total = 0.0;
      for (uint32_t it = 0; it < o.bench + 1; it++) {
        // A dispatch that never completes is asked once more, on a queue of its
        // own: a queue runs its command buffers in submission order, so a wedged
        // one would hold every later launch behind it. The buffers are reset from
        // the pristine copy at the top below, so the second attempt starts from the
        // same bytes the first one did and an accumulating kernel cannot count twice.
        bool reissued = false;
        int guard_ms = 120000;
      again:
        for (size_t b = 0; b < nbind; b++)
          std::memcpy([mbuf[b] contents], pristine[b].data(), host[b].size() * sizeof(float));
        if (o.ptx || spv) {
          uint64_t* pp = (uint64_t*)[par_buf contents];
          for (int i = 0; i < par_n; i++) pp[i] = 0;
          for (const auto& a : o.args) pp[a.first] = a.second;
        }
        if (geom_buf) {
          uint64_t* gg = (uint64_t*)[geom_buf contents];
          gg[0] = o.grid;     // lanes the program is written against
          // A kernel that holds a 64-bit atomic is dispatched one warp per
          // threadgroup (see below), so the group it is told about is one warp:
          // anything else and `ctaid * ntid + tid` stops being the global lane.
          gg[1] = kern.wide ? 32u : o.threads;  // lanes in one threadgroup
          gg[2] = lock_buf ? (uint64_t)(uintptr_t)[lock_buf contents] : 0;
        }
        // Every dispatch starts with every lock free, so a launch that ended
        // holding one cannot stall the next.
        if (lock_buf) std::memset([lock_buf contents], 0, 1024 * sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [q commandBuffer];
        // Tile memory, before any fragment reads it. Both are encoders in one
        // command buffer, so the compute pass finishes before the draw starts.
        if (tpso) {
          id<MTLComputeCommandEncoder> ten = [cb computeCommandEncoder];
          [ten setComputePipelineState:tpso];
          for (size_t b = 0; b < nbind; b++) [ten setBuffer:mbuf[b] offset:0 atIndex:b];
          for (size_t t = 0; t < mtex.size(); t++) [ten setTexture:mtex[t] atIndex:t];
          for (size_t sm = 0; sm < msamp.size(); sm++) [ten setSamplerState:msamp[sm] atIndex:sm];
          [ten dispatchThreads:MTLSizeMake(o.grid, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tile_threads, 1, 1)];
          [ten endEncoding];
        }
        if (o.render) {
          MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
          rpd.colorAttachments[0].texture = target;
          rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
          rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
          rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
          id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
          [enc setRenderPipelineState:rpso];
          for (size_t b = 0; b < nbind; b++) [enc setFragmentBuffer:mbuf[b] offset:0 atIndex:b];
          for (size_t t = 0; t < mtex.size(); t++) [enc setFragmentTexture:mtex[t] atIndex:t];
          [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [enc endEncoding];
        } else {
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:pso];
          for (size_t b = 0; b < nbind; b++) [enc setBuffer:mbuf[b] offset:0 atIndex:b];
          // The scalar parameters sit after the buffers the kernel declared.
          if (o.ptx || spv) [enc setBuffer:par_buf offset:0 atIndex:nbind];
          if (geom_buf) [enc setBuffer:geom_buf offset:0 atIndex:nbind + 1];
          // A per-lane kernel has no geometry to carry, so that same slot holds
          // the lock table a 64-bit atomic takes: one binding after the scalars
          // either way.
          else if (lock_buf) [enc setBuffer:lock_buf offset:0 atIndex:nbind + 1];
          // A warp kernel carries geometry at +1, so its lock table is bound one
          // past that. It used to be handed over as a raw address inside the
          // geometry buffer instead, which is memory the kernel never declared —
          // and a spinlock on it lost increments under load.
          if (kern.gpu_warp && lock_buf) [enc setBuffer:lock_buf offset:0 atIndex:nbind + 2];
          for (size_t t = 0; t < mtex.size(); t++) [enc setTexture:mtex[t] atIndex:t];
          for (size_t sm = 0; sm < msamp.size(); sm++) [enc setSamplerState:msamp[sm] atIndex:sm];
          // A binding the kernel writes is `access::read_write`, and Metal cannot
          // filter through that mode. The module declares a second, sampled view of
          // the same texture one past the last binding; binding the one handle to
          // both gives the filter the same storage the writes reach.
          for (size_t t = 0; t < mtex.size(); t++) {
            if (t >= kern.tex_rw.size() || !kern.tex_rw[t]) continue;
            const NSUInteger at = (NSUInteger)(kern.ntex + (int)t);
            [enc setTexture:mtex[t] atIndex:at];
            [enc setSamplerState:(t < msamp.size() ? msamp[t] : make_sampler(0, 0)) atIndex:at];
          }
          // A warp-backend kernel runs a whole 32-lane warp per thread, so the
          // launch is one thread per warp and the group is the same fraction of
          // the threads per group. The kernel multiplies both back to get the
          // lane counts the program is written against.
          if (kern.gpu_warp) {
            // A thread is one warp here, so the launch is counted in warps. The
            // group count is spelled out rather than left to `dispatchThreads`:
            // a group of one thread is the common case (one warp per
            // threadgroup is what a Triton tile is), and a derived count put
            // every warp in the first group, which reads as "one tile ran".
            {
              const uint32_t warps = (o.grid + 31u) / 32u;
              // A group is never wider than the warps the launch actually has:
              // with a grid smaller than one block, a group of four threads would
              // run three warps the program does not have, and the CPU twin —
              // which launches `ceil(live/32)` warps — runs only the first.
              uint32_t per = std::max(1u, std::min(o.threads / 32u, warps));
              // A kernel that relays between lanes has one slice of threadgroup
              // memory per warp, and the translator sized that array: a wider
              // group would put two warps in one slice, which is a wrong answer
              // rather than a slow one. The bound is the kernel's, not a guess.
              if (kern.slot_per) per = std::max(1u, std::min(per, kern.slot_per));
              // A wide kernel moves its atomics from one thread (see below), so its
              // group is one thread.
              if (kern.wide) per = 1u;
              // One warp per threadgroup was tried and is worse: at 128 warps it
              // stops completing where four warps per group answers in 22 ms, so a
              // group of one thread costs whatever the per-group `threadgroup`
              // memory costs and buys nothing. The shared arena is not a reason to
              // widen the dispatch either — it is zeroed before the body runs (see
              // the emitter), so a warp cannot read a peer's half-written slice.
              // A kernel that needs the 64-bit pair is walked by one thread — that
              // is what gives the pair a single writer — so it is dispatched once.
              const bool one_walker = kern.wide && spv;
              uint64_t* cg = (uint64_t*)[geom_buf contents];
              cg[0] = o.grid;                   // lanes the kernel is written against
              cg[1] = 32u;                      // lanes in one threadgroup, which is one warp
              cg[2] = one_walker ? warps : 0u;  // warps the walk covers
              const uint32_t groups = one_walker ? 1u : (warps + per - 1u) / per;
              const uint32_t threads = one_walker ? 1u : per;
              [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            }
          } else {
            [enc dispatchThreads:MTLSizeMake(o.grid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(o.threads, 1, 1)];
          }
          [enc endEncoding];
        }
        [cb commit];
        gpu_trace_at(reissued ? "the second launch is committed" : "the launch is committed");
        // A dispatch the driver never finishes would block here for ever: poll, and
        // drop the GPU column rather than hang the run.
        // 120 s, not 30: the first dispatch of a kernel on this host compiles its
        // pipeline lazily, and one launch of the wide 64-bit path was measured past
        // 30 s while the same launch warm took 66 ms.
        for (int spin = 0; spin < guard_ms; spin++) {
          const MTLCommandBufferStatus st = cb.status;
          if (st == MTLCommandBufferStatusCompleted || st == MTLCommandBufferStatusError)
            break;
          [NSThread sleepForTimeInterval:0.001];
        }
        // A command buffer that did not complete left its work partly done, and a
        // partly-done dispatch reads as a wrong answer from the kernel rather than
        // as a failed launch. Say which it was.
        if (cb.status != MTLCommandBufferStatusCompleted) {
          if (!reissued) {
            // The launch has been seen never to finish while the same dispatch in a
            // second process answers in milliseconds, so this is one more question,
            // not a longer wait: a fresh queue, a fresh command buffer, the same
            // bindings, and a shorter guard because a real hang hangs twice.
            gpu_trace_at("the dispatch did not complete; issuing it again on a new queue");
            reissued = true;
            guard_ms = 20000;
            q = [dev newCommandQueue];
            goto again;
          }
          gpu_trace_at("the dispatch did not complete a second time");
          // A dispatch the driver never finished is not an answer: its buffers still
          // hold whatever the last launch left there, and comparing those bytes
          // reported the kernel as wrong. This host's first dispatch of a kernel
          // compiles its pipeline lazily and can be far slower than the launch, so
          // say the launch is missing rather than that the kernel disagrees.
          gpu_refusal = std::string("the gpu did not complete the dispatch: ") +
                        (cb.error ? cb.error.localizedDescription.UTF8String
                                  : "no error reported");
          std::fprintf(stderr, "weft: %s\n", gpu_refusal.c_str());
          break;
        }

        gpu_trace_at("the dispatch completed");
        double d = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (it > 0) {
          total += d;
          if (gpu_best < 0 || d < gpu_best) gpu_best = d;
        }
      }
      gpu_avg = total / (double)o.bench;

      if (gpu_refusal.empty()) {
        for (size_t b = 0; b < nbind; b++) {
          gpu_out[b].resize(host[b].size());
          std::memcpy(gpu_out[b].data(), [mbuf[b] contents], host[b].size() * sizeof(float));
        }
        if (o.render) {
          gpu_out[colour_idx].assign(host[colour_idx].size(), 0.0f);
          [target getBytes:gpu_out[colour_idx].data()
               bytesPerRow:(NSUInteger)o.rw * 4 * sizeof(float)
                fromRegion:MTLRegionMake2D(0, 0, o.rw, o.rh)
               mipmapLevel:0];
        }
      }
      gpu_ran = gpu_refusal.empty();
    }
#else
    if (!o.cpu_only) {
      std::printf("%-9s %-34s %s\n", "gpu", "unavailable",
                  "this build has no gpu backend: Metal is Apple-only, so there is "
                  "nothing to diff against");
      std::printf("%-9s %-34s %s\n", "", "",
                  "the kernel still ran on the cpu: compare a second front end's dump "
                  "with --dump and cmp");
    }
#endif

    if (!gpu_refusal.empty() && !gpu_ran)
      std::printf("%-9s %-34s %s\n", "gpu", "no column for this kernel", gpu_refusal.c_str());

    // ---- compare ----
    double worst_abs = 0.0, worst_rel = 0.0;
    size_t checked = 0;
    for (size_t b = 0; b < nbuf && gpu_ran; b++) {
      for (size_t i = 0; i < gpu_out[b].size(); i++) {
        double a = gpu_out[b][i], c = cpu_out[b][i];
        double ad = std::fabs(a - c);
        double denom = std::fmax(std::fabs(a), 1e-30);
        if (ad > worst_abs) worst_abs = ad;
        if (ad / denom > worst_rel) worst_rel = ad / denom;
        checked++;
      }
    }

    for (const auto& d : o.gpu_dumps) {
      if (d.first < 0 || (size_t)d.first >= nbuf || !gpu_ran) {
        std::fprintf(stderr, "weft: --dump-gpu %d: no such buffer, or no gpu run\n", d.first);
        return 2;
      }
      if (!write_bin(d.second, gpu_out[d.first].data(), gpu_out[d.first].size() * sizeof(float))) {
        std::fprintf(stderr, "weft: cannot write %s\n", d.second.c_str());
        return 1;
      }
    }

    for (const auto& d : o.dumps) {
      if (d.first < 0 || (size_t)d.first >= nbuf) {
        std::fprintf(stderr, "weft: --dump %d: no such buffer\n", d.first);
        return 2;
      }
      if (!write_bin(d.second, cpu_out[d.first].data(), cpu_out[d.first].size() * sizeof(float))) {
        std::fprintf(stderr, "weft: cannot write %s\n", d.second.c_str());
        return 1;
      }
    }

    // ---- the image ----
    // A render pass's colour target is the frame; --out is how it leaves the
    // process. The gpu form exists so the two files can be compared with `cmp`,
    // which is the same claim as `compare` in a form that opens in a viewer.
    if (!o.out.empty()) {
      if (!o.render) {
        std::fprintf(stderr, "weft: --out needs a render pass (--render WxH)\n");
        return 2;
      }
      if (!write_png(o.out, cpu_out[colour_idx], o.rw, o.rh)) {
        std::fprintf(stderr, "weft: cannot write %s\n", o.out.c_str());
        return 1;
      }
      std::printf("%-9s %-34s %ux%u  (cpu)\n", "image", o.out.c_str(), o.rw, o.rh);
    }
    if (!o.out_gpu.empty()) {
      if (!o.render || !gpu_ran) {
        std::fprintf(stderr, "weft: --out-gpu needs a render pass and a gpu run\n");
        return 2;
      }
      if (!write_png(o.out_gpu, gpu_out[colour_idx], o.rw, o.rh)) {
        std::fprintf(stderr, "weft: cannot write %s\n", o.out_gpu.c_str());
        return 1;
      }
      std::printf("%-9s %-34s %ux%u  (gpu)\n", "image", o.out_gpu.c_str(), o.rw, o.rh);
    }

    double cpu_avg = sum_cpu / (double)o.bench;
    char cpu_label[64];
    std::snprintf(cpu_label, sizeof cpu_label, "%s, %u thread%s",
                  o.cpu_only ? "cpu only" : "cpu", threads, threads == 1 ? "" : "s");
    std::printf("%-9s %-34s %8.3f ms  (best %.3f)\n", "cpu", cpu_label, cpu_avg, best_cpu);
    if (gpu_ran) {
      std::printf("%-9s %-34s %8.3f ms  (best %.3f)\n", "gpu", gpu_name.c_str(), gpu_avg, gpu_best);
      const double ratio = cpu_avg / std::fmax(gpu_avg, 1e-9);
      std::printf("%-9s cpu is %.2fx %s than the gpu\n", "", ratio >= 1.0 ? ratio : 1.0 / ratio,
                  ratio >= 1.0 ? "slower" : "faster");
      std::printf("%-9s %s values  ·  max |diff| %.3g  ·  max rel %.3g%s\n", "compare",
                  fmt_uint(checked).c_str(), worst_abs, worst_rel,
                  worst_rel <= 1e-4 ? "  — agree" : "  — DISAGREE");
    }
    std::printf("\n");
    return 0;
#if defined(__APPLE__)
  }
#endif
}
