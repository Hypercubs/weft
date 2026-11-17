#if defined(__APPLE__) && !defined(WEFT_NO_METAL)
#define WEFT_METAL 1
#else
#define WEFT_METAL 0
#endif

#if WEFT_METAL
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#include <signal.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <direct.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#endif

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
#include "weft_f64.hpp"
#include "translate_ptx.hpp"
#include "cuda_gpu.hpp"
#include "translate_spirv.hpp"
#include <weft/weft.h>
#if !defined(_WIN32)
#include "vmetal.hpp"
#endif

namespace {

struct GuardTag {
  const char* base = nullptr;
  size_t bytes = 0;
  int index = -1;
  bool caller = true;
};

GuardTag g_guards[64];
int g_nguards = 0;
volatile sig_atomic_t g_guard_reported = 0;
uint64_t g_guard_grid = 0;
uint64_t g_guard_threads = 0;

#if defined(_WIN32)
// No guard page here: an overrun is the host's to report.
void* guard_alloc(size_t bytes) {
  void* p = std::malloc(bytes ? bytes : 1);
  if (!p) {
    std::fprintf(stderr, "weft: out of memory for %zu bytes of buffer\n", bytes);
    std::abort();
  }
  return p;
}

void guard_free(void* p, size_t) { std::free(p); }
// No guard page to walk on this platform, so no pattern to check either.
bool guard_past(int, size_t*) { return false; }

void guard_install() {}

#else

constexpr unsigned char kGuardByte = 0xA5;

size_t g_page_bytes = 0;
size_t guard_page() {
  if (!g_page_bytes) g_page_bytes = (size_t)getpagesize();
  return g_page_bytes;
}

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
  std::memset(p, kGuardByte, pages * page);
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
    char msg[512];
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
    if (g_guards[i].caller) {
      n += guard_put(msg + n, ". --buffers ");
      n += guard_num(msg + n, (uint64_t)g_guards[i].index);
      n += guard_put(msg + n, " is too small for this launch.\n");
    } else {
      n += guard_put(msg + n, ". That block is sized by the launcher and its guard page sits "
                             "at its end, so there is no room to grow it: this run is refused "
                             "rather than answered with bytes the GPU leaves undefined.\n");
    }
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

bool guard_past(int index, size_t* at) {
  if (index < 0 || index >= g_nguards) return false;
  const char* base = g_guards[index].base;
  if (!base) return false;
  const size_t page = guard_page();
  const size_t len = g_guards[index].bytes;
  const size_t usable = ((len + page - 1) / page) * page;
  size_t past = 0;
  for (size_t i = len; i < usable; i++)
    if ((unsigned char)base[i] != kGuardByte) past = i + 1;
  if (!past) return false;
  if (at) *at = past;
  return true;
}

#endif

void guard_tag(const void* base, size_t bytes, int index, bool caller) {
  if (!base || index < 0 || index >= 64) return;
  guard_install();
  g_guards[index].base = (const char*)base;
  g_guards[index].bytes = bytes;
  g_guards[index].index = index;
  g_guards[index].caller = caller;
  if (index >= g_nguards) g_nguards = index + 1;
}

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

#if defined(_WIN32)
void* lib_open(const char* path) { return (void*)LoadLibraryA(path); }
void* lib_sym(void* h, const char* name) { return (void*)GetProcAddress((HMODULE)h, name); }
const char* lib_error() { return "the module did not load"; }
void make_dir(const char* path) { _mkdir(path); }
bool file_readable(const char* path) { return _access(path, 4) == 0; }
long proc_id() { return (long)_getpid(); }
#else
// A float as IEEE half bits, written out here because gcc's C++ front end did
// not accept _Float16 until 13: the bytes a half texture carries do not change.
static uint16_t half_bits_of(float v) {
  uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  const uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
  const int32_t e = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
  const uint32_t m = u & 0x7fffffu;
  if (e >= 0x1f) {
    const bool nan = (u & 0x7f800000u) == 0x7f800000u && m != 0;
    return (uint16_t)(sign | 0x7c00u | (nan ? 0x200u : 0u));
  }
  if (e <= 0) {
    if (e < -10) return sign;
    const uint32_t full = m | 0x800000u;
    const int shift = 14 - e;
    uint32_t h = full >> shift;
    const uint32_t rest = full & ((1u << shift) - 1u);
    const uint32_t mid = 1u << (shift - 1);
    if (rest > mid || (rest == mid && (h & 1u) != 0u)) h += 1u;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (m >> 13);
  const uint32_t rest = m & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1u) != 0u)) h += 1u;
  return (uint16_t)(sign | h);
}

static void print_devices() {
#if WEFT_METAL
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  std::printf("%-14s %s\n", "metal", d ? [[d name] UTF8String] : "none");
#endif
  std::printf("%-14s %s\n", "virtual-metal", vmetal::device_name());
}

void* lib_open(const char* path) { return dlopen(path, RTLD_NOW); }
void* lib_sym(void* h, const char* name) { return dlsym(h, name); }
const char* lib_error() { return dlerror(); }
void make_dir(const char* path) { ::mkdir(path, 0755); }
bool file_readable(const char* path) { return ::access(path, R_OK) == 0; }
long proc_id() { return (long)::getpid(); }
#endif

const char* kUsage =
    "weft: run GPU compute kernels on CPU cores\n"
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
    "  --virtual-gpu     run the gpu column on the virtual Metal device\n"
    "  --list-devices    show the gpu devices this host has\n"
    "  --exact           compare the two columns word for word, NaN and zero signs too\n"
    "                    the cpu path times 1/2/4/8 threads on this kernel and keeps\n"
    "                    the fastest, because the right count is a property of the\n"
    "                    kernel: atomics want one, elementwise wants two to four. The\n"
    "                    count it kept is printed on the cpu line.\n"
    "  --emit            print the generated C++ and exit\n"
    "  --emit-gpu        print the Metal source the GPU is handed, and exit\n"
    "  --emit-ir         print the module contract as text: bindings, scalars,\n"
    "                    textures and launch geometry per kernel, and exit\n"
    "  --emit-ir-json    the same contract as one JSON object\n"
    "  --load B=path     fill buffer B from a file of float32 values\n"
    "  --dump B=path     write the CPU's buffer B out as floats\n"
    "  --dump-gpu B=path write the GPU's buffer B out as floats\n"
    "\n"
    "The same kernel source is compiled twice: once by Metal for the GPU, once\n"
    "translated to C++ for the CPU. Both outputs are compared and both are timed.\n";

std::string exe_dir() {
#if defined(_WIN32)
  char buf[4096];
  const DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof buf);
  if (n == 0 || n >= sizeof buf) return std::string();
  std::string p(buf, (size_t)n);
#elif defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string p(size ? size : 1, '\0');
  if (_NSGetExecutablePath(&p[0], &size) != 0) return std::string();
  p.resize(std::strlen(p.c_str()));
#else
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return std::string();
  std::string p(buf, (size_t)n);
#endif
  const size_t slash = p.find_last_of("/\\");
  return slash == std::string::npos ? std::string() : p.substr(0, slash);
}

// Where runtime/ and include/ live.
std::string runtime_root() {
  if (const char* env = std::getenv("WEFT_ROOT")) return env;
  const std::string dir = exe_dir();
  if (!dir.empty() && file_readable((dir + "/runtime/weft_runtime.hpp").c_str())) return dir;
  return WEFT_ROOT;
}

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
void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
  _putenv_s(k, v.c_str());
#else
  setenv(k, v.c_str(), 1);
#endif
}

#if WEFT_METAL
const char* kFullScreenVertex = R"(
struct WeftVertexOut { float4 pos [[position]]; };
vertex WeftVertexOut weft_vertex(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    WeftVertexOut o;
    o.pos = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    return o;
}
)";

std::string PointVertexSource(uint32_t w, uint32_t h) {
  const std::string sw = std::to_string(w), sh = std::to_string(h);
  return "struct WeftVertexOut { float4 pos [[position]]; float size [[point_size]]; };\n"
         "vertex WeftVertexOut weft_points_vertex(uint vid [[vertex_id]]) {\n"
         "    float2 p = float2(float(vid % " + sw + "u), float(vid / " + sw + "u));\n"
         "    WeftVertexOut o;\n"
         "    o.pos = float4((p.x + 0.5f) / " + sw + ".0f * 2.0f - 1.0f,\n"
         "                   1.0f - (p.y + 0.5f) / " + sh + ".0f * 2.0f, 0.0f, 1.0f);\n"
         "    o.size = 1.0f;\n"
         "    return o;\n"
         "}\n";
}

void LowerStageIn(std::string& src, uint32_t w, uint32_t h) {
  const size_t si = src.find("[[stage_in]]");
  if (si == std::string::npos) return;
  std::string head = src.substr(0, si);
  while (!head.empty() && std::isspace((unsigned char)head.back())) head.pop_back();
  size_t cut = head.find_last_of(" \t\r\n");
  if (cut == std::string::npos) return;
  const std::string pname = head.substr(cut + 1);
  head = head.substr(0, cut);
  while (!head.empty() && std::isspace((unsigned char)head.back())) head.pop_back();
  cut = head.find_last_of(" \t\r\n");
  std::string type = head.substr(cut == std::string::npos ? 0 : cut + 1);
  const size_t lp = type.find_last_of("(,");
  if (lp != std::string::npos) type = type.substr(lp + 1);
  if (type.empty() || pname.empty()) return;

  const size_t def = src.find("struct " + type);
  if (def == std::string::npos) return;
  const size_t open = src.find('{', def);
  if (open == std::string::npos || open > si) return;
  size_t depth = 0, close = std::string::npos;
  for (size_t i = open; i < src.size(); i++) {
    if (src[i] == '{') depth++;
    else if (src[i] == '}' && --depth == 0) { close = i; break; }
  }
  if (close == std::string::npos) return;
  std::string pro = "    " + type + " " + pname + ";\n";
  {
    const std::string fields = src.substr(open + 1, close - open - 1);
    size_t at = 0;
    while (true) {
      const size_t semi = fields.find(';', at);
      if (semi == std::string::npos) break;
      std::string f = fields.substr(at, semi - at);
      at = semi + 1;
      bool is_position = false;
      const size_t lb = f.find("[["), rb = f.find("]]");
      if (lb != std::string::npos && rb > lb) {
        if (f.substr(lb + 2, rb - lb - 2).find("position") != std::string::npos) is_position = true;
        f = f.substr(0, lb) + f.substr(rb + 2);
      }
      std::string t, n, word;
      for (char c : f) {
        if (std::isspace((unsigned char)c)) {
          if (!word.empty()) { if (t.empty()) t = word; n = word; word.clear(); }
        } else {
          word.push_back(c);
        }
      }
      if (!word.empty()) { if (t.empty()) t = word; n = word; }
      if (n.empty()) continue;
      if (is_position) {
        pro += "    " + pname + "." + n + " = weft_px;\n";
      } else if (t == "float2") {
        pro += "    " + pname + "." + n + " = float2(weft_px.x / " + std::to_string(w) +
               ".0f, 1.0f - weft_px.y / " + std::to_string(h) + ".0f);\n";
      } else {
        return;
      }
    }
  }

  const size_t frag = src.rfind("fragment", si);
  if (frag == std::string::npos) return;
  const size_t paren = src.find('(', frag);
  if (paren == std::string::npos || paren > si) return;
  depth = 0;
  size_t end_paren = std::string::npos;
  for (size_t i = paren; i < src.size(); i++) {
    if (src[i] == '(') depth++;
    else if (src[i] == ')' && --depth == 0) { end_paren = i; break; }
  }
  if (end_paren == std::string::npos || end_paren < si) return;
  // A fragment that already names its position cannot name it twice.
  if (src.substr(paren, end_paren - paren).find("[[position]]") != std::string::npos) return;

  size_t pstart = paren + 1;
  for (size_t i = paren + 1; i < si; i++)
    if (src[i] == ',') pstart = i + 1;
  while (pstart < si && std::isspace((unsigned char)src[pstart])) pstart++;
  size_t pend = si + 12;
  while (pend < end_paren && std::isspace((unsigned char)src[pend])) pend++;
  if (pend < end_paren && src[pend] == ',') {
    pend++;
    while (pend < end_paren && std::isspace((unsigned char)src[pend])) pend++;
  } else {
    size_t s = pstart;
    while (s > paren + 1 && std::isspace((unsigned char)src[s - 1])) s--;
    if (s > paren + 1 && src[s - 1] == ',') pstart = s - 1;
  }
  std::string before = src.substr(paren + 1, pstart - paren - 1);
  std::string after = src.substr(pend, end_paren - pend);
  while (!before.empty() && std::isspace((unsigned char)before.back())) before.pop_back();
  while (!after.empty() && std::isspace((unsigned char)after.front())) after.erase(0, 1);
  std::string list;
  if (!before.empty()) {
    list = before;
    if (list.back() != ',') list += ',';
    list += ' ';
  }
  list += "float4 weft_px [[position]]";
  if (!after.empty()) list += ", " + after;

  src = src.substr(0, paren + 1) + list + src.substr(end_paren);
  // The body's brace, now that the parameter list has moved.
  const size_t body = src.find('{', paren + 1 + list.size());
  if (body == std::string::npos) return;
  src.insert(body + 1, "\n" + pro);
}

std::string FullScreenVertexFor(const std::string& src) {
  const size_t si = src.find("[[stage_in]]");
  if (si == std::string::npos) return std::string();
  // `FragmentIn in [[stage_in]]`: the type is the word before the name.
  std::string head = src.substr(0, si);
  while (!head.empty() && std::isspace((unsigned char)head.back())) head.pop_back();
  size_t cut = head.find_last_of(" \t\r\n");
  if (cut == std::string::npos) return std::string();
  head = head.substr(0, cut);
  while (!head.empty() && std::isspace((unsigned char)head.back())) head.pop_back();
  cut = head.find_last_of(" \t\r\n");
  std::string type = head.substr(cut == std::string::npos ? 0 : cut + 1);
  const size_t lp = type.find_last_of("(,");
  if (lp != std::string::npos) type = type.substr(lp + 1);
  if (type.empty()) return std::string();

  const size_t def = src.find("struct " + type);
  if (def == std::string::npos) return std::string();
  const size_t open = src.find('{', def);
  if (open == std::string::npos) return std::string();
  size_t depth = 0, close = std::string::npos;
  for (size_t i = open; i < src.size(); i++) {
    if (src[i] == '{') depth++;
    else if (src[i] == '}' && --depth == 0) { close = i; break; }
  }
  if (close == std::string::npos) return std::string();

  std::string out = "vertex " + type + " weft_vertex(uint vid [[vertex_id]]) {\n"
                    "    float2 p = float2(float((vid << 1) & 2), float(vid & 2));\n"
                    "    " + type + " o;\n";
  const std::string fields = src.substr(open + 1, close - open - 1);
  size_t at = 0;
  while (true) {
    const size_t semi = fields.find(';', at);
    if (semi == std::string::npos) break;
    std::string f = fields.substr(at, semi - at);
    at = semi + 1;
    // The attribute says what the field is; the words say its type and name.
    bool is_position = false;
    const size_t lb = f.find("[["), rb = f.find("]]");
    if (lb != std::string::npos && rb > lb) {
      if (f.substr(lb + 2, rb - lb - 2).find("position") != std::string::npos) is_position = true;
      f = f.substr(0, lb) + f.substr(rb + 2);
    }
    std::string t, n, w;
    for (char c : f) {
      if (std::isspace((unsigned char)c)) {
        if (!w.empty()) { if (t.empty()) t = w; n = w; w.clear(); }
      } else {
        w.push_back(c);
      }
    }
    if (!w.empty()) { if (t.empty()) t = w; n = w; }
    if (n.empty()) continue;
    if (is_position) {
      out += "    o." + n + " = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);\n";
    } else if (t == "float2") {
      out += "    o." + n + " = p;\n";
    } else {
      return std::string();
    }
  }
  return out + "    return o;\n}\n";
}
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
  bool bufs_stated = false;  // --buffers was given, so its counts are the caller's
  std::vector<std::pair<int, uint32_t>> widths;  // --buffer-bytes overrides
  struct TexBind { uint32_t w, h, layers, d; bool cube; bool one_d; };
  std::vector<TexBind> textures;  // bound at index 0, 1, ...
  uint32_t texel_bytes = 4;
  std::vector<std::pair<int, int>> samplers;            // filter, address mode
  std::vector<std::pair<int, uint64_t>> args;  // scalar kernel parameters
  std::vector<std::pair<int, std::string>> dumps, gpu_dumps;
  std::vector<std::pair<int, std::string>> loads;  // raw float32 in
  bool cpu_only = false, emit = false, emit_gpu = false, emit_ir = false,
       emit_ir_json = false, ptx = false;
  bool virtual_gpu = false;  // --virtual-gpu: the gpu column on the virtual device
  bool exact = false;
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

// A renderer that cannot save its frame has no output.
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
  if (argc == 2 && std::string(argv[1]) == "--list-devices") {
    print_devices();
    return 0;
  }
#if WEFT_METAL
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
      else if (a == "--virtual-gpu") o.virtual_gpu = true;
      else if (a == "--list-devices") {
        print_devices();
        return 0;
      }
      else if (a == "--exact") o.exact = true;
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
      else if (a == "--emit-ir") o.emit_ir = true;
      else if (a == "--emit-ir-json") o.emit_ir_json = true;
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
          // A cube array is six faces a layer: `WxHc*N` is N cubes.
          const uint32_t tl =
              cube ? 6u * (star == std::string::npos
                               ? 1u
                               : (uint32_t)std::strtoul(rest.substr(star + 1).c_str(), nullptr, 10))
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
      else if (a == "--texel-bytes") {
        const std::string v = next("--texel-bytes");
        const unsigned long n = std::strtoul(v.c_str(), nullptr, 10);
        if (n != 2 && n != 4) {
          std::fprintf(stderr, "weft: --texel-bytes takes 2 (a half image) or 4, got '%s'\n",
                       v.c_str());
          return 2;
        }
        o.texel_bytes = (uint32_t)n;
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
        o.bufs_stated = true;
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
    if ((!o.out.empty() || !o.out_gpu.empty()) && !o.render) {
      std::fprintf(stderr, "weft: --out needs a render pass (--render WxH)\n");
      return 2;
    }
    const bool spv = !o.render && o.file.size() > 4 && o.file.compare(o.file.size() - 4, 4, ".spv") == 0;
    o.front = spv ? "spir-v" : (o.ptx ? "ptx" : "metal");

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

    std::string gpu_refusal;
    if ((o.ptx || spv) && !o.cpu_only) {
      TranslateResult g = o.ptx ? translate_ptx_gpu(src, o.widths)
                                : translate_spirv_gpu(src, o.widths);
      if (g.ok) {
        tr.gpu_src = g.gpu_src;
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

    if (o.emit_ir_json) { std::fputs(emit_ir_json(tr.kernels).c_str(), stdout); return 0; }
    if (o.emit_ir) { std::fputs(emit_ir(tr.kernels).c_str(), stdout); return 0; }
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

    if (o.texel_bytes == 2 && !o.ptx && !spv &&
        (tr.gpu_src.find("texture2d<float>") != std::string::npos ||
         tr.gpu_src.find("texture2d_array<float>") != std::string::npos ||
         tr.gpu_src.find("texture3d<float>") != std::string::npos ||
         tr.gpu_src.find("texturecube<float>") != std::string::npos ||
         tr.gpu_src.find("texture1d<float>") != std::string::npos)) {
      std::fprintf(stderr,
                   "weft: --texel-bytes 2 binds a half image, but %s reads a float "
                   "texture; a float texture needs four-byte texels\n",
                   kern.name.c_str());
      return 2;
    }
    if (!o.threads_set && kern.tg_hint) o.threads = kern.tg_hint;
    if (!o.threads_set && !kern.tg_hint && kern.def_threads) o.threads = kern.def_threads;
    if (kern.fixed_threads && o.threads != kern.fixed_threads) {
      std::printf("threads   %u given, this kernel declares a %u-lane group: using %u\n",
                  o.threads, kern.fixed_threads, kern.fixed_threads);
      o.threads = kern.fixed_threads;
    }
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
    if (!o.threads_set && kern.mesh) o.threads = 32;
    if (kern.warp_only) {
      const uint32_t block = kern.tg_hint ? kern.tg_hint : 32u;
      if (o.threads != block) {
        std::printf("threads   %u given, this kernel is a %u-lane warp block: using %u\n",
                    o.threads, block, block);
        o.threads = block;
      }
    }
    if (kern.tg_hint && o.grid < kern.tg_hint) {
      std::fprintf(stderr,
                   "weft: kernel '%s' is a %u-lane block and --grid gave %u: give it at "
                   "least one full block (--grid %u or more)\n",
                   kern.name.c_str(), kern.tg_hint, o.grid, kern.tg_hint);
      return 2;
    }
    int tile_kidx = -1;
    if (o.render)
      for (size_t i = 0; i < tr.kernels.size(); i++)
        if (tr.kernels[i].tile) tile_kidx = (int)i;
    int task_kidx = -1;
    for (size_t i = 0; i < tr.kernels.size(); i++)
      if (tr.kernels[i].task) task_kidx = (int)i;
    if (o.bufs.empty()) o.bufs.assign((size_t)kern.nbufs, (size_t)o.grid);

    const std::string root = runtime_root();
    std::string build = std::string(root) + "/build";
    make_dir(build.c_str());
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
      // The module's ABI: a cached dylib read through a header it was not
      // compiled against is a wrong answer, not a reload.
      hash = fnv(read_file(std::string(root) + "/include/weft/weft.h", ok), hash);
    }
    // The generated module is compiled by this host's C++ compiler.
    const char* cxx_env = std::getenv("WEFT_CXX");
#if !defined(_WIN32)
    auto compiler_ok = [&](const std::string& c) {
      const std::string probe = build + "/compiler_probe." + std::to_string(proc_id()) + ".cpp";
      if (!write_file(probe, "_Float16 h = (_Float16)1.5f;\n"
                             "struct P { int x, y; };\n"
                             "int main() { P p(1, 2); return (int)h + p.x; }\n"))
        return false;
      const bool ok = std::system((c + " -std=c++20 -fsyntax-only \"" + probe +
                                   "\" >/dev/null 2>&1").c_str()) == 0;
      std::remove(probe.c_str());
      return ok;
    };
    auto on_path = [](const char* c) {
      return std::system((std::string("command -v ") + c + " >/dev/null 2>&1").c_str()) == 0;
    };
#endif
#if defined(_WIN32)
    // cmd.exe runs the probe, so ask the compiler for its own version.
    const std::string cxx =
        cxx_env ? std::string(cxx_env)
                : (std::system("clang++ --version >NUL 2>&1") == 0 ? "clang++" : "c++");
    const char* kSharedFlag = "-shared -Wl,--export-all-symbols";
    const char* kLibExt = ".dll";
#else
    std::string cxx = cxx_env ? std::string(cxx_env) : "";
    if (cxx.empty()) {
      cxx = on_path("clang++") ? "clang++" : "c++";
      if (!compiler_ok(cxx)) {
        for (const char* c : {"c++", "clang++", "g++-13", "g++-12"}) {
          if (cxx == c || !on_path(c)) continue;
          if (compiler_ok(c)) { cxx = c; break; }
        }
      }
    }
#if WEFT_METAL
    const char* kSharedFlag = "-dynamiclib";
    const char* kLibExt = ".dylib";
#else
    const char* kSharedFlag = "-shared";
    const char* kLibExt = ".so";
#endif
#endif
    hash = fnv(cxx + " -std=c++20 -O2 -fno-strict-aliasing -fsigned-char " + kSharedFlag + " -fPIC " + root, hash);
    char hexkey[32];
    std::snprintf(hexkey, sizeof hexkey, "%016llx", (unsigned long long)hash);
    const std::string cache = build + "/cache";
    make_dir(cache.c_str());
    // One source file per process: two runs must not compile each other's module.
    const std::string gen = build + "/generated." + std::to_string(proc_id()) + ".cpp";
    const std::string lib = cache + "/" + hexkey + kLibExt;
    const bool cached = file_readable(lib.c_str());

    double t0 = now_ms();
    if (!cached) {
      if (!write_file(gen, tr.code)) {
        std::fprintf(stderr, "weft: cannot write %s\n", gen.c_str());
        return 1;
      }
      const std::string tmp = lib + ".tmp." + std::to_string(proc_id());
      const std::string base = cxx + " -std=c++20 -O2 -fno-strict-aliasing -fsigned-char " + kSharedFlag + " -fPIC -I\"" + root +
                               "/runtime\" -I\"" + root + "/include\" \"" + root +
                               "/runtime/weft_runtime.cpp\" \"" + gen + "\" -o \"" + tmp + "\"";
      const std::string logpath = build + "/clang." + std::to_string(proc_id()) + ".log";
      int rc = std::system((base + " 2> \"" + logpath + "\"").c_str());
      std::string log;
      {
        bool lok = false;
        log = read_file(logpath, lok);
      }
      if (rc != 0 && log.find("co_await") != std::string::npos) {
        rc = std::system((base + " -fcoroutines 2> \"" + logpath + "\"").c_str());
        bool lok = false;
        log = read_file(logpath, lok);
      }
      std::remove(gen.c_str());
      std::remove(logpath.c_str());
      if (rc != 0) {
        std::remove(tmp.c_str());
        std::fprintf(stderr, "weft: compiling the translated kernel failed (%s)\n%s\n",
                     cxx.c_str(), log.c_str());
        return 1;
      }
      std::rename(tmp.c_str(), lib.c_str());
    }
    const double compile_ms = cached ? 0.0 : now_ms() - t0;

    void* handle = lib_open(lib.c_str());
    if (!handle) { std::fprintf(stderr, "weft: %s\n", lib_error()); return 1; }
    auto run_cpu = (void (*)(int, void**, uint32_t, uint32_t))lib_sym(handle, "weft_run");
    auto tile_reset = (void (*)())lib_sym(handle, "weft_tile_reset");
    auto tile_violations = (uint64_t (*)())lib_sym(handle, "weft_tile_violations");
    auto kernel_count = (int (*)())lib_sym(handle, "weft_kernel_count");
    auto kernel_at = (const weft_kernel_info* (*)(int))lib_sym(handle, "weft_kernel_at");
    auto set_param = (void (*)(int, uint64_t))lib_sym(handle, "weft_set_param");
    auto fault_word = (uint32_t (*)())lib_sym(handle, "weft_fault_word");
    auto fault_lane = (uint32_t (*)())lib_sym(handle, "weft_fault_lane");
    auto fault_reset = (void (*)())lib_sym(handle, "weft_fault_reset");
    if (set_param)
      for (const auto& a : o.args) set_param(a.first, a.second);

    std::vector<std::vector<float>> texdata(o.textures.size());
    for (size_t t = 0; t < o.textures.size(); t++) {
      const uint32_t outer = o.textures[t].d > 0 ? o.textures[t].d : o.textures[t].layers;
      const size_t n = (size_t)o.textures[t].w * o.textures[t].h * outer;
      texdata[t].resize(n);
      for (size_t i = 0; i < n; i++) texdata[t][i] = det_value((uint32_t)i, (uint32_t)(100 + t));
    }
    std::vector<std::vector<unsigned char>> texbytes(o.textures.size());
    for (size_t t = 0; t < texdata.size(); t++) {
      texbytes[t].resize(texdata[t].size() * o.texel_bytes);
      for (size_t i = 0; i < texdata[t].size(); i++) {
        if (o.texel_bytes == 2) {
          const uint16_t h = half_bits_of(texdata[t][i]);
          std::memcpy(&texbytes[t][2 * i], &h, 2);
        } else {
          std::memcpy(&texbytes[t][4 * i], &texdata[t][i], 4);
        }
      }
    }
    auto set_texture =
        (void (*)(int, void*, uint32_t, uint32_t, uint32_t))lib_sym(handle, "weft_set_texture");
    auto set_texture_layers = (void (*)(int, void*, uint32_t, uint32_t, uint32_t, uint32_t))
        lib_sym(handle, "weft_set_texture_layers");
    if (set_texture_layers)
      for (size_t t = 0; t < o.textures.size(); t++)
        set_texture_layers((int)t, texbytes[t].data(), o.textures[t].w, o.textures[t].h,
                           o.texel_bytes,
                           o.textures[t].d > 0 ? o.textures[t].d : o.textures[t].layers);
    else if (set_texture)
      for (size_t t = 0; t < o.textures.size(); t++) {
        if (o.textures[t].layers != 1) {
          std::fprintf(stderr, "weft: this module cannot bind an array texture\n");
          return 2;
        }
        set_texture((int)t, texbytes[t].data(), o.textures[t].w, o.textures[t].h, o.texel_bytes);
      }
    if (kern.ntex > (int)o.textures.size()) {
      std::fprintf(stderr, "weft: kernel '%s' reads %d texture%s, --textures gave %zu\n",
                   kern.name.c_str(), kern.ntex, kern.ntex == 1 ? "" : "s", o.textures.size());
      return 2;
    }
    auto set_sampler = (void (*)(int, int, int))lib_sym(handle, "weft_set_sampler");
    if (set_sampler) {
      const size_t want = (size_t)kern.nsamp > o.samplers.size() ? (size_t)kern.nsamp : o.samplers.size();
      for (size_t s = 0; s < want; s++) {
        const std::pair<int, int> v = s < o.samplers.size() ? o.samplers[s] : std::pair<int, int>{0, 0};
        set_sampler((int)s, v.first, v.second);
      }
    }
    if (!run_cpu || !kernel_count) { std::fprintf(stderr, "weft: bad kernel library\n"); return 1; }
    if (kernel_count() != (int)tr.kernels.size()) { std::fprintf(stderr, "weft: kernel table mismatch\n"); return 1; }

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
      std::printf("task      %u threadgroup%s of %u lanes  meshgrid it set: %u mesh%s\n",
                  task_groups, task_groups == 1 ? "" : "s", o.threads, set,
                  set == 1 ? "" : "es");
      o.grid = set * 32u;
      o.threads = 32u;
      o.threads_set = true;
    }

    const bool tile_mem = o.render && tile_kidx >= 0 && kern.tile_idx >= 0;
    const size_t ntile = tile_mem ? 1 : 0;
    const int tile_run = tile_mem ? tile_kidx : -1;
    const TranslatedKernel* tstage = tile_mem ? &tr.kernels[tile_kidx] : nullptr;
    const size_t nb = (o.render && tstage)
                          ? std::max((size_t)kern.nbufs, (size_t)tstage->nbufs)
                          : (size_t)kern.nbufs;
    const uint64_t meshes = kern.mesh ? ((uint64_t)o.grid + o.threads - 1) / o.threads : 0;
    const size_t ncolour = (o.render && kern.colour_targets > 0) ? (size_t)kern.colour_targets : 1u;
    // A `[[depth(n)]]` field is one more buffer, one float per pixel.
    const size_t ndepth = (o.render && kern.depth_idx >= 0) ? 1u : 0u;
    const size_t nbuf = o.render ? (nb + ntile + ncolour + ndepth) : (size_t)kern.nargs;
    const size_t nbind = nbuf - (o.render ? (ncolour + ndepth) : 0u);
    const size_t colour_idx = nbind;
    const size_t depth_idx = colour_idx + ncolour;
    if (o.bufs.size() != nb) {
      std::fprintf(stderr, "weft: kernel '%s' takes %zu buffers, --buffers gave %zu\n",
                   kern.name.c_str(), nb, o.bufs.size());
      return 2;
    }
    const uint32_t tile_threads = 32 * 32;
    if (tile_run >= 0 && kidx == tile_run) {
      std::fprintf(stderr,
                   "weft: '%s' is the pass's tile stage, not its fragment: --kernel has to "
                   "name a fragment\n",
                   kern.name.c_str());
      return 2;
    }
    if (tile_run >= 0 && tr.kernels[tile_run].ntex > (int)o.textures.size()) {
      std::fprintf(stderr, "weft: the tile stage '%s' reads %d texture%s, --textures gave %zu\n",
                   tr.kernels[tile_run].name.c_str(), tr.kernels[tile_run].ntex,
                   tr.kernels[tile_run].ntex == 1 ? "" : "s", o.textures.size());
      return 2;
    }
    std::vector<uint32_t> elem(nbuf, 4);
    if (kernel_at) {
      const weft_kernel_info* wi = kernel_at(tile_run >= 0 ? tile_run : kidx);
      for (size_t b = 0; wi && wi->buf_bytes && b < nb; b++)
        if (wi->buf_bytes[b]) elem[b] = wi->buf_bytes[b];
      const weft_kernel_info* fi = kernel_at(kidx);
      if (ntile && nbind > 0 && fi && fi->buf_bytes && kern.tile_idx >= 0)
        elem[nbind - 1] = fi->buf_bytes[kern.tile_idx];
      if (!o.render)
        for (size_t b = nb; fi && fi->buf_bytes && b < nbind; b++)
          if (fi->buf_bytes[b]) elem[b] = fi->buf_bytes[b];
    }
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
        bytes = (kern.mesh ? meshes : std::max<uint64_t>(1, (o.grid + o.threads - 1) / o.threads)) *
                (uint64_t)elem[b];
      else if ((int)b == kern.props_idx)
        bytes = 12ull;  // x, y, z
      else if (ntile && b == nbind - 1)
        bytes = tiles_x * tiles_y * 1024ull * (elem[b] ? elem[b] : 16ull);
      else
        bytes = (uint64_t)o.rw * o.rh * 16;
      const size_t n = (size_t)((bytes + 3) / 4);
      host[b].resize(n);
      for (size_t i = 0; i < n; i++) host[b][i] = det_value((uint32_t)i, (uint32_t)b);
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
    if (o.render)
      for (size_t c = 0; c < ncolour; c++)
        host[colour_idx + c].assign((size_t)o.rw * o.rh * 4, 0.0f);
    // The depth buffer starts at the attachment's clear depth, 1.0.
    if (ndepth) host[depth_idx].assign((size_t)o.rw * o.rh, 1.0f);
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
    std::vector<GuardedFloats> pristine = host;
    std::vector<void*> ptrs(nbuf);
    for (size_t b = 0; b < nbuf; b++) ptrs[b] = host[b].data();
    for (size_t b = 0; b < nbuf; b++)
      guard_tag(host[b].data(), host[b].size() * sizeof(float), (int)b, b < nb);
    // A render pass hands each stage a table in its own argument order.
    std::vector<size_t> frag_slot(nbuf);
    for (size_t b = 0; b < nbuf; b++) frag_slot[b] = b;
    bool frag_remap = false;
    if (o.render && kern.colour_targets > 0 && kernel_at) {
      const size_t first_colour = (size_t)kern.nargs - ndepth - ncolour;
      if (ntile && kern.tile_idx >= 0) {
        frag_slot[(size_t)kern.tile_idx] = nbind - 1;
        if ((size_t)kern.tile_idx != nbind - 1) frag_remap = true;
      }
      for (size_t c = 0; c < ncolour && first_colour + c < nbuf; c++) {
        frag_slot[first_colour + c] = colour_idx + c;
        if (frag_slot[first_colour + c] != first_colour + c) frag_remap = true;
      }
    }
    std::vector<void*> fptrs(nbuf, nullptr);
    for (size_t b = 0; b < nbuf; b++)
      fptrs[b] = b < ptrs.size() ? ptrs[frag_slot[b]] : nullptr;
    std::vector<void*>& kptrs = frag_remap ? fptrs : ptrs;
    g_guard_grid = o.grid;
    g_guard_threads = o.threads;

    const bool report = !o.emit_gpu;
    if (report) std::printf("weft  gpu on cpu\n\n");
    if (kern.mesh && report)
      std::printf("mesh      %d vertices and %d primitives per mesh  %llu mesh%s for this grid\n",
                  kern.mesh_nv, kern.mesh_np, (unsigned long long)meshes,
                  meshes == 1 ? "" : "es");
    if (o.render) {
      if (report) {
        std::printf("fragment  %s  target %ux%u  %zu buffer%s\n", kern.name.c_str(), o.rw,
                    o.rh, nb, nb == 1 ? "" : "s");
        // Which engine computes the fragment's inputs.
        if (!tr.vertex_name.empty())
          std::printf("%-9s %s\n", "interp",
                      tr.lowered ? "weft, the pass interpolates its own stage_in"
                                 : "hardware, the file's vertex stage feeds the interpolator");
      }
    }
    else {
      // A front end that types its pointers names the element (`float4 16B`).
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
        std::printf("kernel    %s  grid %s  threadgroup %u  %zu buffer%s (%s)\n",
                    kern.name.c_str(), fmt_uint(o.grid).c_str(), o.threads, nb,
                    nb == 1 ? "" : "s", types.c_str());
      if (report && unknown_width)
        std::printf("note      * nothing typed reaches the buffers marked *, so the element "
                    "width is\n"
                    "           unknown: 4 bytes is assumed. State one with "
                    "`--buffer-bytes I=N[,J=N]`.\n");
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
      std::printf("translate %zu kernel%s, %d with barriers  %s  %s\n",
                  tr.kernels.size(), tr.kernels.size() == 1 ? "" : "s", kern.sync ? 1 : 0,
                  timing, o.front.c_str());

    std::vector<std::vector<float>> cpu_out(nbuf);
    for (size_t b = 0; b < nbuf; b++) cpu_out[b].assign(host[b].size(), 0.0f);

    auto launch_once = [&]() -> double {
      for (size_t b = 0; b < nbuf; b++)
        std::memcpy(host[b].data(), pristine[b].data(), host[b].size() * sizeof(float));
      double s = now_ms();
      if (tile_reset) tile_reset();
      if (fault_reset) fault_reset();
      if (tile_run >= 0) run_cpu(tile_run, ptrs.data(), o.grid, tile_threads);
      run_cpu(kidx, kptrs.data(), o.grid, o.threads);
      return now_ms() - s;
    };

    const uint32_t hw = std::max(1u, (uint32_t)std::thread::hardware_concurrency());
    std::vector<uint32_t> cands{1};
    if (hw > 1) cands.push_back(2);
    if (hw > 5) cands.push_back(hw / 2);
    if (hw > 2) cands.push_back(hw);
    uint32_t threads = hw;
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
    size_t growths = 0;
    for (;;) {
      best_cpu = 1e18;
      sum_cpu = 0.0;
      for (uint32_t it = 0; it < (o.emit_gpu ? 0u : o.bench + 1); it++) {
        for (size_t b = 0; b < nbuf; b++)
          std::memcpy(host[b].data(), pristine[b].data(), host[b].size() * sizeof(float));
        double s = now_ms();
        if (tile_reset) tile_reset();
        if (tile_run >= 0) run_cpu(tile_run, ptrs.data(), o.grid, tile_threads);
        run_cpu(kidx, kptrs.data(), o.grid, o.threads);
        double d = now_ms() - s;
        for (size_t b = 0; b < nbuf; b++)
          std::memcpy(cpu_out[b].data(), host[b].data(), host[b].size() * sizeof(float));
        if (it > 0) {  // first run is warm-up
          sum_cpu += d;
          if (d < best_cpu) best_cpu = d;
        }
      }

      bool grew = false;
      for (size_t b = 0; b < nbuf; b++) {
        size_t at = 0;
        if (!guard_past((int)b, &at)) continue;
        const size_t per = elem[b] ? (size_t)elem[b] : 1;
        const size_t need = (at + per - 1) / per;
        const size_t was = host[b].size();
        if (b < nb) o.bufs[b] = need;
        const size_t words = (need * per + sizeof(float) - 1) / sizeof(float);
        host[b].resize(words, 0.0f);
        std::memcpy(host[b].data(), pristine[b].data(),
                    std::min(was, pristine[b].size()) * sizeof(float));
        for (size_t i = was; i < words; i++) {
          if (o.render && b >= colour_idx) host[b][i] = 0.0f;
          else if (ndepth && b == depth_idx) host[b][i] = 1.0f;
          else if ((int)b == kern.payload_idx) host[b][i] = 0.0f;
          else host[b][i] = det_value((uint32_t)i, (uint32_t)b);
        }
        pristine[b] = host[b];
        cpu_out[b].assign(words, 0.0f);
        ptrs[b] = host[b].data();
        guard_tag(host[b].data(), host[b].size() * sizeof(float), (int)b, b < nb);
        if (b < nb) {
          std::string fits;
          for (size_t i = 0; i < nb; i++)
            fits += (i ? "," : "") + std::to_string(i == b ? need : o.bufs[i]);
          std::printf("note      buffer %zu was %zu element%s%s and the body writes %zu bytes: "
                      "using %zu. Pass --buffers %s to size it yourself.\n",
                      b, was, was == 1 ? "" : "s", o.bufs_stated ? " from --buffers" : "", at,
                      need, fits.c_str());
        } else {
          std::printf("note      buffer %zu is sized by the launcher and the body writes %zu "
                      "bytes: using %zu.\n",
                      b, at, need);
        }
        grew = true;
      }
      if (!grew) break;
      // A body that keeps reaching further would turn one run into a loop.
      if (++growths > 4) {
        std::fprintf(stderr,
                     "weft: the launch kept writing past a block after it grew four times; "
                     "this run is refused rather than answered on a guess\n");
        return 2;
      }
    }

    bool tile_undef = false;
    if (tile_violations) {
      const unsigned long long hits = (unsigned long long)tile_violations();
      if (hits) {
        tile_undef = true;
        std::printf("note      %s reads tile memory outside its own 32x32 tile (%llu "
                    "access%s).\n"
                    "          On the GPU that read is another tile's, which may not have "
                    "been written yet:\n"
                    "          the run happens, and there is no single answer to "
                    "compare.\n",
                    kern.name.c_str(), hits, hits == 1 ? "" : "es");
      }
    }

    // The GPU column is Metal's, so it exists only in the Apple build.
    double gpu_avg = -1.0, gpu_best = -1.0;
    std::vector<std::vector<float>> gpu_out(nbuf);
    std::string gpu_name = "unavailable";
    bool gpu_ran = false;

    if (o.virtual_gpu) {
      vmetal::ModuleApi api;
      api.kernel_count = kernel_count;
      api.kernel_at = kernel_at;
      api.run = run_cpu;
      api.set_param = set_param;
      api.set_texture = set_texture;
      api.set_texture_layers = set_texture_layers;
      api.set_sampler = set_sampler;
      vmetal::Device dev = vmetal::Device::adopt(api);
      std::string perr;
      vmetal::PipelineState pipe = dev.new_pipeline(kern.name.c_str(), &perr);
      vmetal::PipelineState tile_pipe;
      if (pipe.valid() && tile_run >= 0)
        tile_pipe = dev.new_pipeline(tr.kernels[tile_run].name.c_str(), &perr);
      if (!pipe.valid() || (tile_run >= 0 && !tile_pipe.valid())) {
        gpu_refusal = perr;
      } else {
        for (size_t b = 0; b < nbuf; b++)
          std::memcpy(host[b].data(), pristine[b].data(), host[b].size() * sizeof(float));
        std::vector<vmetal::Buffer> vb(nbuf);
        for (size_t b = 0; b < nbuf; b++)
          vb[b] = vmetal::Device::wrap(ptrs[b], host[b].size() * sizeof(float));
        std::vector<vmetal::Texture> vt(kern.tex_name.size());
        for (size_t tx = 0; tx < vt.size(); tx++)
          vt[tx] = dev.new_texture(o.textures[tx].w, o.textures[tx].h,
                                   o.textures[tx].d > 0 ? o.textures[tx].d : o.textures[tx].layers,
                                   o.texel_bytes, texbytes[tx].data());
        vmetal::CommandQueue q = dev.new_queue();
        double virt_ms = 0.0;
        auto record = [&](vmetal::PipelineState p, uint32_t grid, uint32_t threads) {
          vmetal::CommandBuffer c = q.command_buffer();
          vmetal::Encoder enc = c.compute_encoder();
          enc.set_pipeline(p);
          for (size_t b = 0; b < nbuf; b++)
            enc.set_buffer((int)b,
                           vb[frag_remap && p.index == pipe.index ? frag_slot[b] : b]);
          for (size_t tx = 0; tx < vt.size(); tx++) enc.set_texture((int)tx, vt[tx]);
          const size_t want =
              (size_t)kern.nsamp > o.samplers.size() ? (size_t)kern.nsamp : o.samplers.size();
          for (size_t s = 0; s < want; s++) {
            const std::pair<int, int> v =
                s < o.samplers.size() ? o.samplers[s] : std::pair<int, int>{0, 0};
            enc.set_sampler((int)s, v.first, v.second);
          }
          enc.dispatch_threads(grid, threads);
          enc.end_encoding();
          c.commit();
          c.wait_until_completed();
          virt_ms += c.elapsed_ms();
          return c;
        };
        vmetal::CommandBuffer cb;
        if (tile_pipe.valid()) cb = record(tile_pipe, o.grid, tile_threads);
        if (!tile_pipe.valid() || cb.ok()) cb = record(pipe, o.grid, o.threads);
        if (!cb.ok()) {
          gpu_refusal = cb.error();
        } else {
          gpu_name = dev.name() + std::string(" (") + std::to_string(o.threads) +
                     " threads/group)";
          gpu_avg = virt_ms;
          gpu_best = virt_ms;
          for (size_t b = 0; b < nbuf; b++) {
            gpu_out[b].resize(host[b].size());
            std::memcpy(gpu_out[b].data(), ptrs[b], host[b].size() * sizeof(float));
          }
          gpu_ran = true;
        }
      }
    }

#if WEFT_METAL
    const bool gpu_trace = std::getenv("WEFT_GPU_TRACE") != nullptr;
    const double gpu_t0 = now_ms();
    auto gpu_trace_at = [&](const char* what) {
      if (gpu_trace) std::fprintf(stderr, "weft: gpu %8.3f ms  %s\n", now_ms() - gpu_t0, what);
    };

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
    if (!o.cpu_only && !o.virtual_gpu) {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev) {
        std::fprintf(stderr, "weft: no Metal device (use --cpu-only)\n");
        return 1;
      }
      MTLSize max_tg = [dev maxThreadsPerThreadgroup];
      gpu_name = std::string([[dev name] UTF8String]) + " (" + std::to_string(max_tg.width) +
                 " threads/group)";
      std::string gpu_src = tr.gpu_src;
      if (o.render && (tr.vertex_name.empty() || tr.lowered)) {
        if (!tr.lowered) LowerStageIn(gpu_src, o.rw, o.rh);
        if (!tr.lowered && tr.points) {
          gpu_src += PointVertexSource(o.rw, o.rh);
        } else if (!tr.lowered && gpu_src.find("[[stage_in]]") != std::string::npos) {
          const std::string vs = FullScreenVertexFor(gpu_src);
          gpu_src += vs.empty() ? kFullScreenVertex : vs;
        } else {
          gpu_src += kFullScreenVertex;
        }
      }
      if (o.ptx || spv) gpu_src = std::string();
      if (gpu_src.empty()) gpu_src = tr.gpu_src;
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
        gpu_src.insert(at, std::string(kWWeftF64) + kWSoftMath);
      }
      if (o.emit_gpu) {
        std::fputs(gpu_src.c_str(), stdout);
        return 0;
      }
      NSString* ns = [[NSString alloc] initWithBytes:gpu_src.data()
                                              length:gpu_src.size()
                                            encoding:NSUTF8StringEncoding];
      NSError* err = nil;
      // Metal compiles with fast math by default.
      MTLCompileOptions* copts = [MTLCompileOptions new];
      if (@available(macOS 15.0, *)) {
        copts.mathMode = MTLMathModeSafe;  // IEEE: no approximations, no reassociation
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        copts.fastMathEnabled = NO;       // the same setting, before mathMode existed
#pragma clang diagnostic pop
      }
      const bool reuse_build = probed_pso != nil && probed_dev == dev && probed_src == gpu_src;
      id<MTLLibrary> mtllib = reuse_build ? probed_lib : nil;
      if (!mtllib) {
        gpu_trace_at("compiling the source");
        // Some Metal versions ship without the matrix helpers a shader may call.
        static NSString* const kMslInverse =
            @"inline float2x2 inverse(float2x2 m) {\n"
             "  const float d = m[0].x * m[1].y - m[1].x * m[0].y;\n"
             "  return float2x2(float2(m[1].y, -m[0].y), float2(-m[1].x, m[0].x)) * (1.0f / d);\n"
             "}\n"
             "inline float3x3 inverse(float3x3 m) {\n"
             "  const float d = dot(m[0], cross(m[1], m[2]));\n"
             "  return float3x3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1])) * (1.0f / d);\n"
             "}\n"
             "inline float4x4 inverse(float4x4 m) {\n"
             "  float a[4][8];\n"
             "  for (int i = 0; i < 4; i++)\n"
             "    for (int j = 0; j < 4; j++) { a[i][j] = m[j][i]; a[i][j + 4] = (i == j) ? 1.0f : 0.0f; }\n"
             "  for (int col = 0; col < 4; col++) {\n"
             "    int piv = col;\n"
             "    for (int r = col + 1; r < 4; r++)\n"
             "      if (abs(a[r][col]) > abs(a[piv][col])) piv = r;\n"
             "    if (a[piv][col] == 0.0f) return float4x4(0.0f);\n"
             "    for (int k = 0; k < 8; k++) { const float t = a[col][k]; a[col][k] = a[piv][k]; a[piv][k] = t; }\n"
             "    const float s = 1.0f / a[col][col];\n"
             "    for (int k = 0; k < 8; k++) a[col][k] *= s;\n"
             "    for (int r = 0; r < 4; r++) {\n"
             "      if (r == col) continue;\n"
             "      const float f = a[r][col];\n"
             "      for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];\n"
             "    }\n"
             "  }\n"
             "  return float4x4(float4(a[0][4], a[1][4], a[2][4], a[3][4]),\n"
             "                  float4(a[0][5], a[1][5], a[2][5], a[3][5]),\n"
             "                  float4(a[0][6], a[1][6], a[2][6], a[3][6]),\n"
             "                  float4(a[0][7], a[1][7], a[2][7], a[3][7]));\n"
             "}\n";
        static NSString* const kMslTranspose =
            @"inline float2x2 transpose(float2x2 m) {\n"
             "  return float2x2(float2(m[0].x, m[1].x), float2(m[0].y, m[1].y));\n"
             "}\n"
             "inline float3x3 transpose(float3x3 m) {\n"
             "  return float3x3(float3(m[0].x, m[1].x, m[2].x), float3(m[0].y, m[1].y, m[2].y),\n"
             "                  float3(m[0].z, m[1].z, m[2].z));\n"
             "}\n"
             "inline float4x4 transpose(float4x4 m) {\n"
             "  return float4x4(float4(m[0].x, m[1].x, m[2].x, m[3].x),\n"
             "                  float4(m[0].y, m[1].y, m[2].y, m[3].y),\n"
             "                  float4(m[0].z, m[1].z, m[2].z, m[3].z),\n"
             "                  float4(m[0].w, m[1].w, m[2].w, m[3].w));\n"
             "}\n";
        static NSString* const kMslDeterminant =
            @"inline float determinant(float2x2 m) { return m[0].x * m[1].y - m[1].x * m[0].y; }\n"
             "inline float determinant(float3x3 m) { return dot(m[0], cross(m[1], m[2])); }\n"
             "inline float determinant(float4x4 m) {\n"
             "  const float3 r0 = float3(m[1].x, m[2].x, m[3].x);\n"
             "  const float3 r1 = float3(m[1].y, m[2].y, m[3].y);\n"
             "  const float3 r2 = float3(m[1].z, m[2].z, m[3].z);\n"
             "  const float3 r3 = float3(m[1].w, m[2].w, m[3].w);\n"
             "  return m[0].x * dot(r1, cross(r2, r3)) - m[0].y * dot(r0, cross(r2, r3)) +\n"
             "         m[0].z * dot(r0, cross(r1, r3)) - m[0].w * dot(r0, cross(r1, r2));\n"
             "}\n";
        auto compile_msl = [&](NSString* s) -> id<MTLLibrary> {
          id<MTLLibrary> lib = [dev newLibraryWithSource:s options:copts error:&err];
          if (lib) return lib;
          NSString* why = err ? [err localizedDescription] : @"";
          NSMutableString* add = [NSMutableString string];
          for (const char* n : {"inverse", "transpose", "determinant"}) {
            NSString* key = [NSString stringWithFormat:@"undeclared identifier '%s'", n];
            if (![why containsString:key]) continue;
            [add appendString:std::strcmp(n, "inverse") == 0
                                  ? kMslInverse
                                  : (std::strcmp(n, "transpose") == 0 ? kMslTranspose
                                                                      : kMslDeterminant)];
          }
          if (add.length == 0) return nil;
          const NSRange um = [s rangeOfString:@"using namespace metal;"];
          if (um.location == NSNotFound) return nil;
          NSMutableString* fixed = [s mutableCopy];
          [fixed insertString:add atIndex:NSMaxRange(um)];
          err = nil;
          return [dev newLibraryWithSource:fixed options:copts error:&err];
        };
        mtllib = compile_msl(ns);
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
        const std::string vname = tr.lowered                  ? std::string("weft_vertex")
                                  : !tr.vertex_name.empty()   ? tr.vertex_name
                                  : tr.points                 ? std::string("weft_points_vertex")
                                                              : std::string("weft_vertex");
        const std::string& fname = kern.name;
        id<MTLFunction> vf =
            [mtllib newFunctionWithName:[NSString stringWithUTF8String:vname.c_str()]];
        id<MTLFunction> ff = nil;
        if (tr.function_constants.empty()) {
          ff = [mtllib newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
        } else {
          MTLFunctionConstantValues* fcv = [MTLFunctionConstantValues new];
          for (const auto& fc : tr.function_constants) {
            if (fc.second == "bool") {
              bool v = false;
              [fcv setConstantValue:&v type:MTLDataTypeBool atIndex:(NSUInteger)fc.first];
            } else if (fc.second == "float" || fc.second == "half") {
              float v = 0.0f;
              [fcv setConstantValue:&v type:MTLDataTypeFloat atIndex:(NSUInteger)fc.first];
            } else if (fc.second == "uint") {
              uint32_t v = 0u;
              [fcv setConstantValue:&v type:MTLDataTypeUInt atIndex:(NSUInteger)fc.first];
            } else {
              int v = 0;
              [fcv setConstantValue:&v type:MTLDataTypeInt atIndex:(NSUInteger)fc.first];
            }
          }
          ff = [mtllib newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]
                             constantValues:fcv
                                      error:&err];
        }
        if (!vf || !ff) {
          std::fprintf(stderr, "weft: the library has no %s\n",
                       vf ? kern.name.c_str() : vname.c_str());
          return 1;
        }
        // The tile stage runs as its own pass before the draw.
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
        for (size_t c = 0; c < ncolour; c++)
          rpd.colorAttachments[c].pixelFormat = MTLPixelFormatRGBA32Float;
        if (ndepth) rpd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
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

      // The fragment's own buffers.
      std::vector<id<MTLBuffer>> mbuf(nbind);
      for (size_t b = 0; b < nbind; b++) {
        mbuf[b] = [dev newBufferWithLength:host[b].size() * sizeof(float)
                                   options:MTLResourceStorageModeShared];
      }
      id<MTLBuffer> par_buf = nil;
      int par_n = 1;
      if (o.ptx || spv) {
        for (const auto& a : o.args) par_n = std::max(par_n, a.first + 1);
        par_buf = [dev newBufferWithLength:(size_t)par_n * sizeof(uint64_t)
                                   options:MTLResourceStorageModeShared];
      }
      // A warp-backend kernel needs the launch it was given.
      id<MTLBuffer> geom_buf = nil;
      if (kern.gpu_warp)
        geom_buf = [dev newBufferWithLength:4 * sizeof(uint64_t)
                                   options:MTLResourceStorageModeShared];
      // The lock words, and one past them for the fault a lane reports.
      id<MTLBuffer> lock_buf = [dev newBufferWithLength:1026 * sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
      if (lock_buf) std::memset([lock_buf contents], 0, 1026 * sizeof(uint32_t));
      std::vector<id<MTLTexture>> targets(ncolour, nil);
      if (o.render) {
        for (size_t c = 0; c < ncolour; c++) {
          MTLTextureDescriptor* td = [MTLTextureDescriptor
              texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                           width:o.rw
                                          height:o.rh
                                       mipmapped:NO];
          td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
          td.storageMode = MTLStorageModeShared;
          targets[c] = [dev newTextureWithDescriptor:td];
        }
      }
      id<MTLTexture> depth_target = nil;
      if (ndepth) {
        MTLTextureDescriptor* dd = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                         width:o.rw
                                        height:o.rh
                                     mipmapped:NO];
        dd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        dd.storageMode = MTLStorageModeShared;
        depth_target = [dev newTextureWithDescriptor:dd];
      }
      id<MTLDepthStencilState> dss = nil;
      if (ndepth) {
        MTLDepthStencilDescriptor* dsd = [MTLDepthStencilDescriptor new];
        const std::string& dq = kern.depth_qual;
        dsd.depthCompareFunction =
            dq == "less"            ? MTLCompareFunctionLess
            : dq == "less_equal"    ? MTLCompareFunctionLessEqual
            : dq == "greater"       ? MTLCompareFunctionGreater
            : dq == "greater_equal" ? MTLCompareFunctionGreaterEqual
            : dq == "equal"         ? MTLCompareFunctionEqual
            : dq == "not_equal"     ? MTLCompareFunctionNotEqual
            : dq == "never"         ? MTLCompareFunctionNever
                                    : MTLCompareFunctionAlways;
        dsd.depthWriteEnabled = YES;
        dss = [dev newDepthStencilStateWithDescriptor:dsd];
      }
      std::vector<id<MTLTexture>> mtex(o.textures.size());
      for (size_t t = 0; t < o.textures.size(); t++) {
        const Options::TexBind& tb = o.textures[t];
        // A volume is its own descriptor.
        MTLTextureDescriptor* td;
        if (tb.d > 0) {
          td = [[MTLTextureDescriptor alloc] init];
          td.pixelFormat =
              o.texel_bytes == 2 ? MTLPixelFormatR16Float : MTLPixelFormatR32Float;
          td.textureType = MTLTextureType3D;
          td.width = tb.w;
          td.height = tb.h;
          td.depth = tb.d;
          td.mipmapLevelCount = 1;
        } else {
          td = [MTLTextureDescriptor
              texture2DDescriptorWithPixelFormat:(o.texel_bytes == 2 ? MTLPixelFormatR16Float
                                                                    : MTLPixelFormatR32Float)
                                           width:tb.w
                                          height:tb.h
                                       mipmapped:NO];
        }
        // An array binding is the texture type the kernel declares.
        if (tb.one_d) {
          td.textureType = MTLTextureType1D;
        } else if (tb.cube && tb.layers > 6) {
          td.textureType = MTLTextureTypeCubeArray;
          td.arrayLength = tb.layers / 6;
        } else if (tb.cube) {
          td.textureType = MTLTextureTypeCube;
        } else if (tb.layers > 1) {
          td.textureType = MTLTextureType2DArray;
          td.arrayLength = tb.layers;
        }
        // Writable, because a kernel may hold a storage image.
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        mtex[t] = [dev newTextureWithDescriptor:td];
        if (tb.d > 0) {
          id<MTLBuffer> src = [dev newBufferWithBytes:texbytes[t].data()
                                               length:texbytes[t].size()
                                              options:MTLResourceStorageModeShared];
          id<MTLCommandQueue> q = [dev newCommandQueue];
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
          [bl copyFromBuffer:src
                  sourceOffset:0
              sourceBytesPerRow:(NSUInteger)tb.w * o.texel_bytes
            sourceBytesPerImage:(NSUInteger)tb.w * tb.h * o.texel_bytes
                    sourceSize:MTLSizeMake(tb.w, tb.h, tb.d)
                     toTexture:mtex[t]
              destinationSlice:0
              destinationLevel:0
             destinationOrigin:MTLOriginMake(0, 0, 0)];
          [bl endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
        } else if (tb.layers > 1) {
          const size_t slice = (size_t)tb.w * tb.h * o.texel_bytes;
          for (uint32_t l = 0; l < tb.layers; l++)
            [mtex[t] replaceRegion:MTLRegionMake3D(0, 0, 0, tb.w, tb.h, 1)
                       mipmapLevel:0
                             slice:l
                         withBytes:texbytes[t].data() + (size_t)l * slice
                       bytesPerRow:(NSUInteger)tb.w * o.texel_bytes
                      bytesPerImage:slice];
        }
        else
          [mtex[t] replaceRegion:MTLRegionMake2D(0, 0, tb.w, tb.h)
                     mipmapLevel:0
                       withBytes:texbytes[t].data()
                     bytesPerRow:(NSUInteger)tb.w * o.texel_bytes];
      }

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
          gg[1] = kern.wide ? 32u : o.threads;  // lanes in one threadgroup
          gg[2] = lock_buf ? (uint64_t)(uintptr_t)[lock_buf contents] : 0;
        }
        if (lock_buf) std::memset([lock_buf contents], 0, 1026 * sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [q commandBuffer];
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
          for (size_t c = 0; c < ncolour; c++) {
            rpd.colorAttachments[c].texture = targets[c];
            rpd.colorAttachments[c].loadAction = MTLLoadActionClear;
            rpd.colorAttachments[c].storeAction = MTLStoreActionStore;
            rpd.colorAttachments[c].clearColor = MTLClearColorMake(0, 0, 0, 0);
          }
          if (ndepth) {
            rpd.depthAttachment.texture = depth_target;
            rpd.depthAttachment.loadAction = MTLLoadActionClear;
            rpd.depthAttachment.storeAction = MTLStoreActionStore;
            rpd.depthAttachment.clearDepth = 1.0;
          }
          id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
          [enc setRenderPipelineState:rpso];
          if (dss) [enc setDepthStencilState:dss];
          for (size_t b = 0; b < nbind; b++) {
            const size_t src = frag_slot[b];
            if (src >= nbind) continue;
            [enc setFragmentBuffer:mbuf[src] offset:0 atIndex:b];
          }
          for (size_t b = 0; b < nbind; b++) [enc setVertexBuffer:mbuf[b] offset:0 atIndex:b];
          for (size_t t = 0; t < mtex.size(); t++) [enc setFragmentTexture:mtex[t] atIndex:t];
          for (int slot : tr.fragment_sampler_slots)
            [enc setFragmentSamplerState:make_sampler(0, 0) atIndex:(NSUInteger)slot];
          for (size_t t = 0; t < mtex.size(); t++) [enc setVertexTexture:mtex[t] atIndex:t];
          for (int slot : tr.vertex_sampler_slots)
            [enc setVertexSamplerState:make_sampler(0, 0) atIndex:(NSUInteger)slot];
          // Without a vertex stage the pass draws one point per pixel.
          if (tr.points && !tr.lowered)
            [enc drawPrimitives:MTLPrimitiveTypePoint
                    vertexStart:0
                    vertexCount:tr.vertex_name.empty()
                                    ? (NSUInteger)((size_t)o.rw * o.rh)
                                    : (NSUInteger)3];
          else
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [enc endEncoding];
        } else {
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:pso];
          for (size_t b = 0; b < nbind; b++) [enc setBuffer:mbuf[b] offset:0 atIndex:b];
          // The scalar parameters sit after the buffers the kernel declared.
          if (o.ptx || spv) [enc setBuffer:par_buf offset:0 atIndex:nbind];
          if (geom_buf) [enc setBuffer:geom_buf offset:0 atIndex:nbind + 1];
          else if (lock_buf) [enc setBuffer:lock_buf offset:0 atIndex:nbind + 1];
          if (kern.gpu_warp && lock_buf) [enc setBuffer:lock_buf offset:0 atIndex:nbind + 2];
          for (size_t t = 0; t < mtex.size(); t++) [enc setTexture:mtex[t] atIndex:t];
          for (size_t sm = 0; sm < msamp.size(); sm++) [enc setSamplerState:msamp[sm] atIndex:sm];
          for (size_t t = 0; t < mtex.size(); t++) {
            if (t >= kern.tex_rw.size() || !kern.tex_rw[t]) continue;
            const NSUInteger at = (NSUInteger)(kern.ntex + (int)t);
            [enc setTexture:mtex[t] atIndex:at];
            [enc setSamplerState:(t < msamp.size() ? msamp[t] : make_sampler(0, 0)) atIndex:at];
          }
          if (kern.gpu_warp) {
            {
              const uint32_t warps = (o.grid + 31u) / 32u;
              uint32_t per = std::max(1u, std::min(o.threads / 32u, warps));
              if (kern.slot_per) per = std::max(1u, std::min(per, kern.slot_per));
              if (kern.wide) per = 1u;
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
        for (int spin = 0; spin < guard_ms; spin++) {
          const MTLCommandBufferStatus st = cb.status;
          if (st == MTLCommandBufferStatusCompleted || st == MTLCommandBufferStatusError)
            break;
          [NSThread sleepForTimeInterval:0.001];
        }
        if (cb.status != MTLCommandBufferStatusCompleted) {
          if (!reissued) {
            gpu_trace_at("the dispatch did not complete; issuing it again on a new queue");
            reissued = true;
            guard_ms = 20000;
            q = [dev newCommandQueue];
            goto again;
          }
          gpu_trace_at("the dispatch did not complete a second time");
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
          for (size_t c = 0; c < ncolour; c++) {
            gpu_out[colour_idx + c].assign(host[colour_idx + c].size(), 0.0f);
            [targets[c] getBytes:gpu_out[colour_idx + c].data()
                     bytesPerRow:(NSUInteger)o.rw * 4 * sizeof(float)
                      fromRegion:MTLRegionMake2D(0, 0, o.rw, o.rh)
                     mipmapLevel:0];
          }
          if (ndepth) {
            gpu_out[depth_idx].assign(host[depth_idx].size(), 0.0f);
            [depth_target getBytes:gpu_out[depth_idx].data()
                       bytesPerRow:(NSUInteger)o.rw * sizeof(float)
                        fromRegion:MTLRegionMake2D(0, 0, o.rw, o.rh)
                       mipmapLevel:0];
          }
        }
      }
      if (gpu_refusal.empty() && lock_buf) {
        const uint32_t fault = ((const uint32_t*)[lock_buf contents])[1024];
        const uint32_t at = ((const uint32_t*)[lock_buf contents])[1025];
        if (fault)
          gpu_refusal =
              (fault == 2u
                   ? std::string("a lane trapped, and the run stopped there")
                   : std::string("a lane stopped: recursion ran past the 8 frames "
                                 "this device keeps")) +
              " (lane " + std::to_string(at) + ")";
      }
      gpu_ran = gpu_refusal.empty();
    }
#else
    if (!o.cpu_only && !o.virtual_gpu) {
      if (spv) {
        gpu_refusal = "the cuda column runs ptx, not spir-v";
      } else if (!o.ptx) {
        gpu_refusal = "the cuda column runs ptx, not " + o.front;
      } else if (o.render) {
        gpu_refusal = "the cuda column has no render stage";
      } else if (kern.tex_name.size() > o.textures.size()) {
        gpu_refusal = "a texture the kernel declares was not bound";
      } else {
        std::vector<std::vector<unsigned char>> in(nbuf);
        for (size_t b = 0; b < nbuf; b++) {
          in[b].resize(pristine[b].size() * sizeof(float));
          std::memcpy(in[b].data(), pristine[b].data(), in[b].size());
        }
        std::vector<weft_cuda::Texture> tex(kern.tex_name.size());
        for (size_t t = 0; t < tex.size(); t++) {
          tex[t].name = kern.tex_name[t];
          tex[t].bytes = texbytes[t];
          tex[t].w = o.textures[t].w;
          tex[t].h = o.textures[t].h;
          tex[t].layers = o.textures[t].layers;
          tex[t].depth = o.textures[t].d;
          tex[t].texel = o.texel_bytes;
          tex[t].one_d = o.textures[t].one_d;
          tex[t].cube = o.textures[t].cube;
          if (t < o.samplers.size()) {
            tex[t].filter = o.samplers[t].first;
            tex[t].addr = o.samplers[t].second;
          }
        }
        weft_cuda::Column cu = weft_cuda::run(src, kern.name, kern.params, in, tex, o.args,
                                              o.grid, o.threads, o.bench);
        if (!cu.ok) {
          gpu_refusal = cu.why;
        } else {
          gpu_name = cu.device + " (" + std::to_string(o.threads) + " threads/block)";
          gpu_avg = cu.avg_ms;
          gpu_best = cu.best_ms;
          for (size_t b = 0; b < nbuf; b++) {
            const size_t bytes = host[b].size() * sizeof(float);
            gpu_out[b].resize(host[b].size());
            if (b < cu.out.size() && cu.out[b].size() >= bytes)
              std::memcpy(gpu_out[b].data(), cu.out[b].data(), bytes);
          }
          gpu_ran = true;
        }
      }
    }
#endif

    const uint32_t stopped = fault_word ? fault_word() : 0u;
    if (stopped != 0u) {
      const uint32_t at = fault_lane ? fault_lane() : 0u;
      std::fprintf(stderr, "weft: kernel %s: %s (lane %u)\n", kern.name.c_str(),
                   stopped == 2u
                       ? "a lane trapped, and the run stopped there"
                       : "a lane stopped: recursion ran past the 8 frames this device keeps",
                   at);
      return 1;
    }
    if (!gpu_refusal.empty() && !gpu_ran)
      std::printf("%-9s %-34s %s\n", "gpu", "no column for this kernel", gpu_refusal.c_str());

    double worst_abs = 0.0, worst_rel = 0.0;
    size_t checked = 0, bitdiff = 0;
    for (size_t b = 0; b < nbuf && gpu_ran; b++) {
      for (size_t i = 0; i < gpu_out[b].size(); i++) {
        double a = gpu_out[b][i], c = cpu_out[b][i];
        double ad = std::fabs(a - c);
        double denom = std::fmax(std::fabs(a), 1e-30);
        if (ad > worst_abs) worst_abs = ad;
        if (ad / denom > worst_rel) worst_rel = ad / denom;
        if (o.exact) {
          uint32_t wa = 0, wc = 0;
          float fa = (float)gpu_out[b][i], fc = (float)cpu_out[b][i];
          std::memcpy(&wa, &fa, 4);
          std::memcpy(&wc, &fc, 4);
          if (wa != wc) bitdiff++;
        }
        checked++;
      }
    }

    bool unstable = false;
    if (gpu_ran && (o.exact ? bitdiff != 0 : worst_rel > 1e-4)) {
      const uint32_t was = threads;
      const uint32_t hw_threads = std::max(1u, (uint32_t)std::thread::hardware_concurrency());
      auto cpu_again = [&](uint32_t workers) {
        set_env("WEFT_THREADS", std::to_string(workers));
        for (size_t b = 0; b < nbuf; b++)
          std::memcpy(host[b].data(), pristine[b].data(), host[b].size() * sizeof(float));
        if (tile_reset) tile_reset();
        if (tile_run >= 0) run_cpu(tile_run, ptrs.data(), o.grid, tile_threads);
        run_cpu(kidx, kptrs.data(), o.grid, o.threads);
        for (size_t b = 0; b < nbuf; b++)
          if (std::memcmp(cpu_out[b].data(), host[b].data(), host[b].size() * sizeof(float)) != 0)
            return false;
        return true;
      };
      // No single sample is enough.
      if (o.virtual_gpu) {
        unstable = true;
      } else {
        unstable = !cpu_again(1u);
        for (uint32_t tries = 0; !unstable && tries < 4; tries++)
          unstable = !cpu_again(was == 1u ? hw_threads : was);
        set_env("WEFT_THREADS", std::to_string(was));
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

    if (!o.out.empty()) {
      if (!o.render) {
        std::fprintf(stderr, "weft: --out needs a render pass (--render WxH)\n");
        return 2;
      }
      // The first target is the frame `--out` names.
      for (size_t c = 0; c < ncolour; c++) {
        const std::string path = c == 0 ? o.out : (o.out + "." + std::to_string(c) + ".png");
        if (!write_png(path, cpu_out[colour_idx + c], o.rw, o.rh)) {
          std::fprintf(stderr, "weft: cannot write %s\n", path.c_str());
          return 1;
        }
        std::printf("%-9s %-34s %ux%u  (cpu)\n", "image", path.c_str(), o.rw, o.rh);
      }
      // Depth is one float per pixel: it leaves as a grey image beside the frame.
      if (ndepth) {
        std::vector<float> dpx((size_t)o.rw * o.rh * 4);
        for (size_t i = 0; i < (size_t)o.rw * o.rh; i++) {
          dpx[i * 4 + 0] = dpx[i * 4 + 1] = dpx[i * 4 + 2] = cpu_out[depth_idx][i];
          dpx[i * 4 + 3] = 1.0f;
        }
        const std::string path = o.out + ".depth.png";
        if (!write_png(path, dpx, o.rw, o.rh)) {
          std::fprintf(stderr, "weft: cannot write %s\n", path.c_str());
          return 1;
        }
        std::printf("%-9s %-34s %ux%u  (cpu)\n", "depth", path.c_str(), o.rw, o.rh);
      }
    }
    if (!o.out_gpu.empty()) {
      if (!o.render || !gpu_ran) {
        std::fprintf(stderr, "weft: --out-gpu needs a render pass and a gpu run\n");
        return 2;
      }
      for (size_t c = 0; c < ncolour; c++) {
        const std::string path =
            c == 0 ? o.out_gpu : (o.out_gpu + "." + std::to_string(c) + ".png");
        if (!write_png(path, gpu_out[colour_idx + c], o.rw, o.rh)) {
          std::fprintf(stderr, "weft: cannot write %s\n", path.c_str());
          return 1;
        }
        std::printf("%-9s %-34s %ux%u  (gpu)\n", "image", path.c_str(), o.rw, o.rh);
      }
      if (ndepth) {
        std::vector<float> dpx((size_t)o.rw * o.rh * 4);
        for (size_t i = 0; i < (size_t)o.rw * o.rh; i++) {
          dpx[i * 4 + 0] = dpx[i * 4 + 1] = dpx[i * 4 + 2] = gpu_out[depth_idx][i];
          dpx[i * 4 + 3] = 1.0f;
        }
        const std::string path = o.out_gpu + ".depth.png";
        if (!write_png(path, dpx, o.rw, o.rh)) {
          std::fprintf(stderr, "weft: cannot write %s\n", path.c_str());
          return 1;
        }
        std::printf("%-9s %-34s %ux%u  (gpu)\n", "depth", path.c_str(), o.rw, o.rh);
      }
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
      if (tile_undef) {
        std::printf("%-9s %s values  (not compared: this body reads tile memory outside its "
                    "tile, which the GPU leaves undefined)\n",
                    "compare", fmt_uint(checked).c_str());
      } else if (unstable) {
        std::printf("%-9s %s values  %s differ: the cpu answered differently on a second run, "
                    "so this kernel's answer depends on the order its threads run in\n",
                    "order", fmt_uint(checked).c_str(), fmt_uint(bitdiff).c_str());
      } else if (o.exact) {
        std::printf("%-9s %s values  %s differ: %s\n", "compare",
                    fmt_uint(checked).c_str(), fmt_uint(bitdiff).c_str(),
                    bitdiff == 0 ? "the words agree" : "DISAGREE");
      } else {
        std::printf("%-9s %s values  max |diff| %.3g  max rel %.3g%s\n", "compare",
                    fmt_uint(checked).c_str(), worst_abs, worst_rel,
                    worst_rel <= 1e-4 ? ": agree" : ": DISAGREE");
      }
    }
    std::printf("\n");
    return 0;
#if WEFT_METAL
  }
#endif
}
