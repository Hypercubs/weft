#pragma once
// The GPU column on a host whose driver is CUDA's. PTX is that driver's own
// input, so the kernel that runs is the one CUDA would run: the same text, not a
// second translation of it.
//
// The driver is opened at run time and the entry points used here are declared
// below, so this tree builds without the CUDA toolkit, its headers, or a device.
// A host with no driver reports why and the run stays a CPU one.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "translate.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace weft_cuda {

// One texture binding as the launcher has it: the bytes of the texels in the
// order the CPU runtime indexes them, the shape, and the sampler state the
// command line asked for. The CUDA column uploads these into a CUDA array.
struct Texture {
  std::string name;                    // the `.tex` the module declares
  std::vector<unsigned char> bytes;    // row-major texels, layer after layer
  uint32_t w = 0, h = 1, layers = 1, depth = 0;  // depth > 0 for a volume
  uint32_t texel = 4;                  // bytes per texel: 4 for a float, 2 for a half
  bool one_d = false, cube = false;
  int filter = 0;  // 0 nearest, 1 linear
  int addr = 0;    // 0 clamp to edge, 1 clamp to zero, 2 repeat
};

struct Column {
  bool ok = false;
  std::string why;     // why there is no column, when there is none
  std::string device;  // the device's own name
  double avg_ms = -1.0;
  double best_ms = -1.0;
  std::vector<std::vector<unsigned char>> out;  // one blob per buffer, same size in
};

// A `.surf` declaration is a surface, not a sampled texture. The front end
// numbers both in binding order, so the launcher hands one over like a texture
// and the module has no texture reference for it: say that, rather than asking
// the driver for a name it never declared.
inline bool is_surface(const std::string& ptx, const std::string& name) {
  size_t at = 0;
  while ((at = ptx.find(".surf", at)) != std::string::npos) {
    const size_t eol = ptx.find('\n', at);
    const std::string line =
        ptx.substr(at, eol == std::string::npos ? std::string::npos : eol - at);
    if (line.find(name) != std::string::npos) return true;
    at += 5;
  }
  return false;
}

namespace detail {

using Result = int;  // CUDA_SUCCESS is zero
using Device = int;
using DevicePtr = unsigned long long;
using Context = void*;
using Module = void*;
using Function = void*;
using Array = void*;
using Texref = void*;

// The driver's structs, field for field: the layout is the ABI, so the enum
// fields are `int` and the padding that follows each of them is the header's.
struct ArrayDesc {
  size_t width;
  size_t height;
  int format;
  unsigned channels;
};

struct ArrayDesc3D {
  size_t width;
  size_t height;
  size_t depth;
  int format;
  unsigned channels;
  unsigned flags;
};

struct Copy2D {
  size_t srcX, srcY;
  int srcType;
  const void* srcHost;
  DevicePtr srcDevice;
  Array srcArray;
  size_t srcPitch;
  size_t dstX, dstY;
  int dstType;
  void* dstHost;
  DevicePtr dstDevice;
  Array dstArray;
  size_t dstPitch;
  size_t widthBytes, height;
};

struct Copy3D {
  size_t srcX, srcY, srcZ;
  size_t srcLod;
  int srcType;
  const void* srcHost;
  DevicePtr srcDevice;
  Array srcArray;
  void* reserved0;
  size_t srcPitch, srcHeight;
  size_t dstX, dstY, dstZ;
  size_t dstLod;
  int dstType;
  void* dstHost;
  DevicePtr dstDevice;
  Array dstArray;
  void* reserved1;
  size_t dstPitch, dstHeight;
  size_t widthBytes, height, depth;
};

// The transcription, checked at compile time: a missing, extra or narrower field
// would move everything after it, and the driver reads by offset.
static_assert(sizeof(ArrayDesc) == 24, "CUDA_ARRAY_DESCRIPTOR is 24 bytes");
static_assert(offsetof(ArrayDesc, format) == 16, "the format follows the extents");
static_assert(sizeof(ArrayDesc3D) == 40, "CUDA_ARRAY3D_DESCRIPTOR is 40 bytes");
static_assert(offsetof(ArrayDesc3D, flags) == 32, "the flags are the last field");
static_assert(sizeof(Copy2D) == 128, "CUDA_MEMCPY2D is 128 bytes");
static_assert(offsetof(Copy2D, srcPitch) == 48, "the pitch follows the source handle");
static_assert(offsetof(Copy2D, widthBytes) == 112, "the extent ends the struct");
static_assert(sizeof(Copy3D) == 200, "CUDA_MEMCPY3D is 200 bytes");
static_assert(offsetof(Copy3D, srcArray) == 56, "the source array is the third pointer");
static_assert(offsetof(Copy3D, depth) == 192, "the depth is the last field");

// The values the header gives the three enums this uses.
constexpr int kMemHost = 0x01;
constexpr int kMemArray = 0x03;
constexpr int kFmtHalf = 0x10;
constexpr int kFmtFloat = 0x20;
constexpr int kAddrWrap = 0;
constexpr int kAddrClamp = 1;
constexpr int kAddrBorder = 3;
constexpr int kFilterPoint = 0;
constexpr int kFilterLinear = 1;
constexpr unsigned kNormalized = 0x02;
constexpr unsigned kLayered = 0x01;
constexpr unsigned kCubemap = 0x04;

// Only the calls a launch needs. Each is looked up under its versioned name
// first: the driver exports `cuMemAlloc_v2` and friends, not the unsuffixed name.
struct Driver {
  void* lib = nullptr;
  Result (*init)(unsigned) = nullptr;
  Result (*dev_count)(int*) = nullptr;
  Result (*dev_get)(Device*, int) = nullptr;
  Result (*dev_name)(char*, int, Device) = nullptr;
  Result (*ctx_create)(Context*, unsigned, Device) = nullptr;
  Result (*ctx_destroy)(Context) = nullptr;
  Result (*mem_alloc)(DevicePtr*, size_t) = nullptr;
  Result (*mem_free)(DevicePtr) = nullptr;
  Result (*mem_up)(DevicePtr, const void*, size_t) = nullptr;
  Result (*mem_down)(void*, DevicePtr, size_t) = nullptr;
  Result (*mod_load)(Module*, const void*) = nullptr;
  Result (*mod_unload)(Module) = nullptr;
  Result (*fn_get)(Function*, Module, const char*) = nullptr;
  Result (*fn_attr)(int*, int, Function) = nullptr;
  Result (*launch)(Function, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                   unsigned, void*, void**, void**) = nullptr;
  Result (*sync)() = nullptr;
  Result (*err_string)(Result, const char**) = nullptr;
  Result (*texref_get)(Texref*, Module, const char*) = nullptr;
  Result (*texref_array)(Texref, Array, unsigned) = nullptr;
  Result (*texref_filter)(Texref, int) = nullptr;
  Result (*texref_mode)(Texref, int, int) = nullptr;
  Result (*texref_flags)(Texref, unsigned) = nullptr;
  Result (*texref_border)(Texref, float*) = nullptr;
  Result (*array_create)(Array*, const ArrayDesc*) = nullptr;
  Result (*array3d_create)(Array*, const ArrayDesc3D*) = nullptr;
  Result (*array_destroy)(Array) = nullptr;
  Result (*copy2d)(const Copy2D*) = nullptr;
  Result (*copy3d)(const Copy3D*) = nullptr;
};

inline void* open_library() {
#if defined(_WIN32)
  return (void*)LoadLibraryA("nvcuda.dll");
#else
  return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

template <class T>
inline void pick(void* lib, const char* name, const char* legacy, T& slot) {
#if defined(_WIN32)
  slot = (T)(void*)GetProcAddress((HMODULE)lib, name);
  if (!slot && legacy) slot = (T)(void*)GetProcAddress((HMODULE)lib, legacy);
#else
  slot = (T)(void*)dlsym(lib, name);
  if (!slot && legacy) slot = (T)(void*)dlsym(lib, legacy);
#endif
}

inline Driver* load(std::string& why) {
  static Driver d;
  static bool tried = false;
  if (!tried) {
    tried = true;
    d.lib = open_library();
    if (d.lib) {
      pick(d.lib, "cuInit", nullptr, d.init);
      pick(d.lib, "cuDeviceGetCount", nullptr, d.dev_count);
      pick(d.lib, "cuDeviceGet", nullptr, d.dev_get);
      pick(d.lib, "cuDeviceGetName", nullptr, d.dev_name);
      pick(d.lib, "cuCtxCreate_v2", "cuCtxCreate", d.ctx_create);
      pick(d.lib, "cuCtxDestroy_v2", "cuCtxDestroy", d.ctx_destroy);
      pick(d.lib, "cuMemAlloc_v2", "cuMemAlloc", d.mem_alloc);
      pick(d.lib, "cuMemFree_v2", "cuMemFree", d.mem_free);
      pick(d.lib, "cuMemcpyHtoD_v2", "cuMemcpyHtoD", d.mem_up);
      pick(d.lib, "cuMemcpyDtoH_v2", "cuMemcpyDtoH", d.mem_down);
      pick(d.lib, "cuModuleLoadData", nullptr, d.mod_load);
      pick(d.lib, "cuModuleUnload", nullptr, d.mod_unload);
      pick(d.lib, "cuModuleGetFunction", nullptr, d.fn_get);
      pick(d.lib, "cuFuncGetAttribute", nullptr, d.fn_attr);
      pick(d.lib, "cuLaunchKernel", nullptr, d.launch);
      pick(d.lib, "cuCtxSynchronize", nullptr, d.sync);
      pick(d.lib, "cuGetErrorString", nullptr, d.err_string);
      pick(d.lib, "cuModuleGetTexRef", nullptr, d.texref_get);
      pick(d.lib, "cuTexRefSetArray", nullptr, d.texref_array);
      pick(d.lib, "cuTexRefSetFilterMode", nullptr, d.texref_filter);
      pick(d.lib, "cuTexRefSetAddressMode", nullptr, d.texref_mode);
      pick(d.lib, "cuTexRefSetFlags", nullptr, d.texref_flags);
      pick(d.lib, "cuTexRefSetBorderColor", nullptr, d.texref_border);
      pick(d.lib, "cuArrayCreate_v2", "cuArrayCreate", d.array_create);
      pick(d.lib, "cuArray3DCreate_v2", "cuArray3DCreate", d.array3d_create);
      pick(d.lib, "cuArrayDestroy", nullptr, d.array_destroy);
      pick(d.lib, "cuMemcpy2D_v2", "cuMemcpy2D", d.copy2d);
      pick(d.lib, "cuMemcpy3D_v2", "cuMemcpy3D", d.copy3d);
    }
  }
  if (!d.lib) {
    why = "no CUDA driver (libcuda) on this host";
    return nullptr;
  }
  if (!d.init || !d.launch || !d.mod_load || !d.mem_alloc || !d.fn_get) {
    why = "the CUDA driver loaded but has no compute entry points";
    return nullptr;
  }
  return &d;
}

inline std::string text(Driver& d, Result r) {
  const char* s = nullptr;
  if (d.err_string && d.err_string(r, &s) == 0 && s) return s;
  return "cuda error " + std::to_string(r);
}

// Device memory, arrays and the module, released however the run leaves.
struct Owned {
  Driver* d;
  std::vector<DevicePtr> mem;
  std::vector<Array> arrays;
  Module mod = nullptr;
  explicit Owned(Driver* driver) : d(driver) {}
  ~Owned() {
    if (d->array_destroy)
      for (Array a : arrays)
        if (a) d->array_destroy(a);
    if (mod && d->mod_unload) d->mod_unload(mod);
    if (d->mem_free)
      for (DevicePtr p : mem)
        if (p) d->mem_free(p);
  }
};

}  // namespace detail

// One launch of a PTX kernel on the first CUDA device, timed the way the Metal
// column is: one warm launch, then `bench` of them, best and mean reported.
// `in` is the bytes of each buffer as the CPU path has them, `textures` are the
// bindings the kernel declares, and `out` is what the kernel left behind, so the
// two columns can be compared byte for byte.
inline Column run(const std::string& ptx, const std::string& kernel,
                  const std::vector<TranslatedKernel::Param>& params,
                  const std::vector<std::vector<unsigned char>>& in,
                  const std::vector<Texture>& textures,
                  const std::vector<std::pair<int, uint64_t>>& args, uint32_t grid,
                  uint32_t threads, uint32_t bench) {
  Column c;
  std::string why;
  detail::Driver* d = detail::load(why);
  if (!d) {
    c.why = why;
    return c;
  }

  detail::Result r = d->init(0);
  if (r) {
    c.why = "cuInit: " + detail::text(*d, r);
    return c;
  }
  int ndev = 0;
  if (d->dev_count) {
    r = d->dev_count(&ndev);
    if (r || ndev <= 0) {
      c.why = r ? "cuDeviceGetCount: " + detail::text(*d, r) : "no CUDA device";
      return c;
    }
  }
  detail::Device dev = 0;
  if (d->dev_get) {
    r = d->dev_get(&dev, 0);
    if (r) {
      c.why = "cuDeviceGet: " + detail::text(*d, r);
      return c;
    }
  }
  detail::Context ctx = nullptr;
  r = d->ctx_create(&ctx, 0, dev);
  if (r || !ctx) {
    c.why = "cuCtxCreate: " + detail::text(*d, r);
    return c;
  }
  struct Ctx {
    detail::Driver* d;
    detail::Context c;
    ~Ctx() {
      if (c && d->ctx_destroy) d->ctx_destroy(c);
    }
  } ctx_guard{d, ctx};

  char name[256] = {0};
  if (d->dev_name) d->dev_name(name, 255, dev);
  c.device = name;

  detail::Owned owned(d);
  r = d->mod_load(&owned.mod, (const void*)ptx.c_str());
  if (r || !owned.mod) {
    c.why = "the driver refused the module: " + detail::text(*d, r);
    return c;
  }
  detail::Function fn = nullptr;
  r = d->fn_get(&fn, owned.mod, kernel.c_str());
  if (r || !fn) {
    c.why = "the driver has no kernel '" + kernel + "'" +
            (r ? ": " + detail::text(*d, r) : "");
    return c;
  }
  // The block the kernel was compiled for is a property of the device, so this
  // is asked rather than assumed: a launch wider than it is refused by the
  // driver, and by then the module has been built for nothing.
  int max_threads = 1024;
  if (d->fn_attr) {
    int mt = 0;
    if (d->fn_attr(&mt, 0, fn) == 0 && mt > 0) max_threads = mt;
  }
  if (threads == 0 || threads > (uint32_t)max_threads) {
    c.why = "'" + kernel + "' takes at most " + std::to_string(max_threads) +
            " threads per block, the launch asked for " + std::to_string(threads);
    return c;
  }
  // A CUDA launch is whole blocks. A partial last block would run lanes past the
  // grid, which is a different launch from the one the CPU column ran.
  if (grid == 0 || grid % threads != 0) {
    c.why = "a CUDA launch is whole blocks: " + std::to_string(grid) +
            " lanes do not divide into blocks of " + std::to_string(threads);
    return c;
  }

  for (const std::vector<unsigned char>& b : in) {
    detail::DevicePtr at = 0;
    r = d->mem_alloc(&at, b.empty() ? 1 : b.size());
    if (r) {
      c.why = "no device memory for a " + std::to_string(b.size()) + " byte buffer: " +
              detail::text(*d, r);
      return c;
    }
    owned.mem.push_back(at);
  }

  // A texture is a CUDA array, one channel wide, plus the texture reference the
  // module declares by that name and the sampler state the command line chose.
  // The array format is the width of the texel the launcher uploaded, so a half
  // texture is read back as the halves it holds rather than widened first.
  // The fixtures declare textures with `.tex`, which PTX keeps for backward
  // compatibility and defines as equal to a module-scope `.global .texref` of the
  // same name, so the driver's reference lookup finds it.
  for (const Texture& t : textures) {
    if (is_surface(ptx, t.name)) {
      c.why = "'" + t.name + "' is a surface, and the cuda column binds sampled textures";
      return c;
    }
    if (!d->texref_get || !d->array_create) {
      c.why = "the driver has no texture reference entry points";
      return c;
    }
    const int format = t.texel == 2 ? detail::kFmtHalf : detail::kFmtFloat;
    const size_t depth = t.depth > 0 ? (size_t)t.depth : (t.cube ? 6u : (size_t)t.layers);
    detail::Array arr = nullptr;
    detail::Result made;
    if (t.one_d) {
      // A 1-D binding is an array with no height, which is what the GPU means
      // by `tex.1d`; the CPU runtime lowers the same thing to one row. The upload
      // is a 2-D copy one row high, which is how a 1-D array is addressed: the
      // array is a rectangle whose height is zero, and the copy covers row zero.
      const detail::ArrayDesc desc{t.w, 0, format, 1u};
      made = d->array_create(&arr, &desc);
    } else if (depth > 1) {
      const detail::ArrayDesc3D desc{t.w, t.h, depth, format, 1u,
                                     t.cube ? detail::kCubemap : detail::kLayered};
      made = d->array3d_create ? d->array3d_create(&arr, &desc) : -1;
    } else {
      const detail::ArrayDesc desc{t.w, t.h, format, 1u};
      made = d->array_create(&arr, &desc);
    }
    if (made || !arr) {
      c.why = "no device array for texture '" + t.name + "': " + detail::text(*d, made);
      return c;
    }
    owned.arrays.push_back(arr);

    const size_t row = (size_t)t.w * t.texel;
    detail::Result copied;
    if (t.one_d) {
      detail::Copy2D cp{};
      cp.srcType = detail::kMemHost;
      cp.srcHost = t.bytes.data();
      cp.srcPitch = row;
      cp.dstType = detail::kMemArray;
      cp.dstArray = arr;
      cp.widthBytes = row;
      cp.height = 1;
      copied = d->copy2d ? d->copy2d(&cp) : -1;
    } else {
      detail::Copy3D cp{};
      cp.srcType = detail::kMemHost;
      cp.srcHost = t.bytes.data();
      cp.srcPitch = row;
      cp.srcHeight = t.h;
      cp.dstType = detail::kMemArray;
      cp.dstArray = arr;
      cp.widthBytes = row;
      cp.height = t.h;
      cp.depth = depth;
      copied = d->copy3d ? d->copy3d(&cp) : -1;
    }
    if (copied) {
      c.why = "the copy into texture '" + t.name + "' failed: " + detail::text(*d, copied);
      return c;
    }

    detail::Texref ref = nullptr;
    r = d->texref_get(&ref, owned.mod, t.name.c_str());
    if (r || !ref) {
      c.why = "the module has no texture '" + t.name + "'" +
              (r ? ": " + detail::text(*d, r) : "");
      return c;
    }
    r = d->texref_array(ref, arr, 0);
    if (r) {
      c.why = "cannot bind texture '" + t.name + "': " + detail::text(*d, r);
      return c;
    }
    // The commands line's two choices, which are the CPU runtime's and Metal's:
    // the filter, and what a coordinate outside the image means.
    if (d->texref_flags) d->texref_flags(ref, detail::kNormalized);
    if (d->texref_filter)
      d->texref_filter(ref, t.filter == 1 ? detail::kFilterLinear : detail::kFilterPoint);
    if (d->texref_mode) {
      const int mode = t.addr == 2 ? detail::kAddrWrap
                       : t.addr == 1 ? detail::kAddrBorder
                                     : detail::kAddrClamp;
      for (int axis = 0; axis < 3; axis++) d->texref_mode(ref, axis, mode);
    }
    if (d->texref_border && t.addr == 1) {
      float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      d->texref_border(ref, zero);
    }
  }

  // One eight-byte cell per parameter and a pointer to it: the driver reads each
  // parameter's own width from the address it is handed, so the cells need no
  // packing and the signature's order is the only thing that matters.
  std::vector<unsigned long long> cells(params.size(), 0);
  std::vector<void*> argv(params.size());
  for (size_t i = 0; i < params.size(); i++) {
    argv[i] = &cells[i];
    const TranslatedKernel::Param& p = params[i];
    if (p.buf >= 0) {
      if ((size_t)p.buf >= owned.mem.size()) {
        c.why = "a parameter binds buffer " + std::to_string(p.buf) +
                ", which this launch did not make";
        return c;
      }
      cells[i] = owned.mem[(size_t)p.buf];
    } else if (p.scalar >= 0) {
      uint64_t raw = 0;
      for (const auto& a : args)
        if (a.first == p.scalar) raw = a.second;
      // The CPU module reads this slot as `(T)raw`, so the driver has to see that
      // same cast at the parameter's own width rather than the raw word.
      if (p.type == "f32") {
        const float v = (float)raw;
        std::memcpy(&cells[i], &v, 4);
      } else if (p.type == "f64") {
        const double v = (double)raw;
        std::memcpy(&cells[i], &v, 8);
      } else if (p.type == "f16") {
        c.why = "a half scalar parameter has no encoding here";
        return c;
      } else if (p.type == "u32" || p.type == "s32" || p.type == "b32") {
        const uint32_t v = (uint32_t)raw;
        std::memcpy(&cells[i], &v, 4);
      } else if (p.type == "u16" || p.type == "s16" || p.type == "b16") {
        const uint16_t v = (uint16_t)raw;
        std::memcpy(&cells[i], &v, 2);
      } else if (p.type == "u8" || p.type == "s8" || p.type == "b8") {
        const uint8_t v = (uint8_t)raw;
        std::memcpy(&cells[i], &v, 1);
      } else if (p.type == "u64" || p.type == "s64" || p.type == "b64") {
        cells[i] = raw;
      } else {
        c.why = "a parameter of type '" + p.type + "' has no encoding here";
        return c;
      }
    }
    // A pointer the body never reads is passed as zero: nothing dereferences it.
  }

  // A dynamic shared array is sized by the launch, and the CPU runtime hands out
  // 48 KiB, so the GPU is given the same window.
  const unsigned dyn_shared =
      ptx.find("extern .shared") != std::string::npos ? 48u * 1024u : 0u;
  const unsigned gx = grid / threads;

  auto now = [] {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
  };
  double total = 0.0, best = -1.0;
  for (uint32_t it = 0; it <= bench; it++) {
    for (size_t b = 0; b < in.size(); b++) {
      if (in[b].empty()) continue;
      r = d->mem_up(owned.mem[b], in[b].data(), in[b].size());
      if (r) {
        c.why = "the input copy failed: " + detail::text(*d, r);
        return c;
      }
    }
    const double t0 = now();
    r = d->launch(fn, gx, 1, 1, threads, 1, 1, dyn_shared, nullptr, argv.data(), nullptr);
    if (r) {
      c.why = "the launch failed: " + detail::text(*d, r);
      return c;
    }
    r = d->sync ? d->sync() : 0;
    const double dt = now() - t0;
    if (r) {
      c.why = "the launch did not complete: " + detail::text(*d, r);
      return c;
    }
    if (it > 0) {
      total += dt;
      if (best < 0 || dt < best) best = dt;
    }
  }
  c.avg_ms = bench ? total / (double)bench : 0.0;
  c.best_ms = best;

  c.out.resize(in.size());
  for (size_t b = 0; b < in.size(); b++) {
    c.out[b].resize(in[b].size());
    if (in[b].empty()) continue;
    r = d->mem_down(c.out[b].data(), owned.mem[b], in[b].size());
    if (r) {
      c.why = "the output copy failed: " + detail::text(*d, r);
      return c;
    }
  }
  c.ok = true;
  return c;
}

}  // namespace weft_cuda
