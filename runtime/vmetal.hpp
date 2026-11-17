#pragma once

#include <weft/weft.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace vmetal {

// The entry points a translated module exports, the ones weft.h declares.
struct ModuleApi {
  int (*kernel_count)() = nullptr;
  const weft_kernel_info* (*kernel_at)(int) = nullptr;
  void (*run)(int, void**, uint32_t, uint32_t) = nullptr;
  void (*set_param)(int, uint64_t) = nullptr;
  void (*set_texture)(int, void*, uint32_t, uint32_t, uint32_t) = nullptr;
  void (*set_texture_layers)(int, void*, uint32_t, uint32_t, uint32_t, uint32_t) = nullptr;
  void (*set_sampler)(int, int, int) = nullptr;
};

inline const char* device_name() { return "weft virtual Metal"; }

inline constexpr size_t kTailGuard = 65536;
inline constexpr unsigned char kTailByte = 0xA5;

struct BufferState {
  std::vector<unsigned char> bytes;
  size_t guard = 0;               // the pad past the binding, filled with kTailByte
  unsigned char* view = nullptr;  // storage the caller owns, when set
  size_t view_len = 0;
};

class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::shared_ptr<BufferState> s) : s_(std::move(s)) {}
  bool valid() const { return s_ != nullptr; }
  size_t length() const {
    return s_ ? (s_->view ? s_->view_len : s_->bytes.size() - s_->guard) : 0;
  }
  // The pad past the binding, which is what the device checks.
  size_t guard() const { return s_ ? s_->guard : 0; }
  unsigned char* contents() { return s_ ? (s_->view ? s_->view : s_->bytes.data()) : nullptr; }
  const unsigned char* contents() const {
    return s_ ? (s_->view ? s_->view : s_->bytes.data()) : nullptr;
  }

 private:
  std::shared_ptr<BufferState> s_;
};

struct TextureState {
  std::vector<unsigned char> bytes;
  uint32_t width = 0, height = 0, layers = 1;
  uint32_t elem = 4;  // bytes per texel
};

class Texture {
 public:
  Texture() = default;
  explicit Texture(std::shared_ptr<TextureState> s) : s_(std::move(s)) {}
  bool valid() const { return s_ != nullptr; }
  uint32_t width() const { return s_ ? s_->width : 0; }
  uint32_t height() const { return s_ ? s_->height : 0; }
  uint32_t layers() const { return s_ ? s_->layers : 0; }
  uint32_t elem() const { return s_ ? s_->elem : 0; }
  size_t byte_length() const {
    return s_ ? (size_t)s_->width * s_->height * s_->layers * s_->elem : 0;
  }
  unsigned char* contents() { return s_ ? s_->bytes.data() : nullptr; }
  const unsigned char* contents() const { return s_ ? s_->bytes.data() : nullptr; }

 private:
  std::shared_ptr<TextureState> s_;
};

struct PipelineState {
  int index = -1;
  std::string name;
  bool valid() const { return index >= 0; }
};

struct DeviceState {
  ModuleApi api;
  std::string name;
#if !defined(_WIN32)
  void* handle = nullptr;
#endif
  ~DeviceState() {
#if !defined(_WIN32)
    if (handle) dlclose(handle);
#endif
  }
};

namespace detail {

inline std::string overrun_past(const std::vector<Buffer>& bufs, const weft_kernel_info* k) {
  for (size_t i = 0; i < bufs.size(); i++) {
    const size_t pad = bufs[i].guard();
    if (!pad) continue;
    const unsigned char* tail = bufs[i].contents() + bufs[i].length();
    size_t past = 0;
    for (size_t j = 0; j < pad; j++)
      if (tail[j] != kTailByte) past = j + 1;
    if (!past) continue;
    std::string s = "the kernel wrote ";
    s += past == 1 ? std::string("1 byte") : std::to_string(past) + " bytes";
    s += " past the end of buffer " + std::to_string(i) + " (" +
         std::to_string(bufs[i].length()) + " bytes bound";
    if (k && k->name) s += ", '" + std::string(k->name) + "'";
    return s + "), which the GPU leaves undefined";
  }
  return std::string();
}

// The kernel an index names, or an invalid pipeline and the reason.
inline PipelineState find_kernel(const std::shared_ptr<DeviceState>& d, const char* name,
                                 std::string* err) {
  PipelineState p;
  if (!d || !d->api.kernel_count || !d->api.kernel_at) {
    if (err) *err = "the device has no module";
    return p;
  }
  const int n = d->api.kernel_count();
  for (int i = 0; i < n; i++) {
    const weft_kernel_info* k = d->api.kernel_at(i);
    if (k && k->name && name && std::string(k->name) == name) {
      p.index = i;
      p.name = k->name;
      return p;
    }
  }
  if (err) *err = std::string("the module has no kernel named '") + (name ? name : "") + "'";
  return p;
}

// A module already on disk: dlopen it and fill in the entry points.
inline bool load_module(const std::string& path, std::shared_ptr<DeviceState>* out,
                        std::string* err) {
#if defined(_WIN32)
  (void)path;
  (void)out;
  if (err) *err = "loading a module by path is not wired up on Windows";
  return false;
#else
  void* h = dlopen(path.c_str(), RTLD_NOW);
  if (!h) {
    const char* why = dlerror();
    if (err) *err = why ? why : "cannot open the module";
    return false;
  }
  ModuleApi api;
  api.kernel_count = (int (*)())dlsym(h, "weft_kernel_count");
  api.kernel_at = (const weft_kernel_info* (*)(int))dlsym(h, "weft_kernel_at");
  api.run = (void (*)(int, void**, uint32_t, uint32_t))dlsym(h, "weft_run");
  api.set_param = (void (*)(int, uint64_t))dlsym(h, "weft_set_param");
  api.set_texture = (void (*)(int, void*, uint32_t, uint32_t, uint32_t))dlsym(h, "weft_set_texture");
  api.set_texture_layers =
      (void (*)(int, void*, uint32_t, uint32_t, uint32_t, uint32_t))
          dlsym(h, "weft_set_texture_layers");
  api.set_sampler = (void (*)(int, int, int))dlsym(h, "weft_set_sampler");
  if (!api.kernel_count || !api.run) {
    dlclose(h);
    if (err) *err = "the module does not export the weft entry points";
    return false;
  }
  auto st = std::make_shared<DeviceState>();
  st->api = api;
  st->name = device_name();
  st->handle = h;
  *out = std::move(st);
  return true;
#endif
}

#if !defined(_WIN32)
inline std::string env(const char* key) {
  const char* v = std::getenv(key);
  return v ? v : "";
}

inline bool exists(const std::string& p) {
  std::error_code ec;
  return !p.empty() && std::filesystem::exists(p, ec);
}

// One argument, quoted so a path with a space survives the shell.
inline std::string quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

// Run a command line and collect what it said.
inline bool run(const std::string& cmd, std::string* log) {
  FILE* p = popen((cmd + " 2>&1").c_str(), "r");
  if (!p) return false;
  char buf[512];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, p)) > 0)
    if (log) log->append(buf, n);
  return pclose(p) == 0;
}

inline std::string temp_path(const char* suffix) {
  static unsigned counter = 0;
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  std::error_code ec;
  std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
  if (ec) dir = ".";
  return (dir / ("weft-vmetal-" + std::to_string(stamp) + "-" + std::to_string(counter++) + suffix))
      .string();
}
#endif

}  // namespace detail

enum class SourceKind { Auto, Metal, PTX, SPIRV };

struct LibraryOptions {
  SourceKind kind = SourceKind::Auto;
  std::vector<std::pair<int, uint32_t>> widths;
};

namespace detail {

inline bool ends_with(const std::string& s, const char* tail) {
  const size_t n = std::strlen(tail);
  return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

inline SourceKind sniff(const std::string& src) {
  static const unsigned char kSpirv[4] = {0x03, 0x02, 0x23, 0x07};
  if (src.size() >= 4 && std::memcmp(src.data(), kSpirv, 4) == 0) return SourceKind::SPIRV;
  size_t at = 0;
  for (int line = 0; line < 400 && at < src.size(); line++) {
    size_t end = src.find('\n', at);
    if (end == std::string::npos) end = src.size();
    size_t b = at;
    while (b < end && (src[b] == ' ' || src[b] == '\t' || src[b] == '\r')) b++;
    const std::string head = src.substr(b, std::min<size_t>(end - b, 12));
    if (head.rfind(".version", 0) == 0 || head.rfind(".target", 0) == 0 ||
        head.rfind(".visible", 0) == 0 || head.rfind(".entry", 0) == 0 ||
        head.rfind(".address_siz", 0) == 0)
      return SourceKind::PTX;
    at = end + 1;
  }
  return SourceKind::Metal;
}

inline SourceKind kind_for_path(const std::string& path) {
  if (ends_with(path, ".spv")) return SourceKind::SPIRV;
  if (ends_with(path, ".ptx")) return SourceKind::PTX;
  return SourceKind::Metal;
}

}  // namespace detail

class BlitEncoder;
class Encoder;
class Library;
class Device;

class CommandBuffer {
 public:
  CommandBuffer() = default;
  explicit CommandBuffer(std::shared_ptr<DeviceState> d) : d_(std::move(d)) {}

  Encoder compute_encoder();
  BlitEncoder blit_encoder();

  void set_pipeline(PipelineState p) { kernel_ = p.index; }
  void commit() {}
  void wait_until_completed();

  bool ok() const { return error_.empty(); }
  const std::string& error() const { return error_; }
  double elapsed_ms() const { return elapsed_ms_; }
  // How many compute dispatches were encoded, not counting blits.
  size_t dispatch_count() const;

  void bind_buffer(int index, Buffer b, size_t off);
  void bind_texture(int index, Texture t);
  void set_sampler(int index, int filter, int address);

  void set_launch(uint32_t grid, uint32_t threads);
  void set_launch_indirect(Buffer args, size_t offset, uint32_t threads);

  // Blit operations.
  void fill_buffer(Buffer b, size_t offset, size_t bytes, unsigned char value);
  void copy_buffer(Buffer src, size_t src_off, Buffer dst, size_t dst_off, size_t bytes);
  void copy_texture(Texture src, Texture dst);
  void copy_buffer_to_texture(Buffer src, size_t off, Texture dst);
  void copy_texture_to_buffer(Texture src, Buffer dst, size_t off);

 private:
  struct Sampler {
    int index = 0, filter = 0, address = 0;
  };
  struct Command {
    enum Kind { Dispatch, Fill, CopyBuf, CopyTex, BufToTex, TexToBuf } kind = Dispatch;
    int kernel = -1;
    uint32_t grid = 0, threads = 0;
    bool indirect = false;
    Buffer arg, src, dst;
    Texture stex, dtex;
    size_t arg_off = 0, soff = 0, doff = 0, bytes = 0;
    unsigned char fill = 0;
  };

  std::shared_ptr<DeviceState> d_;
  std::vector<Buffer> bufs_;
  std::vector<size_t> offs_;
  std::vector<Texture> texs_;
  std::vector<Command> commands_;
  std::vector<Sampler> samplers_;
  int kernel_ = -1;
  bool ran_ = false;  // a wait is a wait; it does not run the buffer twice
  std::string error_;
  double elapsed_ms_ = 0.0;
};

class BlitEncoder {
 public:
  explicit BlitEncoder(CommandBuffer* cb) : cb_(cb) {}
  void fill_buffer(Buffer b, size_t offset, size_t bytes, unsigned char value) {
    if (cb_) cb_->fill_buffer(std::move(b), offset, bytes, value);
  }
  void copy_buffer(Buffer src, size_t src_off, Buffer dst, size_t dst_off, size_t bytes) {
    if (cb_) cb_->copy_buffer(std::move(src), src_off, std::move(dst), dst_off, bytes);
  }
  void copy_texture(Texture src, Texture dst) {
    if (cb_) cb_->copy_texture(std::move(src), std::move(dst));
  }
  void copy_buffer_to_texture(Buffer src, size_t off, Texture dst) {
    if (cb_) cb_->copy_buffer_to_texture(std::move(src), off, std::move(dst));
  }
  void copy_texture_to_buffer(Texture src, Buffer dst, size_t off) {
    if (cb_) cb_->copy_texture_to_buffer(std::move(src), std::move(dst), off);
  }
  void end_encoding() {}

 private:
  CommandBuffer* cb_;
};

class Encoder {
 public:
  explicit Encoder(CommandBuffer* cb) : cb_(cb) {}
  void set_pipeline(PipelineState p) {
    if (cb_) cb_->set_pipeline(p);
  }
  void set_buffer(int index, Buffer b, size_t offset = 0) {
    if (cb_) cb_->bind_buffer(index, std::move(b), offset);
  }
  void set_texture(int index, Texture t) {
    if (cb_) cb_->bind_texture(index, std::move(t));
  }
  void set_sampler(int index, int filter, int address) {
    if (cb_) cb_->set_sampler(index, filter, address);
  }
  // Each call encodes one dispatch.
  void dispatch_threads(uint32_t grid, uint32_t threads) {
    if (cb_) cb_->set_launch(grid, threads);
  }
  void dispatch_threadgroups(uint32_t groups, uint32_t threads) {
    if (cb_) cb_->set_launch(groups * threads, threads);
  }
  void dispatch_threadgroups_indirect(Buffer args, size_t offset, uint32_t threads) {
    if (cb_) cb_->set_launch_indirect(std::move(args), offset, threads);
  }
  void fill_buffer(Buffer b, size_t offset, size_t bytes, unsigned char value) {
    if (cb_) cb_->fill_buffer(std::move(b), offset, bytes, value);
  }
  void copy_buffer(Buffer src, size_t src_off, Buffer dst, size_t dst_off, size_t bytes) {
    if (cb_) cb_->copy_buffer(std::move(src), src_off, std::move(dst), dst_off, bytes);
  }
  void end_encoding() {}

 private:
  CommandBuffer* cb_;
};

inline Encoder CommandBuffer::compute_encoder() { return Encoder(this); }
inline BlitEncoder CommandBuffer::blit_encoder() { return BlitEncoder(this); }

inline size_t CommandBuffer::dispatch_count() const {
  size_t n = 0;
  for (const Command& c : commands_)
    if (c.kind == Command::Dispatch) n++;
  return n;
}

inline void CommandBuffer::bind_buffer(int index, Buffer b, size_t off) {
  if (index < 0) return;
  const size_t n = (size_t)index + 1;
  if (bufs_.size() < n) {
    bufs_.resize(n);
    offs_.resize(n, 0);
  }
  bufs_[(size_t)index] = std::move(b);
  offs_[(size_t)index] = off;
}

inline void CommandBuffer::bind_texture(int index, Texture t) {
  if (index < 0) return;
  const size_t n = (size_t)index + 1;
  if (texs_.size() < n) texs_.resize(n);
  texs_[(size_t)index] = std::move(t);
}

inline void CommandBuffer::set_sampler(int index, int filter, int address) {
  if (index < 0) return;
  for (Sampler& s : samplers_) {
    if (s.index == index) {
      s.filter = filter;
      s.address = address;
      return;
    }
  }
  samplers_.push_back(Sampler{index, filter, address});
}

inline void CommandBuffer::set_launch(uint32_t grid, uint32_t threads) {
  Command c;
  c.kind = Command::Dispatch;
  c.kernel = kernel_;
  c.grid = grid;
  c.threads = threads;
  commands_.push_back(std::move(c));
}

inline void CommandBuffer::set_launch_indirect(Buffer args, size_t offset, uint32_t threads) {
  Command c;
  c.kind = Command::Dispatch;
  c.kernel = kernel_;
  c.threads = threads;
  c.indirect = true;
  c.arg = std::move(args);
  c.arg_off = offset;
  commands_.push_back(std::move(c));
}

inline void CommandBuffer::fill_buffer(Buffer b, size_t offset, size_t bytes,
                                       unsigned char value) {
  Command c;
  c.kind = Command::Fill;
  c.dst = std::move(b);
  c.doff = offset;
  c.bytes = bytes;
  c.fill = value;
  commands_.push_back(std::move(c));
}

inline void CommandBuffer::copy_buffer(Buffer src, size_t src_off, Buffer dst, size_t dst_off,
                                       size_t bytes) {
  Command c;
  c.kind = Command::CopyBuf;
  c.src = std::move(src);
  c.dst = std::move(dst);
  c.soff = src_off;
  c.doff = dst_off;
  c.bytes = bytes;
  commands_.push_back(std::move(c));
}

inline void CommandBuffer::copy_texture(Texture src, Texture dst) {
  Command c;
  c.kind = Command::CopyTex;
  c.stex = std::move(src);
  c.dtex = std::move(dst);
  commands_.push_back(std::move(c));
}

inline void CommandBuffer::copy_buffer_to_texture(Buffer src, size_t off, Texture dst) {
  Command c;
  c.kind = Command::BufToTex;
  c.src = std::move(src);
  c.dtex = std::move(dst);
  c.soff = off;
  commands_.push_back(std::move(c));
}

inline void CommandBuffer::copy_texture_to_buffer(Texture src, Buffer dst, size_t off) {
  Command c;
  c.kind = Command::TexToBuf;
  c.stex = std::move(src);
  c.dst = std::move(dst);
  c.doff = off;
  commands_.push_back(std::move(c));
}

inline void CommandBuffer::wait_until_completed() {
  if (ran_) return;  // the commands ran once; a second wait just waits
  ran_ = true;
  error_.clear();
  if (!d_) {
    error_ = "the command buffer has no module";
    return;
  }
  if (commands_.empty()) {
    error_ = "the command buffer has nothing encoded";
    return;
  }
  const int nk = d_->api.kernel_count ? d_->api.kernel_count() : 0;
  size_t need_bufs = 0;
  bool any_dispatch = false;
  for (const Command& c : commands_) {
    if (c.kind != Command::Dispatch) continue;
    any_dispatch = true;
    if (c.kernel < 0) {
      error_ = "a dispatch was encoded with no pipeline";
      return;
    }
    if (c.kernel >= nk) {
      error_ = "a dispatch names a kernel the module does not have";
      return;
    }
    if (c.threads == 0) {
      error_ = "a dispatch has no threadgroup size";
      return;
    }
    if (!c.indirect && c.grid == 0) {
      error_ = "a dispatch has no geometry";
      return;
    }
    const weft_kernel_info* k = d_->api.kernel_at ? d_->api.kernel_at(c.kernel) : nullptr;
    if (k && k->nbufs > 0 && (size_t)k->nbufs > need_bufs) need_bufs = (size_t)k->nbufs;
  }
  if (any_dispatch && !d_->api.run) {
    error_ = "the command buffer has no module to run a dispatch on";
    return;
  }
  if (bufs_.size() < need_bufs) {
    error_ = "buffer slot " + std::to_string(bufs_.size()) + " was not bound; the kernel reads " +
             std::to_string(need_bufs);
    return;
  }
  std::vector<void*> ptrs(bufs_.size(), nullptr);
  for (size_t i = 0; i < bufs_.size(); i++) {
    if (!bufs_[i].valid()) {
      error_ = "buffer slot " + std::to_string(i) + " was not bound";
      return;
    }
    if (offs_[i] > bufs_[i].length()) {
      error_ = "buffer slot " + std::to_string(i) + " was bound at offset " +
               std::to_string(offs_[i]) + " of " + std::to_string(bufs_[i].length()) + " bytes";
      return;
    }
    unsigned char* p = bufs_[i].contents();
    ptrs[i] = p ? p + offs_[i] : nullptr;
  }
  if (!samplers_.empty()) {
    if (!d_->api.set_sampler) {
      error_ = "a sampler was set and the module does not export weft_set_sampler";
      return;
    }
    for (const Sampler& s : samplers_)
      d_->api.set_sampler(s.index, s.filter, s.address);
  }
  for (size_t t = 0; t < texs_.size(); t++) {
    if (!texs_[t].valid()) continue;
    const uint32_t elem = texs_[t].elem() ? texs_[t].elem() : 4u;
    if (d_->api.set_texture_layers)
      d_->api.set_texture_layers((int)t, texs_[t].contents(), texs_[t].width(), texs_[t].height(),
                                 elem, texs_[t].layers());
    else if (d_->api.set_texture)
      d_->api.set_texture((int)t, texs_[t].contents(), texs_[t].width(), texs_[t].height(), elem);
  }
  elapsed_ms_ = 0.0;
  for (const Command& c : commands_) {
    switch (c.kind) {
      case Command::Fill: {
        if (!c.dst.valid()) {
          error_ = "a fill names a buffer that was not bound";
          return;
        }
        if (c.doff > c.dst.length() || c.bytes > c.dst.length() - c.doff) {
          error_ = "a fill covers bytes past the end of its buffer";
          return;
        }
        Buffer dst = c.dst;  // a handle copy: a const command still writes through it
        if (c.bytes) std::memset(dst.contents() + c.doff, c.fill, c.bytes);
        break;
      }
      case Command::CopyBuf: {
        if (!c.src.valid() || !c.dst.valid()) {
          error_ = "a buffer copy names a buffer that was not bound";
          return;
        }
        if (c.soff > c.src.length() || c.bytes > c.src.length() - c.soff) {
          error_ = "a buffer copy reads past the end of its source";
          return;
        }
        if (c.doff > c.dst.length() || c.bytes > c.dst.length() - c.doff) {
          error_ = "a buffer copy writes past the end of its destination";
          return;
        }
        Buffer dst = c.dst;
        if (c.bytes) std::memmove(dst.contents() + c.doff, c.src.contents() + c.soff, c.bytes);
        break;
      }
      case Command::CopyTex: {
        if (!c.stex.valid() || !c.dtex.valid()) {
          error_ = "a texture copy names a texture that was not bound";
          return;
        }
        if (c.stex.width() != c.dtex.width() || c.stex.height() != c.dtex.height() ||
            c.stex.layers() != c.dtex.layers() || c.stex.elem() != c.dtex.elem()) {
          error_ = "a texture copy needs two textures of the same shape and texel";
          return;
        }
        Texture dst = c.dtex;
        if (c.stex.byte_length())
          std::memcpy(dst.contents(), c.stex.contents(), c.stex.byte_length());
        break;
      }
      case Command::BufToTex: {
        if (!c.src.valid() || !c.dtex.valid()) {
          error_ = "a buffer to texture copy names storage that was not bound";
          return;
        }
        const size_t n = c.dtex.byte_length();
        if (c.soff > c.src.length() || n > c.src.length() - c.soff) {
          error_ = "a buffer to texture copy reads past the end of its source";
          return;
        }
        Texture dst = c.dtex;
        if (n) std::memcpy(dst.contents(), c.src.contents() + c.soff, n);
        break;
      }
      case Command::TexToBuf: {
        if (!c.stex.valid() || !c.dst.valid()) {
          error_ = "a texture to buffer copy names storage that was not bound";
          return;
        }
        const size_t n = c.stex.byte_length();
        if (c.doff > c.dst.length() || n > c.dst.length() - c.doff) {
          error_ = "a texture to buffer copy writes past the end of its destination";
          return;
        }
        Buffer dst = c.dst;
        if (n) std::memcpy(dst.contents() + c.doff, c.stex.contents(), n);
        break;
      }
      case Command::Dispatch: {
        uint32_t grid = c.grid;
        if (c.indirect) {
          if (!c.arg.valid()) {
            error_ = "an indirect dispatch has no group-count buffer";
            return;
          }
          if (c.arg_off + 12 > c.arg.length()) {
            error_ = "an indirect dispatch reads past its group-count buffer";
            return;
          }
          uint32_t groups[3] = {0, 0, 0};
          std::memcpy(groups, c.arg.contents() + c.arg_off, sizeof groups);
          if (groups[1] != 1 || groups[2] != 1) {
            error_ = "indirect groups in y or z are not a grid this engine flattens";
            return;
          }
          grid = groups[0] * c.threads;
          if (grid == 0) {
            error_ = "an indirect dispatch asked for no threads";
            return;
          }
        }
        const auto t0 = std::chrono::steady_clock::now();
        d_->api.run(c.kernel, ptrs.data(), grid, c.threads);
        if (const std::string over = detail::overrun_past(
                bufs_, d_->api.kernel_at ? d_->api.kernel_at(c.kernel) : nullptr);
            !over.empty()) {
          error_ = over;
          return;
        }
        const auto t1 = std::chrono::steady_clock::now();
        elapsed_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
        break;
      }
    }
  }
}

class CommandQueue {
 public:
  CommandQueue() = default;
  explicit CommandQueue(std::shared_ptr<DeviceState> d) : d_(std::move(d)) {}
  CommandBuffer command_buffer() { return CommandBuffer(d_); }

 private:
  std::shared_ptr<DeviceState> d_;
};

class Device {
 public:
  Device() = default;

  // A module already loaded: the caller owns the handle.
  static Device adopt(ModuleApi api, const char* name = device_name()) {
    Device d;
    d.s_ = std::make_shared<DeviceState>();
    d.s_->api = api;
    d.s_->name = name ? name : device_name();
    return d;
  }

  // A module built by `weft-gen` and this host's compiler, loaded from its path.
  static Device open(const std::string& path, std::string* err = nullptr) {
    if (err) err->clear();
    std::shared_ptr<DeviceState> st;
    if (!detail::load_module(path, &st, err)) return Device();
    Device d;
    d.s_ = std::move(st);
    return d;
  }

  static Library new_library(const std::string& source, std::string* err = nullptr);
  static Library new_library(const std::string& source, SourceKind kind,
                             std::string* err = nullptr);
  // Widths for the slots a PTX or SPIR-V source cannot name.
  static Library new_library(const std::string& source, const LibraryOptions& opts,
                             std::string* err = nullptr);
  // The path's suffix says which front end: `.ptx`, `.spv`, anything else Metal.
  static Library new_library_from_file(const std::string& path, std::string* err = nullptr);
  static Library new_library_from_file(const std::string& path, const LibraryOptions& opts,
                                       std::string* err = nullptr);

  bool valid() const { return s_ != nullptr; }
  bool is_virtual() const { return true; }
  const char* name() const { return s_ ? s_->name.c_str() : ""; }

  Buffer new_buffer(size_t bytes, const void* init = nullptr) {
    if (!s_) return Buffer();
    auto st = std::make_shared<BufferState>();
    st->guard = kTailGuard;
    st->bytes.assign(bytes + kTailGuard, kTailByte);
    if (init && bytes) std::memcpy(st->bytes.data(), init, bytes);
    return Buffer(std::move(st));
  }

  static Buffer wrap(void* data, size_t bytes) {
    auto st = std::make_shared<BufferState>();
    st->view = (unsigned char*)data;
    st->view_len = bytes;
    return Buffer(std::move(st));
  }

  Texture new_texture(uint32_t w, uint32_t h, uint32_t layers, uint32_t elem,
                      const void* init = nullptr) {
    if (!s_) return Texture();
    auto st = std::make_shared<TextureState>();
    st->width = w;
    st->height = h;
    st->layers = layers ? layers : 1u;
    st->elem = elem;
    st->bytes.resize((size_t)w * h * st->layers * elem, 0);
    if (init && !st->bytes.empty()) std::memcpy(st->bytes.data(), init, st->bytes.size());
    return Texture(std::move(st));
  }

  // What a C binding asks about the module without holding a pipeline.
  int kernel_count() const { return s_ && s_->api.kernel_count ? s_->api.kernel_count() : 0; }
  const weft_kernel_info* kernel_info(int index) const {
    if (!s_ || !s_->api.kernel_at || index < 0 || index >= kernel_count()) return nullptr;
    return s_->api.kernel_at(index);
  }

  PipelineState new_pipeline(const char* kernel, std::string* err = nullptr) {
    if (err) err->clear();
    return detail::find_kernel(s_, kernel, err);
  }

  CommandQueue new_queue() { return CommandQueue(s_); }

 private:
  friend class Library;
  std::shared_ptr<DeviceState> s_;
};

// A module compiled from source, or opened from disk.
class Library {
 public:
  Library() = default;
  bool valid() const { return s_ != nullptr; }
  int kernel_count() const { return s_ && s_->api.kernel_count ? s_->api.kernel_count() : 0; }
  const weft_kernel_info* kernel_info(int index) const {
    if (!s_ || !s_->api.kernel_at || index < 0 || index >= kernel_count()) return nullptr;
    return s_->api.kernel_at(index);
  }
  PipelineState new_pipeline(const char* name, std::string* err = nullptr) const {
    if (err) err->clear();
    return detail::find_kernel(s_, name, err);
  }
  Device device() const {
    Device d;
    d.s_ = s_;
    return d;
  }
  // The shared object the compiler produced.
  const std::string& module_path() const { return path_; }

 private:
  friend class Device;
  std::shared_ptr<DeviceState> s_;
  std::string path_;
};

inline Library Device::new_library_from_file(const std::string& path, std::string* err) {
  return new_library_from_file(path, LibraryOptions{}, err);
}

inline Library Device::new_library_from_file(const std::string& path, const LibraryOptions& opts,
                                             std::string* err) {
  if (err) err->clear();
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    if (err) *err = "cannot read " + path;
    return Library();
  }
  std::string src;
  char buf[8192];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) src.append(buf, n);
  std::fclose(f);
  LibraryOptions o = opts;
  if (o.kind == SourceKind::Auto) o.kind = detail::kind_for_path(path);
  return new_library(src, o, err);
}

inline Library Device::new_library(const std::string& source, std::string* err) {
  return new_library(source, LibraryOptions{}, err);
}

inline Library Device::new_library(const std::string& source, SourceKind kind,
                                   std::string* err) {
  LibraryOptions opts;
  opts.kind = kind;
  return new_library(source, opts, err);
}

inline Library Device::new_library(const std::string& source, const LibraryOptions& opts,
                                   std::string* err) {
  if (err) err->clear();
  SourceKind kind = opts.kind;
#if defined(_WIN32)
  if (err) *err = "compiling source at run time is not wired up on Windows";
  return Library();
#else
  std::string root = detail::env("WEFT_ROOT");
  if (root.empty()) root = ".";
  std::string gen = detail::env("WEFT_GEN");
  if (gen.empty()) {
    const std::string in_tree = root + "/build/weft-gen";
    if (detail::exists(in_tree)) {
      gen = in_tree;
    } else if (detail::exists("./build/weft-gen")) {
      gen = "./build/weft-gen";
    } else {
      gen = "weft-gen";  // on PATH, if a program installed it there
    }
  }
  std::string cxx = detail::env("WEFT_CXX");
  if (cxx.empty()) {
    for (const char* c : {"/usr/bin/clang++", "/usr/bin/g++", "/usr/bin/c++"})
      if (detail::exists(c)) { cxx = c; break; }
    if (cxx.empty()) cxx = "c++";
  }
  const std::string inc = root + "/include";
  const std::string rt = root + "/runtime";
  if (!detail::exists(inc) || !detail::exists(rt)) {
    if (err)
      *err = "run-time compilation needs the weft tree: set WEFT_ROOT to the directory that has "
             "include/ and runtime/ (tried '" + root + "')";
    return Library();
  }

  if (kind == SourceKind::Auto) kind = detail::sniff(source);
  const char* ext = kind == SourceKind::PTX     ? ".ptx"
                    : kind == SourceKind::SPIRV ? ".spv"
                                                : ".metal";
  const std::string metal = detail::temp_path(ext);
  const std::string cpp = detail::temp_path(".cpp");
  const std::string lib = detail::temp_path(
#if defined(__APPLE__)
      ".dylib"
#else
      ".so"
#endif
  );
  auto drop = [&](bool all) {
    std::error_code ec;
    std::filesystem::remove(metal, ec);
    if (all) {
      std::filesystem::remove(cpp, ec);
      std::filesystem::remove(lib, ec);
    }
  };

  {
    FILE* f = std::fopen(metal.c_str(), "wb");
    if (!f) {
      if (err) *err = "cannot write the source to " + metal;
      return Library();
    }
    std::fwrite(source.data(), 1, source.size(), f);
    std::fclose(f);
  }

  std::string stated;
  for (const auto& w : opts.widths)
    stated += " --buffer-bytes " +
              detail::quote(std::to_string(w.first) + "=" + std::to_string(w.second));
  std::string log;
  if (!detail::run(
          detail::quote(gen) + " " + detail::quote(metal) + stated + " -o " + detail::quote(cpp),
          &log)) {
    drop(true);
    if (err) *err = "the source did not translate: " + log;
    return Library();
  }
  const std::string build =
      detail::quote(cxx) + " -std=c++20 -O2 -fno-strict-aliasing -fsigned-char " +
#if defined(__APPLE__)
      "-dynamiclib "
#else
      "-shared "
#endif
      "-fPIC -I" + detail::quote(inc) + " -I" + detail::quote(rt) + " " + detail::quote(cpp) + " " +
      detail::quote(rt + "/weft_runtime.cpp") + " -o " + detail::quote(lib);
  if (!detail::run(build, &log)) {
    drop(false);
    if (err) *err = "the translation did not build (" + cpp + "): " + log;
    return Library();
  }
  Library out;
  if (!detail::load_module(lib, &out.s_, err)) {
    drop(false);
    if (err) *err = "the built module did not load (" + lib + "): " + (err ? *err : "");
    return Library();
  }
  drop(true);
  return out;
#endif
}

}  // namespace vmetal
