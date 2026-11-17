// The virtual device's Metal rules, checked without a GPU.
#include <vmetal.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int g_runs = 0;
int g_kernel = -1;
uint32_t g_grid = 0, g_threads = 0;
int g_samplers = 0, g_filter = -1, g_address = -1;
unsigned char g_seen[8] = {0};
int g_seen_n = 0;

const weft_kernel_info kInfos[2] = {
    {"two", 2, 0, nullptr, 0, 0},
    {"one", 1, 0, nullptr, 0, 0},
};

int api_count() { return 2; }
const weft_kernel_info* api_at(int i) { return i >= 0 && i < 2 ? &kInfos[i] : nullptr; }
void api_run(int i, void** bufs, uint32_t grid, uint32_t threads) {
  g_runs++;
  g_kernel = i;
  g_grid = grid;
  g_threads = threads;
  const unsigned char first = (bufs && bufs[0]) ? ((const unsigned char*)bufs[0])[0] : 0;
  if (g_seen_n < (int)sizeof g_seen) g_seen[g_seen_n++] = first;
}
void api_sampler(int, int filter, int address) {
  g_samplers++;
  g_filter = filter;
  g_address = address;
}

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL  %s\n", what);
    failures++;
  }
}

void reset() {
  g_runs = 0;
  g_kernel = -1;
  g_grid = g_threads = 0;
  g_samplers = 0;
  g_filter = g_address = -1;
  std::memset(g_seen, 0, sizeof g_seen);
  g_seen_n = 0;
}

// Metal source, compiled at run time by the second half of this probe.
const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void vscale(device const float* x [[buffer(0)]],
                   device float* y [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
  y[gid] = x[gid] * 3.0f + 1.0f;
}
)METAL";

int compile_check() {
  std::string err;
  vmetal::Library lib = vmetal::Device::new_library(kSource, &err);
  if (!lib.valid()) {
    std::printf("FAIL  a run-time library did not build: %s\n", err.c_str());
    return 1;
  }
  vmetal::Device dev = lib.device();
  vmetal::PipelineState pipe = lib.new_pipeline("vscale", &err);
  if (!pipe.valid()) {
    std::printf("FAIL  the run-time library has no vscale: %s\n", err.c_str());
    return 1;
  }
  const uint32_t n = 64;
  std::vector<float> x(n), y(n, 0.0f);
  for (uint32_t i = 0; i < n; i++) x[i] = (float)i + 0.5f;
  vmetal::Buffer bx = dev.new_buffer(n * sizeof(float), x.data());
  vmetal::Buffer by = dev.new_buffer(n * sizeof(float), y.data());
  vmetal::CommandQueue q = dev.new_queue();
  vmetal::CommandBuffer cb = q.command_buffer();
  vmetal::Encoder enc = cb.compute_encoder();
  enc.set_pipeline(pipe);
  enc.set_buffer(0, bx);
  enc.set_buffer(1, by);
  enc.dispatch_threads(n, 16);
  enc.end_encoding();
  cb.commit();
  cb.wait_until_completed();
  if (!cb.ok()) {
    std::printf("FAIL  the run-time library did not run: %s\n", cb.error().c_str());
    return 1;
  }
  const float* got = (const float*)by.contents();
  size_t differ = 0;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t a = 0, b = 0;
    const float want = x[i] * 3.0f + 1.0f;
    std::memcpy(&a, got + i, 4);
    std::memcpy(&b, &want, 4);
    if (a != b) differ++;
  }
  std::printf("compile  %u values  %zu differ: %s\n", n, differ,
              differ == 0 ? "a run-time library runs" : "DISAGREE");
  return differ == 0 ? 0 : 1;
}

bool read_file(const std::string& path, std::string* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char buf[8192];
  size_t n;
  out->clear();
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
  std::fclose(f);
  return true;
}

// The first .spv in `dir`, so the probe names no fixture of its own.
std::string first_spv(const std::string& dir) {
  std::error_code ec;
  std::vector<std::string> names;
  for (std::filesystem::directory_iterator it(dir, ec);
       !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".spv") == 0) names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names.empty() ? std::string() : names.front();
}

std::string run_first(vmetal::Library& lib) {
  vmetal::Device dev = lib.device();
  for (int i = 0; i < lib.kernel_count(); i++) {
    const weft_kernel_info* k = lib.kernel_info(i);
    if (!k || !k->name) continue;
    vmetal::PipelineState p = lib.new_pipeline(k->name, nullptr);
    if (!p.valid()) continue;
    vmetal::CommandQueue q = dev.new_queue();
    vmetal::CommandBuffer cb = q.command_buffer();
    vmetal::Encoder e = cb.compute_encoder();
    e.set_pipeline(p);
    std::vector<vmetal::Buffer> held;
    for (int b = 0; b < k->nbufs; b++) {
      size_t bytes = (k->buf_bytes && k->buf_bytes[b]) ? (size_t)k->buf_bytes[b] : 64;
      if (bytes < (1u << 16)) bytes = 1u << 16;
      held.push_back(dev.new_buffer(bytes));
      e.set_buffer(b, held.back());
    }
    std::vector<vmetal::Texture> images;
    for (int t = 0; t < k->ntex; t++) {
      images.push_back(dev.new_texture(8, 8, 1, 4));
      e.set_texture(t, images.back());
    }
    e.dispatch_threads(1, 1);
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    if (!cb.ok()) return std::string("FAIL  ") + k->name + ": " + cb.error();
    return std::string(k->name) + " ran";
  }
  return "no kernel in the library";
}

int library_check(const char* root, const std::string& rel, const char* label, bool by_bytes) {
  const std::string path = std::string(root) + "/" + rel;
  std::string bytes, err;
  if (!read_file(path, &bytes)) {
    std::printf("FAIL  %s: cannot read %s\n", label, path.c_str());
    return 1;
  }
  vmetal::Library lib = by_bytes ? vmetal::Device::new_library(bytes, &err)
                                 : vmetal::Device::new_library_from_file(path, &err);
  if (!lib.valid()) {
    std::printf("FAIL  %s: %s\n", label, err.c_str());
    return 1;
  }
  const std::string r = run_first(lib);
  if (r.rfind("FAIL", 0) == 0) {
    std::printf("%s\n", r.c_str());
    return 1;
  }
  std::printf("%-7s %d kernels, %s\n", label, lib.kernel_count(), r.c_str());
  return 0;
}

int widths_check(const char* root) {
  const std::string path = std::string(root) + "/kernels/bmsk.ptx";
  std::string err;
  vmetal::Library plain = vmetal::Device::new_library_from_file(path, &err);
  if (!plain.valid()) {
    std::printf("FAIL  widths: %s\n", err.c_str());
    return 1;
  }
  const weft_kernel_info* before = plain.kernel_info(0);
  if (!before || !before->buf_bytes || before->buf_bytes[0] != 0) {
    std::printf("FAIL  widths: the source already names slot 0, so nothing is stated\n");
    return 1;
  }
  vmetal::LibraryOptions opts;
  opts.widths.push_back({0, 4});
  vmetal::Library lib = vmetal::Device::new_library_from_file(path, opts, &err);
  if (!lib.valid()) {
    std::printf("FAIL  widths: %s\n", err.c_str());
    return 1;
  }
  const weft_kernel_info* after = lib.kernel_info(0);
  if (!after || !after->buf_bytes || after->buf_bytes[0] != 4) {
    std::printf("FAIL  widths: slot 0 reported %u, not the stated 4\n",
                (unsigned)(after && after->buf_bytes ? after->buf_bytes[0] : 0));
    return 1;
  }
  const std::string r = run_first(lib);
  if (r.rfind("FAIL", 0) == 0) {
    std::printf("%s\n", r.c_str());
    return 1;
  }
  std::printf("widths  slot 0 was unnamed, stated 4, %s\n", r.c_str());
  return 0;
}

int overrun_check(const char* root) {
  const std::string path = std::string(root) + "/kernels/bmsk.ptx";
  std::string err;
  vmetal::Library lib = vmetal::Device::new_library_from_file(path, &err);
  if (!lib.valid()) {
    std::printf("FAIL  overrun: %s\n", err.c_str());
    return 1;
  }
  vmetal::Device dev = lib.device();
  const weft_kernel_info* k = lib.kernel_info(0);
  vmetal::PipelineState p = lib.new_pipeline(k && k->name ? k->name : "", &err);
  if (!p.valid()) {
    std::printf("FAIL  overrun: %s\n", err.c_str());
    return 1;
  }
  for (int pass = 0; pass < 2; pass++) {
    const size_t bytes = pass == 0 ? 4 : 4096;
    vmetal::CommandQueue q = dev.new_queue();
    vmetal::CommandBuffer cb = q.command_buffer();
    vmetal::Encoder e = cb.compute_encoder();
    e.set_pipeline(p);
    vmetal::Buffer held = dev.new_buffer(bytes);
    e.set_buffer(0, held);
    e.dispatch_threads(1, 1);
    e.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    const std::string why = cb.error();
    if (pass == 0) {
      if (cb.ok()) {
        std::printf("FAIL  overrun: a 4-byte binding took a 76-byte body silently\n");
        return 1;
      }
      if (why.find("past the end of buffer 0") == std::string::npos) {
        std::printf("FAIL  overrun: refused for another reason: %s\n", why.c_str());
        return 1;
      }
    } else if (!cb.ok()) {
      std::printf("FAIL  overrun: a 4096-byte binding was refused: %s\n", why.c_str());
      return 1;
    }
  }
  std::printf("overrun a binding smaller than the body writes is refused, and a 76-byte body runs on 4096\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  vmetal::ModuleApi api;
  api.kernel_count = api_count;
  api.kernel_at = api_at;
  api.run = api_run;
  api.set_sampler = api_sampler;
  vmetal::Device dev = vmetal::Device::adopt(api);
  vmetal::CommandQueue q = dev.new_queue();
  std::string err;
  const vmetal::PipelineState two = dev.new_pipeline("two", &err);
  expect(two.valid(), err.c_str());
  const vmetal::PipelineState one = dev.new_pipeline("one", &err);
  expect(one.valid(), err.c_str());

  vmetal::Buffer a = dev.new_buffer(64);
  vmetal::Buffer b = dev.new_buffer(64);

  reset();
  vmetal::CommandBuffer cb = q.command_buffer();
  vmetal::Encoder enc = cb.compute_encoder();
  enc.set_pipeline(two);
  enc.set_buffer(0, a);
  enc.set_buffer(1, b);
  enc.set_sampler(0, 1, 2);
  enc.dispatch_threads(32, 8);
  enc.dispatch_threads(64, 16);
  enc.end_encoding();
  cb.commit();
  cb.wait_until_completed();
  expect(cb.ok(), cb.error().c_str());
  expect(cb.dispatch_count() == 2, "the command buffer counts both dispatches");
  expect(g_runs == 2, "both dispatches in one command buffer ran");
  expect(g_kernel == 0 && g_grid == 64 && g_threads == 16,
         "the last dispatch ran last, at its own geometry");
  expect(g_samplers == 1 && g_filter == 1 && g_address == 2,
         "the encoder's sampler reached the kernel");

  // A wait is a wait: the commands already ran.
  cb.wait_until_completed();
  expect(g_runs == 2, "a second wait ran the dispatches again");

  reset();
  vmetal::Buffer work = dev.new_buffer(16);
  vmetal::CommandBuffer ordered = q.command_buffer();
  vmetal::Encoder oe = ordered.compute_encoder();
  vmetal::BlitEncoder ob = ordered.blit_encoder();
  oe.set_pipeline(one);
  oe.set_buffer(0, work);
  ob.fill_buffer(work, 0, 16, 0xAB);
  oe.dispatch_threads(4, 1);
  ob.fill_buffer(work, 0, 16, 0xCD);
  oe.dispatch_threads(4, 1);
  ob.end_encoding();
  oe.end_encoding();
  ordered.commit();
  ordered.wait_until_completed();
  expect(ordered.ok(), ordered.error().c_str());
  expect(g_runs == 2 && g_seen[0] == 0xAB && g_seen[1] == 0xCD,
         "a blit between two dispatches landed in order");
  expect(ordered.dispatch_count() == 2, "the blits are not counted as dispatches");

  // A buffer copy moves the bytes named, and a fill past the end is refused.
  vmetal::Buffer dst = dev.new_buffer(16);
  vmetal::CommandBuffer copy = q.command_buffer();
  vmetal::BlitEncoder cp = copy.blit_encoder();
  cp.copy_buffer(work, 0, dst, 8, 4);
  cp.fill_buffer(dst, 0, 8, 0x5A);
  copy.commit();
  copy.wait_until_completed();
  expect(copy.ok(), copy.error().c_str());
  expect(dst.contents()[0] == 0x5A && dst.contents()[7] == 0x5A,
         "a fill covered the bytes it was given");
  expect(dst.contents()[8] == 0xCD && dst.contents()[11] == 0xCD,
         "a buffer copy moved the bytes it was given");

  vmetal::Texture t0 = dev.new_texture(4, 4, 1, 4);
  vmetal::Texture t1 = dev.new_texture(4, 4, 1, 4);
  vmetal::Buffer pix = dev.new_buffer(64);
  for (size_t i = 0; i < 64; i++) pix.contents()[i] = (unsigned char)(i * 3 + 1);
  vmetal::CommandBuffer tc = q.command_buffer();
  vmetal::BlitEncoder tb = tc.blit_encoder();
  tb.copy_buffer_to_texture(pix, 0, t0);
  tb.copy_texture(t0, t1);
  tc.commit();
  tc.wait_until_completed();
  expect(tc.ok(), tc.error().c_str());
  expect(std::memcmp(t0.contents(), pix.contents(), 64) == 0, "a buffer to texture copy landed");
  expect(std::memcmp(t1.contents(), t0.contents(), 64) == 0, "a texture copy landed");

  // The group counts come from a buffer, and the geometry they name is the grid.
  reset();
  uint32_t groups[3] = {4, 1, 1};
  vmetal::Buffer args = dev.new_buffer(sizeof groups, groups);
  vmetal::CommandBuffer ind = q.command_buffer();
  vmetal::Encoder ie = ind.compute_encoder();
  ie.set_pipeline(one);
  ie.set_buffer(0, a);
  ie.dispatch_threadgroups_indirect(args, 0, 8);
  ind.commit();
  ind.wait_until_completed();
  expect(ind.ok(), ind.error().c_str());
  expect(g_runs == 1 && g_grid == 32 && g_threads == 8,
         "an indirect group count became the grid it named");

  reset();
  uint32_t flat[3] = {4, 2, 1};
  vmetal::Buffer flat_args = dev.new_buffer(sizeof flat, flat);
  vmetal::CommandBuffer bad = q.command_buffer();
  vmetal::Encoder be = bad.compute_encoder();
  be.set_pipeline(one);
  be.set_buffer(0, a);
  be.dispatch_threadgroups_indirect(flat_args, 0, 8);
  bad.commit();
  bad.wait_until_completed();
  expect(!bad.ok(), "an indirect dispatch on the y axis was accepted");
  expect(g_runs == 0, "a refused indirect dispatch still ran the kernel");

  // A slot the kernel reads and nobody bound is refused before anything runs.
  reset();
  vmetal::CommandBuffer missing = q.command_buffer();
  vmetal::Encoder menc = missing.compute_encoder();
  menc.set_pipeline(two);
  menc.set_buffer(0, a);
  menc.dispatch_threads(32, 8);
  menc.end_encoding();
  missing.commit();
  missing.wait_until_completed();
  expect(!missing.ok(), "a missing buffer slot was accepted");
  expect(g_runs == 0, "a refused buffer still ran the kernel");

  reset();
  vmetal::CommandBuffer past = q.command_buffer();
  vmetal::Encoder penc = past.compute_encoder();
  penc.set_pipeline(one);
  penc.set_buffer(0, a, a.length() + 4);
  penc.dispatch_threads(32, 8);
  penc.end_encoding();
  past.commit();
  past.wait_until_completed();
  expect(!past.ok(), "an offset past the buffer was accepted");
  expect(g_runs == 0, "a refused offset still ran the kernel");

  // Nothing encoded is not a launch over a guessed geometry.
  reset();
  vmetal::CommandBuffer none = q.command_buffer();
  none.set_pipeline(one);
  none.wait_until_completed();
  expect(!none.ok(), "a command buffer with no dispatch was accepted");

  if (argc > 1) {
#if !defined(_WIN32)
    setenv("WEFT_ROOT", argv[1], 1);
#endif
    failures += compile_check();
    // PTX by its suffix, SPIR-V by its magic through the byte-string form.
    failures += library_check(argv[1], "kernels/cuda/scale.ptx", "ptx", false);
    const std::string spv = first_spv(std::string(argv[1]) + "/kernels/spirv");
    if (spv.empty()) {
      std::printf("spirv   no .spv fixture under kernels/spirv, not checked here\n");
    } else {
      failures += library_check(argv[1], "kernels/spirv/" + spv, "spirv", true);
    }
    failures += widths_check(argv[1]);
    failures += overrun_check(argv[1]);
  }

  std::printf("vmetal  %s, %d failed\n", failures ? "FAILED" : "the device holds Metal's rules",
              failures);
  return failures == 0 ? 0 : 1;
}
