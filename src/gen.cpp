// weft-gen — translate a GPU kernel file to C++ at build time.
//
//     weft-gen kernels/basic.metal --kernel saxpy -o build/saxpy.cpp
//     weft-gen kernels/spirv/blockreduce.spv -o build/reduce.cpp
//     weft-gen kernels/basic.metal --describe
//
// The output is a translation unit that links against runtime/weft_runtime.cpp
// and exposes the ABI in include/weft/weft.h. Doing this at build time is what
// makes the CPU path a dependency instead of a subprocess: no `clang++` at run
// time, no dylib, no dlopen, no Metal device.
//
// `--describe` prints what the module contains, one line per kernel, so a build
// script or a test fixture can size its buffers from the kernel's own
// declarations instead of being told the layout out of band.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "translate.hpp"
#include "translate_ptx.hpp"
#include "translate_spirv.hpp"

namespace {

const char* kUsage =
    "weft-gen — translate a GPU kernel to C++ at build time\n"
    "\n"
    "  weft-gen <file.metal|file.ptx|file.spv> [-o OUT] [--kernel NAME] [--render WxH]\n"
    "  weft-gen <file> --describe\n"
    "\n"
    "  -o PATH        write the C++ to PATH (default: stdout)\n"
    "  --kernel NAME  for a render pass, which fragment is the pass's\n"
    "  --render WxH   translate the file's fragment stage instead of a kernel\n"
    "  --describe     print the module's kernels and their arguments, then exit\n"
    "  --gpu          for a .ptx or a .spv file, write the Metal the GPU is handed\n"
    "                 of the C++ (the same decode, the other back end)\n"
    "  --buffer-bytes I=N[,J=N]  the element width of buffer I, for a front end\n"
    "                 that cannot type its pointers (PTX binds addresses)\n"
    "\n"
    "The output links against runtime/weft_runtime.cpp; include/weft/weft.h is the\n"
    "ABI it exposes. `--describe` is one line per kernel:\n"
    "\n"
    "  kernel <index> <name> <kind> buffers=<n> textures=<n> sync=<0|1> threads=<n> types=<t,...>\n"
    "\n"
    "`threads` is the threadgroup size the module declares (Triton's `.maxntid`), 0\n"
    "when it declares none.\n";

std::string read_file(const std::string& path, bool& ok) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    ok = false;
    return {};
  }
  std::string out;
  char buf[8192];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
  std::fclose(f);
  ok = true;
  return out;
}

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() > n && s.compare(s.size() - n, n, suffix) == 0;
}

const char* kind_of(const TranslatedKernel& k) {
  if (k.tile) return "tile";
  if (k.colour_idx >= 0) return "fragment";
  return "kernel";
}

}  // namespace

int main(int argc, char** argv) {
  std::string file, out, kernel;
  uint32_t rw = 0, rh = 0;
  bool describe = false, render = false, gpu = false;
  // PTX has no typed pointers: a kernel parameter is a .u64 address, so the
  // module cannot say how wide one element is. Rather than guess, weft-gen
  // takes the answer and bakes it into the module, where the ABI reports it.
  std::vector<std::pair<int, uint32_t>> widths;
  auto parse_widths = [&](const std::string& v) {
    size_t pos = 0;
    while (pos < v.size()) {
      const size_t comma = v.find(',', pos);
      const std::string one = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      const size_t eq = one.find('=');
      if (eq == std::string::npos) {
        std::fprintf(stderr, "weft-gen: --buffer-bytes wants I=N[,J=N], got '%s'\n", v.c_str());
        std::exit(2);
      }
      const long idx = std::strtol(one.substr(0, eq).c_str(), nullptr, 10);
      const unsigned long n = std::strtoul(one.substr(eq + 1).c_str(), nullptr, 10);
      if (idx < 0 || n == 0) {
        std::fprintf(stderr, "weft-gen: --buffer-bytes wants a non-negative index and a non-zero width\n");
        std::exit(2);
      }
      widths.push_back({(int)idx, (uint32_t)n});
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  };

  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "weft-gen: %s needs a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-o") {
      out = next("-o");
    } else if (a == "--kernel") {
      kernel = next("--kernel");
    } else if (a == "--buffer-bytes") {
      parse_widths(next("--buffer-bytes"));
    } else if (a == "--describe") {
      describe = true;
    } else if (a == "--gpu") {
      gpu = true;
    } else if (a == "--render") {
      const std::string v = next("--render");
      const size_t x = v.find('x');
      if (x == std::string::npos) {
        std::fprintf(stderr, "weft-gen: --render needs WxH, got '%s'\n", v.c_str());
        return 2;
      }
      rw = (uint32_t)std::strtoul(v.substr(0, x).c_str(), nullptr, 10);
      rh = (uint32_t)std::strtoul(v.substr(x + 1).c_str(), nullptr, 10);
      if (!rw || !rh) {
        std::fprintf(stderr, "weft-gen: --render needs a non-zero WxH\n");
        return 2;
      }
      render = true;
    } else if (a == "-h" || a == "--help") {
      std::fputs(kUsage, stdout);
      return 0;
    } else if (!a.empty() && a[0] != '-') {
      file = a;
    } else {
      std::fprintf(stderr, "weft-gen: unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (file.empty()) {
    std::fputs(kUsage, stderr);
    return 2;
  }

  bool ok = false;
  const std::string src = read_file(file, ok);
  if (!ok) {
    std::fprintf(stderr, "weft-gen: cannot read %s\n", file.c_str());
    return 1;
  }

  TranslateResult tr;
  if (ends_with(file, ".spv")) {
    if (render) {
      std::fprintf(stderr, "weft-gen: SPIR-V has no render stages here\n");
      return 2;
    }
    tr = gpu ? translate_spirv_gpu(src, widths) : translate_spirv(src, widths);
  } else if (ends_with(file, ".ptx")) {
    if (render) {
      std::fprintf(stderr, "weft-gen: PTX has no render stages here\n");
      return 2;
    }
    tr = gpu ? translate_ptx_gpu(src, widths) : translate_ptx(src, widths);
  } else {
    RenderMode rm;
    rm.fragment = kernel;
    rm.width = rw;
    rm.height = rh;
    tr = translate_msl(src, render ? &rm : nullptr);
  }
  if (!tr.ok) {
    std::fprintf(stderr, "weft-gen: %s\n", tr.error.c_str());
    return 1;
  }

  // A stated width wins over anything the front end inferred: it exists to fill
  // the gaps of a front end that has nothing to infer from.
  for (auto& k : tr.kernels) {
    if (k.buf_type.size() < (size_t)(k.nbufs > 0 ? k.nbufs : 0)) k.buf_type.resize(k.nbufs);
    for (const auto& w : widths) {
      if (w.first >= k.nbufs) {
        std::fprintf(stderr, "weft-gen: --buffer-bytes %d: kernel '%s' has %d buffers\n",
                     w.first, k.name.c_str(), k.nbufs);
        return 2;
      }
      k.buf_type[(size_t)w.first] = std::to_string(w.second);
    }
    for (size_t b = 0; b < k.buf_type.size(); b++)
      if (k.buf_type[b].empty()) k.buf_type[b] = "0";  // unknown: the ABI says so
  }

  // --gpu is the only thing that writes Metal: the other front ends hand back a
  // C++ translation unit, and a .metal file's `gpu_src` is the source the GPU
  // compiles, which is not what -o means there.
  if (gpu && !tr.gpu_src.empty()) tr.code = tr.gpu_src;

  if (describe) {
    std::printf("kernels %zu\n", tr.kernels.size());
    for (size_t i = 0; i < tr.kernels.size(); i++) {
      const TranslatedKernel& k = tr.kernels[i];
      std::string types;
      for (size_t b = 0; b < k.buf_type.size(); b++) {
        types += (b ? "," : "") + k.buf_type[b];
      }
      std::printf("kernel %zu %s %s buffers=%d textures=%d sync=%d threads=%u types=%s\n", i,
                  k.name.c_str(), kind_of(k), k.nbufs, k.ntex, k.sync ? 1 : 0, k.tg_hint,
                  types.c_str());
    }
    return 0;
  }

  if (out.empty()) {
    std::fwrite(tr.code.data(), 1, tr.code.size(), stdout);
    return 0;
  }
  FILE* f = std::fopen(out.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "weft-gen: cannot write %s\n", out.c_str());
    return 1;
  }
  std::fwrite(tr.code.data(), 1, tr.code.size(), f);
  std::fclose(f);
  return 0;
}
