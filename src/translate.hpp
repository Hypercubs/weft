#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct TranslatedKernel {
  std::string name;
  int nargs = 0;     // buffers the launcher has to bind, tile memory included
  int nbufs = 0;     // the `[[buffer(N)]]` parameters the kernel declares
  int tile_idx = -1;    // the buffer standing in for tile memory, if it takes one
  int colour_idx = -1;  // a fragment's first colour target
  int colour_targets = 1;
  int depth_idx = -1;
  // The qualifier that field names.
  std::string depth_qual = "any";
  std::vector<std::string> buf_type;
  struct Param {
    std::string type;  // the PTX type token without its dot: "u64", "f32"
    int buf = -1;
    int scalar = -1;
  };
  std::vector<Param> params;
  std::vector<std::string> tex_name;
  int ntex = 0;         // textures the kernel reads, bound with --textures
  int nsamp = 0;        // samplers it reads through, chosen with --samplers
  bool sync = false;
  bool flat = false;
  bool step = false;
  bool warp = false;  // also has a 32-lanes-at-a-time backend
  bool tile = false;  // a render pass's first stage, run before the fragments
  bool mesh = false;
  int mesh_nv = 0;  // vertex slots per mesh
  int mesh_np = 0;  // primitive slots per mesh
  bool task = false;
  int payload_idx = -1;
  int props_idx = -1;
  bool warp_only = false;
  bool gpu_warp = false;
  uint32_t slot_words = 0;
  uint32_t slot_per = 0;
  // The kernel uses a 64-bit atomic.
  bool wide = false;
  uint32_t tg_hint = 0;
  // A group size the module states without depending on it, SPIR-V's `LocalSize`.
  uint32_t def_threads = 0;
  uint32_t fixed_threads = 0;
  // What each texture binding has to be, in binding order.
  std::vector<std::string> tex_shape;
  // Bindings the kernel writes as well as reads, in binding order.
  std::vector<bool> tex_rw;
  // Pointer parameters the body never reads, so nothing is bound for them.
  std::vector<std::string> unused;
};

struct TranslateResult {
  bool ok = false;
  std::string error;
  std::string code;
  std::string gpu_src;
  std::string vertex_name;
  std::vector<int> vertex_sampler_slots;
  bool points = false;
  bool lowered = false;
  std::vector<int> fragment_sampler_slots;
  std::vector<std::pair<int, std::string>> function_constants;
  std::vector<TranslatedKernel> kernels;
};

struct RenderMode {
  std::string fragment;  // empty: the file's only fragment
  uint32_t width = 0, height = 0;
};

std::string rewrite_math_calls(const std::string& text);

// The semantics library a module needs to answer the same way on both targets.
std::string weft_math_preamble();

TranslateResult translate_msl(const std::string& msl, const RenderMode* render = nullptr);

std::string emit_module_abi(const std::vector<TranslatedKernel>& kernels);

std::string emit_ir(const std::vector<TranslatedKernel>& kernels);

// The same contract as JSON, for a consumer that is not this binary.
std::string emit_ir_json(const std::vector<TranslatedKernel>& kernels);
