#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct TranslatedKernel {
  std::string name;
  int nargs = 0;     // buffers the launcher has to bind, tile memory included
  int nbufs = 0;     // the `[[buffer(N)]]` parameters the kernel declares
  int tile_idx = -1;    // the buffer standing in for tile memory, if it takes one
  int colour_idx = -1;  // a fragment's colour target
  // The C++ element type of each `[[buffer(N)]]` parameter, when the front end
  // knows it. The module turns these into `sizeof(...)` so a caller can size a
  // buffer from the declaration instead of being told the layout out of band.
  std::vector<std::string> buf_type;
  int ntex = 0;         // textures the kernel reads, bound with --textures
  int nsamp = 0;        // samplers it reads through, chosen with --samplers
  bool sync = false;
  bool flat = false;
  bool warp = false;  // also has a 32-lanes-at-a-time backend
  bool tile = false;  // a render pass's first stage, run before the fragments
  // A `[[mesh]]` stage: its two outputs (one vertex block and one triangle block
  // per mesh) are created by the launcher, not passed in, because a mesh is a
  // threadgroup and the caller does not know how many the grid makes.
  bool mesh = false;
  int mesh_nv = 0;  // vertex slots per mesh
  int mesh_np = 0;  // primitive slots per mesh
  // A `[[object]]` task stage: it runs before the mesh stage and decides how
  // many meshes that stage runs over.
  bool task = false;
  // The binding a task payload binds at, or -1 when the stage has none, and the
  // same for the `mesh_grid_properties` handle. The launcher binds them by these
  // indices rather than by guessing from the parameter order.
  int payload_idx = -1;
  int props_idx = -1;
  // There is no per-lane body: the kernel uses a warp-wide operation whose
  // operand is another lane's register (a tensor-core fragment), so it runs on
  // the warp backend and the module emits no lane entry for it.
  bool warp_only = false;
  // The GPU form of this kernel is the warp back end — one Metal thread per
  // 32-lane warp — so the launcher dispatches a thirty-second of the grid.
  bool gpu_warp = false;
  // How many words of threadgroup memory one warp's relay needs, and how many
  // warps may share a threadgroup with one slice each. Both are fixed when the
  // kernel is translated, so the array the shader declares and the group the
  // launcher dispatches cannot disagree — a warp reading another warp's slots is
  // a wrong answer, not a slow one.
  uint32_t slot_words = 0;
  uint32_t slot_per = 0;
  // The kernel uses a 64-bit atomic. There is no such instruction here, so the
  // warp back end builds one out of 32-bit atomics under a device-scope lock,
  // and the launcher has to hand the kernel a zeroed table of lock words — the
  // same dispatch width as any other warp kernel.
  bool wide = false;
  // The threadgroup size the module itself declares, when it declares one. Triton
  // writes `.maxntid` from `num_warps`, and that is the only place the launch
  // geometry a kernel was compiled for is recorded — a caller that guesses a
  // different one gets a kernel indexed over a different number of blocks.
  uint32_t tg_hint = 0;
  // Pointer parameters the body never reads, so nothing is bound for them. PTX
  // leaves them out of the buffer order, which is why the count can be smaller
  // than the parameter list the signature shows.
  std::vector<std::string> unused;
};

struct TranslateResult {
  bool ok = false;
  std::string error;
  std::string code;
  // The source Metal compiles: the file as written, except that tile memory is
  // lowered to a per-pixel buffer (see translate.cpp). Equal to the input when
  // the file has no imageblock.
  std::string gpu_src;
  std::vector<TranslatedKernel> kernels;
};

// A render pass: the fragment stage is one invocation per pixel of a width x
// height target, so the target's width is all the translator needs to turn a
// pixel index into `[[position]]`.
struct RenderMode {
  std::string fragment;  // empty: the file's only fragment
  uint32_t width = 0, height = 0;
};

// Metal Shading Language (the compute subset) -> a C++ translation unit that
// links against weft_runtime and exposes the weft_kernel/weft_run entry points.
// With `render` set, the fragment functions become kernels whose grid is the
// target's pixels and whose extra last buffer is the colour target.
// Point a shader's math calls at the portable implementation. Metal's sqrt,
// rsqrt, log2, exp2 and pow are hardware approximations in every math mode, so
// the same rewrite is applied to the source handed to the GPU and to the body
// translated to C++, and both engines run the identical sequence of IEEE
// operations (see src/soft_math.hpp).
std::string rewrite_math_calls(const std::string& text);

TranslateResult translate_msl(const std::string& msl, const RenderMode* render = nullptr);

// The module-level ABI every front end emits after its kernels: the kernel table
// (name, buffer count, element widths) and the C entry points in
// include/weft/weft.h. One function so the three front ends cannot drift apart.
std::string emit_module_abi(const std::vector<TranslatedKernel>& kernels);
