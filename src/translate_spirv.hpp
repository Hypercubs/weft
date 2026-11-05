#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "translate.hpp"

// SPIR-V (the binary form clang emits for OpenCL C, which is what LLVM's SPIR-V
// backend produces for a compute kernel) -> the same C++ runtime the Metal and
// PTX front ends target.
//
// The module is parsed as a binary: types, constants, decorations, entry points,
// and the SSA instructions of each kernel function. Every SSA value becomes a
// field of a per-thread state struct, the same resumable state machine the PTX
// front end emits, so a barrier is still `pc = N; return;` and the values live
// in the blob.
TranslateResult translate_spirv(const std::string& binary,
                              const std::vector<std::pair<int, uint32_t>>& widths = {});

// The same decode, emitted as Metal: the GPU column for a .spv file, the way
// translate_ptx_gpu is for a .ptx. A module that uses an image or a sampler is
// refused by name, a texture binding is a resource the launcher has to name,
// not an address the kernel can compute.
TranslateResult translate_spirv_gpu(const std::string& binary,
                                    const std::vector<std::pair<int, uint32_t>>& widths = {});
