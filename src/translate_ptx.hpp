#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "translate.hpp"

// PTX (CUDA's virtual ISA) -> the same C++ runtime the Metal front end targets.
// This is a scalar-subset translator: it handles the instructions a compute
// kernel actually needs and refuses anything else by name.
TranslateResult translate_ptx(const std::string& ptx,
                              const std::vector<std::pair<int, uint32_t>>& widths = {});

// The same decode, emitted as Metal Shading Language instead of C++: the second
// back end for one front end. The instruction text is the same, the runtime
// behind it is the GPU's, and the result is a kernel the Metal compiler takes , 
// which is what gives PTX (and so Triton) a GPU column of its own instead of a
// hand-written twin standing in for one.
TranslateResult translate_ptx_gpu(const std::string& ptx,
                                  const std::vector<std::pair<int, uint32_t>>& widths = {});
