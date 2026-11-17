#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "translate.hpp"

// PTX (CUDA's virtual ISA) -> the same C++ runtime the Metal front end targets.
TranslateResult translate_ptx(const std::string& ptx,
                              const std::vector<std::pair<int, uint32_t>>& widths = {});

TranslateResult translate_ptx_gpu(const std::string& ptx,
                                  const std::vector<std::pair<int, uint32_t>>& widths = {});
