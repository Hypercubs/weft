#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "translate.hpp"

TranslateResult translate_spirv(const std::string& binary,
                              const std::vector<std::pair<int, uint32_t>>& widths = {});

TranslateResult translate_spirv_gpu(const std::string& binary,
                                    const std::vector<std::pair<int, uint32_t>>& widths = {});
