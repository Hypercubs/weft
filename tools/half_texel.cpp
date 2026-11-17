#include <coroutine>

#include "weft_runtime.hpp"

#include <cstdio>

int main() {
  const unsigned short texels[3] = {0x3C00, 0x7BFF, 0x0001};   // 1.0, 65504, 2^-24
  weft::Texture t{};
  t.data = (unsigned char*)texels;
  t.width = 3;
  t.height = 1;
  t.layers = 1;
  t.elem = 2;   // an R16Float image: two bytes a texel

  std::printf("%g %g %g %zu\n", weft::tex_at_f<_Float16>(t, 0, 0, 0),
              weft::tex_at_f<_Float16>(t, 1, 0, 0), weft::tex_at_f<_Float16>(t, 2, 0, 0),
              sizeof(_Float16));
  return 0;
}
