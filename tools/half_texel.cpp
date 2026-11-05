// A two-byte image texel, read the way a sample reads one.
//
// The runtime's sample path is two steps: `tex_read<T>` copies `sizeof(T)` bytes
// starting at `index * t.elem`, and `tex_at_f<T>` applies `(float)T`. So the type
// chosen for a 16-bit texel has to match the width *and* be a decode, which is why
// it is `_Float16` and not `uint16_t`. Both would pass the memcpy; only one answers
// 1.0 for 0x3C00, and the other answers 15360.
//
// Three patterns, one per case a decode gets wrong: the ordinary value, the largest
// finite half, and the smallest subnormal, a widening that flushed subnormals would
// read the third as zero.
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
