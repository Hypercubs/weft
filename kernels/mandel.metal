#include <metal_stdlib>
using namespace metal;

// A picture, not a number: one thread per pixel, 128x64, escape-time Mandelbrot.
kernel void mandel(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid               [[thread_position_in_grid]]) {
    const uint x = gid % 128u;
    const uint y = gid / 128u;
    const float cx = (float)x * (3.0f / 128.0f) - 2.0f;
    const float cy = (float)y * (1.5f / 64.0f) - 0.75f;

    float zx = 0.0f;
    float zy = 0.0f;
    float n = 0.0f;
    for (uint i = 0u; i < 128u; i++) {
        if (zx * zx + zy * zy <= 4.0f) {
            const float t = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = t;
            n = n + 1.0f;
        }
    }
    out[gid] = n + in[gid] * 0.0f;
}
