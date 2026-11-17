#include <metal_stdlib>
using namespace metal;

fragment float4 mandelfrag(float4 pos [[position]],
                           device const float* in [[buffer(0)]]) {
    const uint x = (uint)pos.x;
    const uint y = (uint)pos.y;
    const float cx = (float)x * (3.0f / 256.0f) - 2.0f;
    const float cy = (float)y * (2.0f / 256.0f) - 1.0f;

    float zx = 0.0f;
    float zy = 0.0f;
    float n = 0.0f;
    for (uint i = 0u; i < 96u; i++) {
        if (zx * zx + zy * zy <= 4.0f) {
            const float t = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = t;
            n = n + 1.0f;
        }
    }
    const float m = n / 96.0f;
    float4 c = float4(m * m, m + 0.15f, 1.0f - m, 1.0f);
    if (n >= 96.0f) c = float4(0.0f, 0.0f, 0.0f, 1.0f);
    return c + float4(in[x % 16u] * 0.0f, 0.0f, 0.0f, 0.0f);
}
