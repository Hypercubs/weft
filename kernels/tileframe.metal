#include <metal_stdlib>
using namespace metal;

struct Cell { float4 c; };

[[tile]] void tileframe(imageblock<Cell> blk [[imageblock_data]],
                        device const float* in [[buffer(0)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tgs [[threads_per_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup float esc[1024];
    const uint x = gid % 128u;
    const uint y = gid / 128u;
    const float cx = (float)x * (3.0f / 128.0f) - 2.0f;
    const float cy = (float)y * (2.0f / 128.0f) - 1.0f;

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
    esc[lid] = n + in[gid] * 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint prev = lid == 0u ? lid : lid - 1u;
    const uint next = (lid + 1u) % tgs;
    blk.c = float4(esc[lid], esc[prev], esc[next], (float)(tgid & 1u));
}

fragment float4 shadetile(float4 pos [[position]],
                          device const float* in [[buffer(0)]],
                          imageblock<Cell> blk [[imageblock_data]]) {
    const uint x = (uint)pos.x;
    const uint y = (uint)pos.y;
    const float n = blk.c.x;
    const float m = n / 96.0f;
    const float neighbours = (blk.c.y + blk.c.z) * 0.5f;
    float4 c = float4(m * m, m * 0.5f + neighbours / 192.0f, 1.0f - m, 1.0f);
    if (n >= 96.0f) c = float4(0.0f, 0.0f, 0.0f, 1.0f);
    return c + float4(in[y * 128u + x] * 0.0f, 0.0f, 0.0f, 0.0f);
}
