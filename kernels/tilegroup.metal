#include <metal_stdlib>
using namespace metal;

struct TileVal { float4 c; };

[[tile]] void tilegroup(imageblock<TileVal> blk [[imageblock_data]],
                        device const float* in [[buffer(0)]],
                        texture2d<float> img [[texture(0)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tgs [[threads_per_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup float part[1024];
    part[lid] = in[gid] * img.read(uint2(gid % 64u, gid / 64u)).x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint peer = (lid + 1u) % tgs;
    blk.c = float4(part[peer] + (float)tgid, part[lid], 1.0f, 0.0f);
}

fragment float4 fraggroup(float4 pos [[position]],
                          device const float* in [[buffer(0)]],
                          imageblock<TileVal> blk [[imageblock_data]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    return float4(blk.c.x + blk.c.y + in[y * 64u + x], blk.c.z, 0.0f, 1.0f);
}

fragment float4 fragtilehop(float4 pos [[position]],
                            device const float* in [[buffer(0)]],
                            imageblock<TileVal> blk [[imageblock_data]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    return blk[y * 64u + (x ^ 32u)].c;
}
