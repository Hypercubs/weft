#include <metal_stdlib>
using namespace metal;

struct Wide {
    float4 c;
    float4 d;
};

[[tile]] void twide(imageblock<Wide> blk [[imageblock_data]],
                    device const float* in [[buffer(0)]],
                    uint gid [[thread_position_in_grid]]) {
    blk.c = float4(in[gid] + float(gid) * 0.001f, 0.0f, 0.0f, 1.0f);
    blk.d = float4(1.0f, 2.0f, 3.0f, 4.0f);
}

fragment float4 widefrag(float4 pos [[position]],
                         imageblock<Wide> blk [[imageblock_data]]) {
    return blk.c + blk.d;
}

fragment float4 wideover(float4 pos [[position]],
                         imageblock<Wide> blk [[imageblock_data]]) {
    const uint i = (uint)pos.y * 32u + (uint)pos.x;
    blk[i + 100u].c = float4(1.0f, 1.0f, 1.0f, 1.0f);
    return blk[i].c;
}
