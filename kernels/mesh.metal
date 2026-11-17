#include <metal_stdlib>
using namespace metal;

struct Vtx {
    float4 p;
    float2 uv;
};

[[mesh]] void meshmaker(mesh<Vtx, 16, 16> out [[mesh]],
                        device const float* in [[buffer(0)]],
                        uint gid  [[thread_position_in_grid]],
                        uint lid  [[thread_index_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]])
{
    if (lid < 16u) {
        const float i = float(lid);
        out.set_vertex(lid, Vtx{float4(i, in[gid], i * 2.0f, 1.0f),
                                float2(i, i + 0.5f)});
    }
    if (lid == 0u) {
        for (uint k = 0u; k < 16u; k++) {
            out.set_primitive(k, uint3(k, (k + 1u) & 15u, (k + 2u) & 15u));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 16u) {
        const Vtx v = out.get_vertex(3u);
        out.set_vertex(4u, Vtx{float4(v.p.x + v.uv.y, v.p.y, v.p.z, v.p.w), v.uv});
    }
}
