// A task stage and the mesh stage it feeds.
#include <metal_stdlib>
using namespace metal;

struct Payload {
    uint base;
    float scale;
};

struct Vtx {
    float4 p;
};

// One payload element per task threadgroup.
[[object]] void taskgen(object_data Payload& p [[payload]],
                        mesh_grid_properties mgp,
                        uint gid  [[thread_position_in_grid]],
                        uint tgid [[threadgroup_position_in_grid]])
{
    if (gid % 32u == 0u) {
        p.base = tgid * 4u;
        p.scale = float(tgid) + 0.5f;
    }
    mgp.set_threadgroups_per_grid(uint3(4u, 1u, 1u));
}

[[mesh]] void meshuse(mesh<Vtx, 8, 8> out [[mesh]],
                      object_data Payload& p [[payload]],
                      device float* result [[buffer(0)]],
                      uint lid  [[thread_index_in_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]])
{
    if (lid < 8u) {
        out.set_vertex(lid, Vtx{float4(float(p.base + lid), p.scale, float(tgid), 1.0f)});
    }
    if (lid == 0u) {
        out.set_primitive(0u, uint3(0u, 1u, 2u));
        result[tgid * 2u] = float(p.base) + p.scale;
        result[tgid * 2u + 1u] = float(tgid);
    }
}
