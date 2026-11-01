// A task stage and the mesh stage it feeds. The task stage runs first, fills the
// payload for its own threadgroup and says how many meshes should follow; the
// mesh stage then runs over that meshgrid and reads the payload of the task group
// that spawned it.
//
// As with the mesh stage, there is no GPU answer to compare against on this
// machine (mesh pipelines need MTLGPUFamilyApple9), so `check.sh` recomputes the
// whole result independently and compares bytes.
#include <metal_stdlib>
using namespace metal;

struct Payload {
    uint base;
    float scale;
};

struct Vtx {
    float4 p;
};

// One payload element per task threadgroup. Written by lane 0, so the answer does
// not depend on which lane a per-lane engine happens to run last.
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

// The mesh stage: 8 vertices and one triangle per mesh, plus a line of results so
// the payload it read is visible in an ordinary buffer too.
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
