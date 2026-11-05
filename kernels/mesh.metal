// A mesh object stage: the 32 lanes of a threadgroup fill one mesh, 16 vertices
// and 16 triangles, and two groups never touch the same slot, because a group's
// block is its own mesh.
//
// There is no GPU answer to compare against (mesh pipelines need
// MTLGPUFamilyApple9, M3 and later; the machine this was written on is an M1), so
// the check is a reference model: `check.sh` recomputes these 16 vertices and 16
// triangles independently and compares the bytes. Everything here is exact in
// float32, an integer, the input, twice the integer, and the integer plus a half,
// all of which are representable, so the comparison can be byte for byte.
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
    // Vertex i of this mesh is written by lane i; the slots the group does not
    // fill (there are 16 vertices and 32 lanes) stay as the launcher left them.
    if (lid < 16u) {
        const float i = float(lid);
        out.set_vertex(lid, Vtx{float4(i, in[gid], i * 2.0f, 1.0f),
                                float2(i, i + 0.5f)});
    }
    // One lane builds the triangle list, and every lane can read back what the
    // group wrote, which is the part a per-lane engine has to get right.
    if (lid == 0u) {
        for (uint k = 0u; k < 16u; k++) {
            out.set_primitive(k, uint3(k, (k + 1u) & 15u, (k + 2u) & 15u));
        }
    }
    // A mesh is threadgroup-visible: a lane reads what another lane wrote, so the
    // group has to arrive here between the write and the read, on the hardware
    // that is what makes the read defined at all.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 16u) {
        const Vtx v = out.get_vertex(3u);
        out.set_vertex(4u, Vtx{float4(v.p.x + v.uv.y, v.p.y, v.p.z, v.p.w), v.uv});
    }
}
