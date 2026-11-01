#include <metal_stdlib>
using namespace metal;

// A render pass whose tile stage reads a texture and waits on its group before it
// fills tile memory. The wait is a suspension, so this is the tile stage through
// the coroutine path; the texture is bound to the tile pass, which is its own
// compute dispatch rather than the draw that follows it.
//
//     weft run kernels/tilegroup.metal --kernel fraggroup --render 64x64 \
//         --buffers 4096 --textures 64x64
//
// The grid is the target's pixels (4096 here) and a tile function's group is the
// tile: 32x32 = 1024 threads, so 4096 pixels is exactly four tiles and every
// group is whole. A tile stage that read a slot its group had not written would
// be reading undefined data, on the GPU as much as here.

struct TileVal { float4 c; };

[[tile]] void tilegroup(imageblock<TileVal> blk [[imageblock_data]],
                        device const float* in [[buffer(0)]],
                        texture2d<float> img [[texture(0)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tgs [[threads_per_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup float part[1024];
    // A tile is a square, so the tile's group is square too: the lane's position
    // in the tile is its position in the target modulo the tile.
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

// The read this lowering cannot model: pixel (x, y) reading pixel (x ^ 32, y),
// which at 64x64 is the tile next door. Inside one tile that read is legal and
// this runtime follows it to the right pixel (fraggroup's neighbour slot is the
// test); across tiles the GPU leaves it undefined, so `weft` refuses the run
// instead of letting both sides agree on a number neither of them computed.
//
//     weft run kernels/tilegroup.metal --kernel fragtilehop --render 64x64 \
//         --buffers 4096
//
// It declares the pass's buffer so it is launchable at all (a pass's stages take
// the same buffers, which the launcher checks before it runs anything); the read
// under test is the tile-memory one.
fragment float4 fragtilehop(float4 pos [[position]],
                            device const float* in [[buffer(0)]],
                            imageblock<TileVal> blk [[imageblock_data]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    // A linear subscript, not a uint2: this host's Metal rejects a coordinate
    // index on an imageblock, and the lowered form is one slot per pixel, so the
    // pixel is named the way the lowering addresses it.
    return blk[y * 64u + (x ^ 32u)].c;
}
