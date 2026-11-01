#include <metal_stdlib>
using namespace metal;

// Two plain elementwise kernels and one that needs the threadgroup machinery:
// shared memory plus a barrier. All three are ordinary Metal — the file is run
// unmodified on the GPU, and translated to C++ for the CPU.

kernel void saxpy(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* out       [[buffer(2)]],
                  uint gid [[thread_position_in_grid]])
{
    out[gid] = 2.0f * a[gid] + 0.5f * b[gid];
}

// The reference for the one SPIR-V module LLVM's backend emits itself: the same
// serial sum, with the trip count baked in because the Metal front end has no
// scalar-parameter path. Its SPIR-V twin reads that count from a parameter.
kernel void serial_sum(device const float* in [[buffer(0)]],
                       device float* out      [[buffer(1)]])
{
    float acc = 0.0f;
    for (uint i = 0; i < 1000u; i++) {
        acc += sqrt(in[i]);
    }
    out[0] = acc;
}

// The reference for the PTX kernel that uses dynamic threadgroup memory and
// cp.async: the same rotation through a 64-entry tile, with the tile declared at
// a fixed size because Metal has no dynamic threadgroup memory.
kernel void async_copy(device const float* in [[buffer(0)]],
                       device float* out      [[buffer(1)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float tile[64];
    tile[lid & 63u] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = tile[(lid + 1u) & 63u];
}

// Textures. `read` is unnormalised and exact; `sample` with a nearest sampler at
// a texel centre is the same texel, so both must agree with the GPU exactly.
// The textures are filled by the launcher (`--textures WxH`), which is what lets
// a texture kernel be compared like any other.
kernel void texmix(device float* out [[buffer(0)]],
                  texture2d<float> img [[texture(0)]],
                  uint gid [[thread_position_in_grid]])
{
    constexpr sampler near(filter::nearest, address::clamp_to_edge);
    uint w = img.get_width();
    uint h = img.get_height();
    uint x = gid % w;
    uint y = (gid / w) % h;
    float a = img.read(uint2(x, y)).x;
    float b = img.sample(near, float2((x + 0.5f) / float(w), (y + 0.5f) / float(h))).x;
    out[gid] = a * 2.0f + b;
}

// Bilinear filtering, at a fraction of a texel off the centres. This one agrees
// to the GPU's filter precision rather than bit for bit: the hardware weights
// are 8-bit fixed point (see the README).
kernel void texlin(device float* out [[buffer(0)]],
                  texture2d<float> img [[texture(0)]],
                  uint gid [[thread_position_in_grid]])
{
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    float u = (float(gid % 256u) + 0.3f) / 256.0f;
    float v = (float((gid / 256u) % 256u) + 0.7f) / 256.0f;
    out[gid] = img.sample(lin, float2(u, v)).x;
}

kernel void stencil5(device const float* in [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     uint gid   [[thread_position_in_grid]],
                     uint total [[threads_per_grid]])
{
    // Written without an early return: a `return` here would stop the kernel
    // being flattened into a loop on the CPU, which costs about 5x.
    if (gid == 0 || gid + 1 >= total) {
        out[gid] = in[gid];
    } else {
        out[gid] = 0.5f * in[gid] + 0.25f * (in[gid - 1] + in[gid + 1]);
    }
}

// One 16-byte vector load and store per thread.
kernel void vec4_scale(device const float4* in  [[buffer(0)]],
                       device float4* out       [[buffer(1)]],
                       uint gid [[thread_position_in_grid]])
{
    out[gid] = in[gid] * 1.5f;
}

// Integer atomics: order-independent, so GPU and CPU must agree exactly.
kernel void atomic_bins(device const float* in [[buffer(0)]],
                        device atomic_uint* bins [[buffer(1)]],
                        uint gid [[thread_position_in_grid]])
{
    uint b = gid & 7u;
    atomic_fetch_add_explicit(&bins[b], 1u, memory_order_relaxed);
}

// Plain multiply, used as the reference for the PTX kernels that take scalar
// parameters and vector loads.
kernel void scale(device const float* in [[buffer(0)]],
                  device float* out       [[buffer(1)]],
                  uint gid [[thread_position_in_grid]])
{
    out[gid] = in[gid] * 1.5f;
}

// One barrier, one round of shared memory: each thread reads back the value its
// mirror image in the group wrote. The PTX front end is checked against this.
kernel void reverse(device const float* in [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint gid    [[thread_position_in_grid]],
                    uint lid    [[thread_position_in_threadgroup]],
                    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float tile[256];

    tile[lid] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = tile[tgsize - 1 - lid];
}

// Threadgroup reduce: each group sums 256 values through shared memory with a
// tree of barriers, then writes one partial sum per group.
kernel void reduce_sum(device const float* in  [[buffer(0)]],
                       device float* out       [[buffer(1)]],
                       uint gid    [[thread_position_in_grid]],
                       uint lid    [[thread_position_in_threadgroup]],
                       uint tgid   [[threadgroup_position_in_grid]],
                       uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float tile[256];

    tile[lid] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            tile[lid] += tile[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out[tgid] = tile[0];
    }
}

// Warp-level inclusive scan — the standard CUDA idiom (Harris's scan), one
// shuffle per doubling. Every lane writes a different value, so a wrong lane
// mapping shows up immediately instead of cancelling out.
kernel void warpscan(device const float* in [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     uint gid [[thread_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]])
{
    float v = in[gid];
    uint lane = lid & 31u;
    for (uint d = 1; d < 32; d <<= 1) {
        float t = simd_shuffle_up(v, d);
        if (lane >= d) v += t;
    }
    out[gid] = v;
}

// The fill variants: a lane whose source falls outside the warp gets the fill
// value instead of its own, which is how CUB's scans handle the edge. The result
// depends on the lane id, so a wrong boundary shows up immediately.
kernel void fillscan(device const float* in [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     uint gid [[thread_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]])
{
    float v = in[gid] + (float)(lid & 31u);
    float u = simd_shuffle_and_fill_up(v, 0.0f, 1u);
    float d = simd_shuffle_and_fill_down(v, -1.0f, 2u);
    out[gid] = u + 0.5f * d;
}

// lop3 and the floating-point class tests, against kernels/logic.ptx. The truth
// tables are written out as boolean expressions, and the eight class tests are
// the same eight constants, weighted so they add up to 63 when all are right.
kernel void logic(device const float* in [[buffer(0)]],
                  device uint* out       [[buffer(1)]],
                  uint gid [[thread_position_in_grid]])
{
    uint i = (uint)(in[gid] * 1000.0f);
    uint a = i * 2654435761u;
    uint b = a >> 13u;
    uint c = a ^ 0x12345678u;
    uint xor3 = a ^ b ^ c;                          // lop3 0x96
    uint maj = (a & b) | (a & c) | (b & c);         // lop3 0xE8
    // The class tests come from the bit pattern rather than from isnan/isfinite:
    // Metal compiles with fast-math, under which those can be folded away, and a
    // test that only works on the CPU side would prove nothing.
    uint one = as_type<uint>(1.0f);
    uint pinf = 0x7f800000u;
    uint pnan = 0x7fc00000u;
    uint psub = 0x00000001u;
    auto is_finite = [](uint b) { return (b & 0x7f800000u) != 0x7f800000u; };
    auto is_inf = [](uint b) { return (b & 0x7fffffffu) == 0x7f800000u; };
    auto is_nan = [](uint b) { return (b & 0x7fffffffu) > 0x7f800000u; };
    auto is_norm = [](uint b) {
        const uint e = b & 0x7f800000u;
        return e != 0u && e != 0x7f800000u;
    };
    auto is_sub = [](uint b) { return (b & 0x7f800000u) == 0u && (b & 0x007fffffu) != 0u; };
    uint cls = 1u   * (is_finite(one) ? 1u : 0u)
             + 2u   * (is_inf(pinf) ? 1u : 0u)
             + 4u   * (is_nan(pnan) ? 1u : 0u)
             + 8u   * (is_norm(one) ? 1u : 0u)
             + 16u  * (is_sub(psub) ? 1u : 0u)
             + 32u  * (!is_nan(pinf) ? 1u : 0u)
             + 64u  * (is_norm(psub) ? 1u : 0u)
             + 128u * (is_finite(pinf) ? 1u : 0u);
    out[gid] = xor3 + maj + cls;
}

// Warp-wide scans. Integer, because a scan of floats is only reproducible if the
// additions happen in the same order the GPU folds them, and this folds
// left-to-right (see the README).
kernel void prefixscan(device const uint* in [[buffer(0)]],
                      device uint* out      [[buffer(1)]],
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]])
{
    uint v = in[gid] + (lid & 31u);
    uint inc = simd_prefix_inclusive_sum(v);
    uint exc = simd_prefix_exclusive_sum(v);
    out[gid] = inc + 3u * exc;
}

// The same scan on floats, where the answer cannot be bit-identical: Metal folds
// a prefix sum in the hardware's tree and this folds it the doubling way — the
// same shape, not the same tree, and float addition is not associative. The
// suite checks how big that gap is instead of pretending there isn't one.
kernel void pfxscan(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid [[thread_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]])
{
    float v = in[gid] * (1.0f + float(lid & 31u) * 0.0625f);
    out[gid] = simd_prefix_inclusive_sum(v) + 3.0f * simd_prefix_exclusive_sum(v);
}

// A whole block reduction the way CUB does it: shfl_down within each warp, one
// shared-memory slot per warp, a barrier, then warp 0 reduces the warp sums.
// This is the shape of a real reduction kernel — shuffles, shared memory and a
// barrier in one body, one output per threadgroup.
kernel void block_reduce(device const float* in  [[buffer(0)]],
                         device float* out       [[buffer(1)]],
                         uint gid    [[thread_position_in_grid]],
                         uint lid    [[thread_position_in_threadgroup]],
                         uint tgid   [[threadgroup_position_in_grid]],
                         uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float warp_sums[32];

    float v = in[gid];
    for (uint d = 16; d > 0; d >>= 1) {
        v += simd_shuffle_down(v, d);
    }

    if ((lid & 31u) == 0u) {
        warp_sums[lid >> 5] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nwarps = (tgsize + 31u) >> 5;
    if (lid < 32u) {
        float t = (lid < nwarps) ? warp_sums[lid] : 0.0f;
        for (uint d = 16; d > 0; d >>= 1) {
            t += simd_shuffle_down(t, d);
        }
        if (lid == 0u) {
            out[tgid] = t;
        }
    }
}

// float3 has a 16-byte stride with 4 bytes of padding — the layout a device
// float3* depends on. Reading the fields into a float output keeps the padding
// out of the comparison.
kernel void vec3_mix(device const float3* in [[buffer(0)]],
                     device float* out        [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    float3 v = in[gid];
    float3 r = v * 2.0f + float3{1.0f, 2.0f, 3.0f};
    out[3 * gid + 0] = r.x + r.y;
    out[3 * gid + 1] = r.z - r.x;
    out[3 * gid + 2] = r.x * r.z;
}

// half storage and conversion. This stays on conversions on purpose: Metal's
// fp16 arithmetic is fast-math and keeps intermediates in higher precision, so a
// chained half expression is not bit-reproducible on a CPU that rounds every
// operation the way the type says it should.
kernel void half_cvt(device const float* in [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    half v = half(in[gid]);
    half2 p = half2(v, v);
    out[gid] = float(p.x) + float(p.y);
}

// fp16 in full: the arithmetic, the library on half arguments, and the vector
// widths. Results leave as floats so what is compared is the half operation and
// not the width of the store.
kernel void halfarith(device float* out      [[buffer(0)]],
                      device const float* in [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  const half a = half(0.5f + float(gid & 1023u) * 0.001f);
  const half b = half(0.75f + float(gid & 255u) * 0.002f);
  const half c = half(in[gid]);
  const uint o = gid * 38u;

  // the arithmetic itself
  out[o + 0u]  = float(a + b);
  out[o + 1u]  = float(a * b);
  out[o + 2u]  = float(a - b);
  out[o + 3u]  = float(a / b);
  out[o + 4u]  = float(a * b + a);              // a compiler may fuse this
  out[o + 5u]  = float(fma(a, b, a));           // one rounding, by definition
  out[o + 6u]  = float(a * b - c * a);          // two products in one expression

  // the library, on half arguments
  out[o + 7u]  = float(sqrt(a));
  out[o + 8u]  = float(rsqrt(a));
  out[o + 9u]  = float(exp2(a));
  out[o + 10u] = float(log2(a));
  out[o + 11u] = float(pow(a, b));
  out[o + 12u] = float(sin(a));
  out[o + 13u] = float(mix(a, b, a));
  out[o + 14u] = float(smoothstep(half(0.5f), half(1.0f), a));

  // the operations with no rounding freedom
  out[o + 15u] = float(abs(-a));
  out[o + 16u] = float(min(a, b));
  out[o + 17u] = float(max(a, b));
  out[o + 18u] = float(clamp(a, half(0.25f), half(0.75f)));
  out[o + 19u] = float(saturate(a));
  out[o + 20u] = float(floor(a * half(10.0f)));
  out[o + 21u] = float(fract(a * half(10.0f)));
  out[o + 22u] = float(sign(a - half(0.9f)));
  out[o + 23u] = float(step(half(1.0f), a));

  // vectors
  const half2 p = half2(a, b);
  const half2 q = half2(b, a);
  const half3 v = half3(a, b, c);
  const half4 w = half4(a, b, c, half(1.0f));
  out[o + 24u] = float((p + q).x);
  out[o + 25u] = float((p * q).y);
  out[o + 26u] = float(dot(p, q));
  out[o + 27u] = float(length(p));
  out[o + 28u] = float(normalize(p).x);
  out[o + 29u] = float(abs(-p).x);
  out[o + 30u] = float(min(p, q).y);
  out[o + 31u] = float(dot(v, v));
  out[o + 32u] = float(length(v));
  out[o + 33u] = float(normalize(v).z);
  out[o + 34u] = float(cross(v, half3(b, c, a)).x);
  out[o + 35u] = float(dot(w, w));
  out[o + 36u] = float(mix(w, half4(c), half4(a)).w);
  out[o + 37u] = float(half(float(a) * 0.5f) + half(float(b) * 0.5f));
}

// fp16 in threadgroup memory, which the fixture above cannot reach: a half is
// rounded once on the way into shared memory, the group's tree rounds once per
// step, and a 64-long chain rounds 64 times — the place where fast math's habit
// of keeping intermediates wide would show up as a last-bit difference. The
// four vector arrays also fix their sizes from inside the kernel, so a half3
// that is 6 bytes on one side and 8 on the other is a check and not a comment.
kernel void half_tg(device const float* in [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint gid    [[thread_position_in_grid]],
                    uint lid    [[thread_position_in_threadgroup]],
                    uint tgsize [[threads_per_threadgroup]])
{
  threadgroup half  tile[64];
  threadgroup half  odd[3];        // an odd element count before a wide type
  threadgroup half2 pairs[32];
  threadgroup half3 triples[32];
  threadgroup half4 quads[32];

  const half v  = half(in[gid]);
  const half v2 = half(float(v) * 0.5f);
  const half v4 = half(float(v) * 0.25f);

  tile[lid] = v;
  if (lid < 3u) odd[lid] = half(float(lid) + v2);
  if (lid < 32u) {
    pairs[lid]   = half2(v, v2);
    triples[lid] = half3(v, v2, v4);
    quads[lid]   = half4(v, v2, v4, half(1.0f));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint m = tgsize - 1u - lid;
  const uint q = lid & 31u;
  const uint o = 16u * gid;
  out[o +  0u] = float(tile[m]);          // the mirror's slot, one half away
  out[o +  1u] = float(tile[lid]);        // and our own, read back after the barrier
  out[o +  2u] = float(odd[lid % 3u]);    // an array of size 3
  out[o +  3u] = float(pairs[m & 31u].x);
  out[o +  4u] = float(pairs[m & 31u].y);
  out[o +  5u] = float(triples[m & 31u].x);
  out[o +  6u] = float(triples[m & 31u].z);   // the element past half3's padding
  out[o +  7u] = float(quads[m & 31u].w);
  out[o +  8u] = float(quads[m & 31u].y + quads[q].z);

  // A fixed order, so the rounding is fixed: 64 half adds.
  half acc = half(0.0f);
  for (uint k = 0u; k < 64u; k++) acc += tile[k];
  out[o +  9u] = float(acc);

  // 64 multiply-adds in a chain, each of which must round to half.
  half chain = half(0.5f);
  for (uint k = 0u; k < 64u; k++) chain = half(chain * half(0.5f) + tile[k] * half(0.125f));
  out[o + 10u] = float(chain);

  // The group's tree in half: six adds, each rounded and stored back. The
  // barrier before it is what makes the sums above well defined — the tree's
  // first write lands in slot 0 before its own barrier, so without this one a
  // lane that starts late reads a half-reduced array. On the GPU that shows up
  // as a group-uniform answer and a per-lane one here, which is exactly how the
  // first draft of this kernel disagreed.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32u; stride > 0u; stride >>= 1) {
    if (lid < stride) tile[lid] = tile[lid] + tile[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  out[o + 11u] = float(tile[0]);          // the whole group
  out[o + 12u] = float(tile[q]);          // and each subtree that still exists

  // Layout, measured by the compiler on each side rather than assumed here.
  out[o + 13u] = float(sizeof(half)) / 2.0f;
  out[o + 14u] = float(sizeof(half3)) / 8.0f;
  out[o + 15u] = float(sizeof(half4)) / 8.0f;
}

// Integer vectors and the 8/16-bit ones: index math plus packed fields. The
// results are written as floats so the comparison against the GPU is a real one
// (comparing raw integer bit patterns as floats would hide an off-by-one).
kernel void int_pack(device const int* in [[buffer(0)]],
                     device float* out     [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    int v = in[gid] & 1023;
    int3 a = int3{v, v + 3, v + 7};
    int3 b = a * 2;
    uint3 u = uint3((uint)v, (uint)v, (uint)v);
    uint3 s = u + uint3(1u, 2u, 3u);
    uchar4 p = uchar4{(uchar)(s.x & 0xFFu), (uchar)(s.y & 0xFFu), (uchar)(s.z & 0xFFu), (uchar)0};
    short2 q = short2((short)p.x, (short)p.w);
    char4 c = char4{(char)1, (char)-2, (char)3, (char)4};
    out[4 * gid + 0] = (float)(b.x + b.z);
    out[4 * gid + 1] = (float)((int)p.x + (int)p.z + (int)q.x);
    out[4 * gid + 2] = (float)((int)c.y + (int)p.y);
    out[4 * gid + 3] = (float)(s.x);
}

// A padded vector inside threadgroup memory: float3 in shared memory needs the
// same 16-byte stride the device buffer has.
kernel void tg_vec3(device const float3* in [[buffer(0)]],
                    device float* out        [[buffer(1)]],
                    uint gid    [[thread_position_in_grid]],
                    uint lid    [[thread_position_in_threadgroup]],
                    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float3 tile[64];

    tile[lid] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float3 v = tile[tgsize - 1 - lid];
    out[3 * gid + 0] = v.x + v.y;
    out[3 * gid + 1] = v.z;
    out[3 * gid + 2] = tile[lid].x;
}

// ---- integer bit manipulation ----------------------------------------------
// The same values as kernels/cuda/bits.ll, written from the definitions instead
// of the way the compiler happened to emit them: the funnel shifts are shifts of
// a 64-bit concatenation, the high word of a product is an actual division, and
// the byte permute is spelled out a byte at a time. check.sh compares the two
// byte for byte, so neither front end can be quietly wrong on its own.
kernel void bits(device const float* in [[buffer(0)]],
                device uint* out       [[buffer(1)]],
                uint gid [[thread_position_in_grid]])
{
    float f = in[gid] * 1000.0f + 100.0f;
    uint i = (uint)f;                       // truncation, like cvt.rzi.s32.f32
    uint a = i * 2654435761u;
    uint s = i & 31u;
    // A funnel shift works on a 64-bit pair and keeps the half the direction
    // names. The pair is {high:low}, so a left shift reads the high half of
    // {a:b} and a right shift the low half of {b:a}.
    uint rot  = (uint)(((((ulong)a << 32) | 305419896ul) << s) >> 32);   // shf.l.wrap
    uint rot2 = (uint)((((ulong)305419896ul << 32) | a) >> s);           // shf.r.wrap
    uint h32 = a / 3u;                         // mul.hi.u32 of a magic number
    ulong big = (ulong)a * 4096ul + (ulong)i;
    uint h64t = (uint)((big / 1000ul) & 0x7ffffffful);   // mul.hi.u64
    uint sdt = (uint)((uint)((long)big / -1000L) & 0xffffu);  // mul.hi.s64
    // prmt.b32 with selector 0x2040: byte0(a), byte0(b), byte0(a), byte2(a).
    uint perm = (a & 0xffu) | ((305419896u & 0xffu) << 8) | ((a & 0xffu) << 16) |
                (((a >> 16) & 0xffu) << 24);
    float nv = (a & 1u) ? as_type<float>(0x7fc00000u) : 1.0f;
    uint nn = ((as_type<uint>(nv) & 0x7fffffffu) > 0x7f800000u) ? 1u : 0u;
    uint irn = (uint)(int)rint(f);            // cvt.rni.f32.f32 + cvt.rzi
    uint irm = (uint)(int)floor(f);           // cvt.rmi.f32.f32 + cvt.rzi
    out[gid] = i + rot + rot2 + h32 + h64t + sdt + perm + nn + irn + irm;
}

// The four rounding converts, for comparison with kernels/round.ptx. `rint`
// rounds to nearest with ties to even, `trunc` is the .rz direction, and floor
// and ceil are .rm and .rp.
kernel void roundk(device const float* in [[buffer(0)]],
                   device int* out        [[buffer(1)]],
                   uint gid [[thread_position_in_grid]])
{
    float f = (in[gid] - 1.0f) * 1000.0f + 0.25f;
    int rn = (int)rint(f);
    int rz = (int)trunc(f);
    int rm = (int)floor(f);
    int rp = (int)ceil(f);
    int t0 = (int)rint(0.5f) * 17;    // 0: ties go to even
    int t1 = (int)rint(1.5f) * 19;    // 2
    int rt = (int)f * 13;             // the plain truncating convert
    out[gid] = rn * 3 + rz * 5 + rm * 7 + rp * 11 + t0 + t1 + rt;
}

// ---- warp votes -------------------------------------------------------------
// Each lane votes on a bit of its own index, so the answer is a constant:
// simd_ballot(bit 0) is 0x55555555, simd_all(index < 32) is true for every lane,
// simd_any(index == 7) is true for every lane.
kernel void votek(device const uint* in [[buffer(0)]],
                  device uint* out      [[buffer(1)]],
                  uint tid              [[thread_position_in_threadgroup]],
                  uint ctaid            [[threadgroup_position_in_grid]],
                  uint ntg              [[threads_per_threadgroup]]) {
  const uint gid = ctaid * ntg + tid;
  const uint lane = tid & 31u;
  const uint bal = static_cast<simd_vote::vote_t>(simd_ballot((lane & 1u) == 0u));
  const bool all = simd_all(lane < 32u);
  const bool any = simd_any(lane == 7u);
  out[gid] = bal + (all ? 1u : 0u) + (any ? 1u : 0u);
}

// ---- 64-bit and boolean vectors ---------------------------------------------
// Metal has vectors of long/ulong and of bool, and they are layout as much as
// arithmetic: long3 is 32 bytes with a hole in it, and a bool vector is one byte
// per lane with no padding at all. The numbers are small enough that every step
// is exact in float, so the GPU and the CPU must land on the same bits.
kernel void widevec(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid               [[thread_position_in_grid]]) {
    // Everything here stays under 2^24, so every partial sum is an exact integer
    // and the GPU's freedom to reassociate float additions cannot show up.
    long base = (long)(gid & 4095u) * 3L;
    long3 lv = long3(base, base + 1L, base + 2L);
    long3 lw = lv * 2L + long3{1L, 1L, 1L};
    ulong4 uv = ulong4((ulong)lw.x, (ulong)lw.y, (ulong)lw.z, (ulong)(gid & 4095u));

    bool4 lt = bool4{lv.x < lw.x, lv.y < lw.y, lv.z < lw.z, uv.w < uv.x};
    bool4 want = bool4{true, false, true, true};
    bool4 both = lt && want;
    int bits = (both.x ? 1 : 0) | (both.y ? 2 : 0) | (both.z ? 4 : 0) | (both.w ? 8 : 0);
    int qual = (all(want) ? 1 : 0) + (any(!lt) ? 1 : 0) + (all(both) ? 1 : 0);

    // Every term is an exact integer under 2^24, including the one taken from the
    // buffer, so the GPU's freedom to reassociate float additions cannot show up
    // — which is what makes this a test of the 64-bit vectors and not of the
    // compiler's fast-math.
    float isum = float(lw.x) + float(lw.y) + float(lw.z) + float(uv.w) + float(bits) +
                 float(qual) + float(sizeof(long3)) + float(sizeof(bool4)) +
                 float(int(in[gid]));
    out[gid] = isum;
}

// ---- simdgroup matrices ------------------------------------------------------
// One pair of 8x8 blocks per threadgroup. The tile is filled with the integer
// part of the input first, so every product and partial sum is exact and the
// hardware's fragment order cannot change a bit — what is being checked is the
// load, the store and the product themselves.
kernel void sgmat(device const float* in [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  uint gid               [[thread_position_in_grid]],
                  uint lid               [[thread_position_in_threadgroup]]) {
    threadgroup float tile[128];
    uint base = (gid / 128u) * 128u;
    if (lid < 128u) tile[lid] = float((int)(in[base + lid] * 6.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 a, b;
    simdgroup_load(a, tile, 8u);
    simdgroup_load(b, tile + 64u, 8u);
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_multiply_accumulate(acc, a, b, acc);
    simdgroup_store(acc, out + base, 8u);
}

// The same multiply with a transposed load on the left operand, so the transpose
// flag and the origin are covered too.
kernel void sgmatT(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid               [[thread_position_in_grid]],
                   uint lid               [[thread_position_in_threadgroup]]) {
    threadgroup float tile[128];
    uint base = (gid / 128u) * 128u;
    if (lid < 128u) tile[lid] = float((int)(in[base + lid] * 6.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 a, b;
    simdgroup_load(a, tile, 8u, ulong2(0, 0), true);
    simdgroup_load(b, tile + 64u, 8u);
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
    simdgroup_multiply_accumulate(acc, a, b, acc);
    simdgroup_store(acc, out + base, 8u);
}

// ---- the rest of the warp/quad library ---------------------------------------
// Each lane computes the same answer for the warp-wide reductions, and its own
// quad's answer for the quad ones.
kernel void grpops(device const uint* in [[buffer(0)]],
                   device uint* out      [[buffer(1)]],
                   uint gid              [[thread_position_in_grid]],
                   uint lane             [[thread_index_in_simdgroup]]) {
    uint first = simd_is_first() ? 1u : 0u;
    uint helper = simd_is_helper_thread() ? 1u : 0u;
    uint s = simd_sum(lane + 1u);                      // 528 in every lane
    uint mx = simd_max(lane * 7u + 1u);                // 218
    uint mn = simd_min(lane + 3u);                     // 3
    uint pr = simd_product(1u + (lane & 1u));          // 2^16
    uint b = simd_broadcast(lane * 5u, 5u);            // lane 5's value: 25
    uint pi = simd_prefix_inclusive_product(1u + (lane & 3u));
    uint pe = simd_prefix_exclusive_sum(1u);           // lane, in every lane
    uint qs = quad_sum(lane & 3u);                     // 6 in every lane of a quad
    uint qb = static_cast<quad_vote::vote_t>(quad_ballot((lane & 1u) == 0u));
    uint qx = quad_max(lane & 3u);                     // 3
    uint ql = quad_prefix_inclusive_sum(1u);           // (lane & 3) + 1
    uint qd = quad_shuffle_and_fill_down(lane, 77u, 1u);
    out[gid] = s + mx + mn + pr + b + pi + pe + qs + qb + qx + ql + qd + first + helper + in[gid];
}

// The Metal twin of kernels/twotile.ptx: two threadgroup tiles, a barrier, and
// one element from each on the way out.
kernel void twotile(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid               [[thread_position_in_grid]],
                    uint lid               [[thread_position_in_threadgroup]]) {
    threadgroup float ta[64];
    threadgroup float tb[64];
    uint i = lid & 63u;
    ta[i] = in[gid];
    tb[i] = in[gid] * 2.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = ta[(i + 1u) & 63u] + tb[(i + 3u) & 63u];
}

// The Metal twin of kernels/spirv/wide.spvasm: a float2 shuffle, and a 64-bit
// value shuffled as its two words (Metal's simd_shuffle is a 32-bit rendezvous,
// which is why the SPIR-V front end publishes a 64-bit value the same way).
kernel void wide_shfl(device const float* in [[buffer(0)]],
                      device float* out      [[buffer(1)]],
                      uint gid               [[thread_position_in_grid]],
                      uint lane              [[thread_index_in_simdgroup]]) {
    float2 v = float2(in[gid], in[gid] + 1.0f);
    float2 p = simd_shuffle(v, (lane + 5u) & 31u);
    long lv = (long)gid * 2654435761L + 7L;
    uint lo = simd_shuffle((uint)lv, (lane + 9u) & 31u);
    uint hi = simd_shuffle((uint)((ulong)lv >> 32), (lane + 9u) & 31u);
    out[gid] = p.x + p.y + float(lo & 0xFFFFu) + float(hi & 0xFFFFu);
}

// Float reductions: the fold is the hardware's butterfly, so a sum and a max are
// bit-exact, not merely close. (A scan is a different tree; see the README.)
kernel void grpsf(device const float* in [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  uint gid               [[thread_position_in_grid]]) {
    float s = simd_sum(in[gid]);
    float m = simd_max(in[gid]);
    float p = simd_product(in[gid] * 0.5f);
    out[gid] = s + m + p;
}

// Vector comparisons return a bool vector, for every vector type Metal defines
// them on — here the float and int ones, which is where the GPU can check it.
kernel void veccmp(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid               [[thread_position_in_grid]]) {
    float4 v = float4(in[gid], in[gid] * 2.0f, in[gid] * 0.5f, 1.0f);
    bool4 ge = v >= float4(0.5f, 1.0f, 0.25f, 0.75f);
    bool4 eq = v == float4(v.x, v.y, v.z, 2.0f);
    int a = (ge.x ? 1 : 0) | (ge.y ? 2 : 0) | (ge.z ? 4 : 0) | (ge.w ? 8 : 0);
    int b = (eq.x ? 1 : 0) | (eq.y ? 2 : 0) | (eq.z ? 4 : 0) | (eq.w ? 8 : 0);

    int3 i3 = int3(3, 1, 2);
    bool3 lt = i3 < int3(2, 2, 2);
    bool3 ne = i3 != int3(3, 1, 2);
    int c = (lt.x ? 1 : 0) | (lt.y ? 2 : 0) | (lt.z ? 4 : 0);
    int d = all(!ne) ? 1 : 0;

    out[gid] = float(a + b * 16 + c * 256 + d * 4096);
}

// The SPIR-V twin of kernels/spirv/global.spvasm: the same arithmetic, with the
// output spelled as a parameter instead of a module-scope global.
kernel void globk(device const float* in  [[buffer(0)]],
                  device float* out       [[buffer(1)]],
                  uint gid                [[thread_position_in_grid]]) {
    out[gid] = in[gid] * 3.0f + 1.0f;
}

// ---- packed and narrow vectors -----------------------------------------------
// A packed vector is a layout: three elements, no padding. A device buffer of
// (x, y, z) float triples is exactly what packed_float3 describes, and a struct of
// them has to come out the same size here as on the GPU — which is what the
// sizeof terms in the result check.
struct PackedTrip { packed_float3 p; packed_char3 c; packed_short3 s; };

kernel void packedk(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid               [[thread_position_in_grid]]) {
    packed_float3 a = packed_float3(in[gid], in[gid] + 1.0f, in[gid] + 2.0f);
    packed_float3 b = a * 2.0f + packed_float3(1.0f, 2.0f, 3.0f);
    packed_char3 c = packed_char3(10, 20, 30) + packed_char3(1, 2, 3);
    packed_short2 d = packed_short2(1000, 2000) * (short)3;
    packed_half2 e = packed_half2((half)1.5, (half)2.5) * (half)4.0;
    float sizes = float(sizeof(packed_float3) + sizeof(packed_char3) +
                        sizeof(packed_short3) + sizeof(packed_half2) * 100 +
                        sizeof(PackedTrip) * 10000);
    out[gid] = b.x / 2.0f + b.y / 3.0f + b.z / 5.0f + float(c.x) / 7.0f +
               float(c.y) / 11.0f + float(c.z) / 13.0f + float(d.x) / 17.0f +
               float(d.y) / 19.0f + float(e.x) / 23.0f + float(e.y) / 29.0f +
               sizes / 31.0f;
}

// The 8- and 16-bit vectors compute element-wise in their own type, so an
// overflow wraps where the hardware wraps.
kernel void narrowv(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid               [[thread_position_in_grid]]) {
    char4 a = char4((char)(int)in[gid], 100, -50, 7);
    char4 b = char4(100, 100, 100, 100);
    char4 cs = a + b;
    char4 cd = a / char4(2, 3, 5, 7);
    short2 s = short2(30000, -30000) * short2(2, 2);
    ushort3 u = ushort3(65000, 1, 2) - ushort3(1, 2, 3);
    uchar4 e = uchar4(200, 100, 50, 10) * (uchar)2;
    out[gid] = float(cs.x) / 2.0f + float(cs.y) / 3.0f + float(cs.z) / 5.0f +
               float(cs.w) / 7.0f + float(cd.x) / 11.0f + float(cd.y) / 13.0f +
               float(cd.z) / 17.0f + float(cd.w) / 19.0f + float(s.x) / 23.0f +
               float(s.y) / 29.0f + float(u.x) / 31.0f + float(u.y) / 37.0f +
               float(u.z) / 41.0f + float(e.x) / 43.0f + float(e.y) / 47.0f +
               float(e.z) / 53.0f + float(e.w) / 59.0f;
}

// ---- a kernel file's own declarations ----------------------------------------
// Structs and functions the file defines for itself are carried over, not
// dropped: a helper that only computes compiles as written, and one that waits on
// its threadgroup is expanded at each call site — a call cannot suspend the
// callee's coroutine, so the rendezvous has to end up inside the kernel body,
// where the resume point lives.
struct Trip { float lo, mid, hi; };

static float blend(Trip t, float w) {
    return t.lo + (t.mid - t.lo) * w + (t.hi - t.lo) * w * w;
}

static float mirror(threadgroup float* tile, uint lid, uint n) {
    tile[lid] = float(lid) * 1.5f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return tile[n - 1u - lid];
}

kernel void helperk(device const float* in  [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint gid                [[thread_position_in_grid]],
                    uint lid                [[thread_position_in_threadgroup]],
                    uint tgs                [[threads_per_threadgroup]]) {
    threadgroup float tile[256];
    Trip t;
    t.lo = in[gid];
    t.mid = in[gid] + 1.0f;
    t.hi = in[gid] + 2.0f;
    out[gid] = blend(t, 0.25f) + mirror(tile, lid, min(tgs, 256u));
}

// ---- a storage image ----------------------------------------------------------
// One lane writes a texel, the next lane reads it back. The value that comes out
// is the neighbour's, so a write that never reached the texture — kept in a
// register, or dropped — answers with the wrong number instead of hiding.
kernel void texrw(device const float* in [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  texture2d<float, access::read_write> img [[texture(0)]],
                  uint gid [[thread_position_in_grid]]) {
    const uint w = img.get_width();
    const uint h = img.get_height();
    const uint x = gid % w, y = (gid / w) % h;
    img.write(float4(in[gid], 0.0f, 0.0f, 0.0f), uint2(x, y));
    threadgroup_barrier(mem_flags::mem_texture);
    out[gid] = img.read(uint2(x ^ 1u, y)).x;
}

// ---- the grid's shape ---------------------------------------------------------
// threadgroups_per_grid is the quotient of the two sizes the launcher knows, and a
// short last threadgroup makes it round up: a grid of 1000 at 96 a group is 11
// groups on the GPU, not 10. That is the case a floor gets wrong.
kernel void counts(device float* out [[buffer(0)]],
                   uint gid  [[thread_position_in_grid]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tpg  [[threadgroups_per_grid]]) {
    // Packed as two 16-bit fields and reinterpreted, so a float cannot round the
    // comparison: the group count and the group index are compared as integers.
    out[gid] = as_type<float>((tpg & 0xffffu) | ((tgid & 0xffffu) << 16));
}

// ---- a tile stage and the fragment that reads it -------------------------------
// A tile function runs once per pixel of a tile before any fragment of that tile,
// and what it leaves in tile memory is what those fragments read. Tile memory is
// an imageblock, which this host's Metal compiler cannot compile, so both sides
// lower it to one element per pixel of the target — the same memory, addressed by
// the pixel — and the tile stage becomes a pass over the same grid that runs
// first. A value the fragment cannot see, or one written at the wrong pixel, is a
// wrong number rather than a hidden one.
struct TileVal { float4 c; };

[[tile]] void tilefill(imageblock<TileVal> blk [[imageblock_data]],
                       device const float* in [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    blk.c = float4(in[gid] * 2.0f + (float)(gid % 8u), (float)(gid % 61u), 0.5f, 1.0f);
}

fragment float4 fragtile(float4 pos [[position]],
                         device const float* in [[buffer(0)]],
                         imageblock<TileVal> blk [[imageblock_data]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    return float4(blk.c.x + in[y * 64u + x], blk.c.y + blk.c.z, 0.0f, 1.0f);
}

// ---- a fragment stage ---------------------------------------------------------
// A fragment runs once per pixel of the target, and `[[position]]` is that
// pixel's centre. The colour goes to an R32 attachment, so the red channel is the
// pixel; the launcher supplies the vertex stage and the target.
fragment float4 fragshade(float4 pos [[position]],
                          device const float* in [[buffer(0)]],
                          texture2d<float> img [[texture(0)]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    return float4(in[y * 64u + x] * 2.0f + img.read(uint2(x, y)).x, 0.0f, 0.0f, 1.0f);
}

// A target that is neither square nor a multiple of anything: 100x40. The row
// stride is the target's width, so a position mapping that is off by one in x or
// in y reads a different input rather than a neighbour in the same row.
fragment float4 fragwide(float4 pos [[position]],
                         device const float* in [[buffer(0)]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    return float4(in[y * 100u + x] + (float)x * 0.5f + (float)y, 0.0f, 0.0f, 1.0f);
}

// The reference for kernels/triton/add.ptx: the same masked elementwise add a
// Triton kernel has after `tl.load(..., mask=..., other=0.0)` and a guarded
// store. Triton passes `n_elements` as a scalar argument; the Metal front end has
// no scalar parameters, so the count is baked in — a grid of 1024 threads, 24 of
// them past the end.
kernel void triton_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out       [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    const uint n = 1000u;
    if (gid < n) out[gid] = a[gid] + b[gid];
}

// The reference for kernels/triton/matmul.ptx: one 16x8 tile of `tl.dot` per
// program, computed the way the PTX fragment layout says it should be — each lane
// owns rows g and g+8 and columns 2t, 2t+1 of the product. The PTX rounds its
// inputs to integers before converting them to halves, so every product and every
// sum of sixteen is exact and the two sides must agree to the bit.
kernel void triton_matmul(device const float* Ain [[buffer(0)]],
                          device const float* Bin [[buffer(1)]],
                          device float* out       [[buffer(2)]],
                          uint gid [[thread_position_in_grid]])
{
    const uint tile = gid >> 5;
    const uint l = gid & 31u;
    auto a = Ain + tile * 256u;   // the 16x16 tile of A this program owns
    auto b = Bin + tile * 128u;   // its 16x8 tile of B
    auto d = out + tile * 128u;   // and the 16x8 tile it writes
    const uint g = l >> 2, t = l & 3u;
    for (uint r = 0; r < 2; r++) {
        const uint row = g + r * 8u;
        for (uint c = 0; c < 2; c++) {
            const uint col = 2u * t + c;
            float acc = 0.0f;
            for (uint k = 0; k < 16u; k++)
                acc += (float)(half)rint(a[row * 16u + k] * 8.0f) *
                       (float)(half)rint(b[k * 8u + col] * 8.0f);
            d[row * 8u + col] = acc;
        }
    }
}

// One float reduced to the 10 mantissa bits tf32 has, rounded to nearest-even:
// the same reduction `mma.sync...tf32` performs on its operands, written out so
// the reference does it explicitly instead of borrowing f32's precision.
static inline float tf32r(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    if ((v.u & 0x7F800000u) == 0x7F800000u) return x;  // inf and NaN are unchanged
    v.u += 0x1000u + ((v.u >> 13) & 1u);
    v.u &= 0xFFFFE000u;
    return v.f;
}

// The reference for kernels/triton/dot32.ptx: one 16x8 tile of a tf32 dot per
// program, computed from the documented fragment layout. Each product of two
// tf32 values needs 22 mantissa bits and so is exact, which leaves the additions
// — and both sides add in the same order — so the two must agree to the bit.
kernel void triton_dot32(device const float* Ain [[buffer(0)]],
                         device const float* Bin [[buffer(1)]],
                         device float* out       [[buffer(2)]],
                         uint gid [[thread_position_in_grid]])
{
    const uint tile = gid >> 5;
    const uint l = gid & 31u;
    auto a = Ain + tile * 128u;   // 16x8
    auto b = Bin + tile * 64u;    // 8x8
    auto d = out + tile * 128u;
    const uint g = l >> 2, t = l & 3u;
    for (uint r = 0; r < 2; r++) {
        const uint row = g + r * 8u;
        for (uint c = 0; c < 2; c++) {
            const uint col = 2u * t + c;
            float acc = 0.0f;
            for (uint k = 0; k < 8u; k++)
                acc += tf32r(a[row * 8u + k]) * tf32r(b[k * 8u + col]);
            d[row * 8u + col] = acc;
        }
    }
}

// The reference for kernels/triton/reduce.ptx: the six warp folds `redux.sync`
// performs, each formed by a shuffle tree. An integer fold (and a min/max) has
// one answer whatever order it is formed in, so this is a reference and not a
// copy of the translation.
kernel void triton_reduce(device const float* in [[buffer(0)]],
                          device float* out       [[buffer(1)]],
                          uint gid [[thread_position_in_grid]])
{
    const float x = in[gid];
    const uint v = (uint)rint(x * 8.0f);
    uint s = v, mx = v, mn = v, an = v, orv = v, xr = v;
    for (int o = 16; o > 0; o >>= 1) {
        const ushort d = (ushort)o;
        s += simd_shuffle_xor(s, d);
        mx = max(mx, simd_shuffle_xor(mx, d));
        mn = min(mn, simd_shuffle_xor(mn, d));
        an &= simd_shuffle_xor(an, d);
        orv |= simd_shuffle_xor(orv, d);
        xr ^= simd_shuffle_xor(xr, d);
    }
    out[gid * 8u + 0u] = (float)s;
    out[gid * 8u + 1u] = (float)mx;
    out[gid * 8u + 2u] = (float)mn;
    out[gid * 8u + 3u] = (float)an;
    out[gid * 8u + 4u] = (float)orv;
    out[gid * 8u + 5u] = (float)xr;
}

// One float rounded to bf16, from the definition: the f32's high 16 bits, with
// round-to-nearest-even on the bit below them. The bf16 sample's twin does this
// itself, so a wrong conversion on the PTX side shows up as a difference.
static inline float bf16r(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    if ((v.u & 0x7F800000u) != 0x7F800000u) v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    v.u &= 0xFFFF0000u;
    return v.f;
}

// The reference for kernels/triton/dotbf16.ptx: the same 16x8 tile of `tl.dot`,
// over bf16 operands. A product of two bf16 values needs 16 mantissa bits and is
// exact, and both sides add the sixteen in the same order, so this must agree to
// the bit.
kernel void triton_dotbf16(device const float* Ain [[buffer(0)]],
                           device const float* Bin [[buffer(1)]],
                           device float* out       [[buffer(2)]],
                           uint gid [[thread_position_in_grid]])
{
    const uint tile = gid >> 5;
    const uint l = gid & 31u;
    auto a = Ain + tile * 256u;
    auto b = Bin + tile * 128u;
    auto d = out + tile * 128u;
    const uint g = l >> 2, t = l & 3u;
    for (uint r = 0; r < 2; r++) {
        const uint row = g + r * 8u;
        for (uint c = 0; c < 2; c++) {
            const uint col = 2u * t + c;
            float acc = 0.0f;
            for (uint k = 0; k < 16u; k++)
                acc += bf16r(a[row * 16u + k]) * bf16r(b[k * 8u + col]);
            d[row * 8u + col] = acc;
        }
    }
}

// A tile stage that waits on its group lives in its own file: a render pass has
// one `[[tile]]` stage, and this file already has one. See kernels/tilegroup.metal.

// ---- the vector math library a shader actually leans on -----------------------
// Every function here is Metal's component-wise definition; the GPU compiles this
// same source against its own library, so a wrong formula or a wrong order of
// operations shows up as a difference rather than as a picture that looks fine.
kernel void vecmath(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid [[thread_position_in_grid]])
{
    float3 a = float3{in[3 * gid], in[3 * gid + 1], in[3 * gid + 2]};
    float3 b = float3{a.y + 0.25f, a.z * 0.5f - 0.5f, a.x + 1.5f};

    float3 n = normalize(a + float3{0.25f, 0.5f, 0.75f});
    float3 m = normalize(b + float3{0.5f, 0.25f, 0.125f});
    float3 cr = cross(n, m);

    float3 blend = mix(a, b, 0.25f);
    float3 clamped = clamp(blend, -1.0f, 1.0f);
    float3 curve = smoothstep(-1.0f, 1.0f, clamped);
    float3 mask = step(0.0f, clamped);
    float3 powered = pow(abs(clamped) + 0.5f, 2.0f);
    float3 parts = fract(powered) + floor(powered) + ceil(powered * 0.5f) + sign(clamped);

    out[4 * gid + 0] = dot(n, m) + length(a) + distance(a, b) + length_squared(cr);
    out[4 * gid + 1] = n.x + cr.y + blend.z + curve.x + mask.y + saturate(length(a) - 1.0f);
    out[4 * gid + 2] = powered.x + parts.y + min(parts.z, clamped.x) + max(parts.x, clamped.z);
    out[4 * gid + 3] = fmod(a.x, 2.0f) + mix(a.x, b.y, 0.75f) + clamp(dot(a, b), 0.0f, 1.0f);
}

// ---- twins for the four PTX samples that had no comparison until now ----
//
// These four sat in kernels/cuda/ with nothing diffing them: the coverage check in
// check.sh names every fixture and requires a comparison, which is what surfaced
// them. Each is written from the PTX instruction by instruction, so an agreement is
// the PTX translator and this kernel arriving at the same number independently.

// `bsum`: shfl.sync.down over five distances, added as float — the sum of the lane
// and the lane `d` above it, clamped at the warp's top.
kernel void bsum(device const float* in [[buffer(0)]],
                 device float* out       [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    float v = in[gid];
    for (uint d = 16u; d >= 1u; d >>= 1) v += simd_shuffle_down(v, d);
    out[gid] = v;
}

// `hsum`: the same fold, but each step multiplies the sum by 0f3F800347 twelve
// times — a value that is exactly 1 + 839/2^23, so the decimal below is the float
// with nothing rounded on the way in.
kernel void hsum(device const float* in [[buffer(0)]],
                 device float* out       [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    const float c = 1.00010001659393310546875f;
    float v = in[gid];
    for (uint d = 16u; d >= 1u; d >>= 1) {
        float t = v + simd_shuffle_down(v, d);
        for (int i = 0; i < 12; i++) t *= c;
        v = t;
    }
    out[gid] = v;
}

// `pred`: the value under @%p. Even lanes compute v*3+1 then +2, odd lanes keep v,
// and both sides of the predicate are live in every warp.
kernel void pred(device const uint* in [[buffer(0)]],
                 device uint* out       [[buffer(1)]],
                 uint gid [[thread_position_in_grid]],
                 uint lid [[thread_index_in_threadgroup]])
{
    const uint v = in[gid];
    out[gid] = ((lid & 1u) == 0u) ? (v * 3u + 1u + 2u) : v;
}

// `rotate`: shfl.sync.idx, the source lane being (lane + 1) & 31 — a rotate of the
// whole warp's values by one lane, wrapped.
kernel void rotate(device const uint* in [[buffer(0)]],
                   device uint* out       [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_index_in_threadgroup]])
{
    out[gid] = simd_shuffle(in[gid], (lid + 1u) & 31u);
}
