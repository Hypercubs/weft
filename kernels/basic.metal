#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* out       [[buffer(2)]],
                  uint gid [[thread_position_in_grid]])
{
    out[gid] = 2.0f * a[gid] + 0.5f * b[gid];
}

kernel void serial_sum(device const float* in [[buffer(0)]],
                       device float* out      [[buffer(1)]])
{
    float acc = 0.0f;
    for (uint i = 0; i < 1000u; i++) {
        acc += sqrt(in[i]);
    }
    out[0] = acc;
}

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

// Textures.
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

kernel void swzcmp(device float* out [[buffer(0)]],
                   uint gid [[thread_position_in_grid]])
{
    float4 v = float4(float(gid & 63u), 1.0f, 2.0f, 3.0f);
    float2 p = float2(0.5f, 0.25f);
    v.xy += p;                         // two components, one statement
    v.zw -= float2(1.0f, 0.5f);
    v.yx *= v.zw;                      // the right-hand side is a swizzle of it
    v.wz /= float2(2.0f, 4.0f);

    int2 q = int2(int(gid), 7);
    q |= int2(1, 2);
    q ^= int2(3, 4);
    q &= int2(0x7f, 0x3f);

    out[gid] = v.x + v.y * 2.0f + v.z * 4.0f + v.w * 8.0f + float(q.x & 255) +
               float(q.y & 255) * 16.0f;
}

// A texture with one row.
kernel void tex1ds(device float* out [[buffer(0)]],
                   texture1d<float> img [[texture(0)]],
                   uint gid [[thread_position_in_grid]])
{
    constexpr sampler near(filter::nearest, address::clamp_to_edge);
    uint w = img.get_width();
    uint x = gid % w;
    float a = img.read(x).x;
    float b = img.sample(near, (x + 0.5f) / float(w)).x;
    out[gid] = a * 2.0f + b;
}

// Bilinear filtering, at a fraction of a texel off the centres.
kernel void texlin(device float* out [[buffer(0)]],
                  texture2d<float> img [[texture(0)]],
                  uint gid [[thread_position_in_grid]])
{
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    float u = (float(gid % 256u) + 0.3f) / 256.0f;
    float v = (float((gid / 256u) % 256u) + 0.7f) / 256.0f;
    out[gid] = img.sample(lin, float2(u, v)).x;
}

kernel void texarray(device float* out [[buffer(0)]],
                     texture2d_array<float> img [[texture(0)]],
                     uint gid [[thread_position_in_grid]])
{
    constexpr sampler near(filter::nearest, address::clamp_to_edge);
    uint w = img.get_width();
    uint h = img.get_height();
    uint n = img.get_array_size();
    uint x = gid % w;
    uint y = (gid / w) % h;
    uint l = (x + y) % n;
    float a = img.read(uint2(x, y), l).x;
    float b = img.sample(near, float2((x + 0.5f) / float(w), (y + 0.5f) / float(h)), l).x;
    out[gid] = a * 2.0f + b + float(l);
}

kernel void tex3d(device float* out [[buffer(0)]],
                  texture3d<float> vol [[texture(0)]],
                  uint gid [[thread_position_in_grid]])
{
    constexpr sampler near(filter::nearest, address::clamp_to_edge);
    uint w = vol.get_width();
    uint h = vol.get_height();
    uint d = vol.get_depth();
    uint x = gid % w;
    uint y = (gid / w) % h;
    uint z = (x + y) % d;
    float a = vol.read(uint3(x, y, z)).x;
    float b = vol.sample(near, float3((x + 0.5f) / float(w), (y + 0.5f) / float(h),
                                      (z + 0.5f) / float(d))).x;
    out[gid] = a * 2.0f + b + float(z);
}

kernel void texcube(device float* out [[buffer(0)]],
                    texturecube<float> cub [[texture(0)]],
                    uint gid [[thread_position_in_grid]])
{
    constexpr sampler near(filter::nearest, address::clamp_to_edge);
    uint w = cub.get_width();
    uint h = cub.get_height();
    uint n = 6u;                            // a cube is six faces, fixed by its type
    uint face = (gid / 1024u) % n;
    uint i = gid & 63u;                     // texel x inside the face
    uint j = (gid >> 6) & 15u;              // texel y inside the face
    float a = (float(i) + 0.5f) / float(w) * 2.0f - 1.0f;   // -1 < a < 1
    float b = (float(j) + 0.5f) / float(h) * 2.0f - 1.0f;
    float3 d;
    if (face == 0u)      d = float3( 1.0f, -b, -a);   // +X
    else if (face == 1u) d = float3(-1.0f, -b,  a);   // -X
    else if (face == 2u) d = float3( a,  1.0f,  b);   // +Y
    else if (face == 3u) d = float3( a, -1.0f, -b);   // -Y
    else if (face == 4u) d = float3( a, -b,  1.0f);   // +Z
    else                 d = float3(-a, -b, -1.0f);   // -Z
    out[gid] = cub.sample(near, d).x + float(face);
}

kernel void stencil5(device const float* in [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     uint gid   [[thread_position_in_grid]],
                     uint total [[threads_per_grid]])
{
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

kernel void scale(device const float* in [[buffer(0)]],
                  device float* out       [[buffer(1)]],
                  uint gid [[thread_position_in_grid]])
{
    out[gid] = in[gid] * 1.5f;
}

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

// lop3 and the floating-point class tests, against kernels/logic.ptx.
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
// The class tests come from the bit pattern rather than from isnan/isfinite.
    uint one = as_type<uint>(1.0f);
    uint pinf = 0x7f800000u;
    uint pnan = 0x7fc00000u;
    uint psub = 0x00000001u;
    uint pzero = 0x00000000u;
    uint nzero = 0x80000000u;
    auto is_finite = [](uint b) { return (b & 0x7f800000u) != 0x7f800000u; };
    auto is_inf = [](uint b) { return (b & 0x7fffffffu) == 0x7f800000u; };
    auto is_nan = [](uint b) { return (b & 0x7fffffffu) > 0x7f800000u; };
    auto is_norm = [](uint b) {
        const uint e = b & 0x7f800000u;
        return e != 0x7f800000u && (e != 0u || (b & 0x007fffffu) == 0u);
    };
    auto is_sub = [](uint b) { return (b & 0x7f800000u) == 0u && (b & 0x007fffffu) != 0u; };
    uint cls = 1u   * (is_finite(one) ? 1u : 0u)
             + 2u   * (is_inf(pinf) ? 1u : 0u)
             + 4u   * (is_nan(pnan) ? 1u : 0u)
             + 8u   * (is_norm(one) ? 1u : 0u)
             + 16u  * (is_sub(psub) ? 1u : 0u)
             + 32u  * (!is_nan(pinf) ? 1u : 0u)
             + 64u  * (is_norm(psub) ? 1u : 0u)
             + 128u * (is_finite(pinf) ? 1u : 0u)
             + 256u * (is_norm(pzero) ? 1u : 0u)
             + 512u * (is_norm(nzero) ? 1u : 0u);
    out[gid] = xor3 + maj + cls;
}

// Warp-wide scans.
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

// The same scan on floats, where the answer cannot be bit-identical.
kernel void pfxscan(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid [[thread_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]])
{
    float v = in[gid] * (1.0f + float(lid & 31u) * 0.0625f);
    out[gid] = simd_prefix_inclusive_sum(v) + 3.0f * simd_prefix_exclusive_sum(v);
}

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

// half storage and conversion.
kernel void half_cvt(device const float* in [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    half v = half(in[gid]);
    half2 p = half2(v, v);
    out[gid] = float(p.x) + float(p.y);
}

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

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32u; stride > 0u; stride >>= 1) {
    if (lid < stride) tile[lid] = tile[lid] + tile[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  out[o + 11u] = float(tile[0]);          // the whole group
  out[o + 12u] = float(tile[q]);          // and each subtree that still exists

  out[o + 13u] = float(sizeof(half)) / 2.0f;
  out[o + 14u] = float(sizeof(half3)) / 8.0f;
  out[o + 15u] = float(sizeof(half4)) / 8.0f;
}

// Integer vectors and the 8/16-bit ones: index math plus packed fields.
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

kernel void bits(device const float* in [[buffer(0)]],
                device uint* out       [[buffer(1)]],
                uint gid [[thread_position_in_grid]])
{
    float f = in[gid] * 1000.0f + 100.0f;
    uint i = (uint)f;                       // truncation, like cvt.rzi.s32.f32
    uint a = i * 2654435761u;
    uint s = i & 31u;
// A funnel shift works on a 64-bit pair and keeps the half the direction names.
    uint rot  = (uint)(((((ulong)a << 32) | 305419896ul) << s) >> 32);   // shf.l.wrap
    uint rot2 = (uint)((((ulong)305419896ul << 32) | a) >> s);           // shf.r.wrap
    uint h32 = a / 3u;                         // mul.hi.u32 of a magic number
    ulong big = (ulong)a * 4096ul + (ulong)i;
    uint h64t = (uint)((big / 1000ul) & 0x7ffffffful);   // mul.hi.u64
    uint sdt = (uint)((uint)((long)big / -1000L) & 0xffffu);  // mul.hi.s64
    uint perm = (a & 0xffu) | ((305419896u & 0xffu) << 8) | ((a & 0xffu) << 16) |
                (((a >> 16) & 0xffu) << 24);
    float nv = (a & 1u) ? as_type<float>(0x7fc00000u) : 1.0f;
    uint nn = ((as_type<uint>(nv) & 0x7fffffffu) > 0x7f800000u) ? 1u : 0u;
    uint irn = (uint)(int)rint(f);            // cvt.rni.f32.f32 + cvt.rzi
    uint irm = (uint)(int)floor(f);           // cvt.rmi.f32.f32 + cvt.rzi
    out[gid] = i + rot + rot2 + h32 + h64t + sdt + perm + nn + irn + irm;
}

// The four rounding converts, for comparison with kernels/round.ptx.
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

// warp votes.
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

// 64-bit and boolean vectors.
kernel void widevec(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid               [[thread_position_in_grid]]) {
    long base = (long)(gid & 4095u) * 3L;
    long3 lv = long3(base, base + 1L, base + 2L);
    long3 lw = lv * 2L + long3{1L, 1L, 1L};
    ulong4 uv = ulong4((ulong)lw.x, (ulong)lw.y, (ulong)lw.z, (ulong)(gid & 4095u));

    bool4 lt = bool4{lv.x < lw.x, lv.y < lw.y, lv.z < lw.z, uv.w < uv.x};
    bool4 want = bool4{true, false, true, true};
    bool4 both = lt && want;
    int bits = (both.x ? 1 : 0) | (both.y ? 2 : 0) | (both.z ? 4 : 0) | (both.w ? 8 : 0);
    int qual = (all(want) ? 1 : 0) + (any(!lt) ? 1 : 0) + (all(both) ? 1 : 0);

    float isum = float(lw.x) + float(lw.y) + float(lw.z) + float(uv.w) + float(bits) +
                 float(qual) + float(sizeof(long3)) + float(sizeof(bool4)) +
                 float(int(in[gid]));
    out[gid] = isum;
}

// simdgroup matrices.
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

// the rest of the warp/quad library.
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

kernel void grpsf(device const float* in [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  uint gid               [[thread_position_in_grid]]) {
    float s = simd_sum(in[gid]);
    float m = simd_max(in[gid]);
    float p = simd_product(in[gid] * 0.5f);
    out[gid] = s + m + p;
}

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

kernel void globk(device const float* in  [[buffer(0)]],
                  device float* out       [[buffer(1)]],
                  uint gid                [[thread_position_in_grid]]) {
    out[gid] = in[gid] * 3.0f + 1.0f;
}

// packed and narrow vectors.
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

kernel void counts(device float* out [[buffer(0)]],
                   uint gid  [[thread_position_in_grid]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tpg  [[threadgroups_per_grid]]) {
    out[gid] = as_type<float>((tpg & 0xffffu) | ((tgid & 0xffffu) << 16));
}

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

// a fragment stage.
fragment float4 fragshade(float4 pos [[position]],
                          device const float* in [[buffer(0)]],
                          texture2d<float> img [[texture(0)]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    return float4(in[y * 64u + x] * 2.0f + img.read(uint2(x, y)).x, 0.0f, 0.0f, 1.0f);
}

// A target that is neither square nor a multiple of anything: 100x40.
fragment float4 fragwide(float4 pos [[position]],
                         device const float* in [[buffer(0)]]) {
    const uint x = (uint)pos.x, y = (uint)pos.y;
    return float4(in[y * 100u + x] + (float)x * 0.5f + (float)y, 0.0f, 0.0f, 1.0f);
}

kernel void triton_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out       [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    const uint n = 1000u;
    if (gid < n) out[gid] = a[gid] + b[gid];
}

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

static inline float tf32r(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    if ((v.u & 0x7F800000u) == 0x7F800000u) return x;  // inf and NaN are unchanged
    v.u += 0x1000u + ((v.u >> 13) & 1u);
    v.u &= 0xFFFFE000u;
    return v.f;
}

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

static inline float bf16r(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    if ((v.u & 0x7F800000u) != 0x7F800000u) v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    v.u &= 0xFFFF0000u;
    return v.f;
}

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

kernel void bsum(device const float* in [[buffer(0)]],
                 device float* out       [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    float v = in[gid];
    for (uint d = 16u; d >= 1u; d >>= 1) v += simd_shuffle_down(v, d);
    out[gid] = v;
}

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

// `pred`: the value under @%p.
kernel void pred(device const uint* in [[buffer(0)]],
                 device uint* out       [[buffer(1)]],
                 uint gid [[thread_position_in_grid]],
                 uint lid [[thread_index_in_threadgroup]])
{
    const uint v = in[gid];
    out[gid] = ((lid & 1u) == 0u) ? (v * 3u + 1u + 2u) : v;
}

kernel void rotate(device const uint* in [[buffer(0)]],
                   device uint* out       [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_index_in_threadgroup]])
{
    out[gid] = simd_shuffle(in[gid], (lid + 1u) & 31u);
}

kernel void localtype(device const float* in [[buffer(0)]],
                      device float* out      [[buffer(1)]],
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]])
{
    struct Tap { float w; float v; };
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    Tap t;
    t.w = in[gid];
    t.v = t.w * 3.0f + 1.0f;
    tiles[lid] = t.w + t.v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Tap u = t;
    const float peer = tiles[(lid + 1u) % n];
    u.w += peer;
    out[gid] = u.w * 0.5f + u.v - peer;
}

kernel void lambdak(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid [[thread_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    tiles[lid] = v + 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a = ([](float v) { return v * 3.0f + 1.0f; })(tiles[(lid + 1u) % n]);
    out[gid] = a + [&](float x) { return x + v * 0.5f; }(tiles[lid]);
}

kernel void lamstore(device const float* in [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     uint gid [[thread_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    auto f = [&](float x) { return x * 2.0f + v; };
    tiles[lid] = v * 0.5f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer * 3.0f;
    out[gid] = f(peer) + v;
}

// the three capture-list shapes.
kernel void lamcap(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    float w = v * 2.0f;
    tiles[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    out[gid] = [=](float x) { return x * v + w; }(peer + 1.0f)
             + [&, w](float x) { return x * 2.0f + w + v; }(peer)
             + [v](float x) { return x + v * 3.0f; }(peer);
}

// a body-declared type with a method.
kernel void localmeth(device const float* in [[buffer(0)]],
                      device float* out      [[buffer(1)]],
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]])
{
    struct Tap { float w; float twice() const { return w * 2.0f; } };
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    Tap t;
    t.w = in[gid];
    tiles[lid] = t.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Tap u = t;
    out[gid] = u.twice() + tiles[(lid + 1u) % n];
}

kernel void stcopy(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    auto f = [=](float x) { return x * 2.0f + v; };
    tiles[lid] = v * 0.5f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer * 3.0f;
    out[gid] = f(peer) + v;
}

kernel void stname(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    float w = v + 7.0f;
    auto g = [v, &w](float x) { return x + v * 2.0f + w; };
    tiles[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer;
    w = peer * 5.0f;
    out[gid] = g(peer) + v + w;
}

// A stored lambda that copies an array local.
kernel void starray(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid [[thread_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float taps[4];
    for (uint i = 0u; i < 4u; i++) taps[i] = in[(gid + i) % 1048576u];
    auto h = [=](float x) { return x * taps[0] + taps[3] - taps[1]; };
    tiles[lid] = taps[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < 4u; i++) taps[i] = tiles[(lid + i) % n];
    out[gid] = h(tiles[lid]);
}

kernel void stinit(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    auto f = [t = v * 2.0f, u = lid + 1u](float x) { return x + t + (float)u; };
    tiles[lid] = v * 0.5f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer * 3.0f;
    out[gid] = f(peer) + v;
}

// a closure that writes its own copy.
kernel void stmut(device const float* in [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  uint gid [[thread_position_in_grid]],
                  uint lid [[thread_position_in_threadgroup]],
                  uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float w = in[gid];
    auto f = [w](float x) mutable { w += x; return w * 2.0f; };
    tiles[lid] = f(1.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    w = peer;
    out[gid] = f(peer) + w;
}

// a closure that reads a kernel parameter.
kernel void stparm(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    auto f = [=](float x) { return x + v * 2.0f + (float)lid + (float)tgs; };
    tiles[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer * 5.0f;
    out[gid] = f(peer) + v;
}

// a lambda inside a stored lambda.
kernel void stnest(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    auto f = [&](float x) {
        auto g = [v](float y) { return y * v; };
        auto h = [&](float y) { return y + v; };
        return g(x) + h(x);
    };
    tiles[lid] = v * 0.25f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer * 7.0f;
    out[gid] = f(peer) + v;
}

kernel void autolocal(device const float* in [[buffer(0)]],
                      device float* out      [[buffer(1)]],
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    auto v = in[gid] * 2.0f;
    auto k = lid + 3u;
    auto m = v * 0.5f + (float)k;
    tiles[lid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer;
    out[gid] = m + v + (float)k;
}

kernel void arrinit(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid [[thread_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float taps[4] = {in[gid], in[(gid + 1u) % 1048576u], 3.0f};
    tiles[lid] = taps[0] + taps[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    taps[0] = peer;
    taps[1] = peer * 2.0f;
    taps[2] = peer * 5.0f;
    out[gid] = taps[0] + taps[1] + taps[2] + taps[3];
}

// a capture that binds a local under another name.
kernel void refcap(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    auto f = [&q = v](float x) { return x * 2.0f + q; };
    tiles[lid] = v * 0.5f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    v = peer * 3.0f;
    out[gid] = f(peer) + v;
}

// a name declared again inside a nested scope.
kernel void shadow(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float v = in[gid];
    tiles[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    {
        float v = peer * 7.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tiles[lid] = v;
        out[gid] = v + peer;
    }
    out[gid] += v + tiles[lid];
}

// address spaces on locals.
kernel void addrspace(device const float* in [[buffer(0)]],
                      constant float* w      [[buffer(1)]],
                      device float* out      [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    device float* o = out;
    constant float* r = w;
    thread float t = r[gid] * 2.0f + in[gid];
    o[gid] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tiles[lid] = o[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[gid] = tiles[(lid + 1u) % n] + t;
}

// a two-dimensional array local.
kernel void arr2d(device const float* in [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  uint gid [[thread_position_in_grid]],
                  uint lid [[thread_position_in_threadgroup]],
                  uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float m[2][2] = {{in[gid], 2.0f}, {3.0f, 4.0f}};
    tiles[lid] = m[0][0] + m[1][1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    m[0][0] = peer;
    m[0][1] = peer * 2.0f;
    m[1][0] = peer * 3.0f;
    m[1][1] = peer * 5.0f;
    out[gid] = m[0][0] + m[0][1] + m[1][0] + m[1][1];
}

// a capture bound to an element.
kernel void refsub(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    float taps[4] = {in[gid], in[gid] * 0.5f, 0.0f, 0.0f};
    auto f = [&q = taps[1]](float x) { return x * 3.0f + q; };
    tiles[lid] = taps[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer = tiles[(lid + 1u) % n];
    taps[1] = peer * 7.0f;
    out[gid] = f(peer) + taps[1];
}

kernel void tgblock(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint gid [[thread_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float tiles[256];
    const uint n = min(tgs, 256u);
    tiles[lid] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        threadgroup float more[256];
        more[lid] = tiles[(lid + 1u) % n] * 2.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out[gid] = more[(lid + 1u) % n] + tiles[lid];
    }
}

kernel void packround(device float* packed   [[buffer(0)]],
                      device float* unpacked [[buffer(1)]],
                      uint gid [[thread_position_in_grid]])
{
    const float a = (float)(gid % 256u) / 255.0f;             // every byte boundary
    const float b = ((float)(gid % 128u) + 0.5f) / 127.0f;    // exact ties
    const float c = -1.0f + (float)(gid % 201u) * 0.01f;      // snorm's negative half
    const float w = (float)(gid % 64u) * 0.017f + 0.001f;
    const float4 v = float4(a, b, c, w);
    const uint pu8 = pack_float_to_unorm4x8(v);
    const uint ps8 = pack_float_to_snorm4x8(v);
    const uint pu16 = pack_float_to_unorm2x16(float2(v.x, v.y));
    const uint ps16 = pack_float_to_snorm2x16(float2(v.x, v.y));
    packed[gid * 4u + 0u] = as_type<float>(pu8);
    packed[gid * 4u + 1u] = as_type<float>(ps8);
    packed[gid * 4u + 2u] = as_type<float>(pu16);
    packed[gid * 4u + 3u] = as_type<float>(ps16);
    const float4 du8 = unpack_unorm4x8_to_float(pu8);
    const float4 ds8 = unpack_snorm4x8_to_float(ps8);
    const float2 du16 = unpack_unorm2x16_to_float(pu16);
    const float2 ds16 = unpack_snorm2x16_to_float(ps16);
    unpacked[gid * 12u + 0u] = du8.x;
    unpacked[gid * 12u + 1u] = du8.y;
    unpacked[gid * 12u + 2u] = du8.z;
    unpacked[gid * 12u + 3u] = du8.w;
    unpacked[gid * 12u + 4u] = ds8.x;
    unpacked[gid * 12u + 5u] = ds8.y;
    unpacked[gid * 12u + 6u] = ds8.z;
    unpacked[gid * 12u + 7u] = ds8.w;
    unpacked[gid * 12u + 8u] = du16.x;
    unpacked[gid * 12u + 9u] = du16.y;
    unpacked[gid * 12u + 10u] = ds16.x;
    unpacked[gid * 12u + 11u] = ds16.y;
}

kernel void choosemask(device const float* in  [[buffer(0)]],
                       device float* out       [[buffer(1)]],
                       uint gid [[thread_position_in_grid]])
{
    const float4 v = float4(in[gid * 4u + 0u], in[gid * 4u + 1u],
                            in[gid * 4u + 2u], in[gid * 4u + 3u]);
    const float3 lo = float3(v.x, v.y, v.z);
    const float3 hi = float3(v.w, v.y + 1.0f, v.x - 1.0f);
    const bool3 mask = lo > hi;
    const float3 a = select(lo, hi, mask);        // a mask picks component by component
    const float3 b = select(lo, hi, v.x > 0.0f);  // one bool picks all three
    out[gid * 8u + 0u] = a.x;
    out[gid * 8u + 1u] = a.y;
    out[gid * 8u + 2u] = a.z;
    out[gid * 8u + 3u] = b.x;
    out[gid * 8u + 4u] = b.y;
    out[gid * 8u + 5u] = b.z;
    out[gid * 8u + 6u] = any(mask) ? 1.0f : 0.0f;
    out[gid * 8u + 7u] = all(lo >= float3(0.0f)) ? 3.0f : 0.0f;
}

// swizzles: a read of part of a value, in any order.
kernel void swizzle(device const float* in  [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint gid [[thread_position_in_grid]])
{
    const float4 v = float4(in[gid * 4u + 0u], in[gid * 4u + 1u],
                            in[gid * 4u + 2u], in[gid * 4u + 3u]);
    const float2 a = v.xy;    // a prefix
    const float3 b = v.zyx;   // three, in the other order
    const float3 c = v.xxx;   // one component, broadcast
    const float2 d = v.wz;    // two that are not adjacent
    const float3 e = b.xyx;   // a swizzle of a swizzle
    out[gid * 8u + 0u] = a.x;
    out[gid * 8u + 1u] = a.y;
    out[gid * 8u + 2u] = b.x;
    out[gid * 8u + 3u] = c.y;
    out[gid * 8u + 4u] = d.x;
    out[gid * 8u + 5u] = d.y;
    out[gid * 8u + 6u] = e.z;
    out[gid * 8u + 7u] = b.z;
}

kernel void swzwrite(device const float* in  [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    float4 v = float4(in[gid], 2.0f, 3.0f, 4.0f);
    v.yw = float2(v.x * 2.0f,
                  v.x * 3.0f);   // y = 2x, w = 3x
    v.xz = v.wy;             // in place: x = 3x, z = 2x

    float3 t = float3(1.0f, 2.0f, 3.0f);
    t.zyx = float3(v.x, v.y,
                   v.z);         // three components, the other order

    out[gid * 8u + 0u] = v.x;
    out[gid * 8u + 1u] = v.y;
    out[gid * 8u + 2u] = v.z;
    out[gid * 8u + 3u] = v.w;
    out[gid * 8u + 4u] = t.x;
    out[gid * 8u + 5u] = t.y;
    out[gid * 8u + 6u] = t.z;
// One component of a swizzle is a plain member store and needs no rewrite.
    float4 u = v;
    u.w = t.x;
    out[gid * 8u + 7u] = u.w;
}
