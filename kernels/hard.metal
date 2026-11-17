
#include <metal_stdlib>
using namespace metal;

static float sd_sphere(float3 p, float3 c, float r) {
    return length(p - c) - r;
}

static float sd_torus(float3 p, float2 t) {
    float2 q = float2(length(float2(p.x, p.z)) - t.x, p.y);
    return length(q) - t.y;
}

static float smaller(float a, float b) {
    return a < b ? a : b;
}

static float scene(float3 p) {
// A sphere resting inside a torus, plus a ground plane.
    float s = sd_sphere(p, float3(0.0f, 0.15f, -3.0f), 0.75f);
    float t = sd_torus(float3(p.x, p.y - 0.15f, p.z + 3.0f), float2(1.25f, 0.22f));
    return smaller(smaller(s, t), p.y + 1.0f);
}

static float3 normal_at(float3 p) {
    const float e = 0.0015f;
    float3 dx = float3(e, 0.0f, 0.0f), dy = float3(0.0f, e, 0.0f), dz = float3(0.0f, 0.0f, e);
    return normalize(float3(scene(p + dx) - scene(p - dx),
                            scene(p + dy) - scene(p - dy),
                            scene(p + dz) - scene(p - dz)));
}

kernel void raymarch(device const float* in [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    const float seed = in[gid];
    const float3 ro = float3(0.0f, 0.35f, 2.0f);
    const float3 rd = normalize(float3((seed - 1.0f) * 1.1f,
                                       (seed - 1.0f) * 0.7f - 0.12f,
                                       -1.0f));

    float t = 0.0f;
    float hit = 0.0f;
    for (uint i = 0u; i < 64u; i++) {
        const float3 pos = float3(fma(rd.x, t, ro.x), fma(rd.y, t, ro.y), fma(rd.z, t, ro.z));
        const float d = scene(pos);
        if (d < 0.0015f) { hit = 1.0f; break; }
        t += d;
        if (t > 40.0f) break;
    }

    const float3 p = float3(fma(rd.x, t, ro.x), fma(rd.y, t, ro.y), fma(rd.z, t, ro.z));
    const float3 n = normal_at(p);
    const float3 l = normalize(float3(0.6f, 0.8f, 0.5f));
    const float ndl = dot(n, l);
    const float spec = dot(reflect(-l, n), -rd);
    const float diff = ndl < 0.0f ? 0.0f : (ndl > 1.0f ? 1.0f : ndl);
    const float s = spec < 0.0f ? 0.0f : (spec > 1.0f ? 1.0f : spec);
    const float s2 = s * s, s4 = s2 * s2, s8 = s4 * s4, s16 = s8 * s8;
    const float shine = (s16 * s8) * s;
    const float fog = 1.0f - smoothstep(6.0f, 20.0f, t);
    out[gid] = hit * (mix(0.08f, diff, 0.85f) + shine * 0.9f) * fog + (1.0f - hit) * 0.02f;
}

kernel void fft(device const float* in  [[buffer(0)]],
                device float* out       [[buffer(1)]],
                uint gid    [[thread_position_in_grid]],
                uint lid    [[thread_position_in_threadgroup]])
{
    threadgroup float fr[256];
    threadgroup float fi[256];

    const uint n = 256u;

// Bit reversal, eight bits, written as a loop so the mapping is visible.
    uint j = 0u, x = lid;
    for (uint b = 0u; b < 8u; b++) { j = (j << 1) | (x & 1u); x >>= 1u; }
    fr[j] = in[2u * gid + 0u];
    fi[j] = in[2u * gid + 1u];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint len = 2u; len <= n; len <<= 1u) {
        const uint span = len >> 1u;   // not `half`: that is a type name in MSL
        const float ang = -6.283185307179586f / float(len);
        if (lid < n / 2u) {
// One butterfly per lane: k indexes the n/2 butterflies of the stage.
            const uint k = lid;
            const uint grp = k / span, off = k % span;
            const uint a = grp * len + off, b = a + span;
            const float wr = cos(ang * float(off));
            const float wi = sin(ang * float(off));
            const float ar = fr[a], ai = fi[a];
            const float br = fr[b], bi = fi[b];
// Explicit fma for the same reason as the raymarcher.
            const float tr = fma(br, wr, -(bi * wi));
            const float ti = fma(br, wi, bi * wr);
            fr[a] = ar + tr;  fi[a] = ai + ti;
            fr[b] = ar - tr;  fi[b] = ai - ti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[2u * gid + 0u] = fr[lid] * fr[lid] + fi[lid] * fi[lid];
    out[2u * gid + 1u] = atan2(fi[lid], fr[lid]);
}
