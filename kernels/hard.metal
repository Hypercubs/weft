// Two kernels that used to be impossible to check, kept as fixtures because
// they are the hardest shapes here: one is a real shader-style workload whose
// answer is nothing but transcendental functions, the other is numerically
// sensitive and synchronises nine times. Both are compared byte for byte
// against the GPU like every other kernel.

#include <metal_stdlib>
using namespace metal;

// ---- a raymarcher ----
//
// This is the shape a portable math library has to get right: one wrong bit in
// `normalize` or `sqrt` sends the ray somewhere else, and after 64 steps the
// answer is not slightly off, it is a different picture. Before the library
// existed this kernel could be run on the CPU but not compared — its GPU answer
// came from hardware approximations no CPU reproduces.

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

    // `ro + rd * t` written as an fma per component. The Metal compiler contracts
    // that product into the add; a translated CPU compiler is free not to, and a
    // single ulp there is multiplied by the 64-step march. An explicit fma is the
    // same instruction on both engines, so this is the one way to make the two
    // answers identical rather than merely close.
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
    // A highlight as repeated squaring rather than `pow(s, 24)`: the GPU folds a
    // constant exponent into multiplies but only for the exponents it was
    // measured on (2, 3, 4), and `pow` at 24 is a general call — 3.6e-05 apart
    // from any CPU implementation. Written out, both engines run the same four
    // multiplies.
    const float s = spec < 0.0f ? 0.0f : (spec > 1.0f ? 1.0f : spec);
    const float s2 = s * s, s4 = s2 * s2, s8 = s4 * s4, s16 = s8 * s8;
    const float shine = (s16 * s8) * s;
    const float fog = 1.0f - smoothstep(6.0f, 20.0f, t);
    out[gid] = hit * (mix(0.08f, diff, 0.85f) + shine * 0.9f) * fog + (1.0f - hit) * 0.02f;
}

// ---- a radix-2 FFT ----
//
// 256 points per threadgroup: a bit-reversed load, then eight butterfly stages,
// each one a barrier and a twiddle factor from the portable `sin`/`cos`. A scan
// is one tree; this is a permuted tree with a transcendental at every node, so
// it fails loudly if either the lane mapping or the trig is a bit off.

kernel void fft(device const float* in  [[buffer(0)]],
                device float* out       [[buffer(1)]],
                uint gid    [[thread_position_in_grid]],
                uint lid    [[thread_position_in_threadgroup]])
{
    // Named `fr`/`fi` rather than `re`/`im`: the portable math preamble owns the
    // short names, and a collision with one of its declarations is a compile
    // error that reads like a type problem.
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
            // Explicit fma for the same reason as the raymarcher: `br*wr - bi*wi`
            // is one rounding on the GPU and two on a CPU that does not contract
            // it, and this kernel has 8 stages of them.
            const float tr = fma(br, wr, -(bi * wi));
            const float ti = fma(br, wi, bi * wr);
            fr[a] = ar + tr;  fi[a] = ai + ti;
            fr[b] = ar - tr;  fi[b] = ai - ti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Power and phase per point: a conjugate-symmetric pair is visible in the
    // dump, and a swapped lane shows up as a different number rather than as a
    // sum that happens to still add up.
    out[2u * gid + 0u] = fr[lid] * fr[lid] + fi[lid] * fi[lid];
    out[2u * gid + 1u] = atan2(fi[lid], fr[lid]);
}
