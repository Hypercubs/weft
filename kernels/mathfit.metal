#include <metal_stdlib>
using namespace metal;

kernel void fnprobe(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
  const float3 a = float3(in[6u*gid+0u], in[6u*gid+1u], in[6u*gid+2u]);
  const float3 b = float3(in[6u*gid+3u], in[6u*gid+4u], in[6u*gid+5u]);
  const float  u = fabs(a.x) + 0.5f;
  const float2 a2 = float2(a.x, a.y), b2 = float2(b.x, b.y);
  const float4 a4 = float4(a, 1.0f), b4 = float4(b, 1.0f);
  const uint o = gid*54u;   // weft_sincos_marker
  out[o+0u]  = sqrt(u);
  out[o+1u]  = 1.0f / u;
  out[o+2u]  = rsqrt(u);
  out[o+3u]  = u / (u + 1.0f);
  out[o+4u]  = exp2(u);
  out[o+5u]  = log2(u);
  out[o+6u]  = length(a2);
  out[o+7u]  = dot(a2, b2);
  out[o+8u]  = length(a4);
  out[o+9u]  = dot(a4, b4);
  out[o+10u] = mix(a.x, b.x, a.y);          // the builtin
  out[o+11u] = a.x + (b.x - a.x) * a.y;     // written form 1
  out[o+12u] = fma(a.y, b.x - a.x, a.x);    // written form 2
  out[o+13u] = a.x * (1.0f - a.y) + b.x * a.y;   // form 3
  out[o+14u] = pow(u, 3.0f);
  out[o+15u] = pow(u, 0.5f);
  out[o+16u] = sqrt(u) * sqrt(u);
  out[o+17u] = fma(u, u, 1.0f);
  out[o+18u] = 1.0f / sqrt(u);
  out[o+19u] = u * rsqrt(u);
  out[o+20u] = normalize(a2).x;
  out[o+21u] = distance(a2, b2);
  out[o+22u] = length(a);
  out[o+23u] = dot(a, a);
  out[o+24u] = sqrt(a.x * a.x + a.y * a.y);
  out[o+25u] = fma(a.y, a.y, a.x * a.x);
  out[o+26u] = 1.0f / sqrt(a.x * a.x + 1.0f);
  out[o+27u] = fma(1.5f, log2(u), 0.0f);
  out[o+28u] = exp2(1.5f * log2(u));
  out[o+29u] = mix(a.x, b.x, a.y) - fma(a.y, b.x - a.x, a.x);   // builtin vs fma form
  out[o+30u] = sqrt(u) - u * rsqrt(u);
  out[o+31u] = (a.x * a.x + a.y * a.y) - fma(a.y, a.y, a.x * a.x);
  // The rest of the approximated set: trigonometry, exp10/log10, powr, hyperbolic.
  out[o+32u] = sin(a.x);
  out[o+33u] = cos(a.y);
  out[o+34u] = tan(a.x * 0.5f);
  out[o+35u] = asin(clamp(a.x, -1.0f, 1.0f));
  out[o+36u] = acos(clamp(a.y, -1.0f, 1.0f));
  out[o+37u] = atan(a.z);
  out[o+38u] = atan2(a.y, a.x);
  out[o+39u] = sinh(a.x);
  out[o+40u] = cosh(a.y);
  out[o+41u] = tanh(a.x);
  out[o+42u] = asinh(a.z);
  out[o+43u] = acosh(fabs(a.x) + 1.0f);
  out[o+44u] = atanh(a.x * 0.5f);
  out[o+45u] = exp10(a.y);
  out[o+46u] = log10(fabs(a.x) + 0.5f);
  out[o+47u] = powr(fabs(a.x) + 0.5f, a.y);
  // The geometric compositions, whose multiply-adds could fuse differently.
  out[o+48u] = cross(a, b).x;
  out[o+49u] = reflect(normalize(a), normalize(b)).x;
  out[o+50u] = refract(normalize(a), normalize(b), 0.75f).x;
  out[o+51u] = faceforward(a, b, a).z;
  // The one function that returns through pointers.
  float sc_s = 0.0f, sc_c = 0.0f;
  sincos(a.x, &sc_s, &sc_c);
  out[o+52u] = sc_s + sc_c;
  // A qualified call: the rewrite drops the namespace, because the portable
  // function is what both engines should run.
  out[o+53u] = metal::cos(a.z);
}
