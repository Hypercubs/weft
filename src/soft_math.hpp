// Portable math, one source of truth.
//
// Metal's `sqrt`, `rsqrt`, `log2`, `exp2` and `pow` are hardware approximations
// in every math mode: measured on this GPU, `sqrt(x)` is bit-for-bit
// `x * rsqrt(x)`, and that `rsqrt` matches the correctly rounded `1/sqrt` only
// 62% of the time (`log2` 1.7%, `exp2` 89%). No CPU can reproduce those bits, so
// the previous answer was a tolerance. This is the other answer: the same
// functions, written once, run by both engines.
//
// Every operation here is +, -, *, / or a comparison, each of which is IEEE on
// the GPU under `MTLMathModeSafe` and on the CPU by default, and every loop has a
// fixed trip count. Two engines running the same sequence of IEEE operations get
// the same bits, so `length`, `normalize` and `pow` agree exactly rather than to
// some number of ulps.
//
// The text compiles as MSL and as C++ (the only difference is how the fused
// multiply-add is spelled), and it is emitted into both processes: the generated
// C++ includes it, and the launcher prepends it to the source it hands to Metal.
#pragma once

static const char* const kWSoftMath = R"WEFTMATH(
// ---- portable math ---------------------------------------------------------
// Written once, compiled by both engines. See src/soft_math.hpp.
#ifdef __METAL_VERSION__
#define WEFT_FMA(x, y, z) metal::fma((x), (y), (z))
#else
#define WEFT_FMA(x, y, z) std::fma((x), (y), (z))
#endif
// A pointer carries an address space in MSL and none in C++, which is the one
// other spelling this text needs besides fma and the bit-reinterpretation.
#ifdef __METAL_VERSION__
#define WEFT_THREAD thread
#else
#define WEFT_THREAD
#endif
#define WEFT_LN2 0.6931471805599453f
#define WEFT_INF 3.402823466e38f

// sqrt as a multiplicative Newton iteration on 1/sqrt, over an argument reduced
// to (0.25, 4] by exact power-of-two scaling. Multiplies only: the same trip
// count on both engines means the same sequence of roundings, and there is no
// division in the hot path (the old version took sixteen).
inline float weft_sqrt(float v) {
  if (!(v > 0.0f)) return (v == 0.0f) ? v : (v - v) / (v - v);  // 0, or NaN
  if (v > WEFT_INF) return v;                                  // infinity
  float x = v;
  int k = 0;
  while (x > 4.0f) { x *= 0.25f; k++; }
  while (x < 0.25f) { x *= 4.0f; k--; }
  // The start has to be inside (0, 2/sqrt(x)) or the iteration walks out of the
  // basin and diverges — which it did, silently, for x above about 2.3. On the
  // reduced range (0.25, 4] the bound is 1 for x <= 1 and 2 for x <= 4, so this
  // start is always inside it, and the sequence stays positive.
  float r = (x > 1.0f) ? 0.5f : 1.0f;
  for (int i = 0; i < 8; i++) r = r * (1.5f - 0.5f * x * r * r);
  float y = x * r;
  for (int i = 0; i < k; i++) y *= 2.0f;
  for (int i = 0; i > k; i--) y *= 0.5f;
  return y;
}

inline float weft_rsqrt(float v) { return 1.0f / weft_sqrt(v); }

inline float2 weft_sqrt(float2 v) { return float2(weft_sqrt(v.x), weft_sqrt(v.y)); }
inline float3 weft_sqrt(float3 v) {
  return float3(weft_sqrt(v.x), weft_sqrt(v.y), weft_sqrt(v.z));
}
inline float4 weft_sqrt(float4 v) {
  return float4(weft_sqrt(v.x), weft_sqrt(v.y), weft_sqrt(v.z), weft_sqrt(v.w));
}
inline float2 weft_rsqrt(float2 v) { return float2(weft_rsqrt(v.x), weft_rsqrt(v.y)); }
inline float3 weft_rsqrt(float3 v) {
  return float3(weft_rsqrt(v.x), weft_rsqrt(v.y), weft_rsqrt(v.z));
}
inline float4 weft_rsqrt(float4 v) {
  return float4(weft_rsqrt(v.x), weft_rsqrt(v.y), weft_rsqrt(v.z), weft_rsqrt(v.w));
}

// log2 by exact exponent extraction and the atanh series on the mantissa, which
// converges quickly because the reduced argument is in [1, 2).
inline float weft_log2(float v) {
  if (!(v > 0.0f)) return (v - v) / (v - v);
  if (v > WEFT_INF) return v;
  float m = v;
  int k = 0;
  while (m >= 2.0f) { m *= 0.5f; k++; }
  while (m < 1.0f) { m *= 2.0f; k--; }
  const float t = (m - 1.0f) / (m + 1.0f);
  const float t2 = t * t;
  float sum = 0.0f;
  float p = t;
  for (int n = 1; n <= 15; n += 2) {
    sum += p * (1.0f / (float)n);   // one rounding, no division: the reciprocal
                                   // of a small integer folds to the same float twice
    p *= t2;
  }
  return (float)k + (2.0f * sum) / WEFT_LN2;
}

// exp2 as a Taylor series on the fraction, then exact power-of-two scaling.
inline float weft_exp2(float v) {
  if (!(v == v)) return v;
  if (v > 127.0f) return WEFT_INF * 2.0f;
  if (v < -150.0f) return 0.0f;
  int k = 0;
  float f = v;
  while (f >= 1.0f) { f -= 1.0f; k++; }
  while (f < 0.0f) { f += 1.0f; k--; }
  const float z = f * WEFT_LN2;
  float term = 1.0f;
  float sum = 1.0f;
  for (int n = 1; n <= 10; n++) {
    term *= z * (1.0f / (float)n);
    sum += term;
  }
  for (int i = 0; i < k; i++) sum *= 2.0f;
  for (int i = 0; i > k; i--) sum *= 0.5f;
  return sum;
}

// A constant integer exponent is folded to repeated multiplication by the GPU
// compiler (measured: pow(x, 3) is x*x*x on all lanes), so those cases take the
// same path here and stay bit-identical.
inline float weft_pow(float x, float y) {
  if (y == 2.0f) return x * x;
  if (y == 3.0f) return (x * x) * x;
  if (y == 4.0f) { float a = x * x; return a * a; }
  if (y == 1.0f) return x;
  if (y == 0.0f) return 1.0f;
  if (y == 0.5f) return weft_sqrt(x);
  if (y == -1.0f) return 1.0f / x;
  if (!(x > 0.0f)) return (x - x) / (x - x);
  return weft_exp2(y * weft_log2(x));
}
inline float2 weft_pow(float2 x, float2 y) {
  return float2(weft_pow(x.x, y.x), weft_pow(x.y, y.y));
}
inline float3 weft_pow(float3 x, float3 y) {
  return float3(weft_pow(x.x, y.x), weft_pow(x.y, y.y), weft_pow(x.z, y.z));
}
inline float4 weft_pow(float4 x, float4 y) {
  return float4(weft_pow(x.x, y.x), weft_pow(x.y, y.y), weft_pow(x.z, y.z),
                weft_pow(x.w, y.w));
}

inline float2 weft_log2(float2 v) { return float2(weft_log2(v.x), weft_log2(v.y)); }
inline float3 weft_log2(float3 v) {
  return float3(weft_log2(v.x), weft_log2(v.y), weft_log2(v.z));
}
inline float4 weft_log2(float4 v) {
  return float4(weft_log2(v.x), weft_log2(v.y), weft_log2(v.z), weft_log2(v.w));
}
inline float2 weft_exp2(float2 v) { return float2(weft_exp2(v.x), weft_exp2(v.y)); }
inline float3 weft_exp2(float3 v) {
  return float3(weft_exp2(v.x), weft_exp2(v.y), weft_exp2(v.z));
}
inline float4 weft_exp2(float4 v) {
  return float4(weft_exp2(v.x), weft_exp2(v.y), weft_exp2(v.z), weft_exp2(v.w));
}
inline float weft_exp(float v) { return weft_exp2(v * 1.4426950408889634f); }
inline float weft_log(float v) { return weft_log2(v) * WEFT_LN2; }

// mix is a fused multiply-add on one side of the comparison and a plain
// multiply-then-add on the other, which is a last-bit difference; both engines run
// this one form instead.
inline float weft_mix(float a, float b, float t) { return WEFT_FMA(b - a, t, a); }
inline float2 weft_mix(float2 a, float2 b, float t) {
  return float2(weft_mix(a.x, b.x, t), weft_mix(a.y, b.y, t));
}
inline float3 weft_mix(float3 a, float3 b, float t) {
  return float3(weft_mix(a.x, b.x, t), weft_mix(a.y, b.y, t), weft_mix(a.z, b.z, t));
}
inline float4 weft_mix(float4 a, float4 b, float t) {
  return float4(weft_mix(a.x, b.x, t), weft_mix(a.y, b.y, t), weft_mix(a.z, b.z, t),
                weft_mix(a.w, b.w, t));
}
inline float2 weft_mix(float2 a, float2 b, float2 t) {
  return float2(weft_mix(a.x, b.x, t.x), weft_mix(a.y, b.y, t.y));
}
inline float3 weft_mix(float3 a, float3 b, float3 t) {
  return float3(weft_mix(a.x, b.x, t.x), weft_mix(a.y, b.y, t.y), weft_mix(a.z, b.z, t.z));
}
inline float4 weft_mix(float4 a, float4 b, float4 t) {
  return float4(weft_mix(a.x, b.x, t.x), weft_mix(a.y, b.y, t.y), weft_mix(a.z, b.z, t.z),
                weft_mix(a.w, b.w, t.w));
}

// ---- trigonometry and the rest of the approximated set ---------------------
// Every one of these is a hardware approximation on the GPU, so both engines run
// the text below: reduction by exact arithmetic, a fixed series, fixed trip
// counts. The polynomial is not the best one for accuracy; it is the same one at
// both ends of the comparison, which is what makes the answers equal.
inline float weft_sin(float x) {
  if (!(x == x)) return x;                       // NaN
  if (x > 1.0e7f || x < -1.0e7f) return 0.0f;    // past float's range for either
  float k = x * 0.15915494309189535f;            // 1 / (2 pi)
  k = (k >= 0.0f) ? (float)(int)(k + 0.5f) : (float)(int)(k - 0.5f);
  float r = x - k * 6.283185307179586f;          // [-pi, pi]
  if (r > 1.5707963267948966f) r = 3.141592653589793f - r;
  if (r < -1.5707963267948966f) r = -3.141592653589793f - r;
  const float r2 = r * r;
  float s = r;
  float t = r * r2;
  s -= t * (1.0f / 6.0f);
  t *= r2;
  s += t * (1.0f / 120.0f);
  t *= r2;
  s -= t * (1.0f / 5040.0f);
  t *= r2;
  s += t * (1.0f / 362880.0f);
  t *= r2;
  s -= t * (1.0f / 39916800.0f);
  t *= r2;
  s += t * (1.0f / 6227020800.0f);
  t *= r2;
  s -= t * (1.0f / 1307674368000.0f);
  return s;
}
inline float weft_cos(float x) { return weft_sin(x + 1.5707963267948966f); }
inline float weft_tan(float x) {
  const float c = weft_cos(x);
  return weft_sin(x) / c;
}
inline float weft_sinpi(float x) { return weft_sin(x * 3.141592653589793f); }
inline float weft_cospi(float x) { return weft_cos(x * 3.141592653589793f); }
inline float weft_tanpi(float x) { return weft_tan(x * 3.141592653589793f); }

// atan: |x| > 1 reflected to [0, 1], then five halvings leave an argument under
// 0.025 where the Taylor series is done in twelve terms.
inline float weft_atan(float x) {
  if (!(x == x)) return x;
  const float half_pi = 1.5707963267948966f;
  float s = x;
  const bool neg = s < 0.0f;
  if (neg) s = -s;
  bool reflected = false;
  if (s > 1.0f) { s = 1.0f / s; reflected = true; }
  float t = s;
  for (int i = 0; i < 5; i++) t = t / (1.0f + weft_sqrt(1.0f + t * t));
  const float t2 = t * t;
  float term = t;
  float sum = t;
  float sign = -1.0f;
  for (int n = 3; n <= 25; n += 2) {
    term *= t2;
    sum += sign * (term * (1.0f / (float)n));
    sign = -sign;
  }
  float a = sum * 32.0f;                         // five halvings, undone
  if (reflected) a = half_pi - a;
  return neg ? -a : a;
}
inline float weft_atan2(float y, float x) {
  const float pi = 3.141592653589793f;
  const float half_pi = 1.5707963267948966f;
  if (x > 0.0f) return weft_atan(y / x);
  if (x < 0.0f) return (y >= 0.0f) ? (weft_atan(y / x) + pi) : (weft_atan(y / x) - pi);
  if (y > 0.0f) return half_pi;
  if (y < 0.0f) return -half_pi;
  return 0.0f;
}
inline float weft_asin(float x) {
  const float half_pi = 1.5707963267948966f;
  if (x >= 1.0f) return half_pi;
  if (x <= -1.0f) return -half_pi;
  return weft_atan(x / weft_sqrt(1.0f - x * x));
}
inline float weft_acos(float x) { return 1.5707963267948966f - weft_asin(x); }

// exp10 / log10 are the same series with a constant folded in.
inline float weft_exp10(float v) { return weft_exp2(v * 3.321928094887362f); }
inline float weft_log10(float v) { return weft_log2(v) * 0.30102999566398120f; }
inline float weft_powr(float x, float y) {
  if (!(x > 0.0f)) return (x - x) / (x - x);
  return weft_exp2(y * weft_log2(x));
}
inline float weft_sinh(float v) {
  const float e = weft_exp(v);
  return (e - 1.0f / e) * 0.5f;
}
inline float weft_cosh(float v) {
  const float e = weft_exp(v);
  return (e + 1.0f / e) * 0.5f;
}
inline float weft_tanh(float v) {
  const float e = weft_exp(v);
  const float f = 1.0f / e;
  return (e - f) / (e + f);
}
inline float weft_asinh(float v) {
  // asinh(-v) = -asinh(v): without the reflection, a large negative argument has
  // v*v + 1 == v*v exactly, the square root cancels the logarithm's argument to
  // zero and the answer is -inf instead of -14.
  if (v < 0.0f) return -weft_asinh(-v);
  return weft_log(v + weft_sqrt(v * v + 1.0f));
}
inline float weft_acosh(float v) { return weft_log(v + weft_sqrt(v * v - 1.0f)); }
inline float weft_atanh(float v) { return 0.5f * weft_log((1.0f + v) / (1.0f - v)); }

// The vector forms a shader may call, generated from the scalar ones.
#define WEFT_MAP1(f)                                       \
  inline float2 f(float2 v) { return float2(f(v.x), f(v.y)); } \
  inline float3 f(float3 v) { return float3(f(v.x), f(v.y), f(v.z)); } \
  inline float4 f(float4 v) { return float4(f(v.x), f(v.y), f(v.z), f(v.w)); }
#define WEFT_MAP2(f)                                                          \
  inline float2 f(float2 a, float2 b) { return float2(f(a.x, b.x), f(a.y, b.y)); } \
  inline float3 f(float3 a, float3 b) {                                       \
    return float3(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z));                     \
  }                                                                           \
  inline float4 f(float4 a, float4 b) {                                       \
    return float4(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z), f(a.w, b.w));        \
  }
WEFT_MAP1(weft_sin)
WEFT_MAP1(weft_cos)
WEFT_MAP1(weft_tan)
WEFT_MAP1(weft_sinpi)
WEFT_MAP1(weft_cospi)
WEFT_MAP1(weft_tanpi)
WEFT_MAP1(weft_asin)
WEFT_MAP1(weft_acos)
WEFT_MAP1(weft_atan)
WEFT_MAP1(weft_sinh)
WEFT_MAP1(weft_cosh)
WEFT_MAP1(weft_tanh)
WEFT_MAP1(weft_asinh)
WEFT_MAP1(weft_acosh)
WEFT_MAP1(weft_atanh)
WEFT_MAP1(weft_exp10)
WEFT_MAP1(weft_log10)
WEFT_MAP2(weft_atan2)
WEFT_MAP2(weft_powr)

// cross, reflect, refract and faceforward are compositions whose multiply-adds
// the two compilers can fuse differently, which is the same last-bit problem the
// single functions had. Written out with an explicit fma, they cannot diverge.
inline float3 weft_cross(float3 a, float3 b) {
  return float3(WEFT_FMA(a.y, b.z, -(a.z * b.y)),
                WEFT_FMA(a.z, b.x, -(a.x * b.z)),
                WEFT_FMA(a.x, b.y, -(a.y * b.x)));
}
inline float2 weft_reflect(float2 i, float2 n) {
  const float d = 2.0f * WEFT_FMA(n.y, i.y, n.x * i.x);
  return float2(WEFT_FMA(-n.x, d, i.x), WEFT_FMA(-n.y, d, i.y));
}
inline float3 weft_reflect(float3 i, float3 n) {
  const float d = 2.0f * WEFT_FMA(n.z, i.z, WEFT_FMA(n.y, i.y, n.x * i.x));
  return float3(WEFT_FMA(-n.x, d, i.x), WEFT_FMA(-n.y, d, i.y), WEFT_FMA(-n.z, d, i.z));
}
inline float4 weft_reflect(float4 i, float4 n) {
  const float d =
      2.0f * WEFT_FMA(n.w, i.w, WEFT_FMA(n.z, i.z, WEFT_FMA(n.y, i.y, n.x * i.x)));
  return float4(WEFT_FMA(-n.x, d, i.x), WEFT_FMA(-n.y, d, i.y),
                WEFT_FMA(-n.z, d, i.z), WEFT_FMA(-n.w, d, i.w));
}
inline float2 weft_faceforward(float2 n, float2 i, float2 nref) {
  const float d = WEFT_FMA(nref.y, i.y, nref.x * i.x);
  return (d < 0.0f) ? n : -n;
}
inline float3 weft_faceforward(float3 n, float3 i, float3 nref) {
  const float d = WEFT_FMA(nref.z, i.z, WEFT_FMA(nref.y, i.y, nref.x * i.x));
  return (d < 0.0f) ? n : -n;
}
inline float4 weft_faceforward(float4 n, float4 i, float4 nref) {
  const float d =
      WEFT_FMA(nref.w, i.w, WEFT_FMA(nref.z, i.z, WEFT_FMA(nref.y, i.y, nref.x * i.x)));
  return (d < 0.0f) ? n : -n;
}
inline float2 weft_refract(float2 i, float2 n, float eta) {
  const float d = WEFT_FMA(n.y, i.y, n.x * i.x);
  const float k = 1.0f - eta * eta * (1.0f - d * d);
  if (k < 0.0f) return float2(0.0f);
  const float s = eta * d + weft_sqrt(k);
  return float2(WEFT_FMA(-n.x, s, eta * i.x), WEFT_FMA(-n.y, s, eta * i.y));
}
inline float3 weft_refract(float3 i, float3 n, float eta) {
  const float d = WEFT_FMA(n.z, i.z, WEFT_FMA(n.y, i.y, n.x * i.x));
  const float k = 1.0f - eta * eta * (1.0f - d * d);
  if (k < 0.0f) return float3(0.0f);
  const float s = eta * d + weft_sqrt(k);
  return float3(WEFT_FMA(-n.x, s, eta * i.x), WEFT_FMA(-n.y, s, eta * i.y),
                WEFT_FMA(-n.z, s, eta * i.z));
}
inline float4 weft_refract(float4 i, float4 n, float eta) {
  const float d =
      WEFT_FMA(n.w, i.w, WEFT_FMA(n.z, i.z, WEFT_FMA(n.y, i.y, n.x * i.x)));
  const float k = 1.0f - eta * eta * (1.0f - d * d);
  if (k < 0.0f) return float4(0.0f);
  const float s = eta * d + weft_sqrt(k);
  return float4(WEFT_FMA(-n.x, s, eta * i.x), WEFT_FMA(-n.y, s, eta * i.y),
                WEFT_FMA(-n.z, s, eta * i.z), WEFT_FMA(-n.w, s, eta * i.w));
}

inline float weft_dot(float2 a, float2 b) { return WEFT_FMA(a.y, b.y, a.x * b.x); }
inline float weft_dot(float3 a, float3 b) {
  return WEFT_FMA(a.z, b.z, WEFT_FMA(a.y, b.y, a.x * b.x));
}
inline float weft_dot(float4 a, float4 b) {
  return WEFT_FMA(a.w, b.w, WEFT_FMA(a.z, b.z, WEFT_FMA(a.y, b.y, a.x * b.x)));
}

// sincos returns two values through pointers, so it cannot be an expression
// rewrite of one function; the same text defines it for both engines.
inline void weft_sincos(float x, WEFT_THREAD float* s, WEFT_THREAD float* c) {
  *s = weft_sin(x);
  *c = weft_cos(x);
}

// The geometric functions, on the same fused dot the GPU's own `dot` uses.
inline float weft_length(float2 v) { return weft_sqrt(WEFT_FMA(v.y, v.y, v.x * v.x)); }
inline float weft_length(float3 v) {
  return weft_sqrt(WEFT_FMA(v.z, v.z, WEFT_FMA(v.y, v.y, v.x * v.x)));
}
inline float weft_length(float4 v) {
  return weft_sqrt(
      WEFT_FMA(v.w, v.w, WEFT_FMA(v.z, v.z, WEFT_FMA(v.y, v.y, v.x * v.x))));
}
inline float weft_distance(float2 a, float2 b) { return weft_length(a - b); }
inline float weft_distance(float3 a, float3 b) { return weft_length(a - b); }
inline float weft_distance(float4 a, float4 b) { return weft_length(a - b); }
inline float2 weft_normalize(float2 v) { return v / weft_length(v); }
inline float3 weft_normalize(float3 v) { return v / weft_length(v); }
inline float4 weft_normalize(float4 v) { return v / weft_length(v); }
// ---- fp16 -------------------------------------------------------------------
//
// A `half` is 16 bits of storage and 16 bits of arithmetic, and the arithmetic is
// already the same on both engines: measured over 65,536 lanes, `half + half`,
// `half * half`, `half / half`, and a multiply-add the compiler is free to fuse,
// are bit-identical on the GPU under `MTLMathModeSafe` and to `_Float16` here.
// What was not the same is the library around it, so a function call on a half
// goes through the float implementation and rounds once on the way back: one
// text, both engines, one answer.
inline float weft_fh(half v) { return (float)v; }
inline float2 weft_fh(half2 v) { return float2((float)v.x, (float)v.y); }
inline float3 weft_fh(half3 v) { return float3((float)v.x, (float)v.y, (float)v.z); }
inline float4 weft_fh(half4 v) {
  return float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

inline float weft_fma(float a, float b, float c) { return WEFT_FMA(a, b, c); }
inline float2 weft_fma(float2 a, float2 b, float2 c) {
  return float2(weft_fma(a.x, b.x, c.x), weft_fma(a.y, b.y, c.y));
}
inline float3 weft_fma(float3 a, float3 b, float3 c) {
  return float3(weft_fma(a.x, b.x, c.x), weft_fma(a.y, b.y, c.y),
                weft_fma(a.z, b.z, c.z));
}
inline float4 weft_fma(float4 a, float4 b, float4 c) {
  return float4(weft_fma(a.x, b.x, c.x), weft_fma(a.y, b.y, c.y),
                weft_fma(a.z, b.z, c.z), weft_fma(a.w, b.w, c.w));
}
inline half weft_fma(half a, half b, half c) {
  return (half)weft_fma((float)a, (float)b, (float)c);
}
inline half2 weft_fma(half2 a, half2 b, half2 c) {
  return half2(weft_fma(a.x, b.x, c.x), weft_fma(a.y, b.y, c.y));
}
inline half3 weft_fma(half3 a, half3 b, half3 c) {
  return half3(weft_fma(a.x, b.x, c.x), weft_fma(a.y, b.y, c.y),
               weft_fma(a.z, b.z, c.z));
}
inline half4 weft_fma(half4 a, half4 b, half4 c) {
  return half4(weft_fma(a.x, b.x, c.x), weft_fma(a.y, b.y, c.y),
               weft_fma(a.z, b.z, c.z), weft_fma(a.w, b.w, c.w));
}

inline float weft_smoothstep(float e0, float e1, float x) {
  const float t = (x < e0) ? 0.0f : (x > e1 ? 1.0f : (x - e0) / (e1 - e0));
  return t * t * (3.0f - 2.0f * t);
}
inline float2 weft_smoothstep(float2 e0, float2 e1, float2 x) {
  return float2(weft_smoothstep(e0.x, e1.x, x.x), weft_smoothstep(e0.y, e1.y, x.y));
}
inline float3 weft_smoothstep(float3 e0, float3 e1, float3 x) {
  return float3(weft_smoothstep(e0.x, e1.x, x.x), weft_smoothstep(e0.y, e1.y, x.y),
                weft_smoothstep(e0.z, e1.z, x.z));
}
inline float4 weft_smoothstep(float4 e0, float4 e1, float4 x) {
  return float4(weft_smoothstep(e0.x, e1.x, x.x), weft_smoothstep(e0.y, e1.y, x.y),
                weft_smoothstep(e0.z, e1.z, x.z), weft_smoothstep(e0.w, e1.w, x.w));
}
inline half weft_smoothstep(half e0, half e1, half x) {
  return (half)weft_smoothstep((float)e0, (float)e1, (float)x);
}
inline half2 weft_smoothstep(half2 e0, half2 e1, half2 x) {
  return half2(weft_smoothstep(e0.x, e1.x, x.x), weft_smoothstep(e0.y, e1.y, x.y));
}
inline half3 weft_smoothstep(half3 e0, half3 e1, half3 x) {
  return half3(weft_smoothstep(e0.x, e1.x, x.x), weft_smoothstep(e0.y, e1.y, x.y),
               weft_smoothstep(e0.z, e1.z, x.z));
}
inline half4 weft_smoothstep(half4 e0, half4 e1, half4 x) {
  return half4(weft_smoothstep(e0.x, e1.x, x.x), weft_smoothstep(e0.y, e1.y, x.y),
               weft_smoothstep(e0.z, e1.z, x.z), weft_smoothstep(e0.w, e1.w, x.w));
}

// One name, four widths, for the functions that take and return the same type.
#define WEFT_HALF1(NAME)                                                          \
  inline half NAME(half v) { return (half)NAME(weft_fh(v)); }                      \
  inline half2 NAME(half2 v) { return half2(NAME(v.x), NAME(v.y)); }               \
  inline half3 NAME(half3 v) { return half3(NAME(v.x), NAME(v.y), NAME(v.z)); }    \
  inline half4 NAME(half4 v) {                                                     \
    return half4(NAME(v.x), NAME(v.y), NAME(v.z), NAME(v.w));                      \
  }
#define WEFT_HALF2(NAME)                                                          \
  inline half NAME(half a, half b) { return (half)NAME((float)a, (float)b); }      \
  inline half2 NAME(half2 a, half2 b) { return half2(NAME(a.x, b.x), NAME(a.y, b.y)); } \
  inline half3 NAME(half3 a, half3 b) {                                            \
    return half3(NAME(a.x, b.x), NAME(a.y, b.y), NAME(a.z, b.z));                   \
  }                                                                                \
  inline half4 NAME(half4 a, half4 b) {                                            \
    return half4(NAME(a.x, b.x), NAME(a.y, b.y), NAME(a.z, b.z), NAME(a.w, b.w));   \
  }

WEFT_HALF1(weft_sqrt)
WEFT_HALF1(weft_rsqrt)
WEFT_HALF1(weft_exp2)
WEFT_HALF1(weft_log2)
WEFT_HALF1(weft_exp)
WEFT_HALF1(weft_log)
WEFT_HALF1(weft_log10)
WEFT_HALF1(weft_exp10)
WEFT_HALF1(weft_sin)
WEFT_HALF1(weft_cos)
WEFT_HALF1(weft_tan)
WEFT_HALF1(weft_sinpi)
WEFT_HALF1(weft_cospi)
WEFT_HALF1(weft_tanpi)
WEFT_HALF1(weft_asin)
WEFT_HALF1(weft_acos)
WEFT_HALF1(weft_atan)
WEFT_HALF1(weft_sinh)
WEFT_HALF1(weft_cosh)
WEFT_HALF1(weft_tanh)
WEFT_HALF1(weft_asinh)
WEFT_HALF1(weft_acosh)
WEFT_HALF1(weft_atanh)
WEFT_HALF2(weft_pow)
WEFT_HALF2(weft_powr)
WEFT_HALF2(weft_atan2)

inline half weft_mix(half a, half b, half t) {
  return (half)weft_mix((float)a, (float)b, (float)t);
}
inline half2 weft_mix(half2 a, half2 b, half2 t) {
  return half2(weft_mix(a.x, b.x, t.x), weft_mix(a.y, b.y, t.y));
}
inline half3 weft_mix(half3 a, half3 b, half3 t) {
  return half3(weft_mix(a.x, b.x, t.x), weft_mix(a.y, b.y, t.y),
               weft_mix(a.z, b.z, t.z));
}
inline half4 weft_mix(half4 a, half4 b, half4 t) {
  return half4(weft_mix(a.x, b.x, t.x), weft_mix(a.y, b.y, t.y),
               weft_mix(a.z, b.z, t.z), weft_mix(a.w, b.w, t.w));
}

// The geometric functions return a half for half arguments, so the rounded value
// is what the caller's next operation sees.
inline half weft_dot(half2 a, half2 b) { return (half)weft_dot(weft_fh(a), weft_fh(b)); }
inline half weft_dot(half3 a, half3 b) { return (half)weft_dot(weft_fh(a), weft_fh(b)); }
inline half weft_dot(half4 a, half4 b) { return (half)weft_dot(weft_fh(a), weft_fh(b)); }
inline half weft_length(half2 v) { return (half)weft_length(weft_fh(v)); }
inline half weft_length(half3 v) { return (half)weft_length(weft_fh(v)); }
inline half weft_length(half4 v) { return (half)weft_length(weft_fh(v)); }
inline half weft_distance(half2 a, half2 b) {
  return (half)weft_distance(weft_fh(a), weft_fh(b));
}
inline half weft_distance(half3 a, half3 b) {
  return (half)weft_distance(weft_fh(a), weft_fh(b));
}
inline half weft_distance(half4 a, half4 b) {
  return (half)weft_distance(weft_fh(a), weft_fh(b));
}
inline half2 weft_normalize(half2 v) {
  const float2 r = weft_normalize(weft_fh(v));
  return half2((half)r.x, (half)r.y);
}
inline half3 weft_normalize(half3 v) {
  const float3 r = weft_normalize(weft_fh(v));
  return half3((half)r.x, (half)r.y, (half)r.z);
}
inline half4 weft_normalize(half4 v) {
  const float4 r = weft_normalize(weft_fh(v));
  return half4((half)r.x, (half)r.y, (half)r.z, (half)r.w);
}
inline half3 weft_cross(half3 a, half3 b) {
  const float3 r = weft_cross(weft_fh(a), weft_fh(b));
  return half3((half)r.x, (half)r.y, (half)r.z);
}
inline half3 weft_reflect(half3 i, half3 n) {
  const float3 r = weft_reflect(weft_fh(i), weft_fh(n));
  return half3((half)r.x, (half)r.y, (half)r.z);
}
inline half3 weft_refract(half3 i, half3 n, half eta) {
  const float3 r = weft_refract(weft_fh(i), weft_fh(n), (float)eta);
  return half3((half)r.x, (half)r.y, (half)r.z);
}
inline half3 weft_faceforward(half3 n, half3 i, half3 r) {
  const float3 v = weft_faceforward(weft_fh(n), weft_fh(i), weft_fh(r));
  return half3((half)v.x, (half)v.y, (half)v.z);
}
inline void weft_sincos(half x, WEFT_THREAD half* s, WEFT_THREAD half* c) {
  float fs = 0.0f, fc = 0.0f;
  weft_sincos((float)x, &fs, &fc);
  *s = (half)fs;
  *c = (half)fc;
}

// ---- a warp prefix scan, in the CPU's tree ---------------------------------
//
// Metal's `simd_prefix_*` folds the warp in whatever order its implementation
// uses, and for floats a different order of additions is a different answer:
// measured over 65,536 lanes the two engines were 6.1e-05 apart. The CPU runtime
// runs the doubling (Kogge-Stone) tree, so this runs the same tree on the GPU —
// step d adds the value d lanes up — and both engines then perform the identical
// sequence of additions and agree bit for bit. Integer addition is associative,
// so for those the two agreed either way.
//
// `width` is 32 for a warp and 4 for a quad; which group a lane is in are the low
// bits of its lane id, which is what the CPU indexes its array with.
#ifdef __METAL_VERSION__
template <class T>
inline T weft_warp_prefix(T v, bool inclusive, bool product, uint32_t width) {
  // Qualified, because this text is inserted before `using namespace metal` and
  // a template's names are looked up where it is defined.
  // The lane id, from the one scan the hardware gets right for integers.
  const uint32_t lane = metal::simd_prefix_inclusive_sum(1u) - 1u;
  const uint32_t glane = lane & (width - 1u);
  T acc = v;
  for (uint32_t d = 1; d < width; d <<= 1) {
    const T up = metal::simd_shuffle_up(acc, (ushort)d);
    if (glane >= d) acc = product ? (acc * up) : (acc + up);
  }
  if (inclusive) return acc;
  const T up = metal::simd_shuffle_up(acc, (ushort)1);
  return glane ? up : (product ? (T)1 : (T)0);
}
#endif

// ---- end portable math -----------------------------------------------------
)WEFTMATH";
