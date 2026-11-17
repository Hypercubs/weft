// Portable math, one source of truth.
#pragma once

static const char* const kWSoftMath = R"WEFTMATH(
// Written once, compiled by both engines. See src/soft_math.hpp.
#ifdef __METAL_VERSION__
#define WEFT_FMA(x, y, z) metal::fma((x), (y), (z))
#define WEFT_FLOOR(x) metal::floor((x))
#else
#define WEFT_FMA(x, y, z) std::fma((x), (y), (z))
#define WEFT_FLOOR(x) std::floor((x))
#endif
// other spelling this text needs besides fma and the bit-reinterpretation.
#ifdef __METAL_VERSION__
#define WEFT_THREAD thread
#else
#define WEFT_THREAD
#endif
// lands on the same answer.
#if defined(__METAL_VERSION__)
#define WEFT_SQRT(x) metal::sqrt(x)
#else
#define WEFT_SQRT(x) std::sqrt(x)
#endif
#ifdef __METAL_VERSION__
static inline uint32_t weft_cas_u32(device uint32_t* p, uint32_t cmp, uint32_t v) {
  uint32_t expect = cmp;
  (void)metal::atomic_compare_exchange_weak_explicit(
      (device metal::atomic_uint*)p, &expect, v, metal::memory_order_relaxed,
      metal::memory_order_relaxed);
  return expect;
}
#define WEFT_DEVICE device
// operations exist twice: a device word and a word in the threadgroup window.
static inline uint32_t weft_aload_u32(device uint32_t* p) {
  return metal::atomic_load_explicit((device metal::atomic_uint*)p, metal::memory_order_relaxed);
}
static inline uint32_t weft_aload_u32(threadgroup uint32_t* p) {
  return metal::atomic_load_explicit((threadgroup metal::atomic_uint*)p,
                                     metal::memory_order_relaxed);
}
static inline uint32_t weft_cas_u32(threadgroup uint32_t* p, uint32_t cmp, uint32_t v) {
  uint32_t expect = cmp;
  (void)metal::atomic_compare_exchange_weak_explicit(
      (threadgroup metal::atomic_uint*)p, &expect, v, metal::memory_order_relaxed,
      metal::memory_order_relaxed);
  return expect;
}
#define WEFT_ALOAD(p) weft_aload_u32(p)
#define WEFT_CAS(p, c, v) weft_cas_u32((p), (c), (v))
#else
// The same two operations from the compiler's own atomics.
static inline uint32_t weft_cas_u32(uint32_t* p, uint32_t cmp, uint32_t v) {
  uint32_t expect = cmp;
  (void)__atomic_compare_exchange_n(p, &expect, v, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return expect;
}
#define WEFT_DEVICE
#define WEFT_ALOAD(p) __atomic_load_n((p), __ATOMIC_RELAXED)
#define WEFT_CAS(p, c, v) weft_cas_u32((p), (c), (v))
#endif

#define WEFT_LN2 0.6931471805599453f
#define WEFT_INF 3.402823466e38f

// A float as its bits and back: Metal has `as_type`, C++ has memcpy.
#ifdef __METAL_VERSION__
#define weft_bits(x) as_type<uint32_t>(x)
#define weft_float(x) as_type<float>(x)
#else
// what makes the load and store exact rather than two halves that can be torn.
struct weft_u128 {
  uint64_t lo;
  uint64_t hi;
};

// instruction here is that wide.
inline char weft_b128_locks[64];

static inline weft_u128 weft_atom_cas_b128(weft_u128* p, weft_u128 c, weft_u128 v) {
  const uintptr_t a = (uintptr_t)p;
  char* l = &weft_b128_locks[((a >> 4) ^ (a >> 12)) & 63u];
  while (__atomic_test_and_set(l, __ATOMIC_ACQUIRE)) {
  }
  const weft_u128 o = *p;
  if (o.lo == c.lo && o.hi == c.hi) *p = v;
  __atomic_clear(l, __ATOMIC_RELEASE);
  return o;
}

static inline weft_u128 weft_atom_exch_b128(weft_u128* p, weft_u128 v) {
  const uintptr_t a = (uintptr_t)p;
  char* l = &weft_b128_locks[((a >> 4) ^ (a >> 12)) & 63u];
  while (__atomic_test_and_set(l, __ATOMIC_ACQUIRE)) {
  }
  const weft_u128 o = *p;
  *p = v;
  __atomic_clear(l, __ATOMIC_RELEASE);
  return o;
}

static inline uint32_t weft_bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return u;
}
static inline float weft_float(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
#endif

// A float as a double and back: exact upward, round-to-nearest-even downward.
static inline uint64_t weft_widen(uint32_t u);
static inline float weft_narrow(uint64_t b);

namespace weft_fp64 {
static inline uint64_t sqrt_bits(uint64_t);
static inline uint64_t exp_bits(uint64_t);
static inline uint64_t exp2_bits(uint64_t);
static inline uint64_t expm1_bits(uint64_t);
static inline uint64_t log_bits(uint64_t);
static inline uint64_t log2_bits(uint64_t);
static inline uint64_t log10_bits(uint64_t);
static inline uint64_t log1p_bits(uint64_t);
static inline uint64_t pow_bits(uint64_t, uint64_t);
static inline uint64_t powr_bits(uint64_t, uint64_t);
static inline uint64_t pown_bits(uint64_t, int32_t);
static inline uint64_t rootn_bits(uint64_t, int32_t);
static inline uint64_t cbrt_bits(uint64_t);
static inline uint64_t hypot_bits(uint64_t, uint64_t);
static inline uint64_t sin_bits(uint64_t);
static inline uint64_t cos_bits(uint64_t);
static inline uint64_t tan_bits(uint64_t);
static inline uint64_t sinpi_bits(uint64_t);
static inline uint64_t cospi_bits(uint64_t);
static inline uint64_t tanpi_bits(uint64_t);
static inline uint64_t atan_bits(uint64_t);
static inline uint64_t atan2_bits(uint64_t, uint64_t);
static inline uint64_t asin_bits(uint64_t);
static inline uint64_t acos_bits(uint64_t);
static inline uint64_t acosh_bits(uint64_t);
static inline uint64_t sinh_bits(uint64_t);
static inline uint64_t cosh_bits(uint64_t);
static inline uint64_t tanh_bits(uint64_t);
static inline uint64_t asinh_bits(uint64_t);
static inline uint64_t atanh_bits(uint64_t);
static inline uint64_t tgamma_bits(uint64_t);
static inline uint64_t erf_bits(uint64_t);
static inline uint64_t erfc_bits(uint64_t);
}  // namespace weft_fp64

// The float names are defined at the end of this text, after the pair paths they call.
inline float weft_sqrt(float);
inline float weft_rsqrt(float);
inline float weft_exp(float);
inline float weft_expm1(float);
inline float weft_log(float);
inline float weft_log2(float);
inline float weft_log10(float);
inline float weft_log1p(float);
inline float weft_sinh(float);
inline float weft_cosh(float);
inline float weft_tanh(float);
inline float weft_asinh(float);
inline float weft_acosh(float);
inline float weft_atanh(float);
inline float weft_cbrt(float);
inline float weft_hypot(float, float);
inline float weft_sin(float);
inline float weft_cos(float);
inline float weft_tan(float);
inline float weft_sinpi(float);
inline float weft_cospi(float);
inline float weft_tanpi(float);
inline float weft_atan(float);
inline float weft_atan2(float, float);
inline float weft_asin(float);
inline float weft_acos(float);
inline float weft_powr(float, float);
inline float weft_pown(float, int);
inline float weft_rootn(float, int);

inline float weft_sqrt_wide(float v) {
  if (v < 0.0f) return (v - v) / (v - v);          // a NaN, and a NaN input too
  return weft_narrow(weft_fp64::sqrt_bits(weft_widen(weft_bits(v))));
}

// narrows to the correctly rounded float.
inline float weft_rsqrt_wide(float v) {
  if (v == 0.0f) return 1.0f / v;                  // ±inf
  if (!(v > 0.0f)) return (v - v) / (v - v);       // a NaN, or a negative argument
  return weft_narrow(weft_fp64::powr_bits(weft_widen(weft_bits(v)),
                                          weft_widen(weft_bits(-0.5f))));
}

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

// float, and `fma` hands it over in one rounding, so a value is rounded once, at the end.
struct weft_df { float hi, lo; };

inline weft_df weft_df_add(float a, float b) {
  const float s = a + b;
  const float bb = s - a;
  return weft_df{s, (a - (s - bb)) + (b - bb)};
}
inline weft_df weft_df_prod(float a, float b) {
  const float p = a * b;
  return weft_df{p, WEFT_FMA(a, b, -p)};
}
inline weft_df weft_df_sub(weft_df a, weft_df b) {
  const weft_df t = weft_df_add(a.hi, -b.hi);
  return weft_df_add(t.hi, t.lo + (a.lo - b.lo));
}
// The quotient to twice a float's precision: what the division left over, divided again.
inline weft_df weft_df_div(float a, float b) {
  const float q = a / b;
  return weft_df{q, WEFT_FMA(-q, b, a) / b};
}

// split that keeps an exponential's argument inside half a unit.
inline float weft_rint_even(float x) {
  const float ax = x < 0.0f ? -x : x;
  if (!(ax < 8388608.0f)) return x;   // every float this large is already an integer
  const int t = (int)x;
  const float d = x - (float)t;
  if (d > 0.5f) return (float)(t + 1);
  if (d < -0.5f) return (float)(t - 1);
  if (d == 0.5f) return (t & 1) ? (float)(t + 1) : (float)t;
  if (d == -0.5f) return (t & 1) ? (float)(t - 1) : (float)t;
  return (float)t;
}
inline float weft_ldexp(float x, int n);

// reach the last bit: the sum carries what each addition drops and the log2(e)
// scaling is split.
inline weft_df weft_log2_pair(float v) {
  if (v == 0.0f) return weft_df{-1.0f / 0.0f, 0.0f};   // log(0) is -inf, not a NaN
  if (!(v > 0.0f)) return weft_df{(v - v) / (v - v), 0.0f};
  if (v > WEFT_INF) return weft_df{v, 0.0f};
  float m = v;
  int k = 0;
  if (!(v > 0.5f && v < 1.5f)) {
    while (m >= 2.0f) { m *= 0.5f; k++; }
    while (m < 1.0f) { m *= 2.0f; k--; }
  }
  const float dd = m - 1.0f;                       // exact for m in [0.5, 2)
  const weft_df t = weft_df_div(dd, 2.0f + dd);
  const float t2 = t.hi * t.hi;
  float p = t.hi, sum = 0.0f, carry = 0.0f;
  for (int n = 1; n <= 15; n += 2) {
    const float y = p * (1.0f / (float)n);
    const float s = sum + y;
    carry += (sum - s) + y;                        // what the addition dropped
    sum = s;
    p *= t2;
  }
  // the low half of the quotient through the derivative of 2 atanh(t)
  const float ln = 2.0f * (sum + (carry + t.lo / (1.0f - t2)));
  const float hi = 1.4426950216293335f, lo = 1.925963033500011e-08f;
  const float ph = ln * hi;
  const weft_df e = weft_df_add((float)k, ph);
  return weft_df_add(e.hi, e.lo + (WEFT_FMA(ln, hi, -ph) + ln * lo));
}
inline float weft_log2_wide(float v) {
  return weft_narrow(weft_fp64::log2_bits(weft_widen(weft_bits(v))));
}

// exp2, and 2^(hi+lo) where the low half cannot move the exponent.
inline weft_df weft_exp2_df(float hi, float lo) {
  if (!(hi == hi)) return weft_df{hi, 0.0f};
  if (hi > 129.0f) return weft_df{WEFT_INF * 2.0f, 0.0f};
  if (hi < -152.0f) return weft_df{0.0f, 0.0f};
  const float kf = weft_rint_even(hi);
  const float f = hi - kf;                          // exact: |f| <= 1/2
  const float z = f * WEFT_LN2;
  const float zlo = WEFT_FMA(f, WEFT_LN2, -z);
  float sum = 1.0f, term = z, carry = 0.0f;
  for (int n = 1; n <= 9; n++) {
    const float s = sum + term;
    carry += (sum - s) + term;
    sum = s;
    term *= z * (1.0f / (float)(n + 1));
  }
  const float fix = carry + sum * (zlo + lo * WEFT_LN2);
  const float v = sum + fix;
  const float err = (sum - v) + fix;               // what that last addition dropped
  const float sc = weft_ldexp(v, (int)kf);         // both scalings are exact
  return weft_df{sc, weft_ldexp(err, (int)kf)};
}
inline float weft_exp2_pair(float hi, float lo) {
  const weft_df r = weft_exp2_df(hi, lo);
  return r.hi + r.lo;
}
inline float weft_exp2(float v) { return weft_exp2_pair(v, 0.0f); }

// same path here and stay bit-identical.
inline float weft_pow(float x, float y) {
  if (y == 2.0f) return x * x;
  if (y == 3.0f) return (x * x) * x;
  if (y == 4.0f) { float a = x * x; return a * a; }
  if (y == 1.0f) return x;
  if (y == 0.0f) return 1.0f;
  if (y == 0.5f) return weft_sqrt(x);
  if (y == -1.0f) return 1.0f / x;
  // (and a negative zero keeps its sign), 0 to a negative one is infinite.
  if (x == 0.0f) {
    if (y > 0.0f) return x;
    if (y < 0.0f) return 1.0f / x;
    return (x - x) / (x - x);
  }
  if (!(x > 0.0f)) return (x - x) / (x - x);
  const weft_df l = weft_log2_pair(x);
  const float p = y * l.hi;
  return weft_exp2_pair(p, WEFT_FMA(y, l.hi, -p) + y * l.lo);
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
inline float weft_exp_wide(float v) {
  return weft_narrow(weft_fp64::exp_bits(weft_widen(weft_bits(v))));
}
inline float weft_log_wide(float v) {
  return weft_narrow(weft_fp64::log_bits(weft_widen(weft_bits(v))));
}

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

// wide road is for what a float cannot settle.
inline float weft_sin_wide(float x) {
  return weft_narrow(weft_fp64::sin_bits(weft_widen(weft_bits(x))));
}
inline float weft_cos_wide(float x) {
  return weft_narrow(weft_fp64::cos_bits(weft_widen(weft_bits(x))));
}
inline float weft_tan_wide(float x) {
  return weft_narrow(weft_fp64::tan_bits(weft_widen(weft_bits(x))));
}

inline float weft_sinpi_wide(float x) {
  return weft_narrow(weft_fp64::sinpi_bits(weft_widen(weft_bits(x))));
}
inline float weft_cospi_wide(float x) {
  return weft_narrow(weft_fp64::cospi_bits(weft_widen(weft_bits(x))));
}
inline float weft_tanpi_wide(float x) {
  return weft_narrow(weft_fp64::tanpi_bits(weft_widen(weft_bits(x))));
}

// this is where the pair path gives up first.
inline float weft_atan_wide(float x) {
  return weft_narrow(weft_fp64::atan_bits(weft_widen(weft_bits(x))));
}
inline float weft_atan2_wide(float y, float x) {
  return weft_narrow(
      weft_fp64::atan2_bits(weft_widen(weft_bits(y)), weft_widen(weft_bits(x))));
}
inline float weft_asin_wide(float x) {
  return weft_narrow(weft_fp64::asin_bits(weft_widen(weft_bits(x))));
}
inline float weft_acos_wide(float x) {
  return weft_narrow(weft_fp64::acos_bits(weft_widen(weft_bits(x))));
}

inline float weft_exp10(float v) {
  if (!(v == v) || v > 3.0e38f || v < -3.0e38f) return weft_exp2(v * 3.321928094887362f);
  const float hi = 3.321928024291992f, lo = 7.059536955011936e-08f;
  const float p = v * hi;
  const float d = WEFT_FMA(v, hi, -p) + v * lo;
  return weft_exp2_pair(p, d);
}
// of an integer does not come back as it. `floor` is exact on both targets.
inline float weft_fract(float v) {
  const float f = v - WEFT_FLOOR(v);
  const float lim = weft_float(0x3f7fffffu);
  return f < lim ? f : lim;
}
inline float weft_log10_wide(float v) {
  return weft_narrow(weft_fp64::log10_bits(weft_widen(weft_bits(v))));
}
inline float weft_powr_wide(float x, float y) {
  if (x == 0.0f) {
    if (y > 0.0f) return 0.0f;
    if (y < 0.0f) return 1.0f / 0.0f;
    return (x - x) / (x - x);          // powr(0, 0) is undefined
  }
  if (!(x > 0.0f)) return (x - x) / (x - x);
  return weft_narrow(weft_fp64::powr_bits(weft_widen(weft_bits(x)),
                                         weft_widen(weft_bits(y))));
}
// whole answer was made of the bits that cancelled: sinh(1e-8) came out 0.
inline float weft_sinh_wide(float v) {
  return weft_narrow(weft_fp64::sinh_bits(weft_widen(weft_bits(v))));
}
inline float weft_cosh_wide(float v) {
  return weft_narrow(weft_fp64::cosh_bits(weft_widen(weft_bits(v))));
}
// than making e and 1/e the same infinity.
inline float weft_tanh_wide(float v) {
  return weft_narrow(weft_fp64::tanh_bits(weft_widen(weft_bits(v))));
}
inline float weft_acosh_wide(float v) {
  return weft_narrow(weft_fp64::acosh_bits(weft_widen(weft_bits(v))));
}
inline float weft_atanh_wide(float v) {
  return weft_narrow(weft_fp64::atanh_bits(weft_widen(weft_bits(v))));
}

// steps and there is no table to get wrong.
inline float weft_ldexp(float x, int n) {
  float s = x;
  for (; n > 0; n--) s *= 2.0f;
  for (; n < 0; n++) s *= 0.5f;
  return s;
}

inline int weft_ilogb(float x) {
  const uint32_t u = weft_bits(x);
  const uint32_t e = (u >> 23) & 0xFFu;
  if (e == 0xFFu) return 0x7FFFFFFF;  // inf or nan
  if (e != 0u) return (int)e - 127;
  if ((u & 0x7FFFFFFFu) == 0u) return -0x7FFFFFFF;  // zero has no exponent
  // is not zero, and subtract what was added on the way.
  float v = (u >> 31) ? -x : x;
  int shift = 0;
  while (((weft_bits(v) >> 23) & 0xFFu) == 0u) {
    v *= 8388608.0f;
    shift += 23;
  }
  return (int)((weft_bits(v) >> 23) & 0xFFu) - 127 - shift;
}

inline float weft_logb(float x) {
  if (!(x == x)) return x;                            // nan
  if (x == 0.0f) return -1.0f / 0.0f;                 // logb(0) is -inf
  if (x - x != 0.0f) return 1.0f / 0.0f;              // inf of either sign
  return (float)weft_ilogb(x);
}

inline float weft_maxmag(float a, float b) {
  if (!(a == a)) return b;
  if (!(b == b)) return a;
  const float aa = a < 0.0f ? -a : a, bb = b < 0.0f ? -b : b;
  // the two, so maxmag(-1, 1) is 1 and maxmag(-inf, inf) is inf.
  if (aa == bb) return a > b ? a : b;
  return aa > bb ? a : b;
}
inline float weft_minmag(float a, float b) {
  if (!(a == a)) return b;
  if (!(b == b)) return a;
  const float aa = a < 0.0f ? -a : a, bb = b < 0.0f ? -b : b;
  if (aa == bb) return a < b ? a : b;
  return aa < bb ? a : b;
}

inline float weft_nextafter(float x, float y) {
  if (!(x == x) || !(y == y)) return x + y;
  if (x == y) return y;
  if (x == 0.0f) return weft_float(y > 0.0f ? 1u : 0x80000001u);
  const bool up = (y > x) == (x > 0.0f);
  return weft_float(weft_bits(x) + (up ? 1u : (uint32_t)-1));
}

inline float weft_pown_wide(float x, int n) {
  return weft_narrow(weft_fp64::pown_bits(weft_widen(weft_bits(x)), (int32_t)n));
}

inline float weft_rootn_wide(float x, int n) {
  return weft_narrow(weft_fp64::rootn_bits(weft_widen(weft_bits(x)), (int32_t)n));
}

inline float weft_trunc_of(float x) {
  const float m = x < 0.0f ? -x : x;
  if (!(m < 8388608.0f)) return x;  // every float this large is already an integer
  const uint32_t u = weft_bits(m);
  const uint32_t e = (u >> 23) & 0xFFu;
  if (e < 127u) return x - x;                      // |x| < 1 has no integer part
  const float t = weft_float(u & (0xFFFFFFFFu << (150u - e)));
  return x < 0.0f ? -t : t;
}

// the emitter can name them separately and build the struct in place.
inline int weft_frexp_e(float x) {
  if (!(x == x) || x - x != 0.0f || x == 0.0f) return 0;  // nan, inf and zero
  return weft_ilogb(x) + 1;
}
inline float weft_frexp_f(float x) { return weft_ldexp(x, -weft_frexp_e(x)); }

inline float weft_modf_i(float x) { return weft_trunc_of(x); }
inline float weft_modf_f(float x) { return x - weft_trunc_of(x); }

inline float weft_sincos_s(float x) { return weft_sin(x); }
inline float weft_sincos_c(float x) { return weft_cos(x); }

// The nearest integer, which is the quotient of a `remquo`.
struct weft_remq { float r; int q; int par; };

inline weft_remq weft_rem_exact(float x, float y) {
  weft_remq out;
  out.r = x;
  out.q = 0;
  out.par = 0;
  const float ax = x < 0.0f ? -x : x;
  const float ay = y < 0.0f ? -y : y;
  if (!(y == y) || !(ay > 0.0f)) return out;      // a NaN or zero divisor: the caller
  if (!(x == x)) return out;                      // decides, and x is already the answer
  if (ax - ax != 0.0f) { out.r = x - x; return out; }   // an infinite dividend
  if (!(ax >= ay)) return out;                    // all of x is the remainder
  const uint32_t ux = weft_bits(ax);
  const uint32_t uy = weft_bits(ay);
  int ex = (int)((ux >> 23) & 0xFFu);
  int ey = (int)((uy >> 23) & 0xFFu);
  uint32_t mx = ux & 0x7FFFFFu;
  uint32_t my = uy & 0x7FFFFFu;
  if (ex != 0) mx |= 0x800000u; else ex = 1;      // a subnormal is m*2^-149 either way
  if (ey != 0) my |= 0x800000u; else ey = 1;
  uint64_t q = 0ull;
  int par = 0;
  while (ex > ey && mx != 0u) {
    mx <<= 1;
    const int sub = (mx >= my) ? 1 : 0;
    if (sub != 0) mx -= my;
    q = q * 2ull + (uint64_t)sub;
    if (q > 0xFFFFFFFFull) q = 0xFFFFFFFFull;
    par = sub;
    ex -= 1;
  }
  while (mx >= my) {
    mx -= my;
    if (q < 0xFFFFFFFFull) q += 1ull;
    par ^= 1;
  }
  out.q = (int)(q > 0x7FFFFFFEull ? 0x7FFFFFFFull : q);
  out.par = par;
  if (mx == 0u) {
    out.r = x < 0.0f ? -0.0f : 0.0f;
    return out;
  }
  const float mag = weft_ldexp((float)mx, ey - 150);
  out.r = x < 0.0f ? -mag : mag;
  return out;
}

// ulp of remainder away on a tie.
inline weft_remq weft_remquo_pair(float x, float y) {
  weft_remq out = weft_rem_exact(x, y);
  const float ay = y < 0.0f ? -y : y;
  const float two = out.r < 0.0f ? -2.0f * out.r : 2.0f * out.r;
  if (two > ay || (two == ay && out.par != 0)) {
    out.r -= (x < 0.0f ? -y : y);
    if (out.q < 0x7FFFFFFE) out.q += 1;
    out.par ^= 1;
  }
  // parity, what a tie is decided on, is the same either way.
  if ((x < 0.0f) != (y < 0.0f)) out.q = -out.q;
  return out;
}
inline float weft_remquo_r(float x, float y) { return weft_remquo_pair(x, y).r; }
inline int weft_remquo_q(float x, float y) { return weft_remquo_pair(x, y).q; }

// once, round to nearest even.
static inline uint64_t weft_widen(uint32_t u) {
  const uint64_t sign = (uint64_t)(u >> 31) << 63;
  const uint32_t e = (u >> 23) & 0xFFu;
  const uint32_t m = u & 0x7FFFFFu;
  if (e == 0xFFu)
    return sign | 0x7FF0000000000000ull |
           (m == 0u ? 0ull : (0x8000000000000ull | ((uint64_t)m << 29)));
  if (e == 0u) {
    if (m == 0u) return sign;
    uint32_t t = m;
    int s = 0;
    while ((t & 0x800000u) == 0u) { t <<= 1; s += 1; }
    return sign | ((uint64_t)(897 - s) << 52) | ((uint64_t)(t & 0x7FFFFFu) << 29);
  }
  return sign | ((uint64_t)(e + 896u) << 52) | ((uint64_t)m << 29);
}
static inline float weft_narrow(uint64_t b) {
  const uint32_t sign = (uint32_t)(b >> 63) << 31;
  const uint32_t de = (uint32_t)((b >> 52) & 0x7FFu);
  const uint64_t dm = b & 0xFFFFFFFFFFFFFull;
  if (de == 0x7FFu) {
    if (dm == 0ull) return weft_float(sign | 0x7F800000u);
    uint32_t p = (uint32_t)(dm >> 29) & 0x7FFFFFu;
    if (p == 0u) p = 0x400000u;                    // a payload that still reads NaN
    return weft_float(sign | 0x7F800000u | p);
  }
  if (de == 0u) return weft_float(sign);           // far below any float: a zero
  const int e = (int)de - 1023;
  const uint64_t m = dm | 0x10000000000000ull;     // the value is m * 2^(e - 52)
  if (e > 127) return weft_float(sign | 0x7F800000u);
  if (e < -150) return weft_float(sign);           // under half the smallest subnormal
  if (e < -126) {
    // The answer is a subnormal: one step more of shifting, and one more rounding.
    const int drop = -97 - e;
    uint64_t q = m >> drop;
    const uint64_t r = m & ((1ull << drop) - 1ull);
    const uint64_t halfq = 1ull << (drop - 1);
    if (r > halfq || (r == halfq && (q & 1ull) != 0ull)) q += 1ull;
    return weft_float(sign | (uint32_t)q);
  }
  uint64_t q = m >> 29;
  const uint64_t r = m & ((1ull << 29) - 1ull);
  if (r > (1ull << 28) || (r == (1ull << 28) && (q & 1ull) != 0ull)) q += 1ull;
  uint32_t fe = (uint32_t)(e + 127);
  if (q == 0x1000000ull) { q = 0x800000ull; fe += 1u; }
  if (fe >= 0xFFu) return weft_float(sign | 0x7F800000u);
  return weft_float(sign | (fe << 23) | ((uint32_t)q & 0x7FFFFFu));
}
inline float weft_tgamma_wide(float v) {
  return weft_narrow(weft_fp64::tgamma_bits(weft_widen(weft_bits(v))));
}
inline float weft_erf_wide(float v) {
  return weft_narrow(weft_fp64::erf_bits(weft_widen(weft_bits(v))));
}
inline float weft_erfc_wide(float v) {
  return weft_narrow(weft_fp64::erfc_bits(weft_widen(weft_bits(v))));
}
inline float weft_cbrt_wide(float v) {
  return weft_narrow(weft_fp64::cbrt_bits(weft_widen(weft_bits(v))));
}
inline float weft_log1p_wide(float v) {
  return weft_narrow(weft_fp64::log1p_bits(weft_widen(weft_bits(v))));
}
inline float weft_hypot_wide(float x, float y) {
  return weft_narrow(weft_fp64::hypot_bits(weft_widen(weft_bits(x)),
                                           weft_widen(weft_bits(y))));
}
// where a pair's own error is largest.
inline float weft_asinh_wide(float v) {
  return weft_narrow(weft_fp64::asinh_bits(weft_widen(weft_bits(v))));
}
inline float weft_remainder(float x, float y) {
  if (!(y == y) || y == 0.0f) return (x - x) / (y - y);   // a zero divisor is a NaN
  return weft_remquo_pair(x, y).r;
}

// it answers e^x - 1 to the last bit of the double rather than to one float.
inline float weft_expm1_wide(float x) {
  return weft_narrow(weft_fp64::expm1_bits(weft_widen(weft_bits(x))));
}

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
// goes through the float function and rounds once on the way back.
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
// The 16-bit float bit helpers.
#ifdef __METAL_VERSION__
#define weft_f16_of(b) as_type<half>(b)
#define weft_f16_bits(h) as_type<uint16_t>(h)
#else
static inline half weft_f16_of(uint16_t b) {
  half h;
  memcpy(&h, &b, 2);
  return h;
}
static inline uint16_t weft_f16_bits(half h) {
  uint16_t b;
  memcpy(&b, &h, 2);
  return b;
}
#endif
// and `half` on a GPU, and this is the one name both texts know.
static inline half weft_h16v(uint16_t b) { return weft_f16_of(b); }

static inline uint16_t weft_f16_add(uint16_t a, uint16_t b) {
  return weft_f16_bits(weft_f16_of(a) + weft_f16_of(b));
}
static inline uint16_t weft_f16_sub(uint16_t a, uint16_t b) {
  return weft_f16_bits(weft_f16_of(a) - weft_f16_of(b));
}
static inline uint16_t weft_f16_mul(uint16_t a, uint16_t b) {
  return weft_f16_bits(weft_f16_of(a) * weft_f16_of(b));
}
// minimum and +0 for a maximum, which the bitwise answer gives exactly.
static inline uint16_t weft_f16_min(uint16_t a, uint16_t b) {
  const half x = weft_f16_of(a), y = weft_f16_of(b);
  if (x != x) return b;
  if (y != y) return a;
  if (x == y) return (uint16_t)(a & b);
  return x < y ? a : b;
}
static inline uint16_t weft_f16_max(uint16_t a, uint16_t b) {
  const half x = weft_f16_of(a), y = weft_f16_of(b);
  if (x != x) return b;
  if (y != y) return a;
  if (x == y) return (uint16_t)(a | b);
  return x < y ? b : a;
}
// neg and abs are the sign bit, so they are exact on a NaN as well.
static inline uint16_t weft_f16_neg(uint16_t a) { return (uint16_t)(a ^ 0x8000u); }
static inline uint16_t weft_f16_abs(uint16_t a) { return (uint16_t)(a & 0x7FFFu); }
// positive zero, not the destination's range.
static inline uint16_t weft_f16_sat(uint16_t a) {
  const half v = weft_f16_of(a);
  if (v != v) return 0u;
  if (v < (half)0) return 0u;
  if (v > (half)1) return 0x3C00u;
  return a;
}
// fma on a half, rounded once.
static inline uint32_t weft_narrow_pack(uint64_t mag, int lsb, bool sticky, int sign,
                                        int mbits, int ebits) {
  const int sbit = ebits + mbits;
  const uint32_t xmask = ((uint32_t)1 << ebits) - 1u;
  const uint32_t lead = (uint32_t)1 << mbits;
  const int bias = (1 << (ebits - 1)) - 1;
  const int sube = 1 - bias - mbits;
  const uint32_t qinf = xmask << mbits;
  const int sub = sube - lsb;         // the subnormal grid, or the coarse one it takes
  int bl = 0;
  for (uint64_t t = mag; t; t >>= 1) bl++;
  int drop = bl - (mbits + 1);        // one implicit bit plus the significand
  if (sub > drop) drop = sub;
  while (drop < 0) { mag <<= 1; lsb--; drop++; }
  uint64_t q = mag;
  int round_bit = 0;
  if (drop > 0) {
    if (drop >= 64) { sticky = sticky || (mag != 0u); q = 0u; }
    else {
      // where nearest-even sometimes keeps the lower candidate.
      const uint64_t low = drop > 1 ? ((1ull << (drop - 1)) - 1ull) : 0ull;
      if ((mag & low) != 0u) sticky = true;
      round_bit = (int)((mag >> (drop - 1)) & 1ull);
      q = mag >> drop;
    }
  }
  if (round_bit && (sticky || (q & 1ull))) q += 1ull;
  const int ex = lsb + drop;          // the value is q * 2^ex
  if (q >= lead) {
    int E = ex + bias + mbits;
    if (q >= 2u * lead) { q = lead; E += 1; }
    if (E >= (int)xmask) return (uint32_t)((sign << sbit) | qinf);
    return (uint32_t)((sign << sbit) | ((uint32_t)E << mbits) | (uint32_t)(q - lead));
  }
  return (uint32_t)((sign << sbit) | (uint32_t)q);
}

// what tells a division or a root whether anything sits below the significand.
static inline uint64_t weft_isqrt64(uint64_t n) {
  uint64_t r = 0;
  uint64_t bit = (uint64_t)1 << 62;
  while (bit > n) bit >>= 2;
  while (bit) {
    if (n >= r + bit) {
      n -= r + bit;
      r = (r >> 1) + bit;
    } else {
      r >>= 1;
    }
    bit >>= 2;
  }
  return r;
}

// with the remainder as the sticky bit.
static inline uint32_t weft_narrow_div(uint32_t ab, uint32_t bb, int mbits, int ebits) {
  const int sbit = ebits + mbits;
  const uint32_t xmask = ((uint32_t)1 << ebits) - 1u;
  const uint32_t mmask = ((uint32_t)1 << mbits) - 1u;
  const uint32_t lead = (uint32_t)1 << mbits;
  const int bias = (1 << (ebits - 1)) - 1;
  const int sube = 1 - bias - mbits;
  const uint32_t qnan = (xmask << mbits) | mmask;
  const uint32_t qinf = xmask << mbits;
  const uint32_t sgn = (uint32_t)1 << sbit;
  const uint32_t ea = (ab >> mbits) & xmask, eb = (bb >> mbits) & xmask;
  const uint32_t fa = ab & mmask, fb = bb & mmask;
  const int rs = (int)((ab ^ bb) >> sbit);
  if ((ea == xmask && fa) || (eb == xmask && fb)) return qnan;
  const bool infa = ea == xmask, infb = eb == xmask;
  const bool zera = ea == 0u && fa == 0u, zerb = eb == 0u && fb == 0u;
  if (infa && infb) return qnan;
  if (infa) return (uint32_t)((rs << sbit) | qinf);
  if (infb) return (uint32_t)(rs ? sgn : 0u);
  if (zerb) return zera ? qnan : (uint32_t)((rs << sbit) | qinf);
  if (zera) return (uint32_t)(rs ? sgn : 0u);
  const uint64_t ma = ea == 0u ? fa : fa + lead;
  const uint64_t mb = eb == 0u ? fb : fb + lead;
  const int xa = ea == 0u ? sube : (int)ea - bias - mbits;
  const int xb = eb == 0u ? sube : (int)eb - bias - mbits;
  const uint64_t num = ma << 48;
  const uint64_t q = num / mb;
  const bool sticky = (num % mb) != 0u;
  return weft_narrow_pack(q, xa - xb - 48, sticky, rs, mbits, ebits);
}

// shifted up for guard bits, and the remainder as sticky.
static inline uint32_t weft_narrow_sqrt(uint32_t ab, int mbits, int ebits) {
  const int sbit = ebits + mbits;
  const uint32_t xmask = ((uint32_t)1 << ebits) - 1u;
  const uint32_t mmask = ((uint32_t)1 << mbits) - 1u;
  const uint32_t lead = (uint32_t)1 << mbits;
  const int bias = (1 << (ebits - 1)) - 1;
  const int sube = 1 - bias - mbits;
  const uint32_t qnan = (xmask << mbits) | mmask;
  const uint32_t qinf = xmask << mbits;
  const uint32_t ea = (ab >> mbits) & xmask;
  const uint32_t fa = ab & mmask;
  if (ea == xmask && fa) return qnan;
  if (ea == xmask) return qinf;
  if ((ab >> sbit) && ab != (uint32_t)(1 << sbit)) return qnan;   // negative, and not -0
  if (ea == 0u && fa == 0u) return ab;                            // a zero keeps its sign
  uint64_t ma = ea == 0u ? fa : fa + lead;
  int xa = ea == 0u ? sube : (int)ea - bias - mbits;
  if (xa & 1) {
    ma <<= 1;
    xa -= 1;
  }
  const int sh = 2 * (mbits + 3);
  const uint64_t n = ma << sh;
  const uint64_t s = weft_isqrt64(n);
  return weft_narrow_pack(s, xa / 2 - (mbits + 3), (n - s * s) != 0u, 0, mbits, ebits);
}

// It is one routine for both narrow floats, at their own widths.
static inline uint32_t weft_narrow_fma(uint32_t ab, uint32_t bb, uint32_t cb, int mbits,
                                      int ebits) {
  const int sbit = ebits + mbits;
  const uint32_t xmask = ((uint32_t)1 << ebits) - 1u;
  const uint32_t mmask = ((uint32_t)1 << mbits) - 1u;
  const uint32_t lead = (uint32_t)1 << mbits;
  const int bias = (1 << (ebits - 1)) - 1;
  const int sube = 1 - bias - mbits;   // the exponent of the subnormal grid's LSB
  const uint32_t qnan = (xmask << mbits) | mmask;
  const uint32_t qinf = xmask << mbits;
  const uint32_t sgn = (uint32_t)1 << sbit;
  const uint32_t ea = (ab >> mbits) & xmask, eb = (bb >> mbits) & xmask,
                 ec = (cb >> mbits) & xmask;
  const uint32_t fa = ab & mmask, fb = bb & mmask, fc = cb & mmask;
  const int sa = (int)(ab >> sbit), sb = (int)(bb >> sbit), sc = (int)(cb >> sbit);
  // answer the canonical NaN: sign 0, exponent all ones, every mantissa bit set.
  if ((ea == xmask && fa) || (eb == xmask && fb) || (ec == xmask && fc)) return qnan;
  const bool infa = ea == xmask, infb = eb == xmask, infc = ec == xmask;
  const bool zera = ea == 0u && fa == 0u, zerb = eb == 0u && fb == 0u;
  if ((infa && zerb) || (infb && zera)) return qnan;
  if (infa || infb) {
    if (infc && ((sa ^ sb) != sc)) return qnan;
    return (uint32_t)(((sa ^ sb) << sbit) | qinf);
  }
  if (infc) return (uint32_t)((sc << sbit) | qinf);

  const uint64_t ma = ea == 0u ? fa : fa + lead;
  const uint64_t mb = eb == 0u ? fb : fb + lead;
  const uint64_t mc = ec == 0u ? fc : fc + lead;
  const int xa = ea == 0u ? sube : (int)ea - bias - mbits;
  const int xb = eb == 0u ? sube : (int)eb - bias - mbits;
  const int xc = ec == 0u ? sube : (int)ec - bias - mbits;
  const uint64_t mp = ma * mb;
  const int ep = xa + xb;
  const bool negp = (sa ^ sb) != 0;
  if (mp == 0u && mc == 0u) return (uint32_t)((negp && sc) ? sgn : 0u);

  // sum entirely, which answered zero.
  int blp = 0, blc = 0;
  for (uint64_t t = mp; t; t >>= 1) blp++;
  for (uint64_t t = mc; t; t >>= 1) blc++;
  int hi = -100000;
  if (mp) hi = ep + blp - 1;
  if (mc) { const int q = xc + blc - 1; if (q > hi) hi = q; }
  const int es = hi - 60;
  bool sticky = false;
  uint64_t pp = 0u, pc = 0u;
  {
    int sh = ep - es;
    if (mp != 0u) {
      if (sh >= 0) pp = mp << sh;
      else if (-sh < 64) {
        if ((mp & ((1ull << -sh) - 1ull)) != 0u) sticky = true;
        pp = mp >> -sh;
      } else sticky = true;
    }
    sh = xc - es;
    if (mc != 0u) {
      if (sh >= 0) pc = mc << sh;
      else if (-sh < 64) {
        if ((mc & ((1ull << -sh) - 1ull)) != 0u) sticky = true;
        pc = mc >> -sh;
      } else sticky = true;
    }
  }
  const int64_t sum = (negp ? -(int64_t)pp : (int64_t)pp) + (sc ? -(int64_t)pc : (int64_t)pc);
  if (sum == 0) return (uint32_t)((negp && sc) ? sgn : 0u);

  const int sign = sum < 0 ? 1 : 0;
  const uint64_t mag = sum < 0 ? (uint64_t)(-sum) : (uint64_t)sum;
  return weft_narrow_pack(mag, es, sticky, sign, mbits, ebits);
}

static inline uint16_t weft_f16_fma(uint16_t ab, uint16_t bb, uint16_t cb) {
  return (uint16_t)weft_narrow_fma(ab, bb, cb, 10, 5);
}
static inline uint16_t weft_bf16_fma(uint16_t ab, uint16_t bb, uint16_t cb) {
  return (uint16_t)weft_narrow_fma(ab, bb, cb, 7, 8);
}
static inline uint16_t weft_f16_div(uint16_t a, uint16_t b) {
  return (uint16_t)weft_narrow_div(a, b, 10, 5);
}
static inline uint16_t weft_f16_rcp(uint16_t a) {
  return (uint16_t)weft_narrow_div(0x3C00u, a, 10, 5);
}
static inline uint16_t weft_f16_sqrt(uint16_t a) {
  return (uint16_t)weft_narrow_sqrt(a, 10, 5);
}
static inline uint16_t weft_bf16_div(uint16_t a, uint16_t b) {
  return (uint16_t)weft_narrow_div(a, b, 7, 8);
}
static inline uint16_t weft_bf16_rcp(uint16_t a) {
  return (uint16_t)weft_narrow_div(0x3F80u, a, 7, 8);
}
static inline uint16_t weft_bf16_sqrt(uint16_t a) {
  return (uint16_t)weft_narrow_sqrt(a, 7, 8);
}
// The packed `f16x2` forms are the same operation twice, low half then high.
static inline uint32_t weft_f16x2_add(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_add((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_add((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_sub(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_sub((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_sub((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_mul(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_mul((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_mul((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_fma(uint32_t a, uint32_t b, uint32_t c) {
  return (uint32_t)weft_f16_fma((uint16_t)a, (uint16_t)b, (uint16_t)c) |
         ((uint32_t)weft_f16_fma((uint16_t)(a >> 16), (uint16_t)(b >> 16),
                                 (uint16_t)(c >> 16)) << 16);
}
static inline uint32_t weft_f16x2_min(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_min((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_min((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_max(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_max((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_max((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_neg(uint32_t a) {
  return weft_f16_neg((uint16_t)a) | ((uint32_t)weft_f16_neg((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_f16x2_abs(uint32_t a) {
  return weft_f16_abs((uint16_t)a) | ((uint32_t)weft_f16_abs((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_f16x2_sat(uint32_t a) {
  return weft_f16_sat((uint16_t)a) | ((uint32_t)weft_f16_sat((uint16_t)(a >> 16)) << 16);
}

static inline uint16_t weft_f16_min_nan(uint16_t a, uint16_t b) {
  const half x = weft_f16_of(a), y = weft_f16_of(b);
  if (x != x || y != y) return 0x7FFFu;
  return weft_f16_min(a, b);
}
static inline uint16_t weft_f16_max_nan(uint16_t a, uint16_t b) {
  const half x = weft_f16_of(a), y = weft_f16_of(b);
  if (x != x || y != y) return 0x7FFFu;
  return weft_f16_max(a, b);
}
static inline uint16_t weft_f16_min_xorabs(uint16_t a, uint16_t b) {
  const uint16_t mag = weft_f16_min((uint16_t)(a & 0x7FFFu), (uint16_t)(b & 0x7FFFu));
  if ((mag & 0x7C00u) == 0x7C00u && (mag & 0x03FFu) != 0u) return 0x7FFFu;
  return (uint16_t)(mag | ((a ^ b) & 0x8000u));
}
static inline uint16_t weft_f16_max_xorabs(uint16_t a, uint16_t b) {
  const uint16_t mag = weft_f16_max((uint16_t)(a & 0x7FFFu), (uint16_t)(b & 0x7FFFu));
  if ((mag & 0x7C00u) == 0x7C00u && (mag & 0x03FFu) != 0u) return 0x7FFFu;
  return (uint16_t)(mag | ((a ^ b) & 0x8000u));
}
static inline uint32_t weft_f16x2_min_nan(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_min_nan((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_min_nan((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_max_nan(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_max_nan((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_max_nan((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_min_xorabs(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_min_xorabs((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_min_xorabs((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_max_xorabs(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_max_xorabs((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_max_xorabs((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}

#ifdef __METAL_VERSION__
#define weft_b16_of(b) as_type<float>((uint32_t)(b) << 16)
#define weft_b16_u32(f) as_type<uint32_t>(f)
#else
static inline float weft_b16_of(uint16_t b) {
  const uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline uint32_t weft_b16_u32(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
#endif
// One rounding back to bf16, nearest even on the bit below the kept ones.
static inline uint16_t weft_bf16_round(float f) {
  uint32_t u = weft_b16_u32(f);
  if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline uint16_t weft_bf16_add(uint16_t a, uint16_t b) {
  return weft_bf16_round(weft_b16_of(a) + weft_b16_of(b));
}
static inline uint16_t weft_bf16_sub(uint16_t a, uint16_t b) {
  return weft_bf16_round(weft_b16_of(a) - weft_b16_of(b));
}
static inline uint16_t weft_bf16_mul(uint16_t a, uint16_t b) {
  return weft_bf16_round(weft_b16_of(a) * weft_b16_of(b));
}
static inline uint16_t weft_bf16_min(uint16_t a, uint16_t b) {
  const float x = weft_b16_of(a), y = weft_b16_of(b);
  if (x != x) return b;
  if (y != y) return a;
  if (x == y) return (uint16_t)(a & b);
  return x < y ? a : b;
}
static inline uint16_t weft_bf16_max(uint16_t a, uint16_t b) {
  const float x = weft_b16_of(a), y = weft_b16_of(b);
  if (x != x) return b;
  if (y != y) return a;
  if (x == y) return (uint16_t)(a | b);
  return x < y ? b : a;
}
static inline uint16_t weft_bf16_neg(uint16_t a) { return (uint16_t)(a ^ 0x8000u); }
static inline uint16_t weft_bf16_abs(uint16_t a) { return (uint16_t)(a & 0x7FFFu); }
static inline uint16_t weft_bf16_sat(uint16_t a) {
  const float v = weft_b16_of(a);
  if (v != v) return 0u;
  if (v < 0.0f) return 0u;
  if (v > 1.0f) return 0x3F80u;
  return a;
}
static inline uint16_t weft_bf16_min_nan(uint16_t a, uint16_t b) {
  const float x = weft_b16_of(a), y = weft_b16_of(b);
  if (x != x || y != y) return 0x7FFFu;
  return weft_bf16_min(a, b);
}
static inline uint16_t weft_bf16_max_nan(uint16_t a, uint16_t b) {
  const float x = weft_b16_of(a), y = weft_b16_of(b);
  if (x != x || y != y) return 0x7FFFu;
  return weft_bf16_max(a, b);
}
static inline uint16_t weft_bf16_min_xorabs(uint16_t a, uint16_t b) {
  const uint16_t mag = weft_bf16_min((uint16_t)(a & 0x7FFFu), (uint16_t)(b & 0x7FFFu));
  if ((mag & 0x7F80u) == 0x7F80u && (mag & 0x007Fu) != 0u) return 0x7FFFu;
  return (uint16_t)(mag | ((a ^ b) & 0x8000u));
}
static inline uint16_t weft_bf16_max_xorabs(uint16_t a, uint16_t b) {
  const uint16_t mag = weft_bf16_max((uint16_t)(a & 0x7FFFu), (uint16_t)(b & 0x7FFFu));
  if ((mag & 0x7F80u) == 0x7F80u && (mag & 0x007Fu) != 0u) return 0x7FFFu;
  return (uint16_t)(mag | ((a ^ b) & 0x8000u));
}
static inline uint32_t weft_bf16x2_add(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_add((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_add((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_sub(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_sub((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_sub((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_mul(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_mul((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_mul((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_fma(uint32_t a, uint32_t b, uint32_t c) {
  return (uint32_t)weft_bf16_fma((uint16_t)a, (uint16_t)b, (uint16_t)c) |
         ((uint32_t)weft_bf16_fma((uint16_t)(a >> 16), (uint16_t)(b >> 16),
                                  (uint16_t)(c >> 16))
          << 16);
}
static inline uint32_t weft_bf16x2_min(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_min((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_min((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_max(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_max((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_max((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_neg(uint32_t a) {
  return weft_bf16_neg((uint16_t)a) | ((uint32_t)weft_bf16_neg((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_abs(uint32_t a) {
  return weft_bf16_abs((uint16_t)a) | ((uint32_t)weft_bf16_abs((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_sat(uint32_t a) {
  return weft_bf16_sat((uint16_t)a) | ((uint32_t)weft_bf16_sat((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_min_nan(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_min_nan((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_min_nan((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_max_nan(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_max_nan((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_max_nan((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_min_xorabs(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_min_xorabs((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_min_xorabs((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_max_xorabs(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_max_xorabs((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_max_xorabs((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_div(uint32_t a, uint32_t b) {
  return (uint32_t)weft_f16_div((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_f16_div((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_f16x2_rcp(uint32_t a) {
  return weft_f16_rcp((uint16_t)a) | ((uint32_t)weft_f16_rcp((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_f16x2_sqrt(uint32_t a) {
  return weft_f16_sqrt((uint16_t)a) | ((uint32_t)weft_f16_sqrt((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_div(uint32_t a, uint32_t b) {
  return (uint32_t)weft_bf16_div((uint16_t)a, (uint16_t)b) |
         ((uint32_t)weft_bf16_div((uint16_t)(a >> 16), (uint16_t)(b >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_rcp(uint32_t a) {
  return weft_bf16_rcp((uint16_t)a) | ((uint32_t)weft_bf16_rcp((uint16_t)(a >> 16)) << 16);
}
static inline uint32_t weft_bf16x2_sqrt(uint32_t a) {
  return weft_bf16_sqrt((uint16_t)a) | ((uint32_t)weft_bf16_sqrt((uint16_t)(a >> 16)) << 16);
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

// is 32 for a warp and 4 for a quad; a lane's group is the low bits of its id.
#ifdef __METAL_VERSION__
template <class T>
inline T weft_warp_prefix(T v, bool inclusive, bool product, uint32_t width) {
  // a template's names are looked up where it is defined.
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

// run this text; `tools/pack_probe.mm` is what says the formula is Metal's.
inline float weft_pack_clamp(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Nearest, and an exact tie away from zero, the rule the spec gives `round()`.
inline int weft_pack_round(float v) {
  return v < 0.0f ? -(int)(0.5f - v) : (int)(v + 0.5f);
}

inline uint weft_pack_u8(float v) {
  return (uint)weft_pack_round(weft_pack_clamp(v, 0.0f, 1.0f) * 255.0f);
}
inline uint weft_pack_s8(float v) {
  return (uint)weft_pack_round(weft_pack_clamp(v, -1.0f, 1.0f) * 127.0f) & 0xFFu;
}
inline uint weft_pack_u16(float v) {
  return (uint)weft_pack_round(weft_pack_clamp(v, 0.0f, 1.0f) * 65535.0f) & 0xFFFFu;
}
inline uint weft_pack_s16(float v) {
  return (uint)weft_pack_round(weft_pack_clamp(v, -1.0f, 1.0f) * 32767.0f) & 0xFFFFu;
}

inline uint weft_pack_unorm4x8(float4 v) {
  return weft_pack_u8(v.x) | (weft_pack_u8(v.y) << 8) | (weft_pack_u8(v.z) << 16) |
         (weft_pack_u8(v.w) << 24);
}
inline uint weft_pack_snorm4x8(float4 v) {
  return weft_pack_s8(v.x) | (weft_pack_s8(v.y) << 8) | (weft_pack_s8(v.z) << 16) |
         (weft_pack_s8(v.w) << 24);
}
inline uint weft_pack_unorm2x16(float2 v) {
  return weft_pack_u16(v.x) | (weft_pack_u16(v.y) << 16);
}
inline uint weft_pack_snorm2x16(float2 v) {
  return weft_pack_s16(v.x) | (weft_pack_s16(v.y) << 16);
}

// A signed byte back to a number: sign-extend, divide, clamp at -1.
inline float weft_unpack_snorm_byte(uint b) {
  const float f = (float)(int)(char)(b & 0xFFu) * (1.0f / 127.0f);
  return f < -1.0f ? -1.0f : f;
}
inline float weft_unpack_snorm_half(uint h) {
  const float f = (float)(int)(short)(h & 0xFFFFu) * (1.0f / 32767.0f);
  return f < -1.0f ? -1.0f : f;
}
inline float4 weft_unpack_unorm4x8(uint p) {
  return float4((float)(p & 0xFFu) * (1.0f / 255.0f),
                (float)((p >> 8) & 0xFFu) * (1.0f / 255.0f),
                (float)((p >> 16) & 0xFFu) * (1.0f / 255.0f),
                (float)((p >> 24) & 0xFFu) * (1.0f / 255.0f));
}
inline float4 weft_unpack_snorm4x8(uint p) {
  return float4(weft_unpack_snorm_byte(p), weft_unpack_snorm_byte(p >> 8),
                weft_unpack_snorm_byte(p >> 16), weft_unpack_snorm_byte(p >> 24));
}
inline float2 weft_unpack_unorm2x16(uint p) {
  return float2((float)(p & 0xFFFFu) * (1.0f / 65535.0f),
                (float)((p >> 16) & 0xFFFFu) * (1.0f / 65535.0f));
}
inline float2 weft_unpack_snorm2x16(uint p) {
  return float2(weft_unpack_snorm_half(p), weft_unpack_snorm_half(p >> 16));
}
// Metal's order, and `any`/`all` fold the mask to one answer.
inline bool weft_any(bool2 v) { return v.x || v.y; }
inline bool weft_any(bool3 v) { return v.x || v.y || v.z; }
inline bool weft_any(bool4 v) { return v.x || v.y || v.z || v.w; }
inline bool weft_all(bool2 v) { return v.x && v.y; }
inline bool weft_all(bool3 v) { return v.x && v.y && v.z; }
inline bool weft_all(bool4 v) { return v.x && v.y && v.z && v.w; }

inline float weft_select(float a, float b, bool c) { return c ? b : a; }
inline half weft_select(half a, half b, bool c) { return c ? b : a; }
inline int weft_select(int a, int b, bool c) { return c ? b : a; }
inline uint weft_select(uint a, uint b, bool c) { return c ? b : a; }
inline float2 weft_select(float2 a, float2 b, bool2 c) {
  return float2(c.x ? b.x : a.x, c.y ? b.y : a.y);
}
inline float3 weft_select(float3 a, float3 b, bool3 c) {
  return float3(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z);
}
inline float4 weft_select(float4 a, float4 b, bool4 c) {
  return float4(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z, c.w ? b.w : a.w);
}
inline half2 weft_select(half2 a, half2 b, bool2 c) {
  return half2(c.x ? b.x : a.x, c.y ? b.y : a.y);
}
inline half3 weft_select(half3 a, half3 b, bool3 c) {
  return half3(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z);
}
inline half4 weft_select(half4 a, half4 b, bool4 c) {
  return half4(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z, c.w ? b.w : a.w);
}
inline int2 weft_select(int2 a, int2 b, bool2 c) {
  return int2(c.x ? b.x : a.x, c.y ? b.y : a.y);
}
inline int3 weft_select(int3 a, int3 b, bool3 c) {
  return int3(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z);
}
inline int4 weft_select(int4 a, int4 b, bool4 c) {
  return int4(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z, c.w ? b.w : a.w);
}
inline uint2 weft_select(uint2 a, uint2 b, bool2 c) {
  return uint2(c.x ? b.x : a.x, c.y ? b.y : a.y);
}
inline uint3 weft_select(uint3 a, uint3 b, bool3 c) {
  return uint3(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z);
}
inline uint4 weft_select(uint4 a, uint4 b, bool4 c) {
  return uint4(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z, c.w ? b.w : a.w);
}

// `select(a, b, x > 0.0f)` rather than building a mask first.
inline float2 weft_select(float2 a, float2 b, bool c) { return c ? b : a; }
inline float3 weft_select(float3 a, float3 b, bool c) { return c ? b : a; }
inline float4 weft_select(float4 a, float4 b, bool c) { return c ? b : a; }
inline half2 weft_select(half2 a, half2 b, bool c) { return c ? b : a; }
inline half3 weft_select(half3 a, half3 b, bool c) { return c ? b : a; }
inline half4 weft_select(half4 a, half4 b, bool c) { return c ? b : a; }
inline int2 weft_select(int2 a, int2 b, bool c) { return c ? b : a; }
inline int3 weft_select(int3 a, int3 b, bool c) { return c ? b : a; }
inline int4 weft_select(int4 a, int4 b, bool c) { return c ? b : a; }
inline uint2 weft_select(uint2 a, uint2 b, bool c) { return c ? b : a; }
inline uint3 weft_select(uint3 a, uint3 b, bool c) { return c ? b : a; }
inline uint4 weft_select(uint4 a, uint4 b, bool c) { return c ? b : a; }

// the fp64 text runs. The test refuses about one input in ten thousand.
#define WEFT_CR_REL 4.0e-12f     // 2^-38 of the answer, for a relative evaluation

#define WEFT_DF_ONE      weft_df{0x1.0000000000000p+0f, 0x0.0p+0f}
#define WEFT_DF_TWO      weft_df{0x1.0000000000000p+1f, 0x0.0p+0f}
#define WEFT_DF_LOG2E    weft_df{0x1.7154760000000p+0f, 0x1.4ae0c00000000p-26f}
#define WEFT_DF_LN2      weft_df{0x1.62e4300000000p-1f, -0x1.05c6100000000p-29f}
#define WEFT_DF_LOG10E   weft_df{0x1.bcb7b20000000p-2f, -0x1.5b235e0000000p-27f}
#define WEFT_DF_PI       weft_df{0x1.921fb60000000p+1f, -0x1.777a5c0000000p-24f}
#define WEFT_DF_PI_2     weft_df{0x1.921fb60000000p+0f, -0x1.777a5c0000000p-25f}
// reduction error at all, and a small result is not charged for one it never had.
#define WEFT_PI_2_RESID  1.77635702e-15f
#define WEFT_DF_INV3     weft_df{0x1.5555560000000p-2f, -0x1.5555560000000p-27f}
#define WEFT_DF_INV5     weft_df{0x1.99999a0000000p-3f, -0x1.99999a0000000p-29f}
#define WEFT_DF_INV7     weft_df{0x1.24924a0000000p-3f, -0x1.b6db6e0000000p-28f}
#define WEFT_DF_INV9     weft_df{0x1.c71c720000000p-4f, -0x1.c71c720000000p-31f}
#define WEFT_DF_INV11    weft_df{0x1.745d180000000p-4f, -0x1.745d180000000p-29f}
#define WEFT_DF_INV13    weft_df{0x1.3b13b20000000p-4f, -0x1.89d89e0000000p-29f}
#define WEFT_DF_INV15    weft_df{0x1.1111120000000p-4f, -0x1.ddddde0000000p-29f}
#define WEFT_DF_INV17    weft_df{0x1.e1e1e20000000p-5f, -0x1.e1e1e20000000p-33f}
#define WEFT_DF_INV19    weft_df{0x1.af286c0000000p-5f, -0x1.af286c0000000p-32f}
#define WEFT_DF_INV21    weft_df{0x1.8618620000000p-5f, -0x1.e79e7a0000000p-31f}
#define WEFT_DF_INV23    weft_df{0x1.642c8600000000p-5f, -0x1.bd37a80000000p-31f}
#define WEFT_DF_IF2      weft_df{0x1.0000000000000p-1f, 0x0.0p+0f}
#define WEFT_DF_IF3      weft_df{0x1.5555560000000p-3f, -0x1.5555560000000p-28f}
#define WEFT_DF_IF4      weft_df{0x1.5555560000000p-5f, -0x1.5555560000000p-30f}
#define WEFT_DF_IF5      weft_df{0x1.1111120000000p-7f, -0x1.ddddde0000000p-32f}
#define WEFT_DF_IF6      weft_df{0x1.6c16c20000000p-10f, -0x1.27d27e0000000p-35f}
#define WEFT_DF_IF7      weft_df{0x1.a01a020000000p-13f, -0x1.7f97fa0000000p-39f}
#define WEFT_DF_IF8      weft_df{0x1.a01a020000000p-16f, -0x1.7f97fa0000000p-42f}
#define WEFT_DF_IF9      weft_df{0x1.71de3a0000000p-19f, 0x1.55b1cc0000000p-45f}
#define WEFT_DF_IF10     weft_df{0x1.27e4fc0000000p-22f, -0x1.10ec140000000p-47f}
#define WEFT_DF_IF11     weft_df{0x1.ae64560000000p-26f, 0x1.fd51380000000p-52f}
#define WEFT_DF_IF12     weft_df{0x1.1eed8e0000000p-29f, 0x1.ff1b140000000p-54f}
#define WEFT_DF_IF13     weft_df{0x1.6124620000000p-33f, -0x1.8af25e0000000p-58f}
#define WEFT_DF_IF14     weft_df{0x1.93974a0000000p-37f, 0x1.180f940000000p-62f}

inline weft_df weft_df_of(float x) { return weft_df{x, 0.0f}; }
inline weft_df weft_df_neg(weft_df a) { return weft_df{-a.hi, -a.lo}; }
inline weft_df weft_df_scale(weft_df a, float k) {
  const weft_df p = weft_df_prod(a.hi, k);
  return weft_df_add(p.hi, p.lo + a.lo * k);
}
// The product of two pairs: the high product exact, the cross terms in the low word.
inline weft_df weft_df_mul(weft_df a, weft_df b) {
  const weft_df p = weft_df_prod(a.hi, b.hi);
  return weft_df_add(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi));
}
inline weft_df weft_df_sum(weft_df a, weft_df b) {
  const weft_df s = weft_df_add(a.hi, b.hi);
  return weft_df_add(s.hi, s.lo + (a.lo + b.lo));
}
inline weft_df weft_df_quot(weft_df a, weft_df b) {
  const weft_df q = weft_df_div(a.hi, b.hi);
  const weft_df r = weft_df_sub(a, weft_df_mul(q, b));
  return weft_df_add(q.hi, q.lo + r.hi / b.hi);
}
// the derivative. The hardware root only has to be close; the correction is exact.
inline weft_df weft_df_sqrtdf(weft_df a) {
  const float s = WEFT_SQRT(a.hi);
  const float d = s + s;
  const weft_df r = weft_df_sub(a, weft_df_prod(s, s));
  return weft_df_add(s, r.hi / d + r.lo / d);
}

inline float weft_next_up(float x) {
  const uint32_t u = weft_bits(x);
  return weft_float(x >= 0.0f ? u + 1u : u - 1u);
}
inline float weft_next_down(float x) {
  const uint32_t u = weft_bits(x);
  return weft_float(x > 0.0f ? u - 1u : u + 1u);
}
// Ziv's test.
inline bool weft_cr(weft_df d, float abs_err) {
  const float hi = d.hi, lo = d.lo;
  if (!(hi == hi) || !(lo == lo)) return false;
  const float ax = hi < 0.0f ? -hi : hi;
  // subnormal and the correction the test reads is the bits that underflowed.
  if (!(ax >= 1.0e-28f) || !(ax < WEFT_INF)) return false;
  const float n = lo > 0.0f ? weft_next_up(hi) : weft_next_down(hi);
  if (!(n == n) || !(n > -WEFT_INF) || !(n < WEFT_INF)) return false;
  const float gap = lo > 0.0f ? n - hi : hi - n;
  const float a = lo < 0.0f ? -lo : lo;
  return a + (ax * WEFT_CR_REL + abs_err) <= 0.5f * gap;
}

// factorial carried in a pair, so the powers are right and only the adding rounds.
inline weft_df weft_exp_series(weft_df z) {
  weft_df p = WEFT_DF_IF12;
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF11);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF10);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF9);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF8);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF7);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF6);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF5);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF4);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF3);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_IF2);
  p = weft_df_sum(weft_df_mul(p, z), WEFT_DF_ONE);
  return weft_df_sum(weft_df_mul(p, z), WEFT_DF_ONE);
}
inline weft_df weft_exp_df(weft_df x) {
  const weft_df d = weft_df_mul(x, WEFT_DF_LOG2E);
  const float k = weft_rint_even(d.hi);
  const weft_df z = weft_df_mul(weft_df_sub(d, weft_df_of(k)), WEFT_DF_LN2);
  weft_df p = weft_exp_series(z);
  p.hi = weft_ldexp(p.hi, (int)k);
  p.lo = weft_ldexp(p.lo, (int)k);
  return p;
}
// (e^x - 1)/x near zero, so the answer's own leading bits are not a difference.
inline weft_df weft_expm1_df(weft_df x) {
  if (!(x.hi < 0.5f) || !(x.hi > -0.5f))
    return weft_df_sub(weft_exp_df(x), WEFT_DF_ONE);
  weft_df p = WEFT_DF_IF13;
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF12);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF11);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF10);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF9);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF8);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF7);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF6);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF5);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF4);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF3);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_IF2);
  p = weft_df_sum(weft_df_mul(p, x), WEFT_DF_ONE);
  return weft_df_mul(x, p);
}
inline float weft_exp(float v) {
  if (!(v > -87.0f) || !(v < 88.0f)) return weft_exp_wide(v);
  const weft_df d = weft_exp_df(weft_df_of(v));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_exp_wide(v);
}
inline float weft_expm1(float x) {
  if (!(x > -87.0f) || !(x < 88.0f)) return weft_expm1_wide(x);
  const weft_df d = weft_expm1_df(weft_df_of(x));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_expm1_wide(x);
}
// a subtraction: sinh = m(2+m)/(2(1+m)), cosh = (2+2m+m^2)/(2(1+m)), tanh their ratio.
inline float weft_sinh(float v) {
  const float ax = v < 0.0f ? -v : v;
  if (!(ax < 88.0f)) return weft_sinh_wide(v);
  const weft_df m = weft_expm1_df(weft_df_of(ax));
  const weft_df num = weft_df_mul(m, weft_df_sum(WEFT_DF_TWO, m));
  const weft_df den = weft_df_scale(weft_df_sum(WEFT_DF_ONE, m), 2.0f);
  weft_df d = weft_df_quot(num, den);
  if (v < 0.0f) d = weft_df_neg(d);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_sinh_wide(v);
}
inline float weft_cosh(float v) {
  const float ax = v < 0.0f ? -v : v;
  if (!(ax < 88.0f)) return weft_cosh_wide(v);
  const weft_df m = weft_expm1_df(weft_df_of(ax));
  const weft_df one_m = weft_df_sum(WEFT_DF_ONE, m);
  const weft_df num = weft_df_sum(weft_df_sum(WEFT_DF_TWO, weft_df_scale(m, 2.0f)),
                                  weft_df_mul(m, m));
  const weft_df d = weft_df_quot(num, weft_df_scale(one_m, 2.0f));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_cosh_wide(v);
}
inline float weft_tanh(float v) {
  const float ax = v < 0.0f ? -v : v;
  if (!(ax < 88.0f)) return weft_tanh_wide(v);
  const weft_df m = weft_expm1_df(weft_df_of(ax));
  const weft_df one_m = weft_df_sum(WEFT_DF_ONE, m);
  const weft_df num = weft_df_mul(m, weft_df_sum(WEFT_DF_TWO, m));
  const weft_df den = weft_df_sum(weft_df_mul(one_m, one_m), WEFT_DF_ONE);
  weft_df d = weft_df_quot(num, den);
  if (v < 0.0f) d = weft_df_neg(d);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_tanh_wide(v);
}

// 2 atanh(t) for |t| <= 1/3, the series that turns a logarithm into additions.
inline weft_df weft_atanh_series(weft_df t) {
  const weft_df t2 = weft_df_mul(t, t);
  weft_df p = WEFT_DF_INV23;
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV21);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV19);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV17);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV15);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV13);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV11);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV9);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV7);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV5);
  p = weft_df_sum(weft_df_mul(p, t2), WEFT_DF_INV3);
  return weft_df_mul(t, weft_df_sum(weft_df_mul(p, t2), WEFT_DF_ONE));
}
inline weft_df weft_log_mant(float m) {
  const float dd = m - 1.0f;                        // exact
  // rounded divisor alone put a hundredth of an ulp into every term.
  const weft_df t = weft_df_quot(weft_df_of(dd), weft_df_add(2.0f, dd));
  return weft_df_scale(weft_atanh_series(t), 2.0f);
}
// along as its own first-order term, and ln2 is a pair so the scaling rounds once.
inline weft_df weft_log_df(weft_df a) {
  const uint32_t u = weft_bits(a.hi);
  int e = (int)((u >> 23) & 0xFFu) - 127;
  float m = weft_float((u & 0x7FFFFFu) | 0x3F800000u);
  float ml = weft_ldexp(a.lo, -e);
  if (m > 1.4142135f) { m *= 0.5f; ml *= 0.5f; e += 1; }
  const weft_df extra = weft_df_div(ml, m);
  const weft_df lm0 = weft_log_mant(m);
  const weft_df lm = weft_df_add(lm0.hi, lm0.lo + extra.hi + extra.lo);
  return weft_df_sum(lm, weft_df_mul(weft_df_of((float)e), WEFT_DF_LN2));
}
inline weft_df weft_log1p_df(weft_df u) {
  if (!(u.hi <= 0.5f) || !(u.hi >= -0.5f))
    return weft_log_df(weft_df_sum(WEFT_DF_ONE, u));
  const weft_df t = weft_df_quot(u, weft_df_sum(WEFT_DF_TWO, u));
  return weft_df_scale(weft_atanh_series(t), 2.0f);
}
inline float weft_log(float v) {
  if (v == 1.0f) return 0.0f;   // an exact zero is not a rounding
  if (!(v > 1.0e-28f) || !(v < 3.0e38f)) return weft_log_wide(v);
  const weft_df d = weft_log_df(weft_df_of(v));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_log_wide(v);
}
inline float weft_log2(float v) {
  if (v == 1.0f) return 0.0f;   // an exact zero is not a rounding
  if (!(v > 1.0e-28f) || !(v < 3.0e38f)) return weft_log2_wide(v);
  const weft_df d = weft_df_mul(weft_log_df(weft_df_of(v)), WEFT_DF_LOG2E);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_log2_wide(v);
}
inline float weft_log10(float v) {
  if (v == 1.0f) return 0.0f;   // an exact zero is not a rounding
  if (!(v > 1.0e-28f) || !(v < 3.0e38f)) return weft_log10_wide(v);
  const weft_df d = weft_df_mul(weft_log_df(weft_df_of(v)), WEFT_DF_LOG10E);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_log10_wide(v);
}
inline float weft_log1p(float v) {
  if (v == 0.0f) return v;   // an exact zero is not a rounding
  if (!(v > -1.0f) || !(v < 3.0e38f)) return weft_log1p_wide(v);
  const weft_df d = weft_log1p_df(weft_df_of(v));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_log1p_wide(v);
}
inline float weft_asinh(float v) {
  if (!(v > -3.0e38f) || !(v < 3.0e38f)) return weft_asinh_wide(v);
  const float ax = v < 0.0f ? -v : v;
  const weft_df a = weft_df_of(ax);
  const weft_df aa = weft_df_mul(a, a);
  const weft_df s = weft_df_sqrtdf(weft_df_sum(aa, WEFT_DF_ONE));
  const weft_df u = weft_df_sum(a, weft_df_quot(aa, weft_df_sum(WEFT_DF_ONE, s)));
  weft_df d = weft_log1p_df(u);
  if (v < 0.0f) d = weft_df_neg(d);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_asinh_wide(v);
}
inline float weft_acosh(float v) {
  if (!(v >= 1.0f) || !(v < 3.0e38f)) return weft_acosh_wide(v);
  if (v == 1.0f) return 0.0f;
  weft_df d;
  if (v <= 2.0f) {
    const float u = v - 1.0f;                       // exact
    const weft_df ud = weft_df_of(u);
    const weft_df w = weft_df_sum(weft_df_scale(ud, 2.0f), weft_df_mul(ud, ud));
    d = weft_log1p_df(weft_df_sum(ud, weft_df_sqrtdf(w)));
  } else {
    const weft_df vd = weft_df_of(v);
    d = weft_log_df(weft_df_sum(vd, weft_df_sqrtdf(
        weft_df_sub(weft_df_mul(vd, vd), WEFT_DF_ONE))));
  }
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_acosh_wide(v);
}
inline float weft_atanh(float v) {
  const float ax = v < 0.0f ? -v : v;
  if (!(ax < 1.0f)) return weft_atanh_wide(v);
  const weft_df t = weft_df_quot(weft_df_scale(weft_df_of(v), 2.0f),
                                 weft_df_sub(WEFT_DF_ONE, weft_df_of(v)));
  weft_df d = weft_df_scale(weft_log1p_df(t), 0.5f);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_atanh_wide(v);
}

inline float weft_sqrt(float v) {
  if (!(v >= 1.0e-28f)) return weft_sqrt_wide(v);
  const weft_df d = weft_df_sqrtdf(weft_df_of(v));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_sqrt_wide(v);
}
inline float weft_rsqrt(float v) {
  if (v == 0.0f) return 1.0f / v;
  if (!(v >= 1.0e-28f)) return weft_rsqrt_wide(v);
  const weft_df d = weft_df_quot(WEFT_DF_ONE, weft_df_sqrtdf(weft_df_of(v)));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_rsqrt_wide(v);
}
inline float weft_cbrt(float v) {
  if (!(v >= 1.0e-28f) || !(v < 3.0e38f)) return weft_cbrt_wide(v);
  const float y0 = weft_exp2(weft_log2(v) * 0.33333334f);
  const weft_df y = weft_df_of(y0);
  const weft_df t = weft_df_quot(weft_df_of(v), weft_df_mul(y, y));
  const weft_df d = weft_df_sum(y, weft_df_mul(weft_df_sub(t, y), WEFT_DF_INV3));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_cbrt_wide(v);
}
inline float weft_hypot(float x, float y) {
  const float ax = x < 0.0f ? -x : x, ay = y < 0.0f ? -y : y;
  if (!(ax == ax) || !(ay == ay)) return weft_hypot_wide(x, y);
  const float m = ax > ay ? ax : ay, n = ax > ay ? ay : ax;
  if (m == 0.0f) return 0.0f;
  if (!(m < 3.0e38f)) return weft_hypot_wide(x, y);
  const weft_df r = weft_df_quot(weft_df_of(n), weft_df_of(m));
  const weft_df s = weft_df_sqrtdf(weft_df_sum(weft_df_mul(r, r), WEFT_DF_ONE));
  const weft_df d = weft_df_mul(weft_df_of(m), s);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_hypot_wide(x, y);
}

// k*pi/2 keeps its digits while k stays under 2^8. Past that the fp64 text reduces.
struct weft_red { weft_df r; int q; float err; };
inline weft_red weft_trig_red(float x) {
  const float k = weft_rint_even(x * 0.63661975f);
  weft_red out;
  out.r = weft_df_sub(weft_df_of(x), weft_df_mul(weft_df_of(k), WEFT_DF_PI_2));
  out.q = ((int)(k - 4.0f * weft_rint_even(k * 0.25f))) & 3;
  out.err = (k < 0.0f ? -k : k) * WEFT_PI_2_RESID;
  return out;
}
// sin r = r(1 - r^2/6 + ... + r^12/13!), the coefficients in pairs.
inline weft_df weft_sin_core(weft_df r, weft_df r2) {
  weft_df p = WEFT_DF_IF13;
  p = weft_df_sum(weft_df_mul(p, r2), weft_df_neg(WEFT_DF_IF11));
  p = weft_df_sum(weft_df_mul(p, r2), WEFT_DF_IF9);
  p = weft_df_sum(weft_df_mul(p, r2), weft_df_neg(WEFT_DF_IF7));
  p = weft_df_sum(weft_df_mul(p, r2), WEFT_DF_IF5);
  p = weft_df_sum(weft_df_mul(p, r2), weft_df_neg(WEFT_DF_IF3));
  return weft_df_mul(r, weft_df_sum(weft_df_mul(p, r2), WEFT_DF_ONE));
}
// cos r = 1 - r^2/2 + ... + r^14/14!
inline weft_df weft_cos_core(weft_df r2) {
  weft_df p = weft_df_neg(WEFT_DF_IF14);
  p = weft_df_sum(weft_df_mul(p, r2), WEFT_DF_IF12);
  p = weft_df_sum(weft_df_mul(p, r2), weft_df_neg(WEFT_DF_IF10));
  p = weft_df_sum(weft_df_mul(p, r2), WEFT_DF_IF8);
  p = weft_df_sum(weft_df_mul(p, r2), weft_df_neg(WEFT_DF_IF6));
  p = weft_df_sum(weft_df_mul(p, r2), WEFT_DF_IF4);
  p = weft_df_sum(weft_df_mul(p, r2), weft_df_neg(WEFT_DF_IF2));
  return weft_df_sum(weft_df_mul(p, r2), WEFT_DF_ONE);
}
// and weft_cr still refuses any lane whose last bit it cannot prove.
inline bool weft_trig_ok(float x) {
  const float ax = x < 0.0f ? -x : x;
  return ax <= 65536.0f;
}
// in Metal, and the two engines have to compile the same text.
struct weft_sc { weft_df s, c; bool ok; float err; };
struct weft_dfr { weft_df d; bool ok; };
inline weft_sc weft_trig_pair(float x) {
  weft_sc out;
  out.ok = false;
  out.err = 0.0f;
  out.s = weft_df_of(0.0f);
  out.c = weft_df_of(0.0f);
  if (!weft_trig_ok(x)) return out;
  const weft_red red = weft_trig_red(x);
  if (!(red.r.hi < 0.786f) || !(red.r.hi > -0.786f)) return out;
  const weft_df r2 = weft_df_mul(red.r, red.r);
  const weft_df si = weft_sin_core(red.r, r2);
  const weft_df co = weft_cos_core(r2);
  if (red.q == 0) { out.s = si; out.c = co; }
  else if (red.q == 1) { out.s = co; out.c = weft_df_neg(si); }
  else if (red.q == 2) { out.s = weft_df_neg(si); out.c = weft_df_neg(co); }
  else { out.s = weft_df_neg(co); out.c = si; }
  out.err = red.err;
  out.ok = true;
  return out;
}
inline float weft_sin(float x) {
  if (x == 0.0f) return x;                 // an exact zero keeps its sign
  const weft_sc p = weft_trig_pair(x);
  if (p.ok && weft_cr(p.s, p.err)) return p.s.hi;
  return weft_sin_wide(x);
}
inline float weft_cos(float x) {
  const weft_sc p = weft_trig_pair(x);
  if (p.ok && weft_cr(p.c, p.err)) return p.c.hi;
  return weft_cos_wide(x);
}
inline float weft_tan(float x) {
  if (x == 0.0f) return x;                 // an exact zero keeps its sign
  const weft_sc p = weft_trig_pair(x);
  if (p.ok) {
    const weft_df c = p.c;
    const weft_df d = weft_df_quot(p.s, c);
    // one, and near a pole that is whole ulps: this is the room the test is given.
    const float ax = x < 0.0f ? -x : x;
    const float dr = 1.6e-15f * (ax + 1.0f);
    const float ac = c.hi < 0.0f ? -c.hi : c.hi;
    const float ad = d.hi < 0.0f ? -d.hi : d.hi;
    if (weft_cr(d, ac > 0.0f ? ad * (dr / ac) : 0.0f)) return d.hi;
  }
  return weft_tan_wide(x);
}
// a zero and not the residue of a rounded pi.
inline weft_dfr weft_sinpi_pair(float x) {
  weft_dfr out;
  out.d = weft_df_of(0.0f);
  out.ok = false;
  const float ax = x < 0.0f ? -x : x;
  if (!(ax < 8388608.0f)) return out;
  const float n = weft_rint_even(x);
  const float f = x - n;                                  // exact
  const int par = ((int)(n - 2.0f * weft_rint_even(n * 0.5f))) & 1;
  const float af = f < 0.0f ? -f : f;
  // and cannot hold the correction at all: that end of the range is the fp64 text's.
  if (n == 0.0f && af < 1.0e-20f) return out;
  weft_df d;
  if (af <= 0.25f) {
    const weft_df r = weft_df_mul(weft_df_of(f), WEFT_DF_PI);
    d = weft_sin_core(r, weft_df_mul(r, r));
  } else {
    const float g = af - 0.5f;                            // exact
    const weft_df r = weft_df_mul(weft_df_of(g), WEFT_DF_PI);
    d = weft_cos_core(weft_df_mul(r, r));
    if (f < 0.0f) d = weft_df_neg(d);
  }
  if (par != 0 && f != 0.0f) d = weft_df_neg(d);
  out.d = d;
  out.ok = true;
  return out;
}
inline float weft_sinpi(float x) {
  const weft_dfr p = weft_sinpi_pair(x);
  if (p.ok && weft_cr(p.d, 6.0e-15f)) return p.d.hi;
  return weft_sinpi_wide(x);
}
inline float weft_cospi(float x) {
  const float ax = x < 0.0f ? -x : x;
  if (!(ax < 8388608.0f)) return weft_cospi_wide(x);
  const float n = weft_rint_even(x);
  const float f = x - n;                                  // exact
  const int par = ((int)(n - 2.0f * weft_rint_even(n * 0.5f))) & 1;
  const float af = f < 0.0f ? -f : f;
  if (n == 0.0f && af < 1.0e-20f) return 1.0f;             // cos of an angle this small
  weft_df d;
  if (af <= 0.25f) {
    const weft_df r = weft_df_mul(weft_df_of(f), WEFT_DF_PI);
    d = weft_cos_core(weft_df_mul(r, r));
  } else {
    const float g = 0.5f - af;                            // exact
    const weft_df r = weft_df_mul(weft_df_of(g), WEFT_DF_PI);
    d = weft_sin_core(r, weft_df_mul(r, r));
  }
  if (par != 0) d = weft_df_neg(d);
  if (weft_cr(d, 6.0e-15f)) return d.hi;
  return weft_cospi_wide(x);
}
inline float weft_tanpi(float x) {
  const float ax = x < 0.0f ? -x : x;
  if (!(ax < 8388608.0f)) return weft_tanpi_wide(x);
  const float n = weft_rint_even(x);
  const float f = x - n;                                  // exact
  const float af = f < 0.0f ? -f : f;
  if (n == 0.0f && af < 1.0e-20f) return weft_tanpi_wide(x);
  const bool cot = af > 0.25f;
  const float g = cot ? (f > 0.0f ? f - 0.5f : f + 0.5f) : f;   // exact, |g| <= 0.25
  const weft_df r = weft_df_mul(weft_df_of(g), WEFT_DF_PI);
  const weft_df r2 = weft_df_mul(r, r);
  const weft_df si = weft_sin_core(r, r2);
  const weft_df co = weft_cos_core(r2);
  const weft_df d = cot ? weft_df_neg(weft_df_quot(co, si)) : weft_df_quot(si, co);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_tanpi_wide(x);
}

// halving through a root so the series sees at most a fifth.
inline weft_df weft_atan_series(weft_df u) {
  const weft_df u2 = weft_df_mul(u, u);
  weft_df p = weft_df_neg(WEFT_DF_INV23);
  p = weft_df_sum(weft_df_mul(p, u2), WEFT_DF_INV21);
  p = weft_df_sum(weft_df_mul(p, u2), weft_df_neg(WEFT_DF_INV19));
  p = weft_df_sum(weft_df_mul(p, u2), WEFT_DF_INV17);
  p = weft_df_sum(weft_df_mul(p, u2), weft_df_neg(WEFT_DF_INV15));
  p = weft_df_sum(weft_df_mul(p, u2), WEFT_DF_INV13);
  p = weft_df_sum(weft_df_mul(p, u2), weft_df_neg(WEFT_DF_INV11));
  p = weft_df_sum(weft_df_mul(p, u2), WEFT_DF_INV9);
  p = weft_df_sum(weft_df_mul(p, u2), weft_df_neg(WEFT_DF_INV7));
  p = weft_df_sum(weft_df_mul(p, u2), WEFT_DF_INV5);
  p = weft_df_sum(weft_df_mul(p, u2), weft_df_neg(WEFT_DF_INV3));
  return weft_df_mul(u, weft_df_sum(weft_df_mul(p, u2), WEFT_DF_ONE));
}
inline weft_df weft_atan_df(weft_df t) {
  const bool neg = t.hi < 0.0f;
  weft_df a = neg ? weft_df_neg(t) : t;
  bool refl = false;
  if (a.hi > 1.0f) { a = weft_df_quot(WEFT_DF_ONE, a); refl = true; }
  const weft_df s = weft_df_sqrtdf(weft_df_sum(weft_df_mul(a, a), WEFT_DF_ONE));
  const weft_df b = weft_df_quot(a, weft_df_sum(WEFT_DF_ONE, s));
  weft_df r = weft_df_scale(weft_atan_series(b), 2.0f);
  if (refl) r = weft_df_sub(WEFT_DF_PI_2, r);
  return neg ? weft_df_neg(r) : r;
}
inline weft_df weft_atan2_df(weft_df y, weft_df x) {
  if (x.hi > 0.0f) return weft_atan_df(weft_df_quot(y, x));
  if (x.hi < 0.0f) {
    const weft_df a = weft_atan_df(weft_df_quot(y, x));
    return y.hi >= 0.0f ? weft_df_sum(a, WEFT_DF_PI) : weft_df_sub(a, WEFT_DF_PI);
  }
  return y.hi > 0.0f ? WEFT_DF_PI_2 : (y.hi < 0.0f ? weft_df_neg(WEFT_DF_PI_2) : weft_df_of(0.0f));
}
inline float weft_atan(float x) {
  const float ax = x < 0.0f ? -x : x;
  if (!(ax < 1.0e10f)) return weft_atan_wide(x);
  const weft_df d = weft_atan_df(weft_df_of(x));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_atan_wide(x);
}
inline float weft_atan2(float y, float x) {
  if (!(y == y) || !(x == x)) return weft_atan2_wide(y, x);
  const float ay = y < 0.0f ? -y : y, ax = x < 0.0f ? -x : x;
  if (!(ay >= 1.0e-28f) || !(ay < 3.0e38f)) return weft_atan2_wide(y, x);
  if (!(ax >= 1.0e-28f) || !(ax < 3.0e38f)) return weft_atan2_wide(y, x);
  const weft_df d = weft_atan2_df(weft_df_of(y), weft_df_of(x));
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_atan2_wide(y, x);
}
inline float weft_asin(float x) {
  const float ax = x < 0.0f ? -x : x;
  if (!(ax <= 1.0f)) return weft_asin_wide(x);
  const weft_df xd = weft_df_of(x);
  const weft_df s = weft_df_sqrtdf(weft_df_sub(WEFT_DF_ONE, weft_df_mul(xd, xd)));
  const weft_df d = weft_atan2_df(xd, s);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_asin_wide(x);
}
inline float weft_acos(float x) {
  const float ax = x < 0.0f ? -x : x;
  if (!(ax <= 1.0f)) return weft_acos_wide(x);
  const weft_df xd = weft_df_of(x);
  const weft_df s = weft_df_sqrtdf(weft_df_sub(WEFT_DF_ONE, weft_df_mul(xd, xd)));
  const weft_df d = weft_atan2_df(s, xd);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_acos_wide(x);
}

inline float weft_pown(float x, int n) {
  if (!(x > 1.0e-28f) || !(x < 3.0e38f) || n == 0 || n > 64 || n < -64)
    return weft_pown_wide(x, n);
  weft_df a = weft_df_of(x), r = WEFT_DF_ONE;
  int m = n < 0 ? -n : n;
  bool safe = true;
  while (m > 0) {
    if ((m & 1) != 0) {
      r = weft_df_mul(r, a);
      // pair's correction is the bits that underflowed. Every step has to stay in band.
      if (!(r.hi > 1.0e-28f) || !(r.hi < 3.0e38f)) { safe = false; break; }
    }
    m >>= 1;
    if (m > 0) {
      a = weft_df_mul(a, a);
      if (!(a.hi > 1.0e-28f) || !(a.hi < 3.0e38f)) { safe = false; break; }
    }
  }
  if (!safe) return weft_pown_wide(x, n);
  if (n < 0) r = weft_df_quot(WEFT_DF_ONE, r);
  if (weft_cr(r, 0.0f)) return r.hi;
  return weft_pown_wide(x, n);
}
inline float weft_rootn(float x, int n) {
  if (!(x > 1.0e-28f) || !(x < 3.0e38f) || n == 0 || n > 64 || n < -64)
    return weft_rootn_wide(x, n);
  const weft_df l = weft_df_quot(weft_log_df(weft_df_of(x)),
                                 weft_df_of((float)(n < 0 ? -n : n)));
  if (!(l.hi > -87.0f) || !(l.hi < 88.0f)) return weft_rootn_wide(x, n);
  weft_df d = weft_exp_df(l);
  if (n < 0) d = weft_df_quot(WEFT_DF_ONE, d);
  if (weft_cr(d, 0.0f)) return d.hi;
  return weft_rootn_wide(x, n);
}
inline float weft_powr(float x, float y) {
  if (x == 0.0f) {
    if (y > 0.0f) return 0.0f;
    if (y < 0.0f) return 1.0f / 0.0f;
    return (x - x) / (x - x);
  }
  if (!(x > 1.0e-28f) || !(x < 3.0e38f)) return weft_powr_wide(x, y);
  const weft_df p = weft_df_mul(weft_log_df(weft_df_of(x)), weft_df_of(y));
  if (!(p.hi > -87.0f) || !(p.hi < 88.0f)) return weft_powr_wide(x, y);
  const weft_df d = weft_exp_df(p);
  const float ap = p.hi < 0.0f ? -p.hi : p.hi;
  if (weft_cr(d, ap * (d.hi < 0.0f ? -d.hi : d.hi) * 1.0e-13f)) return d.hi;
  return weft_powr_wide(x, y);
}
// Three whose fast path is not a reduction away from a series.
#define WEFT_DF_HALF      weft_df{0x1.0000000000000p-1f, 0x0.0p+0f}
#define WEFT_DF_1SQRTPI   weft_df{0x1.20dd760000000p-1f, -0x1.f7ac920000000p-26f}   // 1/sqrt(pi)
#define WEFT_DF_2SQRTPI   weft_df{0x1.20dd760000000p+0f, -0x1.f7ac920000000p-25f}   // 2/sqrt(pi)
#define WEFT_DF_HALF_LN_2PI weft_df{0x1.d67f1c0000000p-1f, 0x1.0c97d60000000p-26f}   // ln(sqrt(2 pi))

// Stirling's coefficients: the Bernoulli numbers over their own indices.
#define WEFT_DF_ST1       weft_df{0x1.5555560000000p-4f, -0x1.5555560000000p-29f}   // 1/12
#define WEFT_DF_ST2       weft_df{-0x1.6c16c20000000p-9f, 0x1.27d27e0000000p-34f}   // -1/360
#define WEFT_DF_ST3       weft_df{0x1.a01a020000000p-11f, -0x1.7f97fa0000000p-37f}   // 1/1260
#define WEFT_DF_ST4       weft_df{-0x1.3813820000000p-11f, 0x1.8fd8fe0000000p-36f}   // -1/1680
#define WEFT_DF_ST5       weft_df{0x1.b951e20000000p-11f, 0x1.631fe40000000p-36f}   // 1/1188
#define WEFT_DF_ST6       weft_df{-0x1.f6ab0e0000000p-10f, 0x1.99b0e00000000p-36f}   // -691/360360
#define WEFT_DF_ST7       weft_df{0x1.a41a420000000p-8f, -0x1.6f96fa0000000p-34f}   // 1/156
#define WEFT_DF_ST8       weft_df{-0x1.e4286c0000000p-6f, -0x1.61ea720000000p-31f}   // -3617/122400

// erf's Taylor coefficients: m_n = (-1)^n / (n! (2n + 1)), as pairs.
#define WEFT_DF_ERF_C0    weft_df{0x1.0000000000000p+0f, 0x0.0p+0f}   // 1/(0!*1)
#define WEFT_DF_ERF_C1    weft_df{-0x1.5555560000000p-2f, 0x1.5555560000000p-27f}   // 1/(1!*3)
#define WEFT_DF_ERF_C2    weft_df{0x1.99999a0000000p-4f, -0x1.99999a0000000p-30f}   // 1/(2!*5)
#define WEFT_DF_ERF_C3    weft_df{-0x1.8618620000000p-6f, 0x1.e79e7a0000000p-32f}   // 1/(3!*7)
#define WEFT_DF_ERF_C4    weft_df{0x1.2f684c0000000p-8f, -0x1.2f684c0000000p-35f}   // 1/(4!*9)
#define WEFT_DF_ERF_C5    weft_df{-0x1.8d30180000000p-11f, -0x1.a6031a0000000p-36f}   // 1/(5!*11)
#define WEFT_DF_ERF_C6    weft_df{0x1.c01c020000000p-14f, -0x1.ff1ff20000000p-41f}   // 1/(6!*13)
#define WEFT_DF_ERF_C7    weft_df{-0x1.bbd77a0000000p-17f, 0x1.99621e0000000p-42f}   // 1/(7!*15)
#define WEFT_DF_ERF_C8    weft_df{0x1.87a0020000000p-20f, -0x1.e17ffa0000000p-46f}   // 1/(8!*17)
#define WEFT_DF_ERF_C9    weft_df{-0x1.3777c60000000p-23f, 0x1.552e660000000p-48f}   // 1/(9!*19)
#define WEFT_DF_ERF_C10   weft_df{0x1.c2e3060000000p-27f, -0x1.6f1e9a0000000p-52f}   // 1/(10!*21)
#define WEFT_DF_ERF_C11   weft_df{-0x1.2b67320000000p-30f, 0x1.eaac180000000p-55f}   // 1/(11!*23)
#define WEFT_DF_ERF_C12   weft_df{0x1.6f448e0000000p-34f, 0x1.3e85e00000000p-62f}   // 1/(12!*25)
#define WEFT_DF_ERF_C13   weft_df{-0x1.a289ee0000000p-38f, -0x1.f903dc0000000p-64f}   // 1/(13!*27)
#define WEFT_DF_ERF_C14   weft_df{0x1.bd577e0000000p-42f, 0x1.9634080000000p-68f}   // 1/(14!*29)
#define WEFT_DF_ERF_C15   weft_df{-0x1.bc62500000000p-46f, -0x1.f628460000000p-71f}   // 1/(15!*31)
#define WEFT_DF_ERF_C16   weft_df{0x1.a173a20000000p-50f, -0x1.3008b60000000p-75f}   // 1/(16!*33)
#define WEFT_DF_ERF_C17   weft_df{-0x1.7271cc0000000p-54f, 0x1.a79c140000000p-82f}   // 1/(17!*35)
#define WEFT_DF_ERF_C18   weft_df{0x1.377c220000000p-58f, -0x1.de1bf00000000p-83f}   // 1/(18!*37)
#define WEFT_DF_ERF_C19   weft_df{-0x1.f1b4080000000p-63f, 0x1.8996b40000000p-88f}   // 1/(19!*39)
#define WEFT_DF_ERF_C20   weft_df{0x1.7abd720000000p-67f, 0x1.2c7db80000000p-94f}   // 1/(20!*41)
#define WEFT_DF_ERF_C21   weft_df{-0x1.13246a0000000p-71f, -0x1.79c37c0000000p-96f}   // 1/(21!*43)
#define WEFT_DF_ERF_C22   weft_df{0x1.7e6b820000000p-76f, -0x1.8fa6580000000p-101f}   // 1/(22!*45)
#define WEFT_DF_ERF_C23   weft_df{-0x1.fd6bec0000000p-81f, 0x1.4d77c20000000p-108f}   // 1/(23!*47)
#define WEFT_DF_ERF_C24   weft_df{0x1.45c0a80000000p-85f, 0x1.c77f2c0000000p-112f}   // 1/(24!*49)
#define WEFT_DF_ERF_C25   weft_df{-0x1.909c9e0000000p-90f, 0x1.c5ce3c0000000p-118f}   // 1/(25!*51)
#define WEFT_DF_ERF_C26   weft_df{0x1.da74600000000p-95f, 0x1.5539760000000p-121f}   // 1/(26!*53)
#define WEFT_DF_ERF_C27   weft_df{-0x1.0eef300000000p-99f, -0x1.f421a60000000p-124f}   // 1/(27!*55)
#define WEFT_DF_ERF_C28   weft_df{0x1.2ac6540000000p-104f, -0x1.e821a00000000p-130f}   // 1/(28!*57)
#define WEFT_DF_ERF_C29   weft_df{-0x1.3e81bc0000000p-109f, 0x1.51fc000000000p-134f}   // 1/(29!*59)
#define WEFT_DF_ERF_C30   weft_df{0x1.4899fc0000000p-114f, 0x1.be00000000000p-139f}   // 1/(30!*61)
#define WEFT_DF_ERF_C31   weft_df{-0x1.486eea0000000p-119f, -0x1.0000000000000p-146f}   // 1/(31!*63)
#define WEFT_DF_ERF_C32   weft_df{0x1.3e53de0000000p-124f, 0x1.0000000000000p-148f}   // 1/(32!*65)
#define WEFT_DF_ERF_C33   weft_df{-0x1.2b77900000000p-129f, 0x0.0p+0f}   // 1/(33!*67)
#define WEFT_DF_ERF_C34   weft_df{0x1.11ae000000000p-134f, 0x0.0p+0f}   // 1/(34!*69)
#define WEFT_DF_ERF_C35   weft_df{-0x1.e680000000000p-140f, 0x0.0p+0f}   // 1/(35!*71)
#define WEFT_DF_ERF_C36   weft_df{0x1.a000000000000p-145f, 0x0.0p+0f}   // 1/(36!*73)

inline weft_df weft_erf_tail(weft_df u, weft_df p) {
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C18);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C17);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C16);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C15);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C14);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C13);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C12);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C11);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C10);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C9);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C8);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C7);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C6);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C5);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C4);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C3);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C2);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C1);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C0);
  return p;
}
inline weft_df weft_erf_series19(weft_df u) { return weft_erf_tail(u, WEFT_DF_ERF_C19); }
inline weft_df weft_erf_series36(weft_df u) {
  weft_df p = WEFT_DF_ERF_C36;
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C35);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C34);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C33);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C32);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C31);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C30);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C29);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C28);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C27);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C26);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C25);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C24);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C23);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C22);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C21);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C20);
  p = weft_df_sum(weft_df_mul(p, u), WEFT_DF_ERF_C19);
  return weft_erf_tail(u, p);
}

// asked, and the sign is the caller's.
inline weft_df weft_erf_series_df(float ax) {
  const weft_df u = weft_df_prod(ax, ax);              // the square is exact
  const weft_df m = ax <= 1.2f ? weft_erf_series19(u) : weft_erf_series36(u);
  return weft_df_mul(weft_df_mul(weft_df_of(ax), WEFT_DF_2SQRTPI), m);
}

// Laplace's continued fraction for erfc, in x rather than in x^2: erfc(x)
// sqrt(pi) e^(x^2) = 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...))))).
inline weft_df weft_erfc_cf_df(float x) {
  const weft_df xd = weft_df_of(x);
  const int n = 9 + (int)(78.0f / x);
  weft_df f = weft_df_of(0.0f);
  for (int i = n; i >= 1; i--) {
    f = weft_df_quot(weft_df_of(0.5f * (float)i), weft_df_sum(xd, f));
  }
  return weft_df_quot(WEFT_DF_ONE, weft_df_sum(xd, f));
}
inline weft_df weft_erfc_df(float x) {
  const weft_df z = weft_df_prod(x, x);
  const weft_df e = weft_exp_df(weft_df_neg(z));
  return weft_df_mul(weft_df_mul(e, weft_erfc_cf_df(x)), WEFT_DF_1SQRTPI);
}

inline float weft_erf(float v) {
  const float ax = v < 0.0f ? -v : v;
  if (!(ax == ax)) return weft_erf_wide(v);
  if (ax == 0.0f) return v;                            // the sign of the zero
  if (!(ax < 4.0f)) return v > 0.0f ? 1.0f : -1.0f;
  weft_df d;
  if (ax <= 2.5f) {
    d = weft_erf_series_df(ax);
  } else {
    d = weft_df_sub(WEFT_DF_ONE, weft_erfc_df(ax));
  }
  if (v < 0.0f) d = weft_df_neg(d);
  const float ad = d.hi < 0.0f ? -d.hi : d.hi;
  // answers it can still certify.
  if (weft_cr(d, ad * 9.1e-13f)) return d.hi;
  return weft_erf_wide(v);
}

inline float weft_erfc(float v) {
  const float ax = v < 0.0f ? -v : v;
  if (!(ax == ax)) return weft_erfc_wide(v);
  if (!(ax < 12.0f)) return v > 0.0f ? 0.0f : 2.0f;
  weft_df d;
  float abs_err = 0.0f;
  if (ax <= 2.5f) {
    // decides with it whether to certify or to fetch.
    weft_df e = weft_erf_series_df(ax);
    if (v < 0.0f) e = weft_df_neg(e);
    d = weft_df_sub(WEFT_DF_ONE, e);
    abs_err = 1.2e-13f;
  } else if (v > 0.0f) {
    d = weft_erfc_df(ax);
  } else {
    d = weft_df_sub(WEFT_DF_TWO, weft_erfc_df(ax));    // erfc(-x) = 2 - erfc(x)
  }
  const float ad = d.hi < 0.0f ? -d.hi : d.hi;
  if (weft_cr(d, abs_err + ad * 9.1e-13f)) return d.hi;
  return weft_erfc_wide(v);
}

inline weft_df weft_lgamma_df(weft_df z) {
  const weft_df w = weft_df_quot(WEFT_DF_ONE, z);
  const weft_df w2 = weft_df_mul(w, w);
  weft_df s = WEFT_DF_ST8;
  s = weft_df_sum(weft_df_mul(s, w2), WEFT_DF_ST7);
  s = weft_df_sum(weft_df_mul(s, w2), WEFT_DF_ST6);
  s = weft_df_sum(weft_df_mul(s, w2), WEFT_DF_ST5);
  s = weft_df_sum(weft_df_mul(s, w2), WEFT_DF_ST4);
  s = weft_df_sum(weft_df_mul(s, w2), WEFT_DF_ST3);
  s = weft_df_sum(weft_df_mul(s, w2), WEFT_DF_ST2);
  s = weft_df_sum(weft_df_mul(s, w2), WEFT_DF_ST1);
  weft_df r = weft_df_mul(weft_df_sub(z, WEFT_DF_HALF), weft_log_df(z));
  r = weft_df_sub(r, z);
  r = weft_df_sum(r, WEFT_DF_HALF_LN_2PI);
  return weft_df_sum(r, weft_df_mul(s, w));
}
// what it climbed is a pair, and one exponential divides out of the logarithm.
inline weft_dfr weft_tgamma_pos_df(weft_df vd) {
  weft_dfr out;
  out.d = weft_df_of(0.0f);
  out.ok = false;
  const float v = vd.hi;
  int k = 0;
  if (v < 8.0f) {
    const float need = 8.0f - v;
    k = (int)need;
    if ((float)k < need) k += 1;
  }
  if (k > 8) return out;
  weft_df p = WEFT_DF_ONE;
  for (int j = 0; j < k; j++) p = weft_df_mul(p, weft_df_sum(vd, weft_df_of((float)j)));
  const weft_df z = weft_df_sum(vd, weft_df_of((float)k));
  const weft_df l = weft_lgamma_df(z);
  const weft_df g = weft_exp_df(l);
  out.d = weft_df_quot(g, p);
  out.ok = true;
  return out;
}
// 2^-43 of ln Gamma, which is 2^-39 of the answer, so the test is told 2^-36.
inline float weft_tgamma(float v) {
  if (!(v == v)) return weft_tgamma_wide(v);
  if (v > 0.0f) {
    if (!(v < 35.0f)) return weft_tgamma_wide(v);
    const weft_dfr p = weft_tgamma_pos_df(weft_df_of(v));
    if (p.ok) {
      const float ad = p.d.hi < 0.0f ? -p.d.hi : p.d.hi;
      if (weft_cr(p.d, ad * 1.5e-11f)) return p.d.hi;
    }
    return weft_tgamma_wide(v);
  }
  // the fp64 text's, and the reflection only sees the rest.
  if (!(v > -30.0f)) return weft_tgamma_wide(v);
  if (v == weft_rint_even(v)) return weft_tgamma_wide(v);
  const weft_dfr s = weft_sinpi_pair(v);
  // 1 - v as a pair.
  const weft_dfr g = weft_tgamma_pos_df(weft_df_sub(WEFT_DF_ONE, weft_df_of(v)));
  if (s.ok && g.ok) {
    const weft_df d = weft_df_quot(WEFT_DF_PI, weft_df_mul(s.d, g.d));
    const float ad = d.hi < 0.0f ? -d.hi : d.hi;
    if (weft_cr(d, ad * 1.5e-11f)) return d.hi;
  }
  return weft_tgamma_wide(v);
}

inline weft_df weft_small_sin_df(weft_df z) {
  const weft_df z2 = weft_df_mul(z, z);
  weft_df term = z;
  weft_df s = z;
  for (int k = 1; k <= 12; k++) {
    term = weft_df_quot(weft_df_mul(term, z2), weft_df_of((float)((2 * k) * (2 * k + 1))));
    s = (k & 1) ? weft_df_sub(s, term) : weft_df_sum(s, term);
  }
  return s;
}

inline weft_df weft_df_renorm(weft_df a) {
  const float s = a.hi + a.lo;
  return weft_df{s, a.lo - (s - a.hi)};
}

// product taken off the end instead of the product itself.
inline weft_dfr weft_lgamma_pos_df(weft_df vd) {
  weft_dfr out;
  out.d = weft_df_of(0.0f);
  out.ok = false;
  const float v = vd.hi;
  int k = 0;
  if (v < 8.0f) {
    const float need = 8.0f - v;
    k = (int)need;
    if ((float)k < need) k += 1;
  }
  if (k > 8) return out;
  weft_df p = WEFT_DF_ONE;
  for (int j = 0; j < k; j++) p = weft_df_mul(p, weft_df_sum(vd, weft_df_of((float)j)));
  const weft_df z = weft_df_sum(vd, weft_df_of((float)k));
  const weft_df l = weft_lgamma_df(z);
  out.d = k > 0 ? weft_df_renorm(weft_df_sub(l, weft_log_df(p))) : l;
  out.ok = true;
  return out;
}

// itself has overflowed, which is why a kernel asks for the log instead.
inline float weft_lgamma(float v) {
  if (!(v == v)) return v + v;                      // NaN in, NaN out
  if (!(v > 0.0f)) {
    if (v == 0.0f) return weft_float(0x7F800000u);  // either zero is the pole
    if (!(v > -8388608.0f)) return weft_float(0x7F800000u);  // all integers: all poles
    if (v == weft_rint_even(v)) return weft_float(0x7F800000u);
    const weft_dfr g = weft_lgamma_pos_df(weft_df_sub(WEFT_DF_ONE, weft_df_of(v)));
    if (!g.ok) return weft_float(0xFF800000u);
    float f = v - WEFT_FLOOR(v);                       // 0 <= f < 1, exact below 2^23
    if (f > 0.5f) f = 1.0f - f;                     // sin(pi f) = sin(pi (1 - f))
    const weft_df a = weft_small_sin_df(weft_df_mul(weft_df_of(f), WEFT_DF_PI));
    const weft_df r = weft_df_sub(weft_log_df(WEFT_DF_PI), weft_log_df(a));
    return weft_df_renorm(weft_df_sub(r, g.d)).hi;
  }
  if (v == 1.0f || v == 2.0f) return 0.0f;          // the two exact zeros
  if (!(v < 4.2e36f)) return weft_float(0x7F800000u);  // ln Gamma past the float's range
  const weft_dfr p = weft_lgamma_pos_df(weft_df_of(v));
  return p.ok ? p.d.hi : weft_float(0x7F800000u);
}

// would have to name the caller's address space, and this text belongs to neither.
inline int weft_lgamma_sign(float v) {
  if (!(v < 0.0f) || !(v > -8388608.0f)) return 1;  // positive, a pole, or not a number
  const float w = WEFT_FLOOR(v);
  if (!(v - w > 0.0f)) return 1;                    // a pole answers the same sign
  return ((int)w & 1) ? -1 : 1;
}

// operator on the pair is the comparison the ISA means.
inline float weft_f16_val(uint16_t a) { return (float)weft_f16_of(a); }
inline float weft_bf16_val(uint16_t a) { return weft_b16_of(a); }

// library inside the one rounding.
inline uint16_t weft_f16_rnd(float v) { return weft_f16_bits((half)v); }
inline uint16_t weft_f16_ex2(uint16_t a) { return weft_f16_rnd(weft_exp2(weft_f16_val(a))); }
inline uint16_t weft_f16_lg2(uint16_t a) { return weft_f16_rnd(weft_log2(weft_f16_val(a))); }
inline uint16_t weft_f16_rsqrt(uint16_t a) {
  return weft_f16_rnd(weft_rsqrt(weft_f16_val(a)));
}
inline uint16_t weft_f16_sin(uint16_t a) { return weft_f16_rnd(weft_sin(weft_f16_val(a))); }
inline uint16_t weft_f16_cos(uint16_t a) { return weft_f16_rnd(weft_cos(weft_f16_val(a))); }
inline uint16_t weft_f16_tanh(uint16_t a) { return weft_f16_rnd(weft_tanh(weft_f16_val(a))); }
inline uint16_t weft_bf16_ex2(uint16_t a) {
  return weft_bf16_round(weft_exp2(weft_bf16_val(a)));
}
inline uint16_t weft_bf16_lg2(uint16_t a) {
  return weft_bf16_round(weft_log2(weft_bf16_val(a)));
}
inline uint16_t weft_bf16_rsqrt(uint16_t a) {
  return weft_bf16_round(weft_rsqrt(weft_bf16_val(a)));
}
inline uint16_t weft_bf16_sin(uint16_t a) {
  return weft_bf16_round(weft_sin(weft_bf16_val(a)));
}
inline uint16_t weft_bf16_cos(uint16_t a) {
  return weft_bf16_round(weft_cos(weft_bf16_val(a)));
}
inline uint16_t weft_bf16_tanh(uint16_t a) {
  return weft_bf16_round(weft_tanh(weft_bf16_val(a)));
}
// The packed forms are the same answer twice, low half then high.
inline uint32_t weft_f16x2_ex2(uint32_t a) {
  return (uint32_t)weft_f16_ex2((uint16_t)a) |
         ((uint32_t)weft_f16_ex2((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_f16x2_lg2(uint32_t a) {
  return (uint32_t)weft_f16_lg2((uint16_t)a) |
         ((uint32_t)weft_f16_lg2((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_f16x2_rsqrt(uint32_t a) {
  return (uint32_t)weft_f16_rsqrt((uint16_t)a) |
         ((uint32_t)weft_f16_rsqrt((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_f16x2_sin(uint32_t a) {
  return (uint32_t)weft_f16_sin((uint16_t)a) |
         ((uint32_t)weft_f16_sin((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_f16x2_cos(uint32_t a) {
  return (uint32_t)weft_f16_cos((uint16_t)a) |
         ((uint32_t)weft_f16_cos((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_f16x2_tanh(uint32_t a) {
  return (uint32_t)weft_f16_tanh((uint16_t)a) |
         ((uint32_t)weft_f16_tanh((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_bf16x2_ex2(uint32_t a) {
  return (uint32_t)weft_bf16_ex2((uint16_t)a) |
         ((uint32_t)weft_bf16_ex2((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_bf16x2_lg2(uint32_t a) {
  return (uint32_t)weft_bf16_lg2((uint16_t)a) |
         ((uint32_t)weft_bf16_lg2((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_bf16x2_rsqrt(uint32_t a) {
  return (uint32_t)weft_bf16_rsqrt((uint16_t)a) |
         ((uint32_t)weft_bf16_rsqrt((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_bf16x2_sin(uint32_t a) {
  return (uint32_t)weft_bf16_sin((uint16_t)a) |
         ((uint32_t)weft_bf16_sin((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_bf16x2_cos(uint32_t a) {
  return (uint32_t)weft_bf16_cos((uint16_t)a) |
         ((uint32_t)weft_bf16_cos((uint16_t)(a >> 16)) << 16);
}
inline uint32_t weft_bf16x2_tanh(uint32_t a) {
  return (uint32_t)weft_bf16_tanh((uint16_t)a) |
         ((uint32_t)weft_bf16_tanh((uint16_t)(a >> 16)) << 16);
}
template <typename W_PTR>
static inline uint16_t weft_atom_add_narrow(W_PTR w, uint32_t shift,
                                            uint16_t v, int bf) {
  uint32_t old = WEFT_ALOAD(w);
  for (;;) {
    const uint16_t cur = (uint16_t)(old >> shift);
    const uint16_t nv = bf ? weft_bf16_add(cur, v) : weft_f16_add(cur, v);
    const uint32_t want = (old & ~(0xFFFFu << shift)) | ((uint32_t)nv << shift);
    const uint32_t seen = WEFT_CAS(w, old, want);
    if (seen == old) return cur;
    old = seen;
  }
}
// The packed form fills its whole word, so the pair goes round as one value.
template <typename W_PTR>
static inline uint32_t weft_atom_add_pack(W_PTR w, uint32_t v, int bf) {
  uint32_t old = WEFT_ALOAD(w);
  for (;;) {
    const uint32_t want = bf ? weft_bf16x2_add(old, v) : weft_f16x2_add(old, v);
    const uint32_t seen = WEFT_CAS(w, old, want);
    if (seen == old) return old;
    old = seen;
  }
}
// `.v8` siblings.
template <typename W_PTR>
static inline uint16_t weft_atom_minmax_narrow(W_PTR w, uint32_t shift,
                                               uint16_t v, int bf, int maxv) {
  uint32_t old = WEFT_ALOAD(w);
  for (;;) {
    const uint16_t cur = (uint16_t)(old >> shift);
    uint16_t nv;
    if (bf) nv = maxv ? weft_bf16_max(cur, v) : weft_bf16_min(cur, v);
    else nv = maxv ? weft_f16_max(cur, v) : weft_f16_min(cur, v);
    const uint32_t want = (old & ~(0xFFFFu << shift)) | ((uint32_t)nv << shift);
    const uint32_t seen = WEFT_CAS(w, old, want);
    if (seen == old) return cur;
    old = seen;
  }
}
// A packed pair is its whole word, so it goes round as one value.
template <typename W_PTR>
static inline uint32_t weft_atom_minmax_pack(W_PTR w, uint32_t v, int bf,
                                             int maxv) {
  uint32_t old = WEFT_ALOAD(w);
  for (;;) {
    uint32_t want;
    if (bf) want = maxv ? weft_bf16x2_max(old, v) : weft_bf16x2_min(old, v);
    else want = maxv ? weft_f16x2_max(old, v) : weft_f16x2_min(old, v);
    const uint32_t seen = WEFT_CAS(w, old, want);
    if (seen == old) return old;
    old = seen;
  }
}
// matched or not, which is what atom.cas promises.
template <typename W_PTR>
static inline uint16_t weft_atom_cas_b16(W_PTR w, uint32_t shift,
                                         uint16_t cmp, uint16_t v) {
  uint32_t old = WEFT_ALOAD(w);
  for (;;) {
    const uint16_t cur = (uint16_t)(old >> shift);
    if (cur != cmp) return cur;
    const uint32_t want = (old & ~(0xFFFFu << shift)) | ((uint32_t)v << shift);
    const uint32_t seen = WEFT_CAS(w, old, want);
    if (seen == old) return cur;
    old = seen;
  }
}
// compare-exchange loop like the 16-bit forms.
template <typename W_PTR>
static inline uint32_t weft_atom_inc_u32(W_PTR p, uint32_t b) {
  uint32_t old = WEFT_ALOAD(p);
  for (;;) {
    const uint32_t nv = (old >= b) ? 0u : old + 1u;
    const uint32_t seen = WEFT_CAS(p, old, nv);
    if (seen == old) return old;
    old = seen;
  }
}
template <typename W_PTR>
static inline uint32_t weft_atom_dec_u32(W_PTR p, uint32_t b) {
  uint32_t old = WEFT_ALOAD(p);
  for (;;) {
    const uint32_t nv = (old == 0u || old > b) ? b : old - 1u;
    const uint32_t seen = WEFT_CAS(p, old, nv);
    if (seen == old) return old;
    old = seen;
  }
}
// A 16-bit integer atomic.
#define WEFT_A16_ADD 0
#define WEFT_A16_SUB 1
#define WEFT_A16_MINU 2
#define WEFT_A16_MAXU 3
#define WEFT_A16_MINS 4
#define WEFT_A16_MAXS 5
#define WEFT_A16_AND 6
#define WEFT_A16_OR 7
#define WEFT_A16_XOR 8
#define WEFT_A16_EXCH 9
#define WEFT_A16_CAS 10
template <typename W_PTR>
static inline uint16_t weft_atom_int16(W_PTR w, uint32_t shift,
                                       uint16_t v, uint16_t cmp, int op) {
  uint32_t old = WEFT_ALOAD(w);
  for (;;) {
    const uint16_t cur = (uint16_t)(old >> shift);
    uint16_t nv;
    switch (op) {
      case WEFT_A16_ADD: nv = (uint16_t)(cur + v); break;
      case WEFT_A16_SUB: nv = (uint16_t)(cur - v); break;
      case WEFT_A16_MINU: nv = (cur < v) ? cur : v; break;
      case WEFT_A16_MAXU: nv = (cur > v) ? cur : v; break;
      case WEFT_A16_MINS: nv = ((int16_t)cur < (int16_t)v) ? cur : v; break;
      case WEFT_A16_MAXS: nv = ((int16_t)cur > (int16_t)v) ? cur : v; break;
      case WEFT_A16_AND: nv = (uint16_t)(cur & v); break;
      case WEFT_A16_OR: nv = (uint16_t)(cur | v); break;
      case WEFT_A16_XOR: nv = (uint16_t)(cur ^ v); break;
      case WEFT_A16_EXCH: nv = v; break;
      default: if (cur != cmp) return cur; nv = v; break;
    }
    const uint32_t want = (old & ~(0xFFFFu << shift)) | ((uint32_t)nv << shift);
    const uint32_t seen = WEFT_CAS(w, old, want);
    if (seen == old) return cur;
    old = seen;
  }
}
// Bit scans, shared by both columns so the two cannot answer differently.
// `__builtin_clz(0)` is undefined, which is why the zero case is answered first.
static inline uint32_t weft_bfind_u32(uint32_t a) {
  return a ? (uint32_t)(31 - __builtin_clz(a)) : 0xFFFFFFFFu;
}
static inline uint32_t weft_bfind_s32(int32_t a) {
  return weft_bfind_u32(a < 0 ? ~(uint32_t)a : (uint32_t)a);
}
static inline uint32_t weft_fns_b32(uint32_t mask, uint32_t base, uint32_t offset) {
  for (uint32_t p = (base & 31u); p < 32u; p++) {
    if (((mask >> p) & 1u) == 0u) continue;
    if (offset == 0u) return p;
    offset--;
  }
  return 0xFFFFFFFFu;
}
)WEFTMATH";
