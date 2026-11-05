#include <string>

#pragma once

// The Metal side of the translated-kernel runtime.
//
// A CPU module links against runtime/weft_runtime.hpp: buffer addresses, atomics,
// a rendezvous for shuffles. An emitted GPU kernel needs the same names with a
// hardware answer behind them, so this is that surface written in Metal. The
// instruction text is then the same for both targets, which is the whole point:
// one decode, two back ends, and the GPU column becomes a comparison of the same
// generated code rather than of a hand-written twin.
//
// Where the hardware changes the shape of an instruction:
//
//   - a shuffle is one instruction (simd_shuffle) instead of publish-yield-read;
//   - a barrier is a real threadgroup barrier instead of `save pc, return`;
//   - threadgroup memory is a `threadgroup` array instead of a shared arena;
//   - a 64-bit atomic does not exist on this host (tools/atom64_probe.mm), so one
//     is built from the 32-bit atomics there are: a lock over the two halves.
//     What bounds that lock is a measurement, and it is one measurement because
//     the four before it were the harness (tools/lock_probe.mm):
//
//       Everything the case touches is in one buffer, so no argument can be
//       misbound, and the kernel writes back the numbers it read so the host
//       checks them instead of inferring them. The same kernel, the same
//       bindings, one thread per threadgroup, 2,048 lanes of work:
//
//         control   no lock at all                        counter exact
//         lone      the lock taken and dropped, 1 thread   counter exact, lock
//                                                          left at 0
//         locked    32 threads contending on that word     never returns
//
//       The only difference between `lone` and `locked` is that a second thread
//       wants the word, so the lock word is right, the launch is right, and the
//       wait is what does not end. It does not end with a bare
//       compare-exchange spin either, or with a device fence in the critical
//       section: a thread that waits for another thread's release is not
//       scheduled again here.
//
//     That is why the pair has one writer rather than a lock: the warp back end
//     is exact because one thread owns all 32 lanes and every access is an atomic
//     from that one thread, and the per-lane back end refuses a pair it would
//     have to lock and keeps the shape that needs no lock, a reduction, which is
//     two 32-bit atomics.
//
// Everything here is `static inline` in one translation unit, so the compiler
// inlines the helpers into the kernel and constant-folds the mode switches.
static const char* kWWeftMslHead = R"WEFMSL(
#include <metal_stdlib>
using namespace metal;

// The helpers the emitted body names. Same names, same answers: the bit-level
// ones are pure integer work, and memcpy becomes as_type because there is no
// memcpy in a shader.
// `as_type` rather than a union: the language has the bitcast, so there is one
// spelling of a reinterpret rather than two.
template <class T, class U>
static inline T weft_bitcast(U u) {
  return as_type<T>(u);
}
static inline float weft_f32(uint32_t b) { return as_type<float>(b); }
static inline uint32_t weft_bits(float f) { return as_type<uint32_t>(f); }
static inline uint32_t weft_bitrev32(uint32_t v) {
  v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
  v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
  v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
  v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
  return (v >> 16) | (v << 16);
}
static inline uint16_t weft_h16_bits(float x) { return as_type<uint16_t>((_Float16)x); }

// A double on this target is its bit pattern: the type itself does not exist here,
// so the constant spelling the emitter uses for one hands the bits back unchanged.
static inline uint64_t weft_f64(uint64_t b) { return b; }

//
// Apple GPUs have no fp64 and Metal rejects the type outright, so a kernel carrying a
// `double` cannot be compiled at all, that is the language, not a missing lowering.
// Software is how every GPU without fp64 carries one, and this is that: a double is
// its bits in a `uint64_t` (which is the type the emitter gives it), and every
// operation here is exact IEEE-754 binary64, round-to-nearest-even, subnormals
// included. Exactness is the requirement rather than a nicety: the CPU column runs
// the same kernel with the host's own doubles, and the two are compared bit for bit.
//
// One working form is shared by everything below:
//
//     value = sig * 2^(e - 55)
//
// `e` is the unbiased exponent and `sig` is an integer whose bit 55 is the leading
// bit, the sign being carried separately. The low three bits of `sig` are guard,
// round and sticky, so one rounding at the end is enough: `round_pack` does it, and
// each operation only has to produce that form exactly.
)WEFMSL";

#include "weft_f64.hpp"

static const char* kWWeftMslTail = R"WEFMSL(
static inline float weft_h16(uint32_t b, int i) {
  return (float)as_type<_Float16>((uint16_t)(b >> (16 * i)));
}
static inline uint32_t weft_f16x2(float hi, float lo) {
  return (uint32_t)weft_h16_bits(lo) | ((uint32_t)weft_h16_bits(hi) << 16);
}
static inline uint16_t weft_bf16_bits(float x) {
  uint32_t u = as_type<uint32_t>(x);
  if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline uint32_t weft_bf16x2(float hi, float lo) {
  return (uint32_t)weft_bf16_bits(lo) | ((uint32_t)weft_bf16_bits(hi) << 16);
}
static inline float weft_bf16(uint32_t b, int i) {
  return as_type<float>((uint32_t)(uint16_t)(b >> (16 * i)) << 16);
}
// `.rz` truncates toward zero, and the 16-bit conversion these engines have rounds
// to nearest: so the truncation is the rounded value walked back one step whenever
// it landed away from zero. A finite value past the format's range truncates to
// that format's largest finite value rather than to an infinity, infinities and
// NaNs have no significand to drop and pass through.
static inline uint16_t weft_h16_bits_rz(float x) {
  uint16_t h = weft_h16_bits(x);
  if ((h & 0x7C00u) == 0x7C00u) {
    const uint32_t ux = weft_bitcast<uint32_t>(x);
    if ((ux & 0x7F800000u) == 0x7F800000u) return h;   // inf or nan
    return (uint16_t)((h & 0x8000u) | 0x7BFFu);        // a finite value too large
  }
  const float back = weft_h16((uint32_t)h, 0);
  const float ax = x < 0.0f ? -x : x, ab = back < 0.0f ? -back : back;
  if (ab > ax && (h & 0x7FFFu) != 0u) h = (uint16_t)((h & 0x8000u) | ((h & 0x7FFFu) - 1u));
  return h;
}
static inline uint16_t weft_bf16_bits_rz(float x) {
  // A bf16 has a float's exponent range, so dropping the low sixteen bits cannot
  // overflow: it only drops their significand tail, and inf and NaN have none.
  return (uint16_t)(weft_bitcast<uint32_t>(x) >> 16);
}
// e4m3 / e5m2 conversion is integer work plus an exact power-of-two scale, so
// float arithmetic here reproduces the CPU's double arithmetic bit for bit:
// dividing by 512 and multiplying by 65536 are exact in both.
static inline uint8_t weft_e4m3_bits(float x) {
  const uint32_t u = as_type<uint32_t>(x);
  const uint8_t sign = (u & 0x80000000u) ? 0x80u : 0u;
  if ((u & 0x7FFFFFFFu) == 0u) return sign;
  if (!isfinite(x)) return (uint8_t)(sign | 0x7Fu);
  const int E = (int)((u >> 23) & 0xFFu) - 127 + 7;
  if (E >= 16) return (uint8_t)(sign | 0x7Eu);
  if (E <= 0) {
    const float q = rint(fabs(x) * 512.0f);
    if (q >= 8.0f) return (uint8_t)(sign | 0x08u);
    return (uint8_t)(sign | (uint8_t)q);
  }
  const uint32_t mant = u & 0x7FFFFFu;
  uint32_t r = mant >> 20;
  const uint32_t rem = mant & 0xFFFFFu;
  if (rem > 0x80000u || (rem == 0x80000u && (r & 1u))) r++;
  const uint32_t out = ((uint32_t)E << 3) + r;
  return (uint8_t)(sign | (uint8_t)(out >= 0x7Fu ? 0x7Eu : out));
}
static inline uint32_t weft_e4m3x2(float hi, float lo) {
  return (uint32_t)weft_e4m3_bits(lo) | ((uint32_t)weft_e4m3_bits(hi) << 8);
}
static inline float weft_e4m3(uint32_t b, int i) {
  const uint32_t v = (b >> (8 * i)) & 0xFFu;
  const uint32_t mag = v & 7u, E = (v >> 3) & 0xFu;
  // The scale is an exponent field rather than a library call: E is 1..15 and the
  // field for 2^(E-10) is E+117, so the multiply is exact. `ldexp` was the one
  // place a decode leaned on the platform's own rounding, and it disagreed with
  // the host's, this is the same arithmetic written out.
  const float d = (E == 0) ? (float)mag / 512.0f
                           : (float)(8u + mag) * as_type<float>((E + 117u) << 23);
  return (float)((v & 0x80u) ? -d : d);
}
static inline uint8_t weft_e5m2_bits(float x) {
  const uint32_t u = as_type<uint32_t>(x);
  const uint8_t sign = (u & 0x80000000u) ? 0x80u : 0u;
  if ((u & 0x7FFFFFFFu) == 0u) return sign;
  if (!isfinite(x)) return (uint8_t)(sign | 0x7Fu);
  const int E = (int)((u >> 23) & 0xFFu) - 127 + 15;
  if (E >= 31) return (uint8_t)(sign | 0x7Bu);
  if (E <= 0) {
    const float q = rint(fabs(x) * 65536.0f);
    if (q >= 4.0f) return (uint8_t)(sign | 0x04u);
    return (uint8_t)(sign | (uint8_t)q);
  }
  const uint32_t mant = u & 0x7FFFFFu;
  uint32_t r = mant >> 21;
  const uint32_t rem = mant & 0x1FFFFFu;
  if (rem > 0x100000u || (rem == 0x100000u && (r & 1u))) r++;
  const uint32_t out = ((uint32_t)E << 2) + r;
  return (uint8_t)(sign | (uint8_t)(out >= 0x7Cu ? 0x7Bu : out));
}
static inline uint32_t weft_e5m2x2(float hi, float lo) {
  return (uint32_t)weft_e5m2_bits(lo) | ((uint32_t)weft_e5m2_bits(hi) << 8);
}
// e2m1, e2m3 and e3m2: the MX formats, what a Blackwell fp4/fp6 convert writes.
// Each is a sign, `eb` exponent bits and `mb` mantissa bits biased 2^(eb-1)-1, and
// none of the three has an infinity or a NaN, so the `.satfinite` the ISA makes
// mandatory on these converts is the whole range rule: a NaN answers the positive
// MAX_NORM, and an infinity, whose magnitude is past MAX_NORM, answers MAX_NORM
// keeping its sign.
static inline uint32_t weft_mx_bits(float x, int eb, int mb) {
  const uint32_t maxbits = (uint32_t)((((1u << eb) - 1u) << mb) | ((1u << mb) - 1u));
  const uint32_t u = as_type<uint32_t>(x);
  const uint32_t sign = u >> 31;
  if (!isfinite(x)) return (x != x) ? maxbits : ((sign << (eb + mb)) | maxbits);
  const uint32_t mag = u & 0x7FFFFFFFu;
  if (mag == 0u) return sign << (eb + mb);
  const int bias = (1 << (eb - 1)) - 1;
  const int E = (int)((mag >> 23) & 0xFFu) - 127 + bias;
  if (E >= (1 << eb) - 1) return (sign << (eb + mb)) | maxbits;
  const uint32_t mant = mag & 0x7FFFFFu;
  uint32_t out;
  if (E <= 0) {
    // A subnormal sits on a grid of 2^(1-bias-mb). Scaling the magnitude onto it
    // is exact, because the scale is a power of two, so the rounding is the same
    // `rint` the e4m3 path uses and the two engines agree bit for bit.
    out = (uint32_t)rint(fabs(x) * (float)(1u << (mb + bias - 1)));
  } else {
    const int sh = 23 - mb;
    uint32_t r = mant >> sh;
    const uint32_t rem = mant & ((1u << sh) - 1u);
    if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (r & 1u))) r++;
    out = ((uint32_t)E << mb) + r;
  }
  // A carry out of the mantissa is a carry into the exponent, and a rounding that
  // reaches 2^mb off the subnormal grid is the smallest normal: both are already
  // the right integer, so only the top of the range needs clamping.
  if (out >= maxbits) out = maxbits;
  return (sign << (eb + mb)) | out;
}
// The packed layouts are the ISA's: e2m1 puts the first source in d[7:4] and the
// second in d[3:0]; the f6 formats put each six-bit value in the low bits of a
// byte, d[13:8] then d[5:0], with the top two bits of each byte zero, which the
// mask leaves zero rather than shifting the value up.
static inline uint8_t weft_e2m1x2(float hi, float lo) {
  return (uint8_t)((weft_mx_bits(lo, 2, 1) & 0xFu) | ((weft_mx_bits(hi, 2, 1) & 0xFu) << 4));
}
static inline uint16_t weft_e2m3x2(float hi, float lo) {
  return (uint16_t)((weft_mx_bits(lo, 2, 3) & 0x3Fu) | ((weft_mx_bits(hi, 2, 3) & 0x3Fu) << 8));
}
static inline uint16_t weft_e3m2x2(float hi, float lo) {
  return (uint16_t)((weft_mx_bits(lo, 3, 2) & 0x3Fu) | ((weft_mx_bits(hi, 3, 2) & 0x3Fu) << 8));
}
static inline uint16_t weft_e2m1x4(float a, float b, float e, float f) {
  return (uint16_t)(((weft_mx_bits(a, 2, 1) & 0xFu) << 12) |
                    ((weft_mx_bits(b, 2, 1) & 0xFu) << 8) |
                    ((weft_mx_bits(e, 2, 1) & 0xFu) << 4) | (weft_mx_bits(f, 2, 1) & 0xFu));
}
static inline uint32_t weft_e2m3x4(float a, float b, float e, float f) {
  return ((weft_mx_bits(a, 2, 3) & 0x3Fu) << 24) | ((weft_mx_bits(b, 2, 3) & 0x3Fu) << 16) |
         ((weft_mx_bits(e, 2, 3) & 0x3Fu) << 8) | (weft_mx_bits(f, 2, 3) & 0x3Fu);
}
static inline uint32_t weft_e3m2x4(float a, float b, float e, float f) {
  return ((weft_mx_bits(a, 3, 2) & 0x3Fu) << 24) | ((weft_mx_bits(b, 3, 2) & 0x3Fu) << 16) |
         ((weft_mx_bits(e, 3, 2) & 0x3Fu) << 8) | (weft_mx_bits(f, 3, 2) & 0x3Fu);
}
// `.rs` is stochastic rounding: the value is truncated onto the destination's grid
// and then stepped one ulp away from zero when adding the random bits to the
// discarded mantissa carries out. A float leaves thirteen bits below a half and
// sixteen below a bf16, and the ISA's figure gives exactly that many random bits per
// value, so at those widths the test is the plain integer carry the ISA describes.
// `dropped << w` against `(2^w - r) << shift` is that carry written as a comparison:
// it answers true for the same `dropped` values (a tie carries, an exact grid value
// does not), and it stays exact when a subnormal widens the dropped field past the
// width of the random bits.
static inline int weft_rs_carry(uint32_t mag, int e, int mb, int emin, uint32_t r, int w) {
  int shift = 23 - mb + (e < emin ? emin - e : 0);
  if (shift <= 0) return 0;
  if (shift > 23) shift = 23;
  const uint64_t dropped = (uint64_t)(mag & ((1u << shift) - 1u));
  const uint64_t needbits = (uint64_t)((1u << w) - (r & ((1u << w) - 1u)));
  return (dropped << w) >= (needbits << shift) ? 1 : 0;
}
static inline uint16_t weft_h16_bits_rs(float x, uint32_t rbits, int satfinite) {
  const uint32_t u = as_type<uint32_t>(x);
  const uint16_t sign = (uint16_t)((u >> 31) ? 0x8000u : 0u);
  const uint32_t mag = u & 0x7FFFFFFFu;
  const int e = (int)((mag >> 23) & 0xFFu) - 127;
  // A NaN keeps its own encoding and an infinity has no significand to round;
  // `.satfinite` is what sends a value past the half range to the largest finite
  // half instead of to an infinity.
  if ((mag & 0x7F800000u) == 0x7F800000u) {
    if (mag > 0x7F800000u) return (uint16_t)(sign | 0x7E00u);
    return (uint16_t)(sign | (satfinite ? 0x7BFFu : 0x7C00u));
  }
  if (e >= 16) return (uint16_t)(sign | (satfinite ? 0x7BFFu : 0x7C00u));
  uint16_t t = weft_h16_bits_rz(x);
  if (weft_rs_carry(mag, e, 10, -14, rbits, 13)) {
    const uint32_t body = (uint32_t)(t & 0x7FFFu) + 1u;
    if (body >= 0x7C00u) return (uint16_t)(sign | (satfinite ? 0x7BFFu : 0x7C00u));
    t = (uint16_t)(sign | body);
  }
  return t;
}
static inline uint16_t weft_bf16_bits_rs(float x, uint32_t rbits, int satfinite) {
  const uint32_t u = as_type<uint32_t>(x);
  const uint16_t sign = (uint16_t)((u >> 31) ? 0x8000u : 0u);
  const uint32_t mag = u & 0x7FFFFFFFu;
  const int e = (int)((mag >> 23) & 0xFFu) - 127;
  // A bf16 has a float's exponent range, so only a NaN or an infinity reaches here:
  // the NaN keeps its encoding, and `.satfinite` clamps the infinity.
  if ((mag & 0x7F800000u) == 0x7F800000u) {
    if (mag > 0x7F800000u) return (uint16_t)(sign | 0x7FC0u);
    return (uint16_t)(sign | (satfinite ? 0x7F7Fu : 0x7F80u));
  }
  uint16_t t = (uint16_t)(mag >> 16);
  if (weft_rs_carry(mag, e, 7, -126, rbits, 16)) {
    const uint32_t body = (uint32_t)t + 1u;
    if (body >= 0x7F80u) return (uint16_t)(sign | (satfinite ? 0x7F7Fu : 0x7F80u));
    t = (uint16_t)body;
  }
  return (uint16_t)(sign | t);
}
static inline float weft_e5m2(uint32_t b, int i) {
  const uint32_t v = (b >> (8 * i)) & 0xFFu;
  const uint32_t mag = v & 3u, E = (v >> 2) & 0x1Fu;
  if (E == 31u) return (mag == 0u) ? ((v & 0x80u) ? -INFINITY : INFINITY) : NAN;
  const float d = (E == 0) ? (float)mag / 65536.0f
                           : (float)(4u + mag) * as_type<float>((E + 110u) << 23);
  return (float)((v & 0x80u) ? -d : d);
}
// The packed narrow formats as one code, so an instruction that names two of them is
// a pair of small tables rather than a case for every combination: 0 e2m1, 1 e2m3,
// 2 e3m2, 3 e4m3, 4 e5m2, 5 ue8m0, 6 e2m1p (one nibble a byte).
static inline float weft_mx_decode(uint32_t v, int eb, int mb) {
  const int bias = (1 << (eb - 1)) - 1;
  const uint32_t mant = v & ((1u << mb) - 1u);
  const uint32_t E = (v >> mb) & ((1u << eb) - 1u);
  // 2^(E-bias-mb) as an exponent field, and E = 0 reads the format's subnormal grid
  // instead of an implicit 1, the same arithmetic the e4m3 decode above is written
  // with, so both engines land on the same float.
  const int ex = (E == 0u ? 1 : (int)E) - bias - mb;
  const float sc = as_type<float>((uint32_t)(ex + 127) << 23);
  const float d = (float)((E == 0u ? 0u : (1u << mb)) + mant) * sc;
  return (v >> (eb + mb)) ? -d : d;
}
static inline float weft_e2m1(uint32_t b, int i) {
  return weft_mx_decode((b >> (4 * i)) & 0xFu, 2, 1);
}
static inline float weft_e2m3(uint32_t b, int i) {
  return weft_mx_decode((b >> (8 * i)) & 0x3Fu, 2, 3);
}
static inline float weft_e3m2(uint32_t b, int i) {
  return weft_mx_decode((b >> (8 * i)) & 0x3Fu, 3, 2);
}
// One lane of a packed four, by code: an eight-bit field, a six-bit one at the low end
// of each byte, a four-bit one at the low end of each half-byte, or one nibble a byte
// when the format is the padded fp4.
static inline float weft_mx_get(uint32_t w, int i, int code) {
  if (code == 0) return weft_mx_decode((w >> (4 * i)) & 0xFu, 2, 1);
  if (code == 1) return weft_mx_decode((w >> (8 * i)) & 0x3Fu, 2, 3);
  if (code == 2) return weft_mx_decode((w >> (8 * i)) & 0x3Fu, 3, 2);
  if (code == 3) return weft_e4m3(w, i);
  if (code == 4) return weft_e5m2(w, i);
  if (code == 6) return weft_mx_decode((w >> (8 * i)) & 0xFu, 2, 1);
  const uint32_t v = (w >> (8 * i)) & 0xFFu;
  if (v == 0xFFu) return NAN;
  // ue8m0 is a bare exponent, 2^(v-127), and v = 0 is the format's smallest value
  // rather than zero: a subnormal in f32, and exact.
  const uint32_t sb = v == 0u ? 0x00400000u : (v << 23);
  return as_type<float>(sb);
}
// The two destinations the narrow arithmetic allows. `.satfinite` sends a value past
// MAX_NORM to MAX_NORM; without it e4m3 answers a NaN, which is what the ISA says and
// what the saturating encoder alone would get wrong.
static inline uint32_t weft_f8_bits(float x, int code, int satfinite) {
  if (code == 3) {
    if (!satfinite && isfinite(x) && fabs(x) > 448.0f) return 0x7Fu;
    return weft_e4m3_bits(x);
  }
  return weft_e5m2_bits(x);
}
// add, sub, mul and fma over four packed narrow values. The ISA computes each lane in
// infinite precision and rounds once to the destination; every one of these formats
// decodes to an exactly representable float and a product or a sum of two of them is
// exact in f32, so the one rounding here is the one the instruction asks for.
static inline uint32_t weft_mx4_op(uint32_t a, uint32_t b, uint32_t c, int ac, int bc,
                                   int cc, int dc, int op, int sat) {
  uint32_t out = 0;
  for (int i = 0; i < 4; i++) {
    const float fa = weft_mx_get(a, i, ac);
    float r;
    if (op == 0) r = fa + weft_mx_get(c, i, cc);
    else if (op == 1) r = fa - weft_mx_get(c, i, cc);
    else if (op == 2) r = fa * weft_mx_get(b, i, bc);
    else r = fa * weft_mx_get(b, i, bc) + weft_mx_get(c, i, cc);
    out |= (uint32_t)(weft_f8_bits(r, dc, sat) & 0xFFu) << (8 * i);
  }
  return out;
}
// `.rz` on a narrow destination: the nearest code, walked one grid step closer to
// zero when that answer landed away from it. The codes are monotone in magnitude and
// the nearest answer is within half a step, so one step is the most this can need,
// and a finite value past the range has already saturated to MAX_NORM, which is
// what truncating it gives too, so the walk only ever happens below saturation.
static inline uint8_t weft_e4m3_bits_rz(float x) {
  const uint8_t c = weft_e4m3_bits(x);
  uint32_t m = c & 0x7Fu;
  if (m != 0u && fabs(weft_e4m3((uint32_t)c, 0)) > fabs(x)) m--;
  return (uint8_t)((c & 0x80u) | m);
}
static inline uint8_t weft_e5m2_bits_rz(float x) {
  const uint8_t c = weft_e5m2_bits(x);
  uint32_t m = c & 0x7Fu;
  if (m != 0u && fabs(weft_e5m2((uint32_t)c, 0)) > fabs(x)) m--;
  return (uint8_t)((c & 0x80u) | m);
}
static inline uint32_t weft_mx_bits_rz(float x, int eb, int mb) {
  const uint32_t mask = (1u << (eb + mb)) - 1u;
  const uint32_t c = weft_mx_bits(x, eb, mb);
  uint32_t m = c & mask;
  if (m != 0u && fabs(weft_mx_decode(m, eb, mb)) > fabs(x)) m--;
  return (c & ~mask) | m;
}
static inline uint32_t weft_e2m1_bits(float x) { return weft_mx_bits(x, 2, 1); }
static inline uint32_t weft_e2m3_bits(float x) { return weft_mx_bits(x, 2, 3); }
static inline uint32_t weft_e3m2_bits(float x) { return weft_mx_bits(x, 3, 2); }
static inline uint32_t weft_e2m1_rz(float x) { return weft_mx_bits_rz(x, 2, 1); }
static inline uint32_t weft_e2m3_rz(float x) { return weft_mx_bits_rz(x, 2, 3); }
static inline uint32_t weft_e3m2_rz(float x) { return weft_mx_bits_rz(x, 3, 2); }
// Two narrow values into one word: the first in the high field, four bits up for the
// nibble-wide fp4 pair and a byte up for the rest.
static inline uint32_t weft_pack2(uint32_t hi, uint32_t lo, int bits) {
  return lo | (hi << bits);
}
// `.scaled::n{1,2}::ue8m0`: a ue8m0 factor is the power of two 2^(s-127), which a
// down conversion divides by and an up conversion multiplies by. Writing the down
// direction as a multiply by 2^(127-s) is the same number, and exact because it is a
// power of two. s = 0x7f is 1.0, the value the operand defaults to. The ISA flushes a
// subnormal input to a zero of its own sign before a scaled down conversion.
static inline float weft_ue8m0(uint32_t s) {
  return as_type<float>(s == 0u ? 0x00400000u : (s << 23));
}
static inline float weft_scale_up(float x, uint32_t s) { return x * weft_ue8m0(s); }
static inline float weft_scale_down(float x, uint32_t s) {
  uint32_t u = as_type<uint32_t>(x);
  if ((u & 0x7F800000u) == 0u && (u & 0x7FFFFFu) != 0u) u &= 0x80000000u;
  x = as_type<float>(u);
  const uint32_t sb = s >= 254u ? (s == 254u ? 0x00400000u : 0x00200000u)
                                : (254u - s) << 23;
  return x * as_type<float>(sb);
}
// `.rs` on a packed four: the truncated code, one step further from zero when the
// integer addition of the random bits to the discarded mantissa carries out. Eight
// random bits per operand, which is what the ISA's figure gives each of the four.
static inline uint32_t weft_rs_step(uint32_t code, uint32_t magbits, uint32_t maxmag,
                                    float x, int mb, int emin, uint32_t r) {
  const uint32_t mag = as_type<uint32_t>(x) & 0x7FFFFFFFu;
  if (mag == 0u || (mag & 0x7F800000u) == 0x7F800000u) return code;
  if (!weft_rs_carry(mag, (int)((mag >> 23) & 0xFFu) - 127, mb, emin, r, 8)) return code;
  const uint32_t mask = (1u << magbits) - 1u;
  const uint32_t m = (code & mask) + 1u;
  return (code & ~mask) | (m > maxmag ? maxmag : m);
}
// The rbits word splits by operand, not by value: the byte-wide destinations take a
// and b from the upper half of the word and e and f from the lower one, the first of
// each pair in the upper byte; the nibble-wide fp4 packs four to a half-word, so the
// same rule puts b's random bits in the low byte of the upper half and e's in the
// high byte of the lower one.
static inline uint32_t weft_mx4(float a, float b, float e, float f, uint32_t r,
                                int eb, int mb, int rs) {
  const int nib = (eb + mb) == 3;
  const float v[4] = {a, b, e, f};
  const int bit[4] = {24, nib ? 8 : 16, nib ? 16 : 8, 0};
  const int put[4] = {nib ? 12 : 24, nib ? 8 : 16, nib ? 4 : 8, 0};
  const uint32_t magbits = (uint32_t)(eb + mb);
  const uint32_t maxmag = (1u << magbits) - 1u;
  const int emin = 2 - (1 << (eb - 1));
  uint32_t w = 0;
  for (int i = 0; i < 4; i++) {
    const uint32_t c = rs ? weft_rs_step(weft_mx_bits_rz(v[i], eb, mb), magbits, maxmag,
                                         v[i], mb, emin, (r >> bit[i]) & 0xFFu)
                          : weft_mx_bits(v[i], eb, mb);
    w |= c << put[i];
  }
  return w;
}
static inline uint32_t weft_f8x4(float a, float b, float e, float f, uint32_t r,
                                 int code, int rs) {
  const float v[4] = {a, b, e, f};
  uint32_t w = 0;
  for (int i = 0; i < 4; i++) {
    const uint32_t rb = (r >> (24 - 8 * i)) & 0xFFu;
    const uint32_t c = code == 3
        ? (rs ? weft_rs_step(weft_e4m3_bits_rz(v[i]), 7u, 0x7Eu, v[i], 3, -6, rb)
              : weft_e4m3_bits(v[i]))
        : (rs ? weft_rs_step(weft_e5m2_bits_rz(v[i]), 7u, 0x7Bu, v[i], 2, -14, rb)
              : weft_e5m2_bits(v[i]));
    w |= c << (24 - 8 * i);
  }
  return w;
}
// The same bytes with an integer accumulator: exact in s32, where a float stops
// being the same number past 2^24.
static inline int32_t weft_s8_i(uint32_t b, int i) {
  return (int32_t)((int8_t)(uint8_t)((b >> (8 * i)) & 0xFFu));
}
static inline int32_t weft_u8_i(uint32_t b, int i) {
  return (int32_t)((b >> (8 * i)) & 0xFFu);
}
// The 8-bit integer operands: one byte, sign-extended or not. Both are exact in
// f32, so an integer tile's products and the sum of 32 of them are exact.
static inline float weft_s8(uint32_t b, int i) {
  return (float)((int)(int8_t)((b >> (8 * i)) & 0xFFu));
}
static inline float weft_u8(uint32_t b, int i) {
  return (float)((b >> (8 * i)) & 0xFFu);
}
// TF32 is a sign, an 8-bit exponent and ten mantissa bits: a float32 with the low
// thirteen dropped, and a multiply sees the rounded value rather than the input.
// The three modes differ only in those thirteen bits, `rna` carries a tie away
// from zero, `rn` to even, `rz` throws them away, and a carry out of the
// significand belongs to the exponent, which is why the add comes before the mask.
// `.relu` maps a negative value to zero first, and `.satfinite` clamps an infinite
// result to the largest finite tf32, with NaN becoming zero. It is written over
// the bits because that is what the translator emits: a `.b32` register round-trips
// through it without becoming a float at all.
static inline uint32_t weft_tf32_bits(uint32_t u, int mode, int satfinite, int relu) {
  if (relu && (u & 0x80000000u) != 0u && (u & 0x7FFFFFFFu) != 0u) u = 0u;
  const uint32_t sign = u & 0x80000000u;
  if ((u & 0x7F800000u) == 0x7F800000u) {
    if (!satfinite) return u;
    return (u & 0x7FFFFFFFu) > 0x7F800000u ? 0u : (sign | 0x7F7FE000u);
  }
  // Nearest-even is `half - 1` plus the kept bit, not `half`: adding 0x1000 by
  // itself carries a tie upward whatever the kept bit is, which is round-half-up.
  // This helper answered that until the tf32 fixture separated the two on a tie.
  if (mode == 1) u += 0x1000u;
  else if (mode == 0) u += 0x0FFFu + ((u >> 13) & 1u);
  u &= 0xFFFFE000u;
  if (satfinite && (u & 0x7F800000u) == 0x7F800000u) u = sign | 0x7F7FE000u;
  return u;
}
static inline float weft_tf32_r(float x, int mode) {
  return as_type<float>(weft_tf32_bits(as_type<uint32_t>(x), mode, 0, 0));
}
static inline float weft_tf32(float x) { return weft_tf32_r(x, 0); }
static inline uint32_t weft_shf_l(uint32_t a, uint32_t b, uint32_t c, int clamp) {
  if (clamp) { if (c >= 32u) return a; } else c &= 31u;
  return c ? ((b << c) | (a >> (32u - c))) : b;
}
static inline uint32_t weft_shf_r(uint32_t a, uint32_t b, uint32_t c, int clamp) {
  if (clamp) { if (c >= 32u) return b; } else c &= 31u;
  return c ? ((a >> c) | (b << (32u - c))) : a;
}
static inline uint32_t weft_lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < 8u; i++) {
    const uint32_t m = ((i & 4u) ? a : ~a) & ((i & 2u) ? b : ~b) & ((i & 1u) ? c : ~c);
    if (lut & (1u << i)) out |= m;
  }
  return out;
}
// prmt.b32: the four bytes of the result are picked out of the eight bytes of
// (a:b), and the mode says how. The generic form (mode -1) gives a nibble a byte
// and its top bit asks for the *sign* of that byte instead of the byte; the six
// named modes read one 2-bit selector `k` and are the ISA's own table:
//   f4e  d.bj = src[(k + j) & 7]     b4e  d.bj = src[(k - j) & 7]
//   rc8  d.bj = src[k]               rc16 d.bj = src[(k & 1) * 2 + (j & 1)]
//   ecl  d.bj = src[max(k, j)]       ecr  d.bj = src[min(k, j)]
static inline uint32_t weft_prmt(uint32_t a, uint32_t b, uint32_t c, int mode) {
  const uint64_t src = (uint64_t)a | ((uint64_t)b << 32);
  uint32_t out = 0;
  if (mode < 0) {
    for (int i = 0; i < 4; i++) {
      const uint32_t sel = (c >> (4 * i)) & 0xFu;
      const uint32_t byte = (uint32_t)((src >> ((sel & 7u) * 8u)) & 0xFFu);
      out |= ((sel & 8u) ? ((byte & 0x80u) ? 0xFFu : 0u) : byte) << (8 * i);
    }
    return out;
  }
  const uint32_t k = c & 3u;
  for (uint32_t j = 0; j < 4u; j++) {
    uint32_t idx;
    if (mode == 0) idx = (k + j) & 7u;
    else if (mode == 1) idx = (k - j) & 7u;
    else if (mode == 2) idx = k;
    else if (mode == 3) idx = k > j ? k : j;
    else if (mode == 4) idx = k < j ? k : j;
    else idx = (k & 1u) * 2u + (j & 1u);
    out |= (uint32_t)((src >> (idx * 8u)) & 0xFFu) << (8u * j);
  }
  return out;
}
// mul.hi.u64 / mul.hi.s64: the high 64 bits of a 64x64 product, built from four
// 32x32 multiplies. The scalar `__int128` in the language would express it in one
// line, but this host's back end gives up on a 128-bit multiply in a kernel of
// real size: the library compiles and the *pipeline* then fails with
// XPC_ERROR_CONNECTION_INTERRUPTED, which is what kept kernels/cuda/bits.ptx off
// the GPU column. Four products and a carry chain cost a few instructions and
// compile everywhere.
static inline uint64_t weft_mulhi_u64(uint64_t a, uint64_t b) {
  const uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32, b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
  const uint64_t p0 = a0 * b0, p1 = a0 * b1, p2 = a1 * b0, p3 = a1 * b1;
  const uint64_t cy = ((p0 >> 32) + (p1 & 0xFFFFFFFFu) + (p2 & 0xFFFFFFFFu)) >> 32;
  return p3 + (p1 >> 32) + (p2 >> 32) + cy;
}
// The signed word is the unsigned product with the two sign extensions taken
// back out: a negative `a` contributed `b` too many times, and vice versa.
static inline int64_t weft_mulhi_s64(int64_t a, int64_t b) {
  uint64_t h = weft_mulhi_u64((uint64_t)a, (uint64_t)b);
  if (a < 0) h -= (uint64_t)b;
  if (b < 0) h -= (uint64_t)a;
  return (int64_t)h;
}

// The packed vector types the SPIR-V front end uses: a <3 x float> in memory is
// 12 bytes, and the state struct has to agree or a load reads the wrong data.
struct weft_f2 { float x, y; };
struct weft_f3 { float x, y, z; };
struct weft_f4 { float x, y, z, w; };
struct weft_i2 { int32_t x, y; };
struct weft_i3 { int32_t x, y, z; };
struct weft_i4 { int32_t x, y, z, w; };
struct weft_u2 { uint32_t x, y; };
struct weft_u3 { uint32_t x, y, z; };
struct weft_u4 { uint32_t x, y, z, w; };
struct weft_l2 { uint64_t x, y; };
struct weft_l3 { uint64_t x, y, z; };
struct weft_l4 { uint64_t x, y, z, w; };
struct weft_ul2 { uint64_t x, y; };
struct weft_ul3 { uint64_t x, y, z; };
struct weft_ul4 { uint64_t x, y, z, w; };
// A vector of doubles: the CPU's struct field for field and byte for byte, with
// the software double's bits as components.
struct weft_d2 { uint64_t x, y; };
struct weft_d3 { uint64_t x, y, z; };
struct weft_d4 { uint64_t x, y, z, w; };

// Matrices, column-major: `m[column][row]`, the way SPIR-V, Metal and every GPU lay
// them out, so a matrix times a vector is a linear combination of its columns. One
// template covers every shape SPIR-V allows, two to four columns of two to four
// rows, because one shader multiplies a 4x4 by a 4-vector and a 3x4 skinning matrix
// by a 3-vector in the same file, and the square-only set refused the second. Every
// accumulation is an explicit fused multiply-add in a fixed order, so the two engines
// run the same instruction sequence and agree to the bit. Everything here takes its
// operands by value: Metal wants an address space on every pointer and every
// reference, and this one text is compiled by both engines.
template <int R, int C>
struct weft_mat { float m[C][R]; };
using weft_f2x2 = weft_mat<2, 2>;
using weft_f2x3 = weft_mat<2, 3>;
using weft_f2x4 = weft_mat<2, 4>;
using weft_f3x2 = weft_mat<3, 2>;
using weft_f3x3 = weft_mat<3, 3>;
using weft_f3x4 = weft_mat<3, 4>;
using weft_f4x2 = weft_mat<4, 2>;
using weft_f4x3 = weft_mat<4, 3>;
using weft_f4x4 = weft_mat<4, 4>;

// The components of a row on their way into a vector, as a value rather than a
// pointer so no address space has to be named.
struct weft_n { float v[4]; };

// The bit a ballot says is there: the lowest and the highest one that is set.
// A loop rather than a builtin, `ctz` is spelled differently on the two targets and
// this runs once per ballot, not once per lane.
static inline uint32_t weft_lsb(uint32_t v) {
  for (uint32_t i = 0; i < 32u; i++)
    if (v & (1u << i)) return i;
  return 0xFFFFFFFFu;
}
static inline uint32_t weft_msb(uint32_t v) {
  for (int i = 31; i >= 0; i--)
    if (v & (1u << (uint32_t)i)) return (uint32_t)i;
  return 0xFFFFFFFFu;
}

// The width of a vector type, so a matrix can name the vector it multiplies and the
// loops below are written once for all nine shapes instead of once per shape.
template <int N> struct weft_vt;
template <> struct weft_vt<2> {
  using type = weft_f2;
  static type mk(weft_n t) { return weft_f2{t.v[0], t.v[1]}; }
};
template <> struct weft_vt<3> {
  using type = weft_f3;
  static type mk(weft_n t) { return weft_f3{t.v[0], t.v[1], t.v[2]}; }
};
template <> struct weft_vt<4> {
  using type = weft_f4;
  static type mk(weft_n t) { return weft_f4{t.v[0], t.v[1], t.v[2], t.v[3]}; }
};
template <class V> struct weft_vlen;
template <> struct weft_vlen<weft_f2> { enum { N = 2 }; };
template <> struct weft_vlen<weft_f3> { enum { N = 3 }; };
template <> struct weft_vlen<weft_f4> { enum { N = 4 }; };

static inline float weft_comp(weft_f2 v, int k) { return k == 0 ? v.x : v.y; }
static inline float weft_comp(weft_f3 v, int k) {
  return k == 0 ? v.x : (k == 1 ? v.y : v.z);
}
static inline float weft_comp(weft_f4 v, int k) {
  return k == 0 ? v.x : (k == 1 ? v.y : (k == 2 ? v.z : v.w));
}

// A matrix from its columns, which is the order a composite build arrives in.
template <class V0, class... Rest>
static inline weft_mat<weft_vlen<V0>::N, 1 + (int)sizeof...(Rest)> weft_mk(V0 p0,
                                                                     Rest... rest) {
  const V0 cols[1 + sizeof...(Rest)] = {p0, rest...};
  weft_mat<weft_vlen<V0>::N, 1 + (int)sizeof...(Rest)> r{};
  for (int c = 0; c < 1 + (int)sizeof...(Rest); c++)
    for (int i = 0; i < weft_vlen<V0>::N; i++) r.m[c][i] = weft_comp(cols[c], i);
  return r;
}

// One column of a matrix, as the vector of its rows.
template <int R, int C>
static inline typename weft_vt<R>::type weft_col(weft_mat<R, C> a, int c) {
  weft_n t{};
  for (int i = 0; i < R; i++) t.v[i] = a.m[c][i];
  return weft_vt<R>::mk(t);
}

// Matrix times vector: row `i` of the result is the dot of row `i` with the vector,
// which is `m[0][i] * v[0]` and then one fused multiply-add a column.
template <int R, int C>
static inline typename weft_vt<R>::type operator*(weft_mat<R, C> a,
                                                 typename weft_vt<C>::type v) {
  weft_n t{};
  for (int i = 0; i < R; i++) {
    float acc = a.m[0][i] * weft_comp(v, 0);
    for (int k = 1; k < C; k++) acc = metal::fma(a.m[k][i], weft_comp(v, k), acc);
    t.v[i] = acc;
  }
  return weft_vt<R>::mk(t);
}

// Vector times matrix: the vector multiplies the rows, so column `c` of the result is
// the dot of the vector with column `c`.
template <int R, int C>
static inline typename weft_vt<C>::type operator*(typename weft_vt<R>::type v,
                                                 weft_mat<R, C> a) {
  weft_n t{};
  for (int c = 0; c < C; c++) {
    float acc = weft_comp(v, 0) * a.m[c][0];
    for (int k = 1; k < R; k++) acc = metal::fma(weft_comp(v, k), a.m[c][k], acc);
    t.v[c] = acc;
  }
  return weft_vt<C>::mk(t);
}

// Matrix times matrix: each column of the result is the first matrix times the
// matching column of the second.
template <int R, int C, int P>
static inline weft_mat<R, P> operator*(weft_mat<R, C> a, weft_mat<C, P> b) {
  weft_mat<R, P> r{};
  for (int c = 0; c < P; c++)
    for (int i = 0; i < R; i++) {
      float acc = a.m[0][i] * b.m[c][0];
      for (int k = 1; k < C; k++) acc = metal::fma(a.m[k][i], b.m[c][k], acc);
      r.m[c][i] = acc;
    }
  return r;
}

template <int R, int C>
static inline weft_mat<R, C> operator*(weft_mat<R, C> a, float s) {
  weft_mat<R, C> r{};
  for (int c = 0; c < C; c++)
    for (int i = 0; i < R; i++) r.m[c][i] = a.m[c][i] * s;
  return r;
}
template <int R, int C>
static inline weft_mat<R, C> operator*(float s, weft_mat<R, C> a) { return a * s; }

template <int R, int C>
static inline weft_mat<C, R> weft_transpose(weft_mat<R, C> a) {
  weft_mat<C, R> r{};
  for (int c = 0; c < C; c++)
    for (int i = 0; i < R; i++) r.m[i][c] = a.m[c][i];
  return r;
}

// A load and a store through an address the kernel computed. Both front ends
// name these, so both targets have them.
template <class T>
static inline T weft_ld(const device void* p) { return *((const device T*)p); }
template <class T>
static inline void weft_st(device void* p, T v) { *((device T*)p) = v; }

#ifdef __METAL_VERSION__
// The GPU back end addresses threadgroup and per-thread memory the same way it
// addresses a buffer, so the load and store have to say which space they read.
// The emitter knows, and picks the overload; the CPU sees only the two above.
template <class T>
static inline T weft_ld(threadgroup const void* p) { return *((threadgroup const T*)p); }
template <class T>
static inline void weft_st(threadgroup void* p, T v) { *((threadgroup T*)p) = v; }
template <class T>
static inline T weft_ld(thread const void* p) { return *((thread const T*)p); }
template <class T>
static inline void weft_st(thread void* p, T v) { *((thread T*)p) = v; }
#endif

namespace weft {

// What a thread knows about itself: the same numbers the CPU runtime hands a
// lane, the buffers it was given, and the scalar parameters.
//
// A buffer is held as the *device pointer* the launcher bound, not as an
// address the host filled in: Metal's address space is the GPU's, so a host
// pointer means nothing inside a kernel (tools/ptrprobe answers that with the
// device's own words). The emitter's address arithmetic works on the uint64
// form of that pointer, which stays inside the GPU's address space when it is
// cast back, and that is the one representation both targets agree on.
#define WEFT_MAX_BUFS 32
struct Ctx {
  uint32_t lid;
  uint32_t tgid;
  uint32_t tg_size;
  uint32_t grid;   // total threads in the launch, as the CPU runtime reports it
  uint64_t gid;    // this thread's index in the launch, 64-bit like SPIR-V's
  device uint8_t* b[WEFT_MAX_BUFS];
  device uint64_t* params;
  // The words a 64-bit read-modify-write locks on, zeroed by the launcher before
  // every dispatch.
  device uint32_t* locks;
};

// How many lanes one SIMD group can hold here. The owner loop that serialises a
// 64-bit atomic across a warp runs this many steps, and the count has to come
// from the launch rather than be the literal 32: a fixed count with a spin loop
// inside it is something the Metal compiler tries to unroll, and on a loop of
// this shape it gives up and the pipeline is refused.
static inline uint32_t lane_span(thread const Ctx& c) {
  return c.tg_size < 32u ? c.tg_size : 32u;
}

static inline uint64_t buf_addr(thread const Ctx& c, int i) {
  return (uint64_t)(uintptr_t)c.b[i];
}

// How many threads of this group are real: the last group of a grid is usually
// short, and a ballot's member mask has to say so.
static inline uint32_t live_threads(uint32_t tgid, uint32_t tg_size, uint32_t grid) {
  const uint64_t base = (uint64_t)tgid * (uint64_t)tg_size;
  if (base >= (uint64_t)grid) return 0u;
  const uint64_t left = (uint64_t)grid - base;
  return (uint32_t)(left < (uint64_t)tg_size ? left : (uint64_t)tg_size);
}

// A declaration `OpVariable` with an initialiser zeroes threadgroup memory
// before the body runs; there is no memset here, so it is a store per byte.
static inline void zero_shared(threadgroup uint8_t* d, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) d[i] = 0u;
}
static inline void zero_thread(thread uint8_t* d, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) d[i] = 0u;
}

//
// An image is the bound texture and the two numbers a kernel asks about it. It
// travels by value because it has to: a texture is an opaque handle, and thread
// space cannot hold an address of one. The access mode is part of the type, so
// `ImageRef` is a sampled texture and `RWImageRef` a storage image; both take a
// `read`, which is what `OpImageRead` is.
struct ImageRef {
  texture2d<float, access::sample> t;
  uint32_t width;
  uint32_t height;
};
struct RWImageRef {
  texture2d<float, access::read_write> t;
  uint32_t width;
  uint32_t height;
};
struct SampledImage {
  ImageRef tex;
  sampler s;
};

// A 2D *array* image: the handle is `texture2d_array`, whose slice count is part
// of the binding, and every read and sample names the slice it wants. The rest
// has the same shape as the plain form, one channel, a texel is channel 0, so
// these are overloads rather than a second type system.
struct ArrayImageRef {
  texture2d_array<float, access::sample> t;
  uint32_t width;
  uint32_t height;
  uint32_t layers;
};
struct RWArrayImageRef {
  texture2d_array<float, access::read_write> t;
  uint32_t width;
  uint32_t height;
  uint32_t layers;
};
struct SampledArrayImage {
  ArrayImageRef tex;
  sampler s;
};

// A 3D image: `texture3d`, whose depth is part of the binding the way a 2D
// array's slice count is, and every read and sample names a plane. The three
// handles have the same shape, so the kernel-side calls are overloads rather than
// a second set of names.
struct Image3DRef {
  texture3d<float, access::sample> t;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
};
struct RWImage3DRef {
  texture3d<float, access::read_write> t;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
};
struct SampledImage3D {
  Image3DRef tex;
  sampler s;
};

// A cube: six square faces, and a sample takes a direction rather than a position.
// A cube is filterable only, there is no texel to address without a face, so
// there is no read or write form of it here.
struct CubeImageRef {
  texturecube<float, access::sample> t;
  uint32_t width;
  uint32_t height;
};
struct SampledCubeImage {
  CubeImageRef tex;
  sampler s;
};

static inline ImageRef make_image(texture2d<float, access::sample> t) {
  return ImageRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height()};
}
static inline ArrayImageRef make_array_image(texture2d_array<float, access::sample> t) {
  return ArrayImageRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height(),
                       (uint32_t)t.get_array_size()};
}
static inline RWArrayImageRef make_rw_array_image(texture2d_array<float, access::read_write> t) {
  return RWArrayImageRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height(),
                         (uint32_t)t.get_array_size()};
}
static inline RWImageRef make_rw_image(texture2d<float, access::read_write> t) {
  return RWImageRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height()};
}
static inline Image3DRef make_image3d(texture3d<float, access::sample> t) {
  return Image3DRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height(),
                    (uint32_t)t.get_depth()};
}
static inline RWImage3DRef make_rw_image3d(texture3d<float, access::read_write> t) {
  return RWImage3DRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height(),
                      (uint32_t)t.get_depth()};
}
static inline CubeImageRef make_cube_image(texturecube<float, access::sample> t) {
  return CubeImageRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height()};
}

// The launcher binds one-channel images, so a texel is channel 0 and the value
// is those four bytes. A read outside the image is nothing, which is what the
// hardware answers too.
// A reference has to name its address space in this language, and what a kernel
// holds is a copy of the handle in thread space.
template <class T, class R>
static inline T tex_read(thread const R& im, uint32_t x, uint32_t y) {
  if (x >= im.width || y >= im.height) return (T)0;
  return weft_bitcast<T>(im.t.read(uint2(x, y)).x);
}

// The array form: the same read, of the slice the kernel names. A layer outside
// the binding is nothing, which is what the CPU runtime answers too.
template <class T, class R>
static inline T tex_read(thread const R& im, uint32_t x, uint32_t y, uint32_t layer) {
  if (x >= im.width || y >= im.height || layer >= im.layers) return (T)0;
  return weft_bitcast<T>(im.t.read(uint2(x, y), layer).x);
}

// The 3D form: one coordinate, three components, and a plane outside the binding
// is nothing, the same answer the CPU runtime gives for the same coordinate.
template <class T>
static inline T tex_read(thread const Image3DRef& im, uint32_t x, uint32_t y, uint32_t z) {
  if (x >= im.width || y >= im.height || z >= im.depth) return (T)0;
  return weft_bitcast<T>(im.t.read(uint3(x, y, z)).x);
}

// A sample goes through the bound sampler state, which is where the filter and
// the address mode live on this target: the CPU runtime models the same two
// choices, and the launcher hands them to both.
template <class T>
static inline T tex_sample(thread const SampledImage& si, float u, float v) {
  return weft_bitcast<T>(si.tex.t.sample(si.s, float2(u, v)).x);
}

template <class T>
static inline T tex_sample(thread const SampledArrayImage& si, float u, float v, uint32_t layer) {
  return weft_bitcast<T>(si.tex.t.sample(si.s, float2(u, v), layer).x);
}

// A 3D sample: one normalised coordinate in three components, and the hardware's
// own trilinear filter. The CPU runtime models the same filter with the same 8-bit
// weights, which is what lets a filtered 3D sample be compared at all.
template <class T>
static inline T tex_sample(thread const SampledImage3D& si, float u, float v, float w) {
  return weft_bitcast<T>(si.tex.t.sample(si.s, float3(u, v, w)).x);
}

// A face and a position on it, back into a direction: the inverse of the rule the
// sampler uses, so PTX's `(s, t, face)` coordinate can name a sample here. This is
// the same text as the CPU runtime's, arithmetic for arithmetic.
static inline void weft_cube_dir(uint32_t face, float s, float t, thread float& dx,
                                 thread float& dy, thread float& dz) {
  switch (face % 6u) {
    case 0: dx = 1.0f; dy = -t; dz = -s; break;
    case 1: dx = -1.0f; dy = -t; dz = s; break;
    case 2: dx = s; dy = 1.0f; dz = t; break;
    case 3: dx = s; dy = -1.0f; dz = -t; break;
    case 4: dx = s; dy = -t; dz = 1.0f; break;
    default: dx = -s; dy = -t; dz = -1.0f; break;
  }
}

// A direction back to a face and a place on it: the inverse of `weft_cube_dir`, and
// the rule the sampler hardware implements, the axis the direction points along most
// strongly picks the face, ties going to X then Y, and the other two components over
// that length place the texel. The same text is in `runtime/weft_runtime.hpp`.
static inline void weft_cube_uv(float dx, float dy, float dz, thread uint32_t& face,
                                thread float& u, thread float& v) {
  const float ax = fabs(dx), ay = fabs(dy), az = fabs(dz);
  float sc = 0.0f, tc = 0.0f, ma = 1.0f;
  if (ax >= ay && ax >= az) {
    ma = ax;
    if (dx >= 0.0f) { face = 0u; sc = -dz; tc = -dy; }
    else { face = 1u; sc = dz; tc = -dy; }
  } else if (ay >= az) {
    ma = ay;
    if (dy >= 0.0f) { face = 2u; sc = dx; tc = dz; }
    else { face = 3u; sc = dx; tc = -dz; }
  } else {
    ma = az;
    if (dz >= 0.0f) { face = 4u; sc = dx; tc = -dy; }
    else { face = 5u; sc = -dx; tc = -dy; }
  }
  if (ma == 0.0f) { u = 0.5f; v = 0.5f; return; }
  u = (sc / ma + 1.0f) * 0.5f;
  v = (tc / ma + 1.0f) * 0.5f;
}

// A cube sample: the direction picks the face and the place on it, which is the
// hardware's own rule and the one the CPU runtime models.
template <class T>
static inline T tex_sample(thread const SampledCubeImage& si, float dx, float dy, float dz) {
  return weft_bitcast<T>(si.tex.t.sample(si.s, float3(dx, dy, dz)).x);
}

// A texel is channel 0 of whatever the module wrote, the format is one channel,
// so the other three have nowhere to go, exactly as the CPU runtime copies only
// the first element.
static inline float tex_chan(float v) { return v; }
static inline float tex_chan(int32_t v) { return as_type<float>(v); }
static inline float tex_chan(uint32_t v) { return as_type<float>(v); }
static inline float tex_chan(int16_t v) { return (float)v; }
static inline float tex_chan(uint16_t v) { return (float)v; }

// The four channels a `tex` instruction returns. The binding is one channel
// wide, so the texel is channel 0 and the rest is the shape the format has.
template <class T>
static inline void tex_sample4(thread const SampledImage& si, float u, float v,
                               thread float* out) {
  out[0] = weft_bitcast<T>(si.tex.t.sample(si.s, float2(u, v)).x);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

template <class T>
static inline void tex_sample4(thread const SampledArrayImage& si, float u, float v, uint32_t layer,
                               thread float* out) {
  out[0] = weft_bitcast<T>(si.tex.t.sample(si.s, float2(u, v), layer).x);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

template <class T>
static inline void tex_sample4(thread const SampledImage3D& si, float u, float v, float w,
                               thread float* out) {
  out[0] = weft_bitcast<T>(si.tex.t.sample(si.s, float3(u, v, w)).x);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

template <class T>
static inline void tex_sample4(thread const SampledCubeImage& si, float dx, float dy, float dz,
                               thread float* out) {
  out[0] = weft_bitcast<T>(si.tex.t.sample(si.s, float3(dx, dy, dz)).x);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}


// `tld4`: the four texels around the coordinate, each on its own channel,
// instead of the one a filter would blend. The footprint is the 2x2 block whose
// base is the texel under the coordinate, which is what the instruction means.
template <class T>
static inline void tex_gather4(thread const SampledArrayImage& si, float u, float v, uint32_t layer,
                               thread float* out) {
  const int x0 = (int)floor(u * (float)si.tex.width - 0.5f);
  const int y0 = (int)floor(v * (float)si.tex.height - 0.5f);
  auto at = [&](int x, int y) -> float {
    if (x < 0 || y < 0 || (uint32_t)x >= si.tex.width || (uint32_t)y >= si.tex.height ||
        layer >= si.tex.layers)
      return 0.0f;
    return weft_bitcast<T>(si.tex.t.read(uint2((uint32_t)x, (uint32_t)y), layer).x);
  };
  out[0] = at(x0, y0);
  out[1] = at(x0 + 1, y0);
  out[2] = at(x0, y0 + 1);
  out[3] = at(x0 + 1, y0 + 1);
}

template <class T>
static inline void tex_gather4(thread const SampledImage& si, float u, float v,
                               thread float* out) {
  const int x0 = (int)floor(u * (float)si.tex.width - 0.5f);
  const int y0 = (int)floor(v * (float)si.tex.height - 0.5f);
  auto at = [&](int x, int y) -> float {
    if (x < 0 || y < 0 || (uint32_t)x >= si.tex.width || (uint32_t)y >= si.tex.height) return 0.0f;
    return weft_bitcast<T>(si.tex.t.read(uint2((uint32_t)x, (uint32_t)y)).x);
  };
  out[0] = at(x0, y0);
  out[1] = at(x0 + 1, y0);
  out[2] = at(x0, y0 + 1);
  out[3] = at(x0 + 1, y0 + 1);
}

// A 3-D gather: the 2x2 footprint of the plane the coordinate names. The plane is
// the nearest one under w, because a gather has no filter to blend along z with,
// and a footprint that leaves the binding answers with nothing, the CPU runtime's
// text for the same instruction, arithmetic for arithmetic.
template <class T>
static inline void tex_gather4(thread const SampledImage3D& si, float u, float v, float w,
                               thread float* out) {
  const int x0 = (int)floor(u * (float)si.tex.width - 0.5f);
  const int y0 = (int)floor(v * (float)si.tex.height - 0.5f);
  const int z = (int)floor(w * (float)si.tex.depth);
  auto at = [&](int x, int y) -> float {
    if (x < 0 || y < 0 || z < 0 || (uint32_t)x >= si.tex.width || (uint32_t)y >= si.tex.height ||
        (uint32_t)z >= si.tex.depth)
      return 0.0f;
    return weft_bitcast<T>(si.tex.t.read(uint3((uint32_t)x, (uint32_t)y, (uint32_t)z)).x);
  };
  out[0] = at(x0, y0);
  out[1] = at(x0 + 1, y0);
  out[2] = at(x0, y0 + 1);
  out[3] = at(x0 + 1, y0 + 1);
}

// A cube gather: a face and a place on it, turned into a layer and a 2-D
// coordinate. A cube texel is addressed by `read(uint2, face)`, so the four reads
// are exact on both engines and no sampler is involved.
template <class T>
static inline void tex_gather4_face(thread const SampledCubeImage& si, float u, float v,
                                    uint32_t face, thread float* out) {
  const int x0 = (int)floor(u * (float)si.tex.width - 0.5f);
  const int y0 = (int)floor(v * (float)si.tex.height - 0.5f);
  const uint32_t f = face % 6u;
  auto at = [&](int x, int y) -> float {
    if (x < 0 || y < 0 || (uint32_t)x >= si.tex.width || (uint32_t)y >= si.tex.height) return 0.0f;
    return weft_bitcast<T>(si.tex.t.read(uint2((uint32_t)x, (uint32_t)y), f).x);
  };
  out[0] = at(x0, y0);
  out[1] = at(x0 + 1, y0);
  out[2] = at(x0, y0 + 1);
  out[3] = at(x0 + 1, y0 + 1);
}

template <class T>
static inline void tex_gather4_cube(thread const SampledCubeImage& si, float s, float v,
                                    uint32_t face, thread float* out) {
  tex_gather4_face<T>(si, (s + 1.0f) * 0.5f, (v + 1.0f) * 0.5f, face, out);
}

// A direction instead of a face and a place: the rule the sampler implements picks
// both, then the same four texels are read off that face.
template <class T>
static inline void tex_gather4_cube_dir(thread const SampledCubeImage& si, float dx, float dy,
                                        float dz, float du, float dv, thread float* out) {
  uint32_t face = 0;
  float u = 0.5f, v = 0.5f;
  weft_cube_uv(dx, dy, dz, face, u, v);
  tex_gather4_face<T>(si, u + du, v + dv, face, out);
}

template <class R, class V>
static inline void tex_write(thread const R& im, uint32_t x, uint32_t y, V v) {
  if (x >= im.width || y >= im.height) return;
  im.t.write(float4(tex_chan(v.x), 0.0f, 0.0f, 0.0f), uint2(x, y));
}

template <class R, class V>
static inline void tex_write(thread const R& im, uint32_t x, uint32_t y, uint32_t layer, V v) {
  if (x >= im.width || y >= im.height || layer >= im.layers) return;
  im.t.write(float4(tex_chan(v.x), 0.0f, 0.0f, 0.0f), uint2(x, y), layer);
}

template <class V>
static inline void tex_write(thread const RWImage3DRef& im, uint32_t x, uint32_t y, uint32_t z, V v) {
  if (x >= im.width || y >= im.height || z >= im.depth) return;
  im.t.write(float4(tex_chan(v.x), 0.0f, 0.0f, 0.0f), uint3(x, y, z));
}

// SPIR-V says `OpAtomicLoad` / `OpAtomicStore` with a type; on the GPU those are
// the atomic instructions over the same 32 bits.
template <class T>
static inline T atomic_load(device void* p) {
  device atomic_uint* a = (device atomic_uint*)p;
  return as_type<T>(atomic_load_explicit(a, memory_order_relaxed));
}
template <class T>
static inline void atomic_store(device void* p, T v) {
  device atomic_uint* a = (device atomic_uint*)p;
  atomic_store_explicit(a, as_type<uint32_t>(v), memory_order_relaxed);
}
static inline uint64_t param(thread const Ctx& c, int i) { return c.params[i]; }

// The hardware's own fetch forms, not a retry loop, and that is not a
// micro-optimisation. A compare-exchange retry costs every contender one line
// transfer per success, so a counter touched by n threads costs about n^2 of them:
// at 4,096 lanes that is 16 million and at 65,536 it is 4 billion, which does not
// finish, it reads as a hang, and it did: the reduction path below (no lock, no
// 64-bit anything) hung at 65,536 lanes with the retry loop and finishes now.
// A CAS loop is only used where no instruction exists: min and max.
static inline uint32_t atom_add_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_add_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_sub_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_sub_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_and_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_and_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_or_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_or_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_xor_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_xor_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_exch_u32(device uint32_t* p, uint32_t v) {
  return atomic_exchange_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
// compare-exchange *is* the instruction for cas, and it hands back the old value
// whether or not it matched, which is the answer this has to return.
static inline uint32_t atom_cas_u32(device uint32_t* p, uint32_t cmp, uint32_t v) {
  uint32_t expect = cmp;
  (void)atomic_compare_exchange_weak_explicit((device atomic_uint*)p, &expect, v,
                                              memory_order_relaxed, memory_order_relaxed);
  return expect;
}
static inline int32_t atom_add_s32(device int32_t* p, int32_t v) {
  return (int32_t)atom_add_u32((device uint32_t*)p, (uint32_t)v);
}
// min and max have instructions here too, for both integer types, the note that
// they are refused at pipeline creation was about *float* atomics, and a probe
// asking for `atomic_fetch_max_explicit` on `atomic_uint` and `atomic_int`
// pipelines on this host. Using them matters for the same reason as add: a retry
// loop on one hot word costs n^2 line transfers, and 1,048,576 lanes is where that
// shows.
static inline int32_t atom_min_s32(device int32_t* p, int32_t v) {
  return atomic_fetch_min_explicit((device atomic_int*)p, v, memory_order_relaxed);
}
static inline int32_t atom_max_s32(device int32_t* p, int32_t v) {
  return atomic_fetch_max_explicit((device atomic_int*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_min_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_min_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_max_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_max_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
// A float add has an instruction as well (`atomic_float`), so it is native.
static inline float atom_add_f32(device float* p, float v) {
  return atomic_fetch_add_explicit((device atomic_float*)p, v, memory_order_relaxed);
}
// Float min and max do not: `atomic_fetch_min_explicit` / `max` are constrained to
// the integer types, and asking for `atomic_float` is a compile error. So this is
// the only retry loop left in the runtime, and it is the one place a very large grid
// could still grind, a hot word costs the number of contenders per success. Nothing
// in `kernels/` emits `atom.min.f32` or `atom.max.f32`; if something does, the pause
// below is what keeps it from spinning on the line. The arithmetic is done on the
// float the bits are, and `+.rn` is the addition this language performs.
static inline float cas_f32(device float* p, float v, int mode) {
  device atomic_uint* a = (device atomic_uint*)p;
  uint32_t old = atomic_load_explicit(a, memory_order_relaxed);
  uint32_t wait = 1u;
  for (;;) {
    const float o = as_type<float>(old);
    // The same rule the CPU's `atom_min_f32` uses, spelled out rather than left to
    // `min`: Metal's `min` is `fmin`, which answers with the non-NaN operand, and
    // the CPU's comparison answers with `v`. Two columns that disagree on a NaN
    // are two engines with two answers.
    const float want = (mode == 1) ? (o < v ? o : v) : (o > v ? o : v);
    if (atomic_compare_exchange_weak_explicit(a, &old, as_type<uint32_t>(want),
                                              memory_order_relaxed, memory_order_relaxed))
      return o;
    for (uint32_t i = 0; i < wait; i++) (void)atomic_load_explicit(a, memory_order_relaxed);
    wait = wait < 512u ? wait * 2u : 512u;
  }
}
static inline float atom_min_f32(device float* p, float v) { return cas_f32(p, v, 1); }
static inline float atom_max_f32(device float* p, float v) { return cas_f32(p, v, 2); }

//
// A `red` produces no result, so it does not need the 64 bits read as one value , 
// it needs the pair to end up holding the fold of everything applied to it. For
// an operation that is order-free that is two 32-bit atomics:
//
//   add/sub  the low half goes first and its carry (or borrow) is applied to the
//            high half. Every thread that wraps the low word adds exactly one to
//            the high word, so the sum is right whatever order the threads land
//            in, and the hardware does the low half atomically, so nothing is
//            lost in between.
//   and/or/xor  each half is folded on its own. The two halves are independent,
//            so the pair ends up holding the fold without either half needing to
//            know about the other.
//
// What this cannot do is hand back the old pair as one value, or compare one.
// That needs the pair read as one value, which no instruction here does, so it
// is a lock over the two halves instead.

//
// There is no 64-bit atomic on this host. `atomic_ulong` is declared, but every
// entry point that could read, write, exchange or compare one is constrained to
// 32-bit types, and the two that are not (`atomic_max_explicit` /
// `atomic_min_explicit`) compile only to be refused by the device at pipeline
// creation, tools/atom64_probe.mm asks for each of those answers separately.
//
// So a 64-bit atomic is built out of the 32-bit ones the hardware does have: a
// lock on the word, held while the pair is read, changed and written back. The
// halves move through `atomic_load_explicit` / `atomic_store_explicit` rather
// than as one 64-bit load and store, because a plain load of a `device` word can
// be served from a core's own cache, which is not the point the atomics agree on.
//
// The wait spins, so no thread may be able to freeze the thread it waits for. Two
// threads in one SIMD group share a program counter, so a kernel that runs one
// hardware lane per thread takes turns first: only the lane whose id matches the
// current step is inside the critical section (the owner loop the translators
// emit), and then nobody waits on a peer that is itself waiting. A warp kernel
// has one thread per warp, so its lanes are data and never contend.
//
// A thread holds at most one lock at a time and a holder is by definition
// running, so it always reaches its release: nothing here can deadlock.
static constant uint32_t kLockWords = 1024u;

static inline device uint32_t* lock_of(device uint32_t* locks, device const void* p) {
  const uint64_t a = (uint64_t)(uintptr_t)p;
  const uint32_t h = (uint32_t)(a >> 3) ^ (uint32_t)(a >> 13);
  return locks + (h & (kLockWords - 1u));
}
// One word per lock, taken by test-and-set. A ticket lock would be fairer and was
// tried first: it deadlocks here, because a ticket is taken behind a thread whose
// threadgroup has not been dispatched yet, and that thread cannot reach its turn
// from off the device. So the wait is a plain spin, made polite: a loser stops
// writing and reads until the word looks free, and waits a jittered number of
// reads before trying again. Writing every time is what makes the line ping-pong
// between thousands of waiters, and retrying in step is what let one waiter lose
// for minutes.
//
// Every operation here is `memory_order_relaxed`, because this host's metal_atomic
// constrains all of them to it. Ordering is not the obstacle, a device-scope
// `atomic_thread_fence` exists and a relaxed store from one threadgroup is seen by
// a waiter in another (tools/lock_probe.mm measures both). The obstacle is
// progress: a lane that wins this section cannot run it until the other lanes in
// its warp leave the spin loop, and they cannot leave until it releases, because a
// warp is one program counter. A threadgroup of one thread does not dodge that
// either, such threadgroups are co-scheduled into one warp and deadlock across
// each other. So the lock is only taken where at most one lane per warp can ever
// wait, which is the single-writer walk the 64-bit pair uses; a per-lane section
// under contention cannot complete on this hardware, and the translator refuses
// that shape by name rather than emit something that hangs.
static inline void lock_take(device uint32_t* l) {
  device atomic_uint* a = (device atomic_uint*)l;
  uint32_t wait = 1u;
  for (;;) {
    uint32_t expect = 0u;
    if (atomic_compare_exchange_weak_explicit(a, &expect, 1u, memory_order_relaxed,
                                             memory_order_relaxed))
      return;
    uint32_t spins = wait + ((uint32_t)(uintptr_t)l & 7u);
    while (spins != 0u && atomic_load_explicit(a, memory_order_relaxed) != 0u) spins--;
    wait = wait < 64u ? wait * 2u : 64u;
  }
}
static inline void lock_drop(device uint32_t* l) {
  atomic_store_explicit((device atomic_uint*)l, 0u, memory_order_relaxed);
}

// The pair is assembled through a `uint2` and `as_type`, not `(hi << 32) | lo`.
// That is measured: the shift-and-or of two atomic half-reads is a construct this
// host's pipeline compiler dies on as soon as it sits in a kernel with a barrier
// (XPC_ERROR_CONNECTION_INTERRUPTED at pipeline creation, 3/3 runs), and the same
// loads joined with `as_type<uint64_t>(uint2{lo, hi})` pipeline. The two are the
// same bits, and tools/msl_bisect.py walks the variants if the compiler changes.
static inline uint64_t load64(device uint64_t* p) {
  device atomic_uint* a = (device atomic_uint*)p;
  const uint32_t lo = atomic_load_explicit(a, memory_order_relaxed);
  const uint32_t hi = atomic_load_explicit(a + 1, memory_order_relaxed);
  return as_type<uint64_t>(uint2{lo, hi});
}
static inline int64_t load64(device int64_t* p) {
  return as_type<int64_t>(load64((device uint64_t*)p));
}
// The halves come out of a `uint2` for the reason load64 states: a 64-bit shift
// feeding an atomic store is the same shape that takes the compiler down here.
static inline void store64(device uint64_t* p, uint64_t v) {
  device atomic_uint* a = (device atomic_uint*)p;
  const uint2 w = as_type<uint2>(v);
  atomic_store_explicit(a, w.x, memory_order_relaxed);
  atomic_store_explicit(a + 1, w.y, memory_order_relaxed);
}
static inline void store64(device int64_t* p, int64_t v) {
  store64((device uint64_t*)p, as_type<uint64_t>(v));
}

//
// A warp kernel's lanes are this thread's data, so a vote is a rendezvous with
// itself: every lane publishes into the warp's slot array, the dispatch brings the
// lanes back, and the answer is folded over the slots. `mask` is the lanes that
// exist, because the last warp of a group is short when the lane count is not a
// multiple of 32 and the slots of lanes that never ran hold nothing.
//
// The array is threadgroup memory rather than thread memory on purpose: a
// loop-carried value over a thread array is the shape this host's compiler dies on
// (see load64 above), and a fold over memory is not.
static inline uint32_t slot_ballot(threadgroup const uint32_t* s, uint32_t mask) {
  uint32_t m = 0u;
  for (uint32_t i = 0u; i < 32u; i++)
    if (((mask >> i) & 1u) != 0u && (s[i] & 1u) != 0u) m |= 1u << i;
  return m;
}
static inline uint32_t slot_all_equal(threadgroup const uint32_t* s, uint32_t mask) {
  uint32_t base = 0u, seen = 0u, same = 1u;
  for (uint32_t i = 0u; i < 32u; i++) {
    if (((mask >> i) & 1u) == 0u) continue;
    if (seen == 0u) { base = s[i]; seen = 1u; }
    else if (s[i] != base) same = 0u;
  }
  return same;
}

// A comparison and a select on the pair are written as two 32-bit words, and that
// is measured rather than stylistic. This host's pipeline compiler dies at
// creation (XPC_ERROR_CONNECTION_INTERRUPTED, 3/3, while the same helper doing an
// add pipelines) when a 64-bit compare or a 64-bit select sits inside a function
// that is inlined into a kernel; the identical code over 32-bit halves and a mask
// select pipelines. tools/msl_bisect.py walks the variants if the compiler ever
// changes.
static inline uint32_t gt_u64(uint64_t a, uint64_t b) {  // a > b, as 1 or 0
  const uint32_t a1 = (uint32_t)(a >> 32), b1 = (uint32_t)(b >> 32);
  const uint32_t a0 = (uint32_t)a, b0 = (uint32_t)b;
  return ((a1 > b1) | ((a1 == b1) & (a0 > b0))) ? 1u : 0u;
}
static inline uint32_t gt_s64(int64_t a, int64_t b) {  // signed: the high word too
  const int32_t a1 = (int32_t)((uint64_t)a >> 32), b1 = (int32_t)((uint64_t)b >> 32);
  const uint32_t a0 = (uint32_t)a, b0 = (uint32_t)b;
  return ((a1 > b1) | ((a1 == b1) & (a0 > b0))) ? 1u : 0u;
}
static inline uint32_t eq_bits64(uint64_t a, uint64_t b) {
  return (((uint32_t)a == (uint32_t)b) & ((uint32_t)(a >> 32) == (uint32_t)(b >> 32))) ? 1u : 0u;
}
// `v` when take is 1, else `o`.
static inline uint64_t pick_u64(uint64_t o, uint64_t v, uint32_t take) {
  const uint32_t m = 0u - take;
  const uint32_t lo = ((uint32_t)v & m) | ((uint32_t)o & ~m);
  const uint32_t hi = ((uint32_t)(v >> 32) & m) | ((uint32_t)(o >> 32) & ~m);
  return ((uint64_t)hi << 32) | (uint64_t)lo;
}
static inline int64_t pick_s64(int64_t o, int64_t v, uint32_t take) {
  return as_type<int64_t>(pick_u64(as_type<uint64_t>(o), as_type<uint64_t>(v), take));
}

// The update with the lock taken and released around it.
#define WEFT_RMW(name, T, expr)                                             \
  static inline T rmw_##name(device uint32_t* locks, device T* p, T v) {    \
    device uint32_t* l = lock_of(locks, p);                                 \
    lock_take(l);                                                           \
    const T o = load64(p);                                                  \
    store64(p, (expr));                                                     \
    lock_drop(l);                                                           \
    return o;                                                               \
  }
// ...and the same update for a caller that is the only writer.
//
// The warp body that moves a 64-bit pair is one thread for the whole launch, so
// nothing else touches the word while it runs: a plain 64-bit load and store is
// the whole operation, and it is one access instead of the four device-scope
// atomics `_raw` needs. Dispatches are serial, so this is also correct across
// them.
#define WEFT_RMW_SOLO(name, T, expr)                    \
  static inline T rmw_##name##_solo(device T* p, T v) { \
    const T o = *p;                                     \
    *p = (expr);                                        \
    return o;                                           \
  }
#define WEFT_RMW_ALL_SOLO(MACRO)                  \
  MACRO(add, uint64_t, o + v)                     \
  MACRO(sub, uint64_t, o - v)                     \
  MACRO(min, uint64_t, (o < v ? o : v))           \
  MACRO(max, uint64_t, (o > v ? o : v))           \
  MACRO(and, uint64_t, o & v)                     \
  MACRO(or, uint64_t, o | v)                      \
  MACRO(xor, uint64_t, o ^ v)                     \
  MACRO(exch, uint64_t, v)                        \
  MACRO(min, int64_t, (o < v ? o : v))            \
  MACRO(max, int64_t, (o > v ? o : v))            \
  MACRO(add, int64_t, o + v)                      \
  MACRO(sub, int64_t, o - v)                      \
  MACRO(and, int64_t, o & v)                      \
  MACRO(or, int64_t, o | v)                       \
  MACRO(xor, int64_t, o ^ v)                      \
  MACRO(exch, int64_t, v)
WEFT_RMW_ALL_SOLO(WEFT_RMW_SOLO)
#undef WEFT_RMW_ALL_SOLO
#undef WEFT_RMW_SOLO
static inline uint64_t rmw_cas_solo(device uint64_t* p, uint64_t c, uint64_t v) {
  const uint64_t o = *p;
  if (o == c) *p = v;
  return o;
}
static inline int64_t rmw_cas_solo(device int64_t* p, int64_t c, int64_t v) {
  const int64_t o = *p;
  if (o == c) *p = v;
  return o;
}

// ...and the same update for a caller that already holds one.
#define WEFT_RMW_RAW(name, T, expr)        \
  static inline T rmw_##name##_raw(device T* p, T v) { \
    const T o = load64(p);                \
    store64(p, (expr));                   \
    return o;                             \
  }
#define WEFT_RMW_ALL(MACRO)                        \
  MACRO(add, uint64_t, o + v)                      \
  MACRO(sub, uint64_t, o - v)                      \
  MACRO(min, uint64_t, pick_u64(o, v, gt_u64(o, v))) \
  MACRO(max, uint64_t, pick_u64(o, v, gt_u64(v, o))) \
  MACRO(and, uint64_t, o & v)                      \
  MACRO(or, uint64_t, o | v)                       \
  MACRO(xor, uint64_t, o ^ v)                      \
  MACRO(exch, uint64_t, v)                         \
  MACRO(min, int64_t, pick_s64(o, v, gt_s64(o, v))) \
  MACRO(max, int64_t, pick_s64(o, v, gt_s64(v, o))) \
  MACRO(add, int64_t, o + v)                       \
  MACRO(sub, int64_t, o - v)                       \
  MACRO(and, int64_t, o & v)                       \
  MACRO(or, int64_t, o | v)                        \
  MACRO(xor, int64_t, o ^ v)                       \
  MACRO(exch, int64_t, v)
WEFT_RMW_ALL(WEFT_RMW)
WEFT_RMW_ALL(WEFT_RMW_RAW)
#undef WEFT_RMW_ALL
#undef WEFT_RMW_RAW
#undef WEFT_RMW

// `atom.cas` keeps the word it found and only writes when the pair matches, so
// it needs the compare as well as the value.
static inline uint64_t rmw_cas(device uint32_t* locks, device uint64_t* p, uint64_t c,
                              uint64_t v) {
  device uint32_t* l = lock_of(locks, p);
  lock_take(l);
  const uint64_t o = load64(p);
  if (eq_bits64(o, c)) store64(p, v);
  lock_drop(l);
  return o;
}
static inline int64_t rmw_cas(device uint32_t* locks, device int64_t* p, int64_t c, int64_t v) {
  device uint32_t* l = lock_of(locks, p);
  lock_take(l);
  const int64_t o = load64(p);
  if (eq_bits64(as_type<uint64_t>(o), as_type<uint64_t>(c))) store64(p, v);
  lock_drop(l);
  return o;
}
static inline uint64_t rmw_cas_raw(device uint64_t* p, uint64_t c, uint64_t v) {
  const uint64_t o = load64(p);
  if (eq_bits64(o, c)) store64(p, v);
  return o;
}
static inline int64_t rmw_cas_raw(device int64_t* p, int64_t c, int64_t v) {
  const int64_t o = load64(p);
  if (eq_bits64(as_type<uint64_t>(o), as_type<uint64_t>(c))) store64(p, v);
  return o;
}
// A 64-bit read-modify-write when this thread is the only one touching the word.
//
// The walk that owns the launch also owns the word, so the value can live in a
// local between the sites that use it and reach the buffer twice per launch
// instead of twice per lane, and each of those trips is an L2 round trip. The
// old value is still handed back in program order, so the sequence both engines
// see is the same one; the emitter only takes this form when nothing else in the
// kernel reads that binding, which is what makes the local a copy rather than a
// cache.
#define WEFT_SH64(name, T, expr)                                              \
  static inline T sh64_##name(thread T& slot, T v) {                          \
    const T o = slot;                                                         \
    slot = (expr);                                                            \
    return o;                                                                 \
  }
#define WEFT_SH64_ALL(MACRO)                \
  MACRO(add, uint64_t, o + v)               \
  MACRO(sub, uint64_t, o - v)               \
  MACRO(and, uint64_t, o & v)               \
  MACRO(or, uint64_t, o | v)                \
  MACRO(xor, uint64_t, o ^ v)               \
  MACRO(min, uint64_t, (o < v ? o : v))     \
  MACRO(max, uint64_t, (o > v ? o : v))     \
  MACRO(exch, uint64_t, v)                  \
  MACRO(min, int64_t, (o < v ? o : v))      \
  MACRO(max, int64_t, (o > v ? o : v))      \
  MACRO(add, int64_t, o + v)                \
  MACRO(sub, int64_t, o - v)                \
  MACRO(and, int64_t, o & v)                \
  MACRO(or, int64_t, o | v)                 \
  MACRO(xor, int64_t, o ^ v)                \
  MACRO(exch, int64_t, v)
WEFT_SH64_ALL(WEFT_SH64)
#undef WEFT_SH64_ALL
#undef WEFT_SH64

static inline device uint32_t* lo32(device uint64_t* p) { return (device uint32_t*)p; }
static inline device uint32_t* hi32(device uint64_t* p) { return (device uint32_t*)p + 1; }
static inline void red_add_u64(device uint64_t* p, uint64_t v) {
  const uint32_t vl = (uint32_t)v, vh = (uint32_t)(v >> 32);
  const uint32_t old = atom_add_u32(lo32(p), vl);
  atom_add_u32(hi32(p), vh + ((old + vl) < old ? 1u : 0u));
}
static inline void red_sub_u64(device uint64_t* p, uint64_t v) {
  const uint32_t vl = (uint32_t)v, vh = (uint32_t)(v >> 32);
  const uint32_t old = atom_sub_u32(lo32(p), vl);
  atom_sub_u32(hi32(p), vh + (old < vl ? 1u : 0u));
}
static inline void red_and_u64(device uint64_t* p, uint64_t v) {
  atom_and_u32(lo32(p), (uint32_t)v);
  atom_and_u32(hi32(p), (uint32_t)(v >> 32));
}
static inline void red_or_u64(device uint64_t* p, uint64_t v) {
  atom_or_u32(lo32(p), (uint32_t)v);
  atom_or_u32(hi32(p), (uint32_t)(v >> 32));
}
static inline void red_xor_u64(device uint64_t* p, uint64_t v) {
  atom_xor_u32(lo32(p), (uint32_t)v);
  atom_xor_u32(hi32(p), (uint32_t)(v >> 32));
}

// `bar.red` publishes one bit per thread, then every thread folds the group.
// The barrier is the hardware's, so this is a publish, a barrier, and a loop , 
// no rendezvous bookkeeping, which is what the CPU side has to do by hand.
static inline uint32_t red_fold(threadgroup uint32_t* slots, thread const Ctx& c, int mode) {
  uint32_t t = 0u;
  if (mode == 0) {
    for (uint32_t i = 0; i < c.tg_size; i++) t += (uint32_t)popcount(slots[i]);
  } else if (mode == 1) {
    t = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < c.tg_size; i++) t &= slots[i];
  } else {
    for (uint32_t i = 0; i < c.tg_size; i++) t |= slots[i];
  }
  return t;
}

// `cp.async` copies into threadgroup memory. The GPU has a real asynchronous
// copy, but the answer after the wait is the same bytes, so the copy is issued
// inline and the commit/wait stay no-ops.
static inline void copy_shared(threadgroup uint8_t* dst, const device void* src, uint32_t n) {
  const device uint8_t* s = (const device uint8_t*)src;
  for (uint32_t i = 0; i < n; i++) dst[i] = s[i];
}

// The bulk store's direction: threadgroup memory back out to the buffer. The GPU
// has a real asynchronous store, but the bytes in the buffer after the wait are the
// same ones, so it is issued inline and the group commit stays a no-op.
static inline void copy_global(device void* dst, threadgroup const uint8_t* src, uint32_t n) {
  device uint8_t* d = (device uint8_t*)dst;
  for (uint32_t i = 0; i < n; i++) d[i] = src[i];
}

//
// The same decode as `runtime/weft_runtime.hpp`, for the other back end. The object a
// `cp.async.bulk.tensor` copy reads is 128 bytes and its layout is weft's own: NVIDIA's
// driver encodes it and does not publish the byte layout, so this side and the CPU side
// read one documented format, and the tag at offset 124 is what says an object is one of
// them (a blob from `cuTensorMapEncodeTiled` does not carry it and is not decoded).
//
//   off   0  u64  address of element (0, 0)
//   off   8  u8   rank, 1 to 5; then the kind, then the swizzle, then a spare byte
//   off  12  u32  element size in bytes: 1, 2, 4, 8 or 16
//   off  16  u32  extent[5]         the tensor's length along each dimension
//   off  40  u64  stride[5]         bytes between neighbours; [0] is one element
//   off 124  u32  tag               0x41544657, 'WFTA'
//
// A box (kind 0) is a rectangular tile: `box[5]` at 80 and `elementStride[5]` at 100.
// A column (kind 1) is what `.im2col` copies, `channelsPerPixel` at 80,
// `pixelsPerColumn` at 84, a pixel's `traversal[3]` at 88, the window's
// `lowerCorner[3]` at 100 and a filter step's `dilation[3]` at 112, and the swizzle
// byte applies to the threadgroup destination either way.
static inline uint32_t tma_word(const device uint8_t* d, uint32_t off) {
  return (uint32_t)d[off] | ((uint32_t)d[off + 1u] << 8) | ((uint32_t)d[off + 2u] << 16) |
         ((uint32_t)d[off + 3u] << 24);
}

static inline uint64_t tma_dword(const device uint8_t* d, uint32_t off) {
  return (uint64_t)tma_word(d, off) | ((uint64_t)tma_word(d, off + 4u) << 32);
}

static inline uint32_t tma_swizzled(uint32_t o, uint32_t mode) {
  if (mode != 32u && mode != 64u && mode != 128u) return o;
  const uint32_t shift = (mode == 128u) ? 7u : (mode == 64u) ? 6u : 5u;
  return o ^ (((o >> shift) & ((mode >> 4) - 1u)) << 4);
}

struct TmaBox {
  uint64_t base;
  uint32_t rank, kind, elem, total, swizzle;
  uint64_t stride[5];
  uint32_t extent[5];
  uint32_t box[5];
  uint32_t step[5];
  uint32_t channels, pixels;
  uint32_t traversal[3];
  uint32_t dilation[3];
  int32_t corner[3];
};

static inline bool tma_decode(const device uint8_t* d, thread TmaBox& m) {
  if (tma_word(d, 124u) != 0x41544657u) return false;
  const uint32_t head = tma_word(d, 8u);
  m.rank = head & 0xFFu;
  m.kind = (head >> 8) & 0xFFu;
  m.swizzle = (head >> 16) & 0xFFu;
  m.elem = tma_word(d, 12u);
  if (m.rank == 0u || m.rank > 5u || m.elem == 0u || m.elem > 16u) return false;
  m.base = tma_dword(d, 0u);
  m.total = 1u;
  m.stride[0] = m.elem;
  for (uint32_t i = 0; i < m.rank; i++) {
    m.extent[i] = tma_word(d, 16u + 4u * i);
    if (m.extent[i] == 0u) return false;
    if (i > 0u) m.stride[i] = tma_dword(d, 40u + 8u * i);
  }
  if (m.kind == 0u) {
    for (uint32_t i = 0; i < m.rank; i++) {
      m.box[i] = tma_word(d, 80u + 4u * i);
      m.step[i] = tma_word(d, 100u + 4u * i);
      if (m.box[i] == 0u) return false;
      m.total *= m.box[i];
    }
    return true;
  }
  if (m.kind == 1u) {
    m.channels = tma_word(d, 80u);
    m.pixels = tma_word(d, 84u);
    if (m.channels == 0u || m.pixels == 0u) return false;
    for (uint32_t i = 0; i < 3u; i++) {
      m.traversal[i] = tma_word(d, 88u + 4u * i);
      m.corner[i] = (int32_t)tma_word(d, 100u + 4u * i);
      m.dilation[i] = tma_word(d, 112u + 4u * i);
      if (m.traversal[i] == 0u) m.traversal[i] = 1u;
      if (m.dilation[i] == 0u) m.dilation[i] = 1u;
    }
    m.total = m.channels * m.pixels;
    return true;
  }
  return false;
}

static inline bool tma_element(thread const TmaBox& m, thread const int32_t* co, uint32_t k,
                               thread uint64_t& addr) {
  uint64_t off = 0;
  uint32_t rest = k;
  bool inside = true;
  for (uint32_t i = 0; i < m.rank; i++) {
    const uint32_t e = rest % m.box[i];
    rest /= m.box[i];
    const int64_t c = (int64_t)co[i] + (int64_t)e * (int64_t)(m.step[i] ? m.step[i] : 1u);
    if (c < 0 || c >= (int64_t)m.extent[i]) {
      inside = false;
      break;
    }
    off += (uint64_t)c * m.stride[i];
  }
  addr = m.base + off;
  return inside;
}

// threadgroup memory <- the tensor, reporting the box's bytes the way the CPU side does.
static inline uint32_t tma_load(threadgroup uint8_t* dst, const device void* map, int32_t c0,
                                int32_t c1, int32_t c2, int32_t c3, int32_t c4) {
  const device uint8_t* d = (const device uint8_t*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != 0u) return 0xFFFFFFFFu;
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  for (uint32_t k = 0; k < m.total; k++) {
    uint64_t addr = 0;
    const bool inside = tma_element(m, co, k, addr);
    const device uint8_t* s = (const device uint8_t*)(uintptr_t)addr;
    const uint32_t o = tma_swizzled(k * m.elem, m.swizzle);
    for (uint32_t b = 0; b < m.elem; b++) dst[o + b] = inside ? s[b] : (uint8_t)0;
  }
  return m.total * m.elem;
}

static inline void tma_store(const device void* map, int32_t c0, int32_t c1, int32_t c2,
                             int32_t c3, int32_t c4, threadgroup const uint8_t* src) {
  const device uint8_t* d = (const device uint8_t*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != 0u) return;
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  for (uint32_t k = 0; k < m.total; k++) {
    uint64_t addr = 0;
    if (!tma_element(m, co, k, addr)) continue;
    device uint8_t* t = (device uint8_t*)(uintptr_t)addr;
    const uint32_t o = tma_swizzled(k * m.elem, m.swizzle);
    for (uint32_t b = 0; b < m.elem; b++) t[b] = src[o + b];
  }
}

// Where pixel `p` of a column lands: the corner advanced by `traversal` in each spatial
// dimension, the first fastest, plus the filter step's `dilation`. False means the pixel
// fell outside the tensor, which is the edge a padded convolution is made of.
static inline bool tma_column_at(thread const TmaBox& m, thread const int32_t* co,
                                 thread const int32_t* of, thread const uint32_t* fit,
                                 uint32_t p, thread uint64_t& at) {
  const uint32_t spatial = m.rank - 2u;  // the batch is the last dimension
  const int64_t batch = co[spatial + 1u];
  if (batch < 0 || batch >= (int64_t)m.extent[spatial + 1u]) return false;
  at = (uint64_t)batch * m.stride[spatial + 1u];
  uint32_t q = p;
  for (uint32_t i = 0; i < spatial; i++) {
    const uint32_t step = q % fit[i];
    q /= fit[i];
    const int64_t at_i = (int64_t)co[i + 1u] * (int64_t)m.traversal[i] + m.corner[i] +
                         (int64_t)step * (int64_t)m.traversal[i] +
                         (int64_t)of[i] * (int64_t)m.dilation[i];
    if (at_i < 0 || at_i >= (int64_t)m.extent[i + 1u]) return false;
    at += (uint64_t)at_i * m.stride[i + 1u];
  }
  return true;
}

// The convolution form: the column a `.im2col` copy asks for. A pixel that lands outside
// the tensor is written as zero, which is what makes padding work.
static inline uint32_t tma_im2col(threadgroup uint8_t* dst, const device void* map,
                                  int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4,
                                  int32_t o0, int32_t o1, int32_t o2) {
  const device uint8_t* d = (const device uint8_t*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != 1u) return 0xFFFFFFFFu;
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  const int32_t of[3] = {o0, o1, o2};
  const uint32_t spatial = m.rank - 2u;
  uint32_t fit[3] = {1u, 1u, 1u};
  for (uint32_t i = 0; i < spatial; i++)
    fit[i] = (m.extent[i + 1u] + m.traversal[i] - 1u) / m.traversal[i];
  for (uint32_t p = 0; p < m.pixels; p++) {
    uint64_t at = 0;
    const bool inside = tma_column_at(m, co, of, fit, p, at);
    for (uint32_t e = 0; e < m.channels; e++) {
      const uint32_t o = tma_swizzled((p * m.channels + e) * m.elem, m.swizzle);
      const int64_t c = (int64_t)co[0] + (int64_t)e;
      if (!inside || c < 0 || c >= (int64_t)m.extent[0]) {
        for (uint32_t b = 0; b < m.elem; b++) dst[o + b] = 0;
        continue;
      }
      const device uint8_t* s =
          (const device uint8_t*)(uintptr_t)(m.base + at + (uint64_t)c * m.stride[0]);
      for (uint32_t b = 0; b < m.elem; b++) dst[o + b] = s[b];
    }
  }
  return m.channels * m.pixels * m.elem;
}

// The same column going back out: threadgroup memory is scattered into the window, and an
// element whose place fell outside the tensor writes nothing.
static inline void tma_im2col_store(const device void* map, int32_t c0, int32_t c1, int32_t c2,
                                    int32_t c3, int32_t c4, int32_t o0, int32_t o1, int32_t o2,
                                    threadgroup const uint8_t* src) {
  const device uint8_t* d = (const device uint8_t*)map;
  TmaBox m;
  if (!tma_decode(d, m) || m.kind != 1u) return;
  const int32_t co[5] = {c0, c1, c2, c3, c4};
  const int32_t of[3] = {o0, o1, o2};
  const uint32_t spatial = m.rank - 2u;
  uint32_t fit[3] = {1u, 1u, 1u};
  for (uint32_t i = 0; i < spatial; i++)
    fit[i] = (m.extent[i + 1u] + m.traversal[i] - 1u) / m.traversal[i];
  for (uint32_t p = 0; p < m.pixels; p++) {
    uint64_t at = 0;
    if (!tma_column_at(m, co, of, fit, p, at)) continue;
    for (uint32_t e = 0; e < m.channels; e++) {
      const int64_t c = (int64_t)co[0] + (int64_t)e;
      if (c < 0 || c >= (int64_t)m.extent[0]) continue;
      device uint8_t* t =
          (device uint8_t*)(uintptr_t)(m.base + at + (uint64_t)c * m.stride[0]);
      const uint32_t o = tma_swizzled((p * m.channels + e) * m.elem, m.swizzle);
      for (uint32_t b = 0; b < m.elem; b++) t[b] = src[o + b];
    }
  }
}

}  // namespace weft
)WEFMSL";

// The Metal side of the translated-kernel runtime, as three pieces joined once.
// The middle one is the software double, which is its own text because the CPU
// column compiles it too (see src/weft_f64.hpp).
static const std::string kWWeftMslJoined =
    std::string(kWWeftMslHead) + kWWeftF64 + kWWeftMslTail;
static const char* kWWeftMslPreamble = kWWeftMslJoined.c_str();
