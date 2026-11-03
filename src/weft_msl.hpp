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
//     have to lock and keeps the shape that needs no lock — a reduction, which is
//     two 32-bit atomics.
//
// Everything here is `static inline` in one translation unit, so the compiler
// inlines the helpers into the kernel and constant-folds the mode switches.
static const char* kWWeftMslPreamble = R"WEFMSL(
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

// ---- 64-bit floating point, over the integers ----
//
// Apple GPUs have no fp64 and Metal rejects the type outright, so a kernel carrying a
// `double` cannot be compiled at all — that is the language, not a missing lowering.
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
namespace weft_fp64 {

constant uint64_t kSign = 0x8000000000000000ull;
constant uint64_t kExpMask = 0x7FF0000000000000ull;
constant uint64_t kFracMask = 0x000FFFFFFFFFFFFFull;
constant uint64_t kHidden = 0x0010000000000000ull;
// The leading bit of the working form. It is not `kHidden`: that is bit 52, the
// significand's leading bit in the *encoding*, and the working form carries three
// spare bits below it, so its leading bit sits at 55. Normalizing on the wrong one
// is an infinite loop, because a value whose low bits are zero can never set bit 52
// by shifting left.
constant uint64_t kLead = 0x0080000000000000ull;
constant uint64_t kInf = 0x7FF0000000000000ull;
constant uint64_t kQNaN = 0x7FF8000000000000ull;
// The leading-zero count of a 64-bit value, in 32-bit halves. `__builtin_clzll` is
// what this host's compiler dies on — at pipeline creation, with
// XPC_ERROR_CONNECTION_INTERRUPTED and nothing else — and the 32-bit builtin is
// fine, so the wide one is spelled out of the narrow one. Measured with
// tools/msl_bisect.py: the two helpers that used it were the only ones of sixteen
// that failed to build.
static inline uint32_t clz64(uint64_t v) {
  const uint32_t hi = (uint32_t)(v >> 32);
  if (hi != 0u) return (uint32_t)__builtin_clz(hi);
  return 32u + (uint32_t)__builtin_clz((uint32_t)(v & 0xFFFFFFFFu));
}

constant uint64_t kOne = 0x3FF0000000000000ull;
constant uint64_t kNegOne = 0xBFF0000000000000ull;
constant uint64_t kHalf = 0x3FE0000000000000ull;

static inline bool is_nan(uint64_t b) {
  return (b & kExpMask) == kExpMask && (b & kFracMask) != 0ull;
}
static inline bool is_inf(uint64_t b) { return (b & ~kSign) == kInf; }
static inline bool is_zero(uint64_t b) { return (b & ~kSign) == 0ull; }
static inline uint64_t neg(uint64_t b) { return b ^ kSign; }
static inline uint64_t abs_bits(uint64_t b) { return b & ~kSign; }

// The working form. A subnormal is normalized here — it has fewer than 53
// significant bits, so shifting it up loses nothing and every operation below can
// then be written once.
static inline void unpack(uint64_t b, thread int32_t& e, thread uint64_t& sig) {
  const int32_t be = (int32_t)((b >> 52) & 0x7FF);
  const uint64_t f = b & kFracMask;
  if (be == 0) {
    e = -1022;
    sig = f << 3;
    if (sig == 0ull) return;
    while ((sig & kLead) == 0ull) { sig <<= 1; e -= 1; }
  } else {
    e = be - 1023;
    sig = (f | kHidden) << 3;
  }
}

// Round the working form into a pattern. A result below the normal range has a
// quantum of 2^-1074, so the significand is shifted down before the same three-bit
// rounding applies; that shift is the whole difference between the two cases.
static inline uint64_t round_pack(uint32_t sign, int32_t e, uint64_t sig) {
  const uint64_t sgn = sign ? kSign : 0ull;
  if (sig == 0ull) return sgn;
  const bool sub = e < -1022;
  if (sub) {
    const int32_t k = -1022 - e;
    if (k >= 64) {
      sig = 1u;  // every bit is below the last place, so the value is tiny and sticky
    } else {
      const uint64_t lost = sig & ((1ull << k) - 1ull);
      sig = (sig >> k) | ((lost != 0ull) ? 1u : 0u);
    }
  }
  const uint64_t s = sig & 1ull;
  const uint64_t r = (sig >> 1) & 1ull;
  const uint64_t g = (sig >> 2) & 1ull;
  sig >>= 3;
  if (g && (r || s || (sig & 1ull))) sig += 1ull;  // nearest, ties to even
  if (sub) return sgn | sig;  // fraction only; a carry out of bit 52 is the smallest
                              // normal, which those same bits already spell
  int32_t be = e + 1023;
  if (sig == (1ull << 53)) { sig >>= 1; be += 1; }
  if (be >= 2047) return sgn | kInf;
  return sgn | ((uint64_t)be << 52) | (sig & kFracMask);
}

static inline uint64_t add(uint64_t a, uint64_t b) {
  if (is_nan(a) || is_nan(b)) return kQNaN;
  const uint32_t sa = (uint32_t)(a >> 63), sb = (uint32_t)(b >> 63);
  const uint32_t ea = (uint32_t)((a >> 52) & 0x7FF), eb = (uint32_t)((b >> 52) & 0x7FF);
  if (ea == 0x7FF) return (eb == 0x7FF && sa != sb) ? kQNaN : a;  // inf + -inf
  if (eb == 0x7FF) return b;
  if (is_zero(a) && is_zero(b)) return (sa == sb) ? a : 0ull;    // (+0) + (-0) = +0
  if (is_zero(a)) return b;
  if (is_zero(b)) return a;
  int32_t ua, ub;
  uint64_t Ma, Mb;
  unpack(a, ua, Ma);
  unpack(b, ub, Mb);
  // Order by magnitude, so the only shifting is the smaller operand coming down.
  const bool a_bigger = (ua > ub) || (ua == ub && Ma >= Mb);
  const uint32_t sign = a_bigger ? sa : sb;
  uint64_t big = a_bigger ? Ma : Mb;
  uint64_t small = a_bigger ? Mb : Ma;
  int32_t e = a_bigger ? ua : ub;
  const int32_t d = a_bigger ? (ua - ub) : (ub - ua);
  if (d > 0) {
    if (d >= 64) {
      small = 1u;  // nothing of it survives except the fact that it was not zero
    } else {
      const uint64_t lost = small & ((1ull << d) - 1ull);
      small = (small >> d) | ((lost != 0ull) ? 1u : 0u);
    }
  }
  uint64_t sum;
  uint32_t out_sign = sign;
  if (sa == sb) {
    sum = big + small;
  } else if (big > small) {
    sum = big - small;
  } else if (small > big) {
    sum = small - big;
    out_sign = a_bigger ? sb : sa;  // the larger magnitude was the negative one
  } else {
    return 0ull;  // equal magnitudes, opposite signs: +0 in round-to-nearest
  }
  if (sum >= (1ull << 56)) {
    sum = (sum >> 1) | (sum & 1ull);  // keep the bit that leaves bit 0 as sticky
    e += 1;
  }
  while ((sum & kLead) == 0ull) { sum <<= 1; e -= 1; }
  return round_pack(out_sign, e, sum);
}

static inline uint64_t sub(uint64_t a, uint64_t b) { return add(a, neg(b)); }

static inline uint64_t mul(uint64_t a, uint64_t b) {
  if (is_nan(a) || is_nan(b)) return kQNaN;
  const uint32_t sa = (uint32_t)(a >> 63), sb = (uint32_t)(b >> 63);
  const uint32_t sign = sa ^ sb;
  const uint32_t ea = (uint32_t)((a >> 52) & 0x7FF), eb = (uint32_t)((b >> 52) & 0x7FF);
  const bool za = is_zero(a), zb = is_zero(b);
  if (ea == 0x7FF || eb == 0x7FF) {
    if (za || zb) return kQNaN;  // an infinity times a zero
    return (sign ? kSign : 0ull) | kInf;
  }
  if (za || zb) return sign ? kSign : 0ull;
  int32_t ua, ub;
  uint64_t ma, mb;
  unpack(a, ua, ma);
  unpack(b, ub, mb);
  // The product of two 56-bit significands needs 112 bits, so it is built from
  // 32-bit halves, and the top of it is shifted down with a sticky bit.
  const uint64_t p0 = (ma & 0xFFFFFFFFull) * (mb & 0xFFFFFFFFull);
  const uint64_t p1 = (ma >> 32) * (mb & 0xFFFFFFFFull);
  const uint64_t p2 = (ma & 0xFFFFFFFFull) * (mb >> 32);
  const uint64_t p3 = (ma >> 32) * (mb >> 32);
  const uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p2 & 0xFFFFFFFFull);
  const uint64_t lo = (mid << 32) | (p0 & 0xFFFFFFFFull);
  const uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
  const int32_t lead = hi ? (127 - (int32_t)clz64(hi)) : (63 - (int32_t)clz64(lo));
  const int32_t sh = lead - 55;  // at most 57 for a product this wide
  uint64_t sig = (hi << (64 - sh)) | (lo >> sh);
  const uint64_t lost = lo & ((1ull << sh) - 1ull);
  sig |= (lost != 0ull) ? 1ull : 0ull;
  // value(a) = m_a * 2^(ua-52) and P = (m_a<<3) * (m_b<<3), so
  // value(a)*value(b) = P * 2^(ua+ub-110) = sig * 2^(sh+ua+ub-110), and the working
  // form is sig * 2^(e-55): e = sh + ua + ub - 55.
  return round_pack(sign, sh + ua + ub - 55, sig);
}

static inline uint64_t div(uint64_t a, uint64_t b) {
  if (is_nan(a) || is_nan(b)) return kQNaN;
  const uint32_t sa = (uint32_t)(a >> 63), sb = (uint32_t)(b >> 63);
  const uint32_t sign = sa ^ sb;
  const uint32_t ea = (uint32_t)((a >> 52) & 0x7FF), eb = (uint32_t)((b >> 52) & 0x7FF);
  if (ea == 0x7FF) return (eb == 0x7FF) ? kQNaN : ((sign ? kSign : 0ull) | kInf);
  if (eb == 0x7FF) return sign ? kSign : 0ull;  // a finite value over an infinity
  if (is_zero(b)) return is_zero(a) ? kQNaN : ((sign ? kSign : 0ull) | kInf);
  if (is_zero(a)) return sign ? kSign : 0ull;
  int32_t ua, ub;
  uint64_t ma, mb;
  unpack(a, ua, ma);
  unpack(b, ub, mb);
  // Long division of the significands: 55 quotient bits, and whatever is left over
  // is below the last of them, which is exactly the sticky bit.
  uint64_t q = 0ull, r = ma;
  if (r >= mb) { r -= mb; q = 1ull; }  // the quotient is at least one
  for (int32_t i = 0; i < 54; i++) {
    q <<= 1;
    r <<= 1;
    if (r >= mb) { r -= mb; q |= 1ull; }
  }
  // The quotient is in [2^54, 2^55) when the dividend's significand is the larger
  // and [2^53, 2^54) when it is not, so the leading bit belongs at 55 — and the
  // sticky bit goes in only after that shift, or it would move with it.
  uint64_t sig = q << 1;
  int32_t e = ua - ub;
  while ((sig & kLead) == 0ull) { sig <<= 1; e -= 1; }
  if (r != 0ull) sig |= 1ull;
  return round_pack(sign, e, sig);
}

static inline bool eq(uint64_t a, uint64_t b) {
  if (is_nan(a) || is_nan(b)) return false;
  if (is_zero(a) && is_zero(b)) return true;  // +0 and -0 are equal
  return a == b;                             // with no NaNs, equal bits are equal values
}
static inline bool ne(uint64_t a, uint64_t b) { return !eq(a, b); }
static inline bool lt(uint64_t a, uint64_t b) {
  if (is_nan(a) || is_nan(b)) return false;
  const uint32_t sa = (uint32_t)(a >> 63), sb = (uint32_t)(b >> 63);
  if (is_zero(a) && is_zero(b)) return false;
  if (sa != sb) return sa != 0u;  // a negative value is below a positive one
  const uint64_t ma = a & ~kSign, mb = b & ~kSign;
  return sa ? (ma > mb) : (ma < mb);
}
static inline bool le(uint64_t a, uint64_t b) { return lt(a, b) || eq(a, b); }
static inline bool gt(uint64_t a, uint64_t b) { return lt(b, a); }
static inline bool ge(uint64_t a, uint64_t b) { return le(b, a); }

// ---- fused multiply-add ----
//
// `a * b + c`, rounded once. A product of two 53-bit significands needs 106 bits and
// the addition can cancel almost all of that away, which is exactly why a multiply and
// then an add is a different answer: no 56-bit working form can hold the intermediate.
// The three values are placed at one scale in a 192-bit accumulator instead, and the
// rounding happens once, at the end.
struct decon {
  uint64_t sig;   // value = sig * 2^exp, with bit 52 set for a non-zero value
  int32_t exp;
};
struct wide {
  uint64_t h, m, l;   // (h:m:l) * 2^exp, the exponent being the caller's business
};

static inline int32_t bitlen64(uint64_t v) { return v == 0ull ? 0 : 64 - (int32_t)clz64(v); }

// A value as an integer significand with its leading bit at 52, so two of them fit a
// product exactly. A subnormal is shifted up until its leading bit is there, which
// loses nothing: it has fewer than 53 significant bits to begin with.
static inline decon decompose(uint64_t b) {
  decon d;
  const uint32_t be = (uint32_t)((b >> 52) & 0x7FF);
  d.sig = b & kFracMask;
  if (be == 0u) {
    d.exp = -1074;
    if (d.sig != 0ull)
      while ((d.sig & kHidden) == 0ull) { d.sig <<= 1; d.exp -= 1; }
  } else {
    d.sig |= kHidden;
    d.exp = (int32_t)be - 1075;
  }
  return d;
}

static inline bool wide_zero(wide w) { return (w.h | w.m | w.l) == 0ull; }

static inline int wide_cmp(wide a, wide b) {
  if (a.h != b.h) return a.h > b.h ? 1 : -1;
  if (a.m != b.m) return a.m > b.m ? 1 : -1;
  if (a.l != b.l) return a.l > b.l ? 1 : -1;
  return 0;
}

static inline wide wide_add(wide a, wide b) {
  wide r;
  const uint64_t l = a.l + b.l;
  const uint64_t carry = (l < a.l) ? 1ull : 0ull;
  const uint64_t m0 = a.m + b.m;
  uint64_t c2 = (m0 < a.m) ? 1ull : 0ull;
  const uint64_t m = m0 + carry;
  if (m < m0) c2 += 1ull;
  r.l = l;
  r.m = m;
  r.h = a.h + b.h + c2;
  return r;
}

static inline wide wide_sub(wide a, wide b) {  // a >= b
  wide r;
  const uint64_t br_l = (a.l < b.l) ? 1ull : 0ull;
  const uint64_t br_m = ((a.m < b.m) || (br_l != 0ull && a.m == b.m)) ? 1ull : 0ull;
  r.l = a.l - b.l;
  r.m = a.m - b.m - br_l;
  r.h = a.h - b.h - br_m;
  return r;
}

// Shift right, folding every dropped bit into bit 0. A sticky bit only has to say
// that something was there, which is all a rounding decision needs from it.
static inline wide wide_shr(wide v, int32_t k) {
  if (k <= 0) return v;
  if (k >= 192) return wide{0ull, 0ull, ((v.h | v.m | v.l) != 0ull) ? 1ull : 0ull};
  const int32_t wd = k >> 6, b = k & 63;
  const uint64_t limb[3] = {v.l, v.m, v.h};
  uint64_t out[3] = {0ull, 0ull, 0ull};
  uint64_t sticky = 0ull;
  for (int32_t i = 0; i < wd; i++) sticky |= limb[i];
  for (int32_t i = 0; i < 3; i++) {
    const uint64_t cur = (i + wd < 3) ? limb[i + wd] : 0ull;
    const uint64_t up = (i + wd + 1 < 3) ? limb[i + wd + 1] : 0ull;
    out[i] = (b == 0) ? cur : ((cur >> b) | (up << (64 - b)));
  }
  if (b != 0 && wd < 3) sticky |= limb[wd] & ((1ull << b) - 1ull);
  wide r;
  r.l = out[0] | ((sticky != 0ull) ? 1ull : 0ull);
  r.m = out[1];
  r.h = out[2];
  return r;
}

// Lay the 128-bit `hi:lo` at the accumulator's scale. A negative shift is a value
// entirely below the window, where only its existence survives.
static inline wide place(uint64_t hi, uint64_t lo, int32_t sh) {
  if (sh < 0) {
    wide v;
    v.h = 0ull;
    v.m = hi;
    v.l = lo;
    return wide_shr(v, -sh);
  }
  const int32_t wd = sh >> 6, b = sh & 63;
  uint64_t p0 = lo, p1 = hi, p2 = 0ull;
  if (b != 0) {
    p2 = hi >> (64 - b);
    p1 = (hi << b) | (lo >> (64 - b));
    p0 = lo << b;
  }
  wide r;
  if (wd == 0) { r.h = p2; r.m = p1; r.l = p0; }
  else if (wd == 1) { r.h = p1; r.m = p0; r.l = 0ull; }
  else if (wd == 2) { r.h = p0; r.m = 0ull; r.l = 0ull; }
  else { r.h = 0ull; r.m = 0ull; r.l = 0ull; }
  return r;
}

static inline int32_t msb192(wide w) {
  if (w.h != 0ull) return 191 - (int32_t)clz64(w.h);
  if (w.m != 0ull) return 127 - (int32_t)clz64(w.m);
  return 63 - (int32_t)clz64(w.l);
}

static inline uint64_t fma(uint64_t a, uint64_t b, uint64_t c) {
  if (is_nan(a) || is_nan(b) || is_nan(c)) return kQNaN;
  const uint32_t sa = (uint32_t)(a >> 63), sb = (uint32_t)(b >> 63), sc = (uint32_t)(c >> 63);
  const uint32_t sp = sa ^ sb;
  const bool za = is_zero(a), zb = is_zero(b), zc = is_zero(c);
  if (is_inf(a) || is_inf(b)) {
    if (za || zb) return kQNaN;                 // an infinity times a zero
    if (is_inf(c) && sc != sp) return kQNaN;    // inf - inf
    return (sp != 0u ? kSign : 0ull) | kInf;
  }
  if (is_inf(c)) return c;
  if (za || zb)
    return zc ? ((sc != 0u && sp != 0u) ? kSign : 0ull) : c;  // a zero product leaves c
  if (zc) return mul(a, b);
  const decon da = decompose(a), db = decompose(b), dc = decompose(c);
  // The product, exactly: 53 by 53 bits, built from 32-bit halves.
  const uint64_t p0 = (da.sig & 0xFFFFFFFFull) * (db.sig & 0xFFFFFFFFull);
  const uint64_t p1 = (da.sig >> 32) * (db.sig & 0xFFFFFFFFull);
  const uint64_t p2 = (da.sig & 0xFFFFFFFFull) * (db.sig >> 32);
  const uint64_t p3 = (da.sig >> 32) * (db.sig >> 32);
  const uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p2 & 0xFFFFFFFFull);
  const uint64_t plo = (mid << 32) | (p0 & 0xFFFFFFFFull);
  const uint64_t phi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
  const int32_t pexp = da.exp + db.exp;
  // One scale for both: the larger operand's top bit lands at bit 190, which leaves
  // 84 bits below it — more than the 106-bit product spans, so nothing is lost before
  // the sum and the rounding happens once, at the end.
  const int32_t tbP = pexp + (phi != 0ull ? 64 + bitlen64(phi) : bitlen64(plo)) - 1;
  const int32_t tbC = dc.exp + bitlen64(dc.sig) - 1;
  const int32_t et = tbP > tbC ? tbP : tbC;
  const int32_t e0 = et - 190;
  const wide P = place(phi, plo, pexp - e0);
  const wide C = place(0ull, dc.sig, dc.exp - e0);
  wide r;
  uint32_t sign;
  if (sp == sc) {
    r = wide_add(P, C);
    sign = sp;
  } else {
    const int cmp = wide_cmp(P, C);
    if (cmp == 0) return 0ull;                 // exact cancellation is +0
    if (cmp > 0) { r = wide_sub(P, C); sign = sp; }
    else { r = wide_sub(C, P); sign = sc; }
  }
  if (wide_zero(r)) return 0ull;
  const int32_t mpos = msb192(r);
  // Into the form `round_pack` takes: the top bit at 55 with three bits of guard,
  // round and sticky under it, and the exponent of that top bit.
  const uint64_t sig = (mpos >= 55) ? wide_shr(r, mpos - 55).l : (r.l << (55 - mpos));
  return round_pack(sign, e0 + mpos, sig);
}

// ---- square root ----
//
// `sqrt` is the one function in the transcendental library that IEEE-754 pins
// exactly: the result is the correctly rounded root, so a software one is not an
// approximation of the host's — it is the same number, and the two can be compared
// bit for bit. Everything else in that library rounds through whatever the vendor
// chose and is therefore not carried here.
//
// The root is built from the significand as an integer: the input is written
// `sig * 2^e` with `sig` in [2^52, 2^53) and `e` even, `sig << 54` is the radicand,
// and `floor(sqrt(that))` is 54 bits of the answer. The restoring loop is the one
// from pencil-and-paper long division, two radicand bits a step — its remainder is
// below `2q + 1`, so everything after the first line fits in 64 bits.
static inline uint64_t bit128(uint64_t hi, uint64_t lo, int b) {
  return b >= 64 ? ((hi >> (b - 64)) & 1ull) : ((lo >> b) & 1ull);
}

// floor(sqrt(hi:lo)); `exact` says the square was a whole number.
static inline uint64_t isqrt128(uint64_t hi, uint64_t lo, thread bool& exact) {
  uint64_t rem = 0ull, q = 0ull;
  for (int i = 53; i >= 0; i--) {
    rem = (rem << 2) | ((bit128(hi, lo, 2 * i + 1) << 1) | bit128(hi, lo, 2 * i));
    const uint64_t trial = (q << 2) | 1ull;
    if (rem >= trial) {
      rem -= trial;
      q = (q << 1) | 1ull;
    } else {
      q <<= 1;
    }
  }
  exact = (rem == 0ull);
  return q;
}

static inline uint64_t sqrt_bits(uint64_t b) {
  if (is_nan(b)) return kQNaN;
  if ((b >> 63) != 0ull) return is_zero(b) ? b : kQNaN;  // -0 is -0; a negative is a NaN
  if (is_zero(b) || is_inf(b)) return b;
  decon d = decompose(b);
  int32_t e = d.exp;
  // An odd exponent moves one bit into the significand, which is exact and makes the
  // halving below a shift rather than a rounding.
  if ((e & 1) != 0) {
    d.sig <<= 1;
    e -= 1;
  }
  bool exact = false;
  const uint64_t q = isqrt128(d.sig >> 10, (d.sig & 0x3FFull) << 54, exact);
  // sqrt(value) = (q + frac) * 2^(e/2 - 27) with q in [2^53, 2^54), so the leading bit
  // of the significand is the exponent below and `q >> 1` is the 53 bits that follow.
  int32_t k = e / 2 + 26;
  uint64_t s = q >> 1;
  const uint64_t guard = q & 1ull;
  if (guard != 0ull && ((exact ? 0ull : 1ull) != 0ull || (s & 1ull) != 0ull)) s += 1ull;
  if (s == (1ull << 53)) {  // the rounding carried out of the field
    s >>= 1;
    k += 1;
  }
  const int32_t biased = k + 1023;
  // No finite double roots below the smallest normal — sqrt(2^-1074) is 2^-537 — so
  // the subnormal branch has nothing to carry.
  if (biased >= 2047) return kInf;
  if (biased <= 0) return kInf;
  return ((uint64_t)biased << 52) | (s & kFracMask);
}

// minNum and maxNum: a NaN operand means the other one, and two NaNs mean a NaN.
// Of two zeros the negative one is the smaller, which is the order IEEE gives them.
static inline uint64_t min_bits(uint64_t a, uint64_t b) {
  if (is_nan(a)) return is_nan(b) ? kQNaN : b;
  if (is_nan(b)) return a;
  if (is_zero(a) && is_zero(b)) return (a & kSign) ? a : b;
  return lt(a, b) ? a : b;
}
static inline uint64_t max_bits(uint64_t a, uint64_t b) {
  if (is_nan(a)) return is_nan(b) ? kQNaN : b;
  if (is_nan(b)) return a;
  if (is_zero(a) && is_zero(b)) return (a & kSign) ? b : a;
  return gt(a, b) ? a : b;
}

static inline uint64_t from_u64(uint64_t m) {
  if (m == 0ull) return 0ull;
  const int32_t lead = 63 - (int32_t)clz64(m);
  uint64_t sig;
  if (lead <= 55) {
    sig = m << (55 - lead);
  } else {
    const int32_t sh = lead - 55;
    const uint64_t lost = m & ((1ull << sh) - 1ull);
    sig = (m >> sh) | ((lost != 0ull) ? 1ull : 0ull);
  }
  return round_pack(0u, lead, sig);
}
static inline uint64_t from_i64(int64_t x) {
  if (x == 0) return 0ull;
  const uint32_t sign = x < 0 ? 1u : 0u;
  const uint64_t m = x < 0 ? ((uint64_t)(-(x + 1)) + 1ull) : (uint64_t)x;
  const int32_t lead = 63 - (int32_t)clz64(m);
  uint64_t sig;
  if (lead <= 55) {
    sig = m << (55 - lead);
  } else {
    const int32_t sh = lead - 55;
    const uint64_t lost = m & ((1ull << sh) - 1ull);
    sig = (m >> sh) | ((lost != 0ull) ? 1ull : 0ull);
  }
  return round_pack(sign, lead, sig);
}
// Toward zero, which is what a cast does. Out-of-range and NaN are undefined in
// SPIR-V; this saturates rather than trapping, so two targets cannot disagree by
// accident of the host's conversion.
static inline uint64_t to_u64(uint64_t b) {
  if (is_nan(b)) return 0ull;
  const uint32_t sign = (uint32_t)(b >> 63);
  if (is_inf(b)) return sign ? 0ull : ~0ull;
  if (is_zero(b)) return 0ull;
  int32_t e;
  uint64_t sig;
  unpack(b, e, sig);
  if (e < 0) return 0ull;
  uint64_t m;
  if (e >= 55) {
    if (e >= 119) return sign ? 0ull : ~0ull;
    m = sig << (e - 55);
  } else {
    m = sig >> (55 - e);
  }
  return sign ? (0ull - m) : m;
}
static inline int64_t to_i64(uint64_t b) {
  if (is_nan(b)) return 0;
  const bool negative = (b >> 63) != 0ull;
  const uint64_t m = to_u64(b & ~kSign);
  if (m > 0x8000000000000000ull) return negative ? (-0x7FFFFFFFFFFFFFFF - 1) : 0x7FFFFFFFFFFFFFFF;
  if (m == 0x8000000000000000ull) return negative ? (-0x7FFFFFFFFFFFFFFF - 1) : 0x7FFFFFFFFFFFFFFF;
  return negative ? -(int64_t)m : (int64_t)m;
}

static inline uint64_t from_f32(uint32_t fb) {
  const uint32_t sign = (fb >> 31) & 1u;
  const uint32_t be = (fb >> 23) & 0xFFu;
  const uint32_t f = fb & 0x7FFFFFu;
  if (be == 0xFFu) return ((uint64_t)sign << 63) | kInf | ((uint64_t)f << 29);
  if (be == 0u) {
    if (f == 0u) return (uint64_t)sign << 63;
    int32_t e = -126;
    uint32_t m = f;
    while ((m & 0x800000u) == 0u) { m <<= 1; e -= 1; }
    return round_pack(sign, e, (uint64_t)m << 32);
  }
  return round_pack(sign, (int32_t)be - 127, ((uint64_t)(f | 0x800000u)) << 32);
}

// Round into binary32: same working form, twenty-nine fewer significand bits, and
// binary32's own bias and subnormal floor.
static inline uint32_t round_pack32(uint32_t sign, int32_t e, uint64_t sig) {
  const uint32_t sgn = sign << 31;
  if (sig == 0ull) return sgn;
  const bool sub = e < -126;
  if (sub) {
    const int32_t k = -126 - e;
    if (k >= 64) {
      sig = 1u;
    } else {
      const uint64_t lost = sig & ((1ull << k) - 1ull);
      sig = (sig >> k) | ((lost != 0ull) ? 1u : 0u);
    }
  }
  const uint64_t lost2 = sig & ((1ull << 29) - 1ull);
  sig = (sig >> 29) | ((lost2 != 0ull) ? 1u : 0ull);
  const uint64_t s = sig & 1ull, r = (sig >> 1) & 1ull, g = (sig >> 2) & 1ull;
  sig >>= 3;
  if (g && (r || s || (sig & 1ull))) sig += 1ull;
  if (sub) {
    if (sig >= (1ull << 23)) return sgn | 0x00800000u;  // the smallest normal
    return sgn | (uint32_t)sig;
  }
  int32_t be = e + 127;
  if (sig == (1ull << 24)) { sig >>= 1; be += 1; }
  if (be >= 255) return sgn | 0x7F800000u;
  return sgn | ((uint32_t)be << 23) | ((uint32_t)sig & 0x7FFFFFu);
}

static inline uint32_t to_f32(uint64_t b) {
  const uint32_t sign = (uint32_t)(b >> 63);
  if (is_nan(b)) return (sign << 31) | 0x7FC00000u;
  if (is_inf(b)) return (sign << 31) | 0x7F800000u;
  if (is_zero(b)) return sign << 31;
  int32_t e;
  uint64_t sig;
  unpack(b, e, sig);
  return round_pack32(sign, e, sig);
}

// Toward zero: the fraction bits below the binary point go. Everything from a value
// already integral to an infinity is its own answer.
static inline uint64_t trunc_bits(uint64_t b) {
  if (is_nan(b) || is_inf(b)) return b;
  const uint32_t be = (uint32_t)((b >> 52) & 0x7FF);
  if (be < 1023u) return b & kSign;  // |x| < 1 has no integer part
  const int32_t e = (int32_t)be - 1023;
  if (e >= 52) return b;
  return b & ~((1ull << (52 - e)) - 1ull);
}
static inline uint64_t floor_bits(uint64_t b) {
  const uint64_t t = trunc_bits(b);
  if (is_nan(b) || is_inf(b) || t == b) return t;
  return ((b >> 63) != 0ull) ? add(t, kNegOne) : t;
}
static inline uint64_t ceil_bits(uint64_t b) {
  const uint64_t t = trunc_bits(b);
  if (is_nan(b) || is_inf(b) || t == b) return t;
  return ((b >> 63) == 0ull) ? add(t, kOne) : t;
}
// To the nearest integer, ties to even — what a conversion to an integer with
// `.rn` means, and what a cast to an integer does on the CPU. `round_bits` below is
// GLSL's Round, which breaks a tie away from zero, so the two are not
// interchangeable: 2.5 is 2 here and 3 there.
static inline uint64_t rne_bits(uint64_t b) {
  if (is_nan(b) || is_inf(b)) return b;
  const uint64_t t = trunc_bits(b);
  if (t == b) return t;
  const uint64_t frac = sub(abs_bits(b), abs_bits(t));
  bool up = gt(frac, kHalf);
  if (!up && eq(frac, kHalf)) up = ((to_i64(t) & 1) != 0);
  if (!up) return t;
  return add(t, ((b >> 63) != 0ull) ? kNegOne : kOne);
}

// GLSL's Round: to the nearest integer, ties away from zero.
static inline uint64_t round_bits(uint64_t b) {
  if (is_nan(b) || is_inf(b)) return b;
  const uint64_t t = trunc_bits(b);
  const uint64_t frac = sub(abs_bits(b), abs_bits(t));
  if (ge(frac, kHalf)) return add(t, ((b >> 63) != 0ull) ? kNegOne : kOne);
  return t;
}

}  // namespace weft_fp64
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
  // the host's — this is the same arithmetic written out.
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
static inline float weft_e5m2(uint32_t b, int i) {
  const uint32_t v = (b >> (8 * i)) & 0xFFu;
  const uint32_t mag = v & 3u, E = (v >> 2) & 0x1Fu;
  if (E == 31u) return (mag == 0u) ? ((v & 0x80u) ? -INFINITY : INFINITY) : NAN;
  const float d = (E == 0) ? (float)mag / 65536.0f
                           : (float)(4u + mag) * as_type<float>((E + 110u) << 23);
  return (float)((v & 0x80u) ? -d : d);
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
static inline float weft_tf32(float x) {
  uint32_t u = as_type<uint32_t>(x);
  if ((u & 0x7F800000u) == 0x7F800000u) return x;
  u += 0x1000u + ((u >> 13) & 1u);
  u &= 0xFFFFE000u;
  return as_type<float>(u);
}
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
static inline uint32_t weft_prmt(uint32_t a, uint32_t b, uint32_t c) {
  const uint64_t src = (uint64_t)a | ((uint64_t)b << 32);
  uint32_t out = 0;
  for (int i = 0; i < 4; i++) {
    const uint32_t sel = (c >> (4 * i)) & 0xFu;
    const uint32_t byte = (uint32_t)((src >> ((sel & 7u) * 8u)) & 0xFFu);
    out |= ((sel & 8u) ? ((byte & 0x80u) ? 0xFFu : 0u) : byte) << (8 * i);
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
// template covers every shape SPIR-V allows — two to four columns of two to four
// rows — because one shader multiplies a 4x4 by a 4-vector and a 3x4 skinning matrix
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
// A loop rather than a builtin — `ctz` is spelled differently on the two targets and
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

// ---- images ----
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
// has the same shape as the plain form — one channel, a texel is channel 0 — so
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
// A cube is filterable only — there is no texel to address without a face — so
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
// is nothing — the same answer the CPU runtime gives for the same coordinate.
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

// A cube sample: the direction picks the face and the place on it, which is the
// hardware's own rule and the one the CPU runtime models.
template <class T>
static inline T tex_sample(thread const SampledCubeImage& si, float dx, float dy, float dz) {
  return weft_bitcast<T>(si.tex.t.sample(si.s, float3(dx, dy, dz)).x);
}

// A texel is channel 0 of whatever the module wrote — the format is one channel,
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
// and a footprint that leaves the binding answers with nothing — the CPU runtime's
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
static inline void tex_gather4_cube(thread const SampledCubeImage& si, float s, float v,
                                    uint32_t face, thread float* out) {
  const int x0 = (int)floor(((s + 1.0f) * 0.5f) * (float)si.tex.width - 0.5f);
  const int y0 = (int)floor(((v + 1.0f) * 0.5f) * (float)si.tex.height - 0.5f);
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

// ---- atomics ----
// The hardware's own fetch forms, not a retry loop, and that is not a
// micro-optimisation. A compare-exchange retry costs every contender one line
// transfer per success, so a counter touched by n threads costs about n^2 of them:
// at 4,096 lanes that is 16 million and at 65,536 it is 4 billion, which does not
// finish — it reads as a hang, and it did: the reduction path below (no lock, no
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
// min and max have instructions here too, for both integer types — the note that
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
// could still grind — a hot word costs the number of contenders per success. Nothing
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

// ---- 64-bit reductions ----
//
// A `red` produces no result, so it does not need the 64 bits read as one value —
// it needs the pair to end up holding the fold of everything applied to it. For
// an operation that is order-free that is two 32-bit atomics:
//
//   add/sub  the low half goes first and its carry (or borrow) is applied to the
//            high half. Every thread that wraps the low word adds exactly one to
//            the high word, so the sum is right whatever order the threads land
//            in — and the hardware does the low half atomically, so nothing is
//            lost in between.
//   and/or/xor  each half is folded on its own. The two halves are independent,
//            so the pair ends up holding the fold without either half needing to
//            know about the other.
//
// What this cannot do is hand back the old pair as one value, or compare one.
// That needs the pair read as one value, which no instruction here does — so it
// is a lock over the two halves instead.

// ---- 64-bit read-modify-write ----
//
// There is no 64-bit atomic on this host. `atomic_ulong` is declared, but every
// entry point that could read, write, exchange or compare one is constrained to
// 32-bit types, and the two that are not (`atomic_max_explicit` /
// `atomic_min_explicit`) compile only to be refused by the device at pipeline
// creation — tools/atom64_probe.mm asks for each of those answers separately.
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
// constrains all of them to it. Ordering is not the obstacle — a device-scope
// `atomic_thread_fence` exists and a relaxed store from one threadgroup is seen by
// a waiter in another (tools/lock_probe.mm measures both). The obstacle is
// progress: a lane that wins this section cannot run it until the other lanes in
// its warp leave the spin loop, and they cannot leave until it releases, because a
// warp is one program counter. A threadgroup of one thread does not dodge that
// either — such threadgroups are co-scheduled into one warp and deadlock across
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

// ---- the relay a vote needs when one thread owns all 32 lanes ----
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
// instead of twice per lane — and each of those trips is an L2 round trip. The
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

// ---- a reduction barrier ----
// `bar.red` publishes one bit per thread, then every thread folds the group.
// The barrier is the hardware's, so this is a publish, a barrier, and a loop —
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

}  // namespace weft
)WEFMSL";

