#pragma once

#include <cstring>
#include <string>

// The software 64-bit float, as one text that both columns compile.

// A GPU with no fp64 cannot carry the type at all, Metal rejects it outright,
// which is the language and not a missing lowering, so the value is carried as
// the `uint64_t` its bits live in and every operation is exact IEEE-754 binary64
// over the integers the hardware does have.

// It lives in its own text because it is not the GPU's alone any more: the CPU
// column runs the same functions for the calls the host's libm cannot be held to
// (a double's `exp`, `log`, `pow` and trigonometry), so a module asking for one
// gets the same number on both targets rather than the GPU's approximation
// against the host's library.

// The text is plain C++: two of Metal's address-space words are keywords there
// and not here, and `weft_f64_cpp_text` below defines them for its length.
static const char* kWWeftF64 = R"WEFFF64(
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
// what this host's compiler dies on, at pipeline creation, with
// XPC_ERROR_CONNECTION_INTERRUPTED and nothing else, and the 32-bit builtin is
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

// The working form. A subnormal is normalized here, it has fewer than 53
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
  // and [2^53, 2^54) when it is not, so the leading bit belongs at 55, and the
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
// `.sat` on a floating-point destination is the [0, 1] clamp with a NaN flushed to
// positive zero, and it is not the destination's range. Written over the comparisons
// above rather than over the sign bit, so -0.0 stays the zero it was, which is what
// the CPU column's own `-0.0 < 0.0` test answers too.
static inline uint64_t sat_bits(uint64_t b) {
  if (is_nan(b)) return 0ull;
  if (lt(b, 0ull)) return 0ull;
  if (gt(b, kOne)) return kOne;
  return b;
}

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
  // 84 bits below it, more than the 106-bit product spans, so nothing is lost before
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

//
// `sqrt` is the one function in the transcendental library that IEEE-754 pins
// exactly: the result is the correctly rounded root, so a software one is not an
// approximation of the host's, it is the same number, and the two can be compared
// bit for bit. Everything else in that library rounds through whatever the vendor
// chose and is therefore not carried here.
//
// The root is built from the significand as an integer: the input is written
// `sig * 2^e` with `sig` in [2^52, 2^53) and `e` even, `sig << 54` is the radicand,
// and `floor(sqrt(that))` is 54 bits of the answer. The restoring loop is the one
// from pencil-and-paper long division, two radicand bits a step, its remainder is
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
  // No finite double roots below the smallest normal, sqrt(2^-1074) is 2^-537, so
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
// fract: the part below the binary point, capped at the largest value under one, which
// is what OpenCL.std defines it as. A NaN and an infinity have no fraction and both
// answer the cap. Everything else subtracts its own floor, and that subtraction is
// exact: above one the two values are the same, and below it the difference is a
// representable neighbour of the smaller magnitude. -0.0 answers +0.0, because
// -0.0 - (-0.0) is a positive zero in round-to-nearest.
static inline uint64_t fract_bits(uint64_t b) {
  const uint64_t cap = 0x3fefffffffffffffull;  // 1 - 2^-53, the largest below 1
  if (((b >> 52) & 0x7FFull) == 0x7FFull) return cap;  // a NaN, or an infinity
  const uint64_t r = sub(b, floor_bits(b));
  return gt(r, cap) ? cap : r;
}

// To the nearest integer, ties to even, what a conversion to an integer with
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

//
// `hi` is the value and `lo` the part that did not fit, so a sum that cancels most
// of `hi` still keeps its answer in `lo`. A little over a hundred bits, which is
// what a reduced trigonometric argument needs, and what lets an exponent be
// rounded once at the end instead of twice on the way through.
struct dd {
  uint64_t hi, lo;
};

static inline dd dd_make(uint64_t hi, uint64_t lo) {
  dd r;
  r.hi = hi;
  r.lo = lo;
  return r;
}
static inline dd dd_of(uint64_t b) { return dd_make(b, 0ull); }
static inline uint64_t dd_bits(dd a) { return add(a.hi, a.lo); }
static inline dd dd_neg(dd a) { return dd_make(neg(a.hi), neg(a.lo)); }
// The sign of a double-double is the sign of its high part: `lo` rides within an ulp
// of it and frequently has the opposite sign, so taking the magnitude of the two words
// separately adds that fraction a second time instead of taking it away, a whole ulp
// of error at the magnitude of `hi`. Negating both words is exact and is the magnitude.
static inline dd dd_abs(dd a) { return (a.hi >> 63) != 0ull ? dd_neg(a) : a; }
static inline bool dd_is_zero(dd a) { return is_zero(a.hi) && is_zero(a.lo); }
static inline bool dd_is_nan(dd a) { return is_nan(a.hi) || is_nan(a.lo); }

// Knuth's two-sum and its fast form, where the answer is exact whatever the
// operands: `s` is the rounded sum and `lo` what the rounding dropped.
static inline dd dd_two_sum(uint64_t a, uint64_t b) {
  const uint64_t s = add(a, b);
  const uint64_t bb = sub(s, a);
  return dd_make(s, add(sub(a, sub(s, bb)), sub(b, bb)));
}
static inline dd dd_fast_sum(uint64_t a, uint64_t b) {
  const uint64_t s = add(a, b);
  return dd_make(s, sub(b, sub(s, a)));
}
static inline dd dd_add(dd a, dd b) {
  const dd s = dd_two_sum(a.hi, b.hi);
  return dd_fast_sum(s.hi, add(add(a.lo, b.lo), s.lo));
}
static inline dd dd_sub(dd a, dd b) { return dd_add(a, dd_neg(b)); }
static inline dd dd_mul(dd a, dd b) {
  const uint64_t p = mul(a.hi, b.hi);
  // The product's dropped part is exact in `fma`, and the three cross terms are
  // what a double-double adds to it.
  return dd_fast_sum(p, add(add(fma(a.hi, b.hi, neg(p)), mul(a.hi, b.lo)), mul(a.lo, b.hi)));
}
static inline dd dd_mul_d(dd a, uint64_t d) {
  const uint64_t p = mul(a.hi, d);
  return dd_fast_sum(p, add(fma(a.hi, d, neg(p)), mul(a.lo, d)));
}
static inline dd dd_div_d(dd a, uint64_t d) {
  const uint64_t q1 = div(a.hi, d);
  const uint64_t q2 = div(add(fma(neg(q1), d, a.hi), a.lo), d);
  return dd_fast_sum(q1, q2);
}
static inline dd dd_div(dd a, dd b) {
  const uint64_t q1 = div(a.hi, b.hi);
  const dd r = dd_sub(a, dd_mul_d(b, q1));
  return dd_add(dd_of(q1), dd_of(div(r.hi, b.hi)));
}
static inline dd dd_sqrt(dd a) {
  if (dd_is_zero(a)) return dd_of(0ull);
  const uint64_t s0 = sqrt_bits(a.hi);
  const dd e = dd_sub(a, dd_mul(dd_of(s0), dd_of(s0)));
  return dd_add(dd_of(s0), dd_div_d(e, add(s0, s0)));
}

// Multiply by a power of two by moving the exponent, which is exact in either
// direction: a value below the normal range is shifted into it and rounded once.
static inline uint64_t ldexp_bits(uint64_t b, int32_t n) {
  if (n == 0 || is_nan(b) || is_inf(b) || is_zero(b)) return b;
  int32_t e;
  uint64_t sig;
  unpack(b, e, sig);
  const int32_t t = e + n;
  if (t > 1200) return (b & kSign) | kInf;
  if (t < -1200) return b & kSign;
  return round_pack((uint32_t)(b >> 63), t, sig);
}
static inline dd dd_ldexp(dd a, int32_t n) {
  return dd_make(ldexp_bits(a.hi, n), ldexp_bits(a.lo, n));
}

//
// A constant has to be as wide as the thing it multiplies, or the rounding it
// carries is the error of the result: `ln2` as one double times an exponent of
// 1,024 leaves an error of 2^-43, which is 300,000 ulps.
static inline dd kLn2d() { return dd_make(0x3fe62e42fefa39efull, 0x3c7abc9e3b39803full); }
static inline dd kInvLn2d() { return dd_make(0x3ff71547652b82feull, 0x3c7777d0ffda0d24ull); }
static inline dd kLn10d() { return dd_make(0x40026bb1bbb55516ull, 0xbcaf48ad494ea3e9ull); }
static inline dd kPid() { return dd_make(0x400921fb54442d18ull, 0x3ca1a62633145c07ull); }
static inline dd kPiOver2d() { return dd_make(0x3ff921fb54442d18ull, 0x3c91a62633145c07ull); }
static inline dd kInvPid() { return dd_make(0x3fd45f306dc9c883ull, 0xbc76b01ec5417056ull); }
static inline uint64_t kSqrt2B() { return 0x3FF6A09E667F3BCDull; }
static inline uint64_t kPiOver2B() { return 0x3ff921fb54442d18ull; }
static inline uint64_t kPiB() { return 0x400921fb54442d18ull; }
static inline uint64_t kTinyB() { return 0x3E40000000000000ull; }  // 2^-27
static inline uint64_t kBigB() { return 0x4190000000000000ull; }   // 2^26
static inline uint64_t kBighB() { return 0x4086300000000000ull; } // 710
static inline uint64_t kPiOver4B() { return 0x3FE921FB54442D18ull; }
static inline uint64_t kThreePiOver4B() { return 0x4002D97C7F3321D2ull; }

//
// e^r by a Taylor series on `r/16`, then four squarings. The reduced argument is
// under 0.022, so sixteen terms reach 2^-132 and the squarings keep a hundred bits
// of them; `n` is applied afterwards, exactly, as an exponent.
static inline dd exp_scaled(dd r, int32_t n) {
  const dd t = dd_div_d(r, from_i64(16));
  dd s = dd_of(kOne);
  dd term = dd_of(kOne);
  for (int32_t k = 1; k <= 16; k++) {
    term = dd_div_d(dd_mul(term, t), from_i64(k));
    s = dd_add(s, term);
  }
  for (int32_t k = 0; k < 4; k++) s = dd_mul(s, s);
  return dd_ldexp(s, n);
}

static inline dd exp_of_dd(dd v) {
  if (dd_is_nan(v)) return dd_of(kQNaN);
  if (is_inf(v.hi)) return (v.hi >> 63) ? dd_of(0ull) : dd_make(kInf, 0ull);
  if (dd_is_zero(v)) return dd_of(kOne);
  // The ends of the range are decided here, before the series: log(DBL_MAX) and
  // log(DBL_TRUE_MIN), so an argument past either one never has to pass through a
  // reduced one that has underflowed or overflowed on its own.
  if (gt(v.hi, 0x40862E42FEFA39EFull)) return dd_make(kInf, 0ull);
  if (lt(v.hi, 0xC0874910D52D3051ull)) return dd_of(0ull);
  int64_t n = to_i64(rne_bits(dd_bits(dd_mul(v, kInvLn2d()))));
  const dd r = dd_sub(v, dd_mul_d(kLn2d(), from_i64(n)));
  return exp_scaled(r, (int32_t)n);
}

static inline dd exp2_of_dd(dd v) {
  if (dd_is_nan(v)) return dd_of(kQNaN);
  if (is_inf(v.hi)) return (v.hi >> 63) ? dd_of(0ull) : dd_make(kInf, 0ull);
  if (ge(v.hi, 0x4090000000000000ull)) return dd_make(kInf, 0ull);  // 2^1024
  if (lt(v.hi, 0xC090CC0000000000ull)) return dd_of(0ull);          // 2^-1075
  int64_t n = to_i64(rne_bits(v.hi));
  const dd r = dd_mul(dd_sub(v, dd_of(from_i64(n))), kLn2d());
  return exp_scaled(r, (int32_t)n);
}

static inline uint64_t exp_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return (x >> 63) ? 0ull : kInf;
  return dd_bits(exp_of_dd(dd_of(x)));
}
static inline uint64_t exp2_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return (x >> 63) ? 0ull : kInf;
  return dd_bits(exp2_of_dd(dd_of(x)));
}
static inline uint64_t exp10_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return (x >> 63) ? 0ull : kInf;
  return dd_bits(exp_of_dd(dd_mul(dd_of(x), kLn10d())));
}
// e^x - 1: the subtraction that would eat the answer for a small x is the series
// instead, and for anything larger `exp`'s own rounding is below an ulp of the
// difference.
static inline uint64_t expm1_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return is_nan(x) ? kQNaN : x;
  const dd v = dd_of(x);
  if (le(abs_bits(x), kHalf)) {
    dd term = v, sum = v;
    for (int32_t k = 2; k <= 20; k++) {
      term = dd_div_d(dd_mul(term, v), from_i64(k));
      sum = dd_add(sum, term);
    }
    return dd_bits(sum);
  }
  // Past the range of exp the answer is the limit itself: an infinity above, and
  // exactly -1 below, where e^x is under half an ulp of zero difference.
  if (gt(x, 0x40862E42FEFA39EFull)) return kInf;
  if (lt(x, 0xC0874910D52D3051ull)) return kNegOne;
  return dd_bits(dd_sub(exp_of_dd(v), dd_of(kOne)));
}

//
// log(m) from atanh: s = (m-1)/(m+1) is under 0.172 once m is inside
// [1/sqrt(2), sqrt(2)), and 2*(s + s^3/3 + s^5/5 + ...) is a series of thirty-one
// terms to reach 2^-130. The exponent goes back on as `k * ln2`, in the same
// double-double, so the sum is rounded once.
static inline dd log_of_dd(dd v) {
  int32_t be = (int32_t)((v.hi >> 52) & 0x7FF);
  int32_t below = 0;
  if (be == 0) {
    v = dd_ldexp(v, 64);
    below = -64;
    be = (int32_t)((v.hi >> 52) & 0x7FF);
  }
  int32_t k = be - 1023 + below;
  dd m = dd_make((v.hi & kFracMask) | 0x3FF0000000000000ull, ldexp_bits(v.lo, -k));
  if (gt(m.hi, kSqrt2B())) {
    m = dd_mul_d(m, kHalf);
    k += 1;
  }
  const dd s = dd_div(dd_sub(m, dd_of(kOne)), dd_add(m, dd_of(kOne)));
  const dd s2 = dd_mul(s, s);
  dd term = s, sum = s;
  for (int32_t j = 3; j <= 63; j += 2) {
    term = dd_mul(term, s2);
    sum = dd_add(sum, dd_div_d(term, from_i64(j)));
  }
  return dd_add(dd_add(sum, sum), dd_mul_d(kLn2d(), from_i64(k)));
}

static inline uint64_t log_bits(uint64_t x) {
  if (is_nan(x) || (x >> 63) != 0ull) return kQNaN;
  if (is_zero(x)) return kSign | kInf;
  if (is_inf(x)) return kInf;
  return dd_bits(log_of_dd(dd_of(x)));
}
static inline uint64_t log2_bits(uint64_t x) {
  if (is_nan(x) || (x >> 63) != 0ull) return kQNaN;
  if (is_zero(x)) return kSign | kInf;
  if (is_inf(x)) return kInf;
  return dd_bits(dd_mul(log_of_dd(dd_of(x)), kInvLn2d()));
}
static inline uint64_t log10_bits(uint64_t x) {
  if (is_nan(x) || (x >> 63) != 0ull) return kQNaN;
  if (is_zero(x)) return kSign | kInf;
  if (is_inf(x)) return kInf;
  return dd_bits(dd_div(log_of_dd(dd_of(x)), kLn10d()));
}
// log(1+x), with the addition done in the double-double so a small x keeps every
// bit it started with.
static inline uint64_t log1p_bits(uint64_t x) {
  if (is_nan(x) || lt(x, kNegOne)) return kQNaN;
  if (eq(x, kNegOne)) return kSign | kInf;
  if (is_zero(x)) return x;
  if (is_inf(x)) return kInf;
  return dd_bits(log_of_dd(dd_add(dd_of(kOne), dd_of(x))));
}

//
// x^y = e^(y log x), with the product kept in the double-double: its absolute
// error is 2^-96 of the exponent, which is far below the last bit of the answer
// even when the answer is at the top of the range.
static inline uint64_t pow_bits(uint64_t x, uint64_t y) {
  if (is_nan(x) || is_nan(y)) return kQNaN;
  if (is_zero(y)) return kOne;
  if (eq(x, kOne)) return kOne;
  if (is_zero(x)) {
    if ((y >> 63) != 0ull) return kSign | kInf;
    return 0ull;
  }
  if (is_inf(y)) {
    const bool big = gt(abs_bits(x), kOne);
    return (big == ((y >> 63) == 0ull)) ? kInf : 0ull;
  }
  if (is_inf(x)) {
    const bool up = gt(y, 0ull);
    if ((x >> 63) == 0ull) return up ? kInf : 0ull;
    // A negative infinity raised to an odd integer is the negative infinity.
    const bool odd = (to_i64(abs_bits(y)) & 1) != 0;
    return up ? ((odd ? kSign : 0ull) | kInf) : ((odd ? kSign : 0ull) | 0ull);
  }
  if ((x >> 63) != 0ull) {
    // A negative base has a real power only for an integer exponent.
    if (ne(rne_bits(y), y)) return kQNaN;
    const bool odd = (to_i64(abs_bits(y)) & 1) != 0;
    const uint64_t r = pow_bits(abs_bits(x), y);
    return odd ? neg(r) : r;
  }
  return dd_bits(exp_of_dd(dd_mul(dd_of(y), log_of_dd(dd_of(x)))));
}
static inline uint64_t powr_bits(uint64_t x, uint64_t y) {
  if (is_nan(x) || is_nan(y)) return kQNaN;
  if (lt(x, 0ull) || (is_zero(x) && is_zero(y))) return kQNaN;
  if (is_zero(y)) return kOne;
  return pow_bits(x, y);
}
static inline uint64_t pown_bits(uint64_t x, int32_t n) {
  return pow_bits(x, from_i64(n));
}
static inline uint64_t rootn_bits(uint64_t x, int32_t n) {
  if (n == 0) return kQNaN;
  const int32_t nn = n < 0 ? -n : n;
  const bool odd = (nn & 1) != 0;
  if (is_nan(x)) return kQNaN;
  // The sign of the zero carries through, in the infinity and in the zero.
  if (is_zero(x)) return (n < 0) ? ((x & kSign) | kInf) : x;
  if (lt(x, 0ull) && !odd) return kQNaN;
  const uint64_t a = abs_bits(x);
  if (is_inf(a)) return (n < 0) ? ((x >> 63) != 0ull ? (kSign | 0ull) : 0ull) : x;
  // A negative index is not the reciprocal of a rounded root, that rounds twice and
  // costs the last bit, which `rootn(0.5, -3)` showed. The signed index goes into the
  // exponent, so exp(log(a)/n) is the answer in one rounding, and the square root of a
  // positive index is the one case the hardware helper is exact for.
  const uint64_t r = (nn == 2 && n > 0)
                         ? sqrt_bits(a)
                         : dd_bits(exp_of_dd(dd_div_d(log_of_dd(dd_of(a)), from_i64(n))));
  return ((x >> 63) != 0ull) ? neg(r) : r;
}
static inline uint64_t cbrt_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x) || is_zero(x)) return x;
  const uint64_t a = abs_bits(x);
  const uint64_t r = dd_bits(exp_of_dd(dd_div_d(log_of_dd(dd_of(a)), from_i64(3))));
  return (x >> 63) != 0ull ? neg(r) : r;
}
static inline uint64_t hypot_bits(uint64_t x, uint64_t y) {
  if (is_nan(x) || is_nan(y)) return kQNaN;
  const uint64_t ax = abs_bits(x), ay = abs_bits(y);
  if (is_inf(ax) || is_inf(ay)) return kInf;
  const uint64_t m = gt(ax, ay) ? ax : ay;
  if (is_zero(m)) return 0ull;
  const int32_t e = (int32_t)((m >> 52) & 0x7FF) - 1023;
  const dd a = dd_ldexp(dd_of(x), -e), b = dd_ldexp(dd_of(y), -e);
  const dd s = dd_add(dd_mul(a, a), dd_mul(b, b));
  return dd_bits(dd_ldexp(dd_sqrt(s), e));
}

//
// The argument is reduced against 2/pi as an integer: a double is a 53-bit
// significand times a power of two, so `x * 2/pi` is a product of that integer with
// the bits of 2/pi, and the bits of the answer that matter are the ones at weights
// 2^1 down to 2^-200. Those are accumulated in a 320-bit window, which is exact:
// every partial product is an integer, and the huge integer parts cancel by
// construction rather than by luck. What comes out is the reduced value with more
// than a hundred and fifty bits of it, which is what a near-multiple of pi/2 needs , 
// a floating-point reduction loses exactly those bits, and the answer with them.
//
// 1,408 bits of 2/pi: a double's exponent reaches 1,023, and the reduced value is
// wanted to 200 bits below its own leading bit.
constant uint64_t kInvPi2Bits[22] = {
    0xa2f9836e4e441529ull, 0xfc2757d1f534ddc0ull, 0xdb6295993c439041ull, 0xfe5163abdebbc561ull,
    0xb7246e3a424dd2e0ull, 0x06492eea09d1921cull, 0xfe1deb1cb129a73eull, 0xe88235f52ebb4484ull,
    0xe99c7026b45f7e41ull, 0x3991d639835339f4ull, 0x9c845f8bbdf9283bull, 0x1ff897ffde05980full,
    0xef2f118b5a0a6d1full, 0x6d367ecf27cb09b7ull, 0x4f463f669e5fea2dull, 0x7527bac7ebe5f17bull,
    0x3d0739f78a5292eaull, 0x6bfb5fb11f8d5d08ull, 0x56033046fc7b6babull, 0xf0cfbc209af4361dull,
    0xa9e391615ee61b08ull, 0x6599855f14a06840ull,
};

static inline uint64_t acc_bit(thread const uint64_t* a, int32_t p) {
  if (p < 0 || p > 319) return 0ull;
  return (a[p >> 6] >> (p & 63)) & 1ull;
}

static inline void place_bits(thread uint64_t* sp, uint64_t v, int32_t at) {
  if (v == 0ull) return;
  const int32_t w = at >> 6, b = at & 63;
  if (w >= 0 && w < 6) sp[w] |= (b == 0) ? v : (v << b);
  if (b != 0 && w + 1 >= 0 && w + 1 < 6) sp[w + 1] |= v >> (64 - b);
}

// Add (hi:lo) * 2^sh into the 320-bit window, keeping every bit that lands inside
// it and a sticky bit for the ones that fall below.
static inline void add_shifted(thread uint64_t* acc, uint64_t hi, uint64_t lo, int32_t sh) {
  uint64_t sp[6] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
  if (sh >= 0) {
    place_bits(sp, lo, sh);
    place_bits(sp, hi, sh + 64);
  } else {
    const int32_t d = -sh;
    const int32_t wd = d >> 6, b = d & 63;
    uint64_t lost = 0ull;
    if (wd >= 2) {
      lost = lo | hi;
    } else {
      if (wd == 1) lost = lo;
      const uint64_t cur0 = (wd == 0) ? lo : (wd == 1 ? hi : 0ull);
      const uint64_t cur1 = (wd == 0) ? hi : 0ull;
      if (b == 0) {
        sp[0] = cur0;
        sp[1] = cur1;
      } else {
        lost |= cur0 & ((1ull << b) - 1ull);
        sp[0] = (cur0 >> b) | (cur1 << (64 - b));
        sp[1] = cur1 >> b;
      }
    }
    sp[0] |= (lost != 0ull) ? 1ull : 0ull;
  }
  uint64_t carry = 0ull;
  for (int32_t t = 0; t < 5; t++) {
    const uint64_t a0 = acc[t];
    const uint64_t s1 = a0 + sp[t];
    const uint64_t c1 = (s1 < a0) ? 1ull : 0ull;
    const uint64_t s2 = s1 + carry;
    const uint64_t c2 = (s2 < s1) ? 1ull : 0ull;
    acc[t] = s2;
    carry = c1 | c2;
  }
}

struct red {
  dd y;
  int32_t q;
};

static inline red reduce_pi2(uint64_t xb) {
  red out;
  int32_t e;
  uint64_t M;
  unpack(xb, e, M);  // value = M * 2^(e-55), M in [2^55, 2^56)
  const int32_t k = e - 55;
  out.q = 0;
  out.y = dd_of(xb);
  if (k < -250) return out;  // x mod pi/2 is x, and sin(x) is x
  uint64_t acc[5] = {0ull, 0ull, 0ull, 0ull, 0ull};
  // The limbs whose bits reach weights between 2^1 and 2^-200. The division
  // truncates toward zero, which can start one limb early and never skips one.
  int32_t first = (k - 384) / 64;
  if (first < 0) first = 0;
  int32_t last = (k + 200) / 64 + 1;
  if (last > 21) last = 21;
  for (int32_t i = first; i <= last; i++) {
    const uint64_t limb = kInvPi2Bits[i];
    if (limb == 0ull) continue;
    const uint64_t p0 = (M & 0xFFFFFFFFull) * (limb & 0xFFFFFFFFull);
    const uint64_t p1 = (M >> 32) * (limb & 0xFFFFFFFFull);
    const uint64_t p2 = (M & 0xFFFFFFFFull) * (limb >> 32);
    const uint64_t p3 = (M >> 32) * (limb >> 32);
    const uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p2 & 0xFFFFFFFFull);
    const uint64_t plo = (mid << 32) | (p0 & 0xFFFFFFFFull);
    const uint64_t phi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    add_shifted(acc, phi, plo, k - 64 * (i + 1) + 318);
  }
  // The window holds the product mod 4: bit 319 is 2^1 and bit 0 is 2^-318.
  const uint32_t q = (uint32_t)((acc[4] >> 62) & 3ull);
  acc[4] &= (1ull << 62) - 1ull;
  bool negative = false;
  if (((acc[4] >> 61) & 1ull) != 0ull) {
    // The fraction is at least a half, so the nearest multiple of pi/2 is one
    // further on and the reduced value is what is left, one minus the fraction,
    // negated, which is the two's complement of the window.
    uint64_t carry = 1ull;
    for (int32_t t = 0; t < 5; t++) {
      const uint64_t v = ~acc[t];
      const uint64_t s2 = v + carry;
      carry = (s2 < v) ? 1ull : 0ull;
      acc[t] = s2;
    }
    acc[4] &= (1ull << 62) - 1ull;
    negative = true;
  }
  out.q = (int32_t)((q + (negative ? 1u : 0u)) & 3u);
  int32_t lead = -1;
  for (int32_t t = 4; t >= 0; t--) {
    if (acc[t] != 0ull) {
      lead = t * 64 + (63 - (int32_t)clz64(acc[t]));
      break;
    }
  }
  if (lead < 0) {
    out.y = dd_of(0ull);
    return out;
  }
  uint64_t top = 0ull, nxt = 0ull;
  for (int32_t t = 0; t < 53; t++) top = (top << 1) | acc_bit(acc, lead - t);
  for (int32_t t = 0; t < 53; t++) nxt = (nxt << 1) | acc_bit(acc, lead - 53 - t);
  const dd mag = dd_make(ldexp_bits(from_u64(top), lead - 318 - 52),
                         ldexp_bits(from_u64(nxt), lead - 318 - 105));
  const dd y = dd_mul(mag, kPiOver2d());
  out.y = negative ? dd_neg(y) : y;
  return out;
}

// sin and cos of a reduced argument, |y| <= pi/4, to about a hundred and twenty
// bits: sixteen terms of the series each, which is two terms past the last bit a
// double can hold.
static inline dd sin_dd(dd y) {
  const dd y2 = dd_mul(y, y);
  dd term = y, sum = y;
  for (int32_t t = 1; t <= 15; t++) {
    term = dd_neg(dd_div_d(dd_mul(term, y2), from_i64((int64_t)(2 * t) * (int64_t)(2 * t + 1))));
    sum = dd_add(sum, term);
  }
  return sum;
}
static inline dd cos_dd(dd y) {
  const dd y2 = dd_mul(y, y);
  dd term = dd_of(kOne), sum = dd_of(kOne);
  for (int32_t t = 1; t <= 16; t++) {
    term = dd_neg(dd_div_d(dd_mul(term, y2), from_i64((int64_t)(2 * t - 1) * (int64_t)(2 * t))));
    sum = dd_add(sum, term);
  }
  return sum;
}

static inline uint64_t sin_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return kQNaN;
  if (is_zero(x)) return x;
  const bool neg_ = (x >> 63) != 0ull;
  const red r = reduce_pi2(abs_bits(x));
  dd v;
  if (r.q == 0) v = sin_dd(r.y);
  else if (r.q == 1) v = cos_dd(r.y);
  else if (r.q == 2) v = dd_neg(sin_dd(r.y));
  else v = dd_neg(cos_dd(r.y));
  return dd_bits(neg_ ? dd_neg(v) : v);
}
static inline uint64_t cos_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return kQNaN;
  const red r = reduce_pi2(abs_bits(x));
  dd v;
  if (r.q == 0) v = cos_dd(r.y);
  else if (r.q == 1) v = dd_neg(sin_dd(r.y));
  else if (r.q == 2) v = dd_neg(cos_dd(r.y));
  else v = sin_dd(r.y);
  return dd_bits(v);
}
static inline uint64_t tan_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return kQNaN;
  if (is_zero(x)) return x;
  const bool neg_ = (x >> 63) != 0ull;
  const red r = reduce_pi2(abs_bits(x));
  dd s, c;
  if (r.q == 0) { s = sin_dd(r.y); c = cos_dd(r.y); }
  else if (r.q == 1) { s = cos_dd(r.y); c = dd_neg(sin_dd(r.y)); }
  else if (r.q == 2) { s = dd_neg(sin_dd(r.y)); c = dd_neg(cos_dd(r.y)); }
  else { s = dd_neg(cos_dd(r.y)); c = sin_dd(r.y); }
  uint64_t v = dd_bits(dd_div(s, c));
  if (neg_) v = neg(v);
  return v;
}

//
// atan is halved five times through atan(t) = 2 atan(t / (1 + sqrt(1+t^2))) until
// the argument is under 0.013, where the series is nine terms to reach 2^-125, and
// the halvings are given back as a multiply by 32.
static inline dd atan_small_dd(dd x) {
  dd t = x;
  for (int32_t k = 0; k < 5; k++) {
    const dd s = dd_sqrt(dd_add(dd_of(kOne), dd_mul(t, t)));
    t = dd_div(t, dd_add(dd_of(kOne), s));
  }
  const dd t2 = dd_mul(t, t);
  dd term = t, sum = t;
  for (int32_t j = 3; j <= 19; j += 2) {
    term = dd_neg(dd_mul(term, t2));
    sum = dd_add(sum, dd_div_d(term, from_i64(j)));
  }
  return dd_mul_d(sum, from_i64(32));
}
static inline dd atan_of_dd(dd x) {
  if (dd_is_nan(x)) return dd_of(kQNaN);
  const bool neg_ = (x.hi >> 63) != 0ull;
  const dd a = dd_abs(x);
  if (lt(a.hi, kTinyB())) return x;
  dd r;
  if (gt(a.hi, kOne)) {
    r = dd_sub(kPiOver2d(), atan_small_dd(dd_div(dd_of(kOne), a)));
  } else {
    r = atan_small_dd(a);
  }
  return neg_ ? dd_neg(r) : r;
}
static inline uint64_t atan_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return ((x >> 63) != 0ull) ? neg(kPiOver2B()) : kPiOver2B();
  if (is_zero(x)) return x;
  return dd_bits(atan_of_dd(dd_of(x)));
}
// The inverse functions are each one double-double computation, and the bits are taken
// from it once at the end. The pi-scaled forms divide by pi inside the same
// double-double: dividing a rounded arctangent by pi rounds twice, which costs a whole
// ulp at the least convenient places.
static inline dd atan2_of_dd(uint64_t y, uint64_t x) {
  const bool ys = (y >> 63) != 0ull, xs = (x >> 63) != 0ull;
  const dd q = dd_div(dd_of(abs_bits(y)), dd_of(abs_bits(x)));
  dd a = atan_of_dd(q);
  if (xs) a = dd_sub(kPid(), a);
  return ys ? dd_neg(a) : a;
}
static inline uint64_t atan2_bits(uint64_t y, uint64_t x) {
  if (is_nan(x) || is_nan(y)) return kQNaN;
  const bool ys = (y >> 63) != 0ull, xs = (x >> 63) != 0ull;
  const bool yz = is_zero(y), xz = is_zero(x);
  if (is_inf(x)) {
    if (is_inf(y)) {
      // The diagonal quadrants: pi/4 or three quarters of it, by the two signs.
      const uint64_t v = xs ? kThreePiOver4B() : kPiOver4B();
      return ys ? neg(v) : v;
    }
    // A finite y over an infinite x is a zero, or pi when that x is negative.
    if (!xs) return ys ? neg(0ull) : 0ull;
    return ys ? neg(kPiB()) : kPiB();
  }
  if (is_inf(y)) return ys ? neg(kPiOver2B()) : kPiOver2B();
  if (xz) {
    if (yz) return xs ? (ys ? neg(kPiB()) : kPiB()) : (ys ? neg(0ull) : 0ull);
    return ys ? neg(kPiOver2B()) : kPiOver2B();
  }
  return dd_bits(atan2_of_dd(y, x));
}

static inline dd asin_of_dd(uint64_t x) {
  // asin(x) = atan(x / sqrt(1 - x^2)), which holds its relative accuracy at both ends
  // of the interval because nothing is subtracted near 0 and the root is the small
  // quantity near 1.
  const bool neg_ = (x >> 63) != 0ull;
  const dd a = dd_abs(dd_of(x));
  const dd s = dd_sqrt(dd_sub(dd_of(kOne), dd_mul(a, a)));
  const dd r = atan_of_dd(dd_div(a, s));
  return neg_ ? dd_neg(r) : r;
}
static inline uint64_t asin_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  const uint64_t a = abs_bits(x);
  if (gt(a, kOne)) return kQNaN;
  if (eq(a, kOne)) return (x >> 63) != 0ull ? neg(kPiOver2B()) : kPiOver2B();
  if (lt(a, kTinyB())) return x;
  return dd_bits(asin_of_dd(x));
}

static inline dd acos_of_dd(uint64_t x) {
  // acos(x) = 2 asin(sqrt((1-x)/2)): the subtraction that loses everything near 1 is
  // halved first, which is why this is not pi/2 - asin(x). The negative half is pi
  // minus the positive one, still in the double-double.
  if ((x >> 63) != 0ull) return dd_sub(kPid(), acos_of_dd(abs_bits(x)));
  const dd h = dd_div_d(dd_sub(dd_of(kOne), dd_of(x)), from_i64(2));
  const dd r = dd_sqrt(h);
  const dd s = dd_sqrt(dd_sub(dd_of(kOne), dd_mul(r, r)));
  const dd t = atan_of_dd(dd_div(r, s));
  return dd_add(t, t);
}
static inline uint64_t acos_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  const uint64_t a = abs_bits(x);
  if (gt(a, kOne)) return kQNaN;
  if (eq(a, kOne)) return (x >> 63) != 0ull ? kPiB() : 0ull;
  return dd_bits(acos_of_dd(x));
}


static inline uint64_t sinh_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return x;
  if (is_zero(x)) return x;
  if (gt(abs_bits(x), kBighB())) return ((x >> 63) != 0ull) ? (kSign | kInf) : kInf;
  uint64_t r;
  if (le(abs_bits(x), kHalf)) {
    const dd v = dd_of(x);
    const dd v2 = dd_mul(v, v);
    dd term = v, sum = v;
    for (int32_t k = 1; k <= 12; k++) {
      term = dd_div_d(dd_mul(term, v2), from_i64((int64_t)(2 * k) * (int64_t)(2 * k + 1)));
      sum = dd_add(sum, term);
    }
    r = dd_bits(sum);
  } else {
    const dd e = exp_of_dd(dd_of(x));
    r = dd_bits(dd_div_d(dd_sub(e, dd_div(dd_of(kOne), e)), from_i64(2)));
  }
  return r;
}
static inline uint64_t cosh_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return is_nan(x) ? kQNaN : kInf;
  if (is_zero(x)) return kOne;
  if (gt(abs_bits(x), kBighB())) return kInf;
  uint64_t r;
  if (le(abs_bits(x), kHalf)) {
    const dd v = dd_of(x);
    const dd v2 = dd_mul(v, v);
    dd term = dd_of(kOne), sum = dd_of(kOne);
    for (int32_t k = 1; k <= 12; k++) {
      term = dd_div_d(dd_mul(term, v2), from_i64((int64_t)(2 * k - 1) * (int64_t)(2 * k)));
      sum = dd_add(sum, term);
    }
    r = dd_bits(sum);
  } else {
    const dd e = exp_of_dd(dd_of(abs_bits(x)));
    r = dd_bits(dd_div_d(dd_add(e, dd_div(dd_of(kOne), e)), from_i64(2)));
  }
  return r;
}
static inline uint64_t tanh_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_zero(x)) return x;
  if (is_inf(x)) return (x >> 63) != 0ull ? kNegOne : kOne;
  const bool neg_ = (x >> 63) != 0ull;
  // Past 40 the exponential is under an ulp of one and the answer is exactly one.
  if (gt(abs_bits(x), 0x4044000000000000ull)) return neg_ ? kNegOne : kOne;
  uint64_t r;
  if (le(abs_bits(x), kHalf)) {
    // sinh and cosh of a small argument, both from their series, divided: the
    // subtraction near zero cancels in neither of them.
    const dd v = dd_of(abs_bits(x));
    const dd v2 = dd_mul(v, v);
    dd a = v, sa = v;
    for (int32_t k = 1; k <= 12; k++) {
      a = dd_div_d(dd_mul(a, v2), from_i64((int64_t)(2 * k) * (int64_t)(2 * k + 1)));
      sa = dd_add(sa, a);
    }
    dd b = dd_of(kOne), ca = dd_of(kOne);
    for (int32_t k = 1; k <= 12; k++) {
      b = dd_div_d(dd_mul(b, v2), from_i64((int64_t)(2 * k - 1) * (int64_t)(2 * k)));
      ca = dd_add(ca, b);
    }
    r = dd_bits(dd_div(sa, ca));
  } else {
    // (1 - e^-2|x|) / (1 + e^-2|x|): for a large argument the exponential is the
    // small one, so nothing near equal is subtracted.
    const dd t = dd_div(dd_of(kOne), exp_of_dd(dd_mul_d(dd_of(abs_bits(x)), from_i64(2))));
    r = dd_bits(dd_div(dd_sub(dd_of(kOne), t), dd_add(dd_of(kOne), t)));
  }
  return neg_ ? neg(r) : r;
}
static inline uint64_t asinh_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x) || is_zero(x)) return x;
  const bool neg_ = (x >> 63) != 0ull;
  const dd a = dd_of(abs_bits(x));
  // Under 2^-27 the x^3/6 term is below half an ulp, and for a subnormal it is not
  // representable at all: the answer is the argument.
  if (lt(a.hi, kTinyB())) return x;
  // Above 2^26 sqrt(x^2+1) is x to the last bit, so log(2x) is the answer and the
  // square that would overflow is never formed.
  if (gt(a.hi, kBigB())) {
    const uint64_t r = dd_bits(dd_add(log_of_dd(a), kLn2d()));
    return neg_ ? neg(r) : r;
  }
  // log(x + sqrt(x^2+1)) written as log1p(x + x^2/(1+sqrt(x^2+1))), so a small x
  // keeps its own bits.
  const dd s = dd_sqrt(dd_add(dd_mul(a, a), dd_of(kOne)));
  const dd t = dd_add(a, dd_div(dd_mul(a, a), dd_add(dd_of(kOne), s)));
  const uint64_t r = dd_bits(log_of_dd(dd_add(dd_of(kOne), t)));
  return neg_ ? neg(r) : r;
}
static inline uint64_t acosh_bits(uint64_t x) {
  if (is_nan(x) || lt(x, kOne)) return kQNaN;
  if (is_inf(x)) return kInf;
  const dd a = dd_of(x);
  if (gt(a.hi, kBigB())) return dd_bits(dd_add(log_of_dd(a), kLn2d()));
  const dd s = dd_sqrt(dd_sub(dd_mul(a, a), dd_of(kOne)));
  return dd_bits(log_of_dd(dd_add(a, s)));
}
static inline uint64_t atanh_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  const uint64_t a = abs_bits(x);
  if (gt(a, kOne)) return kQNaN;
  if (eq(a, kOne)) return (x >> 63) != 0ull ? (kSign | kInf) : kInf;
  if (is_zero(x)) return x;
  // 0.5 * log((1+x)/(1-x)), with both sides of the fraction formed in the
  // double-double.
  const dd one = dd_of(kOne);
  const dd q = dd_div(dd_add(one, dd_of(x)), dd_sub(one, dd_of(x)));
  return dd_bits(dd_mul_d(log_of_dd(q), kHalf));
}

//
// The two that no elementary expression reaches. `erf` is its Maclaurin series where
// the argument is small and the continued fraction of the upper incomplete gamma
// where it is not, since each of those is the one that converges on its own side;
// `tgamma` is Stirling's asymptotic series, carried far enough up the recurrence that
// the series is sharp, and the reflection formula on the other side of zero. All of
// it is double-double over the exact primitives above, which is what holds the last
// bit where a nearly-equal subtraction would otherwise spend it: `erfc(3)` is a
// number 2.2e-5 in size formed from one very nearly itself.
static inline dd kTwoOverSqrtPid() { return dd_make(0x3FF20DD750429B6Dull, 0x3C71AE3A914FED80ull); }
static inline dd kHalfLn2Pid() { return dd_make(0x3FED67F1C864BEB5ull, 0xBC865B5A1B7FF5DFull); }
static inline dd kHalfLnPid() { return dd_make(0x3FE250D048E7A1BDull, 0x3C57ABF2AD8D5088ull); }
static inline uint64_t kSixteenB() { return 0x4030000000000000ull; }

// sin(pi x) with the reduction done on the integer part, so a small fractional part
// keeps its own significant bits instead of being lost on the way into sin(pi*x).
static inline dd sinpi_dd(dd x) {
  const uint64_t n = rne_bits(dd_bits(x));
  const dd r = dd_sub(x, dd_of(n));
  dd s = sin_dd(dd_mul(r, kPid()));
  // The sign is the parity of the whole part, and past 2^52 that whole part is even.
  const int64_t ni = lt(abs_bits(n), 0x4330000000000000ull) ? to_i64(n) : 0;
  if ((ni & 1) != 0) s = dd_neg(s);
  return s;
}

// erf on [-3, 3]: 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1)). The terms peak at
// n = x^2, so the top of the range needs a hundred and fifty of them to reach 2^-100;
// the largest of them is 3,200 times the answer, which costs twelve bits of the
// hundred and six and no more.
static inline dd erf_series(dd x) {
  const dd x2 = dd_mul(x, x);
  dd t = x, sum = x;
  for (int32_t n = 1; n <= 150; n++) {
    t = dd_neg(dd_div_d(dd_mul(t, x2), from_i64(n)));
    sum = dd_add(sum, dd_div_d(t, from_i64(2 * n + 1)));
  }
  return dd_mul(sum, kTwoOverSqrtPid());
}

// erfc on [3, inf): the continued fraction of Gamma(1/2, x^2), in the form that
// converges without ever forming the reciprocal of something near zero. A fixed
// hundred terms, so both engines take the same path and the same number of them.
static inline dd erfc_cf(dd x) {
  const dd z = dd_mul(x, x);
  dd b = dd_add(z, dd_of(0x3FE0000000000000ull));  // z + 1/2
  dd c = dd_of(0x7FEFFFFFFFFFFFFFull);             // effectively unbounded
  dd d = dd_div(dd_of(kOne), b);
  dd h = d;
  for (int32_t i = 1; i <= 120; i++) {
    const dd an = dd_mul_d(dd_of(neg(from_i64((int64_t)i * (int64_t)(2 * i - 1)))), 0x3FE0000000000000ull);
    b = dd_add(b, dd_of(from_i64(2)));
    d = dd_add(dd_add(dd_mul(an, d), b), dd_of(0ull));
    d = dd_div(dd_of(kOne), d);
    c = dd_add(b, dd_div(an, c));
    h = dd_mul(h, dd_mul(d, c));
  }
  const dd lz = log_of_dd(z);
  const dd l = dd_sub(dd_add(dd_neg(z), dd_mul_d(lz, 0x3FE0000000000000ull)), kHalfLnPid());
  return dd_mul(exp_of_dd(l), h);
}

static inline uint64_t erf_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return (x >> 63) != 0ull ? kNegOne : kOne;
  if (is_zero(x)) return x;
  const bool neg_ = (x >> 63) != 0ull;
  const uint64_t a = abs_bits(x);
  dd r;
  if (le(a, 0x4008000000000000ull)) {  // |x| <= 3: the series, where it converges
    r = erf_series(dd_of(a));
  } else {
    if (gt(a, 0x403B400000000000ull)) return neg_ ? kNegOne : kOne;  // under 2^-53 either way
    r = dd_sub(dd_of(kOne), erfc_cf(dd_of(a)));
  }
  return dd_bits(neg_ ? dd_neg(r) : r);
}

static inline uint64_t erfc_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return (x >> 63) != 0ull ? 0x4000000000000000ull : 0ull;  // erfc(-inf) = 2
  const bool neg_ = (x >> 63) != 0ull;
  const uint64_t a = abs_bits(x);
  dd r;
  if (gt(a, 0x403B400000000000ull)) r = dd_of(0ull);       // below the smallest subnormal
  else if (le(a, 0x4008000000000000ull)) r = dd_sub(dd_of(kOne), erf_series(dd_of(a)));
  else r = erfc_cf(dd_of(a));
  if (neg_) r = dd_sub(dd_of(from_i64(2)), r);
  return dd_bits(r);
}

// lgamma(z) for z >= 16: Stirling, with eight Bernoulli terms. The first omitted one
// is under 4e-22 at z = 16, which is six orders below an ulp of any gamma this can
// still return.
static inline dd lgamma_large_dd(dd z) {
  const dd iz = dd_div(dd_of(kOne), z);
  dd r = dd_sub(dd_mul(dd_sub(z, dd_of(0x3FE0000000000000ull)), log_of_dd(z)), z);
  r = dd_add(r, kHalfLn2Pid());
  dd p = dd_mul(iz, iz);   // 1/z^3 after two more multiply, so start from 1/z
  p = iz;
  r = dd_add(r, dd_mul_d(p, 0x3FB5555555555555ull));
  p = dd_mul(dd_mul(p, iz), iz);
  r = dd_add(r, dd_mul_d(p, 0xBF66C16C16C16C17ull));
  p = dd_mul(dd_mul(p, iz), iz);
  r = dd_add(r, dd_mul_d(p, 0x3F4A01A01A01A01Aull));
  p = dd_mul(dd_mul(p, iz), iz);
  r = dd_add(r, dd_mul_d(p, 0xBF43813813813814ull));
  p = dd_mul(dd_mul(p, iz), iz);
  r = dd_add(r, dd_mul_d(p, 0x3F4B951E2B18FF23ull));
  p = dd_mul(dd_mul(p, iz), iz);
  r = dd_add(r, dd_mul_d(p, 0xBF5F6AB0D9993C7Dull));
  p = dd_mul(dd_mul(p, iz), iz);
  r = dd_add(r, dd_mul_d(p, 0x3F7A41A41A41A41Aull));
  p = dd_mul(dd_mul(p, iz), iz);
  r = dd_add(r, dd_mul_d(p, 0xBF9E4286CB0F5398ull));
  return r;
}

// Gamma of a positive argument: up the recurrence until the series is sharp, then the
// logarithms of the steps back down, all in the double-double so the sum of sixteen
// of them costs nothing that can be seen in the answer.
static inline dd tgamma_pos_dd(dd v) {
  dd z = v;
  int32_t k = 0;
  while (lt(dd_bits(z), kSixteenB()) && k < 17) {
    z = dd_add(z, dd_of(kOne));
    k += 1;
  }
  dd l = lgamma_large_dd(z);
  for (int32_t j = k - 1; j >= 0; j--) {
    l = dd_sub(l, log_of_dd(dd_add(v, dd_of(from_i64(j)))));
  }
  return exp_of_dd(l);
}

static inline uint64_t tgamma_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return (x >> 63) != 0ull ? kQNaN : kInf;
  if (is_zero(x)) return (x & kSign) | kInf;
  const bool neg_ = (x >> 63) != 0ull;
  dd r;
  if (!neg_) {
    r = tgamma_pos_dd(dd_of(x));
  } else {
    // A negative integer is a pole, and the reflection formula would divide by a zero
    // sine: NaN, which is what the specification says.
    if (eq(trunc_bits(x), x)) return kQNaN;
    const dd s = sinpi_dd(dd_of(x));
    const dd g = tgamma_pos_dd(dd_sub(dd_of(kOne), dd_of(x)));
    r = dd_div(kPid(), dd_mul(s, g));
  }
  return dd_bits(r);
}

// The pi-scaled inverse forms: one multiply by 1/pi, in the double-double, so the
// answer is the same number the un-scaled form divided by pi would be.
static inline uint64_t asinpi_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  const uint64_t a = abs_bits(x);
  if (gt(a, kOne)) return kQNaN;
  if (eq(a, kOne)) return (x >> 63) != 0ull ? neg(kHalf) : kHalf;
  if (is_zero(x)) return x;
  return dd_bits(dd_mul(asin_of_dd(x), kInvPid()));
}

static inline uint64_t acospi_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  const uint64_t a = abs_bits(x);
  if (gt(a, kOne)) return kQNaN;
  if (eq(a, kOne)) return (x >> 63) != 0ull ? kOne : 0ull;
  return dd_bits(dd_mul(acos_of_dd(x), kInvPid()));
}

static inline uint64_t atanpi_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return (x >> 63) != 0ull ? neg(kHalf) : kHalf;
  if (is_zero(x)) return x;
  return dd_bits(dd_mul(atan_of_dd(dd_of(x)), kInvPid()));
}

static inline uint64_t atan2pi_bits(uint64_t y, uint64_t x) {
  if (is_nan(x) || is_nan(y)) return kQNaN;
  const bool ys = (y >> 63) != 0ull, xs = (x >> 63) != 0ull;
  // The quadrant boundaries are exact fractions of pi, so they are written as the
  // quarters they are rather than as an arctangent scaled afterwards.
  const uint64_t quart = mul(kHalf, kHalf);
  const uint64_t threeq = add(kHalf, quart);
  if (is_inf(x)) {
    if (is_inf(y)) return ys ? neg(xs ? threeq : quart) : (xs ? threeq : quart);
    if (!xs) return ys ? neg(0ull) : 0ull;
    return ys ? neg(kOne) : kOne;
  }
  if (is_inf(y)) return ys ? neg(kHalf) : kHalf;
  if (is_zero(x)) {
    if (is_zero(y)) return ys ? neg(0ull) : 0ull;
    return ys ? neg(kHalf) : kHalf;
  }
  return dd_bits(dd_mul(atan2_of_dd(y, x), kInvPid()));
}

// The pi-scaled sines reduce against the nearest *half* integer, not against the period:
// r = x - k/2 is exact and |r| <= 1/4, so the sine is taken at an argument of at most
// pi/4 and an integer multiple of pi lands on zero exactly. Reducing over the period
// instead leaves sin(pi) as the sine of pi as a double-double, the residue of the
// constant, which is the whole answer where the right one is zero.
static inline dd pi_arg(uint64_t x) {
  const uint64_t k = rne_bits(mul(x, from_i64(2)));  // round(2x), exact
  return dd_mul(dd_sub(dd_of(x), dd_of(mul(k, kHalf))), kPid());  // pi * r, |r| <= 1/4
}
static inline int32_t pi_quad(uint64_t x) {
  // The quadrant is k mod 4 for k = round(2x). Past 2^52 the step is a whole number of
  // halves and every double is an integer, so only the parity of that multiple is left;
  // past 2^53 the multiple is even and every one of these lands on a zero of the sine.
  const uint64_t k = rne_bits(mul(x, from_i64(2)));
  if (lt(abs_bits(k), 0x4330000000000000ull)) return (int32_t)(to_i64(k) & 3);
  if (lt(abs_bits(x), 0x4340000000000000ull)) return (int32_t)((to_i64(x) & 1) << 1);
  return 0;
}
static inline uint64_t sinpi_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return kQNaN;
  const dd t = pi_arg(x);
  const int32_t q = pi_quad(x);
  const dd s = sin_dd(t), c = cos_dd(t);
  return dd_bits((q == 0) ? s : (q == 1) ? c : (q == 2) ? dd_neg(s) : dd_neg(c));
}
static inline uint64_t cospi_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return kQNaN;
  const dd t = pi_arg(x);
  const int32_t q = pi_quad(x);
  const dd s = sin_dd(t), c = cos_dd(t);
  return dd_bits((q == 0) ? c : (q == 1) ? dd_neg(s) : (q == 2) ? dd_neg(c) : s);
}
static inline uint64_t tanpi_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x)) return kQNaN;
  const dd t = pi_arg(x);
  const int32_t q = pi_quad(x);
  const dd s = sin_dd(t), c = cos_dd(t);
  const dd sn = (q == 0) ? s : (q == 1) ? c : (q == 2) ? dd_neg(s) : dd_neg(c);
  const dd cs = (q == 0) ? c : (q == 1) ? dd_neg(s) : (q == 2) ? dd_neg(c) : s;
  return dd_bits(dd_div(sn, cs));
}




static inline uint64_t copysign_bits(uint64_t x, uint64_t y) {
  return (x & ~kSign) | (y & kSign);
}
static inline uint64_t nextafter_bits(uint64_t x, uint64_t y) {
  if (is_nan(x) || is_nan(y)) return kQNaN;
  if (eq(x, y)) return y;
  if (is_zero(x)) return (y & kSign) | 1ull;  // the smallest subnormal of y's sign
  // The encoding is monotone in the magnitude, so a step is one away from zero or
  // one toward it, which of the two depends on the sign of x and the direction.
  const uint64_t mag = x & ~kSign;
  const bool positive = (x >> 63) == 0ull;
  const bool away = (positive == lt(x, y));
  // The step is one in the encoding, and the encoding is an integer: adding one to the
  // magnitude walks away from zero and subtracting one walks toward it, with the sign
  // carried separately. `add` here is the fp64 addition, handing it a bit pattern as
  // if it were a number is what made this return a NaN for every operand pair.
  const uint64_t m = away ? mag + 1ull : mag - 1ull;
  return (x & kSign) | m;
}
static inline uint64_t minmag_bits(uint64_t a, uint64_t b) {
  if (is_nan(a)) return is_nan(b) ? kQNaN : b;
  if (is_nan(b)) return a;
  // Equal magnitudes are settled by the values, not by their magnitudes: the spec
  // says fmin, so minmag(-1, 1) is -1 rather than whichever came second.
  if (eq(abs_bits(a), abs_bits(b))) return lt(a, b) ? a : b;
  return lt(abs_bits(a), abs_bits(b)) ? a : b;
}
static inline uint64_t maxmag_bits(uint64_t a, uint64_t b) {
  if (is_nan(a)) return is_nan(b) ? kQNaN : b;
  if (is_nan(b)) return a;
  // A tie is fmax: maxmag(-1, 1) is 1, and maxmag(-inf, inf) is inf.
  if (eq(abs_bits(a), abs_bits(b))) return gt(a, b) ? a : b;
  return gt(abs_bits(a), abs_bits(b)) ? a : b;
}
static inline uint64_t fdim_bits(uint64_t a, uint64_t b) {
  if (is_nan(a) || is_nan(b)) return kQNaN;
  return gt(a, b) ? sub(a, b) : 0ull;
}
// fmod is exact, so it is long division on the significands rather than a
// subtraction in floating point: `a` and `b` are integers times a power of two, and
// the remainder is `a` doubled `d` times modulo `b`, one step a bit.
static inline uint64_t fmod_bits(uint64_t a, uint64_t b) {
  if (is_nan(a) || is_nan(b)) return kQNaN;
  const uint64_t sign = a & kSign;
  const uint64_t aa = abs_bits(a), ab = abs_bits(b);
  if (is_inf(ab)) return is_inf(aa) ? kQNaN : a;
  if (is_zero(ab)) return kQNaN;
  if (is_inf(aa) || is_zero(aa) || lt(aa, ab)) return a;
  int32_t ea, eb;
  uint64_t ma, mb;
  unpack(aa, ea, ma);
  unpack(ab, eb, mb);
  uint64_t r = ma;
  if (r >= mb) r -= mb;
  const int32_t d = ea - eb;
  for (int32_t i = 0; i < d; i++) {
    r <<= 1;
    if (r >= mb) r -= mb;
  }
  if (r == 0ull) return sign;
  int32_t e = eb;
  while ((r & kLead) == 0ull) {
    r <<= 1;
    e -= 1;
  }
  return round_pack((uint32_t)(sign != 0ull), e, r);
}
// The IEEE remainder is the fmod with the quotient rounded to nearest: the two
// differ by one divisor, and both candidates are exact. Which way the divisor is
// subtracted is the remainder's own sign, not the divisor's, for a positive divisor
// and a negative dividend the nearest multiple is the other one, so subtracting the
// divisor as given would walk away from zero instead of toward it.
static inline uint64_t remainder_bits(uint64_t a, uint64_t b) {
  uint64_t r = fmod_bits(a, b);
  if (is_nan(r) || is_zero(r) || is_inf(b) || is_zero(b)) return r;
  const uint64_t twice = add(abs_bits(r), abs_bits(r));
  if (eq(twice, abs_bits(b))) {
    // Exactly half of the divisor: the tie goes to the even quotient. A tie can only
    // happen when that quotient fits in a significand, any larger and the product
    // with the divisor would not, so its parity is exact here.
    const dd k = dd_div(dd_sub(dd_of(abs_bits(a)), dd_of(mul(abs_bits(b), kHalf))),
                        dd_of(abs_bits(b)));
    if ((to_i64(rne_bits(dd_bits(k))) & 1) != 0)
      r = sub(r, copysign_bits(abs_bits(b), r));
    return r;
  }
  if (gt(twice, abs_bits(b))) r = sub(r, copysign_bits(abs_bits(b), r));
  return r;
}

//
// Each pair is one decision, so both halves come from the same reading of the input:
// a caller cannot be handed a fraction whose exponent came from another lane.
//
// frexp is the encoding's own exponent. Unpacking gives the unbiased exponent e such
// that the value is M * 2^(e-52) with M in [2^52, 2^53), the power of two at or below
// it, so the pair's exponent is one more than that, and the fraction is x with it
// divided out.
static inline uint64_t frexp_e_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x) || is_zero(x)) return 0;
  int32_t e;
  uint64_t m;
  unpack(abs_bits(x), e, m);
  return (uint64_t)(int64_t)(e + 1);
}
static inline uint64_t frexp_f_bits(uint64_t x) {
  if (is_nan(x) || is_inf(x) || is_zero(x)) return x;
  return ldexp_bits(x, -(int32_t)frexp_e_bits(x));
}
// modf is trunc, and the difference: it is under one in magnitude, so the
// subtraction is exact.
static inline uint64_t modf_i_bits(uint64_t x) { return trunc_bits(x); }
static inline uint64_t modf_f_bits(uint64_t x) {
  if (is_nan(x)) return kQNaN;
  if (is_inf(x)) return x & kSign;
  return sub(x, trunc_bits(x));
}
static inline uint64_t sincos_s_bits(uint64_t x) { return sin_bits(x); }
static inline uint64_t sincos_c_bits(uint64_t x) { return cos_bits(x); }
// The quotient of remquo is the integer the remainder is defined by, and the
// specification wants its sign and its low thirty-two bits rather than its magnitude:
// 1e20/3 is 33333333333333333333, an integer an int64 does not hold and whose low
// bits are still defined. They come out of one exact fmod instead, x is n*y + r, so
// x mod 2^32*y is (n mod 2^32)*y + r, and subtracting r and dividing by y leaves the
// low bits themselves. Every step there is exact or within half an ulp of an integer,
// which is why the division can be rounded to nearest afterwards.
static inline int32_t remquo_q_bits(uint64_t x, uint64_t y) {
  if (is_nan(x) || is_nan(y) || is_inf(x) || is_zero(y)) return 0;
  const uint64_t two32 = 0x41F0000000000000ull;
  const uint64_t scaled = mul(y, two32);
  uint64_t q;
  if (is_inf(scaled)) {
    // |y| past 2^991, so |n| is under 2^33 whatever x is: a double holds it whole.
    q = rne_bits(dd_bits(dd_div(dd_of(x), dd_of(y))));
  } else {
    const uint64_t part = sub(fmod_bits(x, scaled), remainder_bits(x, y));
    q = rne_bits(dd_bits(dd_div(dd_of(part), dd_of(y))));
  }
  return (int32_t)(uint32_t)(uint64_t)to_i64(q);
}
static inline uint64_t remquo_r_bits(uint64_t x, uint64_t y) {
  return remainder_bits(x, y);
}

}  // namespace weft_fp64
)WEFFF64";

// The same text for the CPU column. `thread` and `constant` are Metal's
// address spaces; in C++ the first is nothing and the second is `const`. The
// definitions are undone straight after, so the module around this text is not
// compiled with them.
static std::string weft_f64_cpp_text() {
  return "#define thread\n#define constant const\n"
         "template <class T, class U>\n"
         "static inline T as_type(U u) { T t; std::memcpy(&t, &u, sizeof(T)); return t; }\n" +
         std::string(kWWeftF64) + "\n#undef thread\n#undef constant\n";
}
