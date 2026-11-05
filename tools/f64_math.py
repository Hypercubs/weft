#!/usr/bin/env python3
"""Sweep the software double's transcendentals against an exact reference.

The library (`src/weft_f64.hpp`) is one text: the GPU compiles it as MSL and the CPU
column as C++. Neither is a reference for the other, so this compares both to the
number itself, every function is evaluated in `decimal` at 1,200 significant digits
and the answer is rounded once, to nearest, and the library's bits are counted
against it in ulps.

    python3 tools/f64_math.py            # the default sweep
    python3 tools/f64_math.py --verbose  # and every function's worst input

A function that is off by more than an ulp on any input is a failure; the exit status
says whether that happened.
"""

import argparse
import decimal
import math
from fractions import Fraction
import os
import re
import struct
import subprocess
import sys
import tempfile
from decimal import Decimal

DPS = 1200
decimal.getcontext().prec = DPS + 40
decimal.getcontext().Emax = 999999
decimal.getcontext().Emin = -999999

def _atan_inv(n):
    """atan(1/n) as a Decimal, by its series."""
    one = Decimal(1)
    n = Decimal(n)
    total = Decimal(0)
    term = one / n
    n2 = n * n
    k = 0
    while True:
        t = term / (2 * k + 1)
        if t == 0:
            break
        total += t if k % 2 == 0 else -t
        term /= n2
        k += 1
    return total

PI = 16 * _atan_inv(5) - 4 * _atan_inv(239)
TAU = 2 * PI
HALF_PI = PI / 2
QUARTER_PI = PI / 4
# The largest double under one: what `fract` answers for a NaN and for an infinity, and
# the cap on any fraction that would otherwise round up to it.
CAP_UNDER_ONE = Decimal(1) - Decimal(2) ** -53


def d(x):
    """A float or an exact string as a Decimal with every bit the input had."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        return Decimal(x)  # exact: binary64 is a finite decimal
    return Decimal(x)


def _want_nan(want):
    """A reference can say "not a number" as a Decimal or as a float; both count."""
    if isinstance(want, Decimal):
        return want.is_nan()
    return isinstance(want, float) and want != want


def ulps(x, want):
    """Distance from the computed double `x` to the nearest double to `want`."""
    if isinstance(x, float) and math.isnan(x):
        return 0 if _want_nan(want) else float("inf")
    if isinstance(want, Decimal) and want.is_infinite():
        # An infinity is an infinity of *that* sign: -inf and +inf are different answers.
        if not (isinstance(x, float) and math.isinf(x)):
            return float("inf")
        return 0 if (x > 0) == (want > 0) else float("inf")
    try:
        target = float(want)
    except OverflowError:  # an exact value past the range of a double
        target = math.inf if want > 0 else -math.inf
    if x == target:
        return 0
    if isinstance(x, float) and math.isinf(x) and not math.isinf(target):
        # `want` rounded to a double is finite, so the answer is wrong by everything.
        return float("inf")
    if math.isnan(target) or math.isinf(target):
        return float("inf")
    # Count steps, so a half-way tie or a subnormal boundary is a fraction of an ulp.
    lo, hi = (x, target) if x < target else (target, x)
    steps = 0.0
    guard = 0
    while lo < hi and guard < 4000:
        lo = math.nextafter(lo, hi)
        steps += 1
        guard += 1
    if guard >= 4000:  # a huge exponent gap is not an ulp question
        return float("inf")
    return steps


def ref_sin(x):
    x = d(x)
    n = (x / TAU).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)
    r = x - TAU * n
    total, term, k = r, r, 1
    while True:
        term = -term * r * r / ((2 * k) * (2 * k + 1))
        if term == 0 or abs(term) < Decimal(10) ** (-DPS):
            break
        total += term
        k += 1
    return total


def ref_cos(x):
    x = d(x)
    n = (x / TAU).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)
    r = x - TAU * n
    total, term, k = Decimal(1), Decimal(1), 1
    while True:
        term = -term * r * r / ((2 * k - 1) * (2 * k))
        if term == 0 or abs(term) < Decimal(10) ** (-DPS):
            break
        total += term
        k += 1
    return total


def ref_tan(x):
    return ref_sin(x) / ref_cos(x)


def ref_atan(x):
    x = d(x)
    if x < 0:
        return -ref_atan(-x)
    if x > 1:
        return HALF_PI - ref_atan(1 / x)
    # Halve the argument five times and give the factor back, exactly as the library
    # does, so the reference is the series and not a second algorithm.
    t = x
    for _ in range(5):
        t = t / (1 + (1 + t * t).sqrt())
    total, term, k = t, t, 1
    while True:
        term = -term * t * t
        add = term / (2 * k + 1)
        if add == 0 or abs(add) < Decimal(10) ** (-DPS):
            break
        total += add
        k += 1
    return total * 32


def ref_atan2(y, x):
    y, x = d(y), d(x)
    if x > 0:
        return ref_atan(y / x)
    if x < 0:
        return ref_atan(y / x) + (PI if y >= 0 else -PI)
    return HALF_PI if y > 0 else (-HALF_PI if y < 0 else Decimal(0))


def ref_asin(x):
    x = d(x)
    return ref_atan2(x, (1 - x * x).sqrt())


def ref_acos(x):
    x = d(x)
    return ref_atan2((1 - x * x).sqrt(), x)


def ref_pow(x, y):
    x, y = d(x), d(y)
    if x == 0:
        return Decimal(0) if y > 0 else Decimal("Infinity")
    if x < 0:
        raise ValueError("negative base")
    return (y * x.ln()).exp()


def ref_expm1(x):
    return d(x).exp() - 1


def ref_log1p(x):
    return (1 + d(x)).ln()


def ref_cbrt(x):
    x = d(x)
    return -((-x).ln().exp() / 3).exp() if False else (abs(x).ln() / 3).exp() * (1 if x >= 0 else -1)


def ref_hypot(x, y):
    return (d(x) ** 2 + d(y) ** 2).sqrt()


def ref_remainder(x, y):
    x, y = d(x), d(y)
    q = (x / y).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)
    return x - y * q


def ref_fmod(x, y):
    x, y = d(x), d(y)
    q = (x / y).to_integral_value(rounding=decimal.ROUND_DOWN)
    return x - y * q


def ref_fract(x):
    """The part below the binary point, capped at the largest value under one. A NaN
    and an infinity have none and both answer the cap, which is what the instruction
    says; the subtraction itself is exact, so there is nothing here to round."""
    x = d(x)
    if x.is_nan() or x.is_infinite():
        return CAP_UNDER_ONE
    r = x - x.to_integral_value(rounding=decimal.ROUND_FLOOR)
    return CAP_UNDER_ONE if r >= CAP_UNDER_ONE else r


#
# `erf` is the Maclaurin series in decimal, which is the same series the library uses
# only on one side of the range; `tgamma` is Stirling's series with the Bernoulli
# numbers as exact fractions, shifted to a different point and carried to more terms
# than the library, so a wrong coefficient cannot cancel out of both.
def ref_erf(x):
    """The Maclaurin series in decimal. Its terms peak at n = x^2 and are that many
    times the answer, so it is carried out to six, past that the double is 1 exactly
    and there is nothing left to measure."""
    x = d(x)
    if abs(x) > 6:
        return Decimal(1) if x > 0 else Decimal(-1)
    if x == 0:
        return Decimal(0)
    total, term, x2 = x, x, x * x
    n = 0
    while True:
        n += 1
        term = -term * x2 / n
        t = term / (2 * n + 1)
        total += t
        if abs(t) < abs(total) * Decimal(10) ** -160:
            break
    return total * 2 / PI.sqrt()


def _bernoulli(n):
    a = [Fraction(0)] * (n + 1)
    for m in range(n + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
    return a[0]


_BERN = [_bernoulli(2 * k) for k in range(1, 11)]


def ref_lgamma(z):
    z = d(z)
    total = (z - Decimal("0.5")) * z.ln() - z + (2 * PI).ln() / 2
    for k, b in enumerate(_BERN, start=1):
        total += (Decimal(b.numerator) / Decimal(b.denominator)
                  / (2 * k * (2 * k - 1)) / z ** (2 * k - 1))
    return total


def ref_sinpi(x):
    x = d(x)
    n = int(x.to_integral_value(rounding=decimal.ROUND_HALF_EVEN))
    r = x - n
    s = PI * r
    total, term, k = s, s, 1
    while True:
        term = -term * s * s / ((2 * k) * (2 * k + 1))
        if abs(term) < Decimal(10) ** -1100:
            break
        total += term
        k += 1
    return -total if n % 2 else total


def ref_tgamma(x):
    x = d(x)
    if x.is_infinite():
        return Decimal("Infinity") if x > 0 else Decimal("NaN")
    if x == 0:
        # tgamma(+-0) is +-inf: a zero is not a pole, its sign carries through.
        return Decimal("-Infinity") if x.is_signed() else Decimal("Infinity")
    if x < 0 and x == x.to_integral_value():
        return Decimal("NaN")  # the poles: the negative integers
    if x < Decimal("0.5"):
        return PI / (ref_sinpi(x) * ref_tgamma(1 - x))
    z, steps = x, 0
    while z < 40:
        z += 1
        steps += 1
    total = ref_lgamma(z)
    for j in range(steps):
        total -= (x + j).ln()
    return total.exp()


def ref_erfc_tail(x):
    """erfc of a positive x above six, as the continued fraction of the incomplete
    gamma: erfc(x) = exp(-x^2) / (x sqrt(pi)) * 1/(1 + v/(1 + 2v/(1 + 3v/...))) with
    v = 1/(2x^2). Above six v is under 0.014, so the fraction is a hundred terms of
    work rather than the thousands the series would need to climb back down from its
    peak."""
    x = d(x)
    v = 1 / (2 * x * x)
    cf = Decimal(0)
    for k in range(150, 0, -1):
        cf = (k * v) / (1 + cf)
    return (-(x * x)).exp() / (x * PI.sqrt()) / (1 + cf)


def ref_erfc(x):
    # Below six the decimal series keeps every bit of the difference. Above it the
    # difference *is* the answer, 1 - erf would have nothing left to subtract from , 
    # and the continued fraction above gives it directly in decimal, so no reference
    # in this file leans on the host's libm.
    x = d(x)
    if abs(x) > 6:
        tail = ref_erfc_tail(abs(x))
        return tail if x > 0 else 2 - tail
    return 1 - ref_erf(x)


#
# Each reference returns a pair, and both halves are computed in exact arithmetic , 
# a Fraction, not a libm call, so the pair cannot inherit the mistake it is meant to
# catch.  The second half is an integer where the specification says an integer.
def ref_frexp(x):
    """x = f * 2^e with f in [0.5, 1): scaling by two is exact in a Fraction."""
    fr = Fraction(d(x))
    if fr == 0:
        return (Decimal(0), 0)
    e = 0
    while abs(fr) >= 1:
        fr /= 2
        e += 1
    while abs(fr) < Fraction(1, 2):
        fr *= 2
        e -= 1
    return (Decimal(fr.numerator) / Decimal(fr.denominator), e)


def ref_modf(x):
    fr = Fraction(d(x))
    whole = int(fr)  # toward zero, which is what truncation means
    rest = fr - whole
    return (Decimal(rest.numerator) / Decimal(rest.denominator), Decimal(whole))


def ref_sincos(x):
    return (ref_sin(x), ref_cos(x))


def ref_remquo(x, y):
    fx, fy = Fraction(d(x)), Fraction(d(y))
    n = round(fx / fy)  # nearest, ties to even: the quotient the remainder is defined by
    rest = fx - fy * n
    low = n & 0xFFFFFFFF  # the specification asks for the sign and the low bits
    if low >= 1 << 31:
        low -= 1 << 32
    return (Decimal(rest.numerator) / Decimal(rest.denominator), low)


#
# The reduction is exact before the decimal ever sees it: x = n + r with r in [-1/2, 1/2]
# is exact for a double, so sin(pi x) is sin(pi r) with the sign of the parity and
# cos(pi x) is the same with the sign of the parity the other way. That is the whole
# point of these instructions, and a reference that reduced a huge argument in decimal
# would be measuring its own loss rather than the library's.
def ref_sinpi_via(x):
    x = d(x)
    n = int(x.to_integral_value(rounding=decimal.ROUND_HALF_EVEN))
    return n, x - n


def ref_cospi(x):
    n, r = ref_sinpi_via(x)
    v = ref_cos(PI * r)
    return -v if n % 2 else v


def ref_tanpi(x):
    n, r = ref_sinpi_via(x)
    return ref_tan(PI * r)


def ref_asinpi(x):
    return ref_asin(x) / PI


def ref_acospi(x):
    return ref_acos(x) / PI


def ref_atanpi(x):
    return ref_atan(x) / PI


def ref_atan2pi(y, x):
    return ref_atan2(y, x) / PI


#
# Everything here is decided by the encoding rather than by an approximation, so the
# reference is the encoding: the sign bit, the monotone bit pattern, the exact
# difference. None of it leans on libm either.
def _bits(v):
    return struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def _of(b):
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


def _nan(v):
    return isinstance(v, float) and v != v


def ref_sqrt(x):
    x = d(x)
    if x.is_nan() or x < 0:
        return Decimal("NaN")
    return x.sqrt()


def ref_copysign(x, y):
    return _of((_bits(x) & ~(1 << 63)) | (_bits(y) & (1 << 63)))


def ref_fdim(x, y):
    if _nan(x) or _nan(y):
        return Decimal("NaN")
    return d(x) - d(y) if d(x) > d(y) else Decimal(0)


def ref_minmag(x, y):
    if _nan(x):
        return Decimal("NaN") if _nan(y) else y
    if _nan(y):
        return x
    if abs(x) == abs(y):          # a tie is fmin, not simply the second operand
        return x if x < y else y
    return x if abs(x) < abs(y) else y


def ref_maxmag(x, y):
    if _nan(x):
        return Decimal("NaN") if _nan(y) else y
    if _nan(y):
        return x
    if abs(x) == abs(y):
        return x if x > y else y
    return x if abs(x) > abs(y) else y


def ref_nextafter(x, y):
    """One step in the encoding, which is monotone in the value: the next bit pattern,
    with the two zeros wired to the smallest subnormal of the other operand's sign."""
    if _nan(x) or _nan(y):
        return Decimal("NaN")
    bx, by = _bits(x), _bits(y)
    if bx == by:
        return y
    if bx == 0 or bx == (1 << 63):
        return _of((by & (1 << 63)) | 1)
    step = 1 if ((x < y) == (x > 0)) else -1
    return _of(bx + step)


def ref_ldexp(x, n):
    return d(x) * (Decimal(2) ** int(n))


def ref_pown(x, n):
    return d(x) ** int(n)


def ref_rootn(x, n):
    x, n = d(x), int(n)
    if x == 0:
        return Decimal(0) if n > 0 else Decimal("Infinity")
    if x < 0:
        if n % 2 == 0:
            return Decimal("NaN")
        return -(((-x).ln()) / n).exp()
    return (x.ln() / n).exp()


def ref_powr(x, y):
    return Decimal("NaN") if d(x) < 0 else ref_pow(x, y)


# name -> (library function, number of arguments, reference)
FUNCS = {
    "exp": ("exp_bits", 1, lambda a: d(a[0]).exp()),
    "exp2": ("exp2_bits", 1, lambda a: (d(a[0]) * Decimal(2).ln()).exp()),
    "exp10": ("exp10_bits", 1, lambda a: (d(a[0]) * Decimal(10).ln()).exp()),
    "expm1": ("expm1_bits", 1, lambda a: ref_expm1(a[0])),
    "log": ("log_bits", 1, lambda a: d(a[0]).ln()),
    "log2": ("log2_bits", 1, lambda a: d(a[0]).ln() / Decimal(2).ln()),
    "log10": ("log10_bits", 1, lambda a: d(a[0]).ln() / Decimal(10).ln()),
    "log1p": ("log1p_bits", 1, lambda a: ref_log1p(a[0])),
    "pow": ("pow_bits", 2, lambda a: ref_pow(a[0], a[1])),
    "sin": ("sin_bits", 1, lambda a: ref_sin(a[0])),
    "cos": ("cos_bits", 1, lambda a: ref_cos(a[0])),
    "tan": ("tan_bits", 1, lambda a: ref_tan(a[0])),
    "atan": ("atan_bits", 1, lambda a: ref_atan(a[0])),
    "atan2": ("atan2_bits", 2, lambda a: ref_atan2(a[0], a[1])),
    "asin": ("asin_bits", 1, lambda a: ref_asin(a[0])),
    "acos": ("acos_bits", 1, lambda a: ref_acos(a[0])),
    "sinh": ("sinh_bits", 1, lambda a: (d(a[0]).exp() - (-d(a[0])).exp()) / 2),
    "cosh": ("cosh_bits", 1, lambda a: (d(a[0]).exp() + (-d(a[0])).exp()) / 2),
    "tanh": ("tanh_bits", 1, lambda a: (lambda e: (e - 1) / (e + 1))((2 * d(a[0])).exp())),
    "asinh": ("asinh_bits", 1, lambda a: (d(a[0]) + (d(a[0]) ** 2 + 1).sqrt()).ln()),
    "acosh": ("acosh_bits", 1, lambda a: (d(a[0]) + (d(a[0]) ** 2 - 1).sqrt()).ln()),
    "atanh": ("atanh_bits", 1, lambda a: ((1 + d(a[0])) / (1 - d(a[0]))).ln() / 2),
    "cbrt": ("cbrt_bits", 1, lambda a: ref_cbrt(a[0])),
    "hypot": ("hypot_bits", 2, lambda a: ref_hypot(a[0], a[1])),
    "remainder": ("remainder_bits", 2, lambda a: ref_remainder(a[0], a[1])),
    "fmod": ("fmod_bits", 2, lambda a: ref_fmod(a[0], a[1])),
    "fract": ("fract_bits", 1, lambda a: ref_fract(a[0])),
    "erf": ("erf_bits", 1, lambda a: ref_erf(a[0])),
    "erfc": ("erfc_bits", 1, lambda a: ref_erfc(a[0])),
    "tgamma": ("tgamma_bits", 1, lambda a: ref_tgamma(a[0])),
    "frexp": ("frexp_f_bits", 1, lambda a: ref_frexp(a[0])),
    "modf": ("modf_f_bits", 1, lambda a: ref_modf(a[0])),
    "sincos": ("sincos_s_bits", 1, lambda a: ref_sincos(a[0])),
    "remquo": ("remquo_r_bits", 2, lambda a: ref_remquo(a[0], a[1])),
    "sinpi": ("sinpi_bits", 1, lambda a: ref_sinpi(a[0])),
    "cospi": ("cospi_bits", 1, lambda a: ref_cospi(a[0])),
    "tanpi": ("tanpi_bits", 1, lambda a: ref_tanpi(a[0])),
    "asinpi": ("asinpi_bits", 1, lambda a: ref_asinpi(a[0])),
    "acospi": ("acospi_bits", 1, lambda a: ref_acospi(a[0])),
    "atanpi": ("atanpi_bits", 1, lambda a: ref_atanpi(a[0])),
    "atan2pi": ("atan2pi_bits", 2, lambda a: ref_atan2pi(a[0], a[1])),
    "sqrt": ("sqrt_bits", 1, lambda a: ref_sqrt(a[0])),
    "copysign": ("copysign_bits", 2, lambda a: ref_copysign(a[0], a[1])),
    "fdim": ("fdim_bits", 2, lambda a: ref_fdim(a[0], a[1])),
    "minmag": ("minmag_bits", 2, lambda a: ref_minmag(a[0], a[1])),
    "maxmag": ("maxmag_bits", 2, lambda a: ref_maxmag(a[0], a[1])),
    "nextafter": ("nextafter_bits", 2, lambda a: ref_nextafter(a[0], a[1])),
    "ldexp": ("ldexp_bits", 2, lambda a: ref_ldexp(a[0], a[1])),
    "pown": ("pown_bits", 2, lambda a: ref_pown(a[0], a[1])),
    "rootn": ("rootn_bits", 2, lambda a: ref_rootn(a[0], a[1])),
    "powr": ("powr_bits", 2, lambda a: ref_powr(a[0], a[1])),
}

# Inputs each function is swept over. The list is the interesting part of a value's
# range rather than a grid: both signs, zero, a subnormal, the edges of the exponent
# range, exact ties, and the arguments that cancel.
ONE_ARG = [
    0.0, -0.0, 1e-320, -1e-320, 1e-300, 1e-20, 1e-8, 0.125, 0.25, 0.375, 0.5,
    0.75, 0.9999999999999999, 1.0, 1.0000000000000002, 1.5, 2.0, 2.718281828459045,
    3.0, 3.141592653589793, 4.0, 5.5, 10.0, 100.0, 1e5, 1e20, 1e100, 1e300,
    -1.5, -2.0, -3.5, -10.0, -100.0, -1e20, -1e300,
    0.0027397260273972603, 1.7976931348623157e308, 5e-324, 2.2250738585072014e-308,
    # A near-multiple of pi/2: the reduction's whole reason for existing.
    1.5707963267948966, 3.141592653589793, 6.283185307179586, 100.00000000000001,
    1e15 + 0.5, 1e22, 1.0e300,
]
LOG_ARGS = [1e-320, 1e-300, 1e-20, 1e-8, 0.125, 0.5, 0.75, 1.0, 1.0000000000000002,
            1.5, 2.0, 3.0, 10.0, 1e5, 1e20, 1e300, 1.7976931348623157e308]
EDGE_PAIRS = [(0.0, 3.0), (-0.0, 3.0), (0.0, -3.0), (-0.0, -3.0), (1.0, 1.0),
              (-1.0, 1.0), (2.5, -2.5), (5e-324, 2.2250738585072014e-308),
              (1.7976931348623157e308, 1e-300), (float("nan"), 3.0),
              (3.0, float("nan")), (float("nan"), float("nan")),
              (float("inf"), 1.0), (1.0, float("inf")), (-float("inf"), float("inf")),
              (7.0, 3.0), (4.5, 3.0), (1e20, 3.0), (1e-300, 1e300)]
TWO_ARG = [(0.5, 3.0), (2.0, 10.0), (2.0, 0.5), (10.0, -2.0), (1e300, 0.0),
           # the half-way ties, where nearest-even decides the sign of the remainder
           (4.5, 3.0), (-4.5, 3.0), (7.5, 3.0), (1.5, 3.0),
           (3.0, 100.0), (1.0000001, 1e6), (0.9999999, 1e6), (2.0, -1074.0),
           (1e-300, 3.0), (7.0, 0.3333333333333333), (1.5, 1.5)]


def inputs_for(name):
    if name in ("log", "log2", "log10"):
        return [(v,) for v in LOG_ARGS]
    if name == "erf" or name == "erfc":
        return [(v,) for v in [0.0, 1e-300, 1e-8, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
                               2.0, 2.5, 3.0, 3.0001, 3.5, 4.0, 5.0, 6.0, 10.0,
                               20.0, 26.0, 27.0, -0.5, -1.5, -2.5, -3.999, -10.5]]
    if name == "tgamma":
        return [(v,) for v in [1e-300, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0,
                               4.5, 10.5, 25.0, 60.0, 150.0, 170.0, 171.7,
                               -0.5, -1.5, -2.5, -7.25, -20.75, -59.473, -100.5,
                               # the poles: the answer has to be a NaN, on both columns,
                               # rather than a number that happens to agree
                               0.0, -0.0, -1.0, -2.0, -3.0, -100.0,
                               # and the ends, +inf and -inf
                               float("inf"), float("-inf"),
                               # a hair from a pole, where the reflection divides by
                               # a sine that is nearly zero: the answer is huge and its
                               # last bits come from what is left after the cancellation
                               -3.0 + 2.220446049250313e-16, -3.0 - 2.220446049250313e-16,
                               -1.0 + 1e-15, -1.0 - 1e-15, -100.0 + 1e-13,
                               -100.0 - 1e-13, -2.0 + 5e-17]]
    if name in ("frexp", "modf", "sincos", "remquo"):
        vals = [0.0, -0.0, 5e-324, 1e-320, 2.2250738585072014e-308, 1e-8, 0.1, 0.25,
                0.5, 0.75, 0.9999999999999999, 1.0, 1.5, 2.0, 3.0, 10.0, 100.0,
                1e20, 1.7976931348623157e308, -0.1, -1.5, -2.0, -3.5, -100.0,
                4.5, -4.5, 7.5,  # the ties, and both signs of the flat remainder
                -1e20, -1.7976931348623157e308]
        if name == "remquo":
            return [(v, 3.0) for v in vals] + [(v, 0.5) for v in vals[:8]]
        return [(v,) for v in vals]
    if name in ("sinpi", "cospi", "tanpi"):
        vals = [0.0, -0.0, 1e-300, 1e-8, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0,
                100.5, 0.3333333333333333, 1e15 + 0.5, 1e300, -0.5, -1.5, -2.5,
                -100.5, 0.9999999999999999]
        if name == "tanpi":
            # A half integer is a pole: tan is infinite from one side and from the
            # other, so there is no one right answer there and nothing to grade.
            vals = [v for v in vals if abs(abs(v) % 1 - 0.5) > 1e-15]
        return [(v,) for v in vals]
    if name in ("asinpi", "acospi"):
        return [(v,) for v in [0.0, -0.0, 1e-300, 1e-8, 0.25, 0.5, 0.75, 0.9,
                               0.9999999999999999, 1.0, -0.25, -0.9, -1.0]]
    if name == "atanpi":
        return [(v,) for v in [0.0, -0.0, 1e-300, 0.5, 1.0, 2.0, 1e8, 1e300,
                               -0.5, -1.0, -1e300, float("inf"), float("-inf")]]
    if name == "atan2pi":
        return [(y, x) for y, x in [(0.0, 1.0), (-0.0, 1.0), (1.0, 0.0), (-1.0, 0.0),
                                    (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (3.0, 4.0),
                                    (1e300, 1e300), (1e-300, -1.0)]]
    if name == "fract":
        return [(v,) for v in [0.0, -0.0, 5e-324, 1e-320, 1e-300, 0.125, 0.5, 0.75,
                               0.9999999999999999, 1.0, 1.0000000000000002, 1.5, 2.0,
                               3.5, 1e20, 1e300, 1.7976931348623157e308,
                               -0.125, -0.5, -1.5, -2.5, -1e300,
                               -1.7976931348623157e308,
                               float("inf"), float("-inf"), float("nan")]]
    if name in ("copysign", "fdim", "minmag", "maxmag", "nextafter"):
        return EDGE_PAIRS
    if name == "sqrt":
        return [(v,) for v in [0.0, -0.0, 5e-324, 1e-300, 1e-8, 0.5, 1.0, 2.0, 4.0,
                               1e8, 1e300, 1.7976931348623157e308,
                               2.2250738585072014e-308, -0.5, -1.0, -1e300,
                               -1.7976931348623157e308, float("inf")]]
    if name == "ldexp":
        return [(v, n) for v in [0.0, 1.0, -1.5, 0.1, 1e-300, 1e300, 5e-324,
                                 1.7976931348623157e308]
                for n in (0, 1, -1, 12, -12, 1024, -1074, 1075, -2000)]
    if name == "pown":
        return [(v, n) for v in [0.5, 1.5, 2.0, -2.0, 3.0, 1e-10, 1e10, -0.5]
                for n in (0, 1, 2, 3, -1, -2, -3, 5, 7)]
    if name == "rootn":
        return [(v, n) for v in [0.0, 5e-324, 1e-300, 0.5, 1.0, 2.0, 8.0, 27.0,
                                 1e100, 1e300, 1.7976931348623157e308, -8.0, -27.0]
                for n in (1, 2, 3, 4, 5, -3)]
    if name == "powr":
        return [(x, y) for x in [5e-324, 1e-300, 0.5, 1.0, 2.0, 10.0, 1e100, 1e300]
                for y in (0.0, 0.5, 1.5, -1.5, 2.0)]
    if name in ("log1p", "acosh"):
        return [(v,) for v in [0.0, 1e-20, 1e-8, 0.5, 1.0, 2.0, 1e20, 1e300]]
    if name in ("asin", "acos", "atanh"):
        return [(v,) for v in [0.0, 1e-300, 1e-8, 0.25, 0.5, 0.75, 0.9,
                               -0.25, -0.9, 1.0, -1.0, 0.9999999999999999]]
    if name == "tan":
        return [(v,) for v in [0.0, 1e-300, 0.5, 0.7853981633974483, 1.0,
                               1.5707963267948966, 3.141592653589793, 1e10, -0.5]]
    if name in ("pow", "atan2", "hypot", "fmod", "remainder", "nextafter"):
        if name == "atan2":
            return [(y, x) for y, x in [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0),
                                        (-1.0, 0.0), (1.0, 1.0), (-1.0, -1.0),
                                        (1e300, 1e300), (1e-300, -1.0),
                                        (3.0, 4.0), (-3.0, 4.0), (3.0, -4.0)]]
        if name == "atan2_bits":
            return []
        return TWO_ARG
    if name == "acosh":
        return [(v,) for v in [1.0, 1.5, 2.0, 10.0, 1e20, 1e300]]
    return [(v,) for v in ONE_ARG]


def library_text():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "weft_f64.hpp")).read()
    m = re.search(r'R"WEFFF64\((.*)\)WEFFF64"', src, re.S)
    if not m:
        sys.exit("tools/f64_math.py: could not find the library text in src/weft_f64.hpp")
    return m.group(1)


HARNESS = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// The two words Metal spells its address spaces with, for the length of the library.
#define thread
#define constant const
template <class T, class U>
static inline T as_type(U u) { T t; std::memcpy(&t, &u, sizeof(T)); return t; }

%s

static uint64_t bits_of(double v) { uint64_t b; std::memcpy(&b, &v, 8); return b; }
static double of_bits(uint64_t b) { double v; std::memcpy(&v, &b, 8); return v; }

int main() {
  char line[512];
  while (std::fgets(line, sizeof line, stdin)) {
    char fn[64];
    char a[128], b[128];
    a[0] = b[0] = 0;
    const int n = std::sscanf(line, "%%63s %%127s %%127s", fn, a, b);
    if (n < 2) continue;
    const uint64_t x = bits_of(std::strtod(a, nullptr));
    const uint64_t y = bits_of(std::strtod(b, nullptr));
    uint64_t r = 0, r2 = 0;
    const std::string f(fn);
    if (f == "exp") r = weft_fp64::exp_bits(x);
    else if (f == "exp2") r = weft_fp64::exp2_bits(x);
    else if (f == "exp10") r = weft_fp64::exp10_bits(x);
    else if (f == "expm1") r = weft_fp64::expm1_bits(x);
    else if (f == "log") r = weft_fp64::log_bits(x);
    else if (f == "log2") r = weft_fp64::log2_bits(x);
    else if (f == "log10") r = weft_fp64::log10_bits(x);
    else if (f == "log1p") r = weft_fp64::log1p_bits(x);
    else if (f == "sin") r = weft_fp64::sin_bits(x);
    else if (f == "cos") r = weft_fp64::cos_bits(x);
    else if (f == "tan") r = weft_fp64::tan_bits(x);
    else if (f == "atan") r = weft_fp64::atan_bits(x);
    else if (f == "asin") r = weft_fp64::asin_bits(x);
    else if (f == "acos") r = weft_fp64::acos_bits(x);
    else if (f == "sinh") r = weft_fp64::sinh_bits(x);
    else if (f == "cosh") r = weft_fp64::cosh_bits(x);
    else if (f == "tanh") r = weft_fp64::tanh_bits(x);
    else if (f == "asinh") r = weft_fp64::asinh_bits(x);
    else if (f == "acosh") r = weft_fp64::acosh_bits(x);
    else if (f == "atanh") r = weft_fp64::atanh_bits(x);
    else if (f == "erf") r = weft_fp64::erf_bits(x);
    else if (f == "erfc") r = weft_fp64::erfc_bits(x);
    else if (f == "tgamma") r = weft_fp64::tgamma_bits(x);
    else if (f == "cbrt") r = weft_fp64::cbrt_bits(x);
    else if (f == "pow") r = weft_fp64::pow_bits(x, bits_of(std::strtod(b, nullptr)));
    else if (f == "atan2") r = weft_fp64::atan2_bits(x, bits_of(std::strtod(b, nullptr)));
    else if (f == "hypot") r = weft_fp64::hypot_bits(x, bits_of(std::strtod(b, nullptr)));
    else if (f == "fmod") r = weft_fp64::fmod_bits(x, bits_of(std::strtod(b, nullptr)));
    else if (f == "fract") r = weft_fp64::fract_bits(x);
    else if (f == "remainder") r = weft_fp64::remainder_bits(x, bits_of(std::strtod(b, nullptr)));
    else if (f == "nextafter") r = weft_fp64::nextafter_bits(x, y);
    else if (f == "frexp") {
      r = weft_fp64::frexp_f_bits(x);
      r2 = bits_of((double)(int64_t)weft_fp64::frexp_e_bits(x));
    }
    else if (f == "modf") {
      r = weft_fp64::modf_f_bits(x);
      r2 = weft_fp64::modf_i_bits(x);
    }
    else if (f == "sincos") {
      r = weft_fp64::sincos_s_bits(x);
      r2 = weft_fp64::sincos_c_bits(x);
    }
    else if (f == "remquo") {
      r = weft_fp64::remquo_r_bits(x, y);
      r2 = bits_of((double)(int64_t)weft_fp64::remquo_q_bits(x, y));
    }
    else if (f == "sinpi") r = weft_fp64::sinpi_bits(x);
    else if (f == "cospi") r = weft_fp64::cospi_bits(x);
    else if (f == "tanpi") r = weft_fp64::tanpi_bits(x);
    else if (f == "asinpi") r = weft_fp64::asinpi_bits(x);
    else if (f == "acospi") r = weft_fp64::acospi_bits(x);
    else if (f == "atanpi") r = weft_fp64::atanpi_bits(x);
    else if (f == "atan2pi") r = weft_fp64::atan2pi_bits(x, y);
    else if (f == "sqrt") r = weft_fp64::sqrt_bits(x);
    else if (f == "copysign") r = weft_fp64::copysign_bits(x, y);
    else if (f == "fdim") r = weft_fp64::fdim_bits(x, y);
    else if (f == "minmag") r = weft_fp64::minmag_bits(x, y);
    else if (f == "maxmag") r = weft_fp64::maxmag_bits(x, y);
    else if (f == "ldexp") r = weft_fp64::ldexp_bits(x, (int32_t)std::strtod(b, nullptr));
    else if (f == "pown") r = weft_fp64::pown_bits(x, (int32_t)std::strtod(b, nullptr));
    else if (f == "rootn") r = weft_fp64::rootn_bits(x, (int32_t)std::strtod(b, nullptr));
    else if (f == "powr") r = weft_fp64::powr_bits(x, y);
    std::printf("%%016llx %%016llx\n", (unsigned long long)r, (unsigned long long)r2);
  }
  return 0;
}
"""


def build():
    tmp = tempfile.mkdtemp(prefix="f64math.")
    src = os.path.join(tmp, "harness.cpp")
    exe = os.path.join(tmp, "harness")
    open(src, "w").write(HARNESS % library_text())
    cmd = ["clang++", "-std=c++20", "-O2", src, "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("the library did not compile as C++:\n" + r.stderr[:4000])
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()
    exe = build()

    # One process for the whole sweep: the harness reads a line and prints a word.
    lines, plan = [], []
    for name, (fn, arity, ref) in FUNCS.items():
        if args.only and name != args.only:
            continue
        for argv in inputs_for(name):
            lines.append(" ".join([name] + [repr(float(v)) for v in argv]))
            plan.append((name, argv, ref))
    env = dict(os.environ)
    r = subprocess.run([exe], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, env=env)
    out = [ln.split() for ln in r.stdout.splitlines() if ln.strip()]
    if len(out) != len(plan):
        sys.exit(f"the harness answered {len(out)} of {len(plan)} lines:\n{r.stderr[:2000]}")

    def unhex(word):
        return struct.unpack("<d", struct.pack("<Q", int(word, 16)))[0]

    worst = {}
    bad = []
    for (name, argv, ref), words in zip(plan, out):
        got = unhex(words[0])
        try:
            want = ref(argv)
        except (ValueError, ArithmeticError, decimal.InvalidOperation):
            continue
        if want is None:
            continue
        second, got2 = None, None
        if isinstance(want, tuple):
            want, second = want
            got2 = unhex(words[1]) if len(words) > 1 else float("nan")
        if _want_nan(want):
            # A pole, or a negative infinity: the answer is not a number, and the
            # library has to say so. A finite answer here is a failure even when it
            # looks plausible, so it is counted rather than skipped.
            if not (isinstance(got, float) and math.isnan(got)):
                bad.append((name, argv, got, want, float("inf")))
            continue
        u = ulps(got, want)
        if isinstance(second, int):
            # An exponent or a quotient: an integer, compared exactly, not in ulps.
            if got2 != second:
                bad.append((name, argv, got2, Decimal(second), float("inf")))
                u = float("inf")
        elif second is not None:
            u = max(u, ulps(got2, second))
        if name not in worst or u > worst[name][0]:
            worst[name] = (u, argv, got, want)
        if u > 1:
            bad.append((name, argv, got, float(want), u))

    print(f"{'function':10s} {'worst ulp':>9s}   input")
    for name in FUNCS:
        if name not in worst:
            continue
        u, argv, got, want = worst[name]
        if args.verbose or u > 0:
            print(f"{name:10s} {u:9.0f}   {argv} -> {got!r}")
    top = max((w[0] for w in worst.values()), default=0)
    print(f"\n{len(worst)} functions swept, worst {top:.0f} ulp")
    if bad:
        print("\nfailures (more than one ulp):")
        for name, argv, got, want, u in bad[:20]:
            print(f"  {name}({argv}) = {got!r}, exact {want!r}, {u:.0f} ulp off")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
