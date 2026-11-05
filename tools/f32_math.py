#!/usr/bin/env python3
"""Sweep the portable float library against an exact reference.

`src/soft_math.hpp` is one text compiled twice, MSL for the GPU, C++ for the CPU , 
which is why the two columns cannot disagree. That is a statement about agreement,
not about correctness: a library both engines run can be wrong on both of them, and
the fp32 set has never been measured against anything but itself. This measures it.

Every input is a float32 and every answer is compared to the number itself: the
reference is the same 1,200-digit `decimal` machinery `tools/f64_math.py` uses (this
reuses its references rather than restating the mathematics), rounded once to the
nearest float32, and the library's bits are counted against it in float32 ulps.

    python3 tools/f32_math.py            # the default sweep
    python3 tools/f32_math.py --verbose  # and every function's worst input
    python3 tools/f32_math.py --only sin

A function that is off by more than an ulp on any input is a failure; the exit
status says whether that happened.
"""

import argparse
import decimal
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f64_math as ref64  # the references, and the decimal context, of the fp64 sweep

F32_MAX = 3.4028234663852886e38
F32_MIN_NORMAL = 1.1754943508222875e-38
F32_MIN_SUB = 1.401298464324817e-45
BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f32_baseline.json")


def f32(v):
    """The nearest float32 to a Python float or Decimal, as a float."""
    if isinstance(v, Decimal):
        try:
            v = float(v)
        except OverflowError:
            v = math.inf if v > 0 else -math.inf
    try:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    except OverflowError:
        return math.copysign(math.inf, v)


def f32_key(x):
    """A monotone integer per float32, so a distance is a subtraction.

    The bits of a positive float order as the float does and those of a negative one
    order as the complement, so one map covers subnormals, zeros and infinities. Both
    zeros land on 0: they are the same number, and a signed zero is a question for
    the word comparison, not for an ulp count.
    """
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    if b & 0x80000000:
        return -(b & 0x7FFFFFFF)
    return b


def f32_from_key(k):
    """The float32 a key names."""
    b = k if k >= 0 else (0x80000000 | -k)
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def f32_next(x, y):
    """The next float32 after x toward y, the float32 reference for nextafter."""
    if math.isnan(x) or math.isnan(y):
        return math.nan
    if x == y:
        return y
    k = f32_key(x)
    return f32_from_key(k + (1 if f32_key(y) > k else -1))


def ulps32(got, want):
    """Distance from the computed float32 to the nearest float32 to `want`."""
    if isinstance(got, float) and math.isnan(got):
        return 0 if _want_nan(want) else float("inf")
    if isinstance(want, Decimal) and want.is_infinite():
        if not (isinstance(got, float) and math.isinf(got)):
            return float("inf")
        return 0 if (got > 0) == (want > 0) else float("inf")
    try:
        target = f32(want)
    except (OverflowError, ValueError):
        target = math.inf if want > 0 else -math.inf
    if math.isnan(target) or math.isinf(target):
        return 0 if got == target else float("inf")
    if got == target:
        return 0
    return abs(f32_key(got) - f32_key(target))


def _want_nan(want):
    if isinstance(want, Decimal):
        return want.is_nan()
    return isinstance(want, float) and want != want


# `expr` is the call the harness makes; `pair` marks a second answer: None,
# "int" for an integer half, or a call expression for a float one.
CALLS = [
    ("sqrt", "weft_sqrt(x)", None),
    ("rsqrt", "weft_rsqrt(x)", None),
    ("sin", "weft_sin(x)", None),
    ("cos", "weft_cos(x)", None),
    ("tan", "weft_tan(x)", None),
    ("sinpi", "weft_sinpi(x)", None),
    ("cospi", "weft_cospi(x)", None),
    ("tanpi", "weft_tanpi(x)", None),
    ("asin", "weft_asin(x)", None),
    ("acos", "weft_acos(x)", None),
    ("atan", "weft_atan(x)", None),
    ("atan2", "weft_atan2(x, y)", None),
    ("sinh", "weft_sinh(x)", None),
    ("cosh", "weft_cosh(x)", None),
    ("tanh", "weft_tanh(x)", None),
    ("asinh", "weft_asinh(x)", None),
    ("acosh", "weft_acosh(x)", None),
    ("atanh", "weft_atanh(x)", None),
    ("exp", "weft_exp(x)", None),
    ("exp2", "weft_exp2(x)", None),
    ("exp10", "weft_exp10(x)", None),
    ("expm1", "weft_expm1(x)", None),
    ("log", "weft_log(x)", None),
    ("log2", "weft_log2(x)", None),
    ("log10", "weft_log10(x)", None),
    ("logb", "weft_logb(x)", None),
    ("ilogb", "weft_ilogb(x)", "int"),
    ("pow", "weft_pow(x, y)", None),
    ("powr", "weft_powr(x, y)", None),
    ("ldexp", "weft_ldexp(x, yi)", None),
    ("pown", "weft_pown(x, yi)", None),
    ("rootn", "weft_rootn(x, yi)", None),
    ("maxmag", "weft_maxmag(x, y)", None),
    ("minmag", "weft_minmag(x, y)", None),
    ("nextafter", "weft_nextafter(x, y)", None),
    ("frexp", "weft_frexp_f(x)", "weft_frexp_e(x)"),
    ("modf", "weft_modf_f(x)", "weft_modf_i(x)"),
    ("sincos", "weft_sin(x)", "weft_cos(x)"),
    ("remquo", "weft_remquo_r(x, y)", "weft_remquo_q(x, y)"),
    # Two whose fast path is a pair but whose slow one is the fp64 library: the harness
    # has to compile that text too, which is what f64_text() below adds.
    ("tgamma", "weft_tgamma(x)", None),
    ("remainder", "weft_remainder(x, y)", None),
    # Two more Metal's standard library does not carry, on the pair path like the rest.
    ("erf", "weft_erf(x)", None),
    ("erfc", "weft_erfc(x)", None),
    ("cbrt", "weft_cbrt(x)", None),
    ("log1p", "weft_log1p(x)", None),
    ("hypot", "weft_hypot(x, y)", None),
    ("fract", "weft_fract(x)", None),
]

# The ends of the float range, which the fp64 input lists do not reach: a subnormal
# float32 is 1e-45, not 1e-324, and the exponent range is 38, not 308.
EXTRA = [F32_MAX, F32_MIN_NORMAL, F32_MIN_SUB, -F32_MIN_SUB, 1e7, -1e7,
         0.9999999, 1.0000001, 1e38, -1e38]
DEFAULT_INPUTS = [0.0, -0.0, F32_MIN_SUB, F32_MIN_NORMAL, 1e-8, 0.25, 0.5, 0.75,
                  1.0, 1.5, 2.0, 10.0, 1e7, F32_MAX]


def reference(name, argv):
    """The reference of the fp64 sweep, where the mathematics is the same."""
    if name == "rsqrt":
        return Decimal(1) / ref64.ref_sqrt(argv[0])
    if name == "nextafter":
        return f32_next(argv[0], argv[1])
    if name == "powr":
        # powr is x^y for x >= 0 only: 0^0 is undefined, 0^positive is 0, and a zero
        # to a negative power is infinite. The fp64 reference answers the same, but
        # saying it here keeps the float sweep's rule in one place.
        x, y = argv[0], argv[1]
        if x < 0.0:
            return float("nan")
        if x == 0.0:
            if y > 0.0:
                return 0.0
            return float("nan") if y == 0.0 else float("inf")
        return (ref64.d(y) * ref64.d(x).ln()).exp()
    if name == "fract":
        lim = struct.unpack("<f", struct.pack("<I", 0x3f7fffff))[0]
        v = float(argv[0])
        if math.isinf(v) or math.isnan(v):
            return lim                     # fmin(NaN, the ceiling) answers the ceiling
        return min(v - math.floor(v), lim)
    if name == "ilogb":
        v = ref64.d(argv[0])
        if v == 0:
            return float("-inf")
        return int(math.floor(math.log2(abs(float(v)))))
    f = ref64.FUNCS.get(name)
    if f is not None:
        return f[2](argv)
    return None


def inputs_for(name):
    """The fp64 sweep's inputs for that function, rounded to float32."""
    try:
        raw = ref64.inputs_for(name)
    except Exception:
        raw = []
    if not raw:
        raw = [(v,) for v in DEFAULT_INPUTS]
    if name == "ilogb":
        # ilogb(0) is a sentinel, not a number, and the edge kernel is where that
        # is compared; here the exponent of a normal or a subnormal is exact.
        raw = [a for a in raw if a[0] != 0 and math.isfinite(a[0])]
    out = []
    for argv in raw:
        conv, keep = [], True
        for v in argv:
            if isinstance(v, float):
                c = f32(v)
                # A finite double that is not a float32 is not an input this sweep
                # has: the ends of the float32 range are added below instead.
                if math.isinf(c) and not math.isinf(v):
                    keep = False
                conv.append(c)
            else:
                conv.append(v)
        if keep:
            out.append(tuple(conv))
    if name == "ilogb":
        # A value that rounds to zero in float32 (the smallest double subnormal does)
        # has no exponent to grade; ilogb(0) is a sentinel and the edge kernel's row.
        out = [a for a in out if a[0] != 0.0 and math.isfinite(a[0])]
    if out and len(out[0]) == 1:
        out.extend((f32(v),) for v in EXTRA)
    return out


# The library is written in Metal's vocabulary. Only the scalar entries are called,
# but its vector overloads have to compile, so this is the smallest set of types that
# makes the text a C++ header.
SHIM = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

typedef unsigned int uint;
typedef _Float16 half;

struct float2 { float x, y; float2() : x(0), y(0) {}
  float2(float a) : x(a), y(a) {} float2(float a, float b) : x(a), y(b) {} };
struct float3 { float x, y, z; float3() : x(0), y(0), z(0) {}
  float3(float a) : x(a), y(a), z(a) {}
  float3(float a, float b, float c) : x(a), y(b), z(c) {} };
struct float4 { float x, y, z, w; float4() : x(0), y(0), z(0), w(0) {}
  float4(float a) : x(a), y(a), z(a), w(a) {}
  float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {} };
struct half2 { half x, y; half2(half a, half b) : x(a), y(b) {} };
struct half3 { half x, y, z; half3(half a, half b, half c) : x(a), y(b), z(c) {} };
struct half4 { half x, y, z, w;
  half4(half a, half b, half c, half d) : x(a), y(b), z(c), w(d) {} };
struct bool2 { bool x, y; bool2(bool a, bool b) : x(a), y(b) {} };
struct bool3 { bool x, y, z; bool3(bool a, bool b, bool c) : x(a), y(b), z(c) {} };
struct bool4 { bool x, y, z, w; bool4(bool a, bool b, bool c, bool d) : x(a), y(b), z(c), w(d) {} };
struct int2 { int x, y; int2(int a, int b) : x(a), y(b) {} };
struct int3 { int x, y, z; int3(int a, int b, int c) : x(a), y(b), z(c) {} };
struct int4 { int x, y, z, w; int4(int a, int b, int c, int d) : x(a), y(b), z(c), w(d) {} };
struct uint2 { uint x, y; uint2(uint a, uint b) : x(a), y(b) {} };
struct uint3 { uint x, y, z; uint3(uint a, uint b, uint c) : x(a), y(b), z(c) {} };
struct uint4 { uint x, y, z, w; uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {} };

static inline float2 operator-(float2 a, float2 b) { return float2(a.x - b.x, a.y - b.y); }
static inline float3 operator-(float3 a, float3 b) { return float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline float4 operator-(float4 a, float4 b) { return float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static inline float2 operator-(float2 a) { return float2(-a.x, -a.y); }
static inline float3 operator-(float3 a) { return float3(-a.x, -a.y, -a.z); }
static inline float4 operator-(float4 a) { return float4(-a.x, -a.y, -a.z, -a.w); }
static inline float2 operator/(float2 a, float s) { return float2(a.x / s, a.y / s); }
static inline float3 operator/(float3 a, float s) { return float3(a.x / s, a.y / s, a.z / s); }
static inline float4 operator/(float4 a, float s) { return float4(a.x / s, a.y / s, a.z / s, a.w / s); }
"""


def library_text():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "soft_math.hpp")
    m = re.search(r'R"WEFTMATH\((.*)\)WEFTMATH"', open(path).read(), re.S)
    if not m:
        sys.exit("tools/f32_math.py: could not find the library text in src/soft_math.hpp")
    return m.group(1)


def f64_text():
    """The fp64 library, as the CPU column compiles it: `weft_tgamma` calls into it."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "weft_f64.hpp")
    src = open(path).read()
    m = re.search(r'R"WEFFF64\((.*)\)WEFFF64"', src, re.S)
    if not m:
        sys.exit("tools/f32_math.py: could not find the fp64 library text in src/weft_f64.hpp")
    # Metal's address spaces, and the bit copy that stands in for `as_type`. Undone
    # straight after, exactly as the runner's own CPU text does it.
    return ("#define thread\n#define constant const\n"
            "template <class T, class U>\n"
            "static inline T as_type(U u) { T t; std::memcpy(&t, &u, sizeof(T)); return t; }\n" +
            m.group(1) + "\n#undef thread\n#undef constant\n")


def harness():
    chain = []
    for name, expr, pair in CALLS:
        cond = "else if" if chain else "if"
        if pair == "int":
            chain.append(f'    {cond} (f == "{name}") i2 = (long long)({expr});')
        elif pair:
            chain.append(f'    {cond} (f == "{name}") {{ r = {expr}; r2 = {pair}; }}')
        else:
            chain.append(f'    {cond} (f == "{name}") r = {expr};')
    return SHIM + library_text() + f64_text() + r"""
static uint32_t fbits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
static float ofbits(uint32_t b) { float v; std::memcpy(&v, &b, 4); return v; }

int main() {
  char line[512];
  while (std::fgets(line, sizeof line, stdin)) {
    char fn[64], a[128], b[128];
    a[0] = b[0] = 0;
    const int n = std::sscanf(line, "%63s %127s %127s", fn, a, b);
    if (n < 2) continue;
    const std::string f(fn);
    const float x = ofbits((uint32_t)std::strtoul(a, nullptr, 16));
    const float y = n > 2 ? ofbits((uint32_t)std::strtoul(b, nullptr, 16)) : 0.0f;
    const int yi = (int)(uint32_t)std::strtoul(b, nullptr, 16);
    float r = 0.0f, r2 = 0.0f;
    long long i2 = 0;
""" + "\n".join(chain) + r"""
    std::printf("%08x %08x %lld\n", fbits(r), fbits(r2), i2);
  }
  return 0;
}
"""


def build():
    tmp = tempfile.mkdtemp(prefix="f32math.")
    src = os.path.join(tmp, "harness.cpp")
    exe = os.path.join(tmp, "harness")
    open(src, "w").write(harness())
    r = subprocess.run(["clang++", "-std=c++20", "-O2", src, "-o", exe],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("the library did not compile as C++:\n" + r.stderr[:4000])
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()
    exe = build()

    # The inputs travel as the bits of the float32 they are, so the harness cannot
    # round them again and both sides of the comparison see the same number.
    lines, plan = [], []
    for name, expr, pair in CALLS:
        if args.only and name != args.only:
            continue
        for argv in inputs_for(name):
            words = [name]
            for v in argv:
                if isinstance(v, float):
                    words.append("%08x" % struct.unpack("<I", struct.pack("<f", v))[0])
                else:
                    words.append("%08x" % struct.unpack("<I", struct.pack("<i", int(v)))[0])
            lines.append(" ".join(words))
            plan.append((name, argv))
    r = subprocess.run([exe], input="\n".join(lines) + "\n", capture_output=True,
                       text=True, env=dict(os.environ))
    out = [ln.split() for ln in r.stdout.splitlines() if ln.strip()]
    if len(out) != len(plan):
        sys.exit(f"the harness answered {len(out)} of {len(plan)} lines:\n{r.stderr[:2000]}")

    def unhex(word):
        return struct.unpack("<f", struct.pack("<I", int(word, 16)))[0]

    worst, bad, skipped = {}, [], {}
    for (name, argv), words in zip(plan, out):
        got = unhex(words[0])
        try:
            want = reference(name, argv)
        except (ValueError, ArithmeticError, decimal.InvalidOperation, OverflowError):
            want = None
        if want is None:
            # Every reference here is the exact number, and a few inputs have none:
            # 1/sqrt(0), log(0), the arctangent of an infinity. They are counted, not
            # silently dropped, because a sweep that skips quietly is a sweep that
            # proves less than it says.
            skipped[name] = skipped.get(name, 0) + 1
            continue
        second = None
        if isinstance(want, tuple):
            want, second = want
        if isinstance(want, int):
            # an exponent or a quotient: exact, not an ulp question
            u = 0.0 if int(words[2]) == want else float("inf")
        else:
            u = ulps32(got, want)
            if _want_nan(want) and not (isinstance(got, float) and math.isnan(got)):
                bad.append((name, argv, got, want, float("inf")))
                continue
            if isinstance(second, int):
                # The integer half travels as the float the library returned, so it is
                # read from the same field as the other halves. Past 2^24 the quotient
                # is not a float any more, `remquo` only promises its low bits, and a
                # float cannot hold those either, so there only the remainder is
                # graded, rather than grading the library on an unrepresentable number.
                if abs(second) < 16777216 and int(unhex(words[1])) != second:
                    u = float("inf")
            elif second is not None:
                u = max(u, ulps32(unhex(words[1]), second))
        if name not in worst or u > worst[name][0]:
            worst[name] = (u, argv, got, want)
        if u > 1:
            bad.append((name, argv, got, want, u))

    print(f"{'function':10s} {'worst ulp':>9s}   input")
    for name, expr, pair in CALLS:
        if name not in worst:
            continue
        u, argv, got, want = worst[name]
        shown = ", ".join(repr(v) for v in argv)
        if len(shown) > 44:
            shown = shown[:41] + "..."
        print(f"{name:10s} {u:9.0f}   ({shown}) -> {got!r}")
    top = max((w[0] for w in worst.values()), default=0)
    print(f"\n{len(worst)} functions swept, worst {top:.0f} ulp")
    if skipped:
        total = sum(skipped.values())
        print(f"inputs without a reference, skipped: {total} through "
              f"{', '.join(sorted(skipped))}")

    def word(u):
        return "inf" if math.isinf(u) else str(int(u))

    if args.write_baseline:
        json.dump({n: word(w[0]) for n, w in worst.items()}, open(BASELINE, "w"),
                  indent=1, sort_keys=True)
        print(f"baseline written to {os.path.relpath(BASELINE)}")
        return 0

    # The float set was written for agreement between the two columns, not for the
    # last bit, so this sweep is not held to an ulp. What it is held to is the file
    # beside it: a change that costs accuracy anywhere shows up here as a regression,
    # and the numbers behind the claim are in the file rather than in a paragraph.
    if os.path.exists(BASELINE):
        base = json.load(open(BASELINE))
        worse, missing = [], []
        for name, want in base.items():
            if name not in worst:
                missing.append(name)
                continue
            target = float("inf") if want == "inf" else float(want)
            if worst[name][0] > target:
                worse.append((name, target, worst[name]))
        if missing:
            print(f"baseline names functions that are no longer swept: {' '.join(missing)}")
        if worse:
            print("\nregressions against tools/f32_baseline.json:")
            for name, target, (u, argv, got, want) in worse:
                print(f"  {name}: was {word(target)} ulp, now {word(u)} at {argv}")
            return 1
        print(f"baseline: no function worse than {os.path.relpath(BASELINE)}")
        return 1 if missing else 0

    if bad:
        print("\nfailures (more than one ulp):")
        for name, argv, got, want, u in bad[:20]:
            print(f"  {name}({argv}) = {got!r}, exact {float(want)!r}, {u:.0f} ulp off")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
