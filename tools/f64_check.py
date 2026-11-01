#!/usr/bin/env python3
"""Compare the software fp64 in the Metal preamble against this machine's own doubles.

A GPU has no fp64 here, so the preamble carries a double in software. "In software"
is only worth anything if it is exact, and the way to know is to ask for the same
expression twice: once in the shader, once on the host in Python, whose floats are
IEEE-754 binary64 themselves. Every case is compared as *bits*, so a difference of
one ulp shows up.

    python3 tools/f64_check.py            # every operation
    python3 tools/f64_check.py add div    # only those

Each operation runs as one dispatch over all the cases at once (one thread per case),
so the whole sweep is a handful of pipeline builds rather than one per value.
"""

import math
import os
import random
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_SRC = os.path.join(ROOT, "tools", "f64_probe.mm")
PROBE = os.path.join(tempfile.gettempdir(), "weft-f64probe")
WEFT = os.path.join(ROOT, "build", "weft")


def build_probe():
    if os.path.exists(PROBE) and os.path.getmtime(PROBE) > os.path.getmtime(PROBE_SRC):
        return True
    cxx = "/usr/bin/clang++" if os.path.exists("/usr/bin/clang++") else "clang++"
    r = subprocess.run([cxx, "-fobjc-arc", "-std=c++17", "-O1", "-framework", "Metal",
                        "-framework", "Foundation", PROBE_SRC, "-o", PROBE],
                       capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace"))
        return False
    return True


def preamble():
    """Everything the emitter puts above the kernel: the helpers, unmodified."""
    r = subprocess.run([WEFT, "run", "kernels/spirv/fp64.spv", "--kernel", "fp64", "--emit-gpu"],
                       capture_output=True, cwd=ROOT)
    src = r.stdout.decode()
    cut = src.find("kernel void")
    if cut < 0:
        sys.stderr.write("could not read the emitted preamble:\n" + src[:2000])
        sys.exit(1)
    return src[:cut]


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def dbl(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


INF = float("inf")
NAN = float("nan")
MINSUB = 5e-324            # the smallest subnormal
MAXF = 1.7976931348623157e308

# Each case is a pair of doubles. They cover the shapes the arithmetic has to get
# right: exact sums, inexact quotients, a tie that rounds one way and a tie that
# rounds the other, cancellation, both zeros, the top and bottom of the range, the
# subnormals, and the specials.
CASES = [
    (1.0, 2.0), (1.0, 1.0 / 3.0), (0.5, 0.25), (-1.5, 2.5), (1.0, -1.0),
    (3.0, 7.0), (1e300, 1e-300), (1e300, -1e300), (MAXF, MAXF), (MAXF, -MAXF),
    (float(2 ** 53), 1.0), (float(2 ** 53) + 2.0, 1.0), (1.5, 0.5), (2.5, 0.5),
    (0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (MINSUB, MINSUB), (MINSUB, 0.0),
    (2.0 ** -1022, 2.0 ** -1022), (1e-308, 1e-308), (INF, 1.0), (INF, -INF),
    (NAN, 1.0), (1.0, NAN), (1e16, 1.0), (-1e-300, 3.0),
]

# And a few hundred values nobody chose by hand: the point of a rounding bug is that
# it hides in the case you did not think of. Seeded, so a failure is reproducible.
_rng = random.Random(1234)
for _ in range(240):
    while True:
        a = dbl(_rng.getrandbits(64))
        b = dbl(_rng.getrandbits(64))
        if not (math.isnan(a) or math.isnan(b)):
            break
    CASES.append((a, b))


def kernel(op_expr):
    return ("kernel void probe(device uint64_t* in [[buffer(0)]],\n"
            "                    device uint64_t* out [[buffer(1)]],\n"
            "                    uint id [[thread_position_in_grid]]) {\n"
            "  const uint64_t a = in[2u * id];\n"
            "  const uint64_t b = in[2u * id + 1u];\n"
            "  out[id] = " + op_expr + ";\n}\n")


def expect_add(a, b):
    return bits(a + b)


def expect_sub(a, b):
    return bits(a - b)


def expect_mul(a, b):
    return bits(a * b)


def expect_div(a, b):
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return 0x7FF8000000000000
        return bits(math.copysign(INF, a) * math.copysign(1.0, b))
    return bits(a / b)


def expect_lt(a, b):
    if math.isnan(a) or math.isnan(b):
        return 0
    return 1 if a < b else 0


def expect_eq(a, b):
    return 0 if (math.isnan(a) or math.isnan(b)) else (1 if a == b else 0)


def expect_floor(a, _b):
    if math.isnan(a) or math.isinf(a):
        return bits(a)
    return bits(signed_zero(float(math.floor(a)), a))


def expect_ceil(a, _b):
    if math.isnan(a) or math.isinf(a):
        return bits(a)
    return bits(signed_zero(float(math.ceil(a)), a))


def signed_zero(x, a):
    # Truncation and the roundings keep the sign of what they started from: trunc of
    # -0.5 is -0.0 and ceil of -0.5 is -0.0, not +0.0.
    return math.copysign(0.0, a) if x == 0.0 else x


def expect_trunc(a, _b):
    if math.isnan(a) or math.isinf(a):
        return bits(a)
    return bits(signed_zero(float(math.trunc(a)), a))


def expect_round(a, _b):
    # OpenCL's round: nearest, ties away from zero.
    if math.isnan(a) or math.isinf(a):
        return bits(a)
    t = float(math.trunc(a))
    if abs(a - t) >= 0.5:
        t += math.copysign(1.0, a)
    return bits(signed_zero(t, a))


def expect_from_f32(a, _b):
    # The operand is a binary32 pattern in the low half, so it is the words of the
    # double being reinterpreted, not the double's value.
    fb = struct.unpack("<f", struct.pack("<I", bits(a) & 0xFFFFFFFF))[0]
    return bits(float(fb))


def expect_to_f32(a, _b):
    # The operand is the double itself; the result is its binary32 pattern. Above
    # binary32's range the conversion is an infinity, which Python refuses to pack.
    try:
        return struct.unpack("<I", struct.pack("<f", a))[0]
    except OverflowError:
        return 0xFF800000 if a < 0 else 0x7F800000


def expect_from_u64(a, _b):
    # The operand is the integer in the word, so its value is the bits themselves.
    return bits(float(bits(a)))


def expect_to_u64(a, _b):
    # SPIR-V leaves a conversion whose value does not fit undefined, so the cases
    # outside the range are not checked: there is no right answer to compare with.
    if math.isnan(a) or math.isinf(a) or a < 0.0 or a >= 2.0 ** 64:
        return None
    return int(math.trunc(a))


def expect_iden(a, _b):
    return bits(a)


def expect_isinf(a, _b):
    return 1 if math.isinf(a) else 0


def expect_isnan(a, _b):
    return 1 if math.isnan(a) else 0


OPS = {
    "iden": ("a", expect_iden),
    "isinf": ("(uint64_t)weft_fp64::is_inf(a)", expect_isinf),
    "isnan": ("(uint64_t)weft_fp64::is_nan(a)", expect_isnan),
    "add": ("weft_fp64::add(a, b)", expect_add),
    "sub": ("weft_fp64::sub(a, b)", expect_sub),
    "mul": ("weft_fp64::mul(a, b)", expect_mul),
    "div": ("weft_fp64::div(a, b)", expect_div),
    "lt": ("(uint64_t)weft_fp64::lt(a, b)", expect_lt),
    "eq": ("(uint64_t)weft_fp64::eq(a, b)", expect_eq),
    "floor": ("weft_fp64::floor_bits(a)", expect_floor),
    "ceil": ("weft_fp64::ceil_bits(a)", expect_ceil),
    "trunc": ("weft_fp64::trunc_bits(a)", expect_trunc),
    "round": ("weft_fp64::round_bits(a)", expect_round),
    "fromf32": ("weft_fp64::from_f32((uint32_t)a)", expect_from_f32),
    "tof32": ("(uint64_t)weft_fp64::to_f32(a)", expect_to_f32),
    "fromu64": ("weft_fp64::from_u64(a)", expect_from_u64),
    "tou64": ("weft_fp64::to_u64(a)", expect_to_u64),
}


def main():
    want = [w for w in sys.argv[1:] if w in OPS] or list(OPS)
    if not build_probe():
        return 2
    head = preamble()
    words = []
    for a, b in CASES:
        words.append(bits(a))
        words.append(bits(b))
    data = struct.pack("<%dQ" % len(words), *words)
    bad = 0
    for name in want:
        expr, expected = OPS[name]
        path = os.path.join(tempfile.gettempdir(), "f64_%s.metal" % name)
        open(path, "w").write(head + kernel(expr))
        inb = os.path.join(tempfile.gettempdir(), "f64_%s.bin" % name)
        open(inb, "wb").write(data)
        try:
            r = subprocess.run([PROBE, path, inb], capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            print("%-8s the gpu never finished: an unbounded loop in this operation" % name)
            bad += 1
            continue
        if r.returncode != 0:
            first = (r.stderr or r.stdout).decode().strip().splitlines()
            print("%-8s failed: %s" % (name, first[0][:150] if first else "?"))
            bad += 1
            continue
        got = [int(line, 16) for line in r.stdout.decode().split()]
        fails = 0
        for (a, b), g in zip(CASES, got):
            e = expected(a, b)
            if e is None:
                continue  # the specification leaves this conversion undefined
            if g != e:
                fails += 1
                if fails <= 8:
                    print("%-8s %016x %016x  got %016x  want %016x  (%r vs %r)"
                          % (name, bits(a), bits(b), g, e, dbl(g), dbl(e)))
        print("%-8s %s" % (name, "ok  %d cases" % len(got) if not fails
                           else "%d of %d cases wrong" % (fails, len(got))))
        if fails:
            bad += 1
    print("\n%d of %d operations disagree" % (bad, len(want)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
